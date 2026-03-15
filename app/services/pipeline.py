from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections.abc import Awaitable, Callable
from typing import Any

from .audio_extractor import create_temp_dir, extract_full_audio
from .frame_extractor import extract_keyframes
from .transcriber import transcribe_audio
from .translator import translate_segments
from .tts import synthesize_speech
from .video_builder import TTSClip, build_dubbed_audio, build_dubbed_video, get_video_duration

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


async def run_dubbing_pipeline(
    video_path: str,
    source_lang: str,
    target_lang: str,
    output_dir: str,
    on_progress: ProgressCallback,
) -> str:
    """
    Full offline dubbing pipeline:
    1. Extract audio + keyframes from video (parallel).
    2. Transcribe with OpenAI Whisper API.
    3. Translate with GPT-4o using scene keyframes for context.
    4. Generate rate-matched TTS for each segment.
    5. Assemble all TTS clips into a single audio track.
    6. Mux dubbed audio + original video into final MP4.

    Returns the path to the final dubbed MP4 file.
    """
    work_dir = create_temp_dir()
    tts_dir = os.path.join(work_dir, "tts")
    os.makedirs(tts_dir, exist_ok=True)

    try:
        # Step 1: Extract audio and keyframes in parallel
        await on_progress({"type": "step", "step": "extracting", "message": "Extracting audio and keyframes..."})

        audio_task = extract_full_audio(video_path, work_dir)
        frames_task = extract_keyframes(video_path, work_dir)
        duration_task = get_video_duration(video_path)

        audio_path, keyframe_map, total_duration = await asyncio.gather(
            audio_task, frames_task, duration_task
        )

        await on_progress({
            "type": "step",
            "step": "extracting",
            "message": f"Extracted audio and {len(keyframe_map)} keyframes.",
        })

        # Step 2: Transcribe with OpenAI Whisper API
        await on_progress({"type": "step", "step": "transcribing", "message": "Transcribing speech with OpenAI Whisper..."})

        transcription = await transcribe_audio(audio_path, language=source_lang)
        segments = transcription.segments

        if not segments:
            await on_progress({"type": "step", "step": "transcribing", "message": "No speech detected."})
            await on_progress({"type": "error", "message": "No speech found in the video."})
            raise RuntimeError("No speech segments found")

        await on_progress({
            "type": "step",
            "step": "transcribing",
            "message": f"Found {len(segments)} speech segments.",
            "total_segments": len(segments),
        })

        # Step 3: Translate with GPT-4o + keyframes
        await on_progress({"type": "step", "step": "translating", "message": "Translating with GPT-4o (scene-aware)..."})

        translated = await translate_segments(
            segments=segments,
            keyframe_map=keyframe_map,
            source_lang=transcription.language,
            target_lang=target_lang,
        )

        await on_progress({
            "type": "step",
            "step": "translating",
            "message": f"Translated {len(translated)} segments.",
            "segments": [
                {
                    "index": t.index,
                    "start": t.start,
                    "end": t.end,
                    "original": t.original_text,
                    "translated": t.translated_text,
                }
                for t in translated
            ],
        })

        # Step 4: Generate TTS for each segment
        await on_progress({"type": "step", "step": "synthesizing", "message": "Generating dubbed speech..."})

        tts_clips: list[TTSClip] = []
        for i, seg in enumerate(translated):
            if not seg.translated_text.strip():
                continue

            tts_path = os.path.join(tts_dir, f"tts_{seg.index:04d}.mp3")
            await synthesize_speech(
                text=seg.translated_text,
                target_lang=target_lang,
                output_path=tts_path,
                target_duration=seg.duration,
            )
            tts_clips.append(TTSClip(path=tts_path, start_time=seg.start))

            if (i + 1) % 5 == 0 or i == len(translated) - 1:
                await on_progress({
                    "type": "step",
                    "step": "synthesizing",
                    "message": f"Generated speech: {i + 1}/{len(translated)} segments",
                })

        # Step 5: Build dubbed audio track
        await on_progress({"type": "step", "step": "building", "message": "Assembling dubbed audio track..."})

        dubbed_audio_path = os.path.join(work_dir, "dubbed_audio.m4a")
        await build_dubbed_audio(tts_clips, total_duration, dubbed_audio_path)

        # Step 6: Mux final video
        await on_progress({"type": "step", "step": "building", "message": "Building final dubbed video..."})

        final_video_path = os.path.join(output_dir, "dubbed_output.mp4")
        await build_dubbed_video(video_path, dubbed_audio_path, final_video_path)

        await on_progress({"type": "step", "step": "done", "message": "Dubbing complete!"})

        return final_video_path

    except Exception as exc:
        logger.exception("Pipeline error")
        await on_progress({"type": "error", "message": str(exc)})
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
