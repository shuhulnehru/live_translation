from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path

import ffmpeg
from langchain_openai import ChatOpenAI

from ..config import get_settings

logger = logging.getLogger(__name__)
from ..services.frame_extractor import extract_keyframes
from ..services.transcriber import SpeechSegment, transcribe_audio
from ..services.translator import translate_segments
from ..services.tts import synthesize_speech
from ..services.video_builder import (
    TTSClip,
    build_dubbed_audio,
    build_dubbed_video,
    get_video_duration,
)
from .state import AudioClip, Segment


def _llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=settings.openai_api_key)


def detect_subtitles_tool(file_path: str) -> tuple[bool, str | None]:
    """Check embedded subtitle streams and sidecar .srt/.vtt."""
    logger.debug("Checking for subtitles: file_path=%s", file_path)
    p = Path(file_path)
    for ext in (".srt", ".vtt"):
        sidecar = p.with_suffix(ext)
        if sidecar.exists():
            logger.info("Found sidecar subtitles: path=%s", sidecar)
            return True, str(sidecar)

    try:
        probe = ffmpeg.probe(file_path)
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "subtitle":
                logger.info("Found embedded subtitle stream in file_path=%s", file_path)
                return True, None
    except Exception as e:
        logger.debug("Probe for subtitles failed (non-fatal): %s", e)
    logger.debug("No subtitles found for file_path=%s", file_path)
    return False, None


async def transcribe_audio_tool(file_path: str, source_language: str) -> list[Segment]:
    """Whisper transcription with timestamps -> canonical segment list."""
    logger.info("Starting transcription: file_path=%s source_language=%s", file_path, source_language)
    denoised_audio = await _prepare_denoised_audio_for_transcription(file_path)
    voiced_ranges = await _detect_voiced_ranges_ffmpeg(denoised_audio)
    logger.debug("Detected %d voiced ranges for overlap filtering", len(voiced_ranges))
    result = await transcribe_audio(denoised_audio, language=source_language)
    raw_count = len(result.segments)
    out: list[Segment] = []
    skipped_voiced = skipped_short = skipped_empty = 0
    for s in result.segments:
        if not _has_enough_voiced_overlap(s.start, s.end, voiced_ranges, min_overlap_ratio=0.30):
            skipped_voiced += 1
            continue
        if (s.end - s.start) < 0.30:
            skipped_short += 1
            continue
        if len(s.text.strip()) < 2:
            skipped_empty += 1
            continue
        out.append(
            {
                "id": s.index,
                "start_time": s.start,
                "end_time": s.end,
                "duration": max(0.01, s.duration),
                "speaker_id": "SPEAKER_0",
                "original_text": s.text,
            }
        )
    logger.info(
        "Transcription complete: raw_segments=%d kept=%d skipped_voiced=%d skipped_short=%d skipped_empty=%d",
        raw_count, len(out), skipped_voiced, skipped_short, skipped_empty,
    )
    return out


async def _prepare_denoised_audio_for_transcription(file_path: str) -> str:
    """
    Extract and denoise audio before transcription.

    Uses FFmpeg filters:
    - `highpass` to remove low-end rumble
    - `lowpass` to trim very high-frequency hiss
    - `afftdn` spectral denoise for broadband noise reduction
    """
    base = Path(file_path).stem
    parent = Path(file_path).parent
    out_path = str(parent / f"{base}_denoised_16k.wav")

    cmd = (
        ffmpeg.input(file_path)
        .filter("highpass", f=120)
        .filter("lowpass", f=7600)
        .filter("afftdn", nr=18)
        .output(out_path, ac=1, ar=16000, format="wav")
        .overwrite_output()
        .compile()
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0 or not os.path.exists(out_path):
        logger.warning(
            "Denoise failed (returncode=%s), using original file: %s; stderr=%s",
            proc.returncode, file_path, stderr.decode(errors="replace")[-500:] if stderr else "",
        )
        return file_path
    logger.debug("Denoised audio written: path=%s", out_path)
    return out_path


async def _detect_voiced_ranges_ffmpeg(audio_path: str) -> list[tuple[float, float]]:
    """
    Detect voiced ranges by running FFmpeg silencedetect and inverting silence windows.
    Returns [(start, end), ...] time ranges where speech is likely present.
    """
    total_duration = await _probe_duration(audio_path)
    if total_duration <= 0:
        logger.warning("Could not get duration for voiced detection, path=%s", audio_path)
        return []

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        audio_path,
        "-af",
        "silencedetect=noise=-35dB:d=0.35",
        "-f",
        "null",
        "-",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    text = stderr.decode(errors="ignore")

    silence_starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9]*\.?[0-9]+)", text)]
    silence_ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9]*\.?[0-9]+)", text)]

    silences: list[tuple[float, float]] = []
    i = 0
    j = 0
    while i < len(silence_starts) or j < len(silence_ends):
        if i < len(silence_starts) and (j >= len(silence_ends) or silence_starts[i] <= silence_ends[j]):
            start = silence_starts[i]
            end = total_duration
            if j < len(silence_ends):
                end = silence_ends[j]
                j += 1
            i += 1
            silences.append((max(0.0, start), min(total_duration, end)))
        else:
            # Unpaired silence_end; skip it
            j += 1

    silences.sort(key=lambda x: x[0])
    voiced: list[tuple[float, float]] = []
    cursor = 0.0
    for s_start, s_end in silences:
        if s_start > cursor:
            voiced.append((cursor, s_start))
        cursor = max(cursor, s_end)
    if cursor < total_duration:
        voiced.append((cursor, total_duration))

    # Remove tiny intervals unlikely to be real speech.
    filtered = [(a, b) for a, b in voiced if (b - a) >= 0.20]
    logger.debug("Voiced ranges: total_duration=%.2fs count=%d", total_duration, len(filtered))
    return filtered


async def _probe_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    try:
        return float(out.decode().strip())
    except Exception:
        return 0.0


def _has_enough_voiced_overlap(
    seg_start: float,
    seg_end: float,
    voiced_ranges: list[tuple[float, float]],
    min_overlap_ratio: float = 0.30,
) -> bool:
    duration = max(0.0, seg_end - seg_start)
    if duration <= 0:
        return False
    overlap = 0.0
    for v_start, v_end in voiced_ranges:
        start = max(seg_start, v_start)
        end = min(seg_end, v_end)
        if end > start:
            overlap += end - start
    return (overlap / duration) >= min_overlap_ratio


def diarize_speakers_tool(segments: list[Segment]) -> list[Segment]:
    """Placeholder diarization (swap with pyannote output)."""
    logger.debug("Diarizing %d segments (heuristic speaker turns)", len(segments))
    speaker = "SPEAKER_0"
    prev_end = 0.0
    out: list[Segment] = []
    for seg in segments:
        if seg["start_time"] - prev_end > 1.5:
            speaker = "SPEAKER_1" if speaker == "SPEAKER_0" else "SPEAKER_0"
        prev_end = seg["end_time"]
        seg = dict(seg)
        seg["speaker_id"] = speaker
        out.append(seg)
    return out


def _cps(text: str, duration: float) -> float:
    return (len(text) / duration) if duration > 0 else 999.0


def validate_cps_tool(text: str, duration: float) -> tuple[bool, float]:
    cps = _cps(text, duration)
    return cps <= 18.0, cps


async def translate_segment_tool(
    text: str,
    target_language: str,
    max_chars: int,
    context: list[str],
) -> str:
    logger.debug("Translating segment: target=%s max_chars=%d len_text=%d", target_language, max_chars, len(text))
    settings = get_settings()
    if not settings.openai_api_key:
        logger.warning("No OpenAI API key; returning truncated original for segment")
        return text[:max_chars]

    prompt = (
        f"Translate to {target_language}. Keep under {max_chars} chars. "
        "Preserve intent and natural dialogue flow.\n"
        f"Context:\n{json.dumps(context, ensure_ascii=False)}\n"
        f"Text: {text}\n"
        "Return only translated text."
    )
    msg = await asyncio.to_thread(_llm().invoke, prompt)
    translated = (msg.content or "").strip()
    return translated[:max_chars] if len(translated) > max_chars else translated


async def retry_translation_tool(
    text: str,
    target_language: str,
    stricter_max_chars: int,
    context: list[str],
) -> str:
    return await translate_segment_tool(text, target_language, stricter_max_chars, context)


async def extract_voice_sample_tool(
    file_path: str,
    speaker_id: str,
    start: float,
    end: float,
    output_dir: str,
) -> str:
    logger.debug("Extracting voice sample: speaker_id=%s start=%.2f end=%.2f", speaker_id, start, end)
    sample_path = os.path.join(output_dir, f"voice_sample_{speaker_id}.wav")
    duration = max(1.0, min(10.0, end - start))
    cmd = (
        ffmpeg.input(file_path, ss=start, t=duration)
        .output(sample_path, ac=1, ar=16000, format="wav")
        .overwrite_output()
        .compile()
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning("Voice sample extraction failed for speaker_id=%s: %s", speaker_id, stderr.decode(errors="replace")[:200])
    return sample_path


async def clone_voice_tool(sample_path: str, speaker_id: str) -> str:
    """
    Provider hook for Cartesia/ElevenLabs/XTTS.
    Scaffold returns a deterministic local voice id.
    """
    voice_id = f"voice_{speaker_id}"
    logger.debug("Clone voice (scaffold): speaker_id=%s -> voice_id=%s", speaker_id, voice_id)
    return voice_id


async def synthesize_speech_tool(
    text: str,
    voice_id: str,
    language: str,
    output_path: str,
    target_duration: float,
) -> tuple[str, float]:
    # voice_id hook retained for external providers; local TTS uses language voice map.
    path = await synthesize_speech(text, language, output_path, target_duration)
    # quick duration estimate using ffprobe
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    stdout, _ = await proc.communicate()
    try:
        duration = float(stdout.decode().strip())
    except Exception:
        duration = target_duration
    return path, duration


async def fit_audio_duration_tool(clip_path: str, target_duration: float) -> tuple[str, bool]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        clip_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    stdout, _ = await proc.communicate()
    try:
        current = float(stdout.decode().strip())
    except Exception:
        return clip_path, False
    if current <= 0:
        return clip_path, False

    ratio = current / target_duration if target_duration > 0 else 1.0
    if 0.85 <= ratio <= 1.15:
        return clip_path, False

    logger.debug("Fitting clip duration: path=%s current=%.2fs target=%.2fs ratio=%.2f", clip_path, current, target_duration, ratio)
    adjusted = clip_path.replace(".mp3", "_fit.mp3")
    atempo = min(2.0, max(0.5, ratio))
    ff_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        clip_path,
        "-filter:a",
        f"atempo={atempo}",
        adjusted,
    ]
    run = await asyncio.create_subprocess_exec(
        *ff_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await run.communicate()
    if not os.path.exists(adjusted):
        logger.warning("fit_audio_duration output missing, using original: %s", clip_path)
        return clip_path, False
    return adjusted, True


async def normalize_audio_tool(clip_path: str) -> str:
    logger.debug("Normalizing audio: path=%s", clip_path)
    normalized = clip_path.replace(".mp3", "_norm.mp3").replace(".wav", "_norm.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        clip_path,
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        normalized,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if not os.path.exists(normalized):
        logger.warning("Normalize output missing, using original: path=%s", clip_path)
        return clip_path
    return normalized


async def mux_audio_video_tool(
    video_path: str,
    clips: list[AudioClip],
    output_path: str,
    original_audio_volume: float = 0.05,
) -> str:
    if not clips:
        raise ValueError(
            "No audio clips to mux. No speech segments were produced by transcription/translation/synthesis."
        )
    logger.info("Muxing audio+video: video_path=%s output_path=%s clips=%d", video_path, output_path, len(clips))
    tts_clips = [TTSClip(path=c["file_path"], start_time=c["start_time"]) for c in clips]
    tmp_audio = output_path.replace(".mp4", "_dubbed_audio.m4a")
    total_duration = await get_video_duration(video_path)
    await build_dubbed_audio(tts_clips, total_duration, tmp_audio)
    # Current video builder replaces audio fully; ambience mix hook kept via parameter for future implementation.
    await build_dubbed_video(video_path, tmp_audio, output_path)
    logger.info("Mux complete: output_path=%s", output_path)
    return output_path

