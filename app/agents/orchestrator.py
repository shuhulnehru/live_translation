"""LangGraph implementation of Orchestrator + 4 sub-agents."""

from __future__ import annotations

import asyncio
import logging
import os

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

from .state import AudioClip, DubbingState, Segment, TranslatedSegment
from .tool_registry import (
    clone_voice_tool,
    detect_subtitles_tool,
    diarize_speakers_tool,
    extract_voice_sample_tool,
    fit_audio_duration_tool,
    mux_audio_video_tool,
    normalize_audio_tool,
    retry_translation_tool,
    synthesize_speech_tool,
    transcribe_audio_tool,
    translate_segment_tool,
    validate_cps_tool,
)


def _emit(state: DubbingState, step: str, message: str) -> None:
    cb = state.get("on_progress")
    if cb:
        asyncio.create_task(cb({"type": "step", "step": step, "message": message}))


async def orchestrator_agent(state: DubbingState) -> dict:
    file_path = state["file_path"]
    logger.info("Orchestrator: checking subtitles for file_path=%s", file_path)
    has_subtitles, subtitle_path = detect_subtitles_tool(file_path)
    _emit(state, "orchestrator", "Checking subtitles and routing flow...")
    route = "translation (has_subtitles)" if has_subtitles else "transcription (no_subtitles)"
    logger.info("Orchestrator: routing to %s", route)
    return {"has_subtitles": has_subtitles, "subtitles_path": subtitle_path}


def route_by_subtitle_check(state: DubbingState) -> str:
    return "has_subtitles" if state.get("has_subtitles") else "no_subtitles"


async def transcription_agent(state: DubbingState) -> dict:
    logger.info("Transcription agent: starting")
    _emit(state, "transcription", "Transcribing and diarizing audio...")
    segments = await transcribe_audio_tool(state["file_path"], state.get("source_language", "auto"))
    segments = diarize_speakers_tool(segments)
    logger.info("Transcription agent: finished segments=%d", len(segments))
    return {"segments": segments}


async def translation_agent(state: DubbingState) -> dict:
    logger.info("Translation agent: starting target_language=%s", state["target_language"])
    _emit(state, "translation", "Translating with CPS validation...")
    target_lang = state["target_language"]
    segments: list[Segment] = state.get("segments", [])
    translated: list[TranslatedSegment] = []

    for i, seg in enumerate(segments):
        context = [t["translated_text"] for t in translated[-2:]]
        max_chars = int(seg["duration"] * 18)
        text = await translate_segment_tool(seg["original_text"], target_lang, max_chars, context)
        fits, cps = validate_cps_tool(text, seg["duration"])
        retries = 0
        while not fits and retries < 3:
            max_chars = int(max_chars * 0.85)
            text = await retry_translation_tool(seg["original_text"], target_lang, max_chars, context)
            fits, cps = validate_cps_tool(text, seg["duration"])
            retries += 1
        if not fits:
            text = text[: max(10, int(max_chars * 0.75))]
            fits, cps = validate_cps_tool(text, seg["duration"])

        translated.append(
            {
                "id": seg["id"],
                "original_text": seg["original_text"],
                "translated_text": text,
                "char_count": len(text),
                "fits": fits,
                "cps": cps,
            }
        )
    logger.info("Translation agent: finished translated_segments=%d", len(translated))
    return {"translated_segments": translated}


async def synthesis_agent(state: DubbingState) -> dict:
    logger.info("Synthesis agent: starting")
    _emit(state, "synthesis", "Synthesizing per-speaker voice clips...")
    voice_registry = dict(state.get("voice_registry", {}))
    translated = state.get("translated_segments", [])
    segments = {s["id"]: s for s in state.get("segments", [])}
    clips: list[AudioClip] = []
    work_dir = state["work_dir"]
    os.makedirs(work_dir, exist_ok=True)

    for t in translated:
        seg = segments.get(t["id"])
        if not seg:
            continue
        speaker = seg["speaker_id"]
        voice_id = voice_registry.get(speaker)
        if not voice_id:
            sample = await extract_voice_sample_tool(
                state["file_path"], speaker, seg["start_time"], seg["end_time"], work_dir
            )
            voice_id = await clone_voice_tool(sample, speaker)
            voice_registry[speaker] = voice_id

        out_path = os.path.join(work_dir, f"clip_{t['id']:05d}.mp3")
        clip_path, actual_duration = await synthesize_speech_tool(
            t["translated_text"], voice_id, state["target_language"], out_path, seg["duration"]
        )
        ratio = actual_duration / seg["duration"] if seg["duration"] > 0 else 1.0
        adjusted = False
        if ratio < 0.85 or ratio > 1.15:
            clip_path, adjusted = await fit_audio_duration_tool(clip_path, seg["duration"])

        clips.append(
            {
                "id": t["id"],
                "file_path": clip_path,
                "duration": seg["duration"],
                "start_time": seg["start_time"],
                "adjusted": adjusted,
            }
        )
    logger.info("Synthesis agent: finished clips=%d", len(clips))
    return {"voice_registry": voice_registry, "audio_clips": clips}


async def audio_mix_agent(state: DubbingState) -> dict:
    logger.info("Audio mix agent: starting")
    _emit(state, "audio_mix", "Normalizing and muxing final output...")
    clips: list[AudioClip] = []
    for clip in state.get("audio_clips", []):
        normalized = await normalize_audio_tool(clip["file_path"])
        updated = dict(clip)
        updated["file_path"] = normalized
        clips.append(updated)

    if not clips:
        msg = (
            "No speech segments to dub. Transcription may have found no voiced content in the video "
            "(e.g. music-only, or all segments filtered out). Try a file with clear speech."
        )
        logger.error("%s", msg)
        raise ValueError(msg)

    out_dir = state["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "dubbed_output.mp4")
    final = await mux_audio_video_tool(
        state["file_path"],
        clips,
        output_path,
        original_audio_volume=0.05,
    )
    logger.info("Audio mix agent: finished output_path=%s", final)
    return {"audio_clips": clips, "output_path": final}


def build_graph():
    graph = StateGraph(DubbingState)

    graph.add_node("orchestrator", orchestrator_agent)
    graph.add_node("transcription", transcription_agent)
    graph.add_node("translation", translation_agent)
    graph.add_node("synthesis", synthesis_agent)
    graph.add_node("audio_mix", audio_mix_agent)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_by_subtitle_check,
        {"has_subtitles": "translation", "no_subtitles": "transcription"},
    )
    graph.add_edge("transcription", "translation")
    graph.add_edge("translation", "synthesis")
    graph.add_edge("synthesis", "audio_mix")
    graph.add_edge("audio_mix", END)
    return graph


def create_orchestrator():
    return build_graph().compile()


def get_orchestrator():
    if not hasattr(get_orchestrator, "_graph"):
        get_orchestrator._graph = create_orchestrator()  # type: ignore[attr-defined]
    return get_orchestrator._graph  # type: ignore[attr-defined]
