from __future__ import annotations

import asyncio
import os
import shutil

from langgraph.graph import END, START, StateGraph

from ..agents.state import EmotionVideoState
from ..agents.tool_registry import translate_segment_tool
from ..models.emotion_classifier import EmotionClassifier
from ..models.tts_emotion import EmotionalTTS
from ..services.audio_extractor import extract_full_audio
from ..services.face_restore import run_face_restore_strict
from ..services.face_routing import run_face_routing_strict
from ..services.lipsync import run_lipsync_refinement
from ..services.timing import align_audio_duration_strict
from ..services.transcriber import transcribe_audio
from ..services.video_builder import build_dubbed_video


emotion_classifier = EmotionClassifier()
emotional_tts = EmotionalTTS()


async def detect_emotion(state: EmotionVideoState) -> dict:
    audio_path = state.get("audio_input_path", "")
    if not audio_path:
        audio_path = await extract_full_audio(state["file_path"], state["work_dir"])
    pred = await emotion_classifier.predict(
        prompt_text=state.get("prompt_text", ""),
        audio_path=audio_path,
    )
    return {
        "audio_input_path": audio_path,
        "detection": {
            "label": pred.label,
            "confidence": pred.confidence,
            "source": pred.source,
        }
    }


async def transcript_conversion_agent(state: EmotionVideoState) -> dict:
    source_lang = state.get("source_language", "auto")
    target_lang = state.get("target_language", "en")
    transcript_text = state.get("prompt_text", "").strip()

    if not transcript_text:
        transcription = await transcribe_audio(state["audio_input_path"], language=source_lang)
        transcript_text = " ".join(seg.text.strip() for seg in transcription.segments if seg.text.strip())
        source_dialogue_duration = sum(max(0.0, seg.duration) for seg in transcription.segments)
    else:
        # Prompt-only mode: use generated duration as reference.
        source_dialogue_duration = 0.0
    if not transcript_text:
        raise RuntimeError("transcription produced empty dialogue text")

    translated_text = transcript_text
    if transcript_text and source_lang != target_lang:
        max_chars = max(64, int(len(transcript_text) * 1.3))
        translated_text = await translate_segment_tool(
            transcript_text,
            target_lang,
            max_chars,
            context=[],
        )

    detection = state["detection"]
    audio_path = os.path.join(state["output_dir"], f"{state['request_id']}_dialogue.mp3")
    await emotional_tts.synthesize(
        text=translated_text or transcript_text,
        emotion=detection["label"],
        target_language=target_lang,
        output_path=audio_path,
    )
    generated_duration = await emotional_tts.probe_duration(audio_path)
    if source_dialogue_duration <= 0:
        source_dialogue_duration = generated_duration

    return {
        "transcript_text": transcript_text,
        "translated_text": translated_text,
        "dialogue_audio_path": audio_path,
        "source_dialogue_duration": source_dialogue_duration,
        "generated_dialogue_duration": generated_duration,
    }


def _language_note(target_language: str) -> str:
    notes = {
        "hi": "Hindi style: slightly expressive and warm delivery.",
        "en": "English style: neutral clarity with moderate pacing.",
        "ja": "Japanese style: polite cadence and softer emphasis.",
        "ko": "Korean style: balanced prosody and clear diction.",
        "es": "Spanish style: energetic pacing with expressive intonation.",
    }
    return notes.get(target_language, "Balanced delivery and natural facial emphasis.")


async def facial_correction_agent(state: EmotionVideoState) -> dict:
    emotion = state["detection"]["label"]
    target_language = state.get("target_language", "en")
    language_note = _language_note(target_language)
    corrected_path = os.path.join(state["output_dir"], f"{state['request_id']}_face_corrected.mp4")

    # Practical facial-expression proxy: emotion-aware visual grading pass.
    filters = {
        "happy": "eq=saturation=1.25:contrast=1.08:brightness=0.03,unsharp=5:5:0.4",
        "sad": "eq=saturation=0.82:contrast=0.96:brightness=-0.02",
        "angry": "eq=saturation=1.18:contrast=1.2:brightness=-0.01,unsharp=5:5:0.5",
        "calm": "eq=saturation=0.92:contrast=1.0:brightness=0.01",
        "neutral": "eq=saturation=1.0:contrast=1.0:brightness=0",
    }
    vf = filters.get(emotion, filters["neutral"])

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        state["file_path"],
        "-vf",
        vf,
        "-an",
        corrected_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"facial correction failed: {stderr.decode(errors='ignore')}")
    return {"corrected_video_path": corrected_path, "language_style_note": language_note}


async def timing_align_agent(state: EmotionVideoState) -> dict:
    aligned_path = os.path.join(state["output_dir"], f"{state['request_id']}_aligned_dialogue.mp3")
    out_path, ratio, stretched = await align_audio_duration_strict(
        input_audio_path=state["dialogue_audio_path"],
        target_duration=state["source_dialogue_duration"],
        output_audio_path=aligned_path,
        tolerance=0.12,
    )
    return {
        "aligned_audio_path": out_path,
        "timing_ratio": ratio,
        "timing_stretch_applied": stretched,
    }


async def speaker_face_routing_agent(state: EmotionVideoState) -> dict:
    metadata_path = os.path.join(state["output_dir"], f"{state['request_id']}_face_tracks.json")
    out_path = await run_face_routing_strict(
        input_video_path=state["corrected_video_path"],
        output_metadata_path=metadata_path,
        source_language=state.get("source_language", "auto"),
        target_language=state.get("target_language", "en"),
    )
    return {"face_routing_metadata_path": out_path}


async def face_restore_agent(state: EmotionVideoState) -> dict:
    restored_path = os.path.join(state["output_dir"], f"{state['request_id']}_restored.mp4")
    out_path = await run_face_restore_strict(
        input_video_path=state["lipsynced_video_path"],
        output_video_path=restored_path,
    )
    return {"restored_video_path": out_path}


async def final_refinement_agent(state: EmotionVideoState) -> dict:
    refined_path = os.path.join(state["output_dir"], f"{state['request_id']}_refined.mp4")
    final_path = os.path.join(state["output_dir"], f"{state['request_id']}_final.mp4")

    # Refinement stage after lip sync; keep hook for future gesture-level post-processing.
    shutil.copy2(state["restored_video_path"], refined_path)
    merged = await build_dubbed_video(refined_path, state["aligned_audio_path"], final_path)

    return {
        "refined_video_path": refined_path,
        "output": {
            "emotion": state["detection"]["label"],
            "confidence": state["detection"]["confidence"],
            "transcript_text": state.get("transcript_text", ""),
            "translated_text": state.get("translated_text", ""),
            "dialogue_audio_path": state["dialogue_audio_path"],
            "aligned_audio_path": state["aligned_audio_path"],
            "corrected_video_path": state["corrected_video_path"],
            "face_routing_metadata_path": state["face_routing_metadata_path"],
            "lipsynced_video_path": state["lipsynced_video_path"],
            "restored_video_path": state["restored_video_path"],
            "refined_video_path": refined_path,
            "final_video_path": merged,
        },
    }


async def lipsync_refinement_agent(state: EmotionVideoState) -> dict:
    lip_path = os.path.join(state["output_dir"], f"{state['request_id']}_lipsynced.mp4")
    output_path, _ = await run_lipsync_refinement(
        input_video_path=state["corrected_video_path"],
        input_audio_path=state["aligned_audio_path"],
        output_video_path=lip_path,
        face_routing_metadata_path=state["face_routing_metadata_path"],
    )
    return {"lipsynced_video_path": output_path}


def build_emotion_graph():
    graph = StateGraph(EmotionVideoState)

    graph.add_node("detectEmotion", detect_emotion)
    graph.add_node("transcriptConversionAgent", transcript_conversion_agent)
    graph.add_node("facialCorrectionAgent", facial_correction_agent)
    graph.add_node("timingAlignAgent", timing_align_agent)
    graph.add_node("speakerFaceRoutingAgent", speaker_face_routing_agent)
    graph.add_node("lipSyncRefinementAgent", lipsync_refinement_agent)
    graph.add_node("faceRestoreAgent", face_restore_agent)
    graph.add_node("finalRefinementAgent", final_refinement_agent)

    graph.add_edge(START, "detectEmotion")
    graph.add_edge("detectEmotion", "transcriptConversionAgent")
    graph.add_edge("detectEmotion", "facialCorrectionAgent")
    graph.add_edge("transcriptConversionAgent", "timingAlignAgent")
    graph.add_edge("facialCorrectionAgent", "speakerFaceRoutingAgent")
    graph.add_edge("timingAlignAgent", "lipSyncRefinementAgent")
    graph.add_edge("speakerFaceRoutingAgent", "lipSyncRefinementAgent")
    graph.add_edge("lipSyncRefinementAgent", "faceRestoreAgent")
    graph.add_edge("faceRestoreAgent", "finalRefinementAgent")
    graph.add_edge("finalRefinementAgent", END)
    return graph


def create_emotion_orchestrator():
    return build_emotion_graph().compile()


def get_emotion_orchestrator():
    if not hasattr(get_emotion_orchestrator, "_graph"):
        get_emotion_orchestrator._graph = create_emotion_orchestrator()  # type: ignore[attr-defined]
    return get_emotion_orchestrator._graph  # type: ignore[attr-defined]
