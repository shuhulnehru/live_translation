from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections.abc import Awaitable, Callable
from typing import Any

from ..agents.orchestrator import get_orchestrator
from .audio_extractor import create_temp_dir

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]

NO_SPEECH_MESSAGE_PREFIX = "No speech segments to dub"


class NoSpeechSegmentsError(RuntimeError):
    """Raised when the pipeline cannot find any dub-able speech segments."""


async def run_dubbing_pipeline(
    video_path: str,
    source_lang: str,
    target_lang: str,
    output_dir: str,
    on_progress: ProgressCallback,
) -> str:
    """
    Run the dubbing pipeline via the LangGraph Orchestrator Agent.

    Flow: Extract -> Transcription (+ Diarization) -> Translation (+ Validation)
    -> Synthesis (+ Duration Fit) -> Final Dubbed Video.

    Returns the path to the final dubbed MP4 file.
    """
    work_dir = create_temp_dir()
    logger.info(
        "Pipeline starting: video_path=%s source_lang=%s target_lang=%s output_dir=%s",
        video_path, source_lang, target_lang, output_dir,
    )

    try:
        orchestrator = get_orchestrator()
        initial_state: dict[str, Any] = {
            "file_path": video_path,
            "target_language": target_lang,
            "source_language": source_lang,
            "output_dir": output_dir,
            "work_dir": work_dir,
            "voice_registry": {},
            "segments": [],
            "translated_segments": [],
            "audio_clips": [],
            "errors": [],
            "on_progress": on_progress,
        }

        final_state = await orchestrator.ainvoke(initial_state)

        final_path = final_state.get("output_path") or ""
        if not final_path or not os.path.exists(final_path):
            raise RuntimeError(final_state.get("error", "Pipeline failed to produce output"))

        logger.info("Pipeline complete: output_path=%s", final_path)
        await on_progress({"type": "step", "step": "done", "message": "Dubbing complete!"})
        return final_path

    except ValueError as exc:
        message = str(exc)
        if message.startswith(NO_SPEECH_MESSAGE_PREFIX):
            logger.warning("Pipeline stopped: %s", message)
            await on_progress({"type": "error", "message": message})
            raise NoSpeechSegmentsError(message) from exc
        logger.exception("Pipeline error")
        await on_progress({"type": "error", "message": message})
        raise
    except Exception as exc:
        logger.exception("Pipeline error")
        await on_progress({"type": "error", "message": str(exc)})
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
