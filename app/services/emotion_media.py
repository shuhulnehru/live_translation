from __future__ import annotations

import os
import shutil
import uuid
from typing import Any

from ..orchestrator.langgraph_flow import get_emotion_orchestrator
from .audio_extractor import create_temp_dir


async def run_emotion_media_pipeline(
    file_path: str,
    output_dir: str,
    prompt_text: str = "",
    source_language: str = "auto",
    target_language: str = "en",
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    work_dir = create_temp_dir()
    request_id = uuid.uuid4().hex[:12]
    initial_state: dict[str, Any] = {
        "request_id": request_id,
        "file_path": file_path,
        "prompt_text": prompt_text,
        "source_language": source_language,
        "target_language": target_language,
        "output_dir": output_dir,
        "work_dir": work_dir,
        "errors": [],
    }
    try:
        graph = get_emotion_orchestrator()
        final_state = await graph.ainvoke(initial_state)
        return {
            "request_id": request_id,
            "detection": final_state.get("detection", {}),
            "language_style_note": final_state.get("language_style_note", ""),
            "output": final_state.get("output", {}),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
