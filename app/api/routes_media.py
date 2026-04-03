from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..services.emotion_media import run_emotion_media_pipeline

router = APIRouter(prefix="/media", tags=["emotion-media"])

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "emotion_media")
os.makedirs(OUTPUT_DIR, exist_ok=True)
JOB_INDEX: dict[str, dict[str, str]] = {}


class EmotionMediaRequest(BaseModel):
    file_path: str = Field(min_length=1, max_length=1024)
    prompt_text: str = Field(default="", max_length=300)
    source_language: str = Field(default="auto", min_length=2, max_length=8)
    target_language: str = Field(default="en", min_length=2, max_length=8)


@router.post("/compose")
async def compose_emotion_media(payload: EmotionMediaRequest) -> dict[str, Any]:
    result = await run_emotion_media_pipeline(
        file_path=payload.file_path,
        prompt_text=payload.prompt_text,
        source_language=payload.source_language,
        output_dir=OUTPUT_DIR,
        target_language=payload.target_language,
    )
    request_id = result["request_id"]
    output = result.get("output", {})
    JOB_INDEX[request_id] = {
        "dialogue": output.get("dialogue_audio_path", ""),
        "aligned_audio": output.get("aligned_audio_path", ""),
        "corrected": output.get("corrected_video_path", ""),
        "face_tracks": output.get("face_routing_metadata_path", ""),
        "lipsynced": output.get("lipsynced_video_path", ""),
        "restored": output.get("restored_video_path", ""),
        "refined": output.get("refined_video_path", ""),
        "final": output.get("final_video_path", ""),
    }
    return {
        "request_id": request_id,
        "emotion": output.get("emotion"),
        "confidence": output.get("confidence"),
        "language_style_note": result.get("language_style_note"),
        "transcript_text": output.get("transcript_text"),
        "translated_text": output.get("translated_text"),
        "assets": {
            "dialogue_audio_url": f"/media/assets/{request_id}/dialogue",
            "aligned_audio_url": f"/media/assets/{request_id}/aligned_audio",
            "corrected_video_url": f"/media/assets/{request_id}/corrected",
            "face_tracks_url": f"/media/assets/{request_id}/face_tracks",
            "lipsynced_video_url": f"/media/assets/{request_id}/lipsynced",
            "restored_video_url": f"/media/assets/{request_id}/restored",
            "refined_video_url": f"/media/assets/{request_id}/refined",
            "final_video_url": f"/media/assets/{request_id}/final",
        },
    }


@router.get("/assets/{request_id}/{asset_type}")
async def get_media_asset(request_id: str, asset_type: str) -> FileResponse:
    job = JOB_INDEX.get(request_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown request_id")
    if asset_type not in {"dialogue", "aligned_audio", "corrected", "face_tracks", "lipsynced", "restored", "refined", "final"}:
        raise HTTPException(
            status_code=400,
            detail="asset_type must be dialogue|aligned_audio|corrected|face_tracks|lipsynced|restored|refined|final",
        )
    path = job.get(asset_type, "")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Asset not found")
    if asset_type in {"dialogue", "aligned_audio"}:
        media_type = "audio/mpeg"
    elif asset_type == "face_tracks":
        media_type = "application/json"
    else:
        media_type = "video/mp4"
    return FileResponse(path=path, media_type=media_type, filename=os.path.basename(path))
