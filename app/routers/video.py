from __future__ import annotations

import json
import logging
import os
import uuid

import aiofiles
from fastapi import APIRouter, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..services.pipeline import NoSpeechSegmentsError, run_dubbing_pipeline

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

router = APIRouter(prefix="/video", tags=["video"])


@router.post("/upload")
async def upload_video(file: UploadFile) -> dict:
    """Accept a video upload and return a job_id for the dubbing session."""
    job_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    async with aiofiles.open(file_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            await out.write(chunk)

    return {"job_id": job_id, "filename": file.filename, "path": file_path}


@router.get("/download/{job_id}")
async def download_dubbed_video(job_id: str) -> FileResponse:
    """Serve the final dubbed MP4 for download/playback."""
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    video_path = os.path.join(job_output_dir, "dubbed_output.mp4")

    if not os.path.exists(video_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Dubbed video not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"dubbed_{job_id}.mp4",
    )


@router.websocket("/dub/{job_id}")
async def dub_websocket(websocket: WebSocket, job_id: str) -> None:
    """
    WebSocket endpoint that drives the dubbing pipeline.

    Client sends a JSON message with source/target language. The server
    streams progress updates as the pipeline runs. When complete, sends
    a download URL for the final dubbed video.
    """
    await websocket.accept()

    try:
        init_msg = await websocket.receive_text()
        config = json.loads(init_msg)
        source_lang = config.get("source_lang", "auto")
        target_lang = config.get("target_lang", "hi")

        video_files = [
            f for f in os.listdir(UPLOAD_DIR) if f.startswith(job_id)
        ]
        if not video_files:
            await websocket.send_json({"type": "error", "message": "Video not found"})
            await websocket.close()
            return

        video_path = os.path.join(UPLOAD_DIR, video_files[0])

        job_output_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_output_dir, exist_ok=True)

        async def on_progress(data: dict) -> None:
            await websocket.send_json(data)

        final_path = await run_dubbing_pipeline(
            video_path=video_path,
            source_lang=source_lang,
            target_lang=target_lang,
            output_dir=job_output_dir,
            on_progress=on_progress,
        )

        await websocket.send_json({
            "type": "complete",
            "message": "Dubbing complete! Your video is ready.",
            "download_url": f"/video/download/{job_id}",
        })

    except WebSocketDisconnect:
        logger.info("Client disconnected from dub session %s", job_id)
    except NoSpeechSegmentsError as exc:
        logger.warning("No speech found for job %s: %s", job_id, exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    except Exception:
        logger.exception("WebSocket error for job %s", job_id)
        try:
            await websocket.send_json({"type": "error", "message": "Internal server error"})
        except Exception:
            pass
    finally:
        # Clean up uploaded source file (keep output for download)
        try:
            for f in os.listdir(UPLOAD_DIR):
                if f.startswith(job_id):
                    os.remove(os.path.join(UPLOAD_DIR, f))
        except Exception:
            pass
