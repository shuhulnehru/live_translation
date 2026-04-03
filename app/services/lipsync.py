from __future__ import annotations

import asyncio
import os
import shlex

from ..config import get_settings


async def run_lipsync_refinement(
    input_video_path: str,
    input_audio_path: str,
    output_video_path: str,
    face_routing_metadata_path: str,
) -> tuple[str, bool]:
    """
    Run external lip-sync model (Wav2Lip-style).
    No fallback behavior.
    """
    settings = get_settings()
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    script_path = settings.lipsync_script_path
    checkpoint_path = settings.lipsync_checkpoint_path
    python_bin = settings.lipsync_python_bin or "python3"
    extra_args = settings.lipsync_extra_args
    enabled = bool(settings.lipsync_enabled)

    if not enabled:
        raise RuntimeError("lipsync is disabled in configuration")
    if not script_path or not checkpoint_path:
        raise RuntimeError("lipsync script/checkpoint are not configured")
    if not os.path.exists(script_path):
        raise RuntimeError(f"lipsync script not found: {script_path}")

    cmd = [
        python_bin,
        script_path,
        "--checkpoint_path",
        checkpoint_path,
        "--face",
        input_video_path,
        "--audio",
        input_audio_path,
        "--face_tracks",
        face_routing_metadata_path,
        "--outfile",
        output_video_path,
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not os.path.exists(output_video_path):
        raise RuntimeError(f"lipsync failed: {stderr.decode(errors='ignore')}")

    return output_video_path, True
