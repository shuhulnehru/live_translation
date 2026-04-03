from __future__ import annotations

import asyncio
import json
import os
import shlex

from ..config import get_settings


async def run_face_routing_strict(
    input_video_path: str,
    output_metadata_path: str,
    source_language: str,
    target_language: str,
) -> str:
    """
    Execute external face detection+tracking script (e.g., RetinaFace + tracker).
    No fallback behavior: configuration and execution must succeed.
    """
    settings = get_settings()
    script_path = settings.face_routing_script_path
    python_bin = settings.face_routing_python_bin
    extra_args = settings.face_routing_extra_args

    if not script_path:
        raise RuntimeError("face routing script is not configured")
    if not os.path.exists(script_path):
        raise RuntimeError(f"face routing script not found: {script_path}")

    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
    cmd = [
        python_bin,
        script_path,
        "--video",
        input_video_path,
        "--out",
        output_metadata_path,
        "--source_language",
        source_language,
        "--target_language",
        target_language,
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"face routing failed: {stderr.decode(errors='ignore')}")
    if not os.path.exists(output_metadata_path):
        raise RuntimeError("face routing metadata file not produced")

    # Validate JSON schema basics.
    with open(output_metadata_path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "tracks" not in payload:
        raise RuntimeError("face routing metadata missing required 'tracks' field")
    return output_metadata_path
