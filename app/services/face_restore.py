from __future__ import annotations

import asyncio
import os
import shlex

from ..config import get_settings


async def run_face_restore_strict(
    input_video_path: str,
    output_video_path: str,
) -> str:
    """
    Execute external face restoration script (e.g., GFPGAN/CodeFormer wrapper).
    No fallback behavior.
    """
    settings = get_settings()
    script_path = settings.face_restore_script_path
    python_bin = settings.face_restore_python_bin
    extra_args = settings.face_restore_extra_args

    if not script_path:
        raise RuntimeError("face restore script is not configured")
    if not os.path.exists(script_path):
        raise RuntimeError(f"face restore script not found: {script_path}")

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    cmd = [
        python_bin,
        script_path,
        "--input",
        input_video_path,
        "--output",
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
        raise RuntimeError(f"face restore failed: {stderr.decode(errors='ignore')}")
    return output_video_path
