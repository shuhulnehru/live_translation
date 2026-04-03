from __future__ import annotations

import asyncio
import os
import re


async def evaluate_audio_quality(audio_path: str) -> tuple[float | None, bool]:
    """
    Return (loudness_lufs, clipping_detected) using ffmpeg astats.
    """
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i",
        audio_path,
        "-af",
        "astats=metadata=1:reset=1",
        "-f",
        "null",
        "-",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not stderr:
        return None, False

    text = stderr.decode(errors="ignore")
    rms_matches = re.findall(r"RMS level dB:\s*(-?[0-9]+(?:\.[0-9]+)?)", text)
    max_matches = re.findall(r"Max level dB:\s*(-?[0-9]+(?:\.[0-9]+)?)", text)

    loudness = float(rms_matches[-1]) if rms_matches else None
    clipping = False
    if max_matches:
        clipping = float(max_matches[-1]) >= -0.2
    return loudness, clipping


async def compose_preview_video(
    image_path: str,
    audio_path: str,
    output_path: str,
) -> str | None:
    """
    Create MP4 preview by looping image over audio duration.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        image_path,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        _ = stderr.decode(errors="ignore")
        return None
    return output_path
