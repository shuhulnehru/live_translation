from __future__ import annotations

import asyncio
import os
import shutil


async def probe_audio_duration(audio_path: str) -> float:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {audio_path}")
    return float(out.decode().strip())


def _atempo_chain(ratio: float) -> str:
    # ffmpeg atempo accepts 0.5 to 2.0 per stage
    stages: list[float] = []
    x = ratio
    while x > 2.0:
        stages.append(2.0)
        x /= 2.0
    while x < 0.5:
        stages.append(0.5)
        x /= 0.5
    stages.append(x)
    return ",".join(f"atempo={s:.6f}" for s in stages)


async def align_audio_duration_strict(
    input_audio_path: str,
    target_duration: float,
    output_audio_path: str,
    tolerance: float = 0.12,
) -> tuple[str, float, bool]:
    if target_duration <= 0:
        raise ValueError("target_duration must be > 0")
    current_duration = await probe_audio_duration(input_audio_path)
    if current_duration <= 0:
        raise ValueError("input audio duration must be > 0")

    ratio = current_duration / target_duration
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

    if abs(ratio - 1.0) <= tolerance:
        shutil.copy2(input_audio_path, output_audio_path)
        return output_audio_path, ratio, False

    # We need playback speed = current / target to hit target duration.
    atempo_filter = _atempo_chain(ratio)
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        input_audio_path,
        "-filter:a",
        atempo_filter,
        output_audio_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not os.path.exists(output_audio_path):
        msg = stderr.decode(errors="ignore")
        raise RuntimeError(f"audio timing alignment failed: {msg}")
    return output_audio_path, ratio, True
