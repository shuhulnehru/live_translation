from __future__ import annotations

import asyncio
import os
import tempfile

import ffmpeg


async def extract_full_audio(video_path: str, output_dir: str) -> str:
    """Extract the full audio track from a video file as 16 kHz mono WAV."""
    audio_path = os.path.join(output_dir, "full_audio.wav")
    cmd = (
        ffmpeg.input(video_path)
        .output(audio_path, ac=1, ar=16000, format="wav")
        .overwrite_output()
        .compile()
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {stderr.decode()}")
    return audio_path


def create_temp_dir() -> str:
    """Create a temporary working directory for a dubbing job."""
    return tempfile.mkdtemp(prefix="dubbing_")
