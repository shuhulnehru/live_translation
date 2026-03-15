from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TTSClip:
    """A TTS audio clip with its placement in the timeline."""

    path: str
    start_time: float  # seconds offset in the video


async def build_dubbed_audio(
    clips: list[TTSClip],
    total_duration: float,
    output_path: str,
) -> str:
    """
    Build a single audio track by placing each TTS clip at its correct
    timestamp offset using FFmpeg adelay + amix filters.

    Creates a silent base track of total_duration, then overlays every
    TTS clip at its start_time.
    """
    if not clips:
        raise ValueError("No TTS clips to assemble")

    valid_clips = [c for c in clips if os.path.exists(c.path) and c.path]
    if not valid_clips:
        raise ValueError("No valid TTS clip files found")

    # Build the FFmpeg command using filter_complex.
    # Strategy: create a silent audio base, then for each clip add an
    # adelay filter and amix them all together.
    inputs = ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono:d={total_duration}"]

    for clip in valid_clips:
        inputs.extend(["-i", clip.path])

    filter_parts: list[str] = []
    amix_inputs: list[str] = ["[0:a]"]

    for i, clip in enumerate(valid_clips):
        input_idx = i + 1
        delay_ms = int(clip.start_time * 1000)
        filter_parts.append(
            f"[{input_idx}:a]adelay={delay_ms}|{delay_ms},apad[d{i}]"
        )
        amix_inputs.append(f"[d{i}]")

    n_inputs = len(amix_inputs)
    amix_str = "".join(amix_inputs)
    filter_parts.append(
        f"{amix_str}amix=inputs={n_inputs}:duration=first:dropout_transition=0[out]"
    )

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ac", "2",
        "-ar", "44100",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Audio muxing failed: {stderr.decode()}")

    return output_path


async def build_dubbed_video(
    original_video: str,
    dubbed_audio: str,
    output_path: str,
) -> str:
    """
    Replace the original video's audio track with the dubbed audio,
    producing the final MP4.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", original_video,
        "-i", dubbed_audio,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Video assembly failed: {stderr.decode()}")

    return output_path


async def get_video_duration(video_path: str) -> float:
    """Get the duration of a video file in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    return float(stdout.decode().strip())
