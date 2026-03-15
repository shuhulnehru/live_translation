from __future__ import annotations

import asyncio
import base64
import math
import os

import ffmpeg


async def extract_keyframes(
    video_path: str,
    output_dir: str,
    fps: int = 1,
) -> dict[float, str]:
    """
    Extract JPEG keyframes from a video at the given fps rate.

    Returns a mapping of timestamp (seconds) -> file path.
    """
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    pattern = os.path.join(frames_dir, "frame_%05d.jpg")

    cmd = (
        ffmpeg.input(video_path)
        .filter("fps", fps=fps)
        .output(pattern, qscale=4)
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
        raise RuntimeError(f"Keyframe extraction failed: {stderr.decode()}")

    timestamp_map: dict[float, str] = {}
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    for i, fname in enumerate(frame_files):
        timestamp = float(i) / fps
        timestamp_map[timestamp] = os.path.join(frames_dir, fname)

    return timestamp_map


def get_closest_frame(
    timestamp_map: dict[float, str],
    target_time: float,
) -> str | None:
    """Return the frame path closest to target_time."""
    if not timestamp_map:
        return None
    closest_ts = min(timestamp_map.keys(), key=lambda t: abs(t - target_time))
    return timestamp_map[closest_ts]


def get_frame_for_segment(
    timestamp_map: dict[float, str],
    start: float,
    end: float,
) -> str | None:
    """Pick the frame closest to the midpoint of a speech segment."""
    mid = (start + end) / 2.0
    return get_closest_frame(timestamp_map, mid)


def encode_frame_base64(frame_path: str) -> str:
    """Read a JPEG file and return its base64-encoded string."""
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
