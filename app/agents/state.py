from __future__ import annotations

from typing import Any, TypedDict


class Segment(TypedDict):
    id: int
    start_time: float
    end_time: float
    duration: float
    speaker_id: str
    original_text: str


class TranslatedSegment(TypedDict):
    id: int
    original_text: str
    translated_text: str
    char_count: int
    fits: bool
    cps: float


class AudioClip(TypedDict):
    id: int
    file_path: str
    duration: float
    start_time: float
    adjusted: bool


class DubbingState(TypedDict, total=False):
    # Canonical schema requested
    file_path: str
    target_language: str
    source_language: str
    has_subtitles: bool
    segments: list[Segment]
    translated_segments: list[TranslatedSegment]
    voice_registry: dict[str, str]
    audio_clips: list[AudioClip]
    output_path: str
    errors: list[str]

    # Internal runtime fields
    work_dir: str
    output_dir: str
    audio_path: str
    keyframe_map: dict[float, str]
    original_video_duration: float
    subtitles_path: str | None
    on_progress: Any
