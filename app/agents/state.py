from __future__ import annotations

from typing import Any, Literal, TypedDict


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


EmotionLabel = Literal["happy", "sad", "angry", "calm", "neutral"]


class EmotionDetection(TypedDict):
    label: EmotionLabel
    confidence: float
    source: Literal["text", "audio", "multimodal", "fallback"]


class EmotionAudioAsset(TypedDict):
    path: str
    duration_seconds: float
    sample_rate: int
    loudness_lufs: float | None
    clipping_detected: bool


class EmotionImageAsset(TypedDict):
    path: str
    mime_type: str
    width: int
    height: int
    style_tags: list[str]


class EmotionOutput(TypedDict):
    emotion: EmotionLabel
    confidence: float
    image_path: str
    audio_path: str
    preview_video_path: str | None
    quality_passed: bool
    fallback_used: bool


class EmotionMediaState(TypedDict, total=False):
    request_id: str
    prompt_text: str
    audio_input_path: str | None
    target_language: str
    output_dir: str

    detection: EmotionDetection
    image_asset: EmotionImageAsset
    audio_asset: EmotionAudioAsset

    min_confidence: float
    min_duration_seconds: float
    max_duration_seconds: float
    quality_notes: list[str]
    fallback_used: bool
    route: str

    output: EmotionOutput


class EmotionVideoOutput(TypedDict):
    emotion: EmotionLabel
    confidence: float
    transcript_text: str
    translated_text: str
    dialogue_audio_path: str
    aligned_audio_path: str
    corrected_video_path: str
    face_routing_metadata_path: str
    lipsynced_video_path: str
    restored_video_path: str
    refined_video_path: str
    final_video_path: str


class EmotionVideoState(TypedDict, total=False):
    request_id: str
    file_path: str
    prompt_text: str
    source_language: str
    target_language: str
    output_dir: str
    work_dir: str

    audio_input_path: str
    detection: EmotionDetection
    transcript_text: str
    translated_text: str
    dialogue_audio_path: str
    source_dialogue_duration: float
    generated_dialogue_duration: float
    aligned_audio_path: str
    timing_ratio: float
    timing_stretch_applied: bool
    corrected_video_path: str
    face_routing_metadata_path: str
    lipsynced_video_path: str
    restored_video_path: str
    refined_video_path: str
    language_style_note: str
    errors: list[str]

    output: EmotionVideoOutput
