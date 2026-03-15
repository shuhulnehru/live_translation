from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from ..config import get_settings


@dataclass
class SpeechSegment:
    """A single speech segment with precise start/end from Whisper."""

    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionResult:
    segments: list[SpeechSegment] = field(default_factory=list)
    language: str = ""


def _get_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


async def transcribe_audio(
    audio_path: str,
    language: str | None = None,
) -> TranscriptionResult:
    """
    Transcribe audio using the OpenAI Whisper API.

    Returns per-segment results with timestamps. Much faster than local
    Whisper since it runs on OpenAI's infrastructure.
    """
    lang_arg = None if language in (None, "auto") else language

    def _call_api() -> TranscriptionResult:
        client = _get_client()

        with open(audio_path, "rb") as audio_file:
            kwargs: dict = {
                "model": "whisper-1",
                "file": audio_file,
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
            }
            if lang_arg:
                kwargs["language"] = lang_arg

            response = client.audio.transcriptions.create(**kwargs)

        segments: list[SpeechSegment] = []
        detected_lang = getattr(response, "language", "") or ""

        raw_segments = getattr(response, "segments", []) or []
        for idx, seg in enumerate(raw_segments):
            text = seg.get("text", "").strip() if isinstance(seg, dict) else getattr(seg, "text", "").strip()
            start = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
            end = seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0)

            if not text:
                continue

            segments.append(SpeechSegment(
                index=idx,
                start=float(start),
                end=float(end),
                text=text,
            ))

        return TranscriptionResult(segments=segments, language=detected_lang)

    return await asyncio.to_thread(_call_api)
