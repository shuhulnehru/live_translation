"""Tools used by the dubbing agents: Diarization, Validation. Duration Fit is built into TTS."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI

from ..config import get_settings
from ..services.transcriber import SpeechSegment


@dataclass
class SegmentWithSpeaker(SpeechSegment):
    """Speech segment with optional speaker label from diarization."""

    speaker_id: str = "SPEAKER_0"


def diarization_tool(segments: list[SpeechSegment]) -> list[SegmentWithSpeaker]:
    """
    Add speaker labels to transcribed segments (diarization).
    Uses heuristics when pyannote is not available.
    """
    result: list[SegmentWithSpeaker] = []
    speaker = "SPEAKER_0"
    prev_end = 0.0

    for seg in segments:
        # Alternate speaker on long pauses
        if seg.start - prev_end > 1.5:
            speaker = "SPEAKER_1" if speaker == "SPEAKER_0" else "SPEAKER_0"
        prev_end = seg.end
        result.append(SegmentWithSpeaker(
            index=seg.index,
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker_id=speaker,
        ))

    return result


def validation_tool(original_text: str, translated_text: str, target_lang: str) -> str:
    """
    Validate and optionally improve a translation.
    Returns the validated (possibly improved) translated text.
    """
    settings = get_settings()
    if not settings.openai_api_key or not translated_text.strip():
        return translated_text

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=settings.openai_api_key)
    prompt = f"""You are a translation validator. Given:
- Original: "{original_text}"
- Translation to {target_lang}: "{translated_text}"

If the translation is accurate and natural, return it unchanged.
If there are grammar or fluency issues, return an improved version.
Output ONLY the final translated text, no explanation."""

    response = llm.invoke(prompt)
    return (response.content or translated_text).strip()


apply_diarization = diarization_tool
apply_validation = validation_tool
