from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from openai import OpenAI

from ..config import get_settings
from .frame_extractor import encode_frame_base64, get_frame_for_segment
from .transcriber import SpeechSegment

logger = logging.getLogger(__name__)

BATCH_SIZE = 10


@dataclass
class TranslatedSegment:
    index: int
    start: float
    end: float
    original_text: str
    translated_text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def _get_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


SYSTEM_PROMPT = """You are a professional dubbing translator. You will receive dialogue segments from a video along with a screenshot of the scene for each segment.

Your task:
1. Translate each dialogue segment to {target_lang}.
2. Look at each scene screenshot to understand the visual context -- character emotions, actions, and setting.
3. Preserve the emotion and tone of the original dialogue. If a character is shouting, the translation should convey shouting. If they are whispering or sad, reflect that.
4. Keep the translated text roughly the same length as the original so it fits the same time window when spoken aloud.
5. Return ONLY a JSON array of objects, one per segment, each with:
   - "index": the segment index (integer)
   - "translated_text": the translated dialogue (string)

Example output:
[
  {{"index": 0, "translated_text": "translated text here"}},
  {{"index": 1, "translated_text": "translated text here"}}
]

Do NOT include any explanation or markdown formatting. Return ONLY the JSON array."""


async def translate_segments(
    segments: list[SpeechSegment],
    keyframe_map: dict[float, str],
    source_lang: str,
    target_lang: str,
) -> list[TranslatedSegment]:
    """
    Translate speech segments using GPT-4o with visual context from keyframes.
    Processes in batches of BATCH_SIZE to stay within token limits.
    """
    results: list[TranslatedSegment] = []

    for i in range(0, len(segments), BATCH_SIZE):
        batch = segments[i : i + BATCH_SIZE]
        batch_results = await _translate_batch(
            batch, keyframe_map, source_lang, target_lang
        )
        results.extend(batch_results)

    return results


async def _translate_batch(
    segments: list[SpeechSegment],
    keyframe_map: dict[float, str],
    source_lang: str,
    target_lang: str,
) -> list[TranslatedSegment]:
    """Send a batch of segments + their keyframes to GPT-4o for translation."""

    def _call_api() -> list[TranslatedSegment]:
        client = _get_client()

        user_content: list[dict] = []

        for seg in segments:
            user_content.append({
                "type": "text",
                "text": (
                    f"[Segment {seg.index}] "
                    f"({seg.start:.1f}s - {seg.end:.1f}s): "
                    f'"{seg.text}"'
                ),
            })

            frame_path = get_frame_for_segment(keyframe_map, seg.start, seg.end)
            if frame_path:
                b64 = encode_frame_base64(frame_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "low",
                    },
                })

        lang_names = {
            "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu",
            "bn": "Bengali", "es": "Spanish", "fr": "French", "de": "German",
            "ja": "Japanese", "ko": "Korean", "zh-CN": "Chinese (Simplified)",
        }
        target_name = lang_names.get(target_lang, target_lang)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(target_lang=target_name),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=0.3,
            max_tokens=4096,
        )

        raw_text = response.choices[0].message.content or "[]"
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            translations = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error("GPT-4o returned invalid JSON: %s", raw_text[:500])
            translations = []

        trans_map = {t["index"]: t["translated_text"] for t in translations}

        results: list[TranslatedSegment] = []
        for seg in segments:
            results.append(TranslatedSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                original_text=seg.text,
                translated_text=trans_map.get(seg.index, seg.text),
            ))

        return results

    return await asyncio.to_thread(_call_api)
