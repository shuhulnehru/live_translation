from __future__ import annotations

import edge_tts

VOICE_MAP: dict[str, str] = {
    "hi": "hi-IN-SwaraNeural",
    "en": "en-US-AriaNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "bn": "bn-IN-TanishaaNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh-CN": "zh-CN-XiaoxiaoNeural",
}

DEFAULT_VOICE = "hi-IN-SwaraNeural"

# Estimated characters-per-second for natural speech by language.
# Used to predict TTS duration and compute rate adjustment.
CHARS_PER_SECOND: dict[str, float] = {
    "hi": 12.0,
    "en": 14.0,
    "ta": 10.0,
    "te": 10.0,
    "bn": 11.0,
    "es": 15.0,
    "fr": 15.0,
    "de": 14.0,
    "ja": 8.0,
    "ko": 9.0,
    "zh-CN": 5.0,
}


def _estimate_rate_pct(text: str, target_duration: float, lang: str) -> int:
    """
    Estimate what SSML rate percentage is needed so the TTS output
    fits within target_duration seconds.

    Returns a percentage relative to normal speed (100 = normal).
    """
    cps = CHARS_PER_SECOND.get(lang, 13.0)
    estimated_natural_duration = len(text) / cps

    if estimated_natural_duration <= 0 or target_duration <= 0:
        return 100

    ratio = estimated_natural_duration / target_duration
    rate_pct = int(ratio * 100)
    return max(50, min(200, rate_pct))


def _build_prosody(rate_pct: int, text: str) -> tuple[str, str]:
    """Return edge-tts rate/pitch strings."""
    pitch = "+0Hz"
    if text.rstrip().endswith("?"):
        pitch = "+10Hz"
    elif text.rstrip().endswith("!"):
        pitch = "+5Hz"

    sign = "+" if rate_pct >= 100 else ""
    rate = f"{sign}{rate_pct - 100}%"
    return rate, pitch


async def synthesize_speech(
    text: str,
    target_lang: str,
    output_path: str,
    target_duration: float | None = None,
) -> str:
    """
    Generate speech audio from translated text using Edge TTS.

    If target_duration is provided, adjusts speaking rate via SSML
    so the output roughly fits the original dialogue window.
    Single-pass approach (no re-generation).
    """
    if not text.strip():
        return ""

    voice = VOICE_MAP.get(target_lang, DEFAULT_VOICE)

    if target_duration and target_duration > 0:
        rate_pct = _estimate_rate_pct(text, target_duration, target_lang)
        if abs(rate_pct - 100) > 10:
            rate, pitch = _build_prosody(rate_pct, text)
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await communicate.save(output_path)
            return output_path

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path
