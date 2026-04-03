from __future__ import annotations

import asyncio
import os

import edge_tts

from ..agents.state import EmotionLabel

VOICE_BY_LANG: dict[str, str] = {
    "en": "en-US-AriaNeural",
    "hi": "hi-IN-SwaraNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
}

PROSODY_BY_EMOTION: dict[EmotionLabel, tuple[str, str]] = {
    "happy": ("+10%", "+15Hz"),
    "sad": ("-12%", "-10Hz"),
    "angry": ("+18%", "+20Hz"),
    "calm": ("-8%", "-5Hz"),
    "neutral": ("+0%", "+0Hz"),
}


class EmotionalTTS:
    async def synthesize(
        self,
        text: str,
        emotion: EmotionLabel,
        target_language: str,
        output_path: str,
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not text.strip():
            text = "This is an emotionally neutral narration."
        voice = VOICE_BY_LANG.get(target_language, VOICE_BY_LANG["en"])
        rate, pitch = PROSODY_BY_EMOTION.get(emotion, PROSODY_BY_EMOTION["neutral"])
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
        await communicate.save(output_path)
        return output_path

    async def probe_duration(self, audio_path: str) -> float:
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
        try:
            return float(out.decode().strip())
        except Exception:
            return 0.0
