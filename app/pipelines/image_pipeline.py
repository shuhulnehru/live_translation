from __future__ import annotations

import html
import os
from pathlib import Path

from ..agents.state import EmotionLabel

THEME_BY_EMOTION: dict[EmotionLabel, tuple[str, str, list[str]]] = {
    "happy": ("#FDE68A", "#92400E", ["warm", "bright", "uplifting"]),
    "sad": ("#BFDBFE", "#1E3A8A", ["cool", "low-contrast", "soft"]),
    "angry": ("#FCA5A5", "#7F1D1D", ["high-contrast", "bold", "intense"]),
    "calm": ("#BBF7D0", "#14532D", ["balanced", "soft", "minimal"]),
    "neutral": ("#E5E7EB", "#111827", ["clean", "simple", "neutral"]),
}


async def generate_emotion_image(
    output_dir: str,
    request_id: str,
    emotion: EmotionLabel,
    prompt_text: str,
) -> dict:
    """
    Create a lightweight SVG card for very low-latency image output.
    """
    os.makedirs(output_dir, exist_ok=True)
    bg, fg, tags = THEME_BY_EMOTION.get(emotion, THEME_BY_EMOTION["neutral"])
    title = f"{emotion.upper()} MODE"
    subtitle = prompt_text.strip() or "Emotion-aware generated frame"
    subtitle = html.escape(subtitle[:96])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <rect width="1280" height="720" fill="{bg}" />
  <rect x="60" y="60" width="1160" height="600" rx="24" fill="white" opacity="0.65" />
  <text x="100" y="190" fill="{fg}" font-family="Arial, sans-serif" font-size="72" font-weight="700">{title}</text>
  <text x="100" y="270" fill="{fg}" font-family="Arial, sans-serif" font-size="36">{subtitle}</text>
  <text x="100" y="620" fill="{fg}" font-family="Arial, sans-serif" font-size="28">Emotion-aware media pipeline</text>
</svg>
"""
    image_path = Path(output_dir) / f"{request_id}_frame.svg"
    image_path.write_text(svg, encoding="utf-8")

    return {
        "path": str(image_path),
        "mime_type": "image/svg+xml",
        "width": 1280,
        "height": 720,
        "style_tags": tags,
    }
