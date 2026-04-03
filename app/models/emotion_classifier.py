from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agents.state import EmotionLabel
from ..config import get_settings

logger = logging.getLogger(__name__)

# Hugging Face id2label strings vary by checkpoint; normalize to our EmotionLabel set.
_SER_TO_EMOTION: dict[str, EmotionLabel] = {
    # SUPERB wav2vec2-base-superb-er
    "neu": "neutral",
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
    # Common full names (e.g. ehcalabres RAVDESS-style)
    "neutral": "neutral",
    "happy": "happy",
    "angry": "angry",
    "calm": "calm",
    "disgust": "angry",
    "fearful": "sad",
    "surprised": "happy",
    "fear": "sad",
    "surprise": "happy",
}

_ser_pipeline: Any = None
_ser_load_error: str | None = None
_ser_lock = threading.Lock()


@dataclass
class EmotionPrediction:
    label: EmotionLabel
    confidence: float
    source: str


def _normalize_ser_label(raw: str) -> EmotionLabel:
    key = raw.lower().strip()
    return _SER_TO_EMOTION.get(key, "neutral")


def _get_ser_pipeline() -> Any:
    """Lazy-load Hugging Face audio-classification pipeline (pretrained SER)."""
    global _ser_pipeline, _ser_load_error

    settings = get_settings()
    if not settings.ser_enabled:
        raise RuntimeError("SER is disabled in settings")

    with _ser_lock:
        if _ser_pipeline is not None:
            return _ser_pipeline
        if _ser_load_error is not None:
            raise RuntimeError(f"SER model load previously failed: {_ser_load_error}")
        try:
            from transformers import pipeline
        except ImportError as exc:
            _ser_load_error = str(exc)
            raise RuntimeError(f"SER requires torch/transformers: {exc}") from exc

        device = settings.ser_device

        try:
            _ser_pipeline = pipeline(
                "audio-classification",
                model=settings.ser_model_id,
                device=device,
            )
            logger.info(
                "Loaded pretrained SER model: %s (device=%s)",
                settings.ser_model_id,
                "cpu" if device < 0 else device,
            )
        except Exception as exc:
            _ser_load_error = str(exc)
            logger.exception("Failed to load SER model %s", settings.ser_model_id)
            raise RuntimeError(f"failed to load SER model: {settings.ser_model_id}") from exc

        return _ser_pipeline


def _run_ser_sync(audio_path: str) -> EmotionPrediction | None:
    pipe = _get_ser_pipeline()
    try:
        out = pipe(str(audio_path))
        if not out:
            raise RuntimeError("SER returned empty predictions")
        top = out[0]
        raw_label = str(top.get("label", ""))
        score = float(top.get("score", 0.0))
        label = _normalize_ser_label(raw_label)
        confidence = min(0.95, max(0.0, score))
        return EmotionPrediction(label=label, confidence=confidence, source="audio")
    except Exception as exc:
        logger.exception("SER inference failed for %s", audio_path)
        raise RuntimeError(f"SER inference failed for {audio_path}") from exc


class EmotionClassifier:
    """
    Strict inference wrapper using pretrained Hugging Face SER on audio.
    No fallback heuristics are used.
    """

    async def predict(self, prompt_text: str, audio_path: str | None = None) -> EmotionPrediction:
        _ = prompt_text  # retained for interface compatibility
        if not audio_path:
            raise RuntimeError("audio_path is required for strict SER prediction")
        audio_pred = await self._predict_from_audio(audio_path)
        if audio_pred is None:
            raise RuntimeError("SER produced no prediction")
        return audio_pred

    async def _predict_from_audio(self, audio_path: str) -> EmotionPrediction | None:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"audio input not found: {audio_path}")

        ser_pred = await asyncio.to_thread(_run_ser_sync, str(path))
        return ser_pred
