from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class SupportedSourceLang(str, Enum):
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE = "zh"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ENGLISH = "en"
    HINDI = "hi"
    AUTO = "auto"


class SupportedTargetLang(str, Enum):
    HINDI = "hi"
    ENGLISH = "en"
    TAMIL = "ta"
    TELUGU = "te"
    BENGALI = "bn"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE_SIMPLIFIED = "zh-CN"


class LanguagePair(BaseModel):
    source: SupportedSourceLang = SupportedSourceLang.AUTO
    target: SupportedTargetLang = SupportedTargetLang.HINDI


class ChunkStatus(str, Enum):
    EXTRACTING = "extracting"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    SYNTHESIZING = "synthesizing"
    DONE = "done"
    ERROR = "error"


class ChunkResult(BaseModel):
    chunk_index: int
    start_time: float
    end_time: float
    original_text: str = ""
    translated_text: str = ""
    status: ChunkStatus = ChunkStatus.EXTRACTING


class DubJobRequest(BaseModel):
    languages: LanguagePair = LanguagePair()


class WSMessage(BaseModel):
    """Messages sent over WebSocket to the client."""

    type: str
    data: dict | None = None
