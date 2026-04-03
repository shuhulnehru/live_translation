from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings


def _load_secrets() -> dict[str, str]:
    """Load secrets from YAML file. Path from SECRETS_FILE env or default secrets.yaml in project root."""
    path = os.environ.get("SECRETS_FILE")
    if not path:
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / "secrets.yaml"
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return {k: str(v) for k, v in (data or {}).items() if v}


class Settings(BaseSettings):
    app_name: str = "Video Dubbing Studio"
    debug: bool = True
    openai_api_key: str = ""
    default_source_lang: str = "auto"
    default_target_lang: str = "hi"

    # Speech emotion recognition (Hugging Face audio-classification)
    ser_enabled: bool = True
    ser_model_id: str = "superb/wav2vec2-base-superb-er"
    ser_device: int = -1  # -1 = CPU, 0 = first CUDA device

    # Optional external lip-sync model integration (e.g., Wav2Lip-style script)
    lipsync_enabled: bool = False
    lipsync_python_bin: str = "python3"
    lipsync_script_path: str = ""
    lipsync_checkpoint_path: str = ""
    lipsync_extra_args: str = ""

    # Face routing (multi-face detection/tracking)
    face_routing_python_bin: str = "python3"
    face_routing_script_path: str = ""
    face_routing_extra_args: str = ""

    # Face restoration (GFPGAN/CodeFormer wrapper)
    face_restore_python_bin: str = "python3"
    face_restore_script_path: str = ""
    face_restore_extra_args: str = ""

    class Config:
        env_file = None  # No .env for secrets; use secrets.yaml instead
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    secrets = _load_secrets()
    # Map YAML keys (snake_case) to Settings field names
    overrides = {}
    if "openai_api_key" in secrets:
        overrides["openai_api_key"] = secrets["openai_api_key"]
    for key in (
        "ser_enabled",
        "ser_model_id",
        "ser_device",
        "lipsync_enabled",
        "lipsync_python_bin",
        "lipsync_script_path",
        "lipsync_checkpoint_path",
        "lipsync_extra_args",
        "face_routing_python_bin",
        "face_routing_script_path",
        "face_routing_extra_args",
        "face_restore_python_bin",
        "face_restore_script_path",
        "face_restore_extra_args",
    ):
        if key in secrets:
            overrides[key] = secrets[key]
    return Settings(**overrides)
