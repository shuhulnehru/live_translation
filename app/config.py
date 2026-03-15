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
    return Settings(**overrides)
