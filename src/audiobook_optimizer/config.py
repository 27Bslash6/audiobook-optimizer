"""Configuration using pydantic-settings."""

import os
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_prefix="AUDIOBOOK_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Directories
    source_dir: Path = Path("/home/fish/media/data/torrents/prowlarr")
    output_dir: Path = Path("/home/fish/media/data/media/books/audiobooks")

    # Conversion settings
    bitrate: int = 64  # kbps - 64 is fine for speech
    mono: bool = True  # Mono is recommended for audiobooks
    preserve_source: bool = True  # Keep source files after processing

    # FFmpeg paths
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    # AI settings
    ai_enabled: bool | None = None  # None = auto-detect from API key
    ai_backend: Literal["api", "cli"] = "api"  # "api" uses PydanticAI, "cli" uses claude CLI

    @field_validator("ai_enabled", mode="before")
    @classmethod
    def auto_detect_ai(cls, v):
        """Auto-enable AI if ANTHROPIC_API_KEY is set and ai_enabled not explicit."""
        if v is None:
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        return v


def get_settings() -> Settings:
    """Get settings instance."""
    return Settings()


def ai_available() -> bool:
    """Check if AI features are available (API key present)."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))
