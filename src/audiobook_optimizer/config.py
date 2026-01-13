"""Configuration using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AISettings(BaseSettings):
    """AI settings - separate class to avoid AUDIOBOOK_ prefix on API key."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: SecretStr | None = None


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_prefix="AUDIOBOOK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
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

    @model_validator(mode="after")
    def _auto_detect_ai(self) -> "Settings":
        """Auto-enable AI if ANTHROPIC_API_KEY is set and ai_enabled not explicit."""
        if self.ai_enabled is None:
            self.ai_enabled = AISettings().anthropic_api_key is not None
        return self


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


@lru_cache
def get_ai_settings() -> AISettings:
    """Get cached AI settings instance."""
    return AISettings()


def ai_available() -> bool:
    """Check if AI features are available (API key present)."""
    return get_ai_settings().anthropic_api_key is not None
