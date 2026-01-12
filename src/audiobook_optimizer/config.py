"""Configuration using pydantic-settings."""

from pathlib import Path

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


def get_settings() -> Settings:
    """Get settings instance."""
    return Settings()
