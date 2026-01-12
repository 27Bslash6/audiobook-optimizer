"""Domain models for audiobook processing.

These are pure data classes with no infrastructure dependencies.
The domain represents the business logic of audiobook management.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ProcessingStatus(Enum):
    """Status of audiobook processing."""

    PENDING = "pending"
    SCANNING = "scanning"
    VALIDATING = "validating"
    CONVERTING = "converting"
    TAGGING = "tagging"
    ORGANIZING = "organizing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AudioFormat(Enum):
    """Supported audio formats."""

    MP3 = "mp3"
    M4A = "m4a"
    M4B = "m4b"
    FLAC = "flac"
    OGG = "ogg"
    OPUS = "opus"
    WAV = "wav"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, ext: str) -> "AudioFormat":
        """Get format from file extension."""
        ext = ext.lower().lstrip(".")
        return cls._value2member_map_.get(ext, cls.UNKNOWN)  # type: ignore[return-value]


@dataclass
class Chapter:
    """A chapter marker in an audiobook."""

    title: str
    start_ms: int
    end_ms: int | None = None

    @property
    def duration_ms(self) -> int | None:
        """Duration in milliseconds."""
        if self.end_ms is None:
            return None
        return self.end_ms - self.start_ms


@dataclass
class AudioFile:
    """A single audio file with metadata."""

    path: Path
    format: AudioFormat
    duration_ms: int
    bitrate: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    title: str | None = None
    track_number: int | None = None
    disc_number: int | None = None
    chapters: list[Chapter] = field(default_factory=list)


@dataclass
class AudiobookMetadata:
    """Metadata for an audiobook."""

    title: str
    author: str
    series: str | None = None
    series_number: float | None = None
    narrator: str | None = None
    year: int | None = None
    description: str | None = None
    genre: str = "Audiobook"
    cover_path: Path | None = None

    @property
    def display_name(self) -> str:
        """Human-readable name for the audiobook."""
        if self.series and self.series_number is not None:
            # Format: "Series 01 - Title" or "Series 1.5 - Title"
            num = int(self.series_number) if self.series_number == int(self.series_number) else self.series_number
            return f"{self.series} {num:02d} - {self.title}" if isinstance(num, int) else f"{self.series} {num} - {self.title}"
        return self.title

    @property
    def folder_name(self) -> str:
        """Sanitized folder name for organization."""
        name = self.display_name
        # Remove/replace problematic characters
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            name = name.replace(char, '-')
        return name.strip()


@dataclass
class AudiobookSource:
    """A discovered audiobook source in the prowlarr directory."""

    source_path: Path
    audio_files: list[AudioFile]
    metadata: AudiobookMetadata | None = None
    total_duration_ms: int = 0
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: str | None = None

    @property
    def file_count(self) -> int:
        """Number of audio files."""
        return len(self.audio_files)

    @property
    def total_size_bytes(self) -> int:
        """Total size of all audio files."""
        return sum(f.path.stat().st_size for f in self.audio_files if f.path.exists())

    @property
    def primary_format(self) -> AudioFormat:
        """Most common format among files."""
        if not self.audio_files:
            return AudioFormat.UNKNOWN
        formats = [f.format for f in self.audio_files]
        return max(set(formats), key=formats.count)


@dataclass
class AIDecisionInfo:
    """Summary of AI decision for display."""

    action: str  # remux, transcode, skip
    target_bitrate_kbps: int
    preserve_stereo: bool
    quality_warnings: list[str]
    reasoning: str
    confidence: float


@dataclass
class ProcessingResult:
    """Result of processing an audiobook."""

    source: AudiobookSource
    output_path: Path | None = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: str | None = None
    chapters_preserved: int = 0
    duration_ms: int = 0
    was_remuxed: bool = False  # True if stream copy was used (no re-encoding)
    ai_decision: AIDecisionInfo | None = None  # AI advisor decision if used

    @property
    def success(self) -> bool:
        """Whether processing completed successfully."""
        return self.status == ProcessingStatus.COMPLETED and self.output_path is not None
