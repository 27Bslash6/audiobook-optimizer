"""Port interfaces (driven/driving adapters contracts).

Ports define what the domain needs from the outside world.
Adapters implement these interfaces with concrete technology.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from audiobook_optimizer.domain.models import (
    AudiobookMetadata,
    AudiobookSource,
    AudioFile,
    Chapter,
    ProcessingResult,
)


class AudioScanner(ABC):
    """Port for discovering audio files in a directory."""

    @abstractmethod
    def scan_directory(self, path: Path) -> Iterator[AudiobookSource]:
        """Scan directory for audiobook sources.

        Yields AudiobookSource for each discovered audiobook folder.
        Does not extract full metadata - just discovers structure.
        """

    @abstractmethod
    def is_audiobook_directory(self, path: Path) -> bool:
        """Check if directory appears to contain audiobook content."""


class MetadataExtractor(ABC):
    """Port for extracting metadata from audio files."""

    @abstractmethod
    def extract_file_info(self, path: Path) -> AudioFile:
        """Extract audio file information (duration, format, embedded metadata)."""

    @abstractmethod
    def extract_chapters(self, path: Path) -> list[Chapter]:
        """Extract chapter information from audio file if present."""

    @abstractmethod
    def extract_cover(self, path: Path) -> bytes | None:
        """Extract embedded cover art from audio file."""

    @abstractmethod
    def infer_metadata(self, source: AudiobookSource) -> AudiobookMetadata:
        """Infer audiobook metadata from folder name, filenames, and embedded tags."""


class AudioConverter(ABC):
    """Port for converting audio files to M4B."""

    @abstractmethod
    def convert_to_m4b(
        self,
        source_files: list[AudioFile],
        output_path: Path,
        chapters: list[Chapter] | None = None,
        bitrate: int = 64,
        mono: bool = True,
    ) -> tuple[Path, bool]:
        """Convert multiple audio files to single M4B.

        Args:
            source_files: Ordered list of audio files to concatenate
            output_path: Where to write the M4B file
            chapters: Chapter markers (if None, creates one chapter per source file)
            bitrate: Target bitrate in kbps (default 64 for audiobooks)
            mono: Convert to mono (recommended for speech)

        Returns:
            Tuple of (path to created M4B, was_remuxed flag)

        Raises:
            ConversionError: If conversion fails
        """

    @abstractmethod
    def probe_duration(self, path: Path) -> int:
        """Get duration in milliseconds using ffprobe."""


class AudioTagger(ABC):
    """Port for writing metadata and cover art to M4B files."""

    @abstractmethod
    def apply_metadata(self, path: Path, metadata: AudiobookMetadata) -> None:
        """Write metadata tags to M4B file."""

    @abstractmethod
    def embed_cover(self, path: Path, cover_data: bytes, mime_type: str = "image/jpeg") -> None:
        """Embed cover art in M4B file."""

    @abstractmethod
    def write_chapters(self, path: Path, chapters: list[Chapter]) -> None:
        """Write chapter markers to M4B file."""


class FileOrganizer(ABC):
    """Port for organizing processed audiobooks into library structure."""

    @abstractmethod
    def organize(self, source: Path, metadata: AudiobookMetadata, library_root: Path) -> Path:
        """Move/copy processed audiobook to organized library location.

        Returns the final destination path.
        """

    @abstractmethod
    def compute_destination(self, metadata: AudiobookMetadata, library_root: Path) -> Path:
        """Compute where an audiobook would be organized without moving it."""


class ProcessingPipeline(ABC):
    """Port for the full processing pipeline."""

    @abstractmethod
    def process(self, source: AudiobookSource) -> ProcessingResult:
        """Process a single audiobook through the full pipeline."""

    @abstractmethod
    def process_batch(self, sources: list[AudiobookSource]) -> list[ProcessingResult]:
        """Process multiple audiobooks."""
