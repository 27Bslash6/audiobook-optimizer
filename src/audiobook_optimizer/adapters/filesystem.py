"""Filesystem adapter for scanning and organizing audiobooks."""

import re
import shutil
from pathlib import Path
from typing import Iterator

from audiobook_optimizer.domain.models import (
    AudiobookMetadata,
    AudiobookSource,
    AudioFile,
    AudioFormat,
    Chapter,
)
from audiobook_optimizer.ports.interfaces import AudioScanner, FileOrganizer, MetadataExtractor

# Audio extensions we care about
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".m4b", ".flac", ".ogg", ".opus", ".wav"}

# Patterns that indicate NOT an audiobook (use word boundaries to avoid false positives)
SKIP_PATTERNS = [
    re.compile(r"\bEPUB\b", re.IGNORECASE),
    re.compile(r"\bebook\b", re.IGNORECASE),
    re.compile(r"\bEmulator\b", re.IGNORECASE),
    re.compile(r"\b(?:PCSX2?|PSX|N64|SNES|NES|Bios)\b", re.IGNORECASE),
    re.compile(r"\b(?:Soundtrack|OST|Music|Album)\b|MP3.*\d{3}k", re.IGNORECASE),
]

# Patterns for inferring metadata from folder/file names
SERIES_PATTERNS = [
    # "Discworld 01 - The Colour of Magic"
    re.compile(r"^(?P<series>.+?)\s+(?P<num>\d+(?:\.\d+)?)\s*[-–]\s*(?P<title>.+)$"),
    # "01 - The Colour of Magic"
    re.compile(r"^(?P<num>\d+(?:\.\d+)?)\s*[-–]\s*(?P<title>.+)$"),
    # "Book 1 - Title"
    re.compile(r"^(?:Book|Vol\.?|Volume)\s*(?P<num>\d+(?:\.\d+)?)\s*[-–:]\s*(?P<title>.+)$", re.IGNORECASE),
]

AUTHOR_PATTERNS = [
    # "Author - Title"
    re.compile(r"^(?P<author>[^-]+?)\s*[-–]\s*(?P<title>.+)$"),
    # "Title by Author"
    re.compile(r"^(?P<title>.+?)\s+by\s+(?P<author>.+)$", re.IGNORECASE),
    # "Author, Name - Title" (for "Parenti, Michael" style)
    re.compile(r"^(?P<author>[A-Z][a-z]+,\s*[A-Z][a-z]+)\s*[-–]\s*(?P<title>.+)$"),
]


class FilesystemScanner(AudioScanner):
    """Scan filesystem for audiobook sources."""

    def __init__(self, min_files: int = 1, min_duration_ms: int = 60000):
        """
        Args:
            min_files: Minimum audio files to consider a valid audiobook
            min_duration_ms: Minimum total duration (default 1 minute)
        """
        self.min_files = min_files
        self.min_duration_ms = min_duration_ms

    def scan_directory(self, path: Path) -> Iterator[AudiobookSource]:
        """Scan directory for audiobook sources.

        Handles both:
        - Directories containing audio files (multi-file audiobooks)
        - Standalone audio files at the scan root (single-file audiobooks)
        """
        if not path.is_dir():
            return

        # If THIS directory is an audiobook, yield it as a unit
        # (don't split into individual files)
        if self.is_audiobook_directory(path):
            audio_files = self._find_audio_files(path)
            if audio_files:
                yield AudiobookSource(
                    source_path=path,
                    audio_files=audio_files,
                )
            return  # Don't descend further

        # This is a "root" directory (like prowlarr) - scan contents
        # First, yield standalone audio files as individual audiobooks
        for item in sorted(path.iterdir()):
            if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS:
                yield AudiobookSource(
                    source_path=path,  # Use parent dir for context
                    audio_files=[AudioFile(
                        path=item,
                        format=AudioFormat.from_extension(item.suffix),
                        duration_ms=0,
                    )],
                )

        # Then, scan subdirectories
        for child in sorted(path.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                if self._should_skip(child):
                    continue
                if self.is_audiobook_directory(child):
                    audio_files = self._find_audio_files(child)
                    if audio_files:
                        yield AudiobookSource(
                            source_path=child,
                            audio_files=audio_files,
                        )
                else:
                    # Recurse into non-audiobook directories
                    yield from self.scan_directory(child)

    def _should_skip(self, path: Path) -> bool:
        """Check if directory should be skipped (not audiobook content)."""
        name = path.name
        return any(pattern.search(name) for pattern in SKIP_PATTERNS)

    def is_audiobook_directory(self, path: Path) -> bool:
        """Check if directory contains audiobook content."""
        if not path.is_dir():
            return False

        # Skip directories that match exclusion patterns
        if self._should_skip(path):
            return False

        audio_files = list(self._iter_audio_files(path, recursive=False))
        if len(audio_files) >= self.min_files:
            return True

        # Check for audiobook indicators in folder name
        indicators = ["audiobook", "audio book", "narrated", "unabridged", "abridged"]
        folder_lower = path.name.lower()
        return any(ind in folder_lower for ind in indicators)

    def _iter_audio_files(self, path: Path, recursive: bool = True) -> Iterator[Path]:
        """Iterate over audio files in directory."""
        pattern = "**/*" if recursive else "*"
        for file in path.glob(pattern):
            if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:
                yield file

    def _find_audio_files(self, path: Path) -> list[AudioFile]:
        """Find and create AudioFile objects for all audio in directory."""
        files = []
        for audio_path in sorted(self._iter_audio_files(path)):
            files.append(AudioFile(
                path=audio_path,
                format=AudioFormat.from_extension(audio_path.suffix),
                duration_ms=0,  # Will be populated by metadata extractor
            ))
        return files


class FilesystemMetadataExtractor(MetadataExtractor):
    """Extract metadata from filesystem and embedded tags."""

    def __init__(self):
        # Import mutagen lazily to avoid import errors if not installed
        try:
            import mutagen
            self._mutagen = mutagen
        except ImportError:
            self._mutagen = None

    def extract_file_info(self, path: Path) -> AudioFile:
        """Extract audio file information."""
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        format_ = AudioFormat.from_extension(path.suffix)
        duration_ms = 0
        bitrate = None
        sample_rate = None
        channels = None
        title = None
        track_number = None

        if self._mutagen:
            try:
                audio = self._mutagen.File(path)
                if audio:
                    duration_ms = int(audio.info.length * 1000) if hasattr(audio.info, "length") else 0
                    raw_bitrate = getattr(audio.info, "bitrate", None)
                    # Mutagen returns bitrate in bps for some formats, kbps for others
                    # Normalize to kbps (reasonable audiobook bitrates are 32-320 kbps)
                    if raw_bitrate and raw_bitrate > 1000:
                        bitrate = raw_bitrate // 1000
                    else:
                        bitrate = raw_bitrate
                    sample_rate = getattr(audio.info, "sample_rate", None)
                    channels = getattr(audio.info, "channels", None)

                    # Try to get title from tags
                    if hasattr(audio, "tags") and audio.tags:
                        title = self._get_tag(audio.tags, ["TIT2", "title", "\xa9nam"])
                        track = self._get_tag(audio.tags, ["TRCK", "tracknumber", "trkn"])
                        if track:
                            # Handle "1/10" format
                            track_number = int(str(track).split("/")[0])
            except Exception:
                pass  # Fall back to defaults

        return AudioFile(
            path=path,
            format=format_,
            duration_ms=duration_ms,
            bitrate=bitrate,
            sample_rate=sample_rate,
            channels=channels,
            title=title,
            track_number=track_number,
        )

    def _get_tag(self, tags, keys: list[str]) -> str | None:
        """Get first matching tag from multiple possible keys."""
        for key in keys:
            if key in tags:
                value = tags[key]
                # Handle mutagen's list-wrapped values
                if isinstance(value, list):
                    value = value[0] if value else None
                return str(value) if value else None
        return None

    def extract_chapters(self, path: Path) -> list[Chapter]:
        """Extract chapter information from audio file."""
        # Most chapter info needs ffprobe - delegate to ffmpeg adapter
        return []

    def extract_cover(self, path: Path) -> bytes | None:
        """Extract embedded cover art from audio file."""
        if not self._mutagen:
            return None

        try:
            audio = self._mutagen.File(path)
            if not audio or not hasattr(audio, "tags") or not audio.tags:
                return None

            # MP3 (ID3)
            for key in audio.tags:
                if key.startswith("APIC"):
                    return audio.tags[key].data

            # MP4/M4A/M4B
            if "covr" in audio.tags:
                covers = audio.tags["covr"]
                if covers:
                    return bytes(covers[0])

            # FLAC
            if hasattr(audio, "pictures") and audio.pictures:
                return audio.pictures[0].data

        except Exception:
            pass

        return None

    def infer_metadata(self, source: AudiobookSource) -> AudiobookMetadata:
        """Infer audiobook metadata from folder name, filename, and embedded tags."""
        # For single-file audiobooks, use the filename (without extension)
        if len(source.audio_files) == 1:
            name_source = source.audio_files[0].path.stem
        else:
            name_source = source.source_path.name

        # Clean up common filename/folder patterns
        name_source = self._clean_name(name_source)

        # Try series patterns first
        for pattern in SERIES_PATTERNS:
            match = pattern.match(name_source)
            if match:
                groups = match.groupdict()
                return AudiobookMetadata(
                    title=groups.get("title", name_source).strip(),
                    author=self._infer_author_from_files(source) or "Unknown Author",
                    series=groups.get("series", "").strip() or None,
                    series_number=float(groups["num"]) if "num" in groups else None,
                )

        # Try author patterns
        for pattern in AUTHOR_PATTERNS:
            match = pattern.match(name_source)
            if match:
                groups = match.groupdict()
                return AudiobookMetadata(
                    title=groups.get("title", name_source).strip(),
                    author=groups.get("author", "Unknown Author").strip(),
                )

        # Fallback: use name as title, try to get author from files
        return AudiobookMetadata(
            title=name_source,
            author=self._infer_author_from_files(source) or "Unknown Author",
        )

    def _clean_name(self, name: str) -> str:
        """Clean up common filename/folder patterns."""
        # Iteratively clean until no more changes (handles nested patterns)
        prev = None
        while prev != name:
            prev = name
            # Remove "by dessalines" or similar uploader tags at end
            name = re.sub(r"\s+by\s+\w+$", "", name, flags=re.IGNORECASE)
            # Remove audiobook indicators
            suffixes = [" Audiobook", " [audiobook]", " (Audiobook)", " - audiobook", " audiobook"]
            for suffix in suffixes:
                if name.lower().endswith(suffix.lower()):
                    name = name[:-len(suffix)]
            # Remove quality/format tags like "(Stevens) 32k 12.58.21 {179mb}"
            name = re.sub(r"\s*\([^)]+\)\s*\d+k\s*[\d.]+\s*\{[^}]+\}$", "", name)
            # Remove year prefix like "2008 - "
            name = re.sub(r"^\d{4}\s*[-–]\s*", "", name)
        return name.strip()

    def _infer_author_from_files(self, source: AudiobookSource) -> str | None:
        """Try to get author from embedded tags in audio files."""
        if not self._mutagen or not source.audio_files:
            return None

        for audio in source.audio_files[:3]:  # Check first few files
            try:
                f = self._mutagen.File(audio.path)
                if f and hasattr(f, "tags") and f.tags:
                    author = self._get_tag(f.tags, ["TPE1", "artist", "\xa9ART", "author", "albumartist"])
                    if author and author.lower() not in ["various", "unknown", "various artists"]:
                        return author
            except Exception:
                continue

        return None

    def find_cover_file(self, source: AudiobookSource) -> Path | None:
        """Find cover image file in audiobook directory."""
        cover_names = ["cover", "folder", "front", "artwork", "album"]
        image_exts = [".jpg", ".jpeg", ".png"]

        for name in cover_names:
            for ext in image_exts:
                cover_path = source.source_path / f"{name}{ext}"
                if cover_path.exists():
                    return cover_path

        # Also check for any image file
        for ext in image_exts:
            for path in source.source_path.glob(f"*{ext}"):
                return path

        return None


class FilesystemOrganizer(FileOrganizer):
    """Organize audiobooks into library structure."""

    def __init__(self, use_hardlinks: bool = True):
        self.use_hardlinks = use_hardlinks

    def organize(self, source: Path, metadata: AudiobookMetadata, library_root: Path) -> Path:
        """Move processed audiobook to organized library location."""
        destination = self.compute_destination(metadata, library_root)
        destination.mkdir(parents=True, exist_ok=True)

        dest_file = destination / source.name
        if dest_file.exists():
            dest_file.unlink()

        if self.use_hardlinks:
            try:
                dest_file.hardlink_to(source)
            except OSError:
                # Fallback to copy if hardlink fails (cross-device)
                shutil.copy2(source, dest_file)
        else:
            shutil.move(str(source), str(dest_file))

        return dest_file

    def compute_destination(self, metadata: AudiobookMetadata, library_root: Path) -> Path:
        """Compute organized location for audiobook."""
        # Use folder_name from metadata (handles series formatting)
        folder_name = metadata.folder_name
        return library_root / folder_name
