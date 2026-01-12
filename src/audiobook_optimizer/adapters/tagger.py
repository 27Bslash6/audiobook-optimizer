"""Mutagen adapter for tagging M4B files."""

from pathlib import Path

from audiobook_optimizer.domain.models import AudiobookMetadata, Chapter
from audiobook_optimizer.ports.interfaces import AudioTagger


class TaggerError(Exception):
    """Tagging operation failed."""


class MutagenTagger(AudioTagger):
    """Tag M4B files using mutagen."""

    def __init__(self):
        try:
            from mutagen.mp4 import MP4, MP4Cover

            self._MP4 = MP4
            self._MP4Cover = MP4Cover
        except ImportError as e:
            raise TaggerError("mutagen is required for tagging: pip install mutagen") from e

    def apply_metadata(self, path: Path, metadata: AudiobookMetadata) -> None:
        """Write metadata tags to M4B file.

        M4B uses MP4 container with these common atoms:
        - \xa9nam: Title
        - \xa9ART: Artist (Author for audiobooks)
        - \xa9alb: Album (Series or Title)
        - \xa9wrt: Composer (Narrator)
        - \xa9gen: Genre
        - \xa9day: Year
        - aART: Album Artist
        - trkn: Track number (Series number)
        - disk: Disc number
        - desc: Description
        """
        audio = self._MP4(path)

        # Core metadata
        audio["\xa9nam"] = metadata.title
        audio["\xa9ART"] = metadata.author
        audio["aART"] = metadata.author  # Album artist

        # Album: use series if available, otherwise title
        if metadata.series:
            audio["\xa9alb"] = metadata.series
            # Use track number for series position
            if metadata.series_number is not None:
                num = int(metadata.series_number)
                audio["trkn"] = [(num, 0)]  # (track, total)
        else:
            audio["\xa9alb"] = metadata.title

        # Optional metadata
        if metadata.narrator:
            audio["\xa9wrt"] = metadata.narrator

        audio["\xa9gen"] = metadata.genre

        if metadata.year:
            audio["\xa9day"] = str(metadata.year)

        if metadata.description:
            audio["desc"] = metadata.description

        audio.save()

    def embed_cover(self, path: Path, cover_data: bytes, mime_type: str = "image/jpeg") -> None:
        """Embed cover art in M4B file.

        MP4Cover format codes:
        - FORMAT_JPEG = 13
        - FORMAT_PNG = 14
        """
        audio = self._MP4(path)

        # Determine format from mime type
        if "png" in mime_type.lower():
            image_format = self._MP4Cover.FORMAT_PNG
        else:
            image_format = self._MP4Cover.FORMAT_JPEG

        cover = self._MP4Cover(cover_data, imageformat=image_format)
        audio["covr"] = [cover]
        audio.save()

    def write_chapters(self, path: Path, chapters: list[Chapter]) -> None:
        """Write chapter markers to M4B file.

        Note: MP4/M4B chapter support via mutagen is limited.
        For full chapter support, we rely on ffmpeg during conversion.
        This method is provided for completeness but may not be fully functional.
        """
        # Chapters are better handled during conversion with ffmpeg metadata
        # Mutagen doesn't have great MP4 chapter support
        pass

    def read_existing_metadata(self, path: Path) -> dict:
        """Read existing metadata from M4B file."""
        audio = self._MP4(path)

        return {
            "title": self._get_tag(audio, "\xa9nam"),
            "author": self._get_tag(audio, "\xa9ART"),
            "album": self._get_tag(audio, "\xa9alb"),
            "narrator": self._get_tag(audio, "\xa9wrt"),
            "genre": self._get_tag(audio, "\xa9gen"),
            "year": self._get_tag(audio, "\xa9day"),
            "description": self._get_tag(audio, "desc"),
            "has_cover": "covr" in audio,
        }

    def _get_tag(self, audio, key: str) -> str | None:
        """Get tag value, handling mutagen's list wrapping."""
        if key in audio:
            val = audio[key]
            return str(val[0]) if isinstance(val, list) else str(val)
        return None
