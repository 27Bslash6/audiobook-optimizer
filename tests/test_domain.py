"""Tests for domain models and metadata inference."""

import pytest
from pathlib import Path

from audiobook_optimizer.domain.models import (
    AudioFormat,
    AudiobookMetadata,
    Chapter,
)
from audiobook_optimizer.adapters.filesystem import (
    FilesystemMetadataExtractor,
    SERIES_PATTERNS,
    AUTHOR_PATTERNS,
    SKIP_PATTERNS,
)


class TestAudioFormat:
    """Test AudioFormat enum."""

    @pytest.mark.parametrize("ext,expected", [
        (".mp3", AudioFormat.MP3),
        (".MP3", AudioFormat.MP3),
        ("mp3", AudioFormat.MP3),
        (".m4a", AudioFormat.M4A),
        (".m4b", AudioFormat.M4B),
        (".flac", AudioFormat.FLAC),
        (".ogg", AudioFormat.OGG),
        (".xyz", AudioFormat.UNKNOWN),
    ])
    def test_from_extension(self, ext: str, expected: AudioFormat):
        assert AudioFormat.from_extension(ext) == expected


class TestChapter:
    """Test Chapter model."""

    def test_duration_with_end(self):
        ch = Chapter(title="Chapter 1", start_ms=0, end_ms=60000)
        assert ch.duration_ms == 60000

    def test_duration_without_end(self):
        ch = Chapter(title="Chapter 1", start_ms=0)
        assert ch.duration_ms is None


class TestAudiobookMetadata:
    """Test AudiobookMetadata model."""

    def test_display_name_standalone(self):
        meta = AudiobookMetadata(title="The Great Book", author="John Smith")
        assert meta.display_name == "The Great Book"

    def test_display_name_with_series(self):
        meta = AudiobookMetadata(
            title="The Colour of Magic",
            author="Terry Pratchett",
            series="Discworld",
            series_number=1,
        )
        assert meta.display_name == "Discworld 01 - The Colour of Magic"

    def test_display_name_with_decimal_series(self):
        meta = AudiobookMetadata(
            title="Interlude",
            author="Author",
            series="Series",
            series_number=1.5,
        )
        assert meta.display_name == "Series 1.5 - Interlude"

    def test_folder_name_sanitizes_chars(self):
        meta = AudiobookMetadata(title="What: A Story?", author="Who")
        assert ":" not in meta.folder_name
        assert "?" not in meta.folder_name


class TestNameCleaning:
    """Test _clean_name logic in FilesystemMetadataExtractor."""

    @pytest.fixture
    def extractor(self):
        return FilesystemMetadataExtractor()

    @pytest.mark.parametrize("dirty,clean", [
        ("Title [audiobook]", "Title"),
        ("Title Audiobook", "Title"),
        ("Title (Audiobook)", "Title"),
        ("Title by dessalines", "Title"),
        ("Title by uploader", "Title"),
        ("2008 - Title", "Title"),
        ("Michael Parenti - Friendly Feudalism - The Tibet Myth [audiobook] by dessalines",
         "Michael Parenti - Friendly Feudalism - The Tibet Myth"),
    ])
    def test_clean_name(self, extractor, dirty: str, clean: str):
        assert extractor._clean_name(dirty) == clean


class TestSeriesPatterns:
    """Test series detection patterns."""

    @pytest.mark.parametrize("name,series,num,title", [
        ("Discworld 01 - The Colour of Magic", "Discworld", "01", "The Colour of Magic"),
        ("Discworld 1 - Title", "Discworld", "1", "Title"),
        ("Series 12.5 - Half Book", "Series", "12.5", "Half Book"),
        ("01 - First Book", None, "01", "First Book"),
        ("Book 3 - Third", None, "3", "Third"),
        ("Volume 2 - Second", None, "2", "Second"),
    ])
    def test_series_patterns(self, name: str, series: str | None, num: str, title: str):
        matched = False
        for pattern in SERIES_PATTERNS:
            match = pattern.match(name)
            if match:
                groups = match.groupdict()
                assert groups.get("title", "").strip() == title
                if series:
                    assert groups.get("series", "").strip() == series
                assert groups.get("num") == num
                matched = True
                break
        assert matched, f"No pattern matched: {name}"


class TestAuthorPatterns:
    """Test author detection patterns."""

    @pytest.mark.parametrize("name,author,title", [
        ("Michael Parenti - Friendly Feudalism", "Michael Parenti", "Friendly Feudalism"),
        ("The Great Book by John Smith", "John Smith", "The Great Book"),
        ("Smith, John - A Title", "Smith, John", "A Title"),
    ])
    def test_author_patterns(self, name: str, author: str, title: str):
        matched = False
        for pattern in AUTHOR_PATTERNS:
            match = pattern.match(name)
            if match:
                groups = match.groupdict()
                assert groups.get("author", "").strip() == author
                assert groups.get("title", "").strip() == title
                matched = True
                break
        assert matched, f"No pattern matched: {name}"


class TestSkipPatterns:
    """Test non-audiobook detection."""

    @pytest.mark.parametrize("name", [
        "Some EPUB Collection",
        "N64 Emulator ROMs",
        "PCSX2 BIOS Files",
        "Album OST Soundtrack",
        "Game Music MP3 320k",
    ])
    def test_skip_patterns_match(self, name: str):
        matched = any(p.search(name) for p in SKIP_PATTERNS)
        assert matched, f"Should skip: {name}"

    @pytest.mark.parametrize("name", [
        "Discworld 01 - The Colour of Magic",
        "Michael Parenti - Friendly Feudalism",
    ])
    def test_skip_patterns_dont_match_audiobooks(self, name: str):
        matched = any(p.search(name) for p in SKIP_PATTERNS)
        assert not matched, f"Should NOT skip: {name}"
