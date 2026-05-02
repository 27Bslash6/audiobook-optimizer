"""Tests for domain models and metadata inference."""

import pytest

from audiobook_optimizer.adapters.filesystem import (
    AUTHOR_PATTERNS,
    SERIES_PATTERNS,
    SKIP_PATTERNS,
    FilesystemMetadataExtractor,
    FilesystemScanner,
)
from audiobook_optimizer.domain.models import (
    AudiobookMetadata,
    AudioFormat,
    Chapter,
)


class TestAudioFormat:
    """Test AudioFormat enum."""

    @pytest.mark.parametrize(
        "ext,expected",
        [
            (".mp3", AudioFormat.MP3),
            (".MP3", AudioFormat.MP3),
            ("mp3", AudioFormat.MP3),
            (".m4a", AudioFormat.M4A),
            (".m4b", AudioFormat.M4B),
            (".flac", AudioFormat.FLAC),
            (".ogg", AudioFormat.OGG),
            (".xyz", AudioFormat.UNKNOWN),
        ],
    )
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

    @pytest.mark.parametrize(
        "dirty,clean",
        [
            ("Title [audiobook]", "Title"),
            ("Title Audiobook", "Title"),
            ("Title (Audiobook)", "Title"),
            ("Title by dessalines", "Title"),
            ("Title by uploader", "Title"),
            ("2008 - Title", "Title"),
            (
                "Michael Parenti - Friendly Feudalism - The Tibet Myth [audiobook] by dessalines",
                "Michael Parenti - Friendly Feudalism - The Tibet Myth",
            ),
        ],
    )
    def test_clean_name(self, extractor, dirty: str, clean: str):
        assert extractor._clean_name(dirty) == clean


class TestSeriesPatterns:
    """Test series detection patterns."""

    @pytest.mark.parametrize(
        "name,series,num,title",
        [
            ("Discworld 01 - The Colour of Magic", "Discworld", "01", "The Colour of Magic"),
            ("Discworld 1 - Title", "Discworld", "1", "Title"),
            ("Series 12.5 - Half Book", "Series", "12.5", "Half Book"),
            ("01 - First Book", None, "01", "First Book"),
            ("Book 3 - Third", None, "3", "Third"),
            ("Volume 2 - Second", None, "2", "Second"),
        ],
    )
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

    @pytest.mark.parametrize(
        "name,author,title",
        [
            ("Michael Parenti - Friendly Feudalism", "Michael Parenti", "Friendly Feudalism"),
            ("The Great Book by John Smith", "John Smith", "The Great Book"),
            ("Smith, John - A Title", "Smith, John", "A Title"),
        ],
    )
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


class TestBitrateLogic:
    """Test bitrate calculation logic."""

    def test_effective_bitrate_never_upscales(self):
        """Verify we never waste space upscaling low-bitrate sources."""
        from unittest.mock import MagicMock

        from audiobook_optimizer.adapters.ffmpeg import FFmpegConverter

        converter = FFmpegConverter.__new__(FFmpegConverter)
        converter.ffprobe = "ffprobe"

        # Mock AudioFile with known bitrates
        mock_files = [
            MagicMock(bitrate=24),  # 24kbps source
            MagicMock(bitrate=32),  # 32kbps source
        ]

        # Target is 64kbps, but source min is 24kbps
        # Should use 24kbps, not 64kbps
        result = converter._calculate_effective_bitrate(mock_files, target_bitrate=64)
        assert result == 24, "Should use source bitrate when lower than target"

    def test_effective_bitrate_downscales_high_sources(self):
        """Verify we compress high-bitrate sources to target."""
        from unittest.mock import MagicMock

        from audiobook_optimizer.adapters.ffmpeg import FFmpegConverter

        converter = FFmpegConverter.__new__(FFmpegConverter)
        converter.ffprobe = "ffprobe"

        mock_files = [
            MagicMock(bitrate=192),
            MagicMock(bitrate=128),
        ]

        # Target is 64kbps, source min is 128kbps
        # Should use 64kbps (compress down)
        result = converter._calculate_effective_bitrate(mock_files, target_bitrate=64)
        assert result == 64, "Should use target bitrate when lower than source"


class TestSkipPatterns:
    """Test non-audiobook detection."""

    @pytest.mark.parametrize(
        "name",
        [
            "Some EPUB Collection",
            "N64 Emulator ROMs",
            "PCSX2 BIOS Files",
            "Album OST Soundtrack",
            "Game Music MP3 320k",
        ],
    )
    def test_skip_patterns_match(self, name: str):
        matched = any(p.search(name) for p in SKIP_PATTERNS)
        assert matched, f"Should skip: {name}"

    @pytest.mark.parametrize(
        "name",
        [
            "Discworld 01 - The Colour of Magic",
            "Michael Parenti - Friendly Feudalism",
        ],
    )
    def test_skip_patterns_dont_match_audiobooks(self, name: str):
        matched = any(p.search(name) for p in SKIP_PATTERNS)
        assert not matched, f"Should NOT skip: {name}"


class TestShouldSkipOverride:
    """Test that 'audiobook' in folder name overrides skip patterns."""

    @pytest.fixture
    def scanner(self):
        return FilesystemScanner()

    @pytest.mark.parametrize(
        "name",
        [
            "Thinking, Fast and Slow [Audiobook + ePub] by Daniel Kahneman",
            "Title AUDIOBOOK (MP3) 2003",
            "[Audiobook] Some EPUB Collection",
            "Some Great audiobook OST included",
        ],
    )
    def test_audiobook_in_name_overrides_skip(self, scanner, tmp_path, name: str):
        d = tmp_path / name
        d.mkdir()
        assert not scanner._should_skip(d), f"Should NOT skip: {name}"

    @pytest.mark.parametrize(
        "name",
        [
            "Some EPUB Collection",
            "Album OST Soundtrack",
            "N64 Emulator ROMs",
        ],
    )
    def test_skip_still_works_without_audiobook(self, scanner, tmp_path, name: str):
        d = tmp_path / name
        d.mkdir()
        assert scanner._should_skip(d), f"Should skip: {name}"


class TestDiscSetDetection:
    """Test disc/CD subdirectory merging."""

    @pytest.fixture
    def scanner(self):
        return FilesystemScanner()

    def _make_disc_tree(self, tmp_path, disc_names, files_per_disc=3):
        """Helper to create a directory tree with disc subdirs and audio files."""
        for disc_name in disc_names:
            disc_dir = tmp_path / disc_name
            disc_dir.mkdir()
            for i in range(files_per_disc):
                (disc_dir / f"track_{i:02d}.mp3").write_bytes(b"\x00" * 100)

    def test_disc_subdirs_detected_as_audiobook(self, scanner, tmp_path):
        book_dir = tmp_path / "Author - Title"
        book_dir.mkdir()
        self._make_disc_tree(book_dir, ["Disc 1", "Disc 2", "Disc 3"])

        assert scanner.is_audiobook_directory(book_dir)

    def test_cd_subdirs_detected(self, scanner, tmp_path):
        book_dir = tmp_path / "Book Name"
        book_dir.mkdir()
        self._make_disc_tree(book_dir, ["CD 1", "CD 2"])

        assert scanner.is_audiobook_directory(book_dir)

    def test_part_subdirs_detected(self, scanner, tmp_path):
        book_dir = tmp_path / "Book Name"
        book_dir.mkdir()
        self._make_disc_tree(book_dir, ["Part 1", "Part 2", "Part 3"])

        assert scanner.is_audiobook_directory(book_dir)

    def test_single_disc_not_detected_as_set(self, scanner, tmp_path):
        book_dir = tmp_path / "Book Name"
        book_dir.mkdir()
        self._make_disc_tree(book_dir, ["Disc 1"])

        assert not scanner.is_audiobook_directory(book_dir)

    def test_non_disc_subdirs_not_merged(self, scanner, tmp_path):
        """Subdirs without disc pattern should NOT be treated as disc set."""
        book_dir = tmp_path / "Collection"
        book_dir.mkdir()
        self._make_disc_tree(book_dir, ["Book One", "Book Two"])

        assert not scanner.is_audiobook_directory(book_dir)

    def test_disc_set_scan_yields_single_source(self, scanner, tmp_path):
        """Scanning a disc-set directory should yield one AudiobookSource with all files."""
        book_dir = tmp_path / "Author - Title"
        book_dir.mkdir()
        self._make_disc_tree(book_dir, ["Disc 1", "Disc 2"], files_per_disc=3)

        sources = list(scanner.scan_directory(book_dir))
        assert len(sources) == 1
        assert len(sources[0].audio_files) == 6  # 3 files × 2 discs

    def test_disc_set_files_are_sorted(self, scanner, tmp_path):
        """Files from disc set should be in disc order."""
        book_dir = tmp_path / "Author - Title"
        book_dir.mkdir()
        self._make_disc_tree(book_dir, ["Disc 2", "Disc 1"], files_per_disc=2)

        sources = list(scanner.scan_directory(book_dir))
        paths = [f.path for f in sources[0].audio_files]
        # Disc 1 files should come before Disc 2 files
        assert "Disc 1" in str(paths[0])
        assert "Disc 2" in str(paths[-1])
