"""Integration tests with real audio files.

These tests use actual audio from LibriVox (public domain) to verify
the full processing pipeline works with real-world data.

Fixtures are downloaded once and cached in tests/.fixtures/
"""

from pathlib import Path

import pytest

from audiobook_optimizer.adapters.ffmpeg import FFmpegConverter
from audiobook_optimizer.adapters.filesystem import FilesystemMetadataExtractor, FilesystemScanner
from audiobook_optimizer.adapters.tagger import MutagenTagger
from audiobook_optimizer.domain.models import AudioFile, AudioFormat, ProcessingStatus
from audiobook_optimizer.services.processor import AudiobookProcessor


class TestFFmpegWithRealAudio:
    """Test FFmpeg operations with real audio files."""

    @pytest.fixture
    def converter(self) -> FFmpegConverter:
        """Create FFmpegConverter instance."""
        return FFmpegConverter()

    def test_probe_single_file(self, converter: FFmpegConverter, meditations_single_chapter):
        """Probe a real MP3 file and verify data extraction."""
        probe_data = converter.probe_file(meditations_single_chapter)

        # Verify structure
        assert "format" in probe_data
        assert "streams" in probe_data
        assert "duration" in probe_data["format"]

        # Verify audio stream exists
        audio_streams = [s for s in probe_data["streams"] if s.get("codec_type") == "audio"]
        assert len(audio_streams) == 1

        audio_stream = audio_streams[0]
        assert audio_stream["codec_name"] == "mp3"
        assert int(audio_stream["channels"]) in (1, 2)  # Mono or stereo

    def test_probe_duration(self, converter: FFmpegConverter, meditations_single_chapter):
        """Verify duration extraction from real file."""
        duration_ms = converter.probe_duration(meditations_single_chapter)

        # LibriVox chapters are typically 5-30 minutes
        assert duration_ms > 60_000  # At least 1 minute
        assert duration_ms < 3600_000  # Less than 1 hour per chapter

    def test_convert_single_file_to_m4b(self, converter: FFmpegConverter, meditations_single_chapter, temp_output_dir):
        """Convert a single MP3 to M4B."""
        output_path = temp_output_dir / "test_single.m4b"

        audio_file = AudioFile(
            path=meditations_single_chapter,
            format=AudioFormat.MP3,
            duration_ms=converter.probe_duration(meditations_single_chapter),
        )

        result_path, was_remuxed = converter.convert_to_m4b(
            source_files=[audio_file],
            output_path=output_path,
            bitrate=48,  # Lower bitrate for faster test
            mono=True,
        )

        assert result_path.exists()
        assert result_path.suffix == ".m4b"
        assert was_remuxed is False  # MP3 source requires transcoding

        # Verify output is valid audio
        probe_data = converter.probe_file(result_path)
        audio_stream = next(s for s in probe_data["streams"] if s["codec_type"] == "audio")
        assert audio_stream["codec_name"] == "aac"
        assert int(audio_stream["channels"]) == 1  # Mono requested

    def test_convert_multiple_files_with_chapters(self, converter: FFmpegConverter, meditations_few_chapters, temp_output_dir):
        """Convert multiple MP3s to M4B with proper chapters."""
        output_path = temp_output_dir / "test_multi.m4b"

        # Build AudioFile list
        audio_files = []
        for path in meditations_few_chapters:
            duration = converter.probe_duration(path)
            audio_files.append(
                AudioFile(
                    path=path,
                    format=AudioFormat.MP3,
                    duration_ms=duration,
                )
            )

        result_path, was_remuxed = converter.convert_to_m4b(
            source_files=audio_files,
            output_path=output_path,
            bitrate=48,
            mono=True,
        )

        assert result_path.exists()

        # Verify chapters were embedded
        probe_data = converter.probe_file(result_path)
        chapters = probe_data.get("chapters", [])
        assert len(chapters) == len(meditations_few_chapters), "Should have one chapter per source file"

        # Verify chapters are sequential and non-overlapping
        for i, ch in enumerate(chapters):
            assert float(ch["start_time"]) >= 0
            if i > 0:
                prev_end = float(chapters[i - 1]["end_time"])
                this_start = float(ch["start_time"])
                assert abs(this_start - prev_end) < 0.1, "Chapters should be contiguous"

    def test_bitrate_never_upscales(self, converter: FFmpegConverter, meditations_single_chapter):
        """Verify effective bitrate calculation respects source quality."""
        # LibriVox 64kbps source
        audio_file = AudioFile(
            path=meditations_single_chapter,
            format=AudioFormat.MP3,
            duration_ms=60000,
            bitrate=64,  # Source is 64kbps
        )

        # Request 128kbps - should NOT upscale
        effective = converter._calculate_effective_bitrate([audio_file], target_bitrate=128)
        assert effective == 64, "Should not upscale 64kbps source to 128kbps"

        # Request 48kbps - should downscale
        effective = converter._calculate_effective_bitrate([audio_file], target_bitrate=48)
        assert effective == 48, "Should downscale to 48kbps when target is lower"

    def test_aac_source_remuxes_without_reencoding(
        self, converter: FFmpegConverter, meditations_single_chapter, temp_output_dir
    ):
        """Verify AAC sources are remuxed (stream copied), not transcoded.

        This tests the 'prefers remux over transcode' claim in the architecture.
        """
        # Step 1: Create an AAC source file by transcoding MP3 -> M4A
        aac_source = temp_output_dir / "source.m4a"
        mp3_duration = converter.probe_duration(meditations_single_chapter)
        mp3_file = AudioFile(
            path=meditations_single_chapter,
            format=AudioFormat.MP3,
            duration_ms=mp3_duration,
        )

        # Convert to mono AAC first (simulating an AAC audiobook source)
        converter.convert_to_m4b(
            source_files=[mp3_file],
            output_path=aac_source,
            bitrate=64,
            mono=True,
        )

        # Step 2: Now convert this AAC source to M4B - should REMUX
        m4b_output = temp_output_dir / "remuxed.m4b"
        aac_file = AudioFile(
            path=aac_source,
            format=AudioFormat.M4A,
            duration_ms=converter.probe_duration(aac_source),
        )

        result_path, was_remuxed = converter.convert_to_m4b(
            source_files=[aac_file],
            output_path=m4b_output,
            bitrate=64,
            mono=True,  # Source is already mono, should remux
        )

        assert result_path.exists()
        assert was_remuxed is True, "AAC mono source should be remuxed, not transcoded"

        # Verify output codec is still AAC
        probe_data = converter.probe_file(result_path)
        audio_stream = next(s for s in probe_data["streams"] if s["codec_type"] == "audio")
        assert audio_stream["codec_name"] == "aac"

    def test_stereo_to_mono_forces_transcode(self, converter: FFmpegConverter, meditations_single_chapter, temp_output_dir):
        """Verify stereo AAC source is transcoded when mono output is requested."""
        # Step 1: Create stereo AAC source
        stereo_aac = temp_output_dir / "stereo_source.m4a"
        mp3_file = AudioFile(
            path=meditations_single_chapter,
            format=AudioFormat.MP3,
            duration_ms=converter.probe_duration(meditations_single_chapter),
        )

        # Convert to STEREO AAC
        converter.convert_to_m4b(
            source_files=[mp3_file],
            output_path=stereo_aac,
            bitrate=64,
            mono=False,  # Keep stereo
        )

        # Verify it's actually stereo
        probe = converter.probe_file(stereo_aac)
        stereo_stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")
        assert int(stereo_stream["channels"]) == 2, "Test setup: should be stereo"

        # Step 2: Convert stereo AAC to mono M4B - must transcode
        m4b_output = temp_output_dir / "forced_transcode.m4b"
        aac_file = AudioFile(
            path=stereo_aac,
            format=AudioFormat.M4A,
            duration_ms=converter.probe_duration(stereo_aac),
        )

        result_path, was_remuxed = converter.convert_to_m4b(
            source_files=[aac_file],
            output_path=m4b_output,
            bitrate=64,
            mono=True,  # Request mono from stereo source
        )

        assert result_path.exists()
        assert was_remuxed is False, "Stereo->mono conversion requires transcoding"

        # Verify output is mono
        out_probe = converter.probe_file(result_path)
        out_stream = next(s for s in out_probe["streams"] if s["codec_type"] == "audio")
        assert int(out_stream["channels"]) == 1


class TestMetadataExtractionWithRealAudio:
    """Test metadata extraction from real audio files."""

    @pytest.fixture
    def extractor(self) -> FilesystemMetadataExtractor:
        return FilesystemMetadataExtractor()

    def test_extract_file_info(self, extractor: FilesystemMetadataExtractor, meditations_single_chapter):
        """Extract real metadata from LibriVox MP3."""
        info = extractor.extract_file_info(meditations_single_chapter)

        assert info.format == AudioFormat.MP3
        # Duration should be reasonable for an audiobook chapter (1 min to 1 hour)
        assert 60_000 < info.duration_ms < 3_600_000, f"Duration {info.duration_ms}ms out of expected range"
        # LibriVox uses 64kbps, allow some tolerance for header overhead
        assert info.bitrate is not None
        assert 60 <= info.bitrate <= 68, f"Expected ~64kbps, got {info.bitrate}kbps"
        # Standard audio sample rates
        assert info.sample_rate in (22050, 44100, 48000), f"Unexpected sample rate: {info.sample_rate}"
        # Mono or stereo only
        assert info.channels in (1, 2), f"Unexpected channel count: {info.channels}"

    def test_extract_metadata_from_directory(self, extractor: FilesystemMetadataExtractor, meditations_audiobook):
        """Infer metadata from real audiobook directory."""
        # Create a minimal AudiobookSource
        scanner = FilesystemScanner()
        sources = list(scanner.scan_directory(meditations_audiobook))

        assert len(sources) >= 1, "Should find at least one audiobook"

        source = sources[0]
        metadata = extractor.infer_metadata(source)

        # LibriVox naming is usually "Title - Author" or similar
        assert metadata.title, "Should extract title"
        assert metadata.author, "Should have author (even if 'Unknown Author')"


class TestFilesystemScannerWithRealAudio:
    """Test filesystem scanning with real audiobook structure."""

    def test_scan_finds_audiobook(self, meditations_audiobook):
        """Verify scanner finds the audiobook directory."""
        scanner = FilesystemScanner()
        sources = list(scanner.scan_directory(meditations_audiobook))

        # Should find at least one source
        assert len(sources) >= 1

        source = sources[0]
        assert len(source.audio_files) > 0, "Should have audio files"
        assert all(f.format == AudioFormat.MP3 for f in source.audio_files)

    def test_is_audiobook_directory(self, meditations_audiobook):
        """Verify audiobook directory detection."""
        scanner = FilesystemScanner()

        # The meditations directory should be recognized as an audiobook
        assert scanner.is_audiobook_directory(meditations_audiobook)


class TestMutagenTaggerWithRealAudio:
    """Test metadata tagging on real converted files."""

    def test_apply_metadata_to_converted_m4b(self, meditations_single_chapter, temp_output_dir):
        """Apply metadata to a real converted M4B."""
        from audiobook_optimizer.domain.models import AudiobookMetadata

        converter = FFmpegConverter()
        tagger = MutagenTagger()

        # First convert to M4B
        output_path = temp_output_dir / "tagged_test.m4b"
        audio_file = AudioFile(
            path=meditations_single_chapter,
            format=AudioFormat.MP3,
            duration_ms=converter.probe_duration(meditations_single_chapter),
        )

        converter.convert_to_m4b(
            source_files=[audio_file],
            output_path=output_path,
            bitrate=48,
            mono=True,
        )

        # Apply metadata
        metadata = AudiobookMetadata(
            title="Meditations",
            author="Marcus Aurelius",
            narrator="LibriVox Volunteers",
            series=None,
            year=180,  # Original work date
        )

        tagger.apply_metadata(output_path, metadata)

        # Verify metadata was written
        import mutagen

        audio = mutagen.File(output_path)
        assert audio is not None

        # MP4/M4B uses specific tag keys
        assert "\xa9nam" in audio.tags  # Title
        assert "\xa9ART" in audio.tags  # Artist/Author


class TestFullProcessingPipeline:
    """End-to-end processing pipeline tests."""

    def test_process_single_file_audiobook(self, meditations_single_chapter, temp_output_dir, tmp_path):
        """Process a single audio file through the complete pipeline."""
        # Set up source directory structure
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create a subfolder like a real audiobook
        book_dir = source_dir / "Marcus Aurelius - Meditations"
        book_dir.mkdir()

        # Copy (or link) the test file
        import shutil

        dest_file = book_dir / meditations_single_chapter.name
        shutil.copy(meditations_single_chapter, dest_file)

        # Process
        processor = AudiobookProcessor(
            source_dir=source_dir,
            output_dir=temp_output_dir,
            bitrate=48,
            mono=True,
            preserve_source=True,
        )

        sources = processor.scan()
        assert len(sources) == 1

        result = processor.process(sources[0])

        assert result.status == ProcessingStatus.COMPLETED, f"Processing failed: {result.error_message}"
        assert result.output_path is not None
        assert result.output_path.exists()
        assert result.output_path.suffix == ".m4b"
        assert result.duration_ms > 0
        assert result.was_remuxed is False  # MP3 requires transcoding

    def test_process_multi_file_audiobook(self, meditations_few_chapters, temp_output_dir, tmp_path):
        """Process multiple files into a single M4B."""
        # Set up source directory
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        book_dir = source_dir / "Meditations by Marcus Aurelius"
        book_dir.mkdir()

        # Copy test files
        import shutil

        for i, path in enumerate(meditations_few_chapters):
            dest = book_dir / f"chapter_{i + 1:02d}.mp3"
            shutil.copy(path, dest)

        processor = AudiobookProcessor(
            source_dir=source_dir,
            output_dir=temp_output_dir,
            bitrate=48,
            mono=True,
            preserve_source=True,
        )

        sources = processor.scan()
        assert len(sources) == 1

        result = processor.process(sources[0])

        assert result.status == ProcessingStatus.COMPLETED, f"Processing failed: {result.error_message}"
        assert result.output_path.exists()

        # Verify chapters
        converter = FFmpegConverter()
        probe_data = converter.probe_file(result.output_path)
        chapters = probe_data.get("chapters", [])
        assert len(chapters) == len(meditations_few_chapters)

    def test_processing_preserves_source(self, meditations_single_chapter, temp_output_dir, tmp_path):
        """Verify source files are preserved when preserve_source=True."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        book_dir = source_dir / "Test Audiobook"
        book_dir.mkdir()

        import shutil

        source_file = book_dir / "test.mp3"
        shutil.copy(meditations_single_chapter, source_file)

        processor = AudiobookProcessor(
            source_dir=source_dir,
            output_dir=temp_output_dir,
            bitrate=48,
            mono=True,
            preserve_source=True,
        )

        sources = processor.scan()
        processor.process(sources[0])

        # Source should still exist
        assert source_file.exists(), "Source file should be preserved"


class TestCachingBehavior:
    """Test that caching works correctly with real files."""

    def test_ffprobe_caching(self, meditations_single_chapter):
        """Verify ffprobe results are cached and cache hits are recorded."""
        from audiobook_optimizer.adapters.ffmpeg import _probe_file_cached

        # Clear L1 cache (L2/Redis may persist across runs)
        _probe_file_cached.cache_clear()

        converter = FFmpegConverter()

        # First probe
        result1 = converter.probe_file(meditations_single_chapter)
        info_after_first = _probe_file_cached.cache_info()

        # Second probe - should hit cache (L1 or L2)
        result2 = converter.probe_file(meditations_single_chapter)
        info_after_second = _probe_file_cached.cache_info()

        # After two probes of the same file, total hits should increase
        # (second call hits either L1 or L2 cache)
        total_hits = info_after_second.hits
        assert total_hits > info_after_first.hits, "Second call should be a cache hit"

        # Results must be identical
        assert result1 == result2

        # Verify returned data is valid probe output
        assert "format" in result1
        assert "duration" in result1["format"]
        assert float(result1["format"]["duration"]) > 0


class TestEdgeCases:
    """Test edge cases and error handling with real files."""

    def test_handles_special_characters_in_path(self, meditations_single_chapter, temp_output_dir, tmp_path):
        """Verify handling of paths with special characters."""
        # Create a path with special characters
        weird_dir = tmp_path / "Author's Book - Title (2023)"
        weird_dir.mkdir()

        import shutil

        weird_file = weird_dir / "chapter 01 - intro.mp3"
        shutil.copy(meditations_single_chapter, weird_file)

        converter = FFmpegConverter()
        audio_file = AudioFile(
            path=weird_file,
            format=AudioFormat.MP3,
            duration_ms=converter.probe_duration(weird_file),
        )

        output_path = temp_output_dir / "special_chars_test.m4b"
        result_path, _ = converter.convert_to_m4b(
            source_files=[audio_file],
            output_path=output_path,
            bitrate=48,
            mono=True,
        )

        assert result_path.exists()

    def test_handles_unicode_in_metadata(self, meditations_single_chapter, temp_output_dir):
        """Verify Unicode metadata is handled correctly."""
        from audiobook_optimizer.domain.models import AudiobookMetadata

        converter = FFmpegConverter()
        tagger = MutagenTagger()

        output_path = temp_output_dir / "unicode_test.m4b"
        audio_file = AudioFile(
            path=meditations_single_chapter,
            format=AudioFormat.MP3,
            duration_ms=converter.probe_duration(meditations_single_chapter),
        )

        converter.convert_to_m4b(
            source_files=[audio_file],
            output_path=output_path,
            bitrate=48,
            mono=True,
        )

        # Unicode metadata
        metadata = AudiobookMetadata(
            title="Méditations",
            author="Marc-Aurèle",
            narrator="Narráteur Français",
        )

        tagger.apply_metadata(output_path, metadata)

        # Verify it can be read back
        import mutagen

        audio = mutagen.File(output_path)
        assert "Méditations" in str(audio.tags.get("\xa9nam", ""))


class TestErrorScenarios:
    """Test error handling and recovery."""

    def test_empty_source_files_raises(self):
        """Verify empty source list raises FFmpegError."""
        from audiobook_optimizer.adapters.ffmpeg import FFmpegError

        converter = FFmpegConverter()
        with pytest.raises(FFmpegError, match="No source files"):
            converter.convert_to_m4b(
                source_files=[],
                output_path=Path("/tmp/should_not_exist.m4b"),
                bitrate=64,
                mono=True,
            )

    def test_nonexistent_source_file_raises(self, temp_output_dir):
        """Verify processing nonexistent file raises FileNotFoundError."""
        converter = FFmpegConverter()
        fake_file = AudioFile(
            path=Path("/nonexistent/file.mp3"),
            format=AudioFormat.MP3,
            duration_ms=60000,
        )

        # The converter tries to stat the file for mtime before probing
        with pytest.raises(FileNotFoundError):
            converter.convert_to_m4b(
                source_files=[fake_file],
                output_path=temp_output_dir / "test.m4b",
                bitrate=64,
                mono=True,
            )

    def test_corrupt_file_metadata_extraction(self, temp_output_dir):
        """Verify metadata extractor handles corrupt/invalid files gracefully."""
        # Create a file with invalid audio content
        corrupt_file = temp_output_dir / "corrupt.mp3"
        corrupt_file.write_bytes(b"this is not audio data at all")

        extractor = FilesystemMetadataExtractor()
        # Should not crash, returns file with default/zero values
        info = extractor.extract_file_info(corrupt_file)

        assert info.path == corrupt_file
        assert info.format == AudioFormat.MP3
        # Duration will be 0 for unreadable files
        assert info.duration_ms == 0

    def test_processor_handles_no_audio_files(self, temp_output_dir, tmp_path):
        """Verify processor handles directories with no audio gracefully."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create a directory with only non-audio files
        book_dir = source_dir / "Fake Audiobook"
        book_dir.mkdir()
        (book_dir / "readme.txt").write_text("Not an audiobook")
        (book_dir / "cover.jpg").write_bytes(b"\xff\xd8\xff")  # Fake JPEG header

        processor = AudiobookProcessor(
            source_dir=source_dir,
            output_dir=temp_output_dir,
            bitrate=48,
            mono=True,
        )

        sources = processor.scan()
        # Should find no audiobooks (directory has no audio files)
        assert len(sources) == 0

    def test_probe_nonexistent_file_raises(self):
        """Verify probing nonexistent file raises FileNotFoundError."""
        converter = FFmpegConverter()

        # probe_file stats the file for mtime first, so FileNotFoundError
        with pytest.raises(FileNotFoundError):
            converter.probe_file(Path("/nonexistent/file.mp3"))


class TestParallelProcessing:
    """Test thread safety of parallel audiobook processing."""

    def test_parallel_ffmpeg_calls_are_safe(self, meditations_few_chapters, temp_output_dir):
        """Verify multiple FFmpeg conversions can run concurrently."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        converter = FFmpegConverter()

        def convert_one(idx: int, source_path: Path) -> tuple[int, Path, bool]:
            """Convert a single file in a thread."""
            output = temp_output_dir / f"parallel_test_{idx}.m4b"
            audio_file = AudioFile(
                path=source_path,
                format=AudioFormat.MP3,
                duration_ms=converter.probe_duration(source_path),
            )
            result_path, was_remuxed = converter.convert_to_m4b(
                source_files=[audio_file],
                output_path=output,
                bitrate=48,
                mono=True,
            )
            return idx, result_path, result_path.exists()

        # Run 3 conversions in parallel
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(convert_one, i, path): i for i, path in enumerate(meditations_few_chapters)}
            for future in as_completed(futures):
                results.append(future.result())

        # All should succeed
        assert len(results) == 3
        for idx, path, exists in results:
            assert exists, f"Output {idx} should exist"
            # Verify output is valid
            probe = converter.probe_file(path)
            assert probe["format"]["format_name"] == "mov,mp4,m4a,3gp,3g2,mj2"

    def test_parallel_probing_is_thread_safe(self, meditations_few_chapters):
        """Verify ffprobe caching works correctly under concurrent access."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from audiobook_optimizer.adapters.ffmpeg import _probe_file_cached

        _probe_file_cached.cache_clear()
        converter = FFmpegConverter()

        def probe_file(path: Path) -> dict:
            """Probe a file in a thread."""
            return converter.probe_file(path)

        # Probe all files multiple times concurrently
        all_paths = meditations_few_chapters * 3  # 9 probes total, 3 unique files
        results = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(probe_file, p) for p in all_paths]
            for future in as_completed(futures):
                results.append(future.result())

        # All probes should succeed
        assert len(results) == 9

        # Cache should show hits (each file probed 3x, so 6 hits expected)
        info = _probe_file_cached.cache_info()
        assert info.hits >= 6, f"Expected at least 6 cache hits, got {info.hits}"

    def test_processor_parallel_via_batch(self, meditations_few_chapters, temp_output_dir, tmp_path):
        """Test processor.process_batch simulates parallel-like behavior."""
        import shutil

        # Set up multiple audiobook directories
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        for i, chapter in enumerate(meditations_few_chapters):
            book_dir = source_dir / f"Audiobook {i + 1}"
            book_dir.mkdir()
            shutil.copy(chapter, book_dir / "chapter.mp3")

        processor = AudiobookProcessor(
            source_dir=source_dir,
            output_dir=temp_output_dir,
            bitrate=48,
            mono=True,
        )

        sources = processor.scan()
        assert len(sources) == 3

        # Process all at once (sequential but tests the batch interface)
        results = processor.process_batch(sources)

        assert len(results) == 3
        assert all(r.status == ProcessingStatus.COMPLETED for r in results)
        assert all(r.output_path and r.output_path.exists() for r in results)

    def test_parallel_processing_handles_failures(self, meditations_single_chapter, temp_output_dir):
        """Verify parallel processing handles mixed success/failure gracefully."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        converter = FFmpegConverter()

        def process_task(task_id: int) -> tuple[int, bool, str]:
            """Process a task that may succeed or fail."""
            if task_id == 1:
                # This one will fail - nonexistent file
                try:
                    converter.probe_file(Path("/nonexistent/fail.mp3"))
                    return task_id, True, "unexpected success"
                except FileNotFoundError:
                    return task_id, False, "file not found"
            else:
                # These succeed
                output = temp_output_dir / f"task_{task_id}.m4b"
                audio_file = AudioFile(
                    path=meditations_single_chapter,
                    format=AudioFormat.MP3,
                    duration_ms=60000,
                )
                converter.convert_to_m4b(
                    source_files=[audio_file],
                    output_path=output,
                    bitrate=48,
                    mono=True,
                )
                return task_id, True, "success"

        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(process_task, i): i for i in range(3)}
            for future in as_completed(futures):
                task_id, success, msg = future.result()
                results[task_id] = (success, msg)

        # Task 1 should fail, others succeed
        assert results[0][0] is True
        assert results[1][0] is False
        assert results[2][0] is True


class TestMetadataTaggingComplete:
    """Test complete metadata flow including series and year."""

    def test_series_metadata_is_tagged(self, meditations_single_chapter, temp_output_dir):
        """Verify series information is written to M4B tags."""
        from audiobook_optimizer.domain.models import AudiobookMetadata

        converter = FFmpegConverter()
        tagger = MutagenTagger()

        output_path = temp_output_dir / "series_test.m4b"
        audio_file = AudioFile(
            path=meditations_single_chapter,
            format=AudioFormat.MP3,
            duration_ms=converter.probe_duration(meditations_single_chapter),
        )

        converter.convert_to_m4b(
            source_files=[audio_file],
            output_path=output_path,
            bitrate=48,
            mono=True,
        )

        # Apply metadata with series info
        metadata = AudiobookMetadata(
            title="The Colour of Magic",
            author="Terry Pratchett",
            series="Discworld",
            series_number=1,
            year=1983,
        )

        tagger.apply_metadata(output_path, metadata)

        # Verify metadata was written correctly
        import mutagen

        audio = mutagen.File(output_path)
        assert audio is not None

        # Title
        assert "The Colour of Magic" in str(audio.tags.get("\xa9nam", ""))
        # Author
        assert "Terry Pratchett" in str(audio.tags.get("\xa9ART", ""))
        # Album (series name)
        assert "Discworld" in str(audio.tags.get("\xa9alb", ""))
        # Track number
        track_tag = audio.tags.get("trkn")
        if track_tag:
            assert track_tag[0][0] == 1  # Track 1
