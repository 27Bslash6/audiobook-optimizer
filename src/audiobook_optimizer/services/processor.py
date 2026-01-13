"""Processing service that orchestrates audiobook conversion."""

import tempfile
from pathlib import Path

from audiobook_optimizer.adapters.ai_advisor import AudioDecision, AudioQualityAdvisor, ProcessingAction
from audiobook_optimizer.adapters.ffmpeg import FFmpegConverter, FFmpegError
from audiobook_optimizer.adapters.filesystem import FilesystemMetadataExtractor, FilesystemOrganizer, FilesystemScanner
from audiobook_optimizer.adapters.tagger import MutagenTagger, TaggerError
from audiobook_optimizer.domain.models import AIDecisionInfo, AudiobookSource, Chapter, ProcessingResult, ProcessingStatus
from audiobook_optimizer.ports.interfaces import ProcessingPipeline


class AudiobookProcessor(ProcessingPipeline):
    """Full audiobook processing pipeline."""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        bitrate: int = 64,
        mono: bool = True,
        preserve_source: bool = True,
        use_ai_advisor: bool = False,
    ):
        """
        Args:
            source_dir: Where to scan for audiobooks (prowlarr dir)
            output_dir: Where to put processed audiobooks (library)
            bitrate: Target bitrate in kbps (used as fallback if AI not enabled)
            mono: Convert to mono (used as fallback if AI not enabled)
            preserve_source: Keep source files after processing
            use_ai_advisor: Use AI to make quality decisions
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.bitrate = bitrate
        self.mono = mono
        self.preserve_source = preserve_source
        self.use_ai_advisor = use_ai_advisor

        # Initialize adapters
        self.scanner = FilesystemScanner()
        self.metadata_extractor = FilesystemMetadataExtractor()
        self.converter = FFmpegConverter()
        self.tagger = MutagenTagger()
        self.organizer = FilesystemOrganizer(use_hardlinks=preserve_source)

        # Optional AI advisor
        self.advisor: AudioQualityAdvisor | None = None
        if use_ai_advisor:
            self.advisor = AudioQualityAdvisor()

    def scan(self) -> list[AudiobookSource]:
        """Scan source directory for audiobooks."""
        return list(self.scanner.scan_directory(self.source_dir))

    def process(self, source: AudiobookSource) -> ProcessingResult:
        """Process a single audiobook through the full pipeline."""
        result = ProcessingResult(source=source)

        try:
            # Step 1: Extract metadata from files
            source.status = ProcessingStatus.SCANNING
            self._populate_file_info(source)

            # Step 2: Validate we have audio to process
            source.status = ProcessingStatus.VALIDATING
            if not source.audio_files:
                raise ValueError("No audio files found")

            # Step 3: Get AI decision if advisor enabled
            ai_decision: AudioDecision | None = None
            target_bitrate = self.bitrate
            target_mono = self.mono

            if self.advisor:
                ai_decision = self._get_ai_decision(source)
                result.ai_decision = AIDecisionInfo(
                    action=ai_decision.action.value,
                    target_bitrate_kbps=ai_decision.target_bitrate_kbps,
                    preserve_stereo=ai_decision.preserve_stereo,
                    quality_warnings=ai_decision.quality_warnings,
                    reasoning=ai_decision.reasoning,
                    confidence=ai_decision.confidence,
                )
                # Use AI recommendations
                target_bitrate = ai_decision.target_bitrate_kbps
                target_mono = not ai_decision.preserve_stereo

                # Handle SKIP action
                if ai_decision.action == ProcessingAction.SKIP:
                    result.status = ProcessingStatus.SKIPPED
                    result.error_message = ai_decision.reasoning
                    source.status = ProcessingStatus.SKIPPED
                    return result

            # Step 4: Infer metadata from folder/files (skip if already set, e.g., by AI verifier)
            if source.metadata is None:
                source.metadata = self.metadata_extractor.infer_metadata(source)

            # Step 5: Extract chapters from source files if they have them
            all_chapters = self._collect_chapters(source)

            # Step 6: Convert to M4B (prefers remux if source is compatible AAC)
            source.status = ProcessingStatus.CONVERTING
            temp_dir = Path(tempfile.mkdtemp(prefix="audiobook_"))
            temp_m4b = temp_dir / f"{source.metadata.folder_name}.m4b"

            _, was_remuxed = self.converter.convert_to_m4b(
                source_files=source.audio_files,
                output_path=temp_m4b,
                chapters=all_chapters if all_chapters else None,
                bitrate=target_bitrate,
                mono=target_mono,
            )
            result.was_remuxed = was_remuxed

            # Step 6: Apply metadata tags
            source.status = ProcessingStatus.TAGGING
            self.tagger.apply_metadata(temp_m4b, source.metadata)

            # Step 7: Embed cover art if available
            cover_data = self._get_cover(source)
            if cover_data:
                self.tagger.embed_cover(temp_m4b, cover_data)

            # Step 8: Organize to library
            source.status = ProcessingStatus.ORGANIZING
            final_path = self.organizer.organize(temp_m4b, source.metadata, self.output_dir)

            # Cleanup temp directory
            temp_m4b.unlink(missing_ok=True)
            temp_dir.rmdir()

            # Success
            result.output_path = final_path
            result.status = ProcessingStatus.COMPLETED
            result.duration_ms = self.converter.probe_duration(final_path)
            result.chapters_preserved = len(all_chapters) if all_chapters else len(source.audio_files)
            source.status = ProcessingStatus.COMPLETED

        except (FFmpegError, TaggerError, ValueError, OSError) as e:
            result.status = ProcessingStatus.FAILED
            result.error_message = str(e)
            source.status = ProcessingStatus.FAILED
            source.error_message = str(e)

        return result

    def process_batch(self, sources: list[AudiobookSource]) -> list[ProcessingResult]:
        """Process multiple audiobooks."""
        return [self.process(source) for source in sources]

    def _populate_file_info(self, source: AudiobookSource) -> None:
        """Populate duration and other info for all audio files."""
        total_duration = 0
        updated_files = []

        for audio in source.audio_files:
            # Try mutagen first (faster)
            info = self.metadata_extractor.extract_file_info(audio.path)
            if info.duration_ms == 0:
                # Fallback to ffprobe
                info.duration_ms = self.converter.probe_duration(audio.path)

            updated_files.append(info)
            total_duration += info.duration_ms

        source.audio_files = updated_files
        source.total_duration_ms = total_duration

    def _collect_chapters(self, source: AudiobookSource) -> list[Chapter]:
        """Collect chapters from source files, preserving existing chapter markers."""
        all_chapters = []
        current_offset = 0

        for audio in source.audio_files:
            # Probe for embedded chapters
            probe_data = self.converter.probe_file(audio.path)
            file_chapters = self.converter.extract_chapters_from_probe(probe_data)

            if file_chapters:
                # Offset chapters to account for previous files
                for ch in file_chapters:
                    all_chapters.append(
                        Chapter(
                            title=ch.title,
                            start_ms=ch.start_ms + current_offset,
                            end_ms=(ch.end_ms + current_offset) if ch.end_ms else None,
                        )
                    )
            else:
                # Create a chapter for this file
                title = audio.title or audio.path.stem
                all_chapters.append(
                    Chapter(
                        title=title,
                        start_ms=current_offset,
                        end_ms=current_offset + audio.duration_ms,
                    )
                )

            current_offset += audio.duration_ms

        return all_chapters

    def _get_cover(self, source: AudiobookSource) -> bytes | None:
        """Get cover art from files or directory."""
        # Try folder image first
        cover_path = self.metadata_extractor.find_cover_file(source)
        if cover_path:
            return cover_path.read_bytes()

        # Try embedded cover from first audio file
        if source.audio_files:
            return self.metadata_extractor.extract_cover(source.audio_files[0].path)

        return None

    def _get_ai_decision(self, source: AudiobookSource) -> AudioDecision:
        """Get AI advisor's decision on how to process this audiobook."""
        if not self.advisor:
            raise RuntimeError("AI advisor not initialized")

        # Get codec info from first file (representative sample)
        first_file = source.audio_files[0]
        probe_data = self.converter.probe_file(first_file.path)
        audio_stream = next(
            (s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"),
            {},
        )

        codec_info = {
            "codec": audio_stream.get("codec_name", "unknown"),
            "channels": audio_stream.get("channels", 2),
            "sample_rate": int(audio_stream.get("sample_rate", 44100)),
        }

        # Build analysis
        analysis = self.advisor.analyze_source(source, codec_info)

        # Get decision from AI
        return self.advisor.get_decision(analysis)
