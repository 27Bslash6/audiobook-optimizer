"""FFmpeg adapter for audio conversion and probing."""

import json
import subprocess
import tempfile
from pathlib import Path

from audiobook_optimizer.domain.models import AudioFile, AudioFormat, Chapter
from audiobook_optimizer.ports.interfaces import AudioConverter


class FFmpegError(Exception):
    """FFmpeg operation failed."""


class FFmpegConverter(AudioConverter):
    """Audio converter using ffmpeg/ffprobe."""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path
        self._verify_tools()

    def _verify_tools(self) -> None:
        """Verify ffmpeg and ffprobe are available."""
        for tool in [self.ffmpeg, self.ffprobe]:
            try:
                subprocess.run([tool, "-version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise FFmpegError(f"{tool} not found or not working: {e}") from e

    def probe_duration(self, path: Path) -> int:
        """Get duration in milliseconds using ffprobe."""
        cmd = [
            self.ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration_sec = float(data["format"]["duration"])
        return int(duration_sec * 1000)

    def probe_file(self, path: Path) -> dict:
        """Get full probe info for a file."""
        cmd = [
            self.ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def extract_chapters_from_probe(self, probe_data: dict) -> list[Chapter]:
        """Extract chapters from ffprobe data."""
        chapters = []
        for ch in probe_data.get("chapters", []):
            # ffprobe gives time in seconds as float
            start_ms = int(float(ch["start_time"]) * 1000)
            end_ms = int(float(ch["end_time"]) * 1000)
            title = ch.get("tags", {}).get("title", f"Chapter {len(chapters) + 1}")
            chapters.append(Chapter(title=title, start_ms=start_ms, end_ms=end_ms))
        return chapters

    def _can_remux(self, source_files: list[AudioFile], mono: bool) -> tuple[bool, str]:
        """Check if source files can be remuxed (stream copied) without re-encoding.

        Returns (can_remux, reason).
        Remux is possible when all files are AAC with compatible channels.
        """
        if not source_files:
            return False, "no files"

        codecs = set()
        channels = set()

        for audio in source_files:
            probe = self.probe_file(audio.path)
            audio_stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), None)
            if not audio_stream:
                return False, f"no audio stream in {audio.path.name}"

            codec = audio_stream.get("codec_name", "unknown")
            codecs.add(codec)
            channels.add(audio_stream.get("channels", 0))

        # Check codec homogeneity
        if len(codecs) > 1:
            return False, f"mixed codecs: {codecs}"

        codec = codecs.pop()
        # Only AAC can be remuxed into M4B (MP4 container)
        if codec not in ("aac", "alac"):
            return False, f"codec {codec} requires transcoding to AAC"

        # Check channel compatibility
        if mono and channels != {1}:
            return False, "source is stereo, mono requested"

        return True, "remux possible"

    def convert_to_m4b(
        self,
        source_files: list[AudioFile],
        output_path: Path,
        chapters: list[Chapter] | None = None,
        bitrate: int = 64,
        mono: bool = True,
    ) -> tuple[Path, bool]:
        """Convert multiple audio files to single M4B.

        Uses ffmpeg concat demuxer for gapless concatenation.
        Prefers remuxing (stream copy) when source is already AAC.

        Returns (output_path, was_remuxed).
        """
        if not source_files:
            raise FFmpegError("No source files provided")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build chapter list if not provided (one chapter per source file)
        if chapters is None:
            chapters = self._build_chapters_from_files(source_files)

        # Check if we can remux instead of re-encoding
        can_remux, remux_reason = self._can_remux(source_files, mono)

        # Step 1: Create concat list file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            concat_file = Path(f.name)
            for audio in source_files:
                # Escape single quotes in path for ffmpeg concat
                escaped = str(audio.path).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        # Step 2: Create metadata file with chapters
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            metadata_file = Path(f.name)
            f.write(";FFMETADATA1\n")
            for ch in chapters:
                f.write("\n[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={ch.start_ms}\n")
                if ch.end_ms is not None:
                    f.write(f"END={ch.end_ms}\n")
                # Escape special chars in title
                title = ch.title.replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\\", "\\\\")
                f.write(f"title={title}\n")

        try:
            # Step 3: Build ffmpeg command
            cmd = [
                self.ffmpeg,
                "-y",  # Overwrite output
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-i", str(metadata_file),
                "-map_metadata", "1",
                "-map", "0:a",  # Only audio from concat input
            ]

            if can_remux:
                # Stream copy - no re-encoding
                cmd.extend(["-c:a", "copy"])
            else:
                # Re-encode to AAC
                target_channels = 1 if mono else 2
                cmd.extend([
                    "-c:a", "aac",
                    "-b:a", f"{bitrate}k",
                    "-ac", str(target_channels),
                ])

            cmd.extend(["-f", "ipod", str(output_path)])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise FFmpegError(f"ffmpeg failed: {result.stderr}")

            return output_path, can_remux

        finally:
            # Cleanup temp files
            concat_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)

    def _build_chapters_from_files(self, files: list[AudioFile]) -> list[Chapter]:
        """Build chapter list with one chapter per source file."""
        chapters = []
        current_ms = 0

        for i, audio in enumerate(files, 1):
            # Use file's title or generate from filename
            title = audio.title or audio.path.stem
            # Clean up common patterns
            title = title.replace("_", " ").strip()

            chapters.append(Chapter(
                title=title,
                start_ms=current_ms,
                end_ms=current_ms + audio.duration_ms,
            ))
            current_ms += audio.duration_ms

        return chapters

    def embed_cover_art(self, m4b_path: Path, cover_path: Path) -> Path:
        """Embed cover art into M4B file.

        Creates a new file with cover embedded, then replaces original.
        FFmpeg with ipod format: -map 1:0 -c:v mjpeg -disposition:v:0 attached_pic
        """
        temp_output = m4b_path.with_suffix(".tmp.m4b")

        cmd = [
            self.ffmpeg,
            "-y",
            "-i", str(m4b_path),
            "-i", str(cover_path),
            "-map", "0:a",
            "-map", "1:0",
            "-c:a", "copy",  # Don't re-encode audio
            "-c:v", "mjpeg",
            "-disposition:v:0", "attached_pic",
            "-f", "ipod",
            str(temp_output),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            temp_output.unlink(missing_ok=True)
            raise FFmpegError(f"Failed to embed cover: {result.stderr}")

        # Replace original with new file
        temp_output.replace(m4b_path)
        return m4b_path
