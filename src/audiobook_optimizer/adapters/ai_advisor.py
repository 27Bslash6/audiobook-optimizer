"""AI-powered audio quality advisor using pydantic-ai.

Uses Claude Haiku for cost-effective, fast decisions about audio processing.
Requires ANTHROPIC_API_KEY environment variable (or .env file).
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from audiobook_optimizer.domain.models import AudiobookSource, AudioFormat

# Load .env from project root if it exists
_env_file = Path(__file__).parent.parent.parent.parent / ".env"
if _env_file.exists():
    load_dotenv(_env_file)


class AdvisorError(Exception):
    """Error from AI advisor."""


class ProcessingAction(str, Enum):
    """Recommended processing action."""

    REMUX = "remux"  # Stream copy, no quality loss
    TRANSCODE = "transcode"  # Re-encode to target bitrate
    SKIP = "skip"  # Don't process (already optimal or problematic)


class AudioDecision(BaseModel):
    """Structured output for AI audio quality decisions."""

    action: ProcessingAction = Field(
        description="The recommended processing action"
    )
    target_bitrate_kbps: int = Field(
        ge=32,
        le=320,
        description="Recommended target bitrate in kbps (32-320)",
    )
    preserve_stereo: bool = Field(
        default=False,
        description="Whether to preserve stereo (False = convert to mono for speech)",
    )
    quality_warnings: list[str] = Field(
        default_factory=list,
        description="Any quality concerns the user should know about",
    )
    reasoning: str = Field(
        description="Brief explanation of the decision",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this recommendation (0-1)",
    )


@dataclass
class AudioAnalysis:
    """Audio characteristics for AI analysis."""

    source_format: str
    source_codec: str
    source_bitrate_kbps: int
    source_channels: int
    source_sample_rate: int
    total_duration_hours: float
    file_count: int
    total_size_mb: float
    has_chapters: bool
    is_homogeneous: bool  # All files same format/codec


class AudioQualityAdvisor:
    """AI advisor for audio processing decisions using pydantic-ai.

    Uses Claude Haiku for fast, cost-effective decisions about:
    - Whether to remux or transcode
    - Optimal target bitrate
    - Stereo vs mono for audiobooks
    - Quality concerns
    """

    def __init__(self, model: str = "anthropic:claude-3-5-haiku-latest"):
        """Initialize advisor with specified model.

        Args:
            model: pydantic-ai model string (default: claude-haiku-4.5)

        Raises:
            AdvisorError: If ANTHROPIC_API_KEY environment variable is not set.
        """
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise AdvisorError(
                "ANTHROPIC_API_KEY required for AI advisor. "
                "Set env var or create .env file. Use --ai-verify instead (Claude CLI) if unavailable."
            )

        self._agent = Agent(
            model,
            output_type=AudioDecision,
            instructions=self._build_instructions(),
        )

    def _build_instructions(self) -> str:
        """System instructions for the audio advisor."""
        return """You are an audio quality expert helping optimize audiobooks.

CORE PRINCIPLES:
- Audiobooks are SPEECH, not music. 64-96kbps AAC is typically transparent for speech.
- The user is sensitive to audio quality - prioritize quality over file size.
- Remuxing (stream copy) preserves original quality with zero generation loss.
- Transcoding introduces quality loss - only do it when necessary.

DECISION FRAMEWORK:
1. REMUX when:
   - Source is already AAC at acceptable bitrate (64-128kbps)
   - Source is ALAC (lossless) - though consider transcoding to save space
   - Quality would be lost by transcoding

2. TRANSCODE when:
   - Source is MP3 (transcode to AAC for better M4B compatibility)
   - Source bitrate is excessive for speech (>160kbps wastes space)
   - Source is stereo but mono would suffice (halves bitrate)
   - Source is low quality (below 48kbps) - flag this as a warning

3. SKIP when:
   - Source is already M4B at good quality
   - Files are corrupted or problematic

BITRATE GUIDELINES for AAC audiobooks:
- 48kbps mono: Minimum acceptable, slight artifacts possible
- 64kbps mono: Good quality for speech, recommended minimum
- 96kbps mono: Excellent quality, recommended for quality-conscious users
- 128kbps mono: Overkill for speech but safe choice
- Stereo: Double these values

QUALITY WARNINGS to flag:
- Source below 64kbps (already compressed, quality loss likely)
- Upsampling detected (e.g., 64kbps source to 128kbps target = waste)
- Mixed quality sources (inconsistent listening experience)
- Lossy-to-lossy transcoding (quality degradation)"""

    def analyze_source(self, source: AudiobookSource, codec_info: dict[str, str | int]) -> AudioAnalysis:
        """Extract audio characteristics from an AudiobookSource.

        Args:
            source: The audiobook source to analyze
            codec_info: Dict with 'codec', 'channels', 'sample_rate' from ffprobe

        Returns:
            AudioAnalysis with source characteristics
        """
        # Get primary format
        primary_format = source.primary_format

        # Calculate totals
        total_size_bytes = source.total_size_bytes
        total_duration_ms = source.total_duration_ms

        # Get average bitrate across files
        bitrates = [f.bitrate for f in source.audio_files if f.bitrate]
        avg_bitrate = sum(bitrates) // len(bitrates) if bitrates else 0

        # Check if all files are same format
        formats = {f.format for f in source.audio_files}
        is_homogeneous = len(formats) == 1

        # Check for existing chapters
        has_chapters = any(f.chapters for f in source.audio_files)

        return AudioAnalysis(
            source_format=primary_format.value,
            source_codec=str(codec_info.get("codec", "unknown")),
            source_bitrate_kbps=avg_bitrate,
            source_channels=int(codec_info.get("channels", 2)),
            source_sample_rate=int(codec_info.get("sample_rate", 44100)),
            total_duration_hours=total_duration_ms / (1000 * 60 * 60),
            file_count=source.file_count,
            total_size_mb=total_size_bytes / (1024 * 1024),
            has_chapters=has_chapters,
            is_homogeneous=is_homogeneous,
        )

    def get_decision(self, analysis: AudioAnalysis) -> AudioDecision:
        """Get AI decision for how to process the audio.

        Args:
            analysis: AudioAnalysis with source characteristics

        Returns:
            AudioDecision with recommended action, bitrate, and warnings
        """
        prompt = f"""Analyze this audiobook and recommend processing:

SOURCE CHARACTERISTICS:
- Format: {analysis.source_format}
- Codec: {analysis.source_codec}
- Bitrate: {analysis.source_bitrate_kbps} kbps
- Channels: {analysis.source_channels} ({'stereo' if analysis.source_channels > 1 else 'mono'})
- Sample rate: {analysis.source_sample_rate} Hz
- Duration: {analysis.total_duration_hours:.1f} hours
- Files: {analysis.file_count}
- Total size: {analysis.total_size_mb:.1f} MB
- Has embedded chapters: {analysis.has_chapters}
- Homogeneous format: {analysis.is_homogeneous}

TARGET: M4B (AAC in MP4 container) optimized for audiobook playback.

Recommend the best processing approach considering:
1. Should we REMUX (stream copy) or TRANSCODE?
2. What target bitrate optimizes quality vs size?
3. Should we keep stereo or convert to mono?
4. Any quality concerns the user should know about?

Remember: User is quality-sensitive. When in doubt, preserve quality."""

        result = self._agent.run_sync(prompt)
        return result.output

    def get_decision_async(self, analysis: AudioAnalysis):
        """Async version of get_decision for pipeline integration."""
        prompt = f"""Analyze this audiobook and recommend processing:

SOURCE CHARACTERISTICS:
- Format: {analysis.source_format}
- Codec: {analysis.source_codec}
- Bitrate: {analysis.source_bitrate_kbps} kbps
- Channels: {analysis.source_channels} ({'stereo' if analysis.source_channels > 1 else 'mono'})
- Sample rate: {analysis.source_sample_rate} Hz
- Duration: {analysis.total_duration_hours:.1f} hours
- Files: {analysis.file_count}
- Total size: {analysis.total_size_mb:.1f} MB
- Has embedded chapters: {analysis.has_chapters}
- Homogeneous format: {analysis.is_homogeneous}

TARGET: M4B (AAC in MP4 container) optimized for audiobook playback.

Recommend the best processing approach."""

        return self._agent.run(prompt)


def create_advisor(model: str | None = None) -> AudioQualityAdvisor:
    """Factory function to create an AudioQualityAdvisor.

    Args:
        model: Optional model override (default: claude-haiku-4.5)

    Returns:
        Configured AudioQualityAdvisor instance
    """
    if model:
        return AudioQualityAdvisor(model=model)
    return AudioQualityAdvisor()
