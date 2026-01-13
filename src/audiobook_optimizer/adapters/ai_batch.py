"""Batch AI verification for audiobook metadata and quality decisions.

Sends all audiobooks in one API call for efficient verification.
Uses deterministic results as baseline, AI validates/tweaks.
Results are cached to avoid redundant API calls.
"""

from pathlib import Path

from cachekit import cache
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from audiobook_optimizer.config import ai_available
from audiobook_optimizer.domain.models import AudiobookMetadata, AudiobookSource

# Load .env from project root
_env_file = Path(__file__).parent.parent.parent.parent / ".env"
if _env_file.exists():
    load_dotenv(_env_file)


class AudiobookVerification(BaseModel):
    """Verification result for a single audiobook."""

    index: int = Field(description="Index in the input list (0-based)")
    title: str = Field(description="Corrected title")
    author: str = Field(description="Corrected author (FirstName LastName format)")
    series: str | None = Field(default=None, description="Series name if part of a series")
    series_number: float | None = Field(default=None, description="Position in series")
    quality_ok: bool = Field(default=True, description="Whether quality settings are acceptable")
    quality_note: str | None = Field(default=None, description="Note about quality if not ok")
    changes: list[str] = Field(default_factory=list, description="List of changes made")


class BatchVerificationResult(BaseModel):
    """Result of batch verification."""

    audiobooks: list[AudiobookVerification] = Field(description="Verified audiobooks")
    summary: str = Field(description="Brief summary of verification")


_INSTRUCTIONS = """You are an audiobook metadata expert. You will receive a batch of audiobooks
with their inferred metadata and quality settings. Your job is to:

1. VERIFY metadata is correct (title, author, series)
2. FIX obvious errors (swapped author/title, missing series detection, typos)
3. VALIDATE quality decisions (bitrate choices)

METADATA RULES:
- Author format: "FirstName LastName" (not "LastName, FirstName")
- Series names: Clean (e.g., "Discworld" not "Discworld Series")
- Remove quality tags from titles: "(Stevens) 32k", "{179mb}", etc.
- Remove year prefixes unless part of actual title
- Detect series from context (e.g., "Book 1", "Vol. 2", numbered titles)

QUALITY RULES:
- Bitrate capping is CORRECT: never upscale (24kbps source → 24kbps output is right)
- Flag if source quality is very low (<32kbps) - user should know
- Remux (stream copy) for AAC sources is preferred

Only include audiobooks in your response that need changes OR have quality notes.
For audiobooks that are perfect, you can omit them (assume unchanged)."""


def _format_items(items: list[dict]) -> str:
    """Format items for prompt."""
    lines = []
    for item in items:
        lines.append(f"[{item['index']}] {item['folder']}")
        lines.append(f"    Files: {item['file_count']} ({', '.join(item['files'][:3])}{'...' if len(item['files']) > 3 else ''})")
        lines.append(f"    Inferred: \"{item['inferred']['title']}\" by {item['inferred']['author']}")
        if item["inferred"]["series"]:
            lines.append(f"    Series: {item['inferred']['series']} #{item['inferred']['series_number']}")
        lines.append(f"    Quality: {item['quality'].get('action', 'transcode')} → {item['quality'].get('effective_bitrate', '?')}kbps (source: {item['quality'].get('bitrate', '?')}kbps)")
        lines.append("")
    return "\n".join(lines)


@cache(ttl=86400 * 7, namespace="ai_verify")
def _verify_batch_cached(items: list[dict], model: str) -> list[dict]:
    """Cached AI verification call. Cachekit auto-hashes the items list.

    Auto-detects Redis from CACHEKIT_REDIS_URL env var, falls back to L1 (memory).

    Args:
        items: List of dicts with folder, files, inferred, quality
        model: Model identifier

    Returns:
        List of verification dicts (serializable)
    """
    # Add indices for prompt
    for i, item in enumerate(items):
        item["index"] = i

    prompt = f"""Verify this batch of {len(items)} audiobooks:

{_format_items(items)}

Return verification results for any audiobooks that need corrections or have quality notes.
Omit audiobooks that are already correct."""

    agent = Agent(
        model,
        output_type=BatchVerificationResult,
        instructions=_INSTRUCTIONS,
    )
    result = agent.run_sync(prompt)

    # Return as list of dicts (serializable for cache)
    return [v.model_dump() for v in result.output.audiobooks]


class BatchAIVerifier:
    """Batch verify audiobook metadata and quality decisions using AI."""

    def __init__(self, model: str = "anthropic:claude-3-5-haiku-latest"):
        """Initialize verifier.

        Raises:
            RuntimeError: If ANTHROPIC_API_KEY not set.
        """
        if not ai_available():
            raise RuntimeError("ANTHROPIC_API_KEY required for AI verification")
        self._model = model

    def verify_batch(
        self,
        audiobooks: list[tuple[AudiobookSource, AudiobookMetadata, dict]],
    ) -> dict[int, AudiobookVerification]:
        """Verify a batch of audiobooks. Results are cached for 7 days.

        Args:
            audiobooks: List of (source, inferred_metadata, quality_info) tuples.
                       quality_info should have: bitrate, effective_bitrate, action

        Returns:
            Dict mapping index to verification result (only for items with changes/notes)
        """
        if not audiobooks:
            return {}

        # Build items list - cachekit will JSON-serialize for cache key
        items = []
        for source, metadata, quality in audiobooks:
            items.append({
                "folder": source.source_path.name,
                "files": [f.path.name for f in source.audio_files[:5]],
                "file_count": len(source.audio_files),
                "inferred": {
                    "title": metadata.title,
                    "author": metadata.author,
                    "series": metadata.series,
                    "series_number": metadata.series_number,
                },
                "quality": quality,
            })

        # Call cached function
        result_dicts = _verify_batch_cached(items, self._model)

        # Convert back to AudiobookVerification objects
        return {d["index"]: AudiobookVerification(**d) for d in result_dicts}


def apply_verification(
    metadata: AudiobookMetadata,
    verification: AudiobookVerification,
) -> AudiobookMetadata:
    """Apply AI verification to metadata, returning updated copy."""
    return AudiobookMetadata(
        title=verification.title,
        author=verification.author,
        series=verification.series,
        series_number=verification.series_number,
        narrator=metadata.narrator,
        year=metadata.year,
        description=metadata.description,
        genre=metadata.genre,
        cover_path=metadata.cover_path,
    )
