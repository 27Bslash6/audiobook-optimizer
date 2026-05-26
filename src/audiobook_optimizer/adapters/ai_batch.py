"""Batch AI verification for audiobook metadata and quality decisions.

Sends all audiobooks in one API call for efficient verification.
Uses deterministic results as baseline, AI validates/tweaks.
Results are cached to avoid redundant API calls.
"""

from cachekit import cache
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from audiobook_optimizer.config import ai_available, get_settings
from audiobook_optimizer.domain.models import AudiobookMetadata, AudiobookSource


class AudiobookVerification(BaseModel):
    """Verification result for a single audiobook."""

    index: int = Field(description="Index in the input list (0-based)")
    title: str = Field(description="Corrected title")
    author: str = Field(description="Corrected author (FirstName LastName format)")
    series: str | None = Field(default=None, description="Series name if part of a series")
    series_number: float | None = Field(default=None, description="Position in series")
    narrator: str | None = Field(
        default=None,
        description=(
            "Narrator/reader name if confidently known. Often the embedded `composer` tag — leave null when not confident."
        ),
    )
    year: int | None = Field(
        default=None,
        description=(
            "Original publication year (4-digit) if confidently known. The "
            "embedded `date` tag is usually the recording year — leave null "
            "when ambiguous."
        ),
    )
    quality_ok: bool = Field(default=True, description="Whether quality settings are acceptable")
    quality_note: str | None = Field(default=None, description="Note about quality if not ok")
    changes: list[str] = Field(default_factory=list, description="List of changes made")


class BatchVerificationResult(BaseModel):
    """Result of batch verification."""

    # default_factory=list lets the LLM omit `audiobooks` when it judges that
    # no corrections are needed — without this, pydantic raises "Field required"
    # on perfectly valid "summary-only" responses and the entire AI verification
    # falls through to filename-based metadata. See lab task #19.
    audiobooks: list[AudiobookVerification] = Field(default_factory=list, description="Verified audiobooks")
    summary: str = Field(description="Brief summary of verification")


_INSTRUCTIONS = """You are an audiobook metadata expert. You will receive a batch of audiobooks
with their inferred metadata, embedded source tags, and quality settings. Your job is to:

1. VERIFY metadata is correct (title, author, series, narrator, year)
2. FIX obvious errors (swapped author/title, missing series detection, typos,
   leading track numbers like "01" baked into titles, quality/size tags in titles)
3. VALIDATE quality decisions (bitrate choices)

METADATA RULES:
- Author format: "FirstName LastName" (not "LastName, FirstName")
- Series names: clean and consistent ("Discworld", not "Discworld Series",
  not "Discworld (Books)"). Use "The Hitchhiker's Guide to the Galaxy" (with
  the leading "The" and the apostrophe) for that series, not "Hitchhikers...".
- Remove quality/size tags ("(Stevens) 32k 12.58.21 {179mb}"), bitrate
  ({179mb}, 64k), and disc/track prefixes from titles. Example:
  "01Hitchhiker's Guide to the Galaxy (audiobook)" -> "The Hitchhiker's Guide to the Galaxy".
- Remove year prefixes ("2008 - ...") unless that year is part of the actual title.
- Detect series from context: numbered prefixes ("Book 1", "Vol. 2"),
  parent directory names, and the embedded `album` tag (often holds the series
  name when the book is part of one).

USING EMBEDDED TAGS:
- The embedded audio tags (artist, albumartist, album, composer, date,
  comment, genre, title) are the richest signal — prefer them over folder-name
  guesses when they are sane.
- The embedded `composer` tag is FREQUENTLY the narrator — return it as
  `narrator` when it is a plausible person name distinct from the author.
- The embedded `date` tag is usually the (re)recording year — return it as
  `year` only when it is a 4-digit year and plausible for that work.
- Embedded tags CAN be wrong: a re-recording or a mislabeled rip may carry
  the wrong artist/album (e.g. a Hitchhiker's rip tagged with Stephen Fry
  or Martin Freeman as composer when the user wants the Douglas-Adams-narrated
  edition). Sanity-check against the author's known works — if the embedded
  artist/album/composer contradicts a famous title, trust the title.
- Return `narrator` and `year` ONLY when reasonably confident; leave null
  otherwise (downstream code falls back to the existing metadata value).

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
        parent = item.get("parent_folder")
        head = f"[{item['index']}] {item['folder']}"
        if parent:
            head += f"  (parent dir: {parent})"
        lines.append(head)
        files_preview = ", ".join(item["files"][:3]) + ("..." if len(item["files"]) > 3 else "")
        files_line = f"    Files: {item['file_count']} ({files_preview})"
        if item.get("total_hours") is not None:
            files_line += f", ~{item['total_hours']}h total"
        lines.append(files_line)
        tags = item.get("embedded_tags") or {}
        present = {k: v for k, v in tags.items() if v}
        if present:
            lines.append("    Embedded tags: " + "; ".join(f"{k}={v}" for k, v in present.items()))
        lines.append(f'    Inferred: "{item["inferred"]["title"]}" by {item["inferred"]["author"]}')
        if item["inferred"]["series"]:
            lines.append(f"    Series: {item['inferred']['series']} #{item['inferred']['series_number']}")
        q = item["quality"]
        lines.append(
            f"    Quality: {q.get('action', 'transcode')} → {q.get('effective_bitrate', '?')}kbps "
            f"(source: {q.get('bitrate', '?')}kbps)"
        )
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
    # Build indexed copies (don't mutate caller's dicts)
    indexed_items = [{"index": i, **item} for i, item in enumerate(items)]

    prompt = f"""Verify this batch of {len(indexed_items)} audiobooks:

{_format_items(indexed_items)}

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

    def __init__(self, model: str | None = None):
        """Initialize verifier.

        Raises:
            RuntimeError: If ANTHROPIC_API_KEY not set.
        """
        if not ai_available():
            raise RuntimeError("ANTHROPIC_API_KEY required for AI verification")
        self._model = model or get_settings().ai_model

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

        # Local extractor: reuses mutagen tag-reading from FilesystemMetadataExtractor
        # so the prompt carries the richest available signal (artist/album/composer/date/...).
        # Local import to avoid a circular module-load dependency.
        from audiobook_optimizer.adapters.filesystem import FilesystemMetadataExtractor

        extractor = FilesystemMetadataExtractor()

        # Build items list - cachekit will JSON-serialize for cache key. Changing
        # the dict shape (parent_folder, total_hours, embedded_tags) naturally
        # invalidates stale cache entries from the pre-enrichment build.
        items = []
        for source, metadata, quality in audiobooks:
            tags = extractor.read_embedded_tags(source)
            # Prefer the AudiobookSource.total_duration_ms when populated; fall back
            # to summing individual files. Both are in milliseconds.
            total_ms = source.total_duration_ms or sum(f.duration_ms for f in source.audio_files if f.duration_ms)
            total_hours = round(total_ms / 3_600_000, 1) if total_ms else None
            # Truncate long tag values (comment can be a multi-paragraph description)
            # to keep prompt size + cache key bounded.
            safe_tags = {k: (v[:300] if isinstance(v, str) else v) for k, v in tags.items()}
            items.append(
                {
                    "folder": source.source_path.name,
                    "parent_folder": source.source_path.parent.name,
                    "files": [f.path.name for f in source.audio_files[:5]],
                    "file_count": len(source.audio_files),
                    "total_hours": total_hours,
                    "embedded_tags": safe_tags,
                    "inferred": {
                        "title": metadata.title,
                        "author": metadata.author,
                        "series": metadata.series,
                        "series_number": metadata.series_number,
                    },
                    "quality": quality,
                }
            )

        # Call cached function
        result_dicts = _verify_batch_cached(items, self._model)

        # Convert back to AudiobookVerification objects
        return {d["index"]: AudiobookVerification(**d) for d in result_dicts}


def apply_verification(
    metadata: AudiobookMetadata,
    verification: AudiobookVerification,
) -> AudiobookMetadata:
    """Apply AI verification to metadata, returning updated copy.

    AI-returned `narrator` / `year` win when present, otherwise fall back to
    the inferred values — never overwrite known data with null.
    """
    return AudiobookMetadata(
        title=verification.title,
        author=verification.author,
        series=verification.series,
        series_number=verification.series_number,
        narrator=verification.narrator or metadata.narrator,
        year=verification.year or metadata.year,
        description=metadata.description,
        genre=metadata.genre,
        cover_path=metadata.cover_path,
    )
