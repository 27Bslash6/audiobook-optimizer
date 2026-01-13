# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
uv sync                                    # Install dependencies
uv run audiobook-optimizer --help          # CLI help
uv run pytest                              # Run all tests
uv run pytest tests/test_domain.py -v      # Single test file
uv run pytest -k "test_clean_name"         # Tests matching pattern
uv run ruff check src/ tests/              # Lint
uv run ruff format src/ tests/             # Format
uv run basedpyright src/                   # Type check (strict)
```

## Architecture

Hexagonal architecture with strict dependency flow: **Domain → Ports ← Adapters ← Services ← CLI**

```
cli.py                    # Typer commands, coordinates processor
├── config.py             # Pydantic-settings (env: AUDIOBOOK_*, .env)
├── services/processor.py # Orchestration pipeline (no business logic here)
└── adapters/             # External integrations
    ├── ffmpeg.py         # FFmpeg/FFprobe with cachekit caching
    ├── filesystem.py     # Scanner, metadata extractor, organizer
    ├── tagger.py         # Mutagen for M4B tagging
    ├── ai_batch.py       # PydanticAI batch verification (Claude API)
    └── ai_metadata.py    # Single-item AI verification

ports/interfaces.py       # Abstract contracts (AudioScanner, AudioConverter, etc.)
domain/models.py          # Pure dataclasses, no external deps
```

**Key invariants:**
- Domain models never import from adapters
- All FFmpeg/external tools go through adapters
- Services orchestrate; adapters do the work

## Critical Behaviors

**Smart encoding** - The converter (`adapters/ffmpeg.py`) decides remux vs transcode:
- AAC mono sources → stream copy (no quality loss, fast)
- MP3/FLAC/stereo → transcode to AAC (64kbps mono default)
- Never upscale: `min(source_bitrate, target_bitrate)` prevents wasting space

**Metadata inference** (`adapters/filesystem.py`):
- Series patterns: `"Discworld 01 - Title"` → series + number + title
- Author patterns: `"Author - Title"`, `"Title by Author"`
- Parent folder detection: `/Terry Pratchett/Book/` → author = Terry Pratchett
- Cleanup: strips `[audiobook]`, `by uploader`, quality tags like `(32k){179mb}`

**Skip detection** - Uses word-boundary patterns to avoid processing:
- EPUBs, ebooks, emulators, soundtracks/OSTs
- Located in `SKIP_PATTERNS` (filesystem.py:21)

## Caching

Uses `cachekit` for L1 (memory) + optional L2 (Redis):
- `@cache(namespace="ffprobe")` - ffprobe results keyed by path+mtime
- `@cache(ttl=86400*7, namespace="ai_verify")` - AI verification (7 days)
- Set `CACHEKIT_REDIS_URL=redis://...` for L2, otherwise L1-only
- Debug with `--debug/-vv` flag to see hit/miss stats

## Configuration

Environment variables (prefix `AUDIOBOOK_`):
```
AUDIOBOOK_BITRATE       # Default: 64 (kbps)
AUDIOBOOK_MONO          # Default: true
AUDIOBOOK_AI_ENABLED    # Auto-detected from ANTHROPIC_API_KEY
```

Note: Source/output paths are CLI arguments, not env vars.

Supports `.env` file in project root.

## Testing Patterns

```python
# Mock external deps, not domain logic
from unittest.mock import MagicMock

converter = FFmpegConverter.__new__(FFmpegConverter)  # Skip __init__ validation
converter.ffprobe = "ffprobe"
mock_files = [MagicMock(bitrate=24), MagicMock(bitrate=32)]
```

Test files mirror src: `tests/test_domain.py` → `domain/models.py`

## Common Extension Points

**Adding a new audio format:**
1. Add to `AudioFormat` enum (domain/models.py)
2. Add extension to `AUDIO_EXTENSIONS` (filesystem.py)
3. Update `_can_remux` logic if codec can stream-copy (ffmpeg.py)

**Adding metadata patterns:**
1. Add regex to `SERIES_PATTERNS` or `AUTHOR_PATTERNS` (filesystem.py)
2. Add test case in `tests/test_domain.py`

**New CLI command:**
1. Add `@app.command()` function in cli.py
2. Use `console.print()` for output, `Progress()` for long ops

## External Dependencies

**Required:** `ffmpeg`, `ffprobe` in PATH
**Optional:** `claude` CLI (for legacy AI verification), `ANTHROPIC_API_KEY` (for PydanticAI)
