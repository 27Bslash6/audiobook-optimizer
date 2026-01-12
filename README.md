# audiobook-optimizer

Convert scattered audiobook files (MP3, M4A, FLAC) into organized M4B audiobooks with chapters and metadata.

## Features

- **Smart encoding**: AAC sources remux (no quality loss), others transcode to 64kbps mono
- **Series detection**: Parses folder names like `Discworld 01 - The Colour of Magic`
- **Chapter preservation**: Keeps source chapters or creates one per file
- **Cover art**: Embeds from `folder.jpg` or extracted from source files
- **AI metadata verification**: Optional Claude-powered title/author correction

## Requirements

- Python 3.13+
- `ffmpeg` and `ffprobe` in PATH
- `claude` CLI (optional, for AI features)

## Install

```bash
cd /home/fish/media/data/tools/audiobook-optimizer
uv sync
```

## Usage

```bash
# Scan for audiobooks (shows what would be processed)
uv run audiobook-optimizer scan /path/to/downloads

# Scan with details (bitrate, size, remux/transcode status)
uv run audiobook-optimizer scan /path/to/downloads -v

# Process audiobooks to M4B
uv run audiobook-optimizer process /path/to/downloads /path/to/output

# Preview without processing
uv run audiobook-optimizer process /path/to/downloads /path/to/output --dry-run

# Inspect single audiobook
uv run audiobook-optimizer info /path/to/audiobook/folder

# AI-verify metadata for all discovered audiobooks
uv run audiobook-optimizer verify /path/to/downloads
```

## Architecture

Hexagonal architecture with clean separation:

```
domain/models.py      - AudiobookSource, Chapter, AudiobookMetadata
ports/interfaces.py   - AudioScanner, AudioConverter, AudioTagger contracts
adapters/             - FFmpeg, Mutagen, Filesystem implementations
services/processor.py - Pipeline orchestration
```

## Output Format

- Single `.m4b` file per book
- Embedded chapters, cover art, and metadata
- 64kbps mono AAC (optimal for speech, ~50% smaller than 128kbps MP3)
- Organized as `Author - Title.m4b` or `Series 01 - Title.m4b`
