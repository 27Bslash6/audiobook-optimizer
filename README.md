# audiobook-optimizer

Convert scattered audiobook files (MP3, M4A, FLAC) into organized M4B audiobooks with chapters and metadata.

## Features

- **Smart encoding**: AAC sources remux (no quality loss), others transcode to 64kbps mono
- **Series detection**: Parses folder names like `Discworld 01 - The Colour of Magic`
- **Chapter preservation**: Keeps source chapters or creates one per file
- **Cover art**: Embeds from `folder.jpg` or extracted from source files
- **AI metadata verification**: Optional Claude-powered title/author correction with batch processing
- **Intelligent caching**: Redis-backed caching for FFmpeg and AI results to speed up repeat operations
- **Quality advisor**: AI-driven decisions on optimal encoding settings

## Requirements

- Python 3.13+
- `ffmpeg` and `ffprobe` in PATH
- `ANTHROPIC_API_KEY` environment variable (optional, for AI features)
- Redis (optional, for L2 caching - set `CACHEKIT_REDIS_URL`)

## Install

```bash
git clone https://github.com/YOUR_USERNAME/audiobook-optimizer.git
cd audiobook-optimizer
cp .env.example .env  # Edit with your paths and optional API keys
uv sync
```

### Optional AI Setup

For AI metadata verification, set your Anthropic API key:

```bash
# In .env file
ANTHROPIC_API_KEY=your_api_key_here

# Or export in shell
export ANTHROPIC_API_KEY=your_api_key_here
```

### Optional Caching Setup

For persistent caching across runs (recommended for large libraries):

```bash
# Install and start Redis
sudo apt install redis-server  # Debian/Ubuntu
brew install redis             # macOS
redis-server

# Set in .env file
CACHEKIT_REDIS_URL=redis://localhost:6379
```

## Usage

```bash
# Scan for audiobooks (shows what would be processed)
uv run audiobook-optimizer scan /path/to/downloads

# Scan with detailed info (bitrate, size, remux/transcode status, AI corrections)
uv run audiobook-optimizer scan /path/to/downloads -v

# Process audiobooks to M4B (AI auto-enabled if API key set)
uv run audiobook-optimizer process /path/to/downloads /path/to/output

# Process with specific AI setting
uv run audiobook-optimizer process /path/to/downloads /path/to/output --ai
uv run audiobook-optimizer process /path/to/downloads /path/to/output --no-ai

# Preview without processing
uv run audiobook-optimizer process /path/to/downloads /path/to/output --dry-run

# Inspect single audiobook with AI verification
uv run audiobook-optimizer info /path/to/audiobook/folder --ai-verify

# AI-verify metadata for all discovered audiobooks
uv run audiobook-optimizer verify /path/to/downloads

# Enable debug output (cache statistics, performance metrics)
uv run audiobook-optimizer --debug scan /path/to/downloads -v
```

### Global Options

- `--ai/--no-ai`: Enable/disable AI metadata verification (default: auto-detect from API key)
- `--debug`: Show cache performance and debugging information at end of operations

## Architecture

Hexagonal architecture with clean separation:

```
domain/models.py      - AudiobookSource, Chapter, AudiobookMetadata
ports/interfaces.py   - AudioScanner, AudioConverter, AudioTagger contracts
adapters/             - FFmpeg, Mutagen, Filesystem, AI implementations
services/processor.py - Pipeline orchestration
```

### AI Integration

- **Batch verification**: Processes multiple audiobooks in single API calls
- **Intelligent caching**: Redis-backed cache for both FFmpeg probe results and AI responses
- **Quality advisor**: AI analyzes source quality to suggest optimal encoding settings
- **Automatic fallback**: Works without AI - gracefully degrades to rule-based processing

### Performance Features

- **L1 Cache**: In-process memory cache for current session
- **L2 Cache**: Optional Redis cache for persistent storage across runs
- **Batch processing**: Reduces API calls by grouping audiobook verification
- **Smart probing**: Caches FFmpeg probe results to avoid repeated file analysis

## Output Format

- Single `.m4b` file per book
- Embedded chapters, cover art, and metadata
- 64kbps mono AAC (optimal for speech, ~50% smaller than 128kbps MP3)
- Organized as `Author - Title.m4b` or `Series 01 - Title.m4b`
