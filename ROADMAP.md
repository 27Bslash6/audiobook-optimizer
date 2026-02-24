# Roadmap

Development priorities for audiobook-optimizer, ordered by impact and feasibility.

## Now: Foundation

### Testing
- Integration tests with real audio files
- Adapter mocks (ffmpeg, mutagen, AI APIs)
- CLI scenario coverage
- Error scenarios (corrupt files, network failures, permissions)

### Error Handling
- Retry logic for transient failures (ffmpeg crashes, API rate limits)
- Pre-flight checks (validate tools/permissions before processing)
- Clear error messages with actionable guidance

## Next: Core Enhancements

### Audio Processing
- Silence detection (trim leading/trailing)
- Volume normalization across files
- Quality analysis (detect clipping, low bitrate sources)

### Metadata
- Audible/Goodreads lookup integration
- Better narrator detection from tags/filenames
- Language detection

### Performance
- Parallel audiobook processing
- Incremental processing (skip already-processed)
- Resume interrupted operations

## Later: Maybe

- Custom output naming patterns
- Duplicate detection
- Watch mode for new files
- PyPI package distribution

## Status

### Done
- Core processing pipeline (remux/transcode decision logic)
- AI metadata verification (PydanticAI batch + caching)
- CLI commands: scan, process, info, verify
- Hexagonal architecture
- Two-level caching (cachekit L1/L2)

### Current Focus
- Test coverage expansion
- Error handling improvements
