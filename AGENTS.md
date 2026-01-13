# AGENTS.md - Development Guidelines

This file contains conventions and commands for agentic coding agents working in this audiobook-optimizer repository.

## Build/Test/Lint Commands

```bash
# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_domain.py

# Run specific test method
uv run pytest tests/test_domain.py::TestAudioFormat::test_from_extension

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=audiobook_optimizer

# Lint code (ruff)
uv run ruff check src/ tests/

# Format code (ruff format)
uv run ruff format src/ tests/

# Type checking (basedpyright)
uv run basedpyright src/

# Install dependencies
uv sync

# Run the CLI
uv run audiobook-optimizer --help
```

## Project Architecture

This is a Python audiobook processing application using **hexagonal/clean architecture**:

```
domain/models.py      - Core business logic and data models (pure Python)
ports/interfaces.py   - Abstract contracts for external dependencies
adapters/            - Concrete implementations (ffmpeg, filesystem, mutagen, AI)
services/processor.py - Orchestration and pipeline logic
cli.py               - Typer-based command-line interface
config.py            - Pydantic settings management
```

**Key principles:**
- Domain models have NO external dependencies
- All external integrations go through adapters
- Ports define interfaces, adapters implement them
- Services orchestrate but don't contain business logic

## Code Style Guidelines

### Python Version & Formatting
- **Python 3.13+** required
- Line length: **129 characters** (configured in ruff)
- Use `uv run ruff format` for auto-formatting
- Use `uv run ruff check` for linting

### Type Hints
- **Standard type checking** enabled (basedpyright)
- Use `|` for union types, not `typing.Union`
- All function parameters and return values must be typed
- Use `list[str]` instead of `List[str]` (Python 3.9+ syntax)

### Imports
- Use `isort` conventions (handled by ruff)
- Group imports: stdlib, third-party, local
- Use absolute imports for local modules: `from audiobook_optimizer.domain.models import AudiobookSource`

### Naming Conventions
- **Classes**: PascalCase (e.g., `AudiobookProcessor`, `AudioFile`)
- **Functions/variables**: snake_case (e.g., `scan_directory`, `audio_files`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `AUDIO_EXTENSIONS`, `SERIES_PATTERNS`)
- **Private methods**: underscore prefix (e.g., `_clean_name`, `_populate_file_info`)
- **Enum members**: UPPER_SNAKE_CASE (e.g., `ProcessingStatus.COMPLETED`)

### Domain Models
- Use `@dataclass` for pure data classes
- Use `@property` for computed attributes
- Enums inherit from `str` when appropriate for serialization
- Domain models must NOT import from adapters

### Error Handling
- Use specific exception classes in adapters (e.g., `FFmpegError`, `TaggerError`)
- Raise descriptive exceptions with context
- Handle external service failures gracefully
- Log warnings with `rich.console.print()` for CLI output

### File Organization
- Each module has a single responsibility
- Keep interfaces in `ports/` directory
- Implementations in `adapters/` directory
- Tests in `tests/` directory, mirroring src structure

### CLI Code (cli.py)
- Use `typer` for command definition
- Use `rich` for formatted output
- Group related functionality
- Provide helpful error messages
- Use progress indicators for long operations

### Configuration
- Use `pydantic-settings` for environment-based config
- Prefix env vars with `AUDIOBOOK_` (e.g., `AUDIOBOOK_BITRATE`)
- Support `.env` file for local development
- Provide sensible defaults in settings

### Testing
- Use `pytest` with parametrized tests for multiple cases
- Mock external dependencies (ffmpeg, filesystem)
- Test domain logic separately from integration
- Use descriptive test method names
- Test both happy path and error conditions

### Dependencies
- **Core**: Pydantic, Typer, Rich, Mutagen, Python-dotenv
- **AI**: PydanticAI (optional), Claude CLI (optional)
- **External**: FFmpeg/FFprobe (must be in PATH)
- **Testing**: pytest

## Common Patterns

### Adapters Pattern
```python
class FFmpegConverter(AudioConverter):
    def convert_to_m4b(self, ...) -> Path:
        # Implementation details
        pass
```

### Error Handling Pattern
```python
try:
    result = converter.convert_to_m4b(...)
except FFmpegError as e:
    console.print(f"[red]Conversion failed: {e}[/red]")
    raise
```

### CLI Progress Pattern
```python
with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
    task = progress.add_task("Processing...", total=None)
    result = processor.process(source)
```

### Domain Model Pattern
```python
@dataclass
class AudiobookSource:
    source_path: Path
    audio_files: list[AudioFile]
    metadata: AudiobookMetadata | None = None
    
    @property
    def file_count(self) -> int:
        return len(self.audio_files)
```

## Important Notes

- **Never commit API keys** or `.env` files
- **FFmpeg/FFprobe must be installed** and in PATH
- **AI features are optional** - work without ANTHROPIC_API_KEY
- **Test with both stereo and mono** audio processing
- **Verify bitrate calculations** never upscale low-bitrate sources
- **Maintain hexagonal architecture** - domain stays pure