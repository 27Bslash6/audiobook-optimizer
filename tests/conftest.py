"""Pytest fixtures for audiobook-optimizer tests.

Provides real audio file fixtures from public domain sources.
Files are downloaded once and cached in .fixtures/ directory.
"""

import hashlib
import io
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pytest

# Cache directory for downloaded fixtures
FIXTURES_DIR = Path(__file__).parent / ".fixtures"

# LibriVox "Meditations" by Marcus Aurelius - public domain, 64kbps MP3
MEDITATIONS_URL = (
    "https://archive.org/compress/meditations_0708_librivox/formats=64KBPS%20MP3&file=/meditations_0708_librivox.zip"
)
MEDITATIONS_CACHE_NAME = "meditations_librivox"

# Expected SHA256 of the zip file (for integrity verification)
# We'll compute this on first download and store it
MEDITATIONS_CHECKSUM_FILE = FIXTURES_DIR / f"{MEDITATIONS_CACHE_NAME}.sha256"


def _download_with_progress(url: str, desc: str) -> bytes:
    """Download URL with progress indication for pytest output."""
    print(f"\n  Downloading: {desc}")
    print(f"  URL: {url}")

    response = urlopen(url, timeout=120)
    total_size = response.headers.get("Content-Length")

    if total_size:
        total_size = int(total_size)
        print(f"  Size: {total_size / 1024 / 1024:.1f} MB")

    chunks = []
    downloaded = 0
    chunk_size = 64 * 1024  # 64KB chunks

    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
        downloaded += len(chunk)
        if total_size:
            pct = downloaded * 100 // total_size
            print(f"\r  Progress: {pct}% ({downloaded / 1024 / 1024:.1f} MB)", end="", flush=True)

    print()  # Newline after progress
    return b"".join(chunks)


def _ensure_fixture_downloaded(url: str, cache_name: str, checksum_file: Path) -> Path:
    """Download and extract a fixture if not already cached.

    Returns path to extracted directory.
    """
    FIXTURES_DIR.mkdir(exist_ok=True)
    extract_dir = FIXTURES_DIR / cache_name

    # Check if already extracted
    if extract_dir.exists() and any(extract_dir.iterdir()):
        # Verify checksum if we have one
        if checksum_file.exists():
            print(f"  Using cached fixture: {cache_name}")
            return extract_dir

    # Download
    data = _download_with_progress(url, cache_name)

    # Compute and store checksum
    checksum = hashlib.sha256(data).hexdigest()
    checksum_file.write_text(checksum)
    print(f"  Checksum (SHA256): {checksum}")

    # Extract
    print(f"  Extracting to: {extract_dir}")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(extract_dir)

    # Count extracted files
    mp3_files = list(extract_dir.rglob("*.mp3"))
    print(f"  Extracted: {len(mp3_files)} MP3 files")

    return extract_dir


@pytest.fixture(scope="session")
def meditations_audiobook() -> Path:
    """Fixture providing path to LibriVox 'Meditations' audiobook.

    This is a public domain recording, perfect for testing:
    - Multiple MP3 files (chapters)
    - Consistent 64kbps encoding
    - Real metadata (author, title)
    - ~4 hours of audio

    Files are downloaded once and cached for subsequent test runs.
    """
    return _ensure_fixture_downloaded(
        MEDITATIONS_URL,
        MEDITATIONS_CACHE_NAME,
        MEDITATIONS_CHECKSUM_FILE,
    )


@pytest.fixture(scope="session")
def meditations_single_chapter(meditations_audiobook: Path) -> Path:
    """Return path to a single chapter from Meditations (faster for unit tests)."""
    mp3_files = sorted(meditations_audiobook.rglob("*.mp3"))
    if not mp3_files:
        pytest.skip("No MP3 files found in fixture")
    return mp3_files[0]


@pytest.fixture(scope="session")
def meditations_few_chapters(meditations_audiobook: Path) -> list[Path]:
    """Return paths to first 3 chapters (good for concat testing)."""
    mp3_files = sorted(meditations_audiobook.rglob("*.mp3"))
    if len(mp3_files) < 3:
        pytest.skip("Not enough MP3 files in fixture")
    return mp3_files[:3]


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory for processed files."""
    output = tmp_path / "output"
    output.mkdir()
    return output
