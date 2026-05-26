"""Tests for the AI batch verification path (ai_batch.py).

Fully offline — `Agent.run_sync` and `_verify_batch_cached` are mocked via
`monkeypatch`, no Anthropic / gateway / network calls. Real audio files are
not needed: `read_embedded_tags` is stubbed where exercised.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from audiobook_optimizer.adapters import ai_batch
from audiobook_optimizer.adapters.ai_batch import (
    AudiobookVerification,
    BatchAIVerifier,
    BatchVerificationResult,
    _format_items,
    _verify_batch_cached,
    apply_verification,
)
from audiobook_optimizer.domain.models import (
    AudiobookMetadata,
    AudiobookSource,
    AudioFile,
    AudioFormat,
)


def _make_source(folder: str, filenames: list[str], duration_ms: int = 3_600_000) -> AudiobookSource:
    """Build an in-memory AudiobookSource pointing at fake paths (no I/O)."""
    base = Path("/staging") / folder
    audio_files = [
        AudioFile(
            path=base / name,
            format=AudioFormat.MP3,
            duration_ms=duration_ms // max(len(filenames), 1),
            bitrate=64,
            sample_rate=44100,
            channels=1,
        )
        for name in filenames
    ]
    return AudiobookSource(
        source_path=base,
        audio_files=audio_files,
        total_duration_ms=duration_ms,
    )


class TestAudiobookVerificationModel:
    """The model has narrator/year fields, defaulting to None."""

    def test_has_narrator_and_year_fields(self):
        v = AudiobookVerification(index=0, title="t", author="a", narrator="Rob Inglis", year=1968)
        assert v.narrator == "Rob Inglis"
        assert v.year == 1968

    def test_narrator_year_default_none(self):
        v = AudiobookVerification(index=0, title="t", author="a")
        assert v.narrator is None
        assert v.year is None


class TestBatchVerificationResultDefaults:
    """audiobooks defaults to an empty list so the LLM can return summary-only.

    Pydantic raised "Field required" when the LLM judged that no corrections
    were needed and returned just `{"summary": "..."}` — the AI verification
    would then fall through to filename-based metadata. The fix is to give
    `audiobooks` a default empty list so an omitted/empty value is valid.
    """

    def test_audiobooks_defaults_to_empty_list(self):
        # Schema must accept a summary-only response — this is what the LLM
        # actually returned in the wild when nothing needed correcting.
        r = BatchVerificationResult(summary="1 audiobook verified, no corrections needed")
        assert r.audiobooks == []
        assert r.summary == "1 audiobook verified, no corrections needed"

    def test_audiobooks_can_still_be_populated(self):
        v = AudiobookVerification(index=0, title="t", author="a")
        r = BatchVerificationResult(audiobooks=[v], summary="1 verified")
        assert r.audiobooks == [v]


class TestApplyVerification:
    """apply_verification: AI value wins when present, else fall back to existing metadata."""

    def _base(self, **kw) -> AudiobookMetadata:
        defaults = dict(title="Old Title", author="Old Author", series=None, series_number=None)
        defaults.update(kw)
        return AudiobookMetadata(**defaults)

    def test_applies_ai_narrator_and_year(self):
        meta = self._base()  # no narrator/year on the input
        v = AudiobookVerification(
            index=0,
            title="New Title",
            author="New Author",
            narrator="Douglas Adams",
            year=1979,
        )
        result = apply_verification(meta, v)
        assert result.title == "New Title"
        assert result.author == "New Author"
        assert result.narrator == "Douglas Adams"
        assert result.year == 1979

    def test_falls_back_when_ai_null(self):
        meta = self._base(narrator="Existing Narrator", year=2001)
        v = AudiobookVerification(
            index=0,
            title="New Title",
            author="New Author",
            narrator=None,
            year=None,
        )
        result = apply_verification(meta, v)
        # AI returned null → keep the existing values, don't overwrite with None
        assert result.narrator == "Existing Narrator"
        assert result.year == 2001

    def test_preserves_description_genre_cover(self):
        meta = self._base(description="A book.", genre="SciFi", cover_path=Path("/tmp/c.jpg"))
        v = AudiobookVerification(index=0, title="T", author="A")
        result = apply_verification(meta, v)
        assert result.description == "A book."
        assert result.genre == "SciFi"
        assert result.cover_path == Path("/tmp/c.jpg")


class TestEnrichedItems:
    """verify_batch enriches each item with parent_folder, total_hours, embedded_tags."""

    def test_items_carry_enrichment(self, monkeypatch):
        # Stub everything that would touch the network or filesystem
        monkeypatch.setattr(ai_batch, "ai_available", lambda: True)
        # The verifier reads settings.ai_model — stub get_settings to a known value
        monkeypatch.setattr(
            ai_batch,
            "get_settings",
            lambda: SimpleNamespace(ai_model="anthropic:claude-haiku-4-5"),
        )

        # Stub the embedded-tag reader on the local extractor
        fake_tags = {
            "artist": "Douglas Adams",
            "albumartist": None,
            "album": "The Hitchhiker's Guide to the Galaxy",
            "composer": "Stephen Fry",  # the "wrong narrator" smell test
            "date": "2005",
            "comment": "Some descriptive blurb.",
            "genre": "Audiobook",
            "title": "01Hitchhiker's Guide to the Galaxy",
        }
        from audiobook_optimizer.adapters import filesystem as fs_mod

        monkeypatch.setattr(
            fs_mod.FilesystemMetadataExtractor,
            "read_embedded_tags",
            lambda self, source: fake_tags,
        )

        # Capture what gets passed into the cached verify call
        captured = {}

        def fake_cached(items, model):
            captured["items"] = items
            captured["model"] = model
            return []  # no AI corrections needed for this assertion

        monkeypatch.setattr(ai_batch, "_verify_batch_cached", fake_cached)

        source = _make_source(
            "01Hitchhiker's Guide to the Galaxy",
            ["01Hitchhiker's Guide to the Galaxy (audiobook).mp3"],
            duration_ms=3_600_000,  # exactly 1 hour
        )
        metadata = AudiobookMetadata(title="01Hitchhiker's Guide to the Galaxy", author="Unknown")
        quality = {"bitrate": 64, "effective_bitrate": 64, "action": "transcode"}

        verifier = BatchAIVerifier()
        verifier.verify_batch([(source, metadata, quality)])

        assert "items" in captured, "verify_batch did not call the cached function"
        item = captured["items"][0]
        assert item["folder"] == "01Hitchhiker's Guide to the Galaxy"
        assert item["parent_folder"] == "staging"
        assert item["total_hours"] == 1.0
        assert item["embedded_tags"]["composer"] == "Stephen Fry"
        assert item["embedded_tags"]["date"] == "2005"
        assert item["embedded_tags"]["title"] == "01Hitchhiker's Guide to the Galaxy"
        # Inferred is preserved
        assert item["inferred"]["title"] == "01Hitchhiker's Guide to the Galaxy"
        # Model threaded through
        assert captured["model"] == "anthropic:claude-haiku-4-5"


class TestFormatItems:
    """_format_items renders enriched fields when present."""

    def test_renders_embedded_tags_and_total_hours(self):
        items = [
            {
                "index": 0,
                "folder": "Some Book",
                "parent_folder": "staging",
                "files": ["chapter1.mp3"],
                "file_count": 1,
                "total_hours": 5.5,
                "embedded_tags": {
                    "artist": "Author Name",
                    "composer": "Narrator Name",
                    "date": None,
                    "album": None,
                },
                "inferred": {
                    "title": "Some Book",
                    "author": "Author Name",
                    "series": None,
                    "series_number": None,
                },
                "quality": {"bitrate": 64, "effective_bitrate": 64, "action": "remux"},
            }
        ]
        rendered = _format_items(items)
        assert "(parent dir: staging)" in rendered
        assert "~5.5h total" in rendered
        assert "composer=Narrator Name" in rendered
        # Null tag values are suppressed
        assert "date=" not in rendered
        assert "album=" not in rendered


class TestRunSyncMocked:
    """End-to-end: a stubbed PydanticAI Agent returns narrator/year and they flow back."""

    def test_returns_expanded_verification(self, monkeypatch):
        # Build a fake BatchVerificationResult with narrator+year populated
        fake_verifications = [
            AudiobookVerification(
                index=0,
                title="The Hitchhiker's Guide to the Galaxy",
                author="Douglas Adams",
                series="The Hitchhiker's Guide to the Galaxy",
                series_number=1.0,
                narrator="Douglas Adams",
                year=1979,
                changes=["normalized title", "added narrator from composer tag", "added year"],
            )
        ]
        fake_result_output = BatchVerificationResult(
            audiobooks=fake_verifications,
            summary="1 audiobook corrected",
        )
        fake_agent_result = SimpleNamespace(output=fake_result_output)

        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def run_sync(self, prompt):
                return fake_agent_result

        # Patch the Agent class inside ai_batch's namespace (where it's imported as `Agent`)
        monkeypatch.setattr(ai_batch, "Agent", FakeAgent)

        # Try the __wrapped__ escape hatch (functools.wraps preserves the inner fn);
        # if cachekit doesn't expose it, call _verify_batch_cached directly — Redis
        # is not configured under pytest so it falls through to in-memory L1.
        inner = getattr(_verify_batch_cached, "__wrapped__", _verify_batch_cached)

        items = [
            {
                "folder": "01Hitchhiker's Guide to the Galaxy",
                "parent_folder": "staging",
                "files": ["01Hitchhiker's Guide to the Galaxy (audiobook).mp3"],
                "file_count": 1,
                "total_hours": 5.0,
                "embedded_tags": {"composer": "Stephen Fry", "date": "2005"},
                "inferred": {
                    "title": "01Hitchhiker's Guide to the Galaxy",
                    "author": "Unknown",
                    "series": None,
                    "series_number": None,
                },
                "quality": {"bitrate": 64, "effective_bitrate": 64, "action": "transcode"},
            }
        ]
        out = inner(items, "anthropic:claude-haiku-4-5")
        assert len(out) == 1
        assert out[0]["narrator"] == "Douglas Adams"
        assert out[0]["year"] == 1979
        assert out[0]["title"] == "The Hitchhiker's Guide to the Galaxy"
        assert out[0]["series"] == "The Hitchhiker's Guide to the Galaxy"


class TestReadEmbeddedTagsContract:
    """The new FilesystemMetadataExtractor.read_embedded_tags returns the expected keys
    and degrades to all-None when mutagen can't read the file (no audio at the path)."""

    def test_returns_all_none_when_file_missing(self, tmp_path):
        from audiobook_optimizer.adapters.filesystem import FilesystemMetadataExtractor

        source = _make_source("nonexistent", ["nope.mp3"])
        extractor = FilesystemMetadataExtractor()
        tags = extractor.read_embedded_tags(source)
        # Contract: same keys every time, even on failure
        expected_keys = {"artist", "albumartist", "album", "composer", "date", "comment", "genre", "title"}
        assert set(tags.keys()) == expected_keys
        assert all(v is None for v in tags.values())

    def test_returns_all_none_when_no_audio_files(self):
        from audiobook_optimizer.adapters.filesystem import FilesystemMetadataExtractor

        empty_source = AudiobookSource(
            source_path=Path("/nothing"),
            audio_files=[],
            total_duration_ms=0,
        )
        extractor = FilesystemMetadataExtractor()
        tags = extractor.read_embedded_tags(empty_source)
        assert all(v is None for v in tags.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
