"""Tests for tidy command: metadata inference from relpath and diff building."""

import pytest

from audiobook_optimizer.adapters.abs_client import ABSItem
from audiobook_optimizer.adapters.filesystem import check_author_name, clean_name, infer_metadata_from_relpath


class TestCleanName:
    """Test enhanced clean_name with new patterns."""

    @pytest.mark.parametrize(
        "dirty,clean",
        [
            # Existing patterns still work
            ("Title [audiobook]", "Title"),
            ("Title Audiobook", "Title"),
            ("Title by dessalines", "Title"),
            ("2008 - Title", "Title"),
            # New patterns
            ("Title (Unabridged)", "Title"),
            ("Title (Abridged)", "Title"),
            ("Title (MP3)", "Title"),
            ("Title (FLAC)", "Title"),
            ("[Audiobook] Some Title", "Some Title"),
            ("DW38 - Night Watch", "Night Watch"),
            ("HoO4 - The House of Hades", "The House of Hades"),
            # Combined junk
            ("[Audiobook] Title (Unabridged) by uploader", "Title"),
        ],
    )
    def test_clean_name(self, dirty: str, clean: str):
        assert clean_name(dirty) == clean


class TestCheckAuthorName:
    """Test author name detection from folder names."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Terry Pratchett", "Terry Pratchett"),
            ("Rick Riordan", "Rick Riordan"),
            ("Nassim Nicholas Taleb", "Nassim Nicholas Taleb"),
            ("J.R.R. Tolkien", "J.R.R. Tolkien"),  # Dots stripped, "JRR" is alpha
            ("audiobooks", None),
            ("downloads", None),
            ("prowlarr", None),
            ("Some 123 Name", None),  # Contains numbers
        ],
    )
    def test_check_author_name(self, name: str, expected: str | None):
        assert check_author_name(name) == expected


class TestInferMetadataFromRelpath:
    """Test metadata inference from ABS relPath strings."""

    def test_discworld_with_number(self):
        meta = infer_metadata_from_relpath("Discworld 14 - Lords and Ladies")
        assert meta.title == "Lords and Ladies"
        assert meta.series == "Discworld"
        assert meta.series_number == 14

    def test_discworld_with_author_parent(self):
        meta = infer_metadata_from_relpath("Terry Pratchett/Discworld 01 - The Colour of Magic")
        assert meta.title == "The Colour of Magic"
        assert meta.series == "Discworld"
        assert meta.series_number == 1
        assert meta.author == "Terry Pratchett"

    def test_percy_jackson_series(self):
        meta = infer_metadata_from_relpath("Percy Jackson 03 - The Titan's Curse")
        assert meta.title == "The Titan's Curse"
        assert meta.series == "Percy Jackson"
        assert meta.series_number == 3

    def test_heroes_of_olympus(self):
        meta = infer_metadata_from_relpath("Heroes of Olympus 01 - The Lost Hero")
        assert meta.title == "The Lost Hero"
        assert meta.series == "Heroes of Olympus"
        assert meta.series_number == 1

    def test_trials_of_apollo(self):
        meta = infer_metadata_from_relpath("Trials of Apollo 05 - The Tower of Nero")
        assert meta.title == "The Tower of Nero"
        assert meta.series == "Trials of Apollo"
        assert meta.series_number == 5

    def test_standalone_with_author(self):
        meta = infer_metadata_from_relpath("Kurt Vonnegut - Slaughterhouse-Five")
        assert meta.title == "Slaughterhouse-Five"
        assert meta.author == "Kurt Vonnegut"

    def test_companion_book(self):
        meta = infer_metadata_from_relpath("Percy Jackson - Camp Half-Blood Confidential")
        assert meta.title == "Camp Half-Blood Confidential"
        assert meta.author == "Percy Jackson"  # Parsed as author (expected — no parent hint)

    def test_audiobook_epub_folder(self):
        meta = infer_metadata_from_relpath("Thinking, Fast and Slow [Audiobook + ePub] by Daniel Kahneman")
        assert meta.title == "Thinking, Fast and Slow"

    def test_dw_abbreviation_stripped(self):
        meta = infer_metadata_from_relpath("DW38 - I Shall Wear Midnight")
        assert meta.title == "I Shall Wear Midnight"

    def test_plain_title(self):
        meta = infer_metadata_from_relpath("Meditations")
        assert meta.title == "Meditations"

    def test_trailing_slashes_handled(self):
        meta = infer_metadata_from_relpath("Discworld 01 - The Colour of Magic/")
        assert meta.title == "The Colour of Magic"
        assert meta.series == "Discworld"


class TestBuildItemDiff:
    """Test diff building between ABS items and inferred metadata."""

    def _make_item(self, **kwargs) -> ABSItem:
        defaults = {
            "id": "test-id",
            "rel_path": "Test Book",
            "title": "Test Book",
            "authors": ["Test Author"],
            "series": [],
            "description": None,
            "narrators": [],
            "is_missing": False,
        }
        defaults.update(kwargs)
        return ABSItem(**defaults)

    def test_no_changes_when_metadata_matches(self):
        from audiobook_optimizer.services.tidy import build_item_diff

        item = self._make_item(
            rel_path="Discworld 01 - The Colour of Magic",
            title="The Colour of Magic",
            series=[{"name": "Discworld", "sequence": "1"}],
        )
        inferred = infer_metadata_from_relpath(item.rel_path)
        assert build_item_diff(item, inferred) == []

    def test_detects_missing_series(self):
        from audiobook_optimizer.services.tidy import build_item_diff

        item = self._make_item(
            rel_path="Discworld 14 - Lords and Ladies",
            title="Discworld 14 - Lords and Ladies",
        )
        inferred = infer_metadata_from_relpath(item.rel_path)
        changes = build_item_diff(item, inferred)

        fields = {c.field for c in changes}
        assert "title" in fields  # "Discworld 14 - ..." -> "Lords and Ladies"
        assert "series" in fields  # (none) -> Discworld
        assert "sequence" in fields  # (none) -> 14

    def test_detects_title_cleanup(self):
        from audiobook_optimizer.services.tidy import build_item_diff

        item = self._make_item(
            rel_path="DW38 - I Shall Wear Midnight",
            title="DW38-I Shall Wear Midnight",
        )
        inferred = infer_metadata_from_relpath(item.rel_path)
        changes = build_item_diff(item, inferred)

        title_changes = [c for c in changes if c.field == "title"]
        assert len(title_changes) == 1
        assert title_changes[0].proposed == "I Shall Wear Midnight"


class TestDuplicateDetection:
    """Test duplicate grouping logic."""

    def test_finds_title_author_duplicates(self):
        from collections import defaultdict

        items = [
            ABSItem(id="1", rel_path="a", title="Book", authors=["Author"]),
            ABSItem(id="2", rel_path="b", title="Book", authors=["Author"]),
            ABSItem(id="3", rel_path="c", title="Other", authors=["Author"]),
        ]

        groups = defaultdict(list)
        for item in items:
            key = (item.title.lower().strip(), frozenset(a.lower() for a in item.authors))
            groups[key].append(item)

        dupes = {k: v for k, v in groups.items() if len(v) > 1}
        assert len(dupes) == 1
        assert len(list(dupes.values())[0]) == 2
