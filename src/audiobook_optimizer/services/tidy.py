"""Tidy service: infer metadata from ABS library and compute diffs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from audiobook_optimizer.adapters.abs_client import ABSClient, ABSItem
from audiobook_optimizer.adapters.filesystem import infer_metadata_from_relpath
from audiobook_optimizer.domain.models import AudiobookMetadata


@dataclass
class MetadataChange:
    """A single proposed metadata change for an ABS library item."""

    item: ABSItem
    field: str
    current: str | None
    proposed: str


@dataclass
class TidyResult:
    """Result of analyzing an ABS library for metadata issues."""

    changes: list[MetadataChange] = field(default_factory=list)
    updates: dict[str, dict] = field(default_factory=dict)  # item_id -> metadata patch
    duplicate_groups: dict[tuple, list[ABSItem]] = field(default_factory=dict)


def format_sequence(num: float) -> str:
    """Format a series number as a clean string (no trailing .0)."""
    return str(int(num)) if num == int(num) else str(num)


def build_item_diff(item: ABSItem, inferred: AudiobookMetadata) -> list[MetadataChange]:
    """Compare ABS item metadata with inferred metadata. Returns list of changes."""
    changes: list[MetadataChange] = []

    # Title: prefer inferred if it's cleaner
    if inferred.title and inferred.title != item.title:
        if len(inferred.title) >= 3 and inferred.title != "Unknown Author":
            changes.append(MetadataChange(item=item, field="title", current=item.title, proposed=inferred.title))

    # Series: add if inferred has series but ABS doesn't
    if inferred.series and not item.series_name:
        changes.append(MetadataChange(item=item, field="series", current=item.series_name, proposed=inferred.series))
        if inferred.series_number is not None:
            changes.append(
                MetadataChange(
                    item=item, field="sequence", current=item.series_sequence, proposed=format_sequence(inferred.series_number)
                )
            )
    elif inferred.series_number is not None and item.series_name and not item.series_sequence:
        changes.append(
            MetadataChange(
                item=item, field="sequence", current=item.series_sequence, proposed=format_sequence(inferred.series_number)
            )
        )

    return changes


def analyze_library(items: list[ABSItem]) -> TidyResult:
    """Analyze all library items and compute metadata diffs."""
    result = TidyResult()

    for item in items:
        if item.is_missing:
            continue

        inferred = infer_metadata_from_relpath(item.rel_path)
        item_changes = build_item_diff(item, inferred)

        if not item_changes:
            continue

        result.changes.extend(item_changes)

        # Build ABS update payload
        meta_patch: dict = {}
        for change in item_changes:
            if change.field == "title":
                meta_patch["title"] = change.proposed
            elif change.field == "series":
                meta_patch["series"] = [{"name": change.proposed, "sequence": ""}]
            elif change.field == "sequence":
                if "series" in meta_patch:
                    meta_patch["series"][0]["sequence"] = change.proposed
                elif inferred.series:
                    meta_patch["series"] = [{"name": inferred.series, "sequence": change.proposed}]

        if meta_patch:
            result.updates[item.id] = meta_patch

    return result


def find_duplicates(items: list[ABSItem]) -> dict[tuple, list[ABSItem]]:
    """Find duplicate items by title + author."""
    groups: dict[tuple, list[ABSItem]] = defaultdict(list)
    for item in items:
        if item.is_missing:
            continue
        key = (item.title.lower().strip(), frozenset(a.lower() for a in item.authors))
        groups[key].append(item)
    return {k: v for k, v in groups.items() if len(v) > 1}


def apply_updates(client: ABSClient, updates: dict[str, dict]) -> int:
    """Apply metadata patches to ABS. Returns count of updated items."""
    for item_id, meta in updates.items():
        client.update_item(item_id, meta)
    return len(updates)


def remove_duplicates(client: ABSClient, duplicate_groups: dict[tuple, list[ABSItem]]) -> list[ABSItem]:
    """Remove duplicate items, keeping the first in each group. Returns deleted items."""
    deleted = []
    for group in duplicate_groups.values():
        for item in group[1:]:  # Keep first, delete rest
            client.delete_item(item.id)
            deleted.append(item)
    return deleted
