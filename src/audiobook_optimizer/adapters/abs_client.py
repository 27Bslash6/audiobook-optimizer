"""Audiobookshelf API client adapter."""

import re
from dataclasses import dataclass, field

import httpx

_SERIES_NAME_RE = re.compile(r"^(.+?)\s*#(\d+(?:\.\d+)?)$")


def _parse_series_name(series_name: str) -> list[dict]:
    """Parse ABS seriesName string like 'Discworld #6' into structured format."""
    if not series_name:
        return []
    match = _SERIES_NAME_RE.match(series_name)
    if match:
        return [{"name": match.group(1).strip(), "sequence": match.group(2)}]
    return [{"name": series_name, "sequence": ""}]


class ABSApiError(Exception):
    """Raised on non-2xx responses from Audiobookshelf API."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"ABS API error {status_code}: {message}")


@dataclass
class ABSItem:
    """A library item from Audiobookshelf."""

    id: str
    rel_path: str
    title: str
    authors: list[str] = field(default_factory=list)
    series: list[dict] = field(default_factory=list)  # [{"name": "...", "sequence": "1"}]
    description: str | None = None
    narrators: list[str] = field(default_factory=list)
    is_missing: bool = False

    @property
    def series_name(self) -> str | None:
        return self.series[0]["name"] if self.series else None

    @property
    def series_sequence(self) -> str | None:
        return self.series[0].get("sequence") if self.series else None

    @property
    def author_name(self) -> str:
        return ", ".join(self.authors) if self.authors else "Unknown Author"


@dataclass
class ABSAuthor:
    """An author from Audiobookshelf."""

    id: str
    name: str
    description: str | None = None
    image_path: str | None = None
    num_books: int = 0

    @property
    def has_image(self) -> bool:
        return self.image_path is not None


class ABSClient:
    """Thin httpx client for the Audiobookshelf API."""

    def __init__(self, url: str, api_key: str, library_id: str):
        self._library_id = library_id
        self._client = httpx.Client(
            base_url=url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        resp = self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            raise ABSApiError(resp.status_code, resp.text[:200])
        return resp

    def get_items(self) -> list[ABSItem]:
        """Fetch all library items."""
        resp = self._request("GET", f"/api/libraries/{self._library_id}/items", params={"limit": 0})
        items = []
        for raw in resp.json().get("results", []):
            meta = raw.get("media", {}).get("metadata", {})
            # Parse authors: full format has list of dicts, minified has flat string
            if "authors" in meta and isinstance(meta["authors"], list):
                authors = [a["name"] for a in meta["authors"] if a.get("name")]
            else:
                author_str = meta.get("authorName", "")
                authors = [a.strip() for a in author_str.split(",")] if author_str else []
            # Parse series: full format has list of dicts, minified has "seriesName" like "Discworld #6"
            if "series" in meta and isinstance(meta["series"], list):
                series = [{"name": s["name"], "sequence": s.get("sequence", "")} for s in meta["series"] if s.get("name")]
            else:
                series = _parse_series_name(meta.get("seriesName", ""))
            items.append(
                ABSItem(
                    id=raw["id"],
                    rel_path=raw.get("relPath", ""),
                    title=meta.get("title", ""),
                    authors=authors,
                    series=series,
                    description=meta.get("description"),
                    narrators=(
                        meta.get("narrators", [])
                        if isinstance(meta.get("narrators"), list)
                        else [meta["narratorName"]]
                        if meta.get("narratorName")
                        else []
                    ),
                    is_missing=raw.get("isMissing", False),
                )
            )
        return items

    def get_authors(self) -> list[ABSAuthor]:
        """Fetch all library authors."""
        resp = self._request("GET", f"/api/libraries/{self._library_id}/authors")
        authors = []
        for raw in resp.json().get("authors", []):
            authors.append(
                ABSAuthor(
                    id=raw["id"],
                    name=raw.get("name", ""),
                    description=raw.get("description"),
                    image_path=raw.get("imagePath"),
                    num_books=raw.get("numBooks", 0),
                )
            )
        return authors

    def update_item(self, item_id: str, metadata: dict) -> None:
        """Update metadata for a single library item."""
        self._request("PATCH", f"/api/items/{item_id}/media", json={"metadata": metadata})

    def match_item(self, item_id: str, provider: str = "audible") -> bool:
        """Trigger metadata match from external provider. Returns True if updated."""
        resp = self._request("POST", f"/api/items/{item_id}/match", json={"provider": provider})
        return resp.json().get("updated", False)

    def match_author(self, author_id: str, author_name: str) -> bool:
        """Trigger author match from Audible. Returns True if updated."""
        resp = self._request("POST", f"/api/authors/{author_id}/match", json={"q": author_name})
        return resp.json().get("updated", False)

    def delete_item(self, item_id: str) -> None:
        """Delete a library item from ABS database (files remain on disk)."""
        self._request("DELETE", f"/api/items/{item_id}")

    def scan_library(self) -> None:
        """Trigger a library scan."""
        self._request("POST", f"/api/libraries/{self._library_id}/scan")

    def close(self) -> None:
        self._client.close()
