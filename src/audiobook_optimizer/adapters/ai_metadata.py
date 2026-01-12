"""AI-powered metadata verification using Claude CLI."""

import json
import subprocess
from dataclasses import dataclass

from audiobook_optimizer.domain.models import AudiobookMetadata, AudiobookSource


@dataclass
class MetadataVerification:
    """Result of AI metadata verification."""

    original: AudiobookMetadata
    suggested: AudiobookMetadata
    confidence: float  # 0-1
    reasoning: str
    changes_made: list[str]

    @property
    def has_changes(self) -> bool:
        """Whether AI suggested any changes."""
        return bool(self.changes_made)


class ClaudeMetadataVerifier:
    """Verify and correct audiobook metadata using Claude CLI."""

    def __init__(self, claude_path: str = "claude", model: str = "haiku"):
        self.claude_path = claude_path
        self.model = model
        self._verify_claude()

    def _verify_claude(self) -> None:
        """Verify claude CLI is available."""
        try:
            result = subprocess.run(
                [self.claude_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError("claude CLI not working")
        except FileNotFoundError as e:
            raise RuntimeError(f"claude CLI not found at {self.claude_path}") from e

    def verify_metadata(self, source: AudiobookSource, inferred: AudiobookMetadata) -> MetadataVerification:
        """Use Claude to verify and potentially correct metadata."""
        # Build context for Claude
        file_names = [f.path.name for f in source.audio_files[:10]]
        if len(source.audio_files) > 10:
            file_names.append(f"... and {len(source.audio_files) - 10} more files")

        prompt = f"""Analyze this audiobook and verify/correct the metadata.

FOLDER: {source.source_path.name}
FILES: {json.dumps(file_names)}

INFERRED METADATA:
- Title: {inferred.title}
- Author: {inferred.author}
- Series: {inferred.series or "None"}
- Series Number: {inferred.series_number or "None"}

Respond with ONLY valid JSON (no markdown, no explanation outside JSON):
{{
    "title": "corrected title",
    "author": "corrected author name",
    "series": "series name or null",
    "series_number": number or null,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of any corrections",
    "changes": ["list of what was changed, empty if nothing"]
}}

Rules:
- Author should be "FirstName LastName" format (not "LastName, FirstName")
- Series names should be clean (e.g., "Discworld" not "Discworld Series")
- Remove quality tags like "(Stevens) 32k 12.58.21 {{179mb}}" from titles
- Remove year prefixes like "2008 - " from titles unless it's part of the actual title
- If unsure, keep original values and set low confidence"""

        try:
            result = subprocess.run(
                [
                    self.claude_path,
                    "-p",
                    "--model", self.model,
                    "--dangerously-skip-permissions",
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                # Fall back to original metadata on error
                return MetadataVerification(
                    original=inferred,
                    suggested=inferred,
                    confidence=0.5,
                    reasoning=f"Claude CLI failed: {result.stderr[:200]}",
                    changes_made=[],
                )

            # Parse response
            response_text = result.stdout.strip()
            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            data = json.loads(response_text)

            suggested = AudiobookMetadata(
                title=data.get("title", inferred.title),
                author=data.get("author", inferred.author),
                series=data.get("series"),
                series_number=data.get("series_number"),
                narrator=inferred.narrator,
                year=inferred.year,
                description=inferred.description,
                genre=inferred.genre,
                cover_path=inferred.cover_path,
            )

            return MetadataVerification(
                original=inferred,
                suggested=suggested,
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", ""),
                changes_made=data.get("changes", []),
            )

        except json.JSONDecodeError as e:
            return MetadataVerification(
                original=inferred,
                suggested=inferred,
                confidence=0.3,
                reasoning=f"Failed to parse Claude response: {e}",
                changes_made=[],
            )
        except subprocess.TimeoutExpired:
            return MetadataVerification(
                original=inferred,
                suggested=inferred,
                confidence=0.3,
                reasoning="Claude CLI timed out",
                changes_made=[],
            )
        except Exception as e:
            return MetadataVerification(
                original=inferred,
                suggested=inferred,
                confidence=0.3,
                reasoning=f"Error: {e}",
                changes_made=[],
            )
