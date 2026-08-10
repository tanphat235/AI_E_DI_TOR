"""Project manifest - the inventory of what the user actually provided.

A manifest is produced by ``aive project scan`` and is deliberately dumb: it
records which files exist and how big they were, and nothing about their content.
Content lives in the analysis models, which are expensive to produce; the manifest
is cheap and is what tells us whether those expensive results are still valid.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal, Self

from pydantic import Field, model_validator

from app.models.common import AiveModel, MediaKind, MediaRef

MANIFEST_SCHEMA_VERSION: Final = "1.0"
"""Version of the manifest document format.

``Final`` so the type narrows to ``Literal["1.0"]`` for the pinned field below.
"""


class MediaEntry(AiveModel):
    """One media file found in the project, with the facts needed to detect change.

    ``size_bytes`` and ``modified_at`` exist purely for cache invalidation. Video
    analysis costs minutes; re-running it because a file's mtime is unknown is
    worse than storing two extra fields.
    """

    ref: MediaRef
    kind: MediaKind
    size_bytes: int = Field(ge=0)
    modified_at: datetime

    def is_unchanged_from(self, other: MediaEntry) -> bool:
        """True when the file looks byte-identical to a previously scanned entry."""
        return (
            self.ref == other.ref
            and self.size_bytes == other.size_bytes
            and self.modified_at == other.modified_at
        )


class ProjectManifest(AiveModel):
    """The inventory of a project folder at scan time."""

    schema_version: Literal["1.0"] = MANIFEST_SCHEMA_VERSION
    project_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    name: str = Field(min_length=1)
    created_at: datetime
    scanned_at: datetime
    narration: MediaEntry | None = Field(
        default=None,
        description="The narration track. None means the project is not yet editable.",
    )
    raw_clips: tuple[MediaEntry, ...] = ()
    music: tuple[MediaEntry, ...] = ()
    capcut_templates: tuple[MediaRef, ...] = Field(
        default=(),
        description="Template project directories under capcut/, for Phase 8 export.",
    )

    @model_validator(mode="after")
    def _validate_unique_refs(self) -> Self:
        seen: set[MediaRef] = set()
        for entry in (*self.raw_clips, *self.music):
            if entry.ref in seen:
                msg = f"duplicate media entry: {entry.ref}"
                raise ValueError(msg)
            seen.add(entry.ref)
        return self

    @property
    def is_editable(self) -> bool:
        """True when the project has the minimum viable inputs: narration and footage."""
        return self.narration is not None and len(self.raw_clips) > 0

    @property
    def total_raw_duration_hint(self) -> int:
        """Combined size of raw footage in bytes.

        Named a *hint* because size is not duration. It is here only so the CLI can
        warn about an hour of 4K footage before the user waits on analysis.
        """
        return sum(entry.size_bytes for entry in self.raw_clips)

    def all_entries(self) -> tuple[MediaEntry, ...]:
        """Every media entry, narration first."""
        narration = (self.narration,) if self.narration is not None else ()
        return (*narration, *self.raw_clips, *self.music)


__all__ = ["MANIFEST_SCHEMA_VERSION", "MediaEntry", "ProjectManifest"]
