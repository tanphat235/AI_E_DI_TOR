"""Subtitle boundary.

Subtitles live in their own package rather than under ``renderer/`` because they are
useful without a video: for a CapCut export, a YouTube upload, or review before
anything is encoded.

A writer takes **cues, not an Edit Plan**. It needs the timings and the text and
nothing else, and narrowing the input that far means a writer can be tested with three
hand-written cues instead of a whole plan. Cue times are already in timeline space when
they reach a writer; if a writer ever needs to map time, something upstream is wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from app.config.settings import SubtitleSettings
from app.models.common import SubtitleFormat
from app.models.edit_plan import SubtitleCue


@runtime_checkable
class SubtitleWriter(Protocol):
    """Serialises cues to a subtitle file."""

    @property
    def format(self) -> SubtitleFormat:
        """Which container this writer produces."""
        ...

    @property
    def supports_word_timings(self) -> bool:
        """Whether this format can express per-word (karaoke) timing.

        SRT cannot; ASS can. Exposed so a caller can decide whether generating word
        timings is worth the cost, rather than computing them and having them silently
        dropped.
        """
        ...

    def write(
        self,
        cues: tuple[SubtitleCue, ...],
        destination: Path,
        *,
        style: SubtitleSettings,
    ) -> Path:
        """Write ``cues`` to ``destination`` and return the path written."""
        ...


__all__ = ["SubtitleWriter"]
