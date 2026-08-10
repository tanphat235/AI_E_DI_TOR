"""Renderer boundary (implemented in Phase 7).

A renderer consumes an :class:`~app.models.edit_plan.EditPlan` and nothing else.
It has no access to a transcript, a scene analysis, or a settings object beyond its
own render options - and it must never import from :mod:`app.analysis`.

That restriction is the architecture's main load-bearing constraint. It is what
allows the AI director to be replaced, retuned, or removed entirely without a
single change to rendering, and it is what makes a plan reproducible: the same plan
plus the same media yields the same video, forever.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.models.common import SubtitleFormat
from app.models.edit_plan import EditPlan

ProgressCallback = Callable[[float, str], None]
"""Reports progress as ``(fraction_complete, description)``.

Injected rather than baked in, because the console, the desktop UI and a test all
want to observe a forty-minute render differently.
"""


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """Everything a render needs beyond the plan itself.

    Encoder settings live here rather than in the plan, which is what lets the same
    plan be rendered as a fast draft for review and as a final master for delivery.
    """

    plan: EditPlan
    project_root: Path
    destination: Path
    subtitle_formats: tuple[SubtitleFormat, ...] = ()
    burn_in_subtitles: bool = False
    draft: bool = False
    """Trade quality for speed: lower resolution and a faster preset. Reviewing an
    edit does not need a visually lossless encode."""


@dataclass(frozen=True, slots=True)
class RenderResult:
    """What a render produced."""

    video: Path
    subtitles: tuple[Path, ...] = ()
    duration: float = 0.0
    elapsed: float = 0.0
    log_file: Path | None = None


@runtime_checkable
class Renderer(Protocol):
    """Turns an Edit Plan into a video file."""

    @property
    def name(self) -> str:
        """Renderer identity, e.g. ``ffmpeg``."""
        ...

    def preflight(self, request: RenderRequest) -> tuple[str, ...]:
        """Check the request without encoding anything.

        Returns a tuple of blocking problems; empty means good to go. Existence of
        every source file, writability of the destination, presence of the required
        binaries. Called before rendering because discovering a missing file forty
        minutes into an encode is unacceptable.
        """
        ...

    def render(
        self,
        request: RenderRequest,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> RenderResult:
        """Encode the plan."""
        ...


# `SubtitleWriter` used to live here. It moved to `app.subtitles.base` in Phase 2,
# because subtitles are useful without a video and a writer needs only cues - not a
# whole Edit Plan - to do its job.


__all__ = [
    "ProgressCallback",
    "RenderRequest",
    "RenderResult",
    "Renderer",
]
