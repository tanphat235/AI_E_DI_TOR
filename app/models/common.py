"""Primitives shared by every AIVE model.

Two conventions hold across the whole codebase and are enforced here rather than
by comment:

* **Time is always seconds as a float.** Never frames, never a timecode string,
  never milliseconds. Frame numbers are a renderer concern and are derived at the
  last possible moment from the output fps.
* **Media paths are always relative to the project root**, in POSIX form. An Edit
  Plan must survive being moved between machines and between Windows and macOS.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class AiveModel(BaseModel):
    """Base class for every model in AIVE.

    ``frozen`` - analysis results and Edit Plans are *values*, not mutable state.
    Every transformation returns a new object, which is what makes the Rule
    Engine's normalise step auditable: you always retain the before and the after.

    ``extra="forbid"`` - Edit Plans are authored by a language model. A misspelled
    key must fail loudly instead of being silently discarded and quietly changing
    the edit. This single setting is the difference between "the plan was wrong"
    and "the plan was wrong and nobody noticed".
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )


# --------------------------------------------------------------------------- #
# Scalar aliases
# --------------------------------------------------------------------------- #

Seconds = Annotated[float, Field(ge=0.0)]
"""A non-negative duration or offset in seconds."""

Score = Annotated[float, Field(ge=0.0, le=1.0)]
"""A normalised score. 0.0 is worst, 1.0 is best - always, for every metric.

Metrics whose natural scale is inverted (blur variance, for instance) are
normalised at the point of measurement so that consumers never need to remember
which direction is good.
"""

Decibels = Annotated[float, Field(ge=-120.0, le=24.0)]
"""A gain or level in dB. Negative attenuates."""

Attenuation = Annotated[float, Field(ge=-120.0, le=0.0)]
"""A gain that may only reduce level. Used for ducking and music trim."""

# CAUTION for anyone extending these aliases.
#
# Pydantic does NOT merge constraints between an Annotated alias and the Field on
# the assignment. Given `x: Seconds = Field(default=0.0, le=10.0)`, the alias's
# bounds win and the `le=10.0` is discarded *silently* - no error, no warning, and
# a schema that claims a limit it does not enforce.
#
# So: either use a bare alias with no extra constraints, or declare the field as a
# plain `float`/`int` and put every constraint on the assignment. Never mix the two.
# Regression coverage lives in tests/unit/test_models_edit_plan.py.


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Severity(StrEnum):
    """Severity of a validation finding."""

    ERROR = "error"
    """The plan cannot be rendered. Rendering must abort."""

    WARNING = "warning"
    """The plan is renderable but an editor would likely object."""

    INFO = "info"
    """A normalisation that was applied automatically."""


class TransitionKind(StrEnum):
    """Transition between two adjacent timeline clips.

    ``CUT`` is a first-class member rather than ``None``: an explicit hard cut is
    an editorial decision, and making it nameable lets the AI director state it
    deliberately instead of by omission.
    """

    CUT = "cut"
    FADE = "fade"
    DISSOLVE = "dissolve"
    CROSSFADE = "crossfade"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"

    @property
    def is_instant(self) -> bool:
        """True when the transition occupies no timeline duration."""
        return self is TransitionKind.CUT


class MotionLevel(StrEnum):
    """Coarse bucket for how much movement a scene contains."""

    STATIC = "static"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CameraMove(StrEnum):
    """Dominant camera movement detected across a scene."""

    STATIC = "static"
    PAN = "pan"
    TILT = "tilt"
    ZOOM = "zoom"
    HANDHELD = "handheld"
    UNKNOWN = "unknown"


class ShotType(StrEnum):
    """Framing of a shot, in the vocabulary a human editor uses."""

    EXTREME_WIDE = "extreme_wide"
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    DRONE = "drone"
    UNKNOWN = "unknown"


class AspectRatio(StrEnum):
    """Delivery aspect ratio."""

    LANDSCAPE = "16:9"
    VERTICAL = "9:16"
    SQUARE = "1:1"
    PORTRAIT_4_5 = "4:5"

    @property
    def ratio(self) -> float:
        """Width divided by height."""
        width, height = (float(part) for part in self.value.split(":"))
        return width / height


class SubtitleFormat(StrEnum):
    """Subtitle container to emit."""

    SRT = "srt"
    """Plain, universally supported, cue-level timing only."""

    ASS = "ass"
    """Styled; carries word-level karaoke timing when the transcript has it."""


class MusicMood(StrEnum):
    """Mood vocabulary for matching a music bed to narration.

    Deliberately small. A short, closed vocabulary is something both a classical
    audio-feature heuristic and an AI director can agree on; free-text moods
    cannot be matched reliably by either.
    """

    CALM = "calm"
    UPLIFTING = "uplifting"
    ENERGETIC = "energetic"
    DRAMATIC = "dramatic"
    TENSE = "tense"
    MELANCHOLIC = "melancholic"
    PLAYFUL = "playful"
    NEUTRAL = "neutral"


class MediaKind(StrEnum):
    """Role a media file plays in a project."""

    NARRATION = "narration"
    RAW_VIDEO = "raw_video"
    MUSIC = "music"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #


class TimeRange(AiveModel):
    """A half-open interval ``[start, end)`` in seconds.

    Half-open is what makes adjacent ranges composable: ``[0, 5)`` followed by
    ``[5, 9)`` covers nine seconds with no overlap and no gap, so the Rule Engine
    can check continuity with plain equality instead of an epsilon.
    """

    start: Seconds = Field(description="Inclusive start, in seconds.")
    end: float = Field(gt=0.0, description="Exclusive end, in seconds.")

    @model_validator(mode="after")
    def _validate_order(self) -> Self:
        if self.end <= self.start:
            msg = f"end ({self.end}) must be greater than start ({self.start})"
            raise ValueError(msg)
        return self

    @property
    def duration(self) -> float:
        """Length of the interval in seconds."""
        return self.end - self.start

    def contains(self, moment: float) -> bool:
        """True when ``moment`` falls inside the half-open interval."""
        return self.start <= moment < self.end

    def overlaps(self, other: TimeRange, *, tolerance: float = 0.0) -> bool:
        """True when the two intervals share more than ``tolerance`` seconds.

        ``tolerance`` exists because float timestamps that came from separate
        analysis passes are rarely bit-identical. A 1 ms shared sliver is a
        rounding artefact, not an overlap an editor would care about.
        """
        return self.start < other.end - tolerance and other.start < self.end - tolerance

    def intersection(self, other: TimeRange) -> TimeRange | None:
        """The shared interval, or ``None`` when they do not genuinely overlap."""
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if end <= start:
            return None
        return TimeRange(start=start, end=end)

    def shifted(self, delta: float) -> TimeRange:
        """A copy moved along the time axis by ``delta`` seconds."""
        start = self.start + delta
        if start < 0.0:
            msg = f"shifting by {delta} would move start to {start}, before zero"
            raise ValueError(msg)
        return TimeRange(start=start, end=self.end + delta)

    def clamped_to(self, limit: TimeRange) -> TimeRange | None:
        """This range restricted to ``limit``, or ``None`` if it falls outside.

        Used when a plan references a source range that runs past the real
        duration of the media file.
        """
        return self.intersection(limit)

    def __str__(self) -> str:
        return f"{self.start:.3f}-{self.end:.3f}"


# --------------------------------------------------------------------------- #
# Media references
# --------------------------------------------------------------------------- #


def map_to_timeline(moment: float, kept_ranges: tuple[TimeRange, ...]) -> float | None:
    """Map a source timestamp into timeline time, or ``None`` if it was cut.

    Narration cleanup removes silence and filler, so source time and timeline time
    diverge: a word spoken at 41.2 s in the recording may land at 33.8 s in the finished
    video. This is the single function that defines that relationship.

    It lives here, in the layer that depends on nothing, precisely because three
    otherwise-unrelated places need it - cleanup, subtitle building, and the Edit Plan's
    narration track. Any of them owning it would drag a dependency across a boundary;
    two of them implementing it would eventually disagree, and a subtitle timed by a
    subtly different mapping is the kind of bug nobody notices until the export.
    """
    elapsed = 0.0
    for kept in kept_ranges:
        if kept.contains(moment):
            return elapsed + (moment - kept.start)
        elapsed += kept.duration
    return None


class MediaRef(AiveModel):
    """A pointer to a media file, always relative to the project root.

    Relative-only is both a portability and a safety property. Portability,
    because an Edit Plan can be committed, shared, and re-rendered elsewhere.
    Safety, because the plan is untrusted input authored by a language model:
    without this constraint a plan could name ``C:/Windows/System32/...`` and the
    renderer would dutifully read it.
    """

    path: Path = Field(description="POSIX-style path relative to the project root.")

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_string(cls, value: object) -> object:
        """Allow ``"raw/001.mp4"`` as well as ``{"path": "raw/001.mp4"}``.

        The Edit Plan is authored by a language model, and this class's design goal is
        forgiving to write, strict to accept. A forty-clip plan mentions a media reference
        eighty times, and requiring the wrapper object at every one of them buys no safety -
        every constraint below still applies - while costing a whole validation round trip
        the first time the shorthand is reached for.
        """
        if isinstance(value, str):
            return {"path": value}
        return value

    @field_validator("path", mode="after")
    @classmethod
    def _must_be_relative(cls, value: Path) -> Path:
        if value.is_absolute() or value.drive or value.root:
            msg = f"media path must be relative to the project root, got {str(value)!r}"
            raise ValueError(msg)
        if ".." in value.parts:
            msg = f"media path must not traverse upwards, got {str(value)!r}"
            raise ValueError(msg)
        if not value.parts or str(value) in {".", ""}:
            msg = "media path must not be empty"
            raise ValueError(msg)
        # Re-parse from POSIX form so Windows and POSIX hosts agree on the value.
        return Path(value.as_posix())

    @field_serializer("path")
    def _serialize_path(self, value: Path) -> str:
        """Always emit forward slashes, whatever platform wrote the file."""
        return value.as_posix()

    @property
    def name(self) -> str:
        """Final path component, e.g. ``001.mp4``."""
        return self.path.name

    @property
    def stem(self) -> str:
        """Filename without its extension, e.g. ``001``."""
        return self.path.stem

    @property
    def suffix(self) -> str:
        """Lowercased extension including the dot, e.g. ``.mp4``."""
        return self.path.suffix.lower()

    def resolve_within(self, root: Path) -> Path:
        """Absolute path of this reference inside ``root``.

        Raises :class:`ValueError` if the result escapes ``root``. The relative
        path validator already blocks the obvious cases; this re-checks after
        symlink resolution, which is the only way to be sure.
        """
        root = root.resolve()
        candidate = (root / self.path).resolve()
        if candidate != root and root not in candidate.parents:
            msg = f"{str(self.path)!r} resolves outside the project root {str(root)!r}"
            raise ValueError(msg)
        return candidate

    @classmethod
    def from_path(cls, path: Path, *, root: Path) -> MediaRef:
        """Build a reference for ``path`` by making it relative to ``root``."""
        try:
            relative = path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            msg = f"{str(path)!r} is not inside the project root {str(root)!r}"
            raise ValueError(msg) from exc
        return cls(path=relative)

    def __str__(self) -> str:
        return self.path.as_posix()


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


class Issue(AiveModel):
    """A single finding from validation or normalisation.

    ``code`` is a stable machine-readable slug so the AI director can branch on
    the failure without parsing prose, while ``hint`` tells it what to change.
    That pairing is what lets the agent self-correct rather than retry blindly.
    """

    code: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_.]+$",
        description="Stable slug, e.g. 'clip.too_short' or 'source.out_of_bounds'.",
    )
    severity: Severity
    message: str = Field(min_length=1, description="Human-readable description.")
    hint: str | None = Field(
        default=None,
        description="Actionable next step for whoever authored the plan.",
    )
    location: str | None = Field(
        default=None,
        description="Where the problem is, e.g. 'clips[3]' or 'subtitles[12]'.",
    )

    def __str__(self) -> str:
        where = f" at {self.location}" if self.location else ""
        return f"[{self.severity}] {self.code}{where}: {self.message}"


__all__ = [
    "AiveModel",
    "AspectRatio",
    "Attenuation",
    "CameraMove",
    "Decibels",
    "Issue",
    "MediaKind",
    "MediaRef",
    "MotionLevel",
    "MusicMood",
    "Score",
    "Seconds",
    "Severity",
    "ShotType",
    "SubtitleFormat",
    "TimeRange",
    "TransitionKind",
    "map_to_timeline",
]
