"""The Edit Plan - AIVE's central contract.

Everything upstream produces it; everything downstream consumes it. The AI
director authors it, the Rule Engine validates and normalises it, the FFmpeg
renderer and the CapCut exporter read it. No renderer ever talks to an analyser,
and no analyser knows a renderer exists.

Two properties make that work in practice:

**The plan is authored by a language model, so it must be forgiving to write and
strict to accept.** Forgiving: ``timeline_start`` is optional, because making an
LLM do cumulative floating-point arithmetic across forty clips is asking for
off-by-a-frame errors. List the clips in order and the Rule Engine places them.
Strict: ``extra="forbid"`` plus a pinned ``schema_version`` means a malformed plan
is rejected at the door rather than half-rendered.

**Validation is split in two.** The models here enforce *well-formedness* -
unique ids, a transition that has duration only if it is not a cut, monotonic
ranges. The Rule Engine enforces *admissibility* - whether a clip is long enough
to read, whether the source range exists in the actual file, whether two shots are
the same take. A plan can be well-formed and still be a bad edit; only the second
check knows the difference.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal, Self

from pydantic import Field, model_validator

from app.models.common import (
    AiveModel,
    AspectRatio,
    Attenuation,
    Decibels,
    Issue,
    MediaRef,
    Score,
    Seconds,
    Severity,
    TimeRange,
    TransitionKind,
    map_to_timeline,
)

EDIT_PLAN_SCHEMA_VERSION: Final = "1.0"
"""Version of the Edit Plan document format.

Declared ``Final`` so the type narrows to ``Literal["1.0"]`` and can be used as the
default for the pinned field below without widening it back to ``str``.

Pinned as a ``Literal`` on :class:`EditPlan`, so an older exporter *refuses* a
newer plan instead of silently misreading fields it does not understand. Bump this
only for breaking changes, and add a migration when you do.
"""


# --------------------------------------------------------------------------- #
# Output format
# --------------------------------------------------------------------------- #


class OutputSpec(AiveModel):
    """The delivery format the edit is cut for.

    Only properties that change *editorial* decisions live here. Frame rate
    affects how long a transition can be; aspect ratio affects whether a wide shot
    survives a crop. Encoder settings - CRF, preset, bitrate - are deliberately
    absent: they belong to config, so the same plan can be rendered once as a fast
    draft and again as a final master without editing the plan.
    """

    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE
    width: int = Field(default=3840, gt=0)
    height: int = Field(default=2160, gt=0)
    fps: float = Field(default=60.0, gt=0.0, le=240.0)

    @property
    def frame_duration(self) -> float:
        """Length of a single frame in seconds."""
        return 1.0 / self.fps

    @property
    def actual_ratio(self) -> float:
        """Width over height as configured, which may differ from ``aspect_ratio``."""
        return self.width / self.height


# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #


class Transition(AiveModel):
    """A transition applied at one edge of a clip."""

    kind: TransitionKind
    # Plain `float`, not the `Seconds` alias: constraints do not merge across an
    # Annotated alias, so an upper bound must be declared here to be enforced.
    duration: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Length of the transition in seconds. Must be 0 for a hard cut.",
    )

    @model_validator(mode="after")
    def _validate_duration_matches_kind(self) -> Self:
        if self.kind.is_instant and self.duration != 0.0:
            msg = (
                f"a {self.kind} transition is instantaneous; "
                f"duration must be 0, got {self.duration}"
            )
            raise ValueError(msg)
        if not self.kind.is_instant and self.duration <= 0.0:
            msg = f"a {self.kind} transition needs a positive duration, got {self.duration}"
            raise ValueError(msg)
        return self

    @classmethod
    def cut(cls) -> Transition:
        """A hard cut - the default, and never wrong."""
        return cls(kind=TransitionKind.CUT, duration=0.0)


class FramingSpec(AiveModel):
    """Reframing applied to a clip: a slow push, a crop to vertical, a Ken Burns move.

    Present in the schema from day one but not consumed by any Phase 7 renderer.
    It is here so that auto-zoom, face-tracked reframing and 9:16 repurposing can
    be added later as *renderer* features against a plan format that already
    describes them, rather than as a schema break.
    """

    zoom_start: float = Field(default=1.0, ge=1.0, le=4.0, description="1.0 is no zoom.")
    zoom_end: float = Field(default=1.0, ge=1.0, le=4.0)
    anchor_x: float = Field(default=0.5, ge=0.0, le=1.0, description="0 is left, 1 is right.")
    anchor_y: float = Field(default=0.5, ge=0.0, le=1.0, description="0 is top, 1 is bottom.")
    crop_to: AspectRatio | None = Field(
        default=None,
        description="Reframe to this ratio. None keeps the project's output ratio.",
    )

    @property
    def is_static(self) -> bool:
        """True when nothing actually moves, so the renderer can skip the filter."""
        return self.zoom_start == self.zoom_end == 1.0 and self.crop_to is None


class TimelineClip(AiveModel):
    """One clip on the timeline: a slice of a source file, in a position.

    ``timeline_start`` is optional on purpose. When every clip omits it, the plan
    is a simple ordered list and the Rule Engine packs the clips end to end,
    accounting for transition overlap. Supply it only to place a clip
    deliberately - an insert that must land on a beat, for instance.
    """

    id: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9._#-]+$",
        description="Unique within the plan. Used by reports to point at a clip.",
    )
    source: MediaRef
    source_range: TimeRange = Field(description="The slice to take, in *source file* time.")
    timeline_start: Seconds | None = Field(
        default=None,
        description=(
            "Position on the timeline in seconds. Omit to have the Rule Engine pack "
            "this clip after the previous one."
        ),
    )
    speed: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Playback rate. 2.0 is double speed, halving the timeline duration.",
    )
    transition_in: Transition | None = None
    transition_out: Transition | None = None
    framing: FramingSpec | None = None
    mute_source_audio: bool = Field(
        default=True,
        description=(
            "B-roll audio is usually wind and handling noise competing with the "
            "narration, so silence is the safe default. Set False to keep sync sound."
        ),
    )

    # -- Provenance: why this clip is here ---------------------------------- #
    reason: str = Field(
        min_length=1,
        description=(
            "Why this clip was chosen, in one sentence. Not decoration: it is how a "
            "human reviews the director's judgement in the UI, and how the director "
            "re-reads its own reasoning on a later pass."
        ),
    )
    confidence: Score = Field(default=1.0, description="Director's confidence in this choice.")
    scene_key: str | None = Field(
        default=None,
        description="Analysed scene this came from, e.g. '001#3'. Links plan back to analysis.",
    )
    beat_index: int | None = Field(
        default=None,
        ge=0,
        description="Narration beat this clip illustrates.",
    )

    @property
    def timeline_duration(self) -> float:
        """How long this clip occupies the timeline, after the speed change."""
        return self.source_range.duration / self.speed

    @property
    def is_placed(self) -> bool:
        """True when an explicit ``timeline_start`` was supplied."""
        return self.timeline_start is not None

    @property
    def timeline_range(self) -> TimeRange:
        """Where this clip sits on the timeline.

        Raises :class:`ValueError` when the clip has not been placed yet - call the
        Rule Engine's normalise step first.
        """
        if self.timeline_start is None:
            msg = f"clip {self.id!r} has no timeline_start; normalise the plan before placing it"
            raise ValueError(msg)
        return TimeRange(
            start=self.timeline_start,
            end=self.timeline_start + self.timeline_duration,
        )


# --------------------------------------------------------------------------- #
# Subtitles
# --------------------------------------------------------------------------- #


class SubtitleCue(AiveModel):
    """One subtitle, timed against the **timeline**, not the source narration.

    This is the single easiest thing to get wrong in the whole format. Narration
    cleanup removes silence and filler, so a word spoken at 41.2 s in
    ``narration.wav`` may land at 33.8 s in the finished video. Cues carry the
    *timeline* value. ``aive subtitle build`` performs that mapping, which is why
    the director should generate cues with the tool rather than by hand.
    """

    range: TimeRange = Field(description="When the cue is on screen, in timeline time.")
    text: str = Field(min_length=1, description="Cue text; a newline separates two lines.")
    style_id: str = Field(
        default="default",
        description="Named style from the subtitle config. Resolved at render time.",
    )
    words: tuple[tuple[str, float, float], ...] = Field(
        default=(),
        description=(
            "Optional word-level karaoke timing as (text, start, end) in timeline "
            "time. Consumed only by the ASS writer; SRT ignores it."
        ),
    )

    @property
    def line_count(self) -> int:
        return len(self.text.splitlines()) or 1


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


class DuckingSpec(AiveModel):
    """How music gets out of the way of narration.

    Stated as a target: *drop the bed by this many dB while narration plays, over these
    ramp times*. Not as compressor settings, and that distinction was earned. Phase 7
    originally rendered this with FFmpeg's ``sidechaincompress``, on the reasoning that a
    compressor is what an audio engine offers natively. Measurement showed the attenuation
    saturating near 10 dB whatever ratio it was given, so a plan asking for -12 dB and a
    plan asking for -20 dB produced the same output and the field was decorative.

    A consumer does not need a compressor, because it does not need to *detect* speech:
    :attr:`NarrationTrack.kept_ranges` states exactly when narration plays. Both the FFmpeg
    renderer and the CapCut exporter derive their envelope from that one field, so they
    cannot drift apart — the property the compressor framing was originally meant to
    protect, obtained by deriving rather than by delegating.
    """

    gain_db: Attenuation = Field(
        default=-12.0,
        description="How far the music drops while narration is present. Exact, not a target.",
    )
    threshold_db: float = Field(
        default=-30.0,
        ge=-90.0,
        le=0.0,
        description=(
            "Narration level above which ducking engages. Retained for exporters whose "
            "audio engine really is a compressor; the FFmpeg renderer derives the ducked "
            "interval from the narration track instead and ignores this."
        ),
    )
    attack: float = Field(
        default=0.15, ge=0.0, le=5.0, description="Time to reach full attenuation."
    )
    release: float = Field(
        default=0.60, ge=0.0, le=10.0, description="Time to recover after speech."
    )


class MusicCue(AiveModel):
    """A music bed placed under a stretch of the timeline."""

    track: MediaRef
    timeline_range: TimeRange = Field(description="Where the bed plays, in timeline time.")
    source_offset: Seconds = Field(
        default=0.0,
        description=(
            "Where to start inside the music file. Use it to skip an ambient intro "
            "so the bed lands with the cut instead of fading up over it."
        ),
    )
    gain_db: Decibels = Field(default=0.0, description="Static trim applied before ducking.")
    fade_in: float = Field(default=0.0, ge=0.0, le=30.0)
    fade_out: float = Field(default=0.0, ge=0.0, le=30.0)
    ducking: DuckingSpec | None = Field(
        default=None,
        description="None disables ducking for this cue - correct only where no narration plays.",
    )

    @model_validator(mode="after")
    def _validate_fades_fit(self) -> Self:
        total_fade = self.fade_in + self.fade_out
        if total_fade > self.timeline_range.duration:
            msg = (
                f"fades ({total_fade:.2f}s) exceed the cue duration "
                f"({self.timeline_range.duration:.2f}s)"
            )
            raise ValueError(msg)
        return self


class NarrationTrack(AiveModel):
    """The narration, and which parts of it survive cleanup.

    ``kept_ranges`` are in *source* time and are laid end to end on the timeline in
    order. That is what produces the timeline-versus-source offset subtitles have
    to account for.
    """

    source: MediaRef
    kept_ranges: tuple[TimeRange, ...] = Field(
        min_length=1,
        description="Ascending, non-overlapping ranges of the source to keep.",
    )
    gain_db: Decibels = Field(default=0.0)

    @model_validator(mode="after")
    def _validate_ranges_ordered(self) -> Self:
        previous: TimeRange | None = None
        for current in self.kept_ranges:
            if previous is not None and current.start < previous.end:
                msg = (
                    f"kept_ranges must be ascending and non-overlapping: "
                    f"{current} starts before {previous} ends"
                )
                raise ValueError(msg)
            previous = current
        return self

    @property
    def timeline_duration(self) -> float:
        """Narration length after cleanup - the natural length of the finished video."""
        return sum(item.duration for item in self.kept_ranges)

    def source_to_timeline(self, moment: float) -> float | None:
        """Map a source timestamp to its timeline position.

        Returns ``None`` when ``moment`` falls in a removed gap.

        Delegates to :func:`app.models.common.map_to_timeline` rather than repeating the
        arithmetic, so this and the analysis pipeline can never disagree about where a
        word lands.
        """
        return map_to_timeline(moment, self.kept_ranges)


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


class EditPlan(AiveModel):
    """A complete, renderer-agnostic description of a finished video.

    Well-formedness is enforced here. Admissibility - clip lengths, source bounds,
    duplicate shots, transition sanity against real durations - is the Rule
    Engine's job, because it needs config thresholds and probe data that a model
    has no access to.
    """

    schema_version: Literal["1.0"] = EDIT_PLAN_SCHEMA_VERSION
    project_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    created_by: str = Field(
        min_length=1,
        description="Who authored this, e.g. 'claude-code', 'heuristic', 'human'.",
    )
    created_at: datetime | None = None
    output: OutputSpec = Field(default_factory=OutputSpec)
    narration: NarrationTrack | None = Field(
        default=None,
        description="None is legal: a music-only montage has no narration.",
    )
    clips: tuple[TimelineClip, ...] = Field(
        min_length=1,
        description="Editorial order. List order is authoritative, not timeline_start.",
    )
    subtitles: tuple[SubtitleCue, ...] = ()
    music: tuple[MusicCue, ...] = ()
    notes: str | None = Field(
        default=None,
        description="The director's overall editorial rationale, for human review.",
    )

    @model_validator(mode="after")
    def _validate_unique_clip_ids(self) -> Self:
        seen: set[str] = set()
        for clip in self.clips:
            if clip.id in seen:
                msg = f"duplicate clip id {clip.id!r}"
                raise ValueError(msg)
            seen.add(clip.id)
        return self

    @property
    def is_placed(self) -> bool:
        """True when every clip has an explicit timeline position."""
        return all(clip.is_placed for clip in self.clips)

    @property
    def timeline_duration(self) -> float:
        """Length of the finished video in seconds.

        Sums clip durations and subtracts transition overlaps, so it is correct
        whether or not the plan has been placed yet.
        """
        total = sum(clip.timeline_duration for clip in self.clips)
        overlap = sum(
            clip.transition_in.duration for clip in self.clips[1:] if clip.transition_in is not None
        )
        return max(0.0, total - overlap)

    def clip_by_id(self, clip_id: str) -> TimelineClip | None:
        return next((clip for clip in self.clips if clip.id == clip_id), None)

    @property
    def sources(self) -> tuple[MediaRef, ...]:
        """Every distinct media file the plan needs, narration and music included.

        Used to preflight a render: check all of these exist before spending an
        hour encoding, and to know what to copy into a CapCut draft.
        """
        refs: list[MediaRef] = [clip.source for clip in self.clips]
        if self.narration is not None:
            refs.append(self.narration.source)
        refs.extend(cue.track for cue in self.music)
        return tuple(dict.fromkeys(refs))


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #


class EditPlanReport(AiveModel):
    """The Rule Engine's verdict on a plan.

    A single list of :class:`~app.models.common.Issue` rather than parallel error
    and warning lists, so that a new severity never changes the shape of the
    document the AI director has learned to read.
    """

    plan_project_id: str = Field(min_length=1)
    issues: tuple[Issue, ...] = ()
    normalised: bool = Field(
        default=False,
        description="True when the reported plan was rewritten, not just inspected.",
    )

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is Severity.WARNING)

    @property
    def ok(self) -> bool:
        """True when the plan can be rendered. Warnings do not block."""
        return not self.errors


__all__ = [
    "EDIT_PLAN_SCHEMA_VERSION",
    "DuckingSpec",
    "EditPlan",
    "EditPlanReport",
    "FramingSpec",
    "MusicCue",
    "NarrationTrack",
    "OutputSpec",
    "SubtitleCue",
    "TimelineClip",
    "Transition",
]
