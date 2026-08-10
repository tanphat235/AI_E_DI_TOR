"""Tests for the Edit Plan - the contract everything else depends on.

These tests defend three properties that matter more than the rest:

* **Round-trip stability.** A plan written to disk and read back must be identical,
  or a render is not reproducible.
* **Strict acceptance.** The plan is authored by a language model. A typo'd key or a
  future schema version must be rejected at the door, not half-rendered.
* **Author-friendly optionality.** ``timeline_start`` may be omitted, because
  expecting an LLM to sum forty float durations correctly is a bug waiting to
  happen.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.models.common import AspectRatio, Issue, MediaRef, Severity, TimeRange, TransitionKind
from app.models.edit_plan import (
    EDIT_PLAN_SCHEMA_VERSION,
    DuckingSpec,
    EditPlan,
    EditPlanReport,
    FramingSpec,
    MusicCue,
    NarrationTrack,
    OutputSpec,
    SubtitleCue,
    TimelineClip,
    Transition,
)


def _clip(
    clip_id: str = "c1", *, start: float = 0.0, end: float = 4.0, **kwargs: object
) -> TimelineClip:
    return TimelineClip(
        id=clip_id,
        source=MediaRef(path="raw/001.mp4"),
        source_range=TimeRange(start=start, end=end),
        reason="test clip",
        **kwargs,  # type: ignore[arg-type]
    )


class TestTransition:
    def test_cut_helper(self) -> None:
        cut = Transition.cut()
        assert cut.kind is TransitionKind.CUT
        assert cut.duration == 0.0

    def test_cut_may_not_have_duration(self) -> None:
        with pytest.raises(ValidationError, match="instantaneous"):
            Transition(kind=TransitionKind.CUT, duration=0.5)

    def test_non_cut_requires_duration(self) -> None:
        with pytest.raises(ValidationError, match="positive duration"):
            Transition(kind=TransitionKind.DISSOLVE)

    def test_valid_dissolve(self) -> None:
        assert Transition(kind=TransitionKind.DISSOLVE, duration=0.4).duration == 0.4


class TestFramingSpec:
    def test_default_is_static(self) -> None:
        assert FramingSpec().is_static

    def test_a_zoom_is_not_static(self) -> None:
        assert not FramingSpec(zoom_start=1.0, zoom_end=1.4).is_static

    def test_a_crop_is_not_static(self) -> None:
        assert not FramingSpec(crop_to=AspectRatio.VERTICAL).is_static

    def test_zoom_below_one_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FramingSpec(zoom_start=0.5)


class TestTimelineClip:
    def test_timeline_duration_ignores_speed_of_one(self) -> None:
        assert _clip(start=10.0, end=16.0).timeline_duration == pytest.approx(6.0)

    def test_speed_compresses_timeline_duration(self) -> None:
        clip = _clip(start=0.0, end=6.0, speed=2.0)
        assert clip.timeline_duration == pytest.approx(3.0)

    def test_unplaced_clip_reports_it(self) -> None:
        clip = _clip()
        assert not clip.is_placed
        with pytest.raises(ValueError, match="no timeline_start"):
            _ = clip.timeline_range

    def test_placed_clip_exposes_its_range(self) -> None:
        clip = _clip(start=10.0, end=14.0, timeline_start=7.0)
        assert clip.is_placed
        assert clip.timeline_range == TimeRange(start=7.0, end=11.0)

    def test_reason_is_mandatory(self) -> None:
        """Provenance is not optional: a choice with no stated reason is unreviewable."""
        with pytest.raises(ValidationError):
            TimelineClip(
                id="c1",
                source=MediaRef(path="raw/001.mp4"),
                source_range=TimeRange(start=0.0, end=1.0),
            )  # type: ignore[call-arg]

    def test_source_audio_is_muted_by_default(self) -> None:
        assert _clip().mute_source_audio is True

    @pytest.mark.parametrize("bad_id", ["has space", "has/slash", ""])
    def test_id_must_be_a_safe_token(self, bad_id: str) -> None:
        with pytest.raises(ValidationError):
            _clip(bad_id)


class TestNarrationTrack:
    def test_timeline_duration_excludes_removed_gaps(self, narration_track: NarrationTrack) -> None:
        # kept 0-4 and 6-10, so 8 seconds survive out of 10.
        assert narration_track.timeline_duration == pytest.approx(8.0)

    def test_maps_source_time_into_timeline_time(self, narration_track: NarrationTrack) -> None:
        assert narration_track.source_to_timeline(2.0) == pytest.approx(2.0)
        # 7.0 is 1.0s into the second kept range, which begins at timeline 4.0.
        assert narration_track.source_to_timeline(7.0) == pytest.approx(5.0)

    def test_a_moment_inside_a_removed_gap_has_no_timeline_position(
        self, narration_track: NarrationTrack
    ) -> None:
        assert narration_track.source_to_timeline(5.0) is None

    def test_rejects_overlapping_kept_ranges(self) -> None:
        with pytest.raises(ValidationError, match="ascending and non-overlapping"):
            NarrationTrack(
                source=MediaRef(path="narration.wav"),
                kept_ranges=(TimeRange(start=0.0, end=5.0), TimeRange(start=4.0, end=8.0)),
            )

    def test_rejects_descending_kept_ranges(self) -> None:
        with pytest.raises(ValidationError):
            NarrationTrack(
                source=MediaRef(path="narration.wav"),
                kept_ranges=(TimeRange(start=6.0, end=9.0), TimeRange(start=0.0, end=4.0)),
            )

    def test_requires_at_least_one_kept_range(self) -> None:
        with pytest.raises(ValidationError):
            NarrationTrack(source=MediaRef(path="narration.wav"), kept_ranges=())


class TestMusicCue:
    def test_fades_may_not_exceed_the_cue(self) -> None:
        with pytest.raises(ValidationError, match="exceed the cue duration"):
            MusicCue(
                track=MediaRef(path="music/calm.mp3"),
                timeline_range=TimeRange(start=0.0, end=3.0),
                fade_in=2.0,
                fade_out=2.0,
            )

    def test_fades_that_fit_are_accepted(self) -> None:
        cue = MusicCue(
            track=MediaRef(path="music/calm.mp3"),
            timeline_range=TimeRange(start=0.0, end=30.0),
            fade_in=1.5,
            fade_out=2.0,
            ducking=DuckingSpec(),
        )
        assert cue.ducking is not None
        assert cue.ducking.gain_db == -12.0

    def test_ducking_gain_must_attenuate(self) -> None:
        """Ducking that *boosts* music under speech is never what was meant."""
        with pytest.raises(ValidationError):
            DuckingSpec(gain_db=6.0)


class TestConstraintsAreActuallyEnforced:
    """Regression cover for a silent pydantic pitfall.

    Constraints do not merge across an ``Annotated`` alias: given
    ``x: Seconds = Field(le=10.0)``, the alias's bounds win and the ``le`` is
    dropped with no error. These fields were all written that way originally and
    silently accepted absurd values. See the caution note in
    :mod:`app.models.common`.
    """

    def test_transition_duration_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            Transition(kind=TransitionKind.DISSOLVE, duration=60.0)

    def test_ducking_attack_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            DuckingSpec(attack=90.0)

    def test_ducking_release_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            DuckingSpec(release=90.0)

    def test_music_fade_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            MusicCue(
                track=MediaRef(path="music/calm.mp3"),
                timeline_range=TimeRange(start=0.0, end=600.0),
                fade_in=120.0,
            )

    @pytest.mark.parametrize(
        "field",
        ["duration"],
    )
    def test_the_json_schema_advertises_the_real_bound(self, field: str) -> None:
        """A schema claiming a limit it does not enforce misleads the plan author."""
        schema = Transition.model_json_schema()
        assert schema["properties"][field]["maximum"] == 10.0


class TestOutputSpec:
    def test_frame_duration(self) -> None:
        assert OutputSpec(fps=25.0).frame_duration == pytest.approx(0.04)

    def test_actual_ratio_reflects_the_pixel_dimensions(self) -> None:
        spec = OutputSpec(width=1080, height=1920, aspect_ratio=AspectRatio.VERTICAL)
        assert spec.actual_ratio == pytest.approx(9 / 16)


class TestEditPlan:
    def test_minimal_plan_is_valid(self, minimal_plan: EditPlan) -> None:
        assert minimal_plan.schema_version == EDIT_PLAN_SCHEMA_VERSION
        assert minimal_plan.timeline_duration == pytest.approx(4.0)
        assert not minimal_plan.is_placed

    def test_requires_at_least_one_clip(self) -> None:
        with pytest.raises(ValidationError):
            EditPlan(project_id="test", created_by="pytest", clips=())

    def test_clip_ids_must_be_unique(self) -> None:
        with pytest.raises(ValidationError, match="duplicate clip id"):
            EditPlan(
                project_id="test",
                created_by="pytest",
                clips=(_clip("dup"), _clip("dup", start=5.0, end=9.0)),
            )

    def test_timeline_duration_subtracts_transition_overlap(self) -> None:
        plan = EditPlan(
            project_id="test",
            created_by="pytest",
            clips=(
                _clip("c1", start=0.0, end=5.0),
                _clip(
                    "c2",
                    start=0.0,
                    end=5.0,
                    transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.5),
                ),
            ),
        )
        # 5 + 5 = 10, minus the half-second the dissolve overlaps.
        assert plan.timeline_duration == pytest.approx(9.5)

    def test_sources_are_deduplicated_and_ordered(self, narration_track: NarrationTrack) -> None:
        plan = EditPlan(
            project_id="test",
            created_by="pytest",
            narration=narration_track,
            clips=(_clip("c1"), _clip("c2", start=5.0, end=9.0)),
            music=(
                MusicCue(
                    track=MediaRef(path="music/calm.mp3"),
                    timeline_range=TimeRange(start=0.0, end=8.0),
                ),
            ),
        )
        # Both clips share one source file, so it must appear once.
        assert plan.sources == (
            MediaRef(path="raw/001.mp4"),
            MediaRef(path="narration.wav"),
            MediaRef(path="music/calm.mp3"),
        )

    def test_clip_lookup(self, minimal_plan: EditPlan) -> None:
        assert minimal_plan.clip_by_id("c001") is not None
        assert minimal_plan.clip_by_id("nope") is None

    def test_is_placed_only_when_every_clip_is(self) -> None:
        plan = EditPlan(
            project_id="test",
            created_by="pytest",
            clips=(_clip("c1", timeline_start=0.0), _clip("c2", start=5.0, end=9.0)),
        )
        assert not plan.is_placed

    # -- Serialisation contract -------------------------------------------- #

    def test_round_trips_exactly(self, minimal_plan: EditPlan) -> None:
        blob = minimal_plan.model_dump_json()
        assert EditPlan.model_validate_json(blob) == minimal_plan

    def test_rejects_an_unknown_key(self, minimal_plan: EditPlan) -> None:
        """A typo must fail loudly rather than silently change the edit."""
        payload = json.loads(minimal_plan.model_dump_json())
        payload["clipz"] = []
        with pytest.raises(ValidationError):
            EditPlan.model_validate(payload)

    def test_rejects_an_unknown_key_on_a_nested_model(self, minimal_plan: EditPlan) -> None:
        payload = json.loads(minimal_plan.model_dump_json())
        payload["clips"][0]["transiton_in"] = None
        with pytest.raises(ValidationError):
            EditPlan.model_validate(payload)

    def test_refuses_a_future_schema_version(self, minimal_plan: EditPlan) -> None:
        """An old build must refuse a newer plan, not misread it."""
        payload = json.loads(minimal_plan.model_dump_json())
        payload["schema_version"] = "2.0"
        with pytest.raises(ValidationError):
            EditPlan.model_validate(payload)

    def test_schema_version_may_be_omitted_when_authoring(self) -> None:
        """Ergonomics: an author should not have to remember the version."""
        plan = EditPlan(project_id="test", created_by="claude-code", clips=(_clip(),))
        assert plan.schema_version == EDIT_PLAN_SCHEMA_VERSION

    def test_json_schema_keeps_the_required_set_small(self) -> None:
        """The smaller the required set, the more likely a first draft validates."""
        schema = EditPlan.model_json_schema()
        assert set(schema["required"]) == {"project_id", "created_by", "clips"}

    def test_paths_serialise_posix_style(self, minimal_plan: EditPlan) -> None:
        assert "raw/001.mp4" in minimal_plan.model_dump_json()


class TestSubtitleCue:
    def test_line_count(self) -> None:
        assert SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="one line").line_count == 1
        assert SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="two\nlines").line_count == 2

    def test_default_style(self) -> None:
        cue = SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="hello")
        assert cue.style_id == "default"


class TestEditPlanReport:
    def test_partitions_by_severity(self) -> None:
        report = EditPlanReport(
            plan_project_id="test",
            issues=(
                Issue(code="a.b", severity=Severity.ERROR, message="broken"),
                Issue(code="c.d", severity=Severity.WARNING, message="odd"),
                Issue(code="e.f", severity=Severity.INFO, message="normalised"),
            ),
        )
        assert len(report.errors) == 1
        assert len(report.warnings) == 1
        assert not report.ok

    def test_warnings_alone_do_not_block_a_render(self) -> None:
        report = EditPlanReport(
            plan_project_id="test",
            issues=(Issue(code="c.d", severity=Severity.WARNING, message="odd"),),
        )
        assert report.ok

    def test_an_empty_report_is_ok(self) -> None:
        assert EditPlanReport(plan_project_id="test").ok
