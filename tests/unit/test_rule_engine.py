"""Tests for the Rule Engine: filters, rules, normalisers and the façade.

Everything here is pure computation over models, so no media is touched. That is by
design: the Rule Engine's job is to judge a plan against *measured facts*, and the facts
arrive as a :class:`RuleContext` the test can construct directly. Retuning a threshold and
re-checking should take milliseconds, not a re-analysis.

The tests are organised around what each rule *protects against*, because every one of
them corresponds to a mistake a language model plausibly makes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.rules import RuleSettings
from app.config.settings import SubtitleSettings
from app.models.common import (
    CameraMove,
    MediaRef,
    MotionLevel,
    Severity,
    ShotType,
    TimeRange,
    TransitionKind,
)
from app.models.edit_plan import (
    EditPlan,
    MusicCue,
    NarrationTrack,
    OutputSpec,
    SubtitleCue,
    TimelineClip,
    Transition,
)
from app.models.media import MediaProbe, VideoStreamInfo
from app.models.video import (
    ClipAnalysis,
    DuplicateGroup,
    FootageAnalysis,
    MotionStats,
    QualityScores,
    Scene,
    SceneTags,
)
from app.rule_engine.context import RuleContext
from app.rule_engine.engine import DefaultRuleEngine
from app.rule_engine.filters import (
    DuplicateFilter,
    DurationFilter,
    QualityFilter,
    filter_scenes,
)
from app.rule_engine.normalizers import (
    ClipPlacementNormalizer,
    MusicFitNormalizer,
    SourceRangeClampNormalizer,
    SubtitleTimingNormalizer,
    TransitionClampNormalizer,
    is_monotonic,
)
from app.rule_engine.rules import (
    ClipDurationRule,
    DuplicateUsageRule,
    MusicRule,
    NarrationCoverageRule,
    OutputSpecRule,
    SourceBoundsRule,
    SourceExistsRule,
    SubtitleRule,
    TimelineContinuityRule,
    TransitionFitRule,
)

CLIP_A = MediaRef(path="raw/001.mp4")
CLIP_B = MediaRef(path="raw/002.mp4")
MUSIC = MediaRef(path="music/calm.mp3")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _probe(ref: MediaRef, duration: float) -> MediaProbe:
    return MediaProbe(
        source=ref,
        duration=duration,
        size_bytes=1024,
        video=VideoStreamInfo(width=1920, height=1080, fps=30.0, codec="h264"),
    )


def _context(
    *,
    durations: dict[MediaRef, float] | None = None,
    suppressed: frozenset[str] = frozenset(),
    project_root: Path | None = None,
    rules: RuleSettings | None = None,
    subtitle: SubtitleSettings | None = None,
) -> RuleContext:
    probes = {ref: _probe(ref, duration) for ref, duration in (durations or {}).items()}
    return RuleContext(
        rules=rules or RuleSettings(),
        subtitle=subtitle or SubtitleSettings(),
        probes=probes,
        suppressed_scene_keys=suppressed,
        project_root=project_root,
    )


def _clip(
    clip_id: str,
    *,
    source: MediaRef = CLIP_A,
    start: float = 0.0,
    end: float = 4.0,
    **kwargs: object,
) -> TimelineClip:
    return TimelineClip(
        id=clip_id,
        source=source,
        source_range=TimeRange(start=start, end=end),
        reason="test",
        **kwargs,  # type: ignore[arg-type]
    )


def _plan(*clips: TimelineClip, **kwargs: object) -> EditPlan:
    return EditPlan(
        project_id="test",
        created_by="pytest",
        clips=clips or (_clip("c1"),),
        **kwargs,  # type: ignore[arg-type]
    )


def _scene(
    index: int,
    *,
    clip: MediaRef = CLIP_A,
    duration: float = 3.0,
    blur: float = 0.9,
    brightness: float = 0.6,
    stability: float = 0.9,
    overall: float = 0.85,
) -> Scene:
    return Scene(
        clip=clip,
        index=index,
        range=TimeRange(start=0.0, end=duration),
        quality=QualityScores(
            blur=blur, brightness=brightness, exposure=0.9, stability=stability, overall=overall
        ),
        motion=MotionStats(
            level=MotionLevel.LOW, mean_magnitude=1.0, camera_move=CameraMove.STATIC
        ),
        tags=SceneTags(provider="test"),
        shot_type=ShotType.MEDIUM,
    )


def _codes(issues: object) -> set[str]:
    return {issue.code for issue in issues}  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


class TestRuleContext:
    def test_durations_come_from_the_footage_analysis(self) -> None:
        from datetime import UTC, datetime

        footage = FootageAnalysis(
            clips=(
                ClipAnalysis(
                    clip=CLIP_A,
                    probe=_probe(CLIP_A, 30.0),
                    scenes=(),
                    analyzer_version="test/1",
                    analyzed_at=datetime.now(UTC),
                ),
            )
        )
        context = RuleContext.build(RuleSettings(), footage=footage)
        assert context.duration_of(CLIP_A) == 30.0

    def test_suppressed_keys_come_from_the_footage_analysis(self) -> None:
        footage = FootageAnalysis(
            duplicates=(
                DuplicateGroup(representative="001#0", duplicates=("002#0",), similarity=0.99),
            )
        )
        context = RuleContext.build(RuleSettings(), footage=footage)
        assert context.suppressed_scene_keys == frozenset({"002#0"})

    def test_an_unmeasured_file_is_none_not_an_error(self) -> None:
        """Validating before analysis must still work, just with fewer checks."""
        assert _context().duration_of(CLIP_A) is None

    def test_resolve_needs_a_project_root(self) -> None:
        assert _context().resolve(CLIP_A) is None

    def test_resolve_returns_an_absolute_path(self, tmp_path: Path) -> None:
        resolved = _context(project_root=tmp_path).resolve(CLIP_A)
        assert resolved == (tmp_path / "raw" / "001.mp4").resolve()


# --------------------------------------------------------------------------- #
# Scene filters (pre-planning)
# --------------------------------------------------------------------------- #


class TestQualityFilter:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"blur": 0.1}, "quality.too_soft"),
            ({"brightness": 0.05}, "quality.too_dark"),
            ({"brightness": 0.99}, "quality.too_bright"),
            ({"stability": 0.1}, "quality.unstable"),
            ({"overall": 0.2}, "quality.below_floor"),
        ],
    )
    def test_each_metric_is_checked_separately(self, kwargs: dict, expected: str) -> None:
        """A shot can be sharp, bright and unusably shaky; an average would pass it."""
        issue = QualityFilter().reject_reason(_scene(0, **kwargs), rules=RuleSettings())
        assert issue is not None
        assert issue.code == expected

    def test_a_good_scene_passes(self) -> None:
        assert QualityFilter().reject_reason(_scene(0), rules=RuleSettings()) is None
        assert QualityFilter().is_eligible(_scene(0), rules=RuleSettings())

    def test_rejection_is_a_warning_not_an_error(self) -> None:
        """Coverage beats perfection: a soft shot is better than a gap."""
        issue = QualityFilter().reject_reason(_scene(0, blur=0.1), rules=RuleSettings())
        assert issue is not None
        assert issue.severity is Severity.WARNING
        assert "gap is worse" in (issue.hint or "")


class TestDurationFilter:
    def test_a_scene_shorter_than_the_minimum_clip_is_rejected(self) -> None:
        issue = DurationFilter().reject_reason(_scene(0, duration=0.5), rules=RuleSettings())
        assert issue is not None
        assert issue.code == "scene.too_short"

    def test_a_long_enough_scene_passes(self) -> None:
        assert DurationFilter().reject_reason(_scene(0, duration=5.0), rules=RuleSettings()) is None

    def test_the_threshold_follows_config(self) -> None:
        rules = RuleSettings(min_clip_duration=4.0)
        assert not DurationFilter().is_eligible(_scene(0, duration=3.0), rules=rules)


class TestDuplicateFilter:
    def test_a_suppressed_scene_is_rejected(self) -> None:
        filter_ = DuplicateFilter(frozenset({"001#0"}))
        issue = filter_.reject_reason(_scene(0), rules=RuleSettings())
        assert issue is not None
        assert issue.code == "scene.duplicate"

    def test_the_surviving_take_is_named_in_the_message(self) -> None:
        """Actionable: the director needs to know what to use instead."""
        filter_ = DuplicateFilter(frozenset({"001#0"}), keeper_of={"001#0": "002#3"})
        issue = filter_.reject_reason(_scene(0), rules=RuleSettings())
        assert issue is not None
        assert "002#3" in issue.message

    def test_a_scene_that_is_not_a_duplicate_passes(self) -> None:
        filter_ = DuplicateFilter(frozenset({"999#9"}))
        assert filter_.reject_reason(_scene(0), rules=RuleSettings()) is None


class TestFilterScenes:
    def _footage(self, *scenes: Scene, duplicates: tuple = ()) -> FootageAnalysis:
        from datetime import UTC, datetime

        return FootageAnalysis(
            clips=(
                ClipAnalysis(
                    clip=CLIP_A,
                    probe=_probe(CLIP_A, 30.0),
                    scenes=scenes,
                    analyzer_version="test/1",
                    analyzed_at=datetime.now(UTC),
                ),
            ),
            duplicates=duplicates,
        )

    def test_eligible_and_rejected_are_partitioned(self) -> None:
        footage = self._footage(_scene(0), _scene(1, blur=0.05))
        report = filter_scenes(footage, rules=RuleSettings())
        assert len(report.eligible) == 1
        assert len(report.rejected) == 1

    def test_all_reasons_are_reported_not_only_the_first(self) -> None:
        """Retuning one threshold and re-running to find the next wastes the user's time."""
        footage = self._footage(_scene(0, duration=0.4, blur=0.05))
        verdict = filter_scenes(footage, rules=RuleSettings()).rejected[0]
        assert {"scene.too_short", "quality.too_soft"} <= _codes(verdict.issues)

    def test_rejected_scenes_are_kept_not_deleted(self) -> None:
        """A rejected scene is still usable when nothing else covers a beat."""
        footage = self._footage(_scene(0, blur=0.05))
        report = filter_scenes(footage, rules=RuleSettings())
        assert len(report.verdicts) == 1
        assert report.verdict_for("001#0") is not None

    def test_eligible_duration_is_summed(self) -> None:
        footage = self._footage(_scene(0, duration=3.0), _scene(1, duration=5.0))
        report = filter_scenes(footage, rules=RuleSettings())
        assert report.total_eligible_duration == pytest.approx(8.0)

    def test_no_scenes(self) -> None:
        report = filter_scenes(FootageAnalysis(), rules=RuleSettings())
        assert report.eligible == ()
        assert report.total_eligible_duration == 0.0


# --------------------------------------------------------------------------- #
# Plan rules (post-planning)
# --------------------------------------------------------------------------- #


class TestSourceExistsRule:
    def test_a_missing_file_is_an_error(self, tmp_path: Path) -> None:
        issues = SourceExistsRule().check(_plan(), _context(project_root=tmp_path))
        assert _codes(issues) == {"source.missing"}

    def test_an_existing_file_passes(self, tmp_path: Path) -> None:
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "001.mp4").write_bytes(b"\0")
        assert SourceExistsRule().check(_plan(), _context(project_root=tmp_path)) == ()

    def test_without_a_project_root_the_check_is_skipped(self) -> None:
        """A plan can legitimately be validated for shape alone."""
        assert SourceExistsRule().check(_plan(), _context()) == ()


class TestSourceBoundsRule:
    def test_a_range_past_the_end_is_an_error(self) -> None:
        """The mistake a model actually makes: 52.0 looks fine on a 30-second clip."""
        plan = _plan(_clip("c1", start=45.0, end=52.0))
        issues = SourceBoundsRule().check(plan, _context(durations={CLIP_A: 30.0}))
        assert _codes(issues) == {"source.out_of_bounds"}
        assert issues[0].severity is Severity.ERROR

    def test_a_start_past_the_end_is_an_error(self) -> None:
        plan = _plan(_clip("c1", start=45.0, end=48.0))
        issues = SourceBoundsRule().check(plan, _context(durations={CLIP_A: 30.0}))
        assert _codes(issues) == {"source.out_of_bounds"}

    def test_a_range_inside_the_file_passes(self) -> None:
        plan = _plan(_clip("c1", start=5.0, end=9.0))
        assert SourceBoundsRule().check(plan, _context(durations={CLIP_A: 30.0})) == ()

    def test_an_unmeasured_clip_reports_that_the_check_was_skipped(self) -> None:
        """Silence would imply the range had been verified."""
        issues = SourceBoundsRule().check(_plan(), _context())
        assert _codes(issues) == {"source.unverified"}
        assert issues[0].severity is Severity.INFO

    def test_a_range_ending_exactly_at_the_duration_passes(self) -> None:
        plan = _plan(_clip("c1", start=0.0, end=30.0))
        assert SourceBoundsRule().check(plan, _context(durations={CLIP_A: 30.0})) == ()


class TestClipDurationRule:
    def test_too_short_is_an_error(self) -> None:
        """Below a second a cut reads as a glitch, not a shot."""
        plan = _plan(_clip("c1", start=0.0, end=0.4))
        issues = ClipDurationRule().check(plan, _context())
        assert _codes(issues) == {"clip.too_short"}
        assert issues[0].severity is Severity.ERROR

    def test_too_long_is_only_a_warning(self) -> None:
        """A long hold is sometimes right; refusing to render it would overrule the editor."""
        plan = _plan(_clip("c1", start=0.0, end=20.0))
        issues = ClipDurationRule().check(plan, _context())
        assert _codes(issues) == {"clip.too_long"}
        assert issues[0].severity is Severity.WARNING

    def test_speed_is_accounted_for(self) -> None:
        """A 2s range at 4x is 0.5s on the timeline, and that is what matters."""
        plan = _plan(_clip("c1", start=0.0, end=2.0, speed=4.0))
        assert _codes(ClipDurationRule().check(plan, _context())) == {"clip.too_short"}

    def test_a_reasonable_clip_passes(self) -> None:
        assert ClipDurationRule().check(_plan(_clip("c1", end=4.0)), _context()) == ()


class TestTimelineContinuityRule:
    def test_an_unplaced_plan_is_reported_as_info(self) -> None:
        """Telling the user to normalise is more useful than calling the plan broken."""
        issues = TimelineContinuityRule().check(_plan(), _context())
        assert _codes(issues) == {"plan.unplaced"}
        assert issues[0].severity is Severity.INFO

    def test_correctly_packed_clips_pass(self) -> None:
        plan = _plan(
            _clip("c1", end=4.0, timeline_start=0.0),
            _clip("c2", end=4.0, timeline_start=4.0),
        )
        assert TimelineContinuityRule().check(plan, _context()) == ()

    def test_a_dissolve_overlap_is_expected_not_an_error(self) -> None:
        """That overlap is what a dissolve physically is."""
        plan = _plan(
            _clip("c1", end=4.0, timeline_start=0.0),
            _clip(
                "c2",
                end=4.0,
                timeline_start=3.6,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.4),
            ),
        )
        assert TimelineContinuityRule().check(plan, _context()) == ()

    def test_overlap_beyond_the_transition_is_an_error(self) -> None:
        plan = _plan(
            _clip("c1", end=4.0, timeline_start=0.0),
            _clip("c2", end=4.0, timeline_start=2.0),
        )
        issues = TimelineContinuityRule().check(plan, _context())
        assert _codes(issues) == {"clip.overlap"}
        assert issues[0].severity is Severity.ERROR

    def test_a_gap_is_a_warning(self) -> None:
        plan = _plan(
            _clip("c1", end=4.0, timeline_start=0.0),
            _clip("c2", end=4.0, timeline_start=6.0),
        )
        issues = TimelineContinuityRule().check(plan, _context())
        assert "clip.gap" in _codes(issues)

    def test_opening_on_black_is_a_warning(self) -> None:
        plan = _plan(_clip("c1", end=4.0, timeline_start=2.0))
        assert "plan.late_start" in _codes(TimelineContinuityRule().check(plan, _context()))


class TestTransitionFitRule:
    def test_a_transition_consuming_too_much_of_a_clip_is_an_error(self) -> None:
        """A dissolve over most of a shot means the shot never resolves."""
        plan = _plan(
            _clip("c1", end=2.0),
            _clip(
                "c2",
                end=2.0,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=1.5),
            ),
        )
        issues = TransitionFitRule().check(plan, _context())
        assert _codes(issues) == {"transition.too_long"}

    def test_the_hint_names_the_admissible_duration(self) -> None:
        plan = _plan(
            _clip("c1", end=2.0),
            _clip(
                "c2",
                end=2.0,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=1.5),
            ),
        )
        issues = TransitionFitRule().check(plan, _context())
        assert "0.50s" in (issues[0].hint or "")

    def test_a_short_transition_on_long_clips_passes(self) -> None:
        plan = _plan(
            _clip("c1", end=6.0),
            _clip(
                "c2",
                end=6.0,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.4),
            ),
        )
        assert TransitionFitRule().check(plan, _context()) == ()

    def test_a_hard_cut_is_never_too_long(self) -> None:
        plan = _plan(_clip("c1", end=2.0), _clip("c2", end=2.0, transition_in=Transition.cut()))
        assert TransitionFitRule().check(plan, _context()) == ()

    def test_the_shorter_of_the_two_clips_governs(self) -> None:
        plan = _plan(
            _clip("c1", end=1.6),
            _clip(
                "c2",
                end=20.0,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.5),
            ),
        )
        assert _codes(TransitionFitRule().check(plan, _context())) == {"transition.too_long"}


class TestDuplicateUsageRule:
    def test_using_a_suppressed_scene_is_a_warning(self) -> None:
        plan = _plan(_clip("c1", scene_key="002#1"))
        issues = DuplicateUsageRule().check(plan, _context(suppressed=frozenset({"002#1"})))
        assert "scene.duplicate_used" in _codes(issues)

    def test_reusing_the_same_scene_twice_is_a_warning(self) -> None:
        """Whatever the duplicate groups say, the audience notices a repeat."""
        plan = _plan(_clip("c1", scene_key="001#0"), _clip("c2", scene_key="001#0"))
        issues = DuplicateUsageRule().check(plan, _context(suppressed=frozenset({"x#0"})))
        assert "scene.reused" in _codes(issues)

    def test_distinct_scenes_pass(self) -> None:
        plan = _plan(_clip("c1", scene_key="001#0"), _clip("c2", scene_key="001#1"))
        assert DuplicateUsageRule().check(plan, _context(suppressed=frozenset({"x#0"}))) == ()

    def test_clips_without_a_scene_key_are_not_checked(self) -> None:
        assert DuplicateUsageRule().check(_plan(), _context(suppressed=frozenset({"a#0"}))) == ()

    def test_nothing_suppressed_means_nothing_to_check(self) -> None:
        plan = _plan(_clip("c1", scene_key="001#0"))
        assert DuplicateUsageRule().check(plan, _context()) == ()


class TestNarrationCoverageRule:
    def _narration(self, duration: float) -> NarrationTrack:
        return NarrationTrack(
            source=MediaRef(path="narration.wav"),
            kept_ranges=(TimeRange(start=0.0, end=duration),),
        )

    def test_narration_outliving_the_picture_is_an_error(self) -> None:
        """The video would end on black with someone still talking."""
        plan = _plan(_clip("c1", end=4.0), narration=self._narration(20.0))
        issues = NarrationCoverageRule().check(plan, _context())
        assert _codes(issues) == {"narration.uncovered"}
        assert issues[0].severity is Severity.ERROR

    def test_matching_durations_pass(self) -> None:
        plan = _plan(_clip("c1", end=8.0), narration=self._narration(8.0))
        assert NarrationCoverageRule().check(plan, _context()) == ()

    def test_a_long_picture_tail_is_only_a_warning(self) -> None:
        plan = _plan(_clip("c1", end=8.0), narration=self._narration(3.0))
        issues = NarrationCoverageRule().check(plan, _context())
        assert _codes(issues) == {"narration.trailing_picture"}
        assert issues[0].severity is Severity.WARNING

    def test_no_narration_means_nothing_to_cover(self) -> None:
        assert NarrationCoverageRule().check(_plan(), _context()) == ()

    def test_a_small_shortfall_is_tolerated(self) -> None:
        """A tenth of a second is rounding, not a failure."""
        plan = _plan(_clip("c1", end=8.0), narration=self._narration(8.2))
        assert NarrationCoverageRule().check(plan, _context()) == ()


class TestSubtitleRule:
    def _plan_with(self, *cues: SubtitleCue) -> EditPlan:
        return _plan(_clip("c1", end=20.0), subtitles=cues)

    def test_a_cue_too_brief_to_read_is_flagged(self) -> None:
        plan = self._plan_with(SubtitleCue(range=TimeRange(start=0.0, end=0.2), text="Hi"))
        assert "subtitle.too_brief" in _codes(SubtitleRule().check(plan, _context()))

    def test_a_cue_held_too_long_is_flagged(self) -> None:
        plan = self._plan_with(SubtitleCue(range=TimeRange(start=0.0, end=15.0), text="Long"))
        assert "subtitle.too_long" in _codes(SubtitleRule().check(plan, _context()))

    def test_overlapping_cues_are_an_error(self) -> None:
        plan = self._plan_with(
            SubtitleCue(range=TimeRange(start=0.0, end=3.0), text="First"),
            SubtitleCue(range=TimeRange(start=2.0, end=5.0), text="Second"),
        )
        issues = SubtitleRule().check(plan, _context())
        assert "subtitle.overlap" in _codes(issues)

    def test_a_cue_past_the_end_of_the_video_names_the_real_cause(self) -> None:
        """The classic authoring mistake: cues timed against the raw narration."""
        plan = self._plan_with(SubtitleCue(range=TimeRange(start=50.0, end=52.0), text="Late"))
        issues = SubtitleRule().check(plan, _context())
        past = next(issue for issue in issues if issue.code == "subtitle.past_end")
        assert past.severity is Severity.ERROR
        assert "TIMELINE time" in (past.hint or "")

    def test_a_long_line_is_flagged(self) -> None:
        plan = self._plan_with(SubtitleCue(range=TimeRange(start=0.0, end=3.0), text="x" * 80))
        assert "subtitle.line_too_long" in _codes(SubtitleRule().check(plan, _context()))

    def test_too_many_lines_is_flagged(self) -> None:
        plan = self._plan_with(SubtitleCue(range=TimeRange(start=0.0, end=3.0), text="a\nb\nc\nd"))
        assert "subtitle.too_many_lines" in _codes(SubtitleRule().check(plan, _context()))

    def test_good_cues_pass(self) -> None:
        plan = self._plan_with(
            SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="First line."),
            SubtitleCue(range=TimeRange(start=2.5, end=4.5), text="Second line."),
        )
        assert SubtitleRule().check(plan, _context()) == ()

    def test_no_subtitles_is_fine(self) -> None:
        assert SubtitleRule().check(_plan(), _context()) == ()


class TestMusicRule:
    def _cue(self, **kwargs: object) -> MusicCue:
        defaults: dict[str, object] = {
            "track": MUSIC,
            "timeline_range": TimeRange(start=0.0, end=5.0),
        }
        return MusicCue(**{**defaults, **kwargs})  # type: ignore[arg-type]

    def test_an_offset_past_the_end_of_the_track_is_an_error(self) -> None:
        """The field that goes wrong: it yields silence, not an FFmpeg error."""
        plan = _plan(_clip("c1", end=10.0), music=(self._cue(source_offset=20.0),))
        issues = MusicRule().check(plan, _context(durations={MUSIC: 8.0}))
        assert _codes(issues) == {"music.offset_past_end"}
        assert issues[0].severity is Severity.ERROR

    def test_a_bed_that_runs_out_is_a_warning(self) -> None:
        plan = _plan(_clip("c1", end=10.0), music=(self._cue(source_offset=7.0),))
        assert "music.too_short" in _codes(
            MusicRule().check(plan, _context(durations={MUSIC: 8.0}))
        )

    def test_a_cue_starting_after_the_video_ends_is_a_warning(self) -> None:
        plan = _plan(
            _clip("c1", end=4.0),
            music=(self._cue(timeline_range=TimeRange(start=30.0, end=35.0)),),
        )
        assert "music.past_end" in _codes(
            MusicRule().check(plan, _context(durations={MUSIC: 60.0}))
        )

    def test_a_fitting_cue_passes(self) -> None:
        plan = _plan(_clip("c1", end=10.0), music=(self._cue(source_offset=2.0),))
        assert MusicRule().check(plan, _context(durations={MUSIC: 60.0})) == ()

    def test_an_unmeasured_track_skips_the_track_checks(self) -> None:
        plan = _plan(_clip("c1", end=10.0), music=(self._cue(source_offset=999.0),))
        assert MusicRule().check(plan, _context()) == ()

    def test_no_music_is_fine(self) -> None:
        assert MusicRule().check(_plan(), _context()) == ()


class TestOutputSpecRule:
    def test_a_mislabelled_aspect_ratio_is_a_warning(self) -> None:
        """The renderer follows width and height; the label would mislead a reader."""
        from app.models.common import AspectRatio

        plan = _plan(output=OutputSpec(aspect_ratio=AspectRatio.VERTICAL, width=1920, height=1080))
        assert "output.aspect_mismatch" in _codes(OutputSpecRule().check(plan, _context()))

    def test_a_consistent_spec_passes(self) -> None:
        plan = _plan(output=OutputSpec(width=1920, height=1080))
        assert OutputSpecRule().check(plan, _context()) == ()

    def test_a_sub_frame_transition_is_flagged(self) -> None:
        plan = _plan(
            _clip("c1", end=4.0),
            _clip(
                "c2",
                end=4.0,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.01),
            ),
            output=OutputSpec(fps=30.0),
        )
        assert "transition.sub_frame" in _codes(OutputSpecRule().check(plan, _context()))


# --------------------------------------------------------------------------- #
# Normalisers
# --------------------------------------------------------------------------- #


class TestClipPlacementNormalizer:
    def test_clips_are_packed_end_to_end(self) -> None:
        """The headline feature: the director lists clips and omits the arithmetic."""
        plan = _plan(_clip("c1", end=4.0), _clip("c2", end=3.0), _clip("c3", end=2.0))
        placed, issues = ClipPlacementNormalizer().normalize(plan, _context())
        assert [clip.timeline_start for clip in placed.clips] == [0.0, 4.0, 7.0]
        assert len(issues) == 3
        assert all(issue.severity is Severity.INFO for issue in issues)

    def test_a_transition_pulls_its_clip_back_over_the_previous_one(self) -> None:
        """That overlap is what a dissolve physically is."""
        plan = _plan(
            _clip("c1", end=4.0),
            _clip(
                "c2",
                end=4.0,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.4),
            ),
        )
        placed, _issues = ClipPlacementNormalizer().normalize(plan, _context())
        assert placed.clips[1].timeline_start == pytest.approx(3.6)
        assert placed.timeline_duration == pytest.approx(7.6)

    def test_the_first_clips_transition_does_not_pull_it_negative(self) -> None:
        """A fade from black is not an overlap with a previous clip."""
        plan = _plan(
            _clip(
                "c1",
                end=4.0,
                transition_in=Transition(kind=TransitionKind.FADE, duration=1.0),
            )
        )
        placed, _issues = ClipPlacementNormalizer().normalize(plan, _context())
        assert placed.clips[0].timeline_start == 0.0

    def test_an_explicit_position_is_respected(self) -> None:
        """The director may need an insert to land on a beat."""
        plan = _plan(_clip("c1", end=4.0), _clip("c2", end=2.0, timeline_start=10.0))
        placed, _issues = ClipPlacementNormalizer().normalize(plan, _context())
        assert placed.clips[1].timeline_start == 10.0

    def test_an_already_placed_plan_is_untouched(self) -> None:
        plan = _plan(_clip("c1", end=4.0, timeline_start=0.0))
        placed, issues = ClipPlacementNormalizer().normalize(plan, _context())
        assert placed is plan
        assert issues == ()

    def test_speed_is_accounted_for_in_placement(self) -> None:
        plan = _plan(_clip("c1", end=8.0, speed=2.0), _clip("c2", end=2.0))
        placed, _issues = ClipPlacementNormalizer().normalize(plan, _context())
        assert placed.clips[1].timeline_start == pytest.approx(4.0)

    def test_placement_is_monotonic(self) -> None:
        plan = _plan(*[_clip(f"c{index}", end=2.0) for index in range(6)])
        placed, _issues = ClipPlacementNormalizer().normalize(plan, _context())
        assert is_monotonic(placed)


class TestSourceRangeClampNormalizer:
    def test_a_range_past_the_end_is_shortened(self) -> None:
        plan = _plan(_clip("c1", start=5.0, end=40.0))
        clamped, issues = SourceRangeClampNormalizer().normalize(
            plan, _context(durations={CLIP_A: 30.0})
        )
        assert clamped.clips[0].source_range.end == 30.0
        assert _codes(issues) == {"normalize.clamped_source"}

    def test_a_range_starting_past_the_end_is_left_alone(self) -> None:
        """Nothing to salvage; inventing a range would substitute a shot nobody chose."""
        plan = _plan(_clip("c1", start=45.0, end=50.0))
        clamped, issues = SourceRangeClampNormalizer().normalize(
            plan, _context(durations={CLIP_A: 30.0})
        )
        assert clamped.clips[0].source_range.start == 45.0
        assert issues == ()

    def test_an_in_bounds_range_is_untouched(self) -> None:
        plan = _plan(_clip("c1", start=0.0, end=4.0))
        clamped, issues = SourceRangeClampNormalizer().normalize(
            plan, _context(durations={CLIP_A: 30.0})
        )
        assert clamped is plan
        assert issues == ()

    def test_an_unmeasured_clip_is_untouched(self) -> None:
        plan = _plan(_clip("c1", start=0.0, end=999.0))
        clamped, issues = SourceRangeClampNormalizer().normalize(plan, _context())
        assert clamped is plan
        assert issues == ()


class TestTransitionClampNormalizer:
    def test_an_oversized_transition_is_shortened(self) -> None:
        plan = _plan(
            _clip("c1", end=4.0),
            _clip(
                "c2",
                end=4.0,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=3.0),
            ),
        )
        clamped, issues = TransitionClampNormalizer().normalize(plan, _context())
        transition = clamped.clips[1].transition_in
        assert transition is not None
        assert transition.duration == pytest.approx(1.0)  # 4.0 * 0.25
        assert transition.kind is TransitionKind.DISSOLVE
        assert _codes(issues) == {"normalize.clamped_transition"}

    def test_a_transition_that_cannot_fit_a_frame_becomes_a_cut(self) -> None:
        """It would render as a hard cut anyway; saying so is more honest."""
        plan = _plan(
            _clip("c1", start=0.0, end=1.3),
            _clip(
                "c2",
                start=0.0,
                end=1.3,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=1.0),
            ),
            output=OutputSpec(fps=1.0),  # one frame is a whole second
        )
        clamped, issues = TransitionClampNormalizer().normalize(plan, _context())
        transition = clamped.clips[1].transition_in
        assert transition is not None
        assert transition.kind is TransitionKind.CUT
        assert _codes(issues) == {"normalize.transition_to_cut"}

    def test_a_fitting_transition_is_untouched(self) -> None:
        plan = _plan(
            _clip("c1", end=6.0),
            _clip(
                "c2",
                end=6.0,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.4),
            ),
        )
        clamped, issues = TransitionClampNormalizer().normalize(plan, _context())
        assert clamped is plan
        assert issues == ()


class TestSubtitleTimingNormalizer:
    def test_a_brief_cue_is_extended(self) -> None:
        plan = _plan(
            _clip("c1", end=20.0),
            subtitles=(SubtitleCue(range=TimeRange(start=0.0, end=0.1), text="Hi"),),
        )
        fixed, issues = SubtitleTimingNormalizer().normalize(plan, _context())
        assert fixed.subtitles[0].range.duration == pytest.approx(0.7)
        assert _codes(issues) == {"normalize.extended_cue"}

    def test_overlapping_cues_are_separated(self) -> None:
        plan = _plan(
            _clip("c1", end=20.0),
            subtitles=(
                SubtitleCue(range=TimeRange(start=0.0, end=3.0), text="First"),
                SubtitleCue(range=TimeRange(start=2.0, end=5.0), text="Second"),
            ),
        )
        fixed, issues = SubtitleTimingNormalizer().normalize(plan, _context())
        assert fixed.subtitles[0].range.end <= fixed.subtitles[1].range.start
        assert "normalize.separated_cues" in _codes(issues)

    def test_an_impossible_cue_is_dropped_rather_than_flashed(self) -> None:
        plan = _plan(
            _clip("c1", end=20.0),
            subtitles=(
                SubtitleCue(range=TimeRange(start=0.0, end=3.0), text="First"),
                SubtitleCue(range=TimeRange(start=0.0, end=0.5), text="Impossible"),
            ),
        )
        fixed, issues = SubtitleTimingNormalizer().normalize(plan, _context())
        assert len(fixed.subtitles) == 1
        assert "normalize.dropped_cue" in _codes(issues)

    def test_good_cues_are_untouched(self) -> None:
        plan = _plan(
            _clip("c1", end=20.0),
            subtitles=(
                SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="First"),
                SubtitleCue(range=TimeRange(start=2.5, end=4.5), text="Second"),
            ),
        )
        fixed, issues = SubtitleTimingNormalizer().normalize(plan, _context())
        assert fixed is plan
        assert issues == ()


class TestMusicFitNormalizer:
    def test_a_cue_past_the_end_of_the_video_is_shortened(self) -> None:
        plan = _plan(
            _clip("c1", end=5.0),
            music=(MusicCue(track=MUSIC, timeline_range=TimeRange(start=0.0, end=30.0)),),
        )
        fixed, issues = MusicFitNormalizer().normalize(plan, _context(durations={MUSIC: 60.0}))
        assert fixed.music[0].timeline_range.end == pytest.approx(5.0)
        assert _codes(issues) == {"normalize.shortened_music"}

    def test_a_cue_longer_than_the_remaining_track_is_shortened(self) -> None:
        plan = _plan(
            _clip("c1", end=30.0),
            music=(
                MusicCue(
                    track=MUSIC,
                    timeline_range=TimeRange(start=0.0, end=20.0),
                    source_offset=7.0,
                ),
            ),
        )
        fixed, issues = MusicFitNormalizer().normalize(plan, _context(durations={MUSIC: 8.0}))
        assert fixed.music[0].timeline_range.end == pytest.approx(1.0)
        assert issues

    def test_fades_are_scaled_so_the_model_stays_valid(self) -> None:
        """A normaliser must not produce a plan that fails validation."""
        plan = _plan(
            _clip("c1", end=30.0),
            music=(
                MusicCue(
                    track=MUSIC,
                    timeline_range=TimeRange(start=0.0, end=20.0),
                    source_offset=7.0,
                    fade_in=2.0,
                    fade_out=2.0,
                ),
            ),
        )
        fixed, _issues = MusicFitNormalizer().normalize(plan, _context(durations={MUSIC: 8.0}))
        cue = fixed.music[0]
        assert cue.fade_in + cue.fade_out <= cue.timeline_range.duration

    def test_a_fitting_cue_is_untouched(self) -> None:
        plan = _plan(
            _clip("c1", end=30.0),
            music=(MusicCue(track=MUSIC, timeline_range=TimeRange(start=0.0, end=10.0)),),
        )
        fixed, issues = MusicFitNormalizer().normalize(plan, _context(durations={MUSIC: 60.0}))
        assert fixed is plan
        assert issues == ()


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


class TestRuleEngine:
    @pytest.fixture
    def engine(self) -> DefaultRuleEngine:
        return DefaultRuleEngine()

    def test_a_good_plan_validates_cleanly(self, engine: DefaultRuleEngine) -> None:
        plan = _plan(
            _clip("c1", start=0.0, end=4.0, timeline_start=0.0),
            _clip("c2", start=10.0, end=14.0, timeline_start=4.0),
        )
        report = engine.validate(plan, _context(durations={CLIP_A: 30.0}))
        assert report.ok
        assert report.errors == ()

    def test_validate_changes_nothing(self, engine: DefaultRuleEngine) -> None:
        plan = _plan(_clip("c1", end=4.0))
        engine.validate(plan, _context())
        assert plan.clips[0].timeline_start is None

    def test_normalize_places_and_then_validates(self, engine: DefaultRuleEngine) -> None:
        plan = _plan(_clip("c1", end=4.0), _clip("c2", end=4.0))
        normalised, report = engine.normalize(plan, _context(durations={CLIP_A: 30.0}))
        assert normalised.is_placed
        assert report.normalised is True
        assert report.ok

    def test_normalisation_reports_every_change(self, engine: DefaultRuleEngine) -> None:
        """A silent fix-up is indistinguishable from a bug."""
        plan = _plan(_clip("c1", start=0.0, end=40.0))
        _normalised, report = engine.normalize(plan, _context(durations={CLIP_A: 30.0}))
        changes = [issue for issue in report.issues if issue.severity is Severity.INFO]
        assert any(issue.code == "normalize.clamped_source" for issue in changes)

    def test_normalisation_resolves_interacting_fixes(self, engine: DefaultRuleEngine) -> None:
        """Clamping a source shortens a clip, which changes the transition cap.

        One pass would leave the transition over its (new) limit, so the chain repeats.
        """
        plan = _plan(
            _clip("c1", start=0.0, end=4.0),
            _clip(
                "c2",
                start=0.0,
                end=40.0,  # clamps to 6.0, so the cap becomes 1.5
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=3.0),
            ),
        )
        normalised, report = engine.normalize(plan, _context(durations={CLIP_A: 6.0}))
        transition = normalised.clips[1].transition_in
        assert transition is not None
        assert transition.duration <= 1.0 + 1e-6  # 0.25 * min(4.0, 6.0)
        assert "transition.too_long" not in _codes(report.issues)

    def test_normalising_an_already_valid_plan_is_a_no_op(self, engine: DefaultRuleEngine) -> None:
        plan = _plan(
            _clip("c1", start=0.0, end=4.0, timeline_start=0.0),
            _clip("c2", start=10.0, end=14.0, timeline_start=4.0),
        )
        normalised, report = engine.normalize(plan, _context(durations={CLIP_A: 30.0}))
        assert normalised == plan
        assert [issue for issue in report.issues if issue.severity is Severity.INFO] == []

    def test_errors_normalisation_cannot_fix_survive(self, engine: DefaultRuleEngine) -> None:
        """Normalisation never invents editorial intent."""
        plan = _plan(_clip("c1", start=0.0, end=0.3))
        _normalised, report = engine.normalize(plan, _context(durations={CLIP_A: 30.0}))
        assert "clip.too_short" in _codes(report.errors)
        assert not report.ok

    def test_every_rule_and_normaliser_is_registered(self, engine: DefaultRuleEngine) -> None:
        assert len(engine.rule_names()) == 10
        assert len(engine.normalizer_names()) == 5

    def test_source_clamping_runs_before_placement(self, engine: DefaultRuleEngine) -> None:
        """Placing first would compute positions from durations about to change."""
        plan = _plan(_clip("c1", start=0.0, end=40.0), _clip("c2", start=0.0, end=4.0))
        normalised, _report = engine.normalize(plan, _context(durations={CLIP_A: 10.0}))
        # c1 clamps to 10s, so c2 must be placed at 10.0 - not at 40.0.
        assert normalised.clips[1].timeline_start == pytest.approx(10.0)
