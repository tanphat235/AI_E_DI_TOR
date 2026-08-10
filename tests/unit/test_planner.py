"""Tests for the planning surface: ranking, coverage, the brief, and the baseline draft.

The recurring theme is **honesty about what is known**. The ranking cannot tell whether a
shot illustrates a line, so the tests assert that it says so rather than producing a
confident-looking number; the draft is not an edit, so the tests assert that its provenance
and its ``reason`` fields admit it.

All pure computation over models - no media, no decoding.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.config.settings import AiveSettings, PlannerSettings
from app.models.common import (
    CameraMove,
    MediaRef,
    MotionLevel,
    Severity,
    ShotType,
    TimeRange,
    TransitionKind,
)
from app.models.media import MediaProbe, VideoStreamInfo
from app.models.planning import CoverageReport
from app.models.speech import (
    NarrationAnalysis,
    NarrationBeat,
    SpeechCleanupReport,
    Transcript,
    TranscriptSegment,
)
from app.models.video import (
    ClipAnalysis,
    DuplicateGroup,
    FootageAnalysis,
    MotionStats,
    QualityScores,
    Scene,
    SceneTags,
)
from app.planner.brief import BriefBuilder
from app.planner.candidates import (
    CandidateRanker,
    duration_fit,
    keyword_overlap,
    variety_bonus,
)
from app.planner.draft import CREATED_BY, HeuristicDrafter

NARRATION = MediaRef(path="narration.wav")
CLIP_A = MediaRef(path="raw/001.mp4")
CLIP_B = MediaRef(path="raw/002.mp4")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _scene(
    index: int,
    *,
    clip: MediaRef = CLIP_A,
    start: float = 0.0,
    duration: float = 4.0,
    quality: float = 0.85,
    shot: ShotType = ShotType.MEDIUM,
    objects: tuple[str, ...] = (),
    setting: str | None = None,
) -> Scene:
    return Scene(
        clip=clip,
        index=index,
        range=TimeRange(start=start, end=start + duration),
        quality=QualityScores(
            blur=0.9, brightness=0.6, exposure=0.9, stability=0.9, overall=quality
        ),
        motion=MotionStats(
            level=MotionLevel.LOW, mean_magnitude=1.0, camera_move=CameraMove.STATIC
        ),
        tags=SceneTags(provider="test", objects=objects, setting=setting),
        shot_type=shot,
    )


def _beat(
    index: int,
    *,
    text: str = "Prepare the soil well.",
    start: float = 0.0,
    duration: float = 2.0,
    timeline_start: float | None = 0.0,
    keywords: tuple[str, ...] = ("prepare", "soil"),
) -> NarrationBeat:
    timeline = (
        None
        if timeline_start is None
        else TimeRange(start=timeline_start, end=timeline_start + duration)
    )
    return NarrationBeat(
        index=index,
        range=TimeRange(start=start, end=start + duration),
        text=text,
        keywords=keywords,
        timeline_range=timeline,
    )


def _footage(*scenes: Scene, duplicates: tuple = ()) -> FootageAnalysis:
    by_clip: dict[MediaRef, list[Scene]] = {}
    for scene in scenes:
        by_clip.setdefault(scene.clip, []).append(scene)
    return FootageAnalysis(
        clips=tuple(
            ClipAnalysis(
                clip=clip,
                probe=MediaProbe(
                    source=clip,
                    duration=60.0,
                    size_bytes=1024,
                    video=VideoStreamInfo(width=1920, height=1080, fps=30.0, codec="h264"),
                ),
                scenes=tuple(clip_scenes),
                analyzer_version="test/1",
                analyzed_at=datetime.now(UTC),
            )
            for clip, clip_scenes in by_clip.items()
        ),
        duplicates=duplicates,
    )


def _narration(*beats: NarrationBeat, kept: float = 10.0) -> NarrationAnalysis:
    return NarrationAnalysis(
        source=NARRATION,
        transcript=Transcript(
            source=NARRATION,
            language="en",
            duration=kept + 2.0,
            model_name="test/fake",
            segments=(
                TranscriptSegment(
                    index=0, range=TimeRange(start=0.0, end=kept), text="Prepare the soil."
                ),
            ),
        ),
        cleanup=SpeechCleanupReport(
            source=NARRATION,
            original_duration=kept + 2.0,
            kept_ranges=(TimeRange(start=0.0, end=kept),),
        ),
        beats=beats,
        analyzer_version="test/1",
        analyzed_at=datetime.now(UTC),
    )


def _permissive_settings(**planner: object) -> AiveSettings:
    """Settings whose quality floors admit the synthetic scenes used here."""
    return AiveSettings(
        rules={"min_overall_quality": 0.1, "min_blur_score": 0.0, "min_clip_duration": 0.5},
        planner=planner or {},
    )


# --------------------------------------------------------------------------- #
# Scoring components
# --------------------------------------------------------------------------- #


class TestKeywordOverlap:
    def test_matching_objects_score(self) -> None:
        score, matched = keyword_overlap(("soil", "spade"), _scene(0, objects=("soil", "hands")))
        assert matched == 1
        assert score == pytest.approx(0.5)

    def test_the_setting_counts_as_a_tag(self) -> None:
        score, matched = keyword_overlap(("garden",), _scene(0, setting="garden"))
        assert matched == 1
        assert score == 1.0

    def test_matching_is_case_insensitive(self) -> None:
        _score, matched = keyword_overlap(("Soil",), _scene(0, objects=("SOIL",)))
        assert matched == 1

    def test_a_scene_with_no_tags_scores_zero(self) -> None:
        """The classical CV provider names nothing, so this is every scene in practice.

        Zero means "no evidence of a match", not "no match" - which is why the ranker
        reports it in words rather than leaving the reader to infer it from a low score.
        """
        score, matched = keyword_overlap(("soil",), _scene(0))
        assert score == 0.0
        assert matched == 0

    def test_a_beat_with_no_keywords_scores_zero(self) -> None:
        assert keyword_overlap((), _scene(0, objects=("soil",))) == (0.0, 0)


class TestDurationFit:
    def test_an_exact_match_is_perfect(self) -> None:
        assert duration_fit(4.0, 4.0, tolerance=2.0) == 1.0

    def test_a_slightly_longer_scene_is_perfect(self) -> None:
        """Extra material is a choice of in and out points, not a problem."""
        assert duration_fit(5.5, 4.0, tolerance=2.0) == 1.0

    def test_a_much_longer_scene_is_only_gently_penalised(self) -> None:
        score = duration_fit(30.0, 4.0, tolerance=2.0)
        assert 0.5 <= score < 1.0

    def test_a_shorter_scene_is_penalised_in_proportion(self) -> None:
        """A shortfall is a hole in the edit, so this direction genuinely hurts."""
        assert duration_fit(2.0, 4.0, tolerance=2.0) == pytest.approx(0.5)

    def test_a_far_too_short_scene_scores_zero(self) -> None:
        assert duration_fit(0.1, 10.0, tolerance=2.0) < 0.05

    def test_a_beat_wanting_nothing_always_fits(self) -> None:
        assert duration_fit(3.0, 0.0, tolerance=2.0) == 1.0


class TestVarietyBonus:
    def test_a_different_shot_type_scores_full(self) -> None:
        assert variety_bonus(ShotType.CLOSE_UP, ShotType.WIDE) == 1.0

    def test_repeating_the_previous_shot_scores_zero(self) -> None:
        """Three consecutive wides read as laziness even when each is individually best."""
        assert variety_bonus(ShotType.WIDE, ShotType.WIDE) == 0.0

    def test_no_previous_shot_is_neutral(self) -> None:
        assert variety_bonus(ShotType.WIDE, None) == 0.5

    def test_unknown_is_neutral_not_good(self) -> None:
        """Most scenes are unknown with the classical provider; rewarding that is noise."""
        assert variety_bonus(ShotType.UNKNOWN, ShotType.WIDE) == 0.5
        assert variety_bonus(ShotType.WIDE, ShotType.UNKNOWN) == 0.5


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


class TestCandidateRanker:
    @pytest.fixture
    def ranker(self) -> CandidateRanker:
        return CandidateRanker(PlannerSettings())

    def test_a_shortlist_is_returned_per_beat(self, ranker: CandidateRanker) -> None:
        scenes = tuple(_scene(index, start=index * 5.0) for index in range(20))
        results = ranker.rank_all((_beat(0),), scenes)
        assert len(results) == 1
        assert len(results[0].candidates) == PlannerSettings().candidates_per_beat

    def test_the_shortlist_size_is_configurable(self) -> None:
        ranker = CandidateRanker(PlannerSettings(candidates_per_beat=2))
        scenes = tuple(_scene(index, start=index * 5.0) for index in range(10))
        assert len(ranker.rank_all((_beat(0),), scenes)[0].candidates) == 2

    def test_candidates_are_ordered_by_score(self, ranker: CandidateRanker) -> None:
        scenes = (
            _scene(0, quality=0.3),
            _scene(1, start=10.0, quality=0.95),
            _scene(2, start=20.0, quality=0.6),
        )
        candidates = ranker.rank_all((_beat(0),), scenes)[0].candidates
        scores = [candidate.score for candidate in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_higher_quality_wins_all_else_equal(self, ranker: CandidateRanker) -> None:
        scenes = (_scene(0, quality=0.4), _scene(1, start=10.0, quality=0.95))
        best = ranker.rank_all((_beat(0),), scenes)[0].best
        assert best is not None
        assert best.scene_key == "001#1"

    def test_a_cut_beat_gets_no_candidates(self, ranker: CandidateRanker) -> None:
        """It needs no picture, so offering options would be noise."""
        results = ranker.rank_all((_beat(0, timeline_start=None),), (_scene(0),))
        assert results[0].candidates == ()
        assert results[0].needs_footage is False

    def test_every_candidate_explains_itself(self, ranker: CandidateRanker) -> None:
        """A number with no explanation invites blind trust or blanket dismissal."""
        candidates = ranker.rank_all((_beat(0),), (_scene(0),))[0].candidates
        assert candidates[0].reasons

    def test_the_missing_semantic_tags_are_stated_not_implied(
        self, ranker: CandidateRanker
    ) -> None:
        """A reader who does not know the provider has no tags would misread a low score."""
        candidates = ranker.rank_all((_beat(0),), (_scene(0),))[0].candidates
        assert any("no semantic tags" in reason for reason in candidates[0].reasons)

    def test_keyword_matches_are_named_in_the_reasons(self, ranker: CandidateRanker) -> None:
        scenes = (_scene(0, objects=("soil",)),)
        candidates = ranker.rank_all((_beat(0),), scenes)[0].candidates
        assert any("keyword" in reason for reason in candidates[0].reasons)

    def test_a_shot_offered_earlier_is_demoted(self, ranker: CandidateRanker) -> None:
        """Reusing a shot the audience just saw is the clearest sign of an automated edit."""
        only = (_scene(0),)
        results = ranker.rank_all((_beat(0), _beat(1, timeline_start=2.0)), only)
        first = results[0].candidates[0]
        second = results[1].candidates[0]
        assert second.score < first.score
        assert any("already offered" in reason for reason in second.reasons)

    def test_the_reuse_penalty_is_configurable(self) -> None:
        strict = CandidateRanker(PlannerSettings(reuse_penalty=1.0))
        results = strict.rank_all((_beat(0), _beat(1, timeline_start=2.0)), (_scene(0),))
        assert results[1].candidates[0].score == 0.0

    def test_variety_prefers_a_different_framing(self) -> None:
        """With variety the only differentiator, the changed shot type must win."""
        ranker = CandidateRanker(
            PlannerSettings(
                keyword_weight=0.0, quality_weight=0.0, duration_weight=0.0, variety_weight=1.0
            )
        )
        scenes = (
            _scene(0, shot=ShotType.WIDE),
            _scene(1, start=10.0, shot=ShotType.CLOSE_UP),
        )
        results = ranker.rank_all((_beat(0), _beat(1, timeline_start=2.0)), scenes)
        # The first beat picks one; the second must prefer the other framing.
        assert results[0].best is not None
        assert results[1].best is not None
        assert results[0].best.shot_type is not results[1].best.shot_type

    def test_a_scene_too_short_for_the_beat_ranks_below_one_that_fits(
        self, ranker: CandidateRanker
    ) -> None:
        scenes = (
            _scene(0, duration=0.6, quality=0.9),
            _scene(1, start=10.0, duration=5.0, quality=0.9),
        )
        best = ranker.rank_all((_beat(0, duration=4.0),), scenes)[0].best
        assert best is not None
        assert best.scene_key == "001#1"

    def test_candidates_carry_the_facts_needed_to_judge_them(self, ranker: CandidateRanker) -> None:
        """The brief must stand alone; no cross-referencing footage.json."""
        candidate = ranker.rank_all((_beat(0),), (_scene(0),))[0].candidates[0]
        assert candidate.clip == CLIP_A
        assert candidate.quality > 0.0
        assert candidate.duration > 0.0
        assert candidate.shot_type is ShotType.MEDIUM

    def test_no_scenes_yields_no_candidates(self, ranker: CandidateRanker) -> None:
        assert ranker.rank_all((_beat(0),), ())[0].candidates == ()

    def test_scores_stay_within_range(self, ranker: CandidateRanker) -> None:
        scenes = tuple(_scene(index, start=index * 5.0) for index in range(8))
        for beat in ranker.rank_all((_beat(0), _beat(1, timeline_start=2.0)), scenes):
            for candidate in beat.candidates:
                assert 0.0 <= candidate.score <= 1.0


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


class TestCoverageReport:
    def _report(self, **kwargs: object) -> CoverageReport:
        defaults: dict[str, object] = {
            "narration_duration": 10.0,
            "eligible_footage_duration": 30.0,
            "beats_total": 5,
            "beats_needing_footage": 4,
            "beats_with_candidates": 4,
            "eligible_scenes": 6,
            "rejected_scenes": 1,
        }
        return CoverageReport(**{**defaults, **kwargs})  # type: ignore[arg-type]

    def test_footage_ratio(self) -> None:
        assert self._report().footage_ratio == pytest.approx(3.0)

    def test_a_covered_project_is_feasible(self) -> None:
        assert self._report().feasible is True

    def test_an_uncovered_beat_makes_it_infeasible(self) -> None:
        assert self._report(beats_with_candidates=3).feasible is False

    def test_no_beats_is_infeasible(self) -> None:
        assert self._report(beats_needing_footage=0, beats_with_candidates=0).feasible is False

    def test_uncovered_beats_are_counted(self) -> None:
        assert self._report(beats_with_candidates=1).uncovered_beats == 3

    def test_zero_narration_does_not_divide_by_zero(self) -> None:
        assert self._report(narration_duration=0.0).footage_ratio == 0.0


class TestBriefBuilder:
    def _build(self, settings: AiveSettings | None = None, **kwargs: object) -> object:
        resolved = settings or _permissive_settings()
        return BriefBuilder(resolved).build(**kwargs)  # type: ignore[arg-type]

    def test_the_brief_gathers_beats_candidates_and_constraints(self) -> None:
        brief = self._build(
            project_id="demo",
            narration=_narration(_beat(0), _beat(1, timeline_start=2.0)),
            footage=_footage(_scene(0), _scene(1, start=10.0)),
        )
        assert brief.project_id == "demo"  # type: ignore[attr-defined]
        assert len(brief.beats) == 2  # type: ignore[attr-defined]
        assert brief.constraints.min_clip_duration > 0.0  # type: ignore[attr-defined]
        assert brief.coverage.eligible_scenes == 2  # type: ignore[attr-defined]

    def test_only_eligible_scenes_become_candidates(self) -> None:
        """Offering a shot the Rule Engine would reject wastes the director's attention."""
        settings = AiveSettings(rules={"min_overall_quality": 0.8, "min_blur_score": 0.0})
        brief = self._build(
            settings,
            project_id="demo",
            narration=_narration(_beat(0)),
            footage=_footage(_scene(0, quality=0.9), _scene(1, start=10.0, quality=0.2)),
        )
        offered = {
            candidate.scene_key
            for beat in brief.beats  # type: ignore[attr-defined]
            for candidate in beat.candidates
        }
        assert offered == {"001#0"}

    def test_a_duplicate_scene_is_never_offered(self) -> None:
        brief = self._build(
            project_id="demo",
            narration=_narration(_beat(0)),
            footage=_footage(
                _scene(0),
                _scene(1, start=10.0),
                duplicates=(
                    DuplicateGroup(representative="001#0", duplicates=("001#1",), similarity=0.99),
                ),
            ),
        )
        offered = {
            candidate.scene_key
            for beat in brief.beats  # type: ignore[attr-defined]
            for candidate in beat.candidates
        }
        assert "001#1" not in offered

    def test_insufficient_footage_is_an_error(self) -> None:
        brief = self._build(
            project_id="demo",
            narration=_narration(_beat(0, duration=30.0), kept=30.0),
            footage=_footage(_scene(0, duration=2.0)),
        )
        codes = {issue.code for issue in brief.coverage.issues}  # type: ignore[attr-defined]
        assert "coverage.insufficient_footage" in codes

    def test_tight_footage_is_a_warning(self) -> None:
        brief = self._build(
            project_id="demo",
            narration=_narration(_beat(0, duration=10.0), kept=10.0),
            footage=_footage(_scene(0, duration=13.0)),
        )
        issues = {issue.code: issue for issue in brief.coverage.issues}  # type: ignore[attr-defined]
        assert "coverage.tight_footage" in issues
        assert issues["coverage.tight_footage"].severity is Severity.WARNING

    def test_no_usable_footage_is_an_error(self) -> None:
        settings = AiveSettings(rules={"min_overall_quality": 0.99})
        brief = self._build(
            settings,
            project_id="demo",
            narration=_narration(_beat(0)),
            footage=_footage(_scene(0, quality=0.5)),
        )
        codes = {issue.code for issue in brief.coverage.issues}  # type: ignore[attr-defined]
        assert "coverage.no_usable_footage" in codes
        assert brief.coverage.feasible is False  # type: ignore[attr-defined]

    def test_no_beats_is_an_error(self) -> None:
        brief = self._build(project_id="demo", narration=_narration(), footage=_footage(_scene(0)))
        codes = {issue.code for issue in brief.coverage.issues}  # type: ignore[attr-defined]
        assert "coverage.no_beats" in codes

    def test_eligible_scenes_matching_no_beat_are_listed(self) -> None:
        """Often the most interesting B-roll, so it is reported rather than dropped."""
        settings = _permissive_settings(candidates_per_beat=1)
        brief = self._build(
            settings,
            project_id="demo",
            narration=_narration(_beat(0)),
            footage=_footage(*(_scene(index, start=index * 6.0) for index in range(4))),
        )
        assert len(brief.unused_scenes) == 3  # type: ignore[attr-defined]

    def test_the_brief_round_trips_through_json(self) -> None:
        from app.models.planning import PlanningBrief

        brief = self._build(
            project_id="demo",
            narration=_narration(_beat(0)),
            footage=_footage(_scene(0)),
        )
        restored = PlanningBrief.model_validate_json(brief.model_dump_json())  # type: ignore[attr-defined]
        assert restored == brief


# --------------------------------------------------------------------------- #
# The heuristic baseline
# --------------------------------------------------------------------------- #


class TestHeuristicDrafter:
    def _brief_and_narration(
        self, *, settings: AiveSettings | None = None
    ) -> tuple[object, NarrationAnalysis]:
        resolved = settings or _permissive_settings()
        narration = _narration(
            _beat(0, duration=2.0, timeline_start=0.0),
            _beat(1, start=3.0, duration=2.0, timeline_start=2.5),
            kept=5.0,
        )
        footage = _footage(
            _scene(0, duration=6.0, shot=ShotType.WIDE),
            _scene(1, start=10.0, duration=6.0, shot=ShotType.CLOSE_UP),
        )
        brief = BriefBuilder(resolved).build(
            project_id="demo", narration=narration, footage=footage
        )
        return brief, narration

    def test_one_clip_per_beat_that_needs_footage(self) -> None:
        settings = _permissive_settings()
        brief, narration = self._brief_and_narration(settings=settings)
        plan = HeuristicDrafter(settings).draft(brief, narration=narration)  # type: ignore[arg-type]
        assert len(plan.clips) == 2

    def test_the_provenance_is_honest(self) -> None:
        """A reviewer must be able to see the reason fields are mechanical."""
        settings = _permissive_settings()
        brief, narration = self._brief_and_narration(settings=settings)
        plan = HeuristicDrafter(settings).draft(brief, narration=narration)  # type: ignore[arg-type]
        assert plan.created_by == CREATED_BY == "heuristic"
        assert "not an edit" in (plan.notes or "")
        assert all("revise" in clip.reason.lower() for clip in plan.clips)

    def test_clips_carry_provenance_back_to_the_analysis(self) -> None:
        settings = _permissive_settings()
        brief, narration = self._brief_and_narration(settings=settings)
        plan = HeuristicDrafter(settings).draft(brief, narration=narration)  # type: ignore[arg-type]
        for clip in plan.clips:
            assert clip.scene_key is not None
            assert clip.beat_index is not None

    def test_the_plan_is_left_unplaced_for_the_rule_engine(self) -> None:
        """Duplicating the placement arithmetic would be a second implementation to maintain."""
        settings = _permissive_settings()
        brief, narration = self._brief_and_narration(settings=settings)
        plan = HeuristicDrafter(settings).draft(brief, narration=narration)  # type: ignore[arg-type]
        assert not plan.is_placed

    def test_clips_are_stretched_to_cover_the_gaps_between_beats(self) -> None:
        """Beats are separated by pauses that survived cleanup.

        Sizing clips to the beats alone leaves black between every line - which is what
        the first version of this drafter did.
        """
        settings = _permissive_settings()
        brief, narration = self._brief_and_narration(settings=settings)
        plan = HeuristicDrafter(settings).draft(brief, narration=narration)  # type: ignore[arg-type]
        # Beat 0 runs 0.0-2.0 but the next begins at 2.5, so the clip must span 2.5s.
        assert plan.clips[0].timeline_duration >= 2.5

    def test_the_picture_covers_the_narration(self) -> None:
        settings = _permissive_settings()
        brief, narration = self._brief_and_narration(settings=settings)
        plan = HeuristicDrafter(settings).draft(brief, narration=narration)  # type: ignore[arg-type]
        assert plan.timeline_duration >= narration.cleanup.kept_duration - 0.5

    def test_the_narration_track_is_attached(self) -> None:
        settings = _permissive_settings()
        brief, narration = self._brief_and_narration(settings=settings)
        plan = HeuristicDrafter(settings).draft(brief, narration=narration)  # type: ignore[arg-type]
        assert plan.narration is not None
        assert plan.narration.source == NARRATION

    def test_the_source_range_is_taken_from_the_middle_of_a_scene(self) -> None:
        """The start of a shot is where the camera is still settling."""
        settings = _permissive_settings()
        brief, narration = self._brief_and_narration(settings=settings)
        plan = HeuristicDrafter(settings).draft(brief, narration=narration)  # type: ignore[arg-type]
        clip = plan.clips[0]
        assert clip.source_range.start > 0.0

    def test_the_first_clip_has_no_incoming_transition(self) -> None:
        settings = _permissive_settings()
        brief, narration = self._brief_and_narration(settings=settings)
        plan = HeuristicDrafter(settings).draft(brief, narration=narration)  # type: ignore[arg-type]
        assert plan.clips[0].transition_in is None

    def test_the_configured_transition_is_applied_between_clips(self) -> None:
        settings = _permissive_settings()
        brief, narration = self._brief_and_narration(settings=settings)
        plan = HeuristicDrafter(settings).draft(brief, narration=narration)  # type: ignore[arg-type]
        transition = plan.clips[1].transition_in
        assert transition is not None
        assert transition.kind is settings.rules.default_transition

    def test_a_cut_default_means_no_transition_object(self) -> None:
        settings = AiveSettings(
            rules={
                "min_overall_quality": 0.1,
                "min_blur_score": 0.0,
                "min_clip_duration": 0.5,
                "default_transition": TransitionKind.CUT,
            }
        )
        brief, narration = self._brief_and_narration(settings=settings)
        plan = HeuristicDrafter(settings).draft(brief, narration=narration)  # type: ignore[arg-type]
        assert plan.clips[1].transition_in is None

    def test_nothing_to_draft_raises_a_clear_error(self) -> None:
        settings = AiveSettings(rules={"min_overall_quality": 0.99})
        narration = _narration(_beat(0))
        footage = _footage(_scene(0, quality=0.2))
        brief = BriefBuilder(settings).build(
            project_id="demo", narration=narration, footage=footage
        )
        with pytest.raises(ValueError, match="no beat has a usable candidate"):
            HeuristicDrafter(settings).draft(brief, narration=narration)

    def test_the_draft_survives_the_rule_engine(self) -> None:
        """The point of a baseline: a structurally valid plan to revise."""
        from app.rule_engine.context import RuleContext
        from app.rule_engine.engine import DefaultRuleEngine

        settings = _permissive_settings()
        brief, narration = self._brief_and_narration(settings=settings)
        plan = HeuristicDrafter(settings).draft(brief, narration=narration)  # type: ignore[arg-type]

        context = RuleContext(
            rules=settings.rules,
            subtitle=settings.subtitle,
            probes={
                CLIP_A: MediaProbe(source=CLIP_A, duration=60.0, size_bytes=1),
                NARRATION: MediaProbe(source=NARRATION, duration=7.0, size_bytes=1),
            },
        )
        normalised, report = DefaultRuleEngine().normalize(plan, context)
        assert normalised.is_placed
        assert report.ok, [issue.message for issue in report.errors]
