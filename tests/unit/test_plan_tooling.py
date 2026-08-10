"""Tests for Edit Plan tooling: loading, migration, review, diff and cue attachment.

Four things are under test, and each closes a specific gap Phase 6 found:

* **Cue attachment.** ``EditPlan.subtitles`` existed since Phase 1 and both the renderer and
  the exporter read it, but nothing ever wrote to it.
* **Migration.** Phase 1's docstring promised "add a migration when you do" without any
  mechanism to add one to.
* **Review.** A plan could only be read as raw JSON, despite the docs promising the ``reason``
  fields are what a user reads.
* **Diff.** No way to see what a second pass, or ``rules normalize``, actually changed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

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
    EDIT_PLAN_SCHEMA_VERSION,
    EditPlan,
    NarrationTrack,
    OutputSpec,
    SubtitleCue,
    TimelineClip,
    Transition,
)
from app.models.media import MediaProbe, VideoStreamInfo
from app.models.migrations import (
    PlanMigrationError,
    migrate,
    needs_migration,
    registered_versions,
)
from app.models.plan_review import PlanDiff, PlanReview
from app.models.speech import Transcript, TranscriptSegment, Word
from app.models.video import (
    ClipAnalysis,
    DuplicateGroup,
    FootageAnalysis,
    MotionStats,
    QualityScores,
    Scene,
    SceneTags,
)
from app.plan.diff import diff_plans
from app.plan.loader import PlanLoadError, load_plan
from app.plan.review import build_review, longest_run
from app.subtitles.attach import attach_subtitles, cues_past_end

NARRATION = MediaRef(path="narration.wav")
CLIP_A = MediaRef(path="raw/001.mp4")
CLIP_B = MediaRef(path="raw/002.mp4")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _clip(
    clip_id: str,
    *,
    source: MediaRef = CLIP_A,
    start: float = 0.0,
    end: float = 4.0,
    reason: str = "a properly considered editorial reason",
    **kwargs: object,
) -> TimelineClip:
    return TimelineClip(
        id=clip_id,
        source=source,
        source_range=TimeRange(start=start, end=end),
        reason=reason,
        **kwargs,  # type: ignore[arg-type]
    )


def _plan(*clips: TimelineClip, **kwargs: object) -> EditPlan:
    fields: dict[str, object] = {
        "project_id": "test",
        "created_by": "pytest",
        "clips": clips or (_clip("c1"),),
    }
    fields.update(kwargs)
    return EditPlan(**fields)  # type: ignore[arg-type]


def _scene(clip: MediaRef, index: int, *, shot: ShotType = ShotType.MEDIUM) -> Scene:
    return Scene(
        clip=clip,
        index=index,
        range=TimeRange(start=index * 5.0, end=index * 5.0 + 4.0),
        quality=QualityScores(blur=0.9, brightness=0.6, exposure=0.9, stability=0.9, overall=0.85),
        motion=MotionStats(
            level=MotionLevel.LOW, mean_magnitude=1.0, camera_move=CameraMove.STATIC
        ),
        tags=SceneTags(provider="test"),
        shot_type=shot,
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


def _transcript(*, words: bool = True) -> Transcript:
    timings = (
        (
            Word(text="First,", start=0.0, end=0.4),
            Word(text="prepare", start=0.5, end=1.0),
            Word(text="the", start=1.05, end=1.2),
            Word(text="soil.", start=1.25, end=1.8),
        )
        if words
        else ()
    )
    return Transcript(
        source=NARRATION,
        language="en",
        duration=10.0,
        model_name="test/fake",
        segments=(
            TranscriptSegment(
                index=0,
                range=TimeRange(start=0.0, end=1.8),
                text="First, prepare the soil.",
                words=timings,
            ),
            TranscriptSegment(
                index=1,
                range=TimeRange(start=6.0, end=7.5),
                text="Then water it well.",
                words=(
                    (
                        Word(text="Then", start=6.0, end=6.3),
                        Word(text="water", start=6.35, end=6.8),
                        Word(text="it", start=6.85, end=7.0),
                        Word(text="well.", start=7.05, end=7.5),
                    )
                    if words
                    else ()
                ),
            ),
        ),
    )


def _codes(issues: object) -> set[str]:
    return {issue.code for issue in issues}  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Cue attachment - the contract gap
# --------------------------------------------------------------------------- #


class TestAttachSubtitles:
    def test_cues_land_on_the_plan(self) -> None:
        """The gap: nothing populated this field, and both consumers read it."""
        plan = _plan(_clip("c1", end=10.0))
        updated, count = attach_subtitles(plan, _transcript(), settings=SubtitleSettings())
        assert count == 2
        assert len(updated.subtitles) == 2
        assert plan.subtitles == ()  # the original is untouched

    def test_the_plans_own_kept_ranges_define_the_clock(self) -> None:
        """The plan is authoritative: a director may have trimmed further than cleanup did."""
        plan = _plan(
            _clip("c1", end=10.0),
            narration=NarrationTrack(
                source=NARRATION,
                # 2.0-6.0 removed, so the second segment shifts back by 4s.
                kept_ranges=(TimeRange(start=0.0, end=2.0), TimeRange(start=6.0, end=10.0)),
            ),
        )
        updated, _count = attach_subtitles(plan, _transcript(), settings=SubtitleSettings())
        assert updated.subtitles[1].range.start == pytest.approx(2.0)

    def test_without_a_narration_track_source_time_is_used(self) -> None:
        """Correct for a music-only montage, and a reasonable fallback otherwise."""
        plan = _plan(_clip("c1", end=10.0))
        updated, _count = attach_subtitles(plan, _transcript(), settings=SubtitleSettings())
        assert updated.subtitles[1].range.start == pytest.approx(6.0)

    def test_existing_cues_are_replaced_not_merged(self) -> None:
        """Two sets of cues for the same words would both render."""
        plan = _plan(
            _clip("c1", end=10.0),
            subtitles=(SubtitleCue(range=TimeRange(start=0.0, end=1.0), text="stale"),),
        )
        updated, _count = attach_subtitles(plan, _transcript(), settings=SubtitleSettings())
        assert "stale" not in {cue.text for cue in updated.subtitles}

    def test_karaoke_timings_follow_the_setting(self) -> None:
        plan = _plan(_clip("c1", end=10.0))
        without, _ = attach_subtitles(plan, _transcript(), settings=SubtitleSettings())
        with_karaoke, _ = attach_subtitles(
            plan, _transcript(), settings=SubtitleSettings(karaoke=True)
        )
        assert without.subtitles[0].words == ()
        assert with_karaoke.subtitles[0].words

    def test_word_timings_can_be_forced(self) -> None:
        plan = _plan(_clip("c1", end=10.0))
        updated, _count = attach_subtitles(
            plan, _transcript(), settings=SubtitleSettings(), include_word_timings=True
        )
        assert updated.subtitles[0].words

    def test_stranded_cues_are_detectable(self) -> None:
        """The symptom of cues on the wrong clock: they fall off the end."""
        plan = _plan(_clip("c1", end=2.0))  # picture is only 2s
        updated, _count = attach_subtitles(plan, _transcript(), settings=SubtitleSettings())
        assert cues_past_end(updated)

    def test_a_covered_plan_strands_nothing(self) -> None:
        plan = _plan(_clip("c1", end=20.0))
        updated, _count = attach_subtitles(plan, _transcript(), settings=SubtitleSettings())
        assert cues_past_end(updated) == ()


# --------------------------------------------------------------------------- #
# Migration - the promised mechanism
# --------------------------------------------------------------------------- #


class TestMigrations:
    def test_a_current_document_needs_no_migration(self) -> None:
        assert needs_migration({"schema_version": EDIT_PLAN_SCHEMA_VERSION}) is False

    def test_a_document_with_no_version_is_treated_as_current(self) -> None:
        """The field has a model default so a hand-authored plan need not carry it."""
        assert needs_migration({"project_id": "x"}) is False

    def test_an_older_document_needs_migration(self) -> None:
        assert needs_migration({"schema_version": "0.9"}) is True

    def test_migrating_a_current_document_is_a_no_op(self) -> None:
        document = {"schema_version": EDIT_PLAN_SCHEMA_VERSION, "project_id": "x"}
        result, chain = migrate(document)
        assert result == document
        assert chain == (EDIT_PLAN_SCHEMA_VERSION,)

    def test_an_unknown_version_is_refused_with_an_actionable_message(self) -> None:
        """A half-understood plan renders a confidently wrong video."""
        with pytest.raises(PlanMigrationError, match="does not know"):
            migrate({"schema_version": "99.0"})

    def test_a_registered_chain_is_followed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exercises the mechanism with a temporary migration, since none exist yet."""
        import app.models.migrations as migrations

        monkeypatch.setattr(migrations, "_MIGRATIONS", {})

        @migrations.register("0.8", "0.9")
        def _first(document: dict) -> dict:
            document["added_by_first"] = True
            return document

        @migrations.register("0.9", EDIT_PLAN_SCHEMA_VERSION)
        def _second(document: dict) -> dict:
            document["added_by_second"] = True
            return document

        result, chain = migrations.migrate({"schema_version": "0.8", "project_id": "x"})
        assert result["added_by_first"] is True
        assert result["added_by_second"] is True
        assert result["schema_version"] == EDIT_PLAN_SCHEMA_VERSION
        assert chain == ("0.8", "0.9", EDIT_PLAN_SCHEMA_VERSION)

    def test_a_duplicate_registration_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app.models.migrations as migrations

        monkeypatch.setattr(migrations, "_MIGRATIONS", {})
        migrations.register("0.8", "0.9")(lambda document: document)
        with pytest.raises(ValueError, match="already registered"):
            migrations.register("0.8", "1.0")(lambda document: document)

    def test_migration_does_not_mutate_its_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A caller holding the original document must still see the original."""
        import app.models.migrations as migrations

        monkeypatch.setattr(migrations, "_MIGRATIONS", {})
        migrations.register("0.9", EDIT_PLAN_SCHEMA_VERSION)(
            lambda document: {**document, "upgraded": True}
        )
        original = {"schema_version": "0.9", "project_id": "x"}
        migrations.migrate(original)
        assert "upgraded" not in original

    def test_no_migrations_are_registered_yet(self) -> None:
        """One schema version exists, so there is nothing to migrate. Documents the state."""
        assert registered_versions() == ()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


class TestLoadPlan:
    def test_a_valid_plan_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.json"
        path.write_text(_plan().model_dump_json(), encoding="utf-8")
        loaded = load_plan(path)
        assert loaded.plan.project_id == "test"
        assert loaded.was_migrated is False

    def test_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(PlanLoadError, match="no Edit Plan at"):
            load_plan(tmp_path / "absent.json")

    def test_malformed_json_says_so(self, tmp_path: Path) -> None:
        """ "Not valid JSON" and "missing a field" call for different responses."""
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(PlanLoadError, match="not valid JSON"):
            load_plan(path)

    def test_a_json_array_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "array.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(PlanLoadError, match="must contain a JSON object"):
            load_plan(path)

    def test_an_invalid_plan_reports_validation_separately(self, tmp_path: Path) -> None:
        path = tmp_path / "invalid.json"
        path.write_text('{"project_id": "x"}', encoding="utf-8")
        with pytest.raises(PlanLoadError, match="not a valid Edit Plan"):
            load_plan(path)

    def test_a_future_version_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "future.json"
        document = json.loads(_plan().model_dump_json())
        document["schema_version"] = "99.0"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(PlanLoadError, match="does not know"):
            load_plan(path)

    def test_an_unreadable_file_is_reported_as_such(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A permissions failure is not a malformed plan, and the message must not say it is."""
        path = tmp_path / "locked.json"
        path.write_text("{}", encoding="utf-8")

        def _deny(*_args: object, **_kwargs: object) -> str:
            raise PermissionError("access denied")

        monkeypatch.setattr(Path, "read_text", _deny)
        with pytest.raises(PlanLoadError, match="could not read"):
            load_plan(path)

    def test_a_migrated_plan_reports_where_it_came_from(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.models.migrations as migrations

        monkeypatch.setattr(migrations, "_MIGRATIONS", {})
        migrations.register("0.9", EDIT_PLAN_SCHEMA_VERSION)(lambda document: document)

        document = json.loads(_plan().model_dump_json())
        document["schema_version"] = "0.9"
        path = tmp_path / "old.json"
        path.write_text(json.dumps(document), encoding="utf-8")

        loaded = load_plan(path)
        assert loaded.was_migrated is True
        assert loaded.migrated_from == "0.9"


# --------------------------------------------------------------------------- #
# Review
# --------------------------------------------------------------------------- #


class TestStatistics:
    def test_basic_measurements(self) -> None:
        plan = _plan(
            _clip("c1", end=4.0), _clip("c2", start=5.0, end=7.0), _clip("c3", start=8.0, end=14.0)
        )
        stats = build_review(plan).statistics
        assert stats.clip_count == 3
        assert stats.total_duration == pytest.approx(12.0)
        assert stats.shortest_clip == pytest.approx(2.0)
        assert stats.longest_clip == pytest.approx(6.0)
        assert stats.median_clip_duration == pytest.approx(4.0)

    def test_pacing_is_per_minute_so_projects_compare(self) -> None:
        plan = _plan(*[_clip(f"c{i}", start=i * 5.0, end=i * 5.0 + 2.0) for i in range(6)])
        stats = build_review(plan).statistics
        # 6 clips over 12s is 30 cuts/min.
        assert stats.cuts_per_minute == pytest.approx(30.0)

    def test_transitions_are_counted_by_kind(self) -> None:
        plan = _plan(
            _clip("c1", end=6.0),
            _clip(
                "c2",
                start=8.0,
                end=14.0,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.4),
            ),
        )
        counts = build_review(plan).statistics.transition_counts
        assert counts["cut"] == 1
        assert counts["dissolve"] == 1

    def test_distinct_sources_and_scenes(self) -> None:
        plan = _plan(
            _clip("c1", source=CLIP_A, scene_key="001#0"),
            _clip("c2", source=CLIP_A, start=5.0, end=9.0, scene_key="001#1"),
            _clip("c3", source=CLIP_B, scene_key="002#0"),
        )
        stats = build_review(plan).statistics
        assert stats.distinct_sources == 2
        assert stats.distinct_scenes == 3

    def test_shot_types_come_from_the_footage_analysis(self) -> None:
        """The plan is decoupled from the analysis, so the mapping is rebuilt."""
        plan = _plan(_clip("c1", scene_key="001#0"))
        footage = _footage(_scene(CLIP_A, 0, shot=ShotType.WIDE))
        stats = build_review(plan, footage=footage).statistics
        assert stats.shot_type_counts.get("wide") == 1

    def test_coverage_delta(self) -> None:
        plan = _plan(
            _clip("c1", end=4.0),
            narration=NarrationTrack(
                source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=10.0),)
            ),
        )
        stats = build_review(plan).statistics
        assert stats.coverage_delta == pytest.approx(-6.0)

    def test_no_narration_means_no_delta(self) -> None:
        assert build_review(_plan()).statistics.coverage_delta is None

    def test_subtitles_are_counted(self) -> None:
        plan = _plan(
            _clip("c1", end=10.0),
            subtitles=(SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="hi"),),
        )
        assert build_review(plan).statistics.subtitle_count == 1


class TestLongestRun:
    def test_an_empty_sequence(self) -> None:
        assert longest_run([]) == 0

    def test_all_the_same(self) -> None:
        assert longest_run(["a", "a", "a"]) == 3

    def test_the_longest_of_several_runs(self) -> None:
        assert longest_run(["a", "b", "b", "b", "a"]) == 3

    def test_no_repeats(self) -> None:
        assert longest_run(["a", "b", "c"]) == 1


class TestEditorialNotes:
    def test_a_mechanical_rhythm_is_flagged(self) -> None:
        """The signature of durations computed rather than chosen."""
        plan = _plan(*[_clip(f"c{i}", start=i * 5.0, end=i * 5.0 + 3.0) for i in range(6)])
        assert "review.mechanical_rhythm" in _codes(build_review(plan).notes)

    def test_varied_lengths_are_not_flagged(self) -> None:
        plan = _plan(
            _clip("c1", end=1.5),
            _clip("c2", start=5.0, end=11.0),
            _clip("c3", start=15.0, end=18.0),
            _clip("c4", start=20.0, end=27.0),
        )
        assert "review.mechanical_rhythm" not in _codes(build_review(plan).notes)

    def test_too_few_clips_to_judge_rhythm(self) -> None:
        plan = _plan(_clip("c1", end=3.0), _clip("c2", start=5.0, end=8.0))
        assert "review.mechanical_rhythm" not in _codes(build_review(plan).notes)

    def test_repeated_framing_is_flagged(self) -> None:
        """Three consecutive wides read as laziness even if each was the best choice."""
        plan = _plan(
            *[
                _clip(f"c{i}", start=i * 5.0, end=i * 5.0 + 4.0, scene_key=f"001#{i}")
                for i in range(3)
            ]
        )
        footage = _footage(*[_scene(CLIP_A, i, shot=ShotType.WIDE) for i in range(3)])
        assert "review.repeated_framing" in _codes(build_review(plan, footage=footage).notes)

    def test_varied_framing_is_not_flagged(self) -> None:
        plan = _plan(
            *[
                _clip(f"c{i}", start=i * 5.0, end=i * 5.0 + 4.0, scene_key=f"001#{i}")
                for i in range(3)
            ]
        )
        footage = _footage(
            _scene(CLIP_A, 0, shot=ShotType.WIDE),
            _scene(CLIP_A, 1, shot=ShotType.CLOSE_UP),
            _scene(CLIP_A, 2, shot=ShotType.MEDIUM),
        )
        assert "review.repeated_framing" not in _codes(build_review(plan, footage=footage).notes)

    def test_unknown_shot_types_do_not_count_as_a_run(self) -> None:
        """With the classical provider most scenes are unknown; a run of those says nothing."""
        plan = _plan(*[_clip(f"c{i}", start=i * 5.0, end=i * 5.0 + 4.0) for i in range(5)])
        assert "review.repeated_framing" not in _codes(build_review(plan).notes)

    def test_a_dissolve_on_every_join_is_flagged(self) -> None:
        """A dissolve should mean time passed; everywhere it means slideshow."""
        dissolve = Transition(kind=TransitionKind.DISSOLVE, duration=0.4)
        plan = _plan(
            *[
                _clip(f"c{i}", start=i * 10.0, end=i * 10.0 + 8.0, transition_in=dissolve)
                for i in range(4)
            ]
        )
        assert "review.every_cut_is_a_transition" in _codes(build_review(plan).notes)

    def test_all_hard_cuts_is_informational_not_a_warning(self) -> None:
        plan = _plan(*[_clip(f"c{i}", start=i * 10.0, end=i * 10.0 + 5.0) for i in range(4)])
        notes = {issue.code: issue for issue in build_review(plan).notes}
        assert "review.no_transitions" in notes
        assert notes["review.no_transitions"].severity is Severity.INFO

    def test_a_mix_of_cuts_and_dissolves_draws_no_comment(self) -> None:
        """Which is the point: the notes fire on monotony, not on transitions existing."""
        dissolve = Transition(kind=TransitionKind.DISSOLVE, duration=0.4)
        plan = _plan(
            _clip("c1", end=6.0),
            _clip("c2", start=8.0, end=13.0, transition_in=dissolve),
            _clip("c3", start=15.0, end=22.0),
            _clip("c4", start=25.0, end=29.0, transition_in=dissolve),
        )
        codes = _codes(build_review(plan).notes)
        assert "review.every_cut_is_a_transition" not in codes
        assert "review.no_transitions" not in codes

    def test_a_picture_that_matches_the_narration_draws_no_coverage_comment(self) -> None:
        plan = _plan(
            _clip("c1", end=10.0),
            narration=NarrationTrack(
                source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=10.0),)
            ),
        )
        codes = _codes(build_review(plan).notes)
        assert "review.narration_uncovered" not in codes
        assert "review.long_tail" not in codes

    def test_footage_usage_needs_provenance_on_the_plan(self) -> None:
        """Without scene_key there is no way to know which scenes were used."""
        footage = _footage(*[_scene(CLIP_A, index) for index in range(10)])
        plan = _plan(_clip("c1"))
        assert "review.footage_underused" not in _codes(build_review(plan, footage=footage).notes)

    def test_underused_footage_is_flagged(self) -> None:
        plan = _plan(_clip("c1", scene_key="001#0"))
        footage = _footage(*[_scene(CLIP_A, index) for index in range(10)])
        assert "review.footage_underused" in _codes(build_review(plan, footage=footage).notes)

    def test_footage_usage_needs_the_analysis(self) -> None:
        plan = _plan(_clip("c1", scene_key="001#0"))
        assert "review.footage_underused" not in _codes(build_review(plan).notes)

    def test_suppressed_duplicates_do_not_count_as_available(self) -> None:
        """Using 1 of 2 usable scenes is not "ignoring the footage" just because 8 are dupes."""
        plan = _plan(_clip("c1", scene_key="001#0"))
        footage = _footage(
            *[_scene(CLIP_A, index) for index in range(10)],
            duplicates=(
                DuplicateGroup(
                    representative="001#0",
                    duplicates=tuple(f"001#{index}" for index in range(2, 10)),
                    similarity=0.99,
                ),
            ),
        )
        assert "review.footage_underused" not in _codes(build_review(plan, footage=footage).notes)

    def test_a_heuristic_plan_is_flagged_loudly(self) -> None:
        plan = _plan(_clip("c1"), created_by="heuristic")
        notes = {issue.code: issue for issue in build_review(plan).notes}
        assert "review.heuristic_plan" in notes
        assert notes["review.heuristic_plan"].severity is Severity.WARNING

    def test_thin_reasons_are_flagged(self) -> None:
        """The reason is what a reviewer reads; one word is not a reason."""
        plan = _plan(_clip("c1", reason="good"))
        assert "review.thin_reasons" in _codes(build_review(plan).notes)

    def test_proper_reasons_are_not_flagged(self) -> None:
        plan = _plan(_clip("c1", reason="Wide shot establishes the garden before the detail."))
        assert "review.thin_reasons" not in _codes(build_review(plan).notes)

    def test_missing_provenance_is_flagged(self) -> None:
        plan = _plan(_clip("c1"), _clip("c2", start=5.0, end=9.0))
        assert "review.no_provenance" in _codes(build_review(plan).notes)

    def test_partial_provenance_is_not_flagged(self) -> None:
        plan = _plan(_clip("c1", scene_key="001#0"), _clip("c2", start=5.0, end=9.0))
        assert "review.no_provenance" not in _codes(build_review(plan).notes)

    def test_uncovered_narration_is_flagged(self) -> None:
        plan = _plan(
            _clip("c1", end=4.0),
            narration=NarrationTrack(
                source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=20.0),)
            ),
        )
        assert "review.narration_uncovered" in _codes(build_review(plan).notes)

    def test_a_long_tail_is_informational(self) -> None:
        plan = _plan(
            _clip("c1", end=20.0),
            narration=NarrationTrack(
                source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=5.0),)
            ),
        )
        notes = {issue.code: issue for issue in build_review(plan).notes}
        assert "review.long_tail" in notes
        assert notes["review.long_tail"].severity is Severity.INFO

    def test_notes_are_never_errors(self) -> None:
        """Admissibility belongs to the Rule Engine; these are observations."""
        plan = _plan(_clip("c1", reason="x"), created_by="heuristic")
        for issue in build_review(plan).notes:
            assert issue.severity is not Severity.ERROR


class TestReviewRendering:
    def test_the_timeline_carries_the_authors_reasons(self) -> None:
        plan = _plan(_clip("c1", reason="Establishes the garden before any detail."))
        review = build_review(plan)
        assert review.timeline[0].reason == "Establishes the garden before any detail."

    def test_an_unplaced_plan_says_so(self) -> None:
        review = build_review(_plan(_clip("c1")))
        assert review.is_placed is False
        assert review.timeline[0].timeline_range is None

    def test_a_placed_plan_shows_timeline_positions(self) -> None:
        review = build_review(_plan(_clip("c1", end=4.0, timeline_start=0.0)))
        assert review.is_placed is True
        assert review.timeline[0].timeline_range == TimeRange(start=0.0, end=4.0)

    def test_the_authors_own_notes_are_carried_through(self) -> None:
        plan = _plan(_clip("c1"), notes="Chronological build.")
        assert build_review(plan).plan_notes == "Chronological build."

    def test_the_review_round_trips_through_json(self) -> None:
        review = build_review(_plan(_clip("c1")))
        assert PlanReview.model_validate_json(review.model_dump_json()) == review


# --------------------------------------------------------------------------- #
# Diff
# --------------------------------------------------------------------------- #


class TestDiff:
    def test_identical_plans(self) -> None:
        plan = _plan(_clip("c1"), _clip("c2", start=5.0, end=9.0))
        result = diff_plans(plan, plan)
        assert result.is_identical

    def test_an_added_clip(self) -> None:
        before = _plan(_clip("c1"))
        after = _plan(_clip("c1"), _clip("c2", start=5.0, end=9.0))
        result = diff_plans(before, after)
        assert result.added == ("c2",)
        assert result.removed == ()

    def test_a_removed_clip(self) -> None:
        before = _plan(_clip("c1"), _clip("c2", start=5.0, end=9.0))
        after = _plan(_clip("c1"))
        assert diff_plans(before, after).removed == ("c2",)

    def test_inserting_a_clip_does_not_report_the_rest_as_changed(self) -> None:
        """Positional diffing would bury the one real change in a wall of false ones."""
        before = _plan(_clip("c1"), _clip("c2", start=5.0, end=9.0))
        after = _plan(
            _clip("c1"), _clip("cNEW", start=20.0, end=24.0), _clip("c2", start=5.0, end=9.0)
        )
        result = diff_plans(before, after)
        assert result.added == ("cNEW",)
        assert result.changed == ()

    def test_a_changed_source_range_is_reported_with_both_values(self) -> None:
        before = _plan(_clip("c1", start=0.0, end=4.0))
        after = _plan(_clip("c1", start=2.0, end=6.0))
        change = diff_plans(before, after).changed[0]
        assert change.clip_id == "c1"
        assert "source_range" in change.fields
        assert "0.00-4.00" in change.before
        assert "2.00-6.00" in change.after

    def test_a_changed_reason_is_reported(self) -> None:
        before = _plan(_clip("c1", reason="the original considered reason"))
        after = _plan(_clip("c1", reason="a revised and better reason"))
        assert "reason" in diff_plans(before, after).changed[0].fields

    def test_a_changed_transition_is_reported(self) -> None:
        before = _plan(_clip("c1"))
        after = _plan(
            _clip("c1", transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.4))
        )
        change = diff_plans(before, after).changed[0]
        assert "transition_in" in change.fields
        assert "dissolve" in change.after

    def test_swapping_the_source_file_names_both(self) -> None:
        before = _plan(_clip("c1", source=CLIP_A))
        after = _plan(_clip("c1", source=CLIP_B))
        change = diff_plans(before, after).changed[0]
        assert change.fields == ("source",)
        assert "001.mp4" in change.before
        assert "002.mp4" in change.after

    def test_placing_a_previously_unplaced_clip_reads_as_unplaced_to_a_position(self) -> None:
        """The common case after `rules normalize`, and None needs its own rendering."""
        before = _plan(_clip("c1", end=4.0))
        after = _plan(_clip("c1", end=4.0, timeline_start=2.5))
        change = diff_plans(before, after).changed[0]
        assert change.before == "tl=unplaced"
        assert change.after == "tl=2.500"

    def test_a_field_without_special_formatting_still_renders(self) -> None:
        """speed, scene_key, beat_index and framing all fall through to the default."""
        before = _plan(_clip("c1", speed=1.0, scene_key="001#0"))
        after = _plan(_clip("c1", speed=0.5, scene_key="001#4"))
        change = diff_plans(before, after).changed[0]
        assert "speed=1.0" in change.before
        assert "speed=0.5" in change.after
        assert "scene_key=001#4" in change.after

    def test_a_long_reason_is_truncated_rather_than_flooding_the_digest(self) -> None:
        long_reason = "Holds on the hands in the soil while the line lands, instead of cutting away"
        after = _plan(_clip("c1", reason=long_reason))
        change = diff_plans(_plan(_clip("c1")), after).changed[0]
        assert change.after.endswith('…"')
        assert len(change.after) < len(long_reason)

    def test_confidence_alone_is_not_a_change(self) -> None:
        """It is the author's estimate, not a property of the edit."""
        before = _plan(_clip("c1", confidence=0.5))
        after = _plan(_clip("c1", confidence=0.9))
        assert diff_plans(before, after).is_identical

    def test_a_sub_millisecond_timing_difference_is_not_a_change(self) -> None:
        """Re-normalising can shift a float in its last bit."""
        before = _plan(_clip("c1", end=4.0, timeline_start=1.0))
        after = _plan(_clip("c1", end=4.0, timeline_start=1.0000001))
        assert diff_plans(before, after).is_identical

    def test_a_real_placement_change_is_reported(self) -> None:
        before = _plan(_clip("c1", end=4.0, timeline_start=1.0))
        after = _plan(_clip("c1", end=4.0, timeline_start=3.0))
        assert "timeline_start" in diff_plans(before, after).changed[0].fields

    def test_reordering_is_reported_once(self) -> None:
        before = _plan(_clip("c1"), _clip("c2", start=5.0, end=9.0))
        after = _plan(_clip("c2", start=5.0, end=9.0), _clip("c1"))
        result = diff_plans(before, after)
        assert result.reordered is True
        assert result.changed == ()

    def test_an_insertion_is_not_reported_as_a_reordering(self) -> None:
        before = _plan(_clip("c1"), _clip("c2", start=5.0, end=9.0))
        after = _plan(
            _clip("c1"), _clip("cX", start=20.0, end=24.0), _clip("c2", start=5.0, end=9.0)
        )
        assert diff_plans(before, after).reordered is False

    def test_plan_level_changes_are_listed(self) -> None:
        before = _plan(_clip("c1"), notes="first")
        after = _plan(_clip("c1"), notes="second", output=OutputSpec(fps=24.0))
        fields = diff_plans(before, after).summary_fields
        assert "notes" in fields
        assert "output" in fields

    def test_attaching_subtitles_shows_as_one_plan_level_change(self) -> None:
        """The realistic case: `plan subtitles` must not appear to touch every clip."""
        before = _plan(_clip("c1", end=10.0))
        after, _count = attach_subtitles(before, _transcript(), settings=SubtitleSettings())
        result = diff_plans(before, after)
        assert result.summary_fields == ("subtitles",)
        assert result.changed == ()

    def test_created_at_alone_is_not_a_change(self) -> None:
        """It changes on every write; reporting it would train readers to ignore the list."""
        before = _plan(_clip("c1"), created_at=datetime(2026, 1, 1, tzinfo=UTC))
        after = _plan(_clip("c1"), created_at=datetime(2026, 6, 1, tzinfo=UTC))
        assert diff_plans(before, after).is_identical

    def test_duration_change_is_reported(self) -> None:
        before = _plan(_clip("c1", end=4.0))
        after = _plan(_clip("c1", end=8.0))
        result = diff_plans(before, after)
        assert result.duration_delta == pytest.approx(4.0)

    def test_the_diff_round_trips_through_json(self) -> None:
        result = diff_plans(_plan(_clip("c1")), _plan(_clip("c1", start=1.0, end=5.0)))
        assert PlanDiff.model_validate_json(result.model_dump_json()) == result
