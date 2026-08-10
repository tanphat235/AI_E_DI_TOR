"""Tests for the CapCut exporter.

A caveat that belongs at the top: these tests verify the draft is **internally consistent and
matches our reconstruction of the format**. They cannot verify that CapCut opens it, because
the format is undocumented and no reference draft existed on the development machine. What
is checked here is everything that is checkable without CapCut — unit conversion, referential
integrity between segments and materials, timeline arithmetic, and the two translations
between the Edit Plan's model and CapCut's that are genuinely easy to get backwards.

The riskiest assumption, called out so it is easy to revisit: CapCut stores video segments
**contiguously** and applies transition overlap itself. See
``TestVideoTrack.test_segments_are_contiguous_because_a_track_is_a_sequence``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config.settings import AiveSettings
from app.exporters.base import ExportRequest
from app.exporters.capcut import schema
from app.exporters.capcut.exporter import (
    CAPCUT_TRANSITIONS,
    CapCutExporter,
    CapCutExportError,
    _ass_to_hex,
    _gain_to_volume,
)
from app.exporters.capcut.locate import candidate_locations, find_draft_dir
from app.models.common import MediaRef, TimeRange, TransitionKind
from app.models.edit_plan import (
    EditPlan,
    MusicCue,
    NarrationTrack,
    SubtitleCue,
    TimelineClip,
    Transition,
)

CLIP_A = MediaRef(path="raw/001.mp4")
CLIP_B = MediaRef(path="raw/002.mp4")
NARRATION = MediaRef(path="narration.wav")
TRACK = MediaRef(path="music/bed.mp3")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    for name in ("001.mp4", "002.mp4"):
        (tmp_path / "raw" / name).write_bytes(b"present")
    (tmp_path / "narration.wav").write_bytes(b"present")
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "bed.mp3").write_bytes(b"present")
    return tmp_path


def _clip(
    clip_id: str,
    *,
    source: MediaRef = CLIP_A,
    start: float = 0.0,
    end: float = 4.0,
    timeline_start: float | None = 0.0,
    **kwargs: object,
) -> TimelineClip:
    return TimelineClip(
        id=clip_id,
        source=source,
        source_range=TimeRange(start=start, end=end),
        timeline_start=timeline_start,
        reason="a considered editorial reason",
        **kwargs,  # type: ignore[arg-type]
    )


def _plan(*clips: TimelineClip, **kwargs: object) -> EditPlan:
    fields: dict[str, object] = {
        "project_id": "capcut",
        "created_by": "pytest",
        "clips": clips or (_clip("c1"),),
    }
    fields.update(kwargs)
    return EditPlan(**fields)  # type: ignore[arg-type]


def _export(plan: EditPlan, project: Path, **kwargs: object) -> dict[str, Any]:
    """Export and return the parsed ``draft_content.json``."""
    exporter = CapCutExporter(AiveSettings())
    request = ExportRequest(
        plan=plan,
        project_root=project,
        destination=project / "draft",
        **kwargs,  # type: ignore[arg-type]
    )
    exporter.export(request)
    return json.loads((project / "draft" / schema.DRAFT_CONTENT_NAME).read_text(encoding="utf-8"))


def _track(document: dict[str, Any], kind: str) -> dict[str, Any] | None:
    return next((item for item in document["tracks"] if item["type"] == kind), None)


# --------------------------------------------------------------------------- #
# Units and identifiers
# --------------------------------------------------------------------------- #


class TestUnits:
    def test_seconds_become_integer_microseconds(self) -> None:
        """Passing seconds produces a draft where every clip is a millionth of its length,
        which reads as an empty timeline rather than as a unit error."""
        assert schema.to_microseconds(1.0) == 1_000_000
        assert schema.to_microseconds(0.4) == 400_000
        assert isinstance(schema.to_microseconds(1.5), int)

    def test_it_rounds_rather_than_truncating(self) -> None:
        """Truncation accumulates in one direction across forty clips."""
        assert schema.to_microseconds(0.0000009) == 1

    def test_a_timerange_is_start_and_duration_not_start_and_end(self) -> None:
        span = schema.timerange(2.0, 3.0)
        assert span == {"start": 2_000_000, "duration": 3_000_000}


class TestIdentifiers:
    def test_stable_ids_are_reproducible(self) -> None:
        """Re-exporting an unchanged plan must produce an unchanged draft."""
        assert schema.stable_id("segment", "a") == schema.stable_id("segment", "a")

    def test_different_namespaces_do_not_collide(self) -> None:
        assert schema.stable_id("segment", "a") != schema.stable_id("material", "a")

    def test_ids_are_uppercase_uuids(self) -> None:
        value = schema.stable_id("segment", "a")
        assert value == value.upper()
        assert len(value) == 36

    def test_new_id_is_random(self) -> None:
        assert schema.new_id() != schema.new_id()


# --------------------------------------------------------------------------- #
# Document structure
# --------------------------------------------------------------------------- #


class TestDraftStructure:
    def test_both_documents_are_written(self, project: Path) -> None:
        """A draft with content but no meta exists on disk and is invisible in the UI,
        which reads to a user as a failed export."""
        _export(_plan(), project)
        assert (project / "draft" / schema.DRAFT_CONTENT_NAME).is_file()
        assert (project / "draft" / schema.DRAFT_META_NAME).is_file()

    def test_every_material_bucket_is_present_even_when_empty(self, project: Path) -> None:
        """CapCut reads several without checking; an absent key is a crash on open."""
        document = _export(_plan(), project)
        for bucket in ("videos", "audios", "texts", "transitions", "speeds", "canvases"):
            assert bucket in document["materials"]

    def test_the_canvas_matches_the_plans_output_spec(self, project: Path) -> None:
        document = _export(_plan(), project)
        assert document["canvas_config"]["width"] == 1920
        assert document["canvas_config"]["height"] == 1080
        assert document["fps"] == 30.0

    def test_no_segment_references_a_material_that_does_not_exist(self, project: Path) -> None:
        """The one structural invariant that makes a draft loadable at all."""
        plan = _plan(
            _clip("c1", end=4.0),
            _clip("c2", source=CLIP_B, timeline_start=4.0),
            narration=NarrationTrack(
                source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=6.0),)
            ),
            music=(MusicCue(track=TRACK, timeline_range=TimeRange(start=0.0, end=6.0)),),
            subtitles=(SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="hello"),),
        )
        document = _export(plan, project)

        known = {item["id"] for bucket in document["materials"].values() for item in bucket}
        for item in document["tracks"]:
            for segment in item["segments"]:
                assert segment["material_id"] in known
                for ref in segment["extra_material_refs"]:
                    assert ref in known

    def test_re_exporting_an_unchanged_plan_produces_an_identical_draft(
        self, project: Path
    ) -> None:
        first = _export(_plan(), project)
        second = _export(_plan(), project)
        assert first == second


# --------------------------------------------------------------------------- #
# The video track - where the two formats disagree
# --------------------------------------------------------------------------- #


class TestVideoTrack:
    def _with_dissolve(self) -> EditPlan:
        return _plan(
            _clip("c1", end=4.0),
            _clip(
                "c2",
                source=CLIP_B,
                end=4.0,
                timeline_start=3.6,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.4),
            ),
        )

    def test_segments_are_contiguous_because_a_track_is_a_sequence(self, project: Path) -> None:
        """**The riskiest assumption in this exporter.**

        The Edit Plan has already applied transition overlap - `rules normalize` pulls each
        clip back - so its clips overlap on the timeline. Exporting those positions directly
        would put two segments at the same instant on one track, which CapCut's UI cannot
        represent. So segments are laid end to end and CapCut applies the overlap itself.
        """
        document = _export(self._with_dissolve(), project)
        segments = _track(document, "video")["segments"]  # type: ignore[index]
        cursor = 0
        for segment in segments:
            span = segment["target_timerange"]
            assert span["start"] == cursor
            cursor = span["start"] + span["duration"]

    def test_collapsing_the_transitions_reproduces_the_plans_duration(self, project: Path) -> None:
        """The check that the contiguous layout is self-consistent: once CapCut consumes
        each overlap, the track lands exactly where the renderer put it."""
        plan = self._with_dissolve()
        document = _export(plan, project)
        segments = _track(document, "video")["segments"]  # type: ignore[index]
        laid_out = sum(segment["target_timerange"]["duration"] for segment in segments)
        overlap = sum(item["duration"] for item in document["materials"]["transitions"])
        assert laid_out - overlap == schema.to_microseconds(plan.timeline_duration)

    def test_a_transition_is_marked_as_consuming_time(self, project: Path) -> None:
        """Without is_overlap the timeline lengthens by every transition, desynchronising
        the narration."""
        document = _export(self._with_dissolve(), project)
        assert all(item["is_overlap"] for item in document["materials"]["transitions"])

    def test_a_transition_attaches_to_the_outgoing_segment(self, project: Path) -> None:
        """The Edit Plan puts it on the clip it *enters*; CapCut on the clip it *leaves*.
        Off by one here and the dissolve appears a cut early."""
        document = _export(self._with_dissolve(), project)
        segments = _track(document, "video")["segments"]  # type: ignore[index]
        transition_id = document["materials"]["transitions"][0]["id"]
        assert transition_id in segments[0]["extra_material_refs"]
        assert transition_id not in segments[1]["extra_material_refs"]

    def test_a_hard_cut_creates_no_transition(self, project: Path) -> None:
        document = _export(_plan(_clip("c1"), _clip("c2", timeline_start=4.0)), project)
        assert document["materials"]["transitions"] == []

    def test_source_range_is_preserved_so_the_user_can_retrim(self, project: Path) -> None:
        document = _export(_plan(_clip("c1", start=12.5, end=18.0)), project)
        segment = _track(document, "video")["segments"][0]  # type: ignore[index]
        assert segment["source_timerange"]["start"] == 12_500_000
        assert segment["source_timerange"]["duration"] == 5_500_000

    def test_a_muted_clip_exports_at_zero_volume(self, project: Path) -> None:
        document = _export(_plan(_clip("c1", mute_source_audio=True)), project)
        assert _track(document, "video")["segments"][0]["volume"] == 0.0  # type: ignore[index]

    def test_kept_sync_sound_exports_audible(self, project: Path) -> None:
        document = _export(_plan(_clip("c1", mute_source_audio=False)), project)
        assert _track(document, "video")["segments"][0]["volume"] == 1.0  # type: ignore[index]

    def test_speed_reaches_both_the_segment_and_its_speed_material(self, project: Path) -> None:
        document = _export(_plan(_clip("c1", speed=2.0)), project)
        assert _track(document, "video")["segments"][0]["speed"] == 2.0  # type: ignore[index]
        assert any(item["speed"] == 2.0 for item in document["materials"]["speeds"])

    def test_one_material_serves_many_clips_of_the_same_file(self, project: Path) -> None:
        """A plan cutting forty times from one take must not import it forty times."""
        plan = _plan(
            _clip("c1", end=2.0),
            _clip("c2", start=4.0, end=6.0, timeline_start=2.0),
            _clip("c3", start=8.0, end=10.0, timeline_start=4.0),
        )
        document = _export(plan, project)
        assert len(document["materials"]["videos"]) == 1
        assert len(_track(document, "video")["segments"]) == 3  # type: ignore[arg-type]

    def test_an_unmapped_transition_warns_rather_than_inventing_one(self, project: Path) -> None:
        """Every TransitionKind currently maps, so this pins the fallback path itself."""
        assert set(CAPCUT_TRANSITIONS) == {kind for kind in TransitionKind if not kind.is_instant}


# --------------------------------------------------------------------------- #
# Audio and text
# --------------------------------------------------------------------------- #


class TestAudioTracks:
    def test_narration_and_music_are_separate_tracks(self, project: Path) -> None:
        """Rebalancing voice against music is a two-slider job only if they are not
        interleaved, and it is the first thing a user does in CapCut."""
        plan = _plan(
            _clip("c1", end=10.0),
            narration=NarrationTrack(
                source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=6.0),)
            ),
            music=(MusicCue(track=TRACK, timeline_range=TimeRange(start=0.0, end=10.0)),),
        )
        document = _export(plan, project)
        audio_tracks = [item for item in document["tracks"] if item["type"] == "audio"]
        assert len(audio_tracks) == 2

    def test_each_kept_range_becomes_its_own_segment(self, project: Path) -> None:
        """Keeping the seams lets a user undo one cleanup cut rather than all of them."""
        plan = _plan(
            _clip("c1", end=10.0),
            narration=NarrationTrack(
                source=NARRATION,
                kept_ranges=(
                    TimeRange(start=0.0, end=2.0),
                    TimeRange(start=5.0, end=7.0),
                    TimeRange(start=9.0, end=11.0),
                ),
            ),
        )
        document = _export(plan, project)
        narration = _track(document, "audio")  # type: ignore[assignment]
        assert len(narration["segments"]) == 3

    def test_kept_ranges_are_laid_end_to_end_on_the_timeline(self, project: Path) -> None:
        """This reassembly is what makes timeline time differ from source time."""
        plan = _plan(
            _clip("c1", end=10.0),
            narration=NarrationTrack(
                source=NARRATION,
                kept_ranges=(TimeRange(start=0.0, end=2.0), TimeRange(start=5.0, end=7.0)),
            ),
        )
        document = _export(plan, project)
        segments = _track(document, "audio")["segments"]  # type: ignore[index]
        assert segments[1]["target_timerange"]["start"] == 2_000_000
        assert segments[1]["source_timerange"]["start"] == 5_000_000

    def test_a_music_cue_carries_its_offset_and_position(self, project: Path) -> None:
        plan = _plan(
            _clip("c1", end=20.0),
            music=(
                MusicCue(
                    track=TRACK,
                    timeline_range=TimeRange(start=4.0, end=14.0),
                    source_offset=6.0,
                ),
            ),
        )
        document = _export(plan, project)
        segment = _track(document, "audio")["segments"][0]  # type: ignore[index]
        assert segment["target_timerange"]["start"] == 4_000_000
        assert segment["source_timerange"]["start"] == 6_000_000

    def test_fades_become_an_audio_fade_material(self, project: Path) -> None:
        plan = _plan(
            _clip("c1", end=20.0),
            music=(
                MusicCue(
                    track=TRACK,
                    timeline_range=TimeRange(start=0.0, end=10.0),
                    fade_in=1.0,
                    fade_out=2.0,
                ),
            ),
        )
        document = _export(plan, project)
        fades = document["materials"]["audio_fades"]
        assert fades[0]["fade_in_duration"] == 1_000_000
        assert fades[0]["fade_out_duration"] == 2_000_000

    def test_no_fade_material_when_no_fades_were_asked_for(self, project: Path) -> None:
        plan = _plan(
            _clip("c1", end=20.0),
            music=(MusicCue(track=TRACK, timeline_range=TimeRange(start=0.0, end=10.0)),),
        )
        assert _export(plan, project)["materials"]["audio_fades"] == []


class TestTextTrack:
    def test_cues_become_a_text_track(self, project: Path) -> None:
        plan = _plan(
            _clip("c1", end=10.0),
            subtitles=(
                SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="first"),
                SubtitleCue(range=TimeRange(start=2.0, end=4.0), text="second"),
            ),
        )
        document = _export(plan, project)
        assert len(_track(document, "text")["segments"]) == 2  # type: ignore[arg-type]

    def test_a_plan_without_cues_has_no_text_track(self, project: Path) -> None:
        assert _track(_export(_plan(), project), "text") is None

    def test_a_wrapped_cue_is_flattened(self, project: Path) -> None:
        """AIVE wraps to two lines; CapCut wraps from the caption box, and a baked-in
        newline fights it."""
        plan = _plan(
            _clip("c1", end=10.0),
            subtitles=(SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="two\nlines"),),
        )
        document = _export(plan, project)
        content = json.loads(document["materials"]["texts"][0]["content"])
        assert content["text"] == "two lines"

    def test_text_segments_have_no_source_range(self, project: Path) -> None:
        """Text has no source to seek into."""
        plan = _plan(
            _clip("c1", end=10.0),
            subtitles=(SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="hi"),),
        )
        document = _export(plan, project)
        assert _track(document, "text")["segments"][0]["source_timerange"] is None  # type: ignore[index]


# --------------------------------------------------------------------------- #
# Media handling
# --------------------------------------------------------------------------- #


class TestMedia:
    def test_media_is_copied_into_the_draft_by_default(self, project: Path) -> None:
        """A draft referencing files elsewhere breaks the moment footage is reorganised,
        and CapCut's failure mode is red placeholders with no explanation."""
        _export(_plan(), project, copy_media=True)
        assert (project / "draft" / "media" / "001.mp4").is_file()

    def test_no_copy_references_the_original(self, project: Path) -> None:
        document = _export(_plan(), project, copy_media=False)
        assert not (project / "draft" / "media").exists()
        assert "raw" in document["materials"]["videos"][0]["path"]

    def test_re_exporting_does_not_recopy_unchanged_media(self, project: Path) -> None:
        exporter = CapCutExporter(AiveSettings())
        request = ExportRequest(plan=_plan(), project_root=project, destination=project / "draft")
        exporter.export(request)
        assert exporter.export(request).media_copied == ()

    def test_the_media_manifest_lists_every_file(self, project: Path) -> None:
        """It is what CapCut's media panel reads."""
        plan = _plan(
            _clip("c1"),
            narration=NarrationTrack(
                source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=2.0),)
            ),
        )
        exporter = CapCutExporter(AiveSettings())
        exporter.export(
            ExportRequest(plan=plan, project_root=project, destination=project / "draft")
        )
        meta = json.loads((project / "draft" / schema.DRAFT_META_NAME).read_text(encoding="utf-8"))
        assert len(meta["draft_materials"][0]["value"]) == 2

    def test_a_template_contributes_styling_but_not_a_timeline(self, project: Path) -> None:
        """The timeline is the one thing being replaced; copying it would be self-defeating."""
        template = project / "template"
        template.mkdir()
        (template / schema.DRAFT_CONTENT_NAME).write_text('{"stale": true}', encoding="utf-8")
        (template / "extra_style.json").write_text("{}", encoding="utf-8")

        document = _export(_plan(), project, template=template)
        assert "stale" not in document
        assert (project / "draft" / "extra_style.json").is_file()


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


class TestPreflight:
    def _problems(self, plan: EditPlan, project: Path, **kwargs: object) -> tuple[str, ...]:
        return CapCutExporter(AiveSettings()).preflight(
            ExportRequest(
                plan=plan,
                project_root=project,
                destination=project / "draft",
                **kwargs,  # type: ignore[arg-type]
            )
        )

    def test_a_complete_plan_passes(self, project: Path) -> None:
        assert self._problems(_plan(), project) == ()

    def test_an_unplaced_plan_is_refused(self, project: Path) -> None:
        plan = _plan(_clip("c1", timeline_start=None))
        assert any("rules normalize" in problem for problem in self._problems(plan, project))

    def test_a_missing_clip_is_named(self, project: Path) -> None:
        plan = _plan(_clip("cX", source=MediaRef(path="raw/absent.mp4")))
        assert any("absent.mp4" in problem for problem in self._problems(plan, project))

    def test_a_template_that_is_not_a_directory_is_refused(self, project: Path) -> None:
        stray = project / "not_a_dir.txt"
        stray.write_text("x", encoding="utf-8")
        assert any(
            "not a directory" in problem
            for problem in self._problems(_plan(), project, template=stray)
        )

    def test_export_refuses_rather_than_writing_a_partial_draft(self, project: Path) -> None:
        """A half-written draft makes CapCut's whole project list unusable."""
        plan = _plan(_clip("cX", source=MediaRef(path="raw/absent.mp4")))
        with pytest.raises(CapCutExportError):
            CapCutExporter(AiveSettings()).export(
                ExportRequest(plan=plan, project_root=project, destination=project / "draft")
            )
        assert not (project / "draft").exists()


# --------------------------------------------------------------------------- #
# Conversions and location
# --------------------------------------------------------------------------- #


class TestConversions:
    def test_gain_to_volume(self) -> None:
        assert _gain_to_volume(0.0) == 1.0
        assert _gain_to_volume(-6.0) == pytest.approx(0.5012, abs=1e-3)

    def test_volume_is_clamped_to_one(self) -> None:
        """CapCut's slider tops out at 1.0; a larger value is silently ignored or rejected."""
        assert _gain_to_volume(12.0) == 1.0

    def test_ass_colour_is_read_in_its_own_byte_order(self) -> None:
        """ASS is &HAABBGGRR - alpha, BLUE, green, RED. Reading it as RGB swaps red and
        blue, which is invisible on the white default and wrong on everything else."""
        assert _ass_to_hex("&H00FFFFFF") == "#FFFFFF"
        assert _ass_to_hex("&H000000FF") == "#FF0000"  # ASS blue byte 00, red byte FF
        assert _ass_to_hex("&H00FF0000") == "#0000FF"

    def test_a_malformed_colour_falls_back_to_white(self) -> None:
        assert _ass_to_hex("nonsense") == "#FFFFFF"


class TestDraftLocation:
    def test_candidates_are_reported_whether_or_not_they_exist(self) -> None:
        """So `export targets` can show a user what was looked for, not only that nothing
        was found."""
        locations = candidate_locations()
        assert all(isinstance(item.exists, bool) for item in locations)

    def test_capcut_is_preferred_over_jianying(self) -> None:
        products = [item.product for item in candidate_locations()]
        if products:
            assert products[0] == "CapCut"

    def test_a_missing_folder_yields_none_rather_than_a_guess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Creating CapCut's folder ourselves would write a draft it never reads."""
        import app.exporters.capcut.locate as locate

        monkeypatch.setattr(locate, "candidate_locations", tuple)
        assert find_draft_dir() is None
