"""Tests for the renderer: progress parsing, preflight, and command assembly.

None of these run FFmpeg. Progress parsing is a pure function over lines; preflight is
filesystem checks; the command is a list built from settings. What *is* exercised against
the real binary lives in the Phase 7 verification notes, not here — a unit suite that shells
out to an encoder is a unit suite that is skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.config.settings import AiveSettings
from app.models.common import MediaRef, SubtitleFormat, TimeRange
from app.models.edit_plan import (
    EditPlan,
    MusicCue,
    NarrationTrack,
    SubtitleCue,
    TimelineClip,
)
from app.renderer.base import RenderRequest
from app.renderer.ffmpeg.progress import ProgressUpdate, parse_progress
from app.renderer.ffmpeg.renderer import FFmpegRenderer, RenderError
from app.services.ffmpeg_locator import FFmpegLocator

CLIP = MediaRef(path="raw/001.mp4")
NARRATION = MediaRef(path="narration.wav")


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #


def _block(**fields: object) -> list[str]:
    lines = [f"{key}={value}" for key, value in fields.items()]
    lines.append("progress=continue")
    return lines


class TestParseProgress:
    def test_one_block_yields_one_update(self) -> None:
        updates = list(
            parse_progress(_block(frame=120, fps=30.0, out_time_us=4000000, speed="2.0x"))
        )
        assert len(updates) == 1
        assert updates[0].frame == 120
        assert updates[0].out_time == pytest.approx(4.0)
        assert updates[0].speed == pytest.approx(2.0)

    def test_the_terminating_block_is_marked_finished(self) -> None:
        lines = [*_block(out_time_us=1000000), "out_time_us=2000000", "progress=end"]
        updates = list(parse_progress(lines))
        assert updates[-1].finished is True

    def test_out_time_ms_is_read_as_microseconds(self) -> None:
        """FFmpeg's field is misnamed and has been for years. Reading it as milliseconds
        makes a render appear to finish a thousand times over."""
        updates = list(parse_progress(["out_time_ms=5000000", "progress=continue"]))
        assert updates[0].out_time == pytest.approx(5.0)

    def test_a_timecode_is_parsed_when_no_integer_field_is_present(self) -> None:
        updates = list(parse_progress(["out_time=00:01:30.500000", "progress=continue"]))
        assert updates[0].out_time == pytest.approx(90.5)

    def test_a_malformed_timecode_yields_zero_rather_than_raising(self) -> None:
        updates = list(parse_progress(["out_time=nonsense", "progress=continue"]))
        assert updates[0].out_time == 0.0

    def test_unknown_keys_are_ignored(self) -> None:
        """FFmpeg adds fields between versions; a render must not fail over a new column."""
        updates = list(
            parse_progress(["some_new_field=7", "out_time_us=1000000", "progress=continue"])
        )
        assert updates[0].out_time == pytest.approx(1.0)

    def test_speed_na_is_treated_as_unknown(self) -> None:
        updates = list(parse_progress(["speed=N/A", "out_time_us=0", "progress=continue"]))
        assert updates[0].speed == 0.0

    def test_blank_and_keyless_lines_are_skipped(self) -> None:
        updates = list(
            parse_progress(["", "   ", "garbage", "out_time_us=1000000", "progress=continue"])
        )
        assert len(updates) == 1

    def test_an_incomplete_trailing_block_is_not_emitted(self) -> None:
        """Half a block is not a progress report; emitting it would show a backwards jump."""
        assert list(parse_progress(["frame=10", "fps=25"])) == []


class TestProgressFraction:
    def test_it_is_a_fraction_of_the_expected_duration(self) -> None:
        update = ProgressUpdate(out_time=5.0, frame=0, fps=0.0, speed=1.0, finished=False)
        assert update.fraction(10.0) == pytest.approx(0.5)

    def test_it_never_reaches_one_before_the_process_exits(self) -> None:
        """out_time can overshoot on the final flush; a bar at 100% while the encoder is
        still working is worse than one sitting at 99%."""
        update = ProgressUpdate(out_time=99.0, frame=0, fps=0.0, speed=1.0, finished=False)
        assert update.fraction(10.0) < 1.0

    def test_finishing_reports_exactly_one(self) -> None:
        update = ProgressUpdate(out_time=0.0, frame=0, fps=0.0, speed=1.0, finished=True)
        assert update.fraction(10.0) == 1.0

    def test_an_unknown_total_reports_zero_rather_than_dividing(self) -> None:
        update = ProgressUpdate(out_time=5.0, frame=0, fps=0.0, speed=1.0, finished=False)
        assert update.fraction(0.0) == 0.0

    def test_eta_uses_ffmpegs_own_speed(self) -> None:
        """Derived from encoding rate, not elapsed wall time, so a slow start does not
        poison the estimate for the whole run."""
        update = ProgressUpdate(out_time=10.0, frame=0, fps=0.0, speed=2.0, finished=False)
        assert update.eta(30.0) == pytest.approx(10.0)

    def test_eta_is_unknown_until_a_speed_is_reported(self) -> None:
        update = ProgressUpdate(out_time=1.0, frame=0, fps=0.0, speed=0.0, finished=False)
        assert update.eta(30.0) is None

    def test_a_finished_render_has_no_eta(self) -> None:
        update = ProgressUpdate(out_time=30.0, frame=0, fps=0.0, speed=2.0, finished=True)
        assert update.eta(30.0) is None


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project whose media actually exists on disk."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "001.mp4").write_bytes(b"not really video, but present")
    (tmp_path / "narration.wav").write_bytes(b"present")
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "bed.mp3").write_bytes(b"present")
    return tmp_path


def _renderer() -> FFmpegRenderer:
    settings = AiveSettings()
    return FFmpegRenderer(settings, FFmpegLocator(settings.media))


def _plan(*clips: TimelineClip, **kwargs: object) -> EditPlan:
    fields: dict[str, object] = {
        "project_id": "render",
        "created_by": "pytest",
        "clips": clips
        or (
            TimelineClip(
                id="c1",
                source=CLIP,
                source_range=TimeRange(start=0.0, end=4.0),
                timeline_start=0.0,
                reason="a considered editorial reason",
            ),
        ),
    }
    fields.update(kwargs)
    return EditPlan(**fields)  # type: ignore[arg-type]


def _request(plan: EditPlan, root: Path, **kwargs: object) -> RenderRequest:
    return RenderRequest(
        plan=plan,
        project_root=root,
        destination=root / "output" / "final.mp4",
        **kwargs,  # type: ignore[arg-type]
    )


def _problems(plan: EditPlan, root: Path, **kwargs: object) -> tuple[str, ...]:
    return _renderer().preflight(_request(plan, root, **kwargs))


class TestPreflight:
    def test_a_complete_plan_passes(self, project: Path) -> None:
        assert _problems(_plan(), project) == ()

    def test_an_unplaced_plan_is_refused_with_the_command_that_fixes_it(
        self, project: Path
    ) -> None:
        plan = _plan(
            TimelineClip(
                id="c1",
                source=CLIP,
                source_range=TimeRange(start=0.0, end=4.0),
                reason="a considered editorial reason",
            )
        )
        problems = _problems(plan, project)
        assert any("rules normalize" in problem for problem in problems)

    def test_a_missing_clip_is_named(self, project: Path) -> None:
        plan = _plan(
            TimelineClip(
                id="cX",
                source=MediaRef(path="raw/absent.mp4"),
                source_range=TimeRange(start=0.0, end=4.0),
                timeline_start=0.0,
                reason="a considered editorial reason",
            )
        )
        problems = _problems(plan, project)
        assert any("cX" in problem and "absent.mp4" in problem for problem in problems)

    def test_a_missing_file_used_by_many_clips_is_reported_once(self, project: Path) -> None:
        """Forty clips off one missing take should not produce forty lines."""
        missing = MediaRef(path="raw/gone.mp4")
        clips = tuple(
            TimelineClip(
                id=f"c{index}",
                source=missing,
                source_range=TimeRange(start=0.0, end=2.0),
                timeline_start=index * 2.0,
                reason="a considered editorial reason",
            )
            for index in range(5)
        )
        problems = [p for p in _problems(_plan(*clips), project) if "gone.mp4" in p]
        assert len(problems) == 1

    def test_a_missing_narration_is_reported(self, project: Path) -> None:
        plan = _plan(
            narration=NarrationTrack(
                source=MediaRef(path="absent.wav"),
                kept_ranges=(TimeRange(start=0.0, end=2.0),),
            )
        )
        assert any("absent.wav" in problem for problem in _problems(plan, project))

    def test_a_missing_music_track_is_reported_with_its_cue_index(self, project: Path) -> None:
        plan = _plan(
            music=(
                MusicCue(
                    track=MediaRef(path="music/absent.mp3"),
                    timeline_range=TimeRange(start=0.0, end=4.0),
                ),
            )
        )
        assert any("music cue 0" in problem for problem in _problems(plan, project))

    def test_burn_in_without_cues_is_refused(self, project: Path) -> None:
        """Silently rendering no subtitles when asked to burn them in is the worst outcome."""
        problems = _problems(_plan(), project, burn_in_subtitles=True)
        assert any("plan subtitles" in problem for problem in problems)

    def test_burn_in_with_cues_passes(self, project: Path) -> None:
        plan = _plan(subtitles=(SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="hello"),))
        assert _problems(plan, project, burn_in_subtitles=True) == ()

    def test_a_destination_that_is_a_directory_is_refused(self, project: Path) -> None:
        destination = project / "output" / "final.mp4"
        destination.mkdir(parents=True)
        request = RenderRequest(plan=_plan(), project_root=project, destination=destination)
        assert any("is a directory" in problem for problem in _renderer().preflight(request))

    def test_preflight_creates_the_output_directory(self, project: Path) -> None:
        """So a first render does not fail on a folder it was always going to make."""
        _problems(_plan(), project)
        assert (project / "output").is_dir()

    def test_preflight_leaves_no_probe_file_behind(self, project: Path) -> None:
        _problems(_plan(), project)
        assert not any(
            p.name.startswith(".aive-write-test") for p in (project / "output").iterdir()
        )

    def test_render_refuses_rather_than_encoding_a_broken_request(self, project: Path) -> None:
        plan = _plan(
            TimelineClip(
                id="c1",
                source=MediaRef(path="raw/absent.mp4"),
                source_range=TimeRange(start=0.0, end=4.0),
                timeline_start=0.0,
                reason="a considered editorial reason",
            )
        )
        with pytest.raises(RenderError, match=re.escape("absent.mp4")):
            _renderer().render(_request(plan, project))


# --------------------------------------------------------------------------- #
# Command
# --------------------------------------------------------------------------- #


class TestCommand:
    def _command(self, plan: EditPlan, root: Path, *, draft: bool = False) -> list[str]:
        renderer = _renderer()
        settings = AiveSettings()
        from app.renderer.ffmpeg.graph import FilterGraphBuilder

        graph = FilterGraphBuilder(settings, draft=draft).build(
            plan, project_root=root, subtitle_file=None
        )
        return renderer._command("ffmpeg", graph, _request(plan, root, draft=draft))

    def test_progress_is_routed_to_ffmpegs_stdout(self, project: Path) -> None:
        """Leaving its stderr a clean log - the same discipline AIVE applies to itself."""
        command = self._command(_plan(), project)
        assert command[command.index("-progress") + 1] == "pipe:1"

    def test_a_draft_uses_the_draft_encoder_settings(self, project: Path) -> None:
        command = self._command(_plan(), project, draft=True)
        assert command[command.index("-crf") + 1] == "30"
        assert command[command.index("-preset") + 1] == "veryfast"

    def test_a_master_uses_the_delivery_settings(self, project: Path) -> None:
        command = self._command(_plan(), project)
        assert command[command.index("-crf") + 1] == "18"
        assert command[command.index("-preset") + 1] == "medium"

    def test_a_silent_plan_disables_audio_rather_than_encoding_silence(self, project: Path) -> None:
        command = self._command(_plan(), project)
        assert "-an" in command
        assert "-c:a" not in command

    def test_a_plan_with_narration_encodes_audio(self, project: Path) -> None:
        plan = _plan(
            narration=NarrationTrack(source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=3.0),))
        )
        command = self._command(plan, project)
        assert "-an" not in command
        assert command[command.index("-c:a") + 1] == "aac"

    def test_the_output_is_bounded_by_the_computed_duration(self, project: Path) -> None:
        """Cheap insurance against a filter's duration accounting disagreeing with ours."""
        command = self._command(_plan(), project)
        assert command[command.index("-t") + 1].startswith("4.0")

    def test_faststart_is_requested(self, project: Path) -> None:
        """Without it the result must be fully downloaded before it will scrub."""
        command = self._command(_plan(), project)
        assert command[command.index("-movflags") + 1] == "+faststart"

    def test_the_filter_graph_is_passed_as_one_argument(self, project: Path) -> None:
        command = self._command(_plan(), project)
        assert command.count("-filter_complex") == 1

    def test_the_destination_is_last(self, project: Path) -> None:
        command = self._command(_plan(), project)
        assert command[-1].endswith("final.mp4")


class TestSubtitleSidecars:
    def test_requested_formats_are_written_beside_the_video(self, project: Path) -> None:
        plan = _plan(subtitles=(SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="hello"),))
        request = _request(plan, project, subtitle_formats=(SubtitleFormat.SRT, SubtitleFormat.ASS))
        (project / "output").mkdir(exist_ok=True)
        written = _renderer()._write_subtitles(request)
        assert {path.suffix for path in written} == {".srt", ".ass"}
        assert all(path.is_file() for path in written)

    def test_a_plan_without_cues_writes_nothing(self, project: Path) -> None:
        request = _request(_plan(), project, subtitle_formats=(SubtitleFormat.SRT,))
        assert _renderer()._write_subtitles(request) == ()

    def test_burn_in_reuses_an_ass_sidecar_rather_than_writing_a_second(
        self, project: Path
    ) -> None:
        plan = _plan(subtitles=(SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="hello"),))
        request = _request(
            plan, project, subtitle_formats=(SubtitleFormat.ASS,), burn_in_subtitles=True
        )
        (project / "output").mkdir(exist_ok=True)
        written = _renderer()._write_subtitles(request)
        assert _renderer()._subtitle_to_burn(request, written) == written[0]

    def test_burn_in_writes_an_ass_when_only_srt_was_asked_for(self, project: Path) -> None:
        """Burning in is a styling operation, and SRT can express no styling."""
        plan = _plan(subtitles=(SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="hello"),))
        request = _request(
            plan, project, subtitle_formats=(SubtitleFormat.SRT,), burn_in_subtitles=True
        )
        (project / "output").mkdir(exist_ok=True)
        written = _renderer()._write_subtitles(request)
        burned = _renderer()._subtitle_to_burn(request, written)
        assert burned is not None
        assert burned.suffix == ".ass"
        assert burned.is_file()

    def test_nothing_is_burned_in_unless_asked(self, project: Path) -> None:
        plan = _plan(subtitles=(SubtitleCue(range=TimeRange(start=0.0, end=2.0), text="hello"),))
        request = _request(plan, project)
        assert _renderer()._subtitle_to_burn(request, ()) is None
