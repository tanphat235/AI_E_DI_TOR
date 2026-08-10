"""Tests for the FFmpeg filter graph.

The graph builder is pure, which is the whole reason it was written that way: a filter graph
is otherwise verifiable only by rendering video and looking at it. Everything here runs
without FFmpeg on the machine.

Two facts below were established by measuring the real binary rather than by reading docs,
and the tests pin them so a refactor cannot quietly undo them:

* ``xfade`` has no ``zoomout``, so :data:`TransitionKind.ZOOM_OUT` must substitute and warn.
* A compressor cannot deliver a fixed attenuation, so ducking is a volume envelope. See
  :func:`app.renderer.ffmpeg.graph.ducking_expression`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.settings import AiveSettings
from app.models.common import MediaRef, TimeRange, TransitionKind
from app.models.edit_plan import (
    DuckingSpec,
    EditPlan,
    MusicCue,
    NarrationTrack,
    OutputSpec,
    SubtitleCue,
    TimelineClip,
    Transition,
)
from app.renderer.ffmpeg.graph import (
    XFADE_TRANSITIONS,
    FilterGraphBuilder,
    GraphInput,
    db_to_linear,
    ducking_expression,
    escape_filter_value,
    frames_for,
)

ROOT = Path("/project") if Path("/").exists() else Path("C:/project")
CLIP_A = MediaRef(path="raw/001.mp4")
CLIP_B = MediaRef(path="raw/002.mp4")
NARRATION = MediaRef(path="narration.wav")
TRACK = MediaRef(path="music/bed.mp3")


def _settings(**overrides: object) -> AiveSettings:
    return AiveSettings(**overrides)  # type: ignore[arg-type]


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
        "project_id": "graph",
        "created_by": "pytest",
        "clips": clips or (_clip("c1"),),
    }
    fields.update(kwargs)
    return EditPlan(**fields)  # type: ignore[arg-type]


def _build(plan: EditPlan, *, draft: bool = False, subtitle_file: str | None = None):
    return FilterGraphBuilder(_settings(), draft=draft).build(
        plan, project_root=ROOT, subtitle_file=subtitle_file
    )


def _graph_text(plan: EditPlan, **kwargs: object) -> str:
    return _build(plan, **kwargs).filter_complex()  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


class TestInputs:
    def test_each_clip_becomes_its_own_seeked_input(self) -> None:
        """Input-level seeking is what makes a cut from the middle of a long take cheap."""
        graph = _build(_plan(_clip("c1", start=30.0, end=34.0)))
        args = graph.inputs[0].to_args()
        assert args[:4] == ["-ss", "30.000000", "-t", "4.000000"]
        assert args[-2] == "-i"

    def test_a_zero_start_emits_no_seek(self) -> None:
        """`-ss 0` is a no-op that makes every command line harder to read."""
        assert "-ss" not in GraphInput(path=Path("a.mp4"), start=0.0, duration=2.0).to_args()

    def test_a_whole_file_input_has_neither_seek_nor_limit(self) -> None:
        assert GraphInput(path=Path("n.wav")).to_args() == ["-i", str(Path("n.wav"))]

    def test_narration_is_read_whole_because_it_is_reassembled_in_the_graph(self) -> None:
        plan = _plan(
            _clip("c1"),
            narration=NarrationTrack(
                source=NARRATION,
                kept_ranges=(TimeRange(start=0.0, end=2.0), TimeRange(start=5.0, end=7.0)),
            ),
        )
        graph = _build(plan)
        narration_input = graph.inputs[-1]
        assert narration_input.start is None
        assert narration_input.duration is None

    def test_a_music_cue_is_seeked_to_its_source_offset(self) -> None:
        """Skipping an ambient intro is the point of source_offset."""
        plan = _plan(
            _clip("c1", end=10.0),
            music=(
                MusicCue(
                    track=TRACK,
                    timeline_range=TimeRange(start=0.0, end=10.0),
                    source_offset=12.5,
                ),
            ),
        )
        cue_input = _build(plan).inputs[-1]
        assert cue_input.start == pytest.approx(12.5)
        assert cue_input.duration == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #


class TestVideoNormalisation:
    def test_every_clip_is_normalised_to_the_same_geometry(self) -> None:
        """xfade and concat both refuse mismatched inputs, and name neither clip nor field."""
        text = _graph_text(_plan(_clip("c1"), _clip("c2", source=CLIP_B, timeline_start=4.0)))
        assert text.count("scale=1920:1080:force_original_aspect_ratio=decrease") == 2
        assert text.count("setsar=1") == 2
        assert text.count("fps=30") == 2

    def test_footage_is_letterboxed_rather_than_stretched(self) -> None:
        """Distorting faces to fill a frame is never the right default."""
        text = _graph_text(_plan())
        assert "force_original_aspect_ratio=decrease" in text
        assert "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black" in text

    def test_timestamps_are_rebased_before_anything_else(self) -> None:
        chain = _graph_text(_plan()).split(";")[0]
        assert chain.index("setpts=PTS-STARTPTS") < chain.index("fps=")

    def test_speed_is_applied_after_rebasing(self) -> None:
        """The divisor has to act on a zero-based timestamp, not the source's."""
        chain = _graph_text(_plan(_clip("c1", speed=2.0))).split(";")[0]
        assert chain.index("setpts=PTS-STARTPTS") < chain.index("setpts=PTS/2")

    def test_unit_speed_adds_no_filter(self) -> None:
        assert "PTS/1" not in _graph_text(_plan(_clip("c1", speed=1.0)))

    def test_the_output_is_forced_to_the_configured_pixel_format(self) -> None:
        assert "format=yuv420p" in _graph_text(_plan())


class TestVideoJoining:
    def test_a_hard_cut_concatenates(self) -> None:
        text = _graph_text(_plan(_clip("c1"), _clip("c2", timeline_start=4.0)))
        assert "concat=n=2:v=1:a=0" in text
        assert "xfade" not in text

    def test_a_transition_crossfades(self) -> None:
        plan = _plan(
            _clip("c1", end=4.0),
            _clip(
                "c2",
                timeline_start=3.6,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.4),
            ),
        )
        assert "xfade=transition=dissolve:duration=0.400000" in _graph_text(plan)

    def test_the_crossfade_offset_tracks_the_accumulated_stream(self) -> None:
        """xfade positions its fade within its *first* input, which is everything joined so
        far - not the incoming clip, and not the plan's timeline_start."""
        plan = _plan(
            _clip("c1", end=5.0),
            _clip(
                "c2",
                timeline_start=4.5,
                end=4.0,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.5),
            ),
        )
        # Accumulator is 5.0 long; the fade starts 0.5 before its end.
        assert "offset=4.500000" in _graph_text(plan)

    def test_offsets_stay_correct_across_three_transitions(self) -> None:
        dissolve = Transition(kind=TransitionKind.DISSOLVE, duration=0.5)
        plan = _plan(
            _clip("c1", end=4.0),
            _clip("c2", end=4.0, timeline_start=3.5, transition_in=dissolve),
            _clip("c3", end=4.0, timeline_start=7.0, transition_in=dissolve),
        )
        text = _graph_text(plan)
        # 4.0 - 0.5 = 3.5, then (4.0 + 4.0 - 0.5) - 0.5 = 7.0
        assert "offset=3.500000" in text
        assert "offset=7.000000" in text

    def test_the_reported_duration_accounts_for_transition_overlap(self) -> None:
        plan = _plan(
            _clip("c1", end=4.0),
            _clip(
                "c2",
                end=4.0,
                timeline_start=3.6,
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.4),
            ),
        )
        assert _build(plan).duration == pytest.approx(7.6)

    def test_every_mapped_transition_name_is_one_ffmpeg_really_has(self) -> None:
        """Verified against `ffmpeg -h filter=xfade` on the vendored 7.1 build."""
        real = {
            "fade",
            "dissolve",
            "slideleft",
            "slideright",
            "slideup",
            "slidedown",
            "zoomin",
        }
        assert set(XFADE_TRANSITIONS.values()) <= real

    def test_zoom_out_substitutes_and_says_so(self) -> None:
        """xfade implements zoomin and has no inverse; rendering a zoom *in* silently
        would be worse than either substituting or failing."""
        plan = _plan(
            _clip("c1", end=4.0),
            _clip(
                "c2",
                timeline_start=3.6,
                transition_in=Transition(kind=TransitionKind.ZOOM_OUT, duration=0.4),
            ),
        )
        graph = _build(plan)
        assert "zoomout" not in graph.filter_complex()
        assert any("zoom_out" in warning for warning in graph.warnings)
        assert any("c2" in warning for warning in graph.warnings)

    def test_a_gap_is_filled_with_black(self) -> None:
        """A hole in a filter graph is not black, it is everything after it pulled forward."""
        plan = _plan(_clip("c1", end=4.0), _clip("c2", timeline_start=6.0))
        text = _graph_text(plan)
        assert "tpad=stop_mode=add:stop_duration=2.000000" in text

    def test_sub_frame_float_noise_is_not_treated_as_a_gap(self) -> None:
        """Placement sums floats; a microsecond of drift must not add a black frame."""
        plan = _plan(_clip("c1", end=4.0), _clip("c2", timeline_start=4.0000001))
        assert "tpad" not in _graph_text(plan)


class TestDraftMode:
    def test_draft_halves_the_geometry(self) -> None:
        graph = _build(_plan(), draft=True)
        assert (graph.width, graph.height) == (960, 540)

    def test_draft_dimensions_are_even(self) -> None:
        """libx264 with yuv420p cannot encode an odd dimension."""
        plan = _plan(output=OutputSpec(width=1079, height=607))
        graph = FilterGraphBuilder(_settings(), draft=True).build(
            plan, project_root=ROOT, subtitle_file=None
        )
        assert graph.width % 2 == 0
        assert graph.height % 2 == 0

    def test_a_full_render_keeps_the_plans_geometry(self) -> None:
        graph = _build(_plan(output=OutputSpec(width=3840, height=2160)))
        assert (graph.width, graph.height) == (3840, 2160)


class TestSubtitleBurnIn:
    def test_a_bare_filename_is_used(self) -> None:
        """FFmpeg runs in the file's directory, so no drive letter reaches the graph parser."""
        text = _graph_text(_plan(), subtitle_file="subtitle.ass")
        assert "ass=filename=subtitle.ass" in text
        assert ":" not in text.split("ass=filename=")[1].split("[")[0]

    def test_burn_in_is_the_last_video_stage(self) -> None:
        graph = _build(_plan(), subtitle_file="s.ass")
        assert graph.video_label == "vsub"


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


class TestNarration:
    def test_kept_ranges_are_trimmed_and_concatenated(self) -> None:
        """This reassembly *is* the cleanup: the gaps are what was removed."""
        plan = _plan(
            _clip("c1", end=10.0),
            narration=NarrationTrack(
                source=NARRATION,
                kept_ranges=(
                    TimeRange(start=0.0, end=2.0),
                    TimeRange(start=5.0, end=7.0),
                    TimeRange(start=9.0, end=12.0),
                ),
            ),
        )
        text = _graph_text(plan)
        assert "asplit=3" in text
        assert "atrim=start=5.000000:end=7.000000" in text
        assert "concat=n=3:v=0:a=1" in text

    def test_a_single_kept_range_needs_no_split_or_concat(self) -> None:
        plan = _plan(
            _clip("c1", end=10.0),
            narration=NarrationTrack(
                source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=8.0),)
            ),
        )
        text = _graph_text(plan)
        assert "asplit" not in text
        assert "concat=n=1" not in text

    def test_narration_gain_is_applied(self) -> None:
        plan = _plan(
            _clip("c1", end=10.0),
            narration=NarrationTrack(
                source=NARRATION,
                kept_ranges=(TimeRange(start=0.0, end=8.0),),
                gain_db=-3.0,
            ),
        )
        assert "volume=-3.00dB" in _graph_text(plan)


class TestMusicAndDucking:
    def _plan_with_music(self, **cue_kwargs: object) -> EditPlan:
        return _plan(
            _clip("c1", end=10.0),
            narration=NarrationTrack(
                source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=6.0),)
            ),
            music=(
                MusicCue(
                    track=TRACK,
                    timeline_range=TimeRange(start=0.0, end=10.0),
                    **cue_kwargs,  # type: ignore[arg-type]
                ),
            ),
        )

    def test_fades_are_placed_relative_to_the_cue(self) -> None:
        text = _graph_text(self._plan_with_music(fade_in=1.0, fade_out=2.0))
        assert "afade=t=in:st=0:d=1.000000" in text
        assert "afade=t=out:st=8.000000:d=2.000000" in text

    def test_a_delayed_cue_is_moved_to_its_timeline_position(self) -> None:
        plan = _plan(
            _clip("c1", end=20.0),
            music=(MusicCue(track=TRACK, timeline_range=TimeRange(start=4.5, end=12.0)),),
        )
        assert "adelay=4500:all=1" in _graph_text(plan)

    def test_delay_moves_every_channel(self) -> None:
        """Without all=1 only the first channel moves and the bed tears in half."""
        plan = _plan(
            _clip("c1", end=20.0),
            music=(MusicCue(track=TRACK, timeline_range=TimeRange(start=2.0, end=10.0)),),
        )
        assert "adelay=2000:all=1" in _graph_text(plan)

    def test_ducking_is_a_volume_envelope_not_a_compressor(self) -> None:
        """Measured: sidechaincompress saturates near 10 dB whatever ratio it is given, so
        it cannot honour a gain_db contract. See ducking_expression."""
        text = _graph_text(self._plan_with_music(ducking=DuckingSpec()))
        assert "sidechaincompress" not in text
        assert "volume='if(lt(t," in text

    def test_the_envelope_covers_exactly_the_narration(self) -> None:
        text = _graph_text(self._plan_with_music(ducking=DuckingSpec()))
        # Narration runs 0 to 6.0 on the timeline, so the bed recovers at 6.0.
        assert "lt(t,6.000000)" in text

    def test_the_envelope_is_evaluated_per_frame(self) -> None:
        """eval=once would freeze the expression at its first-frame value: no ducking."""
        assert "eval=frame" in _graph_text(self._plan_with_music(ducking=DuckingSpec()))

    def test_ducking_is_applied_after_the_delay(self) -> None:
        """The envelope is in timeline time, so `t` must already mean timeline time."""
        plan = _plan(
            _clip("c1", end=20.0),
            narration=NarrationTrack(
                source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=6.0),)
            ),
            music=(
                MusicCue(
                    track=TRACK,
                    timeline_range=TimeRange(start=3.0, end=15.0),
                    ducking=DuckingSpec(),
                ),
            ),
        )
        chain = next(line for line in _graph_text(plan).split(";") if "adelay" in line)
        assert chain.index("adelay") < chain.index("volume='if")

    def test_ducking_without_narration_warns_rather_than_silently_doing_nothing(self) -> None:
        plan = _plan(
            _clip("c1", end=10.0),
            music=(
                MusicCue(
                    track=TRACK,
                    timeline_range=TimeRange(start=0.0, end=10.0),
                    ducking=DuckingSpec(),
                ),
            ),
        )
        graph = _build(plan)
        assert any("nothing to duck under" in warning for warning in graph.warnings)

    def test_music_and_narration_are_mixed_without_renormalising(self) -> None:
        """amix defaults to dividing by the input count, which would halve the narration."""
        text = _graph_text(self._plan_with_music())
        assert "amix=inputs=2:duration=longest:normalize=0" in text


class TestDuckingExpression:
    def test_the_plateau_is_exactly_the_requested_attenuation(self) -> None:
        """Verified against ffmpeg: this envelope delivers 12.000 dB where the compressor
        managed 8.94 dB and saturated near 10."""
        expression = ducking_expression(gain_db=-12.0, start=0.0, end=5.0, attack=0.15, release=0.6)
        assert f"{db_to_linear(-12.0):.6f}" in expression

    def test_it_ramps_rather_than_steps(self) -> None:
        """An instantaneous 12 dB step is an audible click - worse than not ducking."""
        expression = ducking_expression(gain_db=-12.0, start=0.0, end=5.0, attack=0.15, release=0.6)
        assert "(t-0.000000)/0.150000" in expression
        assert "(t-5.000000)/0.600000" in expression

    def test_it_returns_to_unity_after_the_release(self) -> None:
        expression = ducking_expression(gain_db=-12.0, start=0.0, end=5.0, attack=0.1, release=0.5)
        assert expression.endswith("1))))")

    def test_a_zero_ramp_does_not_divide_by_zero(self) -> None:
        expression = ducking_expression(gain_db=-6.0, start=0.0, end=4.0, attack=0.0, release=0.0)
        assert "/0.000000" not in expression

    @pytest.mark.parametrize("gain_db", [-3.0, -6.0, -12.0, -24.0])
    def test_every_requested_depth_appears_verbatim(self, gain_db: float) -> None:
        """The point of the change: -24 dB really means -24, not the compressor's ~-10."""
        expression = ducking_expression(
            gain_db=gain_db, start=0.0, end=4.0, attack=0.1, release=0.2
        )
        assert f"{db_to_linear(gain_db):.6f}" in expression


class TestSourceAudio:
    def test_a_muted_clip_contributes_no_audio(self) -> None:
        assert "sa0" not in _graph_text(_plan(_clip("c1", mute_source_audio=True)))

    def test_kept_sync_sound_gets_its_own_input(self) -> None:
        """The video input is already consumed; a filter pad cannot be read twice."""
        graph = _build(_plan(_clip("c1", mute_source_audio=False)))
        assert len(graph.inputs) == 2
        assert "sa0" in graph.filter_complex()

    def test_sync_sound_is_attenuated_by_config(self) -> None:
        """Under narration it is texture, not content."""
        assert "volume=-6.00dB" in _graph_text(_plan(_clip("c1", mute_source_audio=False)))


class TestSilentPlans:
    def test_a_plan_with_no_audio_at_all_reports_none(self) -> None:
        """So the renderer can pass -an instead of encoding a silent track."""
        assert _build(_plan(_clip("c1", mute_source_audio=True))).audio_label is None

    def test_a_plan_with_narration_has_an_audio_label(self) -> None:
        plan = _plan(
            _clip("c1", end=10.0),
            narration=NarrationTrack(
                source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=5.0),)
            ),
        )
        assert _build(plan).audio_label == "aout"

    def test_the_mix_is_trimmed_to_the_picture(self) -> None:
        """Narration longer than the picture would otherwise extend the file."""
        plan = _plan(
            _clip("c1", end=4.0),
            narration=NarrationTrack(
                source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=30.0),)
            ),
        )
        assert "atrim=end=4.000000" in _graph_text(plan)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_db_to_linear(self) -> None:
        assert db_to_linear(0.0) == pytest.approx(1.0)
        assert db_to_linear(-6.0) == pytest.approx(0.5012, abs=1e-4)
        assert db_to_linear(-20.0) == pytest.approx(0.1)

    def test_filter_escaping_neutralises_a_windows_path(self) -> None:
        """An unescaped drive letter truncates the graph at the colon."""
        escaped = escape_filter_value("C:\\Users\\me\\subtitle.ass")
        assert "\\:" in escaped
        assert "\\\\" not in escaped

    def test_frames_for_rounds_up(self) -> None:
        assert frames_for(1.01, 30.0) == 31

    def test_frames_for_never_returns_zero(self) -> None:
        """A zero denominator in a progress fraction is not a useful thing to produce."""
        assert frames_for(0.0, 30.0) == 1


class TestGraphShape:
    def test_the_filter_complex_is_semicolon_separated(self) -> None:
        graph = _build(_plan(_clip("c1"), _clip("c2", timeline_start=4.0)))
        assert graph.filter_complex().count(";") == len(graph.filters) - 1

    def test_framing_is_reported_as_unimplemented_rather_than_dropped(self) -> None:
        """A plan whose Ken Burns move silently vanished should say so."""
        from app.models.edit_plan import FramingSpec

        plan = _plan(_clip("c1", framing=FramingSpec(zoom_start=1.0, zoom_end=1.4)))
        assert any("FramingSpec" in warning for warning in _build(plan).warnings)

    def test_static_framing_draws_no_comment(self) -> None:
        from app.models.edit_plan import FramingSpec

        plan = _plan(_clip("c1", framing=FramingSpec()))
        assert _build(plan).warnings == ()

    def test_subtitles_on_the_plan_do_not_burn_in_unless_asked(self) -> None:
        plan = _plan(
            _clip("c1"),
            subtitles=(SubtitleCue(range=TimeRange(start=0.0, end=1.0), text="hi"),),
        )
        assert "ass=" not in _graph_text(plan)
