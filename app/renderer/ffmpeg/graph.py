"""Building an FFmpeg filter graph from an Edit Plan.

This module is the whole reason the renderer is testable. It is **pure**: an Edit Plan and
some settings in, a description of inputs and a filter string out. No subprocess, no
filesystem, no FFmpeg on the machine. Every graph-shaping decision below can be asserted
against in a unit test, which is the only practical way to have confidence in a filter
graph — the alternative is rendering forty seconds of video and looking at it.

## How the video chain is built

Each clip becomes its own FFmpeg **input** with ``-ss``/``-t`` in front of it, rather than
one input per file with ``trim`` in the graph. Input-level seeking lets FFmpeg skip decoding
everything before the cut; ``trim`` decodes from zero and throws the result away. On a plan
that takes four seconds from the middle of a twenty-minute take, that is the difference
between a render and a wait.

Each input is then normalised to identical geometry, frame rate, pixel format and sample
aspect. This is not tidiness: ``xfade`` and ``concat`` both *refuse* inputs that differ, and
the failure message names neither the clip nor the property, so mixed phone and camera
footage would otherwise produce an unattributable error twenty clips in.

The clips are then folded left into one stream:

* a **hard cut** appends with ``concat``
* a **transition** overlaps with ``xfade``, whose ``offset`` is where the crossfade starts
  *within the accumulated stream* — so the accumulator has to track its own length, which
  is what ``_VideoFold`` does
* a **gap** (a plan that deliberately leaves a hole) is filled with black via ``tpad``,
  because a hole in a filter graph is not silence-and-black, it is a shorter video with
  everything after it pulled forward

## How the audio chain is built

Narration is a single file with ``kept_ranges``, so it is split, trimmed once per range, and
concatenated — that reassembly *is* the cleanup, and it is what makes timeline time diverge
from source time.

Music beds are trimmed, faded, delayed to their timeline position, and ducked. Ducking is a
``volume`` envelope over timeline time, not ``sidechaincompress`` — see
:func:`ducking_expression` for the measurements that forced that choice, and why the envelope
is exact rather than approximate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from app.config.settings import AiveSettings
from app.models.common import TransitionKind
from app.models.edit_plan import EditPlan, MusicCue, TimelineClip

GRAPH_VERSION = "ffmpeg-graph/1"

XFADE_TRANSITIONS: dict[TransitionKind, str] = {
    TransitionKind.FADE: "fade",
    TransitionKind.DISSOLVE: "dissolve",
    # A video crossfade and a video fade are the same operation; the distinction exists in
    # the plan because it is meaningful for *audio*, where a crossfade is not a fade.
    TransitionKind.CROSSFADE: "fade",
    TransitionKind.SLIDE_LEFT: "slideleft",
    TransitionKind.SLIDE_RIGHT: "slideright",
    TransitionKind.SLIDE_UP: "slideup",
    TransitionKind.SLIDE_DOWN: "slidedown",
    TransitionKind.ZOOM_IN: "zoomin",
}
"""Plan transitions that map onto an ``xfade`` transition of the same meaning.

:data:`TransitionKind.ZOOM_OUT` is absent because ``xfade`` implements ``zoomin`` and has no
inverse. Asking for it gets the configured substitute plus a warning naming the clip —
silently rendering a zoom *in* where the plan said out would be worse than either.
"""


def escape_filter_value(value: str) -> str:
    """Escape a string for use as a filter option value.

    The filtergraph parser treats ``:`` as an option separator, ``,`` as a filter separator
    and ``\\`` as an escape, so a Windows path dropped in raw truncates the graph at the
    drive letter.

    Used for defence in depth only: :class:`~app.renderer.ffmpeg.renderer.FFmpegRenderer`
    runs FFmpeg with its working directory set to the folder holding the subtitle file and
    passes a bare filename, which avoids the problem rather than escaping it.
    """
    return (
        value.replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )


def ducking_expression(
    *, gain_db: float, start: float, end: float, attack: float, release: float
) -> str:
    """A ``volume`` expression that attenuates by exactly ``gain_db`` between two times.

    **Why not ``sidechaincompress``.** :class:`~app.models.edit_plan.DuckingSpec` specifies a
    fixed attenuation in dB, and a compressor cannot deliver one — its output depends on how
    far the sidechain exceeds the threshold, and measurement shows FFmpeg's implementation
    saturating near 10 dB whatever ratio it is given:

    ==========  ===========
    ``ratio``   attenuation
    ==========  ===========
    3           6.96 dB
    7           8.94 dB
    12          9.57 dB
    20          9.91 dB
    ==========  ===========

    So a plan asking for the default -12 dB would silently get about -9, and a plan asking
    for -20 would get the same -9. The contract would be decorative.

    **Why automation is exact here, and does not drift.** The renderer does not have to
    *detect* speech: cleanup lays the narration's kept ranges end to end, so on the timeline
    narration is one contiguous block from 0 to
    :attr:`~app.models.edit_plan.NarrationTrack.timeline_duration`. That interval is a fact
    stated by the plan, so every consumer derives the same envelope from the same field —
    which is the property the sidechain framing was originally chosen to protect.

    ``attack`` and ``release`` become the ramp lengths, so both fields still mean what they
    say. Ramping rather than switching matters: an instantaneous 12 dB step is audible as a
    click, which is the one artefact worse than not ducking.
    """
    gain = db_to_linear(gain_db)
    attack = max(attack, 0.001)
    release = max(release, 0.001)
    duck_in = start + attack
    duck_out = end + release

    # Nested if() rather than min/max arithmetic: FFmpeg's expression evaluator has both,
    # but a reader debugging a render needs to see the five phases spelled out.
    return (
        f"if(lt(t,{start:.6f}),1,"
        f"if(lt(t,{duck_in:.6f}),1+({gain:.6f}-1)*(t-{start:.6f})/{attack:.6f},"
        f"if(lt(t,{end:.6f}),{gain:.6f},"
        f"if(lt(t,{duck_out:.6f}),{gain:.6f}+(1-{gain:.6f})*(t-{end:.6f})/{release:.6f},"
        f"1))))"
    )


def db_to_linear(db: float) -> float:
    """Decibels to linear amplitude. ``sidechaincompress`` wants a 0-1 threshold, not dB."""
    return float(10.0 ** (db / 20.0))


@dataclass(frozen=True, slots=True)
class GraphInput:
    """One FFmpeg ``-i`` and the seek arguments in front of it."""

    path: Path
    start: float | None = None
    """``-ss``. ``None`` for a file read whole, which is how narration and music arrive."""
    duration: float | None = None
    """``-t``, applied to the input so decoding stops at the cut."""

    def to_args(self) -> list[str]:
        """Arguments for this input, seek first.

        ``-ss`` *before* ``-i`` is what makes seeking cheap. FFmpeg still lands on the exact
        frame — it decodes from the preceding keyframe and discards — so accuracy is not
        the thing being traded away.
        """
        args: list[str] = []
        if self.start is not None and self.start > 0.0:
            args += ["-ss", f"{self.start:.6f}"]
        if self.duration is not None:
            args += ["-t", f"{self.duration:.6f}"]
        args += ["-i", str(self.path)]
        return args


@dataclass(frozen=True, slots=True)
class RenderGraph:
    """A complete filter graph, ready to become an FFmpeg command line."""

    inputs: tuple[GraphInput, ...]
    filters: tuple[str, ...]
    video_label: str
    audio_label: str | None
    """``None`` when the plan produces no audio at all — a silent montage with every clip
    muted. The renderer passes ``-an`` rather than encoding a silent track."""
    duration: float
    width: int
    height: int
    fps: float
    warnings: tuple[str, ...] = ()
    """Things the plan asked for that this renderer will not do. Reported, never silently
    dropped: a plan whose framing move vanished should say so."""

    def filter_complex(self) -> str:
        """The graph as a single ``-filter_complex`` argument."""
        return ";".join(self.filters)


@dataclass
class _VideoFold:
    """Accumulator for the left fold over clips.

    Its whole job is remembering how long the joined stream is, because ``xfade`` positions
    its crossfade relative to the *start of its first input* — which is timeline zero once
    the clips are chained. Deriving that from the plan's ``timeline_start`` instead would
    be correct only while placement stays exactly packed.
    """

    label: str
    duration: float
    counter: int = 0
    filters: list[str] = field(default_factory=list)

    def next_label(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"


class FilterGraphBuilder:
    """Turns an Edit Plan into a :class:`RenderGraph`."""

    def __init__(self, settings: AiveSettings, *, draft: bool = False) -> None:
        self._settings = settings
        self._draft = draft

    # -- Public -------------------------------------------------------------- #

    def build(
        self, plan: EditPlan, *, project_root: Path, subtitle_file: str | None
    ) -> RenderGraph:
        """Build the graph for ``plan``.

        Args:
            plan: A **placed** plan. Placement is the renderer's precondition, not its job;
                :class:`~app.renderer.ffmpeg.renderer.FFmpegRenderer` refuses an unplaced
                plan in preflight with a pointer at ``aive rules normalize``.
            project_root: Used to resolve every :class:`~app.models.common.MediaRef`.
            subtitle_file: Bare filename of an ASS file to burn in, or ``None``. A filename
                rather than a path because FFmpeg runs with its working directory set to
                the file's folder; see :func:`escape_filter_value`.
        """
        width, height, fps = self._geometry(plan)
        inputs: list[GraphInput] = []
        warnings: list[str] = []

        video = self._build_video(
            plan,
            project_root=project_root,
            inputs=inputs,
            warnings=warnings,
            width=width,
            height=height,
            fps=fps,
        )
        filters = list(video.filters)
        video_label = video.label

        if subtitle_file is not None:
            burned = "vsub"
            filters.append(f"[{video_label}]ass=filename={subtitle_file}[{burned}]")
            video_label = burned

        audio_label, audio_filters, audio_warnings = self._build_audio(
            plan, project_root=project_root, inputs=inputs, total=video.duration
        )
        filters.extend(audio_filters)
        warnings.extend(audio_warnings)

        return RenderGraph(
            inputs=tuple(inputs),
            filters=tuple(filters),
            video_label=video_label,
            audio_label=audio_label,
            duration=video.duration,
            width=width,
            height=height,
            fps=fps,
            warnings=tuple(warnings),
        )

    # -- Geometry ------------------------------------------------------------ #

    def _geometry(self, plan: EditPlan) -> tuple[int, int, float]:
        """Output size and frame rate, after any draft downscale.

        Dimensions are forced even because ``libx264`` with ``yuv420p`` cannot encode an odd
        one, and a 0.5 scale of 1080 is 540 but of 1079 would be 539.5.
        """
        output = plan.output
        render = self._settings.render
        if not self._draft:
            return output.width, output.height, output.fps

        scale = render.draft_scale
        width = max(2, int(output.width * scale) // 2 * 2)
        height = max(2, int(output.height * scale) // 2 * 2)
        return width, height, render.draft_fps_or or output.fps

    # -- Video --------------------------------------------------------------- #

    def _build_video(
        self,
        plan: EditPlan,
        *,
        project_root: Path,
        inputs: list[GraphInput],
        warnings: list[str],
        width: int,
        height: int,
        fps: float,
    ) -> _VideoFold:
        fold = _VideoFold(label="", duration=0.0)
        labels: list[str] = []

        # Every clip gets its own input and its own normalising chain first, so the graph
        # reads top to bottom: all the sources, then the joins between them.
        for index, clip in enumerate(plan.clips):
            inputs.append(
                GraphInput(
                    path=clip.source.resolve_within(project_root),
                    start=clip.source_range.start,
                    duration=clip.source_range.duration,
                )
            )
            label = f"v{index}"
            labels.append(label)
            fold.filters.append(
                self._normalise_clip(index, clip, label=label, width=width, height=height, fps=fps)
            )
            if clip.framing is not None and not clip.framing.is_static:
                warnings.append(
                    f"clip {clip.id}: framing (zoom/crop) is described in the plan but not "
                    f"applied - this renderer does not implement FramingSpec"
                )

        fold.label = labels[0]
        fold.duration = plan.clips[0].timeline_duration

        for clip, label in zip(plan.clips[1:], labels[1:], strict=True):
            self._join(fold, clip=clip, incoming=label, fps=fps, warnings=warnings)

        # A final format pass. The encoder needs a pixel format it can take, and xfade can
        # promote to a higher bit depth than the input on some builds.
        final = fold.next_label("vout")
        fold.filters.append(f"[{fold.label}]format={self._settings.output.pixel_format}[{final}]")
        fold.label = final
        return fold

    def _normalise_clip(
        self,
        index: int,
        clip: TimelineClip,
        *,
        label: str,
        width: int,
        height: int,
        fps: float,
    ) -> str:
        """One clip's chain: rebase, retime, fit, pad, and pin the sample aspect.

        ``scale`` with ``force_original_aspect_ratio=decrease`` followed by ``pad`` letterboxes
        rather than stretching. Given the plan states its delivery aspect ratio and the
        footage may not match it, distorting faces to fill the frame is never the right
        default; the plan's ``FramingSpec`` is where a deliberate crop would be expressed.
        """
        stages = ["setpts=PTS-STARTPTS"]
        if abs(clip.speed - 1.0) > 1e-6:
            # Applied after rebasing so the divisor acts on a zero-based timestamp.
            stages.append(f"setpts=PTS/{clip.speed:.6f}")
        stages += [
            f"fps={fps:g}",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
            "setsar=1",
        ]
        return f"[{index}:v]{','.join(stages)}[{label}]"

    def _join(
        self,
        fold: _VideoFold,
        *,
        clip: TimelineClip,
        incoming: str,
        fps: float,
        warnings: list[str],
    ) -> None:
        """Append one clip to the accumulated stream."""
        # Half a frame: below that a "gap" is float noise from placement arithmetic, not an
        # editorial hole, and padding it would add a black frame nobody asked for.
        tolerance = 0.5 / fps if fps > 0 else 0.001
        if clip.timeline_start is not None:
            gap = clip.timeline_start - fold.duration
            if gap > tolerance:
                padded = fold.next_label("vgap")
                fold.filters.append(
                    f"[{fold.label}]tpad=stop_mode=add:stop_duration={gap:.6f}[{padded}]"
                )
                fold.label = padded
                fold.duration += gap

        # concat and xfade both refuse inputs whose timebases differ, and a chain of many
        # concats can drift its output timebase away from the per-clip 1/fps one (observed:
        # FFmpeg rebasing an accumulated stream to 1/1000000 several joins in). Rebasing both
        # operands immediately before every join is cheap and makes the join correct
        # regardless of how its inputs got here.
        main = fold.next_label("vtb")
        fold.filters.append(f"[{fold.label}]settb=1/{fps:g}[{main}]")
        rebased_incoming = fold.next_label("vtb")
        fold.filters.append(f"[{incoming}]settb=1/{fps:g}[{rebased_incoming}]")

        transition = clip.transition_in
        duration = transition.duration if transition is not None else 0.0
        joined = fold.next_label("vj")

        if transition is None or duration <= 0.0:
            fold.filters.append(f"[{main}][{rebased_incoming}]concat=n=2:v=1:a=0[{joined}]")
            fold.duration += clip.timeline_duration
        else:
            name = XFADE_TRANSITIONS.get(transition.kind)
            if name is None:
                substitute = self._settings.render.default_transition
                name = XFADE_TRANSITIONS.get(substitute, "dissolve")
                warnings.append(
                    f"clip {clip.id}: ffmpeg's xfade has no {transition.kind.value!r} "
                    f"transition, so {substitute.value!r} was used instead"
                )
            offset = max(0.0, fold.duration - duration)
            fold.filters.append(
                f"[{main}][{rebased_incoming}]"
                f"xfade=transition={name}:duration={duration:.6f}:offset={offset:.6f}"
                f"[{joined}]"
            )
            fold.duration += clip.timeline_duration - duration

        fold.label = joined

    # -- Audio --------------------------------------------------------------- #

    def _build_audio(
        self,
        plan: EditPlan,
        *,
        project_root: Path,
        inputs: list[GraphInput],
        total: float,
    ) -> tuple[str | None, list[str], list[str]]:
        """Narration, music beds and any kept source audio, mixed."""
        filters: list[str] = []
        warnings: list[str] = []
        rate = self._settings.output.audio_sample_rate
        stems: list[str] = []

        # Narration occupies one contiguous timeline block, because cleanup lays its kept
        # ranges end to end. That interval is all a ducking envelope needs, so no copy of
        # the narration stream is routed anywhere except the mix.
        narration_end: float | None = None
        if plan.narration is not None:
            narration_index = len(inputs)
            inputs.append(GraphInput(path=plan.narration.source.resolve_within(project_root)))
            stems.append(
                self._build_narration(plan, index=narration_index, filters=filters, rate=rate)
            )
            narration_end = plan.narration.timeline_duration

        for position, cue in enumerate(plan.music):
            index = len(inputs)
            inputs.append(
                GraphInput(
                    path=cue.track.resolve_within(project_root),
                    start=cue.source_offset,
                    duration=cue.timeline_range.duration,
                )
            )
            stems.append(
                self._build_music_cue(
                    cue,
                    position=position,
                    index=index,
                    filters=filters,
                    rate=rate,
                    narration_end=narration_end,
                    warnings=warnings,
                )
            )

        stems.extend(
            self._build_source_audio(
                plan, project_root=project_root, inputs=inputs, filters=filters, rate=rate
            )
        )

        if not stems:
            return None, filters, warnings
        if len(stems) == 1:
            final = "aout"
            filters.append(
                f"[{stems[0]}]aresample={rate},apad=whole_dur={total:.6f},"
                f"atrim=end={total:.6f}[{final}]"
            )
            return final, filters, warnings

        final = "aout"
        filters.append(
            "".join(f"[{stem}]" for stem in stems)
            # normalize=0 is essential: amix otherwise divides every input by the number of
            # inputs, so adding a music bed would quietly halve the narration.
            + f"amix=inputs={len(stems)}:duration=longest:normalize=0,"
            f"aresample={rate},apad=whole_dur={total:.6f},atrim=end={total:.6f}[{final}]"
        )
        return final, filters, warnings

    def _build_narration(self, plan: EditPlan, *, index: int, filters: list[str], rate: int) -> str:
        """Reassemble the narration from its kept ranges.

        This concatenation *is* the cleanup: the removed silence, fillers and retakes are
        the gaps between these ranges, and laying the survivors end to end is what makes
        timeline time differ from source time.
        """
        narration = plan.narration
        assert narration is not None  # guarded by the caller

        ranges = narration.kept_ranges
        if len(ranges) == 1:
            single = ranges[0]
            base = "n0"
            filters.append(
                f"[{index}:a]atrim=start={single.start:.6f}:end={single.end:.6f},"
                f"asetpts=N/SR/TB[{base}]"
            )
            joined = base
        else:
            labels = [f"nr{position}" for position in range(len(ranges))]
            filters.append(
                f"[{index}:a]asplit={len(ranges)}" + "".join(f"[{name}_s]" for name in labels)
            )
            for label, kept in zip(labels, ranges, strict=True):
                filters.append(
                    f"[{label}_s]atrim=start={kept.start:.6f}:end={kept.end:.6f},"
                    f"asetpts=N/SR/TB[{label}]"
                )
            joined = "ncat"
            filters.append(
                "".join(f"[{name}]" for name in labels)
                + f"concat=n={len(labels)}:v=0:a=1[{joined}]"
            )

        final = "narr"
        filters.append(f"[{joined}]aresample={rate},volume={narration.gain_db:.2f}dB[{final}]")
        return final

    def _build_music_cue(
        self,
        cue: MusicCue,
        *,
        position: int,
        index: int,
        filters: list[str],
        rate: int,
        narration_end: float | None,
        warnings: list[str],
    ) -> str:
        """One music bed: trim, level, fade, place, then duck.

        Order is load-bearing. Fades are expressed relative to the cue, so they go on before
        ``adelay``; ducking is expressed in timeline time, so it goes on after, once the
        stream's ``t`` means what the envelope thinks it means.
        """
        span = cue.timeline_range
        label = f"m{position}"
        stages = [
            "asetpts=N/SR/TB",
            f"aresample={rate}",
            f"volume={cue.gain_db:.2f}dB",
        ]
        if cue.fade_in > 0.0:
            stages.append(f"afade=t=in:st=0:d={cue.fade_in:.6f}")
        if cue.fade_out > 0.0:
            start = max(0.0, span.duration - cue.fade_out)
            stages.append(f"afade=t=out:st={start:.6f}:d={cue.fade_out:.6f}")
        if span.start > 0.0:
            # all=1 delays every channel; without it only the first is moved and the bed
            # arrives as a stereo image torn in half.
            stages.append(f"adelay={round(span.start * 1000)}:all=1")

        ducking = cue.ducking
        if ducking is not None and narration_end is None:
            warnings.append(
                f"music cue {position} asks for ducking but the plan has no narration "
                f"track, so there is nothing to duck under"
            )
        elif ducking is not None and narration_end is not None:
            expression = ducking_expression(
                gain_db=ducking.gain_db,
                start=0.0,
                end=narration_end,
                attack=ducking.attack,
                release=ducking.release,
            )
            # eval=frame, not the default eval=once: the expression is a function of t, and
            # evaluated once it would freeze at whatever value t had on the first frame.
            stages.append(f"volume='{expression}':eval=frame")

        filters.append(f"[{index}:a]{','.join(stages)}[{label}]")
        return label

    def _build_source_audio(
        self,
        plan: EditPlan,
        *,
        project_root: Path,
        inputs: list[GraphInput],
        filters: list[str],
        rate: int,
    ) -> list[str]:
        """Sync sound from clips that asked to keep it.

        Each such clip needs a *second* input on the same file and range: the video input is
        already consumed by the video chain, and a filter output pad cannot be read twice.
        Attenuated by config, because sync sound under narration is texture rather than
        content.
        """
        labels: list[str] = []
        gain = self._settings.render.keep_source_audio_gain_db
        for position, clip in enumerate(plan.clips):
            if clip.mute_source_audio or clip.timeline_start is None:
                continue
            index = len(inputs)
            inputs.append(
                GraphInput(
                    path=clip.source.resolve_within(project_root),
                    start=clip.source_range.start,
                    duration=clip.source_range.duration,
                )
            )
            label = f"sa{position}"
            stages = ["asetpts=N/SR/TB", f"aresample={rate}", f"volume={gain:.2f}dB"]
            if clip.timeline_start > 0.0:
                stages.append(f"adelay={round(clip.timeline_start * 1000)}:all=1")
            filters.append(f"[{index}:a]{','.join(stages)}[{label}]")
            labels.append(label)
        return labels


def frames_for(duration: float, fps: float) -> int:
    """Frame count for a duration, used to report progress as a fraction."""
    return max(1, math.ceil(duration * fps))


__all__ = [
    "GRAPH_VERSION",
    "XFADE_TRANSITIONS",
    "FilterGraphBuilder",
    "GraphInput",
    "RenderGraph",
    "db_to_linear",
    "ducking_expression",
    "escape_filter_value",
    "frames_for",
]
