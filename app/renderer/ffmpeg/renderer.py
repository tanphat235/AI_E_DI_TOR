"""The FFmpeg renderer.

Everything difficult about turning a plan into a video lives in
:mod:`app.renderer.ffmpeg.graph`, which is pure and unit-tested. This module does the parts
that need the outside world: locate the binary, check the request before spending an hour on
it, run the process, read its progress, and keep the log.

Three decisions here are worth knowing about.

**Preflight is a real gate, not a courtesy.** Discovering a missing source file forty minutes
into an encode is unacceptable, so every source is checked for existence, the plan is checked
for placement, and the destination is checked for writability *before* FFmpeg starts.

**FFmpeg runs with its working directory set to the output folder.** Filter graphs and
Windows paths do not mix: ``:`` separates filter options, so ``C:/x.ass`` truncates the graph
at the drive letter. Escaping works but is fragile across FFmpeg versions; running in the
right directory and passing a bare filename removes the problem instead of managing it.

**The full FFmpeg stderr goes to a log file, always.** A failed render's message is the only
evidence of what went wrong, and it is routinely a hundred lines of filter-graph diagnostics
that have no place on a terminal.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from app.config.settings import AiveSettings
from app.models.common import SubtitleFormat
from app.renderer.base import ProgressCallback, RenderRequest, RenderResult
from app.renderer.ffmpeg.graph import GRAPH_VERSION, FilterGraphBuilder, RenderGraph
from app.renderer.ffmpeg.progress import parse_progress
from app.services.ffmpeg_locator import FFmpegLocator, FFmpegNotFoundError
from app.utils.logging import get_logger

logger = get_logger(__name__)

RENDERER_VERSION = f"ffmpeg/1+{GRAPH_VERSION}"


class RenderError(RuntimeError):
    """Raised when FFmpeg fails. Carries the tail of the log, which is the diagnosis."""

    def __init__(self, message: str, *, log_file: Path | None = None) -> None:
        super().__init__(message)
        self.log_file = log_file


class FFmpegRenderer:
    """A :class:`~app.renderer.base.Renderer` over the FFmpeg binary."""

    def __init__(self, settings: AiveSettings, locator: FFmpegLocator) -> None:
        self._settings = settings
        self._locator = locator

    @property
    def name(self) -> str:
        return "ffmpeg"

    @property
    def version(self) -> str:
        return RENDERER_VERSION

    # -- Preflight ----------------------------------------------------------- #

    def preflight(self, request: RenderRequest) -> tuple[str, ...]:
        """Everything that would stop this render, found before it starts."""
        problems: list[str] = []
        plan = request.plan

        try:
            self._locator.locate()
        except FFmpegNotFoundError as exc:
            problems.append(str(exc))

        if not plan.is_placed:
            problems.append(
                "the plan has no timeline positions, so clips cannot be laid out; "
                "run `aive rules normalize` first"
            )

        # Deduplicated: a plan reusing one source forty times should report the missing
        # file once, not forty times.
        seen: set[Path] = set()
        for clip in plan.clips:
            try:
                resolved = clip.source.resolve_within(request.project_root)
            except ValueError as exc:
                problems.append(f"clip {clip.id}: {exc}")
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if not resolved.is_file():
                problems.append(f"clip {clip.id}: {clip.source} does not exist")

        if plan.narration is not None:
            narration = plan.narration.source.resolve_within(request.project_root)
            if not narration.is_file():
                problems.append(f"narration {plan.narration.source} does not exist")

        for position, cue in enumerate(plan.music):
            track = cue.track.resolve_within(request.project_root)
            if not track.is_file():
                problems.append(f"music cue {position}: {cue.track} does not exist")

        problems.extend(self._check_destination(request.destination))

        if request.burn_in_subtitles and not plan.subtitles:
            problems.append(
                "subtitle burn-in was asked for but the plan carries no cues; "
                "run `aive plan subtitles` first"
            )

        return tuple(problems)

    @staticmethod
    def _check_destination(destination: Path) -> list[str]:
        """Whether we can actually write the output, tested rather than assumed."""
        parent = destination.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return [f"cannot create {parent}: {exc}"]

        if destination.is_dir():
            return [f"{destination} is a directory"]

        probe = parent / f".aive-write-test-{destination.name}"
        try:
            probe.touch()
            probe.unlink()
        except OSError as exc:
            return [f"cannot write to {parent}: {exc}"]
        return []

    # -- Render -------------------------------------------------------------- #

    def render(
        self,
        request: RenderRequest,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> RenderResult:
        """Encode the plan.

        Raises:
            RenderError: preflight found a blocking problem, or FFmpeg exited non-zero.
        """
        problems = self.preflight(request)
        if problems:
            msg = "; ".join(problems)
            raise RenderError(msg)

        tools = self._locator.locate()
        subtitle_paths = self._write_subtitles(request)
        burn_in = self._subtitle_to_burn(request, subtitle_paths)

        graph = FilterGraphBuilder(self._settings, draft=request.draft).build(
            request.plan,
            project_root=request.project_root,
            subtitle_file=burn_in.name if burn_in is not None else None,
        )
        for warning in graph.warnings:
            logger.warning("%s", warning)

        command = self._command(str(tools.ffmpeg.path), graph, request)
        log_file = self._log_path(request)
        started = time.monotonic()

        logger.info(
            "Rendering %d clip(s) to %s at %dx%d",
            len(request.plan.clips),
            request.destination.name,
            graph.width,
            graph.height,
        )
        logger.debug("Filter graph: %s", graph.filter_complex())

        self._run(
            command,
            graph=graph,
            log_file=log_file,
            work_dir=request.destination.parent,
            on_progress=on_progress,
        )

        elapsed = time.monotonic() - started
        logger.info("Rendered %s in %.1fs", request.destination.name, elapsed)
        return RenderResult(
            video=request.destination,
            subtitles=subtitle_paths,
            duration=graph.duration,
            elapsed=elapsed,
            log_file=log_file,
        )

    # -- Command ------------------------------------------------------------- #

    def _command(self, ffmpeg: str, graph: RenderGraph, request: RenderRequest) -> list[str]:
        """Assemble the argv.

        Kept separate from :meth:`_run` so a test can assert on the command without a binary
        on the machine, and so `--dry-run` can print it.
        """
        output = self._settings.output
        render = self._settings.render

        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            # Progress goes to FFmpeg's stdout, leaving its stderr a clean log.
            "-progress",
            "pipe:1",
            "-loglevel",
            "warning",
        ]
        command.append("-y" if render.overwrite else "-n")

        for graph_input in graph.inputs:
            command += graph_input.to_args()

        command += ["-filter_complex", graph.filter_complex()]
        command += ["-map", f"[{graph.video_label}]"]
        if graph.audio_label is not None:
            command += ["-map", f"[{graph.audio_label}]"]

        command += [
            "-c:v",
            output.video_codec,
            "-crf",
            str(render.draft_crf if request.draft else output.video_crf),
            "-preset",
            render.draft_preset if request.draft else output.video_preset,
            "-pix_fmt",
            output.pixel_format,
            "-r",
            f"{graph.fps:g}",
        ]
        if graph.audio_label is not None:
            command += [
                "-c:a",
                output.audio_codec,
                "-b:a",
                output.audio_bitrate,
                "-ar",
                str(output.audio_sample_rate),
            ]
        else:
            command.append("-an")

        if render.threads > 0:
            command += ["-threads", str(render.threads)]

        # Bounds the output even if a filter's own duration accounting disagrees with ours -
        # cheap insurance against a trailing frozen frame.
        command += ["-t", f"{graph.duration:.6f}"]
        # +faststart moves the index to the front, so the result streams and scrubs properly
        # in a browser rather than needing a full download first.
        #
        # Absolute, and it has to be: _run sets FFmpeg's working directory to the log folder
        # so the subtitle filter can take a bare filename (see the module docstring). A
        # relative `-o` is then resolved against *that* folder, and `aive render -o
        # ./out/x.mp4` failed with "No such file or directory" while an absolute path worked.
        # Every other path in this command already comes from resolve_within(); this one
        # comes straight from the caller.
        command += ["-movflags", "+faststart", str(request.destination.resolve())]
        return command

    def _run(
        self,
        command: list[str],
        *,
        graph: RenderGraph,
        log_file: Path,
        work_dir: Path,
        on_progress: ProgressCallback | None,
    ) -> None:
        """Run FFmpeg, streaming progress and capturing the log.

        ``work_dir`` is the folder the bare subtitle filename in the graph resolves against,
        so it must be the one the ASS file was written to.
        """
        render = self._settings.render
        interval = render.progress_interval
        last_report = 0.0

        log_file.parent.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8", errors="replace") as log:
            log.write(" ".join(command) + "\n\n")
            log.flush()
            try:
                # argv list, never a shell string: nothing here is interpolated by a shell.
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    # See the module docstring: this is what keeps the subtitle filename
                    # free of drive letters and backslashes. It must be the folder holding
                    # the ASS file - the *output* folder, not the log folder one level down,
                    # which is where this used to point and which made burn-in impossible.
                    cwd=str(work_dir),
                )
            except OSError as exc:
                msg = f"could not start ffmpeg: {exc}"
                raise RenderError(msg, log_file=log_file) from exc

            try:
                assert process.stdout is not None  # stdout=PIPE guarantees it
                for update in parse_progress(process.stdout):
                    now = time.monotonic()
                    throttled = now - last_report < interval and not update.finished
                    if on_progress is None or throttled:
                        continue
                    last_report = now
                    eta = update.eta(graph.duration)
                    detail = f"{update.out_time:.1f}s of {graph.duration:.1f}s"
                    if eta is not None:
                        detail += f", ~{eta:.0f}s left"
                    on_progress(update.fraction(graph.duration), detail)

                returncode = process.wait(timeout=render.timeout or None)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                msg = f"ffmpeg exceeded the {render.timeout:.0f}s render timeout"
                raise RenderError(msg, log_file=log_file) from None
            except BaseException:
                # Covers KeyboardInterrupt as well as errors: a detached FFmpeg would keep
                # writing to a file the user believes was abandoned.
                process.kill()
                process.wait()
                raise

        if returncode != 0:
            msg = f"ffmpeg exited {returncode}; see {log_file}{_log_tail(log_file)}"
            raise RenderError(msg, log_file=log_file)

    # -- Subtitles ----------------------------------------------------------- #

    def _write_subtitles(self, request: RenderRequest) -> tuple[Path, ...]:
        """Write the requested sidecar subtitle files.

        Imported here rather than at module scope: the subtitle writers are the one place a
        renderer touches something outside the plan, and keeping the import local makes the
        dependency visible at the call site.
        """
        if not request.subtitle_formats or not request.plan.subtitles:
            return ()

        from app.subtitles.registry import writer_for

        style = self._settings.subtitle
        written: list[Path] = []
        destination_dir = request.destination.parent
        for fmt in request.subtitle_formats:
            path = destination_dir / f"{request.destination.stem}.{fmt.value}"
            writer_for(fmt, style).write(request.plan.subtitles, path, style=style)
            written.append(path)
        return tuple(written)

    def _subtitle_to_burn(self, request: RenderRequest, written: tuple[Path, ...]) -> Path | None:
        """The ASS file to burn in, writing one if the caller only asked for SRT.

        ASS rather than SRT because burning in is a styling operation: font, size, outline
        and position all come from config, and SRT can express none of them.
        """
        if not request.burn_in_subtitles or not request.plan.subtitles:
            return None

        for path in written:
            if path.suffix == f".{SubtitleFormat.ASS.value}":
                return path

        from app.subtitles.registry import writer_for

        style = self._settings.subtitle
        path = request.destination.parent / f"{request.destination.stem}.burn.ass"
        writer_for(SubtitleFormat.ASS, style).write(request.plan.subtitles, path, style=style)
        return path

    def _log_path(self, request: RenderRequest) -> Path:
        suffix = "draft" if request.draft else "final"
        return request.destination.parent / "logs" / f"render-{suffix}.log"


def _log_tail(log_file: Path, *, lines: int = 6) -> str:
    """The last few log lines, for an error message.

    A path alone is not actionable when the caller is an agent reading stdout, and the
    real diagnosis is almost always in FFmpeg's final lines.
    """
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    tail = [line for line in content[-lines:] if line.strip()]
    return "\n  " + "\n  ".join(tail) if tail else ""


__all__ = ["RENDERER_VERSION", "FFmpegRenderer", "RenderError"]
