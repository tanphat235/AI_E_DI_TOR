"""``aive analyze`` - turn raw media into structured facts.

Three subcommands, one per kind of input: ``audio`` for the narration (Phase 2), ``video``
for the footage (Phase 3), ``music`` for the bed library (Phase 7).

The output contract matters as much as the analysis. The complete document goes to
``.aive/narration.json``; stdout gets a digest. Narration text is the one part of the
digest that cannot be compressed - the director genuinely needs to read what is being
said - so everything around it is kept terse.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from app.analysis.audio.analyzer import TrackFailure
from app.analysis.audio.base import AudioDependencyMissingError
from app.analysis.speech.base import ProgressCallback
from app.analysis.speech.beats import BEATS_VERSION, NarrationBeatBuilder
from app.analysis.speech.cleanup import CLEANUP_VERSION
from app.analysis.speech.whisper_recognizer import (
    RECOGNIZER_VERSION,
    SpeechDependencyMissingError,
)
from app.analysis.vision.frames import FrameReadError
from app.analysis.vision.probe import ProbeError, VideoDependencyMissingError
from app.analysis.vision.scenes import SceneDetectionError
from app.cli.output import ExitCode, emit_digest, emit_error, emit_json, note, write_json_file
from app.config.settings import AiveSettings, load_settings
from app.models.audio import MusicLibrary, MusicTrack
from app.models.common import MediaRef, TimeRange
from app.models.speech import NarrationAnalysis, SpeechCleanupReport, Transcript
from app.models.video import ClipAnalysis, FootageAnalysis
from app.services.container import build_container
from app.services.paths import ProjectPaths
from app.utils.logging import get_logger

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, help="Analyse project media into structured facts.")

ANALYSIS_VERSION = f"{RECOGNIZER_VERSION}+{CLEANUP_VERSION}+{BEATS_VERSION}"
"""Composite version stamped into the result.

A change to any of the three stages invalidates a cached document, because a
transcript, its cleanup and its beats are only coherent when produced together.
"""


@app.command("audio")
def audio(
    project: Annotated[Path, typer.Argument(help="Project directory.")],
    model: Annotated[
        str | None,
        typer.Option("--model", help="Override the recogniser size, e.g. tiny, small, large-v3."),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", help="Force a language code instead of autodetecting."),
    ] = None,
    no_cleanup: Annotated[
        bool,
        typer.Option("--no-cleanup", help="Transcribe only; keep every second of narration."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-analyse even when a valid cached result exists."),
    ] = False,
    full: Annotated[
        bool,
        typer.Option("--full", help="Print the whole document instead of a digest."),
    ] = False,
) -> None:
    """Transcribe the narration, decide what to cut, and split it into beats."""
    paths = ProjectPaths.for_root(project)
    if not paths.exists():
        emit_error(
            "project.not_initialised",
            f"{paths.root} is not an AIVE project",
            hint=f"initialise it with: aive project init {paths.root.as_posix()}",
            exit_code=ExitCode.NOT_FOUND,
        )

    narration_path = paths.find_narration()
    if narration_path is None:
        emit_error(
            "narration.missing",
            f"no narration audio found in {paths.root}",
            hint=(
                "put narration.wav at the project root, or any audio file in audio/. "
                "Accepted stems: narration, voice, voiceover, vo."
            ),
            exit_code=ExitCode.NOT_FOUND,
        )

    overrides: dict[str, object] = {}
    if model is not None:
        overrides["model"] = model
    if language is not None:
        overrides["language"] = language
    container = build_container(
        paths.root,
        settings=None if not overrides else _with_speech_overrides(paths.root, overrides),
    )
    settings = container.settings

    cached = _load_cached(paths, narration_path, expected_model=settings.speech.model)
    if cached is not None and not force:
        note(f"Reusing cached analysis from {paths.narration_file}")
        note("  pass --force to re-transcribe")
        _report(cached, paths, full=full, reused=True)
        return

    ref = paths.to_ref(narration_path)
    recognizer = container.recognizer
    cleaner = container.cleaner
    if recognizer is None or cleaner is None:  # pragma: no cover - always wired
        emit_error(
            "internal.not_wired",
            "the speech components are not available in this build",
            exit_code=ExitCode.INTERNAL,
        )

    note(f"Transcribing {ref} with {settings.speech.model} (this can take a while)")
    try:
        transcript = recognizer.transcribe(
            narration_path, ref=ref, on_progress=_transcription_progress()
        )
    except SpeechDependencyMissingError as exc:
        emit_error(
            "dependency.missing",
            str(exc).splitlines()[0],
            hint='install it with: pip install -e ".[speech]"',
            exit_code=ExitCode.ENVIRONMENT,
        )
    except (FileNotFoundError, ValueError) as exc:
        emit_error(
            "narration.unreadable",
            str(exc),
            hint="check the file is real audio and not zero bytes",
            exit_code=ExitCode.INVALID_INPUT,
        )

    if not transcript.segments:
        emit_error(
            "narration.no_speech",
            f"no speech was recognised in {ref}",
            hint=(
                "check the file actually contains speech, and try --language to skip "
                "autodetection on a short or noisy recording"
            ),
            exit_code=ExitCode.INVALID_INPUT,
            details={"duration": transcript.duration, "language": transcript.language},
        )

    if no_cleanup:
        cleanup = _keep_everything(transcript)
        note("Cleanup skipped: keeping the full narration")
    else:
        cleanup = cleaner.clean(transcript, audio=narration_path)

    beats = NarrationBeatBuilder().build(transcript, cleanup)

    analysis = NarrationAnalysis(
        source=ref,
        transcript=transcript,
        cleanup=cleanup,
        beats=beats,
        analyzer_version=ANALYSIS_VERSION,
        analyzed_at=datetime.now(UTC),
    )
    write_json_file(analysis, paths.narration_file)
    _report(analysis, paths, full=full, reused=False)


@app.command("video")
def video(
    project: Annotated[Path, typer.Argument(help="Project directory.")],
    clip: Annotated[
        str | None,
        typer.Option("--clip", help="Analyse only this clip, by filename."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-analyse clips even when a valid cached result exists."),
    ] = False,
    full: Annotated[
        bool,
        typer.Option("--full", help="Print the whole document instead of a digest."),
    ] = False,
) -> None:
    """Detect scenes in the raw footage and measure each one.

    This is the slowest command in AIVE: it decodes every clip. Results are cached per
    clip, so adding one new clip re-analyses only that clip.
    """
    paths = ProjectPaths.for_root(project)
    if not paths.exists():
        emit_error(
            "project.not_initialised",
            f"{paths.root} is not an AIVE project",
            hint=f"initialise it with: aive project init {paths.root.as_posix()}",
            exit_code=ExitCode.NOT_FOUND,
        )

    clips = list(paths.find_raw_clips())
    if clip is not None:
        clips = [path for path in clips if path.name == clip or path.stem == clip]
        if not clips:
            emit_error(
                "clip.not_found",
                f"no raw clip matching {clip!r}",
                hint=f"available: {', '.join(p.name for p in paths.find_raw_clips()) or 'none'}",
                exit_code=ExitCode.NOT_FOUND,
            )
    if not clips:
        emit_error(
            "footage.missing",
            f"no video files found in {paths.raw}",
            hint=f"put footage in {paths.raw.as_posix()} (mp4, mov, mkv, avi, webm)",
            exit_code=ExitCode.NOT_FOUND,
        )

    container = build_container(paths.root)
    analyzer = container.footage
    if analyzer is None:  # pragma: no cover - always wired for a project
        emit_error(
            "internal.not_wired",
            "video analysis is not available in this build",
            exit_code=ExitCode.INTERNAL,
        )

    cached = {} if force else _load_cached_footage(paths, analyzer.version)
    if cached:
        note(f"Reusing {len(cached)} cached clip analysis result(s); pass --force to redo")

    note(f"Analysing {len(clips)} clip(s). This decodes every file, so it can take a while.")
    try:
        footage = analyzer.analyze_project(clips, cached=cached)
    except VideoDependencyMissingError as exc:
        emit_error(
            "dependency.missing",
            str(exc).splitlines()[0],
            hint='install it with: pip install -e ".[video]"',
            exit_code=ExitCode.ENVIRONMENT,
        )
    except (ProbeError, SceneDetectionError, FrameReadError) as exc:
        emit_error(
            "footage.unreadable",
            str(exc),
            hint="check the file is real video; try re-encoding it if the codec is unusual",
            exit_code=ExitCode.INVALID_INPUT,
        )

    if not footage.scenes:
        emit_error(
            "footage.no_scenes",
            "no scenes were detected in any clip",
            hint="the footage may be zero-length or unreadable; check aive doctor",
            exit_code=ExitCode.INVALID_INPUT,
        )

    write_json_file(footage, paths.footage_analysis_file)
    _report_footage(footage, paths, settings=container.settings, full=full)


def _load_cached_footage(
    paths: ProjectPaths, expected_version: str
) -> dict[MediaRef, ClipAnalysis]:
    """Per-clip analyses that are still trustworthy.

    Invalidated per clip rather than wholesale, so adding a clip to a forty-clip project
    costs one analysis instead of forty. A clip is reused when the analyser version
    matches and the file has not been touched since the cache was written.
    """
    cache_file = paths.footage_analysis_file
    if not cache_file.is_file():
        return {}
    try:
        footage = FootageAnalysis.model_validate_json(cache_file.read_text(encoding="utf-8"))
        cache_mtime = cache_file.stat().st_mtime
    except (OSError, ValueError) as exc:
        logger.debug("Ignoring unreadable footage cache %s: %s", cache_file, exc)
        return {}

    reusable: dict[MediaRef, ClipAnalysis] = {}
    for analysis in footage.clips:
        if analysis.analyzer_version != expected_version:
            continue
        try:
            source = paths.resolve(analysis.clip)
            if not source.is_file() or source.stat().st_mtime > cache_mtime:
                continue
        except (OSError, ValueError):
            continue
        reusable[analysis.clip] = analysis
    return reusable


def _report_footage(
    footage: FootageAnalysis,
    paths: ProjectPaths,
    *,
    settings: AiveSettings,
    full: bool,
) -> None:
    suppressed = footage.suppressed_scene_keys
    floor = settings.rules.min_overall_quality
    low = [s for s in footage.scenes if s.quality.overall < floor and s.key not in suppressed]

    note("")
    note(f"  clips:      {len(footage.clips)}")
    note(f"  scenes:     {len(footage.scenes)}")
    note(f"  duplicates: {len(footage.duplicates)} group(s), {len(suppressed)} scene(s) suppressed")
    note(f"  below the quality floor ({floor:.2f}): {len(low)} scene(s)")
    note(f"  usable:     {len(footage.scenes) - len(suppressed) - len(low)}")
    note(f"  document -> {paths.footage_analysis_file}")

    if footage.failed:
        # Never silent. A run that quietly dropped three clips would read as "these are
        # all your options" and the director would plan around footage it never saw.
        note(f"\n  {len(footage.failed)} clip(s) could NOT be analysed:")
        for failure in footage.failed:
            note(f"    {failure.clip}: {failure.error}")

    tagged = [s for s in footage.scenes if s.tags.people_count]
    if not tagged:
        note("\n  Note: no faces were detected, so people_count is 0 and shot_type is")
        note("  unknown throughout. The classical provider reads only faces; that is its")
        note("  documented limit, not a failure.")

    if full:
        emit_json(footage)
        return
    emit_digest(_footage_digest(footage, paths, floor=floor))


def _footage_digest(footage: FootageAnalysis, paths: ProjectPaths, *, floor: float) -> list[str]:
    """A compact, line-oriented summary of every scene.

    Grouped by clip with an ``@`` header, so the source path is written once instead of
    on every scene line. On a 200-scene project that alone saves a few thousand tokens.
    """
    suppressed = footage.suppressed_scene_keys
    lines = [
        f"# footage={paths.footage_analysis_file.as_posix()} clips={len(footage.clips)} "
        f"scenes={len(footage.scenes)} dup_groups={len(footage.duplicates)} "
        f"suppressed={len(suppressed)} failed={len(footage.failed)} "
        f"quality_floor={floor:.2f}"
    ]
    for analysis in footage.clips:
        stream = analysis.probe.video
        geometry = (
            f"{stream.display_size[0]}x{stream.display_size[1]} {stream.fps:.2f}fps"
            if stream is not None
            else "no video"
        )
        rotation = f" rot={stream.rotation}" if stream is not None and stream.rotation else ""
        lines.append(
            f"@ {analysis.clip} {geometry}{rotation} {analysis.probe.duration:.2f}s "
            f"scenes={len(analysis.scenes)}"
        )
        for scene in analysis.scenes:
            quality = scene.quality
            motion = scene.motion
            flags = []
            if scene.key in suppressed:
                flags.append("DUP")
            elif quality.overall < floor:
                flags.append("LOW")
            marker = f" {'/'.join(flags)}" if flags else ""
            lines.append(
                f"{scene.key} {scene.range.start:.2f}-{scene.range.end:.2f} "
                f"d={scene.range.duration:.1f} q={quality.overall:.2f} "
                f"blur={quality.blur:.2f} br={quality.brightness:.2f} "
                f"ex={quality.exposure:.2f} st={quality.stability:.2f} "
                f"mot={motion.level.value} cam={motion.camera_move.value} "
                f"shot={scene.shot_type.value} ppl={_people(scene)}{marker}"
            )

    # Failures get their own `!` lines so a director scanning the digest cannot miss
    # that some footage was never offered to it.
    lines.extend(f"! {failure.clip} {failure.error}" for failure in footage.failed)

    if footage.duplicates:
        lines.append(
            "# duplicates: "
            + " ".join(
                f"{key}<-{group.representative}"
                for group in footage.duplicates
                for key in group.duplicates
            )
        )
    return lines


def _people(scene: object) -> str:
    """People count, or ``?`` when the provider did not measure it."""
    count = getattr(getattr(scene, "tags", None), "people_count", None)
    return "?" if count is None else str(count)


def _with_speech_overrides(project_root: Path, overrides: dict[str, object]) -> AiveSettings:
    """Settings with CLI speech flags applied on top of every config layer."""
    return load_settings(project_root, speech=overrides)


PROGRESS_INTERVAL = 15.0
"""Seconds between transcription progress lines.

Whisper yields a segment every few seconds of audio, which on a long file is several
per second of wall time. Printing all of them buries the rest of the log; printing
none is what made a thirty-five-minute run indistinguishable from a hang.
"""


def _transcription_progress() -> ProgressCallback:
    """A throttled stderr reporter for transcription.

    Throttling lives here rather than in the recogniser deliberately: the recogniser
    reports every segment and stays ignorant of presentation, which is what lets the
    desktop UI drive a smooth progress bar from the same callback.

    The ETA is derived from elapsed wall time rather than from a speed field, because
    faster-whisper has no equivalent of FFmpeg's ``speed``.

    The clock starts at the *first* report, not here. Loading a model takes tens of
    seconds and downloading one takes minutes, none of which is decoding: timing from
    construction charged all of it to the first percent and produced "~21m left" on a
    file that finished in 68 seconds. The first line therefore carries no estimate,
    which is the honest thing to show when nothing has been measured yet.
    """
    started: float | None = None
    last_printed = 0.0

    def report(fraction: float, detail: str) -> None:
        nonlocal started, last_printed
        now = time.monotonic()
        if started is None:
            started = now
        if fraction < 1.0 and now - last_printed < PROGRESS_INTERVAL:
            return
        last_printed = now

        elapsed = now - started
        suffix = ""
        if 0.0 < fraction < 1.0 and elapsed > 0.0:
            remaining = elapsed / fraction - elapsed
            suffix = f", ~{_duration(remaining)} left"
        note(f"  {fraction * 100:5.1f}%  {detail}{suffix}")

    return report


def _duration(seconds: float) -> str:
    """A rough human duration: ``45s``, ``12m``, ``1h20m``."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    return f"{total // 3600}h{total % 3600 // 60:02d}m"


def _keep_everything(transcript: Transcript) -> SpeechCleanupReport:
    """A cleanup report that removes nothing.

    Used by ``--no-cleanup`` so the rest of the pipeline needs no special case: a
    report keeping the whole file makes source time and timeline time coincide.
    """
    return SpeechCleanupReport(
        source=transcript.source,
        original_duration=transcript.duration,
        kept_ranges=(TimeRange(start=0.0, end=transcript.duration),),
    )


def _load_cached(
    paths: ProjectPaths, narration_path: Path, *, expected_model: str
) -> NarrationAnalysis | None:
    """A cached analysis, if one exists and is still trustworthy.

    Invalidated by a version bump, a different model, or narration audio newer than the
    cache. Transcription costs minutes, so this is worth getting right - but reusing a
    stale result would silently mis-time every subtitle, so the checks are conservative
    and any doubt re-runs.
    """
    cache_file = paths.narration_file
    if not cache_file.is_file():
        return None
    try:
        analysis = NarrationAnalysis.model_validate_json(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("Ignoring unreadable cache %s: %s", cache_file, exc)
        return None

    if analysis.analyzer_version != ANALYSIS_VERSION:
        logger.debug(
            "Cache is from analyser %s, current is %s", analysis.analyzer_version, ANALYSIS_VERSION
        )
        return None
    if not analysis.transcript.model_name.endswith(f"/{expected_model}"):
        logger.debug(
            "Cache used %s, configured model is %s", analysis.transcript.model_name, expected_model
        )
        return None
    try:
        if narration_path.stat().st_mtime > cache_file.stat().st_mtime:
            logger.debug("Narration is newer than the cache")
            return None
    except OSError:
        return None
    return analysis


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _report(analysis: NarrationAnalysis, paths: ProjectPaths, *, full: bool, reused: bool) -> None:
    cleanup = analysis.cleanup
    note("")
    note(f"  language:   {analysis.transcript.language} ({analysis.transcript.model_name})")
    note(f"  duration:   {cleanup.original_duration:.2f}s")
    note(
        f"  kept:       {cleanup.kept_duration:.2f}s "
        f"({_percent(cleanup.kept_duration, cleanup.original_duration)}) "
        f"in {len(cleanup.kept_ranges)} range(s)"
    )
    note(
        f"  removed:    {cleanup.removed_duration:.2f}s "
        f"- {len(cleanup.silences)} silence, {len(cleanup.fillers)} filler, "
        f"{len(cleanup.repetitions)} repetition"
    )
    note(f"  beats:      {len(analysis.beats)} ({len(analysis.surviving_beats)} surviving)")
    note(f"  words:      {len(analysis.transcript.words())}")
    note(f"  document -> {paths.narration_file}")
    if not analysis.transcript.has_word_timings:
        note("\n  WARNING: no word timings, so filler removal was skipped and cue timing")
        note("  is approximate. Set speech.word_timestamps = true.")
    if not reused:
        note("\nNext: aive subtitle build " + paths.root.as_posix())

    if full:
        emit_json(analysis)
        return
    emit_digest(_digest(analysis, paths))


def _digest(analysis: NarrationAnalysis, paths: ProjectPaths) -> list[str]:
    """A compact, line-oriented summary.

    One line per beat: index, source range, timeline range, keywords, then the text.
    The text is the payload the director reasons about, so it is never truncated;
    everything else is abbreviated to leave room for it.
    """
    cleanup = analysis.cleanup
    transcript = analysis.transcript
    lines = [
        f"# narration={analysis.source} lang={transcript.language} "
        f"model={transcript.model_name} dur={cleanup.original_duration:.2f} "
        f"kept={cleanup.kept_duration:.2f} beats={len(analysis.beats)} "
        f"surviving={len(analysis.surviving_beats)} "
        f"wordtimings={str(transcript.has_word_timings).lower()} "
        f"doc={paths.narration_file.as_posix()}"
    ]
    for beat in analysis.beats:
        timeline = (
            f"{beat.timeline_range.start:.2f}-{beat.timeline_range.end:.2f}"
            if beat.timeline_range is not None
            else "CUT"
        )
        keywords = ",".join(beat.keywords) or "-"
        lines.append(
            f"b{beat.index:03d} src={beat.range.start:.2f}-{beat.range.end:.2f} "
            f"tl={timeline} kw={keywords} | {beat.text}"
        )
    lines.append(
        "# cuts: "
        + " ".join(
            [
                *(
                    f"silence:{span.range.start:.2f}-{span.range.end:.2f}"
                    for span in cleanup.silences
                ),
                *(f"filler:{span.range.start:.2f}={span.text}" for span in cleanup.fillers),
                *(
                    f"retake:{span.range.start:.2f}-{span.range.end:.2f}"
                    for span in cleanup.repetitions
                ),
            ]
        )
    )
    return lines


def _percent(part: float, whole: float) -> str:
    return f"{(part / whole * 100.0):.0f}%" if whole > 0 else "n/a"


# --------------------------------------------------------------------------- #
# Phase 7 - music
# --------------------------------------------------------------------------- #


@app.command("music")
def music(
    project: Annotated[Path, typer.Argument(help="Project directory.")],
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-analyse tracks even when a valid cached result exists."),
    ] = False,
    full: Annotated[
        bool,
        typer.Option("--full", help="Print the whole library instead of a digest."),
    ] = False,
) -> None:
    """Describe the tracks in `music/` so a bed can be chosen deliberately.

    Measures tempo, energy, brightness, integrated loudness and where each intro ends,
    and infers a ranked mood pair. **Nothing here selects music** - the mood labels are a
    heuristic over three numbers, and choosing a bed against the narration's emotion is an
    editorial decision the digest informs rather than makes.

    Loudness is the one number measured exactly, by FFmpeg's EBU R128 meter, because
    ducking is a comparison and both sides must use the same ruler.
    """
    paths = ProjectPaths.for_root(project)
    if not paths.exists():
        emit_error(
            "project.not_initialised",
            f"{paths.root} is not an AIVE project",
            hint=f"initialise it with: aive project init {paths.root.as_posix()}",
            exit_code=ExitCode.NOT_FOUND,
        )

    tracks = paths.find_music()
    if not tracks:
        emit_error(
            "music.missing",
            f"no audio files found in {paths.music}",
            hint=f"put music in {paths.music.as_posix()} (mp3, wav, m4a, flac, ogg)",
            exit_code=ExitCode.NOT_FOUND,
        )

    container = build_container(paths.root)
    analyzer = container.music
    if analyzer is None:  # pragma: no cover - always wired for a project
        emit_error(
            "internal.not_wired",
            "music analysis is not available in this build",
            exit_code=ExitCode.INTERNAL,
        )

    cached = {} if force else _load_cached_music(paths, analyzer.version)
    pending = [path for path in tracks if paths.to_ref(path) not in cached]
    if cached:
        note(f"Reusing {len(cached)} cached track result(s); pass --force to redo")

    note(f"Analysing {len(pending)} track(s) of {len(tracks)}.")
    try:
        analysed, failures = analyzer.analyze_library(
            tuple((path, paths.to_ref(path)) for path in pending)
        )
    except AudioDependencyMissingError as exc:
        emit_error(
            "dependency.missing",
            str(exc).splitlines()[0],
            hint='install it with: pip install -e ".[audio]"',
            exit_code=ExitCode.ENVIRONMENT,
        )

    # Cached first, then freshly analysed, then re-sorted by name so the digest order
    # matches the folder order however the work was split.
    merged = {track.source: track for track in (*cached.values(), *analysed.tracks)}
    library = MusicLibrary(
        tracks=tuple(sorted(merged.values(), key=lambda track: str(track.source)))
    )

    if not library.tracks:
        emit_error(
            "music.unreadable",
            f"none of the {len(tracks)} file(s) in {paths.music.name} could be analysed",
            hint="check they are real audio files; run aive doctor to verify ffmpeg",
            exit_code=ExitCode.INVALID_INPUT,
            details={"failed": [failure.path.name for failure in failures]},
        )

    write_json_file(library, paths.music_library_file)
    _report_music(library, failures, paths, full=full)


def _load_cached_music(paths: ProjectPaths, expected_version: str) -> dict[MediaRef, MusicTrack]:
    """Track analyses that are still trustworthy.

    Per track rather than wholesale, for the same reason as footage: adding one file to a
    twenty-track library should cost one analysis. A track is reused when the analyser
    version matches and the file has not been touched since the cache was written.
    """
    cache_file = paths.music_library_file
    if not cache_file.is_file():
        return {}
    try:
        library = MusicLibrary.model_validate_json(cache_file.read_text(encoding="utf-8"))
        cache_mtime = cache_file.stat().st_mtime
    except (OSError, ValueError) as exc:
        logger.debug("Ignoring unreadable music cache %s: %s", cache_file, exc)
        return {}

    reusable: dict[MediaRef, MusicTrack] = {}
    for track in library.tracks:
        if track.analyzer_version != expected_version:
            continue
        try:
            source = paths.resolve(track.source)
            if not source.is_file() or source.stat().st_mtime > cache_mtime:
                continue
        except (OSError, ValueError):
            continue
        reusable[track.source] = track
    return reusable


def _report_music(
    library: MusicLibrary,
    failures: tuple[TrackFailure, ...],
    paths: ProjectPaths,
    *,
    full: bool,
) -> None:
    note(f"Analysed {len(library.tracks)} track(s), {library.total_duration:.0f}s of music")
    for failure in failures:
        note(f"  ! {failure.path.name}: {failure.error}")
    note("")
    note("  Mood labels are a heuristic over tempo, level and brightness - a starting")
    note("  point, not a verdict. Match the music to the narration's emotion, not the")
    note("  footage's.")

    if full:
        emit_json(library)
        return
    emit_digest(_music_digest(library, failures, paths))


def _music_digest(
    library: MusicLibrary,
    failures: tuple[TrackFailure, ...],
    paths: ProjectPaths,
) -> list[str]:
    """One line per track.

    ``lufs`` is the field that matters for placing a bed: two tracks at the same gain but
    six LUFS apart will sit completely differently under the same narration.
    """
    lines = [
        f"# music={paths.music_library_file.as_posix()} tracks={len(library.tracks)} "
        f"failed={len(failures)} total={library.total_duration:.1f}"
    ]
    for track in library.tracks:
        lines.append(
            f"{track.source} d={track.duration:.1f} "
            f"bpm={f'{track.bpm:.0f}' if track.bpm is not None else '?'} "
            f"energy={track.energy:.2f} "
            f"lufs={f'{track.loudness_lufs:.1f}' if track.loudness_lufs is not None else '?'} "
            f"intro={f'{track.intro_end:.1f}' if track.intro_end is not None else '-'} "
            f"mood={','.join(mood.value for mood in track.moods) or '-'} "
            f"tags={','.join(track.tags) or '-'}"
        )
    lines.extend(f"! {failure.path.name} {failure.error}" for failure in failures)
    return lines


__all__ = ["ANALYSIS_VERSION", "app"]
