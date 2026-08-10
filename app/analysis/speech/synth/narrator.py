"""Script in, narration out.

This is where synthesis earns its place in the pipeline, and the reason is worth stating
before the code: **a synthesised narration needs no recogniser.**

The record-then-transcribe path exists because a human recording is opaque — AIVE has to run
Whisper to find out what was said, guess at word boundaries, then detect and cut the "um"s,
the pauses and the retakes. None of that applies here. The text is the input. There are no
fillers, no retakes, and no dead air. The service reports word offsets directly.

So this produces the same :class:`~app.models.speech.NarrationAnalysis` document that
``aive analyze audio`` produces, with three differences that all favour it:

* **No 1.5 GB model download**, and no minutes of transcription.
* **Exact timings** rather than a recogniser's estimate.
* **Exact text** — a recogniser mishears proper nouns; this cannot.

Timeline time and source time are therefore identical, because nothing was cut. That is the
one place a reader of the rest of this codebase should slow down: everywhere else those two
clocks diverge, and the divergence is the single easiest thing to get wrong. Here it is
genuinely absent, and :func:`narrate` states it in the cleanup report rather than leaving it
implied.

Beats come from the existing :class:`~app.analysis.speech.beats.NarrationBeatBuilder`, not
from a second implementation. A script line is already a beat, so it would have been easy to
build them here — and then keyword extraction would drift from the recorded path, and the
director would get different candidate rankings for the same words depending on how the audio
was made.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.analysis.speech.beats import BEATS_VERSION, NarrationBeatBuilder
from app.analysis.speech.synth.base import (
    SpeechSynthesizer,
    SynthesisError,
    SynthesizedLine,
    WordTiming,
)
from app.analysis.speech.synth.script import SCRIPT_VERSION, ScriptLine
from app.models.common import MediaRef, TimeRange
from app.models.speech import (
    NarrationAnalysis,
    SpeechCleanupReport,
    Transcript,
    TranscriptSegment,
    Word,
)
from app.services.ffmpeg_locator import FFmpegLocator, FFmpegNotFoundError
from app.utils.logging import get_logger, stage

logger = get_logger(__name__)


def narrator_version(backend: str) -> str:
    """Stamped into the analysis so a cached document from another backend is not trusted.

    The backend is part of it because voices differ in pace: re-narrating the same script
    with a different voice produces different beat timings, and a plan built against the old
    ones would drift.
    """
    return f"narrate/1+{backend}+{SCRIPT_VERSION}+{BEATS_VERSION}"


@dataclass(frozen=True, slots=True)
class NarrationResult:
    """What :func:`narrate` produced."""

    audio: Path
    analysis: NarrationAnalysis
    lines: tuple[SynthesizedLine, ...]
    backend: str

    @property
    def duration(self) -> float:
        return self.analysis.cleanup.original_duration


def narrate(
    lines: tuple[ScriptLine, ...],
    *,
    synthesizer: SpeechSynthesizer,
    voice: str,
    destination: Path,
    locator: FFmpegLocator,
    language: str = "vi",
    rate: float = 1.0,
    gap: float = 0.35,
    work_dir: Path | None = None,
) -> NarrationResult:
    """Speak every line, join them, and build the narration analysis.

    Args:
        lines: Parsed script beats.
        synthesizer: Backend to speak with.
        voice: Voice name the backend understands.
        destination: Where the joined narration goes, e.g. ``project/narration.wav``.
        locator: For the FFmpeg used to measure and join.
        language: Recorded on the transcript. Also what a caller used to pick ``voice``.
        rate: Speaking rate multiplier.
        gap: Silence inserted after each line, in seconds. A script read with no pause
            between sentences sounds hurried, and the pause is also where a cut can land.
        work_dir: Where per-line audio goes. Defaults to a folder beside ``destination``.

    Raises:
        SynthesisError: a line could not be spoken, or the parts could not be joined.
    """
    if not lines:
        msg = "no script lines to narrate"
        raise SynthesisError(msg)

    staging = work_dir or destination.parent / ".aive" / "narration_parts"
    staging.mkdir(parents=True, exist_ok=True)

    spoken: list[SynthesizedLine] = []
    with stage(logger, f"Speaking {len(lines)} line(s) with {synthesizer.name}"):
        for line in lines:
            part = staging / f"line_{line.index:04d}.wav"
            result = synthesizer.speak(line.text, voice=voice, destination=part, rate=rate)
            # Always measured, never trusted from the backend. SAPI reports nothing, and a
            # backend that does report a duration is describing what it *sent*, not what
            # landed in the file after container overhead.
            measured = _measure(part, locator)
            spoken.append(
                SynthesizedLine(
                    index=line.index,
                    text=line.text,
                    audio=part,
                    duration=measured,
                    words=result.words,
                )
            )
            logger.debug(
                "Line %d: %.2fs, %d word timing(s)", line.index, measured, len(result.words)
            )

    _join(spoken, destination=destination, gap=gap, locator=locator)
    total = _measure(destination, locator)

    analysis = _build_analysis(
        spoken,
        audio=destination,
        total=total,
        gap=gap,
        language=language,
        backend=synthesizer.name,
    )
    logger.info(
        "Narrated %d line(s) into %.2fs of audio at %s",
        len(spoken),
        total,
        destination.name,
    )
    return NarrationResult(
        audio=destination, analysis=analysis, lines=tuple(spoken), backend=synthesizer.name
    )


# --------------------------------------------------------------------------- #
# Joining and measuring
# --------------------------------------------------------------------------- #


def _join(
    lines: list[SynthesizedLine], *, destination: Path, gap: float, locator: FFmpegLocator
) -> None:
    """Concatenate the parts with ``gap`` seconds of silence after each.

    ``apad`` per input rather than separate silence files: it appends exactly the padding
    asked for inside the same filter graph, so there is nothing to clean up and no chance of
    a stray silence file being left behind and concatenated twice.
    """
    try:
        tools = locator.locate()
    except FFmpegNotFoundError as exc:
        msg = f"joining narration needs ffmpeg: {exc}"
        raise SynthesisError(msg) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [str(tools.ffmpeg.path), "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for line in lines:
        command += ["-i", str(line.audio)]

    chains = [
        # Mono 48 kHz throughout: the parts may come back at whatever rate the voice uses,
        # and concat refuses inputs whose formats differ.
        f"[{index}:a]aresample=48000,aformat=channel_layouts=mono"
        + (f",apad=pad_dur={gap:.4f}" if gap > 0.0 else "")
        + f"[a{index}]"
        for index in range(len(lines))
    ]
    joined = "".join(f"[a{index}]" for index in range(len(lines)))
    graph = ";".join([*chains, f"{joined}concat=n={len(lines)}:v=0:a=1[out]"])

    command += ["-filter_complex", graph, "-map", "[out]", str(destination)]

    import subprocess

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        msg = f"could not join narration parts: {exc}"
        raise SynthesisError(msg) from exc

    if completed.returncode != 0:
        msg = f"ffmpeg failed joining narration: {completed.stderr.strip()[-300:]}"
        raise SynthesisError(msg)


def _measure(path: Path, locator: FFmpegLocator) -> float:
    """Real duration of an audio file.

    PyAV first because it needs no subprocess, then FFmpeg. The vendored FFmpeg wheel ships
    no ``ffprobe``, which is why neither path may assume it exists.
    """
    try:
        import av

        with av.open(str(path)) as container:
            if container.duration:
                return float(container.duration) / 1_000_000.0
            if container.streams.audio:
                audio = container.streams.audio[0]
                if audio.duration and audio.time_base:
                    return float(audio.duration * audio.time_base)
    except Exception as exc:
        logger.debug("PyAV could not measure %s: %s", path.name, exc)

    return _measure_with_ffmpeg(path, locator)


def _measure_with_ffmpeg(path: Path, locator: FFmpegLocator) -> float:
    """Duration by decoding to null and reading the reported output time."""
    import re
    import subprocess

    try:
        tools = locator.locate()
    except FFmpegNotFoundError as exc:
        msg = f"cannot measure {path.name} without ffmpeg: {exc}"
        raise SynthesisError(msg) from exc

    completed = subprocess.run(
        [str(tools.ffmpeg.path), "-nostdin", "-hide_banner", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120.0,
        check=False,
    )
    match = re.search(r"time=(\d+):(\d+):([\d.]+)", completed.stderr)
    if match is None:
        msg = f"could not determine the duration of {path.name}"
        raise SynthesisError(msg)
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


# --------------------------------------------------------------------------- #
# Building the analysis
# --------------------------------------------------------------------------- #


def _shift_words(words: tuple[WordTiming, ...], *, offset: float, limit: float) -> tuple[Word, ...]:
    """Move word offsets onto the joined file's clock, dropping any that collapse.

    Backends report offsets relative to their own line, so they need shifting. They are also
    clamped to the line's end, because a service occasionally reports a final word running a
    few milliseconds past the audio it actually produced, and a ``Word`` outside its segment
    fails validation.

    The clamp is why the emptiness check has to happen *after* it. Testing the unclamped
    values - as an earlier version did - passes a word whose clamped start already equals the
    segment end, and ``Word`` requires ``end > start``: the very last word of a line was
    enough to fail the whole narration.
    """
    shifted: list[Word] = []
    for item in words:
        start = min(offset + item.start, limit)
        end = min(offset + item.end, limit)
        if end <= start:
            continue
        shifted.append(Word(text=item.text, start=start, end=end))
    return tuple(shifted)


def _build_analysis(
    lines: list[SynthesizedLine],
    *,
    audio: Path,
    total: float,
    gap: float,
    language: str,
    backend: str,
) -> NarrationAnalysis:
    """Assemble the document the rest of the pipeline reads."""
    ref = MediaRef(path=audio.name)

    segments: list[TranscriptSegment] = []
    cursor = 0.0
    for line in lines:
        end = cursor + line.duration
        segments.append(
            TranscriptSegment(
                index=line.index,
                range=TimeRange(start=cursor, end=end),
                text=line.text,
                words=_shift_words(line.words, offset=cursor, limit=end),
            )
        )
        cursor = end + gap

    transcript = Transcript(
        source=ref,
        language=language,
        duration=total,
        model_name=f"synth/{backend}",
        segments=tuple(segments),
    )

    cleanup = SpeechCleanupReport(
        source=ref,
        original_duration=total,
        # One range covering everything. Nothing was removed, so timeline time *equals*
        # source time - the only place in AIVE where those two clocks coincide, and worth
        # being explicit about rather than leaving a reader to infer it from an empty list of
        # silences.
        kept_ranges=(TimeRange(start=0.0, end=total),),
    )

    return NarrationAnalysis(
        source=ref,
        transcript=transcript,
        cleanup=cleanup,
        # The shared builder, not a local reimplementation: keyword extraction has to match
        # the recorded path or the same words would rank footage differently.
        beats=NarrationBeatBuilder().build(transcript, cleanup),
        analyzer_version=narrator_version(backend),
        analyzed_at=datetime.now(UTC),
    )


def clean_parts(work_dir: Path) -> None:
    """Remove the per-line audio.

    Optional, and off by default in the CLI: when a voice mispronounces one word, having the
    individual lines on disk means re-speaking that line rather than the whole script.
    """
    if work_dir.is_dir():
        shutil.rmtree(work_dir, ignore_errors=True)


__all__ = ["NarrationResult", "clean_parts", "narrate", "narrator_version"]
