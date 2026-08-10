"""``aive narrate`` - a written script becomes the narration track.

The alternative entry point to the pipeline. Normally a user records themselves and AIVE
transcribes it; here they write the words and AIVE speaks them.

**This replaces `analyze audio` rather than feeding it.** Because the text is the input and
the backend reports word offsets, there is nothing left for a recogniser to discover: the
command writes both ``narration.wav`` and ``.aive/narration.json``, so the next step is
`plan brief`. No 1.5 GB model, no transcription pass, and exact timings instead of estimated
ones.

One thing the command is deliberately loud about: the `edge` backend **calls the network**,
which nothing else in AIVE does. It is not the default, and choosing it prints a line saying
so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from app.analysis.speech.synth.base import (
    SpeechSynthesizer,
    SynthesisDependencyMissingError,
    SynthesisError,
    Voice,
    VoiceNotFoundError,
)
from app.analysis.speech.synth.narrator import narrate
from app.analysis.speech.synth.script import ScriptError, ScriptLine, load_script
from app.cli.output import ExitCode, emit_digest, emit_error, emit_json, note, write_json_file
from app.config.settings import AiveSettings, load_settings
from app.models.speech import NarrationAnalysis
from app.services.ffmpeg_locator import FFmpegLocator
from app.services.paths import ProjectPaths
from app.utils.logging import get_logger

logger = get_logger(__name__)


def narrate_command(
    project: Annotated[Path, typer.Argument(help="Project directory.")],
    script: Annotated[
        Path | None,
        typer.Option("--script", "-s", help="Script file. Defaults to <project>/script.txt."),
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="sapi (offline) or edge (network, Vietnamese voices)."),
    ] = None,
    voice: Annotated[
        str | None, typer.Option("--voice", help="Voice name. Omit to pick one for --language.")
    ] = None,
    language: Annotated[
        str | None, typer.Option("--language", help="Script language, e.g. vi or en.")
    ] = None,
    rate: Annotated[
        float | None, typer.Option("--rate", help="Speaking rate multiplier. 1.0 is natural.")
    ] = None,
    gap: Annotated[
        float | None, typer.Option("--gap", help="Silence after each line, in seconds.")
    ] = None,
    list_voices: Annotated[
        bool, typer.Option("--list-voices", help="List the backend's voices and stop.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the parsed beats and stop. Speaks nothing.")
    ] = False,
    full: Annotated[bool, typer.Option("--full", help="Emit the analysis as JSON.")] = False,
) -> None:
    """Speak a script and write it as the project's narration.

    The script is plain text: one paragraph per beat, blank line between beats, lines
    starting with `#` ignored. A beat is what footage gets matched against, so paragraph
    breaks are editorial decisions - put one where you want a cut to be possible.

    Writes `narration.wav` and `.aive/narration.json`. Go straight to `aive plan brief`
    afterwards; `analyze audio` is not needed and would only re-derive what is already known.
    """
    paths = ProjectPaths.for_root(project)
    settings = load_settings(paths.root)
    options = _resolve(
        settings, backend=backend, voice=voice, language=language, rate=rate, gap=gap
    )

    synthesizer = _build_backend(options.backend)
    if list_voices:
        _report_voices(synthesizer, options.language)
        return

    if not paths.exists():
        emit_error(
            "project.not_initialised",
            f"{paths.root} is not an AIVE project",
            hint=f"initialise it with: aive project init {paths.root.as_posix()}",
            exit_code=ExitCode.NOT_FOUND,
        )

    script_path = script or paths.root / "script.txt"
    try:
        lines = load_script(script_path, max_words_per_line=settings.tts.max_words_per_line)
    except ScriptError as exc:
        emit_error(
            "script.unusable",
            str(exc),
            hint=(
                "write one paragraph per beat, with a blank line between beats. "
                f"Expected at {script_path.as_posix()}"
            ),
            exit_code=ExitCode.NOT_FOUND if "no script at" in str(exc) else ExitCode.INVALID_INPUT,
        )

    if dry_run:
        _report_dry_run(lines, options, script_path)
        return

    chosen_voice = options.voice or _default_voice(synthesizer, options.language)
    if synthesizer.requires_network:
        note(
            f"The {options.backend!r} backend calls the network. Nothing else in AIVE does; "
            "use --backend sapi to stay offline."
        )
    note(f"Speaking {len(lines)} beat(s) as {chosen_voice} at {options.rate:g}x")

    destination = paths.root / "narration.wav"
    try:
        result = narrate(
            lines,
            synthesizer=synthesizer,
            voice=chosen_voice,
            destination=destination,
            locator=FFmpegLocator(settings.media),
            language=options.language,
            rate=options.rate,
            gap=options.gap,
        )
    except VoiceNotFoundError as exc:
        emit_error(
            "voice.not_found",
            str(exc),
            hint="run: aive narrate . --list-voices",
            exit_code=ExitCode.INVALID_INPUT,
        )
    except SynthesisDependencyMissingError as exc:
        emit_error(
            "dependency.missing",
            str(exc),
            hint='install it with: pip install -e ".[tts]"',
            exit_code=ExitCode.ENVIRONMENT,
        )
    except SynthesisError as exc:
        emit_error(
            "narrate.failed",
            str(exc),
            hint="check the voice name and, for the edge backend, your connection",
            exit_code=ExitCode.INVALID_INPUT,
        )

    write_json_file(result.analysis, paths.narration_file)
    _report(result.analysis, result, paths, script_path, full=full)


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Options:
    """Resolved settings for one run. CLI flags win over config, config over defaults."""

    backend: str
    voice: str | None
    language: str
    rate: float
    gap: float


def _resolve(
    settings: AiveSettings,
    *,
    backend: str | None,
    voice: str | None,
    language: str | None,
    rate: float | None,
    gap: float | None,
) -> _Options:
    tts = settings.tts
    return _Options(
        backend=(backend or tts.backend).lower(),
        voice=voice or tts.voice,
        language=language or tts.language,
        rate=rate if rate is not None else tts.rate,
        gap=gap if gap is not None else tts.gap,
    )


def _build_backend(name: str) -> SpeechSynthesizer:
    """Construct a backend by name.

    Imported here rather than at module scope so that ``aive narrate --help`` works without
    the optional ``edge-tts`` package installed.
    """
    if name == "sapi":
        from app.analysis.speech.synth.sapi import SapiSynthesizer

        return SapiSynthesizer()
    if name == "edge":
        from app.analysis.speech.synth.edge import EdgeSynthesizer

        return EdgeSynthesizer()

    emit_error(
        "backend.unknown",
        f"unknown TTS backend {name!r}",
        hint="available: sapi (offline), edge (network, Vietnamese voices)",
        exit_code=ExitCode.USAGE,
    )
    raise AssertionError  # pragma: no cover - emit_error exits


def _default_voice(synthesizer: SpeechSynthesizer, language: str) -> str:
    """Ask the backend for a voice, turning its refusal into a structured error."""
    try:
        # Both backends expose this; it is not on the protocol because "pick something
        # sensible" is a convenience rather than part of the contract.
        return str(synthesizer.default_voice(language))  # type: ignore[attr-defined]
    except VoiceNotFoundError as exc:
        emit_error(
            "voice.not_found",
            str(exc),
            hint="run: aive narrate . --list-voices",
            exit_code=ExitCode.INVALID_INPUT,
        )
        raise AssertionError from None  # pragma: no cover - emit_error exits


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _report_voices(synthesizer: SpeechSynthesizer, language: str) -> None:
    try:
        voices = synthesizer.voices()
    except SynthesisDependencyMissingError as exc:
        emit_error(
            "dependency.missing",
            str(exc),
            hint='install it with: pip install -e ".[tts]"',
            exit_code=ExitCode.ENVIRONMENT,
        )
    except SynthesisError as exc:
        emit_error("voices.unavailable", str(exc), exit_code=ExitCode.ENVIRONMENT)

    wanted = language.split("-")[0].lower()
    matching = [item for item in voices if item.language == wanted]
    note(f"{len(voices)} voice(s) from {synthesizer.name!r}, {len(matching)} for {wanted!r}")
    if not matching and synthesizer.name == "sapi":
        note("")
        note("  No voice for this language is installed in Windows. Add one under")
        note("  Settings > Time & Language > Speech > Add voices, or use --backend edge.")

    emit_digest(
        [
            f"# voices backend={synthesizer.name} total={len(voices)} "
            f"language={wanted} matching={len(matching)} "
            f"network={str(synthesizer.requires_network).lower()}",
            *(
                f"{'*' if item.language == wanted else ' '} {item.name} {item.locale} "
                f"{item.gender or '-'}"
                for item in _ordered(voices, wanted)
            ),
        ]
    )


def _ordered(voices: tuple[Voice, ...], language: str) -> list[Voice]:
    """Matching voices first: with 322 to choose from, the relevant ones must not scroll off."""
    return sorted(voices, key=lambda item: (item.language != language, item.locale, item.name))


def _report_dry_run(lines: tuple[ScriptLine, ...], options: _Options, script_path: Path) -> None:
    note(f"{len(lines)} beat(s) parsed from {script_path.name}. Nothing was spoken.")
    emit_digest(
        [
            f"# dry_run script={script_path.as_posix()} beats={len(lines)} "
            f"words={sum(item.word_count for item in lines)} "
            f"backend={options.backend} language={options.language}",
            *(f"b{item.index:03d} words={item.word_count:3} | {item.text}" for item in lines),
        ]
    )


def _report(
    analysis: NarrationAnalysis,
    result: object,
    paths: ProjectPaths,
    script_path: Path,
    *,
    full: bool,
) -> None:
    from app.analysis.speech.synth.narrator import NarrationResult

    assert isinstance(result, NarrationResult)
    transcript = analysis.transcript

    note("")
    note(f"  audio:    {result.audio}")
    note(f"  duration: {result.duration:.2f}s across {len(analysis.beats)} beat(s)")
    note(f"  analysis: {paths.narration_file}")
    note("")
    note("  Next: aive analyze video, then aive plan brief.")
    note("  `analyze audio` is NOT needed - the timings above are exact, not recognised.")

    if full:
        emit_json(analysis)
        return

    emit_digest(
        [
            f"# narration={analysis.source} script={script_path.as_posix()} "
            f"backend={result.backend} lang={transcript.language} "
            f"dur={result.duration:.2f} beats={len(analysis.beats)} "
            f"wordtimings={str(transcript.has_word_timings).lower()} "
            f"doc={paths.narration_file.as_posix()}",
            *(
                f"b{beat.index:03d} src={beat.range.start:.2f}-{beat.range.end:.2f} "
                f"tl={beat.timeline_range.start:.2f}-{beat.timeline_range.end:.2f}"
                if beat.timeline_range is not None
                else f"b{beat.index:03d} tl=CUT"
                for beat in analysis.beats
            ),
        ]
    )


__all__ = ["narrate_command"]
