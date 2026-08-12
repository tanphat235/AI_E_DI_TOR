"""Speech recognition via faster-whisper.

The recogniser is the one component that loads a large model, so it is constructed
lazily: importing this module must not pull ``ctranslate2`` into memory, and building
a :class:`FasterWhisperRecognizer` must not download 3 GB of weights. Both happen on
the first call to :meth:`~FasterWhisperRecognizer.transcribe`.

Two behaviours here matter more than the transcription itself.

**Hallucination filtering.** Whisper does not fail silently on non-speech audio; it
invents plausible text. Over room tone it produces "Thank you for watching", over
music it produces song lyrics. Left in, that text becomes a subtitle and a narration
beat the director tries to find footage for. Three defences are applied: the Silero
VAD pre-pass, a ``no_speech_prob`` ceiling, and a ``compression_ratio`` ceiling that
catches the degenerate repetition loop.

**Word timings are load-bearing.** They are what make karaoke subtitles possible and,
more importantly, what make filler-word removal safe: without them, excising "um"
means cutting the whole segment and losing the words around it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.analysis.speech.base import ProgressCallback
from app.config.settings import SpeechSettings, WhisperDevice
from app.models.common import MediaRef, TimeRange
from app.models.speech import Transcript, TranscriptSegment, Word
from app.utils.logging import get_logger, stage

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from faster_whisper import WhisperModel

logger = get_logger(__name__)

RECOGNIZER_VERSION = "faster-whisper/1"
"""Bumped when a change here would alter output for identical input."""

_PROGRESS_CEILING = 0.999
"""Cap on reported progress before the generator is exhausted.

The same reasoning as the renderer's: a bar that reads 100% while the model is still
decoding is worse than one that sits at 99.9%. Whisper's final segment can also end
slightly past ``info.duration``, which would otherwise report above 100%.
"""


class SpeechDependencyMissingError(RuntimeError):
    """Raised when ``faster-whisper`` is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "faster-whisper is not installed, so speech recognition is unavailable.\n"
            'Install it with: pip install -e ".[speech]"'
        )


_CUDA_RUNTIME_LIBRARIES: tuple[str, ...] = ("cublas64_12", "cudnn_ops64_9")
"""CUDA libraries ctranslate2 loads lazily, at the first inference rather than at load.

Their absence is the single most common GPU failure, and the reason
:meth:`FasterWhisperRecognizer._cuda_available` cannot simply trust a device count:
``get_cuda_device_count()`` reports that a *GPU* exists, not that the CUDA *runtime* is
installed. A machine with an NVIDIA card and no CUDA toolkit - the default state of
most Windows machines - counts one device and then dies mid-transcription.
"""

_LIBRARY_LOAD_MARKERS: tuple[str, ...] = (
    "is not found or cannot be loaded",
    "cublas",
    "cudnn",
    "cuda",
    "no kernel image is available",
)
"""Substrings identifying a CUDA-environment failure rather than a real bug.

Matched on the message because ctranslate2 raises a bare ``RuntimeError`` for all of
them, so there is no exception type to catch.
"""


class FasterWhisperRecognizer:
    """A :class:`~app.analysis.speech.base.SpeechRecognizer` backed by faster-whisper."""

    def __init__(self, settings: SpeechSettings) -> None:
        self._settings = settings
        self._model: WhisperModel | None = None
        self._forced_device: str | None = None
        """Set to "cpu" after a GPU failure, so the retry and every later call stay on
        the CPU instead of rediscovering the same broken environment."""

    @property
    def name(self) -> str:
        """Recorded in ``Transcript.model_name``, e.g. ``faster-whisper/medium``."""
        return f"faster-whisper/{self._settings.model}"

    # -- Model lifecycle ---------------------------------------------------- #

    def _load_model(self) -> WhisperModel:
        """Load the model, once.

        Cached on the instance rather than globally: a long-lived process such as the
        desktop UI should pay the load cost once, but two recognisers configured with
        different model sizes must not share state.
        """
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise SpeechDependencyMissingError from exc

        device, compute_type = self._resolve_device()
        logger.info(
            "Loading %s on %s (%s)",
            self._settings.model,
            device,
            compute_type,
            extra={"model": self._settings.model, "device": device},
        )
        kwargs: dict[str, Any] = {
            "device": device,
            "compute_type": compute_type,
        }
        if self._settings.cpu_threads > 0:
            kwargs["cpu_threads"] = self._settings.cpu_threads
        if self._settings.download_root is not None:
            kwargs["download_root"] = str(self._settings.download_root)

        self._model = WhisperModel(self._settings.model, **kwargs)
        return self._model

    def _resolve_device(self) -> tuple[str, str]:
        """Pick a device and a matching compute type.

        ``auto`` is resolved here rather than deferred to ctranslate2 because the two
        settings are coupled: ``float16`` is a large speed win on a GPU and either
        unsupported or actively slower on a CPU. Getting that pairing wrong is the
        most common cause of "why is this taking twenty minutes".
        """
        device = self._settings.device
        compute_type = self._settings.compute_type

        if self._forced_device is not None:
            resolved_device = self._forced_device
        elif device is WhisperDevice.AUTO:
            resolved_device = "cuda" if self._cuda_available() else "cpu"
        else:
            resolved_device = device.value

        if compute_type == "auto":
            resolved_compute = "float16" if resolved_device == "cuda" else "int8"
        else:
            resolved_compute = compute_type

        return resolved_device, resolved_compute

    @staticmethod
    def _cuda_available() -> bool:
        """Whether CUDA is genuinely usable, not merely present.

        Two separate questions, and asking only the first is a trap. A device count
        above zero means a GPU exists; it says nothing about whether cuBLAS and cuDNN
        are installed, and ctranslate2 loads those lazily at the first inference. So a
        machine with an NVIDIA card and no CUDA toolkit passes a naive check and then
        fails ten seconds into a transcription.

        Asked of ctranslate2 rather than torch, because torch is not a dependency and
        adding gigabytes to answer one question would be absurd.
        """
        try:
            import ctranslate2
        except ImportError:  # pragma: no cover - ctranslate2 ships with faster-whisper
            return False
        try:
            if ctranslate2.get_cuda_device_count() <= 0:
                return False
        except Exception:
            return False

        missing = [name for name in _CUDA_RUNTIME_LIBRARIES if not _can_load_library(name)]
        if missing:
            logger.info(
                "A CUDA device is present but %s could not be loaded; using the CPU. "
                "Install the CUDA 12 runtime and cuDNN 9 to enable GPU transcription.",
                ", ".join(missing),
                extra={"missing_libraries": ",".join(missing)},
            )
            return False
        return True

    # -- Transcription ------------------------------------------------------ #

    def transcribe(
        self,
        audio: Path,
        *,
        ref: MediaRef,
        on_progress: ProgressCallback | None = None,
    ) -> Transcript:
        """Recognise speech in ``audio``.

        Args:
            audio: Absolute path to read.
            ref: Project-relative reference recorded in the result, so the transcript
                stays portable while the read stays absolute.
            on_progress: Called with ``(fraction, detail)`` as each segment arrives.
                Progress is measured in *audio position*, not segments decoded: the
                segment count is unknown until the end, but the duration is known up
                front, which is what makes a percentage meaningful at all.
        """
        if not audio.is_file():
            msg = f"narration audio not found: {audio}"
            raise FileNotFoundError(msg)

        with stage(logger, f"Transcribing {ref}", audio=str(audio)):
            segments, info = self._run(audio, on_progress=on_progress)

        kept, dropped = self._filter_hallucinations(segments)
        if dropped:
            logger.warning(
                "Dropped %d likely-hallucinated segment(s)",
                len(dropped),
                extra={"dropped": len(dropped)},
            )

        # `info.duration` is the duration ctranslate2 actually decoded, which is the
        # honest denominator for cleanup accounting.
        duration = float(getattr(info, "duration", 0.0))
        if duration <= 0.0:
            duration = max((segment.range.end for segment in kept), default=0.0)
        if duration <= 0.0:
            msg = f"could not determine a duration for {audio}; the file may be empty"
            raise ValueError(msg)

        return Transcript(
            source=ref,
            language=str(getattr(info, "language", None) or self._settings.language or "unknown"),
            language_probability=_optional_score(getattr(info, "language_probability", None)),
            duration=duration,
            model_name=self.name,
            segments=tuple(
                # Re-index so numbering stays contiguous after dropped segments.
                segment.model_copy(update={"index": position})
                for position, segment in enumerate(kept)
            ),
        )

    def _run(
        self, audio: Path, *, on_progress: ProgressCallback | None = None
    ) -> tuple[list[TranscriptSegment], Any]:
        """Transcribe, falling back to the CPU if the GPU turns out to be unusable.

        The library check in :meth:`_cuda_available` catches the common case up front,
        but not every one: a driver too old for the toolkit, an out-of-memory GPU, or a
        card whose compute capability ctranslate2 has no kernel for all fail only once
        inference starts. Retrying on the CPU turns each of those from a lost run into a
        slower one.

        The retry happens once and only when the device was CUDA, so a genuine bug on
        the CPU path still surfaces as an error rather than looping.
        """
        try:
            return self._attempt(audio, on_progress=on_progress)
        except RuntimeError as exc:
            device, _ = self._resolve_device()
            if device != "cuda" or not _is_cuda_environment_failure(exc):
                raise
            logger.warning(
                "GPU transcription failed (%s); retrying on the CPU. "
                "This is a CUDA installation problem, not a problem with your audio.",
                exc,
            )
            # Drop the GPU model so the retry rebuilds on the CPU.
            self._model = None
            self._forced_device = "cpu"
            return self._attempt(audio, on_progress=on_progress)

    def _attempt(
        self, audio: Path, *, on_progress: ProgressCallback | None = None
    ) -> tuple[list[TranscriptSegment], Any]:
        model = self._load_model()
        settings = self._settings
        raw_segments, info = model.transcribe(
            str(audio),
            language=settings.language,
            beam_size=settings.beam_size,
            word_timestamps=settings.word_timestamps,
            vad_filter=settings.vad_filter,
            condition_on_previous_text=settings.condition_on_previous_text,
        )
        # `info` is returned eagerly - language detection and the VAD pre-pass have
        # already run - so its duration is available as a denominator before a single
        # segment has been decoded. That is what makes a percentage possible here.
        total = float(getattr(info, "duration", 0.0) or 0.0)
        # faster-whisper returns a lazy generator: recognition runs as it is consumed,
        # so any device failure surfaces here rather than at the call above.
        segments = self._collect_segments(raw_segments, total=total, on_progress=on_progress)
        if on_progress is not None:
            on_progress(1.0, f"{_timecode(total)} of {_timecode(total)}")
        return segments, info

    def _collect_segments(
        self,
        raw_segments: Any,
        *,
        total: float = 0.0,
        on_progress: ProgressCallback | None = None,
    ) -> list[TranscriptSegment]:
        """Convert faster-whisper segments into our models.

        This loop is where recognition actually happens - the generator decodes on
        demand - which makes it the only place that can report progress.
        """
        collected: list[TranscriptSegment] = []
        for index, raw in enumerate(raw_segments):
            text = str(raw.text).strip()
            start, end = float(raw.start), float(raw.end)
            if on_progress is not None and total > 0.0:
                # Reported before the skip below: a discarded artefact still cost the
                # time to decode, so it is honest progress.
                fraction = min(_PROGRESS_CEILING, max(0.0, end / total))
                on_progress(fraction, f"{_timecode(end)} of {_timecode(total)}")
            if not text or end <= start:
                # Zero-length or empty segments are artefacts, not speech.
                continue
            collected.append(
                TranscriptSegment(
                    index=index,
                    range=TimeRange(start=start, end=end),
                    text=text,
                    words=self._collect_words(raw),
                    avg_logprob=_optional_float(getattr(raw, "avg_logprob", None)),
                    no_speech_prob=_optional_score(getattr(raw, "no_speech_prob", None)),
                    compression_ratio=_optional_float(getattr(raw, "compression_ratio", None)),
                )
            )
        return collected

    @staticmethod
    def _collect_words(raw_segment: Any) -> tuple[Word, ...]:
        """Extract word timings, tolerating their absence."""
        raw_words = getattr(raw_segment, "words", None)
        if not raw_words:
            return ()
        words: list[Word] = []
        for raw in raw_words:
            text = str(raw.word).strip()
            start, end = float(raw.start), float(raw.end)
            if not text or end <= start:
                # Whisper occasionally emits a zero-width word at a segment boundary.
                continue
            words.append(
                Word(
                    text=text,
                    start=start,
                    end=end,
                    probability=_clamp_score(getattr(raw, "probability", 1.0)),
                )
            )
        return tuple(words)

    def _filter_hallucinations(
        self, segments: list[TranscriptSegment]
    ) -> tuple[list[TranscriptSegment], list[TranscriptSegment]]:
        """Split segments into keepers and likely fabrications.

        Returns both halves rather than silently discarding, so the caller can report
        how much was dropped. A quiet drop of half the narration would be far worse
        than the hallucination it prevented.
        """
        kept: list[TranscriptSegment] = []
        dropped: list[TranscriptSegment] = []
        for segment in segments:
            if self._is_probably_hallucinated(segment):
                dropped.append(segment)
                logger.debug(
                    "Dropping segment %d: no_speech=%s compression=%s text=%r",
                    segment.index,
                    segment.no_speech_prob,
                    segment.compression_ratio,
                    segment.text[:60],
                )
            else:
                kept.append(segment)
        return kept, dropped

    def _is_probably_hallucinated(self, segment: TranscriptSegment) -> bool:
        settings = self._settings
        if (
            segment.no_speech_prob is not None
            and segment.no_speech_prob > settings.max_no_speech_prob
        ):
            return True
        return (
            segment.compression_ratio is not None
            and segment.compression_ratio > settings.max_compression_ratio
        )


def _can_load_library(name: str) -> bool:
    """Whether a shared library can actually be loaded.

    Loading it is the only honest test. Searching ``PATH`` would miss libraries
    installed into a package directory (the ``nvidia-*`` pip wheels put them there),
    and would also report success for a file that exists but is the wrong architecture.
    """
    import ctypes
    import ctypes.util

    candidates = [name]
    if sys.platform == "win32":
        candidates.append(f"{name}.dll")
    else:
        # POSIX names carry the soname version, e.g. libcublas.so.12.
        stem = name.rstrip("0123456789_")
        located = ctypes.util.find_library(stem)
        if located is not None:
            candidates.append(located)

    for candidate in candidates:
        try:
            ctypes.CDLL(candidate)
        except OSError:
            continue
        else:
            return True
    return False


def _is_cuda_environment_failure(exc: BaseException) -> bool:
    """Whether a ``RuntimeError`` is a CUDA setup problem rather than a real bug.

    ctranslate2 raises a bare ``RuntimeError`` for every device failure, so the message
    is the only signal available. Deliberately narrow: a false positive here would
    silently retry on the CPU and hide an actual defect.
    """
    message = str(exc).casefold()
    return any(marker in message for marker in _LIBRARY_LOAD_MARKERS)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def _optional_score(value: object) -> float | None:
    return None if value is None else _clamp_score(value)


def _clamp_score(value: object) -> float:
    """Coerce into 0..1.

    Whisper occasionally reports a probability a hair outside the range, and a
    validation error on a rounding artefact after a twenty-minute transcription would
    be an unreasonable way to lose the result.
    """
    return min(1.0, max(0.0, float(value)))  # type: ignore[arg-type]


def _timecode(seconds: float) -> str:
    """Format a position as ``H:MM:SS``.

    Bare seconds are unreadable at this scale: "8420.3s of 12970.3s" says far less
    about how much of a lecture is left than "2:20:20 of 3:36:10" does.
    """
    total = max(0, int(seconds))
    return f"{total // 3600:d}:{total % 3600 // 60:02d}:{total % 60:02d}"


__all__ = ["RECOGNIZER_VERSION", "FasterWhisperRecognizer", "SpeechDependencyMissingError"]
