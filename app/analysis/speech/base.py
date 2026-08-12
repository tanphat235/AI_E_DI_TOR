"""Speech recognition boundary (implemented in Phase 2).

Declared as a :class:`typing.Protocol` rather than an abstract base class, so an
implementation never has to import or inherit from AIVE. That keeps the recogniser
- a heavyweight thing that loads a multi-hundred-megabyte model - swappable, and
lets tests substitute a plain object returning a canned
:class:`~app.models.speech.Transcript` with no ML dependency at all.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.models.common import MediaRef
from app.models.speech import SilenceSpan, SpeechCleanupReport, Transcript

ProgressCallback = Callable[[float, str], None]
"""Reports progress as ``(fraction_complete, description)``.

Mirrors :data:`app.renderer.base.ProgressCallback` in shape but is declared separately:
analysis must not import from the renderer, and the two are free to diverge.

Injected rather than baked in for the same reason the renderer's is - transcribing a
three-hour lecture takes tens of minutes, and the console, the desktop UI and a test
each want to observe that differently.
"""


@runtime_checkable
class SilenceDetector(Protocol):
    """Finds stretches of an audio file quiet enough to cut.

    Separate from :class:`NarrationCleaner` because it is the only part that must
    actually decode audio. Keeping it behind its own protocol means cleanup logic -
    the part with all the judgement in it - can be tested with a list of spans and no
    audio file at all.
    """

    @property
    def name(self) -> str:
        """Detector identity, e.g. ``ffmpeg-silencedetect``."""
        ...

    def detect(
        self,
        audio: Path,
        *,
        threshold_db: float,
        min_duration: float,
    ) -> tuple[SilenceSpan, ...]:
        """Return quiet spans, ascending and non-overlapping.

        Args:
            audio: Absolute path to the audio to scan.
            threshold_db: Level below which audio counts as silence, in dBFS.
            min_duration: Ignore quiet spans shorter than this. Shorter gaps are the
                natural rhythm of speech; cutting them makes narration breathless.
        """
        ...


@runtime_checkable
class SpeechRecognizer(Protocol):
    """Turns an audio file into a timed transcript."""

    @property
    def name(self) -> str:
        """Identity recorded in ``Transcript.model_name``, e.g. ``faster-whisper/medium``.

        Stored with the result so a transcript produced by a small model can be
        distinguished from one produced by a large model months later.
        """
        ...

    def transcribe(
        self,
        audio: Path,
        *,
        ref: MediaRef,
        on_progress: ProgressCallback | None = None,
    ) -> Transcript:
        """Recognise speech in ``audio``.

        Args:
            audio: Absolute path to the audio file to read.
            ref: Project-relative reference recorded in the result, so the
                transcript stays portable while the read stays absolute.
            on_progress: Called as recognition advances. Optional because a
                recogniser fast enough to need no reporting is a legitimate
                implementation, and every existing caller predates this.
        """
        ...


@runtime_checkable
class NarrationCleaner(Protocol):
    """Decides which parts of the narration survive.

    Separate from :class:`SpeechRecognizer` because it is pure analysis over an
    existing transcript plus the audio's levels - no model required. Splitting them
    means cleanup thresholds can be retuned and re-run in milliseconds instead of
    re-transcribing.
    """

    def clean(self, transcript: Transcript, *, audio: Path) -> SpeechCleanupReport:
        """Propose silence, filler and repetition removals for ``transcript``."""
        ...


__all__ = ["NarrationCleaner", "ProgressCallback", "SilenceDetector", "SpeechRecognizer"]
