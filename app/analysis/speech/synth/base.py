"""Speech synthesis boundary.

The mirror of :mod:`app.analysis.speech.base`, which turns audio into text. This turns text
into audio, and it changes something important downstream: **when AIVE generates the
narration, it already knows what was said and when.**

That is worth stating plainly, because it removes a whole stage. The record-then-transcribe
path has to run a 1.5 GB Whisper model, guess at word boundaries, and then clean out the
"um"s and the retakes. A synthesised narration has no fillers to remove, no retakes, and the
service reports word offsets directly — so :mod:`.narrator` builds the same
:class:`~app.models.speech.NarrationAnalysis` document with exact timings and no model
download at all.

Two backends ship, and the difference between them is a real trade-off rather than a
preference:

* :mod:`.sapi` uses Windows' own ``System.Speech``. No dependency, no download, **no network
  call** — it shells out to PowerShell the same way the renderer shells out to FFmpeg. But it
  can only use voices installed in Windows.
* :mod:`.edge` uses Microsoft Edge's read-aloud service. Free, no account and no API key, and
  it has the only good Vietnamese voices available. **It calls the network**, which is the one
  thing the rest of AIVE never does, so it is an opt-in extra and ``aive doctor`` says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


class SynthesisError(RuntimeError):
    """Raised when a line could not be synthesised. The text or the voice is the problem."""


class SynthesisDependencyMissingError(RuntimeError):
    """Raised when a backend's package is absent.

    Its own type so the CLI maps it to exit 5 (environment) rather than 4 (bad input): the
    fix is ``pip install``, not a different script.
    """


class VoiceNotFoundError(SynthesisError):
    """Raised when the requested voice is not available to this backend.

    Separate because it is the most common failure and the most fixable — the message lists
    what *is* available, so a typo resolves itself.
    """


@dataclass(frozen=True, slots=True)
class Voice:
    """One voice a backend can use."""

    name: str
    """The identifier to pass back in, e.g. ``vi-VN-HoaiMyNeural``."""
    locale: str
    """BCP-47 tag, e.g. ``vi-VN``. Used to pick a sensible default per language."""
    gender: str = ""
    description: str = ""

    @property
    def language(self) -> str:
        """Just the language subtag, e.g. ``vi``."""
        return self.locale.split("-")[0].lower()


@dataclass(frozen=True, slots=True)
class WordTiming:
    """One word, timed relative to the start of its own line."""

    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class SynthesizedLine:
    """One line of script, spoken.

    ``duration`` is measured from the produced audio rather than estimated from the text.
    That measurement is what lets the timeline be built from real numbers: a plan whose clip
    lengths came from a guess at speaking rate drifts against the voice by the tenth
    sentence.
    """

    index: int
    text: str
    audio: Path
    duration: float
    words: tuple[WordTiming, ...] = ()
    """Empty when the backend cannot report word offsets. Karaoke subtitles need these;
    everything else works from ``duration`` alone."""


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Turns a line of text into an audio file."""

    @property
    def name(self) -> str:
        """Backend identity, e.g. ``edge`` or ``sapi``."""
        ...

    @property
    def requires_network(self) -> bool:
        """Whether speaking a line makes a network call.

        Exposed on the protocol, not buried in a docstring, because AIVE's core promise is
        that it makes none. A caller that cares has to be able to ask.
        """
        ...

    def voices(self) -> tuple[Voice, ...]:
        """Every voice available. Empty when they cannot be enumerated."""
        ...

    def speak(
        self, text: str, *, voice: str, destination: Path, rate: float = 1.0
    ) -> SynthesizedLine:
        """Synthesise ``text`` to ``destination``.

        Args:
            text: One line. Splitting a script into lines is :mod:`.script`'s job.
            voice: A name from :meth:`voices`.
            destination: Where to write. The suffix chooses the container.
            rate: Speaking rate multiplier. ``1.0`` is the voice's natural pace.

        Raises:
            SynthesisError: the line could not be spoken.
            VoiceNotFoundError: ``voice`` is not available.
        """
        ...


__all__ = [
    "SpeechSynthesizer",
    "SynthesisDependencyMissingError",
    "SynthesisError",
    "SynthesizedLine",
    "Voice",
    "VoiceNotFoundError",
    "WordTiming",
]
