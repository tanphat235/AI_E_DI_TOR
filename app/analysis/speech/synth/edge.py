"""Microsoft Edge read-aloud synthesis.

**This is the one component in AIVE that makes a network call.** Everything else runs
offline, and that is a stated promise of the project, so this backend is an opt-in extra
(``pip install -e ".[tts]"``), announces itself through
:attr:`~app.analysis.speech.synth.base.SpeechSynthesizer.requires_network`, and is flagged by
``aive doctor``.

It is here because it is the only free option with usable Vietnamese voices —
``vi-VN-HoaiMyNeural`` and ``vi-VN-NamMinhNeural``. Verified before adopting it: the library
takes no ``api_key`` and no ``token`` parameter, needs no account, and costs nothing.

The reason to prefer it beyond voice quality: it reports **word offsets**, so a synthesised
narration gets per-word timing straight from the service rather than from a recogniser's
estimate. That is what makes karaoke subtitles exact. Note the boundary mode has to be asked
for — the default is ``SentenceBoundary``, and reading the stream for ``WordBoundary`` events
without setting it yields none, which looks exactly like a language that does not support
them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.analysis.speech.synth.base import (
    SynthesisDependencyMissingError,
    SynthesisError,
    SynthesizedLine,
    Voice,
    VoiceNotFoundError,
    WordTiming,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

EDGE_VERSION = "edge-tts/1"

_TICKS_PER_SECOND = 10_000_000
"""Edge reports offsets in 100-nanosecond ticks, the .NET ``TimeSpan`` unit."""

DEFAULT_VOICES: dict[str, str] = {
    "vi": "vi-VN-HoaiMyNeural",
    "en": "en-US-AriaNeural",
}
"""A sensible voice per language, so ``--voice`` is optional.

Only the two languages this project has actually been used in. A language not listed here
requires an explicit voice rather than getting a guess — picking the alphabetically first
``xx-*`` voice would be arbitrary and occasionally absurd.
"""


class EdgeSynthesizer:
    """A :class:`~app.analysis.speech.synth.base.SpeechSynthesizer` over Edge's service."""

    def __init__(self, *, timeout: float = 60.0) -> None:
        self._timeout = timeout
        self._voices: tuple[Voice, ...] | None = None

    @property
    def name(self) -> str:
        return "edge"

    @property
    def requires_network(self) -> bool:
        return True

    def voices(self) -> tuple[Voice, ...]:
        """Every voice the service offers, fetched once and cached for the process."""
        if self._voices is not None:
            return self._voices

        module = self._module()
        try:
            listed: list[dict[str, Any]] = asyncio.run(module.list_voices())
        except Exception as exc:  # the library raises assorted network error types
            msg = (
                f"could not reach the Edge voice service: {type(exc).__name__}: {exc}. "
                "This backend needs an internet connection; use --backend sapi to stay offline."
            )
            raise SynthesisError(msg) from exc

        self._voices = tuple(
            Voice(
                name=str(item["ShortName"]),
                locale=str(item["Locale"]),
                gender=str(item.get("Gender", "")),
                description=str(item.get("FriendlyName", "")),
            )
            for item in listed
        )
        return self._voices

    def default_voice(self, language: str) -> str:
        """The voice to use when the caller did not name one."""
        chosen = DEFAULT_VOICES.get(language.split("-")[0].lower())
        if chosen is None:
            available = ", ".join(sorted(DEFAULT_VOICES))
            msg = (
                f"no default voice for language {language!r}; pass --voice explicitly. "
                f"Defaults exist for: {available}. `aive narrate --list-voices` shows all."
            )
            raise VoiceNotFoundError(msg)
        return chosen

    def speak(
        self, text: str, *, voice: str, destination: Path, rate: float = 1.0
    ) -> SynthesizedLine:
        """Synthesise one line, capturing audio and word offsets in a single pass."""
        module = self._module()
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            duration_ticks, words = asyncio.run(
                self._stream(module, text, voice=voice, destination=destination, rate=rate)
            )
        except SynthesisError:
            raise
        except Exception as exc:
            # The library surfaces an unknown voice as a generic connection failure, so the
            # message names it as the likeliest cause rather than leaving the user to guess.
            msg = (
                f"could not synthesise with voice {voice!r}: {type(exc).__name__}: {exc}. "
                "Check the voice name with `aive narrate --list-voices`, and that you are online."
            )
            raise SynthesisError(msg) from exc

        if not destination.is_file() or destination.stat().st_size == 0:
            msg = f"the service returned no audio for {text[:40]!r}"
            raise SynthesisError(msg)

        return SynthesizedLine(
            index=0,  # assigned by the narrator, which knows the script order
            text=text,
            audio=destination,
            duration=duration_ticks / _TICKS_PER_SECOND if duration_ticks else 0.0,
            words=words,
        )

    async def _stream(
        self,
        module: Any,
        text: str,
        *,
        voice: str,
        destination: Path,
        rate: float,
    ) -> tuple[int, tuple[WordTiming, ...]]:
        """Write the audio and collect word offsets from one stream."""
        communicate = module.Communicate(
            text,
            voice,
            rate=_rate_string(rate),
            # Must be asked for: the default is SentenceBoundary, and filtering the stream
            # for WordBoundary without this yields nothing at all.
            boundary="WordBoundary",
        )

        words: list[WordTiming] = []
        last_end = 0
        with destination.open("wb") as handle:
            async for chunk in communicate.stream():
                kind = chunk.get("type")
                if kind == "audio":
                    handle.write(chunk["data"])
                elif kind == "WordBoundary":
                    offset = int(chunk.get("offset", 0))
                    length = int(chunk.get("duration", 0))
                    words.append(
                        WordTiming(
                            text=str(chunk.get("text", "")),
                            start=offset / _TICKS_PER_SECOND,
                            end=(offset + length) / _TICKS_PER_SECOND,
                        )
                    )
                    last_end = max(last_end, offset + length)

        return last_end, tuple(words)

    @staticmethod
    def _module() -> Any:
        try:
            import edge_tts
        except ImportError as exc:
            msg = (
                "the edge backend needs the edge-tts package. Install it with: "
                'pip install -e ".[tts]" - or use --backend sapi, which needs nothing '
                "and makes no network call."
            )
            raise SynthesisDependencyMissingError(msg) from exc
        return edge_tts


def _rate_string(rate: float) -> str:
    """A rate multiplier as the percentage delta Edge expects.

    ``1.0`` becomes ``+0%``. The sign is mandatory — a bare ``25%`` is rejected — which is
    the kind of detail worth encoding in a function rather than an f-string at the call site.
    """
    percent = round((rate - 1.0) * 100)
    return f"{percent:+d}%"


__all__ = ["DEFAULT_VOICES", "EDGE_VERSION", "EdgeSynthesizer"]
