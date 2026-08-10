"""Windows speech synthesis, offline.

Shells out to PowerShell's ``System.Speech.Synthesis`` — the same tactic the renderer uses
with FFmpeg, and for the same reason: the capability is already on the machine, so adding a
Python dependency to reach it would be paying for something we already have.

The properties that make it the default: **no dependency, no download, and no network call.**
AIVE's promise is that it runs offline, and this backend keeps it.

The limitation is equally real, and the user finds out immediately rather than after a wasted
render: it can only use voices installed in Windows. A stock Windows install has English
only. Vietnamese needs a voice pack from *Settings → Time & Language → Speech → Add voices*,
and :meth:`SapiSynthesizer.voices` reports exactly what is there so the gap is visible.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.analysis.speech.synth.base import (
    SynthesisError,
    SynthesizedLine,
    Voice,
    VoiceNotFoundError,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

SAPI_VERSION = "sapi/1"

_TIMEOUT = 120.0
"""Per line. Synthesis is local and fast; a minute-long wait means something has hung."""

_LIST_SCRIPT = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.GetInstalledVoices() | ForEach-Object {
  $i = $_.VoiceInfo
  [PSCustomObject]@{ name = $i.Name; locale = $i.Culture.Name; gender = $i.Gender.ToString() }
} | ConvertTo-Json -Compress -AsArray
$s.Dispose()
"""

_SPEAK_SCRIPT = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try { $s.SelectVoice($env:AIVE_VOICE) } catch { Write-Error "voice-not-found"; exit 3 }
$s.Rate = [int]$env:AIVE_RATE
$s.SetOutputToWaveFile($env:AIVE_OUT)
$s.Speak([Console]::In.ReadToEnd())
$s.Dispose()
"""
"""Text arrives on **stdin**, not as an argument.

Deliberate: a narration line contains quotes, apostrophes and non-ASCII, and interpolating it
into a PowerShell string is both a quoting minefield and an injection risk. Reading stdin
sidesteps escaping entirely. The voice, rate and output path go through the environment for
the same reason.
"""


class SapiSynthesizer:
    """A :class:`~app.analysis.speech.synth.base.SpeechSynthesizer` over Windows SAPI."""

    def __init__(self) -> None:
        self._voices: tuple[Voice, ...] | None = None

    @property
    def name(self) -> str:
        return "sapi"

    @property
    def requires_network(self) -> bool:
        return False

    @property
    def available(self) -> bool:
        """Whether this backend can run at all. ``System.Speech`` is Windows-only."""
        return sys.platform == "win32"

    def voices(self) -> tuple[Voice, ...]:
        """Voices installed in Windows, cached for the process."""
        if self._voices is not None:
            return self._voices
        if not self.available:
            self._voices = ()
            return self._voices

        completed = self._powershell(_LIST_SCRIPT)
        if completed.returncode != 0:
            logger.warning("Could not list SAPI voices: %s", completed.stderr.strip()[:200])
            self._voices = ()
            return self._voices

        try:
            listed = json.loads(completed.stdout or "[]")
        except json.JSONDecodeError:
            logger.warning("SAPI voice list was not valid JSON")
            self._voices = ()
            return self._voices

        self._voices = tuple(
            Voice(
                name=str(item.get("name", "")),
                locale=str(item.get("locale", "")),
                gender=str(item.get("gender", "")),
            )
            for item in listed
            if item.get("name")
        )
        return self._voices

    def default_voice(self, language: str) -> str:
        """The first installed voice matching ``language``.

        Raises rather than falling back to an English voice for a Vietnamese script. Reading
        Vietnamese with an American voice produces something confidently unusable, and the
        user would have to listen to a whole render to find out.
        """
        wanted = language.split("-")[0].lower()
        for voice in self.voices():
            if voice.language == wanted:
                return voice.name

        installed = ", ".join(f"{v.name} ({v.locale})" for v in self.voices()) or "none"
        msg = (
            f"no installed Windows voice for language {wanted!r}. Installed: {installed}. "
            "Add one under Settings > Time & Language > Speech > Add voices, or use "
            "--backend edge (needs an internet connection but has Vietnamese voices)."
        )
        raise VoiceNotFoundError(msg)

    def speak(
        self, text: str, *, voice: str, destination: Path, rate: float = 1.0
    ) -> SynthesizedLine:
        """Synthesise one line to a WAV file."""
        if not self.available:
            msg = "the sapi backend needs Windows; use --backend edge on other platforms"
            raise SynthesisError(msg)

        destination.parent.mkdir(parents=True, exist_ok=True)
        completed = self._powershell(
            _SPEAK_SCRIPT,
            stdin=text,
            env_extra={
                "AIVE_VOICE": voice,
                "AIVE_RATE": str(_rate_to_sapi(rate)),
                "AIVE_OUT": str(destination),
            },
        )

        if completed.returncode == 3 or "voice-not-found" in completed.stderr:
            installed = ", ".join(item.name for item in self.voices()) or "none"
            msg = f"Windows has no voice named {voice!r}. Installed: {installed}"
            raise VoiceNotFoundError(msg)
        if completed.returncode != 0:
            msg = f"SAPI failed: {completed.stderr.strip()[:300]}"
            raise SynthesisError(msg)
        if not destination.is_file() or destination.stat().st_size == 0:
            msg = f"SAPI wrote no audio for {text[:40]!r}"
            raise SynthesisError(msg)

        return SynthesizedLine(
            index=0,
            text=text,
            audio=destination,
            # Left at zero: SAPI reports no duration, so the narrator measures the file. It
            # has to measure anyway to be sure, and one source of truth beats two.
            duration=0.0,
            words=(),
        )

    @staticmethod
    def _powershell(
        script: str,
        *,
        stdin: str | None = None,
        env_extra: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        import os

        environment = dict(os.environ)
        if env_extra:
            environment.update(env_extra)

        try:
            return subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    script,
                ],
                input=stdin if stdin is not None else "",
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_TIMEOUT,
                check=False,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            msg = f"could not run PowerShell: {exc}"
            raise SynthesisError(msg) from exc


def _rate_to_sapi(rate: float) -> int:
    """A rate multiplier as SAPI's -10..10 integer scale.

    SAPI's scale is not linear in speed and is not documented as one; each step is roughly a
    third faster. The mapping below is approximate on purpose, and clamped so a caller asking
    for 3x gets the fastest available rather than an error.
    """
    if rate <= 0.0:
        return 0
    import math

    steps = round(math.log(rate) / math.log(1.33))
    return max(-10, min(10, int(steps)))


__all__ = ["SAPI_VERSION", "SapiSynthesizer"]
