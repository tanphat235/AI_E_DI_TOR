"""Speech synthesis: a script becomes a narration track.

The mirror of :mod:`app.analysis.speech`, which turns audio into text.

:mod:`.script` parses the script and imports nothing heavy; :mod:`.base` is the boundary;
:mod:`.sapi` and :mod:`.edge` are the backends; :mod:`.narrator` joins the parts and builds
the :class:`~app.models.speech.NarrationAnalysis` — which, because the text and the timings
are both known, needs no recogniser at all.
"""

from app.analysis.speech.synth.base import (
    SpeechSynthesizer,
    SynthesisDependencyMissingError,
    SynthesisError,
    SynthesizedLine,
    Voice,
    VoiceNotFoundError,
)
from app.analysis.speech.synth.script import ScriptError, ScriptLine, load_script, parse_script

__all__ = [
    "ScriptError",
    "ScriptLine",
    "SpeechSynthesizer",
    "SynthesisDependencyMissingError",
    "SynthesisError",
    "SynthesizedLine",
    "Voice",
    "VoiceNotFoundError",
    "load_script",
    "parse_script",
]
