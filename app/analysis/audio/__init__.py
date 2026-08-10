"""Music library analysis (Phase 7).

Describes the user's music well enough that the director can choose a bed deliberately.
Nothing here selects anything — selection is an editorial decision, and the director makes
it from the digest this produces.
"""

from app.analysis.audio.base import (
    AudioDecodeError,
    AudioDependencyMissingError,
    LoudnessMeter,
    MusicAnalyzer,
)

__all__ = [
    "AudioDecodeError",
    "AudioDependencyMissingError",
    "LoudnessMeter",
    "MusicAnalyzer",
]
