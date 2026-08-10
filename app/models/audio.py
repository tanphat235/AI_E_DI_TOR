"""Music library analysis (Phase 7 input).

The user supplies the music; AIVE's job is to describe each track well enough
that the AI director can pick one deliberately rather than alphabetically.

Descriptors here are the ones classical audio analysis can actually measure
reliably - tempo, energy, integrated loudness - plus a small closed mood
vocabulary. Loudness in particular is measured, not guessed: ducking music under
narration only works if both levels are known in the same units.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.models.common import AiveModel, MediaRef, MusicMood, Score, Seconds
from app.models.media import MediaProbe


class MusicTrack(AiveModel):
    """One analysed track from the user's music folder."""

    source: MediaRef
    probe: MediaProbe
    bpm: float | None = Field(
        default=None,
        gt=0.0,
        description="Estimated tempo. None when detection was not confident.",
    )
    energy: Score = Field(
        default=0.5,
        description="Perceived intensity from spectral features. Drives pacing matches.",
    )
    loudness_lufs: float | None = Field(
        default=None,
        description=(
            "Integrated loudness in LUFS, typically between -30 and -6. Needed to "
            "normalise every track to the same perceived level before ducking, so "
            "swapping the music bed does not change how loud the video feels."
        ),
    )
    moods: tuple[MusicMood, ...] = Field(
        default=(),
        description="Inferred moods, most confident first. Empty means undetermined.",
    )
    tags: tuple[str, ...] = Field(
        default=(),
        description="Free-text labels, e.g. from the filename or ID3 genre.",
    )
    intro_end: Seconds | None = Field(
        default=None,
        description=(
            "Where the track's intro finishes. Starting a music bed after a long "
            "ambient intro is the difference between a bed that supports the edit "
            "and eight seconds of apparent silence."
        ),
    )
    analyzer_version: str = Field(
        default="",
        description=(
            "Version of the analysis pipeline that produced this. Cached results from an "
            "older analyser must be invalidated, not trusted: retuning the energy curve or "
            "the mood table changes what these numbers mean, and the director would still "
            "be reading them as current. Empty only in hand-written fixtures."
        ),
    )
    analyzed_at: datetime | None = None

    @property
    def duration(self) -> float:
        return self.probe.duration

    def matches_mood(self, wanted: MusicMood) -> bool:
        """True when this track carries ``wanted`` as one of its moods."""
        return wanted in self.moods


class MusicLibrary(AiveModel):
    """Every analysed track available to a project."""

    tracks: tuple[MusicTrack, ...] = ()

    def by_mood(self, mood: MusicMood) -> tuple[MusicTrack, ...]:
        """Tracks carrying ``mood``, longest first.

        Longest-first because a bed that outlasts the section needs no loop point,
        and looping is the most audible compromise in an automated edit.
        """
        matches = [track for track in self.tracks if track.matches_mood(mood)]
        return tuple(sorted(matches, key=lambda track: track.duration, reverse=True))

    def find(self, ref: MediaRef) -> MusicTrack | None:
        """Look up a track by its media reference."""
        return next((track for track in self.tracks if track.source == ref), None)

    @property
    def total_duration(self) -> float:
        return sum(track.duration for track in self.tracks)


__all__ = ["MusicLibrary", "MusicTrack"]
