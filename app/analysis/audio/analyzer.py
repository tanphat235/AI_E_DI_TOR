"""The music analysis pipeline.

One track in, one :class:`~app.models.audio.MusicTrack` out, composed from four pieces
that fail independently:

1. **Probe** the container with PyAV, for duration and stream facts.
2. **Decode** to mono numpy once. Every feature is computed from that one array — decoding
   a five-minute track four times to answer four questions would be the obvious mistake.
3. **Measure loudness** with FFmpeg. Separate because it is the one number that must match
   the narration's ruler exactly, so it uses the reference meter rather than ours.
4. **Describe** it: energy from level and brightness, mood from the decision table.

A track that fails is *recorded*, not dropped — the same rule Phase 4 established for
footage. A library that silently omitted the one unreadable file would read as "these are
all your options", and the director would plan around music it was never told about.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.analysis.audio.base import AudioDecodeError, AudioDependencyMissingError
from app.analysis.audio.decode import DECODER_VERSION, decode_mono
from app.analysis.audio.features import FEATURES_VERSION, extract
from app.analysis.audio.loudness import (
    MINIMUM_MEASURABLE_DURATION,
    FFmpegLoudnessMeter,
)
from app.analysis.audio.mood import MOOD_VERSION, classify, energy_score
from app.analysis.vision.probe import ProbeError, PyAvProber
from app.config.settings import AiveSettings
from app.models.audio import MusicLibrary, MusicTrack
from app.models.common import MediaRef
from app.services.ffmpeg_locator import FFmpegLocator
from app.utils.logging import get_logger, stage

logger = get_logger(__name__)

MUSIC_ANALYZER_VERSION = f"music/1+{DECODER_VERSION}+{FEATURES_VERSION}+{MOOD_VERSION}"
"""Composite, so retuning any stage invalidates a cached library.

The loudness meter is deliberately absent from this string: it delegates to FFmpeg, and
folding the FFmpeg build into the cache key would invalidate every project on an unrelated
system update.
"""


class TrackFailure:
    """A track that could not be analysed, and why.

    A plain class rather than a model: it never leaves the process, because
    :class:`~app.models.audio.MusicLibrary` has no failure field. The CLI reports these on
    stderr and in the digest.
    """

    __slots__ = ("error", "path")

    def __init__(self, path: Path, error: str) -> None:
        self.path = path
        self.error = error


class DefaultMusicAnalyzer:
    """A :class:`~app.analysis.audio.base.MusicAnalyzer` over PyAV, numpy and FFmpeg."""

    def __init__(self, settings: AiveSettings, locator: FFmpegLocator) -> None:
        self._settings = settings
        self._music = settings.music
        self._prober = PyAvProber()
        self._loudness = FFmpegLoudnessMeter(locator)

    @property
    def version(self) -> str:
        return MUSIC_ANALYZER_VERSION

    def analyze(self, path: Path, *, ref: MediaRef) -> MusicTrack:
        """Analyse one track.

        Raises:
            AudioDecodeError: the file cannot be read, or carries no audio.
            AudioDependencyMissingError: PyAV or numpy is absent.
        """
        try:
            probe = self._prober.probe(path, ref=ref)
        except ProbeError as exc:
            msg = f"could not probe {path.name}: {exc}"
            raise AudioDecodeError(msg) from exc

        if probe.audio is None:
            msg = f"{path.name} has no audio stream"
            raise AudioDecodeError(msg)

        settings = self._music
        samples = decode_mono(path, sample_rate=settings.analysis_sample_rate)
        features = extract(
            samples,
            sample_rate=settings.analysis_sample_rate,
            bpm_min=settings.bpm_min,
            bpm_max=settings.bpm_max,
            bpm_min_confidence=settings.bpm_min_confidence,
            bpm_min_crest=settings.bpm_min_crest,
            brightness_reference_hz=settings.brightness_reference_hz,
            intro_level_ratio=settings.intro_level_ratio,
            intro_max_fraction=settings.intro_max_fraction,
        )

        energy = energy_score(
            rms_db=features.rms_db, brightness=features.brightness, settings=settings
        )

        # Skipped below the gate's settling time rather than reported as a wrong number:
        # a two-second sting has no meaningful integrated loudness.
        loudness = (
            self._loudness.measure(path) if probe.duration >= MINIMUM_MEASURABLE_DURATION else None
        )
        if loudness is None:
            logger.debug("No loudness for %s; it cannot be level-matched", path.name)

        return MusicTrack(
            source=ref,
            probe=probe,
            bpm=features.tempo.bpm,
            energy=energy,
            loudness_lufs=loudness,
            moods=classify(
                energy=energy,
                brightness=features.brightness,
                bpm=features.tempo.bpm,
                settings=settings,
            ),
            tags=_tags_from_name(path),
            intro_end=features.intro_end,
            analyzer_version=MUSIC_ANALYZER_VERSION,
            analyzed_at=datetime.now(UTC),
        )

    def analyze_library(
        self, tracks: tuple[tuple[Path, MediaRef], ...]
    ) -> tuple[MusicLibrary, tuple[TrackFailure, ...]]:
        """Analyse every track, surviving individual failures.

        One corrupt file must not abort a twelve-track library — the same lesson Phase 4
        learned about footage, applied before it could be relearned here.
        """
        analysed: list[MusicTrack] = []
        failures: list[TrackFailure] = []

        for path, ref in tracks:
            with stage(logger, f"Analysing {path.name}"):
                try:
                    analysed.append(self.analyze(path, ref=ref))
                except AudioDependencyMissingError:
                    # An install problem affects every track, so there is no point
                    # trying the rest.
                    raise
                except (AudioDecodeError, ValueError, OSError) as exc:
                    logger.warning("Skipping %s: %s", path.name, exc)
                    failures.append(TrackFailure(path, str(exc)))

        return MusicLibrary(tracks=tuple(analysed)), tuple(failures)


def _tags_from_name(path: Path) -> tuple[str, ...]:
    """Words from the filename, as free-text tags.

    Library music is named ``uplifting-corporate-loop.mp3`` far more often than it carries
    usable ID3 genre, so the filename is the better signal. Kept separate from ``moods``
    because these are the *user's* words and the moods are ours — the director should be
    able to tell which is which, and trust the filename more.
    """
    import re

    words = re.split(r"[\s_\-.]+", path.stem.lower())
    seen: list[str] = []
    for word in words:
        if len(word) >= 3 and not word.isdigit() and word not in seen:
            seen.append(word)
    return tuple(seen)


__all__ = ["MUSIC_ANALYZER_VERSION", "DefaultMusicAnalyzer", "TrackFailure"]
