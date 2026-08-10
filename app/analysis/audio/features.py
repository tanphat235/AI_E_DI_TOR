"""Music features from a mono waveform.

Pure functions over numpy arrays: no files, no FFmpeg, no settings objects beyond the
numbers passed in. That is what makes this module testable against a synthetic click
train at a known tempo, which is exactly how the tempo estimator is verified.

The chain is the standard one, and each stage is here because the next one needs it:

1. **STFT magnitude** — a spectrogram, hop 512 at 22.05 kHz, so roughly 43 frames a second.
2. **Onset envelope** — positive spectral flux, i.e. how much *more* energy each frame has
   than the one before it. Rising energy is what an ear hears as a beat; falling energy is
   a note ending and must not count, which is why the flux is half-wave rectified.
3. **Tempo** — autocorrelation of that envelope. The lag with the strongest correlation
   inside the plausible BPM window is the beat period.

Step 3 is where an honest limitation lives. Autocorrelation finds *a* periodicity, and
music is periodic at several levels at once, so the half- and double-tempo lags correlate
almost as strongly as the true one. We resolve it by preferring the octave nearest the
centre of the search window, and we report ``None`` when nothing correlates well enough —
a wrong tempo is worse than an absent one, because the director would pace cuts to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

FEATURES_VERSION = "audio-features/1"

FRAME_LENGTH = 1024
"""STFT window in samples. At 22.05 kHz this is ~46 ms — long enough to resolve a bass
note, short enough that a drum hit lands in one or two frames."""

HOP_LENGTH = 512
"""Frames advance by this many samples, giving ~43 analysis frames per second. Fine enough
to place a beat within a fiftieth of a second, coarse enough to autocorrelate cheaply."""

_EPSILON = 1e-10
"""Guard against log(0) and division by a silent frame."""

_NOISE_FLOOR = 10.0 ** (-80.0 / 20.0)
"""Bins quieter than 80 dB below the spectrogram's peak are clamped, not measured.

Eighty decibels is past the noise floor of any real recording, so nothing audible is lost -
but it excludes FFT leakage, which is what a log spectrum otherwise magnifies into a signal.
See :func:`onset_envelope` for the failure this prevents."""


@dataclass(frozen=True, slots=True)
class TempoEstimate:
    """A tempo, and how much to believe it."""

    bpm: float | None
    confidence: float
    """Normalised autocorrelation strength at the winning lag, 0.0 to 1.0."""


@dataclass(frozen=True, slots=True)
class AudioFeatures:
    """Everything :mod:`.analyzer` needs to describe a track."""

    duration: float
    rms_db: float
    """Overall level in dBFS. Not loudness — see :mod:`.loudness` for LUFS."""
    peak_db: float
    brightness: float
    """Spectral centroid normalised against a reference frequency, 0.0 to 1.0."""
    tempo: TempoEstimate
    intro_end: float | None
    """Where the opening quiet section stops, or ``None`` when the track starts at level."""


def spectrogram(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    """Magnitude STFT with a Hann window, shaped ``(frames, bins)``.

    Written out rather than pulled from scipy because it is fifteen lines and the
    alternative is a dependency whose only other use would be this.
    """
    import numpy as np

    if samples.size < FRAME_LENGTH:
        return np.zeros((0, FRAME_LENGTH // 2 + 1), dtype=np.float32)

    window = np.hanning(FRAME_LENGTH).astype(np.float32)
    frame_count = 1 + (samples.size - FRAME_LENGTH) // HOP_LENGTH

    # A strided view rather than a copy: a five-minute track is ~13k frames of 1024
    # samples, and materialising that as a real array costs 50 MB for no benefit.
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(frame_count, FRAME_LENGTH),
        strides=(samples.strides[0] * HOP_LENGTH, samples.strides[0]),
        writeable=False,
    )
    return np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)


def onset_envelope(magnitudes: NDArray[np.float32]) -> NDArray[np.float32]:
    """Half-wave-rectified spectral flux, one value per frame.

    Computed on a log-magnitude spectrogram so that a hi-hat over a loud chord still
    registers. On a linear scale the flux is dominated by whichever band happens to carry
    the most energy, and quiet percussion — the part that actually carries the beat —
    disappears.
    """
    import numpy as np

    if magnitudes.shape[0] < 2:
        return np.zeros(magnitudes.shape[0], dtype=np.float32)

    peak = float(magnitudes.max())
    if peak <= _EPSILON:
        return np.zeros(magnitudes.shape[0], dtype=np.float32)

    # Floor the spectrum a fixed distance below its own peak before taking the log.
    #
    # This line is the difference between a working onset detector and a broken one. An
    # earlier version divided by a tiny epsilon instead, which turns the near-zero bins -
    # FFT leakage and float noise, not sound - into large log values whose frame-to-frame
    # wobble dominates the flux. A pure sine wave then produced a strong periodic envelope
    # and was confidently reported at 86 BPM. Clamping instead discards that noise floor.
    floor = peak * _NOISE_FLOOR
    log_magnitudes = np.log(np.maximum(magnitudes, floor))
    flux = np.diff(log_magnitudes, axis=0)
    envelope = np.maximum(flux, 0.0).sum(axis=1)

    # Prepend a zero so the envelope aligns frame-for-frame with the spectrogram.
    envelope = np.concatenate(([0.0], envelope)).astype(np.float32)

    peak = float(envelope.max())
    return (envelope / peak).astype(np.float32) if peak > 0.0 else envelope


def onset_crest(envelope: NDArray[np.float32]) -> float:
    """Peak-to-mean ratio of an onset envelope: how *sparse* it is.

    The discriminator between music and a sustained sound. A real onset envelope is mostly
    quiet with spikes where the beats are; a steady tone produces a smooth oscillation
    around its mean, from spectral leakage varying with the window phase. Measured on the
    synthetic signals in the test suite:

    ==================  ======
    signal              crest
    ==================  ======
    440 Hz sine         2.80
    1 kHz sine          2.17
    120 BPM clicks      22.59
    90 BPM clicks       30.39
    ==================  ======

    An order of magnitude apart, and scale-invariant, so it survives the envelope being
    normalised. Without this gate a pure sine autocorrelates beautifully and is reported at
    86 BPM with 0.96 confidence — periodicity is real, but it is the *analysis window's*
    periodicity, not the music's.
    """
    if envelope.size == 0:
        return 0.0
    mean = float(envelope.mean())
    return float(envelope.max()) / mean if mean > _EPSILON else 0.0


def estimate_tempo(
    envelope: NDArray[np.float32],
    *,
    sample_rate: int,
    bpm_min: float,
    bpm_max: float,
    min_confidence: float,
    min_crest: float = 4.0,
) -> TempoEstimate:
    """Estimate tempo by autocorrelating an onset envelope.

    Returns ``bpm=None`` when the envelope carries no beat-like structure
    (:func:`onset_crest`), or when the strongest periodicity in the search window is weaker
    than ``min_confidence``. Declining to answer is a feature: the director reads a BPM as
    a fact about the music and paces cuts against it.
    """
    import numpy as np

    frames_per_second = sample_rate / HOP_LENGTH
    if envelope.size < 4 or frames_per_second <= 0.0:
        return TempoEstimate(bpm=None, confidence=0.0)

    # Checked before autocorrelating, not after: a signal with no onsets can still
    # correlate almost perfectly with itself, so the confidence score cannot catch this.
    if onset_crest(envelope) < min_crest:
        return TempoEstimate(bpm=None, confidence=0.0)

    # Remove the mean first. Autocorrelating a strictly-positive signal makes every lag
    # correlate strongly with every other, and the true peak vanishes into the offset.
    centred = envelope - float(envelope.mean())
    energy = float(np.dot(centred, centred))
    if energy <= _EPSILON:
        return TempoEstimate(bpm=None, confidence=0.0)

    correlation = np.correlate(centred, centred, mode="full")[centred.size - 1 :]
    correlation = correlation / energy

    min_lag = max(1, round(frames_per_second * 60.0 / bpm_max))
    max_lag = min(correlation.size - 1, round(frames_per_second * 60.0 / bpm_min))
    if max_lag <= min_lag:
        return TempoEstimate(bpm=None, confidence=0.0)

    window = correlation[min_lag : max_lag + 1]
    best_offset = int(np.argmax(window))
    confidence = float(window[best_offset])
    if confidence < min_confidence:
        return TempoEstimate(bpm=None, confidence=max(0.0, confidence))

    lag = min_lag + best_offset
    bpm = 60.0 * frames_per_second / lag
    return TempoEstimate(
        bpm=_prefer_central_octave(bpm, bpm_min=bpm_min, bpm_max=bpm_max),
        confidence=min(1.0, confidence),
    )


def _prefer_central_octave(bpm: float, *, bpm_min: float, bpm_max: float) -> float:
    """Fold a tempo toward the middle of the search window by halving or doubling.

    Autocorrelation cannot distinguish 85 BPM from 170 BPM — both lags are genuinely
    periodic in the signal, and which one a listener taps is a musical judgement no
    correlation exposes. Choosing the octave nearest the window centre is a convention,
    not a measurement, and it is documented here rather than hidden so that a caller
    reading 170 knows 85 was equally consistent with the audio.
    """
    import math

    centre = math.sqrt(bpm_min * bpm_max)  # geometric, because octaves are multiplicative
    best = bpm
    for candidate in (bpm / 2.0, bpm, bpm * 2.0):
        if not bpm_min <= candidate <= bpm_max:
            continue
        if abs(math.log(candidate / centre)) < abs(math.log(best / centre)):
            best = candidate
    return round(best, 1)


def frame_rms(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    """Per-frame RMS level, on the same frame grid as :func:`spectrogram`."""
    import numpy as np

    if samples.size < FRAME_LENGTH:
        return np.zeros(0, dtype=np.float32)
    frame_count = 1 + (samples.size - FRAME_LENGTH) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(frame_count, FRAME_LENGTH),
        strides=(samples.strides[0] * HOP_LENGTH, samples.strides[0]),
        writeable=False,
    )
    return np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)).astype(np.float32)


def spectral_brightness(
    magnitudes: NDArray[np.float32], *, sample_rate: int, reference_hz: float
) -> float:
    """Mean spectral centroid, normalised against ``reference_hz`` and clamped to 0-1.

    The centroid is the energy-weighted average frequency: high for cymbals and strings,
    low for a bass-heavy pad. It stands in for "brightness" in the mood table, where it
    separates *uplifting* from *melancholic* at otherwise identical tempo and energy.
    """
    import numpy as np

    if magnitudes.shape[0] == 0 or reference_hz <= 0.0:
        return 0.0

    bin_frequencies = np.fft.rfftfreq(FRAME_LENGTH, d=1.0 / sample_rate).astype(np.float32)
    frame_energy = magnitudes.sum(axis=1)

    # Silent frames have no meaningful centroid; averaging their zero in would drag the
    # brightness of a track with long pauses toward nothing.
    voiced = frame_energy > _EPSILON
    if not bool(voiced.any()):
        return 0.0

    centroids = (magnitudes[voiced] * bin_frequencies).sum(axis=1) / frame_energy[voiced]
    mean_centroid = float(centroids.mean())
    return min(1.0, max(0.0, mean_centroid / reference_hz))


def find_intro_end(
    rms: NDArray[np.float32],
    *,
    sample_rate: int,
    level_ratio: float,
    max_fraction: float,
) -> float | None:
    """Where the opening quiet section ends, in seconds, or ``None``.

    Starting a music bed after a long ambient intro is the difference between a bed that
    supports the cut and eight seconds of apparent silence.

    Measured against the track's own median level rather than an absolute threshold: a
    quiet recording is not one long intro. ``max_fraction`` caps the answer for the same
    reason — an ambient piece that never rises above its median would otherwise report
    its entire length as intro.
    """
    import numpy as np

    if rms.size == 0:
        return None

    median = float(np.median(rms))
    if median <= _EPSILON:
        return None

    threshold = median * level_ratio
    above = np.flatnonzero(rms >= threshold)
    if above.size == 0 or above[0] == 0:
        # Either nothing reaches the threshold, or the track is already at level: both
        # mean there is no intro to skip.
        return None

    frames_per_second = sample_rate / HOP_LENGTH
    intro_end = float(above[0]) / frames_per_second
    limit = (rms.size / frames_per_second) * max_fraction
    return intro_end if intro_end <= limit else None


def to_db(amplitude: float) -> float:
    """Linear amplitude to dBFS, floored at -120 rather than diverging to -inf."""
    import math

    return 20.0 * math.log10(max(amplitude, _EPSILON)) if amplitude > 0.0 else -120.0


def extract(
    samples: NDArray[np.float32],
    *,
    sample_rate: int,
    bpm_min: float,
    bpm_max: float,
    bpm_min_confidence: float,
    bpm_min_crest: float,
    brightness_reference_hz: float,
    intro_level_ratio: float,
    intro_max_fraction: float,
) -> AudioFeatures:
    """Run the whole chain over one waveform."""
    import numpy as np

    magnitudes = spectrogram(samples)
    rms = frame_rms(samples)

    return AudioFeatures(
        duration=samples.size / sample_rate if sample_rate > 0 else 0.0,
        rms_db=to_db(float(np.sqrt((samples.astype(np.float64) ** 2).mean())))
        if samples.size
        else -120.0,
        peak_db=to_db(float(np.abs(samples).max())) if samples.size else -120.0,
        brightness=spectral_brightness(
            magnitudes, sample_rate=sample_rate, reference_hz=brightness_reference_hz
        ),
        tempo=estimate_tempo(
            onset_envelope(magnitudes),
            sample_rate=sample_rate,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            min_confidence=bpm_min_confidence,
            min_crest=bpm_min_crest,
        ),
        intro_end=find_intro_end(
            rms,
            sample_rate=sample_rate,
            level_ratio=intro_level_ratio,
            max_fraction=intro_max_fraction,
        ),
    )


__all__ = [
    "FEATURES_VERSION",
    "FRAME_LENGTH",
    "HOP_LENGTH",
    "AudioFeatures",
    "TempoEstimate",
    "estimate_tempo",
    "extract",
    "find_intro_end",
    "frame_rms",
    "onset_crest",
    "onset_envelope",
    "spectral_brightness",
    "spectrogram",
    "to_db",
]
