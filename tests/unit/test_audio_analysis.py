"""Tests for music analysis.

The feature extractors are pure functions over numpy arrays, so they are tested against
**synthesised signals with known answers** — a click train at a stated tempo, a track with a
deliberately quiet opening — rather than against recordings whose truth nobody knows. That is
the only way to distinguish "the tempo estimator works" from "the tempo estimator returns a
number".

The same signals were run through the real pipeline during Phase 7 verification: a 120 BPM
click was reported as 120, and a six-second intro as 6.0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.analysis.audio.features import (
    HOP_LENGTH,
    estimate_tempo,
    extract,
    find_intro_end,
    frame_rms,
    onset_crest,
    onset_envelope,
    spectral_brightness,
    spectrogram,
    to_db,
)
from app.analysis.audio.loudness import parse_loudnorm
from app.analysis.audio.mood import classify, energy_score
from app.config.settings import MusicSettings
from app.models.common import MusicMood

SAMPLE_RATE = 22050


# --------------------------------------------------------------------------- #
# Signal builders - the "known answers"
# --------------------------------------------------------------------------- #


def _click_train(bpm: float, *, duration: float = 20.0, amplitude: float = 0.9) -> np.ndarray:
    """Impulses at a known tempo over a quiet tone, i.e. a signal with one true answer."""
    samples = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
    period = int(SAMPLE_RATE * 60.0 / bpm)
    burst = int(SAMPLE_RATE * 0.02)
    time = np.arange(burst) / SAMPLE_RATE
    click = (amplitude * np.sin(2 * math.pi * 2000 * time)).astype(np.float32)
    for start in range(0, samples.size - burst, period):
        samples[start : start + burst] += click
    # A quiet pad, so the track is not silence between clicks.
    pad_time = np.arange(samples.size) / SAMPLE_RATE
    samples += (0.05 * np.sin(2 * math.pi * 220 * pad_time)).astype(np.float32)
    return samples.astype(np.float32)


def _tone(frequency: float, *, duration: float = 5.0, amplitude: float = 0.5) -> np.ndarray:
    time = np.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * math.pi * frequency * time)).astype(np.float32)


def _quiet_then_loud(intro: float, *, total: float = 20.0) -> np.ndarray:
    samples = _tone(440.0, duration=total, amplitude=1.0)
    boundary = int(intro * SAMPLE_RATE)
    samples[:boundary] *= 0.05
    samples[boundary:] *= 0.8
    return samples


# --------------------------------------------------------------------------- #
# Spectrogram and onsets
# --------------------------------------------------------------------------- #


class TestSpectrogram:
    def test_a_short_signal_yields_no_frames_rather_than_raising(self) -> None:
        assert spectrogram(np.zeros(10, dtype=np.float32)).shape[0] == 0

    def test_frame_count_follows_the_hop(self) -> None:
        samples = _tone(440.0, duration=1.0)
        expected = 1 + (samples.size - 1024) // HOP_LENGTH
        assert spectrogram(samples).shape[0] == expected

    def test_a_pure_tone_peaks_in_the_right_bin(self) -> None:
        magnitudes = spectrogram(_tone(1000.0, duration=1.0))
        peak_bin = int(np.argmax(magnitudes.mean(axis=0)))
        peak_hz = peak_bin * SAMPLE_RATE / 1024
        assert peak_hz == pytest.approx(1000.0, abs=40.0)


class TestOnsetEnvelope:
    def test_it_is_normalised(self) -> None:
        envelope = onset_envelope(spectrogram(_click_train(120.0, duration=5.0)))
        assert envelope.max() == pytest.approx(1.0)

    def test_a_steady_tone_produces_no_beat_like_structure(self) -> None:
        """A sustained tone still wobbles frame to frame - spectral leakage varying with
        the window phase - so the envelope is not flat. What distinguishes it from music is
        that the wobble is *smooth* rather than spiky; see onset_crest."""
        assert onset_crest(onset_envelope(spectrogram(_tone(440.0, duration=20.0)))) < 4.0

    def test_a_click_train_is_sparse(self) -> None:
        assert onset_crest(onset_envelope(spectrogram(_click_train(120.0)))) > 10.0

    def test_crest_of_an_empty_envelope_is_zero(self) -> None:
        assert onset_crest(np.zeros(0, dtype=np.float32)) == 0.0

    def test_crest_of_a_silent_envelope_is_zero(self) -> None:
        assert onset_crest(np.zeros(100, dtype=np.float32)) == 0.0

    def test_too_few_frames_returns_zeros_rather_than_raising(self) -> None:
        assert onset_envelope(np.zeros((1, 513), dtype=np.float32)).size == 1


# --------------------------------------------------------------------------- #
# Tempo - the part with a real right answer
# --------------------------------------------------------------------------- #


class TestTempo:
    @pytest.mark.parametrize("bpm", [90.0, 120.0, 140.0])
    def test_a_click_train_is_measured_at_its_true_tempo(self, bpm: float) -> None:
        estimate = estimate_tempo(
            onset_envelope(spectrogram(_click_train(bpm))),
            sample_rate=SAMPLE_RATE,
            bpm_min=60.0,
            bpm_max=190.0,
            min_confidence=0.15,
        )
        assert estimate.bpm is not None
        assert estimate.bpm == pytest.approx(bpm, rel=0.04)

    @pytest.mark.parametrize("frequency", [440.0, 1000.0])
    def test_a_steady_tone_yields_no_tempo(self, frequency: float) -> None:
        """Declining to answer is the feature: a wrong BPM would be paced against.

        Regression: before the crest gate this returned 86.1 BPM at 0.96 confidence. The
        periodicity was real - it was the analysis window's, not the music's.
        """
        estimate = estimate_tempo(
            onset_envelope(spectrogram(_tone(frequency, duration=20.0))),
            sample_rate=SAMPLE_RATE,
            bpm_min=60.0,
            bpm_max=190.0,
            min_confidence=0.15,
        )
        assert estimate.bpm is None

    def test_a_lowered_crest_gate_lets_the_spurious_tempo_back_through(self) -> None:
        """Pins *why* the gate is what stops it, rather than some other clamp."""
        estimate = estimate_tempo(
            onset_envelope(spectrogram(_tone(440.0, duration=20.0))),
            sample_rate=SAMPLE_RATE,
            bpm_min=60.0,
            bpm_max=190.0,
            min_confidence=0.15,
            min_crest=1.0,
        )
        assert estimate.bpm is not None

    def test_the_confidence_threshold_is_honoured(self) -> None:
        envelope = onset_envelope(spectrogram(_click_train(120.0)))
        assert (
            estimate_tempo(
                envelope,
                sample_rate=SAMPLE_RATE,
                bpm_min=60.0,
                bpm_max=190.0,
                min_confidence=0.999,
            ).bpm
            is None
        )

    def test_an_empty_envelope_is_handled(self) -> None:
        estimate = estimate_tempo(
            np.zeros(0, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
            bpm_min=60.0,
            bpm_max=190.0,
            min_confidence=0.1,
        )
        assert estimate.bpm is None
        assert estimate.confidence == 0.0

    def test_a_silent_envelope_is_handled(self) -> None:
        estimate = estimate_tempo(
            np.zeros(500, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
            bpm_min=60.0,
            bpm_max=190.0,
            min_confidence=0.1,
        )
        assert estimate.bpm is None

    def test_the_result_stays_inside_the_search_window(self) -> None:
        estimate = estimate_tempo(
            onset_envelope(spectrogram(_click_train(150.0))),
            sample_rate=SAMPLE_RATE,
            bpm_min=100.0,
            bpm_max=180.0,
            min_confidence=0.05,
        )
        assert estimate.bpm is not None
        assert 100.0 <= estimate.bpm <= 180.0

    def test_an_inverted_window_is_refused_rather_than_crashing(self) -> None:
        estimate = estimate_tempo(
            onset_envelope(spectrogram(_click_train(120.0))),
            sample_rate=SAMPLE_RATE,
            bpm_min=189.0,
            bpm_max=190.0,
            min_confidence=0.1,
        )
        assert estimate.bpm is None


# --------------------------------------------------------------------------- #
# Level, brightness, intro
# --------------------------------------------------------------------------- #


class TestLevelAndBrightness:
    def test_rms_tracks_amplitude(self) -> None:
        loud = frame_rms(_tone(440.0, amplitude=0.8)).mean()
        quiet = frame_rms(_tone(440.0, amplitude=0.1)).mean()
        assert loud > quiet

    def test_a_short_signal_yields_no_frames(self) -> None:
        assert frame_rms(np.zeros(10, dtype=np.float32)).size == 0

    def test_a_high_tone_is_brighter_than_a_low_one(self) -> None:
        high = spectral_brightness(
            spectrogram(_tone(6000.0)), sample_rate=SAMPLE_RATE, reference_hz=3500.0
        )
        low = spectral_brightness(
            spectrogram(_tone(200.0)), sample_rate=SAMPLE_RATE, reference_hz=3500.0
        )
        assert high > low

    def test_brightness_is_clamped_to_one(self) -> None:
        brightness = spectral_brightness(
            spectrogram(_tone(9000.0)), sample_rate=SAMPLE_RATE, reference_hz=100.0
        )
        assert brightness == 1.0

    def test_silence_has_no_brightness(self) -> None:
        """Averaging silent frames in would drag a track with long pauses toward zero."""
        silence = spectrogram(np.zeros(SAMPLE_RATE, dtype=np.float32))
        assert spectral_brightness(silence, sample_rate=SAMPLE_RATE, reference_hz=3500.0) == 0.0

    def test_to_db_floors_rather_than_diverging(self) -> None:
        assert to_db(0.0) == -120.0
        assert to_db(1.0) == pytest.approx(0.0)
        assert to_db(0.5) == pytest.approx(-6.02, abs=0.01)


class TestIntroDetection:
    def test_a_quiet_opening_is_found_at_the_right_moment(self) -> None:
        """Verified end to end: a synthesised 6 s intro was reported as 6.0."""
        rms = frame_rms(_quiet_then_loud(6.0))
        intro = find_intro_end(rms, sample_rate=SAMPLE_RATE, level_ratio=0.55, max_fraction=0.5)
        assert intro is not None
        assert intro == pytest.approx(6.0, abs=0.2)

    def test_a_track_already_at_level_has_no_intro(self) -> None:
        rms = frame_rms(_tone(440.0, duration=20.0, amplitude=0.8))
        assert (
            find_intro_end(rms, sample_rate=SAMPLE_RATE, level_ratio=0.55, max_fraction=0.5) is None
        )

    def test_an_intro_longer_than_the_cap_is_refused(self) -> None:
        """An ambient piece is quiet throughout; without the cap it is all intro."""
        rms = frame_rms(_quiet_then_loud(15.0, total=20.0))
        assert (
            find_intro_end(rms, sample_rate=SAMPLE_RATE, level_ratio=0.55, max_fraction=0.2) is None
        )

    def test_silence_has_no_intro(self) -> None:
        rms = frame_rms(np.zeros(SAMPLE_RATE * 5, dtype=np.float32))
        assert (
            find_intro_end(rms, sample_rate=SAMPLE_RATE, level_ratio=0.55, max_fraction=0.5) is None
        )

    def test_an_empty_signal_is_handled(self) -> None:
        empty = np.zeros(0, dtype=np.float32)
        assert (
            find_intro_end(empty, sample_rate=SAMPLE_RATE, level_ratio=0.5, max_fraction=0.5)
            is None
        )


# --------------------------------------------------------------------------- #
# The whole chain
# --------------------------------------------------------------------------- #


class TestExtract:
    def _extract(self, samples: np.ndarray):
        return extract(
            samples,
            sample_rate=SAMPLE_RATE,
            bpm_min=60.0,
            bpm_max=190.0,
            bpm_min_confidence=0.15,
            bpm_min_crest=4.0,
            brightness_reference_hz=3500.0,
            intro_level_ratio=0.55,
            intro_max_fraction=0.35,
        )

    def test_it_reports_every_field(self) -> None:
        features = self._extract(_click_train(120.0))
        assert features.duration == pytest.approx(20.0, abs=0.1)
        assert features.tempo.bpm == pytest.approx(120.0, rel=0.04)
        assert -60.0 < features.rms_db < 0.0
        assert 0.0 <= features.brightness <= 1.0

    def test_an_empty_signal_does_not_raise(self) -> None:
        features = self._extract(np.zeros(0, dtype=np.float32))
        assert features.duration == 0.0
        assert features.rms_db == -120.0
        assert features.tempo.bpm is None


# --------------------------------------------------------------------------- #
# Energy and mood
# --------------------------------------------------------------------------- #


class TestEnergyScore:
    def test_the_floor_and_ceiling_map_to_zero_and_one(self) -> None:
        settings = MusicSettings(energy_rms_weight=1.0)
        assert energy_score(rms_db=-40.0, brightness=0.0, settings=settings) == 0.0
        assert energy_score(rms_db=-10.0, brightness=0.0, settings=settings) == 1.0

    def test_it_is_clamped_outside_the_range(self) -> None:
        settings = MusicSettings(energy_rms_weight=1.0)
        assert energy_score(rms_db=-90.0, brightness=0.0, settings=settings) == 0.0
        assert energy_score(rms_db=0.0, brightness=0.0, settings=settings) == 1.0

    def test_brightness_contributes_by_its_configured_weight(self) -> None:
        """A loud bass drone and a loud string section are not equally intense."""
        settings = MusicSettings(energy_rms_weight=0.5)
        dull = energy_score(rms_db=-25.0, brightness=0.0, settings=settings)
        bright = energy_score(rms_db=-25.0, brightness=1.0, settings=settings)
        assert bright > dull
        assert bright - dull == pytest.approx(0.5, abs=0.01)


class TestMoodTable:
    SETTINGS = MusicSettings()

    def test_quiet_and_dark_reads_melancholic(self) -> None:
        moods = classify(energy=0.2, brightness=0.2, bpm=70.0, settings=self.SETTINGS)
        assert moods[0] is MusicMood.CALM
        assert MusicMood.MELANCHOLIC in moods

    def test_quiet_and_bright_reads_uplifting(self) -> None:
        moods = classify(energy=0.2, brightness=0.9, bpm=70.0, settings=self.SETTINGS)
        assert MusicMood.UPLIFTING in moods

    def test_loud_fast_and_bright_reads_energetic(self) -> None:
        moods = classify(energy=0.9, brightness=0.8, bpm=150.0, settings=self.SETTINGS)
        assert moods[0] is MusicMood.ENERGETIC

    def test_loud_slow_and_dark_reads_tense(self) -> None:
        moods = classify(energy=0.9, brightness=0.2, bpm=70.0, settings=self.SETTINGS)
        assert MusicMood.TENSE in moods

    def test_a_missing_tempo_still_produces_moods(self) -> None:
        """Energy and brightness alone still separate calm from energetic."""
        assert classify(energy=0.9, brightness=0.9, bpm=None, settings=self.SETTINGS)

    def test_at_least_two_moods_are_offered(self) -> None:
        """A boundary track is described more truthfully by a pair than by a pick."""
        for energy in (0.1, 0.5, 0.9):
            moods = classify(energy=energy, brightness=0.5, bpm=110.0, settings=self.SETTINGS)
            assert len(moods) >= 2

    def test_no_mood_is_repeated(self) -> None:
        """A duplicate would read as extra confidence rather than one opinion."""
        for energy in (0.1, 0.3, 0.5, 0.7, 0.9):
            for brightness in (0.1, 0.6, 0.9):
                moods = classify(
                    energy=energy, brightness=brightness, bpm=120.0, settings=self.SETTINGS
                )
                assert len(moods) == len(set(moods))

    def test_thresholds_come_from_config(self) -> None:
        loose = MusicSettings(mood_calm_energy=0.8, mood_energetic_energy=0.9)
        assert classify(energy=0.5, brightness=0.5, bpm=100.0, settings=loose)[0] is MusicMood.CALM


# --------------------------------------------------------------------------- #
# Loudness parsing - a format we do not control
# --------------------------------------------------------------------------- #


class TestParseLoudnorm:
    REPORT = """
[Parsed_loudnorm_0 @ 000001]
{
	"input_i" : "-23.45",
	"input_tp" : "-2.10",
	"input_lra" : "7.20",
	"input_thresh" : "-33.50",
	"output_i" : "-24.00",
	"normalization_type" : "dynamic"
}
"""

    def test_it_finds_the_integrated_value(self) -> None:
        assert parse_loudnorm(self.REPORT) == -23.45

    def test_output_without_a_json_block_returns_none(self) -> None:
        assert parse_loudnorm("ffmpeg version 7.1\nno json here") is None

    def test_a_malformed_block_returns_none_rather_than_raising(self) -> None:
        assert parse_loudnorm('{ "input_i" : oops }') is None

    def test_digital_silence_is_treated_as_unmeasurable(self) -> None:
        """-inf and anything under the R128 absolute gate carry no programme material."""
        assert parse_loudnorm('{"input_i" : "-inf"}') is None
        assert parse_loudnorm('{"input_i" : "-99.0"}') is None

    def test_a_missing_field_returns_none(self) -> None:
        assert parse_loudnorm('{"input_tp" : "-2.0"}') is None

    def test_the_value_is_rounded_to_two_places(self) -> None:
        assert parse_loudnorm('{"input_i" : "-18.456789"}') == -18.46
