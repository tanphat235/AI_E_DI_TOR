"""Shot boundary detection via PySceneDetect.

PySceneDetect rather than a hand-rolled histogram comparison: cut detection is a solved
problem, its content-aware detector is well tuned, and reimplementing it would be effort
spent on the least differentiated part of the pipeline.

Two things this module adds on top, and both matter more than they look:

**Short detections are merged, not emitted.** A 0.2-second "scene" is a detector
artefact - a camera flash, a fast whip pan, one corrupt frame. Passing it downstream
would give the director a shot too short to cut to, and would pollute duplicate
detection with near-identical fragments.

**The result always tiles the whole clip.** PySceneDetect returns cut points; this
returns contiguous ranges covering the file end to end, so no footage is silently
invisible to the director. A clip with no detected cuts is one scene, not zero.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from app.analysis.vision.probe import VideoDependencyMissingError
from app.config.settings import VisionSettings
from app.models.common import TimeRange
from app.models.media import MediaProbe
from app.utils.logging import get_logger, stage

logger = get_logger(__name__)

DETECTOR_NAME = "pyscenedetect"

SUPPORTED_DETECTORS = ("content", "adaptive", "threshold")
"""Algorithms exposed through config.

``content`` compares frame-to-frame HSV change and is the right default for edited
footage. ``adaptive`` normalises that against a rolling window, which handles fast
camera movement better and is worth trying on handheld or drone material.
``threshold`` detects fades to black, useful only on footage that already has them.
"""


class SceneDetectionError(RuntimeError):
    """Raised when scene detection could not run."""


class PySceneDetectDetector:
    """A :class:`~app.analysis.vision.base.SceneDetector` backed by PySceneDetect."""

    def __init__(self, settings: VisionSettings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return f"{DETECTOR_NAME}/{self._settings.scene_detector}"

    def detect(self, video: Path, *, probe: MediaProbe) -> tuple[TimeRange, ...]:
        """Return contiguous scene ranges covering the whole clip."""
        if not video.is_file():
            msg = f"video not found: {video}"
            raise FileNotFoundError(msg)

        cuts = self._detect_cuts(video, probe)
        scenes = ranges_from_cuts(cuts, duration=probe.duration)
        merged = merge_short_scenes(scenes, minimum=self._settings.min_scene_duration)
        logger.debug(
            "%s: %d cut(s) -> %d scene(s) after merging",
            video.name,
            len(cuts),
            len(merged),
        )
        return merged

    def _detect_cuts(self, video: Path, probe: MediaProbe) -> list[float]:
        """Raw cut timestamps, in seconds."""
        try:
            from scenedetect import AdaptiveDetector, ContentDetector, ThresholdDetector, detect
        except ImportError as exc:
            raise VideoDependencyMissingError("scenedetect") from exc

        choice = self._settings.scene_detector.strip().lower()
        if choice not in SUPPORTED_DETECTORS:
            msg = f"unknown scene_detector {choice!r}; available: {', '.join(SUPPORTED_DETECTORS)}"
            raise SceneDetectionError(msg)

        # min_scene_len is expressed in frames, so the configured duration has to be
        # converted. Passing it to the detector as well as merging afterwards is not
        # redundant: the detector suppresses the cut, which stops a rapid sequence of
        # them from fragmenting a shot in the first place.
        fps = probe.video.fps if probe.video is not None else 0.0
        min_frames = max(1, round(self._settings.min_scene_duration * fps)) if fps > 0 else 1

        detector: Any
        if choice == "content":
            detector = ContentDetector(
                threshold=self._settings.scene_threshold, min_scene_len=min_frames
            )
        elif choice == "adaptive":
            detector = AdaptiveDetector(min_scene_len=min_frames)
        else:
            detector = ThresholdDetector(min_scene_len=min_frames)

        try:
            with stage(logger, f"Detecting scenes in {video.name}"):
                found = detect(str(video), detector, show_progress=False)
        except Exception as exc:
            msg = f"scene detection failed on {video.name}: {type(exc).__name__}: {exc}"
            raise SceneDetectionError(msg) from exc

        # PySceneDetect returns (start, end) FrameTimecode pairs; only the boundaries
        # between them are cuts, so the first start and the last end are dropped.
        return [_seconds(start) for start, _end in found][1:]


def _seconds(timecode: Any) -> float:
    """Seconds from a PySceneDetect ``FrameTimecode``.

    Prefers the ``seconds`` property and falls back to the deprecated ``get_seconds()``,
    so the module works across the 0.6 and 0.7 lines rather than pinning one.
    """
    value = getattr(timecode, "seconds", None)
    if value is not None:
        return float(value)
    return float(timecode.get_seconds())


def ranges_from_cuts(cuts: list[float], *, duration: float) -> tuple[TimeRange, ...]:
    """Turn cut points into contiguous ranges spanning ``[0, duration)``.

    Out-of-range and duplicate cuts are discarded rather than trusted: a cut at or past
    the duration would produce a zero-length range, which no model will accept.
    """
    if duration <= 0.0:
        return ()

    boundaries = sorted({0.0, *(cut for cut in cuts if 0.0 < cut < duration), duration})
    return tuple(
        TimeRange(start=start, end=end)
        for start, end in itertools.pairwise(boundaries)
        if end > start
    )


def merge_short_scenes(scenes: tuple[TimeRange, ...], *, minimum: float) -> tuple[TimeRange, ...]:
    """Absorb scenes shorter than ``minimum`` into a neighbour.

    Merges forward - a short scene joins the one that follows - so the *start* of a
    kept scene stays exactly on a real cut. Starting a shot mid-action is far more
    visible than ending one slightly late.

    The final scene is a special case: with nothing after it, it merges backwards
    instead. And if every scene is too short the whole clip becomes one scene, which is
    the honest answer for footage that is one continuous take.
    """
    if not scenes:
        return ()

    merged: list[TimeRange] = []
    pending_start: float | None = None

    for scene in scenes:
        start = pending_start if pending_start is not None else scene.start
        candidate = TimeRange(start=start, end=scene.end)
        if candidate.duration < minimum:
            pending_start = start
            continue
        merged.append(candidate)
        pending_start = None

    if pending_start is not None:
        # A short tail. Extend the previous scene over it rather than emitting it.
        tail_end = scenes[-1].end
        if merged:
            last = merged[-1]
            merged[-1] = TimeRange(start=last.start, end=tail_end)
        else:
            merged.append(TimeRange(start=pending_start, end=tail_end))

    return tuple(merged)


__all__ = [
    "DETECTOR_NAME",
    "SUPPORTED_DETECTORS",
    "PySceneDetectDetector",
    "SceneDetectionError",
    "merge_short_scenes",
    "ranges_from_cuts",
]
