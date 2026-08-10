"""The footage analysis pipeline.

Composes the pieces into one clip-in, :class:`~app.models.video.ClipAnalysis`-out
operation. The composition itself is the interesting part, because this is the most
expensive thing AIVE does and the ordering is what keeps it tractable:

1. **Probe** the container - cheap, and its duration bounds everything after it.
2. **Detect scenes** in one pass. PySceneDetect reads the file sequentially, which is
   the fastest way to touch every frame.
3. **Per scene, seek to a few probe points** and read a frame *pair* at each. From the
   first frame of each pair: sharpness, exposure, faces. From the pair: motion. This is
   why there is no separate motion pass.
4. **Write keyframes** from the same probe points.
5. **Hash** the middle keyframe for duplicate detection.

The alternative - one pass per metric - would decode a 4K clip five times. Duplicate
detection runs last and across the whole project, because it is the only step that
compares scenes from *different* clips.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.analysis.vision.classical import ClassicalCvVisionProvider
from app.analysis.vision.duplicates import (
    HASH_VERSION,
    PerceptualDuplicateDetector,
    hash_keyframes,
)
from app.analysis.vision.frames import FramePair, FrameReadError, FrameSampler, downscale
from app.analysis.vision.metrics import METRICS_VERSION, OpenCvQualityAnalyzer
from app.analysis.vision.probe import ProbeError, PyAvProber
from app.analysis.vision.scenes import PySceneDetectDetector, SceneDetectionError
from app.config.settings import AiveSettings
from app.models.common import MediaRef, TimeRange
from app.models.video import ClipAnalysis, ClipFailure, FootageAnalysis, Keyframe, Scene
from app.services.paths import ProjectPaths
from app.utils.logging import get_logger, stage

logger = get_logger(__name__)

FOOTAGE_ANALYZER_VERSION = f"footage/1+{METRICS_VERSION}+{HASH_VERSION}"
"""Stamped into every :class:`ClipAnalysis`.

Composite, so a change to metric computation *or to the hash format* invalidates cached
clip analyses even though the orchestration is unchanged. Both matter: trusting stale
metrics would let the director choose shots on numbers that no longer mean what the
thresholds assume, and trusting a stale hash would silently disable duplicate detection.
"""


class DefaultFootageAnalyzer:
    """A :class:`~app.analysis.vision.base.FootageAnalyzer` over OpenCV and PySceneDetect."""

    def __init__(self, settings: AiveSettings, paths: ProjectPaths) -> None:
        self._settings = settings
        self._paths = paths
        vision = settings.vision
        self._prober = PyAvProber()
        self._scenes = PySceneDetectDetector(vision)
        self._sampler = FrameSampler(analysis_width=vision.downscale_width)
        self._metrics = OpenCvQualityAnalyzer(vision)
        self._vision = ClassicalCvVisionProvider(vision, project_root=paths.root)
        self._duplicates = PerceptualDuplicateDetector(
            similarity_threshold=settings.rules.duplicate_similarity
        )

    @property
    def version(self) -> str:
        return FOOTAGE_ANALYZER_VERSION

    # -- One clip ------------------------------------------------------------ #

    def analyze_clip(self, video: Path, *, ref: MediaRef) -> ClipAnalysis:
        """Probe, detect scenes, and measure every scene of one clip."""
        probe = self._prober.probe(video, ref=ref)
        if probe.video is None:
            msg = f"{video.name} has no video stream"
            raise ValueError(msg)

        scene_ranges = self._scenes.detect(video, probe=probe)
        with stage(logger, f"Measuring {len(scene_ranges)} scene(s) in {video.name}"):
            scenes = tuple(
                self._analyze_scene(video, ref=ref, index=index, scene=scene_range)
                for index, scene_range in enumerate(scene_ranges)
            )

        return ClipAnalysis(
            clip=ref,
            probe=probe,
            scenes=scenes,
            analyzer_version=self.version,
            analyzed_at=datetime.now(UTC),
        )

    def _analyze_scene(
        self,
        video: Path,
        *,
        ref: MediaRef,
        index: int,
        scene: TimeRange,
    ) -> Scene:
        """Measure one scene: sample once, derive everything from those frames."""
        vision = self._settings.vision
        pairs = self._sampler.sample_scene(video, scene=scene, count=vision.samples_per_scene)

        motion = self._metrics.measure_motion(pairs)
        stability = self._metrics.stability_from_motion(motion)
        quality = self._metrics.score_frames(pairs, stability=stability)

        keyframes = self._write_keyframes(ref=ref, index=index, pairs=pairs)
        absolute = [self._paths.resolve(frame.image) for frame in keyframes]

        return Scene(
            clip=ref,
            index=index,
            range=scene,
            keyframes=keyframes,
            quality=quality,
            motion=motion,
            tags=self._vision.describe(keyframes, scene=scene),
            shot_type=self._vision.shot_type_for(keyframes),
            phash=hash_keyframes(absolute),
        )

    def _write_keyframes(
        self,
        *,
        ref: MediaRef,
        index: int,
        pairs: list[FramePair],
    ) -> tuple[Keyframe, ...]:
        """Write stills for a scene and return portable references to them.

        Written from the frames already in memory rather than re-seeking the video: the
        probe points are exactly the frames wanted, so a second decode would be pure
        waste. The full-resolution copy on each pair is what makes that possible without
        also costing detail - ``keyframe_width`` is honoured against the original frame,
        not against the downscaled analysis copy.
        """
        import cv2

        vision = self._settings.vision
        wanted = min(len(pairs), vision.keyframes_per_scene)
        if wanted <= 0:
            return ()

        # Spread the chosen stills across the scene when more pairs were sampled than
        # keyframes are wanted, so a filmstrip shows the shot developing.
        step = max(1, len(pairs) // wanted)
        chosen = pairs[::step][:wanted]

        directory = self._paths.keyframes / ref.stem
        directory.mkdir(parents=True, exist_ok=True)

        keyframes: list[Keyframe] = []
        for position, pair in enumerate(chosen):
            destination = directory / f"{index:04d}_{position}.jpg"
            image = downscale(pair.still, vision.keyframe_width)
            try:
                written = cv2.imwrite(
                    str(destination),
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), vision.keyframe_quality],
                )
            except Exception as exc:
                logger.debug("Could not write %s: %s", destination, exc)
                continue
            if not written:
                continue
            keyframes.append(
                Keyframe(
                    timestamp=pair.timestamp,
                    image=self._paths.to_ref(destination),
                )
            )
        return tuple(keyframes)

    # -- Whole project ------------------------------------------------------- #

    def analyze_project(
        self,
        clips: list[Path],
        *,
        cached: dict[MediaRef, ClipAnalysis] | None = None,
    ) -> FootageAnalysis:
        """Analyse every clip, then find duplicates across all of them.

        ``cached`` lets unchanged clips be reused. Duplicate detection still runs over
        the full set, because adding one clip can create a duplicate of an old one -
        which is precisely the case a per-clip cache would miss.

        **A clip that cannot be analysed is recorded and skipped, not fatal.** Analysing
        forty 4K clips takes minutes; losing all of it because the thirty-eighth file is
        truncated would be an unreasonable way to fail. The failures are returned on the
        document so the user is told which files were dropped and why.
        """
        reusable = cached or {}
        analyses: list[ClipAnalysis] = []
        failures: list[ClipFailure] = []

        for position, video in enumerate(clips, start=1):
            ref = self._paths.to_ref(video)
            existing = reusable.get(ref)
            if existing is not None:
                logger.info("[%d/%d] %s (cached)", position, len(clips), ref)
                analyses.append(existing)
                continue

            logger.info("[%d/%d] %s", position, len(clips), ref)
            try:
                analyses.append(self.analyze_clip(video, ref=ref))
            except (ProbeError, SceneDetectionError, FrameReadError, ValueError, OSError) as exc:
                # Deliberately narrow: a dependency being absent or a genuine bug must
                # still abort, because those affect every clip rather than this one.
                logger.warning("Skipping %s: %s", ref, exc)
                failures.append(ClipFailure(clip=ref, error=f"{type(exc).__name__}: {exc}"))

        footage = FootageAnalysis(clips=tuple(analyses), failed=tuple(failures))
        return footage.model_copy(
            update={"duplicates": self._duplicates.find_duplicates(footage.scenes)}
        )


__all__ = ["FOOTAGE_ANALYZER_VERSION", "DefaultFootageAnalyzer"]
