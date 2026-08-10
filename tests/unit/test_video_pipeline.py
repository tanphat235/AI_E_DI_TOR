"""End-to-end video analysis against real, generated H.264 files.

Marked ``integration`` and deselected by default, because these decode video and take a
few seconds. They earn their keep by asserting on *known* properties: the dark clip must
read as dark, the panning clip as a pan, the duplicated red clip as a duplicate. Unit
tests with synthetic arrays cannot catch a codec, seek or orientation problem, and those
are exactly the failures that reach users.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from app.analysis.vision.analyzer import DefaultFootageAnalyzer
from app.analysis.vision.frames import FrameSampler, open_capture
from app.analysis.vision.probe import PyAvProber, read_rotation
from app.analysis.vision.scenes import PySceneDetectDetector
from app.config.settings import AiveSettings, load_settings
from app.models.common import TimeRange
from app.models.video import FootageAnalysis
from app.services.paths import ProjectPaths

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(video_project: ProjectPaths) -> AiveSettings:
    return load_settings(video_project.root)


@pytest.fixture
def analyzer(settings: AiveSettings, video_project: ProjectPaths) -> DefaultFootageAnalyzer:
    return DefaultFootageAnalyzer(settings, video_project)


class TestProbing:
    def test_container_facts_are_read(self, video_project: ProjectPaths) -> None:
        clip = video_project.raw / "sharp.mp4"
        probe = PyAvProber().probe(clip, ref=video_project.to_ref(clip))
        assert probe.duration == pytest.approx(2.0, abs=0.2)
        assert probe.video is not None
        assert probe.video.width == 640
        assert probe.video.height == 360
        assert probe.video.fps == pytest.approx(25.0, abs=0.1)
        assert probe.video.codec == "h264"
        assert probe.size_bytes > 0

    def test_a_silent_clip_reports_no_audio(self, video_project: ProjectPaths) -> None:
        """Assuming every clip has audio is a classic renderer failure."""
        clip = video_project.raw / "sharp.mp4"
        probe = PyAvProber().probe(clip, ref=video_project.to_ref(clip))
        assert probe.has_audio is False
        assert probe.has_video is True

    def test_a_rotation_flag_is_read_and_swaps_the_display_size(
        self, video_project: ProjectPaths
    ) -> None:
        """The case that made aspect-ratio inference unusable.

        This file reports a 16:9 display aspect despite carrying a 90-degree flag, so any
        approach based on comparing aspects misses it entirely.
        """
        clip = video_project.raw / "rotated.mp4"
        assert read_rotation(clip) in {90, 270}

        probe = PyAvProber().probe(clip, ref=video_project.to_ref(clip))
        assert probe.video is not None
        assert probe.video.rotation in {90, 270}
        # Encoded landscape, displayed portrait.
        assert probe.video.width > probe.video.height
        assert probe.video.display_size == (probe.video.height, probe.video.width)

    def test_an_unrotated_clip_reports_zero(self, video_project: ProjectPaths) -> None:
        assert read_rotation(video_project.raw / "sharp.mp4") == 0


class TestFrameReading:
    def test_frames_come_back_display_oriented(self, video_project: ProjectPaths) -> None:
        """A rotated clip must be measured the way a viewer sees it, not sideways."""
        sampler = FrameSampler(analysis_width=640)
        frame = sampler.sample_frame(video_project.raw / "rotated.mp4", timestamp=1.0)
        assert frame is not None
        height, width = frame.shape[:2]
        assert height > width, "rotated clip should decode as portrait"

    def test_a_frame_pair_is_read_at_each_probe_point(self, video_project: ProjectPaths) -> None:
        sampler = FrameSampler(analysis_width=320)
        pairs = sampler.sample_scene(
            video_project.raw / "panning.mp4",
            scene=TimeRange(start=0.0, end=3.0),
            count=3,
        )
        assert len(pairs) == 3
        assert all(pair.has_pair for pair in pairs)
        # Downscaled on the way out.
        assert pairs[0].first.shape[1] == 320

    def test_an_unopenable_file_raises(self, tmp_path: Path) -> None:
        from app.analysis.vision.frames import FrameReadError

        junk = tmp_path / "junk.mp4"
        junk.write_text("not a video", encoding="utf-8")
        with pytest.raises(FrameReadError, match="could not open"), open_capture(junk):
            pass


class TestSceneDetection:
    def test_three_distinct_scenes_are_found(
        self, video_project: ProjectPaths, settings: AiveSettings
    ) -> None:
        clip = video_project.raw / "three_scenes.mp4"
        probe = PyAvProber().probe(clip, ref=video_project.to_ref(clip))
        scenes = PySceneDetectDetector(settings.vision).detect(clip, probe=probe)
        assert len(scenes) == 3

    def test_detected_scenes_tile_the_whole_clip(
        self, video_project: ProjectPaths, settings: AiveSettings
    ) -> None:
        """A gap would make footage invisible to the director."""
        clip = video_project.raw / "three_scenes.mp4"
        probe = PyAvProber().probe(clip, ref=video_project.to_ref(clip))
        scenes = PySceneDetectDetector(settings.vision).detect(clip, probe=probe)
        assert scenes[0].start == 0.0
        assert scenes[-1].end == pytest.approx(probe.duration, abs=0.1)
        for earlier, later in itertools.pairwise(scenes):
            assert earlier.end == later.start

    def test_a_single_take_is_one_scene(
        self, video_project: ProjectPaths, settings: AiveSettings
    ) -> None:
        clip = video_project.raw / "dark.mp4"
        probe = PyAvProber().probe(clip, ref=video_project.to_ref(clip))
        assert len(PySceneDetectDetector(settings.vision).detect(clip, probe=probe)) == 1


class TestQualityOnRealFootage:
    def test_a_sharp_clip_scores_far_above_a_blurred_one(
        self, analyzer: DefaultFootageAnalyzer, video_project: ProjectPaths
    ) -> None:
        """The same source, one heavily blurred. This is the headline quality claim."""
        sharp = analyzer.analyze_clip(
            video_project.raw / "sharp.mp4",
            ref=video_project.to_ref(video_project.raw / "sharp.mp4"),
        )
        blurred = analyzer.analyze_clip(
            video_project.raw / "blurred.mp4",
            ref=video_project.to_ref(video_project.raw / "blurred.mp4"),
        )
        sharp_blur = sharp.scenes[0].quality.blur
        blurred_blur = blurred.scenes[0].quality.blur
        assert sharp_blur > 0.8
        assert blurred_blur < 0.2
        assert sharp.scenes[0].quality.overall > blurred.scenes[0].quality.overall

    def test_the_raw_variance_is_recorded(
        self, analyzer: DefaultFootageAnalyzer, video_project: ProjectPaths
    ) -> None:
        clip = video_project.raw / "sharp.mp4"
        analysis = analyzer.analyze_clip(clip, ref=video_project.to_ref(clip))
        variance = analysis.scenes[0].quality.blur_variance
        assert variance is not None
        assert variance > 100.0

    def test_a_dark_clip_scores_low_on_brightness(
        self, analyzer: DefaultFootageAnalyzer, video_project: ProjectPaths
    ) -> None:
        clip = video_project.raw / "dark.mp4"
        analysis = analyzer.analyze_clip(clip, ref=video_project.to_ref(clip))
        assert analysis.scenes[0].quality.brightness < 0.25

    def test_a_pan_is_recognised(
        self, analyzer: DefaultFootageAnalyzer, video_project: ProjectPaths
    ) -> None:
        from app.models.common import CameraMove

        clip = video_project.raw / "panning.mp4"
        analysis = analyzer.analyze_clip(clip, ref=video_project.to_ref(clip))
        moves = {scene.motion.camera_move for scene in analysis.scenes}
        assert CameraMove.PAN in moves

    def test_a_pan_stays_stable(
        self, analyzer: DefaultFootageAnalyzer, video_project: ProjectPaths
    ) -> None:
        """Smooth movement is steady footage; only incoherent movement is shake."""
        clip = video_project.raw / "panning.mp4"
        analysis = analyzer.analyze_clip(clip, ref=video_project.to_ref(clip))
        assert analysis.scenes[0].quality.stability > 0.6


class TestKeyframes:
    def test_stills_are_written_and_referenced_portably(
        self, analyzer: DefaultFootageAnalyzer, video_project: ProjectPaths
    ) -> None:
        clip = video_project.raw / "three_scenes.mp4"
        analysis = analyzer.analyze_clip(clip, ref=video_project.to_ref(clip))
        for scene in analysis.scenes:
            assert scene.keyframes
            for keyframe in scene.keyframes:
                resolved = video_project.resolve(keyframe.image)
                assert resolved.is_file()
                assert resolved.stat().st_size > 0
                # Portable: relative, POSIX, inside the cache.
                assert not keyframe.image.path.is_absolute()
                assert keyframe.image.path.as_posix().startswith(".aive/keyframes/")

    def test_keyframe_timestamps_fall_inside_their_scene(
        self, analyzer: DefaultFootageAnalyzer, video_project: ProjectPaths
    ) -> None:
        clip = video_project.raw / "three_scenes.mp4"
        analysis = analyzer.analyze_clip(clip, ref=video_project.to_ref(clip))
        for scene in analysis.scenes:
            for keyframe in scene.keyframes:
                assert scene.range.contains(keyframe.timestamp)


class TestWholeProject:
    def test_every_clip_is_analysed_and_duplicates_found(
        self, analyzer: DefaultFootageAnalyzer, video_project: ProjectPaths
    ) -> None:
        clips = sorted(video_project.find_raw_clips())
        footage = analyzer.analyze_project(clips)

        assert len(footage.clips) == len(clips)
        assert footage.scenes

        # red_again.mp4 is byte-for-byte the same shot as the first scene of
        # three_scenes.mp4, so exactly one of them must be suppressed.
        suppressed = footage.suppressed_scene_keys
        assert suppressed, "the duplicated red clip should have been detected"
        assert any("red_again" in key or "three_scenes" in key for key in suppressed)

    def test_differently_coloured_flat_scenes_are_not_grouped(
        self, analyzer: DefaultFootageAnalyzer, video_project: ProjectPaths
    ) -> None:
        """The regression that motivated adding colour to the hash.

        three_scenes.mp4 opens with a red scene and a green scene. With a structure-only
        hash both were all-zeros and got grouped, suppressing unrelated footage.
        """
        clip = video_project.raw / "three_scenes.mp4"
        footage = analyzer.analyze_project([clip])
        red, green = footage.clips[0].scenes[0], footage.clips[0].scenes[1]
        assert red.phash != green.phash
        assert green.key not in footage.suppressed_scene_keys

    def test_the_analysis_round_trips_through_json(
        self, analyzer: DefaultFootageAnalyzer, video_project: ProjectPaths
    ) -> None:
        """It is written to the cache, so it must survive serialisation exactly."""
        clips = [video_project.raw / "sharp.mp4", video_project.raw / "dark.mp4"]
        footage = analyzer.analyze_project(clips)
        restored = FootageAnalysis.model_validate_json(footage.model_dump_json())
        assert restored == footage

    def test_cached_clips_are_reused_verbatim(
        self, analyzer: DefaultFootageAnalyzer, video_project: ProjectPaths
    ) -> None:
        clip = video_project.raw / "sharp.mp4"
        ref = video_project.to_ref(clip)
        first = analyzer.analyze_project([clip])
        cached = {ref: first.clips[0]}
        second = analyzer.analyze_project([clip], cached=cached)
        assert second.clips[0] is first.clips[0]

    def test_a_clip_with_no_video_stream_is_rejected(
        self, analyzer: DefaultFootageAnalyzer, video_project: ProjectPaths, tmp_path: Path
    ) -> None:
        from app.analysis.vision.probe import ProbeError

        junk = video_project.raw / "broken.mp4"
        junk.write_bytes(b"\0" * 4096)
        with pytest.raises((ValueError, ProbeError)):
            analyzer.analyze_clip(junk, ref=video_project.to_ref(junk))
