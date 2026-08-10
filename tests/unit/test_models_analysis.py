"""Tests for the analysis models: media probes, speech, video, audio, manifest."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.models.audio import MusicLibrary, MusicTrack
from app.models.common import (
    CameraMove,
    MediaKind,
    MediaRef,
    MotionLevel,
    MusicMood,
    ShotType,
    TimeRange,
)
from app.models.media import AudioStreamInfo, MediaProbe, VideoStreamInfo
from app.models.project import MediaEntry, ProjectManifest
from app.models.speech import (
    SilenceSpan,
    SpeechCleanupReport,
    Transcript,
    TranscriptSegment,
    Word,
)
from app.models.video import (
    ClipAnalysis,
    DuplicateGroup,
    FootageAnalysis,
    MotionStats,
    QualityScores,
    Scene,
    SceneTags,
)


def _quality(overall: float = 0.8) -> QualityScores:
    return QualityScores(blur=0.8, brightness=0.7, exposure=0.75, stability=0.9, overall=overall)


def _motion() -> MotionStats:
    return MotionStats(level=MotionLevel.MEDIUM, mean_magnitude=2.5, camera_move=CameraMove.PAN)


def _tags(provider: str = "classical_cv") -> SceneTags:
    return SceneTags(provider=provider, people_count=1, confidence=0.6)


def _scene(index: int, clip: str = "raw/001.mp4", *, start: float = 0.0, end: float = 5.0) -> Scene:
    return Scene(
        clip=MediaRef(path=clip),
        index=index,
        range=TimeRange(start=start, end=end),
        quality=_quality(),
        motion=_motion(),
        tags=_tags(),
        shot_type=ShotType.MEDIUM,
    )


class TestMediaProbe:
    def test_stream_presence(self, video_probe: MediaProbe) -> None:
        assert video_probe.has_video
        assert video_probe.has_audio

    def test_audio_only_file(self) -> None:
        """A narration file has no video stream, and code must cope."""
        probe = MediaProbe(
            source=MediaRef(path="narration.wav"),
            duration=125.0,
            size_bytes=1024,
            audio=AudioStreamInfo(codec="pcm_s16le", sample_rate=48000, channels=1),
        )
        assert not probe.has_video
        assert probe.has_audio

    def test_full_range_matches_duration(self, video_probe: MediaProbe) -> None:
        assert video_probe.full_range == TimeRange(start=0.0, end=30.0)

    def test_rotation_swaps_the_display_size(self) -> None:
        """Phone footage is stored landscape with a rotation flag."""
        stream = VideoStreamInfo(width=1920, height=1080, fps=30.0, codec="h264", rotation=90)
        assert stream.display_size == (1080, 1920)
        assert stream.aspect == pytest.approx(9 / 16)

    def test_no_rotation_keeps_the_encoded_size(self) -> None:
        stream = VideoStreamInfo(width=1920, height=1080, fps=30.0, codec="h264")
        assert stream.display_size == (1920, 1080)

    def test_180_degree_rotation_does_not_swap(self) -> None:
        stream = VideoStreamInfo(width=1920, height=1080, fps=30.0, codec="h264", rotation=180)
        assert stream.display_size == (1920, 1080)

    def test_zero_duration_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MediaProbe(source=MediaRef(path="raw/001.mp4"), duration=0.0, size_bytes=0)


class TestSpeechModels:
    def test_word_range(self) -> None:
        word = Word(text="plant", start=1.0, end=1.4, probability=0.98)
        assert word.range == TimeRange(start=1.0, end=1.4)

    def test_a_word_with_no_duration_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must exceed start"):
            Word(text="plant", start=1.0, end=1.0)

    def test_transcript_text_joins_segments(self) -> None:
        transcript = Transcript(
            source=MediaRef(path="narration.wav"),
            language="en",
            duration=10.0,
            model_name="faster-whisper/medium",
            segments=(
                TranscriptSegment(
                    index=0, range=TimeRange(start=0.0, end=3.0), text="Plant the tree."
                ),
                TranscriptSegment(
                    index=1, range=TimeRange(start=3.0, end=6.0), text="Water it well."
                ),
            ),
        )
        assert transcript.text == "Plant the tree. Water it well."
        assert not transcript.has_word_timings

    def test_has_word_timings_requires_every_segment(self) -> None:
        with_words = TranscriptSegment(
            index=0,
            range=TimeRange(start=0.0, end=1.0),
            text="hi",
            words=(Word(text="hi", start=0.0, end=1.0),),
        )
        without = TranscriptSegment(index=1, range=TimeRange(start=1.0, end=2.0), text="there")
        transcript = Transcript(
            source=MediaRef(path="narration.wav"),
            language="en",
            duration=2.0,
            model_name="test",
            segments=(with_words, without),
        )
        assert not transcript.has_word_timings
        assert len(transcript.words()) == 1

    def test_an_empty_transcript_has_no_word_timings(self) -> None:
        transcript = Transcript(
            source=MediaRef(path="narration.wav"),
            language="en",
            duration=1.0,
            model_name="test",
        )
        assert not transcript.has_word_timings
        assert transcript.text == ""


class TestSpeechCleanupReport:
    def test_duration_accounting(self) -> None:
        report = SpeechCleanupReport(
            source=MediaRef(path="narration.wav"),
            original_duration=20.0,
            silences=(SilenceSpan(range=TimeRange(start=8.0, end=11.0), mean_db=-48.0),),
            kept_ranges=(TimeRange(start=0.0, end=8.0), TimeRange(start=11.0, end=20.0)),
        )
        assert report.kept_duration == pytest.approx(17.0)
        assert report.removed_duration == pytest.approx(3.0)

    def test_overlapping_kept_ranges_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ascending and non-overlapping"):
            SpeechCleanupReport(
                source=MediaRef(path="narration.wav"),
                original_duration=20.0,
                kept_ranges=(TimeRange(start=0.0, end=10.0), TimeRange(start=8.0, end=15.0)),
            )


class TestVideoModels:
    def test_scene_key_is_derived_from_clip_and_index(self) -> None:
        assert _scene(3).key == "001#3"

    def test_scene_key_round_trips_through_json(self) -> None:
        """A derived key must not appear in the payload, or extra=forbid breaks."""
        scene = _scene(3)
        assert Scene.model_validate_json(scene.model_dump_json()) == scene

    def test_quality_scores_must_be_normalised(self) -> None:
        with pytest.raises(ValidationError):
            QualityScores(blur=1.5, brightness=0.5, exposure=0.5, stability=0.5, overall=0.5)

    def test_unmeasured_people_count_is_none_not_zero(self) -> None:
        """None means 'not measured'; 0 means 'measured, nobody there'."""
        tags = SceneTags(provider="classical_cv")
        assert tags.people_count is None
        assert tags.objects == ()

    def test_clip_analysis_rejects_a_scene_from_another_clip(self) -> None:
        with pytest.raises(ValidationError, match="belongs to"):
            ClipAnalysis(
                clip=MediaRef(path="raw/001.mp4"),
                probe=MediaProbe(source=MediaRef(path="raw/001.mp4"), duration=30.0, size_bytes=1),
                scenes=(_scene(0, clip="raw/002.mp4"),),
                analyzer_version="1.0",
                analyzed_at=datetime.now(UTC),
            )

    def test_clip_analysis_rejects_out_of_order_scenes(self) -> None:
        with pytest.raises(ValidationError, match="ascending time order"):
            ClipAnalysis(
                clip=MediaRef(path="raw/001.mp4"),
                probe=MediaProbe(source=MediaRef(path="raw/001.mp4"), duration=30.0, size_bytes=1),
                scenes=(
                    _scene(0, start=10.0, end=15.0),
                    _scene(1, start=0.0, end=5.0),
                ),
                analyzer_version="1.0",
                analyzed_at=datetime.now(UTC),
            )

    def test_usable_duration(self, video_probe: MediaProbe) -> None:
        analysis = ClipAnalysis(
            clip=MediaRef(path="raw/001.mp4"),
            probe=video_probe,
            scenes=(_scene(0, start=0.0, end=5.0), _scene(1, start=5.0, end=12.0)),
            analyzer_version="1.0",
            analyzed_at=datetime.now(UTC),
        )
        assert analysis.usable_duration == pytest.approx(12.0)


class TestDuplicateGroup:
    def test_valid_group(self) -> None:
        group = DuplicateGroup(
            representative="001#3", duplicates=("001#4", "002#0"), similarity=0.95
        )
        assert len(group.duplicates) == 2

    def test_the_representative_may_not_be_its_own_duplicate(self) -> None:
        with pytest.raises(ValidationError, match="must not appear in duplicates"):
            DuplicateGroup(representative="001#3", duplicates=("001#3",), similarity=0.99)

    def test_an_empty_group_is_meaningless(self) -> None:
        with pytest.raises(ValidationError):
            DuplicateGroup(representative="001#3", duplicates=(), similarity=0.99)


class TestFootageAnalysis:
    @pytest.fixture
    def footage(self, video_probe: MediaProbe) -> FootageAnalysis:
        return FootageAnalysis(
            clips=(
                ClipAnalysis(
                    clip=MediaRef(path="raw/001.mp4"),
                    probe=video_probe,
                    scenes=(_scene(0, start=0.0, end=5.0), _scene(1, start=5.0, end=9.0)),
                    analyzer_version="1.0",
                    analyzed_at=datetime.now(UTC),
                ),
            ),
            duplicates=(
                DuplicateGroup(representative="001#0", duplicates=("001#1",), similarity=0.97),
            ),
        )

    def test_flattens_scenes(self, footage: FootageAnalysis) -> None:
        assert [scene.key for scene in footage.scenes] == ["001#0", "001#1"]

    def test_scene_lookup(self, footage: FootageAnalysis) -> None:
        assert footage.scene_by_key("001#1") is not None
        assert footage.scene_by_key("999#9") is None

    def test_suppressed_keys(self, footage: FootageAnalysis) -> None:
        assert footage.suppressed_scene_keys == frozenset({"001#1"})


class TestMusicModels:
    def _track(self, name: str, *, duration: float, moods: tuple[MusicMood, ...]) -> MusicTrack:
        return MusicTrack(
            source=MediaRef(path=f"music/{name}"),
            probe=MediaProbe(
                source=MediaRef(path=f"music/{name}"),
                duration=duration,
                size_bytes=1024,
                audio=AudioStreamInfo(codec="mp3", sample_rate=44100, channels=2),
            ),
            bpm=90.0,
            energy=0.4,
            loudness_lufs=-14.0,
            moods=moods,
        )

    def test_mood_matching(self) -> None:
        track = self._track("calm.mp3", duration=120.0, moods=(MusicMood.CALM,))
        assert track.matches_mood(MusicMood.CALM)
        assert not track.matches_mood(MusicMood.ENERGETIC)

    def test_by_mood_returns_longest_first(self) -> None:
        """A bed that outlasts the section needs no loop point."""
        library = MusicLibrary(
            tracks=(
                self._track("short.mp3", duration=60.0, moods=(MusicMood.CALM,)),
                self._track("long.mp3", duration=240.0, moods=(MusicMood.CALM,)),
                self._track("loud.mp3", duration=180.0, moods=(MusicMood.ENERGETIC,)),
            )
        )
        calm = library.by_mood(MusicMood.CALM)
        assert [track.source.name for track in calm] == ["long.mp3", "short.mp3"]

    def test_find_by_ref(self) -> None:
        track = self._track("calm.mp3", duration=60.0, moods=())
        library = MusicLibrary(tracks=(track,))
        assert library.find(MediaRef(path="music/calm.mp3")) is track
        assert library.find(MediaRef(path="music/absent.mp3")) is None

    def test_total_duration(self) -> None:
        library = MusicLibrary(
            tracks=(
                self._track("a.mp3", duration=60.0, moods=()),
                self._track("b.mp3", duration=90.0, moods=()),
            )
        )
        assert library.total_duration == pytest.approx(150.0)


class TestProjectManifest:
    def _entry(self, path: str, kind: MediaKind, *, size: int = 1024) -> MediaEntry:
        return MediaEntry(
            ref=MediaRef(path=path),
            kind=kind,
            size_bytes=size,
            modified_at=datetime(2026, 8, 6, tzinfo=UTC),
        )

    def test_editable_requires_narration_and_footage(self, now: datetime) -> None:
        base = {"project_id": "demo", "name": "demo", "created_at": now, "scanned_at": now}
        assert not ProjectManifest(**base).is_editable  # type: ignore[arg-type]
        assert not ProjectManifest(
            **base,  # type: ignore[arg-type]
            raw_clips=(self._entry("raw/001.mp4", MediaKind.RAW_VIDEO),),
        ).is_editable
        assert ProjectManifest(
            **base,  # type: ignore[arg-type]
            narration=self._entry("narration.wav", MediaKind.NARRATION),
            raw_clips=(self._entry("raw/001.mp4", MediaKind.RAW_VIDEO),),
        ).is_editable

    def test_duplicate_entries_are_rejected(self, now: datetime) -> None:
        with pytest.raises(ValidationError, match="duplicate media entry"):
            ProjectManifest(
                project_id="demo",
                name="demo",
                created_at=now,
                scanned_at=now,
                raw_clips=(
                    self._entry("raw/001.mp4", MediaKind.RAW_VIDEO),
                    self._entry("raw/001.mp4", MediaKind.RAW_VIDEO),
                ),
            )

    def test_all_entries_puts_narration_first(self, now: datetime) -> None:
        manifest = ProjectManifest(
            project_id="demo",
            name="demo",
            created_at=now,
            scanned_at=now,
            narration=self._entry("narration.wav", MediaKind.NARRATION),
            raw_clips=(self._entry("raw/001.mp4", MediaKind.RAW_VIDEO),),
            music=(self._entry("music/calm.mp3", MediaKind.MUSIC),),
        )
        assert [entry.ref.name for entry in manifest.all_entries()] == [
            "narration.wav",
            "001.mp4",
            "calm.mp3",
        ]

    def test_change_detection(self) -> None:
        """Cache invalidation: analysis costs minutes, so unchanged must be cheap."""
        first = self._entry("raw/001.mp4", MediaKind.RAW_VIDEO, size=1024)
        same = self._entry("raw/001.mp4", MediaKind.RAW_VIDEO, size=1024)
        resized = self._entry("raw/001.mp4", MediaKind.RAW_VIDEO, size=2048)
        assert first.is_unchanged_from(same)
        assert not first.is_unchanged_from(resized)

    def test_project_id_must_be_a_safe_token(self, now: datetime) -> None:
        with pytest.raises(ValidationError):
            ProjectManifest(project_id="has spaces", name="x", created_at=now, scanned_at=now)
