"""Tests for the shared primitives.

``TimeRange`` and ``MediaRef`` are used by every other model, so a defect here
propagates everywhere. The path tests double as security tests: ``MediaRef`` is the
boundary that stops an LLM-authored plan from reading arbitrary files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models.common import (
    AspectRatio,
    Issue,
    MediaRef,
    Severity,
    TimeRange,
    TransitionKind,
)


class TestTimeRange:
    def test_duration_and_repr(self) -> None:
        span = TimeRange(start=12.5, end=18.3)
        assert span.duration == pytest.approx(5.8)
        assert str(span) == "12.500-18.300"

    @pytest.mark.parametrize(
        ("start", "end"),
        [(5.0, 5.0), (5.0, 4.0), (0.0, 0.0)],
    )
    def test_rejects_non_positive_duration(self, start: float, end: float) -> None:
        with pytest.raises(ValidationError):
            TimeRange(start=start, end=end)

    def test_rejects_negative_start(self) -> None:
        with pytest.raises(ValidationError):
            TimeRange(start=-1.0, end=5.0)

    def test_is_frozen(self) -> None:
        span = TimeRange(start=0.0, end=1.0)
        with pytest.raises(ValidationError):
            span.start = 5.0  # type: ignore[misc]

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            TimeRange(start=0.0, end=1.0, duration=1.0)  # type: ignore[call-arg]

    def test_contains_is_half_open(self) -> None:
        span = TimeRange(start=2.0, end=5.0)
        assert span.contains(2.0)
        assert span.contains(4.999)
        # The end is exclusive, which is what lets adjacent ranges abut exactly.
        assert not span.contains(5.0)
        assert not span.contains(1.999)

    def test_adjacent_ranges_do_not_overlap(self) -> None:
        first = TimeRange(start=0.0, end=5.0)
        second = TimeRange(start=5.0, end=9.0)
        assert not first.overlaps(second)
        assert not second.overlaps(first)

    def test_overlap_detection(self) -> None:
        first = TimeRange(start=0.0, end=5.0)
        assert first.overlaps(TimeRange(start=4.0, end=6.0))
        assert not first.overlaps(TimeRange(start=6.0, end=8.0))

    def test_tolerance_ignores_rounding_slivers(self) -> None:
        first = TimeRange(start=0.0, end=5.0)
        sliver = TimeRange(start=4.999, end=8.0)
        assert first.overlaps(sliver)
        # A 1 ms shared edge is float noise from two analysis passes, not an overlap.
        assert not first.overlaps(sliver, tolerance=0.01)

    def test_intersection(self) -> None:
        first = TimeRange(start=0.0, end=5.0)
        assert first.intersection(TimeRange(start=3.0, end=9.0)) == TimeRange(start=3.0, end=5.0)
        assert first.intersection(TimeRange(start=5.0, end=9.0)) is None

    def test_shifted(self) -> None:
        span = TimeRange(start=2.0, end=5.0)
        assert span.shifted(3.0) == TimeRange(start=5.0, end=8.0)
        assert span.shifted(-2.0) == TimeRange(start=0.0, end=3.0)

    def test_shifted_before_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="before zero"):
            TimeRange(start=2.0, end=5.0).shifted(-3.0)

    def test_clamped_to_a_shorter_file(self) -> None:
        requested = TimeRange(start=25.0, end=40.0)
        actual_file = TimeRange(start=0.0, end=30.0)
        assert requested.clamped_to(actual_file) == TimeRange(start=25.0, end=30.0)

    def test_clamped_entirely_outside_returns_none(self) -> None:
        requested = TimeRange(start=45.0, end=50.0)
        assert requested.clamped_to(TimeRange(start=0.0, end=30.0)) is None


class TestMediaRef:
    def test_accepts_a_relative_path(self) -> None:
        ref = MediaRef(path=Path("raw/001.mp4"))
        assert ref.name == "001.mp4"
        assert ref.stem == "001"
        assert ref.suffix == ".mp4"
        assert str(ref) == "raw/001.mp4"

    def test_suffix_is_lowercased(self) -> None:
        assert MediaRef(path="raw/CLIP.MP4").suffix == ".mp4"

    def test_serialises_as_posix(self) -> None:
        """A plan written on Windows must be readable on macOS."""
        ref = MediaRef(path=Path("raw") / "sub" / "001.mp4")
        assert "raw/sub/001.mp4" in ref.model_dump_json()
        assert "\\\\" not in ref.model_dump_json()

    def test_round_trips(self) -> None:
        ref = MediaRef(path="raw/001.mp4")
        assert MediaRef.model_validate_json(ref.model_dump_json()) == ref

    @pytest.mark.parametrize(
        "hostile",
        [
            "C:/Windows/System32/config",
            "/etc/passwd",
            "../../secrets.mp4",
            "raw/../../escape.mp4",
            "",
            ".",
        ],
    )
    def test_rejects_paths_that_escape_the_project(self, hostile: str) -> None:
        """The security boundary: a plan is untrusted, LLM-authored input."""
        with pytest.raises(ValidationError):
            MediaRef(path=hostile)

    def test_resolve_within(self, tmp_path: Path) -> None:
        ref = MediaRef(path="raw/001.mp4")
        assert ref.resolve_within(tmp_path) == (tmp_path / "raw" / "001.mp4").resolve()

    def test_from_path(self, tmp_path: Path) -> None:
        target = tmp_path / "raw" / "001.mp4"
        target.parent.mkdir(parents=True)
        target.touch()
        assert MediaRef.from_path(target, root=tmp_path) == MediaRef(path="raw/001.mp4")

    def test_from_path_outside_root_is_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "elsewhere.mp4"
        with pytest.raises(ValueError, match="not inside the project root"):
            MediaRef.from_path(outside, root=tmp_path)


class TestEnums:
    def test_only_cut_is_instant(self) -> None:
        assert TransitionKind.CUT.is_instant
        assert not TransitionKind.DISSOLVE.is_instant
        assert not TransitionKind.ZOOM_IN.is_instant

    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [
            (AspectRatio.LANDSCAPE, 16 / 9),
            (AspectRatio.VERTICAL, 9 / 16),
            (AspectRatio.SQUARE, 1.0),
            (AspectRatio.PORTRAIT_4_5, 0.8),
        ],
    )
    def test_aspect_ratio_arithmetic(self, ratio: AspectRatio, expected: float) -> None:
        assert ratio.ratio == pytest.approx(expected)

    def test_enums_serialise_as_their_string_value(self) -> None:
        assert TransitionKind.DISSOLVE.value == "dissolve"
        assert f"{AspectRatio.VERTICAL}" == "9:16"


class TestIssue:
    def test_renders_with_location(self) -> None:
        issue = Issue(
            code="clip.too_short",
            severity=Severity.ERROR,
            message="clip is 0.4s, below the 1.2s minimum",
            hint="extend the source range or drop the clip",
            location="clips[3]",
        )
        assert (
            str(issue) == "[error] clip.too_short at clips[3]: clip is 0.4s, below the 1.2s minimum"
        )

    def test_renders_without_location(self) -> None:
        issue = Issue(code="plan.empty", severity=Severity.WARNING, message="no clips")
        assert str(issue) == "[warning] plan.empty: no clips"

    @pytest.mark.parametrize("bad_code", ["Clip.TooShort", "clip too short", "clip-too-short", ""])
    def test_code_must_be_a_stable_slug(self, bad_code: str) -> None:
        """Codes are branched on by the agent, so their shape is enforced."""
        with pytest.raises(ValidationError):
            Issue(code=bad_code, severity=Severity.ERROR, message="x")
