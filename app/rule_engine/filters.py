"""Scene eligibility: what the director should never be offered.

This is the *pre*-planning half of the Rule Engine. It runs over the footage analysis and
decides which scenes are candidates at all, so the director never has to reason about
footage it should not use.

The design principle is that a filter **explains itself**. Each one answers two
questions: is this scene eligible, and if not, why. The second matters more than it looks
- a user whose best shot vanished from the edit deserves "004#2 is 0.8s, below the 1.2s
minimum" rather than silence. That is also why filtering returns a *report* rather than
just a shorter list.

One rule outranks the others: **coverage beats perfection.** A soft shot is better than a
gap, so :func:`filter_scenes` marks scenes as ineligible without deleting them, and the
director is told it may reach for a rejected scene when nothing else covers a beat.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.rules import RuleSettings
from app.models.common import Issue, Severity
from app.models.video import FootageAnalysis, Scene
from app.utils.logging import get_logger

logger = get_logger(__name__)


class QualityFilter:
    """Rejects scenes that fail a technical floor.

    Each metric is checked separately rather than only against the aggregate, because the
    failures are not interchangeable: a shot can be sharp, bright and unusably shaky, and
    a weighted average would let it through while a viewer would not.
    """

    @property
    def name(self) -> str:
        return "quality"

    def is_eligible(self, scene: Scene, *, rules: RuleSettings) -> bool:
        return self.reject_reason(scene, rules=rules) is None

    def reject_reason(self, scene: Scene, *, rules: RuleSettings) -> Issue | None:
        quality = scene.quality
        checks: list[tuple[bool, str, str]] = [
            (
                quality.blur < rules.min_blur_score,
                "quality.too_soft",
                f"blur {quality.blur:.2f} is below the {rules.min_blur_score:.2f} floor",
            ),
            (
                quality.brightness < rules.brightness_min,
                "quality.too_dark",
                f"brightness {quality.brightness:.2f} is below {rules.brightness_min:.2f}",
            ),
            (
                quality.brightness > rules.brightness_max,
                "quality.too_bright",
                f"brightness {quality.brightness:.2f} is above {rules.brightness_max:.2f}",
            ),
            (
                quality.stability < rules.min_stability_score,
                "quality.unstable",
                f"stability {quality.stability:.2f} is below {rules.min_stability_score:.2f}",
            ),
            (
                quality.overall < rules.min_overall_quality,
                "quality.below_floor",
                f"overall {quality.overall:.2f} is below {rules.min_overall_quality:.2f}",
            ),
        ]
        for failed, code, message in checks:
            if failed:
                return Issue(
                    code=code,
                    severity=Severity.WARNING,
                    message=f"{scene.key}: {message}",
                    hint="usable only if no other scene covers the beat - a gap is worse",
                    location=scene.key,
                )
        return None


class DurationFilter:
    """Rejects scenes too short to cut to.

    A scene shorter than ``min_clip_duration`` cannot yield an admissible clip, so
    offering it would only invite a plan the post-planning rules then reject.
    """

    @property
    def name(self) -> str:
        return "duration"

    def is_eligible(self, scene: Scene, *, rules: RuleSettings) -> bool:
        return scene.range.duration >= rules.min_clip_duration

    def reject_reason(self, scene: Scene, *, rules: RuleSettings) -> Issue | None:
        if self.is_eligible(scene, rules=rules):
            return None
        return Issue(
            code="scene.too_short",
            severity=Severity.WARNING,
            message=(
                f"{scene.key}: {scene.range.duration:.2f}s is shorter than the "
                f"{rules.min_clip_duration:.2f}s minimum clip length"
            ),
            hint="no clip cut from this scene could be long enough to read",
            location=scene.key,
        )


class DuplicateFilter:
    """Rejects scenes suppressed as duplicates of a better take.

    Constructed with the suppressed set rather than deriving it, so the filter stays a
    pure decision and the expensive hashing stays in Phase 3 where it belongs.
    """

    def __init__(
        self, suppressed: frozenset[str], *, keeper_of: dict[str, str] | None = None
    ) -> None:
        self._suppressed = suppressed
        self._keeper_of = keeper_of or {}

    @property
    def name(self) -> str:
        return "duplicate"

    def is_eligible(self, scene: Scene, *, rules: RuleSettings) -> bool:
        return scene.key not in self._suppressed

    def reject_reason(self, scene: Scene, *, rules: RuleSettings) -> Issue | None:
        if scene.key not in self._suppressed:
            return None
        keeper = self._keeper_of.get(scene.key)
        instead = f"; use {keeper} instead" if keeper else ""
        return Issue(
            code="scene.duplicate",
            severity=Severity.WARNING,
            message=f"{scene.key} is the same shot as a better take{instead}",
            hint="cutting between two takes of one shot is the clearest sign of an automated edit",
            location=scene.key,
        )


@dataclass(frozen=True, slots=True)
class SceneVerdict:
    """Whether one scene may be used, and why not."""

    scene: Scene
    eligible: bool
    issues: tuple[Issue, ...] = ()

    @property
    def key(self) -> str:
        return self.scene.key


@dataclass(frozen=True, slots=True)
class EligibilityReport:
    """The result of filtering a project's scenes."""

    verdicts: tuple[SceneVerdict, ...] = ()

    @property
    def eligible(self) -> tuple[Scene, ...]:
        return tuple(verdict.scene for verdict in self.verdicts if verdict.eligible)

    @property
    def rejected(self) -> tuple[SceneVerdict, ...]:
        return tuple(verdict for verdict in self.verdicts if not verdict.eligible)

    @property
    def issues(self) -> tuple[Issue, ...]:
        return tuple(issue for verdict in self.verdicts for issue in verdict.issues)

    def verdict_for(self, key: str) -> SceneVerdict | None:
        return next((verdict for verdict in self.verdicts if verdict.key == key), None)

    @property
    def total_eligible_duration(self) -> float:
        """How much usable footage exists.

        Worth reporting: if this is less than the narration's length, the project cannot
        be covered no matter how well the director plans, and saying so early is far
        better than discovering it at render time.
        """
        return sum(scene.range.duration for scene in self.eligible)


def default_filters(footage: FootageAnalysis) -> tuple[object, ...]:
    """The filter set, wired with this project's duplicate groups."""
    keeper_of = {
        key: group.representative for group in footage.duplicates for key in group.duplicates
    }
    return (
        DuplicateFilter(footage.suppressed_scene_keys, keeper_of=keeper_of),
        DurationFilter(),
        QualityFilter(),
    )


def filter_scenes(footage: FootageAnalysis, *, rules: RuleSettings) -> EligibilityReport:
    """Judge every scene in a project.

    Every filter runs even after one rejects, so the report lists *all* the reasons a
    scene was dropped rather than only the first. A user retuning thresholds needs to know
    that a shot is both too dark and too shaky; fixing one and re-running to discover the
    other wastes their time.
    """
    filters = default_filters(footage)
    verdicts: list[SceneVerdict] = []

    for scene in footage.scenes:
        issues = tuple(
            issue
            for issue in (
                filter_.reject_reason(scene, rules=rules)  # type: ignore[attr-defined]
                for filter_ in filters
            )
            if issue is not None
        )
        verdicts.append(SceneVerdict(scene=scene, eligible=not issues, issues=issues))

    report = EligibilityReport(verdicts=tuple(verdicts))
    logger.info(
        "Scene eligibility: %d of %d usable (%.1fs of footage)",
        len(report.eligible),
        len(report.verdicts),
        report.total_eligible_duration,
    )
    return report


__all__ = [
    "DuplicateFilter",
    "DurationFilter",
    "EligibilityReport",
    "QualityFilter",
    "SceneVerdict",
    "default_filters",
    "filter_scenes",
]
