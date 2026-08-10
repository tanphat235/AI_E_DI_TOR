"""Assembling the planning brief, and judging whether a project can be edited at all.

The feasibility check earns its place by being cheap and answering the question that
matters before any others: *is there enough usable footage to cover the narration?* If the
answer is no, no amount of good judgement produces a finished video, and finding that out
before authoring forty clips saves the whole effort.

The brief itself is an assembly job. Its value is not in any computation but in putting
four things that live in four places into one document: what is said, what may be used, how
suitable each option is, and what the result must satisfy.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.config.settings import AiveSettings
from app.models.common import Issue, MediaRef, Severity
from app.models.planning import (
    BeatCandidates,
    CoverageReport,
    PlanConstraints,
    PlanningBrief,
)
from app.models.speech import NarrationAnalysis
from app.models.video import FootageAnalysis
from app.planner.candidates import RANKER_VERSION, CandidateRanker
from app.rule_engine.filters import EligibilityReport, filter_scenes
from app.utils.logging import get_logger

logger = get_logger(__name__)

BRIEF_VERSION = f"brief/1+{RANKER_VERSION}"

_COMFORTABLE_FOOTAGE_RATIO = 2.0
"""Usable footage relative to narration length that gives a director real choice.

Below 1.0 the project cannot be covered without reusing shots. At 1.0 every second of
footage must be used, which forbids selection entirely - the director becomes a
concatenator. Two is where choosing starts to be possible.
"""


class BriefBuilder:
    """Turns analysis documents into a planning brief."""

    def __init__(self, settings: AiveSettings) -> None:
        self._settings = settings
        self._ranker = CandidateRanker(settings.planner)

    @property
    def version(self) -> str:
        return BRIEF_VERSION

    def build(
        self,
        *,
        project_id: str,
        narration: NarrationAnalysis,
        footage: FootageAnalysis,
    ) -> PlanningBrief:
        """Assemble the brief.

        Candidates are drawn only from **eligible** scenes. Offering a shot the Rule Engine
        would reject would waste the director's attention and then fail validation, so the
        filters run first and the rejected scenes are reported separately.
        """
        eligibility = filter_scenes(footage, rules=self._settings.rules)
        beats = self._ranker.rank_all(narration.beats, eligibility.eligible)
        coverage = self._coverage(narration, eligibility, beats)

        return PlanningBrief(
            project_id=project_id,
            created_at=datetime.now(UTC),
            brief_version=self.version,
            coverage=coverage,
            constraints=self._constraints(),
            beats=beats,
            unused_scenes=self._unused(eligibility, beats),
            narration=narration.source,
        )

    # -- Coverage ------------------------------------------------------------ #

    def _coverage(
        self,
        narration: NarrationAnalysis,
        eligibility: EligibilityReport,
        beats: tuple[BeatCandidates, ...],
    ) -> CoverageReport:
        needing = [beat for beat in beats if beat.needs_footage]
        with_candidates = [beat for beat in needing if beat.candidates]

        report = CoverageReport(
            narration_duration=narration.cleanup.kept_duration,
            eligible_footage_duration=eligibility.total_eligible_duration,
            beats_total=len(beats),
            beats_needing_footage=len(needing),
            beats_with_candidates=len(with_candidates),
            eligible_scenes=len(eligibility.eligible),
            rejected_scenes=len(eligibility.rejected),
        )
        return report.model_copy(update={"issues": self._coverage_issues(report, needing)})

    def _coverage_issues(
        self, report: CoverageReport, needing: list[BeatCandidates]
    ) -> tuple[Issue, ...]:
        """Problems worth raising before a single clip is chosen."""
        issues: list[Issue] = []

        if report.eligible_scenes == 0:
            issues.append(
                Issue(
                    code="coverage.no_usable_footage",
                    severity=Severity.ERROR,
                    message=(
                        f"none of the {report.rejected_scenes} analysed scene(s) passed the "
                        "quality filters"
                    ),
                    hint=(
                        "run `aive rules scenes` to see why, then lower the [rules] "
                        "thresholds or shoot more footage"
                    ),
                )
            )
        elif report.footage_ratio < 1.0:
            issues.append(
                Issue(
                    code="coverage.insufficient_footage",
                    severity=Severity.ERROR,
                    message=(
                        f"{report.eligible_footage_duration:.1f}s of usable footage for "
                        f"{report.narration_duration:.1f}s of narration"
                    ),
                    hint="the video cannot be covered without reusing shots",
                )
            )
        elif report.footage_ratio < _COMFORTABLE_FOOTAGE_RATIO:
            issues.append(
                Issue(
                    code="coverage.tight_footage",
                    severity=Severity.WARNING,
                    message=(
                        f"only {report.footage_ratio:.1f}x more footage than narration, so "
                        "there is little room to choose"
                    ),
                    hint="expect to use nearly every scene",
                )
            )

        uncovered = [beat for beat in needing if not beat.candidates]
        if uncovered:
            listed = ", ".join(f"b{beat.beat_index:03d}" for beat in uncovered[:5])
            more = f" and {len(uncovered) - 5} more" if len(uncovered) > 5 else ""
            issues.append(
                Issue(
                    code="coverage.uncovered_beats",
                    severity=Severity.ERROR,
                    message=f"{len(uncovered)} beat(s) have no candidate at all: {listed}{more}",
                    hint="these lines would play over black",
                )
            )

        if report.beats_needing_footage == 0:
            issues.append(
                Issue(
                    code="coverage.no_beats",
                    severity=Severity.ERROR,
                    message="no narration beats survived cleanup, so there is nothing to cut to",
                    hint="check the narration actually contains speech",
                )
            )
        return tuple(issues)

    # -- Constraints --------------------------------------------------------- #

    def _constraints(self) -> PlanConstraints:
        rules = self._settings.rules
        output = self._settings.output
        return PlanConstraints(
            min_clip_duration=rules.min_clip_duration,
            max_clip_duration=rules.max_clip_duration,
            default_transition=rules.default_transition,
            transition_duration=rules.transition_duration,
            max_transition_ratio=rules.max_transition_ratio,
            aspect_ratio=output.aspect_ratio,
            width=output.width,
            height=output.height,
            fps=output.fps,
            min_overall_quality=rules.min_overall_quality,
        )

    @staticmethod
    def _unused(
        eligibility: EligibilityReport, beats: tuple[BeatCandidates, ...]
    ) -> tuple[str, ...]:
        """Eligible scenes that appear in no shortlist.

        Reported rather than dropped. With no semantic tags, ranking is largely driven by
        duration and quality, so a distinctive B-roll shot can easily fail to surface for
        any beat - and it is often the most interesting footage in the project.
        """
        offered = {candidate.scene_key for beat in beats for candidate in beat.candidates}
        return tuple(scene.key for scene in eligibility.eligible if scene.key not in offered)


def narration_ref(narration: NarrationAnalysis) -> MediaRef:
    """The narration this brief was built from."""
    return narration.source


__all__ = ["BRIEF_VERSION", "BriefBuilder"]
