"""The Rule Engine façade.

Two operations, and the relationship between them is the important part:

``validate`` inspects a plan and changes nothing.

``normalize`` rewrites the plan, then validates the **result**. Validating afterwards is
not belt-and-braces: a normaliser can introduce a violation while fixing another - clamping
a source range shortens a clip, which can push a transition over its cap - and the caller
needs the verdict on the plan it is actually going to render, not on the one it wrote.
"""

from __future__ import annotations

from app.models.common import Issue, Severity
from app.models.edit_plan import EditPlan, EditPlanReport
from app.rule_engine.context import RuleContext
from app.rule_engine.normalizers import build_default_normalizers
from app.rule_engine.rules import build_default_rules
from app.utils.logging import get_logger

logger = get_logger(__name__)

RULE_ENGINE_VERSION = "rules/1"

_MAX_NORMALIZE_PASSES = 3
"""How many times the normaliser chain may run.

Normalisers interact: clamping a source range changes a clip's duration, which changes the
transition cap, which changes placement. One pass leaves those interactions unresolved, so
the chain repeats until it reports no further changes.

Bounded rather than looping to convergence, because a pair of normalisers that disagree
would otherwise spin forever - and hanging is a worse failure than an imperfect plan plus a
warning saying so.
"""


class DefaultRuleEngine:
    """A :class:`~app.rule_engine.base.RuleEngine` over the standard rules."""

    def __init__(self) -> None:
        self._rules = build_default_rules()
        self._normalizers = build_default_normalizers()

    @property
    def version(self) -> str:
        return RULE_ENGINE_VERSION

    def validate(self, plan: EditPlan, context: RuleContext) -> EditPlanReport:
        """Check a plan without changing it."""
        issues: list[Issue] = []
        for rule in self._rules:
            found = rule.check(plan, context)  # type: ignore[attr-defined]
            issues.extend(found)

        report = EditPlanReport(plan_project_id=plan.project_id, issues=tuple(issues))
        logger.info(
            "Validated %s: %d error(s), %d warning(s)",
            plan.project_id,
            len(report.errors),
            len(report.warnings),
        )
        return report

    def normalize(self, plan: EditPlan, context: RuleContext) -> tuple[EditPlan, EditPlanReport]:
        """Normalise a plan, then validate the result.

        Returns the rewritten plan and a report combining what was changed (``INFO``) with
        what is still wrong (``WARNING`` and ``ERROR``).
        """
        current = plan
        changes: list[Issue] = []

        for pass_number in range(1, _MAX_NORMALIZE_PASSES + 1):
            pass_changes: list[Issue] = []
            for normalizer in self._normalizers:
                current, found = normalizer.normalize(current, context)  # type: ignore[attr-defined]
                pass_changes.extend(found)

            if not pass_changes:
                break
            changes.extend(pass_changes)
            logger.debug("Normalise pass %d made %d change(s)", pass_number, len(pass_changes))
        else:
            changes.append(
                Issue(
                    code="normalize.not_converged",
                    severity=Severity.WARNING,
                    message=(
                        f"normalisation was still making changes after "
                        f"{_MAX_NORMALIZE_PASSES} passes and was stopped"
                    ),
                    hint="the plan may still violate a rule; check the errors below",
                )
            )

        verdict = self.validate(current, context)
        report = EditPlanReport(
            plan_project_id=current.project_id,
            issues=(*changes, *verdict.issues),
            normalised=True,
        )
        logger.info(
            "Normalised %s: %d change(s), %d error(s) remaining",
            current.project_id,
            len(changes),
            len(report.errors),
        )
        return current, report

    def rule_names(self) -> tuple[str, ...]:
        return tuple(rule.name for rule in self._rules)  # type: ignore[attr-defined]

    def normalizer_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self._normalizers)  # type: ignore[attr-defined]


__all__ = ["RULE_ENGINE_VERSION", "DefaultRuleEngine"]
