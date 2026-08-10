"""Rule Engine boundary (implemented in Phase 4).

The Rule Engine is the deterministic counterweight to the AI director, and it runs
on both sides of planning:

**Before** — :class:`SceneFilter` decides which scenes are even eligible. Blurred,
badly exposed, shaky and duplicate shots are removed from consideration, so the
director never has to reason about footage it should not use.

**After** — :class:`PlanRule` checks what the director actually produced. This is
the load-bearing half, because the plan is untrusted output from a language model.
Overlapping clips, a source range that runs past the end of the file, a
half-second cut, a transition longer than the shot it joins: all are things a
plausible-looking plan does, and all are caught here rather than discovered
halfway through a render.

The split between this and model validation matters. The models enforce
*well-formedness*: unique ids, a positive duration, a transition that has length
only if it is not a cut. The Rule Engine enforces *admissibility*, which needs
config thresholds and real probe data that a model has no access to.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.config.rules import RuleSettings
from app.models.common import Issue
from app.models.edit_plan import EditPlan, EditPlanReport
from app.models.video import Scene
from app.rule_engine.context import RuleContext


@runtime_checkable
class SceneFilter(Protocol):
    """Decides whether a scene is eligible for selection (pre-planning)."""

    @property
    def name(self) -> str:
        """Slug used in the ``Issue.code`` this filter emits, e.g. ``quality``."""
        ...

    def is_eligible(self, scene: Scene, *, rules: RuleSettings) -> bool:
        """True when this scene may be offered to the director."""
        ...

    def reject_reason(self, scene: Scene, *, rules: RuleSettings) -> Issue | None:
        """Why the scene was rejected, or ``None`` when it was not.

        Kept separate from :meth:`is_eligible` so filtering stays cheap while
        remaining fully explainable on demand - a user asking "why was my best
        shot dropped?" deserves a real answer.
        """
        ...


@runtime_checkable
class PlanRule(Protocol):
    """A single check applied to a finished Edit Plan (post-planning)."""

    @property
    def name(self) -> str:
        """Slug used in emitted issue codes, e.g. ``clip_duration``."""
        ...

    def check(self, plan: EditPlan, context: RuleContext) -> tuple[Issue, ...]:
        """Inspect ``plan`` and return any findings.

        The context carries the thresholds *and* the measured facts - probe durations,
        suppressed scene keys, the project root. Probes are what make out-of-bounds
        detection possible at all: a plan claiming to cut from 45.0s to 52.0s of a
        30-second clip is well-formed and completely wrong.

        A rule must return an empty tuple rather than raise when it finds nothing, and
        must tolerate a context missing the facts it wanted - a user may validate before
        running analysis, and a partial check beats refusing to check.
        """
        ...


@runtime_checkable
class PlanNormalizer(Protocol):
    """Rewrites a plan into a renderable form.

    Normalisation is what lets the plan format be forgiving to author. The largest
    example is clip placement: a director lists clips in order and omits
    ``timeline_start`` entirely, and the normaliser packs them end to end,
    subtracting transition overlaps. Expecting a language model to do that
    arithmetic across forty clips is how you get a one-frame gap at clip 31.
    """

    @property
    def name(self) -> str: ...

    def normalize(self, plan: EditPlan, context: RuleContext) -> tuple[EditPlan, tuple[Issue, ...]]:
        """Return a corrected plan and a record of every change made.

        The issues are not optional bookkeeping. A silent fix-up is indistinguishable
        from a bug - the user gets a video that does not match the plan they wrote, with
        nothing to explain why - so every adjustment is reported at
        :attr:`~app.models.common.Severity.INFO` and shown.

        A normaliser that changes nothing must return the plan unmodified and an empty
        tuple, because that is how the engine knows the chain has converged.
        """
        ...


@runtime_checkable
class RuleEngine(Protocol):
    """The façade: validate, normalise, or both."""

    def validate(self, plan: EditPlan, context: RuleContext) -> EditPlanReport:
        """Check a plan without changing it."""
        ...

    def normalize(self, plan: EditPlan, context: RuleContext) -> tuple[EditPlan, EditPlanReport]:
        """Normalise a plan, then validate the result.

        Validating *after* normalising is deliberate: a normaliser can introduce a
        violation while fixing another - clamping a source range shortens a clip, which
        can push a transition over its cap - and the caller needs the verdict on the plan
        it is actually going to render.
        """
        ...


__all__ = ["PlanNormalizer", "PlanRule", "RuleContext", "RuleEngine", "SceneFilter"]
