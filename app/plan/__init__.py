"""Edit Plan tooling (Phase 6).

The plan itself is defined in :mod:`app.models.edit_plan`; validating and normalising it
belongs to :mod:`app.rule_engine`. This package is about *working with* a finished plan:

* :mod:`app.plan.loader` - the one place a plan is read, so migration cannot be forgotten.
* :mod:`app.plan.review` - a plan rendered for a human, plus the editorial observations that
  are invisible in a list of clips and obvious in a finished video.
* :mod:`app.plan.diff` - what changed between two plans, matched by clip id so an insertion
  does not report everything after it as modified.
"""

from __future__ import annotations

from app.plan.diff import DIFF_VERSION, diff_plans
from app.plan.loader import LoadedPlan, PlanLoadError, load_plan
from app.plan.review import REVIEW_VERSION, build_review

__all__ = [
    "DIFF_VERSION",
    "REVIEW_VERSION",
    "LoadedPlan",
    "PlanLoadError",
    "build_review",
    "diff_plans",
    "load_plan",
]
