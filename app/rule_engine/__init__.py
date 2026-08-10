"""The Rule Engine: deterministic guardrails around the AI director (Phase 4).

Runs on both sides of planning. **Before**, :mod:`app.rule_engine.filters` decides which
scenes are eligible, so the director is never offered footage it should not use.
**After**, :mod:`app.rule_engine.rules` checks what it produced - and that half is the
load-bearing one, because an Edit Plan is generated text.
"""

from __future__ import annotations

from app.rule_engine.base import PlanNormalizer, PlanRule, RuleEngine, SceneFilter
from app.rule_engine.context import RuleContext
from app.rule_engine.engine import RULE_ENGINE_VERSION, DefaultRuleEngine
from app.rule_engine.filters import EligibilityReport, SceneVerdict, filter_scenes

__all__ = [
    "RULE_ENGINE_VERSION",
    "DefaultRuleEngine",
    "EligibilityReport",
    "PlanNormalizer",
    "PlanRule",
    "RuleContext",
    "RuleEngine",
    "SceneFilter",
    "SceneVerdict",
    "filter_scenes",
]
