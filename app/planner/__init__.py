"""Support for the AI director (Phase 5).

**AIVE contains no planner.** The director is Claude Code, running outside the app, and the
one genuinely hard decision - which shot serves which line - belongs to it. This package
exists to make that decision well-informed rather than to make it.

Three pieces:

* :mod:`app.planner.candidates` narrows eight thousand beat/scene pairs to a handful per
  beat, and explains every ranking.
* :mod:`app.planner.brief` gathers narration, eligible footage, candidates and the
  constraints a plan must satisfy into one document, and answers whether the project can be
  edited at all before any effort goes into planning it.
* :mod:`app.planner.draft` produces a heuristic baseline: a structurally valid plan to
  revise, explicitly stamped ``created_by: "heuristic"`` so nobody mistakes it for an edit.

The honest limit of all three is the same. Without semantic scene tags - and the classical
CV vision provider produces none - nothing here knows whether a picture illustrates the
words. Ranking answers "which shot is technically suitable and not yet used", which is
useful and is a different question.
"""

from __future__ import annotations

from app.planner.brief import BRIEF_VERSION, BriefBuilder
from app.planner.candidates import RANKER_VERSION, CandidateRanker
from app.planner.draft import DRAFT_VERSION, HeuristicDrafter

__all__ = [
    "BRIEF_VERSION",
    "DRAFT_VERSION",
    "RANKER_VERSION",
    "BriefBuilder",
    "CandidateRanker",
    "HeuristicDrafter",
]
