"""AIVE - a local AI auto video editor.

AIVE is a *toolbox*, not an autonomous application. An AI director agent (Claude
Code) drives it: the agent invokes the CLI, reads the analysis it produces,
authors an :class:`~app.models.edit_plan.EditPlan`, and then asks AIVE to render
or export that plan.

Nothing in this package calls a language model or the network. The intelligence
lives outside; the determinism lives here.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
