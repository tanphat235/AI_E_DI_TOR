"""Everything a rule needs to judge a plan.

The Phase 1 protocol passed ``rules`` and ``probes`` as separate arguments. That was
enough for checks about durations and file bounds, but not for the ones that matter most:
"is this clip a suppressed duplicate?" needs the footage analysis, and "does this file
exist?" needs the project root. Growing the parameter list for each would mean editing
every rule every time a new kind of check arrives.

So the arguments are gathered into one frozen context. Rules take a plan and a context;
adding a new fact is a new field, and existing rules are untouched.

The context is also the boundary that keeps the Rule Engine honest about *where* its
knowledge comes from. It holds facts measured from real media - probe durations,
suppressed scene keys - not opinions. The opinions are the thresholds in
:class:`~app.config.rules.RuleSettings`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.config.rules import RuleSettings
from app.config.settings import SubtitleSettings
from app.models.common import MediaRef
from app.models.media import MediaProbe
from app.models.speech import NarrationAnalysis
from app.models.video import FootageAnalysis


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Thresholds plus measured facts about the project's media."""

    rules: RuleSettings
    subtitle: SubtitleSettings = field(default_factory=SubtitleSettings)

    probes: dict[MediaRef, MediaProbe] = field(default_factory=dict)
    """Real durations and geometry, keyed by media reference.

    This is what makes out-of-bounds detection possible at all: a plan cutting from 45s
    to 52s of a 30-second clip is perfectly well-formed and completely wrong, and only a
    probe can tell the difference.

    Missing entries are not an error. A user may validate a plan without having run
    analysis, and half a check is better than refusing to check anything.
    """

    suppressed_scene_keys: frozenset[str] = frozenset()
    """Scenes ruled out as duplicates of a better take.

    Carried so a rule can catch a plan that uses one. Cutting between two takes of the
    same shot is the most obvious sign no human was involved, and the director is told
    about these in the digest - so using one anyway is worth flagging.
    """

    project_root: Path | None = None
    """Needed to check that referenced media actually exists. ``None`` skips those checks."""

    @classmethod
    def build(
        cls,
        rules: RuleSettings,
        *,
        subtitle: SubtitleSettings | None = None,
        project_root: Path | None = None,
        footage: FootageAnalysis | None = None,
        narration: NarrationAnalysis | None = None,
        extra_probes: dict[MediaRef, MediaProbe] | None = None,
    ) -> RuleContext:
        """Assemble a context from whatever analysis documents are available.

        Deliberately tolerant of missing pieces. Running ``rules validate`` before
        ``analyze video`` should still check clip durations and timeline overlaps; it
        simply cannot check source bounds for clips it has never measured. Refusing to
        validate at all would push the user toward skipping validation entirely.
        """
        probes: dict[MediaRef, MediaProbe] = {}
        if footage is not None:
            probes.update({analysis.clip: analysis.probe for analysis in footage.clips})
        if extra_probes:
            probes.update(extra_probes)

        return cls(
            rules=rules,
            subtitle=subtitle or SubtitleSettings(),
            probes=probes,
            suppressed_scene_keys=(
                footage.suppressed_scene_keys if footage is not None else frozenset()
            ),
            project_root=project_root,
        )

    def probe_for(self, ref: MediaRef) -> MediaProbe | None:
        """The measured facts for one file, or ``None`` if it was never analysed."""
        return self.probes.get(ref)

    def duration_of(self, ref: MediaRef) -> float | None:
        """Real duration of a file, or ``None`` when unmeasured."""
        probe = self.probes.get(ref)
        return None if probe is None else probe.duration

    def resolve(self, ref: MediaRef) -> Path | None:
        """Absolute path for a reference, or ``None`` without a project root."""
        if self.project_root is None:
            return None
        try:
            return ref.resolve_within(self.project_root)
        except ValueError:
            return None


__all__ = ["RuleContext"]
