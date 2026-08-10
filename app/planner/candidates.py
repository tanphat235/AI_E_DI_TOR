"""Ranking scenes against narration beats.

This module narrows a choice; it does not make one. A forty-beat project against two
hundred scenes is eight thousand pairs, and no reader — human or model — holds that at
once. Ranking turns it into a handful of options per beat.

**What the score genuinely knows:**

* whether a scene is long enough for the line it would cover,
* whether it is technically good,
* whether it repeats the previous shot type,
* whether it has already been offered elsewhere.

**What it does not know:** whether the picture illustrates the words. That needs semantic
tags, and the classical-CV vision provider produces none — it can count faces and nothing
else. So ``keyword_weight`` scores against an empty set on most projects and contributes
approximately zero.

This is stated plainly rather than papered over, because the alternative is a number that
looks like semantic matching and is not. The ranking is therefore best understood as *"which
of these shots is technically suitable and not yet used"* — real value, but a different
question from the one an editor asks. Closing that gap is what a CLIP or multimodal
:class:`~app.analysis.vision.base.VisionProvider` would do, and the weight is already
wired for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.settings import PlannerSettings
from app.models.common import ShotType
from app.models.planning import BeatCandidates, SceneCandidate
from app.models.speech import NarrationBeat
from app.models.video import Scene
from app.utils.logging import get_logger

logger = get_logger(__name__)

RANKER_VERSION = "candidates/1"


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """A score and the reasons behind it.

    Kept together because a bare number invites either blind trust or blanket dismissal.
    A director who can see *why* a shot ranked third can disagree with the third place.
    """

    total: float
    reasons: tuple[str, ...] = field(default=())


def keyword_overlap(beat_keywords: tuple[str, ...], scene: Scene) -> tuple[float, int]:
    """Fraction of a beat's keywords found in a scene's tags, and how many matched.

    Returns ``(0.0, 0)`` when the scene has no semantic tags at all, which with the
    classical provider is every scene. That zero is honest: it means "no evidence of a
    match", not "no match".
    """
    if not beat_keywords:
        return 0.0, 0

    tags = {tag.casefold() for tag in (*scene.tags.objects, *scene.tags.actions)}
    if scene.tags.setting:
        tags.add(scene.tags.setting.casefold())
    if scene.tags.caption:
        tags.update(word.casefold() for word in scene.tags.caption.split())
    if not tags:
        return 0.0, 0

    matched = sum(1 for keyword in beat_keywords if keyword.casefold() in tags)
    return matched / len(beat_keywords), matched


def duration_fit(scene_duration: float, wanted: float, *, tolerance: float) -> float:
    """How well a scene's length suits a beat, from 0 to 1.

    Asymmetric, because the two directions are not equally bad. A scene **longer** than
    the beat is fine — the plan simply trims it, and extra material is a choice of in and
    out points. A scene **shorter** than the beat cannot cover it, and the shortfall is a
    hole in the edit.
    """
    if wanted <= 0.0:
        return 1.0
    if scene_duration >= wanted:
        # Long enough. Only an absurd excess costs anything, and gently.
        excess = scene_duration - wanted
        if excess <= tolerance:
            return 1.0
        return max(0.5, 1.0 - (excess - tolerance) / (wanted * 4.0))

    shortfall = wanted - scene_duration
    return max(0.0, 1.0 - shortfall / wanted)


def variety_bonus(candidate: ShotType, previous: ShotType | None) -> float:
    """How much a shot type differs from the one before it.

    Returns 1.0 when there is nothing to differ from, or when the framing changes. Three
    consecutive wides read as laziness even when each is individually the best available
    choice, so this is what makes a ranking prefer to cut *between* framings.

    ``UNKNOWN`` scores neutral rather than well: with the classical provider most scenes
    are unknown, and rewarding that would make the bonus meaningless noise.
    """
    if previous is None or previous is ShotType.UNKNOWN or candidate is ShotType.UNKNOWN:
        return 0.5
    return 1.0 if candidate is not previous else 0.0


class CandidateRanker:
    """Scores scenes against beats and returns a shortlist per beat."""

    def __init__(self, settings: PlannerSettings) -> None:
        self._settings = settings

    @property
    def version(self) -> str:
        return RANKER_VERSION

    def rank_all(
        self,
        beats: tuple[NarrationBeat, ...],
        scenes: tuple[Scene, ...],
    ) -> tuple[BeatCandidates, ...]:
        """Build a shortlist for every beat.

        Beats are processed in order, and two pieces of state carry forward: the shot type
        of the previous beat's best candidate (so variety means something) and the set of
        scenes already offered as a top pick (so the same shot is not proposed everywhere).

        That ordering makes the result depend on beat sequence, which is correct — an edit
        is a sequence, and "don't repeat what we just saw" is inherently positional.
        """
        settings = self._settings
        results: list[BeatCandidates] = []
        already_offered: set[str] = set()
        previous_shot: ShotType | None = None

        for beat in beats:
            if not beat.survives_cleanup:
                # A cut beat needs no picture, so offering options for it would be noise.
                results.append(self._empty(beat))
                continue

            scored = sorted(
                (
                    (self._score(beat, scene, previous_shot, already_offered), scene)
                    for scene in scenes
                ),
                key=lambda pair: pair[0].total,
                reverse=True,
            )
            shortlist = scored[: settings.candidates_per_beat]

            results.append(
                BeatCandidates(
                    beat_index=beat.index,
                    text=beat.text,
                    source_range=beat.range,
                    timeline_range=beat.timeline_range,
                    keywords=beat.keywords,
                    candidates=tuple(
                        SceneCandidate(
                            scene_key=scene.key,
                            clip=scene.clip,
                            range=scene.range,
                            score=min(1.0, max(0.0, breakdown.total)),
                            reasons=breakdown.reasons,
                            quality=scene.quality.overall,
                            shot_type=scene.shot_type,
                            motion=scene.motion.level,
                        )
                        for breakdown, scene in shortlist
                    ),
                )
            )

            if shortlist:
                best = shortlist[0][1]
                already_offered.add(best.key)
                previous_shot = best.shot_type

        logger.info(
            "Ranked %d scene(s) against %d beat(s)",
            len(scenes),
            sum(1 for beat in beats if beat.survives_cleanup),
        )
        return tuple(results)

    # -- Scoring ------------------------------------------------------------- #

    def _score(
        self,
        beat: NarrationBeat,
        scene: Scene,
        previous_shot: ShotType | None,
        already_offered: set[str],
    ) -> ScoreBreakdown:
        settings = self._settings
        wanted = beat.timeline_range.duration if beat.timeline_range else 0.0
        reasons: list[str] = []

        overlap, matched = keyword_overlap(beat.keywords, scene)
        if matched:
            reasons.append(f"matches {matched} keyword(s)")
        elif beat.keywords:
            # Said once per candidate rather than left implicit: a reader who does not know
            # the provider has no tags would read a low score as "wrong shot".
            reasons.append("no semantic tags to match against")

        fit = duration_fit(scene.range.duration, wanted, tolerance=settings.duration_tolerance)
        if wanted > 0.0:
            if scene.range.duration < wanted:
                reasons.append(
                    f"{scene.range.duration:.1f}s is shorter than the {wanted:.1f}s beat"
                )
            elif fit >= 1.0:
                reasons.append(f"{scene.range.duration:.1f}s covers the {wanted:.1f}s beat")

        quality = scene.quality.overall
        if quality >= 0.8:
            reasons.append(f"good quality ({quality:.2f})")
        elif quality < 0.5:
            reasons.append(f"weak quality ({quality:.2f})")

        variety = variety_bonus(scene.shot_type, previous_shot)
        if variety >= 1.0:
            reasons.append(f"{scene.shot_type.value} differs from the previous shot")
        elif variety <= 0.0:
            reasons.append(f"repeats the previous {scene.shot_type.value} shot")

        total = (
            overlap * settings.keyword_weight
            + quality * settings.quality_weight
            + fit * settings.duration_weight
            + variety * settings.variety_weight
        )

        if scene.key in already_offered:
            total *= 1.0 - settings.reuse_penalty
            reasons.append("already offered for an earlier beat")

        return ScoreBreakdown(total=total, reasons=tuple(reasons))

    @staticmethod
    def _empty(beat: NarrationBeat) -> BeatCandidates:
        return BeatCandidates(
            beat_index=beat.index,
            text=beat.text,
            source_range=beat.range,
            timeline_range=None,
            keywords=beat.keywords,
            candidates=(),
        )


__all__ = [
    "RANKER_VERSION",
    "CandidateRanker",
    "ScoreBreakdown",
    "duration_fit",
    "keyword_overlap",
    "variety_bonus",
]
