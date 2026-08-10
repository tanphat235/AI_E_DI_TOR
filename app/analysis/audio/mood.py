"""Inferring mood from measurable features.

This is the weakest link in music analysis and the docstring says so up front. Tempo,
level and spectral brightness are genuine measurements; "uplifting" is not a property of a
waveform. What follows is a decision table over three numbers, and it is a *starting
point for the director*, not a verdict.

Two design choices make that honest rather than merely admitted:

**Moods are returned as a ranked tuple, not a single label.** A track at 118 BPM with
middling energy sits on the boundary between calm and uplifting, and returning both is
truer than picking one.

**Every threshold is config.** The table below reads its numbers from
:class:`~app.config.settings.MusicSettings`, so a user whose library the defaults suit
badly can retune it without editing code — and can see, in ``aive config show``, exactly
what produced a label.

The vocabulary is deliberately tiny (:class:`~app.models.common.MusicMood`). A closed set
of eight is something both this heuristic and an AI director can agree on; free text
cannot be matched by either.
"""

from __future__ import annotations

from app.config.settings import MusicSettings
from app.models.common import MusicMood

MOOD_VERSION = "mood-table/1"


def classify(
    *,
    energy: float,
    brightness: float,
    bpm: float | None,
    settings: MusicSettings,
) -> tuple[MusicMood, ...]:
    """Rank moods for a track, most likely first.

    ``bpm=None`` (tempo could not be established) is handled rather than special-cased
    away: energy and brightness alone still separate calm from energetic, they just cannot
    distinguish *playful* from *uplifting*, so the result is shorter.
    """
    moods: list[MusicMood] = []

    fast = bpm is not None and bpm >= settings.mood_fast_bpm
    slow = bpm is not None and bpm <= settings.mood_slow_bpm
    bright = brightness >= settings.mood_bright_brightness

    if energy <= settings.mood_calm_energy:
        # Quiet material. Brightness is what separates airy from sombre; tempo barely
        # registers at this level, so it is not consulted.
        moods.append(MusicMood.CALM)
        moods.append(MusicMood.UPLIFTING if bright else MusicMood.MELANCHOLIC)
    elif energy >= settings.mood_energetic_energy:
        # Loud material. Fast or bright reads as energetic; loud and dark reads as dramatic,
        # and loud, dark and slow is where tension lives.
        if bright or fast:
            moods.append(MusicMood.ENERGETIC)
            moods.append(MusicMood.PLAYFUL if fast and bright else MusicMood.DRAMATIC)
        else:
            moods.append(MusicMood.DRAMATIC)
            # Slow and dark is tense; at a middling tempo the honest second reading is that
            # it is simply loud. Naming ENERGETIC here rather than repeating DRAMATIC keeps
            # the pair informative - a duplicate would be silently deduped to one label and
            # read as confidence rather than as a second opinion.
            moods.append(MusicMood.TENSE if slow else MusicMood.ENERGETIC)
    # The middle band is where most library music sits, and where a single label is least
    # defensible, so it always returns two.
    elif bright:
        moods.append(MusicMood.UPLIFTING)
        moods.append(MusicMood.PLAYFUL if fast else MusicMood.CALM)
    else:
        moods.append(MusicMood.NEUTRAL)
        moods.append(MusicMood.MELANCHOLIC if slow else MusicMood.DRAMATIC)

    # Dedupe while keeping rank: the branches above can legitimately reach the same mood
    # twice, and a duplicate would read as extra confidence rather than one opinion.
    ranked: list[MusicMood] = []
    for mood in moods:
        if mood not in ranked:
            ranked.append(mood)
    return tuple(ranked)


def energy_score(
    *,
    rms_db: float,
    brightness: float,
    settings: MusicSettings,
) -> float:
    """Perceived intensity, 0.0 to 1.0.

    A weighted blend of level and brightness rather than level alone, because a loud bass
    drone and a loud string section are not equally intense to a listener, and only the
    spectrum tells them apart. The weighting is config; the default leans on level, which
    is the more reliable of the two.
    """
    span = settings.energy_ceiling_db - settings.energy_floor_db
    level = (rms_db - settings.energy_floor_db) / span
    level = min(1.0, max(0.0, level))

    weight = settings.energy_rms_weight
    blended = level * weight + brightness * (1.0 - weight)
    return round(min(1.0, max(0.0, blended)), 3)


__all__ = ["MOOD_VERSION", "classify", "energy_score"]
