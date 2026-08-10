"""Duplicate shot detection by perceptual hashing.

This is the feature that stops an automated edit from looking automated. When someone
records three takes of one sentence, naive selection cuts between all three - and
cutting between near-identical shots is the single most obvious tell that no human was
involved. Suppressing the duplicates is worth more to the finished video than any
quality metric.

The hash has **two halves**, and the second one exists because the first is not enough.

**Structure: a 64-bit dHash on luma.** Each bit records whether a pixel is brighter than
its right-hand neighbour, so it describes composition and survives exposure and
colour-grade differences. Preferred over an *average* hash, which compares to the frame
mean and therefore drifts with exposure.

**Colour: 48 bits of absolute, coarsely quantised colour.** dHash alone has a degenerate
failure that shows up immediately on real footage: a frame with no gradients - a sky, a
wall, a locked-off shot of water, anything out of focus - produces an all-zeros hash. So
*every* flat frame matches every other one. A red frame and a green frame hash
identically and get grouped as the same shot, which would suppress footage that has
nothing in common. Absolute colour is the fix, and it must be absolute rather than
comparative: any measure of spatial *variation* is zero on a uniform frame however it is
computed.

Similarity is Hamming distance over all 112 bits. The threshold lives in config because
it is a genuine editorial trade-off: too low and distinct shots of the same subject get
suppressed, too high and near-identical retakes both survive.

One consequence worth knowing: two structureless frames agree on all 64 structure bits by
construction, so their similarity cannot fall below **0.57** whatever their colours.
``rules.duplicate_similarity`` must stay comfortably above that, or every flat shot in a
project collapses into one group. The 0.90 default has ample margin.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from app.analysis.vision.classical import read_image
from app.analysis.vision.frames import to_grayscale
from app.models.video import DuplicateGroup, Scene
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = get_logger(__name__)

DETECTOR_NAME = "dhash"

HASH_VERSION = "phash/2"
"""Version of the hash *format*, not of the detector.

Bumped from 1 to 2 when colour was appended, which changed the hash from 16 hex
characters to 28. It is part of the footage analyser's cache key for a reason: without
it, a cache written by version 1 would be reused, its 16-character hashes would fail the
length check in :func:`hash_similarity`, and duplicate detection would silently stop
finding anything. A feature that quietly does nothing is worse than one that errors.
"""

HASH_SIDE = 8
"""Produces a 64-bit structure hash from a 9x8 grayscale reduction.

Eight is the standard choice and the right trade-off: 4 collides between unrelated
shots, 16 becomes sensitive to framing differences small enough that a human would call
the shots identical.
"""

COLOUR_GRID = 2
COLOUR_LEVELS_BITS = 4
"""Colour is sampled on a 2x2 grid, each channel quantised to 4 bits (16 levels).

Coarse on purpose. The point is to separate a red shot from a green one, not to
distinguish grades: two takes of the same shot under drifting light must still land in
the same buckets, and 16 levels tolerates a shift of several percent. A 2x2 grid keeps
the colour half (48 bits) smaller than the structure half (64), so composition stays the
dominant signal for footage that actually has some.
"""

STRUCTURE_HEX = 16
COLOUR_HEX = COLOUR_GRID * COLOUR_GRID * 3 * COLOUR_LEVELS_BITS // 4


def dhash(image: np.ndarray) -> str:
    """Structure hash of an image, as 16 hex characters.

    Resized to 9x8 and reduced to whether each pixel is brighter than its right-hand
    neighbour: 8 comparisons per row across 8 rows, so 64 bits.

    Note that this is **all-zeros for any uniform frame**, which is why
    :func:`perceptual_hash` appends colour rather than using this alone.
    """
    import cv2

    grey = to_grayscale(image)
    # INTER_AREA so the reduction averages rather than samples: point sampling a 4K
    # frame down to 9 pixels wide would make the hash depend on which pixels happened
    # to be picked, and two frames of the same shot would hash differently.
    small = cv2.resize(grey, (HASH_SIDE + 1, HASH_SIDE), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:0{STRUCTURE_HEX}x}"


def colour_hash(image: np.ndarray) -> str:
    """Absolute colour signature, as 12 hex characters (48 bits).

    A 2x2 grid of mean channel values, each quantised to 4 bits. Absolute rather than
    compared against anything, so it still carries information on a uniform frame - the
    exact case where the structure hash carries none.
    """
    import cv2
    import numpy as np

    if image.ndim == 2:
        # A greyscale frame has no colour to sample; replicate luma across the channels
        # so the hash stays a fixed width and greys still separate by brightness.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    small = cv2.resize(image, (COLOUR_GRID, COLOUR_GRID), interpolation=cv2.INTER_AREA)
    levels = 1 << COLOUR_LEVELS_BITS
    quantised = (np.asarray(small, dtype=np.uint16) * levels) // 256
    value = 0
    for channel_value in quantised.flatten():
        value = (value << COLOUR_LEVELS_BITS) | min(levels - 1, int(channel_value))
    return f"{value:0{COLOUR_HEX}x}"


def perceptual_hash(image: np.ndarray) -> str:
    """The hash duplicate detection actually uses: structure followed by colour.

    Concatenated into one string so it round-trips through
    :attr:`~app.models.video.Scene.phash` as a single value, and so
    :func:`hash_similarity` weights the two halves simply by their bit counts.
    """
    return dhash(image) + colour_hash(image)


def hash_similarity(first: str, second: str) -> float:
    """Similarity of two hashes, from 0.0 to 1.0.

    Returns 0.0 for malformed or differently sized hashes rather than raising: a hash
    from an older analyser version should not abort a run, it should simply fail to match.
    """
    if not first or not second or len(first) != len(second):
        return 0.0
    try:
        difference = int(first, 16) ^ int(second, 16)
    except ValueError:
        return 0.0
    bits = len(first) * 4
    return 1.0 - (difference.bit_count() / bits)


def hash_keyframes(paths: list[Path]) -> str | None:
    """Hash of the middle keyframe of a scene, or ``None`` if none can be read.

    The middle frame, not the first: the frame right after a cut is the one most likely
    to still be mid-transition, part-way through a dissolve or a camera settling.
    """
    if not paths:
        return None
    middle = paths[len(paths) // 2]
    image = read_image(middle)
    if image is None:
        # Fall back to any readable frame rather than giving up on the scene.
        for candidate in paths:
            image = read_image(candidate)
            if image is not None:
                break
    return None if image is None else perceptual_hash(image)


class PerceptualDuplicateDetector:
    """Groups scenes that are the same shot, by perceptual hash similarity."""

    def __init__(self, *, similarity_threshold: float) -> None:
        self._threshold = similarity_threshold

    @property
    def name(self) -> str:
        return DETECTOR_NAME

    def find_duplicates(self, scenes: tuple[Scene, ...]) -> tuple[DuplicateGroup, ...]:
        """Group near-identical scenes, keeping the best of each group.

        Greedy single-pass clustering: each scene either joins the first group it matches
        or starts its own. Not globally optimal, but the alternative - agglomerative
        clustering - is quadratic and can chain two genuinely different shots together
        through a series of small steps, which is a worse failure than missing one match.
        """
        hashable = [scene for scene in scenes if scene.phash]
        if len(hashable) < 2:
            return ()

        clusters: list[list[Scene]] = []
        for scene in hashable:
            for cluster in clusters:
                if hash_similarity(cluster[0].phash or "", scene.phash or "") >= self._threshold:
                    cluster.append(scene)
                    break
            else:
                clusters.append([scene])

        groups: list[DuplicateGroup] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            representative, duplicates = self._pick_best(cluster)
            groups.append(
                DuplicateGroup(
                    representative=representative.key,
                    duplicates=tuple(scene.key for scene in duplicates),
                    similarity=self._group_similarity(representative, duplicates),
                )
            )

        if groups:
            suppressed = sum(len(group.duplicates) for group in groups)
            logger.info(
                "Found %d duplicate group(s), suppressing %d scene(s)", len(groups), suppressed
            )
        return tuple(groups)

    @staticmethod
    def _pick_best(cluster: list[Scene]) -> tuple[Scene, list[Scene]]:
        """Choose which take survives.

        Ranked by quality first, then by duration. Quality because a sharp take beats a
        soft one; duration as the tiebreak because a longer take gives the director room
        to choose an in and an out point, where a short one forces the cut.
        """
        ordered = sorted(
            cluster,
            key=lambda scene: (scene.quality.overall, scene.range.duration),
            reverse=True,
        )
        return ordered[0], ordered[1:]

    @staticmethod
    def _group_similarity(representative: Scene, duplicates: list[Scene]) -> float:
        """Mean similarity of the group to its representative."""
        scores = [
            hash_similarity(representative.phash or "", scene.phash or "") for scene in duplicates
        ]
        return sum(scores) / len(scores) if scores else 0.0


def group_by_clip(scenes: tuple[Scene, ...]) -> dict[str, list[Scene]]:
    """Scenes bucketed by source clip. Useful for reporting."""
    buckets: dict[str, list[Scene]] = defaultdict(list)
    for scene in scenes:
        buckets[str(scene.clip)].append(scene)
    return dict(buckets)


__all__ = [
    "DETECTOR_NAME",
    "HASH_SIDE",
    "PerceptualDuplicateDetector",
    "colour_hash",
    "dhash",
    "group_by_clip",
    "hash_keyframes",
    "hash_similarity",
    "perceptual_hash",
]
