"""Rule Engine thresholds.

These are the deterministic guardrails that sit either side of the AI director.
Before planning, they decide which scenes are even eligible. After planning, they
decide whether what the director proposed is admissible.

Keeping them in one settings object rather than scattered through the rule
implementations is what makes the editing style *configurable*. A punchy short-form
edit and a slow documentary differ mostly in these numbers, not in code.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from app.models.common import AiveModel, TransitionKind


class RuleSettings(AiveModel):
    """Every threshold the Rule Engine applies."""

    # -- Silence, filler and repetition removal ----------------------------- #

    silence_threshold_db: float = Field(
        default=-35.0,
        ge=-90.0,
        le=0.0,
        description="Level below which narration counts as silence, in dBFS.",
    )
    min_silence_duration: float = Field(
        default=0.35,
        gt=0.0,
        description=(
            "Shorter gaps than this are the natural rhythm of speech. Cutting them "
            "makes narration sound breathless and machine-generated."
        ),
    )
    silence_keep_padding: float = Field(
        default=0.08,
        ge=0.0,
        description="Silence retained at each end of a kept range so cuts do not clip consonants.",
    )
    filler_words: tuple[str, ...] = Field(
        default=("um", "uh", "uhm", "ah", "er", "erm", "eh", "hmm", "mmm"),
        description=(
            "Hesitation sounds that are never meaningful, so they are always removed. "
            "Matched case-insensitively against word-level timings; without word "
            "timings there is no way to excise one word, so filler removal is skipped."
        ),
    )
    ambiguous_filler_words: tuple[str, ...] = Field(
        default=(
            "like",
            "you know",
            "i mean",
            "actually",
            "basically",
            "literally",
            "sort of",
            "kind of",
        ),
        description=(
            "Discourse markers that are *also* ordinary words. Removing 'like' from "
            "'I like gardening' destroys the sentence, so these are only removed when "
            "remove_ambiguous_fillers is enabled."
        ),
    )
    remove_ambiguous_fillers: bool = Field(
        default=False,
        description=(
            "Opt in to removing ambiguous_filler_words. Off by default because a "
            "wrongly cut word is far more damaging than a filler left in."
        ),
    )
    max_repetition_gap: float = Field(
        default=1.5,
        gt=0.0,
        description="Identical phrases closer together than this are treated as a retake.",
    )
    min_repetition_words: int = Field(
        default=2,
        ge=1,
        description=(
            "Shortest phrase treated as a retake. At 1, 'very very good' would lose a "
            "word; at 2, only a genuine restart like 'then water it. then water it "
            "well.' is caught."
        ),
    )
    max_repetition_words: int = Field(
        default=12,
        ge=2,
        description="Longest phrase considered a retake. Bounds the search, nothing more.",
    )
    min_kept_duration: float = Field(
        default=0.20,
        gt=0.0,
        description=(
            "Discard surviving fragments shorter than this. After removals, a 50 ms "
            "sliver between two cuts contains no word - only padding - and plays as a "
            "click."
        ),
    )

    # -- Clip length -------------------------------------------------------- #

    min_clip_duration: float = Field(
        default=1.2,
        gt=0.0,
        description="Below this a cut reads as a glitch rather than a shot.",
    )
    max_clip_duration: float = Field(
        default=8.0,
        gt=0.0,
        description="Above this a single static shot starts to drag.",
    )

    # -- Scene quality floors ----------------------------------------------- #

    min_blur_score: float = Field(default=0.35, ge=0.0, le=1.0)
    brightness_min: float = Field(default=0.15, ge=0.0, le=1.0)
    brightness_max: float = Field(default=0.92, ge=0.0, le=1.0)
    min_stability_score: float = Field(default=0.40, ge=0.0, le=1.0)
    min_overall_quality: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description=(
            "A scene below this is a last resort: usable only when nothing else "
            "covers the narration beat, because a gap is worse than a soft shot."
        ),
    )

    # -- Duplicate detection ------------------------------------------------ #

    duplicate_similarity: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description=(
            "Perceptual-hash similarity above which two scenes are the same shot. "
            "Set too low, distinct shots of the same subject get suppressed; too "
            "high, near-identical retakes both survive."
        ),
    )

    # -- Transitions -------------------------------------------------------- #

    default_transition: TransitionKind = TransitionKind.DISSOLVE
    transition_duration: float = Field(default=0.40, gt=0.0, le=10.0)
    max_transition_ratio: float = Field(
        default=0.25,
        gt=0.0,
        le=0.5,
        description=(
            "A transition may consume at most this fraction of the shorter clip it "
            "joins. Capped at 0.5 because beyond that the two clips would overlap "
            "more than they play."
        ),
    )

    @model_validator(mode="after")
    def _validate_coherent_bounds(self) -> Self:
        if self.min_clip_duration >= self.max_clip_duration:
            msg = (
                f"min_clip_duration ({self.min_clip_duration}) must be less than "
                f"max_clip_duration ({self.max_clip_duration})"
            )
            raise ValueError(msg)
        if self.brightness_min >= self.brightness_max:
            msg = (
                f"brightness_min ({self.brightness_min}) must be less than "
                f"brightness_max ({self.brightness_max})"
            )
            raise ValueError(msg)
        return self

    def transition_fits(self, shorter_clip_duration: float) -> bool:
        """True when the configured transition is short enough for such a clip."""
        return self.transition_duration <= shorter_clip_duration * self.max_transition_ratio

    def clamped_transition_duration(self, shorter_clip_duration: float) -> float:
        """The longest admissible transition for a clip of this length."""
        return min(self.transition_duration, shorter_clip_duration * self.max_transition_ratio)


__all__ = ["RuleSettings"]
