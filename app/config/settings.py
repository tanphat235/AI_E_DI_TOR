"""Layered configuration.

Precedence, lowest to highest:

1. Field defaults in this module
2. ``app/config/default.toml`` — the packaged reference values
3. ``config/aive.toml`` — site override, next to the repo or install
4. ``<project>/aive.toml`` — per-project override
5. ``AIVE_<SECTION>__<FIELD>`` environment variables
6. Explicit keyword overrides, i.e. CLI flags

Merging is per field, so an override file contains only what it changes. This is
the mechanism behind the project's "no hardcoded values" rule: if a number
influences the edit, it is a field here, and every layer can reach it.
"""

from __future__ import annotations

import tomllib
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, ClassVar, Self

from pydantic import BeforeValidator, Field, ValidationError, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from app.config.rules import RuleSettings
from app.models.common import AiveModel, AspectRatio, MusicMood, SubtitleFormat, TransitionKind
from app.models.edit_plan import OutputSpec

PACKAGED_DEFAULTS = Path(__file__).parent / "default.toml"
"""The reference configuration that ships inside the package."""

PROJECT_CONFIG_NAME = "aive.toml"
"""Filename of a per-project override, expected at the project root."""


def _blank_to_none(value: Any) -> Any:
    """Treat an empty or whitespace-only string as "not set".

    TOML has no null. The idiomatic way to write "auto-detect this" in a config
    file is an empty string, so ``ffmpeg_path = ""`` must mean ``None`` and not
    ``Path(".")`` - which is a real, existent directory and would send the FFmpeg
    locator hunting for a binary called ``.``.
    """
    if isinstance(value, str) and not value.strip():
        return None
    return value


OptionalPath = Annotated[Path | None, BeforeValidator(_blank_to_none)]
"""A path that may be omitted, with ``""`` in TOML meaning omitted."""

OptionalStr = Annotated[str | None, BeforeValidator(_blank_to_none)]
"""A string that may be omitted, with ``""`` in TOML meaning omitted."""


class LogLevel(StrEnum):
    """Console verbosity."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class WhisperDevice(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


class AppSettings(AiveModel):
    """Process-level behaviour."""

    log_level: LogLevel = LogLevel.INFO
    log_json_file: bool = Field(
        default=True,
        description="Also write JSON-lines logs to <project>/output/logs/<run_id>.jsonl.",
    )


class MediaSettings(AiveModel):
    """How to find the FFmpeg binaries."""

    ffmpeg_path: OptionalPath = Field(
        default=None,
        description="Explicit ffmpeg binary. Empty auto-resolves: PATH, then the vendored build.",
    )
    ffprobe_path: OptionalPath = Field(default=None, description="Explicit ffprobe binary.")


class SpeechSettings(AiveModel):
    """Phase 2 — faster-whisper recognition."""

    model: str = Field(default="medium", min_length=1, description="Model size or a local path.")
    device: WhisperDevice = WhisperDevice.AUTO
    compute_type: str = Field(default="auto", min_length=1)
    language: OptionalStr = Field(default=None, description="Empty autodetects.")
    word_timestamps: bool = Field(
        default=True,
        description="Required for karaoke ASS subtitles and for safe filler-word excision.",
    )
    beam_size: int = Field(default=5, ge=1, le=20)
    vad_filter: bool = Field(
        default=True,
        description="Silero VAD pre-pass. Strongly recommended: it suppresses the text "
        "Whisper otherwise hallucinates over room tone.",
    )
    download_root: OptionalPath = Field(
        default=None,
        description=(
            "Where to cache model weights. Empty uses the HuggingFace default under "
            "the user profile. Worth setting on a machine whose C: drive is small - "
            "large-v3 is roughly 3 GB."
        ),
    )
    cpu_threads: int = Field(
        default=0,
        ge=0,
        description="0 lets ctranslate2 choose, which is right on all but shared machines.",
    )
    condition_on_previous_text: bool = Field(
        default=False,
        description=(
            "Feed each segment the previous one as context. Improves fluency but is the "
            "main cause of Whisper's runaway repetition loops, so it is off by default: "
            "for narration that gets cut up anyway, robustness beats fluency."
        ),
    )
    max_no_speech_prob: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Drop segments the recogniser itself doubts are speech. This is the main "
            "defence against hallucinated text over music or room tone."
        ),
    )
    max_compression_ratio: float = Field(
        default=2.4,
        gt=1.0,
        description=(
            "Drop segments whose text-to-token ratio exceeds this. Above roughly 2.4 "
            "the recogniser has fallen into a degenerate repetition loop."
        ),
    )


class VisionSettings(AiveModel):
    """Phases 3 to 6 — scene detection and vision understanding."""

    provider: str = Field(
        default="classical_cv",
        min_length=1,
        description="Which VisionProvider to construct. 'classical_cv' downloads no weights.",
    )
    scene_detector: str = Field(default="content", min_length=1)
    scene_threshold: float = Field(default=27.0, gt=0.0)
    min_scene_duration: float = Field(default=0.8, gt=0.0)
    keyframes_per_scene: int = Field(default=3, ge=1, le=20)
    downscale_width: int = Field(
        default=640,
        ge=160,
        description="Analysis resolution. The quality metrics are scale-invariant enough "
        "that full resolution buys nothing but time.",
    )
    samples_per_scene: int = Field(
        default=3,
        ge=1,
        le=20,
        description=(
            "Probe points per scene. At each one a pair of consecutive frames is read, "
            "which is what makes one decode pass yield both stills and motion."
        ),
    )

    # -- Metric normalisation ------------------------------------------------ #

    blur_reference_variance: float = Field(
        default=250.0,
        gt=0.0,
        description=(
            "Laplacian variance treated as fully sharp. Blur is measured as variance, "
            "which is unbounded and depends on content as much as focus - a flat wall "
            "scores low while in perfect focus. So the score is comparative *within a "
            "project*, and this is the divisor that maps it into 0..1. Raise it if "
            "detailed footage is being judged soft."
        ),
    )
    motion_low_threshold: float = Field(
        default=0.5,
        gt=0.0,
        description="Mean flow (px/frame at analysis scale) above which a shot stops being static.",
    )
    motion_medium_threshold: float = Field(default=2.0, gt=0.0)
    motion_high_threshold: float = Field(default=6.0, gt=0.0)
    camera_move_consistency: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "How aligned the tracked points must be to call a move deliberate. Below "
            "this the movement is incoherent, which is what handheld shake looks like."
        ),
    )
    shake_reference: float = Field(
        default=3.0,
        gt=0.0,
        description="Jitter (px/frame) treated as maximally shaky when scoring stability.",
    )
    zoom_radial_threshold: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description=(
            "How radial movement must be to count as a zoom, from 0 to 1. Measured by "
            "projecting each tracked point's displacement onto its outward direction "
            "from the frame centre, so handheld jitter - whose direction is unrelated "
            "to position - averages to nearly zero and is not mistaken for a zoom."
        ),
    )

    # -- Vision provider ----------------------------------------------------- #

    face_cascade: str = Field(
        default="haarcascade_frontalface_default.xml",
        min_length=1,
        description="Cascade bundled with OpenCV. No download; see the note on opencv<5.",
    )
    face_min_size_ratio: float = Field(
        default=0.04,
        gt=0.0,
        le=1.0,
        description="Smallest face to accept, as a fraction of frame height. Filters noise.",
    )
    close_up_face_ratio: float = Field(
        default=0.35,
        gt=0.0,
        le=1.0,
        description="Face height over frame height above which the shot reads as a close-up.",
    )
    medium_face_ratio: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description="Above this a shot is medium; below it, wide.",
    )

    # -- Keyframes ----------------------------------------------------------- #

    keyframe_width: int = Field(
        default=512,
        ge=64,
        description="Width of written stills. Big enough for a future vision model, small "
        "enough that a 200-scene project does not fill a disk.",
    )
    keyframe_quality: int = Field(default=85, ge=1, le=100, description="JPEG quality.")


class PlannerSettings(AiveModel):
    """Phase 5 — how beat/scene candidates are ranked.

    These weights decide the *order* scenes are offered to the AI director, never which
    one it picks. They are config rather than constants because the right balance depends
    on the footage: a project shot with one camera wants variety weighted heavily, while a
    multi-camera shoot does not.

    Note what is missing. There is no semantic weight worth much, because the classical CV
    vision provider produces no object or action tags - so ``keyword_weight`` scores
    against an empty set on most projects and contributes nothing. That is the honest state
    of affairs, and it is the single biggest reason to add a CLIP or multimodal
    :class:`~app.analysis.vision.base.VisionProvider`.
    """

    candidates_per_beat: int = Field(
        default=5,
        ge=1,
        le=50,
        description=(
            "How many options to offer per beat. Forty beats against two hundred scenes is "
            "eight thousand pairs; a shortlist is what makes the brief readable at all."
        ),
    )

    keyword_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for overlap between a beat's keywords and a scene's tags. Currently "
            "scores near zero on most projects: the classical provider names no objects. "
            "Kept weighted so a real vision provider takes effect without retuning."
        ),
    )
    quality_weight: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Weight for the scene's overall quality."
    )
    duration_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for how well the scene's length suits the beat's length.",
    )
    variety_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for differing from the previous beat's shot. Three consecutive wides "
            "read as laziness even when each is individually the best choice."
        ),
    )

    reuse_penalty: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "How much to demote a scene already offered as a top candidate elsewhere. "
            "Large by default: reusing a shot the audience just saw is the most visible "
            "sign of an automated edit."
        ),
    )
    duration_tolerance: float = Field(
        default=2.0,
        gt=0.0,
        description=(
            "Seconds of slack before a length mismatch costs anything. A scene need not "
            "match its beat exactly - the plan trims it - so only a large mismatch matters."
        ),
    )


class SubtitleSettings(AiveModel):
    """Phase 2 — subtitle generation and styling."""

    formats: tuple[SubtitleFormat, ...] = (SubtitleFormat.SRT, SubtitleFormat.ASS)
    max_chars_per_line: int = Field(default=42, ge=10, le=120)
    max_lines: int = Field(default=2, ge=1, le=4)
    min_cue_duration: float = Field(default=0.70, gt=0.0)
    max_cue_duration: float = Field(default=6.00, gt=0.0)
    cue_gap: float = Field(default=0.04, ge=0.0, description="Minimum blank between cues.")
    font_name: str = Field(default="Arial", min_length=1)
    font_size: int = Field(
        default=48,
        gt=0,
        description="Points in a fixed 1920x1080 space, so text scales with the output size.",
    )
    primary_colour: str = Field(
        default="&H00FFFFFF",
        description=(
            "ASS colour in &HAABBGGRR order - alpha, then BLUE, GREEN, RED. Not RGB: a "
            "value copied from a web palette comes out with red and blue swapped."
        ),
    )
    secondary_colour: str = Field(
        default="&H00A0A0A0",
        description=(
            "The not-yet-spoken colour in karaoke. Text starts in this colour and flips "
            "to primary_colour as each word is reached."
        ),
    )
    outline_colour: str = Field(default="&H00000000")
    outline: float = Field(default=2.0, ge=0.0)
    shadow: float = Field(default=0.0, ge=0.0)
    margin_v: int = Field(default=60, ge=0, description="Vertical margin from the frame edge.")
    margin_h: int = Field(default=60, ge=0, description="Left and right margin.")
    alignment: int = Field(
        default=2, ge=1, le=9, description="ASS numpad layout; 2 is bottom centre."
    )
    karaoke: bool = Field(
        default=False,
        description=(
            "Emit per-word highlight timing in ASS. Off by default: it needs word "
            "timings and is a strong stylistic choice - right for short-form video, "
            "wrong for a documentary."
        ),
    )


class MusicSettings(AiveModel):
    """Phase 7 — music level and ducking defaults."""

    target_lufs: float = Field(default=-18.0, le=0.0)
    narration_target_lufs: float = Field(default=-16.0, le=0.0)
    duck_gain_db: float = Field(default=-12.0, le=0.0)
    duck_threshold_db: float = Field(default=-30.0, le=0.0)
    duck_attack: float = Field(default=0.15, gt=0.0)
    duck_release: float = Field(default=0.60, gt=0.0)
    fade_in: float = Field(default=1.50, ge=0.0)
    fade_out: float = Field(default=2.00, ge=0.0)
    default_mood: MusicMood = MusicMood.NEUTRAL

    # -- Analysis ------------------------------------------------------------ #
    analysis_sample_rate: int = Field(
        default=22050,
        ge=8000,
        le=48000,
        description=(
            "Rate to decode at for feature extraction. 22.05 kHz resolves everything "
            "tempo and energy need and halves the decode cost; loudness is measured "
            "separately by FFmpeg at the file's native rate."
        ),
    )
    bpm_min: float = Field(default=60.0, gt=0.0, le=300.0)
    bpm_max: float = Field(default=190.0, gt=0.0, le=400.0)
    bpm_min_confidence: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Autocorrelation peak strength below which no BPM is reported. A wrong "
            "tempo is worse than an absent one: the director would pace cuts to it."
        ),
    )
    bpm_min_crest: float = Field(
        default=4.0,
        ge=1.0,
        description=(
            "Onset-envelope peak-to-mean ratio below which a track is judged to have no "
            "beat. Sustained material measures 2-3 and percussive material 20-30, so the "
            "default sits in the wide gap between them. Without this a pure tone "
            "autocorrelates perfectly and is reported at a confident, meaningless tempo."
        ),
    )
    intro_level_ratio: float = Field(
        default=0.55,
        gt=0.0,
        le=1.0,
        description="RMS below this fraction of the track's median counts as intro.",
    )
    intro_max_fraction: float = Field(
        default=0.35,
        gt=0.0,
        le=1.0,
        description=(
            "Cap on how much of a track may be called intro. A quiet ambient piece is "
            "quiet throughout; without this cap its whole length reads as one long intro."
        ),
    )
    energy_floor_db: float = Field(default=-40.0, le=0.0, description="RMS mapped to energy 0.")
    energy_ceiling_db: float = Field(default=-10.0, le=0.0, description="RMS mapped to energy 1.")
    energy_rms_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight of level in the energy score; the remainder is brightness.",
    )
    brightness_reference_hz: float = Field(
        default=3500.0,
        gt=0.0,
        description="Spectral centroid mapped to brightness 1.0.",
    )

    # -- Mood decision table ------------------------------------------------- #
    mood_calm_energy: float = Field(default=0.35, ge=0.0, le=1.0)
    mood_energetic_energy: float = Field(default=0.68, ge=0.0, le=1.0)
    mood_fast_bpm: float = Field(default=120.0, gt=0.0)
    mood_slow_bpm: float = Field(default=90.0, gt=0.0)
    mood_bright_brightness: float = Field(default=0.55, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_ranges(self) -> Self:
        if self.bpm_min >= self.bpm_max:
            msg = f"bpm_min ({self.bpm_min}) must be below bpm_max ({self.bpm_max})"
            raise ValueError(msg)
        if self.energy_floor_db >= self.energy_ceiling_db:
            msg = (
                f"energy_floor_db ({self.energy_floor_db}) must be below "
                f"energy_ceiling_db ({self.energy_ceiling_db})"
            )
            raise ValueError(msg)
        if self.mood_calm_energy >= self.mood_energetic_energy:
            msg = "mood_calm_energy must be below mood_energetic_energy"
            raise ValueError(msg)
        return self


class TtsSettings(AiveModel):
    """Speech synthesis — turning a written script into a narration track.

    ``backend`` defaults to the offline one. AIVE's promise is that it makes no network call,
    so the backend that breaks that has to be asked for by name rather than fallen into.
    """

    backend: str = Field(
        default="sapi",
        min_length=1,
        description=(
            "'sapi' uses Windows' own voices: offline, no dependency, but only the voices "
            "installed in Windows. 'edge' uses Microsoft Edge's free service - no account "
            "and no API key, and the only good Vietnamese voices - but it CALLS THE NETWORK."
        ),
    )
    voice: OptionalStr = Field(
        default=None,
        description="Voice name. Empty picks a default for the language.",
    )
    language: str = Field(
        default="vi",
        min_length=2,
        description="Language of the script. Chooses the default voice and is recorded on "
        "the transcript.",
    )
    rate: float = Field(
        default=1.0,
        gt=0.0,
        le=3.0,
        description="Speaking rate multiplier. 1.0 is the voice's natural pace.",
    )
    gap: float = Field(
        default=0.35,
        ge=0.0,
        le=5.0,
        description=(
            "Silence after each line, in seconds. A script read with no pause between "
            "sentences sounds hurried, and the pause is also where a cut can land."
        ),
    )
    max_words_per_line: int = Field(
        default=0,
        ge=0,
        description=(
            "Split a long paragraph at sentence boundaries above this many words. 0 keeps "
            "one paragraph as exactly one beat. Really a statement about how long you are "
            "willing to hold a single shot."
        ),
    )
    keep_parts: bool = Field(
        default=True,
        description=(
            "Keep the per-line audio under .aive/. Costs a little disk and means a "
            "mispronounced word is fixed by re-speaking one line, not the whole script."
        ),
    )


class RenderSettings(AiveModel):
    """Phase 7 — how the FFmpeg renderer behaves.

    Distinct from :class:`OutputSettings`, which describes the *delivery format*. These
    are properties of the encode run: how fast, how chatty, and what a draft trades away.
    """

    draft_scale: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Multiplier on the plan's output size in draft mode.",
    )
    draft_crf: int = Field(default=30, ge=0, le=51)
    draft_preset: str = Field(default="veryfast", min_length=1)
    draft_fps: float = Field(
        default=0.0,
        ge=0.0,
        le=240.0,
        description="Frame rate for drafts. 0 keeps the plan's rate.",
    )

    threads: int = Field(
        default=0, ge=0, description="0 lets FFmpeg choose, which is right off shared machines."
    )
    overwrite: bool = Field(default=True, description="Pass -y. The destination is ours.")
    timeout: float = Field(
        default=0.0,
        ge=0.0,
        description="Seconds before a render is abandoned. 0 means no limit - a 4K master "
        "legitimately takes hours, and killing it at an arbitrary deadline loses the work.",
    )
    progress_interval: float = Field(
        default=0.5, gt=0.0, le=60.0, description="Seconds between progress callbacks."
    )

    default_transition: TransitionKind = Field(
        default=TransitionKind.DISSOLVE,
        description="Substituted when a plan asks for a transition FFmpeg cannot perform.",
    )
    keep_source_audio_gain_db: float = Field(
        default=-6.0,
        le=0.0,
        description=(
            "Trim applied to a clip's own audio when it sets mute_source_audio=false. "
            "Attenuated by default: sync sound under narration is texture, not content."
        ),
    )

    @property
    def draft_fps_or(self) -> float | None:
        """The draft frame rate, or ``None`` to keep the plan's."""
        return self.draft_fps if self.draft_fps > 0.0 else None


class OutputSettings(AiveModel):
    """Phase 7 — delivery format and encoder settings.

    Split deliberately: the first four fields are *editorial* and get copied into
    the Edit Plan's :class:`~app.models.edit_plan.OutputSpec`. The encoder fields
    stay here, so the same plan renders as a fast draft or a final master without
    being edited.
    """

    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE
    width: int = Field(default=3840, gt=0)
    height: int = Field(default=2160, gt=0)
    fps: float = Field(default=60.0, gt=0.0, le=240.0)

    video_codec: str = Field(default="libx264", min_length=1)
    video_crf: int = Field(default=18, ge=0, le=51)
    video_preset: str = Field(default="medium", min_length=1)
    pixel_format: str = Field(default="yuv420p", min_length=1)
    audio_codec: str = Field(default="aac", min_length=1)
    audio_bitrate: str = Field(default="192k", min_length=1)
    audio_sample_rate: int = Field(default=48000, gt=0)

    def to_output_spec(self) -> OutputSpec:
        """The editorial subset, as it appears in an Edit Plan."""
        return OutputSpec(
            aspect_ratio=self.aspect_ratio,
            width=self.width,
            height=self.height,
            fps=self.fps,
        )


class CapCutSettings(AiveModel):
    """Phase 8 — CapCut draft export."""

    draft_dir: OptionalPath = Field(
        default=None,
        description="Draft folder. Empty auto-detects the platform location.",
    )
    copy_media: bool = Field(
        default=True,
        description="Copy media into the draft rather than referencing it in place. "
        "Costs disk but survives the originals being moved, which otherwise "
        "silently breaks the CapCut project.",
    )


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #


class AiveSettings(BaseSettings):
    """The merged configuration for one AIVE invocation.

    Build it with :func:`load_settings` rather than constructing it directly;
    direct construction skips every file layer and yields bare field defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIVE_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
        nested_model_default_partial_update=True,
    )

    # Set by `load_settings` on a per-call subclass; see `settings_customise_sources`.
    toml_files: ClassVar[tuple[Path, ...]] = ()

    app: AppSettings = AppSettings()
    media: MediaSettings = MediaSettings()
    speech: SpeechSettings = SpeechSettings()
    vision: VisionSettings = VisionSettings()
    planner: PlannerSettings = PlannerSettings()
    rules: RuleSettings = RuleSettings()
    subtitle: SubtitleSettings = SubtitleSettings()
    music: MusicSettings = MusicSettings()
    tts: TtsSettings = TtsSettings()
    output: OutputSettings = OutputSettings()
    render: RenderSettings = RenderSettings()
    capcut: CapCutSettings = CapCutSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order the sources. Earlier entries win.

        ``toml_files`` is supplied lowest-priority-first for readability, so it is
        reversed here before being spliced in beneath the environment.
        """
        toml_sources = tuple(
            TomlConfigSettingsSource(settings_cls, toml_file=path)
            for path in reversed(cls.toml_files)
        )
        return (init_settings, env_settings, dotenv_settings, *toml_sources)

    @property
    def default_transition(self) -> TransitionKind:
        """Convenience passthrough; transitions are a Rule Engine concern."""
        return self.rules.default_transition


def config_search_paths(project_dir: Path | None = None) -> tuple[Path, ...]:
    """Every config file that would be consulted, lowest priority first.

    Only existing files are returned, so the caller can report exactly which
    layers were actually applied instead of which ones might have been.
    """
    candidates = [PACKAGED_DEFAULTS, Path.cwd() / "config" / PROJECT_CONFIG_NAME]
    if project_dir is not None:
        candidates.append(project_dir / PROJECT_CONFIG_NAME)
    return tuple(path for path in candidates if path.is_file())


@lru_cache(maxsize=8)
def _scoped_settings_class(toml_files: tuple[Path, ...]) -> type[AiveSettings]:
    """A settings class bound to a specific set of TOML layers.

    Pydantic resolves sources in a classmethod, so the file list has to live on the
    class. Building a small subclass per distinct layer set keeps that state out of
    the shared class, which matters as soon as two projects are open at once.
    Cached because building a pydantic model is not free.
    """
    return type(
        "ScopedAiveSettings",
        (AiveSettings,),
        {"toml_files": toml_files, "__module__": __name__},
    )


class ConfigError(RuntimeError):
    """Raised when a configuration file is malformed or names a setting that does not exist.

    Its own type so the CLI can report it as *user input* (exit 4) rather than as an
    internal fault. A typo'd key in ``aive.toml`` used to surface as
    ``internal.unhandled`` with the hint "This is a bug in AIVE" — which sent the user
    looking for a bug in the wrong codebase.
    """


def load_settings(
    project_dir: Path | None = None,
    **overrides: Any,
) -> AiveSettings:
    """Load configuration for ``project_dir``, applying every layer.

    ``overrides`` are CLI flags and win over everything. Pass nested values as
    dicts, e.g. ``load_settings(project, output={"fps": 24})``.

    Raises:
        ConfigError: a layer is unreadable, or names a setting that does not exist.
            ``extra="forbid"`` makes a typo fail loudly rather than silently do nothing,
            which is the behaviour we want - but the failure has to be attributed to the
            file it came from.
    """
    files = config_search_paths(project_dir)
    settings_cls = _scoped_settings_class(files)
    layers = ", ".join(path.name for path in files) or "field defaults"
    try:
        return settings_cls(**overrides)
    except ValidationError as exc:
        msg = f"invalid configuration (layers: {layers}): {_describe(exc)}"
        raise ConfigError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        # Malformed TOML, not a wrong value: a duplicated section header, an unquoted
        # string, a missing bracket. Equally the user's file and equally not our bug, so it
        # gets the same treatment rather than falling through to "report this".
        msg = f"could not parse configuration (layers: {layers}): {exc}"
        raise ConfigError(msg) from exc


def _describe(error: ValidationError) -> str:
    """A ValidationError as one actionable line per problem.

    Pydantic's own rendering spans five lines per error and leads with a URL. What a user
    needs is the setting's dotted name and what was wrong with it.
    """
    parts: list[str] = []
    for item in error.errors():
        location = ".".join(str(piece) for piece in item["loc"])
        detail = item["msg"]
        if item["type"] == "extra_forbidden":
            # The common case, and the one where naming the section helps most: the user
            # guessed a setting name. `aive config show` lists every real one.
            section = location.rsplit(".", 1)[0] if "." in location else location
            detail = f"no such setting; run `aive config show` to list the real [{section}] keys"
        parts.append(f"{location}: {detail}")
    return "; ".join(parts)


__all__ = [
    "PACKAGED_DEFAULTS",
    "PROJECT_CONFIG_NAME",
    "AiveSettings",
    "AppSettings",
    "CapCutSettings",
    "ConfigError",
    "LogLevel",
    "MediaSettings",
    "MusicSettings",
    "OptionalPath",
    "OptionalStr",
    "OutputSettings",
    "PlannerSettings",
    "RenderSettings",
    "RuleSettings",
    "SpeechSettings",
    "SubtitleSettings",
    "TtsSettings",
    "VisionSettings",
    "WhisperDevice",
    "config_search_paths",
    "load_settings",
]
