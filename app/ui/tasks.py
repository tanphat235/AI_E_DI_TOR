"""The pipeline, described as steps a user can run.

**No Qt import anywhere in this module.** That is deliberate and is the same split that
makes the renderer testable: :mod:`app.renderer.ffmpeg.graph` is pure and
:mod:`app.renderer.ffmpeg.renderer` is a thin shell over it. Here, this module knows what
the steps are, what each one needs before it can run, and how to run it; :mod:`app.ui.window`
knows only how to draw a button. Everything worth testing is therefore testable without a
display, a window or an event loop.

The other rule this module enforces: **nothing here reimplements pipeline logic.** Every
step calls the same service the CLI calls. A desktop app that quietly analyses differently
from the command line is two products with one name, and the divergence is always found by
a user comparing them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.models.edit_plan import EditPlan
from app.models.speech import NarrationAnalysis
from app.models.video import FootageAnalysis
from app.services.paths import ProjectPaths
from app.utils.logging import get_logger

logger = get_logger(__name__)

ProgressFn = Callable[[float, str], None]
"""Reports ``(fraction, message)``. Injected, so a step never knows about a progress bar."""


class StepId(StrEnum):
    """Every action the window can start."""

    SCAN = "scan"
    ANALYZE_AUDIO = "analyze_audio"
    ANALYZE_VIDEO = "analyze_video"
    ANALYZE_MUSIC = "analyze_music"
    BRIEF = "brief"
    DRAFT = "draft"
    NORMALIZE = "normalize"
    SUBTITLES = "subtitles"
    RENDER_DRAFT = "render_draft"
    RENDER_FINAL = "render_final"
    EXPORT_CAPCUT = "export_capcut"


@dataclass(frozen=True, slots=True)
class StepResult:
    """What a step produced, in terms a window can display without interpreting."""

    summary: str
    detail: str = ""
    artifact: Path | None = None
    """A file or folder worth offering to open. ``None`` when the step produced none."""


@dataclass(frozen=True, slots=True)
class Step:
    """One runnable action, and what has to be true before it can run."""

    id: StepId
    label: str
    description: str
    requires: tuple[str, ...] = ()
    """Human-readable preconditions, checked by :func:`blockers`."""

    @property
    def is_slow(self) -> bool:
        """Whether this step warrants a confirmation before it takes over the machine."""
        return self.id in {
            StepId.ANALYZE_AUDIO,
            StepId.ANALYZE_VIDEO,
            StepId.RENDER_FINAL,
        }


STEPS: tuple[Step, ...] = (
    Step(
        id=StepId.SCAN,
        label="Scan project",
        description="Find the narration, the footage and the music.",
    ),
    Step(
        id=StepId.ANALYZE_AUDIO,
        label="Analyse narration",
        description="Transcribe, cut silence and fillers, find beats.",
        requires=("a narration file",),
    ),
    Step(
        id=StepId.ANALYZE_VIDEO,
        label="Analyse footage",
        description="Detect scenes, score quality, find duplicate takes.",
        requires=("footage in raw/",),
    ),
    Step(
        id=StepId.ANALYZE_MUSIC,
        label="Analyse music",
        description="Tempo, energy, loudness and where each intro ends.",
        requires=("audio in music/",),
    ),
    Step(
        id=StepId.BRIEF,
        label="Build planning brief",
        description="Join beats to eligible footage and rank candidates.",
        requires=("narration analysis", "footage analysis"),
    ),
    Step(
        id=StepId.DRAFT,
        label="Heuristic baseline plan",
        description="A starting point to revise. Not an edit.",
        requires=("narration analysis", "footage analysis"),
    ),
    Step(
        id=StepId.NORMALIZE,
        label="Normalise plan",
        description="Place clips and clamp what is out of bounds.",
        requires=("an edit plan",),
    ),
    Step(
        id=StepId.SUBTITLES,
        label="Attach subtitles",
        description="Put timed cues into the plan itself.",
        requires=("an edit plan", "narration analysis"),
    ),
    Step(
        id=StepId.RENDER_DRAFT,
        label="Render draft",
        description="Half size, fast preset. For judging the cut.",
        requires=("an edit plan",),
    ),
    Step(
        id=StepId.RENDER_FINAL,
        label="Render final",
        description="Full quality. Slow.",
        requires=("an edit plan",),
    ),
    Step(
        id=StepId.EXPORT_CAPCUT,
        label="Export to CapCut",
        description="A draft you can keep editing by hand.",
        requires=("an edit plan",),
    ),
)

STEPS_BY_ID: dict[StepId, Step] = {step.id: step for step in STEPS}


@dataclass(frozen=True, slots=True)
class ProjectState:
    """What exists in a project right now.

    Read from the filesystem rather than remembered, so the window is correct after a user
    runs a CLI command in another terminal — which they will, because the CLI is the
    primary interface and this is a view onto the same project.
    """

    root: Path
    is_project: bool
    has_narration: bool
    has_footage: bool
    has_music: bool
    has_narration_analysis: bool
    has_footage_analysis: bool
    has_music_analysis: bool
    has_plan: bool
    plan: EditPlan | None = None

    @classmethod
    def read(cls, root: Path) -> ProjectState:
        """Inspect a project directory."""
        paths = ProjectPaths.for_root(root)
        plan = _load_plan_quietly(paths.edit_plan_file)
        return cls(
            root=paths.root,
            is_project=paths.exists(),
            has_narration=paths.find_narration() is not None,
            has_footage=bool(paths.find_raw_clips()),
            has_music=bool(paths.find_music()),
            has_narration_analysis=paths.narration_file.is_file(),
            has_footage_analysis=paths.footage_analysis_file.is_file(),
            has_music_analysis=paths.music_library_file.is_file(),
            has_plan=paths.edit_plan_file.is_file(),
            plan=plan,
        )


def blockers(step: Step, state: ProjectState) -> tuple[str, ...]:
    """What is missing before ``step`` can run.

    Returned as text rather than as a boolean so the window can *say why* a button is
    disabled. "Analyse narration is greyed out" is a support question; "needs a narration
    file" is not.
    """
    if not state.is_project:
        return ("this folder is not an AIVE project",)

    available = {
        "a narration file": state.has_narration,
        "footage in raw/": state.has_footage,
        "audio in music/": state.has_music,
        "narration analysis": state.has_narration_analysis,
        "footage analysis": state.has_footage_analysis,
        "an edit plan": state.has_plan,
    }
    return tuple(name for name in step.requires if not available.get(name, True))


def can_run(step: Step, state: ProjectState) -> bool:
    return not blockers(step, state)


def next_suggested(state: ProjectState) -> Step | None:
    """The step a user most usefully runs next.

    A pipeline with eleven buttons is a pipeline nobody finishes. This walks the same order
    the docs recommend and returns the first thing that is both runnable and not yet done,
    so the window always has one obvious action to highlight.
    """
    if not state.is_project:
        return None

    done = {
        StepId.SCAN: state.has_narration_analysis or state.has_footage_analysis,
        StepId.ANALYZE_AUDIO: state.has_narration_analysis,
        StepId.ANALYZE_VIDEO: state.has_footage_analysis,
        StepId.ANALYZE_MUSIC: state.has_music_analysis or not state.has_music,
        StepId.BRIEF: state.has_plan,
        StepId.DRAFT: state.has_plan,
        StepId.NORMALIZE: state.plan is not None and state.plan.is_placed,
        StepId.SUBTITLES: state.plan is not None and bool(state.plan.subtitles),
    }
    for step in STEPS:
        if done.get(step.id, False):
            continue
        if can_run(step, state):
            return step
    return None


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def run_step(step_id: StepId, root: Path, *, on_progress: ProgressFn | None = None) -> StepResult:
    """Execute a step against a project.

    Every branch calls the same service the CLI does. Blocking, and deliberately so: the
    caller is a worker thread (:mod:`app.ui.workers`), and putting the threading here would
    make this module untestable and drag Qt in with it.
    """
    from app.services.container import build_container

    paths = ProjectPaths.for_root(root)
    container = build_container(paths.root)
    report = on_progress or (lambda _fraction, _message: None)

    logger.info("UI running step %s in %s", step_id.value, paths.root)
    report(0.0, STEPS_BY_ID[step_id].label)

    match step_id:
        case StepId.SCAN:
            return _scan(paths)
        case StepId.ANALYZE_AUDIO:
            return _analyze_audio(paths, container, report=report)
        case StepId.ANALYZE_VIDEO:
            return _analyze_video(paths, container)
        case StepId.ANALYZE_MUSIC:
            return _analyze_music(paths, container)
        case StepId.BRIEF:
            return _brief(paths, container)
        case StepId.DRAFT:
            return _draft(paths, container)
        case StepId.NORMALIZE:
            return _normalize(paths, container)
        case StepId.SUBTITLES:
            return _subtitles(paths, container)
        case StepId.RENDER_DRAFT | StepId.RENDER_FINAL:
            return _render(paths, container, draft=step_id is StepId.RENDER_DRAFT, report=report)
        case StepId.EXPORT_CAPCUT:
            return _export(paths, container)


def _scan(paths: ProjectPaths) -> StepResult:
    clips = paths.find_raw_clips()
    music = paths.find_music()
    narration = paths.find_narration()
    return StepResult(
        summary=f"{len(clips)} clip(s), {len(music)} track(s), "
        f"narration {'found' if narration else 'missing'}",
        detail="\n".join(path.name for path in (*clips, *music)),
        artifact=paths.root,
    )


def _analyze_audio(paths: ProjectPaths, container: object, *, report: ProgressFn) -> StepResult:
    from datetime import UTC, datetime

    from app.analysis.speech.beats import NarrationBeatBuilder
    from app.cli.analyze_cmd import ANALYSIS_VERSION
    from app.models.speech import NarrationAnalysis
    from app.services.container import Container

    assert isinstance(container, Container)
    narration_path = paths.find_narration()
    if narration_path is None:
        msg = "no narration file was found"
        raise FileNotFoundError(msg)

    recognizer, cleaner = container.recognizer, container.cleaner
    if recognizer is None or cleaner is None:  # pragma: no cover - always wired
        msg = "speech analysis is not available in this build"
        raise RuntimeError(msg)

    ref = paths.to_ref(narration_path)
    # Transcription dominates this step, so its fraction is the step's fraction. The
    # cleanup and beat passes that follow are milliseconds by comparison.
    transcript = recognizer.transcribe(narration_path, ref=ref, on_progress=report)
    cleanup = cleaner.clean(transcript, audio=narration_path)
    beats = NarrationBeatBuilder().build(transcript, cleanup)

    analysis = NarrationAnalysis(
        source=ref,
        transcript=transcript,
        cleanup=cleanup,
        beats=beats,
        analyzer_version=ANALYSIS_VERSION,
        analyzed_at=datetime.now(UTC),
    )
    paths.narration_file.parent.mkdir(parents=True, exist_ok=True)
    paths.narration_file.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")

    return StepResult(
        summary=f"{len(analysis.surviving_beats)} beat(s) of "
        f"{cleanup.kept_duration:.1f}s narration",
        detail="\n".join(f"{beat.index:03d} {beat.text}" for beat in analysis.beats),
        artifact=paths.narration_file,
    )


def _analyze_video(paths: ProjectPaths, container: object) -> StepResult:
    from app.services.container import Container

    assert isinstance(container, Container)
    analyzer = container.footage
    if analyzer is None:  # pragma: no cover - always wired for a project
        msg = "video analysis is not available in this build"
        raise RuntimeError(msg)

    clips = list(paths.find_raw_clips())
    footage = analyzer.analyze_project(clips)
    paths.footage_analysis_file.parent.mkdir(parents=True, exist_ok=True)
    paths.footage_analysis_file.write_text(footage.model_dump_json(indent=2), encoding="utf-8")

    return StepResult(
        summary=f"{len(footage.scenes)} scene(s) across {len(footage.clips)} clip(s)",
        detail="\n".join(
            f"{scene.key} q={scene.quality.overall:.2f} {scene.motion.level.value}"
            for scene in footage.scenes
        ),
        artifact=paths.footage_analysis_file,
    )


def _analyze_music(paths: ProjectPaths, container: object) -> StepResult:
    from app.models.audio import MusicLibrary
    from app.services.container import Container

    assert isinstance(container, Container)
    analyzer = container.music
    if analyzer is None:  # pragma: no cover - always wired for a project
        msg = "music analysis is not available in this build"
        raise RuntimeError(msg)

    tracks = tuple((path, paths.to_ref(path)) for path in paths.find_music())
    library, failures = analyzer.analyze_library(tracks)
    paths.music_library_file.parent.mkdir(parents=True, exist_ok=True)
    paths.music_library_file.write_text(library.model_dump_json(indent=2), encoding="utf-8")

    assert isinstance(library, MusicLibrary)
    return StepResult(
        summary=f"{len(library.tracks)} track(s), {len(failures)} unreadable",
        detail="\n".join(
            f"{track.source} bpm={track.bpm or '?'} energy={track.energy:.2f} "
            f"lufs={track.loudness_lufs or '?'}"
            for track in library.tracks
        ),
        artifact=paths.music_library_file,
    )


def _brief(paths: ProjectPaths, container: object) -> StepResult:
    from app.services.container import Container

    assert isinstance(container, Container)
    narration, footage = _load_analyses(paths)
    builder = container.brief
    if builder is None:  # pragma: no cover - always wired
        msg = "the planner is not available in this build"
        raise RuntimeError(msg)

    assembled = builder.build(project_id=paths.root.name, narration=narration, footage=footage)
    paths.planning_brief_file.write_text(assembled.model_dump_json(indent=2), encoding="utf-8")
    coverage = assembled.coverage
    return StepResult(
        summary=f"{len(assembled.beats)} beat(s), feasible={str(coverage.feasible).lower()}",
        detail="\n".join(f"{item.beat_index:03d} {item.text}" for item in assembled.beats),
        artifact=paths.planning_brief_file,
    )


def _draft(paths: ProjectPaths, container: object) -> StepResult:
    from app.services.container import Container

    assert isinstance(container, Container)
    narration, footage = _load_analyses(paths)
    builder, drafter = container.brief, container.drafter
    if builder is None or drafter is None:  # pragma: no cover - always wired
        msg = "the planner is not available in this build"
        raise RuntimeError(msg)

    assembled = builder.build(project_id=paths.root.name, narration=narration, footage=footage)
    plan = drafter.draft(assembled, narration=narration)
    paths.edit_plan_file.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return StepResult(
        summary=f"{len(plan.clips)} clip(s) - a baseline, not an edit",
        detail="Every clip was chosen by duration and quality. Revise the selections "
        "and the reasons before rendering.",
        artifact=paths.edit_plan_file,
    )


def _normalize(paths: ProjectPaths, container: object) -> StepResult:
    from app.models.speech import NarrationAnalysis
    from app.models.video import FootageAnalysis
    from app.plan.loader import load_plan
    from app.rule_engine.context import RuleContext
    from app.services.container import Container

    assert isinstance(container, Container)
    engine = container.rules
    if engine is None:  # pragma: no cover - always wired
        msg = "the rule engine is not available in this build"
        raise RuntimeError(msg)

    plan = load_plan(paths.edit_plan_file).plan
    settings = container.settings
    # Built from whatever analysis exists. RuleContext is deliberately tolerant of missing
    # pieces, so normalising before `analyze video` still places clips - it simply cannot
    # check source bounds for footage it has never measured.
    context = RuleContext.build(
        settings.rules,
        subtitle=settings.subtitle,
        project_root=paths.root,
        footage=(
            FootageAnalysis.model_validate_json(
                paths.footage_analysis_file.read_text(encoding="utf-8")
            )
            if paths.footage_analysis_file.is_file()
            else None
        ),
        narration=(
            NarrationAnalysis.model_validate_json(paths.narration_file.read_text(encoding="utf-8"))
            if paths.narration_file.is_file()
            else None
        ),
    )
    normalised, report = engine.normalize(plan, context)
    paths.edit_plan_file.write_text(normalised.model_dump_json(indent=2), encoding="utf-8")

    return StepResult(
        summary=f"{len(report.issues)} note(s), placed={str(normalised.is_placed).lower()}",
        detail="\n".join(f"{issue.severity.value}: {issue.message}" for issue in report.issues),
        artifact=paths.edit_plan_file,
    )


def _subtitles(paths: ProjectPaths, container: object) -> StepResult:
    from app.models.speech import NarrationAnalysis
    from app.plan.loader import load_plan
    from app.services.container import Container
    from app.subtitles.attach import attach_subtitles

    assert isinstance(container, Container)
    plan = load_plan(paths.edit_plan_file).plan
    narration = NarrationAnalysis.model_validate_json(
        paths.narration_file.read_text(encoding="utf-8")
    )
    updated, count = attach_subtitles(
        plan, narration.transcript, settings=container.settings.subtitle
    )
    paths.edit_plan_file.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
    return StepResult(
        summary=f"{count} cue(s) attached",
        detail="\n".join(
            f"{cue.range.start:.2f}-{cue.range.end:.2f} {cue.text}" for cue in updated.subtitles
        ),
        artifact=paths.edit_plan_file,
    )


def _render(
    paths: ProjectPaths, container: object, *, draft: bool, report: ProgressFn
) -> StepResult:
    from app.plan.loader import load_plan
    from app.renderer.base import RenderRequest
    from app.services.container import Container

    assert isinstance(container, Container)
    renderer = container.renderer
    if renderer is None:  # pragma: no cover - always wired
        msg = "the renderer is not available in this build"
        raise RuntimeError(msg)

    plan = load_plan(paths.edit_plan_file).plan
    destination = paths.output / ("draft.mp4" if draft else paths.final_video.name)
    request = RenderRequest(
        plan=plan, project_root=paths.root, destination=destination, draft=draft
    )

    problems = renderer.preflight(request)
    if problems:
        msg = "; ".join(problems)
        raise ValueError(msg)

    result = renderer.render(request, on_progress=report)
    return StepResult(
        summary=f"{result.duration:.2f}s rendered in {result.elapsed:.1f}s",
        detail=str(result.video),
        artifact=result.video,
    )


def _export(paths: ProjectPaths, container: object) -> StepResult:
    from app.exporters.base import ExportRequest
    from app.exporters.capcut.exporter import default_draft_dir
    from app.plan.loader import load_plan
    from app.services.container import Container

    assert isinstance(container, Container)
    plan = load_plan(paths.edit_plan_file).plan
    root = default_draft_dir(container.settings)
    if root is None:
        msg = (
            "no CapCut draft folder was found. Create one project in CapCut so the "
            "folder exists, then try again."
        )
        raise FileNotFoundError(msg)

    exporter = container.exporters.get("capcut")
    result = exporter.export(
        ExportRequest(
            plan=plan,
            project_root=paths.root,
            destination=root / plan.project_id,
            copy_media=container.settings.capcut.copy_media,
            project_name=plan.project_id,
        )
    )
    detail_lines = [
        str(result.project_dir),
        result.open_hint or "",
        *result.warnings,
    ]
    return StepResult(
        summary=f"CapCut draft ready — restart CapCut, open Drafts → {plan.project_id}",
        detail="\n".join(line for line in detail_lines if line),
        artifact=result.project_dir,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_analyses(paths: ProjectPaths) -> tuple[NarrationAnalysis, FootageAnalysis]:
    """Both analysis documents. Typed rather than ``object``: the loose annotation this
    replaces hid a real error, where a caller read a field ``BeatCandidates`` does not have."""
    narration = NarrationAnalysis.model_validate_json(
        paths.narration_file.read_text(encoding="utf-8")
    )
    footage = FootageAnalysis.model_validate_json(
        paths.footage_analysis_file.read_text(encoding="utf-8")
    )
    return narration, footage


def _load_plan_quietly(path: Path) -> EditPlan | None:
    """The plan, or ``None`` if it is absent or unreadable.

    Never raises. This runs every time the window refreshes, and a plan the user is
    part-way through hand-editing must not crash the app that is showing it to them.
    """
    if not path.is_file():
        return None
    try:
        from app.plan.loader import load_plan

        return load_plan(path).plan
    except Exception as exc:
        logger.debug("Could not load %s: %s", path, exc)
        return None


__all__ = [
    "STEPS",
    "STEPS_BY_ID",
    "ProgressFn",
    "ProjectState",
    "Step",
    "StepId",
    "StepResult",
    "blockers",
    "can_run",
    "next_suggested",
    "run_step",
]
