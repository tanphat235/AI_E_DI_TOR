"""Dependency injection.

A plain frozen dataclass assembled by a factory function - no DI framework. The
brief asks to avoid unnecessary dependencies, and for a container this shallow a
framework would add configuration to learn without removing any work.

What matters is not the mechanism but the direction: nothing constructs its own
collaborators. Every component receives what it needs, every boundary is a
:class:`typing.Protocol`, and so every component can be tested with a hand-written
fake and no media, no models and no FFmpeg.

Phase 1 wires only what Phase 1 implements: settings, paths, the FFmpeg locator and
an empty exporter registry. The optional slots are declared so later phases fill in
a field instead of changing this signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.analysis.audio.analyzer import DefaultMusicAnalyzer
from app.analysis.speech.base import NarrationCleaner, SilenceDetector, SpeechRecognizer
from app.analysis.speech.cleanup import NarrationCleanupService
from app.analysis.speech.silence import FFmpegSilenceDetector
from app.analysis.speech.whisper_recognizer import FasterWhisperRecognizer
from app.analysis.vision.analyzer import DefaultFootageAnalyzer
from app.analysis.vision.base import FootageAnalyzer, MediaProber, VisionProvider
from app.analysis.vision.classical import ClassicalCvVisionProvider
from app.analysis.vision.probe import PyAvProber
from app.config.settings import AiveSettings, load_settings
from app.exporters.base import ExporterRegistry
from app.exporters.capcut.exporter import CapCutExporter
from app.planner.brief import BriefBuilder
from app.planner.draft import HeuristicDrafter
from app.renderer.base import Renderer
from app.renderer.ffmpeg.renderer import FFmpegRenderer
from app.rule_engine.base import RuleEngine
from app.rule_engine.engine import DefaultRuleEngine
from app.services.ffmpeg_locator import FFmpegLocator
from app.services.paths import ProjectPaths


@dataclass(frozen=True, slots=True)
class Container:
    """The assembled object graph for one invocation.

    Optional fields are the phases not yet built. They are ``None`` rather than
    absent so that a command can check for a capability and fail with something
    useful - "video analysis is not available in this build" beats an
    ``AttributeError``.
    """

    settings: AiveSettings
    paths: ProjectPaths | None
    ffmpeg: FFmpegLocator
    exporters: ExporterRegistry = field(default_factory=ExporterRegistry)

    # Phase 2.
    recognizer: SpeechRecognizer | None = None
    cleaner: NarrationCleaner | None = None
    silence: SilenceDetector | None = None

    # Phase 3. These need a project directory, because analysis writes keyframes into
    # its cache, so they are None for project-less commands such as `doctor`.
    prober: MediaProber | None = None
    vision: VisionProvider | None = None
    footage: FootageAnalyzer | None = None

    # Phase 4. Needs no project: it judges a plan against a context the caller supplies.
    rules: RuleEngine | None = None

    # Phase 5. Support for the director, not a planner: these rank and assemble, and make
    # no editorial decision.
    brief: BriefBuilder | None = None
    drafter: HeuristicDrafter | None = None

    # Phase 7. The renderer needs no project - it is handed a request that names its own
    # root - but music analysis writes into the project cache, so that one does.
    renderer: Renderer | None = None
    music: DefaultMusicAnalyzer | None = None

    def require_paths(self) -> ProjectPaths:
        """The project paths, or a clear error when no project was given.

        Commands like ``doctor`` and ``schema`` run without a project, so
        ``paths`` is genuinely optional; this turns that into one explicit failure
        rather than a scattering of ``if self.paths is None`` checks.
        """
        if self.paths is None:
            msg = "this command needs a project directory; pass one as an argument"
            raise ValueError(msg)
        return self.paths


def build_container(
    project_dir: Path | None = None,
    *,
    settings: AiveSettings | None = None,
) -> Container:
    """Assemble the object graph.

    Args:
        project_dir: Project root. ``None`` for commands that need no project.
        settings: Pre-loaded settings. Loaded from the layered config when omitted;
            pass explicitly in tests to avoid depending on files on disk.
    """
    resolved_settings = settings if settings is not None else load_settings(project_dir)
    paths = ProjectPaths.for_root(project_dir) if project_dir is not None else None
    locator = FFmpegLocator(resolved_settings.media)

    # Constructing these is free. The recogniser imports ctranslate2 and downloads
    # weights only on its first `transcribe` call, and the silence detector only shells
    # out to ffmpeg when asked to detect - so `aive doctor` pays nothing for either.
    silence = FFmpegSilenceDetector(locator)

    # Video analysis writes keyframes into the project cache, so it cannot be built
    # without a project. Constructing it is still free - PySceneDetect and OpenCV are
    # imported lazily inside the methods that use them.
    footage = DefaultFootageAnalyzer(resolved_settings, paths) if paths is not None else None
    vision = (
        ClassicalCvVisionProvider(resolved_settings.vision, project_root=paths.root)
        if paths is not None
        else None
    )

    # Registered here rather than at import time, so importing an exporter has no side
    # effects and a test can see exactly which exporters a given run had.
    # The prober is *injected*: an exporter that imports an analyser cannot be swapped out,
    # and would drag PyAV into anything that merely wants to write a project file.
    exporters = ExporterRegistry()
    exporters.register(
        CapCutExporter(resolved_settings, prober=PyAvProber(), ffmpeg=locator)
    )

    return Container(
        settings=resolved_settings,
        paths=paths,
        ffmpeg=locator,
        exporters=exporters,
        recognizer=FasterWhisperRecognizer(resolved_settings.speech),
        silence=silence,
        cleaner=NarrationCleanupService(silence, resolved_settings.rules),
        prober=PyAvProber(),
        rules=DefaultRuleEngine(),
        brief=BriefBuilder(resolved_settings),
        drafter=HeuristicDrafter(resolved_settings),
        vision=vision,
        footage=footage,
        # Constructing both is free: the renderer resolves the binary lazily inside
        # preflight, and the analyser imports numpy only when it decodes.
        renderer=FFmpegRenderer(resolved_settings, locator),
        music=DefaultMusicAnalyzer(resolved_settings, locator),
    )


__all__ = ["Container", "build_container"]
