"""Project folder layout.

Kept out of :mod:`app.models` on purpose. Models are serialisable values that must
survive being copied between machines, so they hold *relative*
:class:`~app.models.common.MediaRef` paths only. This module holds the absolute,
machine-specific side of that mapping, and is the one place allowed to touch the
filesystem to resolve it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.models.common import MediaKind, MediaRef

RAW_DIRNAME = "raw"
MUSIC_DIRNAME = "music"
CAPCUT_DIRNAME = "capcut"
OUTPUT_DIRNAME = "output"
AUDIO_DIRNAME = "audio"
CACHE_DIRNAME = ".aive"

VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mts", ".mpg", ".mpeg"}
)
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"})

NARRATION_STEMS = ("narration", "voice", "voiceover", "vo")
"""Filenames treated as the narration track when found at the project root.

Ordered by preference. ``audio/`` is also searched, because that is the layout in
the original brief and users do both.
"""


def classify(path: Path) -> MediaKind:
    """Best guess at the role of a media file from its extension alone."""
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return MediaKind.RAW_VIDEO
    if suffix in AUDIO_EXTENSIONS:
        return MediaKind.NARRATION if path.stem.lower() in NARRATION_STEMS else MediaKind.MUSIC
    return MediaKind.UNKNOWN


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Absolute paths for one project folder.

    Construct with :meth:`for_root`, which resolves symlinks once so that every
    containment check downstream compares canonical paths.
    """

    root: Path

    @classmethod
    def for_root(cls, root: Path) -> ProjectPaths:
        return cls(root=root.expanduser().resolve())

    # -- Directories -------------------------------------------------------- #

    @property
    def raw(self) -> Path:
        """Raw footage."""
        return self.root / RAW_DIRNAME

    @property
    def music(self) -> Path:
        """User-supplied background music."""
        return self.root / MUSIC_DIRNAME

    @property
    def audio(self) -> Path:
        """Optional folder for narration, as an alternative to a root-level file."""
        return self.root / AUDIO_DIRNAME

    @property
    def capcut(self) -> Path:
        """CapCut template projects and resources."""
        return self.root / CAPCUT_DIRNAME

    @property
    def output(self) -> Path:
        """Renders, subtitles and logs."""
        return self.root / OUTPUT_DIRNAME

    @property
    def logs(self) -> Path:
        return self.output / "logs"

    @property
    def preview(self) -> Path:
        return self.output / "preview"

    @property
    def cache(self) -> Path:
        """Derived analysis. Entirely regenerable, and excluded from git."""
        return self.root / CACHE_DIRNAME

    @property
    def keyframes(self) -> Path:
        """Extracted stills, under the cache so they are never mistaken for input."""
        return self.cache / "keyframes"

    @property
    def analysis(self) -> Path:
        """Per-clip analysis documents."""
        return self.cache / "analysis"

    # -- Well-known files --------------------------------------------------- #

    @property
    def config_file(self) -> Path:
        return self.root / "aive.toml"

    @property
    def manifest_file(self) -> Path:
        return self.cache / "manifest.json"

    @property
    def narration_file(self) -> Path:
        """Transcript, cleanup and beats as one document.

        One file rather than three because the parts are only meaningful together: a
        transcript paired with a stale cleanup report would mis-time every subtitle.
        """
        return self.cache / "narration.json"

    @property
    def music_library_file(self) -> Path:
        return self.cache / "music.json"

    @property
    def footage_analysis_file(self) -> Path:
        return self.cache / "footage.json"

    @property
    def planning_brief_file(self) -> Path:
        """The assembled brief. Derived, so it lives in the cache."""
        return self.cache / "brief.json"

    @property
    def edit_plan_file(self) -> Path:
        return self.root / "edit_plan.json"

    @property
    def final_video(self) -> Path:
        return self.output / "final.mp4"

    def subtitle_file(self, extension: str) -> Path:
        """``output/subtitle.srt`` or ``output/subtitle.ass``."""
        return self.output / f"subtitle.{extension.lstrip('.')}"

    # -- Operations --------------------------------------------------------- #

    def all_directories(self) -> tuple[Path, ...]:
        """Every directory a project needs, in creation order."""
        return (
            self.root,
            self.raw,
            self.music,
            self.capcut,
            self.output,
            self.logs,
            self.preview,
            self.cache,
            self.keyframes,
            self.analysis,
        )

    def ensure(self) -> tuple[Path, ...]:
        """Create any missing directories. Returns the ones actually created."""
        created: list[Path] = []
        for directory in self.all_directories():
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created.append(directory)
        return tuple(created)

    def exists(self) -> bool:
        """True when this looks like an initialised project."""
        return self.root.is_dir() and self.raw.is_dir()

    def to_ref(self, path: Path) -> MediaRef:
        """Convert an absolute path inside the project into a portable reference."""
        return MediaRef.from_path(path, root=self.root)

    def resolve(self, ref: MediaRef) -> Path:
        """Absolute path for a reference, refusing anything outside the project."""
        return ref.resolve_within(self.root)

    # -- Discovery ---------------------------------------------------------- #

    def find_narration(self) -> Path | None:
        """Locate the narration track.

        Looks for a preferred stem at the project root first, then any audio file
        in ``audio/``. Returns ``None`` when there is nothing, and picks
        deterministically when there are several, so repeated scans agree.
        """
        for stem in NARRATION_STEMS:
            matches = sorted(
                candidate
                for candidate in self.root.glob(f"{stem}.*")
                if candidate.suffix.lower() in AUDIO_EXTENSIONS and candidate.is_file()
            )
            if matches:
                return matches[0]
        if self.audio.is_dir():
            matches = sorted(
                candidate
                for candidate in self.audio.iterdir()
                if candidate.suffix.lower() in AUDIO_EXTENSIONS and candidate.is_file()
            )
            if matches:
                return matches[0]
        return None

    def find_raw_clips(self) -> tuple[Path, ...]:
        """Video files under ``raw/``, sorted by name.

        Name order matters: users number their footage ``001.mp4``, ``002.mp4``,
        and that ordering is a real editorial hint about shooting sequence.
        """
        return self._sorted_media(self.raw, VIDEO_EXTENSIONS)

    def find_music(self) -> tuple[Path, ...]:
        """Audio files under ``music/``, sorted by name."""
        return self._sorted_media(self.music, AUDIO_EXTENSIONS)

    def find_capcut_templates(self) -> tuple[Path, ...]:
        """Candidate CapCut template projects under ``capcut/``.

        A CapCut draft is a *directory* containing ``draft_content.json``, so that
        marker file is what identifies one.
        """
        if not self.capcut.is_dir():
            return ()
        return tuple(
            sorted(
                candidate
                for candidate in self.capcut.iterdir()
                if candidate.is_dir() and (candidate / "draft_content.json").is_file()
            )
        )

    @staticmethod
    def _sorted_media(directory: Path, extensions: frozenset[str]) -> tuple[Path, ...]:
        if not directory.is_dir():
            return ()
        return tuple(
            sorted(
                candidate
                for candidate in directory.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in extensions
            )
        )


__all__ = [
    "AUDIO_DIRNAME",
    "AUDIO_EXTENSIONS",
    "CACHE_DIRNAME",
    "CAPCUT_DIRNAME",
    "MUSIC_DIRNAME",
    "NARRATION_STEMS",
    "OUTPUT_DIRNAME",
    "RAW_DIRNAME",
    "VIDEO_EXTENSIONS",
    "ProjectPaths",
    "classify",
]
