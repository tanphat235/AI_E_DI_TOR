"""Turning an Edit Plan into a CapCut draft.

The point of this exporter is the *handoff*. A render is finished; a draft is something the
user can keep editing — retime a cut, restyle a subtitle, swap a shot — with the AI's
decisions already laid out and its reasoning visible in the track names.

Every field name and magic constant lives in :mod:`.schema`; this module contains none.
That separation is what makes a CapCut format change a one-file repair rather than an
archaeology exercise.

## Three decisions worth knowing

**Media is copied by default.** A draft that references files elsewhere on disk breaks
silently the moment the user reorganises their footage, and CapCut's failure mode is a
timeline of red placeholders with no explanation. Copying costs disk; ``copy_media=False``
is available for a user who knows their layout is stable.

**Transitions are attached to the outgoing segment.** CapCut models a transition as a
property of the clip it leaves, whereas the Edit Plan models it as ``transition_in`` on the
clip it enters. Translating between the two is a real off-by-one hazard: attach it to the
wrong segment and the dissolve appears one cut early.

**Nothing is written until everything is built.** The draft directory is assembled in memory
and written at the end, because a half-written draft does not merely fail to open — it makes
CapCut's entire project list unusable until the user finds and deletes it by hand.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.config.settings import AiveSettings
from app.exporters.base import ExportRequest, ExportResult
from app.exporters.capcut import schema
from app.exporters.capcut.locate import find_draft_dir
from app.models.common import AspectRatio, MediaRef, TransitionKind
from app.models.edit_plan import EditPlan, MusicCue, OutputSpec, SubtitleCue
from app.models.media import MediaProbe
from app.services.ffmpeg_locator import FFmpegLocator, FFmpegNotFoundError
from app.utils.logging import get_logger

logger = get_logger(__name__)

EXPORTER_VERSION = f"capcut/1+{schema.TARGET_CAPCUT_VERSION}"

# CapCut's preview player often fails on widths/heights that are not multiples of 8
# (screen captures like 1908x1032 are a common case). Thumbnails can still appear.
_DIM_MULTIPLE = 8

_UNSAFE_MEDIA_CHARS = re.compile(r"[^\w.\-]+", re.UNICODE)

# Template drafts carry a Timelines/ tree CapCut regenerates itself. Copying it leaves
# a second timeline that can open with a black player while the root draft is fine.
_SKIP_TEMPLATE_NAMES = frozenset(
    {
        schema.DRAFT_CONTENT_NAME,
        schema.DRAFT_META_NAME,
        "draft_content.json.bak",
        "Timelines",
        "media",
    }
)

CAPCUT_TRANSITIONS: dict[TransitionKind, str] = {
    TransitionKind.FADE: "Fade",
    TransitionKind.DISSOLVE: "Dissolve",
    TransitionKind.CROSSFADE: "Dissolve",
    TransitionKind.SLIDE_LEFT: "Slide left",
    TransitionKind.SLIDE_RIGHT: "Slide right",
    TransitionKind.SLIDE_UP: "Slide up",
    TransitionKind.SLIDE_DOWN: "Slide down",
    TransitionKind.ZOOM_IN: "Zoom in",
    TransitionKind.ZOOM_OUT: "Zoom out",
}
"""Plan transitions to CapCut's display names.

These are names, not effect ids. CapCut resolves a transition by ``effect_id`` against its
own downloadable library, and an id that is not installed renders as a hard cut. Supplying a
name with an empty id gets a hard cut too, but it gets a *labelled* one the user can see and
replace in the UI — a visible, fixable placeholder rather than a silent omission.
"""


class CapCutExportError(RuntimeError):
    """Raised when a draft cannot be written."""


@dataclass(frozen=True, slots=True)
class _Media:
    """A source file resolved for export, and where it will live in the draft."""

    ref: MediaRef
    source: Path
    destination: Path
    material_id: str
    duration: float
    width: int
    height: int
    has_audio: bool
    normalize_size: tuple[int, int] | None = None
    """When set, the copy pass re-encodes to this even size so CapCut's preview can play it."""

    @property
    def draft_path(self) -> str:
        """The path CapCut stores. Backslashes, because that is what it writes itself."""
        return str(self.destination).replace("/", "\\")

    @property
    def export_width(self) -> int:
        if self.normalize_size is not None:
            return self.normalize_size[0]
        return self.width

    @property
    def export_height(self) -> int:
        if self.normalize_size is not None:
            return self.normalize_size[1]
        return self.height


class MediaFacts(Protocol):
    """The only thing this exporter needs from the outside world: a file's real properties.

    Declared here, and satisfied by :class:`~app.analysis.vision.probe.PyAvProber`, so the
    concrete prober is *injected by the container* rather than imported. An exporter that
    imports an analyser can no longer be swapped out, and it drags PyAV into anything that
    merely wants to write a project file — the rule
    ``tests/unit/test_architecture.py::TestRendererAndExportersConsumeOnlyThePlan`` exists to
    catch exactly the shortcut this replaces.
    """

    def probe(self, source: Path, *, ref: MediaRef) -> MediaProbe: ...


class CapCutExporter:
    """An :class:`~app.exporters.base.Exporter` producing a CapCut draft directory."""

    def __init__(
        self,
        settings: AiveSettings,
        prober: MediaFacts | None = None,
        ffmpeg: FFmpegLocator | None = None,
    ) -> None:
        self._settings = settings
        self._prober = prober
        self._ffmpeg = ffmpeg

    @property
    def name(self) -> str:
        return "capcut"

    @property
    def display_name(self) -> str:
        return "CapCut Desktop"

    @property
    def version(self) -> str:
        return EXPORTER_VERSION

    # -- Preflight ----------------------------------------------------------- #

    def preflight(self, request: ExportRequest) -> tuple[str, ...]:
        """Everything that would stop this export, found before anything is written."""
        problems: list[str] = []
        plan = request.plan

        if not plan.is_placed:
            problems.append(
                "the plan has no timeline positions, so segments cannot be laid out; "
                "run `aive rules normalize` first"
            )

        seen: set[Path] = set()
        for clip in plan.clips:
            try:
                resolved = clip.source.resolve_within(request.project_root)
            except ValueError as exc:
                problems.append(f"clip {clip.id}: {exc}")
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if not resolved.is_file():
                problems.append(f"clip {clip.id}: {clip.source} does not exist")

        if plan.narration is not None:
            narration = plan.narration.source.resolve_within(request.project_root)
            if not narration.is_file():
                problems.append(f"narration {plan.narration.source} does not exist")

        for position, cue in enumerate(plan.music):
            if not cue.track.resolve_within(request.project_root).is_file():
                problems.append(f"music cue {position}: {cue.track} does not exist")

        if request.template is not None and not request.template.is_dir():
            problems.append(f"template {request.template} is not a directory")

        destination = request.destination
        if destination.exists() and not destination.is_dir():
            problems.append(f"{destination} exists and is not a directory")

        return tuple(problems)

    # -- Export -------------------------------------------------------------- #

    def export(self, request: ExportRequest) -> ExportResult:
        """Write the draft.

        Raises:
            CapCutExportError: preflight found a blocking problem, or writing failed.
        """
        problems = self.preflight(request)
        if problems:
            raise CapCutExportError("; ".join(problems))

        plan = request.plan
        name = request.project_name or plan.project_id
        draft_dir = request.destination
        warnings: list[str] = []

        media = self._resolve_media(request)
        written, copied, media = self._write(
            request, draft_dir, media=media, warnings=warnings
        )
        content = self._build_content(plan, name=name, media=media, warnings=warnings)
        meta = self._build_meta(plan, name=name, draft_dir=draft_dir, media=media)

        json_written = self._write_documents(draft_dir, content=content, meta=meta)
        cover = self._ensure_cover(draft_dir, media=media)
        files = [*json_written, *written]
        if cover is not None:
            files.append(cover)

        logger.info("Wrote CapCut draft %s (%d file(s) copied)", draft_dir, len(copied))
        return ExportResult(
            project_dir=draft_dir,
            files_written=tuple(files),
            media_copied=copied,
            warnings=tuple(warnings),
            open_hint=(
                "restart CapCut if it is running, then open Drafts and select "
                f"{name!r} at {draft_dir}. CapCut only refreshes that list at startup."
            ),
        )

    # -- Media --------------------------------------------------------------- #

    def _resolve_media(self, request: ExportRequest) -> dict[MediaRef, _Media]:
        """Every distinct source file the plan uses, with its intrinsic properties.

        Probed once per *file*, not per clip: a plan cutting forty times from one take must
        not open that file forty times.
        """
        plan = request.plan
        refs: list[MediaRef] = []
        for clip in plan.clips:
            if clip.source not in refs:
                refs.append(clip.source)
        if plan.narration is not None and plan.narration.source not in refs:
            refs.append(plan.narration.source)
        for cue in plan.music:
            if cue.track not in refs:
                refs.append(cue.track)

        canvas_w, canvas_h, _ratio = capcut_canvas(plan.output)

        media: dict[MediaRef, _Media] = {}
        for ref in refs:
            source = ref.resolve_within(request.project_root)
            safe_name = safe_media_filename(source.name)
            destination = (
                request.destination / "media" / safe_name if request.copy_media else source
            )
            duration, width, height, has_audio = self._probe(source)
            normalize_size: tuple[int, int] | None = None
            # Pad every video onto the CapCut canvas so the draft uses a standard ratio,
            # not the source's free aspect (screen captures are rarely exactly 16:9).
            if width > 0 and (
                width != canvas_w
                or height != canvas_h
                or _needs_preview_normalize(width, height)
            ):
                normalize_size = (canvas_w, canvas_h)
            media[ref] = _Media(
                ref=ref,
                source=source,
                destination=destination,
                material_id=schema.stable_id("material", str(ref)),
                duration=duration,
                width=width,
                height=height,
                has_audio=has_audio,
                normalize_size=normalize_size,
            )
        return media

    def _probe(self, source: Path) -> tuple[float, int, int, bool]:
        """Duration and geometry, or conservative fallbacks.

        Falls back rather than failing, and does so when no prober was injected at all. A
        draft with an understated duration is inconvenient - the user cannot extend a clip
        past it in the UI - but a draft that was never written because one file would not
        probe is worse.
        """
        fallback = (3600.0, self._settings.output.width, self._settings.output.height, True)
        if self._prober is None:
            logger.debug("No prober available; using fallback media facts for %s", source.name)
            return fallback

        try:
            probe = self._prober.probe(source, ref=MediaRef(path=source.name))
        except (ValueError, OSError, RuntimeError) as exc:
            logger.warning("Could not probe %s: %s", source.name, exc)
            return fallback

        video = probe.video
        if video is None:
            return probe.duration, 0, 0, True
        width, height = video.display_size
        return probe.duration, width, height, probe.has_audio

    # -- draft_content.json -------------------------------------------------- #

    def _build_content(
        self,
        plan: EditPlan,
        *,
        name: str,
        media: dict[MediaRef, _Media],
        warnings: list[str],
    ) -> dict[str, Any]:
        materials: dict[str, list[dict[str, Any]]] = {
            "videos": [],
            "audios": [],
            "texts": [],
            "transitions": [],
            "speeds": [],
            "canvases": [],
            "sound_channel_mappings": [],
            "vocal_separations": [],
            "audio_fades": [],
        }
        tracks: list[dict[str, Any]] = []

        for item in media.values():
            if item.width > 0:
                materials["videos"].append(
                    schema.video_material(
                        material_id=item.material_id,
                        path=item.draft_path,
                        name=item.destination.name,
                        duration=item.duration,
                        width=item.export_width,
                        height=item.export_height,
                        has_audio=item.has_audio,
                    )
                )
            else:
                materials["audios"].append(
                    schema.audio_material(
                        material_id=item.material_id,
                        path=item.draft_path,
                        name=item.destination.name,
                        duration=item.duration,
                    )
                )

        tracks.append(self._video_track(plan, media=media, materials=materials, warnings=warnings))
        tracks.extend(self._audio_tracks(plan, media=media, materials=materials))
        if plan.subtitles:
            tracks.append(self._text_track(plan, materials=materials))

        canvas_w, canvas_h, ratio = capcut_canvas(plan.output)
        return schema.draft_content(
            draft_id=schema.stable_id("draft", plan.project_id),
            name=name,
            width=canvas_w,
            height=canvas_h,
            fps=plan.output.fps,
            duration=plan.timeline_duration,
            materials=materials,
            tracks=tracks,
            ratio=ratio,
        )

    def _video_track(
        self,
        plan: EditPlan,
        *,
        media: dict[MediaRef, _Media],
        materials: dict[str, list[dict[str, Any]]],
        warnings: list[str],
    ) -> dict[str, Any]:
        segments: list[dict[str, Any]] = []
        # Segments are laid end to end, NOT at the plan's timeline_start.
        #
        # The two formats disagree about who owns the transition overlap. The Edit Plan has
        # already applied it: `rules normalize` pulls each clip back by its incoming
        # transition, so consecutive clips overlap on the timeline. CapCut instead stores
        # contiguous segments and applies the overlap itself when a transition is marked
        # `is_overlap`. Exporting the plan's positions directly produces segments that
        # occupy the same instant on one track, which a track - being a sequence - cannot
        # represent.
        #
        # Laid out this way the two agree after CapCut collapses the overlaps: the track's
        # effective length becomes sum(durations) - sum(transitions), which is exactly the
        # plan's timeline duration, so the audio tracks below still line up.
        cursor = 0.0

        for index, clip in enumerate(plan.clips):
            item = media[clip.source]
            key = f"{plan.project_id}:{clip.id}"
            extra_refs = self._helper_materials(key, materials, speed=clip.speed)

            # The Edit Plan puts a transition on the clip it *enters*; CapCut puts it on the
            # segment it *leaves*. Translated here, once, rather than at every use.
            outgoing = plan.clips[index + 1] if index + 1 < len(plan.clips) else None
            if outgoing is not None and outgoing.transition_in is not None:
                transition = outgoing.transition_in
                if not transition.kind.is_instant:
                    label = CAPCUT_TRANSITIONS.get(transition.kind)
                    if label is None:
                        warnings.append(
                            f"clip {clip.id}: no CapCut equivalent for "
                            f"{transition.kind.value!r}; the cut is hard in the draft"
                        )
                    else:
                        transition_id = schema.stable_id("transition", key)
                        materials["transitions"].append(
                            schema.transition_material(
                                material_id=transition_id,
                                name=label,
                                duration=transition.duration,
                            )
                        )
                        extra_refs.append(transition_id)
                        warnings.append(
                            f"clip {clip.id}: the {label!r} transition is a labelled "
                            "placeholder - CapCut resolves transitions from its own "
                            "library, so pick it again in the UI to make it render"
                        )

            segments.append(
                schema.video_segment(
                    segment_id=schema.stable_id("segment", key),
                    material_id=item.material_id,
                    source_start=clip.source_range.start,
                    source_duration=clip.source_range.duration,
                    target_start=cursor,
                    target_duration=clip.timeline_duration,
                    speed=clip.speed,
                    volume=0.0 if clip.mute_source_audio else 1.0,
                    extra_refs=extra_refs,
                    render_index=index,
                )
            )
            cursor += clip.timeline_duration

        return schema.track(
            track_id=schema.stable_id("track", f"{plan.project_id}:video"),
            kind="video",
            segments=segments,
        )

    def _audio_tracks(
        self,
        plan: EditPlan,
        *,
        media: dict[MediaRef, _Media],
        materials: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Narration on one track, music on another.

        Separate tracks, deliberately: the first thing a user does in CapCut is rebalance
        voice against music, and that is a two-slider job only if they are not interleaved.
        """
        tracks: list[dict[str, Any]] = []

        if plan.narration is not None:
            item = media[plan.narration.source]
            segments = []
            # Each kept range becomes its own segment, laid end to end. That reassembly is
            # what removed the silence, and keeping the seams visible lets the user undo an
            # individual cut rather than the whole cleanup.
            cursor = 0.0
            for position, kept in enumerate(plan.narration.kept_ranges):
                key = f"{plan.project_id}:narration:{position}"
                segments.append(
                    schema.audio_segment(
                        segment_id=schema.stable_id("segment", key),
                        material_id=item.material_id,
                        source_start=kept.start,
                        source_duration=kept.duration,
                        target_start=cursor,
                        target_duration=kept.duration,
                        volume=_gain_to_volume(plan.narration.gain_db),
                        extra_refs=self._helper_materials(key, materials),
                        render_index=position,
                    )
                )
                cursor += kept.duration
            tracks.append(
                schema.track(
                    track_id=schema.stable_id("track", f"{plan.project_id}:narration"),
                    kind="audio",
                    segments=segments,
                )
            )

        if plan.music:
            segments = [
                self._music_segment(plan, cue, position, media=media, materials=materials)
                for position, cue in enumerate(plan.music)
            ]
            tracks.append(
                schema.track(
                    track_id=schema.stable_id("track", f"{plan.project_id}:music"),
                    kind="audio",
                    segments=segments,
                )
            )

        return tracks

    def _music_segment(
        self,
        plan: EditPlan,
        cue: MusicCue,
        position: int,
        *,
        media: dict[MediaRef, _Media],
        materials: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        item = media[cue.track]
        key = f"{plan.project_id}:music:{position}"
        extra_refs = self._helper_materials(key, materials)

        if cue.fade_in > 0.0 or cue.fade_out > 0.0:
            fade_id = schema.stable_id("fade", key)
            materials["audio_fades"].append(
                schema.audio_fade(material_id=fade_id, fade_in=cue.fade_in, fade_out=cue.fade_out)
            )
            extra_refs.append(fade_id)

        return schema.audio_segment(
            segment_id=schema.stable_id("segment", key),
            material_id=item.material_id,
            source_start=cue.source_offset,
            source_duration=cue.timeline_range.duration,
            target_start=cue.timeline_range.start,
            target_duration=cue.timeline_range.duration,
            volume=_gain_to_volume(cue.gain_db),
            extra_refs=extra_refs,
            render_index=position,
        )

    def _text_track(
        self, plan: EditPlan, *, materials: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        style = self._settings.subtitle
        segments: list[dict[str, Any]] = []

        for position, cue in enumerate(plan.subtitles):
            key = f"{plan.project_id}:subtitle:{position}"
            material_id = schema.stable_id("text", key)
            materials["texts"].append(
                schema.text_material(
                    material_id=material_id,
                    text=_flatten(cue),
                    font_size=style.font_size,
                    colour=_ass_to_hex(style.primary_colour),
                )
            )
            segments.append(
                schema.text_segment(
                    segment_id=schema.stable_id("segment", key),
                    material_id=material_id,
                    target_start=cue.range.start,
                    target_duration=cue.range.duration,
                    render_index=position,
                )
            )

        return schema.track(
            track_id=schema.stable_id("track", f"{plan.project_id}:text"),
            kind="text",
            segments=segments,
        )

    @staticmethod
    def _helper_materials(
        key: str, materials: dict[str, list[dict[str, Any]]], *, speed: float = 1.0
    ) -> list[str]:
        """The per-segment helper objects CapCut expects, and their ids.

        Speed, canvas, channel mapping and vocal separation are separate materials even at
        default values. Versions differ in how gracefully they handle their absence, and
        supplying them costs a few hundred bytes.
        """
        speed_id = schema.stable_id("speed", key)
        canvas_id = schema.stable_id("canvas", key)
        mapping_id = schema.stable_id("mapping", key)
        vocal_id = schema.stable_id("vocal", key)

        materials["speeds"].append(schema.speed_material(material_id=speed_id, speed=speed))
        materials["canvases"].append(schema.canvas_material(material_id=canvas_id))
        materials["sound_channel_mappings"].append(
            schema.sound_channel_mapping(material_id=mapping_id)
        )
        materials["vocal_separations"].append(schema.vocal_separation(material_id=vocal_id))
        return [speed_id, canvas_id, mapping_id, vocal_id]

    # -- draft_meta_info.json ------------------------------------------------ #

    def _build_meta(
        self,
        plan: EditPlan,
        *,
        name: str,
        draft_dir: Path,
        media: dict[MediaRef, _Media],
    ) -> dict[str, Any]:
        return schema.draft_meta_info(
            draft_id=schema.stable_id("draft", plan.project_id),
            name=name,
            folder=str(draft_dir).replace("/", "\\"),
            root=str(draft_dir.parent).replace("/", "\\"),
            duration=plan.timeline_duration,
            created_us=int(time.time() * schema.MICROSECONDS),
            media=[
                schema.meta_media_item(
                    material_id=item.material_id,
                    path=item.draft_path,
                    duration=item.duration,
                    width=item.export_width,
                    height=item.export_height,
                    kind="video" if item.width > 0 else "music",
                )
                for item in media.values()
            ],
        )

    # -- Writing ------------------------------------------------------------- #

    def _write(
        self,
        request: ExportRequest,
        draft_dir: Path,
        *,
        media: dict[MediaRef, _Media],
        warnings: list[str],
    ) -> tuple[tuple[Path, ...], tuple[Path, ...], dict[MediaRef, _Media]]:
        """Create the directory and install media. JSON is written afterwards."""
        try:
            draft_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            msg = f"could not create {draft_dir}: {exc}"
            raise CapCutExportError(msg) from exc

        # Stale Timelines/ from an earlier template export make CapCut open a black player
        # while the root draft_content is fine. Always clear before writing our timeline.
        _remove_stale_timelines(draft_dir)

        if request.template is not None:
            self._clone_template(request.template, draft_dir)

        copied: list[Path] = []
        updated = dict(media)
        if request.copy_media:
            (draft_dir / "media").mkdir(exist_ok=True)
            for ref, item in media.items():
                try:
                    installed, wrote = self._install_media(item, warnings=warnings)
                except OSError as exc:
                    msg = f"could not copy {item.source.name}: {exc}"
                    raise CapCutExportError(msg) from exc
                updated[ref] = installed
                if wrote:
                    copied.append(installed.destination)

        return (), tuple(copied), updated

    def _write_documents(
        self,
        draft_dir: Path,
        *,
        content: dict[str, Any],
        meta: dict[str, Any],
    ) -> tuple[Path, ...]:
        written: list[Path] = []
        for filename, document in (
            (schema.DRAFT_CONTENT_NAME, content),
            (schema.DRAFT_META_NAME, meta),
        ):
            path = draft_dir / filename
            try:
                path.write_text(
                    json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError as exc:
                msg = f"could not write {filename}: {exc}"
                raise CapCutExportError(msg) from exc
            written.append(path)
        return tuple(written)

    def _install_media(
        self, item: _Media, *, warnings: list[str]
    ) -> tuple[_Media, bool]:
        """Copy or re-encode one media file. Returns (item used, whether written)."""
        destination = item.destination
        if (
            destination.exists()
            and destination.stat().st_mtime >= item.source.stat().st_mtime
            and item.normalize_size is None
        ):
            return item, False

        if item.normalize_size is not None:
            if destination.exists() and destination.stat().st_mtime >= item.source.stat().st_mtime:
                return item, False
            if self._normalize_video(item):
                warnings.append(
                    f"{item.source.name}: re-encoded to {item.normalize_size[0]}x"
                    f"{item.normalize_size[1]} so CapCut preview can play it"
                )
                return item, True
            logger.warning(
                "Could not normalise %s for CapCut preview; copying the original",
                item.source.name,
            )
            warnings.append(
                f"{item.source.name}: could not re-encode for CapCut preview; "
                "preview may stay black — open output/final.mp4 instead if needed"
            )
            shutil.copy2(item.source, destination)
            return (
                _Media(
                    ref=item.ref,
                    source=item.source,
                    destination=item.destination,
                    material_id=item.material_id,
                    duration=item.duration,
                    width=item.width,
                    height=item.height,
                    has_audio=item.has_audio,
                    normalize_size=None,
                ),
                True,
            )

        shutil.copy2(item.source, destination)
        return item, True

    def _normalize_video(self, item: _Media) -> bool:
        """Re-encode to an even canvas CapCut's player accepts. Returns False on failure."""
        if item.normalize_size is None or self._ffmpeg is None:
            return False
        width, height = item.normalize_size
        try:
            tools = self._ffmpeg.resolve()
        except FFmpegNotFoundError:
            return False

        filt = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        )
        command = [
            str(tools.ffmpeg.path),
            "-y",
            "-i",
            str(item.source),
            "-vf",
            filt,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(item.destination),
        ]
        # Drop audio encode when the source has none — AAC on empty fails on some builds.
        if not item.has_audio:
            command = [
                str(tools.ffmpeg.path),
                "-y",
                "-i",
                str(item.source),
                "-vf",
                filt,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-an",
                "-movflags",
                "+faststart",
                str(item.destination),
            ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("CapCut media normalise failed for %s: %s", item.source.name, exc)
            return False
        if completed.returncode != 0 or not item.destination.is_file():
            logger.debug(
                "CapCut media normalise exited %s for %s: %s",
                completed.returncode,
                item.source.name,
                (completed.stderr or "")[-400:],
            )
            return False
        return True

    def _ensure_cover(self, draft_dir: Path, *, media: dict[MediaRef, _Media]) -> Path | None:
        """Write ``draft_cover.jpg`` so CapCut's Drafts list has a thumbnail."""
        cover = draft_dir / "draft_cover.jpg"
        if cover.is_file():
            return None
        if self._ffmpeg is None:
            return None
        video = next((item for item in media.values() if item.width > 0), None)
        if video is None or not video.destination.is_file():
            return None
        try:
            tools = self._ffmpeg.resolve()
        except FFmpegNotFoundError:
            return None
        command = [
            str(tools.ffmpeg.path),
            "-y",
            "-ss",
            "0.5",
            "-i",
            str(video.destination),
            "-frames:v",
            "1",
            "-q:v",
            "4",
            str(cover),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=60, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0 or not cover.is_file():
            return None
        return cover

    @staticmethod
    def _clone_template(template: Path, draft_dir: Path) -> None:
        """Copy a template draft's auxiliary files, but never its timeline.

        A template exists to carry fonts, colour settings and canvas configuration that no
        exporter can reliably invent. Its ``draft_content.json`` is the one thing we are
        replacing, so copying it would be self-defeating. ``Timelines/`` is also skipped:
        CapCut regenerates it, and a copied tree has produced black-preview drafts.
        """
        for item in template.iterdir():
            if item.name in _SKIP_TEMPLATE_NAMES:
                continue
            target = draft_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)


def safe_media_filename(name: str) -> str:
    """A CapCut-safe media basename.

    Spaces and odd punctuation in paths have shown up as black-preview drafts even when
    the timeline thumbnails look fine. Keep letters, digits, ``._-``; collapse the rest.
    """
    stem = Path(name).stem
    suffix = Path(name).suffix
    cleaned = _UNSAFE_MEDIA_CHARS.sub("_", stem).strip("._") or "media"
    return f"{cleaned}{suffix}"


def capcut_canvas(output: OutputSpec) -> tuple[int, int, str]:
    """Pixel size and CapCut ``ratio`` label for a plan's output spec.

    CapCut's project UI expects a *preset* ratio string (``16:9``, ``9:16``, ``1:1``,
    ``4:5``), not ``original``. Pixel size is snapped to a common CapCut resolution for
    that ratio so the preview player and project settings agree.
    """
    ratio = output.aspect_ratio
    if ratio not in {
        AspectRatio.LANDSCAPE,
        AspectRatio.VERTICAL,
        AspectRatio.SQUARE,
        AspectRatio.PORTRAIT_4_5,
    }:
        ratio = AspectRatio.LANDSCAPE

    width = output.width if output.width > 0 else 0
    height = output.height if output.height > 0 else 0
    hint = max(width, height, 1920)
    # Prefer an exact preset when the plan is already close (e.g. 1908x1032 → 1920x1080).
    for preset_w, preset_h in _canvas_presets(ratio):
        if width > 0 and height > 0:
            if abs(width - preset_w) <= 48 and abs(height - preset_h) <= 48:
                return preset_w, preset_h, ratio.value
        elif abs(hint - max(preset_w, preset_h)) <= 48:
            return preset_w, preset_h, ratio.value

    if width <= 0 or height <= 0 or abs((width / height) - ratio.ratio) > 0.02:
        return *_default_canvas_size(ratio, hint), ratio.value

    return _align_dimension(width), _align_dimension(height), ratio.value


def _canvas_presets(ratio: AspectRatio) -> tuple[tuple[int, int], ...]:
    presets: dict[AspectRatio, tuple[tuple[int, int], ...]] = {
        AspectRatio.LANDSCAPE: ((3840, 2160), (1920, 1080), (1280, 720)),
        AspectRatio.VERTICAL: ((1080, 1920), (720, 1280)),
        AspectRatio.SQUARE: ((1080, 1080), (720, 720)),
        AspectRatio.PORTRAIT_4_5: ((1080, 1350), (864, 1080)),
    }
    return presets.get(ratio, ((1920, 1080),))


def _default_canvas_size(ratio: AspectRatio, long_edge_hint: int = 1920) -> tuple[int, int]:
    presets = _canvas_presets(ratio)
    if not presets:
        return (1920, 1080)
    return min(presets, key=lambda size: abs(max(size) - long_edge_hint))

def _align_dimension(value: int, multiple: int = _DIM_MULTIPLE) -> int:
    if value <= 0:
        return value
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def _needs_preview_normalize(width: int, height: int) -> bool:
    return width % _DIM_MULTIPLE != 0 or height % _DIM_MULTIPLE != 0


def _remove_stale_timelines(draft_dir: Path) -> None:
    timelines = draft_dir / "Timelines"
    if timelines.is_dir():
        shutil.rmtree(timelines, ignore_errors=True)


def _gain_to_volume(gain_db: float) -> float:
    """Decibels to CapCut's linear 0-1 volume."""
    return round(min(1.0, max(0.0, float(10.0 ** (gain_db / 20.0)))), 4)


def _flatten(cue: SubtitleCue) -> str:
    """Cue text as one line.

    AIVE wraps cues to two lines for legibility; CapCut does its own wrapping from the
    caption box width, and a baked-in newline fights it.
    """
    return " ".join(cue.text.splitlines())


def _ass_to_hex(colour: str) -> str:
    """ASS ``&HAABBGGRR`` to ``#RRGGBB``.

    ASS orders its bytes alpha, blue, green, red — the reverse of what everyone expects.
    Reading it as RGB swaps red and blue, which is the single most common bug in subtitle
    styling code and is invisible on white text.
    """
    value = colour.strip().lstrip("&").lstrip("Hh")
    # The digits must be *checked*, not merely counted. Slicing a non-hex string of the
    # right length raises nothing and yields nonsense: "nonsense" became "#SEENNS", a
    # string CapCut would read as an unparseable colour rather than as an error.
    if len(value) != 8 or any(character not in "0123456789abcdefABCDEF" for character in value):
        return "#FFFFFF"
    blue, green, red = value[2:4], value[4:6], value[6:8]
    return f"#{red}{green}{blue}".upper()


def default_draft_dir(settings: AiveSettings) -> Path | None:
    """Where CapCut keeps its drafts, from config or by platform detection."""
    configured = settings.capcut.draft_dir
    return Path(configured) if configured is not None else find_draft_dir()


__all__ = [
    "CAPCUT_TRANSITIONS",
    "EXPORTER_VERSION",
    "CapCutExportError",
    "CapCutExporter",
    "capcut_canvas",
    "default_draft_dir",
    "safe_media_filename",
]
