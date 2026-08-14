"""The CapCut draft format.

**This file is reverse-engineered.** CapCut's ``draft_content.json`` is undocumented,
ships as compiled code rather than as a schema, and changes between releases. Everything
here was written against **CapCut 8.6.0.3667** and should be treated as a best-effort
reconstruction, not as a specification.

That is precisely why the format lives in one module. Every field name, every magic
constant and every unit conversion CapCut requires is below; the exporter itself
(:mod:`.exporter`) contains no literals from the format at all. When CapCut changes — and it
will — the repair is here and nothing else moves. This is the isolation
:class:`~app.exporters.base.ExporterRegistry` was designed around.

## What a draft is

A **directory**, not a file, named after the project and placed in CapCut's draft folder::

    com.lveditor.draft/
      My Project/
        draft_content.json     the timeline: materials, tracks, segments
        draft_meta_info.json   project metadata and the media manifest
        media/                 copied source files, when copy_media is on

## The three things easiest to get wrong

**Time is in microseconds, as integers.** Every duration and offset. Passing seconds
produces a draft that opens with every clip one millionth of its length, which looks like
an empty timeline rather than like a unit error. :func:`to_microseconds` is the only place
that conversion happens.

**Materials and segments are separate, joined by id.** A *material* is a source file with
its intrinsic properties; a *segment* is an appearance of that material on a track, with a
``source_timerange`` (where in the file) and a ``target_timerange`` (where on the timeline).
Two segments of the same file share one material. Getting this backwards produces a draft
that opens and then behaves strangely under editing.

**A segment references helper materials by id, in ``extra_material_refs``.** Speed, canvas,
channel mapping and vocal separation are each their own material object, one per segment,
even when they carry default values. CapCut tolerates their absence unevenly across
versions; supplying them is the safe choice.
"""

from __future__ import annotations

import uuid
from typing import Any

# Recorded so a draft carries the assumption it was written under, and so a future
# exporter can branch rather than guess.
TARGET_CAPCUT_VERSION = "8.6.0.3667"
"""The CapCut release this reconstruction was written against."""

DRAFT_CONTENT_NAME = "draft_content.json"
DRAFT_META_NAME = "draft_meta_info.json"

_APP_VERSION = "8.6.0"
_NEW_VERSION = "110.0.0"
_DRAFT_VERSION = 360000
"""CapCut's internal schema number. A draft declaring a *newer* value than the installed
app is refused outright, which is why this is pinned low rather than to something current."""

MICROSECONDS = 1_000_000


def to_microseconds(seconds: float) -> int:
    """Seconds to CapCut's integer microseconds.

    Rounded, not truncated: truncation accumulates across forty clips into a visible drift
    against the narration, and the error is always in the same direction.
    """
    return round(seconds * MICROSECONDS)


def new_id() -> str:
    """A fresh CapCut identifier: an uppercase, hyphenated UUID."""
    return str(uuid.uuid4()).upper()


def stable_id(namespace: str, key: str) -> str:
    """A deterministic identifier derived from ``namespace`` and ``key``.

    Re-exporting an unchanged plan should produce an unchanged draft. Random UUIDs would
    make every export a full rewrite, which defeats diffing and makes the exporter
    untestable — a test could only assert that *some* id was present.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"aive:{namespace}:{key}")).upper()


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #


def timerange(start: float, duration: float) -> dict[str, int]:
    """A CapCut time range. Note it is start+**duration**, not start+end."""
    return {"start": to_microseconds(start), "duration": to_microseconds(duration)}


# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #


def video_material(
    *,
    material_id: str,
    path: str,
    name: str,
    duration: float,
    width: int,
    height: int,
    has_audio: bool,
) -> dict[str, Any]:
    """A source video file.

    ``duration`` is the *file's* length, not the clip's. CapCut uses it to bound trimming in
    the UI, so understating it silently prevents the user from extending a clip by hand —
    which is the main reason they opened the draft.
    """
    return {
        "id": material_id,
        "type": "video",
        "path": path,
        "material_name": name,
        "duration": to_microseconds(duration),
        "width": width,
        "height": height,
        "has_audio": has_audio,
        "category_id": "",
        "category_name": "local",
        "check_flag": 63487,
        "crop": _default_crop(),
        "crop_ratio": "free",
        "crop_scale": 1.0,
        "extra_type_option": 0,
        "is_ai_generate_content": False,
        "is_unified_beauty_mode": False,
        "local_material_id": material_id,
        "media_path": "",
        "reverse_intensifies_path": "",
        "reverse_path": "",
        "source_platform": 0,
        "stable": None,
        "team_id": "",
        "video_algorithm": _default_video_algorithm(),
    }


def audio_material(*, material_id: str, path: str, name: str, duration: float) -> dict[str, Any]:
    """A source audio file: the narration, or a music track."""
    return {
        "id": material_id,
        "type": "extract_music",
        "path": path,
        "name": name,
        "duration": to_microseconds(duration),
        "app_id": 0,
        "category_id": "",
        "category_name": "local",
        "check_flag": 1,
        "effect_id": "",
        "intensifies_path": "",
        "local_material_id": material_id,
        "music_id": material_id,
        "query": "",
        "request_id": "",
        "resource_id": "",
        "search_id": "",
        "source_platform": 0,
        "team_id": "",
        "text_id": "",
        "tone_category_id": "",
        "tone_category_name": "",
        "tone_effect_id": "",
        "tone_effect_name": "",
        "tone_speaker": "",
        "tone_type": "",
        "wave_points": [],
    }


def text_material(*, material_id: str, text: str, font_size: int, colour: str) -> dict[str, Any]:
    """A subtitle.

    CapCut stores styled text as a small JSON *string* under ``content`` — a document inside
    a document. The nesting is CapCut's, not ours.
    """
    return {
        "id": material_id,
        "type": "text",
        "content": _rich_text(text, font_size=font_size, colour=colour),
        "alignment": 1,
        "background_alpha": 1.0,
        "background_color": "",
        "background_style": 0,
        "bold_width": 0.0,
        "border_alpha": 1.0,
        "border_color": "",
        "border_width": 0.08,
        "font_category_id": "",
        "font_category_name": "",
        "font_id": "",
        "font_name": "",
        "font_path": "",
        "font_size": float(font_size),
        "font_source_platform": 0,
        "font_title": "none",
        "force_apply_line_max_width": False,
        "has_shadow": False,
        "italic_degree": 0,
        "letter_spacing": 0.0,
        "line_feed": 1,
        "line_spacing": 0.02,
        "shadow_alpha": 0.8,
        "shadow_angle": -45.0,
        "shadow_color": "",
        "shadow_distance": 5.0,
        "shadow_smoothing": 0.45,
        "shape_clip_x": False,
        "shape_clip_y": False,
        "style_name": "",
        "sub_type": 0,
        "text_alpha": 1.0,
        "text_curve": None,
        "text_preset_resource_id": "",
        "text_size": font_size,
        "typesetting": 0,
        "underline": False,
        "use_effect_default_color": True,
    }


def transition_material(*, material_id: str, name: str, duration: float) -> dict[str, Any]:
    """A transition between two segments.

    ``is_overlap`` is the field that matters: it tells CapCut the transition *consumes* time
    from the outgoing clip rather than inserting new time. False would lengthen the whole
    timeline by the sum of every transition, silently desynchronising the narration.
    """
    return {
        "id": material_id,
        "type": "transition",
        "name": name,
        "duration": to_microseconds(duration),
        "category_id": "",
        "category_name": "",
        "effect_id": "",
        "is_overlap": True,
        "path": "",
        "platform": "all",
        "resource_id": "",
        "source_platform": 0,
    }


def speed_material(*, material_id: str, speed: float) -> dict[str, Any]:
    return {
        "id": material_id,
        "type": "speed",
        "speed": speed,
        "curve_speed": None,
        "mode": 0,
    }


def canvas_material(*, material_id: str) -> dict[str, Any]:
    return {
        "id": material_id,
        "type": "canvas_color",
        "album_image": "",
        "blur": 0.0,
        "color": "",
        "image": "",
        "image_id": "",
        "image_name": "",
        "source_platform": 0,
    }


def sound_channel_mapping(*, material_id: str) -> dict[str, Any]:
    return {"id": material_id, "type": "none", "audio_channel_mapping": 0, "is_config_open": False}


def vocal_separation(*, material_id: str) -> dict[str, Any]:
    return {
        "id": material_id,
        "type": "vocal_separation",
        "choice": 0,
        "production_path": "",
        "removed_sounds": [],
        "time_range": None,
    }


def audio_fade(*, material_id: str, fade_in: float, fade_out: float) -> dict[str, Any]:
    return {
        "id": material_id,
        "type": "audio_fade",
        "fade_in_duration": to_microseconds(fade_in),
        "fade_out_duration": to_microseconds(fade_out),
        "fade_type": 0,
    }


# --------------------------------------------------------------------------- #
# Segments
# --------------------------------------------------------------------------- #


def video_segment(
    *,
    segment_id: str,
    material_id: str,
    source_start: float,
    source_duration: float,
    target_start: float,
    target_duration: float,
    speed: float,
    volume: float,
    extra_refs: list[str],
    render_index: int,
) -> dict[str, Any]:
    """One appearance of a video material on a track."""
    return {
        "id": segment_id,
        "material_id": material_id,
        "source_timerange": timerange(source_start, source_duration),
        "target_timerange": timerange(target_start, target_duration),
        "extra_material_refs": extra_refs,
        "speed": speed,
        "volume": volume,
        "visible": True,
        "clip": _default_clip(),
        "enable_adjust": True,
        "enable_color_correct_adjust": False,
        "enable_color_curves": True,
        "enable_color_match_adjust": False,
        "enable_color_wheels": True,
        "enable_lut": True,
        "enable_smart_color_adjust": False,
        "group_id": "",
        "hdr_settings": {"intensity": 1.0, "mode": 1, "nits": 1000},
        "intensifies_audio": False,
        "is_placeholder": False,
        "is_tone_modify": False,
        "keyframe_refs": [],
        "last_nonzero_volume": volume,
        "render_index": render_index,
        "responsive_layout": _default_responsive_layout(),
        "reverse": False,
        "template_id": "",
        "template_scene": "default",
        "track_attribute": 0,
        "track_render_index": 0,
        "uniform_scale": {"on": True, "value": 1.0},
    }


def audio_segment(
    *,
    segment_id: str,
    material_id: str,
    source_start: float,
    source_duration: float,
    target_start: float,
    target_duration: float,
    volume: float,
    extra_refs: list[str],
    render_index: int,
) -> dict[str, Any]:
    """One appearance of an audio material on a track."""
    return {
        "id": segment_id,
        "material_id": material_id,
        "source_timerange": timerange(source_start, source_duration),
        "target_timerange": timerange(target_start, target_duration),
        "extra_material_refs": extra_refs,
        "speed": 1.0,
        "volume": volume,
        "visible": True,
        "clip": None,
        "enable_adjust": False,
        "enable_color_correct_adjust": False,
        "enable_color_curves": True,
        "enable_color_match_adjust": False,
        "enable_color_wheels": True,
        "enable_lut": False,
        "enable_smart_color_adjust": False,
        "group_id": "",
        "intensifies_audio": False,
        "is_placeholder": False,
        "is_tone_modify": False,
        "keyframe_refs": [],
        "last_nonzero_volume": volume,
        "render_index": render_index,
        "reverse": False,
        "template_id": "",
        "template_scene": "default",
        "track_attribute": 0,
        "track_render_index": 0,
        "uniform_scale": None,
    }


def text_segment(
    *,
    segment_id: str,
    material_id: str,
    target_start: float,
    target_duration: float,
    render_index: int,
) -> dict[str, Any]:
    """A subtitle's appearance on the text track.

    No ``source_timerange``: text has no source to seek into.
    """
    return {
        "id": segment_id,
        "material_id": material_id,
        "source_timerange": None,
        "target_timerange": timerange(target_start, target_duration),
        "extra_material_refs": [],
        "speed": 1.0,
        "volume": 1.0,
        "visible": True,
        "clip": _default_clip(offset_y=-0.72),
        "enable_adjust": False,
        "enable_color_curves": True,
        "enable_color_wheels": True,
        "enable_lut": False,
        "group_id": "",
        "intensifies_audio": False,
        "is_placeholder": False,
        "is_tone_modify": False,
        "keyframe_refs": [],
        "last_nonzero_volume": 1.0,
        "render_index": render_index,
        "reverse": False,
        "template_id": "",
        "template_scene": "default",
        "track_attribute": 0,
        "track_render_index": 0,
        "uniform_scale": {"on": True, "value": 1.0},
    }


def track(*, track_id: str, kind: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
    """A timeline track. ``kind`` is ``video``, ``audio`` or ``text``."""
    return {
        "id": track_id,
        "type": kind,
        "segments": segments,
        "attribute": 0,
        "flag": 0,
        "is_default_name": True,
        "name": "",
    }


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


def draft_content(
    *,
    draft_id: str,
    name: str,
    width: int,
    height: int,
    fps: float,
    duration: float,
    materials: dict[str, list[dict[str, Any]]],
    tracks: list[dict[str, Any]],
    ratio: str = "16:9",
) -> dict[str, Any]:
    """The timeline document.

    Every material bucket is present even when empty. CapCut reads several of them without
    checking, so an absent key is a crash on open rather than a graceful default.

    ``ratio`` must be a CapCut preset label (``16:9``, ``9:16``, ``1:1``, ``4:5``). The value
    ``original`` is deliberately avoided: CapCut's project UI treats preset ratios as the
    supported set, and ``original`` has produced drafts that open with a broken preview.
    """
    buckets: dict[str, list[dict[str, Any]]] = {
        "audio_balances": [],
        "audio_effects": [],
        "audio_fades": [],
        "audio_track_indexes": [],
        "audios": [],
        "beats": [],
        "canvases": [],
        "chromas": [],
        "color_curves": [],
        "digital_humans": [],
        "drafts": [],
        "effects": [],
        "flowers": [],
        "green_screens": [],
        "handwrites": [],
        "hsl": [],
        "images": [],
        "log_color_wheels": [],
        "loudnesses": [],
        "manual_deformations": [],
        "masks": [],
        "material_animations": [],
        "material_colors": [],
        "multi_language_refs": [],
        "placeholders": [],
        "plugin_effects": [],
        "primary_color_wheels": [],
        "realtime_denoises": [],
        "shapes": [],
        "smart_crops": [],
        "smart_relights": [],
        "sound_channel_mappings": [],
        "speeds": [],
        "stickers": [],
        "tail_leaders": [],
        "text_templates": [],
        "texts": [],
        "time_marks": [],
        "transitions": [],
        "video_effects": [],
        "video_trackings": [],
        "videos": [],
        "vocal_beautifys": [],
        "vocal_separations": [],
    }
    buckets.update(materials)

    return {
        "id": draft_id,
        "name": name,
        "duration": to_microseconds(duration),
        "fps": fps,
        "canvas_config": {"width": width, "height": height, "ratio": ratio},
        "materials": buckets,
        "tracks": tracks,
        "color_space": 0,
        "config": _default_config(),
        "cover": None,
        "create_time": 0,
        "extra_info": None,
        "free_render_index_mode_on": False,
        "group_container": None,
        "keyframe_graph_list": [],
        "keyframes": _empty_keyframes(),
        "last_modified_platform": _platform(),
        "mutable_config": None,
        "new_version": _NEW_VERSION,
        "platform": _platform(),
        "relationships": [],
        "render_index_track_mode_on": True,
        "retouch_cover": None,
        "source": "default",
        "static_cover_image_path": "",
        "time_marks": None,
        "update_time": 0,
        "version": _DRAFT_VERSION,
    }


def draft_meta_info(
    *,
    draft_id: str,
    name: str,
    folder: str,
    root: str,
    duration: float,
    created_us: int,
    media: list[dict[str, Any]],
) -> dict[str, Any]:
    """The project-list document.

    This is what CapCut's Drafts screen reads. A draft with a valid ``draft_content.json``
    and no meta file exists on disk but is invisible in the UI, which reads to a user as a
    failed export.
    """
    return {
        "draft_id": draft_id,
        "draft_name": name,
        "draft_fold_path": folder,
        "draft_root_path": root,
        "draft_cover": "draft_cover.jpg",
        "tm_duration": to_microseconds(duration),
        "tm_draft_create": created_us,
        "tm_draft_modified": created_us,
        "tm_draft_removed": 0,
        "draft_materials": [
            {"type": 0, "value": media},
            {"type": 1, "value": []},
            {"type": 2, "value": []},
            {"type": 3, "value": []},
            {"type": 6, "value": []},
            {"type": 7, "value": []},
            {"type": 8, "value": []},
        ],
        "cloud_package_completed_time": "",
        "draft_cloud_capcut_purchase_info": "",
        "draft_cloud_last_action_download": False,
        "draft_cloud_materials": [],
        "draft_cloud_purchase_info": "",
        "draft_cloud_template_id": "",
        "draft_cloud_tutorial_info": "",
        "draft_cloud_videocut_purchase_info": "",
        "draft_deeplink_path": "",
        "draft_enterprise_info": {
            "draft_enterprise_extra": "",
            "draft_enterprise_id": "",
            "draft_enterprise_name": "",
            "enterprise_material": [],
        },
        "draft_is_ai_packaging_used": False,
        "draft_is_ai_shorts": False,
        "draft_is_ai_translate": False,
        "draft_is_article_video_draft": False,
        "draft_is_from_deeplink": "false",
        "draft_is_invisible": False,
        "draft_new_version": "",
        "draft_removable_storage_device": "",
        "draft_segment_extra_info": [],
        "draft_timeline_materials_size_": 0,
        "draft_type": "",
        "tm_draft_cloud_completed": "",
        "tm_draft_cloud_modified": 0,
    }


def meta_media_item(
    *,
    material_id: str,
    path: str,
    duration: float,
    width: int,
    height: int,
    kind: str,
) -> dict[str, Any]:
    """One entry in the draft's media manifest.

    ``metetype`` is CapCut's spelling, not a typo of ours. Renaming it to ``metatype``
    produces a draft whose media panel is empty.
    """
    return {
        "id": material_id,
        "file_Path": path,
        "metetype": kind,
        "duration": to_microseconds(duration),
        "width": width,
        "height": height,
        "create_time": 0,
        "import_time": 0,
        "import_time_ms": 0,
        "item_source": 1,
        "md5": "",
        "roughcut_time_range": {"duration": to_microseconds(duration), "start": 0},
        "sub_time_range": {"duration": -1, "start": -1},
        "type": 0,
    }


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


def _default_crop() -> dict[str, float]:
    return {
        "lower_left_x": 0.0,
        "lower_left_y": 1.0,
        "lower_right_x": 1.0,
        "lower_right_y": 1.0,
        "upper_left_x": 0.0,
        "upper_left_y": 0.0,
        "upper_right_x": 1.0,
        "upper_right_y": 0.0,
    }


def _default_clip(*, offset_y: float = 0.0) -> dict[str, Any]:
    return {
        "alpha": 1.0,
        "flip": {"horizontal": False, "vertical": False},
        "rotation": 0.0,
        "scale": {"x": 1.0, "y": 1.0},
        "transform": {"x": 0.0, "y": offset_y},
    }


def _default_responsive_layout() -> dict[str, Any]:
    return {
        "enable": False,
        "horizontal_pos_layout": 0,
        "size_layout": 0,
        "target_follow": "",
        "vertical_pos_layout": 0,
    }


def _default_video_algorithm() -> dict[str, Any]:
    return {
        "algorithms": [],
        "complement_frame_config": None,
        "deflicker": None,
        "gameplay_configs": [],
        "motion_blur_config": None,
        "noise_reduction": None,
        "path": "",
        "quality_enhance": None,
        "time_range": None,
    }


def _default_config() -> dict[str, Any]:
    return {
        "adjust_max_index": 1,
        "attachment_info": [],
        "combination_max_index": 1,
        "export_range": None,
        "extract_audio_last_index": 1,
        "lyrics_recognition_id": "",
        "lyrics_sync": True,
        "lyrics_taskinfo": [],
        "maintrack_adsorb": True,
        "material_save_mode": 0,
        "multi_language_current": "none",
        "multi_language_list": [],
        "multi_language_main": "none",
        "multi_language_mode": "none",
        "original_sound_last_index": 1,
        "record_audio_last_index": 1,
        "sticker_max_index": 1,
        "subtitle_keywords_config": None,
        "subtitle_recognition_id": "",
        "subtitle_sync": True,
        "subtitle_taskinfo": [],
        "system_font_list": [],
        "video_mute": False,
        "zoom_info_params": None,
    }


def _empty_keyframes() -> dict[str, list[Any]]:
    return {
        "adjusts": [],
        "audios": [],
        "effects": [],
        "filters": [],
        "handwrites": [],
        "stickers": [],
        "texts": [],
        "videos": [],
    }


def _platform() -> dict[str, Any]:
    return {
        "app_id": 359289,
        "app_source": "cc",
        "app_version": _APP_VERSION,
        "device_id": "",
        "hard_disk_id": "",
        "mac_address": "",
        "os": "windows",
        "os_version": "",
    }


def _rich_text(text: str, *, font_size: int, colour: str) -> str:
    """CapCut's nested styled-text document, as a JSON string.

    Colours are floats 0-1 in an ``[r, g, b]`` list, not hex and not 0-255.
    """
    import json

    red, green, blue = _hex_to_floats(colour)
    return json.dumps(
        {
            "text": text,
            "styles": [
                {
                    "fill": {"content": {"solid": {"color": [red, green, blue]}}},
                    "font": {"id": "", "path": ""},
                    "range": [0, len(text)],
                    "size": font_size,
                    "useLetterColor": True,
                }
            ],
        },
        ensure_ascii=False,
    )


def _hex_to_floats(colour: str) -> tuple[float, float, float]:
    """``#RRGGBB`` to three floats. Falls back to white on anything unrecognised."""
    value = colour.lstrip("#")
    if len(value) != 6:
        return (1.0, 1.0, 1.0)
    try:
        return tuple(int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return (1.0, 1.0, 1.0)


__all__ = [
    "DRAFT_CONTENT_NAME",
    "DRAFT_META_NAME",
    "MICROSECONDS",
    "TARGET_CAPCUT_VERSION",
    "audio_fade",
    "audio_material",
    "audio_segment",
    "canvas_material",
    "draft_content",
    "draft_meta_info",
    "meta_media_item",
    "new_id",
    "sound_channel_mapping",
    "speed_material",
    "stable_id",
    "text_material",
    "text_segment",
    "timerange",
    "to_microseconds",
    "track",
    "transition_material",
    "video_material",
    "video_segment",
    "vocal_separation",
]
