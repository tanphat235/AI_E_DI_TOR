"""Configuration for AIVE.

Every tunable value in the pipeline is a field on one of these settings models.
Nothing in ``app/`` may hardcode a threshold, duration, or style constant; if a
number changes the edit, it belongs here.
"""

from __future__ import annotations

from app.config.rules import RuleSettings
from app.config.settings import (
    PACKAGED_DEFAULTS,
    PROJECT_CONFIG_NAME,
    AiveSettings,
    AppSettings,
    CapCutSettings,
    LogLevel,
    MediaSettings,
    MusicSettings,
    OutputSettings,
    SpeechSettings,
    SubtitleSettings,
    VisionSettings,
    WhisperDevice,
    config_search_paths,
    load_settings,
)

__all__ = [
    "PACKAGED_DEFAULTS",
    "PROJECT_CONFIG_NAME",
    "AiveSettings",
    "AppSettings",
    "CapCutSettings",
    "LogLevel",
    "MediaSettings",
    "MusicSettings",
    "OutputSettings",
    "RuleSettings",
    "SpeechSettings",
    "SubtitleSettings",
    "VisionSettings",
    "WhisperDevice",
    "config_search_paths",
    "load_settings",
]
