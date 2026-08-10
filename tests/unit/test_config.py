"""Tests for layered configuration.

The properties under test are precedence and *partial* merge. Partial merge is the
one people get wrong: a project file that sets one rule must not reset the other
twenty to their defaults, or every override becomes a full copy of the config that
silently stops tracking upgrades.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config.rules import RuleSettings
from app.config.settings import (
    PACKAGED_DEFAULTS,
    AiveSettings,
    ConfigError,
    config_search_paths,
    load_settings,
)
from app.models.common import AspectRatio, TransitionKind


class TestPackagedDefaults:
    def test_the_packaged_file_exists_and_ships_inside_the_package(self) -> None:
        """It must be inside `app/` or it will not be in the built wheel."""
        assert PACKAGED_DEFAULTS.is_file()
        assert PACKAGED_DEFAULTS.parent.name == "config"
        assert PACKAGED_DEFAULTS.parent.parent.name == "app"

    def test_it_is_applied(self, isolated_cwd: Path) -> None:
        assert PACKAGED_DEFAULTS in config_search_paths()

    def test_every_field_default_agrees_with_the_packaged_file(self, isolated_cwd: Path) -> None:
        """Drift between the two is how documentation starts lying.

        The TOML is meant to *document* the defaults, so any value it sets should
        match what the model would have produced anyway.
        """
        from_file = load_settings()
        from_code = AiveSettings()
        assert from_file.rules == from_code.rules
        assert from_file.output == from_code.output
        assert from_file.subtitle == from_code.subtitle
        assert from_file.speech == from_code.speech
        assert from_file.vision == from_code.vision
        assert from_file.music == from_code.music


class TestPrecedence:
    def test_a_project_file_beats_the_packaged_defaults(self, tmp_path: Path) -> None:
        (tmp_path / "aive.toml").write_text("[rules]\nmin_clip_duration = 2.5\n", encoding="utf-8")
        assert load_settings(tmp_path).rules.min_clip_duration == 2.5

    def test_env_beats_a_project_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "aive.toml").write_text("[rules]\nmin_clip_duration = 2.5\n", encoding="utf-8")
        monkeypatch.setenv("AIVE_RULES__MIN_CLIP_DURATION", "3.5")
        assert load_settings(tmp_path).rules.min_clip_duration == 3.5

    def test_explicit_overrides_beat_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIVE_RULES__MIN_CLIP_DURATION", "3.5")
        settings = load_settings(tmp_path, rules={"min_clip_duration": 4.5})
        assert settings.rules.min_clip_duration == 4.5

    def test_nested_env_delimiter(
        self, monkeypatch: pytest.MonkeyPatch, isolated_cwd: Path
    ) -> None:
        monkeypatch.setenv("AIVE_OUTPUT__FPS", "24.0")
        assert load_settings().output.fps == 24.0


class TestPartialMerge:
    def test_a_partial_section_leaves_its_siblings_alone(self, tmp_path: Path) -> None:
        (tmp_path / "aive.toml").write_text("[rules]\nmin_clip_duration = 2.5\n", encoding="utf-8")
        rules = load_settings(tmp_path).rules
        assert rules.min_clip_duration == 2.5
        assert rules.max_clip_duration == 8.0
        assert rules.duplicate_similarity == 0.90
        assert rules.filler_words[0] == "um"
        assert len(rules.filler_words) == 9
        assert rules.remove_ambiguous_fillers is False

    def test_touching_one_section_leaves_the_others_alone(self, tmp_path: Path) -> None:
        (tmp_path / "aive.toml").write_text("[output]\nfps = 24.0\n", encoding="utf-8")
        settings = load_settings(tmp_path)
        assert settings.output.fps == 24.0
        assert settings.output.video_crf == 18
        assert settings.rules.min_clip_duration == 1.2

    def test_an_override_of_a_partially_overridden_section_via_cli(self, tmp_path: Path) -> None:
        (tmp_path / "aive.toml").write_text("[rules]\nmax_clip_duration = 12.0\n", encoding="utf-8")
        rules = load_settings(tmp_path, rules={"min_clip_duration": 3.0}).rules
        assert rules.min_clip_duration == 3.0
        assert rules.max_clip_duration == 12.0


class TestTypoDetection:
    """A misspelled key must fail loudly, and must be attributed to the user's file.

    It used to raise a bare ``ValidationError``, which reached the CLI's catch-all handler
    and was reported as ``internal.unhandled`` with the hint "This is a bug in AIVE" — so a
    typo in the user's own ``aive.toml`` sent them looking for a bug in this codebase. The
    detection was right; the attribution was not.
    """

    def test_an_unknown_field_is_rejected(self, tmp_path: Path) -> None:
        """Silently ignoring a misspelled key means the user's setting never applies."""
        (tmp_path / "aive.toml").write_text("[rules]\nmin_clip_duratoin = 2.5\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_settings(tmp_path)

    def test_an_unknown_section_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "aive.toml").write_text("[rulez]\nfoo = 1\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_settings(tmp_path)

    def test_the_message_names_the_setting_and_the_way_out(self, tmp_path: Path) -> None:
        """Pydantic's own rendering is five lines led by a URL. What a user needs is the
        dotted name of the setting and how to find the real one."""
        (tmp_path / "aive.toml").write_text("[rules]\nmin_clip_duratoin = 2.5\n", encoding="utf-8")
        with pytest.raises(ConfigError) as caught:
            load_settings(tmp_path)

        message = str(caught.value)
        assert "min_clip_duratoin" in message
        assert "aive config show" in message
        assert "no such setting" in message

    def test_the_message_names_the_layers_that_were_read(self, tmp_path: Path) -> None:
        """With six layers, "invalid configuration" alone does not say which file to open."""
        (tmp_path / "aive.toml").write_text("[rules]\nnope = 1\n", encoding="utf-8")
        with pytest.raises(ConfigError, match=re.escape("aive.toml")):
            load_settings(tmp_path)

    def test_malformed_toml_is_also_the_users_file(self, tmp_path: Path) -> None:
        """A duplicated section or a missing bracket is a syntax error rather than a wrong
        value, and it reached the same "report a bug" handler until Phase 10."""
        (tmp_path / "aive.toml").write_text("[rules]\nfoo = 1\n[rules]\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="could not parse"):
            load_settings(tmp_path)

    def test_an_out_of_range_value_is_reported_with_its_bound(self, tmp_path: Path) -> None:
        """Not every config error is a typo; a real key with an impossible value must also
        be attributed to the file rather than to AIVE."""
        (tmp_path / "aive.toml").write_text("[output]\nfps = -5\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="fps"):
            load_settings(tmp_path)

    def test_a_valid_file_raises_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "aive.toml").write_text("[rules]\nmin_clip_duration = 2.5\n", encoding="utf-8")
        assert load_settings(tmp_path).rules.min_clip_duration == 2.5


class TestBlankToNone:
    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_a_blank_string_means_unset(self, tmp_path: Path, blank: str) -> None:
        (tmp_path / "aive.toml").write_text(f'[media]\nffmpeg_path = "{blank}"\n', encoding="utf-8")
        assert load_settings(tmp_path).media.ffmpeg_path is None

    def test_blank_language_means_autodetect(self, tmp_path: Path) -> None:
        (tmp_path / "aive.toml").write_text('[speech]\nlanguage = ""\n', encoding="utf-8")
        assert load_settings(tmp_path).speech.language is None

    def test_a_real_path_survives(self, tmp_path: Path) -> None:
        (tmp_path / "aive.toml").write_text(
            '[media]\nffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"\n', encoding="utf-8"
        )
        resolved = load_settings(tmp_path).media.ffmpeg_path
        assert resolved is not None
        assert resolved.name == "ffmpeg.exe"


class TestRuleSettings:
    def test_incoherent_clip_bounds_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="min_clip_duration"):
            RuleSettings(min_clip_duration=9.0, max_clip_duration=4.0)

    def test_incoherent_brightness_bounds_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="brightness_min"):
            RuleSettings(brightness_min=0.9, brightness_max=0.2)

    def test_transition_fits_a_long_clip(self) -> None:
        rules = RuleSettings(transition_duration=0.4, max_transition_ratio=0.25)
        assert rules.transition_fits(4.0)  # 4.0 * 0.25 = 1.0, so 0.4 fits

    def test_transition_does_not_fit_a_short_clip(self) -> None:
        rules = RuleSettings(transition_duration=0.4, max_transition_ratio=0.25)
        assert not rules.transition_fits(1.2)  # 1.2 * 0.25 = 0.3, so 0.4 does not fit

    def test_clamped_transition_duration(self) -> None:
        rules = RuleSettings(transition_duration=0.4, max_transition_ratio=0.25)
        assert rules.clamped_transition_duration(1.2) == pytest.approx(0.3)
        assert rules.clamped_transition_duration(8.0) == pytest.approx(0.4)

    def test_ratio_cannot_exceed_a_half(self) -> None:
        """Beyond 0.5 the clips would overlap more than they play."""
        with pytest.raises(ValidationError):
            RuleSettings(max_transition_ratio=0.8)

    def test_default_transition_is_an_enum_not_a_string(self, isolated_cwd: Path) -> None:
        assert load_settings().default_transition is TransitionKind.DISSOLVE


class TestOutputSettings:
    def test_to_output_spec_keeps_only_editorial_fields(self, isolated_cwd: Path) -> None:
        """Encoder settings must stay out of the plan so one plan renders many ways."""
        spec = load_settings().output.to_output_spec()
        assert spec.aspect_ratio is AspectRatio.LANDSCAPE
        assert spec.fps == 60.0
        assert not hasattr(spec, "video_crf")

    def test_a_vertical_project_config(self, tmp_path: Path) -> None:
        (tmp_path / "aive.toml").write_text(
            '[output]\naspect_ratio = "9:16"\nwidth = 1080\nheight = 1920\n', encoding="utf-8"
        )
        spec = load_settings(tmp_path).output.to_output_spec()
        assert spec.aspect_ratio is AspectRatio.VERTICAL
        assert spec.actual_ratio == pytest.approx(9 / 16)


class TestImmutability:
    def test_settings_are_frozen(self, isolated_cwd: Path) -> None:
        settings = load_settings()
        with pytest.raises(ValidationError):
            settings.rules = RuleSettings()  # type: ignore[misc]
