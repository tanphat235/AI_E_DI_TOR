"""Tests that the thing we ship is the thing we wrote.

These exist because of a bug that every other test passed straight through. `.gitignore`
carried ``**/capcut/*`` to keep users' CapCut *template* folders out of the repo. It also
matched ``app/exporters/capcut/`` — so the entire CapCut exporter was untracked by git and,
because hatchling honours `.gitignore`, absent from the built wheel. The unit suite ran
against the working tree and was perfectly green while the shipped artefact could not
import its own exporter.

The lesson generalises: a test suite verifies the source, and packaging is a separate thing
that can be wrong on its own. So this module checks the *manifest*, not the behaviour.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"


def _source_modules() -> set[str]:
    """Every Python module under ``app/``, as wheel-style relative paths."""
    return {
        path.relative_to(REPO_ROOT).as_posix()
        for path in APP_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }


# --------------------------------------------------------------------------- #
# The source tree is actually tracked
# --------------------------------------------------------------------------- #


class TestNothingSourceIsIgnored:
    """`git check-ignore` over the source tree.

    Cheap, and it catches the failure at its root rather than downstream in a wheel.
    """

    def _ignored(self, paths: list[str]) -> list[str]:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "check-ignore",
                "--no-index",
                *paths,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    @pytest.mark.skipif(
        subprocess.run(["git", "--version"], capture_output=True, check=False).returncode != 0,
        reason="git is not available",
    )
    def test_no_source_module_is_git_ignored(self) -> None:
        """The regression that started this file.

        ``**/capcut/*`` silently swallowed app/exporters/capcut/. Any future rule aimed at
        a *project* directory can make the same mistake, because project folders and source
        folders share names by design - `raw`, `music`, `output`, `capcut`.
        """
        modules = sorted(_source_modules())
        assert modules, "no source modules found; the test is looking in the wrong place"
        assert self._ignored(modules) == []

    @pytest.mark.skipif(
        subprocess.run(["git", "--version"], capture_output=True, check=False).returncode != 0,
        reason="git is not available",
    )
    def test_the_packaged_config_is_not_ignored(self) -> None:
        """``default.toml`` is the reference for every threshold. A wheel without it falls
        back to bare field defaults, which is a *different product* that still starts."""
        assert self._ignored(["app/config/default.toml"]) == []


# --------------------------------------------------------------------------- #
# The manifest declares what we actually use
# --------------------------------------------------------------------------- #


class TestDependencies:
    @pytest.fixture
    def manifest(self) -> dict:
        return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    def test_the_core_install_needs_no_media_library(self, manifest: dict) -> None:
        """`aive doctor` and `aive schema` must work on a bare install, which is what makes
        the extras genuinely optional rather than nominally so."""
        core = " ".join(manifest["project"]["dependencies"]).lower()
        for heavy in ("opencv", "faster-whisper", "pyside6", "numpy", "torch"):
            assert heavy not in core

    def test_no_language_model_dependency_anywhere(self, manifest: dict) -> None:
        """The project's central claim: the shipped package contains no LLM and no API key."""
        project = manifest["project"]
        declared = " ".join(
            [
                *project["dependencies"],
                *(dep for group in project["optional-dependencies"].values() for dep in group),
            ]
        ).lower()
        for forbidden in ("anthropic", "openai", "langchain", "transformers", "litellm"):
            assert forbidden not in declared

    def test_every_declared_extra_is_imported_somewhere(self, manifest: dict) -> None:
        """A dependency nobody imports is weight on every install and a lie in the manifest.

        Phase 10 removed four that had accumulated this way: ffmpeg-python, pysubs2,
        tomli-w and watchdog. The renderer builds argv itself, the subtitle writers are
        hand-rolled, and nothing ever watched a file.
        """
        # Distribution name to the module it provides, where they differ.
        module_names = {
            "opencv-python": "cv2",
            "faster-whisper": "faster_whisper",
            "pydantic-settings": "pydantic_settings",
            "imageio-ffmpeg": "imageio_ffmpeg",
            "ffmpeg-python": "ffmpeg",
            "tomli-w": "tomli_w",
            # Distribution names are case-insensitive, module names are not.
            "pyside6": "PySide6",
        }
        sources = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in APP_ROOT.rglob("*.py")
            if "__pycache__" not in path.parts
        )

        project = manifest["project"]
        extras = project["optional-dependencies"]
        declared = [
            *project["dependencies"],
            *(dep for name, group in extras.items() if name != "dev" for dep in group),
        ]

        unused: list[str] = []
        for requirement in declared:
            # "aive[speech]" style self-references in the `all` extra name no module.
            if requirement.startswith("aive["):
                continue
            name = (
                requirement.split(">=")[0]
                .split("==")[0]
                .split("<")[0]
                .split(";")[0]
                .strip()
                .lower()
            )
            module = module_names.get(name, name.replace("-", "_"))
            if f"import {module}" not in sources and f"from {module}" not in sources:
                unused.append(name)

        assert unused == [], f"declared but never imported: {unused}"

    def test_the_all_extra_covers_every_offline_extra(self, manifest: dict) -> None:
        """`all` exists so a user need not reason about extras; missing one defeats it.

        ``tts`` is the deliberate exception and is asserted separately below. Everything
        else has to be in here, so adding a new extra and forgetting it fails.
        """
        extras = manifest["project"]["optional-dependencies"]
        functional = {
            name for name, group in extras.items() if name not in {"all", "dev", "tts"} and group
        }
        covered = {item.removeprefix("aive[").removesuffix("]") for item in extras["all"]}
        assert functional <= covered

    def test_all_does_not_pull_in_the_network_calling_extra(self, manifest: dict) -> None:
        """AIVE's core promise is that it makes no network call, and `edge-tts` is the one
        component that would. Installing it has to be a deliberate act, so `pip install
        aive[all]` must not do it behind the user's back."""
        extras = manifest["project"]["optional-dependencies"]
        covered = {item.removeprefix("aive[").removesuffix("]") for item in extras["all"]}
        assert "tts" not in covered
        assert extras["tts"], "the tts extra should still exist, just not be in `all`"

    def test_the_version_matches_the_package(self, manifest: dict) -> None:
        """`aive --version` reads ``app.__version__``; the wheel reads pyproject. A mismatch
        means a bug report names a release that was never built."""
        from app import __version__

        assert manifest["project"]["version"] == __version__

    def test_both_entry_points_resolve(self, manifest: dict) -> None:
        """A console script naming a function that does not exist fails only on first run."""
        import importlib

        for target in manifest["project"]["scripts"].values():
            module_name, _, attribute = target.partition(":")
            module = importlib.import_module(module_name)
            assert callable(getattr(module, attribute))


# --------------------------------------------------------------------------- #
# The wheel
# --------------------------------------------------------------------------- #


@pytest.mark.integration
class TestWheelContents:
    """Builds a real wheel. Marked ``integration`` because it shells out and takes seconds.

    Run it with ``pytest -m integration``. It is the only check that sees what a user
    actually installs.
    """

    @pytest.fixture
    def wheel(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        output = tmp_path_factory.mktemp("wheel")
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(output), str(REPO_ROOT)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip(f"could not build a wheel: {result.stderr[-400:]}")
        return next(output.glob("*.whl"))

    def test_every_source_module_is_packaged(self, wheel: Path) -> None:
        packaged = set(zipfile.ZipFile(wheel).namelist())
        missing = sorted(_source_modules() - packaged)
        assert missing == [], f"absent from the wheel: {missing}"

    def test_the_reference_config_is_packaged(self, wheel: Path) -> None:
        assert "app/config/default.toml" in zipfile.ZipFile(wheel).namelist()

    def test_tests_and_docs_are_not_packaged(self, wheel: Path) -> None:
        """A wheel is the application, not the repository."""
        packaged = zipfile.ZipFile(wheel).namelist()
        assert not [name for name in packaged if name.startswith(("tests/", "docs/", "config/"))]

    def test_no_bytecode_is_packaged(self, wheel: Path) -> None:
        packaged = zipfile.ZipFile(wheel).namelist()
        assert not [name for name in packaged if "__pycache__" in name]
