"""Executable architecture rules.

The layering described in ``docs/ARCHITECTURE.md`` is load-bearing, and documentation
does not enforce itself. These tests parse the real source with :mod:`ast` and fail
when a boundary is crossed, so a violation is caught in review rather than discovered
two phases later when it has become expensive to undo.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

import pytest

APP = Path(__file__).resolve().parents[2] / "app"


def _module_files(*relative: str) -> list[Path]:
    """Python sources under the given package paths, excluding caches."""
    files: list[Path] = []
    for part in relative:
        target = APP / part
        if target.is_file():
            files.append(target)
        else:
            files.extend(path for path in target.rglob("*.py") if "__pycache__" not in path.parts)
    return files


def _imported_modules(path: Path) -> set[str]:
    """Every module name imported by ``path``, from real import statements only.

    AST-based rather than textual, so a module name appearing in a docstring or a
    comment cannot produce a false positive.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            found.add(node.module)
    return found


def _all_app_files() -> list[Path]:
    return _module_files(".")


class TestEditPlanIsDecoupled:
    """The single most important rule in the codebase."""

    def test_edit_plan_imports_only_common(self) -> None:
        """If the plan imports a transcript or a scene, it stops being a contract.

        The Edit Plan is what lets the director, the renderer and the exporters be
        replaced independently. That only holds while the plan knows nothing about
        how it was produced.
        """
        imports = _imported_modules(APP / "models" / "edit_plan.py")
        app_imports = {name for name in imports if name.startswith("app.")}
        assert app_imports == {"app.models.common"}, (
            f"edit_plan.py must import only app.models.common, found: {sorted(app_imports)}"
        )

    def test_analysis_models_do_not_import_the_plan(self) -> None:
        """Analysis produces facts; it must not know what a plan looks like."""
        for path in _module_files("models/speech.py", "models/video.py", "models/audio.py"):
            imports = _imported_modules(path)
            assert "app.models.edit_plan" not in imports, f"{path.name} imports the Edit Plan"

    def test_analysis_models_do_not_import_each_other(self) -> None:
        """Speech, video and audio analysis are independent by design."""
        siblings = {"app.models.speech", "app.models.video", "app.models.audio"}
        for path in _module_files("models/speech.py", "models/video.py", "models/audio.py"):
            own = f"app.models.{path.stem}"
            offending = (_imported_modules(path) & siblings) - {own}
            assert not offending, f"{path.name} imports sibling analysis models: {offending}"


class TestRendererAndExportersConsumeOnlyThePlan:
    @pytest.mark.parametrize("package", ["renderer", "exporters"])
    def test_no_analysis_imports(self, package: str) -> None:
        """A renderer that imports an analyser can no longer be swapped out."""
        for path in _module_files(package):
            imports = _imported_modules(path)
            forbidden = {
                name for name in imports if name.startswith(("app.analysis", "app.rule_engine"))
            }
            assert not forbidden, f"{path} imports {forbidden}"

    @pytest.mark.parametrize("package", ["renderer", "exporters"])
    def test_no_analysis_model_imports(self, package: str) -> None:
        for path in _module_files(package):
            imports = _imported_modules(path)
            forbidden = imports & {
                "app.models.speech",
                "app.models.video",
                "app.models.audio",
            }
            assert not forbidden, f"{path} imports analysis models {forbidden}"


class TestSubtitlesAreIndependentOfAnalysis:
    """Subtitles must not drag analysis into whatever consumes them.

    The renderer will need the subtitle *writers* in Phase 7 to burn captions in. If
    ``app.subtitles`` imported ``app.analysis``, the renderer would transitively depend
    on the recogniser - and a package that needs ctranslate2 to write an SRT file is a
    package that has lost its boundary.
    """

    def test_subtitles_does_not_import_analysis(self) -> None:
        for path in _module_files("subtitles"):
            forbidden = {
                name for name in _imported_modules(path) if name.startswith("app.analysis")
            }
            assert not forbidden, f"{path.relative_to(APP)} imports {forbidden}"

    def test_the_timeline_mapping_has_exactly_one_implementation(self) -> None:
        """Two copies would eventually disagree, and mis-time every subtitle.

        ``map_to_timeline`` lives in ``app.models.common`` - the layer that depends on
        nothing - because cleanup, subtitle building and the Edit Plan all need it.
        """
        definitions = [
            path.relative_to(APP)
            for path in _all_app_files()
            if any(
                isinstance(node, ast.FunctionDef) and node.name == "map_to_timeline"
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            )
        ]
        assert definitions == [Path("models/common.py")], (
            f"map_to_timeline must be defined once, in models/common.py; found {definitions}"
        )


class TestHeavyDependenciesAreImportedLazily:
    """OpenCV, PySceneDetect and PyAV must not be imported at module scope.

    Two reasons, both load-bearing. ``aive doctor`` and ``aive schema`` must work on a
    bare install with no ``[video]`` extra, and importing OpenCV costs a few hundred
    milliseconds - which every command would pay, including the ones that never decode a
    frame. Building a container constructs the analyser, so a top-level import here would
    make it unconditional.
    """

    HEAVY: ClassVar[frozenset[str]] = frozenset({"cv2", "scenedetect", "av", "numpy"})

    def test_no_top_level_heavy_imports_in_app(self) -> None:
        violations: list[str] = []
        for path in _all_app_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:  # module scope only, not nested in functions
                if isinstance(node, ast.Import):
                    names = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = {node.module.split(".")[0]}
                else:
                    continue
                offending = names & self.HEAVY
                if offending:
                    violations.append(
                        f"{path.relative_to(APP)}:{node.lineno} imports {sorted(offending)} "
                        "at module scope"
                    )
        assert not violations, "\n".join(violations)

    def test_type_checking_imports_are_allowed(self) -> None:
        """Guard the guard: an `if TYPE_CHECKING` numpy import must not be flagged.

        It is nested inside an `if`, so it is not in `tree.body` as an Import node - which
        is exactly why the check above only inspects module-scope statements.
        """
        source = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import numpy as np\n"
        tree = ast.parse(source)
        module_level = [node for node in tree.body if isinstance(node, ast.Import)]
        assert module_level == []

    def test_the_vision_package_imports_cleanly_without_the_extra(self) -> None:
        """Importing the modules must not require OpenCV to be installed."""
        import importlib

        for module in (
            "app.analysis.vision.probe",
            "app.analysis.vision.frames",
            "app.analysis.vision.metrics",
            "app.analysis.vision.scenes",
            "app.analysis.vision.classical",
            "app.analysis.vision.duplicates",
            "app.analysis.vision.analyzer",
        ):
            assert importlib.import_module(module) is not None


class TestEveryModuleImportsStandalone:
    """Each module must import as the *first* thing a process does.

    Regression cover for a real circular import: the footage analyser imported
    ``app.services.paths``, which executes ``app/services/__init__``, which imported the
    container, which imported the analyser. It worked in the test suite and in the CLI
    only because something else had always imported ``app.services`` first - so
    ``python -c "import app.analysis.vision.analyzer"`` failed while every test passed.

    Each module is imported in a **fresh subprocess** so no earlier import can mask a
    cycle, which is precisely what made the original bug invisible.
    """

    def test_each_module_imports_in_a_clean_interpreter(self) -> None:
        import subprocess
        import sys

        modules = sorted(
            ".".join(("app", *path.relative_to(APP).with_suffix("").parts)).removesuffix(
                ".__init__"
            )
            for path in _all_app_files()
        )
        # Import them one per process. Slower than a single process, and the entire point.
        failures: list[str] = []
        for module in modules:
            completed = subprocess.run(
                [sys.executable, "-c", f"import {module}"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if completed.returncode != 0:
                last = (completed.stderr or "").strip().splitlines()[-1:]
                failures.append(f"{module}: {last[0] if last else 'failed'}")
        assert not failures, "\n".join(failures)


class TestNoLanguageModelDependency:
    """The shipped package must not depend on any LLM API or the network.

    This is the project's central constraint: the intelligence lives in the AI
    director running *outside* the app. Adding one of these would break it.
    """

    FORBIDDEN: ClassVar[frozenset[str]] = frozenset(
        {
            "anthropic",
            "openai",
            "langchain",
            "google.generativeai",
            "cohere",
            "mistralai",
            "ollama",
            "transformers",
            "httpx",
            "requests",
            "aiohttp",
            "urllib.request",
            "socket",
        }
    )

    def test_no_forbidden_imports_anywhere_in_app(self) -> None:
        violations: list[str] = []
        for path in _all_app_files():
            for name in _imported_modules(path):
                root = name.split(".")[0]
                if name in self.FORBIDDEN or root in self.FORBIDDEN:
                    violations.append(f"{path.relative_to(APP)} imports {name}")
        assert not violations, "\n".join(violations)

    def test_declared_dependencies_are_free_of_llm_sdks(self) -> None:
        """Belt and braces: check the manifest, not just the code."""
        pyproject = (APP.parent / "pyproject.toml").read_text(encoding="utf-8")
        for forbidden in ("anthropic", "openai", "langchain", "transformers", "torch"):
            assert f'"{forbidden}' not in pyproject, f"{forbidden} must not be a dependency"


class TestStdoutDiscipline:
    """stdout is a machine contract; only one module may write to it."""

    ALLOWED_STDOUT_WRITERS: ClassVar[frozenset[Path]] = frozenset(
        {
            Path("cli/output.py"),
            # `--version` is machine-readable output and legitimately bypasses the
            # JSON helpers, since a bare version string is the useful form.
            Path("cli/main.py"),
        }
    )

    def test_only_sanctioned_modules_touch_stdout(self) -> None:
        violations: list[str] = []
        for path in _all_app_files():
            relative = path.relative_to(APP)
            if relative in self.ALLOWED_STDOUT_WRITERS:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    violations.append(f"{relative}:{node.lineno} calls print()")
                # Match `sys.stdout` exactly. A bare `.stdout` attribute is usually a
                # subprocess result being *read*, which is entirely legitimate.
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "stdout"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "sys"
                ):
                    violations.append(f"{relative}:{node.lineno} writes to sys.stdout")
        assert not violations, "\n".join(violations)

    def test_the_check_would_actually_catch_a_violation(self) -> None:
        """Guard the guard: a rule that cannot fail is not protecting anything."""
        tree = ast.parse("import sys\nsys.stdout.write('leak')\nprint('leak')\n")
        hits = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "stdout"
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
            )
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            )
        ]
        assert len(hits) == 2


class TestPydanticConstraintPitfall:
    """Guard against a silent pydantic behaviour that already caused a real bug.

    Constraints do not merge across an ``Annotated`` alias. Given
    ``x: Seconds = Field(le=10.0)``, the alias's bounds win and the ``le`` is
    discarded with no error, leaving a schema that advertises a limit it does not
    enforce.
    """

    ALIASES: ClassVar[frozenset[str]] = frozenset({"Seconds", "Score", "Decibels", "Attenuation"})
    CONSTRAINT_KEYWORDS: ClassVar[frozenset[str]] = frozenset(
        {"le", "ge", "lt", "gt", "multiple_of", "max_length", "min_length"}
    )

    def test_no_alias_carries_extra_numeric_constraints(self) -> None:
        violations: list[str] = []
        for path in _module_files("models", "config"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.AnnAssign) or node.value is None:
                    continue
                annotation = node.annotation
                # Unwrap `X | None` so an optional alias is still checked.
                names: set[str] = set()
                for sub in ast.walk(annotation):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)
                if not (names & self.ALIASES):
                    continue
                if not (isinstance(node.value, ast.Call) and _is_field_call(node.value)):
                    continue
                offending = {
                    kw.arg for kw in node.value.keywords if kw.arg in self.CONSTRAINT_KEYWORDS
                }
                if offending:
                    target = getattr(node.target, "id", "?")
                    violations.append(
                        f"{path.relative_to(APP)}:{node.lineno} field {target!r} mixes an "
                        f"Annotated alias with {sorted(offending)} - these are silently dropped"
                    )
        assert not violations, "\n".join(violations)


def _is_field_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "Field"
    return isinstance(func, ast.Attribute) and func.attr == "Field"


class TestPackageLayout:
    def test_reserved_directories_exist(self) -> None:
        """The folder layout is part of the contract, so it is committed empty."""
        for reserved in (
            "ui",
            "analysis/vision/impl",
            "analysis/speech/impl",
            "rule_engine/rules",
            "exporters/capcut",
            "renderer/ffmpeg",
        ):
            assert (APP / reserved).is_dir(), f"missing reserved directory app/{reserved}"

    def test_packaged_config_lives_inside_the_package(self) -> None:
        """Outside `app/` it would not be included in the built wheel."""
        assert (APP / "config" / "default.toml").is_file()

    def test_every_package_has_an_init(self) -> None:
        for directory in APP.rglob("*"):
            if not directory.is_dir() or "__pycache__" in directory.parts:
                continue
            has_python = any(path.suffix == ".py" for path in directory.iterdir())
            if has_python:
                assert (directory / "__init__.py").is_file(), f"{directory} lacks __init__.py"
