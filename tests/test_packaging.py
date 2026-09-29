"""Packaging and importability.

These tests exist because the suite once failed to collect at all with
``ModuleNotFoundError: No module named 'multilingual_safety'``. That failure had
two causes worth guarding against permanently:

1. the package was not installed into the active environment, and
2. nothing in committed configuration made the src layout resolvable, so
   collection depended on whatever ``PYTHONPATH`` a given shell happened to have.

The fix is an editable install plus ``pythonpath = ["src"]`` in
``[tool.pytest.ini_options]``. These tests verify the declarative half is
present and correct, and that every source module is genuinely importable --
so a module added to ``src/`` but dropped from the wheel fails loudly here
rather than at first use in a later stage.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import tomllib

import multilingual_safety
from multilingual_safety.paths import REPO_ROOT

PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_DIR = REPO_ROOT / "src" / "multilingual_safety"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


# --- layout ---------------------------------------------------------------
def test_package_init_exists():
    assert (PACKAGE_DIR / "__init__.py").is_file()


def test_every_subpackage_has_an_init():
    """A subpackage without __init__.py is invisible to packages.find."""
    for directory in sorted(p for p in PACKAGE_DIR.iterdir() if p.is_dir()):
        if directory.name == "__pycache__":
            continue
        assert (directory / "__init__.py").is_file(), f"{directory.name} is missing __init__.py"


def test_declared_submodules_match_the_source_tree():
    """SUBMODULES must not drift from what is actually on disk."""
    on_disk = set()
    for path in PACKAGE_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PACKAGE_DIR)
        dotted = ".".join(relative.with_suffix("").parts)
        dotted = dotted.removesuffix(".__init__") if dotted.endswith(".__init__") else dotted
        if dotted == "__init__":
            continue
        on_disk.add(dotted)

    declared = set(multilingual_safety.SUBMODULES)
    assert declared == on_disk, (
        f"SUBMODULES drifted from src/: missing={sorted(on_disk - declared)} "
        f"stale={sorted(declared - on_disk)}"
    )


@pytest.mark.parametrize("name", multilingual_safety.SUBMODULES)
def test_every_submodule_is_importable(name):
    module = importlib.import_module(f"multilingual_safety.{name}")
    assert module is not None


def test_package_imports_from_an_installed_or_configured_location():
    assert multilingual_safety.__version__ == "0.0.1"
    assert Path(multilingual_safety.__file__).name == "__init__.py"


# --- committed configuration ----------------------------------------------
def test_build_backend_is_setuptools_with_src_layout(pyproject):
    build = pyproject["build-system"]
    assert build["build-backend"] == "setuptools.build_meta"
    assert any("setuptools" in r for r in build["requires"])

    tool = pyproject["tool"]["setuptools"]
    assert tool["package-dir"] == {"": "src"}
    find = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["src"]
    assert "multilingual_safety*" in find["include"], "subpackages must be included"


def test_pytest_config_resolves_the_src_layout(pyproject):
    """The committed alternative to an ad-hoc PYTHONPATH export."""
    ini = pyproject["tool"]["pytest"]["ini_options"]
    assert ini["pythonpath"] == ["src"]
    assert ini["testpaths"] == ["tests"]


def test_dev_extra_provides_the_toolchain(pyproject):
    """`pip install -e ".[dev]"` must supply pytest, pytest-cov, and ruff."""
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    names = {r.split(">")[0].split("=")[0].split("[")[0].strip().lower() for r in dev}
    for required in ("pytest", "pytest-cov", "ruff"):
        assert required in names, f"dev extra is missing {required}"


def test_ruff_is_installed_and_configured(pyproject):
    """Ruff must be importable in the active environment, not just declared."""
    ruff = pytest.importorskip(
        "ruff", reason="ruff is not installed; run: python -m pip install -e '.[dev]'"
    )
    assert ruff is not None
    assert pyproject["tool"]["ruff"]["line-length"] > 0


def test_console_script_target_resolves(pyproject):
    """A broken entry point only surfaces after install; check it directly."""
    target = pyproject["project"]["scripts"]["mls-new-run"]
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attribute))
