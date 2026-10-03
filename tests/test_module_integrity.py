from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
import pkgutil

import afruz_pxrd


def test_all_scientific_sources_compile(project_root: Path):
    paths = [project_root / "main.py", *sorted((project_root / "afruz_pxrd").rglob("*.py"))]
    failures = []
    for path in paths:
        try:
            compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
        except Exception as exc:  # pragma: no cover - failure detail
            failures.append(f"{path.name}: {exc}")
    # gpu_backend.py and crystal_gpu.py are optional, import-safe modules.
    # Phase 13 adds the deterministic, non-mutating Crystal Studio view planner.
    assert len(paths) == 149
    assert not failures, "Source compilation failures:\n" + "\n".join(failures)


def test_every_package_module_imports():
    failures = []
    modules = sorted(
        module.name
        for module in pkgutil.iter_modules(afruz_pxrd.__path__)
        if not module.name.startswith("__")
    )
    for name in modules:
        try:
            importlib.import_module(f"afruz_pxrd.{name}")
        except Exception as exc:  # pragma: no cover - failure detail
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert len(modules) >= 90
    assert not failures, "Package import failures:\n" + "\n".join(failures)


def test_runtime_versions_match_the_lock(project_root: Path):
    expected = {}
    for line in (project_root / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
            expected[name.lower().replace("_", "-")] = version
    checked = {
        "numpy",
        "scipy",
        "pandas",
        "pyside6",
        "pyqtgraph",
        "matplotlib",
        "openpyxl",
        "reportlab",
    }
    actual = {
        name: importlib.metadata.version(name)
        for name in checked
    }
    assert actual == {name: expected[name] for name in checked}


def test_package_version_is_preserved():
    assert afruz_pxrd.__version__ == "23.0.0"
