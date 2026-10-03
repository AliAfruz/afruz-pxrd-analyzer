from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
from typing import Iterable

from .gsasii_qpa_backend import detect_gsasii


@dataclass(frozen=True)
class GSASIIProfile:
    installation_root: str = ""
    python_executable: str = ""
    source_parent: str = ""
    default_instrument_file: str = ""
    timeout_seconds: int = 600

    def to_dict(self) -> dict:
        return asdict(self)


class GSASIIProfileError(ValueError):
    pass


def _clean_path(value: str | Path | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().strip('"')
    return str(Path(text)) if text else ""


def profile_from_installation_root(
    installation_root: str | Path,
    *,
    default_instrument_file: str | Path | None = None,
    timeout_seconds: int = 600,
) -> GSASIIProfile:
    root = Path(_clean_path(installation_root))
    if root.name.lower() == "python.exe":
        root = root.parent
    if root.name.lower() == "gsasii":
        # .../GSAS-II/GSASII -> .../gsas2main
        if root.parent.name.lower() == "gsas-ii":
            root = root.parent.parent
    elif root.name.lower() == "gsas-ii":
        # .../gsas2main/GSAS-II -> .../gsas2main
        root = root.parent

    python_executable = root / ("python.exe" if os.name == "nt" else "bin/python")
    if not python_executable.exists() and os.name != "nt":
        alternative = root / "python"
        if alternative.exists():
            python_executable = alternative
    source_parent = root / "GSAS-II"
    return GSASIIProfile(
        installation_root=str(root),
        python_executable=str(python_executable),
        source_parent=str(source_parent),
        default_instrument_file=_clean_path(default_instrument_file),
        timeout_seconds=max(5, int(timeout_seconds)),
    )


def normalise_profile(profile: GSASIIProfile | dict | None) -> GSASIIProfile:
    if profile is None:
        return GSASIIProfile()
    if isinstance(profile, GSASIIProfile):
        source = profile.to_dict()
    elif isinstance(profile, dict):
        source = dict(profile)
    else:
        raise GSASIIProfileError("GSAS-II profile must be a mapping or GSASIIProfile.")

    root = _clean_path(source.get("installation_root"))
    python_executable = _clean_path(source.get("python_executable"))
    source_parent = _clean_path(
        source.get("source_parent") or source.get("gsasii_parent")
    )
    instrument = _clean_path(source.get("default_instrument_file"))
    timeout = max(5, int(source.get("timeout_seconds", 600) or 600))

    if root and (not python_executable or not source_parent):
        inferred = profile_from_installation_root(
            root,
            default_instrument_file=instrument,
            timeout_seconds=timeout,
        )
        python_executable = python_executable or inferred.python_executable
        source_parent = source_parent or inferred.source_parent

    if not root and python_executable:
        executable_path = Path(python_executable)
        if executable_path.name.lower() == "python.exe":
            root = str(executable_path.parent)

    return GSASIIProfile(
        installation_root=root,
        python_executable=python_executable,
        source_parent=source_parent,
        default_instrument_file=instrument,
        timeout_seconds=timeout,
    )


def validate_profile_layout(profile: GSASIIProfile | dict) -> dict:
    profile = normalise_profile(profile)
    errors: list[str] = []
    warnings: list[str] = []

    executable = Path(profile.python_executable) if profile.python_executable else None
    parent = Path(profile.source_parent) if profile.source_parent else None

    if executable is None:
        errors.append("GSAS-II Python executable is not configured.")
    elif not executable.is_file():
        errors.append(f"Python executable does not exist: {executable}")

    package_scriptable = None if parent is None else parent / "GSASII" / "GSASIIscriptable.py"
    package_index = None if parent is None else parent / "GSASII" / "GSASIIindex.py"
    flat_scriptable = None if parent is None else parent / "GSASIIscriptable.py"
    flat_index = None if parent is None else parent / "GSASIIindex.py"

    layout = None
    if parent is None:
        errors.append("GSAS-II source parent is not configured.")
    elif package_scriptable.is_file() and package_index.is_file():
        layout = "package"
    elif flat_scriptable.is_file() and flat_index.is_file():
        layout = "flat"
    else:
        errors.append(
            "GSAS-II source parent must contain either "
            "GSASII/GSASIIscriptable.py and GSASII/GSASIIindex.py, or the "
            "equivalent flat source files."
        )

    instrument = Path(profile.default_instrument_file) if profile.default_instrument_file else None
    if instrument is not None and not instrument.is_file():
        warnings.append(
            f"Default instrument parameter file does not exist: {instrument}"
        )

    return {
        "valid_layout": not errors,
        "layout": layout,
        "errors": errors,
        "warnings": warnings,
        "profile": profile.to_dict(),
    }


def test_profile_connection(profile: GSASIIProfile | dict) -> dict:
    profile = normalise_profile(profile)
    layout = validate_profile_layout(profile)
    if not layout["valid_layout"]:
        return {
            "available": False,
            "stage": "layout",
            "error": "\n".join(layout["errors"]),
            "layout_check": layout,
            "profile": profile.to_dict(),
        }
    result = detect_gsasii(
        profile.python_executable,
        profile.source_parent or None,
    )
    result["stage"] = "import"
    result["layout_check"] = layout
    result["profile"] = profile.to_dict()
    return result


def suggested_installation_roots(home: str | Path | None = None) -> list[str]:
    home_path = Path(home) if home else Path.home()
    candidates: list[Path] = [
        home_path / "gsas2main",
        home_path / "GSASII",
        home_path / "GSAS-II",
        Path("C:/GSASII"),
        Path("C:/gsas2main"),
    ]
    seen: set[str] = set()
    results: list[str] = []
    for candidate in candidates:
        text = str(candidate)
        key = text.lower()
        if key not in seen:
            seen.add(key)
            results.append(text)
    return results


def first_existing_profile(
    roots: Iterable[str | Path] | None = None,
) -> GSASIIProfile | None:
    roots = list(roots or suggested_installation_roots())
    for root in roots:
        profile = profile_from_installation_root(root)
        if validate_profile_layout(profile)["valid_layout"]:
            return profile
    return None
