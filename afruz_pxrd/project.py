from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import io
import json
import os
import shutil
import tempfile
import zipfile

import numpy as np

from .models import Dataset
from .version import APP_VERSION


PROJECT_VERSION = 2
SUPPORTED_PROJECT_VERSIONS = {1, 2}
REQUIRED_PROJECT_MEMBERS = frozenset({"project.json", "datasets.npz"})
MAXIMUM_MANIFEST_BYTES = 50 * 1024 * 1024


class ProjectError(RuntimeError):
    """Base class for actionable project-file errors."""


class ProjectFormatError(ProjectError):
    """Raised when an `.afz` archive is missing, corrupt, or incomplete."""


class ProjectVersionError(ProjectFormatError):
    """Raised when a project schema cannot be migrated by this release."""


class ProjectSaveError(ProjectError):
    """Raised when an atomic save cannot be completed safely."""


def project_backup_path(path: str | Path) -> Path:
    """Return the stable backup path used before replacing an existing project."""
    project_path = Path(path)
    if project_path.suffix.lower() != ".afz":
        project_path = project_path.with_suffix(".afz")
    return project_path.with_suffix(project_path.suffix + ".bak")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value, key=str)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def migrate_project_manifest(manifest: Mapping) -> dict:
    """Validate and migrate a version 1 or 2 manifest to the current schema."""
    if not isinstance(manifest, Mapping):
        raise ProjectFormatError("project.json must contain a JSON object.")

    migrated = dict(manifest)
    version = migrated.get("project_version")
    if type(version) is not int:
        raise ProjectFormatError(
            "project.json is missing a valid integer 'project_version'."
        )
    if version not in SUPPORTED_PROJECT_VERSIONS:
        supported = ", ".join(str(item) for item in sorted(SUPPORTED_PROJECT_VERSIONS))
        raise ProjectVersionError(
            f"Unsupported project version {version}. This release can open versions {supported}."
        )

    if version == 1:
        # Version 1 used the same two archive members but several record fields
        # and top-level state mappings were optional.
        records = migrated.get("datasets", [])
        if isinstance(records, list):
            normalized_records = []
            for raw_record in records:
                if isinstance(raw_record, Mapping):
                    record = dict(raw_record)
                    record.setdefault("source_path", "")
                    record.setdefault("visible", True)
                    record.setdefault("metadata", {})
                    normalized_records.append(record)
                else:
                    normalized_records.append(raw_record)
            migrated["datasets"] = normalized_records
        migrated.setdefault("ui_state", {})
        migrated.setdefault("analysis_state", {})
        migrated["project_version"] = 2

    return _validate_manifest(migrated)


def _validate_manifest(manifest: dict) -> dict:
    if manifest.get("project_version") != PROJECT_VERSION:
        raise ProjectVersionError(
            f"Project migration did not reach schema version {PROJECT_VERSION}."
        )

    records = manifest.get("datasets")
    if not isinstance(records, list):
        raise ProjectFormatError("project.json field 'datasets' must be a list.")
    for state_key in ("ui_state", "analysis_state"):
        state = manifest.get(state_key, {})
        if not isinstance(state, Mapping):
            raise ProjectFormatError(
                f"project.json field '{state_key}' must be a JSON object."
            )
        manifest[state_key] = dict(state)

    seen_uids: set[str] = set()
    normalized_records: list[dict] = []
    for index, raw_record in enumerate(records, start=1):
        if not isinstance(raw_record, Mapping):
            raise ProjectFormatError(f"Dataset record {index} must be a JSON object.")
        record = dict(raw_record)
        uid = record.get("uid")
        name = record.get("name")
        if not isinstance(uid, str) or not uid.strip():
            raise ProjectFormatError(f"Dataset record {index} is missing a valid UID.")
        if uid in seen_uids:
            raise ProjectFormatError(f"Duplicate dataset UID in project: {uid}")
        if not isinstance(name, str) or not name.strip():
            raise ProjectFormatError(f"Dataset '{uid}' is missing a valid name.")
        metadata = record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ProjectFormatError(f"Dataset '{name}' metadata must be a JSON object.")
        if "has_processed" in record and not isinstance(record["has_processed"], bool):
            raise ProjectFormatError(
                f"Dataset '{name}' field 'has_processed' must be true or false."
            )
        record["metadata"] = dict(metadata)
        record.setdefault("source_path", "")
        record.setdefault("visible", True)
        normalized_records.append(record)
        seen_uids.add(uid)
    manifest["datasets"] = normalized_records
    return manifest


def _read_project_archive(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    project_path = Path(path)
    if not project_path.exists():
        raise FileNotFoundError(f"Project file does not exist: {project_path}")
    if not project_path.is_file():
        raise ProjectFormatError(f"Project path is not a file: {project_path}")
    if not zipfile.is_zipfile(project_path):
        raise ProjectFormatError(
            f"'{project_path.name}' is not a valid .afz ZIP archive. "
            "Choose an Afruz project file or restore its .afz.bak backup."
        )

    try:
        with zipfile.ZipFile(project_path, "r") as archive:
            names = archive.namelist()
            duplicate_required = [
                name for name in REQUIRED_PROJECT_MEMBERS if names.count(name) != 1
            ]
            missing = sorted(REQUIRED_PROJECT_MEMBERS.difference(names))
            if missing:
                raise ProjectFormatError(
                    "Project archive is incomplete; missing required member(s): "
                    + ", ".join(missing)
                )
            if duplicate_required:
                raise ProjectFormatError(
                    "Project archive contains duplicate required member(s): "
                    + ", ".join(sorted(duplicate_required))
                )
            corrupt_member = archive.testzip()
            if corrupt_member:
                raise ProjectFormatError(
                    f"Project archive failed its CRC check at '{corrupt_member}'."
                )
            manifest_info = archive.getinfo("project.json")
            if manifest_info.file_size > MAXIMUM_MANIFEST_BYTES:
                raise ProjectFormatError("project.json is unreasonably large and was rejected.")
            manifest_bytes = archive.read("project.json")
            array_bytes = archive.read("datasets.npz")
    except ProjectFormatError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ProjectFormatError(
            f"Could not read project archive '{project_path.name}': {exc}"
        ) from exc

    try:
        manifest_raw = json.loads(manifest_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ProjectFormatError("project.json is not valid UTF-8 text.") from exc
    except json.JSONDecodeError as exc:
        raise ProjectFormatError(
            f"project.json contains invalid JSON at line {exc.lineno}, column {exc.colno}."
        ) from exc
    manifest = migrate_project_manifest(manifest_raw)

    try:
        with np.load(io.BytesIO(array_bytes), allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    except Exception as exc:
        raise ProjectFormatError(f"datasets.npz could not be decoded safely: {exc}") from exc

    for record in manifest["datasets"]:
        uid = record["uid"]
        name = record["name"]
        required_arrays = (f"{uid}_x", f"{uid}_raw")
        missing_arrays = [key for key in required_arrays if key not in arrays]
        if missing_arrays:
            raise ProjectFormatError(
                f"Dataset '{name}' is missing array(s): {', '.join(missing_arrays)}"
            )
        x = arrays[required_arrays[0]]
        raw = arrays[required_arrays[1]]
        if x.dtype.hasobject or raw.dtype.hasobject:
            raise ProjectFormatError(f"Dataset '{name}' contains unsafe object arrays.")
        if x.ndim != 1 or raw.ndim != 1 or len(x) != len(raw):
            raise ProjectFormatError(
                f"Dataset '{name}' X/raw arrays must be one-dimensional and equal in length."
            )
        processed_key = f"{uid}_processed"
        has_processed = record.get("has_processed", processed_key in arrays)
        record["has_processed"] = bool(has_processed)
        if has_processed:
            if processed_key not in arrays:
                raise ProjectFormatError(
                    f"Dataset '{name}' declares processed data but '{processed_key}' is missing."
                )
            processed = arrays[processed_key]
            if processed.dtype.hasobject or processed.ndim != 1 or len(processed) != len(x):
                raise ProjectFormatError(
                    f"Dataset '{name}' processed array must be numeric, one-dimensional, and aligned."
                )

    return manifest, arrays


def validate_afz(path: str | Path) -> dict:
    """Fully validate an `.afz` archive without constructing GUI state."""
    manifest, arrays = _read_project_archive(path)
    return {
        "path": str(Path(path).resolve()),
        "project_version": manifest["project_version"],
        "dataset_count": len(manifest["datasets"]),
        "array_count": len(arrays),
        "application_version": manifest.get("application_version", "Not recorded"),
    }


def _build_project_payload(
    datasets: list[Dataset], ui_state: dict, analysis_state: dict
) -> tuple[dict, dict[str, np.ndarray]]:
    if not isinstance(ui_state, Mapping) or not isinstance(analysis_state, Mapping):
        raise ProjectSaveError("UI state and analysis state must be dictionaries.")

    arrays: dict[str, np.ndarray] = {}
    dataset_records = []
    seen_uids: set[str] = set()
    for index, dataset in enumerate(datasets, start=1):
        uid = str(dataset.uid or "").strip()
        if not uid:
            raise ProjectSaveError(f"Dataset {index} has no UID.")
        if uid in seen_uids:
            raise ProjectSaveError(f"Duplicate dataset UID cannot be saved: {uid}")
        x = np.asarray(dataset.x)
        raw = np.asarray(dataset.y_raw)
        if x.ndim != 1 or raw.ndim != 1 or len(x) != len(raw) or len(x) < 3:
            raise ProjectSaveError(
                f"Dataset '{dataset.name}' has invalid or misaligned X/raw arrays."
            )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(raw)):
            raise ProjectSaveError(f"Dataset '{dataset.name}' contains non-finite values.")
        arrays[f"{uid}_x"] = x
        arrays[f"{uid}_raw"] = raw
        if dataset.y_processed is not None:
            processed = np.asarray(dataset.y_processed)
            if processed.ndim != 1 or len(processed) != len(x):
                raise ProjectSaveError(
                    f"Dataset '{dataset.name}' processed data is not aligned with X."
                )
            if not np.all(np.isfinite(processed)):
                raise ProjectSaveError(
                    f"Dataset '{dataset.name}' processed data contains non-finite values."
                )
            arrays[f"{uid}_processed"] = processed
        dataset_records.append(
            {
                "uid": uid,
                "name": str(dataset.name),
                "source_path": str(dataset.source_path or ""),
                "visible": bool(dataset.visible),
                "metadata": dict(dataset.metadata),
                "has_processed": dataset.y_processed is not None,
            }
        )
        seen_uids.add(uid)

    manifest = {
        "project_version": PROJECT_VERSION,
        "application_version": APP_VERSION,
        "datasets": dataset_records,
        "ui_state": dict(ui_state),
        "analysis_state": dict(analysis_state),
    }
    return manifest, arrays


def save_afz(
    path: str | Path,
    datasets: list[Dataset],
    ui_state: dict,
    analysis_state: dict,
) -> None:
    """Atomically save a validated project and retain the previous file as `.bak`."""
    project_path = Path(path)
    if project_path.suffix.lower() != ".afz":
        project_path = project_path.with_suffix(".afz")
    temporary_path: Path | None = None
    backup_temporary_path: Path | None = None
    try:
        project_path.parent.mkdir(parents=True, exist_ok=True)
        manifest, arrays = _build_project_payload(datasets, ui_state, analysis_state)
        array_buffer = io.BytesIO()
        np.savez_compressed(array_buffer, **arrays)
        manifest_text = json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{project_path.name}.",
            suffix=".tmp",
            dir=project_path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr("project.json", manifest_text)
            archive.writestr("datasets.npz", array_buffer.getvalue())

        # Never replace a user's project with an archive that this release
        # cannot immediately validate and reopen.
        validate_afz(temporary_path)

        if project_path.exists():
            backup_path = project_backup_path(project_path)
            descriptor, backup_temporary_name = tempfile.mkstemp(
                prefix=f".{backup_path.name}.",
                suffix=".tmp",
                dir=project_path.parent,
            )
            os.close(descriptor)
            backup_temporary_path = Path(backup_temporary_name)
            shutil.copy2(project_path, backup_temporary_path)
            os.replace(backup_temporary_path, backup_path)
            backup_temporary_path = None

        # os.replace is atomic when source and destination are on the same
        # filesystem; the old project remains intact until this exact step.
        os.replace(temporary_path, project_path)
        temporary_path = None
    except ProjectSaveError:
        raise
    except Exception as exc:
        raise ProjectSaveError(
            "Could not save the project safely. The previous project was left "
            f"unchanged. Details: {exc}"
        ) from exc
    finally:
        for candidate in (temporary_path, backup_temporary_path):
            if candidate is not None:
                try:
                    candidate.unlink(missing_ok=True)
                except OSError:
                    pass


def load_afz(path: str | Path) -> tuple[list[Dataset], dict, dict]:
    """Validate, migrate, and load a version 1 or version 2 Afruz project."""
    manifest, arrays = _read_project_archive(path)
    datasets: list[Dataset] = []
    for record in manifest["datasets"]:
        uid = record["uid"]
        name = record["name"]
        try:
            dataset = Dataset(
                uid=uid,
                name=name,
                source_path=str(record.get("source_path", "")),
                visible=bool(record.get("visible", True)),
                metadata=dict(record.get("metadata", {})),
                x=arrays[f"{uid}_x"],
                y_raw=arrays[f"{uid}_raw"],
                y_processed=(
                    arrays[f"{uid}_processed"]
                    if record.get("has_processed", False)
                    else None
                ),
            )
            dataset.validate()
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectFormatError(f"Dataset '{name}' is invalid: {exc}") from exc
        datasets.append(dataset)

    return datasets, dict(manifest["ui_state"]), dict(manifest["analysis_state"])
