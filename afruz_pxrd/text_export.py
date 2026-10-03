from __future__ import annotations

"""Reliable, human-readable TXT exports for Afruz PXRD Analyzer.

All tabular files use UTF-8, LF line endings and one TAB between columns.
Writes are atomic: a temporary file is completed and fsynced before it replaces
its destination. Existing interactive exports are preserved as timestamped
backups rather than being silently overwritten.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

TEXT_EXPORT_SCHEMA_VERSION = 1
NA_TEXT = "NA"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_filename(name: str, fallback: str = "export") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._")
    return text or fallback


def ensure_txt_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".txt")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(
            f"{path.stem}.backup_{stamp}_{counter}{path.suffix}"
        )
        counter += 1
    shutil.copy2(path, candidate)
    return candidate


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    backup: bool = True,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if backup:
        _backup_existing(target)
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(normalized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _plain_value(value: Any) -> Any:
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain_value(item) for item in value]
    if isinstance(value, bytes):
        return {
            "bytes_length": len(value),
            "bytes_sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def clean_cell(value: Any) -> str:
    """Convert one cell to a one-line, tab-safe deterministic string."""
    value = _plain_value(value)
    if value is None:
        return NA_TEXT
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".15g") if math.isfinite(value) else NA_TEXT
    if isinstance(value, (dict, list)):
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    else:
        text = str(value)
    return " ".join(text.replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def _metadata_lines(
    *,
    title: str,
    columns: Sequence[str],
    row_count: int,
    metadata: Mapping[str, Any] | None,
) -> list[str]:
    lines = [
        "# Afruz PXRD Analyzer clean text export",
        f"# text_schema_version: {TEXT_EXPORT_SCHEMA_VERSION}",
        f"# title: {clean_cell(title)}",
        f"# created_utc: {utc_timestamp()}",
        "# encoding: UTF-8",
        "# line_ending: LF",
        "# delimiter: TAB",
        f"# missing_value: {NA_TEXT}",
        f"# row_count: {int(row_count)}",
        f"# column_count: {len(columns)}",
    ]
    for key, value in (metadata or {}).items():
        lines.append(f"# {clean_cell(key)}: {clean_cell(value)}")
    return lines


def write_table_txt(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    title: str = "Afruz table export",
    metadata: Mapping[str, Any] | None = None,
    backup: bool = True,
) -> dict[str, Any]:
    target = ensure_txt_path(path)
    normalized_rows = [dict(row) for row in rows]
    if columns is None:
        ordered: list[str] = []
        for row in normalized_rows:
            for key in row:
                name = str(key)
                if name not in ordered:
                    ordered.append(name)
        columns = ordered or ["value"]
    else:
        columns = [str(column) for column in columns]
    lines = _metadata_lines(
        title=title,
        columns=columns,
        row_count=len(normalized_rows),
        metadata=metadata,
    )
    lines.append("\t".join(clean_cell(column) for column in columns))
    for row in normalized_rows:
        cells = [clean_cell(row.get(column)) for column in columns]
        if len(cells) != len(columns):  # defensive invariant
            raise ValueError("TXT export row has a different width than its header.")
        lines.append("\t".join(cells))
    atomic_write_text(target, "\n".join(lines), backup=backup)
    return {
        "txt_path": str(target),
        "txt_sha256": sha256_file(target),
        "row_count": len(normalized_rows),
        "column_count": len(columns),
        "columns": list(columns),
    }


def write_columns_txt(
    path: str | Path,
    columns: Mapping[str, Sequence[Any]],
    *,
    title: str = "Afruz profile export",
    metadata: Mapping[str, Any] | None = None,
    backup: bool = True,
) -> dict[str, Any]:
    names = [str(name) for name in columns]
    values = [list(columns[name]) for name in columns]
    lengths = {len(column) for column in values}
    if len(lengths) > 1:
        detail = ", ".join(f"{name}={len(column)}" for name, column in zip(names, values))
        raise ValueError(f"Cannot export misaligned columns: {detail}")
    row_count = next(iter(lengths), 0)
    rows = [
        {name: values[col_index][row_index] for col_index, name in enumerate(names)}
        for row_index in range(row_count)
    ]
    return write_table_txt(
        path,
        rows,
        columns=names,
        title=title,
        metadata=metadata,
        backup=backup,
    )


def _flatten_mapping(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    plain = _plain_value(value)
    if isinstance(plain, Mapping):
        rows: list[tuple[str, Any]] = []
        for key in sorted(plain, key=lambda item: str(item)):
            child = f"{prefix}.{key}" if prefix else str(key)
            item = plain[key]
            if isinstance(item, Mapping):
                rows.extend(_flatten_mapping(item, child))
            else:
                rows.append((child, item))
        return rows
    return [(prefix or "value", plain)]


def write_mapping_txt(
    path: str | Path,
    mapping: Mapping[str, Any] | Any,
    *,
    title: str = "Afruz result record",
    metadata: Mapping[str, Any] | None = None,
    backup: bool = True,
) -> dict[str, Any]:
    rows = [{"key": key, "value": value} for key, value in _flatten_mapping(mapping)]
    return write_table_txt(
        path,
        rows,
        columns=["key", "value"],
        title=title,
        metadata=metadata,
        backup=backup,
    )


def write_manifest_txt(
    path: str | Path,
    files: Iterable[str | Path],
    *,
    root: str | Path | None = None,
    title: str = "Afruz export manifest",
    backup: bool = True,
) -> dict[str, Any]:
    base = Path(root).resolve() if root is not None else None
    rows = []
    for item in sorted((Path(file) for file in files), key=lambda p: str(p)):
        if not item.is_file():
            continue
        resolved = item.resolve()
        try:
            display = str(resolved.relative_to(base)) if base is not None else str(resolved)
        except ValueError:
            display = str(resolved)
        rows.append(
            {
                "relative_path": display.replace("\\", "/"),
                "size_bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return write_table_txt(
        path,
        rows,
        columns=["relative_path", "size_bytes", "sha256"],
        title=title,
        backup=backup,
    )
