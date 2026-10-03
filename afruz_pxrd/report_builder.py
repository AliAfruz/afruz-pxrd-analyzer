from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import is_dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import html
import json
import math
import re
from typing import Any

from .text_export import write_columns_txt, write_table_txt, write_mapping_txt, write_manifest_txt
from .version import APP_NAME, APP_VERSION

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


REPORT_SCHEMA_VERSION = 2


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_filename(name: str, fallback: str = "dataset") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._")
    return text or fallback


def _jsonable(value: Any) -> Any:
    if np is not None and isinstance(value, np.ndarray):
        array = np.asarray(value)
        return {
            "array_summary": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "min": _finite_float(np.nanmin(array)) if array.size else None,
            "max": _finite_float(np.nanmax(array)) if array.size else None,
            "mean": _finite_float(np.nanmean(array)) if array.size else None,
            "sha256": hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest(),
        }
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _dataset_name(dataset: Any, index: int) -> str:
    if isinstance(dataset, Mapping):
        return str(dataset.get("name") or dataset.get("dataset_name") or f"Dataset {index}")
    return str(getattr(dataset, "name", f"Dataset {index}"))


def _dataset_uid(dataset: Any, index: int) -> str:
    if isinstance(dataset, Mapping):
        return str(dataset.get("uid") or dataset.get("dataset_uid") or f"dataset-{index}")
    return str(getattr(dataset, "uid", f"dataset-{index}"))


def _dataset_array(dataset: Any, *keys: str) -> Any:
    if isinstance(dataset, Mapping):
        for key in keys:
            if key in dataset:
                return dataset[key]
        return None
    for key in keys:
        if hasattr(dataset, key):
            return getattr(dataset, key)
    return None


def _to_float_array(value: Any) -> list[float]:
    if value is None:
        return []
    if np is not None:
        array = np.asarray(value, dtype=float).ravel()
        return [float(row) if math.isfinite(float(row)) else float("nan") for row in array]
    return [float(row) for row in value]


def export_dataset_points(
    output_dir: str | Path,
    dataset: Any,
    *,
    index: int = 1,
    include_processed: bool = True,
) -> dict[str, Any]:
    """Export all measured points as clean TXT plus a compatibility CSV.

    The CSV is deliberately separate from the JSON report so large experimental
    patterns remain complete without making the report metadata unreadable.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    name = _dataset_name(dataset, index)
    uid = _dataset_uid(dataset, index)
    x = _to_float_array(_dataset_array(dataset, "x", "two_theta", "two_theta_deg"))
    raw = _to_float_array(_dataset_array(dataset, "y_raw", "raw", "raw_intensity", "intensity"))
    processed_value = _dataset_array(dataset, "y_processed", "processed", "processed_intensity")
    processed = _to_float_array(processed_value) if processed_value is not None else []

    if len(x) != len(raw):
        raise ValueError(f"Dataset {name!r} has mismatched x/raw lengths.")
    if processed and len(processed) != len(raw):
        raise ValueError(f"Dataset {name!r} has mismatched processed/raw lengths.")

    filename = f"{index:02d}_{safe_filename(name)}_data_points.csv"
    path = directory / filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["index", "two_theta_deg", "raw_intensity"]
        if include_processed and processed:
            header.extend(["prepared_intensity", "prepared_minus_raw"])
        writer.writerow(header)
        for row_index, two_theta in enumerate(x):
            row = [row_index, two_theta, raw[row_index]]
            if include_processed and processed:
                row.extend([processed[row_index], processed[row_index] - raw[row_index]])
            writer.writerow(row)

    txt_columns = {
        "index": list(range(len(x))),
        "two_theta_deg": x,
        "raw_intensity": raw,
    }
    if include_processed and processed:
        txt_columns["prepared_intensity"] = processed
        txt_columns["prepared_minus_raw"] = [p - r for p, r in zip(processed, raw)]
    txt_result = write_columns_txt(
        path.with_suffix(".txt"),
        txt_columns,
        title=f"Complete diffraction data points — {name}",
        metadata={
            "dataset_uid": uid,
            "dataset_name": name,
            "source": "Afruz complete report package",
        },
        backup=False,
    )

    summary = {
        "dataset_uid": uid,
        "dataset_name": name,
        "point_count": len(x),
        "has_prepared_pattern": bool(processed),
        "csv_path": str(path),
        "csv_sha256": sha256_file(path),
        "txt_path": txt_result["txt_path"],
        "txt_sha256": txt_result["txt_sha256"],
        "two_theta_min": min(x) if x else None,
        "two_theta_max": max(x) if x else None,
        "raw_min": min(raw) if raw else None,
        "raw_max": max(raw) if raw else None,
    }
    if processed:
        summary.update(
            {
                "prepared_min": min(processed),
                "prepared_max": max(processed),
            }
        )
    return summary


def _table_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        if all(isinstance(row, Mapping) for row in value):
            return [dict(row) for row in value]
        return [{"value": row} for row in value]
    if isinstance(value, tuple):
        return _table_rows(list(value))
    if isinstance(value, Mapping):
        for key in (
            "peaks",
            "master_peaks",
            "peak_list",
            "candidates",
            "candidate_cells",
            "phases",
            "results",
            "rows",
            "records",
            "reflections",
            "refinement_results",
            "qpa_results",
            "validation_results",
        ):
            if key in value:
                rows = _table_rows(value[key])
                if rows:
                    return rows
        return [dict(value)]
    if is_dataclass(value):
        return [asdict(value)]
    return [{"value": value}]


def export_table_csv(
    output_dir: str | Path,
    table_name: str,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cleaned_rows = [{str(key): _jsonable(value) for key, value in dict(row).items()} for row in rows]
    path = directory / f"{safe_filename(table_name)}.csv"
    columns: list[str] = []
    for row in cleaned_rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["value"])
        writer.writeheader()
        for row in cleaned_rows:
            writer.writerow({key: _stringify_cell(row.get(key, "")) for key in writer.fieldnames})
    txt_result = write_table_txt(
        path.with_suffix(".txt"),
        cleaned_rows,
        columns=columns or ["value"],
        title=f"Analysis table — {table_name}",
        metadata={"table_name": table_name},
        backup=False,
    )
    return {
        "name": table_name,
        "row_count": len(cleaned_rows),
        "column_count": len(columns),
        "csv_path": str(path),
        "csv_sha256": sha256_file(path),
        "txt_path": txt_result["txt_path"],
        "txt_sha256": txt_result["txt_sha256"],
        "columns": columns,
    }


def _stringify_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _collect_analysis_tables(analysis_state: Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(analysis_state, Mapping):
        return {}
    tables: dict[str, list[dict[str, Any]]] = {}
    preferred_keys = {
        "peak_list",
        "master_peak_list",
        "master_reflection_list",
        "unknown_peaks",
        "candidate_cells",
        "native_indexing",
        "multiphase_unknown_discovery",
        "cif_matches",
        "structure_candidates",
        "pawley_le_bail",
        "whole_pattern",
        "rietveld",
        "qpa",
        "validated_qpa",
        "microstructure",
        "structure_plausibility",
        "validation",
    }
    for key, value in analysis_state.items():
        if key in preferred_keys or any(token in str(key).lower() for token in ("peak", "phase", "cell", "qpa", "rietveld", "validation", "cif", "structure")):
            rows = _table_rows(value)
            if rows:
                tables[str(key)] = rows
    return tables


def build_complete_report_package(
    output_dir: str | Path,
    *,
    project_name: str,
    datasets: Iterable[Any] = (),
    scientific_state: Mapping[str, Any] | Any | None = None,
    analysis_state: Mapping[str, Any] | None = None,
    accepted_results: Mapping[str, Any] | None = None,
    warnings: Iterable[str] = (),
    figures: Iterable[Mapping[str, Any]] = (),
    tables: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    author: str = "",
    notes: str = "",
    include_data_points: bool = True,
    include_html: bool = True,
) -> dict[str, Any]:
    """Create a complete, reproducible Afruz report package.

    The package contains readable reports plus all numerical data as clean tab-delimited TXT (and legacy CSV copies for compatibility). It is
    designed for thesis/paper review, lab records and reproducible sharing.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data_points"
    table_dir = root / "tables"
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    warnings_list = [str(warning) for warning in warnings]
    figure_rows = [dict(row) for row in figures]
    dataset_exports: list[dict[str, Any]] = []
    dataset_list = list(datasets or [])
    if include_data_points:
        for index, dataset in enumerate(dataset_list, start=1):
            dataset_exports.append(export_dataset_points(data_dir, dataset, index=index))

    table_exports: list[dict[str, Any]] = []
    table_sources: dict[str, Iterable[Mapping[str, Any]]] = {}
    table_sources.update(_collect_analysis_tables(analysis_state))
    if tables:
        table_sources.update({str(key): list(value) for key, value in tables.items()})
    for name, rows in sorted(table_sources.items()):
        row_list = [dict(row) for row in rows]
        if row_list:
            table_exports.append(export_table_csv(table_dir, name, row_list))

    payload = {
        "report_type": "Afruz complete scientific report package",
        "schema_version": REPORT_SCHEMA_VERSION,
        "application_name": APP_NAME,
        "application_version": APP_VERSION,
        "project_name": project_name,
        "author": author,
        "created_at": utc_timestamp(),
        "notes": notes,
        "dataset_count": len(dataset_list),
        "dataset_exports": _relative_paths(dataset_exports, root),
        "accepted_results": _jsonable(dict(accepted_results or {})),
        "scientific_state": _jsonable(scientific_state.to_dict() if hasattr(scientific_state, "to_dict") else scientific_state),
        "analysis_state_summary": _jsonable(_summarize_analysis_state(analysis_state)),
        "table_exports": _relative_paths(table_exports, root),
        "figures": _jsonable(figure_rows),
        "warnings": warnings_list,
        "scientific_boundary": (
            "This package records data, selected results, settings and provenance. "
            "It does not replace expert crystallographic review, external validation, "
            "or independent confirmation of an unknown structure."
        ),
    }

    json_path = root / "afruz_complete_report.json"
    markdown_path = root / "afruz_complete_report.md"
    html_path = root / "afruz_complete_report.html"
    manifest_path = root / "manifest.json"
    report_txt_path = root / "afruz_complete_report.txt"
    manifest_txt_path = root / "manifest.txt"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(_render_complete_markdown(payload), encoding="utf-8")
    write_mapping_txt(
        report_txt_path,
        payload,
        title=f"Afruz complete scientific report — {project_name}",
        metadata={
            "application_name": APP_NAME,
            "application_version": APP_VERSION,
            "project_name": project_name,
            "report_schema_version": REPORT_SCHEMA_VERSION,
        },
        backup=False,
    )
    if include_html:
        html_path.write_text(_render_html(_render_complete_markdown(payload)), encoding="utf-8")

    files = [json_path, markdown_path, report_txt_path]
    if include_html:
        files.append(html_path)
    files.extend(Path(row["csv_path"]) for row in dataset_exports)
    files.extend(Path(row["txt_path"]) for row in dataset_exports)
    files.extend(Path(row["csv_path"]) for row in table_exports)
    files.extend(Path(row["txt_path"]) for row in table_exports)

    manifest = {
        "package_type": "Afruz complete scientific report package",
        "schema_version": REPORT_SCHEMA_VERSION,
        "application_name": APP_NAME,
        "application_version": APP_VERSION,
        "project_name": project_name,
        "created_at": payload["created_at"],
        "files": [
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(files)
            if path.exists()
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest_txt(
        manifest_txt_path,
        files + [manifest_path],
        root=root,
        title=f"Afruz complete report manifest — {project_name}",
        backup=False,
    )
    payload["manifest_path"] = str(manifest_path)
    payload["report_txt_path"] = str(report_txt_path)
    payload["manifest_txt_path"] = str(manifest_txt_path)
    payload["manifest_sha256"] = sha256_file(manifest_path)

    return {
        "output_dir": str(root.resolve()),
        "json_path": str(json_path.resolve()),
        "markdown_path": str(markdown_path.resolve()),
        "html_path": str(html_path.resolve()) if include_html else "",
        "manifest_path": str(manifest_path.resolve()),
        "report_txt_path": str(report_txt_path.resolve()),
        "manifest_txt_path": str(manifest_txt_path.resolve()),
        "dataset_exports": dataset_exports,
        "table_exports": table_exports,
        "payload": payload,
    }


def _relative_paths(rows: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        for key in ("csv_path", "txt_path", "json_path", "markdown_path", "html_path"):
            if key in item and item[key]:
                try:
                    item[key] = str(Path(item[key]).resolve().relative_to(root.resolve()))
                except Exception:
                    item[key] = str(item[key])
        out.append(item)
    return out


def _summarize_analysis_state(analysis_state: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(analysis_state, Mapping):
        return {}
    summary = {}
    for key, value in analysis_state.items():
        rows = _table_rows(value)
        if rows:
            summary[str(key)] = {"row_count": len(rows), "type": type(value).__name__}
        else:
            summary[str(key)] = {"type": type(value).__name__}
    return summary


def build_reproducibility_report(
    output_dir: str | Path,
    *,
    project_name: str,
    dataset_summary: dict,
    accepted_results: dict,
    warnings: Iterable[str] = (),
    figures: Iterable[dict] = (),
) -> dict:
    """Backward-compatible Phase 18.0 report entry point.

    The legacy function still writes the original filenames while internally
    benefiting from the more complete Phase 18.1 renderer.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    warnings = list(warnings)
    figures = list(figures)
    payload = {
        "report_type": "Afruz reproducibility report",
        "application_name": APP_NAME,
        "application_version": APP_VERSION,
        "project_name": project_name,
        "dataset_summary": dataset_summary,
        "accepted_results": accepted_results,
        "warnings": warnings,
        "figures": figures,
        "scientific_boundary": (
            "The report records accepted analysis results and provenance; it does not replace expert crystallographic review."
        ),
    }
    json_path = directory / "afruz_reproducibility_report.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = directory / "afruz_reproducibility_report.md"
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "output_dir": str(directory.resolve()),
        "json_path": str(json_path.resolve()),
        "markdown_path": str(markdown_path.resolve()),
        "payload": payload,
    }


def _render_complete_markdown(payload: dict) -> str:
    lines = [
        f"# Afruz complete scientific report — {payload['project_name']}",
        "",
        f"- **Application**: {payload.get('application_name', APP_NAME)} {payload.get('application_version', APP_VERSION)}",
        f"- **Created**: {payload.get('created_at', '')}",
        f"- **Author**: {payload.get('author') or 'Not specified'}",
        f"- **Datasets**: {payload.get('dataset_count', 0)}",
        "",
    ]
    if payload.get("notes"):
        lines.extend(["## Notes", str(payload["notes"]), ""])

    lines.extend(["## Data point exports", ""])
    exports = payload.get("dataset_exports", [])
    if exports:
        lines.append("| Dataset | Points | Prepared | CSV | SHA-256 |")
        lines.append("|---|---:|---|---|---|")
        for row in exports:
            lines.append(
                f"| {row.get('dataset_name','')} | {row.get('point_count',0)} | "
                f"{row.get('has_prepared_pattern', False)} | `{row.get('csv_path','')}` | `{str(row.get('csv_sha256',''))[:12]}…` |"
            )
    else:
        lines.append("No data-point CSV exports were requested or supplied.")
    lines.append("")

    lines.extend(["## Accepted scientific results", ""])
    accepted = payload.get("accepted_results", {}) or {}
    if accepted:
        for stage, record in accepted.items():
            lines.append(f"### {stage}")
            _append_mapping_lines(lines, record)
            lines.append("")
    else:
        lines.append("No accepted results were supplied.")
        lines.append("")

    lines.extend(["## Analysis tables", ""])
    table_exports = payload.get("table_exports", [])
    if table_exports:
        lines.append("| Table | Rows | Columns | CSV | SHA-256 |")
        lines.append("|---|---:|---:|---|---|")
        for row in table_exports:
            lines.append(
                f"| {row.get('name','')} | {row.get('row_count',0)} | {row.get('column_count',0)} | "
                f"`{row.get('csv_path','')}` | `{str(row.get('csv_sha256',''))[:12]}…` |"
            )
    else:
        lines.append("No tabular analysis exports were supplied.")
    lines.append("")

    lines.extend(["## Figures", ""])
    figures = payload.get("figures", []) or []
    if figures:
        for index, figure in enumerate(figures, start=1):
            lines.append(f"{index}. **{figure.get('title', 'Figure')}** — {figure.get('description', '')}")
    else:
        lines.append("No figure metadata were supplied.")
    lines.append("")

    lines.extend(["## Warnings and limitations", ""])
    if payload.get("warnings"):
        for warning in payload["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- No unresolved warnings were supplied to this report builder.")
    lines.append("")
    lines.extend(["## Scientific boundary", payload["scientific_boundary"], ""])
    lines.extend([
        "## Reproducibility files",
        "",
        "This report package includes machine-readable JSON metadata, a SHA-256 manifest, full exported data-point CSV files, and all supplied analysis tables.",
        "",
    ])
    return "\n".join(lines)


def _append_mapping_lines(lines: list[str], value: Any, *, indent: int = 0) -> None:
    prefix = "  " * indent
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, Mapping):
                lines.append(f"{prefix}- **{key}**:")
                _append_mapping_lines(lines, item, indent=indent + 1)
            elif isinstance(item, list):
                lines.append(f"{prefix}- **{key}**: {len(item)} item(s)")
            else:
                lines.append(f"{prefix}- **{key}**: {item}")
    else:
        lines.append(f"{prefix}- {value}")


def _render_html(markdown: str) -> str:
    # Compact, dependency-free Markdown-ish HTML renderer for report portability.
    body_lines = []
    in_table = False
    for line in markdown.splitlines():
        if line.startswith("# "):
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
            continue
        if line.startswith("## "):
            body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
            continue
        if line.startswith("### "):
            body_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", ":", " "} for cell in cells):
                continue
            if not in_table:
                body_lines.append("<table>")
                in_table = True
            tag = "th" if not body_lines[-1].startswith("<tr>") and len(body_lines) >= 1 else "td"
            if any(cell in ("Dataset", "Table") for cell in cells):
                tag = "th"
            body_lines.append("<tr>" + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells) + "</tr>")
            continue
        if in_table:
            body_lines.append("</table>")
            in_table = False
        if line.startswith("- "):
            body_lines.append(f"<p>• {html.escape(line[2:])}</p>")
        elif not line.strip():
            body_lines.append("")
        else:
            body_lines.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        body_lines.append("</table>")
    style = """
body { font-family: Arial, sans-serif; margin: 36px; line-height: 1.45; color: #1f2937; }
h1, h2, h3 { color: #111827; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; }
th, td { border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; font-size: 13px; }
th { background: #f3f4f6; }
code { background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }
"""
    return "<!doctype html><html><head><meta charset='utf-8'><title>Afruz report</title><style>" + style + "</style></head><body>" + "\n".join(body_lines) + "</body></html>"


def _render_markdown(payload: dict) -> str:
    lines = [
        f"# Afruz reproducibility report — {payload['project_name']}",
        "",
        f"- **Application**: {payload.get('application_name', APP_NAME)} {payload.get('application_version', APP_VERSION)}",
        "",
        "## Dataset",
    ]
    for key, value in payload.get("dataset_summary", {}).items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Accepted results"])
    for stage, record in payload.get("accepted_results", {}).items():
        lines.append(f"### {stage}")
        if isinstance(record, dict):
            for key, value in record.items():
                lines.append(f"- **{key}**: {value}")
        else:
            lines.append(f"- {record}")
        lines.append("")
    lines.append("## Warnings")
    if payload.get("warnings"):
        for warning in payload["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- No unresolved warnings were supplied to this report builder.")
    lines.extend(["", "## Scientific boundary", payload["scientific_boundary"], ""])
    return "\n".join(lines)


def create_report_archive(report_dir: str | Path, archive_path: str | Path | None = None) -> dict[str, Any]:
    """Zip an existing complete report package for sharing."""
    import zipfile

    directory = Path(report_dir)
    if not directory.exists():
        raise FileNotFoundError(directory)
    if archive_path is None:
        archive_path = directory.with_suffix(".zip")
    archive = Path(archive_path)
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_handle:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zip_handle.write(path, path.relative_to(directory.parent))
    return {
        "archive_path": str(archive.resolve()),
        "archive_sha256": sha256_file(archive),
        "size_bytes": archive.stat().st_size,
    }
