from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import re
import zipfile
from typing import Any
from xml.sax.saxutils import escape

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


class PlotDataExportError(ValueError):
    """Raised when plot-data workbook export cannot be built safely."""


def _jsonable(value: Any) -> Any:
    if np is not None and isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_filename(name: str, fallback: str = "plot_data_export") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._")
    return text or fallback


def _dataset_field(dataset: Any, *names: str) -> Any:
    if dataset is None:
        return None
    if isinstance(dataset, Mapping):
        for name in names:
            if name in dataset:
                return dataset[name]
        return None
    for name in names:
        if hasattr(dataset, name):
            return getattr(dataset, name)
    return None


def _dataset_name(dataset: Any) -> str:
    return str(_dataset_field(dataset, "name", "dataset_name") or "PXRD dataset")


def _dataset_uid(dataset: Any) -> str:
    return str(_dataset_field(dataset, "uid", "dataset_uid") or "unknown")


def _to_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if np is not None:
        array = np.asarray(value, dtype=float).ravel()
        return [float(item) if math.isfinite(float(item)) else float("nan") for item in array]
    out = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            out.append(float("nan"))
    return out


def _list_or_empty(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _best_full_refinement(result: Mapping | None) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {}
    best_id = ((result.get("best_model") or {}) if isinstance(result.get("best_model"), Mapping) else {}).get("model_id")
    full_results = result.get("full_refinement_results") or []
    if isinstance(full_results, Sequence):
        for row in full_results:
            if isinstance(row, Mapping) and row.get("model_id") == best_id:
                refinement = row.get("refinement")
                if isinstance(refinement, Mapping):
                    return dict(refinement)
        for row in full_results:
            if isinstance(row, Mapping) and isinstance(row.get("refinement"), Mapping):
                return dict(row["refinement"])
    if isinstance(result.get("refinement"), Mapping):
        return dict(result["refinement"])
    return {}


def _pad(values: list[Any], n: int) -> list[Any]:
    return values + [None] * max(0, n - len(values))


def _phase_profile_columns(refinement: Mapping, n: int) -> tuple[list[str], list[list[float]]]:
    headers: list[str] = []
    columns: list[list[float]] = []
    for phase in _list_or_empty(refinement.get("phases")):
        if not isinstance(phase, Mapping):
            continue
        profile = _to_float_list(phase.get("profile_y"))
        if not profile:
            continue
        label = str(phase.get("phase_name") or f"phase_{len(headers) + 1}")
        clean = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") or f"phase_{len(headers) + 1}"
        headers.append(f"phase_{clean}_refined_line_y")
        columns.append(_pad(profile, n))
    return headers, columns


def build_plot_data_workbook_sheets(
    *,
    dataset: Any = None,
    result: Mapping | None = None,
    include_tables: bool = True,
) -> list[dict[str, Any]]:
    """Build Excel sheet payloads for raw/prepared/refined PXRD plot data.

    The main Plot_Data sheet is designed for publication/audit workflows: each
    row is one plotted point and each curve gets its own explicit column.
    """

    result = dict(result or {})
    refinement = _best_full_refinement(result)

    x = _to_float_list(refinement.get("observed_x")) or _to_float_list(_dataset_field(dataset, "x", "two_theta", "two_theta_deg"))
    raw = _to_float_list(_dataset_field(dataset, "y_raw", "raw", "raw_intensity", "intensity"))
    prepared = _to_float_list(_dataset_field(dataset, "y_processed", "processed", "prepared_intensity"))
    observed = _to_float_list(refinement.get("observed_y")) or prepared or raw
    calculated = _to_float_list(refinement.get("calculated_y"))
    background = _to_float_list(refinement.get("background_y"))
    difference = _to_float_list(refinement.get("difference_y"))

    n = max(len(x), len(raw), len(prepared), len(observed), len(calculated), len(background), len(difference))
    if n == 0:
        raise PlotDataExportError("No plot data were available for Excel export.")
    if not x:
        x = list(range(n))
    x = _pad(x, n)
    raw = _pad(raw, n)
    prepared = _pad(prepared, n)
    observed = _pad(observed, n)
    calculated = _pad(calculated, n)
    background = _pad(background, n)
    difference = _pad(difference, n)

    phase_headers, phase_columns = _phase_profile_columns(refinement, n)

    header = [
        "index",
        "two_theta_deg",
        "raw_intensity",
        "smoothed_or_prepared_intensity",
        "fixed_for_refinement_intensity",
        "background_y",
        "background_corrected_fixed_intensity",
        "refined_total_y",
        "difference_observed_minus_refined_y",
        "absolute_difference_y",
    ] + phase_headers

    rows = [header]
    for i in range(n):
        obs = observed[i]
        bkg = background[i]
        diff = difference[i]
        try:
            corrected = None if obs is None or bkg is None else float(obs) - float(bkg)
        except Exception:
            corrected = None
        try:
            abs_diff = None if diff is None else abs(float(diff))
        except Exception:
            abs_diff = None
        rows.append(
            [
                i,
                x[i],
                raw[i],
                prepared[i],
                obs,
                bkg,
                corrected,
                calculated[i],
                diff,
                abs_diff,
            ]
            + [column[i] for column in phase_columns]
        )

    sheets: list[dict[str, Any]] = [
        {
            "name": "Plot_Data",
            "rows": rows,
            "freeze_header": True,
            "description": "Point-by-point curves: raw, prepared/smoothed, background, corrected, refined total and phase lines.",
        }
    ]

    metadata_rows = [
        ["field", "value"],
        ["export_schema", "Afruz plot data Excel export v1"],
        ["created_utc", datetime.now(timezone.utc).replace(microsecond=0).isoformat()],
        ["dataset_name", _dataset_name(dataset)],
        ["dataset_uid", _dataset_uid(dataset)],
        ["point_count", n],
        ["has_raw_intensity", bool(any(item is not None for item in raw))],
        ["has_smoothed_or_prepared_intensity", bool(any(item is not None for item in prepared))],
        ["has_background_curve", bool(any(item is not None for item in background))],
        ["has_refined_total_curve", bool(any(item is not None for item in calculated))],
        ["phase_profile_curve_count", len(phase_headers)],
    ]
    best = result.get("best_model") if isinstance(result.get("best_model"), Mapping) else {}
    if best:
        metadata_rows.extend(
            [
                ["best_model", best.get("model_id")],
                ["best_model_status", best.get("status_label")],
                ["best_model_rwp_percent", best.get("rwp_percent")],
                ["best_model_bic", best.get("bic")],
            ]
        )
    sheets.insert(0, {"name": "Metadata", "rows": metadata_rows, "freeze_header": True})

    if include_tables and result:
        table_map = result.get("report_tables") if isinstance(result.get("report_tables"), Mapping) else {}
        for title, rows_value in (
            ("Ranked_Models", result.get("ranked_models") or table_map.get("phase19_ranked_models")),
            ("Phase_Fractions", (best or {}).get("phase_fractions") or table_map.get("phase19_best_phase_fractions")),
            ("Rejected_Phases", result.get("rejected_phases") or table_map.get("phase19_rejected_phases")),
            ("Candidate_Phases", result.get("candidate_phases") or table_map.get("phase19_candidate_phases")),
        ):
            table_rows = _mapping_rows(rows_value)
            if table_rows:
                sheets.append({"name": title, "rows": table_rows, "freeze_header": True})

        reflection_rows = _mapping_rows(refinement.get("reflections"))
        if reflection_rows:
            sheets.append({"name": "Reflections", "rows": reflection_rows, "freeze_header": True})

        full_rows = result.get("full_refinement_results") or []
        diagnostics = {}
        if isinstance(full_rows, Sequence):
            best_id = best.get("model_id") if isinstance(best, Mapping) else None
            for row in full_rows:
                if isinstance(row, Mapping) and row.get("model_id") == best_id and isinstance(row.get("diagnostics"), Mapping):
                    diagnostics = dict(row["diagnostics"])
                    break
        peak_expl = diagnostics.get("peak_explanation") if isinstance(diagnostics.get("peak_explanation"), Mapping) else {}
        explained = _mapping_rows(peak_expl.get("explained"))
        unexplained = _mapping_rows(peak_expl.get("unexplained"))
        residual = _mapping_rows(diagnostics.get("largest_residual_peaks"))
        if explained:
            sheets.append({"name": "Explained_Peaks", "rows": explained, "freeze_header": True})
        if unexplained:
            sheets.append({"name": "Unexplained_Peaks", "rows": unexplained, "freeze_header": True})
        if residual:
            sheets.append({"name": "Residual_Peaks", "rows": residual, "freeze_header": True})

        warnings = [["warning"]] + [[item] for item in _list_or_empty(result.get("scientific_warnings"))]
        if len(warnings) > 1:
            sheets.append({"name": "Warnings", "rows": warnings, "freeze_header": True})

    return sheets


def _mapping_rows(value: Any) -> list[list[Any]]:
    rows = _list_or_empty(value)
    mappings = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not mappings:
        return []
    columns: list[str] = []
    for row in mappings:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    out = [columns]
    for row in mappings:
        out.append([_cell_value(row.get(column)) for column in columns])
    return out


def _cell_value(value: Any) -> Any:
    value = _jsonable(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def export_plot_data_excel(
    path: str | Path,
    *,
    dataset: Any = None,
    result: Mapping | None = None,
    include_tables: bool = True,
) -> dict[str, Any]:
    """Export all plotted/refined curves and analysis tables to one .xlsx workbook."""

    output = Path(path)
    if output.suffix.lower() != ".xlsx":
        output = output.with_suffix(".xlsx")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheets = build_plot_data_workbook_sheets(dataset=dataset, result=result, include_tables=include_tables)
    write_xlsx(output, sheets)
    return {
        "xlsx_path": str(output),
        "sheet_count": len(sheets),
        "sheets": [sheet["name"] for sheet in sheets],
        "bytes": output.stat().st_size,
    }


def _sheet_name(name: str, used: set[str]) -> str:
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", str(name)).strip() or "Sheet"
    cleaned = cleaned[:31]
    base = cleaned
    counter = 2
    while cleaned in used:
        suffix = f"_{counter}"
        cleaned = (base[: 31 - len(suffix)] + suffix)[:31]
        counter += 1
    used.add(cleaned)
    return cleaned


def _col_name(index: int) -> str:
    index += 1
    letters = ""
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _xml_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 32)
    return escape(text, {'"': '&quot;'})


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _worksheet_xml(rows: Sequence[Sequence[Any]], *, freeze_header: bool = True) -> str:
    row_xml = []
    for r_index, row in enumerate(rows, start=1):
        cells = []
        for c_index, value in enumerate(row):
            ref = f"{_col_name(c_index)}{r_index}"
            style = ' s="1"' if r_index == 1 else ""
            if _is_number(value):
                cells.append(f'<c r="{ref}"{style}><v>{float(value):.15g}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"{style}><is><t>{_xml_text(value)}</t></is></c>')
        row_xml.append(f'<row r="{r_index}">{"".join(cells)}</row>')
    max_col = max((len(row) for row in rows), default=1)
    max_row = max(len(rows), 1)
    dimension = f"A1:{_col_name(max_col - 1)}{max_row}"
    views = ""
    if freeze_header and max_row > 1:
        views = '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
    cols = "".join(f'<col min="{i}" max="{i}" width="16" customWidth="1"/>' for i in range(1, min(max_col, 30) + 1))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/>'
        f'{views}'
        f'<cols>{cols}</cols>'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        '</worksheet>'
    )


def write_xlsx(path: str | Path, sheets: Sequence[Mapping[str, Any]]) -> None:
    """Write a compact, dependency-light Excel workbook from sheet row arrays."""

    used: set[str] = set()
    named_sheets = []
    for index, sheet in enumerate(sheets, start=1):
        rows = sheet.get("rows") or []
        if not rows:
            rows = [["No data"]]
        named_sheets.append(
            {
                "name": _sheet_name(str(sheet.get("name") or f"Sheet{index}"), used),
                "rows": rows,
                "freeze_header": bool(sheet.get("freeze_header", True)),
            }
        )

    path = Path(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml(len(named_sheets)))
        archive.writestr("_rels/.rels", _root_rels_xml())
        archive.writestr("xl/workbook.xml", _workbook_xml(named_sheets))
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(len(named_sheets)))
        archive.writestr("xl/styles.xml", _styles_xml())
        archive.writestr("docProps/core.xml", _core_xml())
        archive.writestr("docProps/app.xml", _app_xml(len(named_sheets)))
        for index, sheet in enumerate(named_sheets, start=1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                _worksheet_xml(sheet["rows"], freeze_header=sheet["freeze_header"]),
            )


def _content_types_xml(sheet_count: int) -> str:
    sheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        f'{sheets}'
        '</Types>'
    )


def _root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        '</Relationships>'
    )


def _workbook_rels_xml(sheet_count: int) -> str:
    rels = ''.join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, sheet_count + 1)
    )
    rels += f'<Relationship Id="rId{sheet_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{rels}'
        '</Relationships>'
    )


def _workbook_xml(sheets: Sequence[Mapping[str, Any]]) -> str:
    sheet_tags = ''.join(
        f'<sheet name="{_xml_text(sheet["name"])}" sheetId="{i}" r:id="rId{i}"/>'
        for i, sheet in enumerate(sheets, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{sheet_tags}</sheets>'
        '</workbook>'
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="1" borderId="0" xfId="0" applyFont="1" applyFill="1"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )


def _core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:creator>Afruz PXRD Analyzer</dc:creator>'
        '<dc:title>Afruz PXRD plot data export</dc:title>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        '</cp:coreProperties>'
    )


def _app_xml(sheet_count: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>Afruz PXRD Analyzer</Application>'
        f'<Worksheets>{sheet_count}</Worksheets>'
        '</Properties>'
    )
