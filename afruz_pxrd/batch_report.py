from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .batch_engine import comparison_rows
from .text_export import write_mapping_txt, write_table_txt, write_manifest_txt
from .version import APP_RELEASE, APP_VERSION


def _safe_filename(text: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(text)
    )
    return safe.strip("._") or "report"


def _comparison_dataframe(batch_result: dict) -> pd.DataFrame:
    rows = comparison_rows(batch_result)
    return pd.DataFrame(rows)


def _stage_dataframe(batch_result: dict) -> pd.DataFrame:
    rows = []
    for dataset in batch_result.get("results", []):
        for stage, details in dataset.get("stages", {}).items():
            rows.append(
                {
                    "dataset": dataset.get("dataset_name"),
                    "stage": stage,
                    "status": details.get("status"),
                    "elapsed_seconds": details.get("elapsed_seconds"),
                    "message": details.get("message"),
                }
            )
    return pd.DataFrame(rows)


def _peak_dataframe(batch_result: dict) -> pd.DataFrame:
    rows = []
    for dataset in batch_result.get("results", []):
        for peak in dataset.get("peak_rows", []):
            rows.append(
                {
                    "dataset": dataset.get("dataset_name"),
                    **peak,
                }
            )
    return pd.DataFrame(rows)


def _fit_dataframe(batch_result: dict) -> pd.DataFrame:
    rows = []
    for dataset in batch_result.get("results", []):
        for group in dataset.get("fit_groups", []):
            for component in group.get("components", []):
                rows.append(
                    {
                        "dataset": dataset.get("dataset_name"),
                        "group_id": group.get("group_id"),
                        "group_model": group.get("model"),
                        "group_r_squared": group.get("r_squared"),
                        "group_rmse": group.get("rmse"),
                        **component,
                    }
                )
    return pd.DataFrame(rows)


def _phase_dataframe(batch_result: dict) -> pd.DataFrame:
    rows = []
    for dataset in batch_result.get("results", []):
        phase_result = dataset.get("phase_identification") or {}
        for candidate in phase_result.get("results", []):
            rows.append(
                {
                    "dataset": dataset.get("dataset_name"),
                    "rank": candidate.get("rank"),
                    "kind": candidate.get("kind"),
                    "phase_count": candidate.get("phase_count"),
                    "candidate": (
                        candidate.get("reference_name")
                        or candidate.get("display_name")
                    ),
                    "score": candidate.get("score"),
                    "matched_count": candidate.get("matched_count"),
                    "observed_coverage_percent": candidate.get(
                        "observed_coverage_percent"
                    ),
                    "reference_coverage_percent": candidate.get(
                        "reference_coverage_percent"
                    ),
                    "mean_absolute_delta_deg": candidate.get(
                        "mean_absolute_delta_deg"
                    ),
                    "zero_shift_deg": candidate.get("zero_shift_deg"),
                    "warnings": "; ".join(candidate.get("warnings", [])),
                }
            )
    return pd.DataFrame(rows)


def _whole_pattern_phase_dataframe(
    batch_result: dict,
) -> pd.DataFrame:
    rows = []
    for dataset in batch_result.get("results", []):
        result = dataset.get("whole_pattern") or {}
        for phase in result.get("phases", []):
            rows.append(
                {
                    "dataset": dataset.get("dataset_name"),
                    "mode": result.get("mode"),
                    "rwp_percent": result.get("rwp_percent"),
                    "rp_percent": result.get("rp_percent"),
                    "zero_shift_deg": result.get("zero_shift_deg"),
                    **phase,
                }
            )
    return pd.DataFrame(rows)


def _whole_pattern_reflection_dataframe(
    batch_result: dict,
) -> pd.DataFrame:
    rows = []
    for dataset in batch_result.get("results", []):
        result = dataset.get("whole_pattern") or {}
        for reflection in result.get("reflections", []):
            rows.append(
                {
                    "dataset": dataset.get("dataset_name"),
                    **reflection,
                }
            )
    return pd.DataFrame(rows)


def _rietveld_phase_dataframe(batch_result: dict) -> pd.DataFrame:
    rows = []
    for dataset in batch_result.get("results", []):
        result = dataset.get("rietveld") or {}
        for phase in result.get("phases", []):
            rows.append(
                {
                    "dataset": dataset.get("dataset_name"),
                    "rwp_percent": result.get("rwp_percent"),
                    "rp_percent": result.get("rp_percent"),
                    "zero_shift_deg": result.get("zero_shift_deg"),
                    **phase,
                }
            )
    return pd.DataFrame(rows)


def _rietveld_reflection_dataframe(batch_result: dict) -> pd.DataFrame:
    rows = []
    for dataset in batch_result.get("results", []):
        result = dataset.get("rietveld") or {}
        for reflection in result.get("reflections", []):
            rows.append({"dataset": dataset.get("dataset_name"), **reflection})
    return pd.DataFrame(rows)


def _qpa_dataframe(batch_result: dict) -> pd.DataFrame:
    rows = []
    for dataset in batch_result.get("results", []):
        qpa = dataset.get("qpa") or {}
        for phase in qpa.get("phases", []):
            rows.append(
                {
                    "dataset": dataset.get("dataset_name"),
                    **phase,
                }
            )
    return pd.DataFrame(rows)


def _error_dataframe(batch_result: dict) -> pd.DataFrame:
    rows = []
    for dataset in batch_result.get("results", []):
        for warning in dataset.get("warnings", []):
            rows.append(
                {
                    "dataset": dataset.get("dataset_name"),
                    "severity": "Warning",
                    "message": warning,
                }
            )
        for error in dataset.get("errors", []):
            rows.append(
                {
                    "dataset": dataset.get("dataset_name"),
                    "severity": "Error",
                    "message": error,
                }
            )
    for error in batch_result.get("errors", []):
        rows.append(
            {
                "dataset": "Batch",
                "severity": "Error",
                "message": error,
            }
        )
    return pd.DataFrame(rows)


def _make_comparison_figure(batch_result: dict, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = _comparison_dataframe(batch_result)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    if frame.empty:
        axis.text(0.5, 0.5, "No completed batch results", ha="center", va="center")
        axis.set_axis_off()
    else:
        x = np.arange(len(frame))
        metric = None
        label = ""
        for candidate, candidate_label in (
            ("primary_peak_position_deg", "Primary peak position (° 2θ)"),
            ("scherrer_mean_nm", "Mean Scherrer size (nm)"),
            ("phase_score", "Phase-identification score"),
            ("peak_count", "Detected peak count"),
        ):
            if candidate in frame and pd.to_numeric(frame[candidate], errors="coerce").notna().any():
                metric = candidate
                label = candidate_label
                break
        if metric is None:
            metric = "elapsed_seconds"
            label = "Analysis time (s)"
        values = pd.to_numeric(frame[metric], errors="coerce")
        axis.plot(x, values, marker="o")
        axis.set_xticks(x)
        axis.set_xticklabels(frame["dataset_name"], rotation=35, ha="right")
        axis.set_ylabel(label)
        axis.set_title("Afruz PXRD batch comparison")
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _make_overlay_figure(batch_result: dict, output_path: Path, svg: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 6))
    offset = 0.0
    plotted = 0
    for dataset in batch_result.get("results", []):
        plot_data = dataset.get("plot_data", {})
        x = np.asarray(plot_data.get("x", []), dtype=float)
        y = np.asarray(plot_data.get("processed_y", []), dtype=float)
        if len(x) < 3 or len(x) != len(y):
            continue
        span = max(float(np.max(y) - np.min(y)), 1.0)
        axis.plot(x, y + offset, linewidth=1.0, label=dataset.get("dataset_name"))
        offset += span * 1.15
        plotted += 1
    if plotted:
        axis.set_xlabel("2θ (degrees)")
        axis.set_ylabel("Intensity + offset")
        axis.set_title("Batch pattern overlay")
        axis.legend(fontsize=7, loc="upper right")
        axis.grid(True, alpha=0.15)
    else:
        axis.text(0.5, 0.5, "No pattern data", ha="center", va="center")
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, format="svg" if svg else None, dpi=180)
    plt.close(figure)


def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    return f"data:{mime};base64,{encoded}"


def write_html_report(
    path: str | Path,
    batch_result: dict,
    comparison_image: Path | None = None,
    overlay_image: Path | None = None,
) -> Path:
    path = Path(path)
    summary = _comparison_dataframe(batch_result)
    errors = _error_dataframe(batch_result)

    recipe = batch_result.get("recipe", {})
    recipe_text = json.dumps(recipe, indent=2, ensure_ascii=False)
    summary_html = (
        summary.fillna("").to_html(index=False, classes="dataframe")
        if not summary.empty
        else "<p>No completed dataset summaries.</p>"
    )
    error_html = (
        errors.fillna("").to_html(index=False, classes="dataframe")
        if not errors.empty
        else "<p>No warnings or errors were recorded.</p>"
    )
    comparison_html = (
        f'<img src="{_image_data_uri(comparison_image)}" alt="comparison">'
        if comparison_image and comparison_image.exists()
        else ""
    )
    overlay_html = (
        f'<img src="{_image_data_uri(overlay_image)}" alt="pattern overlay">'
        if overlay_image and overlay_image.exists()
        else ""
    )

    dataset_sections = []
    for dataset in batch_result.get("results", []):
        metrics = dataset.get("metrics", {})
        stage_frame = pd.DataFrame(
            [
                {
                    "Stage": name,
                    "Status": details.get("status"),
                    "Time (s)": details.get("elapsed_seconds"),
                    "Message": details.get("message"),
                }
                for name, details in dataset.get("stages", {}).items()
            ]
        )
        dataset_sections.append(
            f"""
            <section>
              <h2>{dataset.get('dataset_name')}</h2>
              <p><b>Status:</b> {dataset.get('status')} &nbsp;
                 <b>Elapsed:</b> {dataset.get('elapsed_seconds', 0):.3f} s</p>
              <p><b>Source:</b> {dataset.get('source_path') or 'Embedded/generated'}</p>
              <h3>Key metrics</h3>
              <pre>{json.dumps(metrics, indent=2, ensure_ascii=False)}</pre>
              <h3>Stage trace</h3>
              {stage_frame.fillna('').to_html(index=False, classes='dataframe')}
              <h3>Warnings</h3>
              <p>{'<br>'.join(dataset.get('warnings', [])) or 'None'}</p>
            </section>
            """
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{APP_RELEASE} Batch Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 28px; color: #29251f; }}
h1, h2, h3 {{ color: #8e672a; }}
.card {{ border: 1px solid #c9b89f; padding: 14px; margin: 12px 0; border-radius: 8px; }}
table.dataframe {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
table.dataframe th, table.dataframe td {{ border: 1px solid #c9b89f; padding: 5px; text-align: left; }}
table.dataframe th {{ background: #eadfce; }}
img {{ max-width: 100%; border: 1px solid #c9b89f; margin: 8px 0; }}
pre {{ white-space: pre-wrap; background: #f4efe7; padding: 10px; border-radius: 5px; }}
.small {{ color: #75695c; font-size: 12px; }}
</style>
</head>
<body>
<h1>{APP_RELEASE} — Batch Analysis Report</h1>
<div class="card">
<p><b>Created:</b> {batch_result.get('created_utc')}</p>
<p><b>Application version:</b> {APP_VERSION}</p>
<p><b>Recipe:</b> {recipe.get('name', 'Unnamed')}</p>
<p><b>Recipe fingerprint:</b> {batch_result.get('recipe_fingerprint')}</p>
<p><b>Datasets:</b> {batch_result.get('dataset_count', 0)} &nbsp;
<b>Completed:</b> {batch_result.get('completed_count', 0)} &nbsp;
<b>Failed:</b> {batch_result.get('failed_count', 0)} &nbsp;
<b>Elapsed:</b> {batch_result.get('elapsed_seconds', 0):.3f} s</p>
</div>
<h2>Comparison dashboard</h2>
{comparison_html}
{overlay_html}
{summary_html}
<h2>Warnings and errors</h2>
{error_html}
<h2>Dataset traceability</h2>
{''.join(dataset_sections)}
<h2>Reproducibility recipe</h2>
<pre>{recipe_text}</pre>
<p class="small">Batch comparisons are controlled only when the same recipe fingerprint,
reference library, wavelength, and instrument corrections are used. Results remain
subject to the scientific limitations of the individual analysis modules.</p>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def write_excel_report(path: str | Path, batch_result: dict) -> Path:
    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _comparison_dataframe(batch_result).to_excel(
            writer, sheet_name="Summary", index=False
        )
        _stage_dataframe(batch_result).to_excel(
            writer, sheet_name="Stage Trace", index=False
        )
        _peak_dataframe(batch_result).to_excel(
            writer, sheet_name="Peaks", index=False
        )
        _fit_dataframe(batch_result).to_excel(
            writer, sheet_name="Peak Fits", index=False
        )
        _phase_dataframe(batch_result).to_excel(
            writer, sheet_name="Phase ID", index=False
        )
        _whole_pattern_phase_dataframe(batch_result).to_excel(
            writer,
            sheet_name="Pawley LeBail Phases",
            index=False,
        )
        _whole_pattern_reflection_dataframe(batch_result).to_excel(
            writer,
            sheet_name="Extracted Reflections",
            index=False,
        )
        _rietveld_phase_dataframe(batch_result).to_excel(
            writer,
            sheet_name="Rietveld Phases",
            index=False,
        )
        _rietveld_reflection_dataframe(batch_result).to_excel(
            writer,
            sheet_name="Rietveld Reflections",
            index=False,
        )
        _qpa_dataframe(batch_result).to_excel(
            writer, sheet_name="QPA", index=False
        )
        _error_dataframe(batch_result).to_excel(
            writer, sheet_name="Warnings Errors", index=False
        )
        recipe_frame = pd.DataFrame(
            [
                {
                    "recipe_fingerprint": batch_result.get("recipe_fingerprint"),
                    "recipe_json": json.dumps(
                        batch_result.get("recipe", {}),
                        ensure_ascii=False,
                    ),
                }
            ]
        )
        recipe_frame.to_excel(writer, sheet_name="Recipe", index=False)
        calibration = (
            batch_result.get("recipe", {})
            .get("instrument_calibration", {})
            .get("profile")
        )
        if isinstance(calibration, dict):
            calibration_rows = [
                {
                    "parameter": key,
                    "value": (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                    ),
                }
                for key, value in calibration.items()
                if key != "observations"
            ]
            pd.DataFrame(calibration_rows).to_excel(
                writer,
                sheet_name="Calibration",
                index=False,
            )
    return path


def write_pdf_report(
    path: str | Path,
    batch_result: dict,
    comparison_image: Path | None = None,
    overlay_image: Path | None = None,
) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    path = Path(path)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"{APP_RELEASE} — Batch Analysis Report", styles["Title"]),
        Spacer(1, 4 * mm),
        Paragraph(
            f"Created: {batch_result.get('created_utc')}<br/>"
            f"Application version: {APP_VERSION}<br/>"
            f"Recipe fingerprint: {batch_result.get('recipe_fingerprint')}<br/>"
            f"Datasets: {batch_result.get('dataset_count', 0)}; "
            f"completed: {batch_result.get('completed_count', 0)}; "
            f"failed: {batch_result.get('failed_count', 0)}; "
            f"elapsed: {batch_result.get('elapsed_seconds', 0):.3f} s",
            styles["BodyText"],
        ),
        Spacer(1, 4 * mm),
    ]
    for image_path in (comparison_image, overlay_image):
        if image_path and image_path.exists() and image_path.suffix.lower() == ".png":
            story.append(Image(str(image_path), width=250 * mm, height=130 * mm))
            story.append(Spacer(1, 3 * mm))

    frame = _comparison_dataframe(batch_result)
    if not frame.empty:
        preferred = [
            "dataset_name",
            "status",
            "peak_count",
            "primary_peak_position_deg",
            "primary_peak_fwhm_deg",
            "scherrer_mean_nm",
            "microstrain",
            "identified_phase",
            "phase_score",
        ]
        columns = [column for column in preferred if column in frame.columns]
        compact = frame[columns].fillna("")
        data = [columns] + compact.astype(str).values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eadfce")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(table)

    story.append(PageBreak())
    story.append(Paragraph("Reproducibility recipe", styles["Heading1"]))
    recipe_json = json.dumps(
        batch_result.get("recipe", {}),
        indent=2,
        ensure_ascii=False,
    ).replace("\n", "<br/>").replace(" ", "&nbsp;")
    story.append(Paragraph(recipe_json, styles["Code"]))

    document = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    document.build(story)
    return path


def export_batch_bundle(
    output_directory: str | Path,
    batch_result: dict,
    base_name: str = "afruz_pxrd_batch_report",
) -> dict[str, str]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    base = _safe_filename(base_name)

    paths = {
        "json": output_directory / f"{base}.json",
        "result_txt": output_directory / f"{base}.txt",
        "summary_txt": output_directory / f"{base}_summary.txt",
        "errors_txt": output_directory / f"{base}_warnings_errors.txt",
        "summary_csv": output_directory / f"{base}_summary.csv",
        "errors_csv": output_directory / f"{base}_warnings_errors.csv",
        "excel": output_directory / f"{base}.xlsx",
        "comparison_png": output_directory / f"{base}_comparison.png",
        "overlay_png": output_directory / f"{base}_overlay.png",
        "overlay_svg": output_directory / f"{base}_overlay.svg",
        "html": output_directory / f"{base}.html",
        "pdf": output_directory / f"{base}.pdf",
    }

    paths["json"].write_text(
        json.dumps(batch_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    comparison_frame = _comparison_dataframe(batch_result)
    error_frame = _error_dataframe(batch_result)
    _comparison_dataframe(batch_result).to_csv(paths["summary_csv"], index=False)
    _error_dataframe(batch_result).to_csv(paths["errors_csv"], index=False)
    write_mapping_txt(paths["result_txt"], batch_result, title="Afruz batch complete result", backup=False)
    write_table_txt(paths["summary_txt"], comparison_frame.to_dict("records"), title="Afruz batch comparison summary", backup=False)
    write_table_txt(paths["errors_txt"], error_frame.to_dict("records"), title="Afruz batch warnings and errors", backup=False)
    _make_comparison_figure(batch_result, paths["comparison_png"])
    _make_overlay_figure(batch_result, paths["overlay_png"])
    _make_overlay_figure(batch_result, paths["overlay_svg"], svg=True)
    write_excel_report(paths["excel"], batch_result)
    write_html_report(
        paths["html"],
        batch_result,
        comparison_image=paths["comparison_png"],
        overlay_image=paths["overlay_png"],
    )
    write_pdf_report(
        paths["pdf"],
        batch_result,
        comparison_image=paths["comparison_png"],
        overlay_image=paths["overlay_png"],
    )
    paths["manifest_txt"] = output_directory / f"{base}_manifest.txt"
    write_manifest_txt(
        paths["manifest_txt"],
        [path for key, path in paths.items() if key != "manifest_txt"],
        root=output_directory,
        title="Afruz batch export manifest",
        backup=False,
    )
    return {name: str(path) for name, path in paths.items()}
