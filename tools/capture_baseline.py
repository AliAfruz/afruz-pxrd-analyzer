from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "baseline_artifacts"
WAVELENGTH_ANGSTROM = 1.5406

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def json_default(value: Any):
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gui_only(output: Path) -> int:
    if platform.system() != "Windows":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    output.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "success": False,
        "platform": platform.platform(),
        "python": sys.version,
    }
    started = time.perf_counter()
    try:
        import_started = time.perf_counter()
        from PySide6.QtWidgets import QApplication
        from afruz_pxrd.app import MainWindow

        metrics["application_import_seconds"] = time.perf_counter() - import_started
        app_started = time.perf_counter()
        app = QApplication.instance() or QApplication(["afruz-phase0-baseline"])
        metrics["qapplication_seconds"] = time.perf_counter() - app_started
        window_started = time.perf_counter()
        window = MainWindow()
        metrics["main_window_construction_seconds"] = time.perf_counter() - window_started
        window.show()
        for _ in range(4):
            app.processEvents()
        screenshot_path = output / "gui" / "main_window.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        saved = bool(window.grab().save(str(screenshot_path), "PNG"))
        metrics.update(
            {
                "success": saved,
                "screenshot": str(screenshot_path.relative_to(ROOT)),
                "screenshot_sha256": sha256_file(screenshot_path) if saved else None,
                "window_width": window.width(),
                "window_height": window.height(),
            }
        )
        # The preserved application currently has a known closeEvent defect:
        # it calls ValidatedQPAWidget.is_running(), which is not implemented.
        # Phase 0 records that defect and avoids changing application behavior.
        window.hide()
        window.deleteLater()
        app.processEvents()
        app.quit()
    except Exception as exc:
        metrics["error"] = str(exc)
        metrics["traceback"] = traceback.format_exc()
    metrics["total_gui_smoke_seconds"] = time.perf_counter() - started
    write_json(output / "gui_startup.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if metrics["success"] else 1


def package_versions() -> dict[str, str]:
    names = (
        "PySide6",
        "pyqtgraph",
        "numpy",
        "scipy",
        "pandas",
        "openpyxl",
        "matplotlib",
        "reportlab",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "MISSING"
    return versions


def scientific_source_manifest() -> dict[str, str]:
    files = [ROOT / "main.py", *sorted((ROOT / "afruz_pxrd").glob("*.py"))]
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in files
    }


def compile_sources() -> dict[str, Any]:
    files = [ROOT / "main.py", *sorted((ROOT / "afruz_pxrd").glob("*.py"))]
    errors = []
    started = time.perf_counter()
    for path in files:
        try:
            source = path.read_text(encoding="utf-8-sig")
            compile(source, str(path), "exec")
        except Exception as exc:
            errors.append({"file": str(path.relative_to(ROOT)), "error": str(exc)})
    return {
        "success": not errors,
        "file_count": len(files),
        "elapsed_seconds": time.perf_counter() - started,
        "errors": errors,
    }


def preserve_historical_test_cache(output: Path) -> dict[str, Any]:
    nodeids_path = ROOT / ".pytest_cache" / "v" / "cache" / "nodeids"
    nodeids: list[str] = []
    error = None
    if nodeids_path.exists():
        try:
            loaded = json.loads(nodeids_path.read_text(encoding="utf-8"))
            nodeids = [str(value) for value in loaded if isinstance(value, str)]
        except Exception as exc:
            error = str(exc)
    preserved_path = output / "historical_test_nodeids.json"
    write_json(preserved_path, nodeids)
    return {
        "cached_nodeid_count": len(nodeids),
        "tests_directory_present": (ROOT / "tests").is_dir(),
        "source_cache_present": nodeids_path.exists(),
        "parse_error": error,
        "preserved_file": str(preserved_path.relative_to(ROOT)),
        "preserved_sha256": sha256_file(preserved_path),
    }


def read_table(path: Path):
    import pandas as pd

    return pd.read_csv(path, comment="#", sep="\t")


def statistical_reference_checks() -> dict[str, Any]:
    import numpy as np

    from afruz_pxrd.refinement_statistics import (
        build_refinement_weight_model,
        calculate_profile_statistics,
    )

    directory = ROOT / "Afruz_PXRD_Statistical_Validation_Set"
    raw = read_table(directory / "01_NaCl_raw_counts_with_sigma.txt")
    normalized = read_table(directory / "02_NaCl_normalized_100_with_sigma.txt")
    generating = read_table(directory / "03_NaCl_generating_profile_reference.txt")
    raw_y = raw.iloc[:, 1].to_numpy(dtype=float)
    normalized_y = normalized.iloc[:, 1].to_numpy(dtype=float)
    generating_y = generating.iloc[:, 1].to_numpy(dtype=float)
    nonzero = np.abs(raw_y) > np.finfo(float).eps
    normalization_factor = float(np.median(normalized_y[nonzero] / raw_y[nonzero]))

    cases = []
    expected = {
        "rwp_percent": 6.06433,
        "rexp_percent": 5.96986,
        "goodness_of_fit_sqrt": 1.01582,
        "reduced_chi_square": 1.03190,
    }
    for name, frame, calculated, provenance in (
        ("NaCl raw counts", raw, generating_y, "raw_counts"),
        (
            "NaCl normalized to 100",
            normalized,
            generating_y * normalization_factor,
            "scaled_counts",
        ),
    ):
        observed = frame.iloc[:, 1].to_numpy(dtype=float)
        sigma = frame.iloc[:, 2].to_numpy(dtype=float)
        model = build_refinement_weight_model(
            observed,
            "Poisson-like",
            observed_sigma=sigma,
            provenance=provenance,
        )
        metrics = calculate_profile_statistics(
            observed,
            calculated,
            model,
            parameter_count=8,
        )
        actual = {
            "rwp_percent": metrics["rwp_percent"],
            "rexp_percent": metrics["rexp_percent"],
            "goodness_of_fit_sqrt": metrics["goodness_of_fit_sqrt"],
            "reduced_chi_square": metrics["reduced_chi_square"],
        }
        deltas = {key: actual[key] - expected[key] for key in expected}
        passed = all(abs(value) <= 0.0001 for value in deltas.values())
        cases.append(
            {
                "name": name,
                "success": passed,
                "parameter_count": 8,
                "actual": actual,
                "expected_readme": expected,
                "delta": deltas,
            }
        )
    invariance_delta = {
        key: cases[1]["actual"][key] - cases[0]["actual"][key]
        for key in expected
    }
    return {
        "success": all(case["success"] for case in cases)
        and all(abs(value) <= 1e-6 for value in invariance_delta.values()),
        "normalization_factor": normalization_factor,
        "normalization_invariance_delta": invariance_delta,
        "cases": cases,
    }


def summarize_phase_rows(rows: list[dict]) -> list[dict]:
    selected = []
    for row in rows:
        selected.append(
            {
                "phase_name": row.get("phase_name"),
                "scale_factor": row.get("scale_factor"),
                "scale_fraction_percent": row.get("scale_fraction_percent"),
                "weight_fraction_percent": row.get("weight_fraction_percent"),
                "cell": row.get("cell"),
            }
        )
    return selected


def save_refinement_plot(path: Path, title: str, result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.asarray(result["observed_x"], dtype=float)
    observed = np.asarray(result["observed_y"], dtype=float)
    calculated = np.asarray(result["calculated_y"], dtype=float)
    difference = np.asarray(result["difference_y"], dtype=float)
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    top.plot(x, observed, color="black", linewidth=0.7, label="Observed")
    top.plot(x, calculated, color="#d62728", linewidth=0.8, label="Calculated")
    top.set_ylabel("Intensity")
    top.set_title(title)
    top.legend(loc="upper right")
    bottom.axhline(0.0, color="gray", linewidth=0.6)
    bottom.plot(x, difference, color="#1f77b4", linewidth=0.65)
    bottom.set_xlabel("2θ (degree)")
    bottom.set_ylabel("Difference")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_rietveld_cases(output: Path) -> dict[str, Any]:
    import numpy as np

    from afruz_pxrd.crystallography import load_cif
    from afruz_pxrd.rietveld_refinement import RietveldPhaseSpec, refine_rietveld

    directory = ROOT / "Afruz_PXRD_Statistical_Validation_Set"
    definitions = (
        (
            "nacl_single",
            "01_NaCl_raw_counts_with_sigma.txt",
            (("NaCl", "NaCl_validation_P1.cif"),),
        ),
        (
            "nacl_cscl_mixed",
            "04_NaCl_CsCl_mixed_raw_with_sigma.txt",
            (
                ("NaCl", "NaCl_validation_P1.cif"),
                ("CsCl", "CsCl_validation_P1.cif"),
            ),
        ),
    )
    cases = []
    for key, data_name, phase_files in definitions:
        frame = read_table(directory / data_name)
        x = frame.iloc[:, 0].to_numpy(dtype=float)
        observed = frame.iloc[:, 1].to_numpy(dtype=float)
        sigma = frame.iloc[:, 2].to_numpy(dtype=float)
        specs = [
            RietveldPhaseSpec(
                structure=load_cif(directory / cif_name),
                name=phase_name,
                refine_cell=True,
            )
            for phase_name, cif_name in phase_files
        ]
        started = time.perf_counter()
        result = refine_rietveld(
            x,
            observed,
            specs,
            wavelength_angstrom=WAVELENGTH_ANGSTROM,
            weighting="Poisson-like",
            observed_sigma=sigma,
            intensity_provenance="raw_counts",
            maximum_nonlinear_evaluations=120,
            maximum_optimization_points=1800,
        )
        wall_seconds = time.perf_counter() - started
        result_path = output / "refinements" / f"{key}.json"
        write_json(result_path, result)
        plot_path = output / "plots" / f"{key}.png"
        save_refinement_plot(plot_path, key.replace("_", " ").title(), result)
        cases.append(
            {
                "key": key,
                "completed": True,
                "optimizer_success": bool(result.get("success")),
                "message": str(result.get("message", "")),
                "wall_seconds": wall_seconds,
                "engine_elapsed_seconds": result.get("elapsed_seconds"),
                "rwp_percent": result.get("rwp_percent"),
                "rexp_percent": result.get("rexp_percent"),
                "goodness_of_fit_sqrt": result.get("goodness_of_fit_sqrt"),
                "reduced_chi_square": result.get("reduced_chi_square"),
                "r_squared": result.get("r_squared"),
                "phase_rows": summarize_phase_rows(result.get("phases", [])),
                "result_file": str(result_path.relative_to(ROOT)),
                "plot_file": str(plot_path.relative_to(ROOT)),
                "result_sha256": sha256_file(result_path),
                "plot_sha256": sha256_file(plot_path),
            }
        )
    return {"success": all(case["completed"] for case in cases), "cases": cases}


def dataset_and_cif_checks(output: Path) -> dict[str, Any]:
    import numpy as np

    from afruz_pxrd.crystallography import calculate_powder_pattern, load_cif
    from afruz_pxrd.io_engine import (
        export_dataset_excel,
        export_dataset_txt,
        load_patterns,
    )
    from afruz_pxrd.processing import smart_detect_peaks
    from afruz_pxrd.project import load_afz, save_afz

    synthetic = ROOT / "Afruz_PXRD_Synthetic_Test_Set"
    statistical = ROOT / "Afruz_PXRD_Statistical_Validation_Set"
    pattern_files = [
        synthetic / "01_NaCl_single_clean.txt",
        synthetic / "02_NaCl_CsCl_mixed_clean.txt",
        synthetic / "03_NaCl_CsCl_mixed_with_mismatches.txt",
        statistical / "01_NaCl_raw_counts_with_sigma.txt",
        statistical / "02_NaCl_normalized_100_with_sigma.txt",
        statistical / "04_NaCl_CsCl_mixed_raw_with_sigma.txt",
        statistical / "05_NaCl_CsCl_mixed_with_artifacts.txt",
    ]
    datasets = []
    loaded_objects = []
    for path in pattern_files:
        started = time.perf_counter()
        loaded = load_patterns(path)
        import_seconds = time.perf_counter() - started
        if len(loaded) != 1:
            raise RuntimeError(f"Expected one dataset from {path.name}; received {len(loaded)}")
        dataset = loaded[0]
        peak_started = time.perf_counter()
        peaks, diagnostics = smart_detect_peaks(dataset.x, dataset.y, sensitivity="Balanced")
        peak_seconds = time.perf_counter() - peak_started
        loaded_objects.append(dataset)
        datasets.append(
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "name": dataset.name,
                "points": len(dataset.x),
                "two_theta_min": float(dataset.x[0]),
                "two_theta_max": float(dataset.x[-1]),
                "intensity_min": float(np.min(dataset.y)),
                "intensity_max": float(np.max(dataset.y)),
                "import_seconds": import_seconds,
                "smart_peak_seconds": peak_seconds,
                "smart_peak_count": len(peaks),
                "noise_sigma": diagnostics.get("noise_sigma"),
            }
        )

    export_directory = output / "exports"
    export_directory.mkdir(parents=True, exist_ok=True)
    txt_export = export_directory / "nacl_raw_export.txt"
    xlsx_export = export_directory / "nacl_raw_export.xlsx"
    for generated in [
        txt_export,
        xlsx_export,
        *export_directory.glob("nacl_raw_export.backup_*"),
    ]:
        if generated.is_file():
            generated.unlink()
    export_dataset_txt(loaded_objects[3], txt_export)
    export_dataset_excel(loaded_objects[3], xlsx_export)

    project_path = output / "projects" / "phase0_supplied_patterns.afz"
    project_path.parent.mkdir(parents=True, exist_ok=True)
    save_afz(
        project_path,
        loaded_objects,
        {"baseline": True, "selected_dataset_uid": loaded_objects[0].uid},
        {"baseline_note": "Phase 0 supplied-pattern round-trip"},
    )
    reopened, ui_state, analysis_state = load_afz(project_path)
    project_roundtrip = {
        "success": len(reopened) == len(loaded_objects)
        and all(len(left.x) == len(right.x) for left, right in zip(reopened, loaded_objects)),
        "dataset_count": len(reopened),
        "ui_state": ui_state,
        "analysis_state": analysis_state,
        "file": str(project_path.relative_to(ROOT)),
        "sha256": sha256_file(project_path),
    }

    cif_files = [
        synthetic / "NaCl_ideal_P1.cif",
        synthetic / "CsCl_ideal_P1.cif",
        statistical / "NaCl_validation_P1.cif",
        statistical / "CsCl_validation_P1.cif",
    ]
    cif_rows = []
    for path in cif_files:
        started = time.perf_counter()
        structure = load_cif(path)
        pattern = calculate_powder_pattern(
            structure,
            WAVELENGTH_ANGSTROM,
            4.0,
            80.0,
            intensity_cutoff_percent=0.5,
        )
        cif_rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "formula": structure.get("formula"),
                "space_group": structure.get("space_group"),
                "crystal_system": structure.get("crystal_system"),
                "expanded_atom_count": len(structure.get("atoms", [])),
                "reflection_count": len(pattern),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )

    return {
        "success": len(datasets) == len(pattern_files)
        and all(row["points"] >= 3 for row in datasets)
        and all(row["reflection_count"] > 0 for row in cif_rows)
        and project_roundtrip["success"],
        "datasets": datasets,
        "cifs": cif_rows,
        "project_roundtrip": project_roundtrip,
        "exports": [
            {
                "file": str(txt_export.relative_to(ROOT)),
                "sha256": sha256_file(txt_export),
            },
            {
                "file": str(xlsx_export.relative_to(ROOT)),
                "sha256": sha256_file(xlsx_export),
            },
        ],
    }


def write_csv_summary(output: Path, report: dict[str, Any]) -> None:
    rows = []
    for row in report["data_checks"]["datasets"]:
        rows.append(
            {
                "category": "dataset",
                "name": row["name"],
                "points": row["points"],
                "runtime_seconds": row["import_seconds"] + row["smart_peak_seconds"],
                "metric": row["smart_peak_count"],
            }
        )
    for row in report["rietveld"]["cases"]:
        rows.append(
            {
                "category": "rietveld",
                "name": row["key"],
                "points": "",
                "runtime_seconds": row["wall_seconds"],
                "metric": row["rwp_percent"],
            }
        )
    path = output / "baseline_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("category", "name", "points", "runtime_seconds", "metric"),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output: Path, report: dict[str, Any]) -> None:
    checks = report["checks"]
    lines = [
        "# Afruz PXRD Phase 0 Baseline Report",
        "",
        f"- Overall status: **{'PASS' if report['success'] else 'FAIL'}**",
        f"- Captured UTC: `{report['captured_utc']}`",
        f"- Python: `{report['environment']['python_version']}`",
        f"- Platform: `{report['environment']['platform']}`",
        f"- Total capture runtime: `{report['total_elapsed_seconds']:.3f} s`",
        f"- Historical cached test IDs: `{report['historical_test_cache']['cached_nodeid_count']}` "
        f"(test sources present: `{report['historical_test_cache']['tests_directory_present']}`)",
        "",
        "## Checks",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for name, passed in checks.items():
        lines.append(f"| {name.replace('_', ' ').title()} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Supplied patterns",
            "",
            "| Pattern | Points | Range (°2θ) | Smart peaks | Import + peak time (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["data_checks"]["datasets"]:
        runtime = row["import_seconds"] + row["smart_peak_seconds"]
        lines.append(
            f"| {row['name']} | {row['points']} | {row['two_theta_min']:.3f}–{row['two_theta_max']:.3f} | "
            f"{row['smart_peak_count']} | {runtime:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Statistical reference",
            "",
            "| Case | Rwp (%) | Rexp (%) | GoF sqrt | Reduced χ² |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for case in report["statistical_reference"]["cases"]:
        actual = case["actual"]
        lines.append(
            f"| {case['name']} | {actual['rwp_percent']:.5f} | {actual['rexp_percent']:.5f} | "
            f"{actual['goodness_of_fit_sqrt']:.5f} | {actual['reduced_chi_square']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Rietveld examples",
            "",
            "| Case | Optimizer | Rwp (%) | Rexp (%) | Runtime (s) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for case in report["rietveld"]["cases"]:
        lines.append(
            f"| {case['key']} | {'success' if case['optimizer_success'] else 'completed with warning'} | "
            f"{case['rwp_percent']:.5f} | {case['rexp_percent']:.5f} | {case['wall_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "This report captures deterministic software behavior. The supplied datasets are not certified experimental reference materials, and this baseline does not establish publication readiness.",
            "",
        ]
    )
    markdown = "\n".join(lines)
    (output / "BASELINE_REPORT.md").write_text(markdown, encoding="utf-8")
    escaped = html.escape(markdown)
    (output / "BASELINE_REPORT.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Afruz PXRD Phase 0 Baseline</title>"
        "<style>body{font:15px/1.45 system-ui;max-width:1100px;margin:40px auto;padding:0 24px}"
        "pre{white-space:pre-wrap}</style><pre>" + escaped + "</pre>",
        encoding="utf-8",
    )
    write_csv_summary(output, report)


def capture(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    gui_process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--gui-only", "--output", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
        timeout=180,
    )
    gui_metrics_path = output / "gui_startup.json"
    gui_metrics = (
        json.loads(gui_metrics_path.read_text(encoding="utf-8"))
        if gui_metrics_path.exists()
        else {
            "success": False,
            "stdout": gui_process.stdout,
            "stderr": gui_process.stderr,
            "returncode": gui_process.returncode,
        }
    )

    environment = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": package_versions(),
    }
    source_manifest = scientific_source_manifest()
    write_json(output / "scientific_source_sha256.json", source_manifest)
    historical_test_cache = preserve_historical_test_cache(output)
    syntax = compile_sources()
    data_checks = dataset_and_cif_checks(output)
    statistics = statistical_reference_checks()
    rietveld = run_rietveld_cases(output)
    checks = {
        "gui_startup_and_screenshot": bool(gui_metrics.get("success")),
        "scientific_source_syntax": bool(syntax["success"]),
        "supplied_patterns_cifs_exports_project": bool(data_checks["success"]),
        "documented_statistical_reference": bool(statistics["success"]),
        "nacl_and_mixed_rietveld_examples": bool(rietveld["success"]),
    }
    report = {
        "schema_version": 1,
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "success": all(checks.values()),
        "checks": checks,
        "environment": environment,
        "gui": gui_metrics,
        "historical_test_cache": historical_test_cache,
        "syntax": syntax,
        "data_checks": data_checks,
        "statistical_reference": statistics,
        "rietveld": rietveld,
        "scientific_source_file_count": len(source_manifest),
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "baseline_report.json", report)
    write_reports(output, report)
    print(json.dumps({"success": report["success"], "checks": checks}, indent=2))
    return 0 if report["success"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture the Afruz PXRD Phase 0 baseline.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gui-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    destination = arguments.output.resolve()
    raise SystemExit(gui_only(destination) if arguments.gui_only else capture(destination))
