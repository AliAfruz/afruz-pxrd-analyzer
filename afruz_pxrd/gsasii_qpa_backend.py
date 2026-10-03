from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Callable

import numpy as np


class GSASIIBackendError(RuntimeError):
    pass


RUNNER_SOURCE = r'''
from __future__ import annotations
import json
from pathlib import Path
import sys
import traceback

request_path = Path(sys.argv[1])
request = json.loads(request_path.read_text(encoding="utf-8"))
output_path = Path(request["output_json"])
try:
    parent = request.get("gsasii_parent")
    if parent:
        sys.path.insert(0, parent)
    try:
        import G2script as G2sc
    except ImportError:
        from GSASII import GSASIIscriptable as G2sc

    try:
        G2sc.SetPrintLevel("warn")
    except Exception:
        pass

    project = G2sc.G2Project(newgpx=request["project_file"])
    histogram = project.add_powder_histogram(
        request["data_file"],
        request["instrument_file"],
        phases=[],
        fmthint=request.get("powder_format_hint", "xye"),
    )
    phases = []
    for phase_request in request["phases"]:
        phase = project.add_phase(
            phase_request["cif_file"],
            phasename=phase_request.get("name"),
            histograms=[histogram],
            fmthint="CIF",
        )
        phase.set_HAP_refinements({"Scale": True}, histograms=[histogram])
        phases.append((phase, phase_request))

    histogram.set_refinements({
        "Limits": [request["two_theta_min"], request["two_theta_max"]],
        "Background": {
            "type": "chebyschev-1",
            "no. coeffs": request["background_coefficients"],
            "refine": True,
        },
    })
    project.set_Controls("cycles", int(request.get("cycles", 6)))

    stage_results = []
    recipes = [
        {"name": "scale_background", "action": {}},
    ]
    if request.get("refine_zero", True):
        recipes.append({
            "name": "zero_shift",
            "action": {"set": {"Instrument Parameters": ["Zero"]}},
        })
    if request.get("refine_cells", True):
        recipes.append({
            "name": "unit_cells",
            "phase_action": {"Cell": True},
        })
    if request.get("refine_profile", True):
        recipes.append({
            "name": "profile",
            "action": {"set": {"Instrument Parameters": ["U", "V", "W", "X", "Y", "SH/L"]}},
        })
    if request.get("refine_preferred_orientation", False):
        recipes.append({"name": "preferred_orientation", "hap_action": {"Pref.Ori.": True}})

    for stage in recipes:
        if stage.get("phase_action"):
            for phase, phase_request in phases:
                if phase_request.get("refine_cell", True):
                    phase.set_refinements(stage["phase_action"])
        if stage.get("hap_action"):
            for phase, phase_request in phases:
                if phase_request.get("refine_preferred_orientation", False):
                    phase.set_HAP_refinements(stage["hap_action"], histograms=[histogram])
        action = stage.get("action") or {}
        if action:
            project.do_refinements([action])
        else:
            project.do_refinements([{}])
        stage_results.append({
            "stage": stage["name"],
            "residuals": histogram.residuals(),
            "frozen_variables": project.get_Frozen(),
        })

    mass_fractions = histogram.ComputeMassFracs()
    phase_rows = []
    for phase, phase_request in phases:
        cell = phase.get_cell()
        hap_scale = phase.HAPvalue("Scale", targethistlist=[histogram])
        phase_rows.append({
            "phase_name": phase.name,
            "mass_fraction": mass_fractions.get(phase.name, [None, None])[0],
            "mass_fraction_error": mass_fractions.get(phase.name, [None, None])[1],
            "scale_factor": hap_scale,
            "composition": phase.composition,
            "cell": cell,
            "source_cif": phase_request["cif_file"],
        })

    result = {
        "success": True,
        "engine": "GSAS-II GSASIIscriptable",
        "versions": G2sc.ShowVersions(),
        "project_file": request["project_file"],
        "histogram_name": histogram.name,
        "residuals": histogram.residuals(),
        "phases": phase_rows,
        "stages": stage_results,
        "mass_fraction_method": "GSAS-II ComputeMassFracs with covariance-derived uncertainties",
        "warnings": [],
    }
except Exception as exc:
    result = {
        "success": False,
        "engine": "GSAS-II GSASIIscriptable",
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
if not result["success"]:
    raise SystemExit(2)
'''


def _run_command(
    command: list[str],
    *,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
    )


def detect_gsasii(
    python_executable: str | Path | None = None,
    gsasii_parent: str | Path | None = None,
) -> dict:
    executable = str(python_executable or sys.executable)
    parent = None if not gsasii_parent else str(Path(gsasii_parent))
    code = (
        "import json,sys;"
        + (f"sys.path.insert(0,{parent!r});" if parent else "")
        + "\nlayout='package'\n"
        + "try:\n"
        + " from GSASII import GSASIIscriptable as G2sc\n"
        + " from GSASII import GSASIIindex as G2idx\n"
        + "except ImportError:\n"
        + " layout='flat'\n"
        + " try:\n"
        + "  import GSASIIscriptable as G2sc\n"
        + "  import GSASIIindex as G2idx\n"
        + " except ImportError:\n"
        + "  layout='legacy-shortcut'\n"
        + "  import G2script as G2sc\n"
        + "  import GSASIIindex as G2idx\n"
        + "print(json.dumps({"
        + "'available':True,"
        + "'layout':layout,"
        + "'scriptable_module':str(G2sc.__file__),"
        + "'index_module':str(G2idx.__file__),"
        + "'versions':str(G2sc.ShowVersions())"
        + "}))"
    )
    try:
        completed = _run_command([executable, "-c", code], timeout=30.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "python_executable": executable, "error": str(exc)}
    if completed.returncode != 0:
        return {
            "available": False,
            "python_executable": executable,
            "error": (completed.stderr or completed.stdout).strip(),
        }
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        result = {"available": False, "error": completed.stdout.strip()}
    result["python_executable"] = executable
    result["gsasii_parent"] = parent
    return result


def write_xye(path: str | Path, x: np.ndarray, y: np.ndarray) -> Path:
    path = Path(path)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise GSASIIBackendError("X and Y arrays must be one-dimensional and aligned.")
    sigma = np.sqrt(np.maximum(y - float(np.min(y)) + 1.0, 1.0))
    np.savetxt(path, np.column_stack([x, y, sigma]), fmt="%.10g")
    return path


def _materialize_cif(directory: Path, structure: dict, index: int) -> Path:
    source = structure.get("source_path")
    if source and Path(source).exists():
        return Path(source).resolve()
    raw = structure.get("raw_cif_text")
    if not raw:
        raise GSASIIBackendError(
            f"Phase {structure.get('data_name', index)} has no source CIF or retained CIF text."
        )
    path = directory / f"phase_{index:02d}.cif"
    path.write_text(str(raw), encoding="utf-8")
    return path


def build_request(
    work_directory: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    structures: list[dict],
    *,
    instrument_file: str | Path,
    two_theta_min: float,
    two_theta_max: float,
    background_coefficients: int = 6,
    cycles: int = 6,
    refine_zero: bool = True,
    refine_cells: bool = True,
    refine_profile: bool = True,
    refine_preferred_orientation: bool = False,
    gsasii_parent: str | Path | None = None,
) -> tuple[dict, Path]:
    directory = Path(work_directory)
    directory.mkdir(parents=True, exist_ok=True)
    instrument = Path(instrument_file)
    if not instrument.exists():
        raise GSASIIBackendError(f"GSAS-II instrument parameter file does not exist: {instrument}")
    data_file = write_xye(directory / "observed.xye", x, y)
    phase_requests = []
    for index, structure in enumerate(structures, start=1):
        cif = _materialize_cif(directory, structure, index)
        phase_requests.append({
            "name": structure.get("data_name") or f"Phase {index}",
            "cif_file": str(cif),
            "refine_cell": bool(refine_cells),
            "refine_preferred_orientation": bool(refine_preferred_orientation),
        })
    output_json = directory / "gsasii_qpa_result.json"
    request = {
        "data_file": str(data_file),
        "instrument_file": str(instrument.resolve()),
        "powder_format_hint": "xye",
        "project_file": str((directory / "validated_qpa.gpx").resolve()),
        "output_json": str(output_json.resolve()),
        "phases": phase_requests,
        "two_theta_min": float(two_theta_min),
        "two_theta_max": float(two_theta_max),
        "background_coefficients": int(background_coefficients),
        "cycles": int(cycles),
        "refine_zero": bool(refine_zero),
        "refine_cells": bool(refine_cells),
        "refine_profile": bool(refine_profile),
        "refine_preferred_orientation": bool(refine_preferred_orientation),
        "gsasii_parent": None if not gsasii_parent else str(Path(gsasii_parent)),
    }
    request_path = directory / "gsasii_qpa_request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    runner_path = directory / "run_gsasii_qpa.py"
    runner_path.write_text(RUNNER_SOURCE, encoding="utf-8")
    return request, runner_path


def run_gsasii_qpa(
    python_executable: str | Path,
    request: dict,
    runner_path: str | Path,
    *,
    progress_callback: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    timeout_seconds: float | None = None,
) -> dict:
    runner = Path(runner_path)
    request_path = runner.parent / "gsasii_qpa_request.json"
    if progress_callback:
        progress_callback("Launching GSAS-II refinement backend…")
    started = time.monotonic()
    timeout_seconds = (
        None
        if timeout_seconds is None
        else max(1.0, float(timeout_seconds))
    )
    process = subprocess.Popen(
        [str(python_executable), str(runner), str(request_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
    )
    while process.poll() is None:
        if (
            timeout_seconds is not None
            and time.monotonic() - started > timeout_seconds
        ):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise GSASIIBackendError(
                f"GSAS-II QPA exceeded the configured timeout "
                f"({timeout_seconds:g} s)."
            )
        if cancel_check and cancel_check():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise GSASIIBackendError("GSAS-II QPA was cancelled.")
        try:
            process.wait(timeout=0.15)
        except subprocess.TimeoutExpired:
            continue
    stdout, stderr = process.communicate()
    output_path = Path(request["output_json"])
    if not output_path.exists():
        raise GSASIIBackendError(
            "GSAS-II did not create a result file. " + (stderr or stdout).strip()
        )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["stdout"] = stdout
    result["stderr"] = stderr
    if not result.get("success"):
        raise GSASIIBackendError(result.get("error") or "GSAS-II refinement failed.")
    result["classification"] = (
        "Validated backend calculation — publication use still requires certified-mixture validation"
    )
    result["publication_ready"] = False
    result["validation_required"] = [
        "Certified or gravimetric mixture recovery",
        "Independent refinement review",
        "Specimen-preparation and microabsorption assessment",
        "Archived GSAS-II GPX, instrument file, CIFs, recipe and logs",
    ]
    return result
