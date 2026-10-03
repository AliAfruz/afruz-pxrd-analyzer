from __future__ import annotations

"""GUI-independent comparison and sequential-refinement engine for XRD series."""

from copy import deepcopy
from dataclasses import replace
import math
import re
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cif_library import cell_volume
from .processing import smart_detect_peaks
from .rietveld_refinement import RietveldPhaseSpec, refine_rietveld
from .structure_validation import (
    composition_formula_from_sites,
    ensure_site_formula_metadata,
    formula_weight_from_counts,
    parse_formula_counts,
    structure_refinement_gate,
    validate_crystal_structure,
)
from .text_export import (
    safe_filename,
    write_columns_txt,
    write_manifest_txt,
    write_mapping_txt,
    write_table_txt,
)


NORMALIZATION_MODES = (
    "Maximum intensity",
    "Integrated intensity",
    "Reference-window intensity",
    "None (raw intensity)",
)

REFINEMENT_MODES = (
    "Independent batch",
    "Sequential from reference",
)


class DopingSeriesError(ValueError):
    pass


class DopingSeriesCancelled(RuntimeError):
    pass


def _validate_records(records: Iterable[dict]) -> list[dict]:
    rows = [deepcopy(row) for row in records]
    if len(rows) < 2:
        raise DopingSeriesError("At least two included XRD datasets are required.")
    seen = set()
    for index, row in enumerate(rows):
        uid = str(row.get("dataset_uid") or "")
        if not uid or uid in seen:
            raise DopingSeriesError("Every series dataset must have a unique UID.")
        seen.add(uid)
        x = np.asarray(row.get("x"), dtype=float)
        y = np.asarray(row.get("y"), dtype=float)
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or len(x) < 20:
            raise DopingSeriesError(
                f"{row.get('dataset_name', index)} must contain aligned one-dimensional X/Y arrays."
            )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise DopingSeriesError(f"{row.get('dataset_name', index)} contains non-finite values.")
        if np.any(np.diff(x) <= 0):
            raise DopingSeriesError(f"{row.get('dataset_name', index)} has a non-increasing 2θ axis.")
        try:
            series_value = float(row.get("series_value"))
        except (TypeError, ValueError) as exc:
            raise DopingSeriesError(
                f"{row.get('dataset_name', index)} has no numeric dopant concentration."
            ) from exc
        if not np.isfinite(series_value):
            raise DopingSeriesError(f"{row.get('dataset_name', index)} has a non-finite series value.")
        row["dataset_uid"] = uid
        row["dataset_name"] = str(row.get("dataset_name") or uid)
        row["series_value"] = series_value
        row["x"] = x
        row["y"] = y
        refinement_y = np.asarray(row.get("refinement_y", y), dtype=float)
        if refinement_y.shape != y.shape or not np.all(np.isfinite(refinement_y)):
            raise DopingSeriesError(
                f"{row['dataset_name']} has an invalid original refinement-intensity array."
            )
        row["refinement_y"] = refinement_y
    return sorted(rows, key=lambda row: (row["series_value"], row["dataset_name"]))


def _positive_signal(y: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(y, dtype=float) - float(np.min(y)), 0.0)


def normalization_scale(
    x: np.ndarray,
    y: np.ndarray,
    mode: str,
    reference_window: tuple[float, float] | None = None,
) -> float:
    if mode not in NORMALIZATION_MODES:
        raise DopingSeriesError(f"Unsupported normalization mode: {mode}")
    if mode == "None (raw intensity)":
        return 1.0
    signal = _positive_signal(y)
    if mode == "Maximum intensity":
        scale = float(np.max(signal))
    elif mode == "Integrated intensity":
        scale = float(np.trapezoid(signal, x))
    else:
        if reference_window is None:
            raise DopingSeriesError("Reference-window normalization requires lower and upper 2θ limits.")
        lower, upper = map(float, reference_window)
        mask = (x >= lower) & (x <= upper)
        if np.count_nonzero(mask) < 3:
            raise DopingSeriesError("The normalization window contains fewer than three measured points.")
        scale = float(np.trapezoid(signal[mask], x[mask]))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise DopingSeriesError("A selected pattern has no positive intensity available for normalization.")
    return scale


def _common_grid(records: list[dict]) -> np.ndarray:
    lower = max(float(row["x"][0]) for row in records)
    upper = min(float(row["x"][-1]) for row in records)
    if upper <= lower:
        raise DopingSeriesError("The included patterns have no common 2θ interval.")
    step = max(float(np.median(np.diff(row["x"]))) for row in records)
    count = int(math.floor((upper - lower) / step)) + 1
    if count < 20:
        raise DopingSeriesError("The common angular interval contains fewer than twenty comparison points.")
    return lower + np.arange(count, dtype=float) * step


def _provided_peaks(row: dict) -> list[dict]:
    peaks = []
    for source in row.get("peaks", []) or []:
        if source.get("included") is False or source.get("use") is False:
            continue
        position = source.get("position", source.get("center"))
        if position is None:
            continue
        source_name = str(
            source.get("source")
            or source.get("origin")
            or source.get("method")
            or "Main curated/fitted peak list"
        )
        peaks.append(
            {
                "position_deg": float(position),
                "position_standard_error_deg": source.get(
                    "position_error", source.get("center_error")
                ),
                "intensity": float(source.get("intensity", source.get("height", 0.0))),
                "fwhm_deg": source.get("fwhm"),
                "fwhm_standard_error_deg": source.get("fwhm_error"),
                "prominence": source.get("prominence"),
                "snr": source.get("snr"),
                "confidence": source.get("confidence"),
                "confidence_label": source.get("confidence_label"),
                "source": source_name,
            }
        )
    return sorted(peaks, key=lambda item: item["position_deg"])


def _detected_peaks(row: dict, sensitivity: str) -> list[dict]:
    """Use the same noise-aware Smart Peak Search as the main Peaks workspace."""

    x = row["x"]
    y = row["y"]
    step = float(np.median(np.diff(x)))
    detected, _diagnostics = smart_detect_peaks(x, y, sensitivity=sensitivity)
    rows = []
    for peak in detected:
        rows.append(
            {
                "position_deg": float(peak["position"]),
                "position_standard_error_deg": float(step / math.sqrt(12.0)),
                "intensity": float(peak.get("intensity", 0.0)),
                "prominence": peak.get("prominence"),
                "fwhm_deg": peak.get("fwhm"),
                "fwhm_standard_error_deg": step,
                "snr": peak.get("snr"),
                "confidence": peak.get("confidence"),
                "confidence_label": peak.get("confidence_label"),
                "source": f"Main Smart peak search ({sensitivity}; automatic fallback)",
            }
        )
    return rows


def _match_peak_tracks(
    records: list[dict],
    peaks_by_uid: dict[str, list[dict]],
    tolerance_deg: float,
    reference_uid: str,
    wavelength_angstrom: float,
) -> tuple[list[dict], list[dict]]:
    tracks: list[dict] = []
    observations: list[dict] = []
    for series_index, record in enumerate(records):
        uid = record["dataset_uid"]
        peaks = peaks_by_uid[uid]
        active_tracks = [track for track in tracks if track["last_series_index"] == series_index - 1]
        assignments: dict[int, int] = {}
        if active_tracks and peaks:
            cost = np.asarray(
                [
                    [abs(float(track["last_position_deg"]) - float(peak["position_deg"])) for peak in peaks]
                    for track in active_tracks
                ],
                dtype=float,
            )
            left, right = linear_sum_assignment(cost)
            for track_index, peak_index in zip(left, right):
                if cost[track_index, peak_index] <= tolerance_deg:
                    assignments[int(peak_index)] = int(active_tracks[track_index]["track_id"])
        for peak_index, peak in enumerate(peaks):
            track_id = assignments.get(peak_index)
            if track_id is None:
                track_id = len(tracks) + 1
                tracks.append(
                    {
                        "track_id": track_id,
                        "first_series_index": series_index,
                        "last_series_index": series_index,
                        "last_position_deg": float(peak["position_deg"]),
                    }
                )
            track = tracks[track_id - 1]
            track["last_series_index"] = series_index
            track["last_position_deg"] = float(peak["position_deg"])
            theta = math.radians(float(peak["position_deg"]) / 2.0)
            d_spacing = (
                float(wavelength_angstrom / (2.0 * math.sin(theta)))
                if 0.0 < theta < math.pi / 2.0
                else None
            )
            observations.append(
                {
                    **{key: value for key, value in peak.items() if key != "prominence"},
                    "track_id": track_id,
                    "dataset_uid": uid,
                    "dataset_name": record["dataset_name"],
                    "series_value": record["series_value"],
                    "dopant": record.get("dopant", ""),
                    "series_unit": record.get("series_unit", ""),
                    "d_spacing_angstrom": d_spacing,
                }
            )
    reference = {
        row["track_id"]: row
        for row in observations
        if row["dataset_uid"] == reference_uid
    }
    for row in observations:
        baseline = reference.get(row["track_id"])
        row["position_shift_deg"] = (
            None if baseline is None else float(row["position_deg"] - baseline["position_deg"])
        )
        row["d_spacing_shift_angstrom"] = (
            None
            if baseline is None or row["d_spacing_angstrom"] is None
            else float(row["d_spacing_angstrom"] - baseline["d_spacing_angstrom"])
        )
        row["reference_classification"] = (
            "matched to reference" if baseline is not None else "emergent relative to reference"
        )
    summary = []
    for track in tracks:
        rows = [row for row in observations if row["track_id"] == track["track_id"]]
        summary.append(
            {
                "track_id": track["track_id"],
                "observation_count": len(rows),
                "first_series_value": min(row["series_value"] for row in rows),
                "last_series_value": max(row["series_value"] for row in rows),
                "reference_position_deg": (
                    None if track["track_id"] not in reference else reference[track["track_id"]]["position_deg"]
                ),
                "maximum_absolute_shift_deg": max(
                    (abs(row["position_shift_deg"]) for row in rows if row["position_shift_deg"] is not None),
                    default=None,
                ),
                "classification": (
                    "reference-linked" if track["track_id"] in reference else "emergent"
                ),
            }
        )
    return observations, summary


def compare_doping_series(
    records: Iterable[dict],
    *,
    reference_uid: str | None = None,
    normalization: str = NORMALIZATION_MODES[0],
    reference_window: tuple[float, float] | None = None,
    wavelength_angstrom: float = 1.5406,
    peak_match_tolerance_deg: float = 0.15,
    peak_prominence_percent: float = 2.0,
    peak_detection_sensitivity: str = "Balanced",
) -> dict:
    started = time.perf_counter()
    rows = _validate_records(records)
    reference_uid = str(reference_uid or rows[0]["dataset_uid"])
    if reference_uid not in {row["dataset_uid"] for row in rows}:
        raise DopingSeriesError("The selected reference pattern is not included in the series.")
    common_x = _common_grid(rows)
    profiles = []
    peaks_by_uid = {}
    reference_y = None
    for row in rows:
        scale = normalization_scale(
            row["x"], row["y"], normalization, reference_window
        )
        normalized = np.asarray(row["y"], dtype=float) / scale
        common_y = np.interp(common_x, row["x"], normalized)
        profile = {
            "dataset_uid": row["dataset_uid"],
            "dataset_name": row["dataset_name"],
            "series_value": row["series_value"],
            "dopant": str(row.get("dopant", "")),
            "series_unit": str(row.get("series_unit", "")),
            "normalization_scale": scale,
            "common_y": common_y.tolist(),
        }
        profiles.append(profile)
        if row["dataset_uid"] == reference_uid:
            reference_y = common_y
        provided_peaks = _provided_peaks(row)
        peaks = provided_peaks or _detected_peaks(
            row,
            peak_detection_sensitivity,
        )
        peaks_by_uid[row["dataset_uid"]] = peaks
        profile["comparison_signal"] = str(
            row.get("comparison_signal", "processed" if row.get("use_processed") else "raw")
        )
        profile["peak_count"] = len(peaks)
        profile["peak_source"] = (
            "; ".join(sorted({str(peak.get("source", "Unknown")) for peak in peaks}))
            if peaks
            else f"Main Smart peak search ({peak_detection_sensitivity}; no peaks found)"
        )
    assert reference_y is not None
    for profile in profiles:
        profile["difference_y"] = (
            np.asarray(profile["common_y"], dtype=float) - reference_y
        ).tolist()
    observations, track_summary = _match_peak_tracks(
        rows,
        peaks_by_uid,
        max(float(peak_match_tolerance_deg), 1e-6),
        reference_uid,
        wavelength_angstrom,
    )
    return {
        "schema_version": 1,
        "method": "Doping-series common-grid comparison and uncertainty-aware peak tracking",
        "reference_dataset_uid": reference_uid,
        "normalization": normalization,
        "reference_window_deg": None if reference_window is None else list(reference_window),
        "wavelength_angstrom": float(wavelength_angstrom),
        "peak_match_tolerance_deg": float(peak_match_tolerance_deg),
        "peak_detection_method": f"Main Smart peak search ({peak_detection_sensitivity}) fallback",
        "common_two_theta_deg": common_x.tolist(),
        "profiles": profiles,
        "heatmap_matrix": [profile["common_y"] for profile in profiles],
        "difference_matrix": [profile["difference_y"] for profile in profiles],
        "peak_observations": observations,
        "peak_tracks": track_summary,
        "dataset_count": len(rows),
        "peak_track_count": len(track_summary),
        "elapsed_seconds": float(time.perf_counter() - started),
        "scientific_boundary": (
            "Common-grid interpolation is used only for visualization and difference calculations. "
            "Main prepared signals may be used for comparison and peak tracking, while refinements use "
            "each original measured grid and original intensities. Peak shifts require calibrated geometry; "
            "intensity changes can also reflect texture, absorption, thickness, or sample preparation."
        ),
    }


def _execution_order(records: list[dict], reference_uid: str, mode: str) -> list[dict]:
    if mode == "Independent batch":
        return list(records)
    reference_index = next(index for index, row in enumerate(records) if row["dataset_uid"] == reference_uid)
    return [
        records[reference_index],
        *records[reference_index + 1 :],
        *reversed(records[:reference_index]),
    ]


def _carried_specs(
    phase_specs: list[RietveldPhaseSpec],
    previous_result: dict | None,
) -> list[RietveldPhaseSpec]:
    if not previous_result:
        return deepcopy(phase_specs)
    phases = previous_result.get("phases", [])
    carried = []
    for index, specification in enumerate(phase_specs):
        structure = deepcopy(specification.structure)
        if index < len(phases) and phases[index].get("refined_cell"):
            structure["cell"] = deepcopy(phases[index]["refined_cell"])
        initial_r = specification.initial_preferred_orientation_r
        if index < len(phases) and phases[index].get("march_dollase_r") is not None:
            initial_r = float(phases[index]["march_dollase_r"])
        carried.append(
            replace(
                specification,
                structure=structure,
                initial_preferred_orientation_r=initial_r,
            )
        )
    return carried


def _carried_settings(settings: dict, previous_result: dict | None) -> dict:
    current = deepcopy(settings)
    if not previous_result:
        return current
    profile = previous_result.get("profile", {})
    mapping = {
        "initial_u": "caglioti_u",
        "initial_v": "caglioti_v",
        "initial_w": "caglioti_w",
        "initial_eta": "eta",
        "initial_x": "lorentzian_x",
        "initial_y": "lorentzian_y",
        "initial_asymmetry": "axial_asymmetry",
    }
    for target, source in mapping.items():
        if profile.get(source) is not None:
            current[target] = float(profile[source])
    current["initial_zero_shift"] = float(previous_result.get("zero_shift_deg", 0.0))
    return current


def _quality_gate(result: dict, maximum_rwp_percent: float, maximum_correlation: float) -> dict:
    reasons = []
    if not result.get("success", False):
        reasons.append("optimizer did not report success")
    rwp = result.get("rwp_percent")
    if rwp is None or not np.isfinite(float(rwp)) or float(rwp) > maximum_rwp_percent:
        reasons.append(f"Rwp exceeds {maximum_rwp_percent:g}%")
    correlation = result.get("maximum_absolute_correlation")
    if correlation is not None and float(correlation) > maximum_correlation:
        reasons.append(f"maximum parameter correlation exceeds {maximum_correlation:g}")
    status = "accepted for sequential initialization" if not reasons else "review required"
    return {"passed": not reasons, "status": status, "reasons": reasons}


def _trend_rows(records: list[dict], results_by_uid: dict[str, dict]) -> list[dict]:
    rows = []
    for record in records:
        result = results_by_uid.get(record["dataset_uid"])
        base = {
            "dataset_uid": record["dataset_uid"],
            "dataset_name": record["dataset_name"],
            "dopant": record.get("dopant", ""),
            "series_value": record["series_value"],
            "series_unit": record.get("series_unit", ""),
        }
        if not result:
            rows.append({**base, "status": "failed"})
            continue
        phases = result.get("phases", [])
        primary = phases[0] if phases else {}
        cell = primary.get("refined_cell") or {}
        profile = result.get("profile", {})
        rows.append(
            {
                **base,
                "status": result.get("series_quality_gate", {}).get("status", "completed"),
                "rwp_percent": result.get("rwp_percent"),
                "reduced_chi_square": result.get("reduced_chi_square"),
                "zero_shift_deg": result.get("zero_shift_deg"),
                "cell_a_angstrom": cell.get("a"),
                "cell_b_angstrom": cell.get("b"),
                "cell_c_angstrom": cell.get("c"),
                "cell_alpha_deg": cell.get("alpha"),
                "cell_beta_deg": cell.get("beta"),
                "cell_gamma_deg": cell.get("gamma"),
                "cell_volume_angstrom3": cell_volume(cell) if cell else None,
                "primary_phase_fraction_percent": primary.get("pattern_scale_fraction_percent"),
                "caglioti_u": profile.get("caglioti_u"),
                "caglioti_v": profile.get("caglioti_v"),
                "caglioti_w": profile.get("caglioti_w"),
                "eta": profile.get("eta"),
                "maximum_absolute_correlation": result.get("maximum_absolute_correlation"),
                "elapsed_seconds": result.get("elapsed_seconds"),
            }
        )
    return rows


def refine_doping_series(
    records: Iterable[dict],
    phase_specs: list[RietveldPhaseSpec],
    settings: dict,
    *,
    mode: str = REFINEMENT_MODES[0],
    reference_uid: str | None = None,
    maximum_rwp_percent: float = 30.0,
    maximum_parameter_correlation: float = 0.98,
    continue_on_error: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    started = time.perf_counter()
    rows = _validate_records(records)
    if not phase_specs:
        raise DopingSeriesError("At least one included CIF phase is required for series refinement.")
    if mode not in REFINEMENT_MODES:
        raise DopingSeriesError(f"Unsupported series refinement mode: {mode}")
    reference_uid = str(reference_uid or rows[0]["dataset_uid"])
    if reference_uid not in {row["dataset_uid"] for row in rows}:
        raise DopingSeriesError("The reference dataset is not included in the refinement series.")
    execution = _execution_order(rows, reference_uid, mode)
    results_by_uid: dict[str, dict] = {}
    failures = []
    reference_value = next(
        row["series_value"] for row in rows if row["dataset_uid"] == reference_uid
    )
    reference_accepted = None
    previous_high = None
    previous_low = None
    total = len(execution)
    progress_units_per_dataset = 1000
    progress_total = total * progress_units_per_dataset
    for index, record in enumerate(execution, start=1):
        if cancel_check is not None and cancel_check():
            raise DopingSeriesCancelled("Doping-series refinement was cancelled.")
        dataset_progress_start = (index - 1) * progress_units_per_dataset
        if progress_callback is not None:
            progress_callback(
                dataset_progress_start,
                progress_total,
                f"Preparing {record['dataset_name']} ({index}/{total})",
            )
        previous = None
        if mode == "Sequential from reference":
            if record["series_value"] > reference_value:
                previous = previous_high or reference_accepted
            elif record["series_value"] < reference_value:
                previous = previous_low or reference_accepted
        use_previous = previous is not None
        local_specs = _carried_specs(phase_specs, previous)
        local_settings = _carried_settings(settings, previous)
        local_settings.pop("count_reference", None)
        refinement_y = np.asarray(record.get("refinement_y", record["y"]), dtype=float)
        refinement_provenance = str(
            record.get(
                "refinement_intensity_provenance",
                record.get("intensity_provenance", "unknown"),
            )
        )
        local_settings["intensity_provenance"] = refinement_provenance
        if refinement_provenance == "raw_counts" and np.all(refinement_y >= 0):
            local_settings["count_reference"] = refinement_y

        last_engine_progress_done = -1
        last_engine_progress_time = 0.0

        def report_engine_progress(done, engine_total, message):
            nonlocal last_engine_progress_done, last_engine_progress_time
            if progress_callback is None:
                return
            engine_total = max(1, int(engine_total))
            message = str(message)
            engine_completed = (
                int(done) >= engine_total
                and "refinement completed" in message.lower()
            )
            engine_fraction = max(0.0, min(float(done) / engine_total, 1.0))
            # Keep the final 5% of each dataset for covariance/result assembly.
            # The native engine emits an explicit completion event after that work.
            fraction = 1.0 if engine_completed else 0.95 * engine_fraction
            overall_done = dataset_progress_start + int(
                round(fraction * progress_units_per_dataset)
            )
            now = time.monotonic()
            if overall_done == last_engine_progress_done and not engine_completed:
                return
            if (
                not engine_completed
                and overall_done - last_engine_progress_done < 10
                and now - last_engine_progress_time < 0.25
            ):
                return
            last_engine_progress_done = overall_done
            last_engine_progress_time = now
            progress_callback(
                overall_done,
                progress_total,
                f"{record['dataset_name']} ({index}/{total}) — {message}",
            )

        dataset_failed = False
        try:
            result = refine_rietveld(
                record["x"],
                refinement_y,
                local_specs,
                progress_callback=report_engine_progress,
                cancel_check=cancel_check,
                **local_settings,
            )
            gate = _quality_gate(
                result,
                float(maximum_rwp_percent),
                float(maximum_parameter_correlation),
            )
            result["series_quality_gate"] = gate
            result["series_dataset_uid"] = record["dataset_uid"]
            result["series_dataset_name"] = record["dataset_name"]
            result["series_value"] = record["series_value"]
            result["series_initialization"] = (
                "previous accepted neighbouring refinement" if use_previous else "original CIF/profile settings"
            )
            results_by_uid[record["dataset_uid"]] = result
            if gate["passed"] and mode == "Sequential from reference":
                if record["series_value"] == reference_value:
                    reference_accepted = result
                elif record["series_value"] > reference_value:
                    previous_high = result
                else:
                    previous_low = result
        except DopingSeriesCancelled:
            raise
        except Exception as exc:
            dataset_failed = True
            failures.append(
                {
                    "dataset_uid": record["dataset_uid"],
                    "dataset_name": record["dataset_name"],
                    "series_value": record["series_value"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if not continue_on_error:
                raise
        if progress_callback is not None:
            progress_callback(
                index * progress_units_per_dataset,
                progress_total,
                (
                    f"Failed {record['dataset_name']}; continuing ({index}/{total})"
                    if dataset_failed
                    else f"Completed {record['dataset_name']} ({index}/{total})"
                ),
            )
    return {
        "schema_version": 1,
        "method": mode,
        "reference_dataset_uid": reference_uid,
        "execution_order_uids": [row["dataset_uid"] for row in execution],
        "results_by_uid": results_by_uid,
        "failures": failures,
        "trends": _trend_rows(rows, results_by_uid),
        "dataset_count": len(rows),
        "completed_count": len(results_by_uid),
        "failed_count": len(failures),
        "quality_gate": {
            "maximum_rwp_percent": float(maximum_rwp_percent),
            "maximum_parameter_correlation": float(maximum_parameter_correlation),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "scientific_boundary": (
            "Sequential carry-forward improves initialization but does not validate the CIF, dopant site, "
            "phase assemblage, or structural trend. Each pattern retains an independent quality gate."
        ),
    }


def _replace_cif_scalar(text: str, tag: str, value: float) -> str:
    pattern = re.compile(rf"(?im)^(\s*{re.escape(tag)}\s+)([^\s#]+)")
    replacement = rf"\g<1>{float(value):.10g}"
    return pattern.sub(replacement, text, count=1)


def _set_or_insert_cif_value(text: str, tag: str, value: str) -> str:
    pattern = re.compile(
        rf"(?im)^(\s*{re.escape(tag)}\s+)(?:'[^']*'|\"[^\"]*\"|[^\s#]+)"
    )
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{value}", text, count=1)
    lines = text.splitlines()
    insert_at = 1 if lines and lines[0].lstrip().lower().startswith("data_") else 0
    lines.insert(insert_at, f"{tag} {value}")
    return "\n".join(lines)


def _refined_structure_validation(structure: dict, phase_result: dict) -> tuple[dict, dict]:
    refined = deepcopy(structure)
    refined["cell"] = deepcopy(phase_result.get("refined_cell") or structure.get("cell") or {})
    ensure_site_formula_metadata(refined)
    validation = validate_crystal_structure(refined)
    return validation, structure_refinement_gate(validation)


def refined_cif_text(
    structure: dict,
    phase_result: dict,
    *,
    dataset_name: str,
    software_version: str,
) -> str:
    raw = str(structure.get("raw_cif_text") or "").strip()
    if not raw:
        raise DopingSeriesError("Refined CIF export requires the original CIF text.")
    cell = phase_result.get("refined_cell") or structure.get("cell") or {}
    tags = {
        "_cell_length_a": "a",
        "_cell_length_b": "b",
        "_cell_length_c": "c",
        "_cell_angle_alpha": "alpha",
        "_cell_angle_beta": "beta",
        "_cell_angle_gamma": "gamma",
    }
    for tag, name in tags.items():
        if cell.get(name) is not None:
            raw = _replace_cif_scalar(raw, tag, float(cell[name]))
    formula = str(structure.get("formula") or "").strip()
    formula_derived = not bool(parse_formula_counts(formula))
    if formula_derived:
        formula = composition_formula_from_sites(structure.get("atoms") or [])
    if formula:
        formula_weight = formula_weight_from_counts(parse_formula_counts(formula))
        raw = _set_or_insert_cif_value(raw, "_chemical_formula_sum", f"'{formula}'")
        raw = _set_or_insert_cif_value(raw, "_chemical_formula_weight", f"{formula_weight:.6f}")
        if not structure.get("formula_units_z"):
            raw = _set_or_insert_cif_value(raw, "_cell_formula_units_Z", "1")
    validation, gate = _refined_structure_validation(structure, phase_result)
    status = str(validation.get("status") or "Validation unavailable").replace("'", "''")
    readiness = "yes" if gate["publication_ready"] else "no"
    audit_values = {
        "_audit_creation_method": f"'Afruz PXRD Analyzer {software_version}; Phase 12 doping-series refinement'",
        "_audit_update_record": f"'{str(dataset_name).replace(chr(39), chr(39) * 2)}'",
        "_afruz_global_delta_biso": f"{float(phase_result.get('delta_biso', 0.0)):.10g}",
        "_afruz_pattern_scale_fraction_percent": f"{float(phase_result.get('pattern_scale_fraction_percent', 0.0)):.10g}",
        "_afruz_structure_plausibility_status": f"'{status}'",
        "_afruz_structure_plausibility_score": f"{float(gate['score']):.10g}",
        "_afruz_vdw_overlap_count": str(int(validation.get("vdw_overlap_count", len(validation.get("vdw_overlap_pairs") or [])))),
        "_afruz_hydrogen_bond_count": str(int(validation.get("hydrogen_bond_count", 0))),
        "_afruz_publication_ready": readiness,
        "_afruz_formula_source": f"'{'occupied atom-site cell composition' if formula_derived else 'input CIF'}'",
    }
    for tag, value in audit_values.items():
        raw = _set_or_insert_cif_value(raw, tag, value)
    return raw + (
        "\n# Afruz Phase 12 sequential-refinement audit\n"
        "# Atomic coordinates and occupancies were retained from the input CIF.\n"
        "# Topology-aware plausibility separates bonded 1-3 and D-H...A contacts from true nonbonded overlaps.\n"
        "# A publication_ready=no result is a preserved review model, not a validated crystal structure.\n"
    )


def export_doping_series_bundle(
    directory: str | Path,
    result: dict,
    phase_specs: list[RietveldPhaseSpec],
    *,
    software_version: str,
) -> dict:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    comparison = result.get("comparison", {})
    refinement = result.get("refinement", {})
    files: list[Path] = []
    summary = write_mapping_txt(
        output / "phase12_series_summary.txt",
        {
            "software_version": software_version,
            "comparison": comparison,
            "refinement_summary": {
                key: value
                for key, value in refinement.items()
                if key not in {"results_by_uid"}
            },
        },
        title="Phase 12 doping-series comparison and sequential refinement",
    )
    files.append(Path(summary["txt_path"]))
    peak_export = write_table_txt(
        output / "phase12_peak_tracks.txt",
        comparison.get("peak_observations", []),
        title="Phase 12 matched peak evolution",
    )
    files.append(Path(peak_export["txt_path"]))
    trend_export = write_table_txt(
        output / "phase12_refinement_trends.txt",
        refinement.get("trends", []),
        title="Phase 12 refinement trends",
    )
    files.append(Path(trend_export["txt_path"]))
    common_x = comparison.get("common_two_theta_deg", [])
    profiles = comparison.get("profiles", [])
    profile_columns = {"two_theta_deg": common_x}
    difference_columns = {"two_theta_deg": common_x}
    for profile in profiles:
        key = safe_filename(
            f"{profile.get('series_value')}_{profile.get('dataset_name')}"
        )
        profile_columns[key] = profile.get("common_y", [])
        difference_columns[key] = profile.get("difference_y", [])
    profile_export = write_columns_txt(
        output / "phase12_common_grid_profiles.txt",
        profile_columns,
        title="Normalized common-grid profiles (visualization only)",
    )
    difference_export = write_columns_txt(
        output / "phase12_difference_profiles.txt",
        difference_columns,
        title="Difference profiles relative to the selected reference",
    )
    files.extend([Path(profile_export["txt_path"]), Path(difference_export["txt_path"])])
    refined_root = output / "refined_cifs"
    refined_root.mkdir(exist_ok=True)
    for uid, pattern_result in refinement.get("results_by_uid", {}).items():
        dataset_name = str(pattern_result.get("series_dataset_name", uid))
        for phase_index, phase_result in enumerate(pattern_result.get("phases", [])):
            if phase_index >= len(phase_specs):
                continue
            try:
                validation, publication_gate = _refined_structure_validation(
                    phase_specs[phase_index].structure,
                    phase_result,
                )
                cif_text = refined_cif_text(
                    phase_specs[phase_index].structure,
                    phase_result,
                    dataset_name=dataset_name,
                    software_version=software_version,
                )
            except DopingSeriesError:
                continue
            review_suffix = "_REVIEW_REQUIRED" if not publication_gate["publication_ready"] else ""
            cif_path = refined_root / (
                safe_filename(f"{pattern_result.get('series_value')}_{dataset_name}_phase{phase_index + 1}{review_suffix}")
                + ".cif"
            )
            cif_path.write_text(cif_text, encoding="utf-8", newline="\n")
            files.append(cif_path)
        profile = write_columns_txt(
            output / (safe_filename(f"refinement_{pattern_result.get('series_value')}_{dataset_name}") + ".txt"),
            {
                "two_theta_deg": pattern_result.get("observed_x", []),
                "observed_intensity": pattern_result.get("observed_y", []),
                "calculated_intensity": pattern_result.get("calculated_y", []),
                "background_intensity": pattern_result.get("background_y", []),
                "difference_observed_minus_calculated": pattern_result.get("difference_y", []),
            },
            title=f"Phase 12 Rietveld profile — {dataset_name}",
        )
        files.append(Path(profile["txt_path"]))
    manifest = write_manifest_txt(
        output / "phase12_manifest.txt",
        files,
        root=output,
        title="Phase 12 reproducibility manifest",
    )
    return {
        "directory": str(output.resolve()),
        "manifest_txt": manifest["txt_path"],
        "file_count": len(files) + 1,
        "files": [str(path.resolve()) for path in files],
    }
