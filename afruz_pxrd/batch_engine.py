from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
import time
from typing import Callable, Iterable

import numpy as np

from .batch_recipe import recipe_fingerprint, validate_recipe, utc_now_text
from .crystallography import match_observed_to_reference, refine_unit_cell
from .fitting import fit_detected_peaks
from .models import Dataset
from .phase_identification import identify_phases, ReferenceEntry
from .processing import (
    ProcessingParameters,
    detect_peaks,
    process_pattern,
    smart_detect_peaks,
)
from .quantitative_phase import QPAPhaseSpec, quantify_phases
from .residual_stress import (
    ResidualStressError,
    analyze_sin2psi,
    extract_peak_observation,
    infer_psi_deg,
)
from .size_strain import analyze_size_strain
from .whole_pattern_refinement import (
    WholePatternPhaseSpec,
    refine_whole_pattern,
)
from .refinement_statistics import infer_dataset_statistical_input
from .rietveld_refinement import (
    RietveldPhaseSpec,
    refine_rietveld,
)


BATCH_STAGE_ORDER = (
    "preprocessing",
    "peak_detection",
    "peak_fitting",
    "size_strain",
    "cif_cell",
    "phase_identification",
    "whole_pattern",
    "rietveld",
    "qpa",
)


class BatchCancelled(RuntimeError):
    pass


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _downsample(x: np.ndarray, y: np.ndarray, maximum: int = 2000) -> tuple[list, list]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) <= maximum:
        return x.tolist(), y.tolist()
    indices = np.linspace(0, len(x) - 1, maximum, dtype=int)
    return x[indices].tolist(), y[indices].tolist()


def _experimental_dataset(dataset: Dataset) -> bool:
    metadata = dataset.metadata or {}
    return not (
        metadata.get("analysis_role") == "reference_pattern"
        or metadata.get("plot_style") == "sticks"
    )


def _observed_peaks(
    peaks: list[dict],
    fit_groups: list[dict],
    preference: str,
) -> list[dict]:
    fitted = []
    for group in fit_groups:
        for component in group.get("components", []):
            fitted.append(
                {
                    "position": float(component["center"]),
                    "intensity": float(component.get("amplitude", 0.0)),
                    "position_error": component.get("center_error"),
                    "source": "Fitted",
                }
            )
    detected = [
        {
            "position": float(row["position"]),
            "intensity": float(row.get("intensity", 0.0)),
            "position_error": None,
            "source": row.get("method", "Detected"),
        }
        for row in peaks
    ]
    if preference.startswith("Fitted"):
        return sorted(fitted, key=lambda row: row["position"])
    if preference.startswith("Detected"):
        return sorted(detected, key=lambda row: row["position"])
    return sorted(fitted or detected, key=lambda row: row["position"])


def _primary_peak_metrics(peaks: list[dict], fit_groups: list[dict]) -> dict:
    fitted_components = [
        component
        for group in fit_groups
        for component in group.get("components", [])
    ]
    if fitted_components:
        primary = max(
            fitted_components,
            key=lambda row: float(row.get("amplitude", 0.0)),
        )
        return {
            "primary_peak_position_deg": float(primary["center"]),
            "primary_peak_intensity": float(primary.get("amplitude", 0.0)),
            "primary_peak_fwhm_deg": float(primary.get("fwhm", 0.0)),
            "primary_peak_source": "Fitted",
        }
    if peaks:
        primary = max(peaks, key=lambda row: float(row.get("intensity", 0.0)))
        return {
            "primary_peak_position_deg": float(primary["position"]),
            "primary_peak_intensity": float(primary.get("intensity", 0.0)),
            "primary_peak_fwhm_deg": float(primary.get("fwhm", 0.0)),
            "primary_peak_source": str(primary.get("method", "Detected")),
        }
    return {
        "primary_peak_position_deg": None,
        "primary_peak_intensity": None,
        "primary_peak_fwhm_deg": None,
        "primary_peak_source": None,
    }


def _phase_references_from_candidate(
    candidate: dict,
    references_by_uid: dict[str, ReferenceEntry],
) -> list[QPAPhaseSpec]:
    uids = list(candidate.get("reference_uids", []))
    shifts_by_uid = {}
    if candidate.get("kind") == "single":
        if uids:
            shifts_by_uid[uids[0]] = float(candidate.get("zero_shift_deg", 0.0))
    else:
        shift_values = candidate.get("zero_shift_deg", [])
        if not isinstance(shift_values, list):
            shift_values = [shift_values] * len(uids)
        for uid, shift in zip(uids, shift_values):
            shifts_by_uid[uid] = float(shift)

    specs = []
    for uid in uids:
        reference = references_by_uid.get(uid)
        if reference is None:
            continue
        rir = reference.metadata.get("rir")
        try:
            rir_override = None if rir is None else float(rir)
        except (TypeError, ValueError):
            rir_override = None
        specs.append(
            QPAPhaseSpec(
                reference=reference,
                zero_shift_deg=float(shifts_by_uid.get(uid, 0.0)),
                rir_override=rir_override,
            )
        )
    return specs


def _result_template(dataset: Dataset, fingerprint: str) -> dict:
    return {
        "dataset_uid": dataset.uid,
        "dataset_name": dataset.name,
        "source_path": dataset.source_path,
        "status": "Queued",
        "started_utc": None,
        "finished_utc": None,
        "elapsed_seconds": 0.0,
        "recipe_fingerprint": fingerprint,
        "recipe_deviation": False,
        "stages": {},
        "metrics": {},
        "warnings": [],
        "errors": [],
        "peak_rows": [],
        "fit_groups": [],
        "size_strain": None,
        "cell_refinement": None,
        "phase_identification": None,
        "whole_pattern": None,
        "rietveld": None,
        "qpa": None,
        "plot_data": {},
        "metadata": deepcopy(dataset.metadata),
    }


def analyze_batch(
    datasets: list[Dataset],
    recipe: dict,
    references: list[ReferenceEntry] | None = None,
    reference_pattern: list[dict] | None = None,
    reference_structure: dict | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    recipe = validate_recipe(recipe)
    fingerprint = recipe_fingerprint(recipe)
    references = list(references or [])
    references_by_uid = {entry.uid: entry for entry in references}
    experimental = [dataset for dataset in datasets if _experimental_dataset(dataset)]

    enabled_stages = [
        stage for stage in BATCH_STAGE_ORDER
        if recipe["stages"].get(stage, False)
    ]
    total_steps = max(1, len(experimental) * max(1, len(enabled_stages)))
    completed_steps = 0

    def cancelled() -> bool:
        return bool(cancel_check is not None and cancel_check())

    def emit(message: str, advance: bool = False):
        nonlocal completed_steps
        if advance:
            completed_steps += 1
        if progress_callback is not None:
            progress_callback(
                int(completed_steps),
                int(max(total_steps, completed_steps, 1)),
                message,
            )

    batch_started = time.perf_counter()
    result = {
        "batch_version": 1,
        "created_utc": utc_now_text(),
        "recipe": deepcopy(recipe),
        "recipe_fingerprint": fingerprint,
        "dataset_count": len(experimental),
        "completed_count": 0,
        "failed_count": 0,
        "cancelled": False,
        "results": [],
        "residual_stress": None,
        "errors": [],
        "elapsed_seconds": 0.0,
    }

    for dataset_index, source_dataset in enumerate(experimental, start=1):
        if cancelled():
            result["cancelled"] = True
            break

        dataset = source_dataset.clone()
        dataset.name = source_dataset.name
        dataset.uid = source_dataset.uid
        dataset.metadata = deepcopy(source_dataset.metadata)
        dataset_result = _result_template(source_dataset, fingerprint)
        dataset_result["started_utc"] = utc_now_text()
        dataset_result["status"] = "Running"
        started = time.perf_counter()

        peaks: list[dict] = []
        fit_groups: list[dict] = []
        y_for_analysis = np.asarray(dataset.y_raw, dtype=float).copy()
        preprocessing_aux: dict = {}

        try:
            for stage in enabled_stages:
                if cancelled():
                    raise BatchCancelled("Batch processing was cancelled.")
                label = stage.replace("_", " ").title()
                emit(
                    f"{dataset_index}/{len(experimental)} — {dataset.name}: {label}"
                )
                stage_started = time.perf_counter()
                stage_record = {
                    "status": "Running",
                    "elapsed_seconds": 0.0,
                    "message": "",
                }
                dataset_result["stages"][stage] = stage_record

                try:
                    if stage == "preprocessing":
                        settings = recipe["preprocessing"]
                        params = ProcessingParameters(
                            polynomial_order=int(settings["polynomial_order"]),
                            smoothing_window=int(settings["smoothing_window"]),
                            smoothing_order=int(settings["smoothing_order"]),
                            smoothing_method=str(settings["smoothing_method"]),
                            smoothing_strength=float(settings["smoothing_strength"]),
                            smoothing_peak_protection=bool(settings["smoothing_peak_protection"]),
                            smoothing_peak_preservation=float(settings["smoothing_peak_preservation"]),
                            smoothing_maximum_position_shift_deg=float(settings["smoothing_maximum_position_shift_deg"]),
                            smoothing_maximum_height_change_percent=float(settings["smoothing_maximum_height_change_percent"]),
                            smoothing_maximum_fwhm_change_percent=float(settings["smoothing_maximum_fwhm_change_percent"]),
                            normalize=bool(settings["normalize"]),
                            subtract_background=bool(settings["subtract_background"]),
                            smooth=bool(settings["smooth"]),
                            background_method=str(settings["background_method"]),
                            background_smoothness=float(settings["background_smoothness"]),
                            background_asymmetry=float(settings["background_asymmetry"]),
                            background_iterations=int(settings["background_iterations"]),
                            background_window_degrees=float(settings["background_window_degrees"]),
                            background_percentile=float(settings["background_percentile"]),
                            background_peak_protection=bool(settings["background_peak_protection"]),
                            clip_negative=bool(settings["clip_negative"]),
                        )
                        y_for_analysis, aux = process_pattern(
                            dataset.x,
                            dataset.y_raw,
                            params,
                        )
                        preprocessing_aux = dict(aux)
                        dataset.y_processed = y_for_analysis
                        diagnostics = {}
                        if "background" in aux:
                            bx, by = _downsample(
                                dataset.x,
                                aux.get("background_display", aux["background"]),
                            )
                            dataset_result["plot_data"]["background_x"] = bx
                            dataset_result["plot_data"]["background_y"] = by
                            diagnostics["background"] = _json_safe(
                                aux.get("background_diagnostics", {})
                            )
                        if "smoothed" in aux:
                            sx, sy = _downsample(dataset.x, aux["smoothed"])
                            dataset_result["plot_data"]["smoothed_x"] = sx
                            dataset_result["plot_data"]["smoothed_y"] = sy
                            diagnostics["smoothing"] = _json_safe(
                                aux.get("smoothing_diagnostics", {})
                            )
                        if diagnostics:
                            stage_record["diagnostics"] = diagnostics
                        stage_record["message"] = "Shared advanced preprocessing recipe applied."

                    elif stage == "peak_detection":
                        settings = recipe["peak_detection"]
                        if settings["mode"] == "Regular":
                            peaks = detect_peaks(
                                dataset.x,
                                y_for_analysis,
                                prominence_fraction=float(
                                    settings["prominence_fraction"]
                                ),
                                minimum_distance_points=int(
                                    settings["minimum_distance_points"]
                                ),
                            )
                        else:
                            peaks, diagnostics = smart_detect_peaks(
                                dataset.x,
                                y_for_analysis,
                                sensitivity=str(settings["smart_sensitivity"]),
                            )
                            stage_record["diagnostics"] = _json_safe(diagnostics)
                        stage_record["message"] = f"{len(peaks)} peak(s) detected."

                    elif stage == "peak_fitting":
                        if not peaks:
                            peaks, _ = smart_detect_peaks(
                                dataset.x,
                                y_for_analysis,
                                sensitivity=recipe["peak_detection"][
                                    "smart_sensitivity"
                                ],
                            )
                        settings = recipe["peak_fitting"]
                        fit_groups, diagnostics = fit_detected_peaks(
                            dataset.x,
                            y_for_analysis,
                            peaks,
                            model=str(settings["model"]),
                            window_multiplier=float(
                                settings["window_multiplier"]
                            ),
                        )
                        stage_record["diagnostics"] = _json_safe(diagnostics)
                        stage_record["message"] = (
                            f"{diagnostics['components']} fitted component(s)."
                        )

                    elif stage == "size_strain":
                        if not fit_groups:
                            raise ValueError(
                                "Size/strain requires successful fitted peak profiles."
                            )
                        settings = recipe["size_strain"]
                        size_result = analyze_size_strain(
                            fit_groups,
                            wavelength_angstrom=float(
                                settings["wavelength_angstrom"]
                            ),
                            shape_factor=float(settings["shape_factor"]),
                            instrument_fwhm_deg=float(
                                settings["instrument_fwhm_deg"]
                            ),
                            correction_mode=str(settings["correction_mode"]),
                            instrument_profile=(
                                recipe.get("instrument_calibration", {}).get("profile")
                                if recipe.get("instrument_calibration", {}).get("enabled")
                                else None
                            ),
                        )
                        dataset_result["size_strain"] = _json_safe(size_result)
                        stage_record["message"] = (
                            f"{size_result.get('scherrer_summary', {}).get('valid_peak_count', 0)} "
                            "valid peak(s)."
                        )

                    elif stage == "cif_cell":
                        if not reference_pattern or not reference_structure:
                            raise ValueError(
                                "CIF/cell batch stage requires an active CIF reference."
                            )
                        observed = _observed_peaks(
                            peaks,
                            fit_groups,
                            "Auto",
                        )
                        settings = recipe["cif_cell"]
                        matches = match_observed_to_reference(
                            observed,
                            reference_pattern,
                            tolerance_deg=float(
                                settings["match_tolerance_deg"]
                            ),
                        )
                        cell_result = refine_unit_cell(
                            matches,
                            initial_cell=reference_structure["cell"],
                            crystal_system=str(settings["crystal_system"]),
                            wavelength_angstrom=float(
                                recipe["size_strain"]["wavelength_angstrom"]
                            ),
                            refine_zero_shift=bool(
                                settings["refine_zero_shift"]
                            ),
                        )
                        dataset_result["cell_refinement"] = _json_safe(cell_result)
                        stage_record["message"] = (
                            f"{cell_result['match_count']} reflection(s) refined."
                        )

                    elif stage == "phase_identification":
                        if not references:
                            raise ValueError(
                                "Phase identification requires reference-card or CIF entries."
                            )
                        settings = recipe["phase_identification"]
                        observed = _observed_peaks(
                            peaks,
                            fit_groups,
                            str(settings["peak_source"]),
                        )
                        if len(observed) < 2:
                            raise ValueError(
                                "At least two observed peaks are required."
                            )
                        phase_result = identify_phases(
                            observed,
                            references,
                            target_wavelength_angstrom=float(
                                recipe["size_strain"]["wavelength_angstrom"]
                            ),
                            tolerance_deg=float(settings["tolerance_deg"]),
                            maximum_zero_shift_deg=float(
                                settings["maximum_zero_shift_deg"]
                            ),
                            reference_intensity_cutoff_percent=float(
                                settings[
                                    "reference_intensity_cutoff_percent"
                                ]
                            ),
                            convert_from_d=bool(settings["convert_from_d"]),
                            maximum_phases=int(settings["maximum_phases"]),
                            mixture_pool_size=int(settings["mixture_pool_size"]),
                        )
                        dataset_result["phase_identification"] = _json_safe(
                            phase_result
                        )
                        best = phase_result.get("best") or {}
                        stage_record["message"] = (
                            f"Best candidate: {best.get('reference_name', best.get('display_name', '—'))}"
                        )

                    elif stage == "whole_pattern":
                        if not references:
                            raise ValueError(
                                "Whole-pattern refinement requires phase references."
                            )
                        settings = recipe["whole_pattern"]
                        requested_uids = [
                            str(uid) for uid in settings.get("phase_uids", [])
                        ]
                        if not requested_uids:
                            phase_result = (
                                dataset_result.get("phase_identification") or {}
                            )
                            best = phase_result.get("best") or {}
                            requested_uids = list(
                                best.get("reference_uids", [])
                            )
                            single_uid = best.get("reference_uid")
                            if single_uid and single_uid not in requested_uids:
                                requested_uids.append(single_uid)
                        selected_references = [
                            reference
                            for reference in references
                            if reference.uid in requested_uids
                        ]
                        if not selected_references:
                            raise ValueError(
                                "No whole-pattern phase was selected or resolved."
                            )
                        refine_cell_uids = {
                            str(uid)
                            for uid in settings.get("refine_cell_uids", [])
                        }
                        phase_specs = [
                            WholePatternPhaseSpec(
                                reference=reference,
                                refine_cell=(
                                    reference.uid in refine_cell_uids
                                    if refine_cell_uids
                                    else True
                                ),
                            )
                            for reference in selected_references
                        ]
                        whole_observed = (
                            y_for_analysis
                            if settings["use_processed_pattern"]
                            else dataset.y_raw
                        )
                        whole_statistics = infer_dataset_statistical_input(
                            dataset.y_raw,
                            whole_observed,
                            intensity_unit=str(dataset.metadata.get("intensity_unit", "")),
                            counting_time_s=dataset.metadata.get("counting_time_s"),
                            interpretation="Auto from input metadata",
                            known_processing_scale=preprocessing_aux.get("normalization_factor"),
                        )
                        whole_result = refine_whole_pattern(
                            dataset.x,
                            whole_observed,
                            phase_specs,
                            mode=str(settings["mode"]),
                            wavelength_angstrom=float(
                                settings["wavelength_angstrom"]
                            ),
                            two_theta_min=settings.get("two_theta_min_deg"),
                            two_theta_max=settings.get("two_theta_max_deg"),
                            intensity_cutoff_percent=float(
                                settings[
                                    "reference_intensity_cutoff_percent"
                                ]
                            ),
                            background_order=int(settings["background_order"]),
                            weighting=str(settings["weighting"]),
                            refine_zero_shift=bool(
                                settings["refine_zero_shift"]
                            ),
                            refine_profile=bool(settings["refine_profile"]),
                            refine_eta=bool(settings["refine_eta"]),
                            initial_u=float(settings["initial_u"]),
                            initial_v=float(settings["initial_v"]),
                            initial_w=float(settings["initial_w"]),
                            initial_eta=float(settings["initial_eta"]),
                            cell_tolerance_percent=float(
                                settings["cell_tolerance_percent"]
                            ),
                            extraction_cycles=int(
                                settings["extraction_cycles"]
                            ),
                            maximum_nonlinear_evaluations=int(
                                settings["maximum_nonlinear_evaluations"]
                            ),
                            maximum_optimization_points=int(
                                settings["maximum_optimization_points"]
                            ),
                            count_reference=whole_statistics.get("count_reference"),
                            intensity_scale_factor=float(whole_statistics.get("intensity_scale_factor", 1.0)),
                            intensity_provenance=str(whole_statistics.get("intensity_provenance", "unknown")),
                            statistics_note=str(whole_statistics.get("statistics_note", "")),
                            use_gpu=bool(settings.get("use_gpu", True)),
                            cancel_check=cancelled,
                        )
                        dataset_result["whole_pattern"] = _json_safe(
                            whole_result
                        )
                        stage_record["message"] = (
                            f"Rwp {whole_result['rwp_percent']:.4g}%; "
                            f"{whole_result['reflection_count']} reflections."
                        )

                    elif stage == "rietveld":
                        if not reference_structure:
                            raise ValueError(
                                "Rietveld batch stage requires an active CIF structure with atoms."
                            )
                        settings = recipe["rietveld"]
                        rietveld_observed = (
                            y_for_analysis
                            if settings["use_processed_pattern"]
                            else dataset.y_raw
                        )
                        rietveld_statistics = infer_dataset_statistical_input(
                            dataset.y_raw,
                            rietveld_observed,
                            intensity_unit=str(dataset.metadata.get("intensity_unit", "")),
                            counting_time_s=dataset.metadata.get("counting_time_s"),
                            interpretation="Auto from input metadata",
                            known_processing_scale=preprocessing_aux.get("normalization_factor"),
                        )
                        rietveld_result = refine_rietveld(
                            dataset.x,
                            rietveld_observed,
                            [
                                RietveldPhaseSpec(
                                    structure=deepcopy(reference_structure),
                                    name=reference_structure.get("data_name"),
                                    refine_cell=True,
                                )
                            ],
                            wavelength_angstrom=float(settings["wavelength_angstrom"]),
                            two_theta_min=settings.get("two_theta_min_deg"),
                            two_theta_max=settings.get("two_theta_max_deg"),
                            intensity_cutoff_percent=float(settings["intensity_cutoff_percent"]),
                            background_order=int(settings["background_order"]),
                            weighting=str(settings["weighting"]),
                            robust_loss=str(settings["robust_loss"]),
                            refine_zero_shift=bool(settings["refine_zero_shift"]),
                            refine_profile=bool(settings["refine_profile"]),
                            refine_eta=bool(settings["refine_eta"]),
                            initial_u=float(settings["initial_u"]),
                            initial_v=float(settings["initial_v"]),
                            initial_w=float(settings["initial_w"]),
                            initial_eta=float(settings["initial_eta"]),
                            cell_tolerance_percent=float(settings["cell_tolerance_percent"]),
                            k_alpha2_enabled=bool(settings["k_alpha2_enabled"]),
                            k_alpha2_wavelength_angstrom=float(settings["k_alpha2_wavelength_angstrom"]),
                            k_alpha2_ratio=float(settings["k_alpha2_ratio"]),
                            maximum_nonlinear_evaluations=int(settings["maximum_nonlinear_evaluations"]),
                            maximum_optimization_points=int(settings["maximum_optimization_points"]),
                            count_reference=rietveld_statistics.get("count_reference"),
                            intensity_scale_factor=float(rietveld_statistics.get("intensity_scale_factor", 1.0)),
                            intensity_provenance=str(rietveld_statistics.get("intensity_provenance", "unknown")),
                            statistics_note=str(rietveld_statistics.get("statistics_note", "")),
                            use_gpu=bool(settings.get("use_gpu", True)),
                        )
                        dataset_result["rietveld"] = _json_safe(rietveld_result)
                        stage_record["message"] = (
                            f"Rwp {rietveld_result['rwp_percent']:.4g}%; "
                            f"{rietveld_result['phase_count']} structure phase(s)."
                        )

                    elif stage == "qpa":
                        phase_result = dataset_result.get("phase_identification")
                        best = (
                            None if not phase_result
                            else phase_result.get("best")
                        )
                        if not best:
                            raise ValueError(
                                "QPA requires a successful Phase 5 candidate."
                            )
                        specs = _phase_references_from_candidate(
                            best,
                            references_by_uid,
                        )
                        if not specs:
                            raise ValueError(
                                "No candidate references could be resolved for QPA."
                            )
                        settings = recipe["qpa"]
                        qpa_result = quantify_phases(
                            dataset.x,
                            (
                                y_for_analysis
                                if settings["use_processed_pattern"]
                                else dataset.y_raw
                            ),
                            specs,
                            target_wavelength_angstrom=float(
                                recipe["size_strain"]["wavelength_angstrom"]
                            ),
                            mode=str(settings["mode"]),
                            convert_from_d=bool(settings["convert_from_d"]),
                            reference_intensity_cutoff_percent=float(
                                settings[
                                    "reference_intensity_cutoff_percent"
                                ]
                            ),
                            reference_fwhm_deg=float(
                                settings["reference_fwhm_deg"]
                            ),
                            pseudo_voigt_eta=float(
                                settings["pseudo_voigt_eta"]
                            ),
                            baseline_order=int(settings["baseline_order"]),
                            weighting=str(settings["weighting"]),
                            bootstrap_samples=int(
                                settings["bootstrap_samples"]
                            ),
                        )
                        dataset_result["qpa"] = _json_safe(qpa_result)
                        stage_record["message"] = (
                            f"{len(qpa_result.get('phases', []))} phase fraction(s)."
                        )

                    stage_record["status"] = "Completed"
                except Exception as exc:
                    stage_record["status"] = "Failed"
                    stage_record["message"] = str(exc)
                    dataset_result["warnings"].append(
                        f"{label}: {exc}"
                    )
                    if not recipe["scientific_controls"].get(
                        "continue_on_error", True
                    ):
                        raise
                finally:
                    stage_record["elapsed_seconds"] = float(
                        time.perf_counter() - stage_started
                    )
                    emit(
                        f"{dataset.name}: {label} — {stage_record['status']}",
                        advance=True,
                    )

            dataset_result["peak_rows"] = _json_safe(peaks)
            dataset_result["fit_groups"] = _json_safe(fit_groups)
            dataset_result["metrics"].update(
                {
                    "peak_count": len(peaks),
                    "fitted_component_count": sum(
                        len(group.get("components", []))
                        for group in fit_groups
                    ),
                    **_primary_peak_metrics(peaks, fit_groups),
                }
            )

            size_result = dataset_result.get("size_strain") or {}
            summary = size_result.get("scherrer_summary", {})
            wh = size_result.get("williamson_hall") or {}
            dataset_result["metrics"].update(
                {
                    "scherrer_mean_nm": summary.get("mean_nm"),
                    "scherrer_median_nm": summary.get("median_nm"),
                    "wh_size_nm": wh.get("crystallite_size_nm"),
                    "microstrain": wh.get("microstrain"),
                }
            )

            phase_result = dataset_result.get("phase_identification") or {}
            best = phase_result.get("best") or {}
            dataset_result["metrics"].update(
                {
                    "identified_phase": (
                        best.get("reference_name")
                        or best.get("display_name")
                    ),
                    "phase_score": best.get("score"),
                    "phase_match_count": best.get("matched_count"),
                }
            )

            whole_result = dataset_result.get("whole_pattern") or {}
            whole_fractions = {
                str(row.get("phase_name", "Phase")): row.get(
                    "pattern_fraction_percent"
                )
                for row in whole_result.get("phases", [])
            }
            dataset_result["metrics"].update(
                {
                    "whole_pattern_rwp_percent": whole_result.get(
                        "rwp_percent"
                    ),
                    "whole_pattern_rp_percent": whole_result.get(
                        "rp_percent"
                    ),
                    "whole_pattern_zero_shift_deg": whole_result.get(
                        "zero_shift_deg"
                    ),
                    "whole_pattern_fractions": whole_fractions,
                }
            )

            rietveld_result = dataset_result.get("rietveld") or {}
            rietveld_fractions = {
                str(row.get("phase_name", "Phase")): row.get("pattern_scale_fraction_percent")
                for row in rietveld_result.get("phases", [])
            }
            dataset_result["metrics"].update(
                {
                    "rietveld_rwp_percent": rietveld_result.get("rwp_percent"),
                    "rietveld_rp_percent": rietveld_result.get("rp_percent"),
                    "rietveld_zero_shift_deg": rietveld_result.get("zero_shift_deg"),
                    "rietveld_fractions": rietveld_fractions,
                }
            )

            qpa_result = dataset_result.get("qpa") or {}
            fractions = {}
            for row in qpa_result.get("phases", []):
                fractions[str(row.get("reference_name", "Phase"))] = row.get(
                    "corrected_weight_percent",
                    row.get("weight_percent"),
                )
            dataset_result["metrics"]["phase_fractions"] = fractions

            dataset.metadata["batch_recipe_fingerprint"] = fingerprint
            dataset.metadata["batch_recipe_name"] = recipe.get("name")
            dataset_result["status"] = (
                "Completed with warnings"
                if dataset_result["warnings"]
                else "Completed"
            )
            result["completed_count"] += 1

        except BatchCancelled:
            result["cancelled"] = True
            dataset_result["status"] = "Cancelled"
        except Exception as exc:
            dataset_result["status"] = "Failed"
            dataset_result["errors"].append(str(exc))
            result["failed_count"] += 1

        dataset_result["elapsed_seconds"] = float(
            time.perf_counter() - started
        )
        dataset_result["finished_utc"] = utc_now_text()
        plot_x, raw_y = _downsample(dataset.x, dataset.y_raw)
        _, processed_y = _downsample(dataset.x, y_for_analysis)
        dataset_result["plot_data"].update(
            {
                "x": plot_x,
                "raw_y": raw_y,
                "processed_y": processed_y,
            }
        )
        result["results"].append(dataset_result)

        if result["cancelled"]:
            break

    if (
        recipe["stages"].get("residual_stress", False)
        and not result["cancelled"]
    ):
        stress_settings = recipe["residual_stress"]
        observations = []
        result_by_uid = {
            row["dataset_uid"]: row for row in result["results"]
        }
        for dataset in experimental:
            dataset_result = result_by_uid.get(dataset.uid)
            if not dataset_result or dataset_result["status"] == "Failed":
                continue
            psi_deg, psi_source = infer_psi_deg(
                dataset.metadata,
                dataset.name,
            )
            if psi_deg is None:
                dataset_result["warnings"].append(
                    "Residual stress: ψ angle not found in metadata or dataset name."
                )
                continue
            observation = extract_peak_observation(
                dataset.x,
                (
                    dataset.y_processed
                    if dataset.y_processed is not None
                    else dataset.y_raw
                ),
                target_two_theta_deg=float(
                    stress_settings["target_two_theta_deg"]
                ),
                half_window_deg=float(
                    stress_settings["search_half_window_deg"]
                ),
                source_mode=str(stress_settings["peak_source"]),
                fit_groups=dataset_result.get("fit_groups", []),
                detected_peaks=dataset_result.get("peak_rows", []),
                default_error_deg=float(
                    stress_settings["default_peak_error_deg"]
                ),
            )
            if observation is None:
                continue
            observations.append(
                {
                    "included": True,
                    "dataset_uid": dataset.uid,
                    "dataset_name": dataset.name,
                    "psi_deg": float(psi_deg),
                    "psi_source": psi_source,
                    **observation,
                }
            )
        try:
            result["residual_stress"] = _json_safe(
                analyze_sin2psi(
                    observations,
                    wavelength_angstrom=float(
                        stress_settings["wavelength_angstrom"]
                    ),
                    reference_mode=str(
                        stress_settings["reference_mode"]
                    ),
                    stress_free_two_theta_deg=float(
                        stress_settings["stress_free_two_theta_deg"]
                    ),
                    elastic_mode=str(stress_settings["elastic_mode"]),
                    youngs_modulus_gpa=float(
                        stress_settings["youngs_modulus_gpa"]
                    ),
                    poisson_ratio=float(
                        stress_settings["poisson_ratio"]
                    ),
                    xec_half_s2_per_gpa=float(
                        stress_settings["xec_half_s2_per_gpa"]
                    ),
                    regression_mode=str(
                        stress_settings["regression_mode"]
                    ),
                    azimuth_deg=float(stress_settings["azimuth_deg"]),
                )
            )
        except (ResidualStressError, ValueError) as exc:
            result["errors"].append(f"Residual stress: {exc}")

    result["elapsed_seconds"] = float(time.perf_counter() - batch_started)
    return _json_safe(result)


def comparison_rows(batch_result: dict) -> list[dict]:
    rows = []
    for index, result in enumerate(batch_result.get("results", []), start=1):
        metrics = result.get("metrics", {})
        metadata = result.get("metadata", {})
        row = {
            "sample_index": index,
            "dataset_uid": result.get("dataset_uid"),
            "dataset_name": result.get("dataset_name"),
            "status": result.get("status"),
            "source_path": result.get("source_path"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "recipe_fingerprint": result.get("recipe_fingerprint"),
            "recipe_deviation": result.get("recipe_deviation", False),
            "temperature": metadata.get("temperature"),
            "time": metadata.get("time"),
            "composition": metadata.get("composition"),
            **metrics,
        }
        fractions = metrics.get("phase_fractions") or {}
        for phase_name, value in fractions.items():
            row[f"phase_fraction::{phase_name}"] = value
        whole_fractions = metrics.get("whole_pattern_fractions") or {}
        for phase_name, value in whole_fractions.items():
            row[f"pawley_fraction::{phase_name}"] = value
        rietveld_fractions = metrics.get("rietveld_fractions") or {}
        for phase_name, value in rietveld_fractions.items():
            row[f"rietveld_fraction::{phase_name}"] = value
        rows.append(row)
    return rows
