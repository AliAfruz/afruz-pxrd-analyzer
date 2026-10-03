from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .crystallography import CIFImportError, calculate_powder_pattern, load_cif
from .rietveld_refinement import RietveldError, RietveldPhaseSpec, refine_rietveld
from .structure_validation import parse_formula_counts, validate_crystal_structure
from .text_export import write_mapping_txt, write_table_txt, write_manifest_txt
from .refinement_export import export_refinement_txt_bundle


class MultiComponentRefinementError(ValueError):
    """Raised when a multi-component refinement request is not scientifically valid."""


@dataclass(frozen=True)
class CandidatePhase:
    """One possible crystalline component supplied by the user.

    The structure is expected to follow Afruz CIF dictionaries: ``cell``,
    ``space_group``, ``crystal_system`` and, for Rietveld mode, ``atoms``.
    """

    label: str
    structure: dict
    source: str = "user-supplied CIF"
    role: str = "candidate"
    required: bool = False
    enabled: bool = True
    refine_cell: bool = False
    refine_biso: bool = False
    preferred_orientation_hkl: tuple[int, int, int] | None = None
    refine_preferred_orientation: bool = False


@dataclass(frozen=True)
class MultiComponentSearchSettings:
    wavelength_angstrom: float = 1.5406
    two_theta_min: float | None = None
    two_theta_max: float | None = None
    intensity_cutoff_percent: float = 0.5
    background_order: int = 3
    weighting: str = "Balanced"
    max_phases: int = 3
    max_models: int = 64
    refinement_mode: str = "safe_rietveld"
    refine_zero_shift: bool = False
    refine_profile: bool = False
    refine_eta: bool = False
    use_staged_refinement: bool = False
    maximum_nonlinear_evaluations: int = 40
    maximum_optimization_points: int = 1600
    peak_match_tolerance_deg: float = 0.18
    residual_peak_prominence_fraction: float = 0.08
    minimum_phase_fraction_percent: float = 0.5
    bic_improvement_required: float = 4.0
    false_peak_penalty_threshold: float = 0.02
    mismatch_residual_z_threshold: float = 6.0
    artifact_residual_z_threshold: float = 10.0
    mismatch_window_half_width_deg: float = 0.10
    maximum_auto_excluded_windows: int = 8
    maximum_excluded_point_fraction: float = 0.03


@dataclass(frozen=True)
class PhaseCombinationModel:
    labels: tuple[str, ...]
    required_labels: tuple[str, ...]
    optional_labels: tuple[str, ...]


def import_candidate_phase(
    cif_path: str | Path,
    *,
    label: str | None = None,
    required: bool = False,
    refine_cell: bool = False,
) -> CandidatePhase:
    """Load one CIF as a multi-component candidate phase."""

    path = Path(cif_path)
    if not path.exists():
        raise MultiComponentRefinementError(f"CIF file does not exist: {path}")
    try:
        structure = load_cif(path)
    except CIFImportError as exc:
        raise MultiComponentRefinementError(f"Could not import CIF {path}: {exc}") from exc
    return CandidatePhase(
        label=label or str(structure.get("data_name") or path.stem),
        structure=structure,
        source=str(path),
        required=bool(required),
        refine_cell=bool(refine_cell),
    )


def normalize_candidate_phase(value: CandidatePhase | Mapping) -> CandidatePhase:
    if isinstance(value, CandidatePhase):
        return value
    if isinstance(value, Mapping):
        structure = value.get("structure")
        if not isinstance(structure, Mapping):
            # Treat a raw structure dictionary as a phase when it has a cell.
            if "cell" in value:
                structure = value
            else:
                raise MultiComponentRefinementError("Candidate phase mapping must contain a structure/cell.")
        return CandidatePhase(
            label=str(value.get("label") or value.get("name") or structure.get("data_name") or "Candidate phase"),
            structure=dict(structure),
            source=str(value.get("source") or structure.get("_afruz_origin") or "user-supplied structure"),
            role=str(value.get("role") or "candidate"),
            required=bool(value.get("required", False)),
            enabled=bool(value.get("enabled", True)),
            refine_cell=bool(value.get("refine_cell", False)),
            refine_biso=bool(value.get("refine_biso", False)),
            preferred_orientation_hkl=(
                tuple(int(v) for v in value["preferred_orientation_hkl"])
                if value.get("preferred_orientation_hkl") is not None
                else None
            ),
            refine_preferred_orientation=bool(value.get("refine_preferred_orientation", False)),
        )
    raise MultiComponentRefinementError(f"Unsupported candidate phase type: {type(value)!r}")


def _enabled_phases(phases: Sequence[CandidatePhase | Mapping]) -> list[CandidatePhase]:
    out = [normalize_candidate_phase(phase) for phase in phases]
    out = [phase for phase in out if phase.enabled]
    if not out:
        raise MultiComponentRefinementError("At least one enabled candidate phase is required.")
    labels = [phase.label for phase in out]
    if len(labels) != len(set(labels)):
        raise MultiComponentRefinementError("Candidate phase labels must be unique.")
    return out


def _phase_spec(phase: CandidatePhase) -> RietveldPhaseSpec:
    return RietveldPhaseSpec(
        structure=deepcopy(phase.structure),
        name=phase.label,
        refine_cell=bool(phase.refine_cell),
        refine_biso=bool(phase.refine_biso),
        preferred_orientation_hkl=phase.preferred_orientation_hkl,
        refine_preferred_orientation=bool(phase.refine_preferred_orientation),
        included=True,
    )


def _information_criteria(residual: np.ndarray, parameter_count: int) -> dict:
    residual = np.asarray(residual, dtype=float)
    n = max(1, residual.size)
    sse = max(float(np.sum(residual * residual)), 1e-30)
    k = max(1, int(parameter_count))
    return {
        "sse": sse,
        "aic": float(n * math.log(sse / n) + 2 * k),
        "bic": float(n * math.log(sse / n) + k * math.log(n)),
    }


def _local_maxima(x: np.ndarray, y: np.ndarray, *, prominence_fraction: float) -> list[dict]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return []
    baseline = float(np.percentile(y, 20.0))
    span = max(float(np.max(y) - baseline), 1e-12)
    threshold = baseline + max(0.0, float(prominence_fraction)) * span
    peaks = []
    for index in range(1, len(x) - 1):
        value = float(y[index])
        if value >= threshold and value >= float(y[index - 1]) and value >= float(y[index + 1]):
            if peaks and abs(float(x[index]) - peaks[-1]["two_theta_deg"]) < 0.05:
                if value > peaks[-1]["intensity"]:
                    peaks[-1] = {"index": index, "two_theta_deg": float(x[index]), "intensity": value}
            else:
                peaks.append({"index": index, "two_theta_deg": float(x[index]), "intensity": value})
    return peaks


def _theoretical_phase_peaks(
    phase: CandidatePhase,
    settings: MultiComponentSearchSettings,
    x_min: float,
    x_max: float,
) -> list[dict]:
    try:
        pattern = calculate_powder_pattern(
            phase.structure,
            wavelength_angstrom=settings.wavelength_angstrom,
            two_theta_min=x_min,
            two_theta_max=x_max,
            intensity_cutoff_percent=settings.intensity_cutoff_percent,
        )
    except Exception:
        return []
    rows = []
    for peak in pattern:
        rows.append(
            {
                "phase_label": phase.label,
                "two_theta_deg": float(peak["two_theta"]),
                "relative_intensity": float(peak.get("intensity", 0.0)),
                "hkl_label": str(peak.get("hkl_label") or peak.get("hkl") or ""),
            }
        )
    return rows


def explain_observed_peaks(
    x: Sequence[float],
    observed_y: Sequence[float],
    phases: Sequence[CandidatePhase | Mapping],
    *,
    settings: MultiComponentSearchSettings | None = None,
) -> dict:
    """Explain observed local maxima using calculated phase reflection positions."""

    settings = settings or MultiComponentSearchSettings()
    phase_list = _enabled_phases(phases)
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(observed_y, dtype=float)
    observed_peaks = _local_maxima(
        x_array,
        y_array,
        prominence_fraction=settings.residual_peak_prominence_fraction,
    )
    theoretical = []
    for phase in phase_list:
        theoretical.extend(_theoretical_phase_peaks(phase, settings, float(x_array[0]), float(x_array[-1])))
    explained = []
    unexplained = []
    for peak in observed_peaks:
        matches = [
            row for row in theoretical
            if abs(row["two_theta_deg"] - peak["two_theta_deg"]) <= settings.peak_match_tolerance_deg
        ]
        matches.sort(key=lambda row: (abs(row["two_theta_deg"] - peak["two_theta_deg"]), -row["relative_intensity"]))
        record = {
            **peak,
            "explained": bool(matches),
            "matches": matches[:6],
            "phase_labels": sorted({row["phase_label"] for row in matches}),
        }
        if matches:
            explained.append(record)
        else:
            unexplained.append(record)
    return {
        "observed_peak_count": len(observed_peaks),
        "explained_peak_count": len(explained),
        "unexplained_peak_count": len(unexplained),
        "coverage_percent": 100.0 * len(explained) / len(observed_peaks) if observed_peaks else 0.0,
        "observed_peaks": explained + unexplained,
        "unexplained_peaks": unexplained,
    }


def _fit_model(
    x: np.ndarray,
    y: np.ndarray,
    phases: list[CandidatePhase],
    settings: MultiComponentSearchSettings,
) -> dict:
    specs = [_phase_spec(phase) for phase in phases]
    for spec in specs:
        # Safe model-search default: scales + polynomial background. Cell/profile
        # refinement is opt-in per phase/settings because over-fitting is easy in
        # many-component patterns.
        if settings.refinement_mode.startswith("safe"):
            spec.refine_cell = False
            spec.refine_biso = False
            spec.refine_preferred_orientation = False
    try:
        result = refine_rietveld(
            x,
            y,
            specs,
            wavelength_angstrom=settings.wavelength_angstrom,
            two_theta_min=settings.two_theta_min,
            two_theta_max=settings.two_theta_max,
            intensity_cutoff_percent=settings.intensity_cutoff_percent,
            background_order=settings.background_order,
            weighting=settings.weighting,
            refine_zero_shift=settings.refine_zero_shift,
            refine_profile=settings.refine_profile,
            refine_eta=settings.refine_eta,
            maximum_nonlinear_evaluations=settings.maximum_nonlinear_evaluations,
            maximum_optimization_points=settings.maximum_optimization_points,
            use_staged_refinement=settings.use_staged_refinement,
        )
    except RietveldError as exc:
        raise MultiComponentRefinementError(str(exc)) from exc
    residual = np.asarray(result["difference_y"], dtype=float)
    criteria = _information_criteria(residual, int(result.get("parameter_count", 1)))
    result["information_criteria"] = criteria
    return result


def _background_only_model(x: np.ndarray, y: np.ndarray, order: int = 3) -> dict:
    center = float(np.mean(x))
    span = max(float(np.ptp(x)) / 2.0, np.finfo(float).eps)
    xn = (x - center) / span
    background = np.column_stack([xn ** degree for degree in range(order + 1)])
    coefficients, *_ = np.linalg.lstsq(background, y, rcond=None)
    calculated = background @ coefficients
    residual = y - calculated
    criteria = _information_criteria(residual, len(coefficients))
    sum_wy2 = max(float(np.sum(y * y)), 1e-30)
    rwp = 100.0 * math.sqrt(float(np.sum(residual * residual)) / sum_wy2)
    return {
        "model_id": "background_only",
        "phase_labels": [],
        "rwp_percent": float(rwp),
        "parameter_count": len(coefficients),
        "information_criteria": criteria,
        "calculated_y": calculated.tolist(),
        "difference_y": residual.tolist(),
        "background_coefficients": coefficients.tolist(),
    }


def _combination_plan(phases: list[CandidatePhase], settings: MultiComponentSearchSettings) -> list[PhaseCombinationModel]:
    required = [phase for phase in phases if phase.required]
    optional = [phase for phase in phases if not phase.required]
    max_phases = max(1, min(int(settings.max_phases), len(phases)))
    plan: list[PhaseCombinationModel] = []
    if len(required) > max_phases:
        raise MultiComponentRefinementError("Required phases exceed max_phases.")
    for count in range(0, max_phases - len(required) + 1):
        for subset in combinations(optional, count):
            model_phases = required + list(subset)
            if not model_phases:
                continue
            labels = tuple(phase.label for phase in model_phases)
            plan.append(
                PhaseCombinationModel(
                    labels=labels,
                    required_labels=tuple(phase.label for phase in required),
                    optional_labels=tuple(phase.label for phase in subset),
                )
            )
    plan.sort(key=lambda model: (len(model.labels), model.labels))
    return plan[: max(1, int(settings.max_models))]


def _phase_fraction_rows(result: dict, settings: MultiComponentSearchSettings) -> list[dict]:
    rows = []
    for phase in result.get("phases", []):
        fraction = float(phase.get("pattern_scale_fraction_percent", phase.get("pattern_fraction_percent", 0.0)) or 0.0)
        if fraction >= 20.0:
            support = "Strongly supported phase"
        elif fraction >= 5.0:
            support = "Supported minor phase"
        elif fraction >= settings.minimum_phase_fraction_percent:
            support = "Weak / uncertain phase"
        else:
            support = "Below reporting threshold"
        rows.append(
            {
                "phase_index": phase.get("phase_index"),
                "phase_name": phase.get("phase_name"),
                "formula": phase.get("formula"),
                "scale_factor": phase.get("scale_factor"),
                "pattern_fraction_percent": fraction,
                "semi_quantitative_fraction_percent": fraction,
                "support_label": support,
                "refined_cell": phase.get("refined_cell"),
                "structure_plausibility_status": phase.get("structure_plausibility_status"),
                "warning": "Semi-quantitative crystalline pattern fraction; not a validated weight fraction without calibrated Rietveld/QPA model.",
            }
        )
    return rows


def _component_diagnostics(
    result: dict,
    x: np.ndarray,
    y: np.ndarray,
    phases: list[CandidatePhase],
    settings: MultiComponentSearchSettings,
) -> dict:
    explanation = explain_observed_peaks(x, y, phases, settings=settings)
    residual_y = np.asarray(result.get("difference_y", []), dtype=float)
    residual_peaks = _local_maxima(
        np.asarray(result.get("observed_x", x), dtype=float),
        np.abs(residual_y),
        prominence_fraction=max(0.05, settings.residual_peak_prominence_fraction),
    )
    return {
        "peak_explanation": explanation,
        "residual_peak_count": len(residual_peaks),
        "largest_residual_peaks": sorted(residual_peaks, key=lambda row: row["intensity"], reverse=True)[:20],
    }


def _robust_residual_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = float(np.std(values))
    return max(sigma, 1e-12)


def _half_height_width(x: np.ndarray, y: np.ndarray, index: int) -> tuple[float, int]:
    """Estimate a local residual-peak width without fitting a profile model."""

    if len(x) < 2 or index < 0 or index >= len(x):
        return 0.0, 1
    height = max(float(y[index]), 0.0)
    if height <= 0.0:
        return 0.0, 1
    threshold = 0.5 * height
    left = index
    right = index
    while left > 0 and float(y[left - 1]) >= threshold:
        left -= 1
    while right < len(y) - 1 and float(y[right + 1]) >= threshold:
        right += 1
    return float(x[right] - x[left]), int(right - left + 1)


def build_mismatch_review(
    x: Sequence[float],
    observed_y: Sequence[float],
    refinement_result: Mapping,
    phases: Sequence[CandidatePhase | Mapping],
    *,
    settings: MultiComponentSearchSettings | None = None,
) -> list[dict]:
    """Classify large positive residual peaks after a first-pass refinement.

    A Rietveld refinement operates on profile points, not on a peak list.  The
    returned rows therefore describe reviewable *windows*.  Only isolated,
    high-significance single-point spikes are recommended for automatic
    exclusion.  Unexplained crystalline peaks remain enabled because they may
    represent a missing phase rather than bad data.
    """

    settings = settings or MultiComponentSearchSettings()
    phase_list = _enabled_phases(phases)
    x_array = np.asarray(refinement_result.get("observed_x", x), dtype=float)
    observed = np.asarray(refinement_result.get("observed_y", observed_y), dtype=float)
    calculated = np.asarray(refinement_result.get("calculated_y", []), dtype=float)
    if x_array.ndim != 1 or observed.ndim != 1 or calculated.ndim != 1:
        raise MultiComponentRefinementError("First-pass profile arrays must be one-dimensional.")
    if not (len(x_array) == len(observed) == len(calculated)):
        raise MultiComponentRefinementError("First-pass observed and calculated profile lengths do not match.")
    if len(x_array) < 3:
        return []

    residual = observed - calculated
    positive_residual = np.maximum(residual, 0.0)
    sigma = _robust_residual_sigma(residual)
    step = float(np.median(np.diff(x_array)))
    theoretical = []
    for phase in phase_list:
        theoretical.extend(
            _theoretical_phase_peaks(phase, settings, float(x_array[0]), float(x_array[-1]))
        )

    candidates = _local_maxima(
        x_array,
        positive_residual,
        prominence_fraction=max(0.01, settings.residual_peak_prominence_fraction),
    )
    rows: list[dict] = []
    for peak in candidates:
        index = int(peak["index"])
        z_score = float(positive_residual[index] / sigma)
        if z_score < float(settings.mismatch_residual_z_threshold):
            continue
        width_deg, half_height_points = _half_height_width(x_array, positive_residual, index)
        nearest = None
        if theoretical:
            nearest = min(
                theoretical,
                key=lambda row: abs(float(row["two_theta_deg"]) - float(peak["two_theta_deg"])),
            )
        delta = (
            abs(float(nearest["two_theta_deg"]) - float(peak["two_theta_deg"]))
            if nearest is not None
            else None
        )
        matched = delta is not None and delta <= float(settings.peak_match_tolerance_deg)
        isolated_spike = (
            not matched
            and half_height_points <= 2
            and width_deg <= max(2.5 * abs(step), 0.03)
            and z_score >= float(settings.artifact_residual_z_threshold)
        )
        if matched:
            classification = "Matched reflection; intensity/profile mismatch"
            action = "Keep. Refine scale, cell, profile, texture, or structure model."
            auto_exclude = False
        elif isolated_spike:
            classification = "Isolated spike / artifact candidate"
            action = "Review instrument/raw scan; safe candidate to uncheck for Pass 2."
            auto_exclude = True
        else:
            classification = "Unexplained crystalline peak / possible missing phase"
            action = "Keep. Search for an additional phase before excluding this peak."
            auto_exclude = False

        half_window = max(float(settings.mismatch_window_half_width_deg), 2.0 * abs(step))
        center = float(peak["two_theta_deg"])
        rows.append(
            {
                "peak_id": len(rows) + 1,
                "use_in_second_pass": not auto_exclude,
                "auto_exclude_recommended": auto_exclude,
                "two_theta_deg": center,
                "observed_intensity": float(observed[index]),
                "calculated_intensity": float(calculated[index]),
                "positive_residual": float(positive_residual[index]),
                "residual_z_score": z_score,
                "estimated_half_height_width_deg": width_deg,
                "half_height_point_count": half_height_points,
                "nearest_phase": str(nearest.get("phase_label", "")) if nearest else "",
                "nearest_reflection_two_theta_deg": float(nearest["two_theta_deg"]) if nearest else None,
                "delta_two_theta_deg": delta,
                "classification": classification,
                "recommended_action": action,
                "window_min_deg": max(float(x_array[0]), center - half_window),
                "window_max_deg": min(float(x_array[-1]), center + half_window),
            }
        )

    # Limit automatic exclusions to the strongest isolated spikes.  Other rows
    # remain visible but checked so real minor phases are not silently deleted.
    auto_rows = sorted(
        [row for row in rows if row["auto_exclude_recommended"]],
        key=lambda row: float(row["residual_z_score"]),
        reverse=True,
    )
    allowed_ids = {
        row["peak_id"] for row in auto_rows[: max(0, int(settings.maximum_auto_excluded_windows))]
    }
    for row in rows:
        if row["auto_exclude_recommended"] and row["peak_id"] not in allowed_ids:
            row["use_in_second_pass"] = True
            row["auto_exclude_recommended"] = False
            row["recommended_action"] = "Keep: automatic exclusion limit reached; review manually."
    return rows


def run_multicomponent_second_pass(
    x: Sequence[float],
    y: Sequence[float],
    candidate_phases: Sequence[CandidatePhase | Mapping],
    first_pass_result: Mapping,
    mismatch_review: Sequence[Mapping],
    *,
    settings: MultiComponentSearchSettings | None = None,
) -> dict:
    """Refine the first-pass best model after user-reviewed profile masking."""

    settings = settings or MultiComponentSearchSettings()
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 1 or y_array.ndim != 1 or len(x_array) != len(y_array):
        raise MultiComponentRefinementError("X and Y must be one-dimensional arrays of equal length.")
    best = first_pass_result.get("best_model") or {}
    labels = list(best.get("phase_labels") or [])
    if not labels:
        raise MultiComponentRefinementError("A successful first-pass best model is required before Pass 2.")
    phase_by_label = {phase.label: phase for phase in _enabled_phases(candidate_phases)}
    missing = [label for label in labels if label not in phase_by_label]
    if missing:
        raise MultiComponentRefinementError("Pass 2 is missing candidate phases: " + ", ".join(missing))

    mask = np.ones(len(x_array), dtype=bool)
    excluded_windows = []
    for row in mismatch_review:
        if bool(row.get("use_in_second_pass", True)):
            continue
        low = float(row.get("window_min_deg", row.get("two_theta_deg", 0.0)))
        high = float(row.get("window_max_deg", row.get("two_theta_deg", 0.0)))
        if high < low:
            low, high = high, low
        local = (x_array >= low) & (x_array <= high)
        if not np.any(local):
            continue
        mask &= ~local
        excluded_windows.append(
            {
                "peak_id": row.get("peak_id"),
                "two_theta_deg": row.get("two_theta_deg"),
                "window_min_deg": low,
                "window_max_deg": high,
                "classification": row.get("classification", "User excluded window"),
                "reason": row.get("recommended_action", "Unchecked by user for Pass 2"),
            }
        )

    # Preserve the measured endpoints so display/range metadata still reflects
    # the original experiment, even when an edge window was manually selected.
    if len(mask):
        mask[0] = True
        mask[-1] = True
    excluded_count = int(np.count_nonzero(~mask))
    excluded_fraction = excluded_count / max(1, len(mask))
    if excluded_fraction > float(settings.maximum_excluded_point_fraction):
        raise MultiComponentRefinementError(
            f"Pass 2 would exclude {100.0 * excluded_fraction:.2f}% of profile points, exceeding the "
            f"configured {100.0 * settings.maximum_excluded_point_fraction:.2f}% safety limit."
        )
    if int(np.count_nonzero(mask)) < 80:
        raise MultiComponentRefinementError("Too few measured points remain after mismatch-window review.")

    selected_phases = [phase_by_label[label] for label in labels]
    refinement = _fit_model(x_array[mask], y_array[mask], selected_phases, settings)
    diagnostics = _component_diagnostics(
        refinement,
        x_array[mask],
        y_array[mask],
        selected_phases,
        settings,
    )
    return {
        "success": True,
        "method": "Pass 2 user-reviewed mismatch-window refinement",
        "phase_labels": labels,
        "input_point_count": int(len(x_array)),
        "retained_point_count": int(np.count_nonzero(mask)),
        "excluded_point_count": excluded_count,
        "excluded_point_fraction_percent": 100.0 * excluded_fraction,
        "excluded_windows": excluded_windows,
        "first_pass_rwp_percent": best.get("rwp_percent"),
        "second_pass_fit_rwp_percent": refinement.get("rwp_percent"),
        "second_pass_fit_rp_percent": refinement.get("rp_percent"),
        "comparison_warning": (
            "Pass 1 and Pass 2 R-factors use different point masks and must not be compared as if they were identical datasets."
        ),
        "refinement": refinement,
        "diagnostics": diagnostics,
    }


def _model_record(
    result: dict,
    labels: Sequence[str],
    baseline_bic: float,
    previous_best_bic: float | None,
    settings: MultiComponentSearchSettings,
    diagnostics: dict,
) -> dict:
    bic = float(result["information_criteria"]["bic"])
    delta_vs_background = baseline_bic - bic
    delta_vs_previous = (previous_best_bic - bic) if previous_best_bic is not None else None
    fractions = _phase_fraction_rows(result, settings)
    weak = [row["phase_name"] for row in fractions if row["pattern_fraction_percent"] < settings.minimum_phase_fraction_percent]
    if delta_vs_background >= 25.0 and not weak:
        label = "Strong multiphase model — validate"
    elif delta_vs_background >= settings.bic_improvement_required:
        label = "Possible multiphase model — review"
    else:
        label = "Not justified over simpler/background model"
    if weak:
        label = "Contains weak/unsupported phase — review"
    return {
        "model_id": "+".join(labels),
        "phase_labels": list(labels),
        "phase_count": len(labels),
        "status_label": label,
        "rwp_percent": float(result.get("rwp_percent", 0.0)),
        "rp_percent": float(result.get("rp_percent", 0.0)),
        "r_squared": float(result.get("r_squared", 0.0)),
        "rmse": float(result.get("rmse", 0.0)),
        "bic": bic,
        "aic": float(result["information_criteria"]["aic"]),
        "bic_improvement_vs_background": float(delta_vs_background),
        "bic_improvement_vs_previous_best": None if delta_vs_previous is None else float(delta_vs_previous),
        "phase_fractions": fractions,
        "explained_peak_coverage_percent": diagnostics["peak_explanation"]["coverage_percent"],
        "unexplained_peak_count": diagnostics["peak_explanation"]["unexplained_peak_count"],
        "residual_peak_count": diagnostics["residual_peak_count"],
        "weak_phase_labels": weak,
        "parameter_count": int(result.get("parameter_count", 0)),
        "reflection_count": int(result.get("reflection_count", 0)),
        "warnings": list(result.get("warnings", [])),
    }


def run_multicomponent_refinement(
    x: Sequence[float],
    y: Sequence[float],
    candidate_phases: Sequence[CandidatePhase | Mapping],
    *,
    settings: MultiComponentSearchSettings | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
) -> dict:
    """Search and fit user-supplied CIF phase combinations.

    This is an intentionally conservative model-search wrapper around the
    existing Afruz Rietveld machinery.  The default mode refines scale factors
    and background only, then ranks candidate phase combinations by BIC-like
    parsimony and peak-explanation diagnostics.
    """

    started = time.perf_counter()
    settings = settings or MultiComponentSearchSettings()

    def emit_progress(percent: float, message: str, **extra):
        if progress_callback is None:
            return
        event = {
            "percent": int(max(0, min(100, round(float(percent))))),
            "message": str(message),
        }
        event.update(extra)
        progress_callback(event)

    def check_cancelled():
        if cancel_checker is not None and cancel_checker():
            raise MultiComponentRefinementError("Multiphase search cancelled by user.")

    emit_progress(0, "Validating pattern and candidate CIF phases…")
    settings = settings or MultiComponentSearchSettings()
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 1 or y_array.ndim != 1 or len(x_array) != len(y_array):
        raise MultiComponentRefinementError("X and Y must be one-dimensional arrays of equal length.")
    if len(x_array) < 80:
        raise MultiComponentRefinementError("At least eighty measured points are required.")
    if settings.wavelength_angstrom <= 0:
        raise MultiComponentRefinementError("Wavelength must be positive.")
    phases = _enabled_phases(candidate_phases)
    check_cancelled()
    emit_progress(5, f"Loaded {len(phases)} enabled CIF candidate phase(s).")
    structures_without_atoms = [phase.label for phase in phases if not phase.structure.get("atoms")]
    if structures_without_atoms:
        raise MultiComponentRefinementError(
            "Rietveld mode requires atom coordinates in every CIF. Missing atoms: "
            + ", ".join(structures_without_atoms)
        )

    check_cancelled()
    emit_progress(8, "Fitting background-only reference model…")
    baseline = _background_only_model(x_array, y_array, settings.background_order)
    baseline_bic = float(baseline["information_criteria"]["bic"])
    phase_by_label = {phase.label: phase for phase in phases}
    plan = _combination_plan(phases, settings)
    emit_progress(12, f"Prepared {len(plan)} phase-combination model(s) to test.", total_models=len(plan))
    models = []
    full_results = []
    best_bic: float | None = None
    total_models = max(1, len(plan))
    for model_index, combination_model in enumerate(plan, start=1):
        check_cancelled()
        model_name = "+".join(combination_model.labels)
        emit_progress(
            12 + 76.0 * (model_index - 1) / total_models,
            f"Fitting model {model_index}/{len(plan)}: {model_name}",
            model_index=model_index,
            total_models=len(plan),
            model_id=model_name,
        )
        combo_phases = [phase_by_label[label] for label in combination_model.labels]
        try:
            result = _fit_model(x_array, y_array, combo_phases, settings)
            diagnostics = _component_diagnostics(result, x_array, y_array, combo_phases, settings)
            record = _model_record(
                result,
                combination_model.labels,
                baseline_bic,
                best_bic,
                settings,
                diagnostics,
            )
            models.append(record)
            full_results.append({"model_id": record["model_id"], "refinement": result, "diagnostics": diagnostics})
            bic = float(record["bic"])
            best_bic = bic if best_bic is None else min(best_bic, bic)
            emit_progress(
                12 + 76.0 * model_index / total_models,
                f"Finished model {model_index}/{len(plan)}: {model_name}",
                model_index=model_index,
                total_models=len(plan),
                model_id=model_name,
                rwp_percent=record.get("rwp_percent"),
            )
        except Exception as exc:  # keep search robust across bad candidate CIFs
            models.append(
                {
                    "model_id": "+".join(combination_model.labels),
                    "phase_labels": list(combination_model.labels),
                    "phase_count": len(combination_model.labels),
                    "status_label": "Failed model",
                    "error": str(exc),
                    "bic": None,
                    "rwp_percent": None,
                    "warnings": ["Model failed and was not used for ranking."],
                }
            )
    check_cancelled()
    emit_progress(90, "Ranking tested multiphase models…")
    ranked = sorted(
        [model for model in models if model.get("bic") is not None],
        key=lambda row: (float(row["bic"]), int(row.get("phase_count", 999))),
    )
    failed = [model for model in models if model.get("bic") is None]
    ranked.extend(failed)
    best = ranked[0] if ranked and ranked[0].get("bic") is not None else None
    mismatch_review: list[dict] = []
    if best:
        best_id = best.get("model_id")
        best_full = next(
            (row.get("refinement") for row in full_results if row.get("model_id") == best_id),
            None,
        )
        if isinstance(best_full, Mapping):
            best_labels = list(best.get("phase_labels") or [])
            best_phases = [phase_by_label[label] for label in best_labels if label in phase_by_label]
            mismatch_review = build_mismatch_review(
                x_array,
                y_array,
                best_full,
                best_phases,
                settings=settings,
            )
    rejected = []
    if best:
        best_labels = set(best["phase_labels"])
        for phase in phases:
            if phase.label not in best_labels:
                rejected.append(
                    {
                        "phase_label": phase.label,
                        "reason": "Not present in the best parsimony-ranked model.",
                        "status_label": "Rejected / unsupported candidate",
                    }
                )
        for fraction in best.get("phase_fractions", []):
            if fraction["pattern_fraction_percent"] < settings.minimum_phase_fraction_percent:
                rejected.append(
                    {
                        "phase_label": fraction["phase_name"],
                        "reason": "Refined scale fraction is below the reporting threshold.",
                        "status_label": "Weak / uncertain phase",
                    }
                )
    emit_progress(96, "Preparing phase rejection, fraction and report tables…")
    warnings = [
        "Phase 19.0 results are model-selection guidance, not automatic proof of phase identity.",
        "Default safe mode refines phase scales and background only; inspect profile, calibration, texture, stacking disorder and amorphous content before final claims.",
        "Reported fractions are semi-quantitative crystalline pattern fractions unless a validated full Rietveld/QPA model is accepted.",
    ]
    output = {
        "success": bool(best),
        "method": "Intelligent multiphase Rietveld/Pawley-style CIF model search",
        "phase": "19.0",
        "settings": asdict(settings),
        "candidate_phase_count": len(phases),
        "candidate_phases": [
            {
                "label": phase.label,
                "source": phase.source,
                "role": phase.role,
                "required": phase.required,
                "formula": str(phase.structure.get("formula") or ""),
                "space_group": phase.structure.get("space_group"),
                "crystal_system": phase.structure.get("crystal_system"),
                "atom_count": len(phase.structure.get("atoms", [])),
                "structure_plausibility": validate_crystal_structure(phase.structure),
            }
            for phase in phases
        ],
        "background_only": baseline,
        "tested_model_count": len(models),
        "ranked_models": ranked,
        "best_model": best,
        "rejected_phases": rejected,
        "full_refinement_results": full_results,
        "mismatch_review": mismatch_review,
        "scientific_warnings": warnings,
        "report_tables": {
            "phase19_ranked_models": ranked,
            "phase19_best_phase_fractions": best.get("phase_fractions", []) if best else [],
            "phase19_rejected_phases": rejected,
            "phase19_candidate_phases": [
                {
                    "label": phase.label,
                    "formula": str(phase.structure.get("formula") or ""),
                    "space_group": phase.structure.get("space_group"),
                    "source": phase.source,
                }
                for phase in phases
            ],
            "phase21_mismatch_review": mismatch_review,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    emit_progress(100, "Multiphase model search complete.", best_model=(best or {}).get("model_id"))
    return output


def export_multicomponent_report_bundle(
    output_dir: str | Path,
    result: Mapping,
    *,
    dataset_name: str = "Phase 19 active dataset",
    dataset_uid: str = "",
) -> dict:
    """Write a complete Phase 19 bundle with clean TXT as the primary data format.

    JSON/CSV compatibility copies are retained so older workflows and project
    tests continue to work, but every numerical table is also written as a
    validated UTF-8 tab-delimited TXT file.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "phase19_multicomponent_refinement.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    written = [result_path]

    result_txt = write_mapping_txt(
        root / "phase19_multicomponent_refinement.txt",
        result,
        title=f"Phase 19 intelligent multiphase refinement — {dataset_name}",
        metadata={"dataset_name": dataset_name, "dataset_uid": dataset_uid or "Not recorded"},
        backup=False,
    )
    written.append(Path(result_txt["txt_path"]))

    tables_dir = root / "tables"
    tables_dir.mkdir(exist_ok=True)
    for table_name, rows in (result.get("report_tables") or {}).items():
        rows = list(rows or [])
        if not rows:
            continue
        columns = sorted({key for row in rows if isinstance(row, Mapping) for key in row.keys()})
        path = tables_dir / f"{table_name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(",".join(columns) + "\n")
            for row in rows:
                values = []
                for column in columns:
                    value = row.get(column, "") if isinstance(row, Mapping) else ""
                    if isinstance(value, (dict, list, tuple)):
                        value = json.dumps(value, ensure_ascii=False)
                    text = str(value).replace('"', '""')
                    if any(char in text for char in [",", "\n", '"']):
                        text = f'"{text}"'
                    values.append(text)
                handle.write(",".join(values) + "\n")
        written.append(path)
        txt = write_table_txt(
            tables_dir / f"{table_name}.txt",
            rows,
            columns=columns,
            title=table_name.replace("_", " ").title(),
            metadata={"dataset_name": dataset_name, "dataset_uid": dataset_uid or "Not recorded"},
            backup=False,
        )
        written.append(Path(txt["txt_path"]))

    best = result.get("best_model") or {}
    best_id = best.get("model_id")
    full = next(
        (row.get("refinement") for row in result.get("full_refinement_results", []) if row.get("model_id") == best_id),
        None,
    )
    if isinstance(full, Mapping):
        package = export_refinement_txt_bundle(
            root / "phase19_best_model.txt",
            full,
            dataset_name=dataset_name,
            dataset_uid=dataset_uid,
            refinement_label=f"Phase 19 best model {best_id or ''}".strip(),
            include_reference_image_data=True,
        )
        for key, value in package.items():
            if key.endswith("_txt") and isinstance(value, str):
                path = Path(value)
                if path.exists() and path not in written:
                    written.append(path)

    second_pass = result.get("second_pass_refinement") or {}
    second_refinement = second_pass.get("refinement") if isinstance(second_pass, Mapping) else None
    if isinstance(second_refinement, Mapping):
        package = export_refinement_txt_bundle(
            root / "phase21_second_pass.txt",
            second_refinement,
            dataset_name=dataset_name,
            dataset_uid=dataset_uid,
            refinement_label="Phase 21 reviewed second-pass refinement",
            include_reference_image_data=False,
        )
        for key, value in package.items():
            if key.endswith("_txt") and isinstance(value, str):
                path = Path(value)
                if path.exists() and path not in written:
                    written.append(path)

    readme = root / "README_PHASE19_MULTICOMPONENT_REFINER.txt"
    readme.write_text(
        "Phase 19.5 Intelligent Multiphase Rietveld/Pawley Refiner\n"
        "==========================================================\n\n"
        f"Dataset: {dataset_name}\n"
        f"Best model: {best.get('model_id', 'not available')}\n"
        f"Status: {best.get('status_label', 'not available')}\n"
        f"Rwp (%): {best.get('rwp_percent', 'not available')}\n\n"
        "TXT files are UTF-8, LF-terminated, TAB-delimited and atomically written.\n"
        "The *_profile.txt file contains observed, calculated, background and difference data.\n"
        "The *_bragg_positions.txt file contains phase-resolved Bragg marker positions.\n\n"
        "Scientific boundary: this package ranks user-supplied CIF combinations. "
        "It does not prove phase identity without expert review and accepted refinement.\n",
        encoding="utf-8",
    )
    written.append(readme)

    manifest_txt = write_manifest_txt(
        root / "phase19_manifest.txt",
        written,
        root=root,
        title=f"Phase 19 TXT export manifest — {dataset_name}",
        backup=False,
    )
    written.append(Path(manifest_txt["txt_path"]))

    manifest = []
    import hashlib
    for path in written:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({"relative_path": str(path.relative_to(root)), "sha256": digest, "bytes": path.stat().st_size})
    manifest_path = root / "phase19_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(manifest_path)
    return {"output_dir": str(root), "files": [str(path) for path in written], "manifest": manifest}
