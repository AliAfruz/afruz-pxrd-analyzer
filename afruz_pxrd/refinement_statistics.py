from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

import numpy as np


STATISTICAL_PROVENANCE = (
    "raw_counts",
    "count_rate",
    "scaled_counts",
    "processed_correlated",
    "arbitrary_units",
    "unknown",
)


@dataclass(frozen=True)
class RefinementWeightModel:
    """Weights used for fitting and for crystallographic R statistics.

    ``fit_sqrt_weights`` may be normalized by a constant because a global
    multiplier does not change a least-squares optimum.  ``reporting_weights``
    are never normalized: Rexp and reduced chi-square depend on their absolute
    scale and become meaningless if an arbitrary normalization is applied.
    """

    fit_sqrt_weights: np.ndarray
    reporting_weights: np.ndarray
    sigma: np.ndarray | None
    statistics_valid: bool
    rexp_available: bool
    model_name: str
    provenance: str
    reason: str
    warnings: tuple[str, ...]
    intensity_scale_factor: float


def _validated_array(values: Iterable[float] | np.ndarray, length: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{label} must be a one-dimensional array matching the observed pattern.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values.")
    return array


def _normalize_fit_weights(sqrt_weights: np.ndarray) -> np.ndarray:
    """Normalize only the optimization weights, never the statistical weights."""
    values = np.asarray(sqrt_weights, dtype=float)
    finite = values[np.isfinite(values) & (values > 0)]
    if not finite.size:
        return np.ones_like(values)
    rms = float(math.sqrt(np.mean(finite * finite)))
    if not np.isfinite(rms) or rms <= 0:
        return np.ones_like(values)
    return values / rms


def _empirical_poisson_proxy(y: np.ndarray) -> np.ndarray:
    shifted = y - float(np.min(y))
    positive = shifted[shifted > 0]
    floor = float(np.percentile(positive, 10.0)) if positive.size else 1.0
    floor = max(floor, 1e-12)
    variance = np.maximum(shifted, floor)
    return 1.0 / variance


def build_refinement_weight_model(
    y: Iterable[float] | np.ndarray,
    mode: str,
    *,
    observed_sigma: Iterable[float] | np.ndarray | None = None,
    count_reference: Iterable[float] | np.ndarray | None = None,
    intensity_scale_factor: float = 1.0,
    provenance: str = "raw_counts",
    statistics_note: str | None = None,
) -> RefinementWeightModel:
    """Create fitting and reporting weights without corrupting Rexp.

    Parameters
    ----------
    y
        Observed intensities in the units used by the refinement.
    mode
        ``Poisson-like``, ``Balanced`` or ``Uniform``.
    observed_sigma
        Optional one-sigma uncertainties in the same units as ``y``.
    count_reference
        Raw detector counts before any scalar normalization.  This permits
        exact propagation for a pure scaling transformation.
    intensity_scale_factor
        Multiplicative factor converting count-reference units to ``y`` units.
    provenance
        Describes whether the input is raw/scaled counts or a transformed
        pattern with correlations introduced by smoothing/background work.
    """
    observed = np.asarray(y, dtype=float)
    if observed.ndim != 1 or not len(observed):
        raise ValueError("Observed intensities must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(observed)):
        raise ValueError("Observed intensities contain non-finite values.")
    if provenance not in STATISTICAL_PROVENANCE:
        provenance = "unknown"

    warnings: list[str] = []
    scale = abs(float(intensity_scale_factor))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
        warnings.append("Invalid intensity scale factor was replaced by 1.0 for weighting.")

    if observed_sigma is not None:
        sigma = _validated_array(observed_sigma, len(observed), "Observed sigma")
        bad = sigma <= 0
        if np.any(bad):
            positive = sigma[sigma > 0]
            replacement = float(np.min(positive)) if positive.size else 1.0
            sigma = sigma.copy()
            sigma[bad] = replacement
            warnings.append("Non-positive supplied uncertainties were replaced by the smallest positive sigma.")
        reporting = 1.0 / np.square(sigma)
        return RefinementWeightModel(
            fit_sqrt_weights=_normalize_fit_weights(1.0 / sigma),
            reporting_weights=reporting,
            sigma=sigma,
            statistics_valid=True,
            rexp_available=True,
            model_name="Supplied standard uncertainties",
            provenance=provenance,
            reason=statistics_note or "Rexp is based on supplied one-sigma uncertainties.",
            warnings=tuple(warnings),
            intensity_scale_factor=scale,
        )

    if mode == "Uniform":
        reporting = np.ones_like(observed)
        reason = statistics_note or (
            "Uniform weights are an optimization choice, not a counting-statistics model; "
            "Rexp and chi-square are not statistically defined."
        )
        return RefinementWeightModel(
            fit_sqrt_weights=np.ones_like(observed),
            reporting_weights=reporting,
            sigma=None,
            statistics_valid=False,
            rexp_available=False,
            model_name="Uniform empirical weights",
            provenance=provenance,
            reason=reason,
            warnings=tuple(warnings),
            intensity_scale_factor=scale,
        )

    if mode == "Balanced":
        shifted = observed - float(np.min(observed))
        positive = shifted[shifted > 0]
        floor = float(np.percentile(positive, 10.0)) if positive.size else 1.0
        floor = max(floor, 1e-12)
        high = max(float(np.percentile(shifted, 95.0)), floor)
        sqrt_weights = 1.0 / np.sqrt(0.08 + shifted / high)
        reporting = np.square(sqrt_weights)
        reason = statistics_note or (
            "Balanced weights deliberately redistribute influence across the pattern; "
            "they do not represent detector variances, so Rexp and chi-square are unavailable."
        )
        return RefinementWeightModel(
            fit_sqrt_weights=_normalize_fit_weights(sqrt_weights),
            reporting_weights=reporting,
            sigma=None,
            statistics_valid=False,
            rexp_available=False,
            model_name="Balanced empirical weights",
            provenance=provenance,
            reason=reason,
            warnings=tuple(warnings),
            intensity_scale_factor=scale,
        )

    if mode != "Poisson-like":
        raise ValueError(f"Unsupported refinement weighting mode: {mode}")

    counts = None
    if count_reference is not None:
        counts = _validated_array(count_reference, len(observed), "Count reference")
        negative_count_fraction = float(np.mean(counts < 0))
        if negative_count_fraction:
            warnings.append(
                f"{100.0 * negative_count_fraction:.4g}% of count-reference points were negative; "
                "their Poisson variance was floored at one count."
            )
        variance_counts = np.maximum(counts, 1.0)
        sigma = scale * np.sqrt(variance_counts)
        reporting = 1.0 / np.square(sigma)
        valid = provenance in {"raw_counts", "count_rate", "scaled_counts"}
        if valid:
            if provenance == "raw_counts":
                default_reason = "Poisson variance was calculated from raw detector counts."
            elif provenance == "count_rate":
                default_reason = "Poisson variance was propagated from counts and counting time into count-rate units."
            else:
                default_reason = "Poisson variance was exactly propagated through a scalar intensity normalization."
        else:
            default_reason = (
                "Raw counts were used only to stabilize fitting weights. Smoothing, background estimation, clipping, "
                "or another nonlinear transformation changed the covariance, so Rexp and chi-square are unavailable."
            )
        return RefinementWeightModel(
            fit_sqrt_weights=_normalize_fit_weights(1.0 / sigma),
            reporting_weights=reporting,
            sigma=sigma,
            statistics_valid=valid,
            rexp_available=valid,
            model_name=(
                "Poisson counting statistics"
                if valid
                else "Poisson-derived fit weights; transformed-data statistics invalid"
            ),
            provenance=provenance,
            reason=statistics_note or default_reason,
            warnings=tuple(warnings),
            intensity_scale_factor=scale,
        )

    if provenance == "raw_counts":
        negative_fraction = float(np.mean(observed < 0))
        if negative_fraction:
            warnings.append(
                f"{100.0 * negative_fraction:.4g}% of observed points were negative; "
                "their Poisson variance was floored at one count."
            )
        sigma = np.sqrt(np.maximum(observed, 1.0))
        reporting = 1.0 / np.square(sigma)
        return RefinementWeightModel(
            fit_sqrt_weights=_normalize_fit_weights(1.0 / sigma),
            reporting_weights=reporting,
            sigma=sigma,
            statistics_valid=True,
            rexp_available=True,
            model_name="Poisson counting statistics",
            provenance=provenance,
            reason=statistics_note or "Observed intensities were treated as raw detector counts.",
            warnings=tuple(warnings),
            intensity_scale_factor=1.0,
        )

    reporting = _empirical_poisson_proxy(observed)
    reason = statistics_note or (
        "No raw count reference or supplied uncertainties were available. Poisson-like relative weights were used "
        "for optimization only; Rexp and chi-square are unavailable."
    )
    return RefinementWeightModel(
        fit_sqrt_weights=_normalize_fit_weights(np.sqrt(reporting)),
        reporting_weights=reporting,
        sigma=None,
        statistics_valid=False,
        rexp_available=False,
        model_name="Poisson-like empirical fit weights",
        provenance=provenance,
        reason=reason,
        warnings=tuple(warnings),
        intensity_scale_factor=scale,
    )


def calculate_profile_statistics(
    observed: np.ndarray,
    calculated: np.ndarray,
    weight_model: RefinementWeightModel,
    parameter_count: int,
) -> dict:
    observed = np.asarray(observed, dtype=float)
    calculated = np.asarray(calculated, dtype=float)
    if observed.shape != calculated.shape:
        raise ValueError("Observed and calculated arrays must have identical shapes.")
    residual = observed - calculated
    weights = np.asarray(weight_model.reporting_weights, dtype=float)
    if weights.shape != observed.shape:
        raise ValueError("Reporting weights must match the observed pattern.")

    denominator = max(float(np.sum(weights * observed * observed)), 1e-30)
    weighted_sum_squares = float(np.sum(weights * residual * residual))
    rwp = 100.0 * math.sqrt(weighted_sum_squares / denominator)
    rp = 100.0 * float(np.sum(np.abs(residual))) / max(float(np.sum(np.abs(observed))), 1e-30)
    degrees_of_freedom = max(1, len(observed) - int(parameter_count))

    if weight_model.rexp_available:
        rexp = 100.0 * math.sqrt(degrees_of_freedom / denominator)
        reduced_chi_square = weighted_sum_squares / degrees_of_freedom
        gof_sqrt = math.sqrt(max(reduced_chi_square, 0.0))
    else:
        rexp = None
        reduced_chi_square = None
        gof_sqrt = None

    return {
        "rwp_percent": float(rwp),
        "rp_percent": float(rp),
        "rexp_percent": None if rexp is None else float(rexp),
        "reduced_chi_square": (
            None if reduced_chi_square is None else float(reduced_chi_square)
        ),
        "goodness_of_fit_sqrt": None if gof_sqrt is None else float(gof_sqrt),
        # Backward compatibility: existing Afruz project files used this key
        # for reduced chi-square rather than its square root.
        "goodness_of_fit": (
            None if reduced_chi_square is None else float(reduced_chi_square)
        ),
        "weighted_residual_sum_squares": weighted_sum_squares,
        "weighted_observed_sum_squares": denominator,
        "degrees_of_freedom": int(degrees_of_freedom),
    }


def infer_dataset_statistical_input(
    raw_y: Iterable[float] | np.ndarray,
    observed_y: Iterable[float] | np.ndarray,
    *,
    intensity_unit: str = "",
    counting_time_s: float | None = None,
    interpretation: str = "Auto from input metadata",
    known_processing_scale: float | None = None,
) -> dict:
    """Infer count provenance for the GUI without pretending arbitrary units are counts."""
    raw = np.asarray(raw_y, dtype=float)
    observed = np.asarray(observed_y, dtype=float)
    if raw.shape != observed.shape or raw.ndim != 1:
        raise ValueError("Raw and selected observed intensities must be matching one-dimensional arrays.")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(observed)):
        raise ValueError("Intensity arrays contain non-finite values.")

    normalized_interpretation = str(interpretation or "").strip().lower()
    unit = str(intensity_unit or "").strip().lower().replace(" ", "")
    force_counts = normalized_interpretation.startswith("treat as poisson")
    force_invalid = normalized_interpretation.startswith("no valid")

    is_count_rate = any(token in unit for token in ("counts/s", "count/s", "cps", "countpersecond"))
    is_counts = (
        unit in {"count", "counts", "cts"}
        or "detectorcount" in unit
        or unit.endswith("_counts")
    ) and not is_count_rate

    count_reference = None
    base_scale = 1.0
    base_provenance = "arbitrary_units"
    base_reason = "Input intensity units are not identified as detector counts."

    if force_invalid:
        base_reason = "The user explicitly selected no valid counting-statistics model."
    elif force_counts:
        count_reference = raw.copy()
        base_provenance = "raw_counts"
        base_reason = "The user explicitly declared the raw intensity column to be detector counts."
    elif is_count_rate and counting_time_s is not None and float(counting_time_s) > 0:
        time_s = float(counting_time_s)
        count_reference = np.maximum(raw * time_s, 0.0)
        base_scale = 1.0 / time_s
        base_provenance = "count_rate"
        base_reason = "Count-rate uncertainties were reconstructed from the recorded counting time."
    elif is_count_rate:
        base_reason = "The input is a count rate, but no common counting time is available to reconstruct counts."
    elif is_counts:
        count_reference = raw.copy()
        base_provenance = "raw_counts"
        base_reason = "The file metadata identifies the raw intensity column as detector counts."

    denominator = float(np.dot(raw, raw))
    exact_factor = 1.0
    exact_scaled = bool(np.array_equal(observed, raw))
    if denominator > 0:
        exact_factor = float(np.dot(observed, raw) / denominator)
        mismatch = observed - exact_factor * raw
        relative_error = float(np.linalg.norm(mismatch) / max(np.linalg.norm(observed), 1e-30))
        exact_scaled = bool(exact_factor > 0 and relative_error <= 1e-8)

    if count_reference is None:
        return {
            "count_reference": None,
            "intensity_scale_factor": 1.0,
            "intensity_provenance": "arbitrary_units",
            "statistics_note": base_reason,
            "statistics_expected_valid": False,
            "selected_is_pure_scale": exact_scaled,
        }

    if np.array_equal(observed, raw):
        return {
            "count_reference": count_reference,
            "intensity_scale_factor": base_scale,
            "intensity_provenance": base_provenance,
            "statistics_note": base_reason,
            "statistics_expected_valid": True,
            "selected_is_pure_scale": True,
        }

    if exact_scaled:
        return {
            "count_reference": count_reference,
            "intensity_scale_factor": base_scale * exact_factor,
            "intensity_provenance": "scaled_counts",
            "statistics_note": (
                base_reason
                + f" A pure scalar transformation ({exact_factor:.8g}) was detected and the variance was propagated exactly."
            ),
            "statistics_expected_valid": True,
            "selected_is_pure_scale": True,
        }

    scale = known_processing_scale
    if scale is None or not np.isfinite(float(scale)) or float(scale) <= 0:
        raw_high = float(np.percentile(np.abs(raw), 95.0))
        observed_high = float(np.percentile(np.abs(observed), 95.0))
        scale = (
            base_scale * observed_high / raw_high
            if raw_high > 0 and observed_high > 0
            else base_scale
        )
    else:
        scale = base_scale * float(scale)

    return {
        "count_reference": count_reference,
        "intensity_scale_factor": float(scale),
        "intensity_provenance": "processed_correlated",
        "statistics_note": (
            base_reason
            + " The selected processed pattern is not a pure scalar copy of the raw data. "
            "Poisson-derived weights may stabilize fitting, but smoothing/background/clipping covariance is not propagated."
        ),
        "statistics_expected_valid": False,
        "selected_is_pure_scale": False,
    }
