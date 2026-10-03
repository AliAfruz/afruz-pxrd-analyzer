from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np


PEAK_SOURCE_MODES = (
    "Auto: fitted > detected > raw maximum",
    "Fitted peaks only",
    "Detected peaks only",
    "Raw local maximum",
)

REFERENCE_MODES = (
    "Fit d0 from sin²ψ intercept",
    "Use stress-free 2θ",
)

ELASTIC_MODES = (
    "Isotropic E and ν",
    "X-ray elastic constant ½S₂",
)

REGRESSION_MODES = (
    "Weighted linear",
    "Huber robust",
)


class ResidualStressError(ValueError):
    pass


def two_theta_to_d(
    two_theta_deg: float,
    wavelength_angstrom: float,
) -> float:
    two_theta = float(two_theta_deg)
    wavelength = float(wavelength_angstrom)
    if not np.isfinite(two_theta) or not 0.0 < two_theta < 180.0:
        raise ResidualStressError(
            "Peak position must lie between 0 and 180 degrees 2θ."
        )
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ResidualStressError("Wavelength must be positive.")
    theta = math.radians(two_theta / 2.0)
    return float(wavelength / (2.0 * math.sin(theta)))


def d_to_two_theta(
    d_angstrom: float,
    wavelength_angstrom: float,
) -> float:
    d_value = float(d_angstrom)
    wavelength = float(wavelength_angstrom)
    if not np.isfinite(d_value) or d_value <= 0:
        raise ResidualStressError("d-spacing must be positive.")
    ratio = wavelength / (2.0 * d_value)
    if not 0.0 < ratio < 1.0:
        raise ResidualStressError(
            "The wavelength and d-spacing do not define a physical Bragg angle."
        )
    return float(2.0 * math.degrees(math.asin(ratio)))


def d_uncertainty_from_two_theta(
    two_theta_deg: float,
    two_theta_error_deg: float,
    wavelength_angstrom: float,
) -> float:
    d_value = two_theta_to_d(two_theta_deg, wavelength_angstrom)
    error = abs(float(two_theta_error_deg))
    theta = math.radians(float(two_theta_deg) / 2.0)
    derivative = abs(
        d_value
        * (math.cos(theta) / math.sin(theta))
        * math.pi
        / 360.0
    )
    return float(derivative * error)


def infer_psi_deg(metadata: dict | None, dataset_name: str = "") -> tuple[float, str]:
    metadata = metadata or {}
    for key in (
        "psi_deg",
        "psi",
        "tilt_angle_deg",
        "tilt_angle",
        "chi_deg",
        "chi",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            return numeric, f"metadata:{key}"

    name = dataset_name or ""
    patterns = (
        r"(?i)(?:psi|ψ|tilt|chi)\s*[_:=\- ]?\s*([+\-]?\d+(?:\.\d+)?)",
        r"(?i)([+\-]?\d+(?:\.\d+)?)\s*(?:deg|degree|°)\s*(?:psi|ψ|tilt|chi)?",
    )
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return float(match.group(1)), "dataset name"
    return 0.0, "default; edit required"


def _nearest_fitted_peak(
    fit_groups: Iterable[dict],
    target_two_theta_deg: float,
    half_window_deg: float,
) -> dict | None:
    candidates = []
    for group in fit_groups or []:
        for component in group.get("components", []):
            try:
                center = float(component["center"])
            except (KeyError, TypeError, ValueError):
                continue
            distance = abs(center - target_two_theta_deg)
            if distance <= half_window_deg:
                candidates.append(
                    (
                        distance,
                        {
                            "two_theta_deg": center,
                            "two_theta_error_deg": component.get("center_error"),
                            "source": f"Fitted {component.get('model', 'peak')}",
                        },
                    )
                )
    return min(candidates, key=lambda row: row[0])[1] if candidates else None


def _nearest_detected_peak(
    peaks: Iterable[dict],
    target_two_theta_deg: float,
    half_window_deg: float,
) -> dict | None:
    candidates = []
    for peak in peaks or []:
        value = peak.get("position")
        if value is None:
            continue
        try:
            center = float(value)
        except (TypeError, ValueError):
            continue
        distance = abs(center - target_two_theta_deg)
        if distance <= half_window_deg:
            candidates.append(
                (
                    distance,
                    {
                        "two_theta_deg": center,
                        "two_theta_error_deg": None,
                        "source": f"Detected {peak.get('method', 'peak')}",
                    },
                )
            )
    return min(candidates, key=lambda row: row[0])[1] if candidates else None


def _raw_local_maximum(
    x: np.ndarray,
    y: np.ndarray,
    target_two_theta_deg: float,
    half_window_deg: float,
) -> dict | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (
        (x >= target_two_theta_deg - half_window_deg)
        & (x <= target_two_theta_deg + half_window_deg)
    )
    indices = np.flatnonzero(mask)
    if len(indices) < 3:
        return None
    local_index = int(indices[np.argmax(y[indices])])
    center = float(x[local_index])
    source = "Raw local maximum"

    if 0 < local_index < len(x) - 1:
        local_x = x[local_index - 1 : local_index + 2]
        local_y = y[local_index - 1 : local_index + 2]
        try:
            coefficients = np.polyfit(local_x, local_y, 2)
            curvature, slope, _ = coefficients
            if curvature < 0:
                vertex = -slope / (2.0 * curvature)
                if local_x[0] <= vertex <= local_x[-1]:
                    center = float(vertex)
                    source = "Raw maximum + parabolic refinement"
        except (ValueError, np.linalg.LinAlgError):
            pass

    return {
        "two_theta_deg": center,
        "two_theta_error_deg": None,
        "source": source,
    }


def extract_peak_observation(
    x: np.ndarray,
    y: np.ndarray,
    target_two_theta_deg: float,
    half_window_deg: float,
    source_mode: str = PEAK_SOURCE_MODES[0],
    fit_groups: Iterable[dict] | None = None,
    detected_peaks: Iterable[dict] | None = None,
    default_error_deg: float = 0.01,
) -> dict | None:
    target = float(target_two_theta_deg)
    half_window = float(half_window_deg)
    if half_window <= 0:
        raise ResidualStressError("Peak search half-window must be positive.")
    if source_mode not in PEAK_SOURCE_MODES:
        raise ResidualStressError(f"Unsupported peak source mode: {source_mode}")

    result = None
    if source_mode in (
        "Auto: fitted > detected > raw maximum",
        "Fitted peaks only",
    ):
        result = _nearest_fitted_peak(fit_groups or [], target, half_window)
    if result is None and source_mode in (
        "Auto: fitted > detected > raw maximum",
        "Detected peaks only",
    ):
        result = _nearest_detected_peak(detected_peaks or [], target, half_window)
    if result is None and source_mode in (
        "Auto: fitted > detected > raw maximum",
        "Raw local maximum",
    ):
        result = _raw_local_maximum(x, y, target, half_window)
    if result is None:
        return None

    error = result.get("two_theta_error_deg")
    try:
        error = float(error)
    except (TypeError, ValueError):
        error = float(default_error_deg)
    if not np.isfinite(error) or error <= 0:
        error = float(default_error_deg)
    result["two_theta_error_deg"] = error
    result["target_delta_deg"] = float(result["two_theta_deg"] - target)
    return result


def _linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    y_error: np.ndarray | None,
    robust: bool,
) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([x, np.ones_like(x)])

    base_weights = np.ones_like(y)
    weighted = False
    if (
        y_error is not None
        and len(y_error) == len(y)
        and np.all(np.isfinite(y_error))
        and np.all(y_error > 0)
    ):
        base_weights = 1.0 / np.square(np.asarray(y_error, dtype=float))
        weighted = True

    robust_weights = np.ones_like(y)
    coefficients = np.zeros(2, dtype=float)
    for _ in range(30 if robust else 1):
        weights = base_weights * robust_weights
        normal = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * y)
        coefficients = np.linalg.solve(normal, rhs)
        residual = y - design @ coefficients
        if not robust:
            break
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        scale = max(mad / 0.6744897501960817, np.finfo(float).eps)
        standardized = np.abs(residual) / scale
        new_robust = np.ones_like(standardized)
        mask = standardized > 1.345
        new_robust[mask] = 1.345 / standardized[mask]
        if np.max(np.abs(new_robust - robust_weights)) < 1e-6:
            robust_weights = new_robust
            break
        robust_weights = new_robust

    weights = base_weights * robust_weights
    fitted = design @ coefficients
    residual = y - fitted
    degrees_of_freedom = max(1, len(y) - 2)
    weighted_rss = float(np.sum(weights * residual * residual))
    residual_variance = weighted_rss / degrees_of_freedom
    covariance = np.linalg.inv(
        design.T @ (weights[:, None] * design)
    ) * residual_variance

    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "slope": float(coefficients[0]),
        "intercept": float(coefficients[1]),
        "slope_error": float(math.sqrt(max(0.0, covariance[0, 0]))),
        "intercept_error": float(math.sqrt(max(0.0, covariance[1, 1]))),
        "slope_intercept_covariance": float(covariance[0, 1]),
        "covariance": covariance.tolist(),
        "fitted": fitted,
        "residual": residual,
        "r_squared": float(r_squared),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "weighted": bool(weighted),
        "robust": bool(robust),
        "robust_weights": robust_weights,
    }


def analyze_sin2psi(
    observations: list[dict],
    wavelength_angstrom: float,
    reference_mode: str = REFERENCE_MODES[0],
    stress_free_two_theta_deg: float | None = None,
    elastic_mode: str = ELASTIC_MODES[0],
    youngs_modulus_gpa: float = 200.0,
    poisson_ratio: float = 0.30,
    xec_half_s2_per_gpa: float = 0.0058,
    regression_mode: str = REGRESSION_MODES[0],
    azimuth_deg: float = 0.0,
) -> dict:
    if reference_mode not in REFERENCE_MODES:
        raise ResidualStressError(f"Unsupported reference mode: {reference_mode}")
    if elastic_mode not in ELASTIC_MODES:
        raise ResidualStressError(f"Unsupported elastic mode: {elastic_mode}")
    if regression_mode not in REGRESSION_MODES:
        raise ResidualStressError(f"Unsupported regression mode: {regression_mode}")

    wavelength = float(wavelength_angstrom)
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ResidualStressError("Wavelength must be positive.")

    included = []
    for row in observations:
        if not row.get("included", True):
            continue
        try:
            psi = float(row["psi_deg"])
            two_theta = float(row["two_theta_deg"])
            error = float(row.get("two_theta_error_deg", 0.01))
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(psi) and np.isfinite(two_theta) and np.isfinite(error)):
            continue
        if error <= 0:
            continue
        d_value = two_theta_to_d(two_theta, wavelength)
        d_error = d_uncertainty_from_two_theta(two_theta, error, wavelength)
        included.append(
            {
                **row,
                "psi_deg": psi,
                "sin2psi": float(math.sin(math.radians(psi)) ** 2),
                "two_theta_deg": two_theta,
                "two_theta_error_deg": error,
                "d_angstrom": d_value,
                "d_error_angstrom": d_error,
            }
        )

    if len(included) < 4:
        raise ResidualStressError(
            "At least four included ψ observations are required."
        )

    x = np.asarray([row["sin2psi"] for row in included], dtype=float)
    if len(np.unique(np.round(x, 8))) < 3:
        raise ResidualStressError(
            "At least three unique sin²ψ values are required."
        )
    x_span = float(np.max(x) - np.min(x))
    if x_span < 0.05:
        raise ResidualStressError(
            "The sin²ψ range is too narrow for a stable stress fit."
        )

    d = np.asarray([row["d_angstrom"] for row in included], dtype=float)
    d_error = np.asarray(
        [row["d_error_angstrom"] for row in included], dtype=float
    )
    robust = regression_mode == "Huber robust"

    if reference_mode == "Fit d0 from sin²ψ intercept":
        regression_d = _linear_regression(x, d, d_error, robust)
        d0 = float(regression_d["intercept"])
        d0_error = float(regression_d["intercept_error"])
        if d0 <= 0:
            raise ResidualStressError(
                "The fitted stress-free d-spacing is not positive."
            )
        slope_d = float(regression_d["slope"])
        slope_strain = slope_d / d0
        covariance_sd = float(regression_d["slope_intercept_covariance"])
        variance_slope_strain = (
            regression_d["slope_error"] ** 2 / d0**2
            + slope_d**2 * d0_error**2 / d0**4
            - 2.0 * slope_d * covariance_sd / d0**3
        )
        slope_strain_error = float(
            math.sqrt(max(0.0, variance_slope_strain))
        )
        strain = (d - d0) / d0
        strain_error = d_error / d0
        fitted_strain = slope_strain * x
        strain_intercept = 0.0
        strain_intercept_error = d0_error / d0
        residual = strain - fitted_strain
        r_squared = float(regression_d["r_squared"])
        weighted = bool(regression_d["weighted"])
        robust_weights = np.asarray(regression_d["robust_weights"], dtype=float)
    else:
        if stress_free_two_theta_deg is None:
            raise ResidualStressError(
                "Enter the stress-free 2θ value for the selected reflection."
            )
        d0 = two_theta_to_d(float(stress_free_two_theta_deg), wavelength)
        d0_error = None
        strain = (d - d0) / d0
        strain_error = d_error / d0
        regression_strain = _linear_regression(
            x, strain, strain_error, robust
        )
        slope_strain = float(regression_strain["slope"])
        slope_strain_error = float(regression_strain["slope_error"])
        strain_intercept = float(regression_strain["intercept"])
        strain_intercept_error = float(regression_strain["intercept_error"])
        fitted_strain = np.asarray(regression_strain["fitted"], dtype=float)
        residual = np.asarray(regression_strain["residual"], dtype=float)
        r_squared = float(regression_strain["r_squared"])
        weighted = bool(regression_strain["weighted"])
        robust_weights = np.asarray(
            regression_strain["robust_weights"], dtype=float
        )

    if elastic_mode == "Isotropic E and ν":
        youngs = float(youngs_modulus_gpa)
        poisson = float(poisson_ratio)
        if not np.isfinite(youngs) or youngs <= 0:
            raise ResidualStressError("Young's modulus must be positive.")
        if not np.isfinite(poisson) or not -0.99 < poisson < 0.5:
            raise ResidualStressError(
                "Poisson ratio must lie between -0.99 and 0.5."
            )
        stress_gpa = slope_strain * youngs / (1.0 + poisson)
        stress_error_gpa = (
            slope_strain_error * youngs / (1.0 + poisson)
        )
        elastic_factor = youngs / (1.0 + poisson)
        elastic_factor_description = "E/(1+ν)"
    else:
        half_s2 = float(xec_half_s2_per_gpa)
        if not np.isfinite(half_s2) or half_s2 <= 0:
            raise ResidualStressError(
                "The X-ray elastic constant ½S₂ must be positive."
            )
        stress_gpa = slope_strain / half_s2
        stress_error_gpa = slope_strain_error / half_s2
        elastic_factor = 1.0 / half_s2
        elastic_factor_description = "1/(½S₂)"

    residual_microstrain = residual * 1e6
    rmse_microstrain = float(
        np.sqrt(np.mean(np.square(residual_microstrain)))
    )
    standardized = residual / np.maximum(strain_error, np.finfo(float).eps)

    warnings = []
    if x_span < 0.20:
        warnings.append(
            "Limited sin²ψ span; use larger tilt angles when geometry permits."
        )
    if max(abs(row["psi_deg"]) for row in included) < 30.0:
        warnings.append("Maximum |ψ| is below 30°.")
    if r_squared < 0.90:
        warnings.append("The sin²ψ relation has R² below 0.90.")
    if np.any(np.abs(standardized) > 3.0):
        warnings.append("One or more observations exceed 3σ residual.")
    if robust and np.any(robust_weights < 0.5):
        warnings.append("Huber regression strongly down-weighted observations.")

    pair_splitting = []
    for index, row in enumerate(included):
        psi = float(row["psi_deg"])
        if psi <= 0:
            continue
        candidates = [
            other
            for other in included
            if abs(float(other["psi_deg"]) + psi) <= 0.15
        ]
        if not candidates:
            continue
        other = min(
            candidates,
            key=lambda candidate: abs(float(candidate["psi_deg"]) + psi),
        )
        difference = abs(
            float(row["two_theta_deg"]) - float(other["two_theta_deg"])
        )
        threshold = max(
            0.02,
            3.0
            * math.sqrt(
                float(row["two_theta_error_deg"]) ** 2
                + float(other["two_theta_error_deg"]) ** 2
            ),
        )
        pair_splitting.append(
            {
                "absolute_psi_deg": abs(psi),
                "two_theta_difference_deg": difference,
                "threshold_deg": threshold,
                "flagged": difference > threshold,
            }
        )
    if any(row["flagged"] for row in pair_splitting):
        warnings.append(
            "Positive/negative ψ splitting is larger than expected; shear stress, "
            "texture, alignment, or peak-selection effects may be present."
        )

    for index, row in enumerate(included):
        row["strain"] = float(strain[index])
        row["strain_error"] = float(strain_error[index])
        row["fitted_strain"] = float(fitted_strain[index])
        row["residual_microstrain"] = float(residual_microstrain[index])
        row["standardized_residual"] = float(standardized[index])
        row["regression_weight"] = float(robust_weights[index])
        row["status"] = (
            "Outlier warning"
            if abs(float(standardized[index])) > 3.0
            else "Used"
        )

    return {
        "method": "sin²ψ residual stress",
        "wavelength_angstrom": wavelength,
        "reference_mode": reference_mode,
        "stress_free_two_theta_deg": stress_free_two_theta_deg,
        "elastic_mode": elastic_mode,
        "youngs_modulus_gpa": float(youngs_modulus_gpa),
        "poisson_ratio": float(poisson_ratio),
        "xec_half_s2_per_gpa": float(xec_half_s2_per_gpa),
        "elastic_factor_gpa": float(elastic_factor),
        "elastic_factor_description": elastic_factor_description,
        "regression_mode": regression_mode,
        "azimuth_deg": float(azimuth_deg),
        "d0_angstrom": float(d0),
        "d0_error_angstrom": d0_error,
        "slope_strain": float(slope_strain),
        "slope_strain_error": float(slope_strain_error),
        "strain_intercept": float(strain_intercept),
        "strain_intercept_error": float(strain_intercept_error),
        "stress_mpa": float(stress_gpa * 1000.0),
        "stress_error_mpa": float(stress_error_gpa * 1000.0),
        "stress_sign_convention": (
            "Positive slope/stress is tensile; negative is compressive."
        ),
        "r_squared": r_squared,
        "rmse_microstrain": rmse_microstrain,
        "weighted": weighted,
        "point_count": len(included),
        "psi_min_deg": float(min(row["psi_deg"] for row in included)),
        "psi_max_deg": float(max(row["psi_deg"] for row in included)),
        "sin2psi_min": float(np.min(x)),
        "sin2psi_max": float(np.max(x)),
        "sin2psi_span": x_span,
        "observations": included,
        "pair_splitting": pair_splitting,
        "warnings": warnings,
    }
