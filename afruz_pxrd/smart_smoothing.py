from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter1d,
    median_filter,
    percentile_filter,
)
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.sparse.linalg import spsolve


SMOOTHING_METHODS = (
    "Auto intelligent",
    "Savitzky–Golay",
    "Gaussian",
    "Whittaker–Eilers",
    "Adaptive bilateral",
    "Median–Gaussian hybrid",
)


class SmartSmoothingError(ValueError):
    pass


@dataclass(frozen=True)
class SmoothingParameters:
    method: str = "Auto intelligent"
    strength: float = 45.0
    peak_protection: bool = True
    peak_preservation: float = 85.0
    protection_sigma: float = 4.0
    protection_width_multiplier: float = 2.0
    maximum_position_shift_deg: float = 0.02
    maximum_height_change_percent: float = 8.0
    maximum_fwhm_change_percent: float = 10.0
    automatic_trials_per_method: int = 3

    def normalized(self) -> "SmoothingParameters":
        method = self.method if self.method in SMOOTHING_METHODS else "Auto intelligent"
        return SmoothingParameters(
            method=method,
            strength=float(np.clip(self.strength, 0.0, 100.0)),
            peak_protection=bool(self.peak_protection),
            peak_preservation=float(np.clip(self.peak_preservation, 0.0, 100.0)),
            protection_sigma=float(np.clip(self.protection_sigma, 2.0, 20.0)),
            protection_width_multiplier=float(
                np.clip(self.protection_width_multiplier, 0.75, 8.0)
            ),
            maximum_position_shift_deg=float(
                np.clip(self.maximum_position_shift_deg, 1e-5, 2.0)
            ),
            maximum_height_change_percent=float(
                np.clip(self.maximum_height_change_percent, 0.25, 100.0)
            ),
            maximum_fwhm_change_percent=float(
                np.clip(self.maximum_fwhm_change_percent, 0.25, 100.0)
            ),
            automatic_trials_per_method=int(
                np.clip(self.automatic_trials_per_method, 1, 7)
            ),
        )


def _validate_arrays(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise SmartSmoothingError(
            "X and Y must be one-dimensional arrays of equal length."
        )
    if len(x) < 11:
        raise SmartSmoothingError(
            "Advanced smoothing requires at least 11 pattern points."
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise SmartSmoothingError("Advanced smoothing requires finite X and Y values.")
    differences = np.diff(x)
    if np.any(differences <= 0):
        raise SmartSmoothingError("Two-theta values must increase strictly.")
    return x, y


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float(np.finfo(float).eps)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    sigma = mad / 0.6744897501960817
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        sigma = float(np.std(values))
    return max(sigma, float(np.finfo(float).eps))


def _odd_window(value: float, minimum: int, maximum: int) -> int:
    maximum = max(3, int(maximum))
    if maximum % 2 == 0:
        maximum -= 1
    window = max(int(minimum), int(round(value)))
    if window % 2 == 0:
        window += 1
    return max(3, min(window, maximum))


def _noise_sigma(y: np.ndarray, excluded_mask: np.ndarray | None = None) -> float:
    differences = np.diff(np.asarray(y, dtype=float)) / math.sqrt(2.0)
    if excluded_mask is not None and len(excluded_mask) == len(y):
        valid = ~(excluded_mask[:-1] | excluded_mask[1:])
        if np.count_nonzero(valid) >= 20:
            differences = differences[valid]
    centered = differences - np.median(differences)
    # MAD is already robust to the relatively sparse diffraction-peak edges.
    # Trimming the central distribution biases the noise estimate downward and
    # causes random detector fluctuations to be misclassified as real peaks.
    return _robust_sigma(centered)


def _detect_peak_regions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    protection_sigma: float,
    width_multiplier: float,
) -> tuple[np.ndarray, list[dict], float]:
    point_count = len(y)
    maximum_odd = point_count if point_count % 2 else point_count - 1
    broad_window = _odd_window(point_count * 0.04, 31, min(501, maximum_odd))
    baseline = percentile_filter(y, percentile=20.0, size=broad_window, mode="nearest")
    raw_signal = y - baseline
    noise = _noise_sigma(raw_signal)
    # Peak protection is detected on a very mildly filtered copy. This rejects
    # isolated counting-noise spikes while leaving ordinary PXRD peak widths
    # and centroids essentially unchanged.
    signal = gaussian_filter1d(raw_signal, sigma=1.15, mode="nearest")
    dynamic = max(float(np.max(signal) - np.min(signal)), noise)
    prominence = max(protection_sigma * noise, 0.008 * dynamic)
    distance = max(2, int(round(point_count * 0.001)))
    indices, properties = find_peaks(
        signal,
        prominence=prominence,
        height=max(2.5 * noise, 0.002 * dynamic),
        width=2.5,
        distance=distance,
    )
    mask = np.zeros(point_count, dtype=bool)
    peaks: list[dict] = []
    if not len(indices):
        return mask, peaks, noise

    widths, _, left_ips, right_ips = peak_widths(signal, indices, rel_height=0.5)
    for order, index in enumerate(indices):
        half_span = max(
            2.0,
            0.5 * float(widths[order]) * max(1.0, width_multiplier),
        )
        left = max(0, int(math.floor(float(index) - half_span)))
        right = min(point_count - 1, int(math.ceil(float(index) + half_span)))
        mask[left : right + 1] = True
        peaks.append(
            {
                "index": int(index),
                "two_theta_deg": float(x[index]),
                "prominence": float(properties["prominences"][order]),
                "width_points": float(widths[order]),
                "left_index": left,
                "right_index": right,
                "left_ip": float(left_ips[order]),
                "right_ip": float(right_ips[order]),
            }
        )
    return mask, peaks, noise


def detect_smart_peak_regions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    minimum_signal_to_noise: float = 4.0,
    width_multiplier: float = 1.5,
) -> dict:
    """Return noise-aware peak regions for reuse by discovery workflows.

    This uses the same conservative detector as Smart Smoothing, but returns the
    peak mask and numerical diagnostics without modifying the supplied pattern.
    """
    x, y = _validate_arrays(x, y)
    mask, peaks, noise = _detect_peak_regions(
        x,
        y,
        protection_sigma=max(2.0, float(minimum_signal_to_noise)),
        width_multiplier=max(0.75, float(width_multiplier)),
    )
    dynamic = max(float(np.ptp(y)), noise)
    enriched = []
    for peak in peaks:
        row = dict(peak)
        prominence = float(row.get("prominence", 0.0))
        snr = prominence / max(noise, np.finfo(float).eps)
        prominence_fraction = prominence / max(dynamic, np.finfo(float).eps)
        quality = 100.0 * (
            0.68 * (1.0 - math.exp(-max(0.0, snr - 1.0) / 5.0))
            + 0.32 * math.sqrt(max(0.0, min(1.0, prominence_fraction)))
        )
        row["signal_to_noise"] = float(snr)
        row["quality_score"] = float(np.clip(quality, 0.0, 100.0))
        enriched.append(row)
    return {
        "mask": mask,
        "peaks": enriched,
        "noise_sigma": float(noise),
        "dynamic_range": float(dynamic),
        "method": "Smart Smoothing noise-aware detector",
    }


def _savgol(y: np.ndarray, strength: float) -> np.ndarray:
    maximum = len(y) if len(y) % 2 else len(y) - 1
    maximum = min(maximum, max(9, int(len(y) * 0.025) | 1), 301)
    window = _odd_window(5 + (strength / 100.0) ** 1.4 * (maximum - 5), 5, maximum)
    order = min(3, window - 2)
    return savgol_filter(y, window, order, mode="interp")


def _gaussian(y: np.ndarray, strength: float) -> np.ndarray:
    sigma = 0.25 + (strength / 100.0) ** 1.45 * 5.0
    return gaussian_filter1d(y, sigma=sigma, mode="nearest")


def _whittaker(y: np.ndarray, strength: float) -> np.ndarray:
    point_count = len(y)
    # 10^0.5 to 10^6 gives useful control across typical PXRD scan densities.
    lam = 10.0 ** (0.5 + 5.5 * strength / 100.0)
    difference = sparse.diags(
        [np.ones(point_count - 2), -2.0 * np.ones(point_count - 2), np.ones(point_count - 2)],
        [0, 1, 2],
        shape=(point_count - 2, point_count),
        format="csc",
    )
    system = sparse.eye(point_count, format="csc") + lam * (difference.T @ difference)
    result = spsolve(system, y)
    return np.asarray(result, dtype=float)


def _bilateral(y: np.ndarray, strength: float, noise_sigma: float) -> np.ndarray:
    radius = max(2, int(round(2 + 13 * strength / 100.0)))
    sigma_spatial = max(1.0, radius / 2.2)
    dynamic = max(float(np.ptp(y)), noise_sigma)
    sigma_intensity = max(
        1.5 * noise_sigma,
        dynamic * (0.002 + 0.025 * strength / 100.0),
    )
    offsets = np.arange(-radius, radius + 1, dtype=float)
    spatial_weights = np.exp(-0.5 * np.square(offsets / sigma_spatial))
    padded = np.pad(y, radius, mode="edge")
    result = np.empty_like(y)
    for index in range(len(y)):
        local = padded[index : index + 2 * radius + 1]
        intensity_weights = np.exp(
            -0.5 * np.square((local - y[index]) / sigma_intensity)
        )
        weights = spatial_weights * intensity_weights
        denominator = float(np.sum(weights))
        result[index] = (
            float(np.dot(weights, local)) / denominator
            if denominator > np.finfo(float).eps
            else y[index]
        )
    return result


def _median_gaussian(y: np.ndarray, strength: float) -> np.ndarray:
    median_size = _odd_window(3 + 8 * strength / 100.0, 3, 15)
    sigma = 0.25 + 2.25 * strength / 100.0
    return gaussian_filter1d(
        median_filter(y, size=median_size, mode="nearest"),
        sigma=sigma,
        mode="nearest",
    )


def _method_smooth(
    method: str,
    y: np.ndarray,
    strength: float,
    noise_sigma: float,
) -> np.ndarray:
    if method == "Savitzky–Golay":
        return _savgol(y, strength)
    if method == "Gaussian":
        return _gaussian(y, strength)
    if method == "Whittaker–Eilers":
        return _whittaker(y, strength)
    if method == "Adaptive bilateral":
        return _bilateral(y, strength, noise_sigma)
    if method == "Median–Gaussian hybrid":
        return _median_gaussian(y, strength)
    raise SmartSmoothingError(f"Unknown smoothing method: {method}")


def _peak_aware_blend(
    y: np.ndarray,
    candidate: np.ndarray,
    peak_mask: np.ndarray,
    preservation_percent: float,
) -> np.ndarray:
    if not np.any(peak_mask) or preservation_percent <= 0:
        return candidate
    maximum = len(y) if len(y) % 2 else len(y) - 1
    mild_window = _odd_window(max(5, len(y) * 0.0025), 5, min(21, maximum))
    mild = savgol_filter(y, mild_window, min(3, mild_window - 2), mode="interp")
    transition = max(2.0, mild_window / 2.0)
    distance = distance_transform_edt(~peak_mask)
    soft_mask = np.exp(-0.5 * np.square(distance / transition))
    preservation = preservation_percent / 100.0
    target = preservation * mild + (1.0 - preservation) * candidate
    return candidate * (1.0 - soft_mask) + target * soft_mask


def _local_peak_measurements(
    x: np.ndarray,
    y: np.ndarray,
    peaks: Iterable[dict],
) -> list[dict]:
    results = []
    for peak in peaks:
        left = int(peak["left_index"])
        right = int(peak["right_index"])
        if right - left < 4:
            continue
        local_x = x[left : right + 1]
        local_y = y[left : right + 1]
        edge_count = max(1, min(4, len(local_y) // 4))
        baseline_left = float(np.median(local_y[:edge_count]))
        baseline_right = float(np.median(local_y[-edge_count:]))
        baseline = np.linspace(baseline_left, baseline_right, len(local_y))
        signal = local_y - baseline
        positive = np.clip(signal, 0.0, None)
        local_max = int(np.argmax(signal))
        height = max(float(signal[local_max]), np.finfo(float).eps)
        position = float(local_x[local_max])
        area = float(np.trapezoid(positive, local_x))
        half_height = 0.5 * height
        above = np.flatnonzero(signal >= half_height)
        fwhm = (
            float(local_x[above[-1]] - local_x[above[0]])
            if len(above) >= 2
            else float(np.median(np.diff(local_x)))
        )
        total = float(np.sum(positive))
        centroid = (
            float(np.sum(local_x * positive) / total)
            if total > np.finfo(float).eps
            else position
        )
        results.append(
            {
                "reference_two_theta_deg": float(peak["two_theta_deg"]),
                "position_deg": position,
                "centroid_deg": centroid,
                "height": height,
                "area": max(area, np.finfo(float).eps),
                "fwhm_deg": max(fwhm, float(np.median(np.diff(local_x)))),
            }
        )
    return results


def _lag1_correlation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return 0.0
    first = values[:-1] - np.mean(values[:-1])
    second = values[1:] - np.mean(values[1:])
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.dot(first, second) / denominator)


def _evaluate_candidate(
    x: np.ndarray,
    original: np.ndarray,
    smoothed: np.ndarray,
    peak_mask: np.ndarray,
    peaks: list[dict],
    noise_before: float,
    parameters: SmoothingParameters,
    method: str,
    strength: float,
) -> dict:
    noise_after = _noise_sigma(smoothed, peak_mask)
    noise_reduction = 100.0 * (1.0 - noise_after / max(noise_before, np.finfo(float).eps))
    residual = original - smoothed
    outside = ~peak_mask
    residual_outside = residual[outside] if np.count_nonzero(outside) >= 20 else residual
    residual_correlation = _lag1_correlation(residual_outside)
    roughness_original = _robust_sigma(np.diff(original, n=2))
    roughness_smoothed = _robust_sigma(np.diff(smoothed, n=2))
    roughness_ratio = roughness_smoothed / max(roughness_original, np.finfo(float).eps)

    original_metrics = _local_peak_measurements(x, original, peaks)
    smoothed_metrics = _local_peak_measurements(x, smoothed, peaks)
    comparisons = []
    for raw, smooth in zip(original_metrics, smoothed_metrics):
        comparisons.append(
            {
                "two_theta_deg": raw["reference_two_theta_deg"],
                "position_shift_deg": smooth["position_deg"] - raw["position_deg"],
                "centroid_shift_deg": smooth["centroid_deg"] - raw["centroid_deg"],
                "height_retention_percent": 100.0 * smooth["height"] / raw["height"],
                "area_retention_percent": 100.0 * smooth["area"] / raw["area"],
                "fwhm_change_percent": 100.0 * (
                    smooth["fwhm_deg"] / raw["fwhm_deg"] - 1.0
                ),
            }
        )

    def median_absolute(key: str) -> float:
        return (
            float(np.median([abs(row[key]) for row in comparisons]))
            if comparisons
            else 0.0
        )

    median_position_shift = median_absolute("position_shift_deg")
    median_centroid_shift = median_absolute("centroid_shift_deg")
    median_height_retention = (
        float(np.median([row["height_retention_percent"] for row in comparisons]))
        if comparisons
        else 100.0
    )
    median_area_retention = (
        float(np.median([row["area_retention_percent"] for row in comparisons]))
        if comparisons
        else 100.0
    )
    median_fwhm_change = (
        float(np.median([row["fwhm_change_percent"] for row in comparisons]))
        if comparisons
        else 0.0
    )

    reward = 32.0 * float(np.clip(noise_reduction / 60.0, 0.0, 1.0))
    reward += 10.0 * float(np.clip((1.0 - roughness_ratio) / 0.8, 0.0, 1.0))
    # Residual should be nearly white outside protected peaks. Strong positive
    # correlation means broad signal was removed rather than random noise.
    correlation_penalty = 15.0 * float(np.clip(abs(residual_correlation) / 0.55, 0.0, 1.0))
    position_penalty = 22.0 * float(
        np.clip(
            max(median_position_shift, median_centroid_shift)
            / parameters.maximum_position_shift_deg,
            0.0,
            2.0,
        )
    )
    height_penalty = 18.0 * float(
        np.clip(
            abs(median_height_retention - 100.0)
            / parameters.maximum_height_change_percent,
            0.0,
            2.0,
        )
    )
    area_penalty = 12.0 * float(
        np.clip(abs(median_area_retention - 100.0) / 10.0, 0.0, 2.0)
    )
    fwhm_penalty = 18.0 * float(
        np.clip(
            abs(median_fwhm_change)
            / parameters.maximum_fwhm_change_percent,
            0.0,
            2.0,
        )
    )
    negative_overshoot_fraction = float(
        np.mean(smoothed < (np.min(original) - 3.0 * noise_before))
    )
    overshoot_penalty = min(20.0, 300.0 * negative_overshoot_fraction)
    score = float(
        np.clip(
            55.0
            + reward
            - correlation_penalty
            - position_penalty
            - height_penalty
            - area_penalty
            - fwhm_penalty
            - overshoot_penalty,
            0.0,
            100.0,
        )
    )

    limits_ok = (
        max(median_position_shift, median_centroid_shift)
        <= parameters.maximum_position_shift_deg
        and abs(median_height_retention - 100.0)
        <= parameters.maximum_height_change_percent
        and abs(median_fwhm_change)
        <= parameters.maximum_fwhm_change_percent
    )
    if score >= 70.0 and limits_ok and noise_reduction >= 5.0:
        quality = "Good smart smoothing"
    elif score >= 45.0 and noise_reduction >= 0.0:
        quality = "Review recommended"
    else:
        quality = "Aggressive smoothing — do not apply blindly"

    warnings = []
    if noise_reduction < 5.0:
        warnings.append("Noise reduction is small; smoothing may be unnecessary.")
    if noise_reduction > 95.0:
        warnings.append("Very high apparent noise reduction may indicate oversmoothing.")
    if max(median_position_shift, median_centroid_shift) > parameters.maximum_position_shift_deg:
        warnings.append("Peak-position or centroid shift exceeds the configured tolerance.")
    if abs(median_height_retention - 100.0) > parameters.maximum_height_change_percent:
        warnings.append("Median peak-height change exceeds the configured tolerance.")
    if abs(median_fwhm_change) > parameters.maximum_fwhm_change_percent:
        warnings.append("Median FWHM change exceeds the configured tolerance.")
    if abs(median_area_retention - 100.0) > 10.0:
        warnings.append("Median integrated peak-area change exceeds 10%.")
    if abs(residual_correlation) > 0.45:
        warnings.append("The smoothing residual remains structured rather than noise-like.")
    if negative_overshoot_fraction > 0.001:
        warnings.append("The candidate introduces negative edge or ringing overshoot.")

    return {
        "method": method,
        "strength": float(strength),
        "score": score,
        "quality": quality,
        "noise_sigma_before": float(noise_before),
        "noise_sigma_after": float(noise_after),
        "noise_reduction_percent": float(noise_reduction),
        "roughness_ratio": float(roughness_ratio),
        "residual_lag1_correlation": float(residual_correlation),
        "protected_peak_count": len(peaks),
        "protected_point_fraction": float(np.mean(peak_mask)),
        "median_position_shift_deg": float(median_position_shift),
        "median_centroid_shift_deg": float(median_centroid_shift),
        "median_height_retention_percent": float(median_height_retention),
        "median_area_retention_percent": float(median_area_retention),
        "median_fwhm_change_percent": float(median_fwhm_change),
        "negative_overshoot_fraction": negative_overshoot_fraction,
        "peak_comparisons": comparisons,
        "warnings": warnings,
        "smoothed": np.asarray(smoothed, dtype=float),
    }


def smart_smooth_pattern(
    x: np.ndarray,
    y: np.ndarray,
    parameters: SmoothingParameters | None = None,
) -> dict:
    x, y = _validate_arrays(x, y)
    parameters = (parameters or SmoothingParameters()).normalized()
    peak_mask, peaks, initial_noise = _detect_peak_regions(
        x,
        y,
        protection_sigma=parameters.protection_sigma,
        width_multiplier=parameters.protection_width_multiplier,
    )
    if not parameters.peak_protection:
        peak_mask = np.zeros_like(peak_mask)
    noise_before = _noise_sigma(y, peak_mask)
    if not np.isfinite(noise_before) or noise_before <= 0:
        noise_before = initial_noise

    manual_methods = [method for method in SMOOTHING_METHODS if method != "Auto intelligent"]
    candidates: list[dict] = []
    if parameters.method == "Auto intelligent":
        trial_count = parameters.automatic_trials_per_method
        if trial_count == 1:
            factors = np.asarray([1.0])
        else:
            factors = np.linspace(0.72, 1.28, trial_count)
        for method in manual_methods:
            for factor in factors:
                strength = float(np.clip(parameters.strength * factor, 2.0, 100.0))
                candidate = _method_smooth(method, y, strength, noise_before)
                if parameters.peak_protection:
                    candidate = _peak_aware_blend(
                        y,
                        candidate,
                        peak_mask,
                        parameters.peak_preservation,
                    )
                candidates.append(
                    _evaluate_candidate(
                        x,
                        y,
                        candidate,
                        peak_mask,
                        peaks,
                        noise_before,
                        parameters,
                        method,
                        strength,
                    )
                )
    else:
        candidate = _method_smooth(
            parameters.method,
            y,
            parameters.strength,
            noise_before,
        )
        if parameters.peak_protection:
            candidate = _peak_aware_blend(
                y,
                candidate,
                peak_mask,
                parameters.peak_preservation,
            )
        candidates.append(
            _evaluate_candidate(
                x,
                y,
                candidate,
                peak_mask,
                peaks,
                noise_before,
                parameters,
                parameters.method,
                parameters.strength,
            )
        )

    if not candidates:
        raise SmartSmoothingError("No smoothing candidate could be evaluated.")
    candidates.sort(key=lambda row: row["score"], reverse=True)
    selected = candidates[0]
    smoothed = np.asarray(selected.pop("smoothed"), dtype=float)
    candidate_scores = []
    for candidate in candidates:
        row = dict(candidate)
        row.pop("smoothed", None)
        row.pop("peak_comparisons", None)
        row.pop("warnings", None)
        candidate_scores.append(row)

    diagnostics = dict(selected)
    diagnostics["selected_method"] = selected["method"]
    diagnostics["selected_strength"] = selected["strength"]
    diagnostics["candidate_scores"] = candidate_scores
    diagnostics["scientific_boundary"] = (
        "Smoothing is a visualization and noise-reduction operation. Raw data must "
        "remain available, and publication-critical peak widths, areas and positions "
        "should be fitted from unsmoothed or explicitly validated data."
    )
    return {
        "smoothed": smoothed,
        "residual": y - smoothed,
        "protected_mask": peak_mask,
        "detected_peaks": peaks,
        "diagnostics": diagnostics,
        "parameters": asdict(parameters),
    }
