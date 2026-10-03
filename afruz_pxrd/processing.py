from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from uuid import uuid4
import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import savgol_filter, find_peaks, peak_widths

from .advanced_background import BackgroundParameters, detect_background
from .smart_smoothing import SmoothingParameters, smart_smooth_pattern


@dataclass
class ProcessingParameters:
    polynomial_order: int = 3
    smoothing_window: int = 11
    smoothing_order: int = 3
    smoothing_method: str = "Auto intelligent"
    smoothing_strength: float = 45.0
    smoothing_peak_protection: bool = True
    smoothing_peak_preservation: float = 85.0
    smoothing_maximum_position_shift_deg: float = 0.02
    smoothing_maximum_height_change_percent: float = 8.0
    smoothing_maximum_fwhm_change_percent: float = 10.0
    normalize: bool = False
    subtract_background: bool = False
    smooth: bool = False
    background_method: str = "Polynomial"
    background_smoothness: float = 70.0
    background_asymmetry: float = 0.01
    background_iterations: int = 50
    background_window_degrees: float = 2.0
    background_percentile: float = 20.0
    background_peak_protection: bool = True
    background_protection_sigma: float = 4.0
    background_protection_width_multiplier: float = 2.5
    clip_negative: bool = True


def polynomial_background(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    order = max(1, min(int(order), 12))
    # Robust-enough prototype baseline: fit the lower-intensity subset.
    cutoff = np.quantile(y, 0.35)
    mask = y <= cutoff
    if mask.sum() < order + 2:
        mask = np.ones_like(y, dtype=bool)
    coeff = np.polyfit(x[mask], y[mask], order)
    return np.polyval(coeff, x)


def _valid_savgol_window(length: int, requested: int, polyorder: int) -> int:
    window = max(int(requested), polyorder + 2)
    if window % 2 == 0:
        window += 1
    max_window = length if length % 2 == 1 else length - 1
    return max(polyorder + 2 + ((polyorder + 2) % 2 == 0), min(window, max_window))


def _odd_window(requested: float, minimum: int, maximum: int) -> int:
    """Return an odd window length constrained to the supplied range."""
    maximum = max(3, int(maximum))
    if maximum % 2 == 0:
        maximum -= 1
    window = max(int(minimum), int(round(requested)))
    if window % 2 == 0:
        window += 1
    if window > maximum:
        window = maximum
    return max(3, window)


def _robust_sigma(values: np.ndarray) -> float:
    """Estimate Gaussian-equivalent noise with a median absolute deviation."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float(np.finfo(float).eps)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = mad / 0.6744897501960817
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        sigma = float(np.std(values))
    return max(sigma, float(np.finfo(float).eps))


def process_pattern(
    x: np.ndarray,
    y: np.ndarray,
    params: ProcessingParameters,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    result = np.asarray(y, dtype=float).copy()
    aux: dict[str, np.ndarray] = {}

    if params.subtract_background:
        background_result = detect_background(
            x,
            result,
            BackgroundParameters(
                method=params.background_method,
                smoothness=params.background_smoothness,
                asymmetry=params.background_asymmetry,
                iterations=params.background_iterations,
                window_degrees=params.background_window_degrees,
                percentile=params.background_percentile,
                polynomial_order=params.polynomial_order,
                peak_protection=params.background_peak_protection,
                protection_sigma=params.background_protection_sigma,
                protection_width_multiplier=(
                    params.background_protection_width_multiplier
                ),
                clip_negative=params.clip_negative,
            ),
        )
        baseline = np.asarray(background_result["background"], dtype=float)
        aux["background"] = baseline
        aux["background_diagnostics"] = background_result["diagnostics"]
        aux["background_parameters"] = background_result["parameters"]
        aux["background_protected_mask"] = np.asarray(
            background_result["protected_mask"], dtype=bool
        )
        result = np.asarray(background_result["corrected"], dtype=float)

    if params.smooth:
        smoothing_result = smart_smooth_pattern(
            x,
            result,
            SmoothingParameters(
                method=params.smoothing_method,
                strength=params.smoothing_strength,
                peak_protection=params.smoothing_peak_protection,
                peak_preservation=params.smoothing_peak_preservation,
                maximum_position_shift_deg=(
                    params.smoothing_maximum_position_shift_deg
                ),
                maximum_height_change_percent=(
                    params.smoothing_maximum_height_change_percent
                ),
                maximum_fwhm_change_percent=(
                    params.smoothing_maximum_fwhm_change_percent
                ),
            ),
        )
        aux["smoothing_source"] = np.asarray(result, dtype=float).copy()
        aux["smoothed"] = np.asarray(smoothing_result["smoothed"], dtype=float)
        aux["smoothing_diagnostics"] = smoothing_result["diagnostics"]
        aux["smoothing_parameters"] = smoothing_result["parameters"]
        aux["smoothing_protected_mask"] = np.asarray(
            smoothing_result["protected_mask"], dtype=bool
        )
        result = np.asarray(smoothing_result["smoothed"], dtype=float)

    if params.normalize:
        maximum = float(np.max(np.abs(result)))
        if maximum > 0:
            normalization_factor = 100.0 / maximum
            result = result * normalization_factor

            # Every curve shown beside the normalized result must use the same
            # intensity units.  Previously only ``smoothed`` was normalized,
            # while ``smoothing_source`` (and the displayed background) stayed
            # in raw counts.  PyQtGraph then auto-ranged to the raw-count curve,
            # making the normalized diffraction peaks look visually compressed.
            if "smoothed" in aux:
                aux["smoothed"] = (
                    np.asarray(aux["smoothed"], dtype=float)
                    * normalization_factor
                )
            if "smoothing_source" in aux:
                aux["smoothing_source"] = (
                    np.asarray(aux["smoothing_source"], dtype=float)
                    * normalization_factor
                )
            if "background" in aux:
                # Preserve ``background`` in original count units for scientific
                # diagnostics/export, and provide a separate display curve that
                # matches the normalized processed pattern.
                aux["background_display"] = (
                    np.asarray(aux["background"], dtype=float)
                    * normalization_factor
                )

            aux["normalization_factor"] = float(normalization_factor)
            aux["normalization_reference_maximum"] = float(maximum)
            aux["normalization_target"] = 100.0

    return result, aux


def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    prominence_fraction: float = 0.03,
    minimum_distance_points: int = 5,
) -> list[dict[str, float]]:
    """Regular user-controlled peak detection."""
    if len(y) < 3:
        return []
    dynamic_range = float(np.max(y) - np.min(y))
    prominence = max(dynamic_range * prominence_fraction, np.finfo(float).eps)
    indices, props = find_peaks(
        y,
        prominence=prominence,
        distance=max(1, int(minimum_distance_points)),
    )
    if len(indices) == 0:
        return []

    widths, _, left_ips, right_ips = peak_widths(y, indices, rel_height=0.5)
    rows = []
    for i, idx in enumerate(indices):
        left_x = np.interp(left_ips[i], np.arange(len(x)), x)
        right_x = np.interp(right_ips[i], np.arange(len(x)), x)
        rows.append(
            {
                "position": float(x[idx]),
                "intensity": float(y[idx]),
                "prominence": float(props["prominences"][i]),
                "fwhm": float(right_x - left_x),
                "index": int(idx),
                "method": "Regular",
                "snr": None,
                "confidence": None,
                "confidence_label": "Manual",
            }
        )
    return rows



def normalize_peak_rows(peaks) -> list[dict]:
    """Normalize the application-wide detected/manual peak list.

    Extra fields are retained for forward compatibility. ``peak_uuid`` is the
    stable identity used by table edits, project persistence and downstream
    analyses.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for source in peaks or []:
        if not isinstance(source, dict):
            continue
        row = deepcopy(source)
        try:
            position = float(row.get("position"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(position):
            continue
        row["position"] = position
        for key, default in (
            ("intensity", 0.0),
            ("prominence", 0.0),
            ("fwhm", 0.0),
        ):
            try:
                value = float(row.get(key, default))
            except (TypeError, ValueError):
                value = float(default)
            row[key] = value if np.isfinite(value) else float(default)
        for key in ("snr", "confidence", "position_error"):
            value = row.get(key)
            try:
                value = None if value in (None, "", "—") else float(value)
            except (TypeError, ValueError):
                value = None
            row[key] = value if value is None or np.isfinite(value) else None
        method = str(row.get("method") or row.get("origin") or "Imported")
        row["method"] = method
        row["origin"] = str(row.get("origin") or method)
        row["manual"] = bool(
            row.get("manual", method.lower().startswith("manual"))
        )
        row["protected"] = bool(
            row.get(
                "protected",
                row["manual"] or "master" in method.lower(),
            )
        )
        row["use"] = bool(row.get("use", True))
        row["notes"] = str(row.get("notes", row.get("note", "")))
        row["confidence_label"] = str(row.get("confidence_label", ""))
        try:
            row["index"] = int(row.get("index", 0))
        except (TypeError, ValueError):
            row["index"] = 0
        peak_uuid = str(row.get("peak_uuid") or "").strip()
        if not peak_uuid or peak_uuid in seen:
            peak_uuid = uuid4().hex
        row["peak_uuid"] = peak_uuid
        seen.add(peak_uuid)
        rows.append(row)
    rows.sort(key=lambda row: (row["position"], row["peak_uuid"]))
    return rows


def active_peak_rows(peaks) -> list[dict]:
    """Return included peaks only, preserving their stable identities."""
    return [row for row in normalize_peak_rows(peaks) if row.get("use", True)]


def create_manual_peak(
    x: np.ndarray,
    y: np.ndarray,
    target_two_theta_deg: float,
    *,
    snap_window_deg: float = 0.15,
    peak_uuid: str | None = None,
    notes: str = "",
    origin: str = "Manual",
) -> dict:
    """Create a measured manual peak, snapping to a local maximum.

    The measurement is taken from the active dataset. No synthetic intensity or
    width is invented; FWHM is estimated from local half-height crossings.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("A valid pattern with at least three points is required.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("The pattern contains non-finite values.")
    target = float(target_two_theta_deg)
    if target < float(np.min(x)) or target > float(np.max(x)):
        raise ValueError(
            f"Manual peak position {target:g}° is outside the dataset range "
            f"{float(np.min(x)):g}–{float(np.max(x)):g}°."
        )
    window = max(0.0, float(snap_window_deg))
    mask = np.abs(x - target) <= window
    if not np.any(mask):
        index = int(np.argmin(np.abs(x - target)))
    else:
        candidates = np.flatnonzero(mask)
        index = int(candidates[np.argmax(y[candidates])])

    point_step = float(np.median(np.diff(x)))
    half_points = max(4, int(round(max(window, point_step * 5) / point_step)))
    left = max(0, index - half_points)
    right = min(len(y), index + half_points + 1)
    local_y = y[left:right]
    baseline = float(np.percentile(local_y, 15.0))
    height = max(float(y[index] - baseline), np.finfo(float).eps)
    half_height = baseline + 0.5 * height

    left_index = index
    while left_index > 0 and y[left_index] > half_height:
        left_index -= 1
    right_index = index
    while right_index < len(y) - 1 and y[right_index] > half_height:
        right_index += 1

    left_position = float(x[left_index])
    if left_index < index and y[left_index + 1] != y[left_index]:
        left_position = float(
            x[left_index]
            + (half_height - y[left_index])
            * (x[left_index + 1] - x[left_index])
            / (y[left_index + 1] - y[left_index])
        )
    right_position = float(x[right_index])
    if right_index > index and y[right_index] != y[right_index - 1]:
        right_position = float(
            x[right_index - 1]
            + (half_height - y[right_index - 1])
            * (x[right_index] - x[right_index - 1])
            / (y[right_index] - y[right_index - 1])
        )
    fwhm = max(point_step, right_position - left_position)

    differences = np.diff(y)
    centered = differences - np.median(differences)
    noise_sigma = float(
        np.median(np.abs(centered)) / (0.67448975 * np.sqrt(2.0))
    )
    noise_sigma = max(noise_sigma, np.finfo(float).eps)
    snr = float(height / noise_sigma)
    confidence = float(np.clip(45.0 + 12.0 * np.log1p(snr), 0.0, 99.9))
    confidence_label = (
        "High" if confidence >= 80.0
        else "Medium" if confidence >= 65.0
        else "Manual/Review"
    )
    return normalize_peak_rows(
        [
            {
                "position": float(x[index]),
                "position_error": max(point_step / 2.0, np.finfo(float).eps),
                "intensity": float(y[index]),
                "prominence": height,
                "fwhm": float(fwhm),
                "index": index,
                "method": "Manual",
                "origin": origin,
                "manual": True,
                "protected": True,
                "use": True,
                "snr": snr,
                "confidence": confidence,
                "confidence_label": confidence_label,
                "noise_sigma": float(noise_sigma),
                "peak_uuid": peak_uuid or uuid4().hex,
                "notes": notes,
            }
        ]
    )[0]


def merge_detected_with_manual(
    existing,
    detected,
    *,
    duplicate_tolerance_deg: float = 0.04,
) -> list[dict]:
    """Replace automatic detections while preserving protected/manual peaks."""
    old_rows = normalize_peak_rows(existing)
    new_rows = normalize_peak_rows(detected)
    protected = [
        row for row in old_rows
        if row.get("manual") or row.get("protected")
    ]
    merged = [deepcopy(row) for row in protected]
    tolerance = max(0.0, float(duplicate_tolerance_deg))
    for row in new_rows:
        position = float(row["position"])
        if any(
            abs(position - float(kept["position"]))
            <= max(tolerance, 0.25 * float(kept.get("fwhm", 0.0)))
            for kept in protected
        ):
            continue
        merged.append(row)
    return normalize_peak_rows(merged)


SMART_PROFILES = {
    "Conservative": {
        "height_snr": 6.0,
        "prominence_snr": 5.0,
        "minimum_width_points": 2.0,
    },
    "Balanced": {
        "height_snr": 4.5,
        "prominence_snr": 3.8,
        "minimum_width_points": 1.5,
    },
    "Sensitive": {
        "height_snr": 3.2,
        "prominence_snr": 2.6,
        "minimum_width_points": 1.0,
    },
}


def smart_detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    sensitivity: str = "Balanced",
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """
    Fast, noise-aware PXRD peak detection.

    The algorithm:
    1. Smooths with a data-length-adaptive Savitzky–Golay window.
    2. Estimates high-frequency noise using robust MAD statistics.
    3. Estimates a slowly varying baseline with a percentile filter.
    4. Searches the baseline-corrected signal using SNR-based height,
       prominence, distance, and minimum-width constraints.
    5. Assigns a confidence score from height SNR, prominence SNR, and width.

    This is an engineering heuristic and requires validation for each
    instrument, scan step, line broadening regime, and sample type.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("X and Y must be one-dimensional arrays of equal length.")
    if len(y) < 11:
        return [], {
            "noise_sigma": 0.0,
            "height_threshold": 0.0,
            "prominence_threshold": 0.0,
            "smooth_window": 0,
            "baseline_window": 0,
        }
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Smart peak search requires finite X and Y values.")

    profile = SMART_PROFILES.get(sensitivity, SMART_PROFILES["Balanced"])
    point_count = len(y)
    maximum_odd_window = point_count if point_count % 2 == 1 else point_count - 1

    smooth_window = _odd_window(
        point_count * 0.005,
        minimum=5,
        maximum=min(61, maximum_odd_window),
    )
    smooth_order = min(3, smooth_window - 2)
    smoothed = savgol_filter(y, smooth_window, smooth_order)

    # Estimate high-frequency noise while trimming the largest residuals,
    # which are commonly caused by peaks or isolated detector spikes.
    residual = y - smoothed
    residual_center = residual - np.median(residual)
    residual_cutoff = np.quantile(np.abs(residual_center), 0.80)
    residual_sample = residual_center[np.abs(residual_center) <= residual_cutoff]
    sigma_residual = _robust_sigma(residual_sample)

    # A second independent estimate from first differences improves stability
    # for slowly drifting backgrounds.
    differences = np.diff(y)
    difference_center = differences - np.median(differences)
    difference_cutoff = np.quantile(np.abs(difference_center), 0.70)
    difference_sample = difference_center[
        np.abs(difference_center) <= difference_cutoff
    ]
    sigma_difference = _robust_sigma(difference_sample) / np.sqrt(2.0)

    # Avoid allowing one unstable estimator to dominate.
    noise_sigma = max(
        min(sigma_residual, sigma_difference * 1.5),
        float(np.finfo(float).eps),
    )

    baseline_window = _odd_window(
        point_count * 0.06,
        minimum=31,
        maximum=min(401, maximum_odd_window),
    )
    baseline = percentile_filter(
        smoothed,
        percentile=20,
        size=baseline_window,
        mode="nearest",
    )
    baseline = gaussian_filter1d(
        baseline,
        sigma=max(1.0, baseline_window / 12.0),
    )

    corrected = np.clip(smoothed - baseline, 0.0, None)
    robust_dynamic = max(
        float(np.quantile(corrected, 0.995) - np.median(corrected)),
        noise_sigma,
    )

    height_threshold = max(
        profile["height_snr"] * noise_sigma,
        robust_dynamic * 0.012,
    )
    prominence_threshold = max(
        profile["prominence_snr"] * noise_sigma,
        robust_dynamic * 0.015,
    )
    minimum_distance = max(2, smooth_window // 4)

    indices, properties = find_peaks(
        corrected,
        height=height_threshold,
        prominence=prominence_threshold,
        distance=minimum_distance,
        width=(profile["minimum_width_points"], None),
    )

    diagnostics = {
        "noise_sigma": float(noise_sigma),
        "height_threshold": float(height_threshold),
        "prominence_threshold": float(prominence_threshold),
        "smooth_window": int(smooth_window),
        "baseline_window": int(baseline_window),
    }
    if len(indices) == 0:
        return [], diagnostics

    widths, _, left_ips, right_ips = peak_widths(
        corrected,
        indices,
        rel_height=0.5,
    )

    rows: list[dict[str, float]] = []
    for i, idx in enumerate(indices):
        left_x = np.interp(left_ips[i], np.arange(point_count), x)
        right_x = np.interp(right_ips[i], np.arange(point_count), x)
        height_snr = float(corrected[idx] / noise_sigma)
        prominence_snr = float(properties["prominences"][i] / noise_sigma)
        width_points = float(widths[i])

        height_score = 1.0 - np.exp(
            -max(0.0, height_snr - profile["height_snr"]) / 3.0
        )
        prominence_score = 1.0 - np.exp(
            -max(
                0.0,
                prominence_snr - profile["prominence_snr"],
            )
            / 3.0
        )
        width_score = min(1.0, width_points / 4.0)
        confidence = 50.0 + 50.0 * (
            0.42 * height_score
            + 0.42 * prominence_score
            + 0.16 * width_score
        )
        confidence = float(np.clip(confidence, 0.0, 99.9))
        confidence_label = (
            "High" if confidence >= 80.0
            else "Medium" if confidence >= 65.0
            else "Low"
        )

        rows.append(
            {
                "position": float(x[idx]),
                "intensity": float(y[idx]),
                "prominence": float(properties["prominences"][i]),
                "fwhm": float(right_x - left_x),
                "index": int(idx),
                "method": "Smart",
                "snr": height_snr,
                "confidence": confidence,
                "confidence_label": confidence_label,
                "noise_sigma": float(noise_sigma),
            }
        )

    rows.sort(key=lambda row: row["position"])
    return rows, diagnostics
