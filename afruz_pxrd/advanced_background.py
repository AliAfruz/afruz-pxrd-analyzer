from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.sparse import csc_matrix, diags, eye
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import warnings


BACKGROUND_METHODS = (
    "Auto ensemble",
    "arPLS",
    "AsLS",
    "SNIP",
    "Rolling percentile",
    "Polynomial",
)


class BackgroundError(ValueError):
    pass


@dataclass
class BackgroundParameters:
    method: str = "Auto ensemble"
    smoothness: float = 70.0
    asymmetry: float = 0.01
    iterations: int = 50
    window_degrees: float = 2.0
    percentile: float = 20.0
    polynomial_order: int = 3
    peak_protection: bool = True
    protection_sigma: float = 4.0
    protection_width_multiplier: float = 2.5
    clip_negative: bool = False

    def normalized(self) -> "BackgroundParameters":
        method = self.method if self.method in BACKGROUND_METHODS else "Auto ensemble"
        return BackgroundParameters(
            method=method,
            smoothness=float(np.clip(self.smoothness, 1.0, 100.0)),
            asymmetry=float(np.clip(self.asymmetry, 1e-5, 0.49)),
            iterations=int(np.clip(self.iterations, 5, 250)),
            window_degrees=float(max(0.02, self.window_degrees)),
            percentile=float(np.clip(self.percentile, 1.0, 49.0)),
            polynomial_order=int(np.clip(self.polynomial_order, 1, 12)),
            peak_protection=bool(self.peak_protection),
            protection_sigma=float(np.clip(self.protection_sigma, 1.5, 20.0)),
            protection_width_multiplier=float(
                np.clip(self.protection_width_multiplier, 0.5, 10.0)
            ),
            clip_negative=bool(self.clip_negative),
        )


@dataclass
class BackgroundDiagnostics:
    requested_method: str
    selected_method: str
    quality: str
    score: float
    noise_sigma: float
    protected_peak_count: int
    protected_point_fraction: float
    negative_fraction: float
    severe_negative_fraction: float
    contact_fraction: float
    roughness_ratio: float
    edge_mismatch_ratio: float
    background_area_fraction: float
    corrected_area: float
    raw_area: float
    warnings: list[str]
    candidate_scores: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_pattern(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise BackgroundError("Background detection requires equal-length 1D X and Y arrays.")
    if len(x) < 15:
        raise BackgroundError("At least 15 pattern points are required for advanced background detection.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise BackgroundError("Background detection requires finite X and Y values.")
    if np.any(np.diff(x) <= 0):
        raise BackgroundError("X values must be strictly increasing.")
    return x, y


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(np.finfo(float).eps)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    sigma = mad / 0.6744897501960817
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        sigma = float(np.std(values))
    return max(sigma, float(np.finfo(float).eps))


def estimate_noise_sigma(y: np.ndarray) -> float:
    differences = np.diff(np.asarray(y, dtype=float))
    if differences.size < 3:
        return _robust_sigma(y)
    centered = differences - np.median(differences)
    cutoff = np.quantile(np.abs(centered), 0.70)
    sample = centered[np.abs(centered) <= cutoff]
    return _robust_sigma(sample) / math.sqrt(2.0)


def _odd_window_from_degrees(x: np.ndarray, span_degrees: float, minimum: int = 7) -> int:
    step = float(np.median(np.diff(x)))
    requested = max(minimum, int(round(float(span_degrees) / max(step, 1e-12))))
    if requested % 2 == 0:
        requested += 1
    maximum = len(x) if len(x) % 2 else len(x) - 1
    return max(3, min(requested, maximum))


def _difference_penalty(length: int, order: int = 2) -> csc_matrix:
    if order != 2:
        raise BackgroundError("Only a second-difference smoothness penalty is supported.")
    return diags(
        [np.ones(length - 2), -2.0 * np.ones(length - 2), np.ones(length - 2)],
        [0, 1, 2],
        shape=(length - 2, length),
        format="csc",
    )


def _smoothness_lambda(smoothness: float, length: int) -> float:
    # A logarithmic map is easier to control than a linear lambda slider.
    exponent = 2.0 + 7.0 * float(np.clip(smoothness, 1.0, 100.0)) / 100.0
    # Scale mildly with pattern size so the slider feels similar across scans.
    return float(10.0**exponent * max(1.0, (length / 2000.0) ** 0.8))


def _safe_sparse_solve(matrix: csc_matrix, rhs: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)
        try:
            solution = spsolve(matrix, rhs)
        except (MatrixRankWarning, RuntimeError, ValueError) as exc:
            raise BackgroundError(f"Sparse background solve failed: {exc}") from exc
    solution = np.asarray(solution, dtype=float)
    if not np.all(np.isfinite(solution)):
        raise BackgroundError("Sparse background solve produced non-finite values.")
    return solution


def asls_background(
    y: np.ndarray,
    *,
    smoothness: float = 70.0,
    asymmetry: float = 0.01,
    iterations: int = 30,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    length = len(y)
    penalty = _difference_penalty(length)
    lam = _smoothness_lambda(smoothness, length)
    weights = np.ones(length, dtype=float)
    asymmetry = float(np.clip(asymmetry, 1e-5, 0.49))
    baseline = y.copy()
    for _ in range(int(np.clip(iterations, 5, 250))):
        weight_matrix = diags(weights, 0, shape=(length, length), format="csc")
        baseline = _safe_sparse_solve(
            weight_matrix + lam * (penalty.T @ penalty),
            weights * y,
        )
        residual = y - baseline
        updated = np.where(residual > 0.0, asymmetry, 1.0 - asymmetry)
        if np.max(np.abs(updated - weights)) < 1e-4:
            weights = updated
            break
        weights = updated
    return baseline


def arpls_background(
    y: np.ndarray,
    *,
    smoothness: float = 70.0,
    iterations: int = 50,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    length = len(y)
    penalty = _difference_penalty(length)
    lam = _smoothness_lambda(smoothness, length)
    weights = np.ones(length, dtype=float)
    baseline = y.copy()
    for _ in range(int(np.clip(iterations, 5, 250))):
        weight_matrix = diags(weights, 0, shape=(length, length), format="csc")
        baseline = _safe_sparse_solve(
            weight_matrix + lam * (penalty.T @ penalty),
            weights * y,
        )
        residual = y - baseline
        negative = residual[residual < 0.0]
        if negative.size < max(3, length // 100):
            break
        mean_negative = float(np.mean(negative))
        sigma_negative = max(float(np.std(negative)), np.finfo(float).eps)
        exponent = np.clip(
            2.0 * (residual - (2.0 * sigma_negative - mean_negative)) / sigma_negative,
            -60.0,
            60.0,
        )
        updated = 1.0 / (1.0 + np.exp(exponent))
        denominator = max(float(np.linalg.norm(weights)), np.finfo(float).eps)
        relative_change = float(np.linalg.norm(updated - weights) / denominator)
        weights = updated
        if relative_change < 1e-3:
            break
    return baseline


def rolling_percentile_background(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window_degrees: float = 2.0,
    percentile: float = 20.0,
    smoothness: float = 70.0,
) -> np.ndarray:
    window = _odd_window_from_degrees(x, window_degrees, minimum=9)
    baseline = percentile_filter(
        y,
        percentile=float(np.clip(percentile, 1.0, 49.0)),
        size=window,
        mode="nearest",
    )
    sigma_points = max(1.0, window * (0.04 + 0.18 * smoothness / 100.0))
    return gaussian_filter1d(baseline, sigma=sigma_points, mode="nearest")


def snip_background(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window_degrees: float = 2.0,
    smoothness: float = 70.0,
) -> np.ndarray:
    minimum = float(np.min(y))
    shifted = np.clip(y - minimum, 0.0, None)
    transformed = np.log(np.log(np.sqrt(shifted + 1.0) + 1.0) + 1.0)
    maximum_half_window = max(1, _odd_window_from_degrees(x, window_degrees, 7) // 2)
    clipped = transformed.copy()
    for half_window in range(maximum_half_window, 0, -1):
        left = np.roll(clipped, half_window)
        right = np.roll(clipped, -half_window)
        averaged = 0.5 * (left + right)
        valid = np.arange(half_window, len(y) - half_window)
        clipped[valid] = np.minimum(clipped[valid], averaged[valid])
    baseline = (
        np.square(np.exp(np.exp(clipped) - 1.0) - 1.0) - 1.0 + minimum
    )
    sigma = max(0.75, 0.02 * maximum_half_window * smoothness / 25.0)
    return gaussian_filter1d(baseline, sigma=sigma, mode="nearest")


def polynomial_lower_envelope(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int = 3,
    iterations: int = 8,
) -> np.ndarray:
    order = int(np.clip(order, 1, 12))
    x_center = float(np.mean(x))
    x_scale = max(float(np.ptp(x)) / 2.0, np.finfo(float).eps)
    scaled = (x - x_center) / x_scale
    weights = np.ones_like(y)
    baseline = np.full_like(y, np.median(y))
    for _ in range(int(np.clip(iterations, 3, 30))):
        coefficients = np.polynomial.chebyshev.chebfit(
            scaled,
            y,
            order,
            w=np.sqrt(np.clip(weights, 1e-8, None)),
        )
        baseline = np.polynomial.chebyshev.chebval(scaled, coefficients)
        residual = y - baseline
        sigma = _robust_sigma(residual[residual <= np.quantile(residual, 0.65)])
        weights = np.where(residual > 2.5 * sigma, 0.02, 1.0)
    return baseline


def _peak_protection_mask(
    x: np.ndarray,
    y: np.ndarray,
    params: BackgroundParameters,
    noise_sigma: float,
) -> tuple[np.ndarray, int]:
    if not params.peak_protection:
        return np.zeros(len(y), dtype=bool), 0
    preliminary = rolling_percentile_background(
        x,
        y,
        window_degrees=max(params.window_degrees, 0.5),
        percentile=min(params.percentile, 25.0),
        smoothness=max(params.smoothness, 50.0),
    )
    corrected = y - preliminary
    # Peak protection must not respond to every random local maximum. A small
    # Gaussian filter suppresses point noise while remaining far narrower than
    # the broad background features the model is intended to follow.
    search_signal = gaussian_filter1d(corrected, sigma=1.25, mode="nearest")
    robust_range = max(
        float(np.quantile(search_signal, 0.995) - np.median(search_signal)),
        noise_sigma,
    )
    threshold = max(params.protection_sigma * noise_sigma, 0.015 * robust_range)
    indices, properties = find_peaks(
        search_signal,
        height=threshold,
        prominence=max(0.90 * threshold, 3.0 * noise_sigma),
        distance=3,
        width=(1.5, None),
    )
    mask = np.zeros(len(y), dtype=bool)
    if not len(indices):
        return mask, 0
    widths, _, left_ips, right_ips = peak_widths(search_signal, indices, rel_height=0.20)
    for left, right, width in zip(left_ips, right_ips, widths):
        margin = params.protection_width_multiplier * max(1.0, float(width))
        start = max(0, int(math.floor(left - margin)))
        stop = min(len(y), int(math.ceil(right + margin)) + 1)
        mask[start:stop] = True
    return mask, int(len(indices))


def _interpolate_protected_signal(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return y.copy()
    keep = ~mask
    if np.count_nonzero(keep) < 3:
        return y.copy()
    protected = y.copy()
    protected[mask] = np.interp(x[mask], x[keep], y[keep])
    return protected


def _sanitize_baseline(y: np.ndarray, baseline: np.ndarray, smoothness: float) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=float)
    if baseline.shape != y.shape or not np.all(np.isfinite(baseline)):
        raise BackgroundError("Background method returned an invalid baseline.")
    # Remove isolated solver ripples while preserving broad curvature.
    sigma = 0.25 + 1.25 * smoothness / 100.0
    baseline = gaussian_filter1d(baseline, sigma=sigma, mode="nearest")
    # Do not allow numerical extrapolation far above the measured pattern.
    dynamic = max(float(np.ptp(y)), _robust_sigma(y))
    return np.clip(baseline, float(np.min(y) - 0.10 * dynamic), float(np.max(y) + 0.02 * dynamic))


def _candidate_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
    noise_sigma: float,
    protected_mask: np.ndarray,
) -> dict:
    corrected = y - baseline
    unprotected = ~protected_mask
    if np.count_nonzero(unprotected) < 10:
        unprotected = np.ones(len(y), dtype=bool)
    negative_fraction = float(np.mean(corrected < 0.0))
    severe_negative_fraction = float(np.mean(corrected < -3.0 * noise_sigma))
    contact_fraction = float(
        np.mean(np.abs(corrected[unprotected]) <= 2.0 * noise_sigma)
    )
    baseline_second = np.diff(baseline, n=2)
    signal_scale = max(
        float(np.quantile(y, 0.95) - np.quantile(y, 0.05)),
        noise_sigma,
    )
    roughness_ratio = float(
        np.sqrt(np.mean(np.square(baseline_second))) / signal_scale
    )
    edge_count = max(3, min(len(y) // 20, 50))
    edge_residual = np.concatenate([corrected[:edge_count], corrected[-edge_count:]])
    edge_mismatch_ratio = float(abs(np.median(edge_residual)) / max(signal_scale, noise_sigma))
    raw_area = float(np.trapezoid(np.clip(y - np.min(y), 0.0, None), x))
    baseline_area = float(
        np.trapezoid(np.clip(baseline - np.min(y), 0.0, None), x)
    )
    corrected_area = float(np.trapezoid(np.clip(corrected, 0.0, None), x))
    background_area_fraction = baseline_area / max(raw_area, np.finfo(float).eps)

    # Lower is better. The contact target avoids baselines that remain far below
    # all data while the severe-negative penalty rejects over-subtraction.
    contact_penalty = abs(contact_fraction - 0.30)
    score = (
        320.0 * severe_negative_fraction
        + 55.0 * max(0.0, negative_fraction - 0.42)
        + 22.0 * contact_penalty
        + 2500.0 * roughness_ratio
        + 18.0 * edge_mismatch_ratio
        + 45.0 * max(0.0, background_area_fraction - 0.92)
    )
    return {
        "score": float(score),
        "negative_fraction": negative_fraction,
        "severe_negative_fraction": severe_negative_fraction,
        "contact_fraction": contact_fraction,
        "roughness_ratio": roughness_ratio,
        "edge_mismatch_ratio": edge_mismatch_ratio,
        "background_area_fraction": float(background_area_fraction),
        "corrected_area": corrected_area,
        "raw_area": raw_area,
    }


def _run_method(
    method: str,
    x: np.ndarray,
    protected_y: np.ndarray,
    params: BackgroundParameters,
) -> np.ndarray:
    if method == "arPLS":
        return arpls_background(
            protected_y,
            smoothness=params.smoothness,
            iterations=params.iterations,
        )
    if method == "AsLS":
        return asls_background(
            protected_y,
            smoothness=params.smoothness,
            asymmetry=params.asymmetry,
            iterations=params.iterations,
        )
    if method == "SNIP":
        return snip_background(
            x,
            protected_y,
            window_degrees=params.window_degrees,
            smoothness=params.smoothness,
        )
    if method == "Rolling percentile":
        return rolling_percentile_background(
            x,
            protected_y,
            window_degrees=params.window_degrees,
            percentile=params.percentile,
            smoothness=params.smoothness,
        )
    if method == "Polynomial":
        return polynomial_lower_envelope(
            x,
            protected_y,
            order=params.polynomial_order,
            iterations=max(5, params.iterations // 5),
        )
    raise BackgroundError(f"Unsupported background method: {method}")


def detect_background(
    x: np.ndarray,
    y: np.ndarray,
    params: BackgroundParameters | None = None,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    x, y = _validate_pattern(x, y)
    params = (params or BackgroundParameters()).normalized()
    noise_sigma = estimate_noise_sigma(y)
    protected_mask, protected_peak_count = _peak_protection_mask(
        x,
        y,
        params,
        noise_sigma,
    )
    protected_y = _interpolate_protected_signal(x, y, protected_mask)

    methods = (
        ["arPLS", "AsLS", "SNIP", "Rolling percentile", "Polynomial"]
        if params.method == "Auto ensemble"
        else [params.method]
    )
    candidates: list[dict] = []
    failures: list[str] = []
    for index, method in enumerate(methods, start=1):
        if progress_callback:
            progress_callback(index - 1, len(methods), f"Evaluating {method}")
        try:
            baseline = _run_method(method, x, protected_y, params)
            baseline = _sanitize_baseline(y, baseline, params.smoothness)
            metrics = _candidate_diagnostics(
                x,
                y,
                baseline,
                noise_sigma,
                protected_mask,
            )
            candidates.append({"method": method, "baseline": baseline, **metrics})
        except Exception as exc:  # preserve other candidates in auto mode
            failures.append(f"{method}: {exc}")
    if progress_callback:
        progress_callback(len(methods), len(methods), "Background candidates evaluated")
    if not candidates:
        raise BackgroundError("All background methods failed. " + "; ".join(failures))

    candidates.sort(key=lambda row: row["score"])
    selected = candidates[0]
    baseline = np.asarray(selected["baseline"], dtype=float)

    # In auto mode, median-combine statistically similar top candidates. This
    # reduces method-specific wiggles without blending a clearly poor model.
    if params.method == "Auto ensemble" and len(candidates) >= 2:
        best_score = candidates[0]["score"]
        accepted = [
            row for row in candidates
            if row["score"] <= best_score + max(2.0, 0.20 * max(best_score, 1.0))
        ][:3]
        if len(accepted) >= 2:
            ensemble = np.median(
                np.vstack([row["baseline"] for row in accepted]),
                axis=0,
            )
            ensemble = _sanitize_baseline(y, ensemble, params.smoothness)
            metrics = _candidate_diagnostics(
                x,
                y,
                ensemble,
                noise_sigma,
                protected_mask,
            )
            if metrics["score"] <= selected["score"] + 1.0:
                selected = {
                    "method": "Auto ensemble: " + " + ".join(
                        row["method"] for row in accepted
                    ),
                    "baseline": ensemble,
                    **metrics,
                }
                baseline = ensemble

    corrected = y - baseline
    if params.clip_negative:
        corrected = np.clip(corrected, 0.0, None)

    warnings_list: list[str] = []
    if selected["severe_negative_fraction"] > 0.01:
        warnings_list.append(
            "More than 1% of points fall over three noise standard deviations below zero; inspect possible over-subtraction."
        )
    if selected["roughness_ratio"] > 0.003:
        warnings_list.append(
            "The detected background is comparatively rough; increase smoothness or the background window."
        )
    if selected["contact_fraction"] < 0.08:
        warnings_list.append(
            "The background contacts very few non-peak points and may be too low."
        )
    if selected["contact_fraction"] > 0.65:
        warnings_list.append(
            "The background contacts many points and may be following broad peaks or amorphous scattering."
        )
    if protected_peak_count == 0 and params.peak_protection:
        warnings_list.append(
            "No peak regions were automatically protected; review the preview for broad or weak reflections."
        )
    if failures:
        warnings_list.append("Some candidate methods failed: " + "; ".join(failures))
    if params.clip_negative:
        warnings_list.append(
            "Negative corrected intensities were clipped to zero; this can bias integrated intensities and uncertainty models."
        )

    if selected["score"] <= 12.0 and selected["severe_negative_fraction"] <= 0.005:
        quality = "Good automatic baseline"
    elif selected["score"] <= 28.0 and selected["severe_negative_fraction"] <= 0.02:
        quality = "Review recommended"
    else:
        quality = "Manual review required"

    diagnostics = BackgroundDiagnostics(
        requested_method=params.method,
        selected_method=selected["method"],
        quality=quality,
        score=float(selected["score"]),
        noise_sigma=float(noise_sigma),
        protected_peak_count=protected_peak_count,
        protected_point_fraction=float(np.mean(protected_mask)),
        negative_fraction=float(selected["negative_fraction"]),
        severe_negative_fraction=float(selected["severe_negative_fraction"]),
        contact_fraction=float(selected["contact_fraction"]),
        roughness_ratio=float(selected["roughness_ratio"]),
        edge_mismatch_ratio=float(selected["edge_mismatch_ratio"]),
        background_area_fraction=float(selected["background_area_fraction"]),
        corrected_area=float(selected["corrected_area"]),
        raw_area=float(selected["raw_area"]),
        warnings=warnings_list,
        candidate_scores=[
            {
                key: value
                for key, value in row.items()
                if key != "baseline"
            }
            for row in candidates
        ],
    )
    return {
        "background": baseline,
        "corrected": corrected,
        "protected_mask": protected_mask,
        "diagnostics": diagnostics.to_dict(),
        "parameters": asdict(params),
    }
