from __future__ import annotations

from typing import Callable
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks, peak_widths
from scipy.special import gamma as gamma_function, voigt_profile


SUPPORTED_MODELS = (
    "Gaussian",
    "Lorentzian",
    "Pseudo-Voigt",
)

ADVANCED_MODELS = (
    "Auto",
    "Pseudo-Voigt",
    "Pearson VII",
    "Split Pseudo-Voigt",
    "Voigt",
    "Gaussian",
    "Lorentzian",
)

BASELINE_MODELS = ("Constant", "Linear", "Quadratic")
ROBUST_LOSSES = ("Least squares", "Soft L1", "Huber", "Cauchy")
SELECTION_CRITERIA = ("BIC", "AICc")


class DeconvolutionCancelled(RuntimeError):
    """Raised when the user cancels an advanced deconvolution run."""



def gaussian_height(x, amplitude, center, fwhm):
    fwhm = np.maximum(np.asarray(fwhm), np.finfo(float).eps)
    return amplitude * np.exp(-4.0 * np.log(2.0) * ((x - center) / fwhm) ** 2)


def lorentzian_height(x, amplitude, center, fwhm):
    fwhm = np.maximum(np.asarray(fwhm), np.finfo(float).eps)
    return amplitude / (1.0 + 4.0 * ((x - center) / fwhm) ** 2)


def pseudo_voigt_height(x, amplitude, center, fwhm, eta):
    eta = np.clip(eta, 0.0, 1.0)
    return (
        eta * lorentzian_height(x, amplitude, center, fwhm)
        + (1.0 - eta) * gaussian_height(x, amplitude, center, fwhm)
    )


def pearson_vii_height(x, amplitude, center, fwhm, shape):
    shape = max(float(shape), 0.5001)
    coefficient = 4.0 * (2.0 ** (1.0 / shape) - 1.0)
    return amplitude * (
        1.0 + coefficient * ((x - center) / max(float(fwhm), np.finfo(float).eps)) ** 2
    ) ** (-shape)


def split_pseudo_voigt_height(
    x,
    amplitude,
    center,
    fwhm_left,
    fwhm_right,
    eta,
):
    x = np.asarray(x, dtype=float)
    widths = np.where(x < center, fwhm_left, fwhm_right)
    return pseudo_voigt_height(x, amplitude, center, widths, eta)


def voigt_height(x, amplitude, center, fwhm_g, fwhm_l):
    sigma = max(float(fwhm_g), np.finfo(float).eps) / (
        2.0 * math.sqrt(2.0 * math.log(2.0))
    )
    gamma = max(float(fwhm_l), np.finfo(float).eps) / 2.0
    values = voigt_profile(np.asarray(x, dtype=float) - center, sigma, gamma)
    maximum = float(voigt_profile(0.0, sigma, gamma))
    if maximum <= np.finfo(float).eps:
        return np.zeros_like(np.asarray(x, dtype=float))
    return amplitude * values / maximum


def effective_fwhm(component: dict) -> float:
    model = component["model"]
    if model == "Split Pseudo-Voigt":
        return 0.5 * (
            float(component["fwhm_left"]) + float(component["fwhm_right"])
        )
    if model == "Voigt":
        gaussian = float(component["fwhm_g"])
        lorentzian = float(component["fwhm_l"])
        return float(
            0.5346 * lorentzian
            + math.sqrt(0.2166 * lorentzian * lorentzian + gaussian * gaussian)
        )
    return float(component["fwhm"])


def profile_area(
    model: str,
    amplitude: float,
    fwhm: float,
    eta: float | None = None,
    shape: float | None = None,
    fwhm_left: float | None = None,
    fwhm_right: float | None = None,
    fwhm_g: float | None = None,
    fwhm_l: float | None = None,
) -> float:
    gaussian_area = amplitude * fwhm * math.sqrt(
        math.pi / (4.0 * math.log(2.0))
    )
    lorentzian_area = amplitude * math.pi * fwhm / 2.0

    if model == "Gaussian":
        return float(gaussian_area)
    if model == "Lorentzian":
        return float(lorentzian_area)
    if model == "Pseudo-Voigt":
        eta_value = 0.5 if eta is None else float(np.clip(eta, 0.0, 1.0))
        return float(
            eta_value * lorentzian_area
            + (1.0 - eta_value) * gaussian_area
        )
    if model == "Pearson VII":
        m = max(0.5001, 2.0 if shape is None else float(shape))
        coefficient = 4.0 * (2.0 ** (1.0 / m) - 1.0)
        return float(
            amplitude
            * fwhm
            * math.sqrt(math.pi)
            * float(gamma_function(m - 0.5))
            / (math.sqrt(coefficient) * float(gamma_function(m)))
        )
    if model == "Split Pseudo-Voigt":
        average_width = 0.5 * (
            float(fwhm_left if fwhm_left is not None else fwhm)
            + float(fwhm_right if fwhm_right is not None else fwhm)
        )
        return profile_area(
            "Pseudo-Voigt",
            amplitude,
            average_width,
            eta=eta,
        )
    if model == "Voigt":
        gaussian_width = float(fwhm_g if fwhm_g is not None else fwhm)
        lorentzian_width = float(fwhm_l if fwhm_l is not None else fwhm)
        width = max(
            0.5346 * lorentzian_width
            + math.sqrt(
                0.2166 * lorentzian_width * lorentzian_width
                + gaussian_width * gaussian_width
            ),
            np.finfo(float).eps,
        )
        grid = np.linspace(-60.0 * width, 60.0 * width, 12001)
        values = voigt_height(
            grid,
            amplitude,
            0.0,
            gaussian_width,
            lorentzian_width,
        )
        return float(np.trapezoid(values, grid))
    return float(gaussian_area)


def evaluate_component(x: np.ndarray, component: dict) -> np.ndarray:
    model = component["model"]
    amplitude = float(component["amplitude"])
    center = float(component["center"])

    if model == "Gaussian":
        return gaussian_height(
            x, amplitude, center, float(component["fwhm"])
        )
    if model == "Lorentzian":
        return lorentzian_height(
            x, amplitude, center, float(component["fwhm"])
        )
    if model == "Pseudo-Voigt":
        return pseudo_voigt_height(
            x,
            amplitude,
            center,
            float(component["fwhm"]),
            float(component.get("eta", 0.5)),
        )
    if model == "Pearson VII":
        return pearson_vii_height(
            x,
            amplitude,
            center,
            float(component["fwhm"]),
            float(component.get("shape", 2.0)),
        )
    if model == "Split Pseudo-Voigt":
        return split_pseudo_voigt_height(
            x,
            amplitude,
            center,
            float(component["fwhm_left"]),
            float(component["fwhm_right"]),
            float(component.get("eta", 0.5)),
        )
    if model == "Voigt":
        return voigt_height(
            x,
            amplitude,
            center,
            float(component["fwhm_g"]),
            float(component["fwhm_l"]),
        )
    raise ValueError(f"Unsupported component model: {model}")


def evaluate_fit_baseline(x: np.ndarray, group: dict) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    reference = float(group["reference_x"])

    if "baseline_coefficients" in group:
        coefficients = list(group["baseline_coefficients"])
        offset = float(coefficients[0]) if coefficients else 0.0
        slope = float(coefficients[1]) if len(coefficients) > 1 else 0.0
        curvature = float(coefficients[2]) if len(coefficients) > 2 else 0.0
        delta = x - reference
        return offset + slope * delta + curvature * delta * delta

    return (
        float(group.get("baseline_offset", 0.0))
        + float(group.get("baseline_slope", 0.0)) * (x - reference)
    )


def evaluate_fit_group(x: np.ndarray, group: dict) -> np.ndarray:
    result = evaluate_fit_baseline(x, group)
    for component in group["components"]:
        result = result + evaluate_component(x, component)
    return result


def _initial_fwhm(peak: dict, step: float) -> float:
    value = float(peak.get("fwhm") or 0.0)
    if not np.isfinite(value) or value <= step:
        value = step * 8.0
    return max(value, step * 2.0)


def _cluster_peaks(
    peaks: list[dict],
    step: float,
    window_multiplier: float,
) -> list[list[dict]]:
    ordered = sorted(peaks, key=lambda row: float(row["position"]))
    if not ordered:
        return []

    clusters: list[list[dict]] = [[ordered[0]]]
    current_right = float(ordered[0]["position"]) + (
        _initial_fwhm(ordered[0], step) * window_multiplier / 2.0
    )

    for peak in ordered[1:]:
        center = float(peak["position"])
        half_width = _initial_fwhm(peak, step) * window_multiplier / 2.0
        left = center - half_width
        if left <= current_right:
            clusters[-1].append(peak)
            current_right = max(current_right, center + half_width)
        else:
            clusters.append([peak])
            current_right = center + half_width
    return clusters


# ---------------------------------------------------------------------------
# Standard Phase-2 grouped fitting
# ---------------------------------------------------------------------------

def _component_parameter_count(model: str) -> int:
    return 4 if model == "Pseudo-Voigt" else 3


def _unpack_parameters(model: str, parameters: np.ndarray, component_count: int):
    baseline_offset = float(parameters[0])
    baseline_slope = float(parameters[1])
    components = []
    cursor = 2
    for _ in range(component_count):
        amplitude = float(parameters[cursor])
        center = float(parameters[cursor + 1])
        fwhm = float(parameters[cursor + 2])
        cursor += 3
        eta = None
        if model == "Pseudo-Voigt":
            eta = float(parameters[cursor])
            cursor += 1
        components.append((amplitude, center, fwhm, eta))
    return baseline_offset, baseline_slope, components


def _group_model_factory(
    model: str,
    component_count: int,
    reference_x: float,
) -> Callable:
    def model_function(x_values, *parameters):
        offset, slope, components = _unpack_parameters(
            model,
            np.asarray(parameters, dtype=float),
            component_count,
        )
        result = offset + slope * (x_values - reference_x)
        for amplitude, center, fwhm, eta in components:
            if model == "Gaussian":
                result = result + gaussian_height(
                    x_values, amplitude, center, fwhm
                )
            elif model == "Lorentzian":
                result = result + lorentzian_height(
                    x_values, amplitude, center, fwhm
                )
            else:
                result = result + pseudo_voigt_height(
                    x_values, amplitude, center, fwhm, eta
                )
        return result

    return model_function


def fit_detected_peaks(
    x: np.ndarray,
    y: np.ndarray,
    peaks: list[dict],
    model: str = "Pseudo-Voigt",
    window_multiplier: float = 4.0,
) -> tuple[list[dict], dict]:
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported peak model: {model}")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("X and Y must be one-dimensional arrays of equal length.")
    if len(x) < 10:
        raise ValueError("At least ten data points are required for peak fitting.")
    if not peaks:
        return [], {"groups": 0, "components": 0, "failed_groups": 0}

    differences = np.diff(x)
    positive_steps = differences[differences > 0]
    if positive_steps.size == 0:
        raise ValueError("X values must be strictly increasing.")
    step = float(np.median(positive_steps))
    window_multiplier = float(np.clip(window_multiplier, 1.5, 12.0))

    groups = _cluster_peaks(peaks, step, window_multiplier)
    fitted_groups: list[dict] = []
    failed_groups = 0

    for group_index, group_peaks in enumerate(groups, start=1):
        group_peaks = sorted(group_peaks, key=lambda row: float(row["position"]))
        half_widths = [
            _initial_fwhm(peak, step) * window_multiplier / 2.0
            for peak in group_peaks
        ]
        left = max(
            float(x[0]),
            min(
                float(peak["position"]) - half_width
                for peak, half_width in zip(group_peaks, half_widths)
            ),
        )
        right = min(
            float(x[-1]),
            max(
                float(peak["position"]) + half_width
                for peak, half_width in zip(group_peaks, half_widths)
            ),
        )

        mask = (x >= left) & (x <= right)
        x_window = x[mask]
        y_window = y[mask]
        parameter_count = (
            2 + len(group_peaks) * _component_parameter_count(model)
        )
        if len(x_window) <= parameter_count + 3:
            failed_groups += 1
            continue

        edge_count = max(2, min(10, len(x_window) // 8))
        left_level = float(np.median(y_window[:edge_count]))
        right_level = float(np.median(y_window[-edge_count:]))
        reference_x = float(np.mean(x_window))
        span_x = max(float(x_window[-1] - x_window[0]), step)
        baseline_slope = (right_level - left_level) / span_x
        baseline_offset = (
            left_level
            + baseline_slope * (reference_x - float(x_window[0]))
        )

        dynamic = max(
            float(np.max(y_window) - np.min(y_window)),
            float(np.std(y_window)),
            np.finfo(float).eps,
        )
        baseline_lower = float(np.min(y_window) - 2.0 * dynamic)
        baseline_upper = float(np.max(y_window) + 2.0 * dynamic)
        slope_limit = 10.0 * dynamic / span_x

        initial = [baseline_offset, baseline_slope]
        lower = [baseline_lower, -slope_limit]
        upper = [baseline_upper, slope_limit]

        centers = [float(peak["position"]) for peak in group_peaks]
        for peak_index, peak in enumerate(group_peaks):
            center = centers[peak_index]
            fwhm = _initial_fwhm(peak, step)
            nearest_index = int(np.argmin(np.abs(x_window - center)))
            local_baseline = baseline_offset + baseline_slope * (
                float(x_window[nearest_index]) - reference_x
            )
            amplitude = max(
                float(y_window[nearest_index] - local_baseline),
                dynamic * 0.03,
            )

            center_left = (
                float(x_window[0])
                if peak_index == 0
                else (centers[peak_index - 1] + center) / 2.0
            )
            center_right = (
                float(x_window[-1])
                if peak_index == len(group_peaks) - 1
                else (center + centers[peak_index + 1]) / 2.0
            )
            center_margin = max(step, fwhm * 0.75)
            center_left = max(center_left, center - center_margin)
            center_right = min(center_right, center + center_margin)
            if center_right <= center_left:
                center_left, center_right = center - step, center + step

            initial.extend([amplitude, center, fwhm])
            lower.extend([0.0, center_left, step * 1.05])
            upper.extend(
                [
                    dynamic * 5.0,
                    center_right,
                    min(span_x, max(fwhm * 5.0, step * 12.0)),
                ]
            )
            if model == "Pseudo-Voigt":
                initial.append(0.5)
                lower.append(0.0)
                upper.append(1.0)

        model_function = _group_model_factory(
            model,
            len(group_peaks),
            reference_x,
        )

        try:
            optimized, covariance = curve_fit(
                model_function,
                x_window,
                y_window,
                p0=np.asarray(initial, dtype=float),
                bounds=(
                    np.asarray(lower, dtype=float),
                    np.asarray(upper, dtype=float),
                ),
                maxfev=50000,
            )
        except Exception:
            failed_groups += 1
            continue

        y_fit = model_function(x_window, *optimized)
        residual = y_window - y_fit
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((y_window - np.mean(y_window)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        dof = max(1, len(y_window) - len(optimized))
        reduced_chi_square = float(ss_res / dof)

        if covariance is None or covariance.shape[0] != len(optimized):
            errors = np.full(len(optimized), np.nan)
        else:
            diagonal = np.diag(covariance)
            errors = np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))

        fit_offset, fit_slope, unpacked = _unpack_parameters(
            model,
            optimized,
            len(group_peaks),
        )
        _, _, error_components = _unpack_parameters(
            model,
            errors,
            len(group_peaks),
        )

        components = []
        for component_index, (
            values,
            uncertainties,
            source_peak,
        ) in enumerate(
            zip(unpacked, error_components, group_peaks),
            start=1,
        ):
            amplitude, center, fwhm, eta = values
            amplitude_error, center_error, fwhm_error, eta_error = uncertainties
            component = {
                "component_id": component_index,
                "model": model,
                "source_method": source_peak.get("method", "Regular"),
                "amplitude": float(amplitude),
                "amplitude_error": float(amplitude_error),
                "center": float(center),
                "center_error": float(center_error),
                "fwhm": float(fwhm),
                "fwhm_error": float(fwhm_error),
                "area": profile_area(model, amplitude, fwhm, eta),
                "area_fraction_percent": None,
                "eta": None if eta is None else float(eta),
                "eta_error": None if eta_error is None else float(eta_error),
                "snr": None,
                "separation_ratio": None,
                "quality_flags": [],
            }
            components.append(component)

        total_area = sum(component["area"] for component in components)
        for component in components:
            component["area_fraction_percent"] = (
                100.0 * component["area"] / total_area
                if total_area > 0
                else None
            )

        fitted_groups.append(
            {
                "group_id": group_index,
                "fit_mode": "Standard",
                "model": model,
                "baseline_model": "Linear",
                "baseline_coefficients": [
                    float(fit_offset),
                    float(fit_slope),
                    0.0,
                ],
                "window_min": float(x_window[0]),
                "window_max": float(x_window[-1]),
                "reference_x": reference_x,
                "baseline_offset": float(fit_offset),
                "baseline_slope": float(fit_slope),
                "r_squared": float(r_squared),
                "rmse": rmse,
                "reduced_chi_square": reduced_chi_square,
                "aic": None,
                "aicc": None,
                "bic": None,
                "durbin_watson": None,
                "max_parameter_correlation": None,
                "components": components,
                "quality_flags": [],
            }
        )

    diagnostics = {
        "groups": len(fitted_groups),
        "components": sum(
            len(group["components"]) for group in fitted_groups
        ),
        "failed_groups": failed_groups,
        "candidates": [],
    }
    return fitted_groups, diagnostics


# ---------------------------------------------------------------------------
# Advanced XRD deconvolution
# ---------------------------------------------------------------------------

def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float(np.finfo(float).eps)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = mad / 0.6744897501960817
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        sigma = float(np.std(values))
    return max(sigma, float(np.finfo(float).eps))


def _loss_name(loss: str) -> str:
    return {
        "Least squares": "linear",
        "Soft L1": "soft_l1",
        "Huber": "huber",
        "Cauchy": "cauchy",
    }.get(loss, "soft_l1")


def _baseline_parameter_names(model: str) -> list[str]:
    if model == "Constant":
        return ["offset"]
    if model == "Linear":
        return ["offset", "slope"]
    if model == "Quadratic":
        return ["offset", "slope", "curvature"]
    raise ValueError(f"Unsupported baseline model: {model}")


def _model_parameter_names(model: str) -> list[str]:
    if model in {"Gaussian", "Lorentzian"}:
        return ["amplitude", "center", "fwhm"]
    if model == "Pseudo-Voigt":
        return ["amplitude", "center", "fwhm", "eta"]
    if model == "Pearson VII":
        return ["amplitude", "center", "fwhm", "shape"]
    if model == "Split Pseudo-Voigt":
        return [
            "amplitude",
            "center",
            "fwhm_left",
            "fwhm_right",
            "eta",
        ]
    if model == "Voigt":
        return ["amplitude", "center", "fwhm_g", "fwhm_l"]
    raise ValueError(f"Unsupported advanced model: {model}")


def _supports_shared_width(model: str) -> bool:
    return model in {
        "Gaussian",
        "Lorentzian",
        "Pseudo-Voigt",
        "Pearson VII",
    }


def _advanced_model_candidates(requested_model: str) -> list[str]:
    if requested_model == "Auto":
        return [
            "Pseudo-Voigt",
            "Pearson VII",
            "Split Pseudo-Voigt",
        ]
    if requested_model not in ADVANCED_MODELS:
        raise ValueError(f"Unsupported advanced model: {requested_model}")
    return [requested_model]


def _baseline_initial(
    x: np.ndarray,
    y: np.ndarray,
    baseline_model: str,
    reference_x: float,
) -> tuple[list[float], list[float], list[float]]:
    point_count = len(x)
    edge_count = max(3, min(30, point_count // 7))
    edge_x = np.concatenate([x[:edge_count], x[-edge_count:]])
    edge_y = np.concatenate([y[:edge_count], y[-edge_count:]])
    delta = edge_x - reference_x
    degree = {"Constant": 0, "Linear": 1, "Quadratic": 2}[baseline_model]

    if degree == 0:
        coefficients = [float(np.median(edge_y))]
    else:
        fit = np.polyfit(delta, edge_y, degree)
        if degree == 1:
            coefficients = [float(fit[1]), float(fit[0])]
        else:
            coefficients = [
                float(fit[2]),
                float(fit[1]),
                float(fit[0]),
            ]

    dynamic = max(
        float(np.quantile(y, 0.995) - np.quantile(y, 0.05)),
        float(np.std(y)),
        np.finfo(float).eps,
    )
    span = max(float(x[-1] - x[0]), np.finfo(float).eps)
    lower = [float(np.min(y) - 3.0 * dynamic)]
    upper = [float(np.max(y) + 3.0 * dynamic)]

    if baseline_model in {"Linear", "Quadratic"}:
        slope_limit = 20.0 * dynamic / span
        lower.append(-slope_limit)
        upper.append(slope_limit)
    if baseline_model == "Quadratic":
        curvature_limit = 40.0 * dynamic / (span * span)
        lower.append(-curvature_limit)
        upper.append(curvature_limit)

    return coefficients, lower, upper


def _build_advanced_layout(
    model: str,
    component_count: int,
    baseline_model: str,
    shared_width: bool,
) -> list[tuple[str, int | None, str]]:
    layout: list[tuple[str, int | None, str]] = [
        ("baseline", None, name)
        for name in _baseline_parameter_names(baseline_model)
    ]
    shared = bool(shared_width and _supports_shared_width(model))
    if shared:
        layout.append(("global", None, "shared_fwhm"))

    for component_index in range(component_count):
        for parameter_name in _model_parameter_names(model):
            if shared and parameter_name == "fwhm":
                continue
            layout.append(
                ("component", component_index, parameter_name)
            )
    return layout


def _unpack_advanced(
    values: np.ndarray,
    layout: list[tuple[str, int | None, str]],
    model: str,
    component_count: int,
    baseline_model: str,
) -> tuple[list[float], list[dict]]:
    baseline_map = {
        name: 0.0 for name in _baseline_parameter_names(baseline_model)
    }
    components = [
        {"model": model} for _ in range(component_count)
    ]
    shared_fwhm = None

    for value, descriptor in zip(values, layout):
        scope, component_index, name = descriptor
        if scope == "baseline":
            baseline_map[name] = float(value)
        elif scope == "global":
            shared_fwhm = float(value)
        else:
            components[int(component_index)][name] = float(value)

    if shared_fwhm is not None:
        for component in components:
            component["fwhm"] = shared_fwhm

    coefficients = [
        baseline_map.get("offset", 0.0),
        baseline_map.get("slope", 0.0),
        baseline_map.get("curvature", 0.0),
    ]
    return coefficients, components


def _evaluate_advanced_vector(
    x: np.ndarray,
    values: np.ndarray,
    layout: list[tuple[str, int | None, str]],
    model: str,
    component_count: int,
    baseline_model: str,
    reference_x: float,
) -> np.ndarray:
    coefficients, components = _unpack_advanced(
        values,
        layout,
        model,
        component_count,
        baseline_model,
    )
    delta = np.asarray(x, dtype=float) - reference_x
    result = (
        coefficients[0]
        + coefficients[1] * delta
        + coefficients[2] * delta * delta
    )
    for component in components:
        result = result + evaluate_component(x, component)
    return result


def _seed_strength(peak: dict) -> float:
    for key in ("prominence", "intensity", "amplitude"):
        value = peak.get(key)
        if value is not None:
            try:
                numeric = float(value)
                if np.isfinite(numeric):
                    return numeric
            except (TypeError, ValueError):
                pass
    return 0.0


def _prepare_advanced_initial(
    x: np.ndarray,
    y: np.ndarray,
    seeds: list[dict],
    model: str,
    baseline_model: str,
    shared_width: bool,
    step: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[str, int | None, str]],
    float,
]:
    seeds = sorted(seeds, key=lambda row: float(row["position"]))
    reference_x = float(np.mean(x))
    span = max(float(x[-1] - x[0]), step)
    dynamic = max(
        float(np.quantile(y, 0.995) - np.quantile(y, 0.05)),
        float(np.std(y)),
        np.finfo(float).eps,
    )

    baseline_initial, baseline_lower, baseline_upper = _baseline_initial(
        x,
        y,
        baseline_model,
        reference_x,
    )
    layout = _build_advanced_layout(
        model,
        len(seeds),
        baseline_model,
        shared_width,
    )
    initial = list(baseline_initial)
    lower = list(baseline_lower)
    upper = list(baseline_upper)

    shared = bool(shared_width and _supports_shared_width(model))
    seed_widths = [_initial_fwhm(seed, step) for seed in seeds]
    if shared:
        shared_initial = float(np.median(seed_widths))
        initial.append(shared_initial)
        lower.append(step * 1.05)
        upper.append(min(span, max(shared_initial * 6.0, step * 20.0)))

    baseline_at_seed = lambda position: (
        baseline_initial[0]
        + (
            baseline_initial[1] * (position - reference_x)
            if len(baseline_initial) > 1 else 0.0
        )
        + (
            baseline_initial[2] * (position - reference_x) ** 2
            if len(baseline_initial) > 2 else 0.0
        )
    )

    centers = [float(seed["position"]) for seed in seeds]
    for index, seed in enumerate(seeds):
        center = centers[index]
        fwhm = seed_widths[index]
        nearest = int(np.argmin(np.abs(x - center)))
        amplitude = max(
            float(y[nearest] - baseline_at_seed(center)),
            dynamic * 0.02,
        )

        midpoint_left = (
            float(x[0])
            if index == 0
            else 0.5 * (centers[index - 1] + center)
        )
        midpoint_right = (
            float(x[-1])
            if index == len(seeds) - 1
            else 0.5 * (center + centers[index + 1])
        )
        center_margin = max(step * 2.0, fwhm * 1.25)
        center_lower = max(midpoint_left, center - center_margin)
        center_upper = min(midpoint_right, center + center_margin)
        if center_upper <= center_lower:
            center_lower = max(float(x[0]), center - step * 2.0)
            center_upper = min(float(x[-1]), center + step * 2.0)

        for parameter_name in _model_parameter_names(model):
            if shared and parameter_name == "fwhm":
                continue
            if parameter_name == "amplitude":
                initial.append(amplitude)
                lower.append(0.0)
                upper.append(dynamic * 8.0)
            elif parameter_name == "center":
                initial.append(center)
                lower.append(center_lower)
                upper.append(center_upper)
            elif parameter_name == "fwhm":
                initial.append(fwhm)
                lower.append(step * 1.05)
                upper.append(min(span, max(fwhm * 6.0, step * 20.0)))
            elif parameter_name in {"fwhm_left", "fwhm_right"}:
                initial.append(fwhm)
                lower.append(step * 1.05)
                upper.append(min(span, max(fwhm * 6.0, step * 20.0)))
            elif parameter_name in {"fwhm_g", "fwhm_l"}:
                initial.append(max(step * 1.5, fwhm * 0.65))
                lower.append(step * 0.75)
                upper.append(min(span, max(fwhm * 5.0, step * 18.0)))
            elif parameter_name == "eta":
                initial.append(0.5)
                lower.append(0.0)
                upper.append(1.0)
            elif parameter_name == "shape":
                initial.append(2.0)
                lower.append(0.55)
                upper.append(50.0)

    return (
        np.asarray(initial, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        layout,
        reference_x,
    )


def _information_criteria(
    residual: np.ndarray,
    parameter_count: int,
) -> dict:
    n = int(len(residual))
    k = int(parameter_count)
    rss = max(float(np.sum(np.square(residual))), np.finfo(float).eps)
    aic = n * math.log(rss / n) + 2.0 * k
    if n > k + 1:
        aicc = aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
    else:
        aicc = float("inf")
    bic = n * math.log(rss / n) + k * math.log(n)
    return {
        "rss": rss,
        "aic": float(aic),
        "aicc": float(aicc),
        "bic": float(bic),
    }


def _fit_advanced_candidate(
    x: np.ndarray,
    y: np.ndarray,
    seeds: list[dict],
    model: str,
    baseline_model: str,
    robust_loss: str,
    shared_width: bool,
    step: float,
    random_seed: int,
    progress_step: Callable[[str, bool], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict | None:
    maximum_optimization_points = 900
    stride = max(1, int(math.ceil(len(x) / maximum_optimization_points)))
    x_opt = x[::stride]
    y_opt = y[::stride]
    if x_opt[-1] != x[-1]:
        x_opt = np.append(x_opt, x[-1])
        y_opt = np.append(y_opt, y[-1])

    (
        initial,
        lower,
        upper,
        layout,
        reference_x,
    ) = _prepare_advanced_initial(
        x_opt,
        y_opt,
        seeds,
        model,
        baseline_model,
        shared_width,
        step,
    )

    if len(x_opt) <= len(initial) + 4:
        return None

    preliminary = gaussian_filter1d(y, sigma=1.0)
    noise_sigma = _robust_sigma(y - preliminary)
    f_scale = max(noise_sigma, np.finfo(float).eps)
    rng = np.random.default_rng(random_seed)
    best = None

    for attempt in range(2):
        if cancel_check is not None and cancel_check():
            raise DeconvolutionCancelled(
                "Advanced deconvolution was cancelled."
            )
        if progress_step is not None:
            progress_step(
                f"{model}: optimization start {attempt + 1}/2",
                False,
            )

        start = initial.copy()
        if attempt:
            for parameter_index, descriptor in enumerate(layout):
                scope, _, name = descriptor
                if scope == "component" and name == "center":
                    width_scale = max(
                        step * 2.0,
                        0.05 * float(x[-1] - x[0]),
                    )
                    start[parameter_index] += rng.normal(
                        0.0, width_scale * 0.15
                    )
                elif name in {
                    "fwhm",
                    "shared_fwhm",
                    "fwhm_left",
                    "fwhm_right",
                    "fwhm_g",
                    "fwhm_l",
                }:
                    start[parameter_index] *= float(
                        np.exp(rng.normal(0.0, 0.12))
                    )
                elif scope == "component" and name == "amplitude":
                    start[parameter_index] *= float(
                        np.exp(rng.normal(0.0, 0.10))
                    )
            start = np.clip(start, lower + 1e-12, upper - 1e-12)

        def residual_function(values):
            if cancel_check is not None and cancel_check():
                raise DeconvolutionCancelled(
                    "Advanced deconvolution was cancelled."
                )
            return (
                y_opt
                - _evaluate_advanced_vector(
                    x_opt,
                    values,
                    layout,
                    model,
                    len(seeds),
                    baseline_model,
                    reference_x,
                )
            )

        try:
            result = least_squares(
                residual_function,
                start,
                bounds=(lower, upper),
                loss=_loss_name(robust_loss),
                f_scale=f_scale,
                x_scale="jac",
                max_nfev=8000,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
        except DeconvolutionCancelled:
            raise
        except Exception:
            if progress_step is not None:
                progress_step(
                    f"{model}: optimization {attempt + 1}/2 failed",
                    True,
                )
            continue

        if progress_step is not None:
            progress_step(
                f"{model}: optimization {attempt + 1}/2 completed",
                True,
            )

        fitted = _evaluate_advanced_vector(
            x,
            result.x,
            layout,
            model,
            len(seeds),
            baseline_model,
            reference_x,
        )
        residual = y - fitted
        rss = float(np.sum(np.square(residual)))
        if best is None or rss < best["rss"]:
            best = {
                "result": result,
                "fitted": fitted,
                "residual": residual,
                "rss": rss,
                "layout": layout,
                "reference_x": reference_x,
                "noise_sigma": noise_sigma,
            }

    if best is None:
        return None

    result = best["result"]
    residual = best["residual"]
    criteria = _information_criteria(residual, len(result.x))
    dof = max(1, len(x) - len(result.x))
    optimization_dof = max(1, len(x_opt) - len(result.x))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    reduced_chi_square = float(criteria["rss"] / dof)
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    r_squared = (
        1.0 - criteria["rss"] / ss_tot if ss_tot > 0 else 1.0
    )
    denominator = max(
        float(np.sum(np.square(residual))),
        np.finfo(float).eps,
    )
    durbin_watson = float(
        np.sum(np.square(np.diff(residual))) / denominator
    )

    covariance = None
    parameter_errors = np.full(len(result.x), np.nan)
    maximum_correlation = None
    try:
        jacobian = result.jac
        covariance = (
            np.linalg.pinv(jacobian.T @ jacobian)
            * float(np.sum(np.square(result.fun)))
            / optimization_dof
        )
        diagonal = np.diag(covariance)
        parameter_errors = np.sqrt(
            np.where(diagonal >= 0, diagonal, np.nan)
        )
        standard = np.sqrt(np.maximum(diagonal, np.finfo(float).eps))
        correlation = covariance / np.outer(standard, standard)
        off_diagonal = np.abs(
            correlation - np.eye(len(correlation))
        )
        maximum_correlation = float(np.nanmax(off_diagonal))
    except Exception:
        covariance = None

    baseline_coefficients, components = _unpack_advanced(
        result.x,
        best["layout"],
        model,
        len(seeds),
        baseline_model,
    )
    _, component_errors = _unpack_advanced(
        parameter_errors,
        best["layout"],
        model,
        len(seeds),
        baseline_model,
    )

    built_components = []
    for component_index, (
        component,
        uncertainty,
        source_seed,
    ) in enumerate(
        zip(components, component_errors, seeds),
        start=1,
    ):
        component = dict(component)
        uncertainty = dict(uncertainty)
        component["component_id"] = component_index
        component["source_method"] = source_seed.get(
            "method", "Residual-guided"
        )
        component["amplitude_error"] = uncertainty.get("amplitude")
        component["center_error"] = uncertainty.get("center")

        if model == "Split Pseudo-Voigt":
            component["fwhm"] = effective_fwhm(component)
            left_error = uncertainty.get("fwhm_left")
            right_error = uncertainty.get("fwhm_right")
            if left_error is not None and right_error is not None:
                component["fwhm_error"] = 0.5 * math.sqrt(
                    float(left_error) ** 2 + float(right_error) ** 2
                )
            else:
                component["fwhm_error"] = None
        elif model == "Voigt":
            component["fwhm"] = effective_fwhm(component)
            component["fwhm_error"] = None
        else:
            component["fwhm_error"] = uncertainty.get("fwhm")

        component["eta_error"] = uncertainty.get("eta")
        component["shape_error"] = uncertainty.get("shape")
        component["area"] = profile_area(
            model,
            float(component["amplitude"]),
            float(component["fwhm"]),
            eta=component.get("eta"),
            shape=component.get("shape"),
            fwhm_left=component.get("fwhm_left"),
            fwhm_right=component.get("fwhm_right"),
            fwhm_g=component.get("fwhm_g"),
            fwhm_l=component.get("fwhm_l"),
        )
        component["snr"] = float(
            component["amplitude"] / best["noise_sigma"]
        )
        component["area_fraction_percent"] = None
        component["separation_ratio"] = None
        component["quality_flags"] = []
        built_components.append(component)

    total_area = sum(component["area"] for component in built_components)
    centers = np.asarray(
        [component["center"] for component in built_components],
        dtype=float,
    )

    for index, component in enumerate(built_components):
        if total_area > 0:
            component["area_fraction_percent"] = float(
                100.0 * component["area"] / total_area
            )
        if len(centers) > 1:
            distances = np.abs(centers - centers[index])
            distances[index] = np.inf
            nearest_distance = float(np.min(distances))
            neighbor_index = int(np.argmin(distances))
            mean_width = 0.5 * (
                effective_fwhm(component)
                + effective_fwhm(built_components[neighbor_index])
            )
            component["separation_ratio"] = (
                nearest_distance / mean_width
                if mean_width > 0 else None
            )

        flags = []
        if component["snr"] < 3.0:
            flags.append("weak")
        if (
            component["separation_ratio"] is not None
            and component["separation_ratio"] < 0.65
        ):
            flags.append("strongly overlapped")
        fwhm_error = component.get("fwhm_error")
        if (
            fwhm_error is not None
            and np.isfinite(float(fwhm_error))
            and float(fwhm_error) > 0.5 * effective_fwhm(component)
        ):
            flags.append("width unstable")
        center_error = component.get("center_error")
        if (
            center_error is not None
            and np.isfinite(float(center_error))
            and float(center_error) > 0.25 * effective_fwhm(component)
        ):
            flags.append("center unstable")
        if effective_fwhm(component) <= 1.5 * step:
            flags.append("spike-like width")
        if (
            component.get("area_fraction_percent") is not None
            and float(component["area_fraction_percent"]) < 0.5
        ):
            flags.append("negligible area")
        component["quality_flags"] = flags

    group_flags = []
    if maximum_correlation is not None and maximum_correlation > 0.98:
        group_flags.append("high parameter correlation")
    if durbin_watson < 1.2 or durbin_watson > 2.8:
        group_flags.append("structured residual")
    if any(component["snr"] < 3.0 for component in built_components):
        group_flags.append("contains weak component")
    if any(
        any(
            flag in component.get("quality_flags", [])
            for flag in (
                "spike-like width",
                "negligible area",
                "width unstable",
                "center unstable",
            )
        )
        for component in built_components
    ):
        group_flags.append("degenerate component")

    return {
        "fit_mode": "Advanced",
        "model": model,
        "baseline_model": baseline_model,
        "baseline_coefficients": [
            float(value) for value in baseline_coefficients
        ],
        "reference_x": float(best["reference_x"]),
        "r_squared": float(r_squared),
        "rmse": rmse,
        "reduced_chi_square": reduced_chi_square,
        "aic": criteria["aic"],
        "aicc": criteria["aicc"],
        "bic": criteria["bic"],
        "durbin_watson": durbin_watson,
        "max_parameter_correlation": maximum_correlation,
        "noise_sigma": float(best["noise_sigma"]),
        "components": built_components,
        "quality_flags": group_flags,
        "parameter_count": len(result.x),
        "point_count": len(x),
        "_fitted": best["fitted"],
        "_residual": residual,
    }


def _positive_residual_seed(
    x: np.ndarray,
    y: np.ndarray,
    candidate: dict,
    current_seeds: list[dict],
    step: float,
) -> dict | None:
    residual = np.asarray(candidate["_residual"], dtype=float)
    smoothed = gaussian_filter1d(residual, sigma=1.2)
    positive = np.clip(smoothed, 0.0, None)
    noise = max(float(candidate["noise_sigma"]), np.finfo(float).eps)
    prominence = max(
        2.5 * noise,
        0.05 * float(np.max(positive) - np.min(positive)),
    )
    indices, properties = find_peaks(
        positive,
        prominence=prominence,
        distance=max(2, int(round(3.0))),
        width=(1.0, None),
    )
    if len(indices) == 0:
        return None

    existing_centers = np.asarray(
        [float(seed["position"]) for seed in current_seeds],
        dtype=float,
    )
    existing_widths = np.asarray(
        [_initial_fwhm(seed, step) for seed in current_seeds],
        dtype=float,
    )
    order = np.argsort(properties["prominences"])[::-1]
    widths, _, left_ips, right_ips = peak_widths(
        positive,
        indices,
        rel_height=0.5,
    )

    for ordered_index in order:
        residual_index = int(indices[ordered_index])
        center = float(x[residual_index])
        estimated_width = max(
            step * 2.0,
            float(
                np.interp(
                    right_ips[ordered_index],
                    np.arange(len(x)),
                    x,
                )
                - np.interp(
                    left_ips[ordered_index],
                    np.arange(len(x)),
                    x,
                )
            ),
        )
        minimum_residual_width = max(
            step * 2.5,
            0.12 * float(np.median(existing_widths))
            if len(existing_widths) else step * 2.5,
        )
        if estimated_width < minimum_residual_width:
            continue

        if len(existing_centers):
            nearest = int(np.argmin(np.abs(existing_centers - center)))
            minimum_separation = max(
                step * 2.0,
                0.35 * float(existing_widths[nearest]),
            )
            if abs(existing_centers[nearest] - center) < minimum_separation:
                continue

        return {
            "position": center,
            "intensity": float(y[residual_index]),
            "prominence": float(properties["prominences"][ordered_index]),
            "fwhm": estimated_width,
            "method": "Residual-guided",
        }
    return None


def _candidate_summary(
    group_index: int,
    candidate_index: int,
    candidate: dict,
    chosen: bool,
) -> dict:
    return {
        "group_id": group_index,
        "candidate_id": candidate_index,
        "model": candidate["model"],
        "baseline_model": candidate["baseline_model"],
        "component_count": len(candidate["components"]),
        "parameter_count": candidate["parameter_count"],
        "point_count": candidate["point_count"],
        "aic": candidate["aic"],
        "aicc": candidate["aicc"],
        "bic": candidate["bic"],
        "r_squared": candidate["r_squared"],
        "rmse": candidate["rmse"],
        "durbin_watson": candidate["durbin_watson"],
        "max_parameter_correlation": candidate[
            "max_parameter_correlation"
        ],
        "quality_flags": list(candidate["quality_flags"]),
        "selection_score": candidate.get("selection_score"),
        "chosen": bool(chosen),
    }


def advanced_deconvolve_peaks(
    x: np.ndarray,
    y: np.ndarray,
    peaks: list[dict],
    model: str = "Auto",
    window_multiplier: float = 5.0,
    baseline_model: str = "Quadratic",
    robust_loss: str = "Soft L1",
    maximum_extra_components: int = 2,
    selection_criterion: str = "BIC",
    shared_width: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[list[dict], dict]:
    """
    Residual-guided, model-selected peak deconvolution for overlapping XRD peaks.

    The routine clusters nearby seed peaks, fits multiple model/component
    candidates with robust nonlinear least squares, adds hidden shoulder
    components at positive residual maxima, optionally tests one fewer seed
    component, and selects the preferred solution using BIC or AICc.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("X and Y must be one-dimensional arrays of equal length.")
    if len(x) < 15:
        raise ValueError("At least fifteen points are required for advanced fitting.")
    if not peaks:
        return [], {
            "groups": 0,
            "components": 0,
            "failed_groups": 0,
            "candidates": [],
        }
    if baseline_model not in BASELINE_MODELS:
        raise ValueError(f"Unsupported baseline model: {baseline_model}")
    if robust_loss not in ROBUST_LOSSES:
        raise ValueError(f"Unsupported robust loss: {robust_loss}")
    if selection_criterion not in SELECTION_CRITERIA:
        raise ValueError(
            f"Unsupported selection criterion: {selection_criterion}"
        )

    positive_steps = np.diff(x)
    positive_steps = positive_steps[positive_steps > 0]
    if not len(positive_steps):
        raise ValueError("X values must be strictly increasing.")
    step = float(np.median(positive_steps))
    window_multiplier = float(np.clip(window_multiplier, 2.0, 15.0))
    maximum_extra_components = int(
        np.clip(maximum_extra_components, 0, 5)
    )

    clusters = _cluster_peaks(peaks, step, window_multiplier)
    models = _advanced_model_candidates(model)
    attempts_per_candidate = 2

    # Initial total includes known seed sets and one final-selection step
    # for every group. Hidden shoulder rounds add steps dynamically.
    total_steps = sum(
        (
            (1 + int(len(cluster) > 1))
            * len(models)
            * attempts_per_candidate
        )
        + 1
        for cluster in clusters
    )
    total_steps = max(1, int(total_steps))
    completed_steps = 0

    def cancelled() -> bool:
        return bool(cancel_check is not None and cancel_check())

    def emit_progress(
        message: str,
        advance: bool = False,
        add_total: int = 0,
    ) -> None:
        nonlocal completed_steps, total_steps
        if add_total:
            total_steps += int(add_total)
        if advance:
            completed_steps += 1
        if progress_callback is not None:
            progress_callback(
                int(completed_steps),
                int(max(total_steps, completed_steps, 1)),
                str(message),
            )

    if cancelled():
        raise DeconvolutionCancelled(
            "Advanced deconvolution was cancelled."
        )
    emit_progress(
        f"Prepared {len(clusters)} overlapping peak group(s)."
    )

    fitted_groups = []
    candidate_rows = []
    failed_groups = 0

    for group_index, cluster in enumerate(clusters, start=1):
        if cancelled():
            raise DeconvolutionCancelled(
                "Advanced deconvolution was cancelled."
            )
        emit_progress(
            f"Group {group_index}/{len(clusters)}: preparing fitting window."
        )
        cluster = sorted(cluster, key=lambda row: float(row["position"]))
        half_widths = [
            _initial_fwhm(seed, step) * window_multiplier / 2.0
            for seed in cluster
        ]
        left = max(
            float(x[0]),
            min(
                float(seed["position"]) - half_width
                for seed, half_width in zip(cluster, half_widths)
            ),
        )
        right = min(
            float(x[-1]),
            max(
                float(seed["position"]) + half_width
                for seed, half_width in zip(cluster, half_widths)
            ),
        )
        margin = max(step * 4.0, 0.08 * (right - left))
        left = max(float(x[0]), left - margin)
        right = min(float(x[-1]), right + margin)
        mask = (x >= left) & (x <= right)
        x_window = x[mask]
        y_window = y[mask]

        candidate_seed_sets: list[list[dict]] = [list(cluster)]
        if len(cluster) > 1:
            strongest = sorted(
                cluster,
                key=_seed_strength,
                reverse=True,
            )[: len(cluster) - 1]
            candidate_seed_sets.append(
                sorted(strongest, key=lambda row: float(row["position"]))
            )

        candidates = []
        seen_seed_signatures = set()

        def fit_seed_set(seed_set: list[dict]):
            signature = tuple(
                round(float(seed["position"]), 8)
                for seed in sorted(
                    seed_set,
                    key=lambda row: float(row["position"]),
                )
            )
            if signature in seen_seed_signatures:
                return []
            seen_seed_signatures.add(signature)
            fitted_for_seed_set = []
            for model_index, candidate_model in enumerate(models):
                if cancelled():
                    raise DeconvolutionCancelled(
                        "Advanced deconvolution was cancelled."
                    )

                component_count = len(seed_set)

                def candidate_progress(
                    detail: str,
                    advance: bool,
                    current_model=candidate_model,
                    current_components=component_count,
                ):
                    emit_progress(
                        (
                            f"Group {group_index}/{len(clusters)} — "
                            f"{current_model}, {current_components} component(s): "
                            f"{detail}"
                        ),
                        advance=advance,
                    )

                fitted = _fit_advanced_candidate(
                    x_window,
                    y_window,
                    sorted(
                        seed_set,
                        key=lambda row: float(row["position"]),
                    ),
                    candidate_model,
                    baseline_model,
                    robust_loss,
                    shared_width,
                    step,
                    random_seed=(
                        group_index * 1000
                        + len(seed_set) * 100
                        + model_index
                    ),
                    progress_step=candidate_progress,
                    cancel_check=cancelled,
                )
                if fitted is not None:
                    fitted_for_seed_set.append(fitted)
                    candidates.append(fitted)
            return fitted_for_seed_set

        for seed_set in candidate_seed_sets:
            fit_seed_set(seed_set)

        current_seeds = list(cluster)
        for _ in range(maximum_extra_components):
            if not candidates:
                break
            criterion_key = selection_criterion.lower()

            def candidate_score(row):
                score = float(row[criterion_key])
                flags = set(row.get("quality_flags", []))
                if "degenerate component" in flags:
                    score += 2000.0
                if "contains weak component" in flags:
                    score += 50.0
                if "high parameter correlation" in flags:
                    score += 20.0
                return score

            current_best = min(candidates, key=candidate_score)
            extra_seed = _positive_residual_seed(
                x_window,
                y_window,
                current_best,
                current_seeds,
                step,
            )
            if extra_seed is None:
                emit_progress(
                    f"Group {group_index}/{len(clusters)}: no additional shoulder found."
                )
                break

            emit_progress(
                (
                    f"Group {group_index}/{len(clusters)}: hidden shoulder "
                    f"candidate at {float(extra_seed['position']):.6g}°."
                ),
                add_total=len(models) * attempts_per_candidate,
            )
            current_seeds = sorted(
                current_seeds + [extra_seed],
                key=lambda row: float(row["position"]),
            )
            new_candidates = fit_seed_set(current_seeds)
            if not new_candidates:
                break

        if not candidates:
            failed_groups += 1
            emit_progress(
                f"Group {group_index}/{len(clusters)}: no valid candidate fit.",
                advance=True,
            )
            continue

        criterion_key = selection_criterion.lower()

        def final_candidate_score(row):
            score = float(row[criterion_key])
            flags = set(row.get("quality_flags", []))
            if "degenerate component" in flags:
                score += 2000.0
            if "contains weak component" in flags:
                score += 50.0
            if "high parameter correlation" in flags:
                score += 20.0
            return score

        for candidate in candidates:
            candidate["selection_score"] = final_candidate_score(candidate)
        chosen = min(candidates, key=lambda row: row["selection_score"])
        chosen_index = candidates.index(chosen)

        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_rows.append(
                _candidate_summary(
                    group_index,
                    candidate_index,
                    candidate,
                    chosen=(candidate_index - 1 == chosen_index),
                )
            )

        chosen = dict(chosen)
        chosen.pop("_fitted", None)
        chosen.pop("_residual", None)
        chosen.update(
            {
                "group_id": group_index,
                "window_min": float(x_window[0]),
                "window_max": float(x_window[-1]),
                "selection_criterion": selection_criterion,
                "candidate_count": len(candidates),
                "robust_loss": robust_loss,
                "shared_width": bool(
                    shared_width and _supports_shared_width(chosen["model"])
                ),
            }
        )
        fitted_groups.append(chosen)
        emit_progress(
            (
                f"Group {group_index}/{len(clusters)}: selected "
                f"{chosen['model']} with {len(chosen['components'])} component(s)."
            ),
            advance=True,
        )

    if cancelled():
        raise DeconvolutionCancelled(
            "Advanced deconvolution was cancelled."
        )
    if progress_callback is not None:
        progress_callback(
            int(max(total_steps, completed_steps)),
            int(max(total_steps, completed_steps, 1)),
            "Advanced deconvolution completed.",
        )

    diagnostics = {
        "groups": len(fitted_groups),
        "components": sum(
            len(group["components"]) for group in fitted_groups
        ),
        "failed_groups": failed_groups,
        "candidates": candidate_rows,
        "advanced": True,
        "selection_criterion": selection_criterion,
        "progress_steps_completed": int(completed_steps),
        "progress_steps_planned": int(total_steps),
    }
    return fitted_groups, diagnostics
