from __future__ import annotations

"""Shared crystallographic line-profile and instrument-response models.

The functions in this module are intentionally independent from the Qt user
interface. They are used by instrument calibration, Pawley/Le Bail extraction
and structure-constrained Rietveld refinement so that all three workflows use
one definition of the profile parameters.
"""

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from .instrument_physics import axial_divergence_characteristics


PROFILE_MODELS = (
    "Pseudo-Voigt U-V-W",
    "TCH pseudo-Voigt",
    "Split pseudo-Voigt",
    "Gaussian U-V-W",
    "Lorentzian X-Y",
)

ADVANCED_PROFILE_MODELS = (
    "Pseudo-Voigt U-V-W",
    "TCH pseudo-Voigt",
    "Split pseudo-Voigt",
)


class ProfileModelError(ValueError):
    pass


@dataclass(frozen=True)
class ProfileCharacteristics:
    model: str
    fwhm_deg: float
    gaussian_fwhm_deg: float
    lorentzian_fwhm_deg: float
    eta: float
    left_fwhm_deg: float
    right_fwhm_deg: float
    asymmetry_factor: float


def _theta_rad(two_theta_deg: float | np.ndarray) -> np.ndarray:
    return np.radians(np.asarray(two_theta_deg, dtype=float) / 2.0)


def caglioti_fwhm_deg(
    two_theta_deg: float | np.ndarray,
    u: float,
    v: float,
    w: float,
    minimum_width: float = 0.0,
) -> float | np.ndarray:
    theta = _theta_rad(two_theta_deg)
    tangent = np.tan(theta)
    squared = float(u) * tangent * tangent + float(v) * tangent + float(w)
    width = np.sqrt(np.maximum(squared, max(float(minimum_width), 0.0) ** 2))
    return float(width) if np.ndim(two_theta_deg) == 0 else width


def lorentzian_fwhm_deg(
    two_theta_deg: float | np.ndarray,
    x: float,
    y: float,
    minimum_width: float = 0.0,
) -> float | np.ndarray:
    theta = _theta_rad(two_theta_deg)
    cosine = np.maximum(np.cos(theta), 1e-8)
    tangent = np.maximum(np.tan(theta), 0.0)
    width = np.maximum(
        float(x) / cosine + float(y) * tangent,
        max(float(minimum_width), 0.0),
    )
    return float(width) if np.ndim(two_theta_deg) == 0 else width


def tch_fwhm_eta(
    gaussian_fwhm_deg: float | np.ndarray,
    lorentzian_fwhm_deg_value: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return the Thompson-Cox-Hastings approximate FWHM and mixing.

    The fifth-order approximation combines Gaussian and Lorentzian integral
    breadth contributions into a normalized pseudo-Voigt representation.
    """

    hg = np.maximum(np.asarray(gaussian_fwhm_deg, dtype=float), 0.0)
    hl = np.maximum(np.asarray(lorentzian_fwhm_deg_value, dtype=float), 0.0)
    h5 = (
        hg**5
        + 2.69269 * hg**4 * hl
        + 2.42843 * hg**3 * hl**2
        + 4.47163 * hg**2 * hl**3
        + 0.07842 * hg * hl**4
        + hl**5
    )
    h = np.maximum(h5, 1e-30) ** 0.2
    ratio = np.clip(hl / h, 0.0, 1.0)
    eta = np.clip(
        1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3,
        0.0,
        1.0,
    )
    if np.ndim(gaussian_fwhm_deg) == 0 and np.ndim(lorentzian_fwhm_deg_value) == 0:
        return float(h), float(eta)
    return h, eta


def _split_widths(
    two_theta_deg: float,
    width: float,
    axial_asymmetry: float,
) -> tuple[float, float, float]:
    """Empirical low-angle split-width approximation.

    This is a stable asymmetric profile approximation, not a full
    fundamental-parameters axial-divergence convolution. The result is reported
    explicitly as an empirical split profile in exported diagnostics.
    """

    theta = max(math.radians(float(two_theta_deg) / 2.0), math.radians(0.5))
    low_angle_scale = min(3.0, max(0.0, 1.0 / max(math.tan(theta), 1e-6)) / 5.0)
    factor = float(np.clip(axial_asymmetry, -0.75, 0.75)) * low_angle_scale
    left = max(width * 0.2, width * (1.0 + max(factor, -0.65)))
    right = max(width * 0.2, width * (1.0 - min(factor, 0.65) * 0.55))
    return float(left), float(right), float(factor)


def profile_characteristics(
    two_theta_deg: float,
    *,
    model: str = "Pseudo-Voigt U-V-W",
    u: float = 0.005,
    v: float = 0.0,
    w: float = 0.02,
    eta: float = 0.5,
    x: float = 0.0,
    y: float = 0.0,
    axial_asymmetry: float = 0.0,
    axial_sh_over_l: float = 0.0,
    minimum_width: float = 0.003,
) -> ProfileCharacteristics:
    if model not in PROFILE_MODELS:
        raise ProfileModelError(f"Unsupported profile model: {model}")
    minimum = max(float(minimum_width), 1e-8)
    hg = float(caglioti_fwhm_deg(two_theta_deg, u, v, w, minimum))
    hl = float(lorentzian_fwhm_deg(two_theta_deg, max(x, 0.0), max(y, 0.0), 0.0))

    if model == "Gaussian U-V-W":
        width, mixing = hg, 0.0
    elif model == "Lorentzian X-Y":
        width = max(hl, minimum)
        mixing = 1.0
    elif model == "TCH pseudo-Voigt":
        width, mixing = tch_fwhm_eta(hg, hl)
        width = max(float(width), minimum)
        mixing = float(mixing)
    else:
        width = hg
        mixing = float(np.clip(eta, 0.0, 1.0))

    if float(axial_sh_over_l) > 0.0:
        axial = axial_divergence_characteristics(
            two_theta_deg,
            sh_over_l=axial_sh_over_l,
            base_fwhm_deg=width,
        )
        left = axial["left_fwhm_deg"]
        right = axial["right_fwhm_deg"]
        factor = axial["asymmetry_factor"]
    elif model == "Split pseudo-Voigt" or abs(float(axial_asymmetry)) > 1e-12:
        left, right, factor = _split_widths(
            two_theta_deg,
            width,
            axial_asymmetry,
        )
    else:
        left = right = width
        factor = 0.0

    return ProfileCharacteristics(
        model=model,
        fwhm_deg=float(width),
        gaussian_fwhm_deg=float(hg),
        lorentzian_fwhm_deg=float(hl),
        eta=float(mixing),
        left_fwhm_deg=float(left),
        right_fwhm_deg=float(right),
        asymmetry_factor=float(factor),
    )


def _pv_unnormalized(delta: np.ndarray, width: np.ndarray, eta: float) -> np.ndarray:
    width = np.maximum(np.asarray(width, dtype=float), 1e-12)
    scaled = np.asarray(delta, dtype=float) / width
    gaussian = np.exp(-4.0 * np.log(2.0) * scaled * scaled)
    lorentzian = 1.0 / (1.0 + 4.0 * scaled * scaled)
    return float(np.clip(eta, 0.0, 1.0)) * lorentzian + (
        1.0 - float(np.clip(eta, 0.0, 1.0))
    ) * gaussian


def profile_unit_area(
    x_values: np.ndarray,
    center: float,
    *,
    model: str = "Pseudo-Voigt U-V-W",
    u: float = 0.005,
    v: float = 0.0,
    w: float = 0.02,
    eta: float = 0.5,
    x: float = 0.0,
    y: float = 0.0,
    axial_asymmetry: float = 0.0,
    axial_sh_over_l: float = 0.0,
    minimum_width: float = 0.003,
) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=float)
    characteristics = profile_characteristics(
        center,
        model=model,
        u=u,
        v=v,
        w=w,
        eta=eta,
        x=x,
        y=y,
        axial_asymmetry=axial_asymmetry,
        axial_sh_over_l=axial_sh_over_l,
        minimum_width=minimum_width,
    )
    delta = x_values - float(center)
    widths = np.where(
        delta < 0.0,
        characteristics.left_fwhm_deg,
        characteristics.right_fwhm_deg,
    )
    values = _pv_unnormalized(delta, widths, characteristics.eta)
    width_sum = characteristics.left_fwhm_deg + characteristics.right_fwhm_deg
    gaussian_area = (
        width_sum * math.sqrt(math.pi) / (4.0 * math.sqrt(math.log(2.0)))
    )
    lorentzian_area = math.pi * width_sum / 4.0
    area = (
        characteristics.eta * lorentzian_area
        + (1.0 - characteristics.eta) * gaussian_area
    )
    if not np.isfinite(area) or area <= np.finfo(float).eps:
        return np.zeros_like(x_values)
    return values / area


def information_criteria(
    residuals: np.ndarray,
    parameter_count: int,
) -> dict:
    residuals = np.asarray(residuals, dtype=float)
    n = int(residuals.size)
    k = max(0, int(parameter_count))
    rss = max(float(np.sum(residuals * residuals)), 1e-30)
    aic = n * math.log(rss / max(n, 1)) + 2.0 * k
    correction = (
        2.0 * k * (k + 1.0) / max(n - k - 1.0, 1.0)
        if n > k + 1
        else float("inf")
    )
    bic = n * math.log(rss / max(n, 1)) + k * math.log(max(n, 2))
    return {
        "rss": rss,
        "aic": float(aic),
        "aicc": float(aic + correction),
        "bic": float(bic),
        "observation_count": n,
        "parameter_count": k,
    }


def residual_diagnostics(residuals: np.ndarray) -> dict:
    residuals = np.asarray(residuals, dtype=float)
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        return {
            "rmse": None,
            "mae": None,
            "durbin_watson": None,
            "lag1_autocorrelation": None,
            "runs": 0,
        }
    rmse = float(np.sqrt(np.mean(finite * finite)))
    mae = float(np.mean(np.abs(finite)))
    denominator = float(np.sum(finite * finite))
    durbin_watson = (
        float(np.sum(np.diff(finite) ** 2) / denominator)
        if finite.size > 1 and denominator > 0
        else None
    )
    if finite.size > 2 and np.std(finite[:-1]) > 0 and np.std(finite[1:]) > 0:
        lag1 = float(np.corrcoef(finite[:-1], finite[1:])[0, 1])
    else:
        lag1 = None
    signs = np.signbit(finite)
    runs = 1 + int(np.count_nonzero(signs[1:] != signs[:-1])) if finite.size else 0
    return {
        "rmse": rmse,
        "mae": mae,
        "durbin_watson": durbin_watson,
        "lag1_autocorrelation": lag1,
        "runs": runs,
    }


def _initial_caglioti(two_theta: np.ndarray, widths: np.ndarray) -> tuple[float, float, float]:
    tangent = np.tan(_theta_rad(two_theta))
    design = np.column_stack([tangent * tangent, tangent, np.ones_like(tangent)])
    coefficients, *_ = np.linalg.lstsq(design, widths * widths, rcond=None)
    u, v, w = [float(value) for value in coefficients]
    return max(u, 0.0), v, max(w, 1e-8)


def fit_instrument_width_models(
    two_theta_deg: Iterable[float],
    observed_fwhm_deg: Iterable[float],
    *,
    fwhm_errors_deg: Iterable[float] | None = None,
    left_fwhm_deg: Iterable[float] | None = None,
    right_fwhm_deg: Iterable[float] | None = None,
    candidate_models: Iterable[str] = ADVANCED_PROFILE_MODELS,
) -> dict:
    two_theta = np.asarray(list(two_theta_deg), dtype=float)
    widths = np.asarray(list(observed_fwhm_deg), dtype=float)
    if two_theta.ndim != 1 or widths.ndim != 1 or len(two_theta) != len(widths):
        raise ProfileModelError("Peak positions and widths must be equal one-dimensional arrays.")
    if len(two_theta) < 3 or np.any(widths <= 0) or not np.all(np.isfinite(widths)):
        raise ProfileModelError("At least three finite positive peak widths are required.")
    if fwhm_errors_deg is None:
        errors = np.maximum(0.003, widths * 0.03)
    else:
        errors = np.maximum(np.asarray(list(fwhm_errors_deg), dtype=float), 1e-5)
        if len(errors) != len(widths):
            raise ProfileModelError("FWHM uncertainty count does not match peak count.")
    left = None if left_fwhm_deg is None else np.asarray(list(left_fwhm_deg), dtype=float)
    right = None if right_fwhm_deg is None else np.asarray(list(right_fwhm_deg), dtype=float)
    if left is not None and (right is None or len(left) != len(widths) or len(right) != len(widths)):
        raise ProfileModelError("Left and right width arrays must match the observed widths.")

    u0, v0, w0 = _initial_caglioti(two_theta, widths)
    candidates = []
    for model in candidate_models:
        if model not in ADVANCED_PROFILE_MODELS:
            continue
        if model == "TCH pseudo-Voigt":
            initial = np.asarray([u0 * 0.8, v0, w0 * 0.8, 0.02, 0.001], dtype=float)
            lower = np.asarray([0.0, -5.0, 1e-10, 0.0, 0.0])
            upper = np.asarray([10.0, 5.0, 10.0, 2.0, 2.0])

            def residual(parameters):
                predicted = np.asarray([
                    profile_characteristics(
                        value,
                        model=model,
                        u=parameters[0],
                        v=parameters[1],
                        w=parameters[2],
                        x=parameters[3],
                        y=parameters[4],
                    ).fwhm_deg
                    for value in two_theta
                ])
                # Weak regularization avoids trading arbitrary Gaussian and
                # Lorentzian terms when only total FWHM values are available.
                regularization = np.asarray([
                    parameters[3] / 5.0,
                    parameters[4] / 5.0,
                ])
                return np.concatenate([(widths - predicted) / errors, regularization])

            names = ("caglioti_u", "caglioti_v", "caglioti_w", "lorentzian_x", "lorentzian_y")
        elif model == "Split pseudo-Voigt":
            initial = np.asarray([u0, v0, w0, 0.5, 0.0], dtype=float)
            lower = np.asarray([0.0, -5.0, 1e-10, 0.0, -0.75])
            upper = np.asarray([10.0, 5.0, 10.0, 1.0, 0.75])

            def residual(parameters):
                chars = [
                    profile_characteristics(
                        value,
                        model=model,
                        u=parameters[0],
                        v=parameters[1],
                        w=parameters[2],
                        eta=parameters[3],
                        axial_asymmetry=parameters[4],
                    )
                    for value in two_theta
                ]
                total = (widths - np.asarray([item.fwhm_deg for item in chars])) / errors
                if left is not None and right is not None:
                    split_error = np.maximum(errors, 0.005)
                    total = np.concatenate([
                        total,
                        (left - np.asarray([item.left_fwhm_deg for item in chars])) / split_error,
                        (right - np.asarray([item.right_fwhm_deg for item in chars])) / split_error,
                    ])
                else:
                    total = np.concatenate([total, np.asarray([parameters[4] / 0.25])])
                return total

            names = ("caglioti_u", "caglioti_v", "caglioti_w", "eta", "axial_asymmetry")
        else:
            initial = np.asarray([u0, v0, w0, 0.5], dtype=float)
            lower = np.asarray([0.0, -5.0, 1e-10, 0.0])
            upper = np.asarray([10.0, 5.0, 10.0, 1.0])

            def residual(parameters):
                predicted = caglioti_fwhm_deg(
                    two_theta,
                    parameters[0],
                    parameters[1],
                    parameters[2],
                    1e-8,
                )
                return (widths - predicted) / errors

            names = ("caglioti_u", "caglioti_v", "caglioti_w", "eta")

        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            max_nfev=20000,
        )
        parameters = {name: float(value) for name, value in zip(names, result.x)}
        covariance = np.linalg.pinv(result.jac.T @ result.jac)
        dof = max(1, len(result.fun) - len(result.x))
        covariance *= max(float(np.sum(np.square(result.fun)) / dof), 1.0)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = np.outer(scale, scale)
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0,
        )
        condition_number = float(np.linalg.cond(result.jac.T @ result.jac))
        if not np.isfinite(condition_number):
            condition_number = None
        predicted_chars = [
            profile_characteristics(value, model=model, **{
                "u": parameters.get("caglioti_u", u0),
                "v": parameters.get("caglioti_v", v0),
                "w": parameters.get("caglioti_w", w0),
                "eta": parameters.get("eta", 0.5),
                "x": parameters.get("lorentzian_x", 0.0),
                "y": parameters.get("lorentzian_y", 0.0),
                "axial_asymmetry": parameters.get("axial_asymmetry", 0.0),
            })
            for value in two_theta
        ]
        predicted = np.asarray([item.fwhm_deg for item in predicted_chars])
        raw_residual = widths - predicted
        criteria = information_criteria(raw_residual, len(parameters))
        candidates.append({
            "model": model,
            "success": bool(result.success),
            "message": str(result.message),
            "parameters": parameters,
            "parameter_names": list(names),
            "parameter_standard_errors": {
                name: float(value)
                for name, value in zip(names, standard_errors)
            },
            "covariance_matrix": covariance.tolist(),
            "correlation_matrix": correlation.tolist(),
            "normal_matrix_condition_number": condition_number,
            "calculated_fwhm_deg": predicted.tolist(),
            "width_rmse_deg": float(np.sqrt(np.mean(raw_residual**2))),
            "width_mae_deg": float(np.mean(np.abs(raw_residual))),
            "information_criteria": criteria,
        })

    if not candidates:
        raise ProfileModelError("No supported instrument profile models were requested.")
    candidates.sort(key=lambda row: (row["information_criteria"]["bic"], row["width_rmse_deg"]))
    best = dict(candidates[0])
    best["classification"] = (
        "Preferred by BIC"
        if len(candidates) > 1
        else "Only model evaluated"
    )
    return {
        "best_model": best["model"],
        "best": best,
        "candidates": candidates,
        "observation_count": int(len(widths)),
        "two_theta_min_deg": float(np.min(two_theta)),
        "two_theta_max_deg": float(np.max(two_theta)),
    }



def staged_parameter_groups(parameter_names: Iterable[str]) -> list[dict]:
    """Group nonlinear parameters into a conservative refinement sequence."""

    names = list(parameter_names)
    groups: list[dict] = []
    definitions = (
        (
            "Position and cell",
            lambda name: name == "zero_shift_deg"
            or (name.startswith("phase") and any(
                name.endswith("_" + suffix)
                for suffix in ("a", "b", "c", "alpha", "beta", "gamma")
            )),
        ),
        (
            "Gaussian width",
            lambda name: name in {"caglioti_u", "caglioti_v", "caglioti_w"},
        ),
        (
            "Profile shape",
            lambda name: name in {
                "eta", "lorentzian_x", "lorentzian_y", "axial_asymmetry"
            },
        ),
        (
            "Microstructure and texture",
            lambda name: name.endswith("_delta_biso") or name.endswith("_march_r"),
        ),
    )
    used: set[int] = set()
    for label, predicate in definitions:
        indices = [index for index, name in enumerate(names) if predicate(name)]
        if indices:
            used.update(indices)
            groups.append({"name": label, "indices": indices})
    remaining = [index for index in range(len(names)) if index not in used]
    if remaining:
        groups.append({"name": "Other nonlinear terms", "indices": remaining})
    if len(names) > 1:
        groups.append({"name": "Joint final refinement", "indices": list(range(len(names)))})
    return groups

def staged_profile_refinement_plan(
    *,
    profile_model: str,
    has_instrument_profile: bool,
    refine_cell: bool = True,
    refine_texture: bool = False,
) -> list[dict]:
    if profile_model not in PROFILE_MODELS:
        raise ProfileModelError(f"Unsupported profile model: {profile_model}")
    stages = [
        {
            "stage": 1,
            "name": "Scale and background",
            "parameters": ["scale", "background"],
            "purpose": "Stabilize intensity scale before nonlinear profile terms.",
        },
        {
            "stage": 2,
            "name": "Position terms",
            "parameters": ["zero_shift"] + (["cell"] if refine_cell else []),
            "purpose": "Align reflection positions before broadening refinement.",
        },
        {
            "stage": 3,
            "name": "Gaussian instrument width",
            "parameters": ["U", "V", "W"],
            "purpose": (
                "Verify the calibrated U–V–W profile."
                if has_instrument_profile
                else "Determine the baseline angular width function."
            ),
        },
    ]
    if profile_model == "TCH pseudo-Voigt":
        stages.append({
            "stage": len(stages) + 1,
            "name": "Lorentzian width",
            "parameters": ["X", "Y"],
            "purpose": "Separate Lorentzian broadening after U–V–W is stable.",
        })
    elif profile_model == "Split pseudo-Voigt":
        stages.append({
            "stage": len(stages) + 1,
            "name": "Asymmetry",
            "parameters": ["eta", "axial_asymmetry"],
            "purpose": "Model low-angle asymmetry without shifting peak centroids.",
        })
    else:
        stages.append({
            "stage": len(stages) + 1,
            "name": "Profile mixing",
            "parameters": ["eta"],
            "purpose": "Refine Gaussian/Lorentzian mixing after width convergence.",
        })
    if refine_texture:
        stages.append({
            "stage": len(stages) + 1,
            "name": "Preferred orientation",
            "parameters": ["March-Dollase r"],
            "purpose": "Refine texture only after position and profile stability.",
        })
    return stages
