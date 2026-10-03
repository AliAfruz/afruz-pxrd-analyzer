from __future__ import annotations

"""Instrument-physics helpers shared by calibration and refinements.

The functions here keep the physical parameterization separate from the GUI.
Angles accepted and returned by the public API are in degrees.  Lengths are in
millimetres unless a name explicitly states otherwise.
"""

from copy import deepcopy
import math
from typing import Iterable

import numpy as np


CU_K_ALPHA1_ANGSTROM = 1.5405929
CU_K_ALPHA2_ANGSTROM = 1.5444274
CU_K_ALPHA2_TO_K_ALPHA1 = 0.5

RADIATION_CONFIGURATIONS = (
    "Cu Kα1 monochromated",
    "Cu Kα1/Kα2 doublet",
    "Custom monochromatic",
    "Custom doublet",
)


class InstrumentPhysicsError(ValueError):
    pass


class CalibrationRangeError(InstrumentPhysicsError):
    pass


def radiation_components(
    configuration: str,
    *,
    primary_wavelength_angstrom: float | None = None,
    secondary_wavelength_angstrom: float | None = None,
    secondary_to_primary_ratio: float | None = None,
) -> list[dict]:
    """Return normalized spectral components for a laboratory source.

    The Cu defaults use resolved Kα1 and Kα2 component wavelengths.  A
    monochromated configuration deliberately returns only Kα1; the doublet
    configuration preserves the conventional Kα2/Kα1 integrated-intensity
    ratio as an explicit, user-overridable quantity.
    """

    name = str(configuration or "Cu Kα1 monochromated")
    if name not in RADIATION_CONFIGURATIONS:
        raise InstrumentPhysicsError(
            f"Unsupported radiation configuration: {name}"
        )
    if name.startswith("Cu Kα"):
        primary = CU_K_ALPHA1_ANGSTROM
    else:
        primary = float(primary_wavelength_angstrom or 0.0)
    if primary_wavelength_angstrom is not None:
        primary = float(primary_wavelength_angstrom)
    if not np.isfinite(primary) or primary <= 0:
        raise InstrumentPhysicsError("Primary wavelength must be positive.")

    components = [
        {
            "label": "Kα1" if "doublet" in name or name.startswith("Cu") else "primary",
            "wavelength_angstrom": float(primary),
            "relative_intensity": 1.0,
        }
    ]
    if "doublet" in name:
        secondary = (
            CU_K_ALPHA2_ANGSTROM
            if name.startswith("Cu")
            else float(secondary_wavelength_angstrom or 0.0)
        )
        if secondary_wavelength_angstrom is not None:
            secondary = float(secondary_wavelength_angstrom)
        ratio = (
            CU_K_ALPHA2_TO_K_ALPHA1
            if secondary_to_primary_ratio is None
            else float(secondary_to_primary_ratio)
        )
        if not np.isfinite(secondary) or secondary <= 0:
            raise InstrumentPhysicsError("Secondary wavelength must be positive.")
        if not np.isfinite(ratio) or ratio < 0:
            raise InstrumentPhysicsError(
                "Secondary/primary intensity ratio must be non-negative."
            )
        components.append(
            {
                "label": "Kα2" if name.startswith("Cu") else "secondary",
                "wavelength_angstrom": float(secondary),
                "relative_intensity": float(ratio),
            }
        )
    return components


def spectral_positions_deg(
    primary_two_theta_deg: float,
    components: Iterable[dict],
) -> list[dict]:
    """Map a primary-line Bragg position onto all configured wavelengths."""

    rows = [deepcopy(row) for row in components]
    if not rows:
        raise InstrumentPhysicsError("At least one radiation component is required.")
    primary_wavelength = float(rows[0]["wavelength_angstrom"])
    theta = math.radians(float(primary_two_theta_deg) / 2.0)
    if not 0.0 < theta < math.pi / 2.0:
        raise InstrumentPhysicsError("Bragg position must lie between 0 and 180° 2θ.")
    d_spacing = primary_wavelength / (2.0 * math.sin(theta))
    for row in rows:
        argument = float(row["wavelength_angstrom"]) / (2.0 * d_spacing)
        if argument <= 0.0 or argument >= 1.0:
            raise InstrumentPhysicsError(
                "A configured wavelength has no physical Bragg position for this d-spacing."
            )
        row["two_theta_deg"] = math.degrees(2.0 * math.asin(argument))
    return rows


def spectral_centroid_deg(
    primary_two_theta_deg: float,
    components: Iterable[dict],
) -> float:
    rows = spectral_positions_deg(primary_two_theta_deg, components)
    weights = np.asarray([float(row["relative_intensity"]) for row in rows])
    positions = np.asarray([float(row["two_theta_deg"]) for row in rows])
    return float(np.sum(weights * positions) / np.sum(weights))


def bragg_brentano_transparency_shift_deg(
    two_theta_deg: float | np.ndarray,
    *,
    inverse_linear_absorption_mm: float,
    goniometer_radius_mm: float,
) -> float | np.ndarray:
    """Thick-specimen Bragg–Brentano transparency centroid shift.

    Uses Δ(2θ) = sin(2θ)/(2 μ R), in radians.  The fitted quantity is
    1/μ in mm, which keeps displacement and transparency as separate angular
    basis functions (cos θ and sin 2θ respectively).
    """

    radius = float(goniometer_radius_mm)
    inverse_mu = float(inverse_linear_absorption_mm)
    if radius <= 0:
        raise InstrumentPhysicsError("Goniometer radius must be positive.")
    if inverse_mu < 0 or not np.isfinite(inverse_mu):
        raise InstrumentPhysicsError(
            "Inverse linear absorption coefficient must be finite and non-negative."
        )
    values = np.asarray(two_theta_deg, dtype=float)
    shift_rad = inverse_mu / (2.0 * radius) * np.sin(np.radians(values))
    result = np.degrees(shift_rad)
    return float(result) if np.ndim(two_theta_deg) == 0 else result


def axial_sh_over_l(
    *,
    sample_length_mm: float,
    receiving_slit_length_mm: float,
    goniometer_radius_mm: float,
) -> float:
    """Return the GSAS-style FCJ SH/L geometry ratio.

    GSAS-II defines SH/L as (sample length + receiving-slit length) divided
    by the goniometer diameter.  This parameter records physical geometry and
    replaces an unconstrained generic asymmetry scalar in new profiles.
    """

    sample = float(sample_length_mm)
    slit = float(receiving_slit_length_mm)
    radius = float(goniometer_radius_mm)
    if sample < 0 or slit < 0 or radius <= 0:
        raise InstrumentPhysicsError(
            "Axial lengths must be non-negative and goniometer radius positive."
        )
    return float((sample + slit) / (2.0 * radius))


def axial_divergence_characteristics(
    two_theta_deg: float,
    *,
    sh_over_l: float,
    base_fwhm_deg: float,
) -> dict:
    """Return a stable FCJ/SH-L low-angle asymmetry approximation.

    This is a geometry-constrained reduced model, not a fundamental-parameters
    convolution.  It has the correct limiting behavior: zero at SH/L=0 and a
    tail that weakens strongly as the Bragg angle increases.
    """

    ratio = max(0.0, float(sh_over_l))
    theta = max(math.radians(float(two_theta_deg) / 2.0), math.radians(0.25))
    # SH/L is dimensionless.  cot(theta) supplies the characteristic strong
    # low-angle dependence; clipping keeps the reduced profile numerically safe.
    factor = float(np.clip(2.0 * ratio / max(math.tan(theta), 1e-8), 0.0, 0.85))
    width = max(float(base_fwhm_deg), 1e-12)
    left = width * (1.0 + factor)
    right = width * max(0.35, 1.0 - 0.35 * factor)
    return {
        "sh_over_l": ratio,
        "asymmetry_factor": factor,
        "left_fwhm_deg": float(left),
        "right_fwhm_deg": float(right),
        "classification": "geometry-constrained FCJ/SH-L reduced model",
    }


def validate_calibration_range(
    profile: dict,
    two_theta_deg: float | np.ndarray,
    *,
    allow_extrapolation: bool = False,
) -> dict:
    """Validate angles against the measured calibration interval."""

    values = np.atleast_1d(np.asarray(two_theta_deg, dtype=float))
    lower = profile.get("valid_two_theta_min_deg")
    upper = profile.get("valid_two_theta_max_deg")
    if lower is None or upper is None:
        return {
            "valid": True,
            "extrapolated": False,
            "message": "Profile has no declared angular validity interval.",
        }
    lower = float(lower)
    upper = float(upper)
    outside = values[(values < lower) | (values > upper)]
    if not len(outside):
        return {"valid": True, "extrapolated": False, "message": ""}
    message = (
        f"Requested 2θ values extend outside the calibrated range "
        f"{lower:.6g}–{upper:.6g}°."
    )
    if not allow_extrapolation:
        raise CalibrationRangeError(message)
    return {"valid": False, "extrapolated": True, "message": message}


def validate_profile_compatibility(
    profile: dict,
    *,
    wavelength_angstrom: float,
    relative_tolerance: float = 1e-4,
) -> None:
    """Reject use of a calibration acquired for a different wavelength."""

    calibrated = profile.get("wavelength_angstrom")
    if calibrated is None:
        return
    calibrated = float(calibrated)
    requested = float(wavelength_angstrom)
    if calibrated <= 0 or requested <= 0:
        raise InstrumentPhysicsError("Calibration and analysis wavelengths must be positive.")
    relative = abs(requested - calibrated) / calibrated
    if relative > max(float(relative_tolerance), 0.0):
        raise InstrumentPhysicsError(
            f"Instrument profile wavelength {calibrated:.8g} Å is incompatible "
            f"with analysis wavelength {requested:.8g} Å."
        )


def covariance_block(profile: dict, block: str) -> tuple[list[str], np.ndarray]:
    """Read a named covariance block from a version-3 profile."""

    payload = profile.get(f"{block}_covariance") or {}
    names = [str(value) for value in payload.get("parameter_names", [])]
    matrix = np.asarray(payload.get("matrix", []), dtype=float)
    if matrix.shape != (len(names), len(names)):
        return [], np.empty((0, 0), dtype=float)
    return names, matrix


def numerical_prediction_uncertainty(
    profile: dict,
    block: str,
    evaluator,
) -> float:
    """Propagate one covariance block through a scalar profile prediction."""

    names, covariance = covariance_block(profile, block)
    if not names:
        return 0.0
    base = deepcopy(profile)
    base_prediction = float(evaluator(base))
    gradient = []
    for name in names:
        value = float(base.get(name, 0.0))
        step = max(abs(value) * 1e-5, 1e-8)
        plus = deepcopy(base)
        minus = deepcopy(base)
        plus[name] = value + step
        minus[name] = value - step
        plus_prediction = float(evaluator(plus))
        try:
            minus_prediction = float(evaluator(minus))
        except (ValueError, FloatingPointError):
            gradient.append((plus_prediction - base_prediction) / step)
        else:
            gradient.append((plus_prediction - minus_prediction) / (2.0 * step))
    jacobian = np.asarray(gradient, dtype=float)
    variance = float(jacobian @ covariance @ jacobian)
    return float(math.sqrt(max(0.0, variance)))


def prepare_calibration_prior(
    profile: dict | None,
    parameter_names: Iterable[str],
) -> dict | None:
    """Build a whitened Gaussian prior from calibration covariance blocks.

    Only parameters that are both refined by the downstream engine and present
    in a finite, positive covariance subspace are included.
    """

    if not profile:
        return None
    downstream_names = [str(value) for value in parameter_names]
    selected_names: list[str] = []
    selected_indices: list[int] = []
    selected_means: list[float] = []
    blocks: list[np.ndarray] = []
    for block_name in ("position", "width"):
        names, covariance = covariance_block(profile, block_name)
        keep = [index for index, name in enumerate(names) if name in downstream_names]
        if not keep:
            continue
        submatrix = covariance[np.ix_(keep, keep)]
        if not np.all(np.isfinite(submatrix)):
            continue
        blocks.append(submatrix)
        for index in keep:
            name = names[index]
            selected_names.append(name)
            selected_indices.append(downstream_names.index(name))
            selected_means.append(float(profile.get(name, 0.0)))
    if not blocks:
        return None
    size = sum(block.shape[0] for block in blocks)
    covariance = np.zeros((size, size), dtype=float)
    offset = 0
    for block in blocks:
        count = block.shape[0]
        covariance[offset : offset + count, offset : offset + count] = block
        offset += count
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = max(float(np.max(eigenvalues)), 0.0)
    keep_modes = eigenvalues > max(maximum * 1e-12, 1e-20)
    if not np.any(keep_modes):
        return None
    whitening = (
        (1.0 / np.sqrt(eigenvalues[keep_modes]))[:, None]
        * eigenvectors[:, keep_modes].T
    )
    return {
        "parameter_names": selected_names,
        "parameter_indices": selected_indices,
        "means": selected_means,
        "covariance_matrix": covariance.tolist(),
        "whitening_matrix": whitening.tolist(),
        "effective_rank": int(np.count_nonzero(keep_modes)),
        "profile_fingerprint": profile.get("fingerprint"),
    }


def calibration_prior_residual(
    values: np.ndarray,
    prior: dict | None,
) -> np.ndarray:
    if not prior:
        return np.empty(0, dtype=float)
    selected = np.asarray(values, dtype=float)[
        np.asarray(prior["parameter_indices"], dtype=int)
    ]
    means = np.asarray(prior["means"], dtype=float)
    whitening = np.asarray(prior["whitening_matrix"], dtype=float)
    return whitening @ (selected - means)
