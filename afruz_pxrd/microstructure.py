from __future__ import annotations

"""Advanced crystallite-size, microstrain and texture models.

This module is UI-independent.  It uses peak-profile outputs and optional
instrument profiles to separate instrumental broadening from sample
broadening before estimating isotropic/anisotropic size, strain and texture
metrics.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares, nnls

from .crystal_profiles import profile_characteristics


class MicrostructureError(ValueError):
    pass


MICROSTRUCTURE_VERSION = 1
TEXTURE_MODELS = (
    "None",
    "March-Dollase",
    "Even Legendre P2/P4",
)


@dataclass(frozen=True)
class CorrectedWidth:
    two_theta_deg: float
    theta_rad: float
    observed_fwhm_deg: float
    instrument_gaussian_fwhm_deg: float
    instrument_lorentzian_fwhm_deg: float
    sample_gaussian_fwhm_deg: float | None
    sample_lorentzian_fwhm_deg: float | None
    sample_effective_fwhm_deg: float | None
    note: str


def _as_float(value, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _extract_hkl(row: dict) -> tuple[int, int, int] | None:
    hkl = row.get("hkl") or row.get("hkl_tuple")
    if hkl is None and row.get("h") is not None:
        hkl = (row.get("h"), row.get("k"), row.get("l"))
    if isinstance(hkl, str):
        cleaned = hkl.replace("(", " ").replace(")", " ").replace(",", " ")
        parts = cleaned.split()
    elif isinstance(hkl, Sequence):
        parts = list(hkl)
    else:
        return None
    if len(parts) != 3:
        return None
    try:
        values = tuple(int(round(float(part))) for part in parts)
    except (TypeError, ValueError):
        return None
    if values == (0, 0, 0):
        return None
    return values


def _profile_component_hint(row: dict) -> tuple[float | None, float | None]:
    """Return optional observed Gaussian and Lorentzian FWHM hints."""

    gaussian = _as_float(
        row.get("gaussian_fwhm_deg")
        or row.get("observed_gaussian_fwhm_deg")
        or row.get("beta_gaussian_deg")
    )
    lorentzian = _as_float(
        row.get("lorentzian_fwhm_deg")
        or row.get("observed_lorentzian_fwhm_deg")
        or row.get("beta_lorentzian_deg")
    )
    return gaussian, lorentzian


def instrument_profile_components(
    two_theta_deg: float,
    instrument_profile: dict | None,
    constant_instrument_fwhm_deg: float = 0.0,
) -> tuple[float, float, float, str]:
    """Return total, Gaussian and Lorentzian instrument FWHM values in degrees."""

    constant = max(0.0, float(constant_instrument_fwhm_deg or 0.0))
    if not instrument_profile:
        return constant, constant, 0.0, "constant instrument FWHM"

    model = str(instrument_profile.get("profile_model", "Pseudo-Voigt U-V-W"))
    try:
        chars = profile_characteristics(
            two_theta_deg,
            model=model,
            u=float(instrument_profile.get("caglioti_u", 0.0)),
            v=float(instrument_profile.get("caglioti_v", 0.0)),
            w=float(instrument_profile.get("caglioti_w", 0.0)),
            eta=float(instrument_profile.get("eta", 0.5)),
            x=float(instrument_profile.get("lorentzian_x", 0.0)),
            y=float(instrument_profile.get("lorentzian_y", 0.0)),
            axial_asymmetry=float(instrument_profile.get("axial_asymmetry", 0.0)),
            minimum_width=0.0,
        )
    except Exception as exc:
        return constant, constant, 0.0, f"instrument profile unavailable: {exc}"

    return (
        float(chars.fwhm_deg),
        float(chars.gaussian_fwhm_deg),
        float(chars.lorentzian_fwhm_deg),
        f"profile {model}",
    )


def correct_sample_broadening_components(
    two_theta_deg: float,
    observed_fwhm_deg: float,
    *,
    instrument_profile: dict | None = None,
    constant_instrument_fwhm_deg: float = 0.0,
    observed_gaussian_fwhm_deg: float | None = None,
    observed_lorentzian_fwhm_deg: float | None = None,
) -> CorrectedWidth:
    """Separate instrumental and sample broadening approximately.

    Gaussian parts are deconvolved by quadrature and Lorentzian parts by linear
    subtraction. If observed components are not known, the observed total width
    is used as a conservative effective width with the instrument total removed
    by quadrature.
    """

    two_theta = float(two_theta_deg)
    observed = float(observed_fwhm_deg)
    if not np.isfinite(two_theta) or not (0.0 < two_theta < 180.0):
        return CorrectedWidth(two_theta, math.nan, observed, 0.0, 0.0, None, None, None, "Peak position must be between 0 and 180° 2θ.")
    if not np.isfinite(observed) or observed <= 0:
        return CorrectedWidth(two_theta, math.radians(two_theta / 2.0), observed, 0.0, 0.0, None, None, None, "Observed FWHM is not positive.")

    total_inst, inst_g, inst_l, source = instrument_profile_components(
        two_theta,
        instrument_profile,
        constant_instrument_fwhm_deg,
    )
    note_parts = [source]

    obs_g = _as_float(observed_gaussian_fwhm_deg)
    obs_l = _as_float(observed_lorentzian_fwhm_deg)
    sample_g = None
    sample_l = None

    if obs_g is not None and obs_g > 0:
        difference = obs_g * obs_g - inst_g * inst_g
        if difference > 0:
            sample_g = math.sqrt(difference)
        else:
            note_parts.append("Gaussian component not larger than instrument width.")
    if obs_l is not None and obs_l > 0:
        difference = obs_l - inst_l
        if difference > 0:
            sample_l = difference
        else:
            note_parts.append("Lorentzian component not larger than instrument width.")

    if sample_g is not None or sample_l is not None:
        effective = math.sqrt((sample_g or 0.0) ** 2 + (sample_l or 0.0) ** 2)
    else:
        difference = observed * observed - total_inst * total_inst
        if difference <= 0:
            return CorrectedWidth(
                two_theta,
                math.radians(two_theta / 2.0),
                observed,
                inst_g,
                inst_l,
                None,
                None,
                None,
                "; ".join(note_parts + ["Observed width is not larger than the instrument width."]),
            )
        effective = math.sqrt(difference)
        sample_g = effective
        sample_l = 0.0
        note_parts.append("total-width quadrature approximation")

    return CorrectedWidth(
        two_theta_deg=two_theta,
        theta_rad=math.radians(two_theta / 2.0),
        observed_fwhm_deg=observed,
        instrument_gaussian_fwhm_deg=inst_g,
        instrument_lorentzian_fwhm_deg=inst_l,
        sample_gaussian_fwhm_deg=None if sample_g is None else float(sample_g),
        sample_lorentzian_fwhm_deg=None if sample_l is None else float(sample_l),
        sample_effective_fwhm_deg=float(effective),
        note="; ".join(note_parts),
    )


def reciprocal_direction_cosines(
    hkl: tuple[int, int, int],
    cell: dict | None = None,
) -> np.ndarray:
    """Return a normalized reciprocal-space direction vector.

    For general cells the full metric tensor can be added later; Stage 1 uses
    orthogonal-cell scaling. This already distinguishes plate/needle behavior
    for cubic, tetragonal and orthorhombic systems.
    """

    h, k, l = [float(value) for value in hkl]
    if cell:
        a = max(float(cell.get("a", 1.0) or 1.0), 1e-12)
        b = max(float(cell.get("b", 1.0) or 1.0), 1e-12)
        c = max(float(cell.get("c", 1.0) or 1.0), 1e-12)
        vector = np.asarray([h / a, k / b, l / c], dtype=float)
    else:
        vector = np.asarray([h, k, l], dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 0 or not np.isfinite(norm):
        raise MicrostructureError("Invalid hkl direction.")
    return vector / norm


def _scherrer_size_nm(beta_deg: float, theta_rad: float, wavelength_angstrom: float, shape_factor: float) -> float | None:
    beta_rad = math.radians(float(beta_deg))
    cosine = math.cos(theta_rad)
    if beta_rad <= 0 or cosine <= 0:
        return None
    return float(float(shape_factor) * float(wavelength_angstrom) * 0.1 / (beta_rad * cosine))


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    design = np.column_stack([x, np.ones_like(x)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    residual = y - predicted
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "slope": float(coefficients[0]),
        "intercept": float(coefficients[1]),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        "rmse": float(np.sqrt(np.mean(residual * residual))),
    }


def williamson_hall(peaks: list[dict], *, wavelength_angstrom: float, shape_factor: float = 0.9) -> dict:
    rows = [row for row in peaks if row.get("sample_effective_fwhm_deg")]
    x_values = []
    y_values = []
    for row in rows:
        theta = float(row["theta_rad"])
        beta = math.radians(float(row["sample_effective_fwhm_deg"]))
        x_values.append(4.0 * math.sin(theta))
        y_values.append(beta * math.cos(theta))
    result = {
        "valid": False,
        "point_count": len(x_values),
        "microstrain": None,
        "crystallite_size_nm": None,
        "intercept": None,
        "slope": None,
        "r_squared": None,
        "rmse": None,
        "note": "",
    }
    if len(x_values) < 3:
        result["note"] = "At least three valid peaks are required."
        return result
    fit = _linear_fit(np.asarray(x_values), np.asarray(y_values))
    size = None
    if fit["intercept"] > 0:
        size = shape_factor * wavelength_angstrom * 0.1 / fit["intercept"]
    result.update({
        "valid": bool(size is not None and fit["slope"] >= -1e-8),
        "microstrain": float(fit["slope"]),
        "crystallite_size_nm": None if size is None else float(size),
        "intercept": fit["intercept"],
        "slope": fit["slope"],
        "r_squared": fit["r_squared"],
        "rmse": fit["rmse"],
        "note": "" if size is not None else "Non-positive intercept.",
    })
    return result


def modified_williamson_hall(
    peaks: list[dict],
    *,
    wavelength_angstrom: float,
    shape_factor: float = 0.9,
    contrast_factor: str = "isotropic",
) -> dict:
    """Run a simple modified W-H regression using hkl contrast weights.

    For Stage 17.4 the contrast factor is an explicit diagnostic weight, not a
    dislocation-density determination. It helps reveal whether hkl-dependent
    broadening is present.
    """

    rows = [row for row in peaks if row.get("sample_effective_fwhm_deg") and row.get("hkl")]
    x_values = []
    y_values = []
    for row in rows:
        h, k, l = row["hkl"]
        norm2 = max(h * h + k * k + l * l, 1)
        if contrast_factor == "cubic_edge":
            contrast = (h * h * k * k + h * h * l * l + k * k * l * l) / (norm2 * norm2)
            contrast = max(0.05, float(contrast))
        elif contrast_factor == "c_axis":
            contrast = max(0.05, float(l * l / norm2))
        else:
            contrast = 1.0
        theta = float(row["theta_rad"])
        beta = math.radians(float(row["sample_effective_fwhm_deg"]))
        x_values.append(4.0 * math.sin(theta) * math.sqrt(contrast))
        y_values.append(beta * math.cos(theta))
    result = {
        "valid": False,
        "point_count": len(x_values),
        "contrast_factor": contrast_factor,
        "microstrain_like_slope": None,
        "crystallite_size_nm": None,
        "r_squared": None,
        "rmse": None,
        "note": "",
    }
    if len(x_values) < 3:
        result["note"] = "At least three hkl-assigned peaks are required."
        return result
    fit = _linear_fit(np.asarray(x_values), np.asarray(y_values))
    size = None
    if fit["intercept"] > 0:
        size = shape_factor * wavelength_angstrom * 0.1 / fit["intercept"]
    result.update({
        "valid": bool(size is not None),
        "microstrain_like_slope": fit["slope"],
        "crystallite_size_nm": None if size is None else float(size),
        "intercept": fit["intercept"],
        "r_squared": fit["r_squared"],
        "rmse": fit["rmse"],
    })
    return result


def fit_anisotropic_size_ellipsoid(
    peaks: list[dict],
    *,
    wavelength_angstrom: float,
    shape_factor: float = 0.9,
    cell: dict | None = None,
) -> dict:
    """Fit apparent crystallite size along reciprocal axes.

    The model is:
        1/D_hkl² = u_x²/D_a² + u_y²/D_b² + u_z²/D_c²

    where u is the normalized reciprocal direction vector. This gives a stable
    ellipsoid-like description useful for detecting plate/needle behavior.
    """

    rows = []
    for row in peaks:
        if not row.get("sample_effective_fwhm_deg") or not row.get("hkl"):
            continue
        size = _scherrer_size_nm(row["sample_effective_fwhm_deg"], row["theta_rad"], wavelength_angstrom, shape_factor)
        if size is None or size <= 0:
            continue
        direction = reciprocal_direction_cosines(row["hkl"], cell)
        rows.append((direction, size, row["hkl"]))
    result = {
        "valid": False,
        "point_count": len(rows),
        "size_a_nm": None,
        "size_b_nm": None,
        "size_c_nm": None,
        "anisotropy_ratio": None,
        "rmse_inverse_size": None,
        "note": "",
    }
    if len(rows) < 4:
        result["note"] = "At least four hkl-assigned peaks are required for anisotropic-size fitting."
        return result
    design = np.asarray([direction * direction for direction, _, _ in rows], dtype=float)
    target = np.asarray([1.0 / (size * size) for _, size, _ in rows], dtype=float)
    coefficients, residual_norm = nnls(design, target)
    coefficients = np.maximum(coefficients, 1e-16)
    sizes = 1.0 / np.sqrt(coefficients)
    predicted = design @ coefficients
    rmse = float(np.sqrt(np.mean((target - predicted) ** 2)))
    result.update({
        "valid": True,
        "size_a_nm": float(sizes[0]),
        "size_b_nm": float(sizes[1]),
        "size_c_nm": float(sizes[2]),
        "anisotropy_ratio": float(np.max(sizes) / max(np.min(sizes), 1e-12)),
        "rmse_inverse_size": rmse,
        "note": "Ellipsoid approximation in reciprocal directions.",
    })
    return result


def fit_stephens_like_strain(
    peaks: list[dict],
) -> dict:
    """Fit a nonnegative hkl-dependent broadening fingerprint.

    This is a Stephens-inspired diagnostic using the fourth-order hkl monomials.
    It is intentionally labelled as a fingerprint, not a full symmetry-specific
    Stephens tensor refinement.
    """

    rows = []
    for row in peaks:
        if not row.get("sample_effective_fwhm_deg") or not row.get("hkl"):
            continue
        h, k, l = row["hkl"]
        theta = float(row["theta_rad"])
        beta = math.radians(float(row["sample_effective_fwhm_deg"]))
        y = (beta * math.cos(theta)) ** 2
        scale = max((h * h + k * k + l * l) ** 2, 1.0)
        features = np.asarray([h**4, k**4, l**4, h*h*k*k, h*h*l*l, k*k*l*l], dtype=float) / scale
        rows.append((features, y))
    result = {
        "valid": False,
        "point_count": len(rows),
        "coefficients": {},
        "dominant_term": None,
        "anisotropy_index": None,
        "rmse": None,
        "note": "",
    }
    if len(rows) < 6:
        result["note"] = "At least six hkl-assigned peaks are required for strain-fingerprint fitting."
        return result
    design = np.asarray([features for features, _ in rows], dtype=float)
    target = np.asarray([value for _, value in rows], dtype=float)
    coefficients, _ = nnls(design, target)
    predicted = design @ coefficients
    labels = ("h4", "k4", "l4", "h2k2", "h2l2", "k2l2")
    coeff_map = {label: float(value) for label, value in zip(labels, coefficients)}
    total = float(np.sum(coefficients))
    dominant = max(coeff_map, key=coeff_map.get) if total > 0 else None
    anisotropy = None
    if total > 0:
        anisotropy = float(np.max(coefficients) / max(np.mean(coefficients), 1e-30))
    result.update({
        "valid": True,
        "coefficients": coeff_map,
        "dominant_term": dominant,
        "anisotropy_index": anisotropy,
        "rmse": float(np.sqrt(np.mean((target - predicted) ** 2))),
        "note": "Diagnostic Stephens-like fingerprint; not a full tensor refinement.",
    })
    return result


def march_dollase_factor(
    hkl: tuple[int, int, int],
    preferred_axis: tuple[int, int, int] = (0, 0, 1),
    r: float = 1.0,
    cell: dict | None = None,
) -> float:
    direction = reciprocal_direction_cosines(hkl, cell)
    axis = reciprocal_direction_cosines(preferred_axis, cell)
    cos_alpha = float(abs(np.dot(direction, axis)))
    sin2 = max(0.0, 1.0 - cos_alpha * cos_alpha)
    r = max(float(r), 1e-6)
    return float((r * r * cos_alpha * cos_alpha + sin2 / r) ** (-1.5))


def fit_march_dollase_texture(
    reflections: list[dict],
    *,
    preferred_axis: tuple[int, int, int] = (0, 0, 1),
    cell: dict | None = None,
) -> dict:
    """Fit March-Dollase r to observed/calculated intensity ratios."""

    rows = []
    for row in reflections:
        hkl = row.get("hkl") or _extract_hkl(row)
        observed = _as_float(row.get("observed_intensity") or row.get("intensity") or row.get("area"))
        calculated = _as_float(row.get("calculated_intensity") or row.get("reference_intensity") or row.get("untextured_intensity"))
        if hkl and observed is not None and calculated is not None and observed > 0 and calculated > 0:
            rows.append((hkl, observed / calculated))
    result = {
        "valid": False,
        "point_count": len(rows),
        "preferred_axis": tuple(int(v) for v in preferred_axis),
        "r": None,
        "texture_strength": None,
        "rmse_log_ratio": None,
        "note": "",
    }
    if len(rows) < 3:
        result["note"] = "At least three hkl intensity ratios are required for March-Dollase fitting."
        return result

    ratios = np.asarray([value for _, value in rows], dtype=float)
    log_ratios = np.log(np.maximum(ratios / np.median(ratios), 1e-12))

    def residual(p):
        r = float(p[0])
        factors = np.asarray([march_dollase_factor(hkl, preferred_axis, r, cell) for hkl, _ in rows], dtype=float)
        predicted = np.log(np.maximum(factors / np.median(factors), 1e-12))
        return log_ratios - predicted

    fit = least_squares(residual, x0=np.asarray([1.0]), bounds=(0.2, 5.0), loss="soft_l1")
    r = float(fit.x[0])
    result.update({
        "valid": bool(fit.success),
        "r": r,
        "texture_strength": float(abs(math.log(r))),
        "rmse_log_ratio": float(np.sqrt(np.mean(residual([r]) ** 2))),
        "note": "r=1 indicates no March-Dollase texture; deviations require scientific review.",
    })
    return result


def legendre_texture_factors(
    hkls: Iterable[tuple[int, int, int]],
    *,
    axis: tuple[int, int, int] = (0, 0, 1),
    c2: float = 0.0,
    c4: float = 0.0,
    cell: dict | None = None,
) -> list[float]:
    axis_vector = reciprocal_direction_cosines(axis, cell)
    factors = []
    for hkl in hkls:
        direction = reciprocal_direction_cosines(hkl, cell)
        mu = float(abs(np.dot(direction, axis_vector)))
        p2 = 0.5 * (3.0 * mu * mu - 1.0)
        p4 = (35.0 * mu**4 - 30.0 * mu * mu + 3.0) / 8.0
        factors.append(float(max(0.0, 1.0 + float(c2) * p2 + float(c4) * p4)))
    return factors


def analyze_advanced_microstructure(
    peak_rows: list[dict],
    *,
    wavelength_angstrom: float,
    shape_factor: float = 0.9,
    instrument_profile: dict | None = None,
    constant_instrument_fwhm_deg: float = 0.0,
    cell: dict | None = None,
    preferred_axis: tuple[int, int, int] = (0, 0, 1),
) -> dict:
    """Run the Phase 17.4 microstructure analysis pipeline."""

    if wavelength_angstrom <= 0:
        raise MicrostructureError("Wavelength must be positive.")
    corrected = []
    for index, row in enumerate(peak_rows, start=1):
        two_theta = _as_float(row.get("two_theta_deg") or row.get("center"))
        observed_width = _as_float(row.get("observed_fwhm_deg") or row.get("fwhm") or row.get("fwhm_deg"))
        if two_theta is None or observed_width is None:
            continue
        obs_g, obs_l = _profile_component_hint(row)
        width = correct_sample_broadening_components(
            two_theta,
            observed_width,
            instrument_profile=instrument_profile,
            constant_instrument_fwhm_deg=constant_instrument_fwhm_deg,
            observed_gaussian_fwhm_deg=obs_g,
            observed_lorentzian_fwhm_deg=obs_l,
        )
        hkl = _extract_hkl(row)
        result_row = {
            "peak_number": row.get("peak_number", index),
            "two_theta_deg": width.two_theta_deg,
            "theta_rad": width.theta_rad,
            "hkl": hkl,
            "observed_fwhm_deg": width.observed_fwhm_deg,
            "instrument_gaussian_fwhm_deg": width.instrument_gaussian_fwhm_deg,
            "instrument_lorentzian_fwhm_deg": width.instrument_lorentzian_fwhm_deg,
            "sample_gaussian_fwhm_deg": width.sample_gaussian_fwhm_deg,
            "sample_lorentzian_fwhm_deg": width.sample_lorentzian_fwhm_deg,
            "sample_effective_fwhm_deg": width.sample_effective_fwhm_deg,
            "scherrer_size_nm": None,
            "valid": width.sample_effective_fwhm_deg is not None,
            "note": width.note,
        }
        if width.sample_effective_fwhm_deg is not None:
            result_row["scherrer_size_nm"] = _scherrer_size_nm(
                width.sample_effective_fwhm_deg,
                width.theta_rad,
                wavelength_angstrom,
                shape_factor,
            )
        corrected.append(result_row)

    valid_sizes = np.asarray([row["scherrer_size_nm"] for row in corrected if row.get("scherrer_size_nm")], dtype=float)
    isotropic = {
        "valid_peak_count": int(valid_sizes.size),
        "mean_size_nm": float(np.mean(valid_sizes)) if valid_sizes.size else None,
        "median_size_nm": float(np.median(valid_sizes)) if valid_sizes.size else None,
        "min_size_nm": float(np.min(valid_sizes)) if valid_sizes.size else None,
        "max_size_nm": float(np.max(valid_sizes)) if valid_sizes.size else None,
    }

    md_reflections = []
    for original, processed in zip(peak_rows, corrected):
        if processed.get("hkl"):
            merged = dict(original)
            merged["hkl"] = processed["hkl"]
            md_reflections.append(merged)

    result = {
        "version": MICROSTRUCTURE_VERSION,
        "wavelength_angstrom": float(wavelength_angstrom),
        "shape_factor": float(shape_factor),
        "instrument_profile_fingerprint": _fingerprint_or_none(instrument_profile),
        "constant_instrument_fwhm_deg": float(constant_instrument_fwhm_deg),
        "corrected_peaks": corrected,
        "isotropic_scherrer": isotropic,
        "williamson_hall": williamson_hall(corrected, wavelength_angstrom=wavelength_angstrom, shape_factor=shape_factor),
        "modified_williamson_hall": modified_williamson_hall(corrected, wavelength_angstrom=wavelength_angstrom, shape_factor=shape_factor),
        "anisotropic_size_ellipsoid": fit_anisotropic_size_ellipsoid(corrected, wavelength_angstrom=wavelength_angstrom, shape_factor=shape_factor, cell=cell),
        "stephens_like_strain": fit_stephens_like_strain(corrected),
        "march_dollase_texture": fit_march_dollase_texture(md_reflections, preferred_axis=preferred_axis, cell=cell),
    }
    result["classification"] = classify_microstructure_result(result)
    return result


def _fingerprint_or_none(profile: dict | None) -> str | None:
    if not profile:
        return None
    payload = dict(profile)
    payload.pop("fingerprint", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def classify_microstructure_result(result: dict) -> dict:
    warnings = []
    if not result.get("instrument_profile_fingerprint") and result.get("constant_instrument_fwhm_deg", 0.0) <= 0:
        warnings.append("No instrument-broadening correction was applied.")
    if result["isotropic_scherrer"].get("valid_peak_count", 0) < 3:
        warnings.append("Too few valid peaks for robust microstructure interpretation.")
    wh = result.get("williamson_hall", {})
    if wh.get("valid") and wh.get("microstrain") is not None and wh.get("microstrain") < -1e-6:
        warnings.append("Williamson-Hall slope is negative; review instrument correction and peak widths.")
    aniso = result.get("anisotropic_size_ellipsoid", {})
    if aniso.get("valid") and aniso.get("anisotropy_ratio", 1.0) > 3.0:
        warnings.append("Large size anisotropy detected; verify hkl assignments and profile model.")
    texture = result.get("march_dollase_texture", {})
    if texture.get("valid") and texture.get("texture_strength", 0.0) > 0.3:
        warnings.append("Preferred orientation may affect intensity-based conclusions.")
    status = "Complete"
    if warnings:
        status = "Complete with warnings"
    if result["isotropic_scherrer"].get("valid_peak_count", 0) == 0:
        status = "Invalid"
    return {"status": status, "warnings": warnings}
