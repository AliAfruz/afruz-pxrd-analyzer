from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .instrument_calibration import (
    CalibrationRangeError,
    instrument_fwhm_deg as calibrated_instrument_fwhm_deg,
    instrument_fwhm_uncertainty_deg,
)
from .microstructure import analyze_advanced_microstructure
from .instrument_physics import validate_profile_compatibility


CORRECTION_MODES = (
    "None",
    "Gaussian quadrature",
    "Lorentzian linear",
)


def correct_instrument_broadening(
    observed_fwhm_deg: float,
    instrument_fwhm_deg: float,
    mode: str,
    observed_error_deg: float | None = None,
    instrument_error_deg: float | None = None,
) -> tuple[float | None, float | None, str]:
    """
    Correct an observed FWHM for instrumental broadening.

    Gaussian quadrature:
        beta_sample = sqrt(beta_observed^2 - beta_instrument^2)

    Lorentzian linear:
        beta_sample = beta_observed - beta_instrument

    The selected approximation must be consistent with the profile model and
    the instrument-standard analysis used by the researcher.
    """
    observed = float(observed_fwhm_deg)
    instrument = max(0.0, float(instrument_fwhm_deg))
    error = None
    if observed_error_deg is not None:
        try:
            candidate = float(observed_error_deg)
            if np.isfinite(candidate) and candidate >= 0:
                error = candidate
        except (TypeError, ValueError):
            pass
    instrument_error = None
    if instrument_error_deg is not None:
        try:
            candidate = float(instrument_error_deg)
            if np.isfinite(candidate) and candidate >= 0:
                instrument_error = candidate
        except (TypeError, ValueError):
            pass

    if not np.isfinite(observed) or observed <= 0:
        return None, None, "Observed FWHM is not positive."

    if mode == "None" or instrument == 0:
        return observed, error, ""

    if mode == "Gaussian quadrature":
        difference = observed * observed - instrument * instrument
        if difference <= 0:
            return None, None, (
                "Observed FWHM is not larger than the instrumental FWHM "
                "under Gaussian quadrature correction."
            )
        corrected = math.sqrt(difference)
        variance = 0.0
        have_uncertainty = False
        if error is not None:
            variance += (observed / corrected * error) ** 2
            have_uncertainty = True
        if instrument_error is not None:
            variance += (instrument / corrected * instrument_error) ** 2
            have_uncertainty = True
        corrected_error = math.sqrt(variance) if have_uncertainty else None
        return corrected, corrected_error, ""

    if mode == "Lorentzian linear":
        corrected = observed - instrument
        if corrected <= 0:
            return None, None, (
                "Observed FWHM is not larger than the instrumental FWHM "
                "under Lorentzian linear correction."
            )
        corrected_error = (
            math.sqrt(
                (0.0 if error is None else error**2)
                + (0.0 if instrument_error is None else instrument_error**2)
            )
            if error is not None or instrument_error is not None
            else None
        )
        return corrected, corrected_error, ""

    raise ValueError(f"Unsupported instrument correction mode: {mode}")


def _weighted_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    y_error: np.ndarray | None,
) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    weighted = (
        y_error is not None
        and len(y_error) == len(y)
        and np.all(np.isfinite(y_error))
        and np.all(y_error > 0)
    )

    design = np.column_stack([x, np.ones_like(x)])
    if weighted:
        weights = 1.0 / np.square(np.asarray(y_error, dtype=float))
        xtwx = design.T @ (weights[:, None] * design)
        xtwy = design.T @ (weights * y)
        coefficients = np.linalg.solve(xtwx, xtwy)
        residuals = y - design @ coefficients
        degrees_of_freedom = max(1, len(y) - 2)
        reduced_chi_square = float(
            np.sum(weights * residuals * residuals) / degrees_of_freedom
        )
        covariance = np.linalg.inv(xtwx) * reduced_chi_square
    else:
        coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ coefficients
        degrees_of_freedom = max(1, len(y) - 2)
        residual_variance = float(
            np.sum(residuals * residuals) / degrees_of_freedom
        )
        covariance = np.linalg.inv(design.T @ design) * residual_variance
        reduced_chi_square = residual_variance

    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    slope_error = float(math.sqrt(max(0.0, covariance[0, 0])))
    intercept_error = float(math.sqrt(max(0.0, covariance[1, 1])))

    ss_res = float(np.sum(residuals * residuals))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "slope": slope,
        "intercept": intercept,
        "slope_error": slope_error,
        "intercept_error": intercept_error,
        "r_squared": float(r_squared),
        "rmse": float(np.sqrt(np.mean(residuals * residuals))),
        "reduced_chi_square": reduced_chi_square,
        "weighted": bool(weighted),
    }


def _flatten_fit_components(fit_groups: Iterable[dict]) -> list[dict]:
    flattened = []
    for group in fit_groups:
        for component in group.get("components", []):
            row = dict(component)
            row["group_id"] = group.get("group_id")
            row["group_r_squared"] = group.get("r_squared")
            if "hkl" not in row and group.get("hkl") is not None:
                row["hkl"] = group.get("hkl")
            flattened.append(row)
    return flattened


def analyze_size_strain(
    fit_groups: list[dict],
    wavelength_angstrom: float,
    shape_factor: float = 0.9,
    instrument_fwhm_deg: float = 0.0,
    correction_mode: str = "None",
    instrument_profile: dict | None = None,
    allow_profile_extrapolation: bool = False,
) -> dict:
    """
    Calculate Scherrer crystallite sizes and a Williamson-Hall regression.

    Williamson-Hall coordinates use:
        y = beta * cos(theta)
        x = 4 * sin(theta)

    Therefore:
        y = K*lambda/D + epsilon*x

    FWHM values must be in 2theta degrees at input and are converted to radians.
    Wavelength is entered in angstrom and reported crystallite sizes are in nm.
    """
    wavelength_angstrom = float(wavelength_angstrom)
    shape_factor = float(shape_factor)
    instrument_fwhm_deg = max(0.0, float(instrument_fwhm_deg))

    if not np.isfinite(wavelength_angstrom) or wavelength_angstrom <= 0:
        raise ValueError("Wavelength must be positive.")
    if not np.isfinite(shape_factor) or shape_factor <= 0:
        raise ValueError("Scherrer shape factor must be positive.")
    if instrument_profile:
        validate_profile_compatibility(
            instrument_profile,
            wavelength_angstrom=wavelength_angstrom,
        )
    if correction_mode not in CORRECTION_MODES:
        raise ValueError(f"Unsupported correction mode: {correction_mode}")

    wavelength_nm = wavelength_angstrom * 0.1
    components = _flatten_fit_components(fit_groups)
    peak_results = []
    valid_for_regression = []

    for peak_number, component in enumerate(
        sorted(components, key=lambda row: float(row["center"])),
        start=1,
    ):
        two_theta_deg = float(component["center"])
        observed_fwhm_deg = float(component["fwhm"])
        observed_error_deg = component.get("fwhm_error")

        local_instrument_fwhm = instrument_fwhm_deg
        local_instrument_error = None
        profile_note = ""
        if instrument_profile:
            try:
                local_instrument_fwhm = float(
                    calibrated_instrument_fwhm_deg(
                        two_theta_deg,
                        instrument_profile,
                        allow_extrapolation=allow_profile_extrapolation,
                    )
                )
                local_instrument_error = float(
                    instrument_fwhm_uncertainty_deg(
                        two_theta_deg,
                        instrument_profile,
                        allow_extrapolation=allow_profile_extrapolation,
                    )
                )
            except CalibrationRangeError as exc:
                local_instrument_fwhm = None
                profile_note = str(exc)
            if allow_profile_extrapolation:
                valid_min = instrument_profile.get("valid_two_theta_min_deg")
                valid_max = instrument_profile.get("valid_two_theta_max_deg")
                if (
                    valid_min is not None
                    and valid_max is not None
                    and not float(valid_min) <= two_theta_deg <= float(valid_max)
                ):
                    profile_note = (
                        "Instrument width is extrapolated outside the "
                        "calibrated 2θ range."
                    )
        if local_instrument_fwhm is None:
            corrected_deg, corrected_error_deg, note = None, None, ""
        else:
            corrected_deg, corrected_error_deg, note = correct_instrument_broadening(
                observed_fwhm_deg,
                local_instrument_fwhm,
                correction_mode,
                observed_error_deg,
                local_instrument_error,
            )

        row = {
            "peak_number": peak_number,
            "group_id": component.get("group_id"),
            "model": component.get("model"),
            "two_theta_deg": two_theta_deg,
            "theta_deg": two_theta_deg / 2.0,
            "observed_fwhm_deg": observed_fwhm_deg,
            "instrument_fwhm_deg": (
                None
                if local_instrument_fwhm is None
                else float(local_instrument_fwhm)
            ),
            "instrument_fwhm_error_deg": local_instrument_error,
            "instrument_uncertainty_propagated": bool(
                local_instrument_error is not None
                and correction_mode != "None"
                and local_instrument_fwhm not in (None, 0.0)
            ),
            "observed_fwhm_error_deg": (
                None
                if observed_error_deg is None
                else float(observed_error_deg)
            ),
            "corrected_fwhm_deg": corrected_deg,
            "corrected_fwhm_error_deg": corrected_error_deg,
            "scherrer_size_nm": None,
            "scherrer_size_error_nm": None,
            "hkl": component.get("hkl"),
            "sample_gaussian_fwhm_deg": component.get("sample_gaussian_fwhm_deg"),
            "sample_lorentzian_fwhm_deg": component.get("sample_lorentzian_fwhm_deg"),
            "wh_x": None,
            "wh_y": None,
            "wh_y_error": None,
            "valid": False,
            "note": "; ".join(
                value for value in (note, profile_note) if value
            ),
        }

        if corrected_deg is None:
            peak_results.append(row)
            continue
        if not (0.0 < two_theta_deg < 180.0):
            row["note"] = "Peak position must lie between 0 and 180 degrees 2theta."
            peak_results.append(row)
            continue

        theta_rad = math.radians(two_theta_deg / 2.0)
        beta_rad = math.radians(corrected_deg)
        beta_error_rad = (
            None
            if corrected_error_deg is None
            else math.radians(corrected_error_deg)
        )
        cosine = math.cos(theta_rad)
        sine = math.sin(theta_rad)
        if beta_rad <= 0 or cosine <= 0:
            row["note"] = "Corrected width or cos(theta) is not positive."
            peak_results.append(row)
            continue

        scherrer_size_nm = (
            shape_factor * wavelength_nm / (beta_rad * cosine)
        )
        scherrer_error_nm = (
            None
            if beta_error_rad is None
            else abs(scherrer_size_nm * beta_error_rad / beta_rad)
        )
        wh_x = 4.0 * sine
        wh_y = beta_rad * cosine
        wh_y_error = (
            None if beta_error_rad is None else beta_error_rad * cosine
        )

        row.update(
            {
                "scherrer_size_nm": float(scherrer_size_nm),
                "scherrer_size_error_nm": (
                    None
                    if scherrer_error_nm is None
                    else float(scherrer_error_nm)
                ),
                "wh_x": float(wh_x),
                "wh_y": float(wh_y),
                "wh_y_error": (
                    None if wh_y_error is None else float(wh_y_error)
                ),
                "valid": True,
                "note": profile_note,
            }
        )
        peak_results.append(row)
        valid_for_regression.append(row)

    sizes = np.asarray(
        [
            row["scherrer_size_nm"]
            for row in valid_for_regression
            if row["scherrer_size_nm"] is not None
        ],
        dtype=float,
    )
    scherrer_summary = {
        "valid_peak_count": int(len(sizes)),
        "mean_nm": float(np.mean(sizes)) if len(sizes) else None,
        "median_nm": float(np.median(sizes)) if len(sizes) else None,
        "standard_deviation_nm": (
            float(np.std(sizes, ddof=1)) if len(sizes) > 1 else None
        ),
        "minimum_nm": float(np.min(sizes)) if len(sizes) else None,
        "maximum_nm": float(np.max(sizes)) if len(sizes) else None,
        "weighted_mean_nm": None,
        "weighted_mean_error_nm": None,
    }

    size_errors = np.asarray(
        [
            row["scherrer_size_error_nm"]
            for row in valid_for_regression
        ],
        dtype=object,
    )
    if len(sizes) and all(
        value is not None and np.isfinite(float(value)) and float(value) > 0
        for value in size_errors
    ):
        numeric_errors = np.asarray(size_errors, dtype=float)
        weights = 1.0 / np.square(numeric_errors)
        scherrer_summary["weighted_mean_nm"] = float(
            np.sum(weights * sizes) / np.sum(weights)
        )
        scherrer_summary["weighted_mean_error_nm"] = float(
            math.sqrt(1.0 / np.sum(weights))
        )

    wh_result = {
        "valid": False,
        "point_count": len(valid_for_regression),
        "x": [row["wh_x"] for row in valid_for_regression],
        "y": [row["wh_y"] for row in valid_for_regression],
        "y_error": [row["wh_y_error"] for row in valid_for_regression],
        "slope": None,
        "slope_error": None,
        "intercept": None,
        "intercept_error": None,
        "microstrain": None,
        "microstrain_error": None,
        "crystallite_size_nm": None,
        "crystallite_size_error_nm": None,
        "r_squared": None,
        "rmse": None,
        "reduced_chi_square": None,
        "weighted": False,
        "note": "",
    }

    if len(valid_for_regression) < 3:
        wh_result["note"] = (
            "At least three valid fitted peaks are required for "
            "Williamson-Hall regression."
        )
    else:
        x = np.asarray(wh_result["x"], dtype=float)
        y = np.asarray(wh_result["y"], dtype=float)
        y_error_values = wh_result["y_error"]
        y_error = None
        if all(
            value is not None
            and np.isfinite(float(value))
            and float(value) > 0
            for value in y_error_values
        ):
            y_error = np.asarray(y_error_values, dtype=float)

        regression = _weighted_linear_regression(x, y, y_error)
        intercept = regression["intercept"]
        size_nm = (
            shape_factor * wavelength_nm / intercept
            if intercept > 0
            else None
        )
        size_error_nm = (
            None
            if size_nm is None
            else abs(
                size_nm
                * regression["intercept_error"]
                / intercept
            )
        )
        wh_result.update(
            {
                "valid": intercept > 0,
                "slope": regression["slope"],
                "slope_error": regression["slope_error"],
                "intercept": intercept,
                "intercept_error": regression["intercept_error"],
                "microstrain": regression["slope"],
                "microstrain_error": regression["slope_error"],
                "crystallite_size_nm": (
                    None if size_nm is None else float(size_nm)
                ),
                "crystallite_size_error_nm": (
                    None
                    if size_error_nm is None
                    else float(size_error_nm)
                ),
                "r_squared": regression["r_squared"],
                "rmse": regression["rmse"],
                "reduced_chi_square": regression["reduced_chi_square"],
                "weighted": regression["weighted"],
                "note": (
                    ""
                    if intercept > 0
                    else "The fitted intercept is not positive; "
                    "a physical crystallite size cannot be calculated."
                ),
            }
        )

    advanced_peak_rows = []
    for peak in peak_results:
        row = dict(peak)
        row["fwhm"] = peak.get("observed_fwhm_deg")
        row["center"] = peak.get("two_theta_deg")
        if row.get("sample_gaussian_fwhm_deg") is None:
            row.pop("sample_gaussian_fwhm_deg", None)
        if row.get("sample_lorentzian_fwhm_deg") is None:
            row.pop("sample_lorentzian_fwhm_deg", None)
        advanced_peak_rows.append(row)

    advanced_microstructure = analyze_advanced_microstructure(
        advanced_peak_rows,
        wavelength_angstrom=wavelength_angstrom,
        shape_factor=shape_factor,
        instrument_profile=instrument_profile,
        constant_instrument_fwhm_deg=instrument_fwhm_deg,
    )

    return {
        "wavelength_angstrom": wavelength_angstrom,
        "shape_factor": shape_factor,
        "instrument_fwhm_deg": instrument_fwhm_deg,
        "instrument_profile": instrument_profile,
        "instrument_profile_fingerprint": (
            None if not instrument_profile
            else instrument_profile.get("fingerprint")
        ),
        "instrument_covariance_propagated": bool(
            any(
                row.get("instrument_uncertainty_propagated", False)
                for row in peak_results
            )
        ),
        "allow_profile_extrapolation": bool(allow_profile_extrapolation),
        "correction_mode": correction_mode,
        "peak_results": peak_results,
        "scherrer_summary": scherrer_summary,
        "williamson_hall": wh_result,
        "advanced_microstructure": advanced_microstructure,
    }
