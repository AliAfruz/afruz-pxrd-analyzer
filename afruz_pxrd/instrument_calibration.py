from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from .crystal_profiles import (
    fit_instrument_width_models,
    profile_characteristics,
)
from .instrument_physics import (
    CalibrationRangeError,
    RADIATION_CONFIGURATIONS,
    axial_sh_over_l,
    bragg_brentano_transparency_shift_deg,
    numerical_prediction_uncertainty,
    radiation_components,
    validate_calibration_range,
)


PROFILE_VERSION = 3
GEOMETRIES = (
    "Bragg-Brentano symmetric",
    "Position-only calibration",
)


class CalibrationError(ValueError):
    pass


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def profile_fingerprint(profile: dict) -> str:
    payload = deepcopy(profile)
    payload.pop("fingerprint", None)
    payload.pop("fingerprint_valid", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def instrument_fwhm_deg(
    two_theta_deg: float | np.ndarray,
    profile: dict,
    *,
    allow_extrapolation: bool = True,
) -> float | np.ndarray:
    validate_calibration_range(
        profile,
        two_theta_deg,
        allow_extrapolation=allow_extrapolation,
    )
    values = np.asarray(two_theta_deg, dtype=float)
    model = str(profile.get("profile_model", "Pseudo-Voigt U-V-W"))
    widths = np.asarray([
        profile_characteristics(
            float(value),
            model=model,
            u=float(profile.get("caglioti_u", 0.0)),
            v=float(profile.get("caglioti_v", 0.0)),
            w=float(profile.get("caglioti_w", 0.0)),
            eta=float(profile.get("eta", 0.5)),
            x=float(profile.get("lorentzian_x", 0.0)),
            y=float(profile.get("lorentzian_y", 0.0)),
            axial_asymmetry=float(profile.get("axial_asymmetry", 0.0)),
            axial_sh_over_l=float(profile.get("axial_sh_over_l", 0.0)),
            minimum_width=0.0,
        ).fwhm_deg
        for value in np.atleast_1d(values)
    ], dtype=float)
    if np.ndim(two_theta_deg) == 0:
        return float(widths[0])
    return widths.reshape(values.shape)


def position_correction_deg(
    two_theta_deg: float | np.ndarray,
    profile: dict,
) -> float | np.ndarray:
    values = np.asarray(two_theta_deg, dtype=float)
    correction = np.full_like(
        values,
        float(profile.get("zero_shift_deg", 0.0)),
        dtype=float,
    )
    if profile.get("geometry") == "Bragg-Brentano symmetric":
        radius = float(profile.get("goniometer_radius_mm", 0.0))
        displacement = float(profile.get("specimen_displacement_mm", 0.0))
        if radius > 0:
            theta = np.radians(values / 2.0)
            correction += (
                -2.0
                * displacement
                / radius
                * np.cos(theta)
                * 180.0
                / math.pi
            )
            inverse_mu = float(
                profile.get("inverse_linear_absorption_mm", 0.0)
            )
            if inverse_mu != 0:
                correction += bragg_brentano_transparency_shift_deg(
                    values,
                    inverse_linear_absorption_mm=inverse_mu,
                    goniometer_radius_mm=radius,
                )
    return float(correction) if np.ndim(two_theta_deg) == 0 else correction


def instrument_fwhm_uncertainty_deg(
    two_theta_deg: float,
    profile: dict,
    *,
    allow_extrapolation: bool = False,
) -> float:
    validate_calibration_range(
        profile,
        two_theta_deg,
        allow_extrapolation=allow_extrapolation,
    )
    return numerical_prediction_uncertainty(
        profile,
        "width",
        lambda candidate: instrument_fwhm_deg(
            two_theta_deg,
            candidate,
            allow_extrapolation=True,
        ),
    )


def position_correction_uncertainty_deg(
    two_theta_deg: float,
    profile: dict,
    *,
    allow_extrapolation: bool = False,
) -> float:
    validate_calibration_range(
        profile,
        two_theta_deg,
        allow_extrapolation=allow_extrapolation,
    )
    return numerical_prediction_uncertainty(
        profile,
        "position",
        lambda candidate: position_correction_deg(two_theta_deg, candidate),
    )


def corrected_reference_position(
    reference_two_theta_deg: float | np.ndarray,
    profile: dict,
) -> float | np.ndarray:
    values = np.asarray(reference_two_theta_deg, dtype=float)
    corrected = values + position_correction_deg(values, profile)
    return float(corrected) if np.ndim(reference_two_theta_deg) == 0 else corrected


def _component_rows(fit_groups: Iterable[dict]) -> list[dict]:
    rows = []
    for group in fit_groups:
        for component in group.get("components", []):
            center = component.get("center")
            fwhm = component.get("fwhm")
            if center is None or fwhm is None:
                continue
            try:
                center = float(center)
                fwhm = float(fwhm)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(center) or not np.isfinite(fwhm) or fwhm <= 0:
                continue
            rows.append(
                {
                    "observed_two_theta_deg": center,
                    "observed_fwhm_deg": fwhm,
                    "center_error_deg": component.get("center_error"),
                    "fwhm_error_deg": component.get("fwhm_error"),
                    "left_fwhm_deg": component.get("left_fwhm_deg"),
                    "right_fwhm_deg": component.get("right_fwhm_deg"),
                    "amplitude": component.get("amplitude"),
                    "model": component.get("model", group.get("model")),
                    "group_id": group.get("group_id"),
                }
            )
    return sorted(rows, key=lambda row: row["observed_two_theta_deg"])


def match_standard_reflections(
    fit_groups: list[dict],
    reference_peaks: list[dict],
    tolerance_deg: float = 0.25,
    intensity_cutoff_percent: float = 1.0,
) -> list[dict]:
    observed = _component_rows(fit_groups)
    if len(observed) < 3:
        raise CalibrationError(
            "At least three fitted standard reflections are required."
        )
    if not reference_peaks:
        raise CalibrationError("A reference standard pattern is required.")

    intensities = np.asarray(
        [float(row.get("intensity", 0.0)) for row in reference_peaks],
        dtype=float,
    )
    maximum = float(np.max(intensities)) if len(intensities) else 0.0
    cutoff = maximum * max(0.0, float(intensity_cutoff_percent)) / 100.0
    references = [
        {
            "reference_two_theta_deg": float(row["two_theta"]),
            "reference_intensity": float(row.get("intensity", 0.0)),
            "hkl_label": str(row.get("hkl_label", "")),
        }
        for row in reference_peaks
        if float(row.get("intensity", 0.0)) >= cutoff
    ]
    candidates = []
    tolerance = max(0.001, float(tolerance_deg))
    for oi, obs in enumerate(observed):
        for ri, ref in enumerate(references):
            delta = (
                obs["observed_two_theta_deg"]
                - ref["reference_two_theta_deg"]
            )
            if abs(delta) <= tolerance:
                candidates.append((abs(delta), oi, ri, delta))
    candidates.sort()

    used_observed = set()
    used_reference = set()
    matches = []
    for _, oi, ri, delta in candidates:
        if oi in used_observed or ri in used_reference:
            continue
        used_observed.add(oi)
        used_reference.add(ri)
        matches.append(
            {
                **observed[oi],
                **references[ri],
                "raw_delta_deg": float(delta),
                "included": True,
            }
        )
    return sorted(matches, key=lambda row: row["reference_two_theta_deg"])


def _position_fit(
    rows: list[dict],
    geometry: str,
    goniometer_radius_mm: float,
    fit_zero_shift: bool,
    fit_displacement: bool,
    fit_transparency: bool,
) -> dict:
    reference = np.asarray(
        [row["reference_two_theta_deg"] for row in rows],
        dtype=float,
    )
    observed = np.asarray(
        [row["observed_two_theta_deg"] for row in rows],
        dtype=float,
    )
    errors = np.asarray(
        [
            float(row.get("center_error_deg") or 0.01)
            for row in rows
        ],
        dtype=float,
    )
    errors = np.maximum(errors, 1e-5)

    parameter_names = []
    initial = []
    lower = []
    upper = []
    if fit_zero_shift:
        parameter_names.append("zero_shift_deg")
        initial.append(float(np.median(observed - reference)))
        lower.append(-2.0)
        upper.append(2.0)
    if fit_displacement:
        if geometry != "Bragg-Brentano symmetric":
            raise CalibrationError(
                "Specimen displacement is supported only for symmetric "
                "Bragg–Brentano geometry."
            )
        if goniometer_radius_mm <= 0:
            raise CalibrationError(
                "A positive goniometer radius is required for displacement fitting."
            )
        parameter_names.append("specimen_displacement_mm")
        initial.append(0.0)
        lower.append(-5.0)
        upper.append(5.0)
    if fit_transparency:
        if geometry != "Bragg-Brentano symmetric":
            raise CalibrationError(
                "Specimen transparency is supported only for symmetric "
                "Bragg–Brentano geometry."
            )
        if goniometer_radius_mm <= 0:
            raise CalibrationError(
                "A positive goniometer radius is required for transparency fitting."
            )
        parameter_names.append("inverse_linear_absorption_mm")
        initial.append(0.1)
        lower.append(0.0)
        upper.append(20.0)

    def model(parameters):
        profile = {
            "geometry": geometry,
            "goniometer_radius_mm": goniometer_radius_mm,
            "zero_shift_deg": 0.0,
            "specimen_displacement_mm": 0.0,
            "inverse_linear_absorption_mm": 0.0,
        }
        for name, value in zip(parameter_names, parameters):
            profile[name] = float(value)
        return reference + position_correction_deg(reference, profile)

    if parameter_names:
        result = least_squares(
            lambda parameters: (observed - model(parameters)) / errors,
            np.asarray(initial, dtype=float),
            bounds=(np.asarray(lower), np.asarray(upper)),
            loss="soft_l1",
            max_nfev=10000,
        )
        parameters = result.x
        calculated = model(parameters)
        jacobian = result.jac
        dof = max(1, len(rows) - len(parameters))
        weighted_ss = float(np.sum(((observed - calculated) / errors) ** 2))
        # Peak-fit uncertainties are treated as absolute standard
        # uncertainties.  Inflate for under-dispersion only; never collapse a
        # calibration covariance to zero merely because synthetic or rounded
        # residuals happen to be tiny.
        covariance = np.linalg.pinv(jacobian.T @ jacobian) * max(
            weighted_ss / dof,
            1.0,
        )
        parameter_errors = np.sqrt(
            np.maximum(np.diag(covariance), 0.0)
        )
    else:
        parameters = np.asarray([], dtype=float)
        parameter_errors = np.asarray([], dtype=float)
        calculated = reference.copy()
        covariance = np.empty((0, 0), dtype=float)

    if len(parameters):
        scales = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = np.outer(scales, scales)
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0,
        )
        condition_number = float(np.linalg.cond(jacobian.T @ jacobian))
        if not np.isfinite(condition_number):
            condition_number = None
    else:
        correlation = np.empty((0, 0), dtype=float)
        condition_number = 0.0

    output = {
        "zero_shift_deg": 0.0,
        "zero_shift_error_deg": None,
        "specimen_displacement_mm": 0.0,
        "specimen_displacement_error_mm": None,
        "inverse_linear_absorption_mm": 0.0,
        "inverse_linear_absorption_error_mm": None,
        "calculated_positions": calculated,
        "covariance": {
            "parameter_names": list(parameter_names),
            "matrix": covariance.tolist(),
            "correlation_matrix": correlation.tolist(),
            "normal_matrix_condition_number": condition_number,
        },
    }
    for index, name in enumerate(parameter_names):
        output[name] = float(parameters[index])
        error_key = {
            "zero_shift_deg": "zero_shift_error_deg",
            "specimen_displacement_mm": "specimen_displacement_error_mm",
            "inverse_linear_absorption_mm": "inverse_linear_absorption_error_mm",
        }[name]
        output[error_key] = float(parameter_errors[index])
    return output


def _caglioti_fit(rows: list[dict]) -> dict:
    theta = np.radians(
        np.asarray(
            [row["reference_two_theta_deg"] for row in rows],
            dtype=float,
        )
        / 2.0
    )
    tangent = np.tan(theta)
    widths = np.asarray(
        [row["observed_fwhm_deg"] for row in rows],
        dtype=float,
    )
    width_errors = np.asarray(
        [
            float(row.get("fwhm_error_deg") or max(0.005, width * 0.03))
            for row, width in zip(rows, widths)
        ],
        dtype=float,
    )
    width_errors = np.maximum(width_errors, 1e-5)

    design = np.column_stack(
        [tangent * tangent, tangent, np.ones_like(tangent)]
    )
    target = widths * widths
    target_error = np.maximum(2.0 * widths * width_errors, 1e-8)
    weights = 1.0 / np.square(target_error)
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_target = target * np.sqrt(weights)

    coefficients, _, _, _ = np.linalg.lstsq(
        weighted_design,
        weighted_target,
        rcond=None,
    )
    u, v, w = [float(value) for value in coefficients]

    predicted_squared = design @ coefficients
    minimum_predicted = float(np.min(predicted_squared))
    if minimum_predicted <= 0:
        # Refit with bounded U/W and a physically permissive V.
        result = least_squares(
            lambda parameters: (
                widths
                - np.sqrt(
                    np.maximum(
                        parameters[0] * tangent * tangent
                        + parameters[1] * tangent
                        + parameters[2],
                        1e-12,
                    )
                )
            )
            / width_errors,
            np.asarray(
                [max(u, 0.0), v, max(w, float(np.min(widths) ** 2 * 0.25))]
            ),
            bounds=(
                np.asarray([0.0, -5.0, 1e-10]),
                np.asarray([10.0, 5.0, 10.0]),
            ),
            loss="soft_l1",
            max_nfev=20000,
        )
        u, v, w = [float(value) for value in result.x]

    predicted = np.sqrt(
        np.maximum(u * tangent * tangent + v * tangent + w, 0.0)
    )
    residuals = widths - predicted
    ss_res = float(np.sum(residuals * residuals))
    ss_tot = float(np.sum((widths - np.mean(widths)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "caglioti_u": u,
        "caglioti_v": v,
        "caglioti_w": w,
        "calculated_widths": predicted,
        "width_rmse_deg": float(np.sqrt(np.mean(residuals * residuals))),
        "width_r_squared": float(r_squared),
    }


def calibrate_instrument(
    fit_groups: list[dict],
    reference_peaks: list[dict],
    *,
    standard_name: str,
    standard_formula: str = "",
    radiation: str = "Cu Kα1",
    wavelength_angstrom: float = 1.5406,
    radiation_configuration: str = "Cu Kα1 monochromated",
    secondary_wavelength_angstrom: float = 1.5444274,
    secondary_to_primary_ratio: float = 0.5,
    geometry: str = "Bragg-Brentano symmetric",
    goniometer_radius_mm: float = 240.0,
    tolerance_deg: float = 0.25,
    intensity_cutoff_percent: float = 1.0,
    fit_zero_shift: bool = True,
    fit_displacement: bool = False,
    fit_transparency: bool = False,
    sample_length_mm: float = 0.0,
    receiving_slit_length_mm: float = 0.0,
    instrument_name: str = "",
    operator: str = "",
    notes: str = "",
) -> dict:
    if geometry not in GEOMETRIES:
        raise CalibrationError(f"Unsupported geometry: {geometry}")
    if wavelength_angstrom <= 0:
        raise CalibrationError("Wavelength must be positive.")
    if radiation_configuration not in RADIATION_CONFIGURATIONS:
        raise CalibrationError(
            f"Unsupported radiation configuration: {radiation_configuration}"
        )
    spectral_components = radiation_components(
        radiation_configuration,
        primary_wavelength_angstrom=wavelength_angstrom,
        secondary_wavelength_angstrom=secondary_wavelength_angstrom,
        secondary_to_primary_ratio=secondary_to_primary_ratio,
    )

    rows = match_standard_reflections(
        fit_groups,
        reference_peaks,
        tolerance_deg=tolerance_deg,
        intensity_cutoff_percent=intensity_cutoff_percent,
    )
    if len(rows) < 3:
        raise CalibrationError(
            "Fewer than three standard reflections were matched. "
            "Fit more standard peaks or increase the verified tolerance."
        )
    if fit_displacement and len(rows) < 5:
        raise CalibrationError(
            "At least five matched reflections spanning a broad 2θ range are "
            "required when fitting specimen displacement."
        )
    if fit_transparency and len(rows) < 7:
        raise CalibrationError(
            "At least seven matched reflections spanning a broad 2θ range are "
            "required when fitting specimen transparency."
        )
    angular_span = float(
        max(row["reference_two_theta_deg"] for row in rows)
        - min(row["reference_two_theta_deg"] for row in rows)
    )
    if fit_transparency and angular_span < 45.0:
        raise CalibrationError(
            "Transparency fitting requires at least 45° of calibrated 2θ span "
            "to reduce correlation with zero shift and displacement."
        )

    position = _position_fit(
        rows,
        geometry=geometry,
        goniometer_radius_mm=float(goniometer_radius_mm),
        fit_zero_shift=fit_zero_shift,
        fit_displacement=fit_displacement,
        fit_transparency=fit_transparency,
    )
    width = _caglioti_fit(rows)
    left_widths = [row.get("left_fwhm_deg") for row in rows]
    right_widths = [row.get("right_fwhm_deg") for row in rows]
    have_split_widths = all(
        value is not None
        and np.isfinite(float(value))
        and float(value) > 0
        for value in left_widths + right_widths
    )
    advanced_width = fit_instrument_width_models(
        [row["reference_two_theta_deg"] for row in rows],
        [row["observed_fwhm_deg"] for row in rows],
        fwhm_errors_deg=[
            float(row.get("fwhm_error_deg") or max(0.005, row["observed_fwhm_deg"] * 0.03))
            for row in rows
        ],
        left_fwhm_deg=(
            [float(value) for value in left_widths]
            if have_split_widths
            else None
        ),
        right_fwhm_deg=(
            [float(value) for value in right_widths]
            if have_split_widths
            else None
        ),
    )
    selected_width = advanced_width["best"]
    selected_parameters = selected_width["parameters"]

    predicted_positions = np.asarray(
        position["calculated_positions"],
        dtype=float,
    )
    predicted_widths = np.asarray(
        selected_width["calculated_fwhm_deg"],
        dtype=float,
    )
    observed_positions = np.asarray(
        [row["observed_two_theta_deg"] for row in rows],
        dtype=float,
    )
    observed_widths = np.asarray(
        [row["observed_fwhm_deg"] for row in rows],
        dtype=float,
    )
    position_residuals = observed_positions - predicted_positions
    width_residuals = observed_widths - predicted_widths

    for index, row in enumerate(rows):
        row["calculated_two_theta_deg"] = float(predicted_positions[index])
        row["position_residual_deg"] = float(position_residuals[index])
        row["calculated_fwhm_deg"] = float(predicted_widths[index])
        row["fwhm_residual_deg"] = float(width_residuals[index])

    position_mae = float(np.mean(np.abs(position_residuals)))
    position_rmse = float(np.sqrt(np.mean(position_residuals ** 2)))
    width_rmse = float(np.sqrt(np.mean(width_residuals ** 2)))
    valid_min = float(min(row["reference_two_theta_deg"] for row in rows))
    valid_max = float(max(row["reference_two_theta_deg"] for row in rows))

    warnings = []
    if len(rows) < 5:
        warnings.append("Only a small number of reflections constrain the profile.")
    if valid_max - valid_min < 30.0:
        warnings.append("The matched 2θ range is narrow for U–V–W calibration.")
    if position_mae > 0.02:
        warnings.append("Mean absolute peak-position residual exceeds 0.02°.")
    if width["width_r_squared"] < 0.90:
        warnings.append("Caglioti width fit has R² below 0.90.")
    if selected_width["model"] == "Split pseudo-Voigt":
        warnings.append(
            "Split pseudo-Voigt asymmetry is empirical unless left/right peak widths were fitted explicitly."
        )
    if len(rows) < 6 and selected_width["model"] == "TCH pseudo-Voigt":
        warnings.append(
            "TCH Gaussian/Lorentzian separation is weakly constrained by fewer than six standard reflections."
        )
    if abs(position["zero_shift_deg"]) > 0.20:
        warnings.append("Large constant zero shift; inspect alignment and wavelength.")
    if fit_zero_shift and fit_displacement:
        warnings.append(
            "Zero shift and specimen displacement can be strongly correlated; "
            "verify with a broad angular standard."
        )
    if fit_transparency and (fit_zero_shift or fit_displacement):
        warnings.append(
            "Transparency, zero shift and displacement share correlated position "
            "information; inspect the stored covariance and correlation matrix."
        )
    position_condition = float(
        position["covariance"].get("normal_matrix_condition_number") or 0.0
    )
    if position_condition > 1e8:
        warnings.append(
            "Position-calibration normal matrix is ill-conditioned; do not "
            "interpret individual correction terms independently."
        )

    geometry_ratio = axial_sh_over_l(
        sample_length_mm=sample_length_mm,
        receiving_slit_length_mm=receiving_slit_length_mm,
        goniometer_radius_mm=goniometer_radius_mm,
    )

    profile = {
        "profile_version": PROFILE_VERSION,
        "created_utc": utc_now_text(),
        "instrument_name": str(instrument_name),
        "operator": str(operator),
        "standard_name": str(standard_name),
        "standard_formula": str(standard_formula),
        "radiation": str(radiation),
        "radiation_configuration": str(radiation_configuration),
        "spectral_components": spectral_components,
        "wavelength_angstrom": float(wavelength_angstrom),
        "secondary_wavelength_angstrom": (
            float(spectral_components[1]["wavelength_angstrom"])
            if len(spectral_components) > 1
            else None
        ),
        "secondary_to_primary_ratio": (
            float(spectral_components[1]["relative_intensity"])
            if len(spectral_components) > 1
            else 0.0
        ),
        "geometry": geometry,
        "goniometer_radius_mm": float(goniometer_radius_mm),
        "fit_zero_shift": bool(fit_zero_shift),
        "fit_specimen_displacement": bool(fit_displacement),
        "fit_specimen_transparency": bool(fit_transparency),
        "zero_shift_deg": float(position["zero_shift_deg"]),
        "zero_shift_error_deg": position["zero_shift_error_deg"],
        "specimen_displacement_mm": float(
            position["specimen_displacement_mm"]
        ),
        "specimen_displacement_error_mm": position[
            "specimen_displacement_error_mm"
        ],
        "inverse_linear_absorption_mm": float(
            position["inverse_linear_absorption_mm"]
        ),
        "inverse_linear_absorption_error_mm": position[
            "inverse_linear_absorption_error_mm"
        ],
        "position_covariance": position["covariance"],
        "profile_model": str(selected_width["model"]),
        "caglioti_u": float(selected_parameters.get("caglioti_u", width["caglioti_u"])),
        "caglioti_v": float(selected_parameters.get("caglioti_v", width["caglioti_v"])),
        "caglioti_w": float(selected_parameters.get("caglioti_w", width["caglioti_w"])),
        "eta": float(selected_parameters.get("eta", 0.5)),
        "lorentzian_x": float(selected_parameters.get("lorentzian_x", 0.0)),
        "lorentzian_y": float(selected_parameters.get("lorentzian_y", 0.0)),
        "axial_asymmetry": float(selected_parameters.get("axial_asymmetry", 0.0)),
        "axial_divergence_model": "FCJ/SH-L geometry-constrained reduced model",
        "sample_length_mm": float(sample_length_mm),
        "receiving_slit_length_mm": float(receiving_slit_length_mm),
        "axial_sh_over_l": float(geometry_ratio),
        "width_covariance": {
            "parameter_names": list(selected_width.get("parameter_names", [])),
            "matrix": selected_width.get("covariance_matrix", []),
            "correlation_matrix": selected_width.get("correlation_matrix", []),
            "normal_matrix_condition_number": selected_width.get(
                "normal_matrix_condition_number"
            ),
        },
        "profile_model_comparison": advanced_width,
        "valid_two_theta_min_deg": valid_min,
        "valid_two_theta_max_deg": valid_max,
        "matched_reflection_count": len(rows),
        "position_mae_deg": position_mae,
        "position_rmse_deg": position_rmse,
        "width_rmse_deg": width_rmse,
        "width_r_squared": float(width["width_r_squared"]),
        "selected_profile_bic": float(selected_width["information_criteria"]["bic"]),
        "observations": rows,
        "warnings": warnings,
        "notes": str(notes),
    }
    profile["fingerprint"] = profile_fingerprint(profile)
    return profile


def quality_check(
    profile: dict,
    current_fit_groups: list[dict],
    reference_peaks: list[dict],
    tolerance_deg: float = 0.25,
) -> dict:
    rows = match_standard_reflections(
        current_fit_groups,
        reference_peaks,
        tolerance_deg=tolerance_deg,
        intensity_cutoff_percent=1.0,
    )
    if len(rows) < 3:
        raise CalibrationError(
            "At least three standard reflections are required for QA."
        )

    position_residuals = []
    width_residuals = []
    for row in rows:
        reference_position = float(row["reference_two_theta_deg"])
        predicted_position = float(
            corrected_reference_position(reference_position, profile)
        )
        predicted_width = float(
            instrument_fwhm_deg(
                reference_position,
                profile,
                allow_extrapolation=False,
            )
        )
        row["calculated_two_theta_deg"] = predicted_position
        row["position_residual_deg"] = (
            float(row["observed_two_theta_deg"]) - predicted_position
        )
        row["calculated_fwhm_deg"] = predicted_width
        row["calculated_position_error_deg"] = (
            position_correction_uncertainty_deg(
                reference_position,
                profile,
                allow_extrapolation=False,
            )
        )
        row["calculated_fwhm_error_deg"] = (
            instrument_fwhm_uncertainty_deg(
                reference_position,
                profile,
                allow_extrapolation=False,
            )
        )
        row["fwhm_residual_deg"] = (
            float(row["observed_fwhm_deg"]) - predicted_width
        )
        position_residuals.append(row["position_residual_deg"])
        width_residuals.append(row["fwhm_residual_deg"])

    position_mae = float(np.mean(np.abs(position_residuals)))
    width_rmse = float(np.sqrt(np.mean(np.square(width_residuals))))
    baseline_position = max(float(profile.get("position_mae_deg", 0.0)), 1e-6)
    baseline_width = max(float(profile.get("width_rmse_deg", 0.0)), 1e-6)
    warnings = []
    if position_mae > max(0.03, baseline_position * 2.0):
        warnings.append("Peak-position drift exceeds the QA threshold.")
    if width_rmse > max(0.03, baseline_width * 2.0):
        warnings.append("Instrument-resolution drift exceeds the QA threshold.")
    status = "Pass" if not warnings else "Review"

    return {
        "checked_utc": utc_now_text(),
        "profile_fingerprint": profile.get("fingerprint"),
        "status": status,
        "matched_reflection_count": len(rows),
        "position_mae_deg": position_mae,
        "width_rmse_deg": width_rmse,
        "observations": rows,
        "warnings": warnings,
    }


def save_profile(path: str | Path, profile: dict) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    saved = deepcopy(profile)
    saved["fingerprint"] = profile_fingerprint(saved)
    path.write_text(
        json.dumps(saved, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _migrate_profile(profile: dict) -> dict:
    """Upgrade legacy calibration JSON in memory without losing provenance."""

    migrated = deepcopy(profile)
    source_version = int(migrated.get("profile_version", 1))
    if source_version >= PROFILE_VERSION:
        return migrated
    migrated.setdefault("radiation_configuration", "Custom monochromatic")
    migrated.setdefault(
        "spectral_components",
        [
            {
                "label": "primary",
                "wavelength_angstrom": float(migrated["wavelength_angstrom"]),
                "relative_intensity": 1.0,
            }
        ],
    )
    migrated.setdefault("secondary_wavelength_angstrom", None)
    migrated.setdefault("secondary_to_primary_ratio", 0.0)
    migrated.setdefault("fit_specimen_transparency", False)
    migrated.setdefault("inverse_linear_absorption_mm", 0.0)
    migrated.setdefault("inverse_linear_absorption_error_mm", None)
    migrated.setdefault("sample_length_mm", 0.0)
    migrated.setdefault("receiving_slit_length_mm", 0.0)
    migrated.setdefault("axial_sh_over_l", 0.0)
    migrated.setdefault(
        "axial_divergence_model",
        "legacy empirical split-width model",
    )

    position_names = []
    position_variances = []
    for name, error_name in (
        ("zero_shift_deg", "zero_shift_error_deg"),
        ("specimen_displacement_mm", "specimen_displacement_error_mm"),
    ):
        error = migrated.get(error_name)
        if error is not None and np.isfinite(float(error)):
            position_names.append(name)
            position_variances.append(float(error) ** 2)
    migrated.setdefault(
        "position_covariance",
        {
            "parameter_names": position_names,
            "matrix": np.diag(position_variances).tolist(),
            "correlation_matrix": np.eye(len(position_names)).tolist(),
            "normal_matrix_condition_number": None,
            "classification": "diagonal covariance reconstructed from legacy errors",
        },
    )
    migrated.setdefault(
        "width_covariance",
        {
            "parameter_names": [],
            "matrix": [],
            "correlation_matrix": [],
            "normal_matrix_condition_number": None,
            "classification": "unavailable in legacy profile",
        },
    )
    migrated["profile_version"] = PROFILE_VERSION
    migrated["migrated_from_profile_version"] = source_version
    return migrated


def load_profile(path: str | Path) -> dict:
    path = Path(path)
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise CalibrationError("Calibration profile must be a JSON object.")
    required = (
        "caglioti_u",
        "caglioti_v",
        "caglioti_w",
        "wavelength_angstrom",
        "valid_two_theta_min_deg",
        "valid_two_theta_max_deg",
    )
    missing = [key for key in required if key not in profile]
    if missing:
        raise CalibrationError(
            "Calibration profile is missing: " + ", ".join(missing)
        )
    source_version = int(profile.get("profile_version", 1))
    expected = profile_fingerprint(profile)
    stored = profile.get("fingerprint")
    fingerprint_valid = stored in (None, expected)
    source_fingerprint = stored or expected
    profile = _migrate_profile(profile)
    profile["fingerprint_valid"] = fingerprint_valid
    profile["source_fingerprint"] = source_fingerprint
    profile["fingerprint"] = (
        profile_fingerprint(profile)
        if source_version < PROFILE_VERSION
        else expected
    )
    return profile
