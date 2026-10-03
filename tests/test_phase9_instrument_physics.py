from __future__ import annotations

import json
import math

import numpy as np
import pytest

from afruz_pxrd.crystal_profiles import caglioti_fwhm_deg, profile_unit_area
from afruz_pxrd.crystallography import refine_unit_cell, two_theta_from_hkl
from afruz_pxrd.instrument_calibration import (
    CalibrationRangeError,
    calibrate_instrument,
    instrument_fwhm_deg,
    instrument_fwhm_uncertainty_deg,
    load_profile,
    position_correction_deg,
    save_profile,
)
from afruz_pxrd.instrument_physics import (
    CU_K_ALPHA1_ANGSTROM,
    CU_K_ALPHA2_ANGSTROM,
    axial_divergence_characteristics,
    axial_sh_over_l,
    prepare_calibration_prior,
    radiation_components,
    spectral_positions_deg,
    validate_calibration_range,
)
from afruz_pxrd.size_strain import correct_instrument_broadening


def _synthetic_calibration():
    reference_positions = np.linspace(20.0, 120.0, 11)
    position_profile = {
        "geometry": "Bragg-Brentano symmetric",
        "goniometer_radius_mm": 240.0,
        "zero_shift_deg": 0.035,
        "specimen_displacement_mm": 0.12,
        "inverse_linear_absorption_mm": 0.65,
    }
    widths = caglioti_fwhm_deg(reference_positions, 0.004, -0.001, 0.012)
    widths = widths * (1.0 + 0.004 * np.sin(np.arange(len(widths))))
    observed_positions = reference_positions + position_correction_deg(
        reference_positions,
        position_profile,
    )
    fit_groups = []
    references = []
    for index, (reference, observed, width) in enumerate(
        zip(reference_positions, observed_positions, widths),
        start=1,
    ):
        fit_groups.append(
            {
                "group_id": index,
                "components": [
                    {
                        "center": float(observed),
                        "fwhm": float(width),
                        "center_error": 0.002,
                        "fwhm_error": 0.001,
                        "model": "Pseudo-Voigt",
                    }
                ],
            }
        )
        references.append(
            {
                "two_theta": float(reference),
                "intensity": 100.0 - index,
                "hkl_label": str(index),
            }
        )
    return calibrate_instrument(
        fit_groups,
        references,
        standard_name="Synthetic metrology standard",
        wavelength_angstrom=CU_K_ALPHA1_ANGSTROM,
        radiation_configuration="Cu Kα1/Kα2 doublet",
        geometry="Bragg-Brentano symmetric",
        goniometer_radius_mm=240.0,
        fit_zero_shift=True,
        fit_displacement=True,
        fit_transparency=True,
        sample_length_mm=12.0,
        receiving_slit_length_mm=10.0,
    )


def test_radiation_configuration_preserves_resolved_cu_doublet():
    components = radiation_components("Cu Kα1/Kα2 doublet")
    assert components[0]["wavelength_angstrom"] == pytest.approx(
        CU_K_ALPHA1_ANGSTROM
    )
    assert components[1]["wavelength_angstrom"] == pytest.approx(
        CU_K_ALPHA2_ANGSTROM
    )
    assert components[1]["relative_intensity"] == pytest.approx(0.5)
    positions = spectral_positions_deg(60.0, components)
    assert positions[1]["two_theta_deg"] > positions[0]["two_theta_deg"]


def test_position_terms_are_separate_and_covariance_is_retained():
    profile = _synthetic_calibration()
    assert profile["profile_version"] == 3
    assert profile["zero_shift_deg"] == pytest.approx(0.035, abs=2e-5)
    assert profile["specimen_displacement_mm"] == pytest.approx(0.12, abs=2e-4)
    assert profile["inverse_linear_absorption_mm"] == pytest.approx(0.65, abs=2e-3)
    covariance = profile["position_covariance"]
    assert covariance["parameter_names"] == [
        "zero_shift_deg",
        "specimen_displacement_mm",
        "inverse_linear_absorption_mm",
    ]
    assert np.asarray(covariance["matrix"]).shape == (3, 3)
    assert profile["width_covariance"]["parameter_names"]
    assert profile["spectral_components"][1]["relative_intensity"] == 0.5


def test_geometry_constrained_axial_profile_is_normalized_and_angle_dependent():
    ratio = axial_sh_over_l(
        sample_length_mm=12.0,
        receiving_slit_length_mm=10.0,
        goniometer_radius_mm=240.0,
    )
    low = axial_divergence_characteristics(
        20.0,
        sh_over_l=ratio,
        base_fwhm_deg=0.12,
    )
    high = axial_divergence_characteristics(
        100.0,
        sh_over_l=ratio,
        base_fwhm_deg=0.12,
    )
    assert low["asymmetry_factor"] > high["asymmetry_factor"] > 0
    x = np.linspace(-20.0, 60.0, 80001)
    values = profile_unit_area(
        x,
        20.0,
        u=0.0,
        v=0.0,
        w=0.12**2,
        eta=0.4,
        axial_sh_over_l=ratio,
    )
    assert np.trapezoid(values, x) == pytest.approx(1.0, rel=4e-3)


def test_range_gate_and_width_prediction_uncertainty():
    profile = _synthetic_calibration()
    validate_calibration_range(profile, [20.0, 120.0])
    with pytest.raises(CalibrationRangeError):
        instrument_fwhm_deg(10.0, profile, allow_extrapolation=False)
    width = instrument_fwhm_deg(60.0, profile, allow_extrapolation=False)
    uncertainty = instrument_fwhm_uncertainty_deg(
        60.0,
        profile,
        allow_extrapolation=False,
    )
    assert width > 0
    assert uncertainty >= 0


def test_instrument_width_uncertainty_reaches_size_strain_correction():
    corrected, propagated, note = correct_instrument_broadening(
        0.20,
        0.12,
        "Gaussian quadrature",
        observed_error_deg=0.01,
        instrument_error_deg=0.02,
    )
    assert note == ""
    assert corrected == pytest.approx(0.16)
    expected = math.sqrt((0.20 / 0.16 * 0.01) ** 2 + (0.12 / 0.16 * 0.02) ** 2)
    assert propagated == pytest.approx(expected)


def test_cell_refinement_applies_position_covariance():
    cell = {
        "a": 5.64,
        "b": 5.64,
        "c": 5.64,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    }
    profile = _synthetic_calibration()
    matches = []
    for hkl in ((1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2)):
        reference = two_theta_from_hkl(cell, hkl, CU_K_ALPHA1_ANGSTROM)
        observed = reference + position_correction_deg(reference, profile)
        matches.append(
            {
                "hkl": hkl,
                "observed_2theta": observed,
                "observed_error": 0.002,
                "initial_delta": observed - reference,
            }
        )
    result = refine_unit_cell(
        matches,
        cell,
        "Cubic",
        CU_K_ALPHA1_ANGSTROM,
        refine_zero_shift=False,
        instrument_profile=profile,
        apply_instrument_position_correction=True,
    )
    assert result["refined_cell"]["a"] == pytest.approx(5.64, abs=2e-5)
    assert result["instrument_position_covariance_propagated"] is True
    assert result["matches"][0]["instrument_position_error_deg"] >= 0


def test_calibration_covariance_builds_downstream_gaussian_prior():
    profile = _synthetic_calibration()
    names = ["zero_shift_deg", "caglioti_u", "caglioti_v", "caglioti_w"]
    prior = prepare_calibration_prior(profile, names)
    assert prior is not None
    assert prior["effective_rank"] >= 1
    assert set(prior["parameter_names"]).issubset(names)


def test_legacy_profile_migrates_to_version_three(tmp_path):
    legacy = {
        "profile_version": 2,
        "wavelength_angstrom": 1.5406,
        "caglioti_u": 0.004,
        "caglioti_v": 0.0,
        "caglioti_w": 0.012,
        "zero_shift_deg": 0.01,
        "zero_shift_error_deg": 0.002,
        "specimen_displacement_mm": 0.0,
        "specimen_displacement_error_mm": None,
        "valid_two_theta_min_deg": 20.0,
        "valid_two_theta_max_deg": 120.0,
    }
    path = tmp_path / "legacy-profile.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    migrated = load_profile(path)
    assert migrated["profile_version"] == 3
    assert migrated["migrated_from_profile_version"] == 2
    assert migrated["position_covariance"]["parameter_names"] == ["zero_shift_deg"]
    assert migrated["width_covariance"]["classification"] == "unavailable in legacy profile"


def test_version_three_profile_fingerprint_round_trip(tmp_path):
    profile = _synthetic_calibration()
    json.dumps(profile, allow_nan=False)
    path = save_profile(tmp_path / "profile.json", profile)
    stored = json.loads(path.read_text(encoding="utf-8"))
    loaded = load_profile(path)
    assert loaded["fingerprint_valid"] is True
    assert loaded["fingerprint"] == stored["fingerprint"]
