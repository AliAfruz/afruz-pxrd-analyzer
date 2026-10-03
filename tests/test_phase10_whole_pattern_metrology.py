from __future__ import annotations

import numpy as np

from afruz_pxrd.refinement_export import export_refinement_txt_bundle
from afruz_pxrd.whole_pattern_metrology import intensity_metrology
from afruz_pxrd.whole_pattern_refinement import (
    _build_peak_basis,
    _reflection_positions,
    _solve_le_bail,
)


def _linear_result(peak_basis: np.ndarray, intensities: np.ndarray) -> dict:
    x = np.linspace(20.0, 30.0, peak_basis.shape[0])
    background = np.ones((len(x), 1))
    coefficients = np.array([4.0])
    observed = peak_basis @ intensities + background @ coefficients
    return intensity_metrology(
        peak_basis,
        background,
        observed,
        np.ones_like(observed),
        intensities,
        coefficients,
        statistics_valid=True,
        method="Pawley decomposition",
        overlap_correlation_threshold=0.90,
    )


def test_separated_reflections_have_finite_covariance_and_full_rank():
    x = np.linspace(-5.0, 5.0, 801)
    peaks = np.column_stack(
        [np.exp(-((x + 2.0) / 0.2) ** 2), np.exp(-((x - 2.0) / 0.2) ** 2)]
    )
    result = _linear_result(peaks, np.array([100.0, 60.0]))

    assert result["effective_independent_reflection_count"] == 2
    assert result["individually_identifiable_count"] == 2
    assert result["unresolved_group_count"] == 0
    assert all(
        row["standard_error"] is not None
        for row in result["reflection_diagnostics"]
    )


def test_coincident_reflections_are_gated_as_not_individually_identifiable():
    x = np.linspace(-3.0, 3.0, 501)
    peak = np.exp(-(x / 0.25) ** 2)
    result = _linear_result(
        np.column_stack([peak, peak]),
        np.array([55.0, 45.0]),
    )

    assert result["effective_independent_reflection_count"] == 1
    assert result["unresolved_group_count"] == 1
    assert result["individually_identifiable_count"] == 0
    assert result["overlap_groups"][0]["intensity_sum"] == 100.0
    assert {
        row["identifiability"] for row in result["reflection_diagnostics"]
    } == {"not-individually-identifiable"}


def test_doublet_basis_is_normalized_to_total_integrated_intensity():
    x = np.linspace(35.0, 45.0, 10001)
    reflection = {
        "two_theta_deg": 40.0,
        "spectral_components": [
            {"two_theta_deg": 40.0, "relative_intensity": 1.0},
            {"two_theta_deg": 40.11, "relative_intensity": 0.5},
        ],
    }
    basis, widths = _build_peak_basis(
        x,
        [reflection],
        u=0.0,
        v=0.0,
        w=0.01,
        eta=0.4,
    )

    np.testing.assert_allclose(
        np.trapezoid(basis[:, 0], x), 1.0, rtol=0.0, atol=4e-3
    )
    assert widths[0] > 0.0


def test_geometry_correction_and_residual_zero_are_kept_separate():
    phases = [
        {
            "phase_index": 0,
            "name": "test",
            "initial_cell": None,
            "reflections": [
                {
                    "initial_two_theta_deg": 40.0,
                    "hkl": None,
                    "hkl_label": "unindexed",
                }
            ],
        }
    ]
    state = {"zero_shift_deg": 0.02, "phase_cells": {0: None}}
    profile = {
        "geometry": "Bragg-Brentano symmetric",
        "goniometer_radius_mm": 240.0,
        "specimen_displacement_mm": 0.1,
        "inverse_linear_absorption_mm": 0.0,
        "zero_shift_deg": 0.3,
    }
    rows, warnings = _reflection_positions(
        phases,
        state,
        1.5406,
        spectral_components=[
            {
                "label": "primary",
                "wavelength_angstrom": 1.5406,
                "relative_intensity": 1.0,
            }
        ],
        instrument_profile=profile,
        apply_instrument_position_correction=True,
    )

    assert not warnings
    row = rows[0]
    assert row["instrument_position_correction_deg"] < 0.0
    assert np.isclose(
        row["two_theta_deg"],
        40.0 + row["instrument_position_correction_deg"] + 0.02,
    )
    assert not np.isclose(row["two_theta_deg"], 40.32)


def test_le_bail_cycle_history_is_reproducible_and_explicit():
    x = np.linspace(-3.0, 3.0, 501)
    peaks = np.column_stack(
        [np.exp(-((x + 0.7) / 0.25) ** 2), np.exp(-((x - 0.7) / 0.25) ** 2)]
    )
    background = np.ones((len(x), 1))
    observed = peaks @ np.array([80.0, 35.0]) + 3.0
    _intensities, _coefficients, _calculated, diagnostics = _solve_le_bail(
        peaks,
        background,
        [{"reference_intensity": 100.0}, {"reference_intensity": 50.0}],
        observed,
        np.ones_like(observed),
        8,
    )

    assert diagnostics["cycles"] == 8
    assert len(diagnostics["history"]) == 8
    assert diagnostics["history"][-1]["weighted_residual_sum_squares"] >= 0.0
    assert isinstance(diagnostics["converged_at_1e-5"], bool)


def test_covariance_and_overlap_groups_are_exported(tmp_path):
    result = {
        "observed_x": [1.0, 2.0],
        "observed_y": [10.0, 11.0],
        "calculated_y": [10.0, 11.0],
        "background_y": [1.0, 1.0],
        "difference_y": [0.0, 0.0],
        "phases": [],
        "reflections": [
            {"phase_name": "A", "hkl_label": "100"},
            {"phase_name": "A", "hkl_label": "101"},
        ],
        "intensity_metrology": {
            "covariance_interpretation": "active-set linear least-squares covariance",
            "intensity_covariance": [[4.0, -1.0], [-1.0, 9.0]],
            "intensity_correlation": [[1.0, -1.0 / 6.0], [-1.0 / 6.0, 1.0]],
            "overlap_groups": [{"group_id": 1, "reflection_indices": [0, 1]}],
            "overlap_correlation_threshold": 0.9,
            "severe_correlation_threshold": 0.98,
        },
    }
    exported = export_refinement_txt_bundle(
        tmp_path / "whole_pattern.txt",
        result,
        dataset_name="synthetic",
        refinement_label="Pawley extraction",
    )

    covariance_text = (tmp_path / "whole_pattern_intensity_covariance.txt").read_text(
        encoding="utf-8"
    )
    assert exported["file_count"] >= 12
    assert "upper triangle" in covariance_text
    assert "-1" in covariance_text
    assert (tmp_path / "whole_pattern_overlap_groups.txt").is_file()
