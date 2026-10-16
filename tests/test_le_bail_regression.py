"""Le Bail area-repartition regressions: no Pawley substitution or data masking."""
from __future__ import annotations

import json

import numpy as np
import pytest

from afruz_pxrd.crystallography import calculate_powder_pattern
from afruz_pxrd.phase_identification import reference_from_cif_pattern
from afruz_pxrd.whole_pattern_refinement import (
    WholePatternCancelled,
    WholePatternPhaseSpec,
    _build_peak_basis,
    _le_bail_initial_intensities,
    _positive_background_basis,
    _solve_le_bail,
    _solve_le_bail_cuda,
    refine_whole_pattern,
)


@pytest.mark.parametrize("step", [0.005, 0.02, 0.08])
@pytest.mark.parametrize("count_scale", [1.0, 1e-6])
def test_default_cycles_recover_areas_independent_of_step_and_units(step, count_scale):
    x = np.arange(5.0, 75.0 + step / 2, step)
    refs = [dict(two_theta_deg=t, reference_intensity=i)
            for t, i in ((21, 100), (34, 40), (53, 30))]
    peaks, _ = _build_peak_basis(x, refs, u=.005, v=0, w=.02, eta=.5)
    background = _positive_background_basis(x, 2)
    true_areas = np.array([850., 220., 450.]) * count_scale
    observed = peaks @ true_areas + background @ (np.array([20., 25., 40.]) * count_scale)
    weights = 1 / np.sqrt(observed)
    intensity, coefficients, calculated, diagnostics = _solve_le_bail(
        peaks, background, refs, observed, weights, cycles=12,
    )
    np.testing.assert_allclose(intensity, true_areas, rtol=3e-3)
    assert np.linalg.norm((observed-calculated)*weights) / np.linalg.norm(observed*weights) < .002
    assert np.min(coefficients) >= 0
    assert diagnostics["algorithm_revision"] == 2
    assert diagnostics["cycles"] == 12
    assert diagnostics["final_background_refitted"]


def test_sampled_profile_initialization_has_no_inverse_step_factor():
    for points in (501, 2001):
        x = np.linspace(5, 55, points)
        refs = [dict(two_theta_deg=30, reference_intensity=100)]
        peaks, _ = _build_peak_basis(x, refs, u=.005, v=0, w=.02, eta=.5)
        observed = 850 * peaks[:, 0] + 20
        initial = _le_bail_initial_intensities(peaks, refs, observed, np.ones(len(x)))
        assert initial[0] == pytest.approx(850, rel=2e-5)


@pytest.mark.parametrize("weights", [np.ones(60), np.tile([1., .1], 30)])
def test_signed_net_noise_is_integrated_before_positivity_not_rectified(weights):
    # A fixed background has already been subtracted. Half the net points are
    # negative, but their signed area is positive and exactly known. A ratio
    # ceiling or pointwise clipping would inflate this weak reflection.
    peaks = np.ones((60, 1))
    net = .5 + 2 * np.tile([-1., 1.], 30)
    intensity, _, calculated, diagnostics = _solve_le_bail(
        peaks, np.empty((60, 0)), [{"reference_intensity": 100}],
        net, weights, cycles=40,
    )
    np.testing.assert_allclose(intensity, [.5], rtol=1e-9)
    np.testing.assert_allclose(calculated, .5, rtol=1e-9)
    assert diagnostics["repartition_weighting"] == "unweighted observed-area partition"
    json.dumps(diagnostics, allow_nan=False)


def test_final_background_is_fitted_against_final_intensities():
    x = np.linspace(-3, 3, 501)
    peaks = np.exp(-(x / .3)**2)[:, None]
    background = np.ones((len(x), 1))
    y = 35 * peaks[:, 0] + 7
    weights = 1 / np.sqrt(y)
    intensity, coefficients, calc, diag = _solve_le_bail(
        peaks, background, [{"reference_intensity": 100}], y, weights, cycles=2,
    )
    expected = np.sum((y-peaks@intensity)*weights**2) / np.sum(weights**2)
    assert coefficients[0] == pytest.approx(expected, abs=1e-9)
    np.testing.assert_allclose(calc, peaks@intensity+background@coefficients)
    assert diag["history"][-1]["weighted_residual_sum_squares"] == pytest.approx(np.sum(((y-calc)*weights)**2))


def test_coincident_reflections_remain_nonidentifiable_and_keep_seed_ratio():
    x = np.linspace(-2, 2, 401)
    peak = np.exp(-(x / .2)**2)
    areas, _, calc, _ = _solve_le_bail(
        np.column_stack([peak, peak]), np.empty((len(x), 0)),
        [{"reference_intensity": 2}, {"reference_intensity": 1}],
        peak * 90, np.ones(len(x)), 12,
    )
    assert areas.sum() == pytest.approx(90, rel=1e-4)
    assert areas[0] / areas[1] == pytest.approx(2)
    np.testing.assert_allclose(calc, peak*90, rtol=1e-4)


@pytest.mark.parametrize("observed", [np.zeros(20), -np.ones(20)])
def test_no_signal_diagnostics_and_profiles_stay_finite(observed):
    areas, coeff, calc, diag = _solve_le_bail(
        np.ones((20, 1)), np.ones((20, 1)),
        [{"reference_intensity": 1}], observed, np.ones(20), 12,
    )
    assert np.all(np.isfinite(calc))
    assert np.all(areas >= 0) and np.all(coeff >= 0)
    json.dumps(diag, allow_nan=False)


@pytest.mark.parametrize("solver", [_solve_le_bail, _solve_le_bail_cuda])
def test_inner_repartition_cancellation_is_not_swallowed(solver):
    count = 0

    def cancel():
        nonlocal count
        count += 1
        return count >= 4

    with pytest.raises(WholePatternCancelled):
        solver(np.ones((60, 1)), np.ones((60, 1)),
               [{"reference_intensity": 1}], np.ones(60)*3,
               np.ones(60), 100, cancel_check=cancel)
    assert count == 4


@pytest.mark.parametrize("use_gpu", [False, True])
@pytest.mark.parametrize("profile", ["Pseudo-Voigt U-V-W", "TCH pseudo-Voigt"])
def test_nacl_default_twelve_cycles_no_longer_gives_large_rwp(
    nacl_raw_frame, nacl_structure, use_gpu, profile,
):
    """End-to-end regression on bundled synthetic NaCl, not a certified standard."""
    x = nacl_raw_frame["two_theta_deg"].to_numpy(float)
    y = nacl_raw_frame["intensity_counts"].to_numpy(float)
    sigma = nacl_raw_frame["sigma_counts"].to_numpy(float)
    original = y.copy()
    pattern = calculate_powder_pattern(nacl_structure, 1.5406, 4, 80)
    reference = reference_from_cif_pattern(nacl_structure, pattern, 1.5406)
    result = refine_whole_pattern(
        x, y, [WholePatternPhaseSpec(reference, refine_cell=True)],
        mode="Le Bail-style extraction", observed_sigma=sigma,
        intensity_provenance="raw_counts", use_gpu=use_gpu,
        profile_model=profile, refine_lorentzian_width=profile == "TCH pseudo-Voigt",
        use_staged_refinement=True, maximum_nonlinear_evaluations=120,
        maximum_optimization_points=1800,
    )
    assert result["success"]
    assert result["rwp_percent"] < (6.3 if profile == "TCH pseudo-Voigt" else 7.2)
    assert result["phases"][0]["refined_cell"]["a"] == pytest.approx(5.6402, abs=.0005)
    assert result["extraction_diagnostics"]["algorithm_revision"] == 2
    assert result["extraction_diagnostics"]["cycles"] == 12
    assert result["data_point_count"] == len(x)
    assert np.array_equal(y, original)
    assert result["observed_y"] == original.tolist()
    assert "Le Bail" in result["mode"]
    if not result["extraction_diagnostics"]["converged_at_1e-5"]:
        assert any("cycling has not converged" in w for w in result["warnings"])
