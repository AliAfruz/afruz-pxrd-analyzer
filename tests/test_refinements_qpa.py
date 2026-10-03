from __future__ import annotations

import numpy as np
import pytest

from afruz_pxrd.crystallography import calculate_powder_pattern
from afruz_pxrd.phase_identification import reference_from_cif_pattern
from afruz_pxrd.rietveld_refinement import RietveldPhaseSpec, refine_rietveld
from afruz_pxrd.validated_qpa import QPAValidationError, compute_hill_howard_qpa
from afruz_pxrd.whole_pattern_refinement import (
    WholePatternPhaseSpec,
    _positive_background_basis,
    _solve_le_bail,
    refine_whole_pattern,
)


WAVELENGTH = 1.5406


def test_le_bail_background_is_constrained_to_nonnegative_counts():
    x = np.linspace(5.0, 35.0, 301)
    peak = np.exp(-0.5 * ((x - 10.0) / 0.15) ** 2)[:, None]
    y = 30.0 + 0.7 * (x - 5.0) + 250.0 * peak[:, 0]
    background_basis = _positive_background_basis(x, 4)
    reflections = [{"reference_intensity": 100.0}]

    _, coefficients, calculated, _ = _solve_le_bail(
        peak,
        background_basis,
        reflections,
        y,
        np.ones_like(y),
        cycles=8,
    )

    fitted_background = background_basis @ coefficients
    assert np.min(fitted_background) >= -1e-10
    assert np.all(np.isfinite(calculated))


def test_pawley_background_is_constrained_to_nonnegative_counts(
    nacl_raw_frame, nacl_structure
):
    x, y, sigma = _arrays(nacl_raw_frame)
    pattern = calculate_powder_pattern(nacl_structure, WAVELENGTH, 4.0, 80.0)
    reference = reference_from_cif_pattern(nacl_structure, pattern, WAVELENGTH)
    assert reference is not None
    result = refine_whole_pattern(
        x,
        y,
        [WholePatternPhaseSpec(reference, refine_cell=False)],
        mode="Pawley decomposition",
        wavelength_angstrom=WAVELENGTH,
        weighting="Poisson-like",
        observed_sigma=sigma,
        intensity_provenance="raw_counts",
        refine_zero_shift=False,
        refine_profile=False,
        extraction_cycles=3,
    )

    assert min(result["background_y"]) >= -1e-10


def test_rietveld_can_freeze_cif_structure_factors(nacl_raw_frame, nacl_structure):
    x, y, sigma = _arrays(nacl_raw_frame)
    result = refine_rietveld(
        x,
        y,
        [
            RietveldPhaseSpec(
                nacl_structure,
                name="NaCl",
                refine_cell=False,
                freeze_structure_factors=True,
            )
        ],
        wavelength_angstrom=WAVELENGTH,
        weighting="Poisson-like",
        observed_sigma=sigma,
        intensity_provenance="raw_counts",
        refine_zero_shift=False,
        refine_profile=False,
        maximum_nonlinear_evaluations=10,
    )

    assert result["success"]
    assert result["phases"][0]["structure_factors_frozen"] is True
    assert min(result["background_y"]) >= -1e-10
    assert any("frozen CIF-derived structure factors" in row for row in result["warnings"])


def test_rietveld_reports_held_out_validation_statistics(
    nacl_raw_frame, nacl_structure
):
    x, y, sigma = _arrays(nacl_raw_frame)
    result = refine_rietveld(
        x,
        y,
        [
            RietveldPhaseSpec(
                nacl_structure,
                name="NaCl",
                refine_cell=False,
                freeze_structure_factors=True,
            )
        ],
        wavelength_angstrom=WAVELENGTH,
        weighting="Poisson-like",
        observed_sigma=sigma,
        intensity_provenance="raw_counts",
        refine_zero_shift=False,
        refine_profile=False,
        maximum_nonlinear_evaluations=10,
        validation_stride=5,
    )

    validation = result["cross_validation"]
    assert validation["enabled"] is True
    assert validation["stride"] == 5
    assert validation["training_point_count"] > validation["validation_point_count"] > 0
    assert np.isfinite(validation["rwp_percent"])
    assert np.isfinite(validation["rp_percent"])
    assert np.isfinite(validation["rmse"])


def _arrays(frame):
    return (
        frame["two_theta_deg"].to_numpy(dtype=float),
        frame["intensity_counts"].to_numpy(dtype=float),
        frame["sigma_counts"].to_numpy(dtype=float),
    )


@pytest.mark.slow
def test_single_phase_rietveld_preserves_phase0_baseline(nacl_raw_frame, nacl_structure):
    x, y, sigma = _arrays(nacl_raw_frame)
    result = refine_rietveld(
        x,
        y,
        [RietveldPhaseSpec(nacl_structure, name="NaCl", refine_cell=True)],
        wavelength_angstrom=WAVELENGTH,
        weighting="Poisson-like",
        observed_sigma=sigma,
        intensity_provenance="raw_counts",
        maximum_nonlinear_evaluations=120,
        maximum_optimization_points=1800,
    )

    assert result["success"]
    assert result["rwp_percent"] == pytest.approx(10.1676404408, abs=0.05)
    assert result["rexp_percent"] == pytest.approx(5.9682850037, abs=0.05)
    assert len(result["phases"]) == 1
    snapshot = result["phases"][0]["structure_snapshot"]
    assert snapshot["atoms"] == nacl_structure["atoms"]
    assert snapshot["atoms"] is not nacl_structure["atoms"]
    assert snapshot["cell"] is not nacl_structure["cell"]


@pytest.mark.slow
def test_two_phase_rietveld_preserves_phase0_baseline(
    mixed_raw_frame, nacl_structure, cscl_structure
):
    x, y, sigma = _arrays(mixed_raw_frame)
    result = refine_rietveld(
        x,
        y,
        [
            RietveldPhaseSpec(nacl_structure, name="NaCl", refine_cell=True),
            RietveldPhaseSpec(cscl_structure, name="CsCl", refine_cell=True),
        ],
        wavelength_angstrom=WAVELENGTH,
        weighting="Poisson-like",
        observed_sigma=sigma,
        intensity_provenance="raw_counts",
        maximum_nonlinear_evaluations=120,
        maximum_optimization_points=1800,
    )

    assert result["success"]
    assert result["rwp_percent"] == pytest.approx(9.1871733347, abs=0.05)
    assert result["rexp_percent"] == pytest.approx(5.9382224912, abs=0.05)
    assert {row["phase_name"] for row in result["phases"]} == {"NaCl", "CsCl"}


@pytest.mark.slow
def test_pawley_decomposition_regression(nacl_raw_frame, nacl_structure):
    x, y, sigma = _arrays(nacl_raw_frame)
    pattern = calculate_powder_pattern(nacl_structure, WAVELENGTH, 4.0, 80.0)
    reference = reference_from_cif_pattern(nacl_structure, pattern, WAVELENGTH)
    assert reference is not None
    result = refine_whole_pattern(
        x,
        y,
        [WholePatternPhaseSpec(reference, refine_cell=True)],
        mode="Pawley decomposition",
        wavelength_angstrom=WAVELENGTH,
        weighting="Poisson-like",
        observed_sigma=sigma,
        intensity_provenance="raw_counts",
        extraction_cycles=6,
        maximum_nonlinear_evaluations=40,
        maximum_optimization_points=1200,
    )

    assert result["success"]
    assert result["rwp_percent"] == pytest.approx(6.7948477025, abs=0.10)
    assert result["rexp_percent"] == pytest.approx(5.9627723028, abs=0.10)


def test_hill_howard_qpa_is_closed_and_explicitly_not_publication_ready(
    nacl_structure, cscl_structure
):
    result = compute_hill_howard_qpa(
        [
            {"phase_name": "NaCl", "scale_factor": 1.0},
            {"phase_name": "CsCl", "scale_factor": 1.0},
        ],
        [nacl_structure, cscl_structure],
        scale_standard_errors=[0.01, 0.01],
    )
    fractions = [row["crystalline_weight_percent"] for row in result["phases"]]
    assert sum(fractions) == pytest.approx(100.0, abs=1e-10)
    assert fractions == pytest.approx([78.0439411775, 21.9560588225], abs=1e-6)
    assert result["publication_ready"] is False
    assert result["uncertainty_method"] == "Independent scale-error propagation"


def test_hill_howard_requires_two_phases(nacl_structure):
    with pytest.raises(QPAValidationError, match="At least two"):
        compute_hill_howard_qpa(
            [{"phase_name": "NaCl", "scale_factor": 1.0}], [nacl_structure]
        )
