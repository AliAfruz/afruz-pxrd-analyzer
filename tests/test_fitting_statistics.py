from __future__ import annotations

import numpy as np
import pytest

from afruz_pxrd.fitting import fit_detected_peaks
from afruz_pxrd.refinement_statistics import (
    build_refinement_weight_model,
    calculate_profile_statistics,
    infer_dataset_statistical_input,
)


def test_gaussian_peak_fit_recovers_known_parameters():
    x = np.linspace(28.0, 32.0, 801)
    y = 5.0 + 100.0 * np.exp(-4.0 * np.log(2.0) * ((x - 30.0) / 0.24) ** 2)
    groups, summary = fit_detected_peaks(
        x,
        y,
        [{"position": 30.01, "fwhm": 0.25, "method": "Regular"}],
        model="Gaussian",
    )

    component = groups[0]["components"][0]
    assert summary["failed_groups"] == 0
    assert groups[0]["r_squared"] == pytest.approx(1.0, abs=1e-10)
    assert component["center"] == pytest.approx(30.0, abs=1e-6)
    assert component["fwhm"] == pytest.approx(0.24, abs=1e-6)
    assert component["amplitude"] == pytest.approx(100.0, rel=1e-6)


def test_profile_statistics_preserve_phase0_reference_values():
    observed = np.array([100.0, 81.0, 64.0, 49.0])
    calculated = np.array([98.0, 82.0, 63.0, 50.0])
    weights = build_refinement_weight_model(
        observed, "Poisson-like", provenance="raw_counts"
    )
    result = calculate_profile_statistics(observed, calculated, weights, parameter_count=2)

    assert weights.statistics_valid
    assert weights.model_name == "Poisson counting statistics"
    assert result["rwp_percent"] == pytest.approx(1.7338059495, abs=1e-4)
    assert result["rp_percent"] == pytest.approx(1.7006802721, abs=1e-4)
    assert result["rexp_percent"] == pytest.approx(8.2478609884, abs=1e-4)
    assert result["reduced_chi_square"] == pytest.approx(0.0441894211, abs=1e-6)


def test_normalized_data_does_not_claim_counting_statistics(
    nacl_raw_frame, nacl_normalized_frame
):
    inferred = infer_dataset_statistical_input(
        nacl_raw_frame["intensity_counts"],
        nacl_normalized_frame["intensity_normalized"],
        intensity_unit="normalized intensity",
        interpretation="Auto from input metadata",
    )
    assert inferred["intensity_provenance"] == "arbitrary_units"
    assert inferred["statistics_expected_valid"] is False


def test_observed_sigma_length_must_match_pattern():
    with pytest.raises(ValueError, match="matching the observed pattern"):
        build_refinement_weight_model(
            np.array([10.0, 20.0, 30.0]),
            "Poisson-like",
            observed_sigma=np.array([1.0, 2.0]),
        )
