from __future__ import annotations

import numpy as np
import pytest

from afruz_pxrd.cif_simulation_widget import CifSimulationWidget, simulated_profile
from afruz_pxrd.crystallography import calculate_powder_pattern
from afruz_pxrd.exclusion_regions import (
    ExclusionRegion,
    attach_exclusion_audit,
    filter_arrays,
    inclusion_mask,
    merge_regions,
)
from afruz_pxrd.workflow import TASK_BY_KEY
from afruz_pxrd.phase_identification import reference_from_cif_pattern
from afruz_pxrd.rietveld_refinement import RietveldPhaseSpec, refine_rietveld
from afruz_pxrd.whole_pattern_refinement import WholePatternPhaseSpec, refine_whole_pattern


def test_exclusion_regions_are_sorted_merged_and_closed():
    merged = merge_regions(
        [
            (25.0, 24.0, "peak B"),
            {"start_deg": 10.0, "end_deg": 11.0, "label": "peak A"},
            ExclusionRegion(10.5, 12.0, "shoulder"),
        ]
    )
    assert [(row.start_deg, row.end_deg) for row in merged] == [
        (10.0, 12.0),
        (24.0, 25.0),
    ]
    x = np.array([9.9, 10.0, 11.0, 12.0, 12.1, 24.0, 25.0, 25.1])
    assert inclusion_mask(x, merged).tolist() == [True, False, False, False, True, False, False, True]


def test_filter_arrays_preserves_source_and_records_audit():
    x = np.linspace(5.0, 80.0, 751)
    y = np.sin(x) + 10.0
    source_x = x.copy()
    source_y = y.copy()
    filtered_x, [filtered_y, missing], audit = filter_arrays(
        x,
        (y, None),
        [(20.0, 25.0, "unindexed feature"), (40.0, 41.0, "holder")],
    )
    assert missing is None
    assert len(filtered_x) == len(filtered_y) == audit["included_point_count"]
    assert audit["excluded_point_count"] > 0
    assert audit["original_point_count"] == len(x)
    assert np.array_equal(x, source_x)
    assert np.array_equal(y, source_y)
    assert not np.any((filtered_x >= 20.0) & (filtered_x <= 25.0))


def test_filter_arrays_rejects_over_exclusion():
    x = np.linspace(5.0, 10.0, 100)
    with pytest.raises(ValueError, match="at least 50"):
        filter_arrays(x, (np.ones_like(x),), [(5.0, 9.0)])


def test_exclusion_audit_is_explicit_in_result_and_warnings():
    result = attach_exclusion_audit(
        {"warnings": ["existing"], "rwp_percent": 4.2},
        {
            "excluded_regions": [
                {"start_deg": 20.0, "end_deg": 21.0, "width_deg": 1.0, "label": "impurity test"}
            ],
            "original_point_count": 1000,
            "included_point_count": 980,
            "excluded_point_count": 20,
            "included_fraction": 0.98,
        },
        method="Exclusion-aware Pawley refinement",
    )
    assert result["excluded_data_point_count"] == 20
    assert result["exclusion_investigation"]["method"].startswith("Exclusion-aware")
    assert any("does not by itself" in warning for warning in result["warnings"])


def test_cif_simulation_preview_preserves_reflection_maxima():
    pattern = [
        {"two_theta": 10.0, "intensity": 100.0},
        {"two_theta": 20.0, "intensity": 50.0},
    ]
    x, y = simulated_profile(pattern, 5.0, 25.0, fwhm_deg=0.1, eta=0.5, point_count=8001)
    assert len(x) == len(y) == 8001
    assert np.max(y) == pytest.approx(100.0)
    assert x[np.argmax(y)] == pytest.approx(10.0, abs=0.01)
    near_second = y[np.argmin(np.abs(x - 20.0))]
    assert near_second == pytest.approx(50.0, rel=0.02)


@pytest.mark.gui
def test_cif_simulation_panel_calculates_in_background(qtbot, nacl_structure):
    class StatusBar:
        def __init__(self):
            self.message = ""

        def showMessage(self, message, *_args):
            self.message = str(message)

    class Window:
        reference_structure = nacl_structure

        def __init__(self):
            self._status = StatusBar()

        def statusBar(self):
            return self._status

    owner = Window()
    widget = CifSimulationWidget(owner)
    qtbot.addWidget(widget)
    widget.minimum.setValue(4.0)
    widget.maximum.setValue(40.0)
    widget.cutoff.setValue(0.5)
    widget.use_gpu.setChecked(False)
    widget.calculate()
    qtbot.waitUntil(lambda: widget._thread is None, timeout=15_000)

    assert widget.pattern
    assert widget.profile_x.shape == widget.profile_y.shape
    assert widget.table is not None
    assert widget.table.rowCount() == len(widget.pattern)
    assert "reflection group" in owner._status.message


def test_new_tasks_are_additive_and_keep_original_refinements():
    assert "pawley_lebail" in TASK_BY_KEY
    assert "rietveld" in TASK_BY_KEY
    assert TASK_BY_KEY["cif_simulator"].workspace == "phase_structure"
    assert TASK_BY_KEY["exclusion_study"].workspace == "refinement_qpa"


@pytest.mark.parametrize("mode", ["Pawley decomposition", "Le Bail-style extraction"])
def test_filtered_pattern_runs_in_both_whole_pattern_methods(
    mode, nacl_raw_frame, nacl_structure
):
    x = nacl_raw_frame["two_theta_deg"].to_numpy(dtype=float)
    y = nacl_raw_frame["intensity_counts"].to_numpy(dtype=float)
    sigma = nacl_raw_frame["sigma_counts"].to_numpy(dtype=float)
    filtered_x, [filtered_y, filtered_sigma], audit = filter_arrays(
        x,
        (y, sigma),
        [(30.0, 32.0, "deliberate investigation interval")],
    )
    pattern = calculate_powder_pattern(nacl_structure, 1.5406, 4.0, 80.0)
    reference = reference_from_cif_pattern(nacl_structure, pattern, 1.5406)
    result = refine_whole_pattern(
        filtered_x,
        filtered_y,
        [WholePatternPhaseSpec(reference, refine_cell=False)],
        mode=mode,
        wavelength_angstrom=1.5406,
        weighting="Poisson-like",
        observed_sigma=filtered_sigma,
        refine_zero_shift=False,
        refine_profile=False,
        extraction_cycles=3,
    )
    audited = attach_exclusion_audit(result, audit, method=mode)
    assert audited["success"]
    assert audited["excluded_data_point_count"] > 0
    assert not np.any(
        (np.asarray(audited["observed_x"]) >= 30.0)
        & (np.asarray(audited["observed_x"]) <= 32.0)
    )


def test_filtered_pattern_runs_in_rietveld(nacl_raw_frame, nacl_structure):
    x = nacl_raw_frame["two_theta_deg"].to_numpy(dtype=float)
    y = nacl_raw_frame["intensity_counts"].to_numpy(dtype=float)
    sigma = nacl_raw_frame["sigma_counts"].to_numpy(dtype=float)
    filtered_x, [filtered_y, filtered_sigma], audit = filter_arrays(
        x,
        (y, sigma),
        [(30.0, 32.0, "deliberate investigation interval")],
    )
    result = refine_rietveld(
        filtered_x,
        filtered_y,
        [
            RietveldPhaseSpec(
                nacl_structure,
                name="NaCl",
                refine_cell=False,
                freeze_structure_factors=True,
            )
        ],
        wavelength_angstrom=1.5406,
        weighting="Poisson-like",
        observed_sigma=filtered_sigma,
        refine_zero_shift=False,
        refine_profile=False,
        maximum_nonlinear_evaluations=5,
    )
    audited = attach_exclusion_audit(result, audit, method="Rietveld")
    assert audited["success"]
    assert audited["excluded_data_point_count"] > 0
