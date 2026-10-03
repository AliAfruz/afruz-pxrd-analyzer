from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt

from afruz_pxrd.doping_series import (
    DopingSeriesError,
    compare_doping_series,
    export_doping_series_bundle,
    refine_doping_series,
    refined_cif_text,
)
from afruz_pxrd.doping_series_widget import DopingSeriesWidget
from afruz_pxrd.doping_series_worker import DopingSeriesWorker
from afruz_pxrd.models import Dataset
from afruz_pxrd.rietveld_refinement import RietveldPhaseSpec


def _record(uid: str, value: float, *, start=10.0, stop=20.0, step=0.02, shift=0.0):
    x = np.arange(start, stop + step / 2.0, step)
    y = 5.0 + 100.0 * np.exp(-0.5 * ((x - (15.0 + shift)) / 0.08) ** 2)
    return {
        "dataset_uid": uid,
        "dataset_name": f"sample-{value:g}",
        "dopant": "Al",
        "series_value": value,
        "series_unit": "at.%",
        "intensity_provenance": "raw_counts",
        "x": x,
        "y": y,
        "peaks": [
            {
                "position": 15.0 + shift,
                "position_error": 0.004,
                "intensity": 105.0,
                "fwhm": 0.19,
                "included": True,
            }
        ],
    }


def _structure():
    raw = """data_demo
_cell_length_a 5.000
_cell_length_b 5.000
_cell_length_c 5.000
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 0 0 0
"""
    return {
        "name": "demo",
        "raw_cif_text": raw,
        "cell": {
            "a": 5.0,
            "b": 5.0,
            "c": 5.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        "atoms": [],
    }


def _fake_result(cell_a: float, *, success=True, rwp=5.0):
    return {
        "success": success,
        "rwp_percent": rwp,
        "reduced_chi_square": 1.2,
        "maximum_absolute_correlation": 0.25,
        "zero_shift_deg": 0.01 * cell_a,
        "profile": {
            "caglioti_u": 0.01,
            "caglioti_v": -0.01,
            "caglioti_w": 0.03 + cell_a / 1000.0,
            "eta": 0.5,
            "lorentzian_x": 0.0,
            "lorentzian_y": 0.0,
            "axial_asymmetry": 0.0,
        },
        "phases": [
            {
                "refined_cell": {
                    "a": cell_a,
                    "b": cell_a,
                    "c": cell_a,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gamma": 90.0,
                },
                "march_dollase_r": 1.0,
                "delta_biso": 0.0,
                "pattern_scale_fraction_percent": 100.0,
            }
        ],
        "observed_x": [10.0, 11.0],
        "observed_y": [5.0, 6.0],
        "calculated_y": [5.1, 5.9],
        "background_y": [1.0, 1.0],
        "difference_y": [-0.1, 0.1],
        "elapsed_seconds": 0.01,
    }


def test_comparison_uses_only_intersection_and_coarsest_step():
    fine = _record("fine", 0.0, start=10.0, stop=20.0, step=0.01)
    coarse = _record("coarse", 1.0, start=11.0, stop=19.0, step=0.025, shift=0.04)
    fine_x_before = fine["x"].copy()
    coarse_x_before = coarse["x"].copy()

    result = compare_doping_series([fine, coarse], reference_uid="fine")

    grid = np.asarray(result["common_two_theta_deg"])
    assert grid[0] >= 11.0
    assert grid[-1] <= 19.0 + 1e-12
    assert np.median(np.diff(grid)) == pytest.approx(0.025)
    assert np.array_equal(fine["x"], fine_x_before)
    assert np.array_equal(coarse["x"], coarse_x_before)
    reference = next(row for row in result["profiles"] if row["dataset_uid"] == "fine")
    assert np.max(np.abs(reference["difference_y"])) == pytest.approx(0.0)
    assert "visualization" in result["scientific_boundary"]


def test_peak_tracking_reports_reference_shifts_and_emergent_tracks():
    reference = _record("u0", 0.0)
    doped = _record("u1", 1.0, shift=0.06)
    doped["peaks"].append(
        {"position": 17.0, "position_error": 0.005, "intensity": 20.0, "included": True}
    )

    result = compare_doping_series(
        [reference, doped], reference_uid="u0", peak_match_tolerance_deg=0.10
    )

    shifted = next(
        row
        for row in result["peak_observations"]
        if row["dataset_uid"] == "u1" and row["reference_classification"] == "matched to reference"
    )
    emergent = next(
        row
        for row in result["peak_observations"]
        if row["dataset_uid"] == "u1" and row["reference_classification"] == "emergent relative to reference"
    )
    assert shifted["position_shift_deg"] == pytest.approx(0.06)
    assert shifted["position_standard_error_deg"] == pytest.approx(0.004)
    assert shifted["d_spacing_shift_angstrom"] is not None
    assert emergent["position_deg"] == pytest.approx(17.0)


def test_noisy_automatic_fallback_uses_main_smart_peak_search():
    rng = np.random.default_rng(20260811)
    x = np.arange(10.0, 80.0, 0.02)

    def noisy_pattern(uid, value, shift):
        y = 180.0 + 0.7 * (x - 10.0) + rng.normal(0.0, 3.0, len(x))
        for center, height, width in (
            (22.0, 150.0, 0.10),
            (38.0, 90.0, 0.14),
            (52.0, 130.0, 0.18),
            (67.0, 70.0, 0.12),
        ):
            y += height * np.exp(-0.5 * ((x - center - shift) / width) ** 2)
        return {
            "dataset_uid": uid,
            "dataset_name": uid,
            "series_value": value,
            "x": x,
            "y": y,
            "peaks": [],
        }

    result = compare_doping_series(
        [noisy_pattern("reference", 0.0, 0.0), noisy_pattern("doped", 1.0, 0.04)],
        reference_uid="reference",
        peak_detection_sensitivity="Balanced",
    )

    assert [profile["peak_count"] for profile in result["profiles"]] == [4, 4]
    assert result["peak_track_count"] == 4
    assert all(
        "Main Smart peak search" in row["source"]
        for row in result["peak_observations"]
    )


def test_sequential_refinement_runs_reference_outward_and_keeps_branches_independent(monkeypatch):
    calls = []

    def fake_refine(x, y, specs, **settings):
        calls.append(
            {
                "cell_a": specs[0].structure["cell"]["a"],
                "initial_zero_shift": settings.get("initial_zero_shift", 0.0),
            }
        )
        return _fake_result(5.1 + 0.1 * len(calls))

    monkeypatch.setattr("afruz_pxrd.doping_series.refine_rietveld", fake_refine)
    records = [_record("u0", 0.0), _record("u1", 1.0), _record("u2", 2.0)]
    phase = RietveldPhaseSpec(_structure(), name="demo")

    result = refine_doping_series(
        records,
        [phase],
        {},
        mode="Sequential from reference",
        reference_uid="u1",
    )

    assert result["execution_order_uids"] == ["u1", "u2", "u0"]
    assert calls[0]["cell_a"] == pytest.approx(5.0)
    assert calls[1]["cell_a"] == pytest.approx(5.2)
    # The lower-concentration branch restarts from the accepted reference,
    # rather than crossing through the high-concentration branch.
    assert calls[2]["cell_a"] == pytest.approx(5.2)
    assert calls[1]["initial_zero_shift"] == pytest.approx(0.052)
    assert result["completed_count"] == 3
    assert not result["failures"]


def test_failed_or_rejected_refinement_is_isolated_and_not_propagated(monkeypatch):
    call_count = 0
    starts = []
    progress_messages = []

    def fake_refine(x, y, specs, **settings):
        nonlocal call_count
        call_count += 1
        starts.append(specs[0].structure["cell"]["a"])
        if call_count == 2:
            raise RuntimeError("synthetic failure")
        return _fake_result(5.0 + 0.1 * call_count)

    monkeypatch.setattr("afruz_pxrd.doping_series.refine_rietveld", fake_refine)
    result = refine_doping_series(
        [_record("u0", 0.0), _record("u1", 1.0), _record("u2", 2.0)],
        [RietveldPhaseSpec(_structure())],
        {},
        mode="Sequential from reference",
        reference_uid="u0",
        progress_callback=lambda _done, _total, message: progress_messages.append(
            message
        ),
    )

    assert result["completed_count"] == 2
    assert result["failed_count"] == 1
    assert result["failures"][0]["dataset_uid"] == "u1"
    assert starts == pytest.approx([5.0, 5.1, 5.1])
    assert any("Failed sample-1; continuing" in message for message in progress_messages)


def test_series_refinement_maps_inner_optimizer_progress_monotonically(monkeypatch):
    events = []
    call_count = 0

    def fake_refine(x, y, specs, **settings):
        nonlocal call_count
        call_count += 1
        callback = settings["progress_callback"]
        callback(0, 4, "Scaling generated pattern")
        callback(2, 4, "Rietveld evaluation 2/4")
        callback(4, 4, "Rietveld refinement completed")
        return _fake_result(5.0 + 0.1 * call_count)

    monkeypatch.setattr("afruz_pxrd.doping_series.refine_rietveld", fake_refine)
    refine_doping_series(
        [_record("u0", 0.0), _record("u1", 1.0)],
        [RietveldPhaseSpec(_structure())],
        {},
        mode="Sequential from reference",
        reference_uid="u0",
        progress_callback=lambda done, total, message: events.append(
            (done, total, message)
        ),
    )

    done_values = [row[0] for row in events]
    assert done_values == sorted(done_values)
    assert {row[1] for row in events} == {2000}
    assert done_values[0] == 0
    assert done_values[-1] == 2000
    assert any(done == 475 and "sample-0" in message for done, _, message in events)
    assert any(done == 1475 and "sample-1" in message for done, _, message in events)
    assert any(
        done == 1000 and "refinement completed" in message.lower()
        for done, _, message in events
    )


def test_refinement_uses_original_intensities_when_comparison_is_prepared(monkeypatch):
    captured = {}

    def fake_refine(x, y, specs, **settings):
        if not captured:
            captured["y"] = np.asarray(y).copy()
            captured["count_reference"] = np.asarray(settings["count_reference"]).copy()
            captured["provenance"] = settings["intensity_provenance"]
        return _fake_result(5.1)

    monkeypatch.setattr("afruz_pxrd.doping_series.refine_rietveld", fake_refine)
    record = _record("u0", 0.0)
    raw = np.asarray(record["y"]) * 11.0
    record["y"] = np.asarray(record["y"]) / np.max(record["y"]) * 100.0
    record["refinement_y"] = raw
    record["intensity_provenance"] = "processed_for_comparison"
    record["refinement_intensity_provenance"] = "raw_counts"

    refine_doping_series(
        [record, _record("u1", 1.0)],
        [RietveldPhaseSpec(_structure())],
        {},
        mode="Independent batch",
        reference_uid="u0",
    )

    assert np.array_equal(captured["y"], raw)
    assert np.array_equal(captured["count_reference"], raw)
    assert captured["provenance"] == "raw_counts"


def test_worker_reserves_comparison_progress_and_scales_refinements(monkeypatch):
    events = []
    finished = []

    monkeypatch.setattr(
        "afruz_pxrd.doping_series_worker.compare_doping_series",
        lambda records, **settings: {"dataset_count": len(records)},
    )

    def fake_series_refinement(records, phases, settings, **kwargs):
        callback = kwargs["progress_callback"]
        callback(0, 2000, "Preparing first")
        callback(1000, 2000, "Completed first")
        callback(2000, 2000, "Completed second")
        return {"completed_count": 2}

    monkeypatch.setattr(
        "afruz_pxrd.doping_series_worker.refine_doping_series",
        fake_series_refinement,
    )
    worker = DopingSeriesWorker([{}, {}], {}, [object()], {}, {})
    worker.progress.connect(
        lambda done, total, message: events.append((done, total, message))
    )
    worker.finished.connect(finished.append)

    worker.run()

    done_values = [row[0] for row in events]
    assert done_values == sorted(done_values)
    assert {row[1] for row in events} == {DopingSeriesWorker.PROGRESS_MAXIMUM}
    assert done_values == [0, 50, 50, 515, 980, 990]
    assert finished and finished[0]["refinement"]["completed_count"] == 2


def test_refined_cif_and_export_bundle_preserve_coordinates_and_create_manifest(tmp_path):
    structure = _structure()
    phase_result = _fake_result(5.25)["phases"][0]
    cif = refined_cif_text(
        structure,
        phase_result,
        dataset_name="sample-1",
        software_version="23.0.0",
    )
    assert "_cell_length_a 5.25" in cif
    assert "Na1 0 0 0" in cif
    assert "Atomic coordinates and occupancies were retained" in cif
    assert "_afruz_structure_plausibility_status" in cif
    assert "_afruz_publication_ready no" in cif

    comparison = compare_doping_series([_record("u0", 0.0), _record("u1", 1.0)])
    pattern = _fake_result(5.25)
    pattern.update(
        {
            "series_dataset_name": "sample-1",
            "series_value": 1.0,
            "series_dataset_uid": "u1",
            "series_quality_gate": {"passed": True, "status": "accepted", "reasons": []},
        }
    )
    bundle = export_doping_series_bundle(
        tmp_path,
        {
            "comparison": comparison,
            "refinement": {
                "results_by_uid": {"u1": pattern},
                "trends": [],
                "completed_count": 1,
                "failed_count": 0,
            },
        },
        [RietveldPhaseSpec(structure)],
        software_version="23.0.0",
    )

    assert Path(bundle["manifest_txt"]).is_file()
    refined_cifs = list((tmp_path / "refined_cifs").glob("*.cif"))
    assert refined_cifs
    assert "REVIEW_REQUIRED" in refined_cifs[0].name
    assert (tmp_path / "phase12_peak_tracks.txt").is_file()
    assert (tmp_path / "phase12_difference_profiles.txt").is_file()


def test_comparison_rejects_duplicate_uids_and_missing_overlap():
    first = _record("same", 0.0, start=10.0, stop=15.0)
    second = _record("same", 1.0, start=20.0, stop=25.0)
    with pytest.raises(DopingSeriesError, match="unique UID"):
        compare_doping_series([first, second])
    second["dataset_uid"] = "other"
    with pytest.raises(DopingSeriesError, match="no common"):
        compare_doping_series([first, second])


@pytest.mark.gui
def test_widget_round_trip_preserves_cif_inclusion_and_refinement_choices(qtbot):
    owner = SimpleNamespace(datasets=[])
    original = DopingSeriesWidget(owner)
    qtbot.addWidget(original)
    original._add_structure(_structure())
    original.cif_table.item(0, 0).setCheckState(Qt.Unchecked)
    original.cif_table.item(0, 3).setCheckState(Qt.Unchecked)
    original.cif_table.item(0, 4).setCheckState(Qt.Checked)

    restored = DopingSeriesWidget(owner)
    qtbot.addWidget(restored)
    restored.set_state(original.get_state())

    assert restored.cif_table.item(0, 0).checkState() == Qt.Unchecked
    assert restored.cif_table.item(0, 3).checkState() == Qt.Unchecked
    assert restored.cif_table.item(0, 4).checkState() == Qt.Checked


@pytest.mark.gui
def test_widget_uses_scrollable_workspace_and_roomy_setup_tabs(qtbot):
    owner = SimpleNamespace(datasets=[])
    widget = DopingSeriesWidget(owner)
    qtbot.addWidget(widget)
    widget.resize(760, 540)
    widget.show()
    qtbot.wait(50)

    assert widget.workspace_scroll.widgetResizable()
    assert widget.workspace_scroll.verticalScrollBar().maximum() > 0
    assert widget.setup_tabs.count() == 3
    assert widget.setup_tabs.tabText(0) == "1. Series Datasets"
    assert widget.setup_tabs.tabText(1) == "2. CIF Models"
    assert widget.setup_tabs.tabText(2) == "3. Refinement Controls"
    assert widget.controls_scroll.widgetResizable()
    widget.setup_tabs.setCurrentIndex(2)
    qtbot.wait(20)
    assert widget.controls_scroll.verticalScrollBar().maximum() > 0
    widget._on_progress(525, 1000, "sample-1 — Rietveld evaluation 20/100")
    assert widget.progress.value() == 525
    assert widget.progress.format() == "Working — %p%"
    assert widget.progress_label.text().startswith("52.5% — sample-1")


@pytest.mark.gui
def test_widget_links_prepared_signal_and_main_peaks_but_preserves_raw_refinement(qtbot):
    x = np.arange(10.0, 20.0, 0.02)
    raw = 20.0 + 200.0 * np.exp(-0.5 * ((x - 15.0) / 0.09) ** 2)
    prepared = np.clip(raw - 20.0, 0.0, None)
    dataset = Dataset(
        name="prepared-sample",
        x=x,
        y_raw=raw,
        y_processed=prepared,
        metadata={
            "intensity_provenance": "raw_counts",
            "processing_provenance": {
                "background_subtracted": True,
                "smoothed": True,
                "normalized": False,
            },
        },
    )
    owner = SimpleNamespace(
        datasets=[dataset],
        fit_groups={},
        peak_rows={
            dataset.uid: [
                {
                    "position": 15.0,
                    "intensity": 200.0,
                    "fwhm": 0.2,
                    "method": "Smart",
                    "use": True,
                }
            ]
        },
    )
    widget = DopingSeriesWidget(owner)
    qtbot.addWidget(widget)

    assert widget.dataset_table.item(0, 5).checkState() == Qt.Checked
    assert "baseline corrected" in widget.dataset_table.item(0, 6).text()
    records = widget._records()
    assert np.array_equal(records[0]["y"], prepared)
    assert np.array_equal(records[0]["refinement_y"], raw)
    assert records[0]["comparison_signal"] == "main prepared pattern"
    assert records[0]["peaks"][0]["source"] == "Smart"


@pytest.mark.gui
def test_linked_smart_search_populates_main_peak_list(qtbot):
    x = np.arange(10.0, 20.0, 0.01)
    raw = 25.0 + 180.0 * np.exp(-0.5 * ((x - 15.0) / 0.10) ** 2)
    dataset = Dataset(name="smart-series", x=x, y_raw=raw)
    owner = SimpleNamespace(datasets=[dataset], fit_groups={}, peak_rows={})
    widget = DopingSeriesWidget(owner)
    qtbot.addWidget(widget)

    widget.run_linked_smart_peak_search()

    rows = owner.peak_rows[dataset.uid]
    assert len(rows) == 1
    assert rows[0]["position"] == pytest.approx(15.0, abs=0.01)
    assert "Main Smart peak search" in rows[0]["source"]
    assert widget.progress.value() == widget.progress.maximum()
