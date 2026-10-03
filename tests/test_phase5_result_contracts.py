from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from afruz_pxrd.application.project_controller import ProjectController
from afruz_pxrd.application.state import ProjectState
from afruz_pxrd.contracts import (
    ContractValidationError,
    PreprocessingResult,
    ResultContractService,
    ResultContractStore,
)
from afruz_pxrd.export_engine import build_export_package
from afruz_pxrd.models import Dataset
from afruz_pxrd.project import save_afz


ENVELOPE_FIELDS = {
    "kind",
    "schema_version",
    "result_id",
    "dataset_id",
    "engine",
    "input_signature",
    "parameters",
    "numerical_outputs",
    "uncertainties",
    "warnings",
    "validation_status",
    "created_at",
    "provenance",
}


def _dataset(uid: str = "dataset-1") -> Dataset:
    return Dataset(
        uid=uid,
        name="NaCl",
        x=np.array([20.0, 21.0, 22.0, 23.0]),
        y_raw=np.array([10.0, 30.0, 20.0, 12.0]),
        y_processed=np.array([9.0, 28.0, 18.0, 10.0]),
        metadata={"wavelength_angstrom": 1.5406},
    )


def _all_payloads() -> dict:
    x = np.array([20.0, 21.0, 22.0, 23.0])
    observed = np.array([9.0, 28.0, 18.0, 10.0])
    calculated = np.array([9.5, 27.5, 18.5, 9.5])
    return {
        "preprocessing": {
            "processed": observed,
            "background": {"background": np.ones(4), "settings": {"method": "ALS"}},
            "smoothing": {"smoothed": observed, "settings": {"window": 5}},
        },
        "peak_list": {
            "rows": [{"position": 21.0, "intensity": 28.0, "fwhm": 0.18}],
            "meta": {"locked": True},
        },
        "peak_fitting": {
            "groups": [{"center": 21.0, "height": 28.0}],
            "candidates": [{"model": "Pseudo-Voigt", "bic": 3.5}],
        },
        "instrument_calibration": {
            "profile": {"u": 0.01, "v": 0.0, "w": 0.02},
            "qa_result": {"status": "passed"},
            "engine_version": "1.0",
        },
        "unit_cell_refinement": {
            "initial_cell": {"a": 5.63},
            "refined_cell": {"a": 5.64},
            "matches": [{"h": 1, "k": 1, "l": 1}],
            "rmse": 0.01,
        },
        "phase_identification": {
            "results": [{"name": "NaCl", "score": 99.0}],
            "best": {"name": "NaCl", "matches": [{"position": 21.0}]},
        },
        "whole_pattern_refinement": {
            "observed_x": x,
            "observed_y": observed,
            "calculated_y": calculated,
            "background_y": np.ones(4),
            "rwp_percent": 4.2,
        },
        "rietveld_refinement": {
            "observed_x": x,
            "observed_y": observed,
            "calculated_y": calculated,
            "background_y": np.ones(4),
            "phases": [{"phase_name": "NaCl", "weight_fraction_percent": 100.0}],
            "rwp_percent": 3.8,
        },
        "qpa": {
            "validated": {
                "mode": "Rietveld",
                "phases": [{"phase_name": "NaCl", "weight_fraction_percent": 100.0}],
                "rwp_percent": 3.8,
            }
        },
        "validation": {
            "rows": [{"dataset_uid": "dataset-1", "status": "passed"}],
            "summary": {"publication_ready": True},
        },
    }


def test_all_ten_contracts_round_trip_with_complete_envelopes():
    store = ResultContractStore()
    ResultContractService.capture_dataset(store, "dataset-1", _all_payloads())

    assert set(store.for_dataset("dataset-1")) == set(ResultContractService.RESULT_KINDS)
    serialized = store.to_dict()
    for contract in serialized["results"].values():
        payload = contract["dataset-1"]
        assert set(payload) == ENVELOPE_FIELDS
        assert payload["dataset_id"] == "dataset-1"
        assert payload["engine"]["name"]
        assert payload["engine"]["version"]

    restored = ResultContractStore.from_dict(serialized)
    preprocessing = restored.get("preprocessing", "dataset-1")
    assert isinstance(preprocessing, PreprocessingResult)
    assert np.array_equal(preprocessing.processed_profile, np.array([9.0, 28.0, 18.0, 10.0]))
    assert restored.get("qpa", "dataset-1").fractions == {"NaCl": 100.0}


def test_missing_fields_and_bad_arrays_fail_at_the_contract_boundary():
    store = ResultContractStore()
    ResultContractService.capture_dataset(store, "dataset-1", _all_payloads())
    payload = store.get("preprocessing", "dataset-1").to_dict()
    payload.pop("engine")
    with pytest.raises(ContractValidationError, match="missing fields.*engine"):
        PreprocessingResult.from_dict(payload)

    misaligned = deepcopy(_all_payloads()["preprocessing"])
    misaligned["background"]["background"] = np.ones(3)
    with pytest.raises(ContractValidationError, match="equal lengths"):
        ResultContractService.capture_dataset(
            ResultContractStore(), "dataset-1", {"preprocessing": misaligned}
        )

    non_finite = deepcopy(_all_payloads()["whole_pattern_refinement"])
    non_finite["observed_y"][0] = np.nan
    with pytest.raises(ContractValidationError, match="non-finite"):
        ResultContractService.capture_dataset(
            ResultContractStore(),
            "dataset-1",
            {"whole_pattern_refinement": non_finite},
        )


def test_result_identity_is_stable_until_inputs_change_and_tracks_provenance():
    store = ResultContractStore()
    payload = _all_payloads()["peak_list"]
    store.record_legacy("peak_list", "dataset-1", payload)
    first = store.get("peak_list", "dataset-1")
    store.record_legacy("peak_list", "dataset-1", deepcopy(payload))
    assert store.get("peak_list", "dataset-1") is first

    changed = deepcopy(payload)
    changed["rows"][0]["intensity"] = 29.0
    store.record_legacy("peak_list", "dataset-1", changed)
    second = store.get("peak_list", "dataset-1")
    assert second.result_id != first.result_id
    assert first.result_id in second.provenance.source_result_ids


def test_project_state_duplicates_removes_and_rejects_orphan_contracts():
    source = _dataset()
    state = ProjectState(datasets=[source])
    ResultContractService.capture_dataset(state.result_contracts, source.uid, _all_payloads())
    clone = source.clone()
    state.datasets.append(clone)
    state.duplicate_dataset_payloads(source.uid, clone.uid)
    state.validate_invariants()
    assert set(state.result_contracts.for_dataset(clone.uid)) == set(ResultContractService.RESULT_KINDS)
    assert state.result_contracts.get("qpa", clone.uid).result_id != state.result_contracts.get("qpa", source.uid).result_id

    state.datasets.remove(source)
    state.remove_dataset_payloads(source.uid)
    state.validate_invariants()
    assert state.result_contracts.for_dataset(source.uid) == {}


def test_legacy_afz_is_migrated_by_gui_independent_project_controller(tmp_path):
    dataset = _dataset()
    payloads = _all_payloads()
    analysis = {
        "background_results": {dataset.uid: payloads["preprocessing"]["background"]},
        "smoothing_results": {dataset.uid: payloads["preprocessing"]["smoothing"]},
        "peak_rows": {dataset.uid: payloads["peak_list"]["rows"]},
        "peak_list_meta_by_uid": {dataset.uid: payloads["peak_list"]["meta"]},
        "fit_groups": {dataset.uid: payloads["peak_fitting"]["groups"]},
        "fit_candidates": {dataset.uid: payloads["peak_fitting"]["candidates"]},
        "cell_refinement_results": {dataset.uid: payloads["unit_cell_refinement"]},
        "phase_identification_results": {dataset.uid: payloads["phase_identification"]},
        "qpa_results": {dataset.uid: payloads["qpa"]["validated"]},
        "instrument_calibration": {**payloads["instrument_calibration"], "reference_uid": dataset.uid},
        "whole_pattern_refinement": {"results_by_uid": {dataset.uid: payloads["whole_pattern_refinement"]}},
        "rietveld_refinement": {"results_by_uid": {dataset.uid: payloads["rietveld_refinement"]}},
        "validation_campaign": {
            "audit_results": [{"dataset_uid": dataset.uid, "status": "passed"}]
        },
    }
    path = tmp_path / "legacy.afz"
    save_afz(path, [dataset], {}, analysis)

    controller = ProjectController()
    opened = controller.open(path)
    assert "typed_result_contracts" not in opened.analysis_state
    assert set(controller.state.result_contracts.for_dataset(dataset.uid)) == set(
        ResultContractService.RESULT_KINDS
    )
    assert np.array_equal(controller.state.datasets[0].y_processed, dataset.y_processed)
    assert opened.analysis_state["peak_rows"] == analysis["peak_rows"]


def test_export_engine_prefers_documented_contracts_over_legacy_results():
    dataset = _dataset()
    store = ResultContractStore()
    ResultContractService.capture_dataset(store, dataset.uid, _all_payloads())
    snapshot = {
        "project_name": "Typed export",
        "datasets": [{
            "uid": dataset.uid,
            "name": dataset.name,
            "x": dataset.x,
            "y_raw": dataset.y_raw,
            "y_processed": dataset.y_processed,
            "metadata": dataset.metadata,
            "analyses": {
                "result_contracts": ResultContractService.export_dataset(store, dataset.uid),
                "cell_refinement": {"legacy_only_marker": 123},
            },
        }],
    }
    package = build_export_package(snapshot)
    titles = [table.title for table in package.tables]
    assert any("Unit-cell refinement result contract" in title for title in titles)
    assert not any(title.startswith("Cell refinement") for title in titles)
    refinement_profiles = [
        table
        for table in package.tables
        if table.category == "refinement" and "profile" in table.title.lower()
    ]
    assert len(refinement_profiles) == 2
    assert all(len(table.rows) == 4 for table in refinement_profiles)
    parameters = {
        row.get("parameter")
        for table in package.tables
        for row in table.rows
        if isinstance(row, dict)
    }
    assert "legacy_only_marker" not in parameters


@pytest.mark.gui
def test_live_window_sync_captures_every_primary_result_family(qtbot):
    from afruz_pxrd.app import MainWindow

    dataset = _dataset()
    payloads = _all_payloads()
    window = MainWindow()
    qtbot.addWidget(window)
    window.project_controller.add_datasets([dataset])
    window.refresh_dataset_list()
    uid = dataset.uid
    window.background_results[uid] = payloads["preprocessing"]["background"]
    window.smoothing_results[uid] = payloads["preprocessing"]["smoothing"]
    window.peak_rows[uid] = payloads["peak_list"]["rows"]
    window.peak_list_meta_by_uid[uid] = payloads["peak_list"]["meta"]
    window.fit_groups[uid] = payloads["peak_fitting"]["groups"]
    window.fit_candidates[uid] = payloads["peak_fitting"]["candidates"]
    window.cell_refinement_results[uid] = payloads["unit_cell_refinement"]
    window.phase_identification_results[uid] = payloads["phase_identification"]
    window.qpa_results[uid] = payloads["qpa"]["validated"]
    window.calibration_widget.profile = payloads["instrument_calibration"]["profile"]
    window.calibration_widget.qa_result = payloads["instrument_calibration"]["qa_result"]
    index = window.calibration_widget.reference_selector.findData(uid)
    if index >= 0:
        window.calibration_widget.reference_selector.setCurrentIndex(index)
    window.whole_pattern_widget.results_by_uid[uid] = payloads["whole_pattern_refinement"]
    window.rietveld_widget.results_by_uid[uid] = payloads["rietveld_refinement"]
    window.validation_widget.audit_results = [{"dataset_uid": uid, "status": "passed"}]

    window._sync_unified_scientific_state(uid)
    assert set(window.project_state.result_contracts.for_dataset(uid)) == set(
        ResultContractService.RESULT_KINDS
    )
    snapshot = window.build_export_snapshot(all_datasets=True)
    assert set(snapshot["datasets"][0]["analyses"]["result_contracts"]) == set(
        ResultContractService.RESULT_KINDS
    )
    assert "typed_result_contracts" in window._analysis_state()
    window.close()
