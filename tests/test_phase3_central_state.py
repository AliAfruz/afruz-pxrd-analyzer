from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from afruz_pxrd.application import ApplicationState, ProjectController, ProjectState
from afruz_pxrd.app import MainWindow
from afruz_pxrd.models import Dataset
from afruz_pxrd.project import ProjectFormatError


def _dataset(name: str = "sample", uid: str = "dataset-1") -> Dataset:
    return Dataset(
        name=name,
        uid=uid,
        x=np.array([10.0, 11.0, 12.0, 13.0]),
        y_raw=np.array([100.0, 120.0, 90.0, 80.0]),
        y_processed=np.array([95.0, 115.0, 85.0, 75.0]),
        metadata={"wavelength_angstrom": 1.5406},
    )


def test_project_state_exposes_typed_scientific_views():
    state = ProjectState()
    assert state.datasets == []
    assert state.preprocessing["background_results"] is state.background_results
    assert state.peak_lists is state.peak_rows
    assert state.refinements["cell"] is state.cell_refinement_results
    assert state.qpa_results == {}
    assert state.scientific_state.by_dataset == {}


def test_controller_dataset_crud_copies_and_purges_all_central_payloads():
    state = ProjectState()
    controller = ProjectController(state, ApplicationState())
    source = _dataset()
    controller.add_datasets([source])
    state.backgrounds[source.uid] = np.array([1.0, 2.0])
    state.background_results[source.uid] = {"method": "test"}
    state.smoothing_results[source.uid] = {"y_smoothed": np.array([3.0, 4.0])}
    state.peak_rows[source.uid] = [{"position": 11.0}]
    state.cell_refinement_results[source.uid] = {"a": 5.64}
    state.qpa_results[source.uid] = {"NaCl": 100.0}
    state.scientific_state.ensure(source.uid).record_result(
        "preparation",
        {"method": "test"},
        reason="unit test",
    )

    controller.rename_dataset(source.uid, "renamed")
    clone = controller.duplicate_dataset(source.uid)

    assert source.name == "renamed"
    assert clone.uid != source.uid
    assert np.array_equal(state.backgrounds[clone.uid], state.backgrounds[source.uid])
    assert state.backgrounds[clone.uid] is not state.backgrounds[source.uid]
    assert state.peak_rows[clone.uid] == state.peak_rows[source.uid]
    assert state.peak_rows[clone.uid] is not state.peak_rows[source.uid]
    assert state.qpa_results[clone.uid] == {"NaCl": 100.0}
    assert clone.uid in state.scientific_state.by_dataset

    removed = controller.remove_dataset(source.uid)
    assert removed.uid == source.uid
    assert [row.uid for row in state.datasets] == [clone.uid]
    for mapping in (
        state.backgrounds,
        state.background_results,
        state.smoothing_results,
        state.peak_rows,
        state.cell_refinement_results,
        state.qpa_results,
    ):
        assert source.uid not in mapping
    assert source.uid not in state.scientific_state.by_dataset
    state.validate_invariants()


def test_controller_save_and_open_are_gui_independent(tmp_path):
    controller = ProjectController()
    dataset = _dataset()
    controller.add_datasets([dataset])
    controller.state.peak_rows[dataset.uid] = [{"position": 11.0}]
    path = controller.save(
        tmp_path / "central-state",
        {"theme_name": "Dark", "workflow_mode": "Expert"},
        {"peak_rows": controller.state.peak_rows},
    )

    assert path == tmp_path / "central-state.afz"
    assert controller.state.current_project == path
    assert not controller.state.dirty

    reopened = ProjectController()
    opened = reopened.open(path)
    assert opened.path == path
    assert opened.ui_state["theme_name"] == "Dark"
    assert reopened.application_state.theme_name == "Dark"
    assert reopened.application_state.workflow_mode == "Expert"
    assert reopened.application_state.ui_preferences == opened.ui_state
    assert opened.analysis_state["peak_rows"][dataset.uid][0]["position"] == 11.0
    assert reopened.state.datasets[0].uid == dataset.uid
    assert np.array_equal(reopened.state.datasets[0].y_processed, dataset.y_processed)


def test_failed_open_does_not_mutate_live_project(tmp_path):
    controller = ProjectController()
    source = _dataset()
    controller.add_datasets([source])
    controller.state.background_results[source.uid] = {"method": "preserve-me"}
    before = controller.state.snapshot()
    invalid = tmp_path / "invalid.afz"
    invalid.write_bytes(b"not a project")

    with pytest.raises(ProjectFormatError):
        controller.open(invalid)

    after = controller.state.snapshot()
    assert after["datasets"][0].uid == before["datasets"][0].uid
    assert after["background_results"] == before["background_results"]
    assert after["dirty"] == before["dirty"]


def test_controller_opens_version_one_afz_without_gui(tmp_path):
    dataset = _dataset()
    manifest = {
        "project_version": 1,
        "datasets": [{"uid": dataset.uid, "name": dataset.name}],
    }
    array_buffer = io.BytesIO()
    np.savez_compressed(
        array_buffer,
        **{
            f"{dataset.uid}_x": dataset.x,
            f"{dataset.uid}_raw": dataset.y_raw,
            f"{dataset.uid}_processed": dataset.y_processed,
        },
    )
    path = tmp_path / "version-1.afz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(manifest))
        archive.writestr("datasets.npz", array_buffer.getvalue())

    opened = ProjectController().open(path)
    assert opened.analysis_state == {}
    assert opened.ui_state == {}
    assert opened.path == path


def test_central_snapshot_drives_undo_and_redo_without_qt():
    app_state = ApplicationState(theme_name="Dark", active_workspace_key="peak_list")
    controller = ProjectController(application_state=app_state)
    dataset = _dataset()
    controller.add_datasets([dataset])
    controller.mark_clean()

    assert controller.checkpoint("Rename dataset")
    controller.rename_dataset(dataset.uid, "changed")
    assert controller.state.datasets[0].name == "changed"
    assert controller.undo() == "Rename dataset"
    assert controller.state.datasets[0].name == "sample"
    assert controller.application_state.theme_name == "Dark"
    assert controller.redo() == "Rename dataset"
    assert controller.state.datasets[0].name == "changed"


def test_snapshot_restores_project_and_application_state():
    controller = ProjectController(
        application_state=ApplicationState(theme_name="Dark", workflow_mode="Expert")
    )
    controller.add_datasets([_dataset()])
    snapshot = controller.create_snapshot(
        analysis_state={"widget": {"value": 4}},
        ui_state={"active_tab": 2},
        selected_row=0,
    )
    controller.state.datasets[0].name = "mutated"
    controller.application_state.theme_name = "Light"

    restored = controller.restore_snapshot(snapshot)
    assert controller.state.datasets[0].name == "sample"
    assert controller.application_state.theme_name == "Dark"
    assert restored == {
        "analysis_state": {"widget": {"value": 4}},
        "ui_state": {"active_tab": 2},
        "selected_row": 0,
    }


@pytest.mark.gui
def test_main_window_routes_legacy_fields_to_central_state(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.project_controller.state is window.project_state
    assert window.project_controller.application_state is window.application_state
    assert window.datasets is window.project_state.datasets
    assert window.qpa_results is window.project_state.qpa_results
    assert window.scientific_state is window.project_state.scientific_state
    assert window.current_theme_name == window.application_state.theme_name
    for legacy_name in (
        "datasets",
        "background_results",
        "peak_rows",
        "qpa_results",
        "scientific_state",
        "current_theme_name",
    ):
        assert legacy_name not in window.__dict__

    window.project_controller.add_datasets([_dataset()])
    assert window.datasets[0].name == "sample"
    window.close()
