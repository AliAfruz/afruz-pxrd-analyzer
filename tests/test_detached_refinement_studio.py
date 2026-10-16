from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QScrollArea

from afruz_pxrd.exclusion_refinement_widget import ExclusionRefinementHubWidget
from afruz_pxrd.models import Dataset
from afruz_pxrd.project import load_afz, save_afz
from afruz_pxrd.workflow import build_task_states, build_workflow_status


@pytest.fixture
def studio_factory(qtbot, nacl_raw_frame, nacl_structure, cscl_structure):
    class Owner:
        reference_structure = None
        save_calls = 0

        def __init__(self):
            self.dataset = Dataset(uid="studio-data", name="Studio validation",
                x=nacl_raw_frame["two_theta_deg"].to_numpy().copy(),
                y_raw=nacl_raw_frame["intensity_counts"].to_numpy().copy())

        def selected_dataset(self):
            return self.dataset

        def _mark_project_dirty(self):
            pass

        def save_project(self):
            self.save_calls += 1

    def build():
        owner = Owner()
        hub = ExclusionRefinementHubWidget(owner)
        qtbot.addWidget(hub)
        workspace = hub.sequential_workspace
        for model in (nacl_structure, cscl_structure):
            workspace._add_structure(deepcopy(model), "fixture.cif")
        workspace.use_gpu.setChecked(False)
        workspace.refine_cell.setChecked(False)
        workspace.refine_profile.setChecked(False)
        workspace.refine_zero.setChecked(False)
        workspace.staged.setChecked(False)
        workspace.max_evaluations.setValue(10)
        workspace.background_order.setValue(2)
        workspace.cutoff.setValue(0.5)
        return hub, owner

    return build


@pytest.mark.gui
def test_separate_window_has_one_shared_state_and_reopens_without_duplicates(qtbot, studio_factory):
    hub, owner = studio_factory()
    workspace = hub.sequential_workspace
    assert not workspace.isVisible()
    hub.open_studio_button.click()
    dialog = hub.studio_dialog
    qtbot.addWidget(dialog)
    assert isinstance(dialog, QDialog) and dialog.isWindow() and not dialog.isModal()
    assert workspace.window() is dialog
    assert dialog.refinement_widget is workspace
    assert "23.5.1" in dialog.windowTitle()
    workspace.exclusion_editor.set_regions([(20, 21, "retain this mask")])
    pending = workspace.queue_checked_candidates()
    dialog.close()
    assert not dialog.isVisible()
    assert [record["status"] for record in pending] == ["Queued", "Queued"]
    assert hub.open_studio() is dialog
    assert workspace.exclusion_editor.regions()[0].label == "retain this mask"
    assert len(workspace._history()) == 2
    workspace.save_project_button.click()
    assert owner.save_calls == 1
    dialog.reject()
    assert not dialog.isVisible()
    assert hub.open_studio() is dialog


@pytest.mark.gui
@pytest.mark.parametrize("size", [(1280, 720), (1366, 800), (1920, 1040)])
def test_plot_and_pinned_queue_actions_are_visible_without_overlap(qtbot, studio_factory, size):
    hub, _ = studio_factory()
    dialog = hub.open_studio()
    qtbot.addWidget(dialog)
    dialog.showNormal()
    dialog.resize(*size)
    qtbot.wait(40)
    workspace = hub.sequential_workspace
    assert workspace.plot.width() >= 380 and workspace.plot.height() >= 500
    splitter = workspace.studio_splitter
    left, plot, right = (splitter.widget(index).geometry() for index in range(3))
    assert left.right() < plot.left() and plot.right() < right.left()
    for button in (workspace.queue_checked_button, workspace.start_queue_button,
                   workspace.pause_queue_button, workspace.next_step_button):
        parent = button.parentWidget()
        assert parent.rect().contains(button.geometry())
        assert not isinstance(parent, QScrollArea)
        assert button.isVisible()
    assert workspace.crystal_studio_button.isVisible()
    assert workspace.crystal_studio_button.parentWidget().rect().contains(workspace.crystal_studio_button.geometry())
    for index in range(workspace.detail_tabs.count()):
        workspace.detail_tabs.setCurrentIndex(index)
        qtbot.wait(5)
        assert workspace.plot.height() >= 500
    dialog.close()


@pytest.mark.gui
def test_hiding_active_studio_does_not_cancel_or_duplicate_native_queue(qtbot, studio_factory):
    hub, _ = studio_factory()
    workspace = hub.sequential_workspace
    dialog = hub.open_studio()
    qtbot.addWidget(dialog)
    original_y = workspace._current_dataset().y_raw.copy()
    records = workspace.queue_checked_candidates()
    workspace.start_queue()
    dialog.close()
    assert not dialog.isVisible()
    qtbot.waitUntil(lambda: not workspace.is_running(), timeout=20_000)
    assert [record["status"] for record in records] == ["Completed", "Completed"]
    assert hub.open_studio() is dialog
    assert np.array_equal(workspace._current_dataset().y_raw, original_y)
    assert len(workspace._history()) == 2


@pytest.mark.gui
def test_hub_afz_roundtrip_retains_queue_and_studio_view(studio_factory, tmp_path):
    hub, owner = studio_factory()
    workspace = hub.sequential_workspace
    records = workspace.queue_checked_candidates()
    workspace.detail_tabs.setCurrentIndex(3)
    workspace.result_tabs.setCurrentIndex(1)
    path = tmp_path / "detached.afz"
    save_afz(path, [owner.dataset], {}, {"exclusion_refinement_investigation": hub.get_state()})
    datasets, _, analysis = load_afz(path)
    restored, new_owner = studio_factory()
    new_owner.dataset = datasets[0]
    restored.set_state(analysis["exclusion_refinement_investigation"])
    assert restored.studio_dialog is None  # Never auto-opens or runs on load.
    loaded = restored.sequential_workspace
    assert not loaded.is_running()
    assert loaded.detail_tabs.currentIndex() == 3 and loaded.result_tabs.currentIndex() == 1
    assert [record["run_id"] for record in loaded._history()] == [record["run_id"] for record in records]
    assert all(record["status"] == "Queued" for record in loaded._history())


def test_open_studio_has_no_global_cif_or_phase_prerequisite():
    states = build_workflow_status({})
    tasks = build_task_states(states, {}, mode="Expert")
    assert tasks["exclusion_study"].enabled


@pytest.mark.gui
@pytest.mark.parametrize("method", ["rietveld", "pawley", "lebail"])
def test_crystal_button_opens_selected_native_result_not_active_cif(qtbot, studio_factory, method):
    hub, owner = studio_factory()
    workspace = hub.sequential_workspace
    studio = hub.open_studio()
    qtbot.addWidget(studio)
    assert not workspace.crystal_studio_button.isEnabled()
    workspace.queue_method.setCurrentIndex(workspace.queue_method.findData(method))
    records = workspace.queue_checked_candidates()
    workspace.start_queue()
    qtbot.waitUntil(lambda: not workspace.is_running(), timeout=25_000)
    workspace.history_table.selectRow(0)
    assert workspace.crystal_studio_button.isEnabled()
    assert records[0]["run_id"] in workspace.crystal_studio_button.toolTip()
    snapshot_before = deepcopy(records[0]["structure_snapshot"])
    # Break/remove the live reference. It must not influence a saved run view.
    workspace.structures[0]["atoms"] = []
    owner.reference_structure = deepcopy(workspace.structures[1])
    workspace.crystal_studio_button.click()
    assert workspace._crystal_model_job is not None
    assert "Preparing" in workspace.crystal_studio_button.text()
    assert not workspace.crystal_studio_button.isEnabled()
    # A selection change during preparation must not redirect the frozen request.
    workspace.history_table.selectRow(1)
    qtbot.waitUntil(lambda: bool(workspace._crystal_dialogs), timeout=30_000)
    crystal = workspace._crystal_dialogs[-1]
    qtbot.addWidget(crystal)
    assert crystal.isVisible() and not crystal.isModal()
    assert records[0]["run_id"] in crystal.windowTitle()
    assert crystal.models[0]["atoms"] == snapshot_before["atoms"]
    assert crystal.models[0]["cell"] == records[0]["result"]["phases"][0]["refined_cell"]
    qtbot.waitUntil(lambda: crystal._last_scene is not None and crystal._preview_job is None, timeout=30_000)
    assert crystal.preview._image is not None
    assert crystal.gpu_view_button.isEnabled()
    assert crystal._last_scene.model["provenance"]["refinement_id"] == records[0]["run_id"]
    assert crystal._last_scene.model["atoms"] == crystal.models[0]["atoms"]
    workspace.history_table.selectRow(1)
    assert crystal.models[0]["provenance"]["refinement_id"] == records[0]["run_id"]
    assert records[0]["structure_snapshot"] == snapshot_before
    crystal.close()
    qtbot.waitUntil(lambda: not workspace._crystal_dialogs)
    studio.close()


@pytest.mark.gui
def test_candidate_preview_targets_selected_row_and_history_button_restores_from_afz(qtbot, studio_factory, tmp_path):
    hub, owner = studio_factory()
    workspace = hub.sequential_workspace
    workspace.candidate_table.selectRow(0)
    assert workspace.candidate_crystal_button.isEnabled()
    workspace.candidate_crystal_button.click()
    qtbot.waitUntil(lambda: bool(workspace._crystal_dialogs), timeout=30_000)
    preview = workspace._crystal_dialogs[-1]
    qtbot.addWidget(preview)
    assert preview.models[0]["atoms"] == workspace.structures[0]["atoms"]
    assert "Not a completed refinement" in preview.provenance.text()
    assert preview.models[0]["provenance"]["kind"] == "cif_reference"
    preview.close()
    records = workspace.queue_checked_candidates()
    workspace.start_queue()
    qtbot.waitUntil(lambda: not workspace.is_running(), timeout=25_000)
    workspace.history_table.selectRow(0)
    path = tmp_path / "crystal-link.afz"
    save_afz(path, [owner.dataset], {}, {"exclusion_refinement_investigation": hub.get_state()})
    datasets, _, analysis = load_afz(path)
    restored, new_owner = studio_factory()
    new_owner.dataset = datasets[0]
    restored.set_state(analysis["exclusion_refinement_investigation"])
    loaded = restored.sequential_workspace
    assert loaded.crystal_studio_button.isEnabled()
    loaded.open_crystal_studio()
    qtbot.waitUntil(lambda: bool(loaded._crystal_dialogs), timeout=30_000)
    crystal = loaded._crystal_dialogs[-1]
    qtbot.addWidget(crystal)
    assert crystal.models[0]["provenance"]["refinement_id"] == records[0]["run_id"]
    crystal._timer.stop()  # Selection/provenance test; CPU rendering tested above.
    crystal.close()


@pytest.mark.gui
def test_crystal_validation_runs_off_gui_thread_and_failure_recovers_controls(qtbot, studio_factory, monkeypatch):
    import threading
    from PySide6.QtWidgets import QMessageBox
    import afruz_pxrd.sequential_refinement_workspace as module

    hub, _ = studio_factory()
    workspace = hub.sequential_workspace
    records = workspace.queue_checked_candidates()
    workspace.start_queue()
    qtbot.waitUntil(lambda: not workspace.is_running(), timeout=25_000)
    workspace.history_table.selectRow(0)
    records[0]["result"]["phases"][0].pop("refined_cell")
    gui_thread = threading.get_ident()
    preparation_threads = []
    real_prepare = module.models_from_refinement_record
    def prepare(record):
        preparation_threads.append(threading.get_ident())
        return real_prepare(record)
    monkeypatch.setattr(module, "models_from_refinement_record", prepare)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, message: warnings.append(message))
    job = workspace.open_crystal_studio()
    assert job is not None and workspace._crystal_model_job is job
    assert workspace.open_crystal_studio() is None  # No duplicate background jobs.
    qtbot.waitUntil(lambda: bool(warnings) and workspace._crystal_model_job is None, timeout=30_000)
    assert preparation_threads and all(value != gui_thread for value in preparation_threads)
    assert "saved fitted unit cell" in warnings[0]
    assert not workspace._crystal_dialogs
    assert workspace.crystal_studio_button.isEnabled()
    assert workspace.crystal_studio_button.text() == "Open Crystal Studio"
