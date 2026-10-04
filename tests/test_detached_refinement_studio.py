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
    assert "23.3.1" in dialog.windowTitle()
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
