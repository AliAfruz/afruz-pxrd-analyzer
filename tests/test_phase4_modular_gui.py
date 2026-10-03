from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from afruz_pxrd.application import ApplicationState, ProjectState, WorkflowController
from afruz_pxrd.app import MainWindow
from afruz_pxrd.io_engine import DataImportError
from afruz_pxrd.models import Dataset
from afruz_pxrd.services import ExportService, ImportService, ReportService
from afruz_pxrd.workspaces.base import WorkspaceAdapter, WorkspaceContract
from afruz_pxrd.workspaces.peaks_workspace import PeaksWorkspace
from afruz_pxrd.workspaces.phase_workspace import PhaseWorkspace
from afruz_pxrd.workspaces.preparation_workspace import PreparationWorkspace
from afruz_pxrd.workspaces.refinement_workspace import RefinementWorkspace
from afruz_pxrd.workspaces.registry import WorkspaceRegistry
from afruz_pxrd.workspaces.validation_workspace import ValidationWorkspace


def _dataset(uid: str = "phase4-dataset") -> Dataset:
    return Dataset(
        uid=uid,
        name="Phase 4 sample",
        x=np.array([10.0, 11.0, 12.0, 13.0]),
        y_raw=np.array([4.0, 8.0, 6.0, 5.0]),
    )


class _DummyWorkspace:
    def __init__(self):
        self.state = None
        self.dataset = None
        self.runs = 0
        self.refreshes = 0

    def set_project_state(self, state):
        self.state = state

    def set_dataset(self, dataset):
        self.dataset = dataset

    def validate_inputs(self):
        return True, ""

    def run_analysis(self):
        self.runs += 1
        return "complete"

    def refresh_results(self):
        self.refreshes += 1


@pytest.mark.parametrize(
    "workspace_type",
    (
        PreparationWorkspace,
        PeaksWorkspace,
        PhaseWorkspace,
        RefinementWorkspace,
        ValidationWorkspace,
    ),
)
def test_each_workspace_type_implements_common_interface(workspace_type):
    state = ProjectState()
    dummy = _DummyWorkspace()
    workspace = workspace_type("test", dummy, project_state=state)
    assert isinstance(workspace, WorkspaceContract)

    dataset = _dataset()
    workspace.set_project_state(state)
    workspace.set_dataset(dataset)
    assert workspace.validate_inputs().valid
    assert workspace.run_analysis() == "complete"
    workspace.refresh_results()
    assert dummy.state is state
    assert dummy.dataset is dataset
    assert dummy.runs == 1
    assert dummy.refreshes == 1


def test_registry_and_workflow_controller_are_gui_independent():
    state = ProjectState()
    application_state = ApplicationState()
    registry = WorkspaceRegistry(state)
    dummy = _DummyWorkspace()
    adapter = registry.register("preparation", dummy)
    controller = WorkflowController(application_state, registry)

    selected = controller.activate("preparation", workspace_key="preparation")
    controller.set_dataset(_dataset())
    assert selected is adapter
    assert application_state.active_workflow_task == "preparation"
    assert registry.validate_inputs("preparation").valid
    assert controller.run_active() == "complete"
    controller.refresh()
    assert dummy.runs == 1
    assert dummy.refreshes == 1


def test_workspace_adapter_emits_lifecycle_signals(qtbot):
    workspace = WorkspaceAdapter("signal-test", _DummyWorkspace())
    with qtbot.waitSignal(workspace.analysisRequested):
        workspace.run_analysis()
    with qtbot.waitSignal(workspace.resultsRefreshed):
        workspace.refresh_results()


def test_import_service_collects_successes_and_actionable_failures():
    dataset = _dataset()

    def loader(path):
        if Path(path).name == "bad.xy":
            raise DataImportError("invalid diffraction columns")
        return [dataset]

    batch = ImportService(loader).load_paths(("good.xy", "bad.xy"))
    assert batch.datasets == (dataset,)
    assert batch.imported_file_count == 1
    assert len(batch.failures) == 1
    assert batch.failures[0].display_text() == "bad.xy: invalid diffraction columns"


def test_export_and_report_services_are_independently_testable(tmp_path):
    text_path = ExportService().export_dataset_text(
        _dataset(), tmp_path / "dataset.txt"
    )
    assert text_path.is_file()
    assert "Phase 4 sample" in text_path.read_text(encoding="utf-8-sig")

    calls = {}

    def builder(output_root, **payload):
        calls["builder"] = (output_root, payload)
        return {"output_dir": tmp_path / "package"}

    def archiver(output_dir):
        calls["archiver"] = output_dir
        return {"archive_path": tmp_path / "package.zip"}

    result = ReportService(builder=builder, archiver=archiver).build_complete_package(
        tmp_path / "report", project_name="Phase4"
    )
    assert calls["builder"][1] == {"project_name": "Phase4"}
    assert calls["archiver"] == tmp_path / "package"
    assert result["archive"]["archive_path"].name == "package.zip"


@pytest.mark.gui
def test_main_window_builds_registry_without_direct_widget_imports(qtbot, project_root):
    window = MainWindow()
    qtbot.addWidget(window)

    assert set(window._workflow_task_widgets) == set(window.workspace_registry.keys)
    assert window.workflow_controller.registry is window.workspace_registry
    assert window.workspace_registry.project_state is window.project_state
    assert all(
        isinstance(window.workspace_registry.adapter(key), WorkspaceContract)
        for key in window.workspace_registry.keys
    )

    app_lines = (project_root / "afruz_pxrd" / "app.py").read_text(
        encoding="utf-8"
    ).splitlines()
    main_lines = (project_root / "afruz_pxrd" / "main_window.py").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(app_lines) < 100
    assert len(main_lines) < 1500
    assert "preview_background" not in MainWindow.__dict__
    assert "quantify_selected_phases" not in MainWindow.__dict__

    dependency_source = (project_root / "afruz_pxrd" / "main_window_dependencies.py").read_text(
        encoding="utf-8"
    )
    for class_name in (
        "InstrumentCalibrationWidget",
        "WholePatternRefinementWidget",
        "RietveldRefinementWidget",
        "ValidationCampaignWidget",
    ):
        assert class_name not in dependency_source
    window.close()
