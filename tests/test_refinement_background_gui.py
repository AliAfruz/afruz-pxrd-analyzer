from copy import deepcopy
import threading

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

from afruz_pxrd.advanced_background import BACKGROUND_METHODS
from afruz_pxrd.app import MainWindow
from afruz_pxrd.models import Dataset
from afruz_pxrd.project import save_afz, load_afz
from afruz_pxrd.refinement_background import BACKGROUND_KEY, background_selection
from afruz_pxrd.refinement_background_widget import RefinementBackgroundControl


@pytest.fixture
def background_owner(qtbot, nacl_raw_frame):
    class Owner:
        background_results = {}
        reference_structure = None
        dirty = 0
        def __init__(self):
            self.dataset = Dataset(uid="shared-bg", name="Shared background fixture",
                x=nacl_raw_frame["two_theta_deg"].to_numpy().copy(),
                y_raw=nacl_raw_frame["intensity_counts"].to_numpy().copy(), metadata={"intensity_unit": "counts"})
        def selected_dataset(self):
            return self.dataset
        def _mark_project_dirty(self):
            self.dirty += 1
    owner = Owner()
    control = RefinementBackgroundControl(owner)
    qtbot.addWidget(control)
    return owner, control


@pytest.mark.gui
def test_dialog_exposes_full_engine_background_worker_and_shared_choice(qtbot, background_owner, monkeypatch):
    import afruz_pxrd.refinement_background_widget as module
    owner, control = background_owner
    dialog = control.open_engine()
    qtbot.addWidget(dialog)
    assert [dialog.method.itemText(index) for index in range(dialog.method.count())] == list(BACKGROUND_METHODS)
    assert set(dialog.values) == {"smoothness", "asymmetry", "iterations", "window_degrees", "percentile", "polynomial_order", "protection_sigma", "protection_width_multiplier"}
    gui_id, threads = threading.get_ident(), []
    real_detect = module.detect_background
    def detect(*args, **kwargs):
        threads.append(threading.get_ident())
        return real_detect(*args, **kwargs)
    monkeypatch.setattr(module, "detect_background", detect)
    dialog.preview_button.click()
    assert dialog._job is not None and not dialog.use_button.isEnabled()
    qtbot.waitUntil(lambda: dialog._job is None, timeout=20_000)
    assert threads and all(value != gui_id for value in threads)
    assert dialog.use_button.isEnabled() and dialog.candidate_table.rowCount() >= 2
    raw_before = owner.dataset.y_raw.copy()
    dialog.use_button.click()
    assert owner.dataset.metadata[BACKGROUND_KEY]["mode"] == "seed_plus_correction"
    assert "Shared:" in control.status.text() and owner.dirty == 1
    assert np.array_equal(owner.dataset.y_raw, raw_before)
    dialog.values["smoothness"].setValue(60)
    assert not dialog.use_button.isEnabled()  # Cannot silently use an outdated preview.
    assert owner.dataset.metadata[BACKGROUND_KEY]["parameters"]["smoothness"] == 70
    dialog.clear_button.click()
    assert BACKGROUND_KEY not in owner.dataset.metadata
    dialog.close()


@pytest.mark.gui
def test_import_main_preview_and_dataset_switch_never_commit_to_wrong_dataset(qtbot, background_owner, monkeypatch):
    owner, control = background_owner
    record = {"background": np.full(len(owner.dataset.x), 100.), "parameters": {"method": "SNIP"}, "diagnostics": {"selected_method": "SNIP"}}
    owner.background_results = {owner.dataset.uid: record}
    dialog = control.open_engine()
    qtbot.addWidget(dialog)
    dialog.import_button.click()
    assert dialog.use_button.isEnabled() and dialog.method.currentText() == "SNIP"
    owner.dataset = Dataset(name="Different raw pattern", x=owner.dataset.x.copy(), y_raw=owner.dataset.y_raw.copy())
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, message: warnings.append(message))
    dialog.use_button.click()
    assert warnings and "dataset changed" in warnings[0]
    assert BACKGROUND_KEY not in owner.dataset.metadata
    dialog.close()


@pytest.mark.gui
def test_background_export_and_manual_peak_share_the_original_data(qtbot, background_owner, tmp_path, monkeypatch):
    owner, control = background_owner
    requested_peaks = []
    owner._context_manual_peak_requested = lambda x, y, **kwargs: requested_peaks.append((x, y, kwargs))
    record = {"background": np.full(len(owner.dataset.x), 100.), "parameters": {"method": "SNIP", "clip_negative": True}}
    owner.background_results = {owner.dataset.uid: record}
    dialog = control.open_engine()
    qtbot.addWidget(dialog)
    dialog.import_button.click()
    dialog.plot.peakAddRequested.emit(30., 200.)
    assert requested_peaks == [(30., 200., {"origin": "Refinement background plot right-click"})]
    target = tmp_path / "baseline.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(target), "Text files (*.txt)"))
    dialog.export_button.click()
    exported = target.read_text(encoding="utf-8-sig")
    assert "starting_background_intensity" in exported and "protected_peak_region" in exported
    assert "refinement_observation_clipping: 0" in exported
    assert "raw observations unchanged" in exported
    assert (tmp_path / "baseline_diagnostics.txt").exists()
    dialog.close()


@pytest.mark.gui
@pytest.mark.parametrize("method", ["rietveld", "pawley", "lebail"])
def test_legacy_investigation_masks_the_same_raw_background(qtbot, background_owner, method):
    from afruz_pxrd.exclusion_refinement_widget import InvestigationRietveldWidget, InvestigationWholePatternWidget
    from afruz_pxrd.refinement_background import background_kwargs
    owner, _ = background_owner
    dataset = owner.dataset
    dataset.y_processed = np.zeros_like(dataset.y_raw)
    curve = np.linspace(100, 120, len(dataset.x))
    dataset.metadata[BACKGROUND_KEY] = background_selection(dataset, {"background": curve, "parameters": {"method": "SNIP"}})
    if method == "rietveld":
        widget = InvestigationRietveldWidget(owner)
        checkbox = widget.use_processed
    else:
        widget = InvestigationWholePatternWidget(owner, "Pawley decomposition" if method == "pawley" else "Le Bail-style extraction")
        checkbox = widget.use_processed_check
    qtbot.addWidget(widget)
    checkbox.setChecked(True)
    widget.exclusion_editor.set_state({"regions_by_uid": {dataset.uid: [{"start_deg": 30, "end_deg": 31, "label": "Shared raw-unit background"}]}})
    filtered, _audit = widget._filtered_dataset(dataset)
    mask = (dataset.x < 30) | (dataset.x > 31)
    assert background_kwargs(filtered)["background_seed_y"] == pytest.approx(curve[mask])
    observed, _statistics = widget._selected_input_and_statistics(filtered)
    assert observed == pytest.approx(dataset.y_raw[mask])
    assert widget._source_snapshot_by_uid[dataset.uid]["y"] == pytest.approx(dataset.y_raw)
    assert not widget._uses_processed()
    assert widget.background_control.main_window is owner
    widget.close()


@pytest.mark.gui
@pytest.mark.parametrize("method", ["rietveld", "pawley", "lebail"])
def test_queue_freezes_shared_raw_background_before_later_choices_and_afz(qtbot, background_owner, nacl_structure, tmp_path, method):
    from afruz_pxrd.sequential_refinement_workspace import SequentialRefinementWorkspace
    owner, _ = background_owner
    widget = SequentialRefinementWorkspace(owner)
    qtbot.addWidget(widget)
    widget._add_structure(nacl_structure, "fixture.cif")
    widget._add_structure(deepcopy(nacl_structure), "fixture-second.cif")
    widget.use_gpu.setChecked(False)
    widget.refine_cell.setChecked(False)
    widget.refine_profile.setChecked(False)
    widget.refine_zero.setChecked(False)
    widget.staged.setChecked(False)
    widget.max_evaluations.setValue(10)
    widget.extraction_cycles.setValue(4)
    widget.queue_method.setCurrentIndex(widget.queue_method.findData(method))
    owner.dataset.y_processed = np.zeros_like(owner.dataset.y_raw)
    widget.use_processed.setChecked(True)
    curve = np.full(len(owner.dataset.x), 300.)
    owner.dataset.metadata[BACKGROUND_KEY] = background_selection(owner.dataset, {"background": curve, "parameters": {"method": "SNIP"}})
    widget.exclusion_editor.set_regions([(30, 31, "Shared impurity hypothesis")])
    records = widget.queue_checked_candidates()
    assert len(records) == 2
    frozen = deepcopy(records[0])
    assert not frozen["settings"]["use_processed"]
    assert frozen["input_snapshot"]["observed_y"] == owner.dataset.y_raw.tolist()
    path = tmp_path / f"{method}-background.afz"
    save_afz(path, [owner.dataset], {}, {"workspace": widget.get_state()})
    data, _, analysis = load_afz(path)
    owner.dataset = data[0]
    widget.set_state(analysis["workspace"])
    records = widget._history()
    owner.dataset.metadata[BACKGROUND_KEY] = background_selection(owner.dataset, {"background": curve / 2, "parameters": {"method": "AsLS"}})
    widget.start_queue()
    qtbot.waitUntil(lambda: not widget.is_running(), timeout=25_000)
    for record in records:
        result = record["result"]
        assert record["status"] == "Completed"
        assert result["background_model"]["parameters"]["method"] == "SNIP"
        assert np.all(np.asarray(result["background_seed_y"]) == 300)
        assert all(not 30 <= value <= 31 for value in result["observed_x"])
        assert record["settings"]["background_seed_y"] == frozen["settings"]["background_seed_y"]
    widget.close()


@pytest.mark.gui
def test_regular_refiners_bypass_subtracted_processed_input_and_share_main_background_choice(qtbot, nacl_raw_frame):
    window = MainWindow()
    qtbot.addWidget(window)
    dataset = Dataset(uid="main-background", name="Main background fixture",
        x=nacl_raw_frame["two_theta_deg"].to_numpy().copy(), y_raw=nacl_raw_frame["intensity_counts"].to_numpy().copy(),
        y_processed=np.ones(len(nacl_raw_frame)), metadata={"intensity_unit": "counts"})
    window.datasets.append(dataset)
    # Dataset selection is isolated from file loading for this UI wiring check.
    window.selected_dataset = lambda: dataset
    window.background_results[dataset.uid] = {"background": np.full(len(dataset.x), 120.), "parameters": {"method": "SNIP"}}
    window.background_refinement_button.click()
    assert dataset.metadata[BACKGROUND_KEY]["parameters"]["method"] == "SNIP"
    for widget, checkbox in ((window.whole_pattern_widget, window.whole_pattern_widget.use_processed_check),
                             (window.rietveld_widget, window.rietveld_widget.use_processed)):
        checkbox.setChecked(True)
        observed, statistics = widget._selected_input_and_statistics(dataset)
        assert observed == pytest.approx(dataset.y_raw)
        assert not statistics["statistics_expected_valid"]
        assert "chosen background/correction" in statistics["selected_input_label"]
        assert "Shared:" in widget.background_control.status.text()
    studio_control = window.exclusion_refinement_widget.sequential_workspace.background_control
    assert "Shared:" in studio_control.status.text()
    dialog = window.rietveld_widget.background_control.open_engine()
    qtbot.addWidget(dialog)
    dialog.clear_button.click()
    for control in (studio_control, window.whole_pattern_widget.background_control, window.rietveld_widget.background_control):
        assert "Ordinary fitted" in control.status.text()
    dialog.close()
    window.close()
