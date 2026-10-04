from copy import deepcopy
import json
from zipfile import ZipFile

import numpy as np
import pytest
from PySide6.QtCore import Qt, Slot

from afruz_pxrd.candidate_queue import (
    can_advance, candidate_run_label, comparison_domain_signature,
    pack_snapshots, parent_seed, unpack_snapshots,
)
from afruz_pxrd.models import Dataset
from afruz_pxrd.project import load_afz, save_afz
import afruz_pxrd.sequential_refinement_workspace as queue_module


@pytest.fixture
def queue_factory(qtbot, nacl_raw_frame, nacl_structure, cscl_structure):
    class Owner:
        reference_structure = None

        def __init__(self, dataset):
            self.dataset = dataset
            self.dirty_count = 0

        def selected_dataset(self):
            return self.dataset

        def _mark_project_dirty(self):
            self.dirty_count += 1

    def build():
        dataset = Dataset(uid="queue-data", name="Queue validation",
                          x=nacl_raw_frame["two_theta_deg"].to_numpy().copy(),
                          y_raw=nacl_raw_frame["intensity_counts"].to_numpy().copy(),
                          source_path="validation.xy")
        owner = Owner(dataset)
        widget = queue_module.SequentialRefinementWorkspace(owner)
        qtbot.addWidget(widget)
        widget._add_structure(nacl_structure, "NaCl.cif")
        widget._add_structure(cscl_structure, "CsCl.cif")
        widget.use_gpu.setChecked(False)
        widget.refine_cell.setChecked(False)
        widget.refine_profile.setChecked(False)
        widget.refine_zero.setChecked(False)
        widget.staged.setChecked(False)
        widget.freeze_structure.setChecked(True)
        widget.max_evaluations.setValue(10)
        widget.extraction_cycles.setValue(1)
        widget.background_order.setValue(2)
        widget.cutoff.setValue(0.5)
        return widget, owner

    return build


@pytest.fixture
def fake_worker(monkeypatch):
    class FakeWorker(queue_module.SequentialRefinementWorker):
        live = 0
        maximum_live = 0

        @Slot()
        def run(self):
            type(self).live += 1
            type(self).maximum_live = max(type(self).maximum_live, type(self).live)
            try:
                if self.structure.get("data_name") == "Bad candidate":
                    self.failed.emit("Deliberate candidate failure")
                    return
                if self._cancel_event.is_set():
                    self.cancelled.emit()
                    return
                cell = deepcopy(self.structure["cell"])
                cell["a"] += 0.01
                self.finished.emit({"success": True, "rwp_percent": 1.0, "rp_percent": 1.0,
                    "observed_x": self.x.tolist(), "observed_y": self.y.tolist(),
                    "calculated_y": self.y.tolist(), "background_y": np.zeros_like(self.y).tolist(),
                    "difference_y": np.zeros_like(self.y).tolist(), "reflections": [],
                    "phases": [{"refined_cell": cell}], "zero_shift_deg": 0.01,
                    "profile": {"caglioti_u": 0.005, "caglioti_v": 0.0, "caglioti_w": 0.02, "eta": 0.5}})
            finally:
                type(self).live -= 1

    monkeypatch.setattr(queue_module, "SequentialRefinementWorker", FakeWorker)
    return FakeWorker


def test_parent_model_and_domain_signature_do_not_mutate_source(nacl_structure):
    original = deepcopy(nacl_structure)
    cell = deepcopy(original["cell"])
    cell["a"] += 0.01
    parent = {"run_id": "AFR-RT-00001", "status": "Completed", "result": {"phases": [{"refined_cell": cell}]}}
    seed, audit = parent_seed(original, parent)
    assert audit["refined_cell_carried"]
    assert seed["cell"]["a"] == cell["a"]
    assert original["cell"] == nacl_structure["cell"]
    assert candidate_run_label(2, 3, "framework A", "rietveld") == "S02 · C03 · RT · framework A"
    assert not can_advance({"status": "Completed", "result": {"success": False}})
    x = np.linspace(5, 80, 301)
    settings = {"two_theta_min": 5, "two_theta_max": 80, "weighting": "Poisson-like"}
    signature = comparison_domain_signature(x, x, settings, [(20, 21, "A")])
    assert signature == comparison_domain_signature(x, x, settings, [(20, 21, "different label")])
    assert signature != comparison_domain_signature(x, x, settings, [(20, 22, "A")])
    bad = deepcopy(parent)
    bad["result"]["phases"][0]["refined_cell"]["a"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        parent_seed(original, bad)


def test_snapshot_pool_shares_fixed_atoms_but_retains_individual_cells(nacl_structure):
    first = deepcopy(nacl_structure)
    second = deepcopy(nacl_structure)
    second["cell"]["a"] += 0.1
    input_snapshot = {"x": [5, 6], "observed_y": [10, 20]}
    source = {"d": [{"structure_snapshot": first, "input_snapshot": input_snapshot},
                    {"structure_snapshot": second, "input_snapshot": input_snapshot}]}
    packed, steps, pool = pack_snapshots(source, {})
    assert len(pool["models"]) == len(pool["inputs"]) == 1
    assert "structure_snapshot" in source["d"][0]
    unpack_snapshots(packed, steps, pool)
    assert packed["d"][0]["structure_snapshot"]["cell"]["a"] != packed["d"][1]["structure_snapshot"]["cell"]["a"]
    assert packed["d"][0]["input_snapshot"] == input_snapshot


@pytest.mark.gui
@pytest.mark.parametrize("method", ["pawley", "lebail", "rietveld"])
def test_real_native_methods_run_two_cifs_serially_with_frozen_masks(qtbot, queue_factory, method):
    widget, owner = queue_factory()
    widget.queue_method.setCurrentIndex(widget.queue_method.findData(method))
    widget.exclusion_editor.set_regions([(30, 32, "shared impurity zone")])
    original_y = owner.dataset.y_raw.copy()
    records = widget.queue_checked_candidates()
    assert len(records) == 2
    assert records[0]["comparison_domain_signature"] == records[1]["comparison_domain_signature"]
    assert [record["candidate_order"] for record in records] == [1, 2]
    assert all(record["parent_run_id"] is None for record in records)
    widget.exclusion_editor.set_regions([(40, 41, "later draft; not this queue")])
    widget.start_queue()
    qtbot.waitUntil(lambda: not widget.is_running(), timeout=20_000)
    assert [record["status"] for record in records] == ["Completed", "Completed"]
    assert records[1]["started_utc"] >= records[0]["completed_utc"]
    for record in records:
        result = record["result"]
        assert result["step_id"] == "S01"
        assert result["excluded_data_point_count"] > 0
        assert not np.any((np.asarray(result["observed_x"]) >= 30) & (np.asarray(result["observed_x"]) <= 32))
        assert record["excluded_regions"][0]["start_deg"] == 30
    assert np.array_equal(owner.dataset.y_raw, original_y)


@pytest.mark.gui
def test_checked_results_create_steps_two_and_three_without_merging_candidates(qtbot, queue_factory, fake_worker):
    widget, _ = queue_factory()
    first = widget.queue_checked_candidates()
    widget.start_queue()
    qtbot.waitUntil(lambda: not widget.is_running(), timeout=5000)
    widget.history_table.item(0, 0).setCheckState(Qt.Checked)
    widget.history_table.item(1, 0).setCheckState(Qt.Checked)
    second_step = widget.prepare_next_step()
    assert second_step["number"] == 2
    assert len(second_step["candidates"]) == 2
    widget.exclusion_editor.set_regions([(20, 21, "Step 2 zone")])
    second = widget.queue_checked_candidates()
    assert [record["parent_run_id"] for record in second] == [record["run_id"] for record in first]
    for parent, child in zip(first, second):
        assert child["structure_snapshot"]["cell"]["a"] == parent["result"]["phases"][0]["refined_cell"]["a"]
        assert child["structure_seed_audit"]["refined_cell_carried"]
        assert child["settings"]["initial_zero_shift"] == 0.01
    assert len({record["comparison_domain_signature"] for record in second}) == 1
    widget.start_queue()
    qtbot.waitUntil(lambda: not widget.is_running(), timeout=5000)
    widget.history_table.item(2, 0).setCheckState(Qt.Checked)
    third_step = widget.prepare_next_step()
    assert third_step["number"] == 3 and len(third_step["candidates"]) == 1
    assert third_step["candidates"][0]["parent_run_id"] == second[0]["run_id"]
    assert fake_worker.maximum_live == 1
    widget.step_selector.setCurrentIndex(0)
    assert widget.exclusion_editor.regions() == []
    widget.step_selector.setCurrentIndex(1)
    assert widget.exclusion_editor.regions()[0].label == "Step 2 zone"


@pytest.mark.gui
def test_pause_and_afz_resume_preserve_frozen_pending_jobs(qtbot, queue_factory, fake_worker, tmp_path):
    widget, owner = queue_factory()
    records = widget.queue_checked_candidates()
    widget.runCompleted.connect(widget.pause_queue)
    widget.start_queue()
    qtbot.waitUntil(lambda: not widget.is_running(), timeout=5000)
    assert [record["status"] for record in records] == ["Completed", "Queued"]
    state = widget.get_state()
    path = tmp_path / "step-queue.afz"
    save_afz(path, [owner.dataset], {}, {"queue": state})
    datasets, _, analysis = load_afz(path)
    restored, new_owner = queue_factory()
    new_owner.dataset = datasets[0]
    restored.set_state(analysis["queue"])
    assert not restored.is_running()
    assert [row["status"] for row in restored._history()] == ["Completed", "Queued"]
    restored.exclusion_editor.set_regions([(40, 41, "not part of frozen queue")])
    restored.start_queue()
    qtbot.waitUntil(lambda: not restored.is_running(), timeout=5000)
    assert all(row["status"] == "Completed" for row in restored._history())
    assert restored._history()[1]["run_id"] == records[1]["run_id"]
    assert restored._history()[1]["excluded_regions"] == []


@pytest.mark.gui
@pytest.mark.parametrize("continue_on_failure", [True, False])
def test_failed_candidates_do_not_block_or_advance(qtbot, queue_factory, fake_worker, continue_on_failure):
    widget, _ = queue_factory()
    widget.structures[0]["data_name"] = "Bad candidate"
    widget.continue_after_failure.setChecked(continue_on_failure)
    records = widget.queue_checked_candidates()
    widget.start_queue()
    qtbot.waitUntil(lambda: not widget.is_running(), timeout=5000)
    assert records[0]["status"] == "Failed"
    assert not can_advance(records[0])
    assert records[1]["status"] == ("Completed" if continue_on_failure else "Queued")


@pytest.mark.gui
def test_cancel_stops_dispatch_and_pending_jobs_can_be_discarded(qtbot, queue_factory, fake_worker):
    widget, _ = queue_factory()
    records = widget.queue_checked_candidates()
    widget.start_queue()
    widget.cancel_current_run()
    qtbot.waitUntil(lambda: not widget.is_running(), timeout=5000)
    assert records[1]["status"] == "Queued"
    widget.discard_pending_queue()
    assert records[1]["status"] == "Discarded before start"
    assert widget.candidate_table.isEnabled()


@pytest.mark.gui
def test_queue_export_records_plan_order_frozen_inputs_and_models(queue_factory, tmp_path):
    widget, _ = queue_factory()
    widget.candidate_table.selectRow(1)
    widget._move_candidate(-1)
    widget.candidate_table.item(0, 3).setText("My CsCl candidate")
    records = widget.queue_checked_candidates()
    assert records[0]["candidate_label"] == "My CsCl candidate"
    assert records[0]["cif_name"].lower().startswith("cscl")
    path = widget.export_full_history(target_path=tmp_path / "queue.zip")
    assert path is not None
    with ZipFile(path) as archive:
        assert "step_candidate_plan.txt" in archive.namelist()
        history = archive.read("refinement_history.txt").decode("utf-8-sig")
        assert "S01" in history and "My CsCl candidate" in history
        for record in records:
            prefix = record["run_id"]
            assert f"{prefix}/frozen_input_pattern.txt" in archive.namelist()
            model = json.loads(archive.read(f"{prefix}/frozen_model_snapshot.json"))
            assert model["cell"] == record["structure_snapshot"]["cell"]


@pytest.mark.gui
def test_running_snapshot_reopens_as_interrupted_and_never_auto_starts(queue_factory):
    widget, _ = queue_factory()
    records = widget.queue_checked_candidates()
    records[0]["status"] = "Running"
    state = widget.get_state()
    assert records[0]["status"] == "Running"
    restored, _ = queue_factory()
    restored.set_state(state)
    assert restored._history()[0]["status"].startswith("Interrupted")
    assert restored._history()[1]["status"] == "Queued"
    assert not restored.is_running()


@pytest.mark.gui
def test_parent_specific_domains_and_gpu_request_are_frozen(qtbot, queue_factory, fake_worker):
    widget, _ = queue_factory()
    first = widget.queue_checked_candidates()
    widget.start_queue()
    qtbot.waitUntil(lambda: not widget.is_running(), timeout=5000)
    first[0]["excluded_regions"] = [{"start_deg": 20, "end_deg": 21, "label": "Parent A zone"}]
    first[1]["excluded_regions"] = [{"start_deg": 30, "end_deg": 31, "label": "Parent B zone"}]
    first[0]["settings"]["two_theta_min"] = 10
    for index in range(2):
        widget.history_table.item(index, 0).setCheckState(Qt.Checked)
    widget.prepare_next_step()
    widget.common_domain.setChecked(False)
    widget.use_gpu.setChecked(True)
    second = widget.queue_checked_candidates()
    assert second[0]["excluded_regions"][0]["start_deg"] == 20
    assert second[1]["excluded_regions"][0]["start_deg"] == 30
    assert second[0]["settings"]["two_theta_min"] == 10
    assert second[0]["comparison_domain_signature"] != second[1]["comparison_domain_signature"]
    assert all(record["settings"]["use_gpu"] for record in second)
    widget.use_gpu.setChecked(False)
    assert all(record["settings"]["use_gpu"] for record in second)


@pytest.mark.gui
def test_frozen_queue_and_historical_plot_survive_live_data_changes(qtbot, queue_factory, fake_worker):
    widget, owner = queue_factory()
    widget.exclusion_editor.set_regions([(20, 21, "frozen excluded points")])
    records = widget.queue_checked_candidates()
    frozen_y = np.asarray(records[0]["input_snapshot"]["observed_y"])
    owner.dataset.y_raw[:] = 12345
    widget.structures[0]["cell"]["a"] += 5
    widget.start_queue()
    qtbot.waitUntil(lambda: not widget.is_running(), timeout=5000)
    assert np.array_equal(records[0]["result"]["observed_y"], frozen_y)
    assert records[0]["result"]["phases"][0]["refined_cell"]["a"] == records[0]["structure_snapshot"]["cell"]["a"] + 0.01
    widget._draw_record(records[0])
    excluded = next(item for item in widget.plot.listDataItems() if item.name() == "Excluded measured points")
    x, y = excluded.getData()
    assert len(x) > 0
    mask = (np.asarray(records[0]["input_snapshot"]["x"]) >= 20) & (np.asarray(records[0]["input_snapshot"]["x"]) <= 21)
    assert np.array_equal(y, frozen_y[mask])


@pytest.mark.gui
def test_loading_another_dataset_does_not_overwrite_saved_step_drafts(queue_factory):
    widget, owner = queue_factory()
    widget.exclusion_editor.set_regions([(20, 21, "Original dataset draft")])
    state = widget.get_state()
    other = deepcopy(owner.dataset)
    other.uid = "other-data"
    owner.dataset = other
    widget.refresh_for_selected_dataset()
    widget.exclusion_editor.set_regions([(30, 31, "Second dataset draft")])
    state_with_two = widget.get_state()
    # A project open changes the owner's selected dataset before state restore.
    owner.dataset.uid = "queue-data"
    widget.set_state(state_with_two)
    assert widget.steps_by_uid["other-data"][0]["draft_exclusions"][0]["start_deg"] == 30
    assert widget.steps_by_uid["queue-data"][0]["draft_exclusions"] == state["steps_by_uid"]["queue-data"][0]["draft_exclusions"]
