from copy import deepcopy

import pytest

from afruz_pxrd.crystal_scene import build_scene, RenderSettings, structure_snapshot
from afruz_pxrd.refinement_crystal import model_from_candidate, models_from_refinement_record


def saved_record(structure, method="rietveld"):
    cell = {**structure["cell"], "a": structure["cell"]["a"] + 0.02}
    return {"run_id": "AFR-RT-00017", "method_key": method, "status": "Completed",
            "dataset_uid": "data-1", "dataset_name": "Measured sample", "step_id": "S02",
            "candidate_uid": "candidate-1", "candidate_label": "Framework A", "label": "Chosen run",
            "parent_run_id": "AFR-RT-00003", "cif_uid": "cif-1", "cif_name": "Frozen model",
            "cif_source": "no-longer-available.cif", "structure_snapshot": deepcopy(structure),
            "settings": {"refine_cell": True}, "excluded_regions": [{"start_deg": 20, "end_deg": 21, "label": "Impurity hypothesis"}],
            "comparison_domain_signature": "saved-domain", "result": {"success": True, "rwp_percent": 4.2,
                "message": "Converged", "phases": [{"phase_name": "Frozen model", "cell_refined": True,
                "refined_cell": cell}]}}


@pytest.mark.parametrize("method", ["rietveld", "pawley", "lebail"])
def test_all_methods_use_fitted_cell_frozen_atoms_and_precise_provenance(nacl_structure, method):
    record = saved_record(nacl_structure, method)
    before = deepcopy(record)
    model = models_from_refinement_record(record)[0]
    assert record == before
    assert model["cell"] == record["result"]["phases"][0]["refined_cell"]
    assert model["atoms"] == record["structure_snapshot"]["atoms"]
    assert model["cell"] != record["structure_snapshot"]["cell"]
    provenance = model["provenance"]
    assert provenance["kind"] == f"{method}_result"
    assert provenance["refinement_id"] == record["run_id"]
    assert provenance["refinement_success"] is True
    assert provenance["input_cell"] == nacl_structure["cell"]
    assert provenance["parent_refinement_id"] == record["parent_run_id"]
    assert provenance["excluded_regions"] == record["excluded_regions"]
    assert provenance["rwp_percent"] == 4.2
    assert "fixed" in provenance["note"]
    assert ("does not determine atomic positions" in provenance["note"]) is (method != "rietveld")
    model["atoms"][0]["element"] = "C"
    model["cell"]["a"] = 99
    assert record == before


def test_native_phase_snapshot_is_authoritative_not_record_or_live_model(nacl_structure, cscl_structure):
    record = saved_record(nacl_structure)
    phase = record["result"]["phases"][0]
    phase["structure_snapshot"] = structure_snapshot(cscl_structure)
    phase["refined_cell"] = deepcopy(cscl_structure["cell"])
    model = models_from_refinement_record(record)[0]
    assert model["atoms"] == cscl_structure["atoms"]
    assert model["cell"] == cscl_structure["cell"]


@pytest.mark.parametrize("fault, message", [
    ("queued", "completed run"), ("atoms", "saved atomic model"),
    ("cell", "saved fitted unit cell"), ("phases", "one saved phase"),
    ("nonfinite", "unit cell"), ("method", "recognized refinement method"),
])
def test_incomplete_or_invalid_records_never_guess_a_structure(nacl_structure, fault, message):
    record = saved_record(nacl_structure)
    if fault == "queued":
        record["status"] = "Queued"
    elif fault == "atoms":
        record.pop("structure_snapshot")
    elif fault == "cell":
        record["result"]["phases"][0].pop("refined_cell")
    elif fault == "phases":
        record["result"]["phases"] *= 2
    elif fault == "nonfinite":
        record["result"]["phases"][0]["refined_cell"]["a"] = float("inf")
    else:
        record["method_key"] = "unknown"
    with pytest.raises(ValueError, match=message):
        models_from_refinement_record(record)


def test_nonconverged_completed_fit_is_inspectable_with_warning(nacl_structure):
    record = saved_record(nacl_structure)
    record["result"]["success"] = False
    model = models_from_refinement_record(record)[0]
    assert "did not converge" in model["provenance"]["note"]
    assert model["provenance"]["refinement_success"] is False


def test_candidate_preview_distinguishes_reference_from_parent_seed(nacl_structure):
    candidate = {"candidate_uid": "c1", "label": "Candidate A", "cif_uid": "cif-1"}
    before = deepcopy(nacl_structure)
    preview = model_from_candidate(nacl_structure, candidate, step_id="S01")
    assert preview["provenance"]["kind"] == "cif_reference"
    assert "Not a completed refinement" in preview["provenance"]["note"]
    assert preview["cell"] == before["cell"]
    parent = saved_record(nacl_structure)
    candidate["parent_run_id"] = parent["run_id"]
    seeded = model_from_candidate(nacl_structure, candidate, parent, step_id="S02")
    assert seeded["provenance"]["kind"] == "refinement_candidate_seed"
    assert parent["run_id"] in seeded["provenance"]["note"]
    assert seeded["cell"] == parent["result"]["phases"][0]["refined_cell"]
    assert seeded["atoms"] == before["atoms"]
    assert nacl_structure == before
    with pytest.raises(ValueError, match="parent run is missing"):
        model_from_candidate(nacl_structure, candidate)
    parent["result"]["phases"] = []
    with pytest.raises(ValueError, match="no saved fitted unit cell"):
        model_from_candidate(nacl_structure, candidate, parent)


def test_excluded_xrd_regions_do_not_delete_crystal_atoms(nacl_structure):
    record = saved_record(nacl_structure)
    settings = RenderSettings(repeats=(1, 1, 1), complete_boundaries=False)
    masked = build_scene(models_from_refinement_record(record)[0], settings)
    record["excluded_regions"] = []
    unmasked = build_scene(models_from_refinement_record(record)[0], settings)
    assert masked.positions == pytest.approx(unmasked.positions)
    assert masked.elements == unmasked.elements
    assert masked.bonds == unmasked.bonds
