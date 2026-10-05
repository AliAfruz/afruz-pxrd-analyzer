"""Frozen, explicitly labelled Crystal Studio models for the step-wise lab."""
from __future__ import annotations

from copy import deepcopy

from .candidate_queue import parent_seed
from .crystal_scene import model_from_structure, models_from_result


METHOD_NAMES = {"pawley": "Pawley", "lebail": "Le Bail", "rietveld": "Rietveld"}


def models_from_refinement_record(record: dict) -> list[dict]:
    """Use this run's saved cell/sites, never the live CIF or another refiner.

    Pawley/Le Bail do not determine atomic coordinates. Their visual model is
    the frozen input CIF's sites in the fitted cell, with that limitation
    recorded in both the visible caption and exported scene provenance.
    """
    result = record.get("result")
    if record.get("status") != "Completed" or not isinstance(result, dict):
        raise ValueError("Select a completed run in History to view its fitted crystal model.")
    method = str(record.get("method_key") or "")
    if method not in METHOD_NAMES:
        raise ValueError("This saved run has no recognized refinement method.")
    phases = result.get("phases") or []
    if len(phases) != 1:
        raise ValueError("A step-wise crystal view requires one saved phase for this one-CIF run.")
    phase = deepcopy(phases[0])
    snapshot = phase.get("structure_snapshot") or record.get("structure_snapshot")
    if not isinstance(snapshot, dict) or not snapshot.get("atoms"):
        raise ValueError("This run has no saved atomic model. Rerun it; the current CIF cannot replace its frozen input.")
    if not isinstance(phase.get("refined_cell"), dict):
        raise ValueError("This run has no saved fitted unit cell. Rerun it before opening its crystal view.")
    phase["structure_snapshot"] = deepcopy(snapshot)
    phase["atoms_fixed"] = True
    phase.setdefault("cell_refined", bool((record.get("settings") or {}).get("refine_cell")))
    phase.setdefault("phase_name", record.get("cif_name") or snapshot.get("data_name") or "Candidate")
    payload = {**result, "phases": [phase]}
    models = models_from_result(payload)
    model = models[0]
    run_id = str(record.get("run_id") or "Saved run")
    model["data_name"] = f"{phase['phase_name']} · {run_id}"
    note = (
        f"{run_id} · {METHOD_NAMES[method]}: saved fitted cell; fractional coordinates and occupancies fixed to the input CIF. "
    )
    if method != "rietveld":
        note += "Cell-only structural visualization; this intensity decomposition does not determine atomic positions. "
    note += "Display radii are illustrative; no thermal ellipsoids are implied."
    if result.get("success") is False:
        note += " Refinement did not converge; inspect fit diagnostics."
    model["provenance"].update(
        kind=f"{method}_result", note=note, atoms_fixed=True,
        refinement_id=run_id, refinement_method=METHOD_NAMES[method],
        parent_refinement_id=record.get("parent_run_id"),
        run_label=record.get("label"), step_id=record.get("step_id"),
        candidate_uid=record.get("candidate_uid"), candidate_label=record.get("candidate_label"),
        source_dataset_uid=record.get("dataset_uid"), source_dataset_name=record.get("dataset_name"),
        cif_uid=record.get("cif_uid"), cif_source=record.get("cif_source"),
        input_cell=deepcopy(snapshot.get("cell")),
        excluded_regions=deepcopy(record.get("excluded_regions") or []),
        comparison_domain_signature=record.get("comparison_domain_signature"),
    )
    return models


def model_from_candidate(structure: dict, candidate: dict, parent: dict | None = None, *, step_id=None) -> dict:
    """Preview a candidate's initial/parent-seeded model, not a new fit."""
    parent_id = candidate.get("parent_run_id")
    if parent_id and (parent is None or parent.get("run_id") != parent_id):
        raise ValueError("The candidate's saved parent run is missing. Select an available completed parent first.")
    seed, audit = parent_seed(structure, parent if parent_id else None)
    if parent_id and not audit["refined_cell_carried"]:
        raise ValueError("The candidate's parent has no saved fitted unit cell. Rerun that parent before previewing a seeded model.")
    model = model_from_structure(seed)
    label = candidate.get("label") or candidate.get("cif_name") or "CIF candidate"
    model["data_name"] = str(label)
    note = f"Candidate preview · {label}: "
    note += f"starting cell from {parent_id}; " if audit["refined_cell_carried"] else "input CIF cell; "
    note += "fixed CIF fractional coordinates and occupancies. Not a completed refinement result. Display radii are illustrative."
    model["provenance"].update(
        kind="refinement_candidate_seed" if parent_id else "cif_reference", note=note,
        candidate_uid=candidate.get("candidate_uid"), candidate_label=label,
        cif_uid=candidate.get("cif_uid"), parent_refinement_id=parent_id,
        step_id=step_id, seed_audit=deepcopy(audit), atoms_fixed=True,
    )
    return model
