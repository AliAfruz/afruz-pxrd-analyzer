"""Non-mutating candidate/step planning for one-CIF-at-a-time refinements."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from .exclusion_regions import merge_regions


def candidate_run_label(step: int, order: int, label: str, method: str) -> str:
    code = {"pawley": "PW", "lebail": "LB", "rietveld": "RT"}[method]
    return f"S{int(step):02d} · C{int(order):02d} · {code} · {str(label).strip() or 'CIF candidate'}"


def can_advance(record: dict) -> bool:
    return (record.get("status") == "Completed" and isinstance(record.get("result"), dict)
            and record["result"].get("success") is not False)


def parent_seed(structure: dict, parent: dict | None) -> tuple[dict, dict]:
    """Carry a parent's valid refined cell, not a residual or combined phase."""
    seed = deepcopy(structure)
    audit = {"kind": "original CIF", "parent_run_id": None, "refined_cell_carried": False}
    if parent is None:
        return seed, audit
    if not can_advance(parent):
        raise ValueError("Only a completed candidate can seed the next step.")
    audit.update(kind="selected parent model", parent_run_id=parent.get("run_id"))
    phases = (parent.get("result") or {}).get("phases") or []
    cell = phases[0].get("refined_cell") if len(phases) == 1 else None
    if isinstance(cell, dict):
        values = {key: float(cell.get(key, (seed.get("cell") or {}).get(key, math.nan)))
                  for key in ("a", "b", "c", "alpha", "beta", "gamma")}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Parent refined cell contains non-finite parameters.")
        if any(values[key] <= 0 for key in ("a", "b", "c")) or any(
            not 0 < values[key] < 180 for key in ("alpha", "beta", "gamma")
        ):
            raise ValueError("Parent refined cell is not physically valid.")
        ca, cb, cg = (math.cos(math.radians(values[key])) for key in ("alpha", "beta", "gamma"))
        if 1 + 2 * ca * cb * cg - ca * ca - cb * cb - cg * cg <= 1e-12:
            raise ValueError("Parent refined cell has zero or invalid volume.")
        seed["cell"] = {**(seed.get("cell") or {}), **values}
        audit.update(refined_cell_carried=True, cell=deepcopy(values))
    return seed, audit


def warm_start_settings(settings: dict, parent: dict | None) -> dict:
    output = deepcopy(settings)
    if parent is None:
        return output
    result = parent.get("result") or {}
    profile = result.get("profile") or {}
    seeds = {"initial_zero_shift": result.get("zero_shift_deg")}
    seeds.update({destination: profile.get(source) for source, destination in (
        ("caglioti_u", "initial_u"), ("caglioti_v", "initial_v"),
        ("caglioti_w", "initial_w"), ("eta", "initial_eta"),
    )})
    for key, value in seeds.items():
        if value is not None:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Parent profile contains a non-finite {key}.")
            output[key] = value
    return output


def comparison_domain_signature(x, y, settings: dict, regions) -> str:
    """Identify matched observations and fit domains, not model goodness."""
    digest = hashlib.sha256()
    for values in (x, y):
        array = np.asarray(values, dtype="<f8")
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError("Queue input must contain finite one-dimensional measurements.")
        digest.update(str(len(array)).encode("ascii"))
        digest.update(array.tobytes())
    domain = {
        "minimum": settings["two_theta_min"], "maximum": settings["two_theta_max"],
        "weighting": settings.get("weighting"), "processed": settings.get("use_processed", False),
        "regions": [[region.start_deg, region.end_deg] for region in merge_regions(regions or [])],
    }
    if settings.get("background_seed_provenance"):
        domain["background_seed_sha256"] = settings["background_seed_provenance"].get("seed_sha256")
    digest.update(json.dumps(domain, sort_keys=True, allow_nan=False).encode("utf-8"))
    return digest.hexdigest()


def pack_snapshots(histories: dict, steps: dict) -> tuple[dict, dict, dict]:
    """Deduplicate fixed atoms and observations in AFZ; cell seeds stay per run.

    MOF structures can contain thousands of atoms. Repeating the same atom
    array for every queued candidate would needlessly inflate project.json.
    """
    from .project import _json_default
    packed_history, packed_steps = deepcopy(histories), deepcopy(steps)
    pool = {"models": {}, "inputs": {}}

    def intern(value, kind):
        encoded = json.dumps(value, sort_keys=True, default=_json_default).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()
        pool[kind].setdefault(key, value)
        return key

    records = [record for rows in packed_history.values() for record in rows]
    candidates = [candidate for rows in packed_steps.values() for step in rows for candidate in step.get("candidates", [])]
    for row in [*records, *candidates]:
        structure = row.pop("structure_snapshot", None)
        if isinstance(structure, dict):
            cell = structure.pop("cell", None)
            row["structure_snapshot_ref"] = intern(structure, "models")
            row["structure_snapshot_cell"] = cell
    for row in records:
        observed = row.pop("input_snapshot", None)
        if isinstance(observed, dict):
            row["input_snapshot_ref"] = intern(observed, "inputs")
    return packed_history, packed_steps, pool


def unpack_snapshots(histories: dict, steps: dict, pool: dict):
    records = [record for rows in histories.values() for record in rows]
    candidates = [candidate for rows in steps.values() for step in rows for candidate in step.get("candidates", [])]
    for row in [*records, *candidates]:
        key = row.pop("structure_snapshot_ref", None)
        if key is not None:
            structure = deepcopy((pool.get("models") or {}).get(key))
            if isinstance(structure, dict):
                structure["cell"] = row.pop("structure_snapshot_cell", None)
            row["structure_snapshot"] = structure
    for row in records:
        key = row.pop("input_snapshot_ref", None)
        if key is not None:
            row["input_snapshot"] = deepcopy((pool.get("inputs") or {}).get(key))
