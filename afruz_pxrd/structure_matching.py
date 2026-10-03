from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .cif_library import (
    CellSearchTolerance,
    candidate_record_from_library_row,
    search_by_cell,
    search_by_peak_list,
)


def match_structure_candidates(
    db_path: str | Path,
    *,
    observed_peaks: Iterable[dict] | None = None,
    accepted_cell: dict | None = None,
    required_elements: Iterable[str] = (),
    peak_tolerance_deg: float = 0.15,
    limit: int = 25,
) -> dict:
    """Rank local CIF candidates using peaks, an indexed cell, or both.

    The function intentionally returns candidates rather than identifications.
    Downstream Pawley/Rietveld validation is still required.
    """
    peak_rows = []
    cell_rows = []
    if observed_peaks is not None:
        peak_rows = search_by_peak_list(
            db_path,
            observed_peaks,
            tolerance_deg=peak_tolerance_deg,
            required_elements=required_elements,
            limit=limit * 4,
        )
    if accepted_cell is not None:
        cell_rows = search_by_cell(
            db_path,
            accepted_cell,
            tolerance=CellSearchTolerance(),
            required_elements=required_elements,
            limit=limit * 4,
        )
    merged: dict[str, dict] = {}
    for row in cell_rows:
        key = row.get("sha256") or row.get("absolute_path")
        merged.setdefault(key, dict(row))
        merged[key]["cell_score_component"] = float(row.get("overall_score", 0.0))
    for row in peak_rows:
        key = row.get("sha256") or row.get("absolute_path")
        merged.setdefault(key, dict(row))
        merged[key]["peak_score_component"] = float(row.get("overall_score", 0.0))
        merged[key]["peak_match"] = row.get("peak_match", {})
    ranked = []
    for row in merged.values():
        peak_score = float(row.get("peak_score_component", 0.0))
        cell_score = float(row.get("cell_score_component", 0.0))
        if observed_peaks is not None and accepted_cell is not None:
            combined = 0.6 * peak_score + 0.4 * cell_score
            mode = "Peak-list + cell consensus"
        elif observed_peaks is not None:
            combined = peak_score
            mode = "Peak-list match"
        else:
            combined = cell_score
            mode = "Cell match"
        context = {
            "search_mode": mode,
            "combined_score": combined,
            "peak_score": peak_score,
            "cell_score": cell_score,
            "peak_match": row.get("peak_match", {}),
        }
        candidate = candidate_record_from_library_row(row, context)
        candidate["overall_score"] = combined
        candidate["classification"] = _classify_candidate(context)
        ranked.append(candidate)
    ranked.sort(key=lambda item: -float(item.get("overall_score", 0.0)))
    return {
        "search_mode": "combined" if observed_peaks is not None and accepted_cell is not None else ("peaks" if observed_peaks is not None else "cell"),
        "candidate_count": len(ranked[:limit]),
        "candidates": ranked[:limit],
        "scientific_boundary": (
            "Structure matching suggests possible CIFs only. Validation requires whole-pattern refinement, chemical plausibility checks, and comparison with alternatives."
        ),
    }


def _classify_candidate(context: dict) -> str:
    score = float(context.get("combined_score", 0.0))
    peak_score = float(context.get("peak_score", 0.0))
    cell_score = float(context.get("cell_score", 0.0))
    if score >= 75 and (peak_score >= 50 or cell_score >= 70):
        return "Strong candidate — validate"
    if score >= 45:
        return "Possible match — review"
    return "Weak match — low priority"

from .cif_library import query_library_candidates, fingerprint_peak_match, _cell_distance_score


def _score_cell_similarity(row: dict, accepted_cell: dict | None) -> float:
    if not accepted_cell:
        return 0.0
    try:
        distance = _cell_distance_score(accepted_cell, row, CellSearchTolerance())
    except Exception:
        return 0.0
    return max(0.0, 100.0 - 12.0 * float(distance))


def _score_chemistry(row: dict, required_elements: Iterable[str], excluded_elements: Iterable[str] = ()) -> float:
    elements = set(row.get("elements", []))
    required = {str(el).strip().capitalize() for el in required_elements if str(el).strip()}
    excluded = {str(el).strip().capitalize() for el in excluded_elements if str(el).strip()}
    if excluded and elements.intersection(excluded):
        return 0.0
    if not required:
        return 60.0 if elements else 0.0
    if required.issubset(elements):
        return 100.0
    overlap = len(required.intersection(elements)) / max(len(required), 1)
    return 100.0 * overlap


def _score_density(row: dict) -> float:
    density = row.get("density_g_cm3")
    try:
        density = float(density)
    except (TypeError, ValueError):
        return 50.0
    if 0.2 <= density <= 12.0:
        return 100.0
    if 0.05 <= density < 0.2 or 12.0 < density <= 25.0:
        return 40.0
    return 0.0


def _score_symmetry(row: dict, accepted_cell: dict | None = None) -> float:
    crystal_system = str(row.get("crystal_system") or "Unknown").lower()
    if crystal_system and crystal_system != "unknown":
        return 80.0
    space_group = str(row.get("space_group") or "Unknown").lower()
    return 60.0 if space_group and space_group != "unknown" else 20.0


def _score_plausibility(row: dict) -> float:
    value = row.get("plausibility_score")
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        status = str(row.get("plausibility_status") or "").lower()
        return 75.0 if "good" in status or "ok" in status else 50.0


def intelligent_match_structure_candidates(
    db_path: str | Path,
    *,
    observed_peaks: Iterable[dict] | None = None,
    accepted_cell: dict | None = None,
    required_elements: Iterable[str] = (),
    excluded_elements: Iterable[str] = (),
    material_types: Iterable[str] = (),
    peak_tolerance_deg: float = 0.15,
    limit: int = 25,
    weights: dict | None = None,
) -> dict:
    """Rank CIF candidates with peak, cell, chemistry, density and plausibility scores."""
    weights = weights or {
        "peak": 0.35,
        "cell": 0.25,
        "chemistry": 0.15,
        "density": 0.10,
        "symmetry": 0.10,
        "plausibility": 0.05,
    }
    rows = query_library_candidates(
        db_path,
        required_elements=required_elements,
        excluded_elements=excluded_elements,
        material_types=material_types,
        limit=max(int(limit) * 200, 1000),
    )
    ranked = []
    for row in rows:
        peak_match = {}
        peak_score = 0.0
        if observed_peaks is not None:
            peak_match = fingerprint_peak_match(
                observed_peaks,
                row.get("fingerprint", {}) or {},
                tolerance_deg=peak_tolerance_deg,
            )
            peak_score = float(peak_match.get("overall_score", 0.0))
        cell_score = _score_cell_similarity(row, accepted_cell)
        chemistry_score = _score_chemistry(row, required_elements, excluded_elements)
        density_score = _score_density(row)
        symmetry_score = _score_symmetry(row, accepted_cell)
        plausibility_score = _score_plausibility(row)
        overall = (
            weights.get("peak", 0.0) * peak_score
            + weights.get("cell", 0.0) * cell_score
            + weights.get("chemistry", 0.0) * chemistry_score
            + weights.get("density", 0.0) * density_score
            + weights.get("symmetry", 0.0) * symmetry_score
            + weights.get("plausibility", 0.0) * plausibility_score
        )
        context = {
            "search_mode": "Intelligent multi-score CIF matching",
            "combined_score": float(overall),
            "peak_score": peak_score,
            "cell_score": cell_score,
            "chemistry_score": chemistry_score,
            "density_score": density_score,
            "symmetry_score": symmetry_score,
            "plausibility_score": plausibility_score,
            "weights": dict(weights),
            "peak_match": peak_match,
            "material_type": row.get("material_type"),
            "family_key": row.get("family_key"),
        }
        candidate = candidate_record_from_library_row(row, context)
        candidate.update(
            {
                "overall_score": float(overall),
                "classification": _classify_intelligent_candidate(context),
                "structure_family_key": row.get("family_key"),
                "material_type": row.get("material_type"),
                "component_scores": {
                    "peak": peak_score,
                    "cell": cell_score,
                    "chemistry": chemistry_score,
                    "density": density_score,
                    "symmetry": symmetry_score,
                    "plausibility": plausibility_score,
                },
            }
        )
        ranked.append(candidate)
    ranked.sort(key=lambda item: -float(item.get("overall_score", 0.0)))
    return {
        "search_mode": "intelligent",
        "candidate_count": len(ranked[:limit]),
        "weights": dict(weights),
        "filters": {
            "required_elements": list(required_elements),
            "excluded_elements": list(excluded_elements),
            "material_types": list(material_types),
        },
        "candidates": ranked[: int(limit)],
        "scientific_boundary": (
            "Intelligent CIF matching ranks possible structures; it does not identify a phase without whole-pattern validation and chemical review."
        ),
    }


def _classify_intelligent_candidate(context: dict) -> str:
    score = float(context.get("combined_score", 0.0))
    peak = float(context.get("peak_score", 0.0))
    cell = float(context.get("cell_score", 0.0))
    chemistry = float(context.get("chemistry_score", 0.0))
    if score >= 80 and chemistry >= 80 and (peak >= 55 or cell >= 80):
        return "Strong structure candidate — validate"
    if score >= 55:
        return "Possible structure match — review"
    return "Weak structure match — low priority"
