"""Crystal-chemistry QA for explicit periodic atom-site models.

The calculations in this module are diagnostic.  They never modify coordinates,
occupancies, the unit cell, or refinement results.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from itertools import product
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .crystallography import ATOMIC_NUMBER, direct_metric_tensor
from .structure_validation import (
    COMMON_METAL_O_RANGES,
    COVALENT_RADII,
    METALS,
    composition_formula_from_sites,
    validate_crystal_structure,
)


REPORT_SCHEMA = "afruz.crystal-chemistry-qa.v1"

# Gagne & Hawthorne, Acta Cryst. B71 (2015) 562-578, table 2 (GRG values).
# These parameter pairs must be used together in s = exp((R0 - R) / B).
# https://doi.org/10.1107/S2052520615016297
BOND_VALENCE_PARAMETERS = {
    ("Fe", 2, "O"): (1.658, 0.447),
    ("Fe", 3, "O"): (1.766, 0.360),
    ("Ni", 2, "O"): (1.689, 0.347),
    ("Ni", 4, "O"): (1.734, 0.335),
}

DEFAULT_OXIDATION_STATES = {"Fe": 3, "Ni": 2}
NONMETAL_OXIDATION_STATES = {
    "H": 1, "B": 3, "C": 4, "N": -3, "O": -2, "F": -1,
    "P": 5, "S": 6, "Cl": -1, "Br": -1, "I": -1,
}

_IMAGE_SHIFTS = np.asarray(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=int,
)


def _site_id(atom: dict, index: int) -> str:
    label = str(atom.get("label") or atom.get("element") or "site")
    return f"{label} [{index + 1}]"


def _basis(cell: dict) -> np.ndarray:
    return np.linalg.cholesky(direct_metric_tensor(cell))


def _minimum_image(
    first: Iterable[float], second: Iterable[float], basis: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return distance, first-to-second fractional vector, and image shift.

    Searching adjacent images is reliable for skewed conventional cells where
    independently rounding fractional components need not find the shortest image.
    """
    raw = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    vectors = raw[None, :] + _IMAGE_SHIFTS
    cartesian = vectors @ basis
    squared = np.einsum("ij,ij->i", cartesian, cartesian)
    best = int(np.argmin(squared))
    return float(math.sqrt(float(squared[best]))), vectors[best], _IMAGE_SHIFTS[best]


def _minimum_image_distance_matrix(
    sources: np.ndarray, targets: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    """Vectorized minimum-image distances for two equally labelled site sets."""
    raw = targets[None, :, :] - sources[:, None, :]
    best = np.full(raw.shape[:2], np.inf, dtype=float)
    for shift in _IMAGE_SHIFTS:
        cartesian = (raw + shift) @ basis
        squared = np.einsum("ijk,ijk->ij", cartesian, cartesian)
        np.minimum(best, squared, out=best)
    return np.sqrt(best)


def _atom_fractional(atom: dict) -> np.ndarray:
    return np.asarray([atom[axis] for axis in ("x", "y", "z")], dtype=float)


def _stats(values: Iterable[float]) -> dict:
    data = np.asarray(list(values), dtype=float)
    if not data.size:
        return {"minimum": None, "mean": None, "maximum": None, "standard_deviation": None}
    return {
        "minimum": float(np.min(data)),
        "mean": float(np.mean(data)),
        "maximum": float(np.max(data)),
        "standard_deviation": float(np.std(data)),
    }


def _coordination_geometry(vectors: list[np.ndarray], distances: list[float]) -> dict:
    angles = []
    for index, first in enumerate(vectors):
        for second in vectors[index + 1:]:
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator <= 0:
                continue
            cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
            angles.append(float(math.degrees(math.acos(cosine))))
    mean_distance = float(np.mean(distances)) if distances else 0.0
    distortion = (
        float(np.mean(np.abs(np.asarray(distances) - mean_distance)) / mean_distance)
        if mean_distance > 0 else None
    )
    result = {
        "assignment": "unclassified",
        "bond_length_distortion_index": distortion,
        "angle_statistics_deg": _stats(angles),
        "cis_angle_rms_deviation_deg": None,
        "trans_angle_rms_deviation_deg": None,
        "ideal_angle_rms_deviation_deg": None,
    }
    if len(vectors) == 6 and len(angles) == 15:
        ordered = np.sort(np.asarray(angles))
        cis = ordered[:12]
        trans = ordered[12:]
        cis_rms = float(np.sqrt(np.mean((cis - 90.0) ** 2)))
        trans_rms = float(np.sqrt(np.mean((trans - 180.0) ** 2)))
        result.update(
            assignment="six-coordinate / octahedral comparison",
            cis_angle_rms_deviation_deg=cis_rms,
            trans_angle_rms_deviation_deg=trans_rms,
            ideal_angle_rms_deviation_deg=float(
                math.sqrt((12.0 * cis_rms**2 + 3.0 * trans_rms**2) / 15.0)
            ),
        )
    elif len(vectors) == 4 and len(angles) == 6:
        values = np.asarray(angles)
        tetra_rms = float(np.sqrt(np.mean((values - 109.4712) ** 2)))
        ordered = np.sort(values)
        square_rms = float(np.sqrt(
            (np.sum((ordered[:4] - 90.0) ** 2) + np.sum((ordered[4:] - 180.0) ** 2)) / 6.0
        ))
        if tetra_rms <= square_rms:
            result.update(assignment="four-coordinate / tetrahedral comparison", ideal_angle_rms_deviation_deg=tetra_rms)
        else:
            result.update(assignment="four-coordinate / square-planar comparison", ideal_angle_rms_deviation_deg=square_rms)
    return result


def _metal_coordination(atoms: list[dict], basis: np.ndarray) -> list[dict]:
    records = []
    oxygen_indices = [
        index for index, atom in enumerate(atoms)
        if str(atom.get("element") or "").capitalize() == "O"
    ]
    for center_index, center in enumerate(atoms):
        element = str(center.get("element") or "").capitalize()
        if element not in METALS:
            continue
        lower, upper = COMMON_METAL_O_RANGES.get(element, (1.65, 2.85))
        center_fractional = _atom_fractional(center)
        ligands = []
        vectors = []
        for oxygen_index in oxygen_indices:
            oxygen = atoms[oxygen_index]
            distance, vector_fractional, image_shift = _minimum_image(
                center_fractional, _atom_fractional(oxygen), basis
            )
            if 0.45 <= distance <= upper:
                vector_cartesian = vector_fractional @ basis
                vectors.append(vector_cartesian)
                ligands.append({
                    "site": _site_id(oxygen, oxygen_index),
                    "label": str(oxygen.get("label") or f"O{oxygen_index + 1}"),
                    "source_site_index": oxygen_index,
                    "periodic_image_shift": [int(value) for value in image_shift],
                    "distance_angstrom": distance,
                    "within_reference_range": bool(distance >= lower),
                })
        order = np.argsort([row["distance_angstrom"] for row in ligands]) if ligands else []
        ligands = [ligands[int(index)] for index in order]
        vectors = [vectors[int(index)] for index in order]
        distances = [row["distance_angstrom"] for row in ligands]
        geometry = _coordination_geometry(vectors, distances)
        records.append({
            "site": _site_id(center, center_index),
            "label": str(center.get("label") or f"{element}{center_index + 1}"),
            "source_site_index": center_index,
            "element": element,
            "occupancy": float(center.get("occupancy", 1.0)),
            "fractional_coordinates": [float(value) for value in center_fractional],
            "oxygen_coordination_number": len(ligands),
            "oxygen_bond_length_statistics_angstrom": _stats(distances),
            "coordination_geometry": geometry,
            "oxygen_ligands": ligands,
        })
    return records


def _bond_valence_analysis(
    metal_sites: list[dict], expected_oxidation_states: dict[str, int]
) -> dict:
    site_rows = []
    by_element: dict[str, list[float]] = defaultdict(list)
    for site in metal_sites:
        element = site["element"]
        distances = [row["distance_angstrom"] for row in site["oxygen_ligands"]]
        available = {}
        for (parameter_element, oxidation, ligand), (r0, b_value) in BOND_VALENCE_PARAMETERS.items():
            if parameter_element != element or ligand != "O":
                continue
            bvs = float(sum(math.exp((r0 - distance) / b_value) for distance in distances))
            available[str(oxidation)] = {
                "oxidation_state": oxidation,
                "bond_valence_sum_vu": bvs,
                "difference_from_assumed_valence_vu": bvs - oxidation,
                "R0_angstrom": r0,
                "B_angstrom": b_value,
            }
        expected = expected_oxidation_states.get(element)
        selected = available.get(str(expected)) if expected is not None else None
        if selected is not None:
            by_element[element].append(float(selected["bond_valence_sum_vu"]))
        site_rows.append({
            "site": site["site"],
            "element": element,
            "oxygen_coordination_number": site["oxygen_coordination_number"],
            "assumed_oxidation_state": expected,
            "selected_bond_valence_sum_vu": None if selected is None else selected["bond_valence_sum_vu"],
            "selected_difference_vu": None if selected is None else selected["difference_from_assumed_valence_vu"],
            "available_parameter_scenarios": available,
        })
    summaries = {}
    for element, values in by_element.items():
        summaries[element] = {
            "assumed_oxidation_state": expected_oxidation_states[element],
            "site_count": len(values),
            "bond_valence_sum_statistics_vu": _stats(values),
        }
    return {
        "equation": "s_ij = exp((R0 - R_ij) / B); BVS_i = sum(s_ij)",
        "parameter_source": {
            "authors": "O. C. Gagne and F. C. Hawthorne",
            "title": "Comprehensive derivation of bond-valence parameters for ion pairs involving oxygen",
            "journal": "Acta Crystallographica Section B 71 (2015) 562-578",
            "doi": "10.1107/S2052520615016297",
            "url": "https://doi.org/10.1107/S2052520615016297",
            "parameter_set": "GRG R0 and B values from table 2",
        },
        "assumed_oxidation_states": expected_oxidation_states,
        "site_results": site_rows,
        "element_summaries": summaries,
        "interpretation_limit": (
            "Bond-valence sums are empirical plausibility diagnostics. Oxidation states are assumptions, "
            "and mixed valence, disorder, protonation and an incomplete coordination shell can change the result."
        ),
    }


def _functional_groups(atoms: list[dict], basis: np.ndarray) -> dict:
    oxygen_indices = [i for i, atom in enumerate(atoms) if str(atom.get("element") or "").capitalize() == "O"]
    hydrogen_indices = [i for i, atom in enumerate(atoms) if str(atom.get("element") or "").capitalize() == "H"]
    carbon_indices = [i for i, atom in enumerate(atoms) if str(atom.get("element") or "").capitalize() == "C"]
    oxygen_hydrogens: dict[int, list[int]] = defaultdict(list)
    hydrogen_assignments = []
    for hydrogen_index in hydrogen_indices:
        candidates = []
        for oxygen_index in oxygen_indices:
            distance, _, shift = _minimum_image(
                _atom_fractional(atoms[hydrogen_index]), _atom_fractional(atoms[oxygen_index]), basis
            )
            if distance <= 1.25:
                candidates.append((distance, oxygen_index, shift))
        if candidates:
            distance, oxygen_index, shift = min(candidates, key=lambda row: row[0])
            oxygen_hydrogens[oxygen_index].append(hydrogen_index)
            hydrogen_assignments.append({
                "hydrogen": _site_id(atoms[hydrogen_index], hydrogen_index),
                "oxygen": _site_id(atoms[oxygen_index], oxygen_index),
                "distance_angstrom": distance,
                "periodic_image_shift": [int(value) for value in shift],
            })

    carbon_oxygen: dict[int, list[tuple[int, float]]] = defaultdict(list)
    oxygen_carbons: dict[int, list[int]] = defaultdict(list)
    for carbon_index in carbon_indices:
        for oxygen_index in oxygen_indices:
            distance, _, _ = _minimum_image(
                _atom_fractional(atoms[carbon_index]), _atom_fractional(atoms[oxygen_index]), basis
            )
            if 1.05 <= distance <= 1.60:
                carbon_oxygen[carbon_index].append((oxygen_index, distance))
                oxygen_carbons[oxygen_index].append(carbon_index)

    oxygen_types = Counter()
    oxygen_rows = []
    for oxygen_index in oxygen_indices:
        h_count = len(oxygen_hydrogens.get(oxygen_index, []))
        c_count = len(oxygen_carbons.get(oxygen_index, []))
        if c_count:
            assignment = "carbon-bound oxygen"
        elif h_count >= 2:
            assignment = "water-like oxygen"
        elif h_count == 1:
            assignment = "hydroxyl-like oxygen"
        else:
            assignment = "unprotonated oxygen / unassigned"
        oxygen_types[assignment] += 1
        oxygen_rows.append({
            "site": _site_id(atoms[oxygen_index], oxygen_index),
            "assigned_hydrogen_count": h_count,
            "nearby_carbon_count": c_count,
            "assignment": assignment,
        })
    carbon_rows = []
    for carbon_index in carbon_indices:
        neighbors = carbon_oxygen.get(carbon_index, [])
        assignment = "carbonate-like CO3 group" if len(neighbors) == 3 else f"C-O coordination {len(neighbors)}"
        carbon_rows.append({
            "site": _site_id(atoms[carbon_index], carbon_index),
            "oxygen_coordination_number": len(neighbors),
            "assignment": assignment,
            "oxygen_distances_angstrom": [float(row[1]) for row in sorted(neighbors, key=lambda row: row[1])],
        })
    return {
        "oxygen_site_assignments": dict(oxygen_types),
        "assigned_o_h_bond_count": len(hydrogen_assignments),
        "unassigned_hydrogen_count": len(hydrogen_indices) - len(hydrogen_assignments),
        "carbonate_like_group_count": sum(row["assignment"].startswith("carbonate-like") for row in carbon_rows),
        "oxygen_sites": oxygen_rows,
        "carbon_sites": carbon_rows,
        "o_h_assignments": hydrogen_assignments,
        "assignment_limit": (
            "Water, hydroxyl and carbonate labels are distance-based connectivity descriptions, "
            "not refined protonation states or spectroscopic identification."
        ),
    }


def _charge_balance(atoms: list[dict], oxidation_states: dict[str, int]) -> dict:
    composition: dict[str, float] = defaultdict(float)
    for atom in atoms:
        element = str(atom.get("element") or "").capitalize()
        if element in ATOMIC_NUMBER:
            composition[element] += max(0.0, float(atom.get("occupancy", 1.0)))
    assumptions = dict(NONMETAL_OXIDATION_STATES)
    assumptions.update(oxidation_states)
    contributions = {}
    unresolved = []
    total = 0.0
    for element, count in sorted(composition.items()):
        if element not in assumptions:
            unresolved.append(element)
            continue
        charge = float(assumptions[element])
        contribution = count * charge
        contributions[element] = {
            "occupied_site_count": count,
            "assumed_oxidation_state": charge,
            "charge_contribution": contribution,
        }
        total += contribution
    if unresolved:
        status = "indeterminate — oxidation-state assumptions are missing"
    elif abs(total) <= 0.20:
        status = "neutral under the stated oxidation-state assumptions"
    else:
        status = "not neutral under the stated oxidation-state assumptions"
    return {
        "status": status,
        "net_charge_per_explicit_cell": total,
        "contributions": contributions,
        "unresolved_elements": unresolved,
        "interpretation_limit": (
            "This is formal electron counting from occupied sites. It does not determine oxidation states "
            "and cannot resolve mixed valence or disordered solvent by itself."
        ),
    }


def _wrapped_signed(vector: np.ndarray) -> np.ndarray:
    value = (np.asarray(vector, dtype=float) + 0.5) % 1.0 - 0.5
    value[np.isclose(value, 0.5, atol=1e-8)] = -0.5
    return value


def _translation_key(vector: np.ndarray) -> tuple[float, float, float]:
    signed = _wrapped_signed(vector)
    inverse = _wrapped_signed(-signed)
    first = tuple(float(value) for value in np.round(signed, 6))
    second = tuple(float(value) for value in np.round(inverse, 6))
    return min(first, second)


def _translation_order(vector: np.ndarray, maximum: int = 12) -> int | None:
    fractions = [Fraction(float(value % 1.0)).limit_denominator(maximum) for value in vector]
    for fraction, value in zip(fractions, vector):
        if abs(float(fraction) - float(value % 1.0)) > 2e-4:
            return None
    order = 1
    for fraction in fractions:
        order = math.lcm(order, fraction.denominator)
    return order if 1 < order <= maximum else None


def _translation_match(
    atoms: list[dict], translation: np.ndarray, basis: np.ndarray, tolerance: float
) -> tuple[float, float | None]:
    return _operation_match(atoms, np.eye(3, dtype=int), translation, basis, tolerance)


def _operation_match(
    atoms: list[dict], operation: np.ndarray, translation: np.ndarray,
    basis: np.ndarray, tolerance: float,
) -> tuple[float, float | None]:
    by_element: dict[str, list[int]] = defaultdict(list)
    for index, atom in enumerate(atoms):
        by_element[str(atom.get("element") or "").capitalize()].append(index)
    matched_weight = 0.0
    total_weight = sum(max(0.0, float(atom.get("occupancy", 1.0))) for atom in atoms)
    matched_distances = []
    for indices in by_element.values():
        size = len(indices)
        sources = np.asarray([
            (_atom_fractional(atoms[index]) @ operation + translation) % 1.0
            for index in indices
        ])
        targets = np.asarray([_atom_fractional(atoms[index]) for index in indices])
        costs = _minimum_image_distance_matrix(sources, targets, basis)
        occupancies = np.asarray([float(atoms[index].get("occupancy", 1.0)) for index in indices])
        incompatible = np.abs(occupancies[:, None] - occupancies[None, :]) > 0.05
        costs[incompatible] = 1e6
        rows, columns = linear_sum_assignment(costs)
        for source_row, target_column in zip(rows, columns):
            distance = float(costs[source_row, target_column])
            if distance <= tolerance:
                source_index = indices[int(source_row)]
                matched_weight += max(0.0, float(atoms[source_index].get("occupancy", 1.0)))
                matched_distances.append(distance)
    fraction = matched_weight / total_weight if total_weight > 0 else 0.0
    rms = float(np.sqrt(np.mean(np.square(matched_distances)))) if matched_distances else None
    return float(fraction), rms


@lru_cache(maxsize=1)
def _small_unimodular_matrices() -> tuple[tuple[int, ...], ...]:
    matrices = []
    for values in product((-1, 0, 1), repeat=9):
        matrix = np.asarray(values, dtype=int).reshape(3, 3)
        determinant = int(round(float(np.linalg.det(matrix))))
        if abs(determinant) == 1:
            matrices.append(values)
    return tuple(matrices)


def _point_symmetry_screen(
    atoms: list[dict], basis: np.ndarray, tolerance: float, metric_tolerance: float = 0.005
) -> list[dict]:
    metric = basis @ basis.T
    scale = max(float(np.max(np.abs(metric))), 1e-12)
    identity = np.eye(3, dtype=int)
    operations = []
    for values in _small_unimodular_matrices():
        operation = np.asarray(values, dtype=int).reshape(3, 3)
        if np.array_equal(operation, identity):
            continue
        mismatch = float(np.max(np.abs(operation @ metric @ operation.T - metric)) / scale)
        if mismatch <= metric_tolerance:
            operations.append((mismatch, operation))
    operations.sort(key=lambda row: row[0])

    groups: dict[tuple[str, float], list[int]] = defaultdict(list)
    for index, atom in enumerate(atoms):
        groups[(
            str(atom.get("element") or "").capitalize(),
            round(float(atom.get("occupancy", 1.0)), 3),
        )].append(index)
    reference = min(groups.values(), key=len)
    anchor_index = reference[0]
    anchor = _atom_fractional(atoms[anchor_index])
    candidates = []
    for metric_mismatch, operation in operations[:64]:
        transformed_anchor = anchor @ operation
        tried = set()
        best = None
        for target_index in reference[:8]:
            translation = _wrapped_signed(_atom_fractional(atoms[target_index]) - transformed_anchor)
            key = tuple(float(value) for value in np.round(translation, 6))
            if key in tried:
                continue
            tried.add(key)
            fraction, rms = _operation_match(atoms, operation, translation, basis, tolerance)
            row = {
                "fractional_matrix": [[int(value) for value in line] for line in operation],
                "fractional_translation": [float(value) for value in translation],
                "matched_occupied_fraction": fraction,
                "rms_mapping_distance_angstrom": rms,
                "metric_relative_mismatch": metric_mismatch,
            }
            if best is None or (fraction, -(rms or math.inf)) > (
                best["matched_occupied_fraction"], -(best["rms_mapping_distance_angstrom"] or math.inf)
            ):
                best = row
        if best is not None and best["matched_occupied_fraction"] >= 0.60:
            candidates.append(best)
    candidates.sort(key=lambda row: (-row["matched_occupied_fraction"], row["rms_mapping_distance_angstrom"] or math.inf))
    return candidates[:16]


def _metric_description(cell: dict, tolerance: float = 0.003) -> str:
    a, b, c = (float(cell[key]) for key in ("a", "b", "c"))
    alpha, beta, gamma = (float(cell[key]) for key in ("alpha", "beta", "gamma"))
    rel_close = lambda first, second: abs(first - second) <= tolerance * max(first, second)
    angle_close = lambda first, second: abs(first - second) <= 0.20
    right = [angle_close(value, 90.0) for value in (alpha, beta, gamma)]
    if rel_close(a, b) and rel_close(b, c) and all(right):
        return "cubic metric"
    if rel_close(a, b) and right[0] and right[1] and angle_close(gamma, 120.0):
        return "hexagonal metric"
    if rel_close(a, b) and all(right):
        return "tetragonal metric"
    if all(right):
        return "orthorhombic metric"
    if sum(right) == 2:
        return "monoclinic-like metric"
    return "triclinic/oblique metric"


def _symmetry_screen(atoms: list[dict], cell: dict, basis: np.ndarray, tolerance: float = 0.12) -> dict:
    groups: dict[tuple[str, float], list[int]] = defaultdict(list)
    for index, atom in enumerate(atoms):
        key = (str(atom.get("element") or "").capitalize(), round(float(atom.get("occupancy", 1.0)), 3))
        groups[key].append(index)
    candidate_vectors = {}
    useful_groups = sorted((indices for indices in groups.values() if len(indices) >= 2), key=len)[:3]
    for indices in useful_groups:
        for anchor_index in indices[:3]:
            anchor = _atom_fractional(atoms[anchor_index])
            for target_index in indices:
                vector = _wrapped_signed(_atom_fractional(atoms[target_index]) - anchor)
                if np.linalg.norm(vector @ basis) <= tolerance:
                    continue
                candidate_vectors.setdefault(_translation_key(vector), vector)
    evaluated = []
    for vector in list(candidate_vectors.values())[:36]:
        fraction, rms = _translation_match(atoms, vector, basis, tolerance)
        if fraction < 0.60:
            continue
        order = _translation_order(vector)
        axis_hint = None
        nonzero = np.flatnonzero(np.abs(vector) > 2e-4)
        if order and len(nonzero) == 1:
            axis = "abc"[int(nonzero[0])]
            reduced_length = float(cell[axis]) / order
            axis_hint = f"possible {order}-fold supercell along {axis}; reduced {axis} about {reduced_length:.5g} Å"
        evaluated.append({
            "fractional_translation": [float(value) for value in vector],
            "matched_occupied_fraction": fraction,
            "rms_mapping_distance_angstrom": rms,
            "commensurate_order": order,
            "cell_reduction_hint": axis_hint,
        })
    evaluated.sort(key=lambda row: (-row["matched_occupied_fraction"], row["rms_mapping_distance_angstrom"] or math.inf))
    strong = [row for row in evaluated if row["matched_occupied_fraction"] >= 0.90]
    point_candidates = _point_symmetry_screen(atoms, basis, tolerance)
    strong_points = [row for row in point_candidates if row["matched_occupied_fraction"] >= 0.90]
    if strong:
        status = "strong non-identity translation found — test a reduced cell and higher space group"
    elif strong_points:
        status = "strong lattice-compatible point operation found — test a higher space group"
    elif evaluated:
        status = "partial translational pseudosymmetry found — expert review recommended"
    elif point_candidates:
        status = "partial lattice-compatible pseudosymmetry found — expert review recommended"
    else:
        status = "no strong candidate operation found at the stated tolerance"
    return {
        "status": status,
        "cell_metric": _metric_description(cell),
        "mapping_tolerance_angstrom": tolerance,
        "translation_candidates": evaluated[:12],
        "strong_translation_count": len(strong),
        "point_operation_candidates": point_candidates,
        "strong_point_operation_count": len(strong_points),
        "scope_limit": (
            "This lightweight screen tests translational repetition and small-integer lattice operations "
            "against element/occupancy-labelled sites. It does not assign a space group or exhaustively test "
            "all settings, systematic absences or coordinate uncertainties. "
            "Confirm candidates with a crystallographic symmetry program and the diffraction data."
        ),
    }


def _refinement_evidence(structure: dict) -> dict:
    provenance = structure.get("provenance") if isinstance(structure.get("provenance"), dict) else {}
    keys = (
        "refinement_success", "refinement_message", "rwp_percent", "rp_percent", "rexp_percent",
        "goodness_of_fit", "reduced_chi_square", "condition_number",
        "maximum_absolute_correlation", "strong_correlation_pairs",
    )
    values = {key: provenance.get(key) for key in keys if provenance.get(key) is not None}
    return {
        "available": bool(values),
        "values": values,
        "interpretation": (
            "Fit statistics and parameter correlations are copied from the saved refinement result when available. "
            "Re-run refinement after changing the structural model to test stability."
        ),
    }


def analyze_crystal_chemistry(
    structure: dict,
    *,
    expected_oxidation_states: dict[str, int] | None = None,
    symmetry_tolerance_angstrom: float = 0.12,
) -> dict:
    """Create a deterministic crystal-chemistry QA record from a model or CIF dict."""
    atoms = [dict(atom) for atom in structure.get("atoms") or []]
    if not atoms:
        raise ValueError("Crystal Chemistry QA requires atomic coordinates.")
    cell = dict(structure.get("cell") or {})
    basis = _basis(cell)
    expected = dict(DEFAULT_OXIDATION_STATES)
    if expected_oxidation_states:
        expected.update({str(key).capitalize(): int(value) for key, value in expected_oxidation_states.items()})

    validation = validate_crystal_structure(structure)
    metal_sites = _metal_coordination(atoms, basis)
    bond_valence = _bond_valence_analysis(metal_sites, expected)
    functional_groups = _functional_groups(atoms, basis)
    charge = _charge_balance(atoms, expected)
    symmetry = _symmetry_screen(atoms, cell, basis, tolerance=float(symmetry_tolerance_angstrom))

    findings = []
    for site in metal_sites:
        coordination = site["oxygen_coordination_number"]
        if site["element"] in {"Fe", "Ni"} and coordination != 6:
            findings.append({
                "severity": "review",
                "code": "metal_coordination",
                "message": f"{site['site']} has {coordination} O neighbors; review the expected six-coordinate shell.",
            })
        geometry = site["coordination_geometry"]
        distortion = geometry.get("bond_length_distortion_index")
        angle_rms = geometry.get("ideal_angle_rms_deviation_deg")
        if coordination == 6 and ((distortion is not None and distortion > 0.06) or (angle_rms is not None and angle_rms > 15.0)):
            findings.append({
                "severity": "review",
                "code": "polyhedral_distortion",
                "message": f"{site['site']} is strongly distorted relative to an ideal octahedron.",
            })
    for row in bond_valence["site_results"]:
        delta = row.get("selected_difference_vu")
        if delta is not None and abs(float(delta)) > 0.65:
            findings.append({
                "severity": "review",
                "code": "bond_valence_mismatch",
                "message": f"{row['site']} BVS differs from assumed {row['assumed_oxidation_state']}+ by {float(delta):+.2f} v.u.",
            })
    if charge["unresolved_elements"]:
        findings.append({"severity": "info", "code": "charge_indeterminate", "message": charge["status"] + "."})
    elif abs(float(charge["net_charge_per_explicit_cell"])) > 0.20:
        findings.append({
            "severity": "review", "code": "charge_imbalance",
            "message": f"Formal net charge is {float(charge['net_charge_per_explicit_cell']):+.3g} per explicit cell under the stated assumptions.",
        })
    if symmetry["strong_translation_count"]:
        best = symmetry["translation_candidates"][0]
        hint = best.get("cell_reduction_hint") or "test a reduced cell"
        findings.append({
            "severity": "review", "code": "hidden_translation",
            "message": f"A translation maps {100.0 * best['matched_occupied_fraction']:.1f}% of occupied sites; {hint}.",
        })
    elif symmetry["strong_point_operation_count"]:
        best = symmetry["point_operation_candidates"][0]
        findings.append({
            "severity": "review", "code": "hidden_point_symmetry",
            "message": (
                f"A lattice-compatible point operation maps {100.0 * best['matched_occupied_fraction']:.1f}% "
                "of occupied sites; test a higher space group."
            ),
        })
    validation_problems = list(validation.get("errors") or []) + list(validation.get("warnings") or [])
    if validation_problems:
        findings.append({
            "severity": "review", "code": "plausibility",
            "message": f"Topology-aware plausibility check reports {len(validation_problems)} issue(s).",
        })
    if not findings:
        findings.append({
            "severity": "pass", "code": "screen_pass",
            "message": "No automatic crystal-chemistry review trigger was found within the stated assumptions.",
        })
    review_count = sum(row["severity"] == "review" for row in findings)
    status = "review recommended" if review_count else "checks passed within stated assumptions"
    coordination_counts: dict[str, Counter] = defaultdict(Counter)
    for site in metal_sites:
        coordination_counts[site["element"]][site["oxygen_coordination_number"]] += 1

    composition = composition_formula_from_sites(atoms)
    return {
        "schema": REPORT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "status": status,
            "review_finding_count": review_count,
            "structure_name": str(structure.get("data_name") or "Crystal structure"),
            "occupied_cell_composition": composition,
            "atom_site_count": len(atoms),
            "metal_site_count": len(metal_sites),
        },
        "structure": {
            "data_name": str(structure.get("data_name") or "Crystal structure"),
            "supplied_formula": str(structure.get("formula") or ""),
            "occupied_cell_composition": composition,
            "space_group": str(structure.get("space_group") or "Unknown"),
            "crystal_system": str(structure.get("crystal_system") or "Unknown"),
            "cell": cell,
            "source_path": str(structure.get("source_path") or ""),
        },
        "coordination_summary": {
            element: {str(coordination): count for coordination, count in sorted(counts.items())}
            for element, counts in sorted(coordination_counts.items())
        },
        "metal_sites": metal_sites,
        "bond_valence": bond_valence,
        "formal_charge_balance": charge,
        "functional_group_screen": functional_groups,
        "symmetry_screen": symmetry,
        "topology_validation": {
            "status": validation.get("status"),
            "scores": validation.get("scores"),
            "errors": validation.get("errors"),
            "warnings": validation.get("warnings"),
            "vdw_overlap_count": validation.get("vdw_overlap_count"),
            "hydrogen_bond_count": validation.get("hydrogen_bond_count"),
            "topology_excluded_pair_count": validation.get("topology_excluded_pair_count"),
        },
        "refinement_evidence": _refinement_evidence(structure),
        "findings": findings,
        "scientific_boundary": (
            "This QA report is a reproducible screening aid. It does not prove atom identity, protonation, "
            "oxidation state, symmetry or structural correctness, and it does not replace diffraction-fit review, "
            "checkCIF/PLATON, spectroscopy, chemical analysis or expert crystallographic assessment."
        ),
    }


def _format_number(value, digits: int = 4, missing: str = "—") -> str:
    if value is None:
        return missing
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_crystal_chemistry_report(report: dict) -> str:
    """Format the full QA record as a readable plain-text laboratory report."""
    summary = report["summary"]
    structure = report["structure"]
    charge = report["formal_charge_balance"]
    groups = report["functional_group_screen"]
    symmetry = report["symmetry_screen"]
    lines = [
        "AFRUZ CRYSTAL CHEMISTRY QA",
        "=" * 78,
        f"Structure: {summary['structure_name']}",
        f"Status: {summary['status'].upper()}",
        f"Occupied cell composition: {summary['occupied_cell_composition']}",
        f"Supplied formula: {structure['supplied_formula'] or 'not supplied'}",
        f"Space group / crystal system: {structure['space_group']} / {structure['crystal_system']}",
        f"Atom sites: {summary['atom_site_count']}   Metal sites: {summary['metal_site_count']}",
        "",
        "FINDINGS",
        "-" * 78,
    ]
    for finding in report["findings"]:
        lines.append(f"[{finding['severity'].upper():6}] {finding['message']}")
    lines.extend(["", "METAL–O COORDINATION AND POLYHEDRAL GEOMETRY", "-" * 78])
    if not report["metal_sites"]:
        lines.append("No metal centers were found.")
    else:
        lines.append("Site                 CN   M–O min / mean / max (Å)   DI        angle RMS (°)")
        for site in report["metal_sites"]:
            bond = site["oxygen_bond_length_statistics_angstrom"]
            geometry = site["coordination_geometry"]
            lines.append(
                f"{site['site'][:20]:20} {site['oxygen_coordination_number']:>2}   "
                f"{_format_number(bond['minimum'], 3):>6} / {_format_number(bond['mean'], 3):>6} / {_format_number(bond['maximum'], 3):>6}   "
                f"{_format_number(geometry['bond_length_distortion_index'], 4):>7}   "
                f"{_format_number(geometry['ideal_angle_rms_deviation_deg'], 2):>8}"
            )
    lines.extend(["", "BOND-VALENCE SUMS", "-" * 78])
    lines.append("Site                 assumed   BVS (v.u.)   delta (v.u.)")
    for row in report["bond_valence"]["site_results"]:
        lines.append(
            f"{row['site'][:20]:20} {str(row['assumed_oxidation_state'] or '—'):>7}   "
            f"{_format_number(row['selected_bond_valence_sum_vu'], 3):>10}   "
            f"{_format_number(row['selected_difference_vu'], 3):>11}"
        )
    source = report["bond_valence"]["parameter_source"]
    lines.append(f"Parameters: {source['authors']}, {source['journal']}, DOI {source['doi']} ({source['parameter_set']}).")
    lines.extend(["", "CONNECTIVITY AND FORMAL CHARGE", "-" * 78])
    for name, count in sorted(groups["oxygen_site_assignments"].items()):
        lines.append(f"{name}: {count}")
    lines.append(f"Assigned O–H bonds: {groups['assigned_o_h_bond_count']}")
    lines.append(f"Carbonate-like CO3 groups: {groups['carbonate_like_group_count']}")
    lines.append(f"Charge screen: {charge['status']} ({charge['net_charge_per_explicit_cell']:+.3g} e/cell)")
    assumptions = ", ".join(
        f"{element}{int(row['assumed_oxidation_state']):+d}"
        for element, row in charge["contributions"].items()
    )
    lines.append(f"Formal oxidation-state assumptions: {assumptions}")
    lines.extend(["", "TRANSLATIONAL SYMMETRY SCREEN", "-" * 78])
    lines.append(f"Cell metric: {symmetry['cell_metric']}")
    lines.append(f"Result: {symmetry['status']}")
    for candidate in symmetry["translation_candidates"][:6]:
        vector = ", ".join(f"{value:+.6f}" for value in candidate["fractional_translation"])
        lines.append(
            f"t = ({vector}); match {100*candidate['matched_occupied_fraction']:.1f}%; "
            f"RMS {_format_number(candidate['rms_mapping_distance_angstrom'], 4)} Å"
            + (f"; {candidate['cell_reduction_hint']}" if candidate.get("cell_reduction_hint") else "")
        )
    for candidate in symmetry["point_operation_candidates"][:6]:
        matrix = "; ".join(" ".join(f"{value:+d}" for value in row) for row in candidate["fractional_matrix"])
        vector = ", ".join(f"{value:+.6f}" for value in candidate["fractional_translation"])
        lines.append(
            f"W = [{matrix}], t = ({vector}); match {100*candidate['matched_occupied_fraction']:.1f}%; "
            f"RMS {_format_number(candidate['rms_mapping_distance_angstrom'], 4)} Å"
        )
    refinement = report["refinement_evidence"]
    lines.extend(["", "REFINEMENT EVIDENCE", "-" * 78])
    if refinement["available"]:
        for key, value in refinement["values"].items():
            lines.append(f"{key}: {value}")
    else:
        lines.append("No saved refinement fit/correlation metrics were attached to this CIF reference.")
    lines.extend([
        "", "METHOD LIMITS", "-" * 78,
        report["bond_valence"]["interpretation_limit"],
        groups["assignment_limit"],
        symmetry["scope_limit"],
        report["scientific_boundary"],
        "", f"Generated (UTC): {report['generated_utc']}",
    ])
    return "\n".join(lines) + "\n"


def export_crystal_chemistry_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    elif suffix in {".txt", ".md"}:
        path.write_text(format_crystal_chemistry_report(report), encoding="utf-8")
    else:
        raise ValueError("Crystal Chemistry QA reports use .json, .txt, or .md files.")
    return path


def export_crystal_chemistry_bundle(structure: dict, figure_path: str | Path) -> tuple[Path, Path]:
    """Write full JSON and readable text QA companions beside a figure."""
    figure_path = Path(figure_path)
    report = analyze_crystal_chemistry(structure)
    stem = figure_path.with_suffix("")
    json_path = Path(str(stem) + ".chemistry-qa.json")
    text_path = Path(str(stem) + ".chemistry-qa.txt")
    export_crystal_chemistry_report(report, json_path)
    export_crystal_chemistry_report(report, text_path)
    return json_path, text_path
