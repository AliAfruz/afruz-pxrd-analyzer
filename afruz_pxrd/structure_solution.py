from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from .text_export import write_table_txt, write_mapping_txt, write_manifest_txt
from scipy.optimize import least_squares

from .crystallography import (
    ATOMIC_NUMBER,
    CIFImportError,
    direct_metric_tensor,
    d_spacing,
    parse_cif_text,
    two_theta_from_hkl,
)
from .phase_revolution import (
    CRYSTAL_SYSTEM_BY_BRAVAIS,
    PhaseRevolutionError,
    candidate_cell,
)
from .structure_validation import (
    composition_formula_from_sites,
    formula_weight_from_counts,
    parse_formula_counts,
    validate_crystal_structure,
)


STRUCTURE_SOLUTION_INTERFACES = (
    "Generic hkl + provisional CIF",
    "SHELXT reflection-data handoff",
    "EXPO powder-structure handoff",
    "FOX / DASH real-space handoff",
)


# This library is intentionally a pre-screening set, not an exhaustive
# International Tables implementation. Centering conditions are applied for all
# entries. Extra conditions are limited to rules that are explicitly encoded
# below. Final assignment must be checked with a crystallographic engine.
SPACE_GROUP_LIBRARY = (
    {"number": 1, "name": "P 1", "bravais": "Triclinic", "rule": "centering"},
    {"number": 2, "name": "P -1", "bravais": "Triclinic", "rule": "centering"},
    {"number": 3, "name": "P 2", "bravais": "Monoclinic-P", "rule": "centering"},
    {"number": 4, "name": "P 21", "bravais": "Monoclinic-P", "rule": "p21_b"},
    {"number": 10, "name": "P 2/m", "bravais": "Monoclinic-P", "rule": "centering"},
    {"number": 11, "name": "P 21/m", "bravais": "Monoclinic-P", "rule": "p21_b"},
    {"number": 13, "name": "P 2/c", "bravais": "Monoclinic-P", "rule": "h0l_hl_even"},
    {"number": 14, "name": "P 21/c", "bravais": "Monoclinic-P", "rule": "p21c_b"},
    {"number": 5, "name": "C 2", "bravais": "Monoclinic-C", "rule": "centering"},
    {"number": 9, "name": "C c", "bravais": "Monoclinic-C", "rule": "centering"},
    {"number": 12, "name": "C 2/m", "bravais": "Monoclinic-C", "rule": "centering"},
    {"number": 15, "name": "C 2/c", "bravais": "Monoclinic-C", "rule": "centering"},
    {"number": 16, "name": "P 2 2 2", "bravais": "Orthorhombic-P", "rule": "centering"},
    {"number": 19, "name": "P 21 21 21", "bravais": "Orthorhombic-P", "rule": "p212121"},
    {"number": 62, "name": "P n m a", "bravais": "Orthorhombic-P", "rule": "centering"},
    {"number": 61, "name": "P b c a", "bravais": "Orthorhombic-P", "rule": "centering"},
    {"number": 20, "name": "C 2 2 21", "bravais": "Orthorhombic-C", "rule": "centering"},
    {"number": 63, "name": "C m c m", "bravais": "Orthorhombic-C", "rule": "centering"},
    {"number": 65, "name": "C m m m", "bravais": "Orthorhombic-C", "rule": "centering"},
    {"number": 71, "name": "I m m m", "bravais": "Orthorhombic-I", "rule": "centering"},
    {"number": 73, "name": "I b c a", "bravais": "Orthorhombic-I", "rule": "centering"},
    {"number": 69, "name": "F m m m", "bravais": "Orthorhombic-F", "rule": "centering"},
    {"number": 75, "name": "P 4", "bravais": "Tetragonal-P", "rule": "centering"},
    {"number": 76, "name": "P 41", "bravais": "Tetragonal-P", "rule": "p41_c"},
    {"number": 77, "name": "P 42", "bravais": "Tetragonal-P", "rule": "p42_c"},
    {"number": 78, "name": "P 43", "bravais": "Tetragonal-P", "rule": "p41_c"},
    {"number": 123, "name": "P 4/m m m", "bravais": "Tetragonal-P", "rule": "centering"},
    {"number": 136, "name": "P 42/m n m", "bravais": "Tetragonal-P", "rule": "p42_c"},
    {"number": 139, "name": "I 4/m m m", "bravais": "Tetragonal-I", "rule": "centering"},
    {"number": 141, "name": "I 41/a m d", "bravais": "Tetragonal-I", "rule": "centering"},
    {"number": 143, "name": "P 3", "bravais": "Trigonal/Hexagonal-P", "rule": "centering"},
    {"number": 147, "name": "P -3", "bravais": "Trigonal/Hexagonal-P", "rule": "centering"},
    {"number": 146, "name": "R 3", "bravais": "Trigonal-R", "rule": "centering"},
    {"number": 148, "name": "R -3", "bravais": "Trigonal-R", "rule": "centering"},
    {"number": 161, "name": "R 3 c", "bravais": "Trigonal-R", "rule": "r3c"},
    {"number": 167, "name": "R -3 c", "bravais": "Trigonal-R", "rule": "r3c"},
    {"number": 168, "name": "P 6", "bravais": "Trigonal/Hexagonal-P", "rule": "centering"},
    {"number": 173, "name": "P 63", "bravais": "Trigonal/Hexagonal-P", "rule": "p63_c"},
    {"number": 191, "name": "P 6/m m m", "bravais": "Trigonal/Hexagonal-P", "rule": "centering"},
    {"number": 194, "name": "P 63/m m c", "bravais": "Trigonal/Hexagonal-P", "rule": "p63_c"},
    {"number": 195, "name": "P 2 3", "bravais": "Cubic-P", "rule": "centering"},
    {"number": 200, "name": "P m -3", "bravais": "Cubic-P", "rule": "centering"},
    {"number": 221, "name": "P m -3 m", "bravais": "Cubic-P", "rule": "centering"},
    {"number": 197, "name": "I 2 3", "bravais": "Cubic-I", "rule": "centering"},
    {"number": 204, "name": "I m -3", "bravais": "Cubic-I", "rule": "centering"},
    {"number": 229, "name": "I m -3 m", "bravais": "Cubic-I", "rule": "centering"},
    {"number": 230, "name": "I a -3 d", "bravais": "Cubic-I", "rule": "centering"},
    {"number": 196, "name": "F 2 3", "bravais": "Cubic-F", "rule": "centering"},
    {"number": 202, "name": "F m -3", "bravais": "Cubic-F", "rule": "centering"},
    {"number": 225, "name": "F m -3 m", "bravais": "Cubic-F", "rule": "centering"},
    {"number": 227, "name": "F d -3 m", "bravais": "Cubic-F", "rule": "fd3m"},
)


class StructureSolutionError(ValueError):
    pass


class StructureSolutionCancelled(RuntimeError):
    pass


def _centering_allowed(bravais_name: str, h: int, k: int, l: int) -> bool:
    if bravais_name.endswith("-P") or bravais_name in {
        "Triclinic",
        "Trigonal/Hexagonal-P",
    }:
        return True
    if bravais_name.endswith("-I"):
        return (h + k + l) % 2 == 0
    if bravais_name.endswith("-F"):
        parity = (h % 2, k % 2, l % 2)
        return parity == (0, 0, 0) or parity == (1, 1, 1)
    if bravais_name.endswith("-A"):
        return (k + l) % 2 == 0
    if bravais_name.endswith("-B"):
        return (h + l) % 2 == 0
    if bravais_name.endswith("-C"):
        return (h + k) % 2 == 0
    if bravais_name == "Trigonal-R":
        return (-h + k + l) % 3 == 0
    return True


def _extra_space_group_allowed(rule: str, h: int, k: int, l: int) -> bool:
    if rule == "centering":
        return True
    if rule == "p21_b":
        return not (h == 0 and l == 0) or k % 2 == 0
    if rule == "h0l_hl_even":
        return not (k == 0) or (h + l) % 2 == 0
    if rule == "p21c_b":
        if h == 0 and l == 0 and k % 2:
            return False
        if k == 0 and (h + l) % 2:
            return False
        return True
    if rule == "p212121":
        if k == 0 and l == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if h == 0 and k == 0:
            return l % 2 == 0
        return True
    if rule == "p41_c":
        return not (h == 0 and k == 0) or l % 4 == 0
    if rule == "p42_c":
        return not (h == 0 and k == 0) or l % 2 == 0
    if rule == "p63_c":
        return not (h == 0 and k == 0) or l % 2 == 0
    if rule == "r3c":
        if h == 0 and k == 0:
            return l % 6 == 0
        return True
    if rule == "fd3m":
        if h == 0 and k == 0:
            return l % 4 == 0
        return True
    return True


def space_group_candidates_for_bravais(bravais_name: str) -> list[dict]:
    return [
        deepcopy(row)
        for row in SPACE_GROUP_LIBRARY
        if row["bravais"] == str(bravais_name)
    ]


def _raw_candidate_reflections(
    candidate: dict,
    *,
    wavelength_angstrom: float,
    two_theta_min: float,
    two_theta_max: float,
    maximum_index: int = 16,
    include_centering_forbidden: bool = False,
) -> list[dict]:
    cell = candidate_cell(candidate)
    bravais = str(candidate["bravais_name"])
    rows = []
    for h in range(maximum_index + 1):
        for k in range(maximum_index + 1):
            for l in range(maximum_index + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                centered = _centering_allowed(bravais, h, k, l)
                if not centered and not include_centering_forbidden:
                    continue
                try:
                    spacing = float(d_spacing(cell, (h, k, l)))
                except (ValueError, np.linalg.LinAlgError):
                    continue
                argument = float(wavelength_angstrom) / (2.0 * spacing)
                if not 0.0 < argument < 1.0:
                    continue
                position = float(2.0 * math.degrees(math.asin(argument)))
                if two_theta_min <= position <= two_theta_max:
                    rows.append(
                        {
                            "h": h,
                            "k": k,
                            "l": l,
                            "hkl_label": f"({h} {k} {l})",
                            "d_spacing_angstrom": spacing,
                            "two_theta_deg": position,
                            "centering_allowed": centered,
                        }
                    )
    rows.sort(key=lambda row: row["two_theta_deg"])
    return rows


def _peak_positions(peaks: Iterable[dict]) -> np.ndarray:
    values = [
        float(row["two_theta_deg"])
        for row in peaks
        if row.get("use", True) and row.get("two_theta_deg") is not None
    ]
    return np.asarray(sorted(values), dtype=float)


def _position_match_count(
    observed: np.ndarray,
    predicted: np.ndarray,
    tolerance_deg: float,
) -> tuple[int, list[dict]]:
    if not len(observed) or not len(predicted):
        return 0, []
    used = set()
    matches = []
    for observed_position in observed:
        order = np.argsort(np.abs(predicted - observed_position))
        selected = None
        for index in order:
            index = int(index)
            if index in used:
                continue
            delta = float(observed_position - predicted[index])
            if abs(delta) <= tolerance_deg:
                selected = index
                used.add(index)
                matches.append(
                    {
                        "observed_two_theta_deg": float(observed_position),
                        "predicted_two_theta_deg": float(predicted[index]),
                        "delta_deg": delta,
                    }
                )
                break
        if selected is None:
            continue
    return len(matches), matches


def screen_space_groups(
    candidate: dict,
    observed_peaks: list[dict],
    *,
    wavelength_angstrom: float,
    tolerance_deg: float = 0.08,
    maximum_index: int = 16,
    space_group_names: Iterable[str] | None = None,
) -> list[dict]:
    observed = _peak_positions(observed_peaks)
    if len(observed) < 5:
        raise StructureSolutionError(
            "At least five included peaks are required for space-group pre-screening."
        )
    groups = space_group_candidates_for_bravais(
        str(candidate["bravais_name"])
    )
    selected_names = set(space_group_names or [])
    if selected_names:
        groups = [row for row in groups if row["name"] in selected_names]
    if not groups:
        raise StructureSolutionError(
            "No curated space-group candidates are available for this Bravais lattice."
        )

    minimum = float(np.min(observed))
    maximum = float(np.max(observed))
    base_reflections = _raw_candidate_reflections(
        candidate,
        wavelength_angstrom=wavelength_angstrom,
        two_theta_min=max(0.01, minimum - 1.0),
        two_theta_max=maximum + 1.0,
        maximum_index=maximum_index,
    )
    results = []
    for group in groups:
        allowed_rows = []
        forbidden_rows = []
        for reflection in base_reflections:
            allowed = _extra_space_group_allowed(
                group["rule"],
                int(reflection["h"]),
                int(reflection["k"]),
                int(reflection["l"]),
            )
            (allowed_rows if allowed else forbidden_rows).append(reflection)

        allowed_positions = np.asarray(
            [row["two_theta_deg"] for row in allowed_rows],
            dtype=float,
        )
        forbidden_positions = np.asarray(
            [row["two_theta_deg"] for row in forbidden_rows],
            dtype=float,
        )
        matched_count, matches = _position_match_count(
            observed,
            allowed_positions,
            tolerance_deg,
        )

        forbidden_conflicts = 0
        conflict_rows = []
        for position in observed:
            allowed_distance = (
                float(np.min(np.abs(allowed_positions - position)))
                if allowed_positions.size
                else math.inf
            )
            forbidden_distance = (
                float(np.min(np.abs(forbidden_positions - position)))
                if forbidden_positions.size
                else math.inf
            )
            if (
                forbidden_distance <= tolerance_deg
                and allowed_distance > tolerance_deg
            ):
                forbidden_conflicts += 1
                conflict_rows.append(
                    {
                        "observed_two_theta_deg": float(position),
                        "nearest_forbidden_delta_deg": forbidden_distance,
                    }
                )

        coverage = matched_count / max(1, len(observed))
        conflict_fraction = forbidden_conflicts / max(1, len(observed))
        complexity_penalty = 0.002 * max(0, len(allowed_rows) - matched_count)
        score = (
            100.0 * coverage
            - 65.0 * conflict_fraction
            - complexity_penalty
        )
        if coverage >= 0.85 and forbidden_conflicts == 0:
            status = "Compatible pre-screen"
        elif coverage >= 0.65 and conflict_fraction <= 0.10:
            status = "Ambiguous"
        else:
            status = "Incompatible pre-screen"
        results.append(
            {
                **group,
                "observed_peak_count": len(observed),
                "matched_peak_count": matched_count,
                "coverage_fraction": float(coverage),
                "forbidden_conflict_count": forbidden_conflicts,
                "forbidden_conflict_fraction": float(conflict_fraction),
                "predicted_allowed_count": len(allowed_rows),
                "screening_score": float(score),
                "status": status,
                "matches": matches,
                "conflicts": conflict_rows,
                "screening_level": (
                    "Curated systematic-absence pre-screen. Confirm the "
                    "space group with a full symmetry engine, International Tables and "
                    "Rietveld/chemical validation."
                ),
            }
        )
    results.sort(
        key=lambda row: (
            row["screening_score"],
            row["coverage_fraction"],
            -row["forbidden_conflict_count"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank
    return results


def _cell_parameterization(candidate: dict):
    system = CRYSTAL_SYSTEM_BY_BRAVAIS.get(
        str(candidate["bravais_name"]),
        "Triclinic",
    )
    cell = candidate_cell(candidate)
    if system == "Cubic":
        names = ["a"]
    elif system in {"Tetragonal", "Hexagonal"}:
        names = ["a", "c"]
    elif system == "Orthorhombic":
        names = ["a", "b", "c"]
    elif system == "Rhombohedral":
        names = ["a", "alpha"]
    elif system == "Monoclinic":
        names = ["a", "b", "c", "beta"]
    else:
        names = ["a", "b", "c", "alpha", "beta", "gamma"]
    values = np.asarray([cell[name] for name in names], dtype=float)
    return system, names, values


def _cell_from_parameter_values(
    system: str,
    names: list[str],
    values: np.ndarray,
) -> dict:
    p = {name: float(value) for name, value in zip(names, values)}
    if system == "Cubic":
        return {
            "a": p["a"], "b": p["a"], "c": p["a"],
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        }
    if system == "Tetragonal":
        return {
            "a": p["a"], "b": p["a"], "c": p["c"],
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        }
    if system == "Hexagonal":
        return {
            "a": p["a"], "b": p["a"], "c": p["c"],
            "alpha": 90.0, "beta": 90.0, "gamma": 120.0,
        }
    if system == "Orthorhombic":
        return {
            "a": p["a"], "b": p["b"], "c": p["c"],
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        }
    if system == "Rhombohedral":
        return {
            "a": p["a"], "b": p["a"], "c": p["a"],
            "alpha": p["alpha"], "beta": p["alpha"], "gamma": p["alpha"],
        }
    if system == "Monoclinic":
        return {
            "a": p["a"], "b": p["b"], "c": p["c"],
            "alpha": 90.0, "beta": p["beta"], "gamma": 90.0,
        }
    return {
        "a": p["a"], "b": p["b"], "c": p["c"],
        "alpha": p["alpha"], "beta": p["beta"], "gamma": p["gamma"],
    }


def _assign_hkls(
    candidate: dict,
    observed_peaks: list[dict],
    wavelength_angstrom: float,
    tolerance_deg: float,
    maximum_index: int,
) -> list[dict]:
    observed = _peak_positions(observed_peaks)
    if not len(observed):
        return []
    predicted = _raw_candidate_reflections(
        candidate,
        wavelength_angstrom=wavelength_angstrom,
        two_theta_min=max(0.01, float(np.min(observed)) - 1.0),
        two_theta_max=float(np.max(observed)) + 1.0,
        maximum_index=maximum_index,
    )
    used = set()
    assignments = []
    for observed_position in observed:
        order = sorted(
            range(len(predicted)),
            key=lambda index: abs(
                predicted[index]["two_theta_deg"] - observed_position
            ),
        )
        for index in order:
            if index in used:
                continue
            delta = observed_position - predicted[index]["two_theta_deg"]
            if abs(delta) <= tolerance_deg:
                used.add(index)
                assignments.append(
                    {
                        **predicted[index],
                        "observed_two_theta_deg": float(observed_position),
                        "initial_delta_deg": float(delta),
                    }
                )
                break
    return assignments


def evaluate_candidate_robustness(
    candidate: dict,
    observed_peaks: list[dict],
    *,
    wavelength_angstrom: float,
    matching_tolerance_deg: float = 0.25,
    position_jitter_sigma_deg: float = 0.003,
    drop_fraction: float = 0.15,
    iterations: int = 100,
    parameter_bound_percent: float = 5.0,
    maximum_index: int = 16,
    random_seed: int = 1405,
    progress_callback=None,
    cancel_check=None,
) -> dict:
    assignments = _assign_hkls(
        candidate,
        observed_peaks,
        wavelength_angstrom,
        matching_tolerance_deg,
        maximum_index,
    )
    if len(assignments) < 5:
        raise StructureSolutionError(
            "Fewer than five peaks could be assigned to the candidate cell."
        )

    system, names, initial = _cell_parameterization(candidate)
    relative_bound = max(0.001, float(parameter_bound_percent) / 100.0)
    lower = []
    upper = []
    for name, value in zip(names, initial):
        if name in {"a", "b", "c"}:
            lower.append(max(0.05, value * (1.0 - relative_bound)))
            upper.append(value * (1.0 + relative_bound))
        else:
            span = max(0.2, 180.0 * relative_bound)
            lower.append(max(35.0, value - span))
            upper.append(min(145.0, value + span))
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    rng = np.random.default_rng(int(random_seed))
    samples = []
    iterations = int(np.clip(iterations, 10, 1000))
    drop_fraction = float(np.clip(drop_fraction, 0.0, 0.45))
    minimum_keep = max(5, int(math.ceil(len(assignments) * (1.0 - drop_fraction))))

    def refine(rows, jitter):
        observed = np.asarray(
            [row["observed_two_theta_deg"] for row in rows],
            dtype=float,
        ) + jitter
        hkls = [
            (int(row["h"]), int(row["k"]), int(row["l"]))
            for row in rows
        ]

        def residual(values):
            cell = _cell_from_parameter_values(system, names, values)
            calculated = []
            for hkl in hkls:
                position = two_theta_from_hkl(
                    cell,
                    hkl,
                    wavelength_angstrom,
                )
                if position is None:
                    return np.full(len(hkls), 1e3)
                calculated.append(position)
            return observed - np.asarray(calculated, dtype=float)

        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            max_nfev=3000,
        )
        cell = _cell_from_parameter_values(system, names, result.x)
        metric = direct_metric_tensor(cell)
        volume = float(math.sqrt(np.linalg.det(metric)))
        return result, cell, volume

    for iteration in range(iterations):
        if cancel_check is not None and cancel_check():
            raise StructureSolutionCancelled(
                "Candidate robustness test was cancelled."
            )
        if progress_callback is not None:
            progress_callback(
                iteration,
                iterations,
                f"Robustness iteration {iteration + 1}/{iterations}",
            )
        indices = np.arange(len(assignments))
        rng.shuffle(indices)
        indices = np.sort(indices[:minimum_keep])
        rows = [assignments[int(index)] for index in indices]
        jitter = rng.normal(
            0.0,
            max(0.0, float(position_jitter_sigma_deg)),
            len(rows),
        )
        try:
            result, cell, volume = refine(rows, jitter)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not result.success or not np.all(np.isfinite(result.x)):
            continue
        samples.append(
            {
                "iteration": iteration + 1,
                "cell": cell,
                "volume_angstrom3": volume,
                "rmse_deg": float(
                    np.sqrt(np.mean(np.square(result.fun)))
                ),
                "used_peak_count": len(rows),
            }
        )

    if progress_callback is not None:
        progress_callback(iterations, iterations, "Robustness test completed.")

    if not samples:
        raise StructureSolutionError(
            "No robustness iteration produced a valid refined cell."
        )

    volumes = np.asarray(
        [row["volume_angstrom3"] for row in samples],
        dtype=float,
    )
    volume_cv_percent = (
        100.0 * float(np.std(volumes, ddof=1)) / float(np.mean(volumes))
        if len(volumes) > 1 and np.mean(volumes) > 0
        else 0.0
    )
    parameter_statistics = {}
    relative_spreads = []
    for name in ("a", "b", "c", "alpha", "beta", "gamma"):
        values = np.asarray(
            [row["cell"][name] for row in samples],
            dtype=float,
        )
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        relative = 100.0 * std / abs(mean) if mean else math.inf
        parameter_statistics[name] = {
            "mean": mean,
            "standard_deviation": std,
            "relative_standard_deviation_percent": relative,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
        if name in names:
            relative_spreads.append(relative)

    success_fraction = len(samples) / iterations
    assignment_coverage = len(assignments) / max(
        1,
        sum(bool(row.get("use", True)) for row in observed_peaks),
    )
    maximum_relative_spread = max(relative_spreads or [0.0])
    median_rmse = float(np.median([row["rmse_deg"] for row in samples]))

    if (
        success_fraction >= 0.85
        and assignment_coverage >= 0.70
        and volume_cv_percent <= 0.75
        and maximum_relative_spread <= 0.75
        and median_rmse <= matching_tolerance_deg / 3.0
    ):
        status = "Stable candidate"
    elif (
        success_fraction >= 0.60
        and assignment_coverage >= 0.50
        and volume_cv_percent <= 2.0
        and maximum_relative_spread <= 2.0
    ):
        status = "Review candidate"
    else:
        status = "Unstable candidate"

    return {
        "status": status,
        "candidate": deepcopy(candidate),
        "assigned_peak_count": len(assignments),
        "included_peak_count": sum(
            bool(row.get("use", True)) for row in observed_peaks
        ),
        "assignment_coverage_fraction": float(assignment_coverage),
        "iterations_requested": iterations,
        "iterations_successful": len(samples),
        "success_fraction": float(success_fraction),
        "volume_mean_angstrom3": float(np.mean(volumes)),
        "volume_standard_deviation_angstrom3": (
            float(np.std(volumes, ddof=1)) if len(volumes) > 1 else 0.0
        ),
        "volume_cv_percent": float(volume_cv_percent),
        "maximum_cell_parameter_rsd_percent": float(maximum_relative_spread),
        "median_position_rmse_deg": median_rmse,
        "parameter_statistics": parameter_statistics,
        "assignments": assignments,
        "sample_preview": samples[: min(25, len(samples))],
        "warning": (
            "Robustness against peak omission and position jitter does not "
            "prove uniqueness. Competing cells and space groups must be tested."
        ),
    }


def _parse_hkl_label(label: str) -> tuple[int, int, int] | None:
    values = re.findall(r"[+-]?\d+", str(label))
    if len(values) < 3:
        return None
    return tuple(int(value) for value in values[:3])


def extract_intensity_table(
    whole_pattern_result: dict,
    *,
    phase_name: str | None = None,
    phase_index: int | None = None,
    overlap_factor: float = 0.75,
) -> list[dict]:
    if not isinstance(whole_pattern_result, dict):
        raise StructureSolutionError("A Phase 10 result is required.")
    rows = []
    for reflection in whole_pattern_result.get("reflections", []):
        if phase_name is not None and reflection.get("phase_name") != phase_name:
            continue
        if phase_index is not None and int(
            reflection.get("phase_index", -1)
        ) != int(phase_index):
            continue
        hkl = _parse_hkl_label(reflection.get("hkl_label", ""))
        if hkl is None:
            continue
        intensity = reflection.get("extracted_intensity")
        if intensity is None:
            continue
        uncertainty = (
            reflection.get("intensity_standard_error")
            or reflection.get("intensity_error")
            or reflection.get("extracted_intensity_error")
        )
        rows.append(
            {
                "phase_name": reflection.get("phase_name", ""),
                "phase_index": reflection.get("phase_index"),
                "h": hkl[0],
                "k": hkl[1],
                "l": hkl[2],
                "hkl_label": reflection.get("hkl_label", ""),
                "two_theta_deg": float(reflection.get("two_theta_deg")),
                "d_spacing_angstrom": reflection.get("d_spacing"),
                "fwhm_deg": float(reflection.get("fwhm_deg", 0.0)),
                "intensity": float(intensity),
                "intensity_uncertainty": (
                    None if uncertainty is None else float(uncertainty)
                ),
                "uncertainty_source": (
                    "Refinement output"
                    if uncertainty is not None
                    else "Not supplied by Phase 10"
                ),
                "overlap_group": None,
                "overlap_flag": False,
            }
        )
    rows.sort(key=lambda row: row["two_theta_deg"])
    group = 0
    previous = None
    for row in rows:
        if previous is None:
            group += 1
        else:
            separation = row["two_theta_deg"] - previous["two_theta_deg"]
            threshold = overlap_factor * max(
                row["fwhm_deg"],
                previous["fwhm_deg"],
                1e-6,
            )
            if separation > threshold:
                group += 1
            else:
                row["overlap_flag"] = True
                previous["overlap_flag"] = True
        row["overlap_group"] = group
        previous = row
    return rows


def export_extracted_intensities(
    directory: str | Path,
    rows: list[dict],
    *,
    fallback_relative_uncertainty: float = 0.10,
) -> dict:
    if not rows:
        raise StructureSolutionError("No extracted intensities are available.")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    txt_path = out / "extracted_intensities.txt"
    csv_path = out / "extracted_intensities.csv"
    json_path = out / "extracted_intensities.json"
    hkl_path = out / "extracted_intensities.hkl"

    fieldnames = [
        "phase_name", "phase_index", "h", "k", "l", "hkl_label",
        "two_theta_deg", "d_spacing_angstrom", "fwhm_deg", "intensity",
        "intensity_uncertainty", "uncertainty_source",
        "overlap_group", "overlap_flag",
    ]
    write_table_txt(
        txt_path,
        rows,
        columns=fieldnames,
        title="Extracted reflection intensities",
        metadata={
            "reflection_count": len(rows),
            "scientific_note": "Uncertainty fallbacks, when used, are export conveniences and not covariance-derived standard uncertainties.",
        },
        backup=False,
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fallback_used = 0
    with hkl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            sigma = row.get("intensity_uncertainty")
            if sigma is None or not np.isfinite(float(sigma)) or float(sigma) <= 0:
                sigma = max(
                    1.0,
                    abs(float(row["intensity"]))
                    * max(0.001, float(fallback_relative_uncertainty)),
                )
                fallback_used += 1
            handle.write(
                f"{int(row['h']):4d}{int(row['k']):4d}{int(row['l']):4d}"
                f"{float(row['intensity']):12.3f}{float(sigma):12.3f}\n"
            )
        handle.write("   0   0   0       0.000       0.000\n")
    return {
        "txt": str(txt_path),
        "csv": str(csv_path),
        "json": str(json_path),
        "hkl": str(hkl_path),
        "reflection_count": len(rows),
        "fallback_uncertainty_count": fallback_used,
        "warning": (
            "Fallback intensity uncertainties are export conveniences, not "
            "refined covariance-derived standard uncertainties."
            if fallback_used
            else ""
        ),
    }


def _formula_elements(formula: str) -> dict[str, float]:
    text = str(formula or "").replace(" ", "")
    if not text:
        return {}
    parts = re.findall(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)", text)
    composition: dict[str, float] = {}
    for symbol, count_text in parts:
        if symbol not in ATOMIC_NUMBER:
            continue
        count = float(count_text) if count_text else 1.0
        composition[symbol] = composition.get(symbol, 0.0) + count
    return composition


def _normalised_composition(composition: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in composition.values())
    if total <= 0:
        return {}
    return {
        key: max(0.0, float(value)) / total
        for key, value in composition.items()
    }


def build_p1_provisional_cif(
    candidate_or_cell: dict,
    atoms: list[dict],
    *,
    title: str = "Afruz provisional structure",
    formula: str = "",
    proposed_space_group: str = "Unresolved",
    provenance: str = "",
) -> str:
    cell = (
        candidate_cell(candidate_or_cell)
        if "a_angstrom" in candidate_or_cell
        else {
            key: float(candidate_or_cell[key])
            for key in ("a", "b", "c", "alpha", "beta", "gamma")
        }
    )
    if not atoms:
        raise StructureSolutionError(
            "At least one provisional atom is required to create a CIF."
        )
    cleaned = []
    for index, atom in enumerate(atoms, start=1):
        element = str(atom.get("element", "")).capitalize()
        if element not in ATOMIC_NUMBER:
            raise StructureSolutionError(
                f"Atom {index} has an unknown element: {element!r}."
            )
        coordinates = [
            float(atom.get(axis, 0.0))
            for axis in ("x", "y", "z")
        ]
        if not all(np.isfinite(coordinates)):
            raise StructureSolutionError(
                f"Atom {index} has non-finite coordinates."
            )
        occupancy = float(atom.get("occupancy", 1.0))
        if not 0.0 <= occupancy <= 1.0:
            raise StructureSolutionError(
                f"Atom {index} occupancy must be between 0 and 1."
            )
        b_iso = float(atom.get("b_iso", 1.0))
        if not np.isfinite(b_iso) or b_iso < 0:
            raise StructureSolutionError(
                f"Atom {index} Biso must be finite and non-negative."
            )
        cleaned.append(
            {
                "label": str(atom.get("label") or f"{element}{index}"),
                "element": element,
                "x": coordinates[0] % 1.0,
                "y": coordinates[1] % 1.0,
                "z": coordinates[2] % 1.0,
                "occupancy": occupancy,
                "b_iso": b_iso,
            }
        )

    safe_title = re.sub(r"[^A-Za-z0-9_]+", "_", title).strip("_") or "provisional"
    formula_value = formula or composition_formula_from_sites(cleaned)
    formula_weight = formula_weight_from_counts(parse_formula_counts(formula_value))
    lines = [
        f"data_{safe_title}",
        f"_chemical_name_common '{title}'",
        f"_chemical_formula_sum '{formula_value}'",
        f"_chemical_formula_weight {formula_weight:.6f}",
        "_cell_formula_units_Z 1",
        "_audit_creation_method 'Afruz PXRD Analyzer Phase Revolution provisional P1 container'",
        f"_afruz_proposed_space_group '{proposed_space_group}'",
        "_afruz_structure_status 'Provisional CIF - not independently validated'",
        f"_afruz_structure_provenance '{provenance or 'Not supplied'}'",
        f"_cell_length_a {cell['a']:.10g}",
        f"_cell_length_b {cell['b']:.10g}",
        f"_cell_length_c {cell['c']:.10g}",
        f"_cell_angle_alpha {cell['alpha']:.10g}",
        f"_cell_angle_beta {cell['beta']:.10g}",
        f"_cell_angle_gamma {cell['gamma']:.10g}",
        "_space_group_crystal_system triclinic",
        "_space_group_name_H-M_alt 'P 1'",
        "loop_",
        "_space_group_symop_id",
        "_space_group_symop_operation_xyz",
        "1 'x,y,z'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_B_iso_or_equiv",
    ]
    for atom in cleaned:
        lines.append(
            f"{atom['label']} {atom['element']} "
            f"{atom['x']:.10f} {atom['y']:.10f} {atom['z']:.10f} "
            f"{atom['occupancy']:.6f} {atom['b_iso']:.6f}"
        )
    lines.extend(
        [
            "",
            "# The proposed higher symmetry is stored as provenance only.",
            "# Coordinates are written in P1 until symmetry has been independently validated.",
        ]
    )
    return "\n".join(lines) + "\n"


def audit_provisional_cif(
    cif_text: str,
    *,
    minimum_contact_angstrom: float = 0.70,
    duplicate_contact_angstrom: float = 0.08,
) -> dict:
    try:
        structure = parse_cif_text(cif_text)
    except CIFImportError as exc:
        return {
            "status": "Invalid provisional CIF",
            "errors": [str(exc)],
            "warnings": [],
            "structure": None,
        }

    errors = []
    warnings = []
    atoms = structure.get("atoms", [])
    if not atoms:
        errors.append("The CIF contains no readable fractional atom sites.")
    for atom in atoms:
        occupancy = float(atom.get("occupancy", 1.0))
        if not 0.0 <= occupancy <= 1.0:
            errors.append(
                f"{atom.get('label')}: occupancy {occupancy:g} is outside 0–1."
            )

    metric = direct_metric_tensor(structure["cell"])
    shortest = math.inf
    duplicate_pairs = []
    short_pairs = []
    for first_index, first in enumerate(atoms):
        first_fractional = np.asarray(
            [first["x"], first["y"], first["z"]],
            dtype=float,
        )
        for second_index in range(first_index + 1, len(atoms)):
            second = atoms[second_index]
            second_fractional = np.asarray(
                [second["x"], second["y"], second["z"]],
                dtype=float,
            )
            delta = first_fractional - second_fractional
            delta -= np.round(delta)
            distance = float(math.sqrt(delta @ metric @ delta))
            shortest = min(shortest, distance)
            pair = {
                "atom_1": first.get("label"),
                "atom_2": second.get("label"),
                "distance_angstrom": distance,
            }
            if distance < duplicate_contact_angstrom:
                duplicate_pairs.append(pair)
            elif distance < minimum_contact_angstrom:
                short_pairs.append(pair)

    if duplicate_pairs:
        errors.append(
            f"{len(duplicate_pairs)} duplicate or nearly duplicate atom pair(s) were found."
        )
    if short_pairs:
        warnings.append(
            f"{len(short_pairs)} unusually short interatomic contact(s) were found."
        )
    if structure.get("space_group") == "P 1":
        warnings.append(
            "The provisional structure is stored in P1. Proposed higher symmetry "
            "has not been applied to the coordinates."
        )

    plausibility = validate_crystal_structure(structure)
    for message in plausibility.get("errors", []):
        if "duplicate" not in message.lower():
            errors.append(message)
    warnings.extend(plausibility.get("warnings", []))
    warnings.extend(plausibility.get("notes", []))

    expected = _normalised_composition(
        _formula_elements(structure.get("formula", ""))
    )
    observed_counts: dict[str, float] = {}
    for atom in atoms:
        element = atom["element"]
        observed_counts[element] = observed_counts.get(element, 0.0) + float(
            atom.get("occupancy", 1.0)
        )
    observed = _normalised_composition(observed_counts)
    formula_mismatch = []
    if expected and observed:
        for element in sorted(set(expected) | set(observed)):
            difference = abs(expected.get(element, 0.0) - observed.get(element, 0.0))
            if difference > 0.05:
                formula_mismatch.append(
                    {
                        "element": element,
                        "formula_fraction": expected.get(element, 0.0),
                        "site_fraction": observed.get(element, 0.0),
                        "absolute_difference": difference,
                    }
                )
        if formula_mismatch:
            warnings.append(
                "The atom-site composition differs from the supplied formula."
            )

    status = (
        "Invalid provisional CIF"
        if errors
        else "Provisional CIF — review required"
        if warnings
        else "Provisional CIF — basic checks passed"
    )
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "structure": structure,
        "atom_count": len(atoms),
        "shortest_contact_angstrom": (
            None if not np.isfinite(shortest) else shortest
        ),
        "duplicate_pairs": duplicate_pairs,
        "short_contact_pairs": short_pairs,
        "formula_mismatch": formula_mismatch,
        "structure_plausibility": plausibility,
        "scientific_boundary": (
            "Basic CIF syntax, occupancy, duplicate-site, contact, density and "
            "coordination checks do not replace checkCIF/PLATON, bond-valence "
            "analysis, expert chemical review or Rietveld validation."
        ),
    }


def import_and_audit_provisional_cif(path: str | Path) -> dict:
    source = Path(path)
    if not source.exists():
        raise StructureSolutionError(f"CIF does not exist: {source}")
    text = source.read_text(encoding="utf-8", errors="replace")
    result = audit_provisional_cif(text)
    result["source_path"] = str(source.resolve())
    result["raw_cif_text"] = text
    return result


def build_phase11_handoff(
    cif_text: str,
    *,
    dataset_uid: str,
    dataset_name: str = "",
    candidate: dict | None = None,
    master_peak_metadata: dict | None = None,
    proposed_space_group: str = "Unresolved",
    provenance: str = "Unknown Discovery structure-solution pathway",
) -> dict:
    """Create a validated, provenance-rich Phase 11 handoff record.

    The handoff deliberately preserves the provisional status of an unknown
    structure. It performs the same syntax/contact audit used by the Structure
    Solution workspace, rejects stale candidate/peak-list combinations, and
    packages the parsed structure in the schema consumed by Phase 11.
    """

    text = str(cif_text or "")
    if not text.strip():
        raise StructureSolutionError(
            "A provisional CIF is required before sending a structure to Phase 11."
        )

    audit = audit_provisional_cif(text)
    errors = list(audit.get("errors") or [])
    structure = audit.get("structure")
    if isinstance(structure, dict) and not structure.get("atoms"):
        raise StructureSolutionError(
            "Phase 11 requires an atomic model. Add or import at least one atom first."
        )
    if errors or not isinstance(structure, dict):
        detail = "; ".join(errors) or "The provisional CIF could not be parsed."
        raise StructureSolutionError(
            "The structure cannot be sent to Phase 11: " + detail
        )

    candidate_record = deepcopy(candidate or {})
    peak_record = deepcopy(master_peak_metadata or {})
    candidate_checksum = candidate_record.get("master_peak_checksum")
    active_checksum = peak_record.get("checksum")
    if (
        candidate_checksum
        and active_checksum
        and str(candidate_checksum) != str(active_checksum)
    ):
        raise StructureSolutionError(
            "The selected candidate cell was indexed from a different master "
            "reflection list. Re-run indexing or restore the indexed peak-list revision."
        )

    raw_cif_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    formula = str(structure.get("formula") or "").strip()
    data_name = str(structure.get("data_name") or "").strip()
    phase_name = formula or data_name or dataset_name or "Unknown phase"
    warnings = list(audit.get("warnings") or [])
    status = (
        "Provisional — review required"
        if warnings
        else "Provisional — basic checks passed"
    )

    plausibility = deepcopy(audit.get("structure_plausibility") or {})
    handoff = {
        "schema": "afruz.phase11_unknown_structure_handoff.v1",
        "source": "Unknown Discovery",
        "source_dataset_uid": str(dataset_uid),
        "source_dataset_name": str(dataset_name or ""),
        "classification": status,
        "provenance": str(provenance or ""),
        "raw_cif_sha256": raw_cif_sha256,
        "master_peak_revision": peak_record.get("revision"),
        "master_peak_checksum": active_checksum or candidate_checksum,
        "candidate_rank": candidate_record.get("rank"),
        "candidate_status": candidate_record.get("status"),
        "candidate_bravais": candidate_record.get("bravais_name"),
        "candidate_cell": (
            candidate_cell(candidate_record)
            if candidate_record and "a_angstrom" in candidate_record
            else None
        ),
        "proposed_space_group": str(proposed_space_group or "Unresolved"),
        "audit_status": audit.get("status"),
        "audit_warnings": warnings,
        "atom_count": int(audit.get("atom_count") or 0),
        "structure_plausibility_status": plausibility.get("status"),
        "structure_plausibility_score": (
            plausibility.get("scores", {}).get("overall")
            if isinstance(plausibility.get("scores"), dict)
            else None
        ),
        "structure_plausibility_warnings": list(plausibility.get("warnings") or []),
        "scientific_boundary": (
            "This is a provisional atomic model transferred for structure-constrained "
            "whole-pattern validation. A successful fit does not independently prove "
            "the structure or space group."
        ),
    }

    phase11_structure = deepcopy(structure)
    phase11_structure["data_name"] = phase_name
    phase11_structure["raw_cif_text"] = text
    phase11_structure["_afruz_origin"] = "Unknown Discovery"
    phase11_structure["_afruz_validation_status"] = status
    phase11_structure["_afruz_proposed_space_group"] = str(
        proposed_space_group or "Unresolved"
    )
    phase11_structure["_afruz_phase11_handoff"] = deepcopy(handoff)
    phase11_structure["_afruz_structure_plausibility"] = plausibility

    return {
        "structure": phase11_structure,
        "audit": deepcopy(audit),
        "handoff": handoff,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_structure_solution_package(
    directory: str | Path,
    *,
    interface: str,
    candidate: dict,
    wavelength_angstrom: float,
    intensity_rows: list[dict],
    formula: str = "",
    z_value: int | None = None,
    proposed_space_group: str = "Unresolved",
    atoms: list[dict] | None = None,
    notes: str = "",
) -> dict:
    if interface not in STRUCTURE_SOLUTION_INTERFACES:
        raise StructureSolutionError(f"Unsupported interface: {interface}")
    if not intensity_rows:
        raise StructureSolutionError(
            "Extracted reflection intensities are required for a structure-solution package."
        )
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    intensity_files = export_extracted_intensities(out, intensity_rows)
    atoms = atoms or [
        {
            "label": "X1",
            "element": next(iter(_formula_elements(formula)), "C"),
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "occupancy": 1.0,
            "b_iso": 1.0,
        }
    ]
    cif_text = build_p1_provisional_cif(
        candidate,
        atoms,
        title="Phase Revolution starting cell",
        formula=formula,
        proposed_space_group=proposed_space_group,
        provenance=(
            "Generated from indexed powder candidate and extracted intensities. "
            "Dummy/start atoms must be replaced by a structure solution."
        ),
    )
    cif_path = out / "starting_cell_p1.cif"
    cif_path.write_text(cif_text, encoding="utf-8")

    request = {
        "classification": "Structure-solution interface package",
        "interface": interface,
        "wavelength_angstrom": float(wavelength_angstrom),
        "candidate_cell": deepcopy(candidate),
        "formula": formula,
        "z_value": None if z_value is None else int(z_value),
        "proposed_space_group": proposed_space_group,
        "reflection_count": len(intensity_rows),
        "input_files": {
            "cif": cif_path.name,
            "txt": Path(intensity_files["txt"]).name,
            "csv": Path(intensity_files["csv"]).name,
            "hkl": Path(intensity_files["hkl"]).name,
            "json": Path(intensity_files["json"]).name,
        },
        "notes": notes,
        "scientific_warning": (
            "This package prepares inputs only. It does not demonstrate that "
            "the selected cell, space group or atomic model is correct."
        ),
    }
    request_path = out / "structure_solution_request.json"
    request_path.write_text(
        json.dumps(request, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    request_txt = write_mapping_txt(
        out / "structure_solution_request.txt",
        request,
        title="Structure-solution handoff request",
        backup=False,
    )

    interface_guidance = {
        "Generic hkl + provisional CIF": (
            "Import the HKL intensities and P1 cell into the selected structure-"
            "solution program. Confirm reflection conventions and uncertainty handling."
        ),
        "SHELXT reflection-data handoff": (
            "Use extracted powder intensities only after reviewing severe overlap. "
            "Create the SHELX INS file manually with the verified composition, Z, "
            "LATT/SYMM instructions and scattering-factor order."
        ),
        "EXPO powder-structure handoff": (
            "Import the powder reflection list, unit cell and candidate symmetry "
            "into EXPO. Recheck peak decomposition and systematic absences in EXPO."
        ),
        "FOX / DASH real-space handoff": (
            "Use the cell and extracted intensities with chemically constrained "
            "molecular fragments. Define rigid bodies, torsions and collision "
            "restraints in the external program."
        ),
    }
    readme = out / "README_STRUCTURE_SOLUTION.txt"
    readme.write_text(
        "\n".join(
            [
                "AFRUZ PHASE REVOLUTION — STRUCTURE-SOLUTION HANDOFF",
                "",
                f"Interface: {interface}",
                "",
                interface_guidance[interface],
                "",
                "Mandatory review:",
                "- confirm wavelength and reflection convention",
                "- inspect overlapping reflections and uncertainties",
                "- test competing cells and space groups",
                "- apply chemical composition and density constraints",
                "- validate any solution by Rietveld refinement",
                "- run independent CIF validation before publication",
                "",
                request["scientific_warning"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = out / "manifest_sha256.csv"
    files = sorted(
        path for path in out.iterdir()
        if path.is_file() and path.name != manifest_path.name
    )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "size_bytes", "sha256"])
        for path in files:
            writer.writerow([path.name, path.stat().st_size, _sha256(path)])

    manifest_txt = write_manifest_txt(
        out / "manifest_sha256.txt",
        files + [manifest_path],
        root=out,
        title="Structure-solution handoff manifest",
        backup=False,
    )

    return {
        "directory": str(out.resolve()),
        "request": str(request_path),
        "request_txt": request_txt["txt_path"],
        "readme": str(readme),
        "manifest": str(manifest_path),
        "manifest_txt": manifest_txt["txt_path"],
        "cif": str(cif_path),
        "intensity_files": intensity_files,
        "interface": interface,
        "classification": "Prepared structure-solution handoff",
    }
