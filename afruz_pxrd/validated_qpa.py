from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any

import numpy as np

from .crystallography import direct_metric_tensor


class QPAValidationError(ValueError):
    pass


# Standard atomic weights (abridged uncertainty notation removed). Values are
# sufficient for unit-cell mass calculations; unsupported elements stop QPA.
ATOMIC_WEIGHT = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831,
    "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998403163, "Ne": 20.1797, "Na": 22.98976928,
    "Mg": 24.305, "Al": 26.9815385, "Si": 28.085, "P": 30.973761998,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.0983,
    "Ca": 40.078, "Sc": 44.955908, "Ti": 47.867, "V": 50.9415,
    "Cr": 51.9961, "Mn": 54.938044, "Fe": 55.845, "Co": 58.933194,
    "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723,
    "Ge": 72.630, "As": 74.921595, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.4678, "Sr": 87.62, "Y": 88.90584,
    "Zr": 91.224, "Nb": 92.90637, "Mo": 95.95, "Tc": 98.0,
    "Ru": 101.07, "Rh": 102.90550, "Pd": 106.42, "Ag": 107.8682,
    "Cd": 112.414, "In": 114.818, "Sn": 118.710, "Sb": 121.760,
    "Te": 127.60, "I": 126.90447, "Xe": 131.293, "Cs": 132.90545196,
    "Ba": 137.327, "La": 138.90547, "Ce": 140.116, "Pr": 140.90766,
    "Nd": 144.242, "Pm": 145.0, "Sm": 150.36, "Eu": 151.964,
    "Gd": 157.25, "Tb": 158.92535, "Dy": 162.500, "Ho": 164.93033,
    "Er": 167.259, "Tm": 168.93422, "Yb": 173.045, "Lu": 174.9668,
    "Hf": 178.49, "Ta": 180.94788, "W": 183.84, "Re": 186.207,
    "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.966569,
    "Hg": 200.592, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98040,
    "Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0,
    "Ra": 226.0, "Ac": 227.0, "Th": 232.0377, "Pa": 231.03588,
    "U": 238.02891, "Np": 237.0, "Pu": 244.0,
}


def cell_volume_angstrom3(cell: dict) -> float:
    return float(math.sqrt(float(np.linalg.det(direct_metric_tensor(cell)))))


def _composition_from_expanded_atoms(atoms: list[dict]) -> dict[str, float]:
    composition: dict[str, float] = {}
    for atom in atoms:
        element = str(atom.get("element", "")).strip()
        if not element:
            continue
        occupancy = float(atom.get("occupancy", 1.0))
        composition[element] = composition.get(element, 0.0) + occupancy
    return composition


def unit_cell_mass_g_mol(structure: dict) -> tuple[float, dict[str, float]]:
    atoms = list(structure.get("atoms", []))
    if not atoms:
        raise QPAValidationError("The CIF structure contains no expanded atomic sites.")
    composition = _composition_from_expanded_atoms(atoms)
    unsupported = sorted(element for element in composition if element not in ATOMIC_WEIGHT)
    if unsupported:
        raise QPAValidationError(
            "Atomic weights are unavailable for: " + ", ".join(unsupported)
        )
    mass = sum(ATOMIC_WEIGHT[element] * amount for element, amount in composition.items())
    if not np.isfinite(mass) or mass <= 0:
        raise QPAValidationError("The calculated unit-cell mass is not positive.")
    return float(mass), composition


def parse_formula(formula: str) -> dict[str, float]:
    clean = str(formula or "").replace(" ", "")
    clean = re.sub(r"[\[\]()]", "", clean)
    tokens = re.findall(r"([A-Z][a-z]?)([-+]?(?:\d+(?:\.\d*)?|\.\d+)?)", clean)
    result: dict[str, float] = {}
    for element, count_text in tokens:
        if element not in ATOMIC_WEIGHT:
            continue
        count = float(count_text) if count_text not in ("", "+", "-") else 1.0
        result[element] = result.get(element, 0.0) + count
    return result


def formula_mass_g_mol(formula: str) -> float | None:
    composition = parse_formula(formula)
    if not composition:
        return None
    return float(sum(ATOMIC_WEIGHT[e] * n for e, n in composition.items()))


def audit_structure(structure: dict, phase_name: str = "Phase") -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    atoms = list(structure.get("atoms", []))
    cell = structure.get("cell")
    if not atoms:
        errors.append("No expanded atomic sites are present.")
    if not isinstance(cell, dict):
        errors.append("A complete unit cell is missing.")

    for index, atom in enumerate(atoms, start=1):
        occupancy = float(atom.get("occupancy", 1.0))
        if occupancy < 0 or occupancy > 1.0 + 1e-8:
            errors.append(f"Atom {index} has occupancy outside 0–1.")
        if atom.get("element") not in ATOMIC_WEIGHT:
            errors.append(f"Atom {index} has unsupported element {atom.get('element')!r}.")

    volume = None
    mass = None
    composition: dict[str, float] = {}
    if not errors:
        try:
            volume = cell_volume_angstrom3(cell)
            mass, composition = unit_cell_mass_g_mol(structure)
        except (ValueError, QPAValidationError, np.linalg.LinAlgError) as exc:
            errors.append(str(exc))

    formula = str(structure.get("formula", "") or "")
    formula_mass = formula_mass_g_mol(formula)
    inferred_z = None
    if mass and formula_mass and formula_mass > 0:
        ratio = mass / formula_mass
        nearest = round(ratio)
        if nearest >= 1 and abs(ratio - nearest) <= 0.03 * nearest:
            inferred_z = int(nearest)
        else:
            warnings.append(
                "The CIF formula does not yield an integer Z consistent with the expanded atoms."
            )
    else:
        warnings.append("Z cannot be inferred from the CIF formula; cell mass is used directly.")

    raw_text = structure.get("raw_cif_text")
    if not raw_text and not structure.get("source_path"):
        warnings.append("The original CIF source is not retained for source traceability.")

    return {
        "phase_name": phase_name,
        "status": "Pass" if not errors else "Fail",
        "errors": errors,
        "warnings": warnings,
        "unit_cell_volume_angstrom3": volume,
        "unit_cell_mass_g_mol": mass,
        "zmv": None if mass is None or volume is None else float(mass * volume),
        "composition_per_cell": composition,
        "formula": formula,
        "formula_mass_g_mol": formula_mass,
        "inferred_z": inferred_z,
        "expanded_atom_count": len(atoms),
    }


def _fraction_jacobian(contributions: np.ndarray, zmv: np.ndarray) -> np.ndarray:
    total = float(np.sum(contributions))
    n = len(contributions)
    jacobian = np.zeros((n, n), dtype=float)
    if total <= 0:
        return jacobian
    for i in range(n):
        for j in range(n):
            jacobian[i, j] = (
                (zmv[i] if i == j else 0.0) * total
                - contributions[i] * zmv[j]
            ) / (total * total)
    return jacobian


def compute_hill_howard_qpa(
    phase_rows: list[dict],
    structures: list[dict],
    *,
    scale_covariance: list[list[float]] | np.ndarray | None = None,
    scale_standard_errors: list[float | None] | None = None,
    engine: str = "Afruz native preview",
    internal_standard_phase: str | None = None,
    internal_standard_known_wt_percent: float | None = None,
) -> dict:
    if len(phase_rows) < 2:
        raise QPAValidationError("At least two crystalline phases are required for QPA.")
    if len(phase_rows) != len(structures):
        raise QPAValidationError("Phase rows and CIF structures are not aligned.")

    audits = [
        audit_structure(structure, str(row.get("phase_name", f"Phase {i+1}")))
        for i, (row, structure) in enumerate(zip(phase_rows, structures))
    ]
    failed = [audit for audit in audits if audit["status"] != "Pass"]
    if failed:
        messages = []
        for audit in failed:
            messages.extend(f"{audit['phase_name']}: {error}" for error in audit["errors"])
        raise QPAValidationError("QPA crystallographic audit failed: " + "; ".join(messages))

    scales = np.asarray([float(row.get("scale_factor", 0.0)) for row in phase_rows])
    if np.any(scales < 0) or not np.any(scales > 0):
        raise QPAValidationError("Refined phase scales must be non-negative with at least one positive scale.")
    zmv = np.asarray([float(audit["zmv"]) for audit in audits])
    contributions = scales * zmv
    total = float(np.sum(contributions))
    if total <= 0:
        raise QPAValidationError("The sum of Hill–Howard mass contributions is not positive.")
    fractions = contributions / total

    covariance = None
    uncertainty_method = "Unavailable"
    if scale_covariance is not None:
        candidate = np.asarray(scale_covariance, dtype=float)
        if candidate.shape == (len(scales), len(scales)) and np.all(np.isfinite(candidate)):
            covariance = candidate
            uncertainty_method = "Scale covariance propagation"
    if covariance is None and scale_standard_errors:
        errors = np.asarray([
            0.0 if value is None else max(0.0, float(value))
            for value in scale_standard_errors
        ])
        if len(errors) == len(scales) and np.any(errors > 0):
            covariance = np.diag(errors * errors)
            uncertainty_method = "Independent scale-error propagation"

    fraction_errors = np.full(len(scales), np.nan)
    if covariance is not None:
        jacobian = _fraction_jacobian(contributions, zmv)
        fraction_covariance = jacobian @ covariance @ jacobian.T
        fraction_errors = np.sqrt(np.maximum(np.diag(fraction_covariance), 0.0))

    phase_results = []
    for index, (row, audit) in enumerate(zip(phase_rows, audits)):
        phase_results.append({
            "phase_name": row.get("phase_name"),
            "formula": row.get("formula") or audit.get("formula"),
            "scale_factor": float(scales[index]),
            "scale_standard_error": (
                None if not scale_standard_errors or index >= len(scale_standard_errors)
                else scale_standard_errors[index]
            ),
            "unit_cell_mass_g_mol": audit["unit_cell_mass_g_mol"],
            "unit_cell_volume_angstrom3": audit["unit_cell_volume_angstrom3"],
            "zmv": audit["zmv"],
            "inferred_z": audit["inferred_z"],
            "mass_contribution": float(contributions[index]),
            "crystalline_weight_fraction": float(fractions[index]),
            "crystalline_weight_percent": float(100.0 * fractions[index]),
            "crystalline_weight_percent_error": (
                None if not np.isfinite(fraction_errors[index])
                else float(100.0 * fraction_errors[index])
            ),
            "absolute_weight_percent": None,
            "absolute_weight_percent_error": None,
        })

    amorphous_percent = None
    standard_details = None
    if internal_standard_phase is not None:
        if internal_standard_known_wt_percent is None:
            raise QPAValidationError("The known internal-standard weight percent is required.")
        known = float(internal_standard_known_wt_percent)
        if not 0 < known < 100:
            raise QPAValidationError("Internal-standard weight percent must be between 0 and 100.")
        match = next(
            (i for i, row in enumerate(phase_results) if row["phase_name"] == internal_standard_phase),
            None,
        )
        if match is None:
            raise QPAValidationError("The selected internal-standard phase is not in the QPA result.")
        refined_standard = float(fractions[match])
        if refined_standard <= 0:
            raise QPAValidationError("The refined internal-standard fraction is not positive.")
        crystalline_total_fraction = (known / 100.0) / refined_standard
        if crystalline_total_fraction > 1.05:
            raise QPAValidationError(
                "Internal-standard correction implies more than 105% crystalline material; "
                "verify the known addition, phase model, absorption, and refinement."
            )
        crystalline_total_fraction = min(crystalline_total_fraction, 1.0)
        amorphous_percent = float(100.0 * (1.0 - crystalline_total_fraction))
        for index, phase in enumerate(phase_results):
            absolute = float(100.0 * fractions[index] * crystalline_total_fraction)
            phase["absolute_weight_percent"] = absolute
            if np.isfinite(fraction_errors[index]):
                phase["absolute_weight_percent_error"] = float(
                    100.0 * crystalline_total_fraction * fraction_errors[index]
                )
        standard_details = {
            "phase_name": internal_standard_phase,
            "known_weight_percent_in_total_mixture": known,
            "refined_crystalline_normalized_percent": 100.0 * refined_standard,
            "crystalline_total_percent": 100.0 * crystalline_total_fraction,
            "amorphous_or_unmodelled_percent": amorphous_percent,
        }

    warnings = []
    for audit in audits:
        warnings.extend(f"{audit['phase_name']}: {warning}" for warning in audit["warnings"])
    warnings.append(
        "Native Afruz scale factors require certified-mixture recovery and independent "
        "experimental validation before publication use."
    )

    return {
        "method": "Hill–Howard Rietveld quantitative phase analysis",
        "equation": "W_p = S_p(ZMV)_p / sum_i[S_i(ZMV)_i]",
        "engine": engine,
        "classification": "Native crystallographic QPA — validation required",
        "publication_ready": False,
        "uncertainty_method": uncertainty_method,
        "phases": phase_results,
        "crystallographic_audits": audits,
        "internal_standard": standard_details,
        "amorphous_or_unmodelled_percent": amorphous_percent,
        "warnings": list(dict.fromkeys(warnings)),
    }


