from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from .crystallography import ATOMIC_NUMBER, direct_metric_tensor

# Compact reference tables. Missing elements fall back to conservative values so
# validation remains available for incomplete or unusual CIFs.
ATOMIC_WEIGHTS = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42,
    "Ag": 107.868, "Cd": 112.414, "In": 114.818, "Sn": 118.710, "Sb": 121.760,
    "Te": 127.60, "I": 126.904, "Xe": 131.293, "Cs": 132.905, "Ba": 137.327,
    "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Sm": 150.36,
    "Eu": 151.964, "Gd": 157.25, "Tb": 158.925, "Dy": 162.500, "Ho": 164.930,
    "Er": 167.259, "Tm": 168.934, "Yb": 173.045, "Lu": 174.967, "Hf": 178.49,
    "Ta": 180.948, "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217,
    "Pt": 195.084, "Au": 196.967, "Hg": 200.592, "Tl": 204.38, "Pb": 207.2,
    "Bi": 208.980, "Th": 232.038, "U": 238.029,
}

COVALENT_RADII = {
    "H": 0.31, "B": 0.85, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53,
    "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32,
    "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42,
    "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Cs": 2.44, "Ba": 2.15,
    "La": 2.07, "Ce": 2.04, "Nd": 2.01, "Sm": 1.98, "Gd": 1.96, "Dy": 1.92,
    "Er": 1.89, "Yb": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51,
    "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Pb": 1.46,
    "Bi": 1.48, "Th": 2.06, "U": 1.96,
}

VDW_RADII = {
    "H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "Na": 2.27, "Mg": 1.73, "K": 2.75, "Ca": 2.31, "Fe": 2.00,
    "Cr": 2.00, "Ni": 1.63, "Cu": 1.40, "Zn": 1.39, "Pb": 2.02,
}

METALS = {
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba", "Sc", "Ti",
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo",
    "Ru", "Rh", "Pd", "Ag", "Cd", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Al", "Ga", "In", "Sn", "Pb", "Bi", "Th", "U",
}

COMMON_METAL_O_RANGES = {
    "Cr": (1.75, 2.35), "Fe": (1.75, 2.45), "Al": (1.70, 2.20),
    "Zn": (1.85, 2.35), "Cu": (1.80, 2.45), "Ni": (1.75, 2.35),
    "Co": (1.75, 2.35), "Mn": (1.80, 2.55), "Ti": (1.70, 2.25),
    "Zr": (1.90, 2.40), "Ca": (2.20, 2.90), "Mg": (1.85, 2.25),
}


def _clean_formula(formula: str) -> str:
    text = str(formula or "")
    text = text.replace("·", "+").replace(".", "+")
    text = re.sub(r"\s+", "", text)
    text = text.replace("'", "").replace('"', "")
    return text


def parse_formula_counts(formula: str) -> dict[str, float]:
    """Parse a practical CIF formula string into element counts.

    Supports simple element counts and one level of parentheses, sufficient for
    validation diagnostics. It deliberately avoids pretending to be a complete
    chemical parser; unparsed pieces are ignored instead of raising.
    """
    text = _clean_formula(formula)
    if not text or text in {"?", "."}:
        return {}

    def merge(target: dict[str, float], source: dict[str, float], factor: float = 1.0) -> None:
        for element, value in source.items():
            if element in ATOMIC_NUMBER:
                target[element] = target.get(element, 0.0) + float(value) * factor

    def parse_segment(segment: str) -> dict[str, float]:
        counts: dict[str, float] = {}
        index = 0
        while index < len(segment):
            if segment[index] == "(":
                depth = 1
                end = index + 1
                while end < len(segment) and depth:
                    if segment[end] == "(":
                        depth += 1
                    elif segment[end] == ")":
                        depth -= 1
                    end += 1
                inner = parse_segment(segment[index + 1:end - 1]) if depth == 0 else {}
                match = re.match(r"(\d+(?:\.\d+)?)", segment[end:])
                factor = float(match.group(1)) if match else 1.0
                merge(counts, inner, factor)
                index = end + (len(match.group(1)) if match else 0)
                continue
            match = re.match(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)", segment[index:])
            if not match:
                index += 1
                continue
            element, number = match.groups()
            if element in ATOMIC_NUMBER:
                counts[element] = counts.get(element, 0.0) + (float(number) if number else 1.0)
            index += len(match.group(0))
        return counts

    total: dict[str, float] = {}
    for piece in re.split(r"[+,]", text):
        coefficient = 1.0
        prefix = re.match(r"^(\d+(?:\.\d+)?)(?=[A-Z(])", piece)
        if prefix:
            coefficient = float(prefix.group(1))
            piece = piece[len(prefix.group(1)):]
        merge(total, parse_segment(piece), coefficient)
    return total


def formula_weight_from_counts(counts: dict[str, float]) -> float:
    return float(sum(ATOMIC_WEIGHTS.get(el, 0.0) * float(n) for el, n in counts.items()))


def composition_formula_from_sites(atoms: Iterable[dict]) -> str:
    """Return a deterministic cell-composition formula from occupied atom sites."""
    counts: dict[str, float] = defaultdict(float)
    for atom in atoms:
        element = str(atom.get("element") or "").capitalize()
        if element in ATOMIC_NUMBER:
            counts[element] += max(0.0, float(atom.get("occupancy", 1.0)))
    order = sorted(counts, key=lambda element: (element != "C", element != "H", element))
    parts = []
    for element in order:
        value = counts[element]
        if math.isclose(value, round(value), abs_tol=1e-8):
            count = str(int(round(value)))
        else:
            count = f"{value:.6g}"
        parts.append(f"{element}{'' if count == '1' else count}")
    return " ".join(parts)


def ensure_site_formula_metadata(structure: dict) -> bool:
    """Fill missing formula metadata from the explicit occupied P1 cell sites.

    Returns True only when metadata was added. Existing formula metadata is
    preserved because its relationship to Z can carry crystallographic meaning.
    """
    if parse_formula_counts(structure.get("formula", "")):
        return False
    formula = composition_formula_from_sites(structure.get("atoms") or [])
    if not formula:
        return False
    structure["formula"] = formula
    structure["formula_units_z"] = 1.0
    structure["formula_weight"] = formula_weight_from_counts(parse_formula_counts(formula))
    structure["_afruz_formula_source"] = "occupied atom-site cell composition"
    return True


def _cell_volume(cell: dict) -> float:
    metric = direct_metric_tensor(cell)
    return float(math.sqrt(float(np.linalg.det(metric))))


def _minimum_image_distance(cell: dict, a: dict, b: dict) -> tuple[float, list[float]]:
    metric = direct_metric_tensor(cell)
    first = np.asarray([a["x"], a["y"], a["z"]], dtype=float)
    second = np.asarray([b["x"], b["y"], b["z"]], dtype=float)
    delta = first - second
    delta -= np.round(delta)
    distance = float(math.sqrt(float(delta @ metric @ delta)))
    return distance, delta.tolist()


def _periodic_neighbor_pairs(
    cell: dict,
    atoms: list[dict],
    cutoff_angstrom: float,
) -> list[tuple[int, int]]:
    """Return candidate periodic pairs without an O(N²) all-pairs scan."""
    metric = direct_metric_tensor(cell)
    basis = np.linalg.cholesky(metric)
    fractional = np.asarray(
        [[float(atom[axis]) for axis in ("x", "y", "z")] for atom in atoms],
        dtype=float,
    )
    fractional %= 1.0
    shifts = np.asarray(
        [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
        ],
        dtype=float,
    )
    expanded = np.concatenate(
        [(fractional + shift) @ basis for shift in shifts],
        axis=0,
    )
    expanded_indices = np.tile(np.arange(len(atoms), dtype=int), len(shifts))
    tree = cKDTree(expanded)
    original = fractional @ basis
    candidate_pairs: set[tuple[int, int]] = set()
    for first, neighbors in enumerate(
        tree.query_ball_point(original, float(cutoff_angstrom))
    ):
        for expanded_index in neighbors:
            second = int(expanded_indices[expanded_index])
            if first == second:
                continue
            candidate_pairs.add(
                (first, second) if first < second else (second, first)
            )
    return sorted(candidate_pairs)


def _composition_fractions(counts: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in counts.values())
    if total <= 0:
        return {}
    return {key: max(0.0, float(value)) / total for key, value in counts.items()}


def validate_crystal_structure(
    structure: dict,
    *,
    duplicate_distance_angstrom: float = 0.08,
    severe_contact_scale: float = 0.45,
    short_contact_scale: float = 0.62,
    bond_scale_min: float = 0.72,
    bond_scale_max: float = 1.32,
) -> dict:
    """Run geometry and chemical plausibility checks for a parsed CIF structure.

    The function returns diagnostics only; it does not prove a structure. It is
    designed to be deterministic, lightweight, and safe for provisional models.
    """
    cell = structure.get("cell") or {}
    atoms = list(structure.get("atoms") or [])
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    if not atoms:
        return {
            "status": "Invalid structure — no atoms",
            "errors": ["No atomic coordinates are available."],
            "warnings": [],
            "notes": [],
            "scores": {"geometry": 0.0, "chemistry": 0.0, "density": 0.0, "overall": 0.0},
            "atom_count": 0,
            "cell_volume_angstrom3": None,
        }

    try:
        volume = _cell_volume(cell)
    except Exception as exc:
        return {
            "status": "Invalid structure — bad cell",
            "errors": [f"Unit-cell metric is invalid: {exc}"],
            "warnings": [],
            "notes": [],
            "scores": {"geometry": 0.0, "chemistry": 0.0, "density": 0.0, "overall": 0.0},
            "atom_count": len(atoms),
            "cell_volume_angstrom3": None,
        }

    element_counts = Counter()
    occupancy_counts: dict[str, float] = defaultdict(float)
    site_mass = 0.0
    unknown_elements = []
    for index, atom in enumerate(atoms, start=1):
        element = str(atom.get("element") or "").capitalize()
        if element not in ATOMIC_NUMBER:
            unknown_elements.append(str(atom.get("label") or index))
            continue
        element_counts[element] += 1
        occupancy = float(atom.get("occupancy", 1.0))
        occupancy_counts[element] += occupancy
        site_mass += ATOMIC_WEIGHTS.get(element, 0.0) * max(0.0, occupancy)
        for axis in ("x", "y", "z"):
            value = float(atom.get(axis, 0.0))
            if not np.isfinite(value):
                errors.append(f"{atom.get('label', index)} has a non-finite {axis} coordinate.")
        if not 0.0 <= occupancy <= 1.0:
            errors.append(f"{atom.get('label', index)} occupancy is outside 0–1.")
    if unknown_elements:
        errors.append(f"{len(unknown_elements)} atom site(s) have unknown element symbols.")

    metric = direct_metric_tensor(cell)
    duplicate_pairs = []
    severe_contacts = []
    short_contacts = []
    vdw_overlaps = []
    hydrogen_bond_pairs = []
    topology_excluded_pairs = []
    bonds = []
    coordination: dict[str, int] = defaultdict(int)
    metal_oxygen = []
    shortest = math.inf
    pair_records = []
    adjacency = [set() for _ in atoms]

    maximum_pair_cutoff = max(
        2.0 * max(COVALENT_RADII.values()) * bond_scale_max,
        2.0 * max(VDW_RADII.values()) * 0.72,
        3.5,
    )
    if len(atoms) > 2000:
        candidate_pairs = _periodic_neighbor_pairs(
            cell,
            atoms,
            maximum_pair_cutoff,
        )
        notes.append(
            f"Large-structure geometry validation used a periodic neighbor search "
            f"({len(candidate_pairs):,} candidate pairs within "
            f"{maximum_pair_cutoff:.2f} Å)."
        )
    else:
        candidate_pairs = [
            (i, j)
            for i in range(len(atoms))
            for j in range(i + 1, len(atoms))
        ]

    for i, j in candidate_pairs:
        first = atoms[i]
        element_1 = str(first.get("element") or "").capitalize()
        radius_1 = COVALENT_RADII.get(element_1, 0.85)
        vdw_1 = VDW_RADII.get(element_1, 1.8)
        second = atoms[j]
        element_2 = str(second.get("element") or "").capitalize()
        radius_2 = COVALENT_RADII.get(element_2, 0.85)
        vdw_2 = VDW_RADII.get(element_2, 1.8)
        distance, _ = _minimum_image_distance(cell, first, second)
        shortest = min(shortest, distance)
        pair = {
            "atom_1": first.get("label", f"{element_1}{i+1}"),
            "atom_2": second.get("label", f"{element_2}{j+1}"),
            "element_1": element_1,
            "element_2": element_2,
            "distance_angstrom": distance,
        }
        covalent_sum = radius_1 + radius_2
        if distance < duplicate_distance_angstrom:
            duplicate_pairs.append(pair)
        elif distance < severe_contact_scale * covalent_sum:
            severe_contacts.append(pair)
        elif distance < short_contact_scale * covalent_sum:
            short_contacts.append(pair)
        is_bond = bond_scale_min * covalent_sum <= distance <= bond_scale_max * covalent_sum
        pair_records.append((i, j, distance, pair, covalent_sum, vdw_1 + vdw_2, is_bond))
        if is_bond:
            bonds.append(pair)
            adjacency[i].add(j)
            adjacency[j].add(i)
            coordination[pair["atom_1"]] += 1
            coordination[pair["atom_2"]] += 1
        if element_1 in METALS and element_2 == "O":
            lo, hi = COMMON_METAL_O_RANGES.get(element_1, (1.65, 2.85))
            if lo <= distance <= hi:
                metal_oxygen.append(pair)
        elif element_2 in METALS and element_1 == "O":
            lo, hi = COMMON_METAL_O_RANGES.get(element_2, (1.65, 2.85))
            if lo <= distance <= hi:
                metal_oxygen.append(pair)

    donor_elements = {"N", "O", "F", "S"}
    acceptor_elements = {"N", "O", "F", "S"}
    hydrogen_donors: dict[int, int] = {}
    for h_index, atom in enumerate(atoms):
        if str(atom.get("element") or "").capitalize() != "H":
            continue
        candidates = []
        for neighbor in adjacency[h_index]:
            element = str(atoms[neighbor].get("element") or "").capitalize()
            if element not in donor_elements:
                continue
            distance, _ = _minimum_image_distance(cell, atom, atoms[neighbor])
            if distance <= 1.35:
                candidates.append((distance, neighbor))
        if candidates:
            hydrogen_donors[h_index] = min(candidates)[1]

    overlap_penalty = 0.0
    for i, j, distance, pair, covalent_sum, vdw_sum, is_bond in pair_records:
        vdw_limit = 0.72 * vdw_sum
        if not (distance < vdw_limit and distance > 1.25 * covalent_sum):
            continue
        if is_bond:
            topology_excluded_pairs.append(dict(pair, reason="direct bond"))
            continue
        shared = adjacency[i] & adjacency[j]
        if shared:
            topology_excluded_pairs.append(dict(pair, reason="shared bonded neighbor (1–3 contact)"))
            continue
        h_index = i if str(atoms[i].get("element") or "").capitalize() == "H" else (
            j if str(atoms[j].get("element") or "").capitalize() == "H" else None
        )
        acceptor = j if h_index == i else i if h_index == j else None
        donor = hydrogen_donors.get(h_index) if h_index is not None else None
        if donor is not None and acceptor is not None and acceptor != donor:
            acceptor_element = str(atoms[acceptor].get("element") or "").capitalize()
            if acceptor_element in acceptor_elements and 1.50 <= distance <= 2.60:
                donor_acceptor, _ = _minimum_image_distance(cell, atoms[donor], atoms[acceptor])
                donor_vector = np.asarray([
                    float(atoms[donor][axis]) - float(atoms[h_index][axis])
                    for axis in ("x", "y", "z")
                ])
                acceptor_vector = np.asarray([
                    float(atoms[acceptor][axis]) - float(atoms[h_index][axis])
                    for axis in ("x", "y", "z")
                ])
                donor_vector -= np.round(donor_vector)
                acceptor_vector -= np.round(acceptor_vector)
                denominator = math.sqrt(
                    float(donor_vector @ metric @ donor_vector)
                    * float(acceptor_vector @ metric @ acceptor_vector)
                )
                cosine = 1.0 if denominator <= 0 else float(
                    np.clip((donor_vector @ metric @ acceptor_vector) / denominator, -1.0, 1.0)
                )
                angle = math.degrees(math.acos(cosine))
                if 2.40 <= donor_acceptor <= 3.40 and angle >= 145.0:
                    hydrogen_bond_pairs.append({
                        "donor": atoms[donor].get("label"),
                        "hydrogen": atoms[h_index].get("label"),
                        "acceptor": atoms[acceptor].get("label"),
                        "h_acceptor_distance_angstrom": distance,
                        "donor_acceptor_distance_angstrom": donor_acceptor,
                        "angle_deg": angle,
                    })
                    continue
        vdw_overlaps.append(pair)
        amount = (vdw_limit - distance) / max(vdw_limit, 1e-9)
        overlap_penalty += amount * amount
    if duplicate_pairs:
        errors.append(f"{len(duplicate_pairs)} duplicate or nearly duplicate atom pair(s) were found.")
    if severe_contacts:
        errors.append(f"{len(severe_contacts)} chemically impossible short contact(s) were found.")
    if short_contacts:
        warnings.append(f"{len(short_contacts)} unusually short contact(s) require review.")
    if vdw_overlaps:
        warnings.append(f"{len(vdw_overlaps)} nonbonded van der Waals overlap(s) require review.")
    if topology_excluded_pairs:
        notes.append(
            f"{len(topology_excluded_pairs)} close bonded/1–3 contact(s) were excluded from the nonbonded-overlap count."
        )
    if hydrogen_bond_pairs:
        notes.append(
            f"{len(hydrogen_bond_pairs)} plausible D–H···A contact(s) were classified as hydrogen bonds, not van der Waals overlaps."
        )

    formula_counts = parse_formula_counts(structure.get("formula", ""))
    if not formula_counts:
        warnings.append(
            f"Chemical formula is missing; occupied sites imply {composition_formula_from_sites(atoms) or 'an unknown composition'}."
        )
    formula_fractions = _composition_fractions(formula_counts)
    site_fractions = _composition_fractions(dict(occupancy_counts))
    formula_mismatch = []
    if formula_fractions and site_fractions:
        for element in sorted(set(formula_fractions) | set(site_fractions)):
            difference = abs(formula_fractions.get(element, 0.0) - site_fractions.get(element, 0.0))
            if difference > 0.08:
                formula_mismatch.append({
                    "element": element,
                    "formula_fraction": formula_fractions.get(element, 0.0),
                    "site_fraction": site_fractions.get(element, 0.0),
                    "absolute_difference": difference,
                })
        if formula_mismatch:
            warnings.append("Atom-site composition does not match the supplied formula fractions.")

    density_from_sites = 1.66053906660 * site_mass / volume if volume > 0 else None
    z_value = structure.get("formula_units_z")
    formula_weight = structure.get("formula_weight") or formula_weight_from_counts(formula_counts)
    density_from_formula_z = None
    if z_value and formula_weight:
        density_from_formula_z = 1.66053906660 * float(z_value) * float(formula_weight) / volume
    density = density_from_formula_z if density_from_formula_z else density_from_sites
    if density is not None:
        if density < 0.20 or density > 12.0:
            warnings.append(f"Estimated density {density:.3g} g/cm³ is outside a broad expected range.")
        elif density < 0.45 or density > 6.5:
            notes.append(f"Estimated density {density:.3g} g/cm³ is unusual; check Z, solvent and composition.")

    elements = set(element_counts)
    metal_elements = sorted(elements & METALS)
    is_mof_like = bool(metal_elements and "O" in elements and "C" in elements)
    metal_coordination_summary = []
    if metal_elements:
        for atom in atoms:
            element = str(atom.get("element") or "").capitalize()
            if element not in METALS:
                continue
            label = atom.get("label")
            oxygen_count = sum(
                1 for pair in metal_oxygen
                if label in {pair["atom_1"], pair["atom_2"]}
            )
            metal_coordination_summary.append({
                "atom": label,
                "element": element,
                "oxygen_coordination": oxygen_count,
                "total_bond_count": coordination.get(label, 0),
            })
            if is_mof_like and oxygen_count == 0:
                warnings.append(f"{label}: no plausible metal–oxygen coordination was detected.")
    if is_mof_like:
        notes.append("MOF-like composition detected; review linker connectivity, metal-node geometry and solvent/guest disorder.")

    geometry_penalty = 35 * len(errors) + 12 * len(short_contacts) + 8 * len(vdw_overlaps)
    chemistry_penalty = 15 * len(formula_mismatch) + (10 if is_mof_like and not metal_oxygen else 0)
    density_penalty = 0
    if density is not None and (density < 0.20 or density > 12.0):
        density_penalty = 35
    elif density is not None and (density < 0.45 or density > 6.5):
        density_penalty = 12
    geometry_score = max(0.0, 100.0 - geometry_penalty)
    chemistry_score = max(0.0, 100.0 - chemistry_penalty)
    density_score = max(0.0, 100.0 - density_penalty)
    overall = float(0.45 * geometry_score + 0.35 * chemistry_score + 0.20 * density_score)

    if errors:
        status = "Invalid structure — geometry errors"
    elif overall >= 85 and not warnings:
        status = "Plausibility passed"
    elif overall >= 65:
        status = "Plausibility review required"
    else:
        status = "Plausibility weak — review before refinement"

    return {
        "status": status,
        "errors": errors,
        "warnings": list(dict.fromkeys(warnings)),
        "notes": list(dict.fromkeys(notes)),
        "scores": {
            "geometry": float(geometry_score),
            "chemistry": float(chemistry_score),
            "density": float(density_score),
            "overall": float(overall),
        },
        "atom_count": len(atoms),
        "element_counts": dict(element_counts),
        "occupancy_counts": dict(occupancy_counts),
        "cell_volume_angstrom3": volume,
        "density_g_cm3": None if density is None else float(density),
        "density_source": "formula_z" if density_from_formula_z else "site_mass",
        "shortest_contact_angstrom": None if not np.isfinite(shortest) else float(shortest),
        "duplicate_pairs": duplicate_pairs[:50],
        "duplicate_pair_count": len(duplicate_pairs),
        "severe_contact_pairs": severe_contacts[:50],
        "severe_contact_count": len(severe_contacts),
        "short_contact_pairs": short_contacts[:50],
        "short_contact_count": len(short_contacts),
        "vdw_overlap_pairs": vdw_overlaps[:50],
        "vdw_overlap_count": len(vdw_overlaps),
        "nonbonded_overlap_penalty": float(overlap_penalty),
        "hydrogen_bond_pairs": hydrogen_bond_pairs[:80],
        "hydrogen_bond_count": len(hydrogen_bond_pairs),
        "topology_excluded_pairs": topology_excluded_pairs[:80],
        "topology_excluded_pair_count": len(topology_excluded_pairs),
        "bond_count": len(bonds),
        "coordination_summary": dict(coordination),
        "metal_oxygen_bonds": metal_oxygen[:80],
        "metal_coordination_summary": metal_coordination_summary,
        "formula_counts": formula_counts,
        "formula_mismatch": formula_mismatch,
        "is_mof_like": is_mof_like,
        "scientific_boundary": (
            "Plausibility checks identify geometry, composition and coordination problems. "
            "They do not replace checkCIF/PLATON, expert chemical review, bond-valence analysis, "
            "or diffraction-based validation."
        ),
    }


def structure_refinement_gate(validation: dict, minimum_score: float = 65.0) -> dict:
    """Turn plausibility diagnostics into an explicit refinement/publication gate.

    Refinement may proceed for a passed or review-required structure. Invalid or
    weak structures require an explicit expert override. Publication-ready is a
    stricter label and is granted only when the plausibility check passes without
    errors or warnings.
    """
    validation = validation if isinstance(validation, dict) else {}
    status = str(validation.get("status") or "Validation unavailable")
    errors = [str(value) for value in validation.get("errors") or []]
    warnings = [str(value) for value in validation.get("warnings") or []]
    scores = validation.get("scores") if isinstance(validation.get("scores"), dict) else {}
    try:
        score = float(scores.get("overall"))
    except (TypeError, ValueError):
        score = 0.0
    blocked = bool(errors) or status.lower().startswith("invalid") or score < float(minimum_score)
    reasons = list(errors)
    if score < float(minimum_score):
        reasons.append(f"Plausibility score {score:.1f} is below the {float(minimum_score):.1f} refinement threshold.")
    reasons.extend(warnings)
    return {
        "passed": not blocked,
        "expert_override_required": blocked,
        "publication_ready": (
            not errors
            and not warnings
            and status.lower().startswith("plausibility passed")
            and score >= float(minimum_score)
        ),
        "status": status,
        "score": score,
        "minimum_score": float(minimum_score),
        "reasons": list(dict.fromkeys(reasons)),
    }
