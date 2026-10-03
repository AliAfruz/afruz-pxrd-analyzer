from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import hashlib
import json
import math
from typing import Iterable

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .crystallography import ATOMIC_NUMBER, direct_metric_tensor, parse_cif_text
from .text_export import write_mapping_txt, write_table_txt, write_manifest_txt
from .structure_validation import (
    ATOMIC_WEIGHTS,
    COVALENT_RADII,
    composition_formula_from_sites,
    formula_weight_from_counts,
    parse_formula_counts,
    validate_crystal_structure,
)


@dataclass(frozen=True)
class DirectSpaceSettings:
    """Controls for the Phase 18.2 direct-space prototype.

    The first implementation is deliberately conservative: P1 atom-site
    placement only, bounded fractional coordinates, robust intensity matching,
    and strong geometry penalties. It creates provisional models for validation,
    not final solved structures.
    """

    max_atoms: int = 18
    random_seed: int = 182
    global_iterations: int = 18
    population_size: int = 5
    local_polish_iterations: int = 120
    collision_weight: float = 40.0
    density_weight: float = 2.0
    preferred_min_density_g_cm3: float = 0.20
    preferred_max_density_g_cm3: float = 8.0
    minimum_reflections: int = 3
    intensity_floor: float = 1e-9
    use_lorentz_weight: bool = False


def stable_json_hash(data: object) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_structure_solution_package(
    output_dir: str | Path,
    *,
    candidate_cell: dict,
    extracted_reflections: Iterable[dict],
    proposed_space_group: str = "P 1",
    formula_hint: str = "",
    source_dataset_uid: str = "",
    master_peak_checksum: str = "",
) -> dict:
    """Create a reproducible no-CIF structure-solution handoff package.

    This does not solve the structure by itself; it creates auditable inputs for
    direct-space or external structure-solution tools.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    reflections = [dict(row) for row in extracted_reflections]
    hkl_path = directory / "extracted_reflections.hkl.csv"
    with hkl_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["h", "k", "l", "d_spacing_angstrom", "intensity", "sigma", "peak_uuid"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in reflections:
            writer.writerow(row)
    hkl_txt = write_table_txt(
        directory / "extracted_reflections.txt",
        reflections,
        columns=["h", "k", "l", "d_spacing_angstrom", "intensity", "sigma", "peak_uuid"],
        title="Structure-solution extracted reflections",
        metadata={"source_dataset_uid": source_dataset_uid},
        backup=False,
    )
    manifest = {
        "package_type": "Afruz provisional structure-solution handoff",
        "candidate_cell": dict(candidate_cell),
        "proposed_space_group": proposed_space_group,
        "formula_hint": formula_hint,
        "source_dataset_uid": source_dataset_uid,
        "master_peak_checksum": master_peak_checksum,
        "reflection_count": len(reflections),
        "reflections_sha256": stable_json_hash(reflections),
        "scientific_boundary": (
            "This package supports structure solution from an indexed cell and extracted intensities. "
            "It is not a solved CIF and must be chemically and diffraction validated."
        ),
    }
    manifest_path = directory / "structure_solution_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_txt = write_mapping_txt(
        directory / "structure_solution_manifest.txt",
        manifest,
        title="Provisional structure-solution handoff manifest",
        backup=False,
    )
    readme_path = directory / "README_STRUCTURE_SOLUTION.txt"
    readme_path.write_text(
        "Afruz structure-solution handoff package\n"
        "=======================================\n\n"
        "Use the HKL CSV, candidate cell, proposed space group and formula hint as inputs for direct-space or external structure-solution tools.\n"
        "A successful structure proposal must return a CIF with atomic coordinates before Phase 11 Rietveld refinement.\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(directory.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "hkl_path": str(hkl_path.resolve()),
        "hkl_txt_path": hkl_txt["txt_path"],
        "manifest_txt_path": manifest_txt["txt_path"],
        "readme_path": str(readme_path.resolve()),
        "manifest": manifest,
    }


def validate_solution_candidate_record(candidate: dict) -> dict:
    has_atoms = int(candidate.get("atom_count") or 0) > 0
    has_cell = bool(candidate.get("cell"))
    if has_atoms and has_cell:
        status = "Ready for provisional Rietveld validation"
        blocking = []
    elif has_cell:
        status = "Cell-only candidate — needs atomic coordinates"
        blocking = ["No atomic coordinates are available."]
    else:
        status = "Invalid candidate — missing cell"
        blocking = ["No unit-cell information is available."]
    return {
        "status": status,
        "blocking_reasons": blocking,
        "can_send_to_rietveld": not blocking,
        "scientific_boundary": (
            "Only candidates with atomic coordinates can enter structure-based Rietveld refinement. Cell-only results belong in Pawley/Le Bail verification."
        ),
    }


def formula_hint_to_atom_sites(
    formula_hint: str,
    *,
    max_atoms: int = 18,
) -> list[dict]:
    """Convert a formula hint into a small set of P1 atom-site templates.

    Counts are rounded to practical integer sites and scaled down if the formula
    is too large for this prototype. Unknown formulas fall back to a single C
    placeholder so that a package can still be created, but warnings will mark
    the result as weak.
    """
    counts = parse_formula_counts(formula_hint)
    if not counts:
        counts = {"C": 1.0}
    integer_counts: dict[str, int] = {}
    for element, value in counts.items():
        if element not in ATOMIC_NUMBER:
            continue
        integer_counts[element] = max(1, int(round(float(value))))
    if not integer_counts:
        integer_counts = {"C": 1}
    total = sum(integer_counts.values())
    if total > max_atoms:
        scale = max_atoms / total
        scaled: dict[str, int] = {}
        for element, value in integer_counts.items():
            scaled[element] = max(1, int(round(value * scale)))
        # If rounding still exceeds the cap, remove the lightest surplus sites.
        while sum(scaled.values()) > max_atoms:
            removable = sorted(
                (el for el, n in scaled.items() if n > 1),
                key=lambda el: ATOMIC_WEIGHTS.get(el, 999.0),
            )
            if not removable:
                break
            scaled[removable[0]] -= 1
        integer_counts = scaled
    sites = []
    for element in sorted(integer_counts, key=lambda el: (-ATOMIC_NUMBER.get(el, 0), el)):
        for index in range(integer_counts[element]):
            sites.append(
                {
                    "label": f"{element}{index + 1}",
                    "element": element,
                    "occupancy": 1.0,
                    "b_iso": 1.0,
                }
            )
    return sites


def _clean_reflections(extracted_reflections: Iterable[dict], settings: DirectSpaceSettings) -> list[dict]:
    rows = []
    for row in extracted_reflections:
        try:
            h = int(row.get("h"))
            k = int(row.get("k"))
            l = int(row.get("l"))
            intensity = float(row.get("intensity"))
        except (TypeError, ValueError):
            continue
        if (h, k, l) == (0, 0, 0) or not np.isfinite(intensity) or intensity <= settings.intensity_floor:
            continue
        sigma = row.get("sigma", None)
        try:
            sigma_value = float(sigma)
        except (TypeError, ValueError):
            sigma_value = math.sqrt(max(intensity, 1.0))
        rows.append(
            {
                "h": h,
                "k": k,
                "l": l,
                "intensity": intensity,
                "sigma": max(sigma_value, settings.intensity_floor),
                "peak_uuid": row.get("peak_uuid", ""),
            }
        )
    rows.sort(key=lambda item: (item["h"] ** 2 + item["k"] ** 2 + item["l"] ** 2, item["h"], item["k"], item["l"]))
    return rows


def _atoms_from_vector(sites: list[dict], vector: np.ndarray) -> list[dict]:
    coords = np.asarray(vector, dtype=float).reshape((len(sites), 3)) % 1.0
    atoms = []
    for site, xyz in zip(sites, coords):
        atom = dict(site)
        atom.update({"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])})
        atoms.append(atom)
    return atoms


def calculated_reflection_intensities(
    atoms: list[dict],
    reflections: Iterable[dict],
) -> np.ndarray:
    """Calculate approximate |F|² intensities for supplied hkl rows.

    This is a lightweight direct-space objective, not a full Rietveld structure
    factor engine. It is suitable for ranking provisional placements.
    """
    values = []
    for reflection in reflections:
        h = int(reflection["h"])
        k = int(reflection["k"])
        l = int(reflection["l"])
        structure_factor = 0.0 + 0.0j
        for atom in atoms:
            element = str(atom.get("element", "C")).capitalize()
            scattering = float(ATOMIC_NUMBER.get(element, 6))
            occupancy = max(0.0, float(atom.get("occupancy", 1.0)))
            phase = 2.0 * math.pi * (
                h * float(atom.get("x", 0.0))
                + k * float(atom.get("y", 0.0))
                + l * float(atom.get("z", 0.0))
            )
            structure_factor += occupancy * scattering * complex(math.cos(phase), math.sin(phase))
        values.append(abs(structure_factor) ** 2)
    return np.asarray(values, dtype=float)


def _optimal_scale(observed: np.ndarray, calculated: np.ndarray, sigma: np.ndarray) -> float:
    weights = 1.0 / np.maximum(sigma, 1e-9) ** 2
    denominator = float(np.sum(weights * calculated * calculated))
    if denominator <= 0 or not np.isfinite(denominator):
        return 0.0
    return max(0.0, float(np.sum(weights * observed * calculated) / denominator))


def _minimum_image_penalty(cell: dict, atoms: list[dict]) -> tuple[float, list[dict]]:
    if len(atoms) < 2:
        return 0.0, []
    metric = direct_metric_tensor(cell)
    penalty = 0.0
    contacts: list[dict] = []
    for i, first in enumerate(atoms):
        for j in range(i + 1, len(atoms)):
            second = atoms[j]
            delta = np.asarray(
                [
                    float(first["x"]) - float(second["x"]),
                    float(first["y"]) - float(second["y"]),
                    float(first["z"]) - float(second["z"]),
                ],
                dtype=float,
            )
            delta -= np.round(delta)
            distance = float(math.sqrt(float(delta @ metric @ delta)))
            r1 = COVALENT_RADII.get(str(first.get("element", "C")).capitalize(), 0.76)
            r2 = COVALENT_RADII.get(str(second.get("element", "C")).capitalize(), 0.76)
            severe_limit = 0.55 * (r1 + r2)
            if distance < severe_limit:
                amount = (severe_limit - distance) / max(severe_limit, 1e-6)
                penalty += amount * amount
                contacts.append(
                    {
                        "atom1": first.get("label", f"A{i+1}"),
                        "atom2": second.get("label", f"A{j+1}"),
                        "distance_angstrom": distance,
                        "limit_angstrom": severe_limit,
                    }
                )
    contacts.sort(key=lambda row: row["distance_angstrom"])
    return penalty, contacts[:10]


def _density_penalty(validation: dict, settings: DirectSpaceSettings) -> float:
    density = validation.get("density_g_cm3")
    try:
        value = float(density)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(value) or value <= 0:
        return 1.0
    if value < settings.preferred_min_density_g_cm3:
        return ((settings.preferred_min_density_g_cm3 - value) / settings.preferred_min_density_g_cm3) ** 2
    if value > settings.preferred_max_density_g_cm3:
        return ((value - settings.preferred_max_density_g_cm3) / settings.preferred_max_density_g_cm3) ** 2
    return 0.0


def score_direct_space_model(
    *,
    cell: dict,
    atoms: list[dict],
    reflections: Iterable[dict],
    settings: DirectSpaceSettings | None = None,
) -> dict:
    settings = settings or DirectSpaceSettings()
    rows = list(reflections)
    observed = np.asarray([float(row["intensity"]) for row in rows], dtype=float)
    sigma = np.asarray([float(row.get("sigma", math.sqrt(max(row["intensity"], 1.0)))) for row in rows], dtype=float)
    calculated = calculated_reflection_intensities(atoms, rows)
    scale = _optimal_scale(observed, calculated, sigma)
    residual = (observed - scale * calculated) / np.maximum(sigma, settings.intensity_floor)
    intensity_chi2 = float(np.mean(residual**2)) if len(residual) else float("inf")
    collision_penalty, short_contacts = _minimum_image_penalty(cell, atoms)
    structure = {"cell": dict(cell), "atoms": atoms, "formula": ""}
    validation = validate_crystal_structure(structure)
    nonbonded_overlap_penalty = float(validation.get("nonbonded_overlap_penalty", 0.0))
    collision_penalty += nonbonded_overlap_penalty
    density_penalty = _density_penalty(validation, settings)
    objective = (
        intensity_chi2
        + settings.collision_weight * float(collision_penalty)
        + settings.density_weight * float(density_penalty)
    )
    correlation = 0.0
    if len(observed) >= 2 and np.std(observed) > 0 and np.std(calculated) > 0:
        correlation = float(np.corrcoef(observed, calculated)[0, 1])
    return {
        "objective": float(objective),
        "intensity_chi2": intensity_chi2,
        "scale": float(scale),
        "correlation": correlation,
        "collision_penalty": float(collision_penalty),
        "nonbonded_overlap_penalty": nonbonded_overlap_penalty,
        "nonbonded_overlap_count": int(validation.get("vdw_overlap_count", 0)),
        "density_penalty": float(density_penalty),
        "short_contacts": short_contacts,
        "calculated_intensities": calculated.tolist(),
        "scaled_calculated_intensities": (scale * calculated).tolist(),
        "residuals": residual.tolist(),
        "validation": validation,
    }


def build_provisional_cif_text(
    *,
    cell: dict,
    atoms: list[dict],
    data_name: str = "afruz_direct_space_solution",
    proposed_space_group: str = "P 1",
    formula_hint: str = "",
) -> str:
    formula = formula_hint or composition_formula_from_sites(atoms) or _composition_from_atoms(atoms)
    formula_weight = formula_weight_from_counts(parse_formula_counts(formula))
    lines = [
        f"data_{_safe_cif_name(data_name)}",
        f"_chemical_formula_sum '{formula}'",
        f"_chemical_formula_weight {formula_weight:.6f}",
        "_cell_formula_units_Z 1",
        "_audit_creation_method 'Afruz direct-space provisional P1 model; site-derived cell composition when no formula was supplied'",
        f"_space_group_name_H-M_alt '{proposed_space_group or 'P 1'}'",
        f"_cell_length_a {float(cell['a']):.8f}",
        f"_cell_length_b {float(cell['b']):.8f}",
        f"_cell_length_c {float(cell['c']):.8f}",
        f"_cell_angle_alpha {float(cell['alpha']):.8f}",
        f"_cell_angle_beta {float(cell['beta']):.8f}",
        f"_cell_angle_gamma {float(cell['gamma']):.8f}",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_B_iso_or_equiv",
    ]
    for atom in atoms:
        lines.append(
            f"{atom.get('label', atom.get('element', 'X'))} "
            f"{atom.get('element', 'C')} "
            f"{float(atom.get('x', 0.0)) % 1.0:.8f} "
            f"{float(atom.get('y', 0.0)) % 1.0:.8f} "
            f"{float(atom.get('z', 0.0)) % 1.0:.8f} "
            f"{float(atom.get('occupancy', 1.0)):.6f} "
            f"{float(atom.get('b_iso', 1.0)):.6f}"
        )
    return "\n".join(lines) + "\n"


def _safe_cif_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value or "solution"))
    return text.strip("_") or "solution"


def _composition_from_atoms(atoms: list[dict]) -> str:
    return composition_formula_from_sites(atoms)


def solve_direct_space_structure(
    *,
    candidate_cell: dict,
    extracted_reflections: Iterable[dict],
    formula_hint: str,
    proposed_space_group: str = "P 1",
    settings: DirectSpaceSettings | None = None,
    progress_callback=None,
) -> dict:
    """Run a lightweight direct-space atom-placement search.

    Prototype limitations:
    * P1 fractional atom placement only;
    * approximate atomic-number structure factors;
    * no molecular rigid bodies or symmetry expansion yet;
    * output is a provisional CIF candidate requiring validation.
    """
    settings = settings or DirectSpaceSettings()
    cell = dict(candidate_cell)
    reflections = _clean_reflections(extracted_reflections, settings)
    if len(reflections) < settings.minimum_reflections:
        raise ValueError(
            f"At least {settings.minimum_reflections} extracted hkl intensities are required."
        )
    sites = formula_hint_to_atom_sites(formula_hint, max_atoms=settings.max_atoms)
    variable_count = 3 * len(sites)
    rng = np.random.default_rng(settings.random_seed)

    def objective(vector: np.ndarray) -> float:
        atoms = _atoms_from_vector(sites, vector)
        return score_direct_space_model(
            cell=cell,
            atoms=atoms,
            reflections=reflections,
            settings=settings,
        )["objective"]

    bounds = [(0.0, 1.0)] * variable_count
    best_vector = rng.random(variable_count)
    best_value = objective(best_vector)
    method_notes = ["random initial placement"]
    if progress_callback:
        progress_callback({"stage": "initial", "objective": best_value})

    if variable_count <= 54 and settings.global_iterations > 0:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=int(settings.global_iterations),
            popsize=max(2, int(settings.population_size)),
            seed=int(settings.random_seed),
            polish=False,
            updating="immediate",
            workers=1,
            tol=0.02,
        )
        if np.isfinite(result.fun) and float(result.fun) < best_value:
            best_vector = np.asarray(result.x, dtype=float)
            best_value = float(result.fun)
            method_notes.append("differential evolution")
        if progress_callback:
            progress_callback({"stage": "global", "objective": best_value})

    if settings.local_polish_iterations > 0:
        local = minimize(
            objective,
            best_vector,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(settings.local_polish_iterations), "ftol": 1e-8},
        )
        if np.isfinite(local.fun) and float(local.fun) < best_value:
            best_vector = np.asarray(local.x, dtype=float)
            best_value = float(local.fun)
            method_notes.append("bounded local polish")
        if progress_callback:
            progress_callback({"stage": "local", "objective": best_value})

    atoms = _atoms_from_vector(sites, best_vector)
    score = score_direct_space_model(
        cell=cell,
        atoms=atoms,
        reflections=reflections,
        settings=settings,
    )
    cif_text = build_provisional_cif_text(
        cell=cell,
        atoms=atoms,
        data_name="afruz_direct_space_solution",
        proposed_space_group=proposed_space_group,
        formula_hint=formula_hint,
    )
    parsed = parse_cif_text(cif_text)
    plausibility = validate_crystal_structure(parsed)
    status = _classify_direct_space_solution(score, plausibility, len(reflections))
    return {
        "status": status,
        "candidate_type": "Direct-space provisional structure",
        "settings": asdict(settings),
        "candidate_cell": cell,
        "proposed_space_group": proposed_space_group,
        "formula_hint": formula_hint,
        "atom_count": len(atoms),
        "reflection_count": len(reflections),
        "objective": score["objective"],
        "intensity_chi2": score["intensity_chi2"],
        "intensity_correlation": score["correlation"],
        "scale": score["scale"],
        "short_contacts": score["short_contacts"],
        "plausibility": plausibility,
        "atoms": atoms,
        "cif_text": cif_text,
        "cif_sha256": hashlib.sha256(cif_text.encode("utf-8")).hexdigest(),
        "reflections_sha256": stable_json_hash(reflections),
        "method_notes": method_notes,
        "scientific_boundary": (
            "This is a direct-space prototype solution. It is a provisional structural hypothesis and must pass Pawley/Le Bail consistency, Rietveld refinement, chemical plausibility review, and competing-model comparison."
        ),
    }


def _classify_direct_space_solution(score: dict, plausibility: dict, reflection_count: int) -> str:
    objective = float(score.get("objective", float("inf")))
    correlation = float(score.get("correlation", 0.0))
    overall = float((plausibility.get("scores") or {}).get("overall", 0.0))
    if reflection_count < 5:
        return "Exploratory — too few reflections"
    if objective < 8.0 and correlation > 0.80 and overall >= 60.0:
        return "Provisional candidate — validate"
    if objective < 25.0 and overall >= 35.0:
        return "Weak provisional candidate — review"
    return "Exploratory candidate — low confidence"


def create_direct_space_solution_package(
    output_dir: str | Path,
    *,
    candidate_cell: dict,
    extracted_reflections: Iterable[dict],
    formula_hint: str,
    proposed_space_group: str = "P 1",
    source_dataset_uid: str = "",
    master_peak_checksum: str = "",
    settings: DirectSpaceSettings | None = None,
) -> dict:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    solution = solve_direct_space_structure(
        candidate_cell=candidate_cell,
        extracted_reflections=extracted_reflections,
        formula_hint=formula_hint,
        proposed_space_group=proposed_space_group,
        settings=settings,
    )
    cif_path = directory / "afruz_direct_space_provisional.cif"
    json_path = directory / "afruz_direct_space_solution.json"
    cif_path.write_text(solution["cif_text"], encoding="utf-8")
    record = dict(solution)
    record.pop("cif_text", None)
    record.update(
        {
            "source_dataset_uid": source_dataset_uid,
            "master_peak_checksum": master_peak_checksum,
            "cif_path": str(cif_path.resolve()),
        }
    )
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    solution_txt = write_mapping_txt(
        directory / "afruz_direct_space_solution.txt",
        record,
        title="Afruz direct-space provisional solution",
        metadata={"source_dataset_uid": source_dataset_uid},
        backup=False,
    )
    readme_path = directory / "README_DIRECT_SPACE_PROTOTYPE.txt"
    readme_path.write_text(
        "Afruz direct-space structure-solution prototype\n"
        "==============================================\n\n"
        "This package contains a provisional P1 atom-placement candidate created from an indexed cell, a formula hint and extracted hkl intensities.\n"
        "It is not a final structure. Validate the CIF chemically and by whole-pattern refinement before using it as evidence.\n",
        encoding="utf-8",
    )
    manifest = {
        "package_type": "Afruz direct-space provisional structure package",
        "source_dataset_uid": source_dataset_uid,
        "master_peak_checksum": master_peak_checksum,
        "formula_hint": formula_hint,
        "candidate_cell": dict(candidate_cell),
        "proposed_space_group": proposed_space_group,
        "status": solution["status"],
        "objective": solution["objective"],
        "cif_sha256": solution["cif_sha256"],
        "solution_json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
        "scientific_boundary": solution["scientific_boundary"],
    }
    manifest_path = directory / "direct_space_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_txt = write_mapping_txt(
        directory / "direct_space_manifest.txt",
        manifest,
        title="Direct-space solution manifest",
        backup=False,
    )
    return {
        "output_dir": str(directory.resolve()),
        "cif_path": str(cif_path.resolve()),
        "solution_json_path": str(json_path.resolve()),
        "solution_txt_path": solution_txt["txt_path"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_txt_path": manifest_txt["txt_path"],
        "readme_path": str(readme_path.resolve()),
        "solution": solution,
        "manifest": manifest,
    }


@dataclass(frozen=True)
class FragmentDirectSpaceSettings:
    """Controls for Phase 18.4 fragment-based direct-space solving.

    This is still a prototype, but it is substantially more crystallographic
    than free P1 atom placement: each fragment is moved by translation,
    orientation and optional torsion variables while internal geometry is
    preserved unless a torsion is explicitly declared.
    """

    random_seed: int = 184
    global_iterations: int = 16
    population_size: int = 5
    local_polish_iterations: int = 120
    maximum_variables: int = 72
    minimum_reflections: int = 3
    intensity_floor: float = 1e-9
    collision_weight: float = 45.0
    density_weight: float = 2.0
    torsion_weight: float = 0.02
    preferred_min_density_g_cm3: float = 0.20
    preferred_max_density_g_cm3: float = 8.0


def cell_cartesian_basis(cell: dict) -> np.ndarray:
    """Return a 3x3 matrix whose columns are the unit-cell basis vectors.

    Fractional coordinates f are converted to Cartesian coordinates by
    ``basis @ f``. The inverse converts Cartesian displacement vectors in
    angstrom to fractional-coordinate displacements.
    """
    a = float(cell["a"])
    b = float(cell["b"])
    c = float(cell["c"])
    alpha = math.radians(float(cell["alpha"]))
    beta = math.radians(float(cell["beta"]))
    gamma = math.radians(float(cell["gamma"]))
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1e-12:
        raise ValueError("Unit-cell gamma angle is singular.")
    ax = np.array([a, 0.0, 0.0], dtype=float)
    by = np.array([b * math.cos(gamma), b * sin_gamma, 0.0], dtype=float)
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0 or not np.isfinite(cz2):
        raise ValueError("Unit-cell basis is not physically valid.")
    cz = math.sqrt(cz2)
    return np.column_stack([ax, by, np.array([cx, cy, cz], dtype=float)])


def euler_rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Return a Z-Y-X Euler rotation matrix in radians."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cg, -sg], [0.0, sg, cg]])
    return rz @ ry @ rx


def _axis_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t*x*x + c, t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c],
        ],
        dtype=float,
    )


def build_rigid_fragment(
    name: str,
    atoms: Iterable[dict],
    *,
    torsions: Iterable[dict] | None = None,
    center: bool = True,
) -> dict:
    """Create a reusable fragment record from Cartesian atom coordinates.

    Atoms must provide ``element`` and Cartesian coordinates ``x``, ``y`` and
    ``z`` in angstrom. The coordinates are stored relative to the fragment
    origin so rotations and translations behave predictably.
    """
    rows: list[dict] = []
    for index, atom in enumerate(atoms):
        element = str(atom.get("element", atom.get("type", "C"))).capitalize()
        if element not in ATOMIC_NUMBER:
            element = "C"
        rows.append(
            {
                "label": str(atom.get("label") or f"{element}{index + 1}"),
                "element": element,
                "x": float(atom.get("x", 0.0)),
                "y": float(atom.get("y", 0.0)),
                "z": float(atom.get("z", 0.0)),
                "occupancy": float(atom.get("occupancy", 1.0)),
                "b_iso": float(atom.get("b_iso", 1.0)),
            }
        )
    if not rows:
        raise ValueError("A fragment must contain at least one atom.")
    coords = np.asarray([[row["x"], row["y"], row["z"]] for row in rows], dtype=float)
    origin = coords.mean(axis=0) if center else np.zeros(3)
    for row in rows:
        row["x"] = float(row["x"] - origin[0])
        row["y"] = float(row["y"] - origin[1])
        row["z"] = float(row["z"] - origin[2])
    torsion_rows = [dict(item) for item in (torsions or [])]
    return {
        "name": str(name or "fragment"),
        "atoms": rows,
        "torsions": torsion_rows,
        "origin_cartesian_angstrom": origin.tolist(),
        "fragment_sha256": stable_json_hash({"atoms": rows, "torsions": torsion_rows}),
    }


def parse_xyz_fragment_text(text: str, *, name: str = "fragment") -> dict:
    """Parse a simple XYZ text block into a centered rigid fragment."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("XYZ fragment text must contain an atom count and atoms.")
    try:
        atom_count = int(lines[0])
    except ValueError as exc:
        raise ValueError("The first XYZ line must be an integer atom count.") from exc
    atom_lines = lines[2:2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError("XYZ fragment text ended before all atoms were read.")
    atoms = []
    for index, line in enumerate(atom_lines):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"XYZ atom line {index + 1} has fewer than 4 columns.")
        atoms.append(
            {
                "label": f"{parts[0].capitalize()}{index + 1}",
                "element": parts[0].capitalize(),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
            }
        )
    return build_rigid_fragment(name, atoms)


def _fragment_atom_index(fragment: dict) -> dict[str, int]:
    return {str(atom.get("label")): index for index, atom in enumerate(fragment.get("atoms", []))}


def _apply_fragment_torsions(fragment: dict, coords: np.ndarray, torsion_angles: Iterable[float]) -> np.ndarray:
    updated = np.asarray(coords, dtype=float).copy()
    label_index = _fragment_atom_index(fragment)
    for torsion, angle in zip(fragment.get("torsions", []) or [], torsion_angles):
        axis_labels = torsion.get("axis_labels") or torsion.get("axis") or []
        affected_labels = torsion.get("affected_labels") or torsion.get("affected") or []
        if len(axis_labels) != 2 or not affected_labels:
            continue
        if axis_labels[0] not in label_index or axis_labels[1] not in label_index:
            continue
        first_index = label_index[axis_labels[0]]
        second_index = label_index[axis_labels[1]]
        origin = updated[first_index].copy()
        axis = updated[second_index] - origin
        rotation = _axis_rotation_matrix(axis, float(angle))
        for label in affected_labels:
            index = label_index.get(str(label))
            if index is None or index in {first_index, second_index}:
                continue
            updated[index] = origin + rotation @ (updated[index] - origin)
    return updated


def transform_fragment_atoms(
    fragment: dict,
    *,
    cell: dict,
    translation_fractional: Iterable[float],
    euler_angles_rad: Iterable[float] = (0.0, 0.0, 0.0),
    torsion_angles_rad: Iterable[float] | None = None,
    label_prefix: str = "",
) -> list[dict]:
    """Place a fragment inside a cell and return fractional atom records."""
    basis = cell_cartesian_basis(cell)
    inv_basis = np.linalg.inv(basis)
    atoms = list(fragment.get("atoms", []))
    coords = np.asarray([[atom["x"], atom["y"], atom["z"]] for atom in atoms], dtype=float)
    coords = _apply_fragment_torsions(fragment, coords, [] if torsion_angles_rad is None else torsion_angles_rad)
    rotation = euler_rotation_matrix(*[float(v) for v in euler_angles_rad])
    rotated = coords @ rotation.T
    translation = np.asarray(list(translation_fractional), dtype=float)
    if translation.shape != (3,):
        raise ValueError("translation_fractional must contain three values.")
    placed = []
    for atom, cart in zip(atoms, rotated):
        frac = (translation + inv_basis @ cart) % 1.0
        label = f"{label_prefix}{atom.get('label', atom.get('element', 'X'))}"
        placed.append(
            {
                "label": label,
                "element": atom.get("element", "C"),
                "x": float(frac[0]),
                "y": float(frac[1]),
                "z": float(frac[2]),
                "occupancy": float(atom.get("occupancy", 1.0)),
                "b_iso": float(atom.get("b_iso", 1.0)),
                "fragment": fragment.get("name", "fragment"),
            }
        )
    return placed


def _fragment_variable_slices(fragments: list[dict]) -> list[dict]:
    slices = []
    cursor = 0
    for fragment in fragments:
        torsion_count = len(fragment.get("torsions", []) or [])
        slices.append(
            {
                "fragment": fragment,
                "translation": slice(cursor, cursor + 3),
                "rotation": slice(cursor + 3, cursor + 6),
                "torsion": slice(cursor + 6, cursor + 6 + torsion_count),
            }
        )
        cursor += 6 + torsion_count
    return slices


def _atoms_from_fragment_vector(cell: dict, fragments: list[dict], vector: np.ndarray) -> list[dict]:
    atoms: list[dict] = []
    for index, entry in enumerate(_fragment_variable_slices(fragments)):
        fragment = entry["fragment"]
        translation = np.asarray(vector[entry["translation"]], dtype=float) % 1.0
        rotation = np.asarray(vector[entry["rotation"]], dtype=float)
        torsion = np.asarray(vector[entry["torsion"]], dtype=float)
        atoms.extend(
            transform_fragment_atoms(
                fragment,
                cell=cell,
                translation_fractional=translation,
                euler_angles_rad=rotation,
                torsion_angles_rad=torsion,
                label_prefix=f"F{index + 1}_",
            )
        )
    return atoms


def _fragment_bounds(fragments: list[dict]) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for fragment in fragments:
        bounds.extend([(0.0, 1.0)] * 3)
        bounds.extend([(0.0, 2.0 * math.pi)] * 3)
        bounds.extend([(-math.pi, math.pi)] * len(fragment.get("torsions", []) or []))
    return bounds


def solve_fragment_direct_space_structure(
    *,
    candidate_cell: dict,
    extracted_reflections: Iterable[dict],
    fragments: Iterable[dict],
    formula_hint: str = "",
    proposed_space_group: str = "P 1",
    settings: FragmentDirectSpaceSettings | None = None,
    progress_callback=None,
) -> dict:
    """Run a fragment-based direct-space search.

    The solver optimizes each fragment as a rigid body with three fractional
    translations, three Euler rotations and optional torsion variables. It uses
    the same approximate intensity/collision/plausibility scoring as the 18.2
    prototype, but preserves molecular geometry within each fragment.
    """
    settings = settings or FragmentDirectSpaceSettings()
    cell = dict(candidate_cell)
    rows = _clean_reflections(
        extracted_reflections,
        DirectSpaceSettings(minimum_reflections=settings.minimum_reflections, intensity_floor=settings.intensity_floor),
    )
    if len(rows) < settings.minimum_reflections:
        raise ValueError(
            f"At least {settings.minimum_reflections} extracted hkl intensities are required."
        )
    fragment_list = [dict(fragment) for fragment in fragments]
    if not fragment_list:
        raise ValueError("At least one molecular or atomic fragment is required.")
    bounds = _fragment_bounds(fragment_list)
    if len(bounds) > settings.maximum_variables:
        raise ValueError(
            f"Fragment model has {len(bounds)} variables, exceeding the configured maximum of {settings.maximum_variables}."
        )
    scoring_settings = DirectSpaceSettings(
        random_seed=settings.random_seed,
        global_iterations=0,
        population_size=settings.population_size,
        local_polish_iterations=0,
        collision_weight=settings.collision_weight,
        density_weight=settings.density_weight,
        preferred_min_density_g_cm3=settings.preferred_min_density_g_cm3,
        preferred_max_density_g_cm3=settings.preferred_max_density_g_cm3,
        minimum_reflections=settings.minimum_reflections,
        intensity_floor=settings.intensity_floor,
    )
    rng = np.random.default_rng(settings.random_seed)

    torsion_slices = [entry["torsion"] for entry in _fragment_variable_slices(fragment_list)]

    def objective(vector: np.ndarray) -> float:
        atoms = _atoms_from_fragment_vector(cell, fragment_list, vector)
        score = score_direct_space_model(
            cell=cell,
            atoms=atoms,
            reflections=rows,
            settings=scoring_settings,
        )
        torsion_penalty = 0.0
        for torsion_slice in torsion_slices:
            values = np.asarray(vector[torsion_slice], dtype=float)
            torsion_penalty += float(np.mean((values / math.pi) ** 2)) if values.size else 0.0
        return float(score["objective"] + settings.torsion_weight * torsion_penalty)

    best_vector = np.asarray([rng.uniform(low, high) for low, high in bounds], dtype=float)
    best_value = objective(best_vector)
    notes = ["fragment random initial placement"]
    if progress_callback:
        progress_callback({"stage": "initial", "objective": best_value})

    if settings.global_iterations > 0:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=int(settings.global_iterations),
            popsize=max(2, int(settings.population_size)),
            seed=int(settings.random_seed),
            polish=False,
            tol=0.02,
            updating="immediate",
            workers=1,
        )
        if np.isfinite(result.fun) and float(result.fun) < best_value:
            best_vector = np.asarray(result.x, dtype=float)
            best_value = float(result.fun)
            notes.append("fragment differential evolution")
        if progress_callback:
            progress_callback({"stage": "global", "objective": best_value})

    if settings.local_polish_iterations > 0:
        local = minimize(
            objective,
            best_vector,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(settings.local_polish_iterations), "ftol": 1e-8},
        )
        if np.isfinite(local.fun) and float(local.fun) < best_value:
            best_vector = np.asarray(local.x, dtype=float)
            best_value = float(local.fun)
            notes.append("fragment bounded local polish")
        if progress_callback:
            progress_callback({"stage": "local", "objective": best_value})

    atoms = _atoms_from_fragment_vector(cell, fragment_list, best_vector)
    score = score_direct_space_model(
        cell=cell,
        atoms=atoms,
        reflections=rows,
        settings=scoring_settings,
    )
    cif_text = build_provisional_cif_text(
        cell=cell,
        atoms=atoms,
        data_name="afruz_fragment_direct_space_solution",
        proposed_space_group=proposed_space_group,
        formula_hint=formula_hint or _composition_from_atoms(atoms),
    )
    parsed = parse_cif_text(cif_text)
    plausibility = validate_crystal_structure(parsed)
    status = _classify_fragment_solution(score, plausibility, len(rows), fragment_list)
    variable_record = _fragment_solution_variables(fragment_list, best_vector)
    fragment_hashes = [fragment.get("fragment_sha256") or stable_json_hash(fragment) for fragment in fragment_list]
    return {
        "status": status,
        "candidate_type": "Fragment direct-space provisional structure",
        "settings": asdict(settings),
        "candidate_cell": cell,
        "proposed_space_group": proposed_space_group,
        "formula_hint": formula_hint,
        "fragment_count": len(fragment_list),
        "fragment_hashes": fragment_hashes,
        "atom_count": len(atoms),
        "reflection_count": len(rows),
        "objective": float(best_value),
        "intensity_chi2": score["intensity_chi2"],
        "intensity_correlation": score["correlation"],
        "scale": score["scale"],
        "short_contacts": score["short_contacts"],
        "plausibility": plausibility,
        "atoms": atoms,
        "fragment_solution_variables": variable_record,
        "cif_text": cif_text,
        "cif_sha256": hashlib.sha256(cif_text.encode("utf-8")).hexdigest(),
        "reflections_sha256": stable_json_hash(rows),
        "method_notes": notes,
        "scientific_boundary": (
            "This fragment direct-space result is a provisional structural hypothesis. "
            "Rigid-body agreement with extracted intensities must be followed by chemical plausibility review, "
            "Pawley/Le Bail consistency, Rietveld validation and competing-model comparison."
        ),
    }


def _fragment_solution_variables(fragments: list[dict], vector: np.ndarray) -> list[dict]:
    records = []
    for index, entry in enumerate(_fragment_variable_slices(fragments)):
        fragment = entry["fragment"]
        records.append(
            {
                "fragment_index": index,
                "fragment_name": fragment.get("name", f"fragment_{index + 1}"),
                "translation_fractional": np.asarray(vector[entry["translation"]], dtype=float).tolist(),
                "euler_angles_rad": np.asarray(vector[entry["rotation"]], dtype=float).tolist(),
                "torsion_angles_rad": np.asarray(vector[entry["torsion"]], dtype=float).tolist(),
            }
        )
    return records


def _classify_fragment_solution(score: dict, plausibility: dict, reflection_count: int, fragments: list[dict]) -> str:
    objective = float(score.get("objective", float("inf")))
    correlation = float(score.get("correlation", 0.0))
    overall = float((plausibility.get("scores") or {}).get("overall", 0.0))
    if reflection_count < 5:
        return "Exploratory fragment candidate — too few reflections"
    if not fragments:
        return "Invalid fragment candidate — no fragments"
    if objective < 10.0 and correlation > 0.75 and overall >= 55.0:
        return "Fragment provisional candidate — validate"
    if objective < 35.0 and overall >= 30.0:
        return "Weak fragment provisional candidate — review"
    return "Exploratory fragment candidate — low confidence"


def create_fragment_direct_space_solution_package(
    output_dir: str | Path,
    *,
    candidate_cell: dict,
    extracted_reflections: Iterable[dict],
    fragments: Iterable[dict],
    formula_hint: str = "",
    proposed_space_group: str = "P 1",
    source_dataset_uid: str = "",
    master_peak_checksum: str = "",
    settings: FragmentDirectSpaceSettings | None = None,
) -> dict:
    """Create a complete Phase 18.4 fragment-solution package."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fragment_list = [dict(fragment) for fragment in fragments]
    solution = solve_fragment_direct_space_structure(
        candidate_cell=candidate_cell,
        extracted_reflections=extracted_reflections,
        fragments=fragment_list,
        formula_hint=formula_hint,
        proposed_space_group=proposed_space_group,
        settings=settings,
    )
    cif_path = directory / "afruz_fragment_direct_space_provisional.cif"
    json_path = directory / "afruz_fragment_direct_space_solution.json"
    fragments_path = directory / "afruz_fragment_library.json"
    manifest_path = directory / "fragment_direct_space_manifest.json"
    readme_path = directory / "README_FRAGMENT_DIRECT_SPACE_PROTOTYPE.txt"
    cif_path.write_text(solution["cif_text"], encoding="utf-8")
    fragments_path.write_text(json.dumps(fragment_list, indent=2, ensure_ascii=False), encoding="utf-8")
    fragments_txt = write_table_txt(
        directory / "afruz_fragment_library.txt",
        fragment_list,
        title="Fragment direct-space library",
        backup=False,
    )
    record = dict(solution)
    record.pop("cif_text", None)
    record.update(
        {
            "source_dataset_uid": source_dataset_uid,
            "master_peak_checksum": master_peak_checksum,
            "cif_path": str(cif_path.resolve()),
            "fragments_path": str(fragments_path.resolve()),
        }
    )
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    solution_txt = write_mapping_txt(
        directory / "afruz_fragment_direct_space_solution.txt",
        record,
        title="Afruz fragment direct-space provisional solution",
        metadata={"source_dataset_uid": source_dataset_uid},
        backup=False,
    )
    readme_path.write_text(
        "Afruz fragment direct-space structure-solution prototype\n"
        "======================================================\n\n"
        "This package contains a provisional CIF created by placing one or more molecular/atomic fragments inside an indexed cell.\n"
        "The fragment geometry is preserved except for declared torsions. The result is not a final structure and must be validated chemically and by whole-pattern refinement.\n",
        encoding="utf-8",
    )
    manifest = {
        "package_type": "Afruz fragment direct-space provisional structure package",
        "source_dataset_uid": source_dataset_uid,
        "master_peak_checksum": master_peak_checksum,
        "formula_hint": formula_hint,
        "candidate_cell": dict(candidate_cell),
        "proposed_space_group": proposed_space_group,
        "status": solution["status"],
        "objective": solution["objective"],
        "fragment_count": solution["fragment_count"],
        "fragment_hashes": solution["fragment_hashes"],
        "cif_sha256": solution["cif_sha256"],
        "solution_json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
        "fragment_library_sha256": hashlib.sha256(fragments_path.read_bytes()).hexdigest(),
        "scientific_boundary": solution["scientific_boundary"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_txt = write_mapping_txt(
        directory / "fragment_direct_space_manifest.txt",
        manifest,
        title="Fragment direct-space solution manifest",
        backup=False,
    )
    return {
        "output_dir": str(directory.resolve()),
        "cif_path": str(cif_path.resolve()),
        "solution_json_path": str(json_path.resolve()),
        "solution_txt_path": solution_txt["txt_path"],
        "fragments_path": str(fragments_path.resolve()),
        "fragments_txt_path": fragments_txt["txt_path"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_txt_path": manifest_txt["txt_path"],
        "readme_path": str(readme_path.resolve()),
        "solution": solution,
        "manifest": manifest,
    }
