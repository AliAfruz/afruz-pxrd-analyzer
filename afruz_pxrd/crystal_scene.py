"""Deterministic crystal geometry, independent of the GUI and refinement solver.

The parser already expands explicit CIF symmetry operations. Do not expand them
again here or infer unprovided atoms from a space-group name.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import product
import colorsys
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .crystallography import ATOMIC_NUMBER, direct_metric_tensor
from .structure_validation import (
    COVALENT_RADII as VALIDATED_COVALENT_RADII,
    METALS as VALIDATED_METALS,
    VDW_RADII as VALIDATED_VDW_RADII,
    ensure_site_formula_metadata,
    structure_refinement_gate,
    validate_crystal_structure,
)


SCIENTIFIC_STYLE = "Scientific • polyhedra + H-bonds"
MOF_STYLE = "MOF • porous framework"
MOF_PORE_STYLE = "MOF • isolated pore / cage"
LEGACY_MOF_PORE_STYLE = "MOF • pore aperture [111]"
# Shared CPU/GPU display convention for the illustrative pore envelope.
PORE_ORANGE_HEX = "#ff9e04"
PORE_ORANGE_RGB = (1.0, 0.62, 0.015)
STYLES = (
    "Cinematic • midnight",
    "Publication • ivory",
    "Studio • slate",
    SCIENTIFIC_STYLE,
    MOF_STYLE,
    MOF_PORE_STYLE,
)
# Display colors and radii are artistic conventions, never measured atom sizes.
ELEMENT_COLORS = {
    "H": "#e7edf4", "C": "#52677e", "N": "#729dfa", "O": "#44cbbd",
    "F": "#8bd6a0", "Na": "#bb9aef", "Mg": "#77d6b1", "Al": "#aab7d2",
    "Si": "#e7b990", "P": "#efae6e", "S": "#edcf72", "Cl": "#79cfa8",
    "K": "#af9bef", "Ca": "#d0c5a5", "Ti": "#abb4e2", "V": "#a1b2c3",
    "Cr": "#83adc6", "Mn": "#c495b7", "Fe": "#e4a278", "Co": "#779bdf",
    "Ni": "#93bc8c", "Cu": "#e0a184", "Zn": "#afbad3", "Zr": "#92bec8",
    "Ce": "#e8be58", "Cs": "#c3a4eb", "Ba": "#aad696", "La": "#d6bd73",
    "W": "#98aabd", "Pt": "#ccd6e2", "Au": "#ebc472", "Pb": "#afa8cb",
}

# Conventional high-contrast crystallographic colors used by the scientific preset.
SCIENTIFIC_ELEMENT_COLORS = {
    # High-contrast CPK/Jmol-like colors: neutral framework carbon, red
    # oxygen, white hydrogen and lavender-blue chromium.  These conventions
    # keep MOF nodes and organic linkers distinct on a journal-white canvas.
    "H": "#ffffff", "C": "#4d5156", "N": "#3456c5", "O": "#e31d24",
    "F": "#65bd45", "Na": "#9254c8", "Mg": "#55a852", "Al": "#a9adb2",
    "Si": "#d2a286", "P": "#e88725", "S": "#e0c423", "Cl": "#43a63c",
    "K": "#8f4fc1", "Ca": "#7eaa69", "Ti": "#9ca4a9", "V": "#909aa0",
    "Cr": "#7889c4", "Mn": "#9b7b9b", "Fe": "#a96f0b", "Co": "#5977a5",
    "Ni": "#a5aaad", "Cu": "#b9784e", "Zn": "#8e9aa8", "Zr": "#73a4aa",
    "Ce": "#d2a423", "Cs": "#8d5bb7", "Ba": "#73a85b", "La": "#bba952",
    "W": "#73828e", "Pt": "#aab1b7", "Au": "#d2a51b", "Pb": "#7f7896",
}

COVALENT_RADII = {
    "H": .31, "C": .76, "N": .71, "O": .66, "F": .57, "Na": 1.66,
    "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "K": 2.03, "Ca": 1.76, "Ti": 1.60, "V": 1.53,
    "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24,
    "Cu": 1.32, "Zn": 1.22, "Zr": 1.75, "Ce": 2.04, "Cs": 2.44,
    "Ba": 2.15, "La": 2.07, "W": 1.62, "Pt": 1.36, "Au": 1.36,
    "Pb": 1.46,
}

# Approximate van der Waals radii used only for the optional pore-envelope
# visualization. They are not refinement parameters or adsorption radii.
PORE_VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10, "P": 1.80,
    "S": 1.80, "Cl": 1.75, "K": 2.75, "Ca": 2.31, "Ti": 2.15,
    "V": 2.05, "Cr": 2.05, "Mn": 2.05, "Fe": 2.00, "Co": 2.00,
    "Ni": 1.97, "Cu": 1.96, "Zn": 2.01, "Zr": 2.30, "Ce": 2.42,
}
COVALENT_RADII.update(VALIDATED_COVALENT_RADII)
PORE_VDW_RADII.update(VALIDATED_VDW_RADII)

METAL_ELEMENTS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs",
    "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
    "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
}
METAL_ELEMENTS.update(VALIDATED_METALS)


def element_color(element: str, scientific: bool = False) -> str:
    """Return a stable color for every valid element, with an obvious X fallback."""
    symbol = str(element or "X").capitalize()
    palette = SCIENTIFIC_ELEMENT_COLORS if scientific else ELEMENT_COLORS
    if symbol in palette:
        return palette[symbol]
    atomic_number = ATOMIC_NUMBER.get(symbol)
    if atomic_number is None:
        return "#d726d9"  # conspicuous unknown-site marker
    # Golden-angle hues prevent adjacent atomic numbers from collapsing into
    # nearly identical fallback colors.  This is a display convention only.
    hue = (atomic_number * 0.618033988749895) % 1.0
    saturation = 0.48 if scientific else 0.42
    value = 0.78 if scientific else 0.84
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return "#{:02x}{:02x}{:02x}".format(
        round(red * 255), round(green * 255), round(blue * 255)
    )


class CrystalSceneError(ValueError):
    pass


@dataclass(frozen=True)
class RenderSettings:
    style: str = STYLES[0]
    repeats: tuple[int, int, int] = (2, 2, 2)
    azimuth: float = 32.0
    elevation: float = 19.0
    zoom: float = 1.0
    atom_scale: float = 1.0
    atom_opacity: float = 1.0
    bond_radius: float = 0.065
    contact_cutoff: float = 0.0  # zero: per-site nearest-shell heuristic
    bonds: bool = True
    unlike_only: bool = True
    polyhedra: bool = True
    center_element: str = "Auto"
    polyhedron_opacity: float = 0.13
    polyhedron_edges: bool = False
    hydrogen_bonds: bool = False
    complete_boundaries: bool = True
    metal_metal_bonds: bool = False
    cell_edges: bool = True
    cell_grid: bool = True
    axes: bool = True
    labels: bool = False
    caption: bool = True
    transparent: bool = False
    hide_hydrogen: bool = False
    pore_volumes: bool = False
    pore_probe_radius: float = 0.0
    pore_opacity: float = 0.88
    pore_max_count: int = 2
    isolate_pore: bool = False
    smart_pore_isolation: bool = True
    isolated_pore_index: int = 0
    pore_shell_thickness: float = 8.0

    def validate(self):
        if self.style not in STYLES:
            raise CrystalSceneError("Choose a supported rendering style.")
        if len(self.repeats) != 3 or any(type(n) is not int or not 1 <= n <= 4 for n in self.repeats):
            raise CrystalSceneError("Repeat counts must be integers between 1 and 4.")
        limits = {"azimuth": (-3600, 3600), "elevation": (-89, 89), "zoom": (0.35, 2.5),
                  "atom_scale": (0.35, 2), "atom_opacity": (0.05, 1.0),
                  "bond_radius": (0.01, 0.2),
                  "contact_cutoff": (0, 12), "polyhedron_opacity": (0, 0.5),
                  "pore_probe_radius": (0, 3), "pore_opacity": (0.05, 1.0),
                  "pore_shell_thickness": (0.5, 12.0)}
        for key, (low, high) in limits.items():
            value = float(getattr(self, key))
            if not np.isfinite(value) or not low <= value <= high:
                raise CrystalSceneError(f"{key} must be between {low} and {high}.")
        if type(self.pore_max_count) is not int or not 1 <= self.pore_max_count <= 24:
            raise CrystalSceneError("pore_max_count must be an integer between 1 and 24.")
        if type(self.isolated_pore_index) is not int or not 0 <= self.isolated_pore_index < 24:
            raise CrystalSceneError("isolated_pore_index must be an integer between 0 and 23.")
        if self.isolate_pore and self.isolated_pore_index >= self.pore_max_count:
            raise CrystalSceneError(
                "The isolated pore number must not exceed the requested pore count."
            )


def cell_basis(cell: dict) -> np.ndarray:
    """Row vectors a,b,c in a right-handed Cartesian frame, in angstroms."""
    values = np.array([cell[k] for k in ("a", "b", "c", "alpha", "beta", "gamma")], float)
    if not np.isfinite(values).all() or np.any(values[:3] <= 0) or np.any(values[3:] <= 0) or np.any(values[3:] >= 180):
        raise CrystalSceneError("The unit cell needs positive lengths and angles between 0 and 180°.")
    try:
        return np.linalg.cholesky(direct_metric_tensor(cell))
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise CrystalSceneError("The unit cell is degenerate or physically invalid.") from exc


def structure_snapshot(structure: dict) -> dict:
    """Freeze the actual solver input, including the expanded atomic model."""
    keys = ("data_name", "formula", "formula_units_z", "formula_weight", "_afruz_formula_source",
            "space_group", "crystal_system", "source_path", "reported_wavelength_angstrom",
            "intensity_model", "xray_scattering_factors",
            "cell", "atoms", "cif_bonds", "symmetry_operations", "raw_cif_text")
    return {key: deepcopy(structure[key]) for key in keys if key in structure}


def model_from_structure(structure: dict, phase: dict | None = None) -> dict:
    model = structure_snapshot(structure)
    if phase is not None:
        if not phase.get("refined_cell"):
            raise CrystalSceneError("This phase has no saved refined unit cell.")
        model["cell"] = deepcopy(phase["refined_cell"])
        model["data_name"] = phase.get("phase_name") or model.get("data_name", "Phase")
        model["provenance"] = {
            "kind": "rietveld_result", "phase_index": phase.get("phase_index"),
            "cell_refined": bool(phase.get("cell_refined")),
            "atoms_fixed": bool(phase.get("atoms_fixed", True)),
            "delta_biso": phase.get("delta_biso", 0),
            "validation_status": phase.get("structure_plausibility_status") or phase.get("validation_status"),
            "structure_warnings": phase.get("structure_plausibility_warnings", []),
            "note": "Saved Rietveld cell; fractional coordinates and occupancies fixed to the input CIF. Display radii are illustrative; no thermal ellipsoids are implied.",
        }
    else:
        model["provenance"] = {"kind": "cif_reference", "note": "Imported CIF reference; not a refinement result. Display radii are illustrative."}
    cell_basis(model["cell"])
    if not model.get("atoms"):
        raise CrystalSceneError("This structure contains no atomic coordinates to render.")
    if ensure_site_formula_metadata(model):
        model["provenance"]["formula_source"] = model.get("_afruz_formula_source")
    validation = validate_crystal_structure(model)
    gate = structure_refinement_gate(validation)
    model["provenance"].update(
        validation_status=validation.get("status"),
        validation_score=validation.get("scores", {}).get("overall"),
        structure_warnings=list(validation.get("errors") or []) + list(validation.get("warnings") or []),
        structure_validation_counts={
            "duplicate_pairs": int(validation.get("duplicate_pair_count", len(validation.get("duplicate_pairs") or []))),
            "severe_contacts": int(validation.get("severe_contact_count", len(validation.get("severe_contact_pairs") or []))),
            "short_contacts": int(validation.get("short_contact_count", len(validation.get("short_contact_pairs") or []))),
            "vdw_overlaps": int(validation.get("vdw_overlap_count", len(validation.get("vdw_overlap_pairs") or []))),
            "hydrogen_bonds": int(validation.get("hydrogen_bond_count", 0)),
            "topology_excluded": int(validation.get("topology_excluded_pair_count", 0)),
        },
        refinement_allowed=bool(gate["passed"]),
        publication_ready=bool(gate["publication_ready"]),
    )
    return model


def models_from_result(result: dict | None) -> list[dict]:
    models = []
    for phase in (result or {}).get("phases", []):
        snapshot = phase.get("structure_snapshot")
        if not snapshot:
            raise CrystalSceneError(
                "This older refinement has no saved atomic snapshot. Run the refinement again to render its result, "
                "or open a CIF here as a clearly labelled reference preview."
            )
        model = model_from_structure(snapshot, phase)
        model["provenance"].update(
            refinement_success=result.get("success"),
            refinement_message=str(result.get("message", "")),
            rwp_percent=result.get("rwp_percent"),
            rp_percent=result.get("rp_percent"),
            rexp_percent=result.get("rexp_percent"),
            goodness_of_fit=result.get("goodness_of_fit"),
            reduced_chi_square=result.get("reduced_chi_square"),
            condition_number=result.get("condition_number"),
            maximum_absolute_correlation=result.get("maximum_absolute_correlation"),
            strong_correlation_pairs=deepcopy(result.get("strong_correlation_pairs")),
        )
        if result.get("success") is False:
            model["provenance"]["note"] += " Refinement did not converge; inspect fit diagnostics."
        models.append(model)
    return models


def display_radius(element: str, scientific: bool = False) -> float:
    # Deliberately artistic: radius is not used in contact inference.
    if scientific:
        return .18 if element == "H" else (.30 if element in {"C", "N", "O", "F"} else .40 + min(ATOMIC_NUMBER.get(element, 20), 80) * .001)
    return 0.26 if element == "H" else (0.35 if element in {"C", "N", "O", "F"} else 0.48 + min(ATOMIC_NUMBER.get(element, 20), 80) * 0.002)


@dataclass
class CrystalScene:
    model: dict
    settings: RenderSettings
    basis: np.ndarray
    positions: np.ndarray
    fractional: np.ndarray
    elements: list[str]
    labels: list[str]
    occupancies: list[float]
    primary: list[bool]
    bonds: list[tuple[int, int]]
    hydrogen_bonds: list[tuple[int, int]]
    triangles: list[np.ndarray]
    triangle_elements: list[str]
    pore_centers: list[np.ndarray]
    pore_radii: list[float]
    edges: list[np.ndarray]
    warnings: list[str]
    center_element: str


def _pore_envelopes_from_sites(
    source_sites: list[tuple[np.ndarray, str, str, float]],
    basis: np.ndarray,
    probe_radius: float,
    maximum_count: int,
) -> tuple[list[np.ndarray], list[float]]:
    """Return periodic maximal-empty-sphere estimates for a unit cell.

    This deliberately lightweight grid method supports a visual pore envelope;
    it is not a replacement for a probe-accessible surface/volume package.
    """
    if not source_sites:
        return [], []
    fractional = np.asarray([site[0] for site in source_sites], dtype=float)
    elements = [site[1] for site in source_sites]
    atomic_radii = np.asarray(
        [
            PORE_VDW_RADII.get(
                element,
                COVALENT_RADII.get(element, 1.15) + .65,
            )
            for element in elements
        ],
        dtype=float,
    )
    shifts = np.asarray(list(product((-1, 0, 1), repeat=3)), dtype=float)
    periodic_fractional = (fractional[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    periodic_positions = periodic_fractional @ basis
    periodic_radii = np.tile(atomic_radii, len(shifts))
    tree = cKDTree(periodic_positions)

    longest_axis = max(float(np.linalg.norm(vector)) for vector in basis)
    grid_size = int(np.clip(math.ceil(longest_axis / 3.5), 12, 26))
    axis = (np.arange(grid_size, dtype=float) + .5) / grid_size
    candidate_fractional = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    candidate_positions = candidate_fractional @ basis
    neighbor_count = min(32, len(periodic_positions))
    distances, indices = tree.query(candidate_positions, k=neighbor_count)
    if neighbor_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    clearances = np.min(distances - periodic_radii[indices], axis=1)
    clearances -= float(probe_radius)
    field = clearances.reshape((grid_size, grid_size, grid_size))
    maxima = np.ones_like(field, dtype=bool)
    for shift in product((-1, 0, 1), repeat=3):
        if shift == (0, 0, 0):
            continue
        maxima &= field >= np.roll(field, shift, axis=(0, 1, 2))
    candidates = np.flatnonzero(maxima.ravel() & (clearances >= 2.0))
    candidates = candidates[np.argsort(clearances[candidates])[::-1]]

    def clearance_at(points: np.ndarray) -> np.ndarray:
        """Periodic atom-surface clearance at one or more fractional points."""
        cartesian = np.mod(np.atleast_2d(points), 1.0) @ basis
        local_distances, local_indices = tree.query(cartesian, k=neighbor_count)
        if neighbor_count == 1:
            local_distances = local_distances[:, None]
            local_indices = local_indices[:, None]
        return (
            np.min(local_distances - periodic_radii[local_indices], axis=1)
            - float(probe_radius)
        )

    pool_fractional: list[np.ndarray] = []
    pool_radii: list[float] = []
    pool_limit = min(96, max(24, maximum_count * 8))
    for index in candidates:
        point = candidate_fractional[index].copy()
        radius = float(clearances[index])
        # Refine the coarse-grid maximum locally. This remains a deliberately
        # lightweight geometric estimate, but avoids reporting a visibly
        # undersized sphere merely because the true center lies between voxels.
        step = 0.5 / grid_size
        for _ in range(4):
            offsets = np.asarray(list(product((-step, 0.0, step), repeat=3)))
            trials = np.mod(point[None, :] + offsets, 1.0)
            trial_radii = clearance_at(trials)
            best = int(np.argmax(trial_radii))
            if float(trial_radii[best]) > radius:
                point = trials[best]
                radius = float(trial_radii[best])
            step *= 0.5
        separated = True
        for other, other_radius in zip(pool_fractional, pool_radii):
            delta = point - other
            delta -= np.round(delta)
            distance = float(np.linalg.norm(delta @ basis))
            if distance < max(3.0, .72 * min(radius, other_radius)):
                separated = False
                break
        if not separated:
            continue
        pool_fractional.append(point)
        pool_radii.append(radius)
        if len(pool_fractional) >= pool_limit:
            break

    # Prefer the largest representative from each distinct cavity-size family
    # before adding symmetry-equivalent copies. The previous center-biased
    # choice could understate the radius in primitive/non-orthogonal cells.
    groups: list[list[int]] = []
    for index, radius in sorted(
        enumerate(pool_radii), key=lambda item: item[1], reverse=True
    ):
        for group in groups:
            if abs(radius - pool_radii[group[0]]) <= .50:
                group.append(index)
                break
        else:
            groups.append([index])
    center_distance = [
        float(np.linalg.norm((point - .5) @ basis))
        for point in pool_fractional
    ]
    chosen: list[int] = []
    for group in groups:
        chosen.append(
            min(group, key=lambda index: (-pool_radii[index], center_distance[index]))
        )
        if len(chosen) >= maximum_count:
            break
    if len(chosen) < maximum_count:
        remaining = sorted(
            (index for index in range(len(pool_fractional)) if index not in chosen),
            key=lambda index: (center_distance[index], -pool_radii[index]),
        )
        chosen.extend(remaining[: maximum_count - len(chosen)])
    chosen = chosen[:maximum_count]
    return (
        [pool_fractional[index] @ basis for index in chosen],
        [pool_radii[index] for index in chosen],
    )


def _detect_pore_envelopes(
    source_sites: list[tuple[np.ndarray, str, str, float]],
    basis: np.ndarray,
    probe_radius: float,
    maximum_count: int,
) -> tuple[list[np.ndarray], list[float]]:
    """Patchable public seam used by scene construction and tests."""
    return _pore_envelopes_from_sites(
        source_sites, basis, probe_radius, maximum_count
    )


def _bond_components(
    atom_count: int,
    bonds: list[tuple[int, int]],
    allowed: set[int] | None = None,
) -> list[list[int]]:
    """Connected components of the displayed contact graph."""
    nodes = set(range(atom_count)) if allowed is None else set(allowed)
    adjacency = [[] for _ in range(atom_count)]
    for first, second in bonds:
        if first in nodes and second in nodes:
            adjacency[first].append(second)
            adjacency[second].append(first)
    components = []
    unseen = set(nodes)
    while unseen:
        start = unseen.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _complete_framework_pore_sites(
    source_sites: list[tuple[np.ndarray, str, str, float]],
    fractional: np.ndarray,
    elements: list[str],
    labels: list[str],
    occupancies: list[float],
    framework_indices: set[int],
) -> tuple[list[tuple[np.ndarray, str, str, float]], int]:
    """Recover a complete unit-cell framework from a local periodic crop.

    Single-pore mode first builds a bounded periodic neighbourhood so that
    bonds crossing a cell face can be classified.  The connected component is
    consequently only a *local* fragment.  Feeding those local coordinates to
    a periodic void search leaves most of a large conventional cell empty and
    can create a fictitious sphere almost half a cell wide.  CIF atom labels
    identify all symmetry-expanded copies of the accepted framework sites, so
    use them to project the classification back onto the complete source cell.

    The coordinate fallback retains support for hand-built/legacy models whose
    labels are absent or ambiguous.  Returned sites are always deduplicated in
    the crystallographic unit cell.
    """
    identities = {
        (str(elements[index]), str(labels[index]))
        for index in framework_indices
    }
    recovered = [
        site
        for site in source_sites
        if (str(site[1]), str(site[2])) in identities
    ]
    used_complete_source = bool(recovered)
    if not recovered:
        recovered = [
            (
                np.mod(np.asarray(fractional[index], dtype=float), 1.0),
                elements[index],
                labels[index],
                occupancies[index],
            )
            for index in sorted(framework_indices)
        ]

    complete: list[tuple[np.ndarray, str, str, float]] = []
    seen = set()
    for frac, element, label, occupancy in recovered:
        frac = np.mod(np.asarray(frac, dtype=float), 1.0)
        frac[np.isclose(frac, 1.0, atol=1e-7, rtol=0)] = 0.0
        key = (
            str(element),
            str(label),
            *np.round(frac, 7),
        )
        if key in seen:
            continue
        seen.add(key)
        complete.append((frac, str(element), str(label), float(occupancy)))
    return complete, len(identities) if used_complete_source else 0


def _metal_connected_framework_sites(
    source_sites: list[tuple[np.ndarray, str, str, float]],
    basis: np.ndarray,
) -> tuple[list[tuple[np.ndarray, str, str, float]], set[tuple[str, str]]]:
    """Classify a symmetry-expanded MOF framework before pore centering.

    A solvated conventional-cell CIF can place thousands of guest sites inside
    the real cages.  Centering the preliminary crop on all sites then selects a
    guest-dependent void rather than a framework cage.  Infer conservative
    covalent-radius contacts in the complete cell, retain components containing
    a metal node, and project their accepted CIF labels back across every
    symmetry-equivalent copy.  The result is a visualization classification;
    it does not modify the stored CIF or claim refined bond orders.
    """
    if len(source_sites) < 4:
        return [], set()
    positions = np.asarray(
        [np.mod(np.asarray(site[0], dtype=float), 1.0) @ basis for site in source_sites]
    )
    pairs = cKDTree(positions).query_pairs(4.5, output_type="ndarray")
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    contacts: list[tuple[int, int]] = []
    if len(pairs):
        distances = np.linalg.norm(
            positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1
        )
        for (first, second), distance in zip(pairs, distances):
            left = str(source_sites[int(first)][1])
            right = str(source_sites[int(second)][1])
            if left in METAL_ELEMENTS and right in METAL_ELEMENTS:
                continue
            cutoff = 1.24 * (
                COVALENT_RADII.get(left, 1.0)
                + COVALENT_RADII.get(right, 1.0)
            )
            if 0.35 <= float(distance) <= cutoff:
                contacts.append((int(first), int(second)))
    components = _bond_components(len(source_sites), contacts)
    framework_indices = {
        index
        for component in components
        if len(component) >= 4
        and any(
            str(source_sites[index][1]) in METAL_ELEMENTS
            for index in component
        )
        for index in component
    }
    identities = {
        (str(source_sites[index][1]), str(source_sites[index][2]))
        for index in framework_indices
    }
    if not identities:
        return [], set()
    return (
        [
            site
            for site in source_sites
            if (str(site[1]), str(site[2])) in identities
        ],
        identities,
    )


def _smart_mof_cage_indices(
    positions: np.ndarray,
    elements: list[str],
    bonds: list[tuple[int, int]],
    distances: np.ndarray,
    outer_radius: float,
    closure_limit: float,
) -> tuple[set[int], int] | None:
    """Select one MOF cage by metal nodes and node-to-node linkers.

    The selection is intentionally topological rather than a pure sphere crop:
    nearby metals are grouped into inorganic nodes, then only organic fragments
    connecting two or more selected nodes are retained. This removes linkers
    pointing into adjacent pores while preserving node oxo/halide ligands.
    """
    candidates = set(np.flatnonzero(distances <= closure_limit).tolist())
    metal_indices = [
        index for index in candidates if elements[index] in METAL_ELEMENTS
    ]
    if not metal_indices:
        return None

    parent = {index: index for index in metal_indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int):
        left, right = find(first), find(second)
        if left != right:
            parent[right] = left

    metal_positions = positions[metal_indices]
    metal_tree = cKDTree(metal_positions)
    for local_first, local_second in metal_tree.query_pairs(5.2):
        first = metal_indices[int(local_first)]
        second = metal_indices[int(local_second)]
        cutoff = float(np.clip(
            1.50
            * (
                COVALENT_RADII.get(elements[first], 1.45)
                + COVALENT_RADII.get(elements[second], 1.45)
            ),
            3.2,
            5.2,
        ))
        if float(np.linalg.norm(positions[first] - positions[second])) <= cutoff:
            union(first, second)

    group_members: dict[int, set[int]] = {}
    for index in metal_indices:
        group_members.setdefault(find(index), set()).add(index)
    selected_groups = {
        root
        for root, members in group_members.items()
        if any(distances[index] <= outer_radius for index in members)
    }
    if len(selected_groups) < 2:
        return None

    metal_group = {
        index: root for root, members in group_members.items() for index in members
    }
    adjacency = [[] for _ in elements]
    for first, second in bonds:
        if first in candidates and second in candidates:
            adjacency[first].append(second)
            adjacency[second].append(first)

    nonmetals = candidates - set(metal_indices)
    nonmetal_bonds = [
        (first, second)
        for first, second in bonds
        if first in nonmetals and second in nonmetals
    ]
    fragments = _bond_components(len(positions), nonmetal_bonds, nonmetals)
    keep = {
        index
        for root in selected_groups
        for index in group_members[root]
    }
    linker_count = 0
    for fragment in fragments:
        neighbouring_groups = {
            metal_group[neighbor]
            for index in fragment
            for neighbor in adjacency[index]
            if neighbor in metal_group
        }
        if not neighbouring_groups or not neighbouring_groups <= selected_groups:
            continue
        contains_carbon = any(elements[index] == "C" for index in fragment)
        if len(neighbouring_groups) >= 2:
            keep.update(fragment)
            if contains_carbon:
                linker_count += 1
        elif not contains_carbon:
            # Retain μ-O/terminal inorganic ligands belonging to a selected node,
            # but not carbon-containing linkers that leave for another pore.
            keep.update(fragment)

    # Keep the complete first coordination sphere even when an outward organic
    # linker is intentionally truncated at the cage boundary. This preserves
    # scientifically meaningful closed coordination polyhedra without drawing
    # the neighbouring pore's full linker network.
    ligand_elements = {"N", "O", "F", "S", "Cl"}
    for root in selected_groups:
        for metal in group_members[root]:
            keep.update(
                neighbor
                for neighbor in adjacency[metal]
                if elements[neighbor] in ligand_elements
            )

    if linker_count < max(1, len(selected_groups) - 1):
        return None
    return keep, linker_count


def _cif_display_bonds(
    model: dict,
    tree: cKDTree,
    positions: np.ndarray,
    labels: list[str],
    elements: list[str],
    primary: np.ndarray,
    complete_boundaries: bool,
) -> tuple[set[tuple[int, int]], int, int]:
    """Map CIF geometric bond rows onto displayed symmetry/periodic copies.

    The CIF's labels and reported distance are honored.  Raw symmetry codes are
    preserved in the model but not guessed here; equivalent displayed pairs are
    selected geometrically.  The result remains a visualization contact, not a
    refined bond-order assignment.
    """
    records = list(model.get("cif_bonds") or [])
    if not records or len(positions) < 2:
        return set(), 0, 0
    by_labels: dict[tuple[str, str], list[tuple[int, dict]]] = {}
    maximum_distance = 0.0
    symmetry_coded = 0
    for index, record in enumerate(records):
        first = str(record.get("label_1") or "")
        second = str(record.get("label_2") or "")
        if not first or not second:
            continue
        key = tuple(sorted((first, second)))
        by_labels.setdefault(key, []).append((index, record))
        try:
            reported = float(record.get("distance"))
        except (TypeError, ValueError):
            reported = float("nan")
        if np.isfinite(reported) and 0.35 <= reported <= 12.0:
            maximum_distance = max(maximum_distance, reported * 1.04 + 0.12)
        if any(
            str(record.get(field) or ".").strip() not in {"", ".", "?", "1_555"}
            for field in ("symmetry_1", "symmetry_2")
        ):
            symmetry_coded += 1
    search_distance = min(12.0, max(6.0, maximum_distance))
    matched_records: set[int] = set()
    matched_pairs: set[tuple[int, int]] = set()
    for first, second in tree.query_pairs(search_distance):
        if complete_boundaries and not (primary[first] or primary[second]):
            continue
        candidates = by_labels.get(tuple(sorted((labels[first], labels[second]))))
        if not candidates:
            continue
        distance = float(np.linalg.norm(positions[first] - positions[second]))
        if distance < 0.35:
            continue
        for record_index, record in candidates:
            try:
                reported = float(record.get("distance"))
            except (TypeError, ValueError):
                reported = float("nan")
            if np.isfinite(reported):
                tolerance = max(0.12, 0.04 * reported)
                accepted = abs(distance - reported) <= tolerance
            else:
                cutoff = 1.24 * (
                    COVALENT_RADII.get(elements[first], 1.15)
                    + COVALENT_RADII.get(elements[second], 1.15)
                )
                accepted = distance <= cutoff
            if accepted:
                matched_pairs.add((int(first), int(second)))
                matched_records.add(record_index)
                break
    return matched_pairs, len(matched_records), symmetry_coded


def build_scene(model: dict, settings: RenderSettings) -> CrystalScene:
    settings.validate()
    basis = cell_basis(model["cell"])
    fractional, elements, labels, occupancies, primary = [], [], [], [], []
    provenance = model.get("provenance", {})
    warnings = [str(warning) for warning in provenance.get("structure_warnings", [])]
    if provenance.get("refinement_success") is False:
        warnings.append("Refinement did not converge; inspect fit diagnostics before using this figure.")
    if provenance.get("validation_status"):
        warnings.append("Structure status: " + str(provenance["validation_status"]))
    seen = set()
    repeats = np.array(settings.repeats)
    # Expanded P1 representations of large frameworks can legitimately contain
    # well over ten thousand non-hydrogen sites. Keep the higher allowance
    # limited to a single-cell MOF view so accidental supercells still fail.
    mof_style = settings.style in {MOF_STYLE, MOF_PORE_STYLE}
    scientific_style = settings.style in {SCIENTIFIC_STYLE, MOF_STYLE, MOF_PORE_STYLE}
    # A bounded one-cell scientific view is also valid for large inorganic,
    # intermetallic and covalent structures.  The GUI enables GPU-first mode
    # for these models; repeated cells retain the stricter accidental-growth
    # guard.
    atom_limit = 18000 if scientific_style and int(np.prod(repeats)) == 1 else 1800
    source_sites = []
    pore_source_sites = []
    normalized_occupancy_count = 0
    preliminary_center: np.ndarray | None = None
    preliminary_radius: float | None = None
    for atom in model.get("atoms", []):
        try:
            frac = np.array([atom[k] for k in ("x", "y", "z")], float)
            occ = float(atom.get("occupancy", 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise CrystalSceneError("An atomic site has missing or invalid coordinates/occupancy.") from exc
        if 1.0 < occ <= 1.001:
            occ = 1.0
            normalized_occupancy_count += 1
        elif -0.001 <= occ < 0.0:
            occ = 0.0
            normalized_occupancy_count += 1
        if not np.isfinite(frac).all() or not np.isfinite(occ) or not 0 <= occ <= 1.0:
            raise CrystalSceneError("Atomic coordinates must be finite and occupancies between 0 and 1.")
        if occ == 0:
            continue
        frac %= 1.0
        frac[np.isclose(frac, 1, atol=1e-7, rtol=0)] = 0
        element = str(atom.get("element", "X"))
        label = str(atom.get("label", element))
        pore_source_sites.append((frac.copy(), element, label, occ))
        if settings.hide_hydrogen and element == "H":
            continue
        source_sites.append((frac, element, label, occ))
        for shift in product(*(range(int(n) + 1) for n in repeats)):
            point = frac + shift
            if np.any(point > repeats + 1e-7):
                continue
            key = (element, *np.round(point, 7))
            if key in seen:
                continue
            seen.add(key)
            fractional.append(point)
            elements.append(element)
            labels.append(label)
            occupancies.append(occ)
            primary.append(True)
            if len(fractional) > atom_limit:
                raise CrystalSceneError(
                    f"This view exceeds {atom_limit:,} atoms. Reduce repeat counts or hide hydrogen atoms."
                )
    preliminary_pore_sites = pore_source_sites
    source_identity_count = len({
        (str(site[1]), str(site[2])) for site in pore_source_sites
    })
    symmetry_expanded_model = bool(
        len(model.get("symmetry_operations") or []) > 1
        and len(pore_source_sites) > 4 * max(1, source_identity_count)
    )
    if (
        settings.isolate_pore
        and settings.smart_pore_isolation
        and symmetry_expanded_model
    ):
        framework_seed_sites, framework_seed_identities = (
            _metal_connected_framework_sites(pore_source_sites, basis)
        )
        if len(framework_seed_sites) >= 4:
            preliminary_pore_sites = framework_seed_sites
            warnings.append(
                f"Preliminary pore centering used {len(framework_seed_sites):,} "
                f"symmetry-expanded sites from {len(framework_seed_identities):,} "
                "metal-connected CIF site families, excluding disconnected "
                "guest/solvent families."
            )
    # A pore cage frequently crosses every face of a primitive cell. A radial
    # crop of only the displayed unit cell therefore produces broken linkers
    # and incomplete metal nodes (especially UiO-type primitive P1 models).
    # Build a bounded periodic neighbourhood around a preliminary void center;
    # the later topology crop selects the final cage from this complete graph.
    if settings.isolate_pore and preliminary_pore_sites:
        preliminary_centers, preliminary_radii = _pore_envelopes_from_sites(
            preliminary_pore_sites,
            basis,
            settings.pore_probe_radius,
            settings.pore_max_count,
        )
        if settings.isolated_pore_index < len(preliminary_centers):
            preliminary_center = np.asarray(
                preliminary_centers[settings.isolated_pore_index], dtype=float
            )
            preliminary_radius = float(
                preliminary_radii[settings.isolated_pore_index]
            )
            # 2.5 Å is the topology closure used below.  In a solvated or
            # disordered MOF the all-site preliminary sphere can be much
            # smaller than the later framework-only sphere, so scale an extra
            # safety margin with the cell instead of assuming a fixed 4 Å.
            # This ensures the second pass actually has the complete shell it
            # may need (MIL-101 is a representative large-cell case).
            longest_lattice_scale = max(
                float(np.linalg.norm(vector)) for vector in basis
            )
            pore_neighbourhood_margin = max(
                4.0,
                min(16.0, 0.15 * longest_lattice_scale),
            )
            neighbourhood_radius = (
                preliminary_radius
                + float(settings.pore_shell_thickness)
                + pore_neighbourhood_margin
            )
            smallest_lattice_scale = max(
                0.5, float(np.min(np.linalg.svd(basis, compute_uv=False)))
            )
            padding = max(
                1,
                int(math.ceil(neighbourhood_radius / smallest_lattice_scale)) + 1,
            )
            fractional = []
            elements = []
            labels = []
            occupancies = []
            primary = []
            seen = set()
            for frac, element, label, occ in source_sites:
                for shift in product(range(-padding, padding + 1), repeat=3):
                    point = frac + np.asarray(shift, dtype=float)
                    cartesian = point @ basis
                    if (
                        float(np.linalg.norm(cartesian - preliminary_center))
                        > neighbourhood_radius
                    ):
                        continue
                    key = (element, *np.round(point, 7))
                    if key in seen:
                        continue
                    seen.add(key)
                    fractional.append(point)
                    elements.append(element)
                    labels.append(label)
                    occupancies.append(occ)
                    # Every periodic copy belongs to the requested cage crop;
                    # primary/helper has no crystallographic meaning here.
                    primary.append(True)
                    if len(fractional) > atom_limit:
                        raise CrystalSceneError(
                            f"This periodic pore neighbourhood exceeds {atom_limit:,} atoms. "
                            "Hide hydrogen atoms or reduce the framework shell thickness."
                        )
            warnings.append(
                "Single-pore mode assembled a bounded periodic neighbourhood "
                "before topology filtering so cages crossing primitive-cell "
                "faces retain complete linkers and coordination nodes."
            )
    if not fractional:
        raise CrystalSceneError("There are no occupied atomic sites to render.")
    if normalized_occupancy_count:
        warnings.append(
            f"Normalized {normalized_occupancy_count} near-bound occupancy value(s) "
            "within 0.001 of 0 or 1 to absorb CIF rounding noise."
        )
    scientific = settings.style in {SCIENTIFIC_STYLE, MOF_STYLE, MOF_PORE_STYLE}
    # Complete coordination at cell faces with only the periodic images close
    # enough to interact with a displayed atom. These helpers remain outside the
    # cell box and are tagged separately from the primary display atoms.
    if scientific and settings.complete_boundaries and not settings.isolate_pore:
        primary_positions = np.asarray(fractional) @ basis
        primary_tree = cKDTree(primary_positions)
        for frac, element, label, occ in source_sites:
            for shift in product(*(range(-1, int(n) + 1) for n in repeats)):
                point = frac + shift
                if np.all(point >= -1e-7) and np.all(point <= repeats + 1e-7):
                    continue
                key = (element, *np.round(point, 7))
                if key in seen:
                    continue
                distance, _ = primary_tree.query(point @ basis, k=1)
                if float(distance) > 4.0:
                    continue
                seen.add(key)
                fractional.append(point)
                elements.append(element)
                labels.append(label)
                occupancies.append(occ)
                primary.append(False)
                if len(fractional) > atom_limit:
                    raise CrystalSceneError(
                        f"This periodic boundary view exceeds {atom_limit:,} atoms. Reduce repeats or disable boundary completion."
                    )
    fractional = np.asarray(fractional)
    positions = fractional @ basis
    primary_array = np.asarray(primary, dtype=bool)
    tree = cKDTree(positions)
    if any(x < 0.99999 for x in occupancies):
        warnings.append("Partial occupancies are preserved in the scene data; spheres show site positions, not occupancy volumes.")
    if tree.query_pairs(1e-5):
        warnings.append("Overlapping/disordered sites are present. Coincident spheres can hide one another.")
    bonds = []
    inferred_bond_count = 0
    cif_bond_count = 0
    cif_symmetry_coded_count = 0
    # Distance-only shell heuristic, explicitly exposed to the user. No guessed valence/radii.
    if len(positions) > 1 and (
        settings.bonds or settings.polyhedra or settings.isolate_pore
    ):
        nearest = np.full(len(positions), np.inf)
        pairs = np.asarray(sorted(tree.query_pairs(settings.contact_cutoff or 6.0)), dtype=int).reshape(-1, 2)
        if len(pairs):
            distances = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
            permitted = distances > 1e-5
            if settings.unlike_only:
                permitted &= np.array([elements[i] != elements[j] for i, j in pairs])
            if scientific and not settings.metal_metal_bonds:
                permitted &= np.array([
                    not (elements[i] in METAL_ELEMENTS and elements[j] in METAL_ELEMENTS)
                    for i, j in pairs
                ])
            if scientific and settings.complete_boundaries and not settings.isolate_pore:
                permitted &= np.array([primary_array[i] or primary_array[j] for i, j in pairs])
            pairs, distances = pairs[permitted], distances[permitted]
            if scientific and not settings.contact_cutoff:
                for (i, j), distance in zip(pairs, distances):
                    cutoff = 1.24 * (COVALENT_RADII.get(elements[i], 1.0) + COVALENT_RADII.get(elements[j], 1.0))
                    if .35 <= distance <= cutoff:
                        bonds.append((int(i), int(j)))
            else:
                for (i, j), distance in zip(pairs, distances):
                    nearest[i] = min(nearest[i], distance)
                    nearest[j] = min(nearest[j], distance)
                for (i, j), distance in zip(pairs, distances):
                    if settings.contact_cutoff or distance <= 1.16 * min(nearest[i], nearest[j]):
                        bonds.append((int(i), int(j)))
        inferred_bond_count = len(bonds)
        cif_bonds, cif_bond_count, cif_symmetry_coded_count = _cif_display_bonds(
            model,
            tree,
            positions,
            labels,
            elements,
            primary_array,
            scientific and settings.complete_boundaries and not settings.isolate_pore,
        )
        bonds = sorted(set(bonds) | cif_bonds)
        contact_limit = 30000 if scientific_style and int(np.prod(repeats)) == 1 else 12000
        if len(bonds) > contact_limit:
            raise CrystalSceneError(
                f"Too many contacts ({len(bonds):,}; limit {contact_limit:,}). "
                "Reduce the cutoff or repeat counts."
            )
        if scientific:
            rule = "covalent-radius limits (×1.24)"
            if settings.contact_cutoff:
                rule = f"the explicit {settings.contact_cutoff:.3g} Å cutoff"
            metal_note = " Metal–metal contacts are suppressed." if not settings.metal_metal_bonds else ""
            boundary_note = " Periodic helper atoms complete coordination across cell faces." if settings.complete_boundaries and not settings.isolate_pore else ""
            warnings.append(f"Inferred solid contacts use {rule}; they are display assignments, not refined bond orders.{metal_note}{boundary_note}")
        else:
            warnings.append("Contacts are a distance-based display heuristic, not refined chemical bonds. Auto uses mutual nearest shells (+16%) within 6 Å.")
        if model.get("cif_bonds"):
            warning = (
                f"Matched {cif_bond_count:,} of {len(model['cif_bonds']):,} CIF geometric "
                f"bond record(s); the scene contains {len(bonds) - inferred_bond_count:,} "
                "additional CIF-supported display contact(s) after deduplication. "
                "CIF records are geometric evidence, not refined bond orders."
            )
            if cif_symmetry_coded_count:
                warning += (
                    f" {cif_symmetry_coded_count:,} record(s) contain non-identity "
                    "symmetry codes; the renderer preserves those codes and maps labels "
                    "by reported distance without claiming full symmetry-code resolution."
                )
            warnings.append(warning)
    hydrogen_bonds = []
    if settings.hydrogen_bonds:
        donor_elements = {"N", "O", "F", "S"}
        acceptor_elements = {"N", "O", "F", "S"}
        for h_index, h_element in enumerate(elements):
            if h_element != "H" or not primary_array[h_index]:
                continue
            nearby = tree.query_ball_point(positions[h_index], 3.5)
            donors = [index for index in nearby if elements[index] in donor_elements and .35 < np.linalg.norm(positions[index] - positions[h_index]) <= 1.30]
            if not donors:
                continue
            donor = min(donors, key=lambda index: np.linalg.norm(positions[index] - positions[h_index]))
            donor_vector = positions[donor] - positions[h_index]
            candidates = []
            for acceptor in nearby:
                if acceptor == donor or elements[acceptor] not in acceptor_elements:
                    continue
                acceptor_vector = positions[acceptor] - positions[h_index]
                distance = float(np.linalg.norm(acceptor_vector))
                if not 1.50 <= distance <= 2.60:
                    continue
                cosine = float(np.dot(donor_vector, acceptor_vector) / (np.linalg.norm(donor_vector) * distance))
                angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
                donor_acceptor = float(np.linalg.norm(positions[donor] - positions[acceptor]))
                if angle >= 145.0 and 2.40 <= donor_acceptor <= 3.40:
                    candidates.append((distance, -angle, acceptor))
            if candidates:
                # One best acceptor per hydrogen avoids dense, mutually
                # incompatible networks in disordered/overlapping guest sites.
                hydrogen_bonds.append((h_index, min(candidates)[2]))
        hydrogen_bonds = sorted(set(hydrogen_bonds))
        if hydrogen_bonds:
            warnings.append("Dashed H···A contacts use H···A 1.50–2.60 Å, D···A 2.40–3.40 Å and D–H···A ≥145°; only the best acceptor per H is shown.")
        elif "H" in elements:
            warnings.append("No hydrogen-bond contacts met the current geometric display criteria.")

    center_element = settings.center_element
    if center_element == "Auto":
        center_element = max(set(elements), key=lambda symbol: (ATOMIC_NUMBER.get(symbol, 0), symbol))
    center_elements = ({element for element in elements if element in METAL_ELEMENTS}
                       if center_element == "All metals" else {center_element})
    triangles = []
    triangle_elements = []
    if settings.polyhedra:
        adjacent = [[] for _ in elements]
        for i, j in bonds:
            adjacent[i].append(j)
            adjacent[j].append(i)
        for index, neighbors in enumerate(adjacent):
            if elements[index] not in center_elements or not primary_array[index]:
                continue
            # Legacy styles omit boundary fragments. The scientific style can
            # instead use explicitly generated periodic neighbors to close them.
            if not (
                scientific and (settings.complete_boundaries or settings.isolate_pore)
            ) and not np.all(
                (fractional[index] > 1e-6) & (fractional[index] < repeats - 1e-6)
            ):
                continue
            if scientific and elements[index] in METAL_ELEMENTS:
                ligand_elements = {"N", "O", "F", "S", "Cl"}
                candidates = [neighbor for neighbor in tree.query_ball_point(positions[index], 4.0)
                              if elements[neighbor] in ligand_elements and neighbor != index]
                if candidates:
                    distances = np.array([np.linalg.norm(positions[neighbor] - positions[index]) for neighbor in candidates])
                    shell = min(float(np.min(distances)) * 1.28, 4.0)
                    neighbors = [neighbor for neighbor, distance in zip(candidates, distances) if distance <= shell]
            if 4 <= len(neighbors) <= 24:
                points = positions[neighbors]
                try:
                    hull = ConvexHull(points)
                except QhullError:
                    continue
                # A hull must actually enclose its central site.
                if np.any(hull.equations[:, :3] @ positions[index] + hull.equations[:, 3] > 1e-6):
                    continue
                faces = [points[face] for face in hull.simplices]
                triangles.extend(faces)
                triangle_elements.extend([elements[index]] * len(faces))
        if not triangles:
            warnings.append("No enclosed coordination polyhedra were found. Adjust the center element/contact cutoff or inspect the CIF connectivity.")
    if scientific and settings.complete_boundaries and not settings.isolate_pore:
        used = set(np.flatnonzero(primary_array).tolist())
        used.update(index for pair in bonds for index in pair)
        used.update(index for pair in hydrogen_bonds for index in pair)
        # Convex-hull vertices normally coincide with bonded ligands, but retain
        # them explicitly if a custom contact cutoff excludes such a bond.
        for triangle in triangles:
            for point in triangle:
                distance, index = tree.query(point, k=1)
                if float(distance) <= 1e-6:
                    used.add(int(index))
        if len(used) < len(positions):
            keep = sorted(used)
            remap = {old: new for new, old in enumerate(keep)}
            positions = positions[keep]
            fractional = fractional[keep]
            elements = [elements[index] for index in keep]
            labels = [labels[index] for index in keep]
            occupancies = [occupancies[index] for index in keep]
            primary = [primary[index] for index in keep]
            primary_array = np.asarray(primary, dtype=bool)
            bonds = [(remap[i], remap[j]) for i, j in bonds]
            hydrogen_bonds = [(remap[i], remap[j]) for i, j in hydrogen_bonds]
    pore_centers: list[np.ndarray] = []
    pore_radii: list[float] = []
    framework_site_identities: set[tuple[str, str]] = set()
    if settings.pore_volumes or settings.isolate_pore:
        pore_detection_sites = pore_source_sites
        if settings.isolate_pore and settings.smart_pore_isolation:
            components = _bond_components(len(positions), bonds)
            metal_components = [
                component
                for component in components
                if len(component) >= 4
                and any(elements[index] in METAL_ELEMENTS for index in component)
            ]
            if metal_components:
                framework_components = metal_components
                framework_kind = "metal-connected"
            else:
                largest = max((len(component) for component in components), default=0)
                minimum = max(4, int(math.ceil(largest * .25)))
                framework_components = [
                    component for component in components if len(component) >= minimum
                ]
                framework_kind = "largest bonded"
            framework_indices = {
                index for component in framework_components for index in component
            }
            framework_site_identities = {
                (str(elements[index]), str(labels[index]))
                for index in framework_indices
            }
            framework_sites, recovered_identity_count = (
                _complete_framework_pore_sites(
                    pore_source_sites,
                    fractional,
                    elements,
                    labels,
                    occupancies,
                    framework_indices,
                )
            )
            if len(framework_sites) >= 4:
                pore_detection_sites = framework_sites
                recovery_note = (
                    f" Recovered the complete unit-cell population for "
                    f"{recovered_identity_count:,} accepted CIF site label(s) "
                    "before periodic void analysis."
                    if recovered_identity_count
                    else ""
                )
                warnings.append(
                    f"Smart pore detection used {len(framework_sites):,} unique "
                    f"sites from {len(framework_components):,} {framework_kind} "
                    "framework fragment(s), excluding disconnected guest, solvent "
                    "and disorder sites from the void estimate."
                    + recovery_note
                )
            else:
                warnings.append(
                    "Smart framework-only pore detection could not identify a "
                    "bonded framework, so all occupied CIF sites were used."
                )
        # Isolation needs more internal candidates than the number ultimately
        # displayed.  Large symmetric MOFs contain several equivalent copies
        # of each cage family; keeping only two size representatives can leave
        # no equivalent close to the periodic neighbourhood assembled above.
        detection_count = (
            max(settings.pore_max_count, 24)
            if settings.isolate_pore
            else settings.pore_max_count
        )
        unit_centers, unit_radii = _detect_pore_envelopes(
            pore_detection_sites,
            basis,
            settings.pore_probe_radius,
            detection_count,
        )
        if (
            settings.isolate_pore
            and preliminary_center is not None
            and preliminary_radius is not None
            and unit_centers
        ):
            inverse_basis = np.linalg.inv(basis)
            equivalents = []
            for center, radius in zip(unit_centers, unit_radii):
                delta_fractional = (
                    np.asarray(center, dtype=float) - preliminary_center
                ) @ inverse_basis
                equivalent = np.asarray(center, dtype=float) - (
                    np.round(delta_fractional) @ basis
                )
                equivalents.append((equivalent, float(radius)))
            requested = settings.isolated_pore_index
            if requested < len(unit_centers):
                # Preserve the requested final cage-size family (index zero is
                # the largest), then choose its symmetry-equivalent occurrence
                # closest to the preliminary neighborhood.  Matching by the
                # smaller guest-filled preliminary radius selected the wrong
                # MIL-101 cage family and produced a one-sided crop.
                target_radius = float(unit_radii[requested])
                same_family = [
                    index
                    for index, (_center, radius) in enumerate(equivalents)
                    if abs(radius - target_radius) <= 0.50
                ]
                matched = min(
                    same_family or list(range(len(equivalents))),
                    key=lambda index: float(
                        np.linalg.norm(
                            equivalents[index][0] - preliminary_center
                        )
                    ),
                )
                unit_centers[requested] = equivalents[matched][0]
                unit_radii[requested] = equivalents[matched][1]
            unit_centers = unit_centers[: settings.pore_max_count]
            unit_radii = unit_radii[: settings.pore_max_count]
        for shift in product(*(range(int(n)) for n in repeats)):
            translation = np.asarray(shift, dtype=float) @ basis
            for center, radius in zip(unit_centers, unit_radii):
                pore_centers.append(center + translation)
                pore_radii.append(radius)
        if pore_centers:
            range_text = f"{min(pore_radii):.2f}–{max(pore_radii):.2f} Å"
            warnings.append(
                f"Yellow pore envelopes show {len(pore_centers)} periodic "
                f"grid-derived maximal empty sphere(s), radii {range_text}, "
                f"after a {settings.pore_probe_radius:.2f} Å probe subtraction. "
                "They are illustrative geometry, not an adsorption-accessible "
                "surface or pore-size distribution."
            )
        else:
            warnings.append(
                "No pore envelope above the 2 Å display threshold was found "
                "with the current probe radius."
            )

    if settings.isolate_pore:
        if not pore_centers:
            raise CrystalSceneError(
                "A single pore cannot be isolated because no pore envelope was found. "
                "Try a smaller probe subtraction or inspect the CIF geometry."
            )
        pore_index = settings.isolated_pore_index
        if pore_index >= len(pore_centers):
            raise CrystalSceneError(
                f"Pore {pore_index + 1} was requested, but only "
                f"{len(pore_centers)} pore envelope(s) were found."
            )
        selected_center = np.asarray(pore_centers[pore_index], dtype=float)
        selected_radius = float(pore_radii[pore_index])
        outer_radius = selected_radius + float(settings.pore_shell_thickness)
        distances = np.linalg.norm(positions - selected_center, axis=1)
        initial = set(np.flatnonzero(distances <= outer_radius).tolist())
        closure_limit = outer_radius + 2.5
        topology_fragment_count = 0
        topology_description = "metal-connected framework fragment(s)"
        discarded_radial_count = 0
        keep = set()
        if settings.smart_pore_isolation and bonds:
            symmetry_complete_shell = bool(
                symmetry_expanded_model
                and framework_site_identities
            )
            candidates = set(
                np.flatnonzero(distances <= closure_limit).tolist()
            )
            if symmetry_complete_shell:
                # Conventional high-symmetry MOF cells repeat each asymmetric
                # CIF label many times.  A graph-only cage extraction can keep
                # one chemically connected arc while dropping the other
                # symmetry-equivalent walls.  Keep the complete framework-
                # labelled radial shell instead: this yields the closed,
                # polyhedral MIL-101 cage while still excluding guest/solvent
                # labels identified above.
                keep = {
                    index
                    for index in candidates
                    if (str(elements[index]), str(labels[index]))
                    in framework_site_identities
                }
                topology_fragment_count = len(framework_site_identities)
                topology_description = "symmetry-expanded framework site family/families"
                discarded_radial_count = len(initial - keep)
            else:
                cage_selection = _smart_mof_cage_indices(
                    positions,
                    elements,
                    bonds,
                    distances,
                    outer_radius,
                    closure_limit,
                )
                if cage_selection is not None:
                    keep, topology_fragment_count = cage_selection
                    topology_description = "node-to-node organic linker fragment(s)"
                    discarded_radial_count = len(initial - keep)
                else:
                    # A large/disordered CIF can split one real framework into
                    # several display-contact components.  The complete-cell
                    # identities recovered above are stronger evidence than
                    # requiring every linker to survive that heuristic graph.
                    identity_framework = {
                        index
                        for index in candidates
                        if (str(elements[index]), str(labels[index]))
                        in framework_site_identities
                    }
                    if identity_framework:
                        keep.update(identity_framework)
                        topology_fragment_count = len(framework_site_identities)
                        topology_description = "complete-cell framework site family/families"
                        discarded_radial_count = len(initial - keep)
                    else:
                        candidate_components = _bond_components(
                            len(positions), bonds, candidates
                        )
                        framework_has_metal = any(
                            len(component) >= 4
                            and any(elements[index] in METAL_ELEMENTS for index in component)
                            for component in candidate_components
                        )
                        accepted = []
                        for component in candidate_components:
                            if not any(index in initial for index in component):
                                continue
                            if framework_has_metal:
                                if not any(
                                    elements[index] in METAL_ELEMENTS for index in component
                                ):
                                    continue
                            elif len(component) < 4:
                                continue
                            accepted.append(component)
                            keep.update(component)
                        topology_fragment_count = len(accepted)
                        discarded_radial_count = len(initial - keep)
        if not keep:
            # Conservative fallback for molecular/metal-free structures whose
            # CIF connectivity cannot be inferred from the display contacts.
            keep = set(initial)
            for first, second in bonds:
                if first in initial and distances[second] <= closure_limit:
                    keep.add(second)
                if second in initial and distances[first] <= closure_limit:
                    keep.add(first)
        if not keep:
            raise CrystalSceneError(
                "The selected pore shell contains no displayed atoms. Increase "
                "the framework shell thickness."
            )
        old_count = len(positions)
        keep = sorted(keep)
        keep_set = set(keep)
        remap = {old: new for new, old in enumerate(keep)}
        geometry_tree = cKDTree(positions)
        isolated_triangles = []
        isolated_triangle_elements = []
        for triangle, element in zip(triangles, triangle_elements):
            vertex_distances, vertex_indices = geometry_tree.query(triangle, k=1)
            if np.all(vertex_distances <= 1e-5) and all(
                int(index) in keep_set for index in np.asarray(vertex_indices).ravel()
            ):
                isolated_triangles.append(np.asarray(triangle, dtype=float))
                isolated_triangle_elements.append(element)
        positions = positions[keep]
        fractional = fractional[keep]
        elements = [elements[index] for index in keep]
        labels = [labels[index] for index in keep]
        occupancies = [occupancies[index] for index in keep]
        primary = [primary[index] for index in keep]
        bonds = [
            (remap[first], remap[second])
            for first, second in bonds
            if first in keep_set and second in keep_set
        ]
        hydrogen_bonds = [
            (remap[first], remap[second])
            for first, second in hydrogen_bonds
            if first in keep_set and second in keep_set
        ]
        triangles = isolated_triangles
        triangle_elements = isolated_triangle_elements
        # Center the cropped cage in both renderers without changing the CIF
        # model stored in the reusable scene document.
        display_center = np.asarray(repeats, dtype=float) @ basis / 2.0
        translation = display_center - selected_center
        positions = positions + translation
        fractional = positions @ np.linalg.inv(basis)
        triangles = [triangle + translation for triangle in triangles]
        pore_centers = [selected_center + translation]
        pore_radii = [selected_radius]
        warnings.append(
            f"Isolated pore {pore_index + 1}: retained {len(positions):,} of "
            f"{old_count:,} displayed atoms around a {selected_radius:.2f} Å "
            f"maximal-empty-sphere estimate with a "
            f"{settings.pore_shell_thickness:.2f} Å framework shell. "
            + (
                f"Topology filtering retained {topology_fragment_count:,} "
                f"{topology_description} and rejected "
                f"{discarded_radial_count:,} disconnected radial site(s). "
                if settings.smart_pore_isolation and topology_fragment_count
                else ""
            )
            + "This is a "
            "visualization crop, not an extracted molecular species; inspect "
            "the crop boundary before structural interpretation."
        )

    edges = []
    if settings.cell_edges and not settings.isolate_pore:
        for axis in range(3):
            other = [k for k in range(3) if k != axis]
            values_u = range(settings.repeats[other[0]] + 1) if settings.cell_grid else (0, settings.repeats[other[0]])
            values_v = range(settings.repeats[other[1]] + 1) if settings.cell_grid else (0, settings.repeats[other[1]])
            for u, v in product(values_u, values_v):
                p = np.zeros(3)
                p[other] = [u, v]
                q = p.copy()
                q[axis] = repeats[axis]
                edges.append(np.array([p @ basis, q @ basis]))
    return CrystalScene(
        deepcopy(model), settings, basis, positions, fractional, elements,
        labels, occupancies, primary, bonds, hydrogen_bonds, triangles,
        triangle_elements, pore_centers, pore_radii, edges, warnings,
        center_element,
    )


def _json_compliant_scene_value(value, path: str, replacements: list[dict]):
    """Convert scientific metadata to strict JSON without hiding non-finite diagnostics."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        representation = (
            "NaN" if math.isnan(value)
            else "Infinity" if value > 0
            else "-Infinity"
        )
        replacements.append({
            "path": path or "$",
            "original_non_finite_value": representation,
            "json_representation": "string",
        })
        return representation
    if isinstance(value, Mapping):
        return {
            str(key): _json_compliant_scene_value(
                item,
                f"{path}.{key}" if path else str(key),
                replacements,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_compliant_scene_value(item, f"{path}[{index}]", replacements)
            for index, item in enumerate(value)
        ]
    if isinstance(value, set):
        return [
            _json_compliant_scene_value(item, f"{path}[{index}]", replacements)
            for index, item in enumerate(sorted(value, key=str))
        ]
    if isinstance(value, Path):
        return str(value)
    return value


def scene_document(scene: CrystalScene, width: int, height: int, dpi: int) -> dict:
    raw = {"schema": "afruz.crystal-scene.v1", "model": scene.model,
           "settings": asdict(scene.settings), "export": {"width": width, "height": height, "dpi": dpi},
           "geometry": {"displayed_atoms": len(scene.positions),
                        "primary_atoms": int(sum(scene.primary)),
                        "periodic_helper_atoms": int(len(scene.primary) - sum(scene.primary)),
                        "displayed_contacts": len(scene.bonds),
                        "displayed_hydrogen_bonds": len(scene.hydrogen_bonds),
                        "displayed_polyhedron_faces": len(scene.triangles),
                        "displayed_pore_envelopes": len(scene.pore_centers),
                        "pore_envelope_radii_angstrom": scene.pore_radii},
           "warnings": scene.warnings,
           "conventions": "Orthographic projection; Cartesian distances in Å. Expanded CIF sites with periodic boundary copies. Illustrative atom sizes, contact/polyhedron display heuristics, and optional grid-derived maximal-empty-sphere pore envelopes."}
    replacements: list[dict] = []
    document = _json_compliant_scene_value(raw, "", replacements)
    if replacements:
        document["serialization_notes"] = [
            "Non-finite scientific diagnostics are stored as explicit JSON strings; "
            "the rendered structure and source refinement values are unchanged.",
            *replacements,
        ]
    return document


def load_scene_document(path: str | Path) -> tuple[dict, RenderSettings]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "afruz.crystal-scene.v1":
        raise CrystalSceneError("This is not an Afruz Crystal Studio scene file.")
    settings = dict(value["settings"])
    if settings.get("style") == LEGACY_MOF_PORE_STYLE:
        settings["style"] = MOF_PORE_STYLE
    settings["repeats"] = tuple(settings.get("repeats", (1, 1, 1)))
    options = RenderSettings(**settings)
    options.validate()
    loaded_model = deepcopy(value["model"])
    previous_provenance = deepcopy(loaded_model.get("provenance") or {})
    model = model_from_structure(loaded_model)
    model["provenance"].update(previous_provenance)
    # Recompute safety fields even for scenes saved by older versions.
    validation = validate_crystal_structure(model)
    gate = structure_refinement_gate(validation)
    model["provenance"].update(
        validation_status=validation.get("status"),
        validation_score=validation.get("scores", {}).get("overall"),
        structure_warnings=list(validation.get("errors") or []) + list(validation.get("warnings") or []),
        structure_validation_counts={
            "duplicate_pairs": int(validation.get("duplicate_pair_count", len(validation.get("duplicate_pairs") or []))),
            "severe_contacts": int(validation.get("severe_contact_count", len(validation.get("severe_contact_pairs") or []))),
            "short_contacts": int(validation.get("short_contact_count", len(validation.get("short_contact_pairs") or []))),
            "vdw_overlaps": int(validation.get("vdw_overlap_count", len(validation.get("vdw_overlap_pairs") or []))),
            "hydrogen_bonds": int(validation.get("hydrogen_bond_count", 0)),
            "topology_excluded": int(validation.get("topology_excluded_pair_count", 0)),
        },
        refinement_allowed=bool(gate["passed"]),
        publication_ready=bool(gate["publication_ready"]),
    )
    build_scene(model, options)
    return model, options
