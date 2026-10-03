"""Deterministic, inspectable view planning for Crystal Studio.

This module is advisory. It never changes coordinates, occupancies, the unit
cell, refinement results, or chemical labels. Its inferred contacts are used
only to classify periodic topology and recommend visualization controls.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import product
import math

import numpy as np
from scipy.spatial import cKDTree

from .crystallography import direct_metric_tensor
from .structure_validation import COVALENT_RADII, METALS


SMART_SCHEMA = "afruz.smart-crystal-view.v1"
DONOR_ELEMENTS = {"N", "O", "F", "P", "S", "Cl", "Br", "I"}
NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"}
MOF_STYLE = "MOF • porous framework"
SCIENTIFIC_STYLE = "Scientific • polyhedra + H-bonds"
PUBLICATION_STYLE = "Publication • ivory"
STUDIO_STYLE = "Studio • slate"


def _basis(cell: dict) -> np.ndarray:
    return np.linalg.cholesky(direct_metric_tensor(cell))


def _occupied_sites(model: dict) -> tuple[list[dict], np.ndarray, list[str]]:
    sites = []
    fractional = []
    elements = []
    for atom in model.get("atoms") or []:
        try:
            occupancy = float(atom.get("occupancy", 1.0))
            point = np.asarray([atom[key] for key in ("x", "y", "z")], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if occupancy <= 0 or not np.isfinite(point).all():
            continue
        sites.append(atom)
        fractional.append(np.mod(point, 1.0))
        elements.append(str(atom.get("element") or "X").capitalize())
    return sites, np.asarray(fractional, dtype=float), elements


def _canonical_edge(first: int, second: int, shift: np.ndarray):
    shift_tuple = tuple(int(value) for value in shift)
    if first < second or (first == second and shift_tuple > (0, 0, 0)):
        return first, second, shift_tuple
    return second, first, tuple(-value for value in shift_tuple)


def _periodic_contact_graph(
    fractional: np.ndarray,
    elements: list[str],
    basis: np.ndarray,
    maximum_sites: int = 6000,
) -> tuple[list[tuple[int, int, tuple[int, int, int], float]], str]:
    """Element-aware periodic display contacts for topology classification."""
    count = len(fractional)
    if count < 1:
        return [], "No occupied sites were available."
    if count > maximum_sites:
        return [], (
            f"Topology graph skipped because {count:,} occupied sites exceed the "
            f"{maximum_sites:,}-site advisory limit."
        )
    shifts = np.asarray(list(product((-1, 0, 1), repeat=3)), dtype=int)
    cartesian = fractional @ basis
    periodic_positions = (
        fractional[None, :, :] + shifts[:, None, :]
    ).reshape(-1, 3) @ basis
    periodic_indices = np.tile(np.arange(count, dtype=int), len(shifts))
    periodic_shifts = np.repeat(shifts, count, axis=0)
    tree = cKDTree(periodic_positions)
    maximum_radius = max(
        (COVALENT_RADII.get(element, 1.15) for element in elements),
        default=1.15,
    )
    search_radius = min(6.0, 1.24 * (maximum_radius + maximum_radius))
    edges: dict[tuple, float] = {}
    for first, position in enumerate(cartesian):
        for periodic_index in tree.query_ball_point(position, search_radius):
            second = int(periodic_indices[periodic_index])
            shift = periodic_shifts[periodic_index]
            if first == second and not np.any(shift):
                continue
            first_element, second_element = elements[first], elements[second]
            if first_element in NOBLE_GASES or second_element in NOBLE_GASES:
                continue
            delta = periodic_positions[periodic_index] - position
            distance = float(np.linalg.norm(delta))
            cutoff = 1.24 * (
                COVALENT_RADII.get(first_element, 1.15)
                + COVALENT_RADII.get(second_element, 1.15)
            )
            if not 0.35 <= distance <= cutoff:
                continue
            key = _canonical_edge(first, second, shift)
            if key not in edges or distance < edges[key]:
                edges[key] = distance
    return [(*key, distance) for key, distance in sorted(edges.items())], (
        "Element-aware periodic distance contacts (covalent radii ×1.24); "
        "display/topology inference, not assigned bond order."
    )


def _component_periodicity(
    atom_count: int,
    edges: list[tuple[int, int, tuple[int, int, int], float]],
) -> tuple[int, list[list[int]], list[list[int]]]:
    adjacency: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(atom_count)]
    for first, second, shift, _distance in edges:
        vector = np.asarray(shift, dtype=int)
        adjacency[first].append((second, vector))
        adjacency[second].append((first, -vector))
    unseen = set(range(atom_count))
    component_sizes = []
    all_cycles: list[np.ndarray] = []
    component_cycles: list[list[int]] = []
    while unseen:
        root = unseen.pop()
        offsets = {root: np.zeros(3, dtype=int)}
        queue = deque([root])
        members = [root]
        cycles = []
        while queue:
            current = queue.popleft()
            for neighbor, shift in adjacency[current]:
                expected = offsets[current] + shift
                if neighbor not in offsets:
                    offsets[neighbor] = expected
                    unseen.discard(neighbor)
                    members.append(neighbor)
                    queue.append(neighbor)
                else:
                    cycle = expected - offsets[neighbor]
                    if np.any(cycle):
                        cycles.append(cycle)
                        all_cycles.append(cycle)
        component_sizes.append(members)
        if cycles:
            unique = np.unique(np.asarray(cycles, dtype=int), axis=0)
            component_cycles.extend(unique.tolist())
    dimension = (
        int(np.linalg.matrix_rank(np.asarray(all_cycles, dtype=float)))
        if all_cycles else 0
    )
    return dimension, component_sizes, component_cycles


def _camera_from_topology(
    basis: np.ndarray,
    dimension: int,
    cycles: list[list[int]],
) -> tuple[float, float, str]:
    direction = np.sum(
        basis / np.linalg.norm(basis, axis=1, keepdims=True), axis=0
    )
    reason = "isometric view from the three crystallographic basis directions"
    cycle_vectors = [np.asarray(cycle, dtype=float) @ basis for cycle in cycles]
    independent = []
    for vector in cycle_vectors:
        if np.linalg.norm(vector) <= 1e-8:
            continue
        trial = np.asarray([*independent, vector], dtype=float)
        if np.linalg.matrix_rank(trial) > len(independent):
            independent.append(vector)
        if len(independent) >= max(1, dimension):
            break
    if dimension == 2 and len(independent) >= 2:
        direction = np.cross(independent[0], independent[1])
        reason = "view normal to the detected two-dimensional periodic plane"
    elif dimension == 1 and independent:
        chain = independent[0] / np.linalg.norm(independent[0])
        axes = np.eye(3)
        reference = axes[int(np.argmin(np.abs(axes @ chain)))]
        direction = np.cross(chain, reference) + .35 * chain
        reason = "oblique view across the detected one-dimensional periodic chain"
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        direction = np.asarray((1.0, 1.0, 1.0))
        norm = math.sqrt(3.0)
    direction /= norm
    if direction[2] < 0:
        direction *= -1
    azimuth = float(math.degrees(math.atan2(direction[1], direction[0])))
    elevation = float(math.degrees(math.atan2(
        direction[2], math.hypot(direction[0], direction[1])
    )))
    return azimuth, float(np.clip(elevation, -80.0, 80.0)), reason


def _classification(
    counts: Counter,
    dimension: int | None,
    atom_count: int,
    volume_per_non_h: float | None,
) -> tuple[str, float, list[str], bool]:
    elements = set(counts)
    metals = elements & METALS
    has_carbon = counts.get("C", 0) > 0
    donors = elements & DONOR_ELEMENTS
    nonmetals = elements - METALS - {"H", "X"}
    evidence = []
    porous_candidate = False
    if metals and has_carbon and donors:
        if dimension is None or dimension >= 1 or atom_count >= 80:
            family = "Metal–organic framework / coordination polymer"
            confidence = .92 if dimension and dimension >= 2 else .82
            porous_candidate = bool(
                (dimension is not None and dimension >= 2)
                and (volume_per_non_h or 0) >= 16.0
            ) or atom_count > 1000
            evidence.append("metal sites coexist with carbon-containing donor linkers")
        else:
            family = "Molecular coordination compound"
            confidence = .78
            evidence.append("metal–ligand composition is present without periodic network cycles")
    elif not metals and has_carbon and dimension is not None and dimension >= 2:
        family = "Covalent organic framework / extended covalent solid"
        confidence = .82
        porous_candidate = (volume_per_non_h or 0) >= 16.0
        evidence.append("carbon-rich nonmetal network has two- or three-dimensional periodicity")
    elif metals and "O" in elements:
        if dimension == 2:
            family = "Layered inorganic / hydroxide / oxide framework"
        elif dimension == 1:
            family = "One-dimensional inorganic coordination chain"
        else:
            family = "Extended inorganic / oxide coordination solid"
        confidence = .84 if dimension is not None else .70
        evidence.append("metal and oxygen sites dominate the inferred coordination network")
    elif elements and elements <= METALS:
        family = "Elemental or intermetallic solid"
        confidence = .88
        evidence.append("all occupied elements are metallic")
    elif dimension == 0:
        family = "Molecular crystal"
        confidence = .86
        evidence.append("no non-zero periodic connectivity cycles were detected")
    elif dimension == 2:
        family = "Layered extended solid"
        confidence = .78
        evidence.append("the inferred network spans two independent lattice directions")
    elif dimension == 1:
        family = "One-dimensional chain solid"
        confidence = .76
        evidence.append("the inferred network spans one lattice direction")
    elif dimension == 3:
        family = "Extended ionic/covalent crystal"
        confidence = .72
        evidence.append("the inferred network spans three independent lattice directions")
    elif has_carbon and nonmetals:
        family = "Molecular or low-dimensional organic crystal"
        confidence = .62
        evidence.append("organic composition is present but topology was inconclusive")
    else:
        family = "General crystalline solid"
        confidence = .50
        evidence.append("composition/topology does not match a specialized safe preset")
    return family, confidence, evidence, porous_candidate


def _recommended_view(
    family: str,
    dimension: int | None,
    cycles: list[list[int]],
    counts: Counter,
    atom_count: int,
    basis: np.ndarray,
    porous_candidate: bool,
) -> dict:
    has_metal = any(element in METALS for element in counts)
    has_donor = any(element in DONOR_ELEMENTS for element in counts)
    is_mof = family.startswith("Metal–organic")
    is_organic_network = family.startswith("Covalent organic")
    is_molecular = "Molecular" in family
    is_metallic = family.startswith("Elemental or intermetallic")
    if is_mof:
        style = MOF_STYLE
    elif is_molecular:
        style = PUBLICATION_STYLE
    elif is_metallic:
        style = STUDIO_STYLE
    else:
        style = SCIENTIFIC_STYLE
    repeats = [1, 1, 1]
    if dimension in {1, 2} and cycles:
        active = [False, False, False]
        for cycle in cycles:
            for axis, value in enumerate(cycle):
                active[axis] |= bool(value)
        repeats = [2 if value else 1 for value in active]
    azimuth, elevation, camera_reason = _camera_from_topology(
        basis, dimension or 0, cycles
    )
    hide_hydrogen = bool((is_mof or is_organic_network) and atom_count > 180)
    polyhedra = bool(has_metal and has_donor and not is_metallic)
    return {
        "style": style,
        "repeats": repeats,
        "azimuth": azimuth,
        "elevation": elevation,
        "zoom": .88 if atom_count > 1500 else .96,
        "atom_scale": .52 if atom_count > 1500 else (.62 if is_mof else .82),
        "bond_radius": .024 if atom_count > 700 else .035,
        "bonds": not is_metallic,
        "unlike_only": not (is_mof or is_organic_network or is_molecular),
        "polyhedra": polyhedra,
        "center_element": "All metals" if has_metal else "Auto",
        "polyhedron_opacity": .30 if is_mof else .38,
        "polyhedron_edges": polyhedra,
        "hydrogen_bonds": bool(is_molecular and counts.get("H") and has_donor),
        "complete_boundaries": atom_count <= 5000,
        "metal_metal_bonds": is_metallic,
        "cell_edges": True,
        "cell_grid": False,
        "axes": True,
        "labels": False,
        "caption": True,
        "hide_hydrogen": hide_hydrogen,
        "pore_volumes": False,
        "isolate_pore": False,
        "smart_pore_isolation": True,
        "gpu_first": atom_count > 1200,
        "porosity_candidate": porous_candidate,
        "camera_reason": camera_reason,
        "lod_reason": (
            "GPU-first mode and reduced atom size are recommended for the large model."
            if atom_count > 1200
            else "Full sphere detail is suitable for this model size."
        ),
    }


def analyze_crystal_model(model: dict, maximum_topology_sites: int = 6000) -> dict:
    sites, fractional, elements = _occupied_sites(model)
    if not sites:
        raise ValueError("Smart Crystal Studio needs at least one occupied finite atom site.")
    basis = _basis(model["cell"])
    counts = Counter(elements)
    edges, contact_note = _periodic_contact_graph(
        fractional, elements, basis, maximum_sites=maximum_topology_sites
    )
    dimension = None
    components: list[list[int]] = []
    cycles: list[list[int]] = []
    if len(sites) <= maximum_topology_sites:
        dimension, components, cycles = _component_periodicity(len(sites), edges)
    non_hydrogen = max(1, sum(value for key, value in counts.items() if key != "H"))
    volume = abs(float(np.linalg.det(basis)))
    volume_per_non_h = volume / non_hydrogen
    family, confidence, evidence, porous_candidate = _classification(
        counts, dimension, len(sites), volume_per_non_h
    )
    explicit_bonds = list(model.get("cif_bonds") or [])
    partial = sum(
        1 for site in sites if float(site.get("occupancy", 1.0)) < .99999
    )
    unknown = sum(element == "X" for element in elements)
    if explicit_bonds:
        evidence.append(f"CIF supplies {len(explicit_bonds):,} explicit geometric bond record(s)")
        confidence = min(.98, confidence + .03)
    if dimension is not None:
        evidence.append(
            f"periodic contact graph dimensionality is {dimension}D"
        )
    recommendation = _recommended_view(
        family, dimension, cycles, counts, len(sites), basis, porous_candidate
    )
    limitations = [
        "Automatic contacts are visualization/topology assignments, not refined bond orders.",
        "Oxidation states, magnetic ordering and protonation are not inferred by the view planner.",
    ]
    if partial:
        limitations.append(
            f"{partial:,} partially occupied site(s) are preserved; mutually exclusive disorder alternatives are not merged."
        )
    if unknown:
        limitations.append(
            f"{unknown:,} site(s) have unknown element symbol X and use conspicuous fallback styling."
        )
    if len(sites) > maximum_topology_sites:
        limitations.append(contact_note)
    if porous_candidate:
        limitations.append(
            "Porosity is only a candidate classification; use a probe-accessible surface/volume program for adsorption claims."
        )
    return {
        "schema": SMART_SCHEMA,
        "structure_name": str(model.get("data_name") or "Crystal structure"),
        "classification": {
            "family": family,
            "confidence": float(confidence),
            "periodic_dimension": dimension,
            "porosity_candidate": bool(porous_candidate),
        },
        "composition": {
            "occupied_site_count": len(sites),
            "element_counts": dict(sorted(counts.items())),
            "partial_occupancy_site_count": partial,
            "unknown_element_site_count": unknown,
            "cell_volume_angstrom3": volume,
            "volume_per_non_hydrogen_site_angstrom3": volume_per_non_h,
        },
        "connectivity": {
            "source": (
                "CIF geometric bond records plus advisory periodic distance topology"
                if explicit_bonds else "advisory periodic distance topology"
            ),
            "cif_bond_record_count": len(explicit_bonds),
            "inferred_periodic_contact_count": len(edges),
            "component_count": len(components) if dimension is not None else None,
            "translation_cycle_vectors": cycles[:24],
            "method_note": contact_note,
        },
        "evidence": evidence,
        "recommendation": recommendation,
        "limitations": limitations,
    }


def format_smart_crystal_report(report: dict) -> str:
    classification = report["classification"]
    composition = report["composition"]
    connectivity = report["connectivity"]
    recommendation = report["recommendation"]
    dimension = classification["periodic_dimension"]
    dimension_text = "not calculated (large-model limit)" if dimension is None else f"{dimension}D"
    lines = [
        "SMART CRYSTAL STUDIO — EXPLAINABLE VIEW PLAN",
        f"Structure: {report['structure_name']}",
        f"Classification: {classification['family']}",
        f"Confidence: {classification['confidence'] * 100:.0f}%",
        f"Periodic topology: {dimension_text}",
        f"Occupied sites: {composition['occupied_site_count']:,}",
        f"Connectivity source: {connectivity['source']}",
        "",
        "Evidence:",
        *[f"  • {item}" for item in report["evidence"]],
        "",
        "Recommended view:",
        f"  • Template: {recommendation['style']}",
        f"  • Repeats a/b/c: {' × '.join(map(str, recommendation['repeats']))}",
        f"  • Camera: azimuth {recommendation['azimuth']:.1f}°, elevation {recommendation['elevation']:.1f}°",
        f"  • Camera reason: {recommendation['camera_reason']}",
        f"  • GPU-first: {'yes' if recommendation['gpu_first'] else 'no'} — {recommendation['lod_reason']}",
        f"  • Coordination polyhedra: {'on' if recommendation['polyhedra'] else 'off'}",
        f"  • Hydrogen display: {'hidden' if recommendation['hide_hydrogen'] else 'shown'}",
        f"  • Pore candidate: {'yes; manual pore analysis available' if recommendation['porosity_candidate'] else 'not asserted'}",
        "",
        "Scientific limits:",
        *[f"  • {item}" for item in report["limitations"]],
        "",
        "Manual controls remain authoritative. Applying this plan changes only the view, never the CIF or refinement result.",
    ]
    return "\n".join(lines)
