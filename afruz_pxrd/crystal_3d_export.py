"""Editable glTF 2.0 export of frozen Crystal Studio geometry (no framebuffer)."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np

from .crystal_gpu import _gpu_theme, _rgba, _unit_icosphere
from .crystal_scene import (
    MOF_PORE_STYLE, MOF_STYLE, PORE_ORANGE_RGB, SCIENTIFIC_STYLE,
    display_radius, element_color, scene_document,
)
from .version import APP_RELEASE


@dataclass(frozen=True)
class ModelExportOptions:
    sphere_subdivisions: int = 3
    atoms: bool = True
    bonds: bool = True
    polyhedra: bool = True
    pores: bool = True
    hydrogen_bonds: bool = True
    cell_edges: bool = True
    background_style: str = "Journal white"
    allow_flagged_structure: bool = False

    def validate(self):
        if type(self.sphere_subdivisions) is not int or not 1 <= self.sphere_subdivisions <= 3:
            raise ValueError("Choose sphere quality 1, 2 or 3.")
        if self.background_style not in {"Journal white", "Dark studio"}:
            raise ValueError("Choose a supported background style.")


def _finite(values, shape=None, name="Geometry"):
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all() or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} must have the expected shape and contain only finite numbers.")
    return result


class _Glb:
    def __init__(self):
        self.binary = bytearray()
        self.doc = {"asset": {"version": "2.0", "generator": APP_RELEASE},
                    "scene": 0, "scenes": [], "nodes": [], "meshes": [],
                    "materials": [], "bufferViews": [], "accessors": []}
        self.materials = {}

    def accessor(self, values, indices=False):
        values = np.ascontiguousarray(values, dtype="<u4" if indices else "<f4")
        if not np.isfinite(values).all() or not len(values):
            raise ValueError("Mesh buffers must be nonempty and finite.")
        while len(self.binary) % 4:
            self.binary.append(0)
        view = {"buffer": 0, "byteOffset": len(self.binary), "byteLength": values.nbytes,
                "target": 34963 if indices else 34962}
        self.binary.extend(values.tobytes())
        view_index = len(self.doc["bufferViews"])
        self.doc["bufferViews"].append(view)
        accessor = {"bufferView": view_index, "componentType": 5125 if indices else 5126,
                    "count": len(values), "type": "SCALAR" if indices else "VEC3",
                    "min": [int(values.min())] if indices else values.min(axis=0).tolist(),
                    "max": [int(values.max())] if indices else values.max(axis=0).tolist()}
        index = len(self.doc["accessors"])
        self.doc["accessors"].append(accessor)
        return index

    def geometry(self, vertices, normals, faces):
        vertices = _finite(vertices, name="Mesh vertices")
        normals = _finite(normals, vertices.shape, "Mesh normals")
        faces = np.asarray(faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Expected 3D vertices and triangular faces.")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("Mesh face indices are out of range.")
        return {"attributes": {"POSITION": self.accessor(vertices), "NORMAL": self.accessor(normals)},
                "indices": self.accessor(faces.reshape(-1), True), "mode": 4}

    def material(self, name, rgba, roughness=.32, double_sided=False):
        rgba = _finite(rgba, (4,), "Material RGBA")
        if np.any(rgba < 0) or np.any(rgba > 1):
            raise ValueError("Material components must be between zero and one.")
        key = (name, *rgba, roughness, double_sided)
        if key not in self.materials:
            # glTF baseColorFactor is linear; Studio color codes are sRGB.
            rgb = [float(c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4) for c in rgba[:3]]
            self.materials[key] = len(self.doc["materials"])
            self.doc["materials"].append({"name": name, "pbrMetallicRoughness": {
                "baseColorFactor": [*rgb, float(rgba[3])], "metallicFactor": 0., "roughnessFactor": roughness},
                "alphaMode": "BLEND" if rgba[3] < .999 else "OPAQUE", "doubleSided": double_sided,
                "extras": {"studio_srgb_rgba": rgba.tolist()}})
        return self.materials[key]

    def mesh(self, name, geometry, material):
        index = len(self.doc["meshes"])
        self.doc["meshes"].append({"name": name, "primitives": [{**geometry, "material": material}]})
        return index

    def node(self, name, **values):
        index = len(self.doc["nodes"])
        self.doc["nodes"].append({"name": name, **values})
        return index

    def bytes(self):
        self.doc["buffers"] = [{"byteLength": len(self.binary)}]
        encoded = json.dumps(self.doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        encoded += b" " * (-len(encoded) % 4)
        binary = bytes(self.binary) + b"\0" * (-len(self.binary) % 4)
        length = 12 + 8 + len(encoded) + 8 + len(binary)
        if length >= 2**32:
            raise ValueError("The GLB exceeds the 4 GiB format limit. Reduce the displayed supercell.")
        return (struct.pack("<4sII", b"glTF", 2, length) + struct.pack("<II", len(encoded), 0x4E4F534A)
                + encoded + struct.pack("<II", len(binary), 0x004E4942) + binary)


def _cylinder(sides=24):
    angles = np.arange(sides) * (2 * np.pi / sides)
    ring = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(sides)))
    bottom, top = ring.copy(), ring.copy()
    bottom[:, 2], top[:, 2] = -.5, .5
    vertices = np.concatenate((bottom, top, bottom, top, [[0, 0, -.5], [0, 0, .5]]))
    normals = np.concatenate((ring, ring, np.tile([0, 0, -1], (sides, 1)), np.tile([0, 0, 1], (sides, 1)), [[0, 0, -1], [0, 0, 1]]))
    faces = []
    for first in range(sides):
        second = (first + 1) % sides
        faces.extend(((first, second, first + sides), (second, second + sides, first + sides),
                      (4 * sides, second + 2 * sides, first + 2 * sides),
                      (4 * sides + 1, first + 3 * sides, second + 3 * sides)))
    return vertices, normals, np.asarray(faces)


def _segment(glb, name, first, second, radius, mesh, extras):
    first, second = _finite(first, (3,), "Segment endpoint"), _finite(second, (3,), "Segment endpoint")
    vector = second - first
    length = float(np.linalg.norm(vector))
    if not np.isfinite(radius) or radius <= 0 or length < 1e-12:
        raise ValueError("A bond or edge has zero length or an invalid display radius.")
    axis = vector / length
    quaternion = np.array([-axis[1], axis[0], 0., 1 + axis[2]])
    quaternion = np.array([1., 0., 0., 0.]) if axis[2] < -1 + 1e-12 else quaternion / np.linalg.norm(quaternion)
    return glb.node(name, mesh=mesh, translation=((first + second) / 2).tolist(),
                    rotation=quaternion.tolist(), scale=[float(radius), float(radius), length], extras=extras)


def build_editable_glb(scene, options=None, *, camera=None, progress=None):
    """Return binary GLB and provenance without changing any scene coordinates."""
    options = options or ModelExportOptions()
    options.validate()
    scene.settings.validate()
    if scene.model.get("provenance", {}).get("publication_ready") is False and not options.allow_flagged_structure:
        raise ValueError("This structure requires review. Explicitly enable the expert override to export it.")
    positions = _finite(scene.positions, name="Atom coordinates")
    count = len(positions)
    if positions.shape != (count, 3) or not count:
        raise ValueError("The scene has no valid atom coordinates.")
    _finite(scene.fractional, (count, 3), "Displayed fractional coordinates")
    _finite(scene.basis, (3, 3), "Cell basis")
    for values in (scene.elements, scene.labels, scene.occupancies, scene.primary):
        if len(values) != count:
            raise ValueError("Atom identities and coordinates are inconsistent.")
    occupancies = _finite(scene.occupancies, (count,), "Occupancies")
    if np.any(occupancies < 0) or np.any(occupancies > 1):
        raise ValueError("Occupancies must be between zero and one.")
    if len(scene.pore_centers) != len(scene.pore_radii) or len(scene.triangles) != len(scene.triangle_elements):
        raise ValueError("Pore or polyhedral geometry is inconsistent.")
    total = count + len(scene.bonds) + len(scene.hydrogen_bonds) + len(scene.triangles) + len(scene.edges) + len(scene.pore_centers)
    if total > 250_000:
        raise ValueError("This editable scene is too large. Reduce repeats or export a single pore/cage.")
    def report(value, message):
        if progress is not None:
            progress(int(value), message)
    report(1, "Preparing shared meshes and materials…")
    glb = _Glb()
    groups = {}
    def add(role, node):
        groups.setdefault(role, []).append(node)
    scientific = scene.settings.style in {SCIENTIFIC_STYLE, MOF_STYLE, MOF_PORE_STYLE}
    theme = _gpu_theme(options.background_style)
    sphere, faces = _unit_icosphere(options.sphere_subdivisions)
    sphere_geometry = glb.geometry(sphere, sphere / np.linalg.norm(sphere, axis=1)[:, None], faces)
    cylinder_geometry = glb.geometry(*_cylinder())
    sphere_meshes = {}
    if options.atoms:
        for index, (point, element) in enumerate(zip(positions, scene.elements)):
            if element not in sphere_meshes:
                mat = glb.material(f"Atom {element}", _rgba(element_color(element, scientific), scene.settings.atom_opacity))
                sphere_meshes[element] = glb.mesh(f"Sphere {element}", sphere_geometry, mat)
            radius = max(.08, 2 * display_radius(element, scientific) * scene.settings.atom_scale)
            add("Atoms", glb.node(f"Atom {index + 1:06d} {scene.labels[index]} ({element})", mesh=sphere_meshes[element],
                translation=point.tolist(), scale=[radius] * 3, extras={"afruz_role": "atom", "scene_index": index,
                    "element": str(element), "source_label": str(scene.labels[index]),
                    "cartesian_angstrom": point.tolist(), "fractional_display": np.asarray(scene.fractional[index]).tolist(),
                    "occupancy": float(occupancies[index]), "periodic_helper": not bool(scene.primary[index]),
                    "display_radius_angstrom": radius}))
            if index % 100 == 0:
                report(5 + 35 * index / max(1, count), "Preparing individually selectable atoms…")
    if options.bonds and scene.bonds:
        mesh = glb.mesh("Bond cylinder", cylinder_geometry, glb.material("Display contacts", theme["bond"]))
        for index, pair in enumerate(scene.bonds):
            if len(pair) != 2 or any(type(value) not in (int, np.int64, np.int32) or not 0 <= value < count for value in pair):
                raise ValueError("A contact references an invalid atom index.")
            first, second = pair
            add("Bonds", _segment(glb, f"Bond {index + 1:06d} {scene.labels[first]}—{scene.labels[second]}", positions[first], positions[second],
                scene.settings.bond_radius, mesh, {"afruz_role": "bond", "atom_indices": [int(first), int(second)],
                    "distance_angstrom": float(np.linalg.norm(positions[second] - positions[first])),
                    "convention": "display contact, not a measured bond order"}))
            if index % 200 == 0:
                report(40 + 20 * index / max(1, len(scene.bonds)), "Preparing editable bond cylinders…")
    if options.hydrogen_bonds and scene.hydrogen_bonds:
        mesh = glb.mesh("H-bond dash", cylinder_geometry, glb.material("Hydrogen-bond contacts", theme["hydrogen_bond"]))
        for index, pair in enumerate(scene.hydrogen_bonds):
            if len(pair) != 2 or any(not 0 <= value < count for value in pair):
                raise ValueError("An H-bond references an invalid atom index.")
            first, second = positions[pair[0]], positions[pair[1]]
            dashes = max(3, min(16, int(np.ceil(np.linalg.norm(second - first) / .25))))
            dash_nodes = []
            for dash in range(dashes):
                start, end = (dash + .15) / dashes, (dash + .65) / dashes
                dash_nodes.append(_segment(glb, f"H-bond {index + 1:06d} dash {dash + 1}", first + start * (second - first), first + end * (second - first),
                    .025, mesh, {"afruz_role": "hydrogen_bond_dash", "atom_indices": [int(value) for value in pair]}))
            add("Hydrogen bonds", glb.node(f"H-bond {index + 1:06d}", children=dash_nodes, extras={"afruz_role": "hydrogen_bond"}))
    if options.polyhedra and scene.triangles:
        report(65, "Preserving the exact polyhedral surfaces…")
        all_triangles = _finite(scene.triangles, (len(scene.triangles), 3, 3), "Polyhedral faces")
        for element in sorted(set(scene.triangle_elements)):
            triangles = all_triangles[np.asarray(scene.triangle_elements) == element]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            norms = np.linalg.norm(normals, axis=1)
            if np.any(norms < 1e-12):
                raise ValueError("A polyhedral surface contains a degenerate triangle.")
            normals /= norms[:, None]
            geometry = glb.geometry(triangles.reshape(-1, 3), np.repeat(normals, 3, axis=0), np.arange(triangles.size // 3).reshape(-1, 3))
            mat = glb.material(f"Polyhedra {element}", _rgba(element_color(element, scientific), max(.08, scene.settings.polyhedron_opacity)), .48, True)
            add("Polyhedra", glb.node(f"Polyhedral surfaces ({element})", mesh=glb.mesh(f"Polyhedra {element}", geometry, mat),
                extras={"afruz_role": "polyhedra", "center_element": str(element), "convention": "coordination hulls grouped by element; editable faces"}))
        if scene.settings.polyhedron_edges:
            mesh = glb.mesh("Polyhedron edge cylinder", cylinder_geometry, glb.material("Polyhedron edges", theme["polyhedron_edge"]))
            seen = set()
            for triangle in all_triangles:
                for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                    key = tuple(sorted((tuple(first), tuple(second))))
                    if key not in seen:
                        seen.add(key)
                        add("Polyhedron edges", _segment(glb, f"Polyhedron edge {len(seen):06d}", first, second,
                            max(.012, scene.settings.bond_radius * .3), mesh, {"afruz_role": "polyhedron_edge"}))
    if options.pores and scene.pore_centers:
        mat = glb.material("Orange illustrative pore envelope", (*PORE_ORANGE_RGB, scene.settings.pore_opacity), .30)
        mesh = glb.mesh("Pore sphere", sphere_geometry, mat)
        for index, (center, radius) in enumerate(zip(scene.pore_centers, scene.pore_radii)):
            center = _finite(center, (3,), "Pore center")
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("Pore radius must be positive and finite.")
            add("Pore envelopes", glb.node(f"Pore envelope {index + 1:03d} — NOT an atom", mesh=mesh,
                translation=center.tolist(), scale=[float(radius)] * 3,
                extras={"afruz_role": "pore_envelope", "radius_angstrom": float(radius),
                    "convention": "grid-derived maximal-empty-sphere estimate; NOT an adsorption-accessible surface or refined atom"}))
    if options.cell_edges and scene.edges:
        mesh = glb.mesh("Cell edge cylinder", cylinder_geometry, glb.material("Unit cell guides", theme["cell_edge"]))
        for index, edge in enumerate(scene.edges):
            edge = _finite(edge, (2, 3), "Cell edge")
            add("Unit cell", _segment(glb, f"Cell edge {index + 1:04d}", *edge, .02, mesh, {"afruz_role": "cell_edge"}))
    if not groups:
        raise ValueError("Select at least one geometry type that exists in the current scene.")
    document = scene_document(scene, 0, 0, 0)
    report(80, "Saving source coordinates and refinement provenance…")
    export_info = {"format": "glTF 2.0 binary (GLB)", "software": APP_RELEASE,
        "options": asdict(options), "angstrom_per_scene_unit": 1.0,
        "physical_to_graphical_magnification": 1e10,
        "units_note": "Graphical-abstract scale: 1 Blender unit numerically represents 1 Å; glTF metre geometry is intentionally magnified 10^10 times.",
        "coordinate_note": "CIF/Studio Cartesian XYZ is Z-up. A rigid -90° X root rotation writes glTF Y-up; Blender's importer restores XYZ without mirroring.",
        "material_note": "Meshes and materials are shared for compactness; make single-user before editing only one atom's mesh/material.",
        "geometry_note": "Atom centers and contacts come directly from the frozen displayed scene. Atom radii, cylinder widths, coordination hulls and pore spheres are display conventions, not extra refined parameters.",
        "expert_override": bool(options.allow_flagged_structure),
        "object_counts": dict(Counter(node.get("extras", {}).get("afruz_role", "other") for node in glb.doc["nodes"]))}
    if camera is not None:
        document["gpu_camera"] = camera
    document["export"] = export_info
    document["conventions"] = "Editable three-dimensional geometry; no 2D projection. " + export_info["geometry_note"]
    # scene_document already sanitizes diagnostics; a live camera must be finite too.
    json.dumps(document, ensure_ascii=False, allow_nan=False)
    group_nodes = [glb.node(role, children=children, extras={"afruz_role": "group"}) for role, children in groups.items()]
    root = glb.node(scene.model.get("data_name", "Afruz crystal"), children=group_nodes,
        rotation=[-math.sqrt(.5), 0., 0., math.sqrt(.5)], extras={
            "afruz_role": "crystal_root", "angstrom_per_scene_unit": 1.,
            "source_cell_json": json.dumps(document["model"].get("cell", {}), allow_nan=False),
            "source_provenance_json": json.dumps(document["model"].get("provenance", {}), ensure_ascii=False, allow_nan=False),
            "scientific_conventions": export_info["geometry_note"], "units_note": export_info["units_note"],
            "structure_review_required": scene.model.get("provenance", {}).get("publication_ready") is False})
    glb.doc["scenes"] = [{"name": scene.model.get("data_name", "Crystal"), "nodes": [root]}]
    glb.doc["asset"]["extras"] = {"afruz_export": export_info}
    data = glb.bytes()
    document["export"]["glb_sha256"] = hashlib.sha256(data).hexdigest()
    return data, document


def _write_bundle(files):
    """Prepare everything first; preserve overwritten files and roll back failures."""
    parent = next(iter(files)).parent
    parent.mkdir(parents=True, exist_ok=True)
    backups, installed = {}, []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with tempfile.TemporaryDirectory(prefix=".afruz-3d-", dir=parent) as temporary:
        staged = {}
        for index, (target, data) in enumerate(files.items()):
            staged[target] = Path(temporary) / str(index)
            with staged[target].open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        try:
            for target in files:
                if target.exists():
                    backup = target.with_name(f"{target.stem}.backup_{stamp}{target.suffix}")
                    if backup.exists():
                        raise FileExistsError(f"Backup already exists: {backup}")
                    shutil.copy2(target, backup)
                    backups[target] = backup
            for target, temporary_path in staged.items():
                os.replace(temporary_path, target)
                installed.append(target)
        except Exception:
            for target in reversed(installed):
                if target in backups:
                    shutil.copy2(backups[target], target)
                else:
                    target.unlink(missing_ok=True)
            raise
    return list(backups.values())


def export_editable_crystal(scene, path, options=None, *, camera=None, progress=None):
    path = Path(path)
    if path.suffix.lower() != ".glb":
        raise ValueError("Editable crystal export requires a .glb filename.")
    data, document = build_editable_glb(scene, options, camera=camera, progress=progress)
    scene_path = path.with_suffix(".crystal.json")
    guide_path = path.with_name(path.stem + "_blender_README.txt")
    guide = f"""{APP_RELEASE} — editable crystal model

1. Blender: File > Import > glTF 2.0 (.glb/.gltf), then choose {path.name}.
2. Use the Outliner groups to select Atoms, Bonds, Polyhedra, Pore envelopes or Unit cell.
3. Select the imported mesh objects (A in a fresh scene), then View > Frame Selected.
   Switch to Material Preview to see colors/transparency.
4. Atoms/bonds/pores are separate objects. Polyhedral faces are grouped by center element.
5. Shared sphere/cylinder meshes keep large MOFs compact. Before editing ONLY one mesh/material,
   use Object > Relations > Make Single User > Object & Data (and copy its material when needed).
6. Save as .blend for Blender-specific editing, lighting, cameras and your graphical abstract.

{document['export']['units_note']}
{document['export']['coordinate_note']}
{document['export']['geometry_note']}
The orange sphere is an illustrative maximal-empty-sphere pore estimate, NOT an atom or an adsorption-accessible surface.
Single-pore scenes are visualization crops, not extracted molecular species. Fractional_display coordinates include any display translation.
Partial occupancies/disorder are retained as metadata, not replaced by an ordered hypothetical structure.
The companion {scene_path.name} preserves the original CIF/refinement model, settings, provenance and warnings.
Source scientific warnings: {json.dumps(document.get('warnings', []), ensure_ascii=False)}
Structure review required: {document['model'].get('provenance', {}).get('publication_ready') is False}
Expert override used: {document['export']['expert_override']}
Lighting and transparency may differ from GPU PNG/TIFF. Bond cylinder width replaces screen-space line width.
Text labels, captions, axes and image backgrounds are not baked into the 3D mesh. No diffraction/refinement calculations were changed.
GLB SHA256: {document['export']['glb_sha256']}
"""
    if progress is not None:
        progress(92, "Writing the complete 3D export bundle…")
    backups = _write_bundle({path: data, scene_path: json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"),
                            guide_path: guide.encode("utf-8")})
    if progress is not None:
        progress(100, "Editable 3D model exported.")
    return {"model_path": path, "scene_path": scene_path, "guide_path": guide_path,
            "backups": backups, "counts": document["export"]["object_counts"], "sha256": document["export"]["glb_sha256"]}
