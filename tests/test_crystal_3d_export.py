from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import struct
import threading

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog

from afruz_pxrd.crystal_3d_export import ModelExportOptions, build_editable_glb, export_editable_crystal
from afruz_pxrd.crystal_scene import PORE_ORANGE_RGB, RenderSettings, SCIENTIFIC_STYLE, build_scene, load_scene_document, model_from_structure


def glb_document(data):
    assert data[:4] == b"glTF"
    version, length = struct.unpack_from("<II", data, 4)
    assert version == 2 and length == len(data)
    json_length, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A and json_length % 4 == 0
    document = json.loads(data[20:20 + json_length])
    offset = 20 + json_length
    binary_length, kind = struct.unpack_from("<II", data, offset)
    assert kind == 0x004E4942 and binary_length % 4 == 0
    return document, data[offset + 8:]


@pytest.fixture
def editable_scene():
    points = [("Cu", "Cu1", (.5, .5, .5)), ("O", "O1", (.62, .62, .62)),
              ("O", "O2", (.38, .38, .62)), ("O", "O3", (.38, .62, .38)), ("O", "O4", (.62, .38, .38))]
    model = model_from_structure({"data_name": "Editable tetrahedral fixture", "cell": dict(a=10, b=10, c=10, alpha=90, beta=90, gamma=90),
        "atoms": [{"element": element, "label": label, "x": point[0], "y": point[1], "z": point[2], "occupancy": .75 if label == "O1" else 1.} for element, label, point in points]})
    model["provenance"].update(publication_ready=True, source="Synthetic geometry test", refinement_background={"diagnostic": float("inf")})
    scene = build_scene(model, RenderSettings(style=SCIENTIFIC_STYLE, repeats=(1, 1, 1), center_element="Cu", complete_boundaries=False, atom_opacity=.6, polyhedron_edges=True))
    # A synthetic illustrative sphere tests export geometry, not pore detection.
    scene.pore_centers, scene.pore_radii = [np.array([5., 5., 5.])], [1.2]
    scene.hydrogen_bonds = [(1, 2)]
    assert scene.triangles and scene.bonds
    return scene


@pytest.mark.parametrize("quality", [1, 2, 3])
def test_glb_preserves_geometry_materials_units_and_separate_objects(editable_scene, quality):
    source = deepcopy(editable_scene)
    data, provenance = build_editable_glb(editable_scene, ModelExportOptions(sphere_subdivisions=quality))
    document, binary = glb_document(data)
    assert document["buffers"][0]["byteLength"] <= len(binary)
    nodes = document["nodes"]
    atoms = [node for node in nodes if node.get("extras", {}).get("afruz_role") == "atom"]
    assert len(atoms) == len(source.positions)
    assert len({node["name"] for node in nodes}) == len(nodes)
    for node in atoms:
        index = node["extras"]["scene_index"]
        assert node["translation"] == source.positions[index].tolist()
        assert node["extras"]["source_label"] == source.labels[index]
        assert node["extras"]["occupancy"] == source.occupancies[index]
        assert node["extras"]["fractional_display"] == source.fractional[index].tolist()
    pore = next(node for node in nodes if node.get("extras", {}).get("afruz_role") == "pore_envelope")
    assert pore["translation"] == [5., 5., 5.] and pore["scale"] == [1.2] * 3
    material = document["materials"][document["meshes"][pore["mesh"]]["primitives"][0]["material"]]
    assert material["extras"]["studio_srgb_rgba"][:3] == list(PORE_ORANGE_RGB)
    assert "NOT an adsorption-accessible" in pore["extras"]["convention"]
    oxygen = next(value for value in document["materials"] if value["name"] == "Atom O")
    assert oxygen["alphaMode"] == "BLEND" and oxygen["pbrMetallicRoughness"]["baseColorFactor"][3] == .6
    assert len({node["mesh"] for node in atoms if node["extras"]["element"] == "O"}) == 1
    assert provenance["export"]["angstrom_per_scene_unit"] == 1.
    root = next(node for node in nodes if node.get("extras", {}).get("afruz_role") == "crystal_root")
    assert root["rotation"] == pytest.approx([-np.sqrt(.5), 0, 0, np.sqrt(.5)])
    assert "Infinity" in root["extras"]["source_provenance_json"]
    assert "serialization_notes" in provenance
    assert provenance["export"]["glb_sha256"] == hashlib.sha256(data).hexdigest()
    assert np.array_equal(source.positions, editable_scene.positions)
    assert source.model["provenance"]["refinement_background"]["diagnostic"] == float("inf")
    for view in document["bufferViews"]:
        assert view["byteOffset"] % 4 == 0 and view["byteOffset"] + view["byteLength"] <= len(binary)
    for accessor in document["accessors"]:
        view = document["bufferViews"][accessor["bufferView"]]
        if accessor["componentType"] == 5126:
            values = np.frombuffer(binary, dtype="<f4", offset=view["byteOffset"], count=accessor["count"] * 3)
            assert np.isfinite(values).all()


def test_export_bundle_is_reopenable_and_preserves_overwritten_files(editable_scene, tmp_path):
    path = tmp_path / "fixture.glb"
    exported = export_editable_crystal(editable_scene, path)
    first = path.read_bytes()
    restored, settings = load_scene_document(exported["scene_path"])
    assert restored["cell"] == editable_scene.model["cell"] and settings == editable_scene.settings
    assert "Make Single User" in exported["guide_path"].read_text(encoding="utf-8")
    second = export_editable_crystal(editable_scene, path, ModelExportOptions(sphere_subdivisions=1))
    assert len(second["backups"]) == 3
    assert next(value for value in second["backups"] if value.suffix == ".glb").read_bytes() == first


@pytest.mark.parametrize("direction", [[0., 0., 2.], [0., 0., -2.], [2., 3., 4.], [-2., 3., -.5]])
def test_bond_cylinder_transforms_preserve_both_endpoints(direction):
    from afruz_pxrd.crystal_3d_export import _Glb, _segment
    first = np.array([3., -7., 19.])
    second = first + direction
    glb = _Glb()
    index = _segment(glb, "Endpoint test", first, second, .08, 0, {})
    node = glb.doc["nodes"][index]
    xyz, w = np.asarray(node["rotation"][:3]), node["rotation"][3]
    for sign, expected in ((-1., first), (1., second)):
        local = np.array([0., 0., sign * node["scale"][2] / 2.])
        rotated = local + 2 * np.cross(xyz, np.cross(xyz, local) + w * local)
        assert rotated + node["translation"] == pytest.approx(expected, abs=1e-12)


def test_bundle_rolls_back_existing_files_if_installing_a_sidecar_fails(editable_scene, tmp_path, monkeypatch):
    import afruz_pxrd.crystal_3d_export as module
    path = tmp_path / "rollback.glb"
    exported = export_editable_crystal(editable_scene, path)
    originals = {exported[key]: exported[key].read_bytes() for key in ("model_path", "scene_path", "guide_path")}
    replace_file, calls = module.os.replace, []
    def fail_second(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("Simulated sidecar write failure")
        return replace_file(source, target)
    monkeypatch.setattr(module.os, "replace", fail_second)
    with pytest.raises(OSError, match="Simulated"):
        export_editable_crystal(editable_scene, path, ModelExportOptions(sphere_subdivisions=1))
    assert all(file.read_bytes() == data for file, data in originals.items())


@pytest.mark.parametrize("damage", ["nan_position", "infinite_pore", "bad_contact", "empty_parts", "flagged"])
def test_invalid_geometry_or_review_gate_does_not_overwrite_existing_files(editable_scene, tmp_path, damage):
    path = tmp_path / "existing.glb"
    path.write_bytes(b"existing model")
    options = ModelExportOptions()
    if damage == "nan_position":
        editable_scene.positions[0, 0] = np.nan
    elif damage == "infinite_pore":
        editable_scene.pore_radii = [np.inf]
    elif damage == "bad_contact":
        editable_scene.bonds.append((0, 999))
    elif damage == "empty_parts":
        options = ModelExportOptions(atoms=False, bonds=False, polyhedra=False, pores=False, hydrogen_bonds=False, cell_edges=False)
    else:
        editable_scene.model["provenance"]["publication_ready"] = False
    with pytest.raises(ValueError):
        export_editable_crystal(editable_scene, path, options)
    assert path.read_bytes() == b"existing model"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.gui
def test_editable_export_dialog_runs_worker_and_retains_frozen_scene(qtbot, editable_scene, tmp_path, monkeypatch):
    import afruz_pxrd.crystal_3d_export_dialog as module
    dialog = module.EditableCrystalExportDialog(editable_scene)
    qtbot.addWidget(dialog)
    path = tmp_path / "threaded.glb"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(path), "Binary glTF 2.0 (*.glb)"))
    ids, gui_id = [], threading.get_ident()
    real_export = module.export_editable_crystal
    def export(*args, **kwargs):
        ids.append(threading.get_ident())
        return real_export(*args, **kwargs)
    monkeypatch.setattr(module, "export_editable_crystal", export)
    editable_scene.positions[0] = [99, 99, 99]
    dialog.export_button.click()
    assert dialog._job is not None and not dialog.controls.isEnabled()
    qtbot.waitUntil(lambda: dialog._job is None, timeout=15_000)
    assert ids and all(value != gui_id for value in ids)
    assert dialog.progress.value() == 100 and path.exists()
    doc, _ = glb_document(path.read_bytes())
    atom = next(node for node in doc["nodes"] if node.get("extras", {}).get("scene_index") == 0)
    assert atom["translation"] == [5., 5., 5.]
    dialog.close()


@pytest.mark.gui
def test_main_studio_and_gpu_entry_points_use_the_prepared_scene(qtbot, editable_scene, monkeypatch):
    from afruz_pxrd.crystal_studio import CrystalStudioDialog
    from afruz_pxrd.crystal_gpu import GpuCrystalViewDialog
    from afruz_pxrd.crystal_3d_export_dialog import EditableCrystalExportDialog
    studio = CrystalStudioDialog(models=[editable_scene.model])
    qtbot.addWidget(studio)
    studio._timer.stop()
    assert not studio.model_export_button.isEnabled()
    studio._gpu_scene_ready(studio._revision, editable_scene)
    dialog = studio.export_editable_model()
    qtbot.addWidget(dialog)
    assert np.array_equal(dialog.scene.positions, editable_scene.positions)
    dialog.close()
    qtbot.wait(10)
    studio.close()
    qtbot.wait(10)
    # Exercise the GPU action without creating an offscreen OpenGL context.
    from PySide6.QtWidgets import QDialog, QComboBox
    from PySide6.QtGui import QVector3D
    owner = QDialog()
    qtbot.addWidget(owner)
    owner._scene, owner._model_export_dialogs = editable_scene, []
    owner.background_combo = QComboBox()
    owner.background_combo.addItem("Journal white")
    class View:
        def cameraParams(self):
            return {"center": QVector3D(), "distance": 30., "azimuth": 40., "elevation": 20., "fov": 60.}
    owner.view = View()
    dialog = GpuCrystalViewDialog.export_editable_model(owner)
    qtbot.addWidget(dialog)
    assert isinstance(dialog, EditableCrystalExportDialog) and dialog.camera["distance"] == 30.
    assert np.array_equal(dialog.scene.positions, editable_scene.positions)
    dialog.close()
    qtbot.wait(10)
    owner.close()
    qtbot.wait(10)


@pytest.mark.gui
def test_flagged_dialog_requires_deliberate_override(qtbot, editable_scene):
    from afruz_pxrd.crystal_3d_export_dialog import EditableCrystalExportDialog
    editable_scene.model["provenance"]["publication_ready"] = False
    dialog = EditableCrystalExportDialog(editable_scene)
    qtbot.addWidget(dialog)
    assert not dialog.export_button.isEnabled()
    dialog.override.setChecked(True)
    assert dialog.export_button.isEnabled()
    data, _ = build_editable_glb(editable_scene, ModelExportOptions(allow_flagged_structure=True))
    doc, _ = glb_document(data)
    assert doc["asset"]["extras"]["afruz_export"]["expert_override"]
    dialog.close()
