from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
from PIL import Image
import pytest

from afruz_pxrd.crystallography import direct_metric_tensor
from afruz_pxrd.crystal_scene import (
    MOF_STYLE, SCIENTIFIC_STYLE, CrystalSceneError, RenderSettings, build_scene, cell_basis,
    load_scene_document, model_from_structure, models_from_result, structure_snapshot,
)
from afruz_pxrd.crystal_render import _Canvas, export_crystal, render_crystal


def test_nonorthogonal_cell_metric_and_handedness():
    cell = dict(a=4.1, b=5.2, c=6.3, alpha=81, beta=104, gamma=119)
    basis = cell_basis(cell)
    assert basis @ basis.T == pytest.approx(direct_metric_tensor(cell))
    assert np.linalg.det(basis) > 0
    assert np.linalg.norm(np.array([1, 0, 0]) @ basis) == pytest.approx(4.1)
    for invalid in (dict(cell, a=-2), dict(cell, gamma=0), dict(cell, alpha=float("nan")), dict(cell, alpha=175, beta=5, gamma=5)):
        with pytest.raises(CrystalSceneError):
            cell_basis(invalid)


def test_result_uses_frozen_atoms_and_refined_cell(nacl_structure):
    source = deepcopy(nacl_structure)
    snapshot = structure_snapshot(source)
    phase = {"structure_snapshot": snapshot, "phase_name": "Saved NaCl", "refined_cell": dict(source["cell"], a=5.9), "atoms_fixed": True}
    source["atoms"][0]["x"] = .123
    model = models_from_result({"phases": [phase]})[0]
    assert model["cell"]["a"] == 5.9
    assert model["atoms"][0]["x"] == nacl_structure["atoms"][0]["x"]
    model["atoms"][0]["x"] = .321
    assert snapshot["atoms"][0]["x"] == nacl_structure["atoms"][0]["x"]
    assert model["provenance"]["kind"] == "rietveld_result"
    with pytest.raises(CrystalSceneError, match="older refinement"):
        models_from_result({"phases": [{"refined_cell": source["cell"]}]})


def test_rietveld_scene_uses_same_geometry_pipeline_as_its_frozen_cif(nacl_structure):
    direct_model = model_from_structure(nacl_structure)
    phase = {
        "structure_snapshot": structure_snapshot(nacl_structure),
        "phase_name": "Rietveld NaCl",
        "refined_cell": deepcopy(nacl_structure["cell"]),
        "atoms_fixed": True,
    }
    refined_model = models_from_result({"success": True, "phases": [phase]})[0]
    settings = RenderSettings(repeats=(1, 1, 1))
    direct_scene = build_scene(direct_model, settings)
    refined_scene = build_scene(refined_model, settings)
    assert refined_scene.positions == pytest.approx(direct_scene.positions)
    assert refined_scene.fractional == pytest.approx(direct_scene.fractional)
    assert refined_scene.bonds == direct_scene.bonds
    assert refined_scene.elements == direct_scene.elements


def test_periodic_boundary_sites_and_nearest_contacts(cscl_structure):
    scene = build_scene(model_from_structure(cscl_structure), RenderSettings(repeats=(1, 1, 1)))
    assert len(scene.positions) == 9  # 8 corner Cs + 1 body-center Cl
    assert len(scene.bonds) == 8
    distances = [np.linalg.norm(scene.positions[i] - scene.positions[j]) for i, j in scene.bonds]
    assert distances == pytest.approx([4.123 * np.sqrt(3) / 2] * 8)
    assert len({(element, *xyz) for element, xyz in zip(scene.elements, scene.positions)}) == 9


def test_snapshot_survives_project_save_load_and_carries_failed_fit_status(tmp_path, cscl_structure):
    from afruz_pxrd.project import save_afz, load_afz
    from afruz_pxrd.models import Dataset
    phase = dict(structure_snapshot=structure_snapshot(cscl_structure), refined_cell=dict(cscl_structure["cell"], a=4.2), phase_name="Saved CsCl")
    result = dict(phases=[phase], success=False, message="Evaluation limit reached")
    dataset = Dataset(name="example", uid="example", x=np.array([10., 11., 12., 13.]), y_raw=np.array([1., 2., 3., 4.]))
    path = tmp_path / "crystal.afz"
    save_afz(path, [dataset], {}, {"rietveld": {"results_by_uid": {dataset.uid: result}}})
    _, _, restored = load_afz(path)
    model = models_from_result(restored["rietveld"]["results_by_uid"][dataset.uid])[0]
    assert model["cell"]["a"] == 4.2
    assert model["atoms"] == cscl_structure["atoms"]
    assert "did not converge" in model["provenance"]["note"]
    assert any("did not converge" in warning for warning in build_scene(model, RenderSettings()).warnings)


def test_polyhedra_enclose_center_and_exclude_boundary_fragments(cscl_structure):
    scene = build_scene(model_from_structure(cscl_structure), RenderSettings(repeats=(1, 1, 1), center_element="Cl"))
    assert len(scene.triangles) == 12  # six cube faces, two triangles each
    assert np.max(np.array(scene.triangles)) == pytest.approx(4.123)
    boundary = build_scene(model_from_structure(cscl_structure), RenderSettings(repeats=(1, 1, 1), center_element="Cs"))
    assert not boundary.triangles


def test_scientific_style_supports_multiple_metal_polyhedra_and_outer_cell_box():
    tetrahedron = ((.10, .10, .10), (.10, -.10, -.10), (-.10, .10, -.10), (-.10, -.10, .10))
    atoms = [
        {"element": "Fe", "label": "Fe1", "x": .25, "y": .25, "z": .25, "occupancy": 1},
        {"element": "Ni", "label": "Ni1", "x": .75, "y": .75, "z": .75, "occupancy": 1},
    ]
    for name, center in (("Fe", np.array([.25, .25, .25])), ("Ni", np.array([.75, .75, .75]))):
        for index, offset in enumerate(tetrahedron):
            point = center + offset
            atoms.append({"element": "O", "label": f"O{name}{index}", "x": point[0], "y": point[1], "z": point[2], "occupancy": 1})
    model = model_from_structure({"data_name": "Fe-Ni polyhedra", "cell": dict(a=10, b=10, c=10, alpha=90, beta=90, gamma=90), "atoms": atoms})
    scene = build_scene(model, RenderSettings(style=SCIENTIFIC_STYLE, repeats=(1, 1, 1),
                                               center_element="All metals", cell_grid=False))
    assert set(scene.triangle_elements) == {"Fe", "Ni"}
    assert len(scene.triangles) == 8
    assert len(scene.edges) == 12
    gridded = build_scene(model, RenderSettings(style=SCIENTIFIC_STYLE, repeats=(2, 2, 2),
                                                 center_element="All metals", cell_grid=True))
    assert len(gridded.edges) == 27


def test_scientific_style_detects_dashed_hydrogen_bonds_separately():
    model = model_from_structure({
        "data_name": "Hydrogen bond test",
        "cell": dict(a=10, b=10, c=10, alpha=90, beta=90, gamma=90),
        "atoms": [
            {"element": "O", "label": "Odonor", "x": .20, "y": .50, "z": .50, "occupancy": 1},
            {"element": "H", "label": "H1", "x": .30, "y": .50, "z": .50, "occupancy": 1},
            {"element": "O", "label": "Oacceptor", "x": .50, "y": .50, "z": .50, "occupancy": 1},
        ],
    })
    scene = build_scene(model, RenderSettings(style=SCIENTIFIC_STYLE, repeats=(1, 1, 1),
                                               polyhedra=False, hydrogen_bonds=True))
    assert scene.hydrogen_bonds
    assert all(scene.elements[h] == "H" and scene.elements[a] == "O" for h, a in scene.hydrogen_bonds)
    assert all(pair not in scene.bonds for pair in scene.hydrogen_bonds)
    assert any("D–H···A" in warning for warning in scene.warnings)


def test_scientific_style_suppresses_metal_metal_contacts_unless_requested():
    model = model_from_structure({
        "data_name": "Metal contact test",
        "cell": dict(a=10, b=10, c=10, alpha=90, beta=90, gamma=90),
        "atoms": [
            {"element": "Fe", "label": "Fe1", "x": .30, "y": .50, "z": .50, "occupancy": 1},
            {"element": "Ni", "label": "Ni1", "x": .60, "y": .50, "z": .50, "occupancy": 1},
        ],
    })
    settings = RenderSettings(style=SCIENTIFIC_STYLE, repeats=(1, 1, 1),
                              polyhedra=False, complete_boundaries=False)
    assert not build_scene(model, settings).bonds
    explicit = build_scene(model, replace(settings, metal_metal_bonds=True))
    assert len(explicit.bonds) == 1


def test_scientific_style_completes_periodic_boundary_polyhedra():
    atoms = [
        {"element": "Fe", "label": "Fe1", "x": .05, "y": .50, "z": .50, "occupancy": 1},
        {"element": "O", "label": "Ox+", "x": .25, "y": .50, "z": .50, "occupancy": 1},
        {"element": "O", "label": "Ox-", "x": .85, "y": .50, "z": .50, "occupancy": 1},
        {"element": "O", "label": "Oy+", "x": .05, "y": .70, "z": .50, "occupancy": 1},
        {"element": "O", "label": "Oy-", "x": .05, "y": .30, "z": .50, "occupancy": 1},
        {"element": "O", "label": "Oz+", "x": .05, "y": .50, "z": .70, "occupancy": 1},
        {"element": "O", "label": "Oz-", "x": .05, "y": .50, "z": .30, "occupancy": 1},
    ]
    model = model_from_structure({"data_name": "Boundary octahedron",
        "cell": dict(a=10, b=10, c=10, alpha=90, beta=90, gamma=90), "atoms": atoms})
    scene = build_scene(model, RenderSettings(style=SCIENTIFIC_STYLE, repeats=(1, 1, 1),
                                               center_element="All metals", complete_boundaries=True))
    fe = next(i for i, (element, primary) in enumerate(zip(scene.elements, scene.primary))
              if element == "Fe" and primary)
    oxygen_neighbors = {j if i == fe else i for i, j in scene.bonds if fe in {i, j} and scene.elements[j if i == fe else i] == "O"}
    assert len(oxygen_neighbors) == 6
    assert any(not value for value in scene.primary)
    assert len(scene.triangles) == 8


def test_hydrogen_bond_assignment_uses_only_best_acceptor():
    model = model_from_structure({
        "data_name": "Competing acceptors",
        "cell": dict(a=10, b=10, c=10, alpha=90, beta=90, gamma=90),
        "atoms": [
            {"element": "O", "label": "D", "x": .20, "y": .50, "z": .50, "occupancy": 1},
            {"element": "H", "label": "H", "x": .30, "y": .50, "z": .50, "occupancy": 1},
            {"element": "O", "label": "A1", "x": .50, "y": .50, "z": .50, "occupancy": 1},
            {"element": "O", "label": "A2", "x": .52, "y": .50, "z": .50, "occupancy": 1},
        ],
    })
    scene = build_scene(model, RenderSettings(style=SCIENTIFIC_STYLE, repeats=(1, 1, 1),
                                               polyhedra=False, hydrogen_bonds=True,
                                               complete_boundaries=False))
    assert len(scene.hydrogen_bonds) == 1


def test_partial_occupancy_zero_sites_and_bad_coordinates(cscl_structure):
    model = model_from_structure(cscl_structure)
    model["atoms"][0]["occupancy"] = 1.00002
    rounded = build_scene(model, RenderSettings(repeats=(1, 1, 1)))
    assert all(occupancy <= 1.0 for occupancy in rounded.occupancies)
    assert any("Normalized" in warning for warning in rounded.warnings)
    model["atoms"][0]["occupancy"] = .5
    scene = build_scene(model, RenderSettings(repeats=(1, 1, 1)))
    assert .5 in scene.occupancies
    assert any("Partial occupancies" in warning for warning in scene.warnings)
    model["atoms"][0]["occupancy"] = 0
    assert len(build_scene(model, RenderSettings()).positions) < len(scene.positions) * 8
    model["atoms"][1]["x"] = float("nan")
    with pytest.raises(CrystalSceneError, match="finite"):
        build_scene(model, RenderSettings())


def test_single_pore_mode_crops_and_centers_framework(monkeypatch):
    import afruz_pxrd.crystal_scene as scene_module

    monkeypatch.setattr(
        scene_module,
        "_detect_pore_envelopes",
        lambda *args, **kwargs: ([np.array([5.0, 5.0, 5.0])], [2.5]),
    )
    model = model_from_structure({
        "data_name": "Pore crop",
        "cell": dict(a=10, b=10, c=10, alpha=90, beta=90, gamma=90),
        "atoms": [
            {"element": "Cr", "label": "Cr1", "x": .50, "y": .50, "z": .80, "occupancy": 1},
            {"element": "O", "label": "O1", "x": .50, "y": .50, "z": .85, "occupancy": 1},
            {"element": "C", "label": "far", "x": .95, "y": .95, "z": .95, "occupancy": 1},
        ],
    })
    model["provenance"]["publication_ready"] = True
    scene = build_scene(
        model,
        RenderSettings(
            style=MOF_STYLE,
            repeats=(1, 1, 1),
            bonds=False,
            polyhedra=False,
            cell_edges=True,
            pore_volumes=False,
            isolate_pore=True,
            smart_pore_isolation=False,
            pore_shell_thickness=1.0,
        ),
    )
    assert scene.elements == ["Cr", "O"]
    assert not scene.edges
    assert scene.pore_centers[0] == pytest.approx([5.0, 5.0, 5.0])
    assert scene.pore_radii == [2.5]
    assert any("visualization crop" in warning for warning in scene.warnings)


def test_smart_pore_mode_excludes_disconnected_guest_sites(monkeypatch):
    import afruz_pxrd.crystal_scene as scene_module

    detected_sites = []

    def fake_detector(sites, *args, **kwargs):
        detected_sites.extend(sites)
        return [np.array([10.0, 10.0, 10.0])], [1.0]

    monkeypatch.setattr(scene_module, "_detect_pore_envelopes", fake_detector)
    model = model_from_structure({
        "data_name": "Framework plus guest",
        "cell": dict(a=20, b=20, c=20, alpha=90, beta=90, gamma=90),
        "atoms": [
            {"element": "Cr", "label": "Cr1", "x": .5, "y": .5, "z": .5, "occupancy": 1},
            {"element": "O", "label": "O1", "x": .6, "y": .5, "z": .5, "occupancy": 1},
            {"element": "O", "label": "O2", "x": .4, "y": .5, "z": .5, "occupancy": 1},
            {"element": "O", "label": "O3", "x": .5, "y": .6, "z": .5, "occupancy": 1},
            {"element": "O", "label": "O4", "x": .5, "y": .4, "z": .5, "occupancy": 1},
            {"element": "C", "label": "guest", "x": .5, "y": .5, "z": .65, "occupancy": 1},
        ],
    })
    scene = build_scene(
        model,
        RenderSettings(
            style=MOF_STYLE,
            repeats=(1, 1, 1),
            polyhedra=False,
            isolate_pore=True,
            smart_pore_isolation=True,
            pore_max_count=1,
            pore_shell_thickness=4.0,
        ),
    )
    assert {site[1] for site in detected_sites} == {"Cr", "O"}
    assert "C" not in scene.elements
    assert any("rejected 1 disconnected radial site" in warning for warning in scene.warnings)


def test_atom_opacity_changes_cpu_raster_alpha():
    model = model_from_structure({
        "data_name": "Transparent atom",
        "cell": dict(a=10, b=10, c=10, alpha=90, beta=90, gamma=90),
        "atoms": [
            {"element": "C", "label": "C1", "x": .5, "y": .5, "z": .5, "occupancy": 1},
        ],
    })
    model["provenance"]["publication_ready"] = True
    scene = build_scene(
        model,
        RenderSettings(
            repeats=(1, 1, 1), bonds=False, polyhedra=False,
            cell_edges=False, axes=False, caption=False, transparent=True,
            atom_opacity=.25,
        ),
    )
    image = np.asarray(render_crystal(scene, 320, 240, antialias=False))
    assert 62 <= int(image[..., 3].max()) <= 64
    with pytest.raises(CrystalSceneError, match="atom_opacity"):
        RenderSettings(atom_opacity=0.0).validate()


def test_same_element_contacts_require_explicit_toggle(cscl_structure):
    model = model_from_structure(cscl_structure)
    model["atoms"] = [model["atoms"][0]]
    settings = RenderSettings(repeats=(1, 1, 1), polyhedra=False)
    assert not build_scene(model, settings).bonds
    assert len(build_scene(model, replace(settings, unlike_only=False)).bonds) == 12
    with pytest.raises(CrystalSceneError):
        build_scene(model, replace(settings, repeats=(0, 1, 1)))


def test_depth_buffer_hides_rear_sphere_and_transparency_composites():
    canvas = _Canvas(200, 200, False, True)
    canvas.sphere(np.array([100, 100, 10]), 30, np.array([1., 0., 0.]))
    before = canvas.rgb[100, 100].copy()
    canvas.sphere(np.array([100, 100, -30]), 30, np.array([0., 1., 0.]))
    assert np.array_equal(canvas.rgb[100, 100], before)
    assert canvas.alpha[0, 0] == 0
    assert canvas.alpha[100, 100] == 1
    canvas.triangle(np.array([[0, 0, 100], [70, 0, 100], [0, 70, 100]]), np.array([.5, .4, .8]), .2)
    assert canvas.alpha[10, 10] == pytest.approx(.2)
    assert canvas.rgb[10, 10] == pytest.approx([.5, .4, .8])


@pytest.mark.parametrize("suffix", [".png", ".tiff"])
def test_exports_have_resolution_dpi_metadata_and_reopenable_scene(tmp_path, cscl_structure, suffix):
    settings = RenderSettings(repeats=(1, 1, 1), transparent=True, caption=False, center_element="Cl")
    scene = build_scene(model_from_structure(cscl_structure), settings)
    scene.model["provenance"]["condition_number"] = float("inf")
    image_path, scene_path = export_crystal(scene, tmp_path / ("figure" + suffix), width=640, height=480, dpi=600)
    with Image.open(image_path) as picture:
        assert picture.size == (640, 480)
        assert picture.info["dpi"][0] == pytest.approx(600, abs=.1)
        assert picture.getextrema()[3] == (0, 255)
        metadata = picture.info["AfruzCrystalScene"] if suffix == ".png" else picture.tag_v2[270]
        embedded = json.loads(metadata)
        assert embedded["schema"] == "afruz.crystal-scene.v1"
        assert embedded["model"]["provenance"]["condition_number"] == "Infinity"
        assert embedded["serialization_notes"]
    model, restored_settings = load_scene_document(scene_path)
    assert restored_settings == settings
    restored = build_scene(model, restored_settings)
    assert restored.positions == pytest.approx(scene.positions)
    assert np.array_equal(np.asarray(render_crystal(restored, 320, 240)), np.asarray(render_crystal(scene, 320, 240)))


@pytest.mark.gui
def test_studio_renders_result_and_updates_style(qtbot, cscl_structure):
    from afruz_pxrd.crystal_studio import CrystalStudioDialog
    phase = dict(structure_snapshot=structure_snapshot(cscl_structure), refined_cell=cscl_structure["cell"], phase_name="CsCl result")
    dialog = CrystalStudioDialog({"phases": [phase]})
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.preview._image is not None and dialog._preview_job is None, timeout=30000)
    assert "fixed" in dialog.provenance.text()
    first = dialog.preview._image.copy()
    dialog.style_combo.setCurrentIndex(1)
    qtbot.waitUntil(lambda: dialog._preview_job is None and dialog.preview._image != first, timeout=30000)
    dialog.style_combo.setCurrentText(SCIENTIFIC_STYLE)
    qtbot.waitUntil(lambda: dialog._preview_job is None and dialog.style_combo.currentText() == SCIENTIFIC_STYLE, timeout=30000)
    assert dialog.center_combo.currentText() == "All metals"
    assert dialog.hydrogen_bonds.isChecked()
    assert dialog.polyhedron_edges.isChecked()
    assert not dialog.cell_grid.isChecked()
    assert [spin.value() for spin in dialog.repeats] == [1, 1, 1]
    assert dialog.export_button.isEnabled()
    assert dialog.qa_button.isEnabled()
    dialog.atom_opacity.setValue(45)
    assert dialog.settings().atom_opacity == pytest.approx(.45)
    dialog.isolate_pore.setChecked(True)
    assert dialog.pore_volumes.isChecked()
    assert dialog.pore_index.isEnabled()
    assert dialog.pore_shell.isEnabled()
    assert dialog.smart_pore.isChecked() and dialog.smart_pore.isEnabled()
    dialog.isolate_pore.setChecked(False)
    assert "frozen coordinates" in dialog.qa_summary.text()
    dialog.qa_button.click()
    qtbot.waitUntil(lambda: dialog._qa_job is None, timeout=30000)
    assert dialog._qa_dialogs
    assert "metal sites" in dialog.qa_summary.text()
    dialog._qa_dialogs[0].close()
    dialog.close()


@pytest.mark.gui
def test_old_result_requires_rerun_instead_of_guessing_atoms(qtbot):
    from afruz_pxrd.crystal_studio import CrystalStudioDialog
    dialog = CrystalStudioDialog({"phases": [{"phase_name": "Old"}]})
    qtbot.addWidget(dialog)
    assert "older refinement" in dialog.provenance.text()
    assert not dialog.export_button.isEnabled()
    dialog.close()


def test_scene_honors_cif_geometric_bond_beyond_inferred_radius():
    from afruz_pxrd.crystallography import parse_cif_text

    structure = parse_cif_text(
        """data_explicit_contact
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.10 0.10 0.10
C2 C 0.40 0.10 0.10
loop_
_geom_bond_atom_site_label_1
_geom_bond_atom_site_label_2
_geom_bond_distance
C1 C2 3.0
"""
    )
    scene = build_scene(
        model_from_structure(structure),
        RenderSettings(
            style=SCIENTIFIC_STYLE, repeats=(1, 1, 1),
            complete_boundaries=False, unlike_only=False, polyhedra=False,
        ),
    )
    assert scene.bonds == [(0, 1)]
    assert any("Matched 1 of 1 CIF geometric bond" in item for item in scene.warnings)


@pytest.mark.gui
def test_studio_applies_explainable_smart_plan_without_changing_model(qtbot, cscl_structure):
    from afruz_pxrd.crystal_intelligence import analyze_crystal_model, format_smart_crystal_report
    from afruz_pxrd.crystal_studio import CrystalStudioDialog

    phase = dict(
        structure_snapshot=structure_snapshot(cscl_structure),
        refined_cell=cscl_structure["cell"], phase_name="Smart CsCl",
    )
    dialog = CrystalStudioDialog({"phases": [phase]})
    qtbot.addWidget(dialog)
    before = deepcopy(dialog.models[0])
    report = analyze_crystal_model(dialog._current_model())
    dialog._smart_report = report
    dialog.smart_summary.setPlainText(format_smart_crystal_report(report))
    dialog._apply_smart_plan(report["recommendation"])
    assert dialog.models[0] == before
    assert dialog.style_combo.currentText() == report["recommendation"]["style"]
    assert dialog.gpu_first.isChecked() == report["recommendation"]["gpu_first"]
    dialog.smart_summary.selectAll()
    assert "Scientific limits:" in dialog.smart_summary.textCursor().selectedText()
    dialog.close()


@pytest.mark.gui
def test_studio_uses_safe_single_cell_preset_for_large_general_structure(
    qtbot, monkeypatch
):
    import afruz_pxrd.crystal_studio as studio_module

    large_result_model = {
        "data_name": "Large Rietveld MOF",
        "cell": dict(a=88, b=88, c=88, alpha=90, beta=90, gamma=90),
        "atoms": [
            {"element": "C", "label": f"C{index}"}
            for index in range(6001)
        ],
        "provenance": {"publication_ready": True},
    }
    monkeypatch.setattr(
        studio_module,
        "models_from_result",
        lambda result: [large_result_model],
    )
    dialog = studio_module.CrystalStudioDialog({"phases": [{}]})
    qtbot.addWidget(dialog)
    assert dialog.style_combo.currentText() == SCIENTIFIC_STYLE
    assert [spin.value() for spin in dialog.repeats] == [1, 1, 1]
    assert dialog.hide_hydrogen.isChecked()
    assert not dialog.complete_boundaries.isChecked()
    assert not dialog.bonds.isChecked()
    assert not dialog.polyhedra.isChecked()
    assert dialog.gpu_first.isChecked()
    dialog.close()


@pytest.mark.gui
def test_studio_opens_cif_on_background_worker(qtbot, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog
    from afruz_pxrd.crystal_studio import CrystalStudioDialog

    cif_path = tmp_path / "background-load.cif"
    cif_path.write_text(
        """data_background
_cell_length_a 5
_cell_length_b 5
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0 0 0
Cl1 Cl 0.5 0.5 0.5
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(cif_path), "Crystal structures (*.cif)"),
    )
    dialog = CrystalStudioDialog()
    qtbot.addWidget(dialog)
    dialog.open_button.click()
    assert dialog._load_job is not None
    assert not dialog.open_button.isEnabled()
    qtbot.waitUntil(
        lambda: (
            dialog._load_job is None
            and dialog._preview_job is None
            and dialog.preview._image is not None
        ),
        timeout=30000,
    )
    assert len(dialog.models) == 1
    assert dialog.open_button.isEnabled()
    assert dialog.preview._image is not None
    dialog.close()


@pytest.mark.gui
def test_studio_locks_flagged_publication_export_until_expert_override(qtbot, nacl_structure):
    from afruz_pxrd.crystal_studio import CrystalStudioDialog
    structure = deepcopy(nacl_structure)
    structure["atoms"].append(deepcopy(structure["atoms"][0]))
    structure["atoms"][-1]["label"] = "duplicate"
    phase = dict(structure_snapshot=structure_snapshot(structure), refined_cell=structure["cell"],
                 phase_name="Flagged result")
    dialog = CrystalStudioDialog({"phases": [phase]})
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.preview._image is not None and dialog._preview_job is None, timeout=30000)
    assert dialog.validation_banner.isVisible()
    assert not dialog.export_button.isEnabled()
    dialog.allow_flagged_export.setChecked(True)
    assert dialog.export_button.isEnabled()
    dialog.close()
