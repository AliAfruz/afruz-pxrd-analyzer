from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from afruz_pxrd.app import MainWindow
from afruz_pxrd.crystallography import load_cif
from afruz_pxrd.crystal_render import render_crystal
from afruz_pxrd.crystal_scene import (
    MOF_PORE_STYLE, MOF_STYLE, RenderSettings, build_scene,
    model_from_structure,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _small_mof_node():
    atoms = [
        {"label": "Cr1", "element": "Cr", "x": .5, "y": .5, "z": .5, "occupancy": 1.0},
        {"label": "O1", "element": "O", "x": .70, "y": .5, "z": .5, "occupancy": 1.0},
        {"label": "O2", "element": "O", "x": .30, "y": .5, "z": .5, "occupancy": 1.0},
        {"label": "O3", "element": "O", "x": .5, "y": .70, "z": .5, "occupancy": 1.0},
        {"label": "O4", "element": "O", "x": .5, "y": .30, "z": .5, "occupancy": 1.0},
        {"label": "O5", "element": "O", "x": .5, "y": .5, "z": .70, "occupancy": 1.0},
        {"label": "O6", "element": "O", "x": .5, "y": .5, "z": .30, "occupancy": 1.0},
        {"label": "C1", "element": "C", "x": .82, "y": .5, "z": .5, "occupancy": 1.0},
        {"label": "H1", "element": "H", "x": .90, "y": .5, "z": .5, "occupancy": 1.0},
    ]
    return {
        "data_name": "MIL-like Cr node",
        "cell": {"a": 10.0, "b": 10.0, "c": 10.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        "atoms": atoms,
    }


def test_mof_style_hides_hydrogen_without_changing_model():
    model = _small_mof_node()
    scene = build_scene(
        model,
        RenderSettings(
            style=MOF_STYLE,
            repeats=(1, 1, 1),
            center_element="All metals",
            hide_hydrogen=True,
            complete_boundaries=False,
        ),
    )
    assert "H" not in scene.elements
    assert any(atom["element"] == "H" for atom in model["atoms"])
    assert scene.settings.style == MOF_STYLE


def test_mof_pore_template_renders_scientific_cr_polyhedra_on_clean_white_canvas():
    scene = build_scene(
        _small_mof_node(),
        RenderSettings(
            style=MOF_PORE_STYLE,
            repeats=(1, 1, 1),
            azimuth=45.0,
            elevation=35.3,
            atom_scale=.55,
            bond_radius=.022,
            unlike_only=False,
            center_element="All metals",
            polyhedron_opacity=.50,
            polyhedron_edges=True,
            complete_boundaries=False,
            cell_edges=False,
            axes=False,
            caption=False,
            hide_hydrogen=True,
            pore_volumes=True,
            pore_probe_radius=0.0,
            pore_opacity=.88,
            pore_max_count=2,
        ),
    )
    assert scene.triangles
    assert scene.pore_centers
    assert all(radius >= 2.0 for radius in scene.pore_radii)
    assert "H" not in scene.elements
    picture = np.asarray(render_crystal(scene, 480, 480, antialias=False))
    assert np.all(picture[0, 0, :3] == 255)
    chromium_blue = (
        (picture[..., 2].astype(int) > picture[..., 0].astype(int) + 18)
        & (picture[..., 2].astype(int) > picture[..., 1].astype(int) + 8)
        & (picture[..., 2] > 90)
    )
    assert np.count_nonzero(chromium_blue) > 50
    yellow = (
        (picture[..., 0] > 130)
        & (picture[..., 1] > 100)
        & (picture[..., 2] < picture[..., 1] * .92)
    )
    assert np.count_nonzero(yellow) > 250


def test_mof_single_cell_accepts_expanded_p1_framework_above_legacy_limit():
    model = _small_mof_node()
    model["atoms"] = [
        {
            **model["atoms"][index % len(model["atoms"])],
            "label": f"site{index}",
            "x": ((index * 17) % 997) / 997,
            "y": ((index * 31) % 991) / 991,
            "z": ((index * 47) % 983) / 983,
        }
        for index in range(6001)
    ]
    scene = build_scene(
        model,
        RenderSettings(
            style=MOF_STYLE,
            repeats=(1, 1, 1),
            bonds=False,
            polyhedra=False,
            complete_boundaries=False,
        ),
    )
    # Sites exactly on a wrapped origin are repeated on the visible cell faces.
    assert 6001 <= len(scene.positions) < 6100


def test_synthetic_periodic_mof_pore_scene_is_finite_and_retains_framework_elements():
    model = model_from_structure(load_cif(
        FIXTURES / "synthetic_cr_organic_framework.cif"
    ))
    scene = build_scene(
        model,
        RenderSettings(
            style=MOF_PORE_STYLE,
            repeats=(1, 1, 1),
            unlike_only=False,
            center_element="All metals",
            hide_hydrogen=True,
            complete_boundaries=False,
            cell_edges=False,
            axes=False,
            caption=False,
            pore_volumes=True,
            pore_max_count=4,
            isolate_pore=True,
            smart_pore_isolation=True,
            pore_shell_thickness=8.0,
        ),
    )
    counts = Counter(scene.elements)
    assert counts["Cr"] >= 1
    assert counts["O"] >= 6
    assert counts["C"] >= 9
    assert scene.bonds
    assert scene.triangles
    assert scene.pore_centers
    assert all(radius > 0 for radius in scene.pore_radii)
    assert np.isfinite(scene.positions).all()
    assert np.isfinite(np.asarray(scene.triangles)).all()


@pytest.mark.gui
def test_crystal_studio_exposes_reference_like_mof_pore_template(qtbot):
    from afruz_pxrd.crystal_studio import CrystalStudioDialog

    dialog = CrystalStudioDialog()
    qtbot.addWidget(dialog)
    dialog.style_combo.setCurrentText(MOF_PORE_STYLE)
    assert dialog.style_combo.currentText() == MOF_PORE_STYLE
    assert [spin.value() for spin in dialog.repeats] == [1, 1, 1]
    assert dialog.atom_scale.value() == pytest.approx(.55)
    assert dialog.bond_radius.value() == pytest.approx(.022)
    assert not dialog.unlike.isChecked()
    assert dialog.opacity.value() == pytest.approx(.50)
    assert dialog.hide_hydrogen.isChecked()
    assert dialog.polyhedra.isChecked()
    assert dialog.polyhedron_edges.isChecked()
    assert dialog.pore_volumes.isChecked()
    assert dialog.pore_probe.value() == pytest.approx(0.0)
    assert dialog.pore_opacity.value() == pytest.approx(.88)
    assert dialog.pore_count.value() == 2
    assert not dialog.complete_boundaries.isChecked()
    assert not dialog.cell_edges.isChecked()
    assert not dialog.axes.isChecked()
    assert not dialog.caption.isChecked()
    assert dialog.azimuth.value() == pytest.approx(45.0)
    assert dialog.elevation.value() == pytest.approx(35.3)
    assert not dialog.gpu_view_button.isEnabled()
    dialog.close()


@pytest.mark.gui
def test_mof_refinement_presets_are_conservative(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.rietveld_widget.range_min.setValue(2.0)
    window.rietveld_widget.range_max.setValue(80.0)
    window.whole_pattern_widget.range_min.setValue(2.0)
    window.whole_pattern_widget.range_max.setValue(80.0)
    window.rietveld_widget._add_structure(_small_mof_node())
    window.rietveld_widget.apply_mof_refinement_preset()
    window.whole_pattern_widget.apply_mof_refinement_preset()

    rietveld = window.rietveld_widget
    whole = window.whole_pattern_widget
    assert rietveld.cutoff.value() == pytest.approx(0.05)
    assert rietveld.range_max.value() == pytest.approx(35.0)
    assert not rietveld.refine_eta.isChecked()
    assert not rietveld.allow_flagged_structures.isChecked()
    assert rietveld.freeze_structure_factors.isChecked()
    assert rietveld.use_cuda.isChecked()
    assert not rietveld.refine_zero.isChecked()
    assert rietveld.background_order.value() == 2
    assert rietveld.validation_stride.value() == 5
    assert rietveld.kalpha2.isChecked()
    assert rietveld.phase_table.item(0, 6).text() == "1 1 1"
    assert rietveld.phase_table.item(0, 7).checkState().value == 2
    assert whole.reference_cutoff.value() == pytest.approx(0.05)
    assert whole.range_max.value() == pytest.approx(35.0)
    assert "Le Bail" in whole.mode.currentText()
    assert whole.use_cuda.isChecked()
    assert whole.extraction_cycles.value() == 20
    window.close()
