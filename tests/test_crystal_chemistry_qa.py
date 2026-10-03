import json
import math

import pytest

from afruz_pxrd.crystal_chemistry import (
    analyze_crystal_chemistry,
    export_crystal_chemistry_bundle,
    export_crystal_chemistry_report,
    format_crystal_chemistry_report,
)


def _cell(a=10.0, b=10.0, c=10.0):
    return {"a": a, "b": b, "c": c, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}


def _ideal_nickel_octahedron():
    # Ni2+-O distance that gives BVS=2 for six equal bonds with the bundled
    # Gagne-Hawthorne (2015) Ni2+-O parameters.
    distance = 1.689 - 0.347 * math.log(2.0 / 6.0)
    offset = distance / 10.0
    atoms = [{"label": "Ni1", "element": "Ni", "x": .5, "y": .5, "z": .5, "occupancy": 1}]
    for label, vector in enumerate(((offset, 0, 0), (-offset, 0, 0), (0, offset, 0),
                                    (0, -offset, 0), (0, 0, offset), (0, 0, -offset)), start=1):
        atoms.append({
            "label": f"O{label}", "element": "O",
            "x": .5 + vector[0], "y": .5 + vector[1], "z": .5 + vector[2], "occupancy": 1,
        })
    return {"data_name": "ideal NiO6", "formula": "Ni O6", "space_group": "P 1", "cell": _cell(), "atoms": atoms}


def test_qa_measures_octahedron_and_bond_valence_without_moving_sites():
    structure = _ideal_nickel_octahedron()
    before = [dict(atom) for atom in structure["atoms"]]
    report = analyze_crystal_chemistry(structure)

    site = report["metal_sites"][0]
    bvs = report["bond_valence"]["site_results"][0]
    assert site["oxygen_coordination_number"] == 6
    assert site["coordination_geometry"]["bond_length_distortion_index"] == pytest.approx(0, abs=1e-12)
    assert site["coordination_geometry"]["ideal_angle_rms_deviation_deg"] == pytest.approx(0, abs=1e-10)
    assert bvs["assumed_oxidation_state"] == 2
    assert bvs["selected_bond_valence_sum_vu"] == pytest.approx(2.0, rel=1e-10)
    assert structure["atoms"] == before


def test_qa_classifies_water_hydroxyl_and_carbonate_connectivity():
    atoms = [
        {"label": "C1", "element": "C", "x": .15, "y": .15, "z": .15, "occupancy": 1},
        {"label": "Oc1", "element": "O", "x": .28, "y": .15, "z": .15, "occupancy": 1},
        {"label": "Oc2", "element": "O", "x": .15, "y": .28, "z": .15, "occupancy": 1},
        {"label": "Oc3", "element": "O", "x": .15, "y": .15, "z": .28, "occupancy": 1},
        {"label": "Ow", "element": "O", "x": .70, "y": .70, "z": .70, "occupancy": 1},
        {"label": "Hw1", "element": "H", "x": .796, "y": .70, "z": .70, "occupancy": 1},
        {"label": "Hw2", "element": "H", "x": .676, "y": .793, "z": .70, "occupancy": 1},
        {"label": "Oh", "element": "O", "x": .48, "y": .48, "z": .48, "occupancy": 1},
        {"label": "Hh", "element": "H", "x": .576, "y": .48, "z": .48, "occupancy": 1},
    ]
    report = analyze_crystal_chemistry({"data_name": "groups", "cell": _cell(), "atoms": atoms})
    groups = report["functional_group_screen"]
    assert groups["carbonate_like_group_count"] == 1
    assert groups["assigned_o_h_bond_count"] == 3
    assert groups["oxygen_site_assignments"] == {
        "carbon-bound oxygen": 3,
        "water-like oxygen": 1,
        "hydroxyl-like oxygen": 1,
    }


def test_qa_detects_exact_half_cell_translation():
    structure = {
        "data_name": "twofold supercell",
        "cell": _cell(a=8, b=4, c=4),
        "atoms": [
            {"label": "Fe1", "element": "Fe", "x": .10, "y": .20, "z": .30, "occupancy": 1},
            {"label": "Fe2", "element": "Fe", "x": .60, "y": .20, "z": .30, "occupancy": 1},
            {"label": "O1", "element": "O", "x": .20, "y": .20, "z": .30, "occupancy": 1},
            {"label": "O2", "element": "O", "x": .70, "y": .20, "z": .30, "occupancy": 1},
        ],
    }
    report = analyze_crystal_chemistry(structure, symmetry_tolerance_angstrom=.02)
    symmetry = report["symmetry_screen"]
    assert symmetry["strong_translation_count"] >= 1
    best = symmetry["translation_candidates"][0]
    assert best["matched_occupied_fraction"] == pytest.approx(1)
    assert best["commensurate_order"] == 2
    assert "supercell along a" in best["cell_reduction_hint"]


def test_qa_detects_lattice_compatible_inversion_operation():
    structure = {
        "data_name": "centrosymmetric P1 listing",
        "cell": _cell(a=7, b=8, c=9),
        "atoms": [
            {"label": "C1", "element": "C", "x": .13, "y": .24, "z": .31, "occupancy": 1},
            {"label": "C2", "element": "C", "x": .87, "y": .76, "z": .69, "occupancy": 1},
            {"label": "O1", "element": "O", "x": .22, "y": .35, "z": .44, "occupancy": 1},
            {"label": "O2", "element": "O", "x": .78, "y": .65, "z": .56, "occupancy": 1},
        ],
    }
    symmetry = analyze_crystal_chemistry(structure, symmetry_tolerance_angstrom=.02)["symmetry_screen"]
    assert symmetry["strong_point_operation_count"] >= 1
    assert symmetry["point_operation_candidates"][0]["matched_occupied_fraction"] == pytest.approx(1)


def test_report_exports_full_json_and_readable_companions(tmp_path):
    structure = _ideal_nickel_octahedron()
    report = analyze_crystal_chemistry(structure)
    text = format_crystal_chemistry_report(report)
    assert "METAL–O COORDINATION" in text
    assert "BOND-VALENCE SUMS" in text
    assert "10.1107/S2052520615016297" in text

    standalone = export_crystal_chemistry_report(report, tmp_path / "qa.json")
    assert json.loads(standalone.read_text(encoding="utf-8"))["schema"] == "afruz.crystal-chemistry-qa.v1"
    json_path, text_path = export_crystal_chemistry_bundle(structure, tmp_path / "figure.png")
    assert json_path.name == "figure.chemistry-qa.json"
    assert text_path.name == "figure.chemistry-qa.txt"
    assert json_path.exists() and text_path.exists()
