import pytest

from afruz_pxrd.crystallography import parse_cif_text
from afruz_pxrd.doping_series import refined_cif_text
from afruz_pxrd.structure_solution import build_p1_provisional_cif
from afruz_pxrd.structure_solving import (
    DirectSpaceSettings,
    build_provisional_cif_text,
    score_direct_space_model,
)
from afruz_pxrd.structure_validation import validate_crystal_structure


def _cell(length=6.0):
    return dict(a=length, b=length, c=length, alpha=90, beta=90, gamma=90)


def _water_hbond_structure():
    # O-H = 0.96 Å, H-O-H = 104.5°, H...O = 1.84 Å.
    return {
        "data_name": "water geometry",
        "formula": "H2 O2",
        "cell": _cell(),
        "atoms": [
            {"label": "Ow", "element": "O", "x": .25, "y": .25, "z": .25, "occupancy": 1},
            {"label": "H1", "element": "H", "x": .41, "y": .25, "z": .25, "occupancy": 1},
            {"label": "H2", "element": "H", "x": .20994, "y": .40490, "z": .25, "occupancy": 1},
            {"label": "Oa", "element": "O", "x": .71667, "y": .25, "z": .25, "occupancy": 1},
        ],
    }


def test_validator_separates_water_one_three_and_hydrogen_bond_contacts():
    validation = validate_crystal_structure(_water_hbond_structure())

    assert validation["vdw_overlap_count"] == 0
    assert validation["topology_excluded_pair_count"] >= 1
    assert validation["hydrogen_bond_count"] == 1
    assert validation["status"] == "Plausibility passed"


def test_true_nonbonded_overlap_remains_a_generator_penalty():
    atoms = [
        {"label": "H1", "element": "H", "x": .20, "y": .20, "z": .20, "occupancy": 1},
        {"label": "H2", "element": "H", "x": .36667, "y": .20, "z": .20, "occupancy": 1},
    ]
    structure = {"formula": "H2", "cell": _cell(), "atoms": atoms}
    validation = validate_crystal_structure(structure)
    assert validation["vdw_overlap_count"] == 1
    assert validation["nonbonded_overlap_penalty"] > 0

    reflections = [
        {"h": 1, "k": 0, "l": 0, "intensity": 100.0, "sigma": 10.0},
        {"h": 0, "k": 1, "l": 0, "intensity": 50.0, "sigma": 7.0},
    ]
    score = score_direct_space_model(
        cell=structure["cell"], atoms=atoms, reflections=reflections,
        settings=DirectSpaceSettings(collision_weight=40.0),
    )
    assert score["nonbonded_overlap_count"] == 1
    assert score["collision_penalty"] >= score["nonbonded_overlap_penalty"] > 0


def test_generated_cifs_fill_site_formula_weight_and_z():
    atoms = _water_hbond_structure()["atoms"]
    direct = parse_cif_text(build_provisional_cif_text(cell=_cell(), atoms=atoms, formula_hint=""))
    manual = parse_cif_text(build_p1_provisional_cif(_cell(), atoms, formula=""))

    for structure in (direct, manual):
        assert structure["formula"] == "H2 O2"
        assert structure["formula_units_z"] == 1
        assert structure["formula_weight"] == pytest.approx(34.014)


def test_refined_export_fills_missing_formula_and_replaces_audit_tags_once():
    raw = """data_demo
_cell_length_a 5
_cell_length_b 5
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_audit_creation_method 'old method'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0 0 0
"""
    source = parse_cif_text(raw)
    first = refined_cif_text(source, {"refined_cell": dict(_cell(5), a=5.1)},
                             dataset_name="sample", software_version="23.0.0")
    parsed = parse_cif_text(first)
    assert parsed["formula"] == "Na"
    assert parsed["formula_units_z"] == 1
    assert parsed["formula_weight"] == pytest.approx(22.99)
    assert first.count("_audit_creation_method") == 1
    assert first.count("_afruz_publication_ready") == 1

    parsed["raw_cif_text"] = first
    second = refined_cif_text(parsed, {"refined_cell": parsed["cell"]},
                              dataset_name="sample-2", software_version="23.0.0")
    assert second.count("_audit_creation_method") == 1
    assert second.count("_afruz_publication_ready") == 1


def test_large_structure_validator_uses_periodic_neighbor_search():
    atoms = []
    side = 13
    for i in range(side):
        for j in range(side):
            for k in range(side):
                atoms.append(
                    {
                        "label": f"C{i}_{j}_{k}",
                        "element": "C",
                        "x": (i + 0.5) / side,
                        "y": (j + 0.5) / side,
                        "z": (k + 0.5) / side,
                        "occupancy": 1.0,
                    }
                )
    validation = validate_crystal_structure(
        {
            "data_name": "large periodic grid",
            "formula": f"C{len(atoms)}",
            "cell": _cell(130.0),
            "atoms": atoms,
        }
    )

    assert validation["atom_count"] == side**3
    assert any(
        "periodic neighbor search" in note
        for note in validation["notes"]
    )
