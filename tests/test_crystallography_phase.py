from __future__ import annotations

import numpy as np
import pytest

import afruz_pxrd.crystallography as crystallography
from afruz_pxrd.crystallography import (
    CIFImportError,
    _structure_factor_intensity,
    _structure_factor_intensities_batch,
    calculate_powder_pattern,
    d_spacing,
    load_cif,
    match_observed_to_reference,
    parse_cif_text,
    refine_unit_cell,
)
from afruz_pxrd.phase_identification import identify_phases, reference_from_cif_pattern


WAVELENGTH = 1.5406


CROMER_MANN_CIF = """data_scattering
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_pd_proc_wavelength 1.250071
loop_
_atom_type_symbol
_atom_type_scat_Cromer_Mann_a1
_atom_type_scat_Cromer_Mann_b1
_atom_type_scat_Cromer_Mann_a2
_atom_type_scat_Cromer_Mann_b2
_atom_type_scat_Cromer_Mann_a3
_atom_type_scat_Cromer_Mann_b3
_atom_type_scat_Cromer_Mann_a4
_atom_type_scat_Cromer_Mann_b4
_atom_type_scat_Cromer_Mann_c
_atom_type_scat_dispersion_real
_atom_type_scat_dispersion_imag
C 2.31 20.8439 1.02 10.2075 1.5886 0.5687 0.865 51.6512 0.2156 0.002 0.003
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0 0 0
"""


def test_cif_cromer_mann_coefficients_drive_cpu_and_batch_structure_factors():
    structure = parse_cif_text(CROMER_MANN_CIF)
    factors = structure["xray_scattering_factors"]
    assert factors["C"]["a"] == pytest.approx([2.31, 1.02, 1.5886, 0.865])
    assert structure["reported_wavelength_angstrom"] == pytest.approx(1.250071)
    assert "Cromer–Mann" in structure["intensity_model"]

    hkl = (1, 0, 0)
    spacing = d_spacing(structure["cell"], hkl)
    two_theta = 2.0 * np.degrees(np.arcsin(WAVELENGTH / (2.0 * spacing)))
    scalar = _structure_factor_intensity(
        structure["atoms"], hkl, spacing, two_theta, WAVELENGTH, factors
    )
    batch, _ = _structure_factor_intensities_batch(
        structure["atoms"],
        [hkl],
        [spacing],
        [two_theta],
        scattering_factors=factors,
    )

    s2 = 1.0 / (4.0 * spacing * spacing)
    carbon = factors["C"]
    f0 = carbon["c"] + sum(
        a * np.exp(-b * s2) for a, b in zip(carbon["a"], carbon["b"])
    )
    amplitude_squared = (f0 + carbon["dispersion_real"]) ** 2 + carbon[
        "dispersion_imag"
    ] ** 2
    theta = np.radians(two_theta / 2.0)
    lp = (1.0 + np.cos(np.radians(two_theta)) ** 2) / (
        np.sin(theta) ** 2 * np.cos(theta)
    )
    assert scalar == pytest.approx(amplitude_squared * lp)
    assert batch[0] == pytest.approx(scalar)


def test_saved_legacy_structure_rehydrates_cif_scattering_metadata():
    structure = parse_cif_text(CROMER_MANN_CIF)
    structure.pop("xray_scattering_factors")
    structure.pop("reported_wavelength_angstrom")

    calculate_powder_pattern(structure, WAVELENGTH, 4.0, 30.0, 0.0)

    assert set(structure["xray_scattering_factors"]) == {"C"}
    assert structure["reported_wavelength_angstrom"] == pytest.approx(1.250071)


def test_large_fd3m_pattern_is_not_silently_truncated_at_index_40(monkeypatch):
    def unit_intensities(atoms, hkls, d_values, two_theta_values, **kwargs):
        hkls = list(hkls)
        return np.ones(len(hkls), dtype=float), {"backend": "test", "used": False}

    monkeypatch.setattr(
        crystallography,
        "_structure_factor_intensities_batch",
        unit_intensities,
    )
    atom = {
        "label": "C1",
        "element": "C",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "occupancy": 1.0,
        "b_iso": 0.0,
    }
    structure = {
        "cell": {
            "a": 88.86899,
            "b": 88.86899,
            "c": 88.86899,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        "space_group": "F d -3 m",
        "crystal_system": "Cubic",
        "atoms": [atom] * 2001,
    }

    pattern = calculate_powder_pattern(
        structure,
        WAVELENGTH,
        1.0,
        120.0,
        intensity_cutoff_percent=0.0,
    )

    assert pattern[-1]["two_theta"] > 119.9
    assert max(max(row["hkl"]) for row in pattern) > 40


@pytest.mark.parametrize(
    "fixture_name,expected_a,minimum_reflections",
    (("nacl_structure", 5.6402, 8), ("cscl_structure", 4.123, 10)),
)
def test_supplied_cifs_parse_and_calculate_patterns(
    request, fixture_name, expected_a, minimum_reflections
):
    structure = request.getfixturevalue(fixture_name)
    assert structure["crystal_system"] == "Cubic"
    assert structure["cell"]["a"] == pytest.approx(expected_a, abs=1e-4)
    assert structure["atoms"]

    pattern = calculate_powder_pattern(structure, WAVELENGTH, 4.0, 80.0)
    assert len(pattern) >= minimum_reflections
    assert all(4.0 <= row["two_theta"] <= 80.0 for row in pattern)
    assert max(row["intensity"] for row in pattern) == pytest.approx(100.0)
    assert all(
        sum(row["equivalent_multiplicities"]) == row["multiplicity_count"]
        for row in pattern
    )


def test_malformed_cif_is_rejected():
    with pytest.raises(CIFImportError, match="required unit-cell"):
        parse_cif_text("data_broken\n_cell_length_a not-a-number\n")


def test_cif_tokenizer_accepts_backslash_escaped_bibliographic_apostrophe():
    structure = parse_cif_text(
        """data_quote
loop_
_publ_author_name
'F\\'erey, G.'
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
Cr1 Cr 0 0 0
"""
    )
    assert structure["atoms"][0]["element"] == "Cr"


def test_cif_rounding_noise_is_normalized_and_special_position_is_deduplicated():
    structure = parse_cif_text(
        """data_rounded_special_position
_cell_length_a 88.86899
_cell_length_b 88.86899
_cell_length_c 88.86899
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_symmetry_equiv_pos_as_xyz
'x,y,z'
'y,x,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
O1 O 0.07916 0.07915 0.17085 1.00002
"""
    )

    # The two operations describe one rounded special-position site, not two
    # oxygen atoms separated by an impossible sub-picometre distance.
    assert len(structure["atoms"]) == 1
    assert structure["atoms"][0]["occupancy"] == 1.0


def test_cif_parser_preserves_partial_charge_and_geometric_bond_evidence():
    structure = parse_cif_text(
        """data_bond_metadata
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
_atom_site_charge
C1 C 0.10 0.10 0.10 0.25
C2 C 0.40 0.10 0.10 -0.25
loop_
_geom_bond_atom_site_label_1
_geom_bond_atom_site_label_2
_geom_bond_distance
_geom_bond_site_symmetry_1
_geom_bond_site_symmetry_2
_geom_bond_type
C1 C2 3.000(5) . 2_565 S
"""
    )
    assert structure["atoms"][0]["charge"] == pytest.approx(.25)
    assert structure["cif_bonds"] == [{
        "label_1": "C1", "label_2": "C2", "distance": 3.0,
        "symmetry_1": ".", "symmetry_2": "2_565", "type": "S",
    }]


def test_phase_identification_recovers_nacl_and_zero_shift(nacl_structure, cscl_structure):
    references = []
    for structure in (nacl_structure, cscl_structure):
        pattern = calculate_powder_pattern(structure, WAVELENGTH, 4.0, 80.0)
        reference = reference_from_cif_pattern(structure, pattern, WAVELENGTH)
        assert reference is not None
        reference.uid = f"reference-{len(references)}"
        references.append(reference)

    observed = [
        {"position": row["two_theta"] + 0.04, "intensity": row["intensity"]}
        for row in references[0].peaks
        if row["intensity"] >= 1.0
    ]
    result = identify_phases(
        observed,
        references,
        WAVELENGTH,
        tolerance_deg=0.2,
        maximum_zero_shift_deg=0.3,
    )

    assert "NaCl" in result["best"]["reference_name"]
    assert result["best"]["matched_count"] == 8
    assert result["best"]["score"] == pytest.approx(100.0, abs=1e-6)
    assert result["best"]["zero_shift_deg"] == pytest.approx(0.04, abs=1e-6)


def test_unit_cell_refinement_recovers_known_cell_and_shift(nacl_structure):
    reference = calculate_powder_pattern(nacl_structure, WAVELENGTH, 4.0, 80.0)
    observed = [
        {"position": row["two_theta"] + 0.025, "intensity": row["intensity"]}
        for row in reference
    ]
    matches = match_observed_to_reference(observed, reference, tolerance_deg=0.2)
    initial_cell = dict(nacl_structure["cell"])
    initial_cell["a"] *= 1.01
    initial_cell["b"] *= 1.01
    initial_cell["c"] *= 1.01

    result = refine_unit_cell(
        matches,
        initial_cell,
        "Cubic",
        WAVELENGTH,
        refine_zero_shift=True,
    )
    assert result["success"]
    assert result["refined_cell"]["a"] == pytest.approx(5.6402, abs=1e-5)
    assert result["zero_shift_deg"] == pytest.approx(0.025, abs=1e-5)
