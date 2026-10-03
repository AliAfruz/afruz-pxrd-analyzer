from copy import deepcopy

import pytest

from afruz_pxrd.rietveld_refinement import (
    RietveldError,
    RietveldPhaseSpec,
    validate_phase_specs_for_refinement,
)
from afruz_pxrd.structure_validation import (
    composition_formula_from_sites,
    structure_refinement_gate,
    validate_crystal_structure,
)


def test_weak_or_invalid_structure_requires_explicit_refinement_override(nacl_structure):
    invalid = deepcopy(nacl_structure)
    invalid["atoms"].append(deepcopy(invalid["atoms"][0]))
    invalid["atoms"][-1]["label"] = "duplicate"
    validation = validate_crystal_structure(invalid)
    gate = structure_refinement_gate(validation)

    assert not gate["passed"]
    assert not gate["publication_ready"]
    assert validation["duplicate_pair_count"] >= 1
    with pytest.raises(RietveldError, match="blocked before refinement"):
        validate_phase_specs_for_refinement([RietveldPhaseSpec(invalid, name="bad CIF")])

    diagnostics = validate_phase_specs_for_refinement([
        RietveldPhaseSpec(invalid, name="bad CIF", allow_flagged_structure=True)
    ])
    assert diagnostics[0]["gate"]["expert_override_required"]


def test_clean_structure_is_refinement_and_publication_ready(nacl_structure):
    structure = deepcopy(nacl_structure)
    structure["formula"] = composition_formula_from_sites(structure["atoms"])
    gate = structure_refinement_gate(validate_crystal_structure(structure))
    assert gate["passed"]
    assert gate["publication_ready"]
