from copy import deepcopy
from pathlib import Path

import pytest

from afruz_pxrd.crystal_intelligence import (
    SMART_SCHEMA,
    analyze_crystal_model,
    format_smart_crystal_report,
)
from afruz_pxrd.crystallography import load_cif


def _cell(a=20.0, b=20.0, c=20.0):
    return dict(a=a, b=b, c=c, alpha=90.0, beta=90.0, gamma=90.0)


def test_smart_view_identifies_a_finite_molecular_crystal_without_mutation():
    model = {
        "data_name": "finite molecule",
        "cell": _cell(),
        "atoms": [
            {"label": "C1", "element": "C", "x": .45, "y": .5, "z": .5, "occupancy": 1},
            {"label": "O1", "element": "O", "x": .51, "y": .5, "z": .5, "occupancy": 1},
        ],
    }
    frozen = deepcopy(model)
    report = analyze_crystal_model(model)
    assert report["schema"] == SMART_SCHEMA
    assert report["classification"]["periodic_dimension"] == 0
    assert report["classification"]["family"] == "Molecular crystal"
    assert report["recommendation"]["style"] == "Publication • ivory"
    assert model == frozen


def test_smart_view_detects_a_two_dimensional_carbon_network():
    model = {
        "data_name": "synthetic sheet",
        "cell": _cell(1.4, 1.4, 10.0),
        "atoms": [
            {"label": "C1", "element": "C", "x": 0, "y": 0, "z": .5, "occupancy": 1},
        ],
    }
    report = analyze_crystal_model(model)
    assert report["classification"]["periodic_dimension"] == 2
    assert report["classification"]["family"].startswith("Covalent organic framework")
    assert report["recommendation"]["repeats"] == [2, 2, 1]
    assert "two-dimensional periodic plane" in report["recommendation"]["camera_reason"]


def test_smart_view_classifies_synthetic_periodic_fixture_as_metal_organic_framework():
    fixture = Path(__file__).parent / "fixtures" / "synthetic_cr_organic_framework.cif"
    model = load_cif(fixture)
    report = analyze_crystal_model(model)
    assert report["classification"]["family"].startswith("Metal–organic framework")
    assert report["classification"]["periodic_dimension"] == 3
    assert report["classification"]["porosity_candidate"]
    assert report["recommendation"]["style"] == "MOF • porous framework"
    assert report["recommendation"]["polyhedra"]
    rendered = format_smart_crystal_report(report)
    assert "Evidence:" in rendered
    assert "Scientific limits:" in rendered
    assert "never the CIF or refinement result" in rendered


def test_large_model_uses_bounded_analysis_and_gpu_first_recommendation():
    model = {
        "data_name": "large framework",
        "cell": _cell(80, 80, 80),
        "atoms": [
            {
                "label": f"C{index}", "element": "C",
                "x": (index % 97) / 97, "y": (index % 89) / 89,
                "z": (index % 83) / 83, "occupancy": 1,
            }
            for index in range(6001)
        ],
    }
    report = analyze_crystal_model(model)
    assert report["classification"]["periodic_dimension"] is None
    assert report["recommendation"]["gpu_first"]
    assert any("6,000-site advisory limit" in item for item in report["limitations"])


def test_unknown_element_receives_an_explicit_scientific_limitation():
    model = {
        "data_name": "unknown site",
        "cell": _cell(),
        "atoms": [
            {"label": "Q1", "element": "X", "x": .5, "y": .5, "z": .5, "occupancy": .5},
        ],
    }
    report = analyze_crystal_model(model)
    assert report["composition"]["unknown_element_site_count"] == 1
    assert any("unknown element symbol X" in item for item in report["limitations"])
    assert any("partially occupied" in item for item in report["limitations"])
