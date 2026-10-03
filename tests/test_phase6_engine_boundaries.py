from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from afruz_pxrd.application.project_controller import ProjectController
from afruz_pxrd.application.state import ProjectState
from afruz_pxrd.contracts import ResultContractService, ResultContractStore
from afruz_pxrd.crystallography import (
    calculate_powder_pattern,
    match_observed_to_reference,
)
from afruz_pxrd.engines import (
    ENGINE_API_VERSION,
    EngineExecutionFailure,
    EngineInputError,
    EngineRegistrationError,
    EngineRequest,
    EngineStatus,
    ScientificEngineService,
    create_default_engine_registry,
)
from afruz_pxrd.models import Dataset
from afruz_pxrd.phase_identification import ReferenceEntry, reference_from_cif_pattern
from afruz_pxrd.quantitative_phase import QPAPhaseSpec, build_reference_basis
from afruz_pxrd.validation_campaign import generate_synthetic_campaign
from afruz_pxrd.version import APP_VERSION


WAVELENGTH = 1.5406


def _dataset(uid: str = "dataset-phase6") -> Dataset:
    x = np.linspace(10.0, 70.0, 1201)
    y = (
        80.0
        + 0.25 * x
        + 900.0 * np.exp(-0.5 * ((x - 24.0) / 0.18) ** 2)
        + 1200.0 * np.exp(-0.5 * ((x - 42.5) / 0.22) ** 2)
        + 650.0 * np.exp(-0.5 * ((x - 58.0) / 0.28) ** 2)
    )
    return Dataset(uid=uid, name="Phase 6", x=x, y_raw=y)


def test_default_registry_has_one_versioned_native_engine_per_contract_kind():
    registry = create_default_engine_registry()
    descriptors = registry.descriptors()
    assert len(descriptors) == 10
    assert {row.result_kind for row in descriptors} == set(
        ResultContractService.RESULT_KINDS
    )
    assert set(registry.defaults()) == set(ResultContractService.RESULT_KINDS)
    assert all(row.api_version == ENGINE_API_VERSION for row in descriptors)
    assert all(row.identity.version == APP_VERSION for row in descriptors)
    assert all(row.required_inputs for row in descriptors)

    with pytest.raises(EngineRegistrationError, match="already registered"):
        registry.register(registry.resolve(result_kind="preprocessing"))


def test_engine_request_round_trip_is_array_safe_and_signature_is_stable():
    first = EngineRequest(
        result_kind="preprocessing",
        dataset_id="dataset-1",
        inputs={"x": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 5.0, 6.0])},
        parameters={"normalize": True},
    )
    second = EngineRequest(
        result_kind="preprocessing",
        dataset_id="dataset-1",
        inputs={"x": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 5.0, 6.0])},
        parameters={"normalize": True},
    )
    assert first.request_id != second.request_id
    assert first.input_signature == second.input_signature

    restored = EngineRequest.from_dict(first.to_dict())
    assert restored.input_signature == first.input_signature
    assert np.array_equal(restored.inputs["x"], first.inputs["x"])
    broken = first.to_dict()
    broken.pop("dataset_id")
    with pytest.raises(EngineInputError, match="missing fields.*dataset_id"):
        EngineRequest.from_dict(broken)
    tampered = first.to_dict()
    tampered["inputs"]["y"]["values"][0] = 99.0
    with pytest.raises(EngineInputError, match="signature does not match"):
        EngineRequest.from_dict(tampered)
    wrong_identity = first.to_dict()
    wrong_identity["dataset_id"] = None
    with pytest.raises(EngineInputError, match="dataset_id is required"):
        EngineRequest.from_dict(wrong_identity)

    reference = ReferenceEntry(
        uid="reference-1",
        name="Reference",
        formula="X",
        source="test",
        wavelength_angstrom=1.5406,
        peaks=[{"position": 20.0, "intensity": 100.0}],
        metadata={},
    )
    qpa_request = EngineRequest(
        result_kind="qpa",
        dataset_id="dataset-1",
        inputs={
            "x": np.array([19.0, 20.0, 21.0]),
            "y": np.array([1.0, 10.0, 1.0]),
            "phase_specs": [QPAPhaseSpec(reference)],
        },
    )
    serialized_qpa = qpa_request.to_dict()
    assert serialized_qpa["inputs"]["phase_specs"][0]["reference"]["uid"] == (
        "reference-1"
    )
    assert EngineRequest.from_dict(serialized_qpa).input_signature == (
        qpa_request.input_signature
    )


def test_project_controller_runs_native_engines_and_chains_typed_provenance():
    dataset = _dataset()
    state = ProjectState(datasets=[dataset])
    controller = ProjectController(project_state=state)
    progress = []

    first = controller.run_engine(
        "preprocessing",
        dataset.uid,
        inputs={"x": dataset.x, "y": dataset.y_raw},
        parameters={"normalize": True},
        progress_callback=lambda completed, total, message: progress.append(
            (completed, total, message)
        ),
    )
    assert first.execution.status == EngineStatus.COMPLETED
    assert first.execution.engine_id == "afruz.native.preprocessing"
    assert first.result.engine.version == APP_VERSION
    assert np.max(first.result.processed_profile) == pytest.approx(100.0)
    assert state.result_contracts.get("preprocessing", dataset.uid) is first.result
    assert progress[0][0] == 0 and progress[-1][0] == progress[-1][1]
    assert state.dirty

    previous_id = first.result.result_id
    second = controller.run_engine(
        "preprocessing",
        dataset.uid,
        inputs={"x": dataset.x, "y": dataset.y_raw},
        parameters={"normalize": False},
    )
    assert previous_id in second.result.provenance.source_result_ids
    assert second.result.input_signature != first.result.input_signature


def test_peak_detection_runs_headlessly_and_rejects_bad_inputs():
    dataset = _dataset()
    service = ScientificEngineService(
        create_default_engine_registry(),
        ResultContractStore(),
    )
    run = service.execute_kind(
        "peak_list",
        dataset.uid,
        inputs={"x": dataset.x, "y": dataset.y_raw},
        parameters={
            "method": "regular",
            "prominence_fraction": 0.03,
            "minimum_distance_points": 8,
        },
    )
    positions = [row.position_deg for row in run.result.peaks]
    for expected in (24.0, 42.5, 58.0):
        assert min(abs(value - expected) for value in positions) < 0.1

    with pytest.raises(EngineExecutionFailure) as failure:
        service.execute_kind(
            "peak_list",
            dataset.uid,
            inputs={"x": dataset.x},
        )
    assert isinstance(failure.value.cause, EngineInputError)
    assert failure.value.execution.status == EngineStatus.FAILED
    assert service.history()[-1] == failure.value.execution


def test_cancellation_is_recorded_and_never_replaces_a_result():
    dataset = _dataset()
    store = ResultContractStore()
    service = ScientificEngineService(create_default_engine_registry(), store)
    with pytest.raises(EngineExecutionFailure) as failure:
        service.execute_kind(
            "preprocessing",
            dataset.uid,
            inputs={"x": dataset.x, "y": dataset.y_raw},
            cancellation_check=lambda: True,
        )
    assert failure.value.execution.status == EngineStatus.CANCELLED
    assert store.for_dataset(dataset.uid) == {}


def test_phase_cell_and_qpa_adapters_use_existing_numerical_engines(
    nacl_structure,
    cscl_structure,
):
    registry = create_default_engine_registry()
    store = ResultContractStore()
    service = ScientificEngineService(registry, store)
    references = []
    patterns = []
    for structure in (nacl_structure, cscl_structure):
        pattern = calculate_powder_pattern(structure, WAVELENGTH, 4.0, 80.0)
        reference = reference_from_cif_pattern(structure, pattern, WAVELENGTH)
        assert reference is not None
        reference.uid = f"reference-{len(references)}"
        references.append(reference)
        patterns.append(pattern)

    observed = [
        {"position": row["two_theta"] + 0.04, "intensity": row["intensity"]}
        for row in patterns[0]
    ]
    phase_run = service.execute_kind(
        "phase_identification",
        "dataset-phase",
        inputs={"observed_peaks": observed, "references": references},
        parameters={
            "target_wavelength_angstrom": WAVELENGTH,
            "tolerance_deg": 0.2,
            "maximum_zero_shift_deg": 0.3,
        },
    )
    assert "NaCl" in phase_run.result.selected_candidate["reference_name"]

    matches = match_observed_to_reference(observed, patterns[0], tolerance_deg=0.2)
    initial_cell = deepcopy(nacl_structure["cell"])
    for key in ("a", "b", "c"):
        initial_cell[key] *= 1.01
    cell_run = service.execute_kind(
        "unit_cell_refinement",
        "dataset-phase",
        inputs={"matches": matches, "initial_cell": initial_cell},
        parameters={
            "crystal_system": "Cubic",
            "wavelength_angstrom": WAVELENGTH,
            "refine_zero_shift": True,
        },
    )
    assert cell_run.result.refined_cell["a"] == pytest.approx(5.6402, abs=1e-5)

    x = np.linspace(20.0, 75.0, 801)
    basis = build_reference_basis(x, references[0].peaks, 0.20)
    qpa_run = service.execute_kind(
        "qpa",
        "dataset-phase",
        inputs={"x": x, "y": 20.0 + 500.0 * basis, "phase_specs": [{"reference": references[0]}]},
        parameters={
            "target_wavelength_angstrom": WAVELENGTH,
            "mode": "Pattern scale fractions",
            "bootstrap_samples": 0,
        },
    )
    assert list(qpa_run.result.fractions.values()) == pytest.approx([100.0])


def test_peak_fit_instrument_and_validation_adapters_run_headlessly(
    nacl_structure,
):
    service = ScientificEngineService(
        create_default_engine_registry(),
        ResultContractStore(),
    )
    dataset = _dataset("dataset-secondary-adapters")
    peak_run = service.execute_kind(
        "peak_list",
        dataset.uid,
        inputs={"x": dataset.x, "y": dataset.y_raw},
        parameters={
            "prominence_fraction": 0.03,
            "minimum_distance_points": 8,
        },
    )
    fit_inputs = [
        {
            "position": row.position_deg,
            "intensity": row.intensity,
            "fwhm": row.fwhm_deg,
        }
        for row in peak_run.result.peaks
    ]
    fit_run = service.execute_kind(
        "peak_fitting",
        dataset.uid,
        inputs={"x": dataset.x, "y": dataset.y_raw, "peaks": fit_inputs},
        parameters={"model": "Pseudo-Voigt", "window_multiplier": 4.0},
    )
    assert fit_run.result.groups

    reference_peaks = calculate_powder_pattern(
        nacl_structure,
        WAVELENGTH,
        4.0,
        80.0,
    )
    calibration_groups = [
        {
            "group_id": index,
            "model": "Pseudo-Voigt",
            "components": [{
                "center": row["two_theta"] + 0.015,
                "center_error": 0.003,
                "fwhm": 0.075 + 0.0008 * row["two_theta"],
                "fwhm_error": 0.003,
                "amplitude": row["intensity"],
                "model": "Pseudo-Voigt",
            }],
        }
        for index, row in enumerate(reference_peaks, start=1)
    ]
    calibration_run = service.execute_kind(
        "instrument_calibration",
        dataset.uid,
        inputs={
            "fit_groups": calibration_groups,
            "reference_peaks": reference_peaks,
        },
        parameters={
            "standard_name": "Synthetic NaCl standard",
            "wavelength_angstrom": WAVELENGTH,
            "fit_zero_shift": True,
            "fit_displacement": False,
        },
    )
    assert calibration_run.result.profile["matched_reflection_count"] >= 3
    assert calibration_run.result.profile["zero_shift_deg"] == pytest.approx(
        0.015,
        abs=0.005,
    )

    validation_run = service.execute_kind(
        "validation",
        dataset.uid,
        inputs={"records": generate_synthetic_campaign(seed=1300)},
        parameters={"campaign_name": "Phase 6 adapter validation"},
    )
    assert validation_run.result.evidence
    assert validation_run.result.summary["campaign_name"] == (
        "Phase 6 adapter validation"
    )


@pytest.mark.slow
def test_whole_pattern_and_rietveld_adapters_preserve_native_baselines(
    nacl_raw_frame,
    nacl_structure,
):
    x = nacl_raw_frame["two_theta_deg"].to_numpy(dtype=float)
    y = nacl_raw_frame["intensity_counts"].to_numpy(dtype=float)
    sigma = nacl_raw_frame["sigma_counts"].to_numpy(dtype=float)
    pattern = calculate_powder_pattern(nacl_structure, WAVELENGTH, 4.0, 80.0)
    reference = reference_from_cif_pattern(
        nacl_structure,
        pattern,
        WAVELENGTH,
    )
    assert reference is not None
    service = ScientificEngineService(
        create_default_engine_registry(),
        ResultContractStore(),
    )

    whole = service.execute_kind(
        "whole_pattern_refinement",
        "dataset-refinement-adapters",
        inputs={
            "x": x,
            "y": y,
            "phase_specs": [{"reference": reference, "refine_cell": True}],
        },
        parameters={
            "mode": "Pawley decomposition",
            "wavelength_angstrom": WAVELENGTH,
            "weighting": "Poisson-like",
            "observed_sigma": sigma,
            "intensity_provenance": "raw_counts",
            "extraction_cycles": 6,
            "maximum_nonlinear_evaluations": 40,
            "maximum_optimization_points": 1200,
        },
    )
    assert whole.result.statistics["rwp_percent"] == pytest.approx(
        6.7948477025,
        abs=0.10,
    )

    rietveld = service.execute_kind(
        "rietveld_refinement",
        "dataset-refinement-adapters",
        inputs={
            "x": x,
            "y": y,
            "phase_specs": [{
                "structure": nacl_structure,
                "name": "NaCl",
                "refine_cell": True,
            }],
        },
        parameters={
            "wavelength_angstrom": WAVELENGTH,
            "weighting": "Poisson-like",
            "observed_sigma": sigma,
            "intensity_provenance": "raw_counts",
            "maximum_nonlinear_evaluations": 120,
            "maximum_optimization_points": 1800,
        },
    )
    assert rietveld.result.statistics["rwp_percent"] == pytest.approx(
        10.1676404408,
        abs=0.05,
    )
    assert rietveld.result.phases[0]["phase_name"] == "NaCl"


def test_new_project_rebinds_engine_service_to_the_new_result_store():
    dataset = _dataset()
    controller = ProjectController(project_state=ProjectState(datasets=[dataset]))
    old_store = controller.state.result_contracts
    controller.new_project()
    assert controller.engine_service.result_store is controller.state.result_contracts
    assert controller.engine_service.result_store is not old_store
    assert controller.state.result_contracts.to_dict()["results"] == {}


def test_headless_engine_result_survives_compatibility_sync_and_afz_round_trip(
    tmp_path,
):
    dataset = _dataset()
    controller = ProjectController(project_state=ProjectState(datasets=[dataset]))
    run = controller.run_engine(
        "preprocessing",
        dataset.uid,
        inputs={"x": dataset.x, "y": dataset.y_raw},
        parameters={"normalize": True},
    )
    ResultContractService.capture_dataset(
        controller.state.result_contracts,
        dataset.uid,
        {},
    )
    assert (
        controller.state.result_contracts.get("preprocessing", dataset.uid).result_id
        == run.result.result_id
    )

    path = controller.save(tmp_path / "headless-engine", {}, {})
    reopened = ProjectController()
    opened = reopened.open(path)
    assert "typed_result_contracts" in opened.analysis_state
    restored = reopened.state.result_contracts.get("preprocessing", dataset.uid)
    assert restored.result_id == run.result.result_id
    assert np.array_equal(restored.processed_profile, run.result.processed_profile)
