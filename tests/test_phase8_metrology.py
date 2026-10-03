from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from afruz_pxrd.application import ProjectController
from afruz_pxrd.contracts import EngineIdentity
from afruz_pxrd.metrology import (
    AcceptanceCriterion,
    AcquisitionMetadata,
    AssessmentStatus,
    ClaimScope,
    EvidenceClass,
    MetrologyRegistry,
    MetrologyValidationError,
    MetricObservation,
    NIST_SRM_640G,
    NIST_SRM_660C,
    StandardCatalogue,
    dataset_data_signature,
)
from afruz_pxrd.models import Dataset
from afruz_pxrd.validation_campaign import (
    analyze_validation_campaign,
    generate_synthetic_campaign,
)


def _dataset(uid: str = "phase8-dataset") -> Dataset:
    return Dataset(
        uid=uid,
        name="NIST standard scan",
        x=np.linspace(20.0, 100.0, 81),
        y_raw=np.linspace(100.0, 180.0, 81),
        metadata={},
    )


def _acquisition(dataset: Dataset, *, temperature_c: float = 22.5) -> AcquisitionMetadata:
    return AcquisitionMetadata(
        instrument_id="LAB-XRD-01",
        instrument_name="Laboratory diffractometer",
        geometry="Bragg-Brentano symmetric",
        radiation="Cu Kalpha1",
        wavelength_angstrom=1.5405929,
        detector="Silicon strip detector",
        optics="Fixed divergence and receiving slits",
        step_size_deg=0.02,
        counting_time_seconds=1.5,
        temperature_c=temperature_c,
        specimen_preparation="Back-loaded and rotated powder",
        specimen_geometry="Flat plate",
        operator="Phase 8 test operator",
        acquired_at="2026-08-11T10:00:00+00:00",
        source_data_signature=dataset_data_signature(dataset.x, dataset.y_raw),
    )


def _lattice_observation(value: float = 5.431120) -> MetricObservation:
    return MetricObservation(
        metric="lattice_parameter_a",
        value=value,
        unit="angstrom",
        standard_uncertainty=0.000050,
        method="Weighted cubic unit-cell refinement",
        source_result_id="cell-result-1",
    )


def _lattice_criterion(tolerance: float = 0.00020) -> AcceptanceCriterion:
    return AcceptanceCriterion(
        metric="lattice_parameter_a",
        absolute_tolerance=tolerance,
        maximum_en=1.0,
        require_measurement_uncertainty=True,
        notes="Pre-registered Phase 8 acceptance rule",
    )


def test_builtin_standard_catalog_preserves_certified_status_and_units():
    catalogue = StandardCatalogue()
    silicon = catalogue.get("NIST-SRM-640g")
    lab6 = catalogue.get("NIST-SRM-660c")

    assert silicon.property("lattice_parameter_a").value == pytest.approx(5.431109)
    assert silicon.property("lattice_parameter_a").expanded_uncertainty == pytest.approx(
        0.000080
    )
    assert silicon.property("lattice_parameter_a").coverage_factor == 2.0
    assert lab6.property("lattice_parameter_a").value == pytest.approx(4.156826)
    assert "non-certified" in " ".join(silicon.notes).lower()

    qpa_set = catalogue.get("NIST-SRM-674b")
    with pytest.raises(MetrologyValidationError, match="has no property"):
        qpa_set.property("phase_purity")


def test_acquisition_data_signature_detects_wrong_source_arrays():
    dataset = _dataset()
    controller = ProjectController()
    controller.add_datasets([dataset])
    acquisition = _acquisition(dataset)

    controller.set_acquisition_metadata(dataset.uid, acquisition)
    assert dataset.metadata["acquisition_metadata"]["instrument_id"] == "LAB-XRD-01"

    changed = _dataset("changed")
    changed.y_raw = changed.y_raw + 1.0
    wrong = _acquisition(changed)
    with pytest.raises(MetrologyValidationError, match="does not match"):
        controller.set_acquisition_metadata(dataset.uid, wrong)


def test_certified_assessment_authorizes_only_supported_claim_scopes():
    dataset = _dataset()
    controller = ProjectController()
    controller.add_datasets([dataset])
    controller.set_acquisition_metadata(dataset.uid, _acquisition(dataset))

    assessment = controller.assess_certified_standard(
        dataset.uid,
        NIST_SRM_640G.standard_id,
        [_lattice_observation()],
        [_lattice_criterion()],
        material_unit_id="SRM-640g bottle test-001 / current certificate",
    )

    assert assessment.status == AssessmentStatus.PASS
    assert assessment.evaluations[0].passed is True
    assert assessment.evaluations[0].en_number is not None
    assert assessment.evaluations[0].en_number < 1.0
    assert ClaimScope.MEASUREMENT_ACCURACY in assessment.allowed_claim_scopes

    claim = controller.authorize_scientific_claim(
        assessment.assessment_id,
        ClaimScope.MEASUREMENT_ACCURACY,
        "Cubic lattice parameter agrees with the certified value under the stated rule.",
    )
    assert claim.to_dict()["status"] == "Validated"

    with pytest.raises(MetrologyValidationError, match="cannot validate"):
        controller.authorize_scientific_claim(
            assessment.assessment_id,
            ClaimScope.ENGINE_AGREEMENT,
            "Unsupported scope",
        )


def test_failed_or_temperature_mismatched_assessment_cannot_validate_claim():
    dataset = _dataset()
    controller = ProjectController()
    controller.add_datasets([dataset])
    controller.set_acquisition_metadata(dataset.uid, _acquisition(dataset))
    failed = controller.assess_certified_standard(
        dataset.uid,
        NIST_SRM_640G.standard_id,
        [_lattice_observation(5.435)],
        [_lattice_criterion()],
        material_unit_id="SRM unit A",
    )
    assert failed.status == AssessmentStatus.FAIL
    with pytest.raises(MetrologyValidationError, match="did not pass"):
        controller.authorize_scientific_claim(
            failed.assessment_id,
            ClaimScope.MEASUREMENT_ACCURACY,
            "Must be rejected",
        )

    missing_uncertainty = controller.assess_certified_standard(
        dataset.uid,
        NIST_SRM_640G.standard_id,
        [MetricObservation("lattice_parameter_a", 5.43111, "angstrom")],
        [_lattice_criterion()],
        material_unit_id="SRM unit A",
    )
    assert missing_uncertainty.status == AssessmentStatus.FAIL
    assert any(
        row["name"] == "measurement_uncertainty_available" and not row["passed"]
        for row in missing_uncertainty.evaluations[0].checks
    )

    controller.set_acquisition_metadata(dataset.uid, _acquisition(dataset, temperature_c=27.0))
    review = controller.assess_certified_standard(
        dataset.uid,
        NIST_SRM_640G.standard_id,
        [_lattice_observation()],
        [_lattice_criterion()],
        material_unit_id="SRM unit A",
    )
    assert review.status == AssessmentStatus.REVIEW
    assert any("temperature" in row.lower() for row in review.warnings)


def test_reference_engine_comparison_is_versioned_and_not_accuracy_evidence():
    dataset = _dataset()
    controller = ProjectController()
    controller.add_datasets([dataset])

    comparison = controller.compare_reference_engine(
        dataset.uid,
        EngineIdentity("afruz.native.unit_cell", "22.6.0"),
        [_lattice_observation(5.43112)],
        EngineIdentity("GSAS-II", "reference-build-2026-08"),
        [
            MetricObservation(
                "lattice_parameter_a",
                5.43110,
                "angstrom",
                standard_uncertainty=0.00004,
            )
        ],
        [AcceptanceCriterion("lattice_parameter_a", absolute_tolerance=0.0001)],
        native_input_signature="shared-signed-input",
        reference_input_signature="shared-signed-input",
    )

    assert comparison.passed
    assert comparison.evidence_class == EvidenceClass.REFERENCE_ENGINE
    assert comparison.native_engine["version"] == "22.6.0"
    assert comparison.reference_engine["name"] == "GSAS-II"
    assert comparison.input_signature == "shared-signed-input"
    assert comparison.allowed_claim_scopes == (ClaimScope.ENGINE_AGREEMENT,)
    with pytest.raises(MetrologyValidationError, match="cannot validate"):
        controller.authorize_scientific_claim(
            comparison.assessment_id,
            ClaimScope.MEASUREMENT_ACCURACY,
            "Engine agreement is not measurement accuracy",
        )

    with pytest.raises(MetrologyValidationError, match="inputs differ"):
        controller.compare_reference_engine(
            dataset.uid,
            EngineIdentity("afruz.native.unit_cell", "22.6.0"),
            [_lattice_observation()],
            EngineIdentity("GSAS-II", "reference-build-2026-08"),
            [_lattice_observation()],
            [AcceptanceCriterion("lattice_parameter_a", absolute_tolerance=0.0001)],
            native_input_signature="native-input",
            reference_input_signature="different-reference-input",
        )


def test_acquisition_change_invalidates_stale_assessments_and_claims():
    dataset = _dataset()
    controller = ProjectController()
    controller.add_datasets([dataset])
    controller.set_acquisition_metadata(dataset.uid, _acquisition(dataset))
    assessment = controller.assess_certified_standard(
        dataset.uid,
        NIST_SRM_640G.standard_id,
        [_lattice_observation()],
        [_lattice_criterion()],
        material_unit_id="SRM unit B",
    )
    controller.authorize_scientific_claim(
        assessment.assessment_id,
        ClaimScope.INSTRUMENT_PERFORMANCE,
        "Instrument line-position performance passed.",
    )

    controller.set_acquisition_metadata(
        dataset.uid,
        AcquisitionMetadata.from_dict(
            {
                **_acquisition(dataset).to_dict(),
                "operator": "Independent operator",
            }
        ),
    )
    assert controller.state.metrology_registry.for_dataset(dataset.uid) == ()
    assert controller.state.metrology_registry.claims_for_dataset(dataset.uid) == ()


def test_metrology_registry_round_trips_through_afz_without_copying_claims(
    tmp_path,
):
    dataset = _dataset()
    controller = ProjectController()
    controller.add_datasets([dataset])
    controller.set_acquisition_metadata(dataset.uid, _acquisition(dataset))
    assessment = controller.assess_certified_standard(
        dataset.uid,
        NIST_SRM_640G.standard_id,
        [_lattice_observation()],
        [_lattice_criterion()],
        material_unit_id="SRM unit C",
    )
    controller.authorize_scientific_claim(
        assessment.assessment_id,
        ClaimScope.MEASUREMENT_ACCURACY,
        "Certified lattice-parameter agreement.",
    )

    path = controller.save(tmp_path / "phase8.afz", {}, {})
    reopened = ProjectController()
    reopened.open(path)
    reopened_registry = reopened.state.metrology_registry
    assert len(reopened_registry.for_dataset(dataset.uid)) == 1
    assert len(reopened_registry.claims_for_dataset(dataset.uid)) == 1
    assert reopened.metrology_service.registry is reopened_registry

    clone = reopened.duplicate_dataset(dataset.uid)
    assert reopened_registry.for_dataset(clone.uid) == ()
    assert reopened_registry.claims_for_dataset(clone.uid) == ()
    assert "acquisition_metadata" not in clone.metadata


def test_corrupt_or_stale_claim_registry_is_rejected():
    dataset = _dataset()
    controller = ProjectController()
    controller.add_datasets([dataset])
    controller.set_acquisition_metadata(dataset.uid, _acquisition(dataset))
    assessment = controller.assess_certified_standard(
        dataset.uid,
        NIST_SRM_640G.standard_id,
        [_lattice_observation()],
        [_lattice_criterion()],
        material_unit_id="SRM unit D",
    )
    claim = controller.authorize_scientific_claim(
        assessment.assessment_id,
        ClaimScope.MEASUREMENT_ACCURACY,
        "Certified lattice-parameter agreement.",
    )
    payload = controller.state.metrology_registry.to_dict()
    forged_pass = deepcopy(payload)
    forged_pass["assessments"][assessment.assessment_id]["evaluations"][0]["checks"][0][
        "passed"
    ] = False
    with pytest.raises(MetrologyValidationError, match="pass flag is inconsistent"):
        MetrologyRegistry.from_dict(forged_pass)

    corrupt = deepcopy(payload)
    corrupt["claims"][claim.claim_id]["assessment_fingerprint"] = "0" * 64

    with pytest.raises(MetrologyValidationError, match="stale"):
        MetrologyRegistry.from_dict(corrupt)


def test_validation_campaign_separates_regression_from_publication_evidence():
    synthetic = analyze_validation_campaign(generate_synthetic_campaign(seed=1300))
    assert synthetic["evidence_classification"] == EvidenceClass.SOFTWARE_REGRESSION.value
    assert synthetic["claim_eligible"] is False
    assert synthetic["scientific_claim_status"].startswith("Not validated")

    experimental_records = generate_synthetic_campaign(seed=1300)
    for row in experimental_records:
        row["replica_type"] = "Independent preparation"
        row["classification"] = "Gravimetric reference mixture"
    experimental = analyze_validation_campaign(
        experimental_records,
        evidence_classification=EvidenceClass.EXPERIMENTAL_VALIDATION.value,
        metrology_assessment_ids=["passing-assessment-id"],
    )
    assert experimental["status"] == "Pass"
    assert experimental["claim_eligible"] is True


def test_catalog_constants_are_immutable_scientific_snapshots():
    assert NIST_SRM_640G.standard_id != NIST_SRM_660C.standard_id
    snapshot = NIST_SRM_640G.to_dict()
    snapshot["properties"]["lattice_parameter_a"]["value"] = 0.0
    assert NIST_SRM_640G.property("lattice_parameter_a").value == pytest.approx(5.431109)
