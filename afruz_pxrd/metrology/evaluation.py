from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

import numpy as np

from ..contracts import EngineIdentity
from .models import (
    AcceptanceCriterion,
    AcquisitionMetadata,
    AssessmentStatus,
    CertifiedStandard,
    ClaimScope,
    EvidenceClass,
    MetrologyAssessment,
    MetrologyValidationError,
    MetricEvaluation,
    MetricObservation,
    ScientificClaim,
)


_UNIT_ALIASES = {
    "a": "angstrom",
    "å": "angstrom",
    "ångström": "angstrom",
    "angstroms": "angstrom",
    "nanometer": "nm",
    "nanometers": "nm",
    "degree": "deg",
    "degrees": "deg",
}


def dataset_data_signature(x: np.ndarray, y: np.ndarray) -> str:
    """Return a stable identity for the exact numerical data used by an assessment."""
    digest = hashlib.sha256()
    for name, value in (("x", x), ("y", y)):
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
            raise MetrologyValidationError(f"{name} must be a finite one-dimensional array")
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    if len(np.asarray(x)) != len(np.asarray(y)):
        raise MetrologyValidationError("x and y must have equal lengths")
    return digest.hexdigest()


def _unit(value: str) -> str:
    normalized = str(value).strip().casefold()
    return _UNIT_ALIASES.get(normalized, normalized)


def _convert(value: float, source_unit: str, target_unit: str) -> float:
    source, target = _unit(source_unit), _unit(target_unit)
    if source == target:
        return float(value)
    if source == "nm" and target == "angstrom":
        return 10.0 * float(value)
    if source == "angstrom" and target == "nm":
        return 0.1 * float(value)
    raise MetrologyValidationError(
        f"incompatible units: '{source_unit}' and '{target_unit}'"
    )


def _unique_by_metric(values: Iterable, label: str) -> dict[str, object]:
    output: dict[str, object] = {}
    for value in values:
        metric = str(value.metric)
        if metric in output:
            raise MetrologyValidationError(f"duplicate {label} metric: {metric}")
        output[metric] = value
    if not output:
        raise MetrologyValidationError(f"at least one {label} is required")
    return output


def evaluate_metric(
    observation: MetricObservation,
    reference: MetricObservation,
    criterion: AcceptanceCriterion,
) -> MetricEvaluation:
    if observation.metric != reference.metric or observation.metric != criterion.metric:
        raise MetrologyValidationError("observation, reference and criterion metrics must match")
    reference_value = _convert(reference.value, reference.unit, observation.unit)
    reference_uncertainty = (
        None
        if reference.standard_uncertainty is None
        else abs(_convert(reference.standard_uncertainty, reference.unit, observation.unit))
    )
    signed_error = observation.value - reference_value
    absolute_error = abs(signed_error)
    relative_error = (
        None
        if abs(reference_value) <= np.finfo(float).eps
        else 100.0 * signed_error / abs(reference_value)
    )

    combined_standard = None
    en_number = None
    if observation.standard_uncertainty is not None and reference_uncertainty is not None:
        combined_standard = float(
            np.hypot(observation.standard_uncertainty, reference_uncertainty)
        )
        if combined_standard > 0:
            # En convention uses expanded uncertainties. Both input uncertainties are
            # standard uncertainties, so k=2 is applied to the combined denominator.
            en_number = absolute_error / (2.0 * combined_standard)

    checks: list[dict] = []
    if criterion.absolute_tolerance is not None:
        checks.append(
            {
                "name": "absolute_error",
                "value": absolute_error,
                "limit": criterion.absolute_tolerance,
                "passed": absolute_error <= criterion.absolute_tolerance,
            }
        )
    if criterion.relative_tolerance_percent is not None:
        relative_magnitude = None if relative_error is None else abs(relative_error)
        checks.append(
            {
                "name": "absolute_relative_error_percent",
                "value": relative_magnitude,
                "limit": criterion.relative_tolerance_percent,
                "passed": (
                    relative_magnitude is not None
                    and relative_magnitude <= criterion.relative_tolerance_percent
                ),
            }
        )
    if criterion.maximum_en is not None:
        checks.append(
            {
                "name": "en_number",
                "value": en_number,
                "limit": criterion.maximum_en,
                "passed": en_number is not None and en_number <= criterion.maximum_en,
            }
        )
    if criterion.require_measurement_uncertainty:
        checks.append(
            {
                "name": "measurement_uncertainty_available",
                "value": observation.standard_uncertainty,
                "limit": "required",
                "passed": observation.standard_uncertainty is not None,
            }
        )
    return MetricEvaluation(
        metric=observation.metric,
        measured_value=observation.value,
        reference_value=reference_value,
        unit=observation.unit,
        signed_error=signed_error,
        absolute_error=absolute_error,
        relative_error_percent=relative_error,
        en_number=en_number,
        passed=all(bool(row["passed"]) for row in checks),
        checks=tuple(checks),
        measurement_standard_uncertainty=observation.standard_uncertainty,
        reference_standard_uncertainty=reference_uncertainty,
    )


def _evaluate_sets(
    observations: Iterable[MetricObservation],
    references: Iterable[MetricObservation],
    criteria: Iterable[AcceptanceCriterion],
) -> tuple[tuple[MetricEvaluation, ...], tuple[AcceptanceCriterion, ...]]:
    observation_map = _unique_by_metric(observations, "observation")
    reference_map = _unique_by_metric(references, "reference")
    criterion_map = _unique_by_metric(criteria, "criterion")
    metrics = set(observation_map)
    if metrics != set(reference_map) or metrics != set(criterion_map):
        raise MetrologyValidationError(
            "observations, references and criteria must cover the same metric IDs"
        )
    ordered = tuple(str(metric) for metric in sorted(metrics))
    evaluations = tuple(
        evaluate_metric(
            observation_map[metric],
            reference_map[metric],
            criterion_map[metric],
        )
        for metric in ordered
    )
    return evaluations, tuple(criterion_map[metric] for metric in ordered)


def assess_certified_standard(
    dataset_id: str,
    standard: CertifiedStandard,
    observations: Iterable[MetricObservation],
    criteria: Iterable[AcceptanceCriterion],
    acquisition: AcquisitionMetadata,
    *,
    material_unit_id: str,
    title: str | None = None,
) -> MetrologyAssessment:
    if not isinstance(standard, CertifiedStandard):
        raise MetrologyValidationError("standard must be a CertifiedStandard")
    if not isinstance(acquisition, AcquisitionMetadata):
        raise MetrologyValidationError(
            "certified-reference assessment requires complete acquisition metadata"
        )
    unit_id = str(material_unit_id).strip()
    if not unit_id:
        raise MetrologyValidationError(
            "physical reference-material unit or certificate identity is required"
        )
    observation_rows = tuple(observations)
    references = tuple(
        MetricObservation(
            metric=row.metric,
            value=standard.property(row.metric).value,
            unit=standard.property(row.metric).unit,
            standard_uncertainty=standard.property(row.metric).standard_uncertainty,
            method=f"Certified value from {standard.standard_id}",
        )
        for row in observation_rows
    )
    evaluations, criterion_rows = _evaluate_sets(
        observation_rows,
        references,
        criteria,
    )
    warnings = list(standard.notes)
    temperature_mismatch = False
    for row in observation_rows:
        property_value = standard.property(row.metric)
        reference_temperature = property_value.reference_temperature_c
        if (
            reference_temperature is not None
            and abs(acquisition.temperature_c - reference_temperature) > 1.0
        ):
            temperature_mismatch = True
            warnings.append(
                f"{row.metric}: acquisition temperature differs from the certified "
                "reference temperature by more than 1 degC; apply a documented correction."
            )
    if temperature_mismatch:
        status = AssessmentStatus.REVIEW
    else:
        status = (
            AssessmentStatus.PASS
            if all(row.passed for row in evaluations)
            else AssessmentStatus.FAIL
        )
    return MetrologyAssessment(
        dataset_id=str(dataset_id),
        title=title or f"{standard.standard_id} certified-reference assessment",
        evidence_class=EvidenceClass.CERTIFIED_REFERENCE,
        status=status,
        evaluations=evaluations,
        criteria=criterion_rows,
        acquisition_fingerprint=acquisition.fingerprint,
        input_signature=acquisition.source_data_signature,
        standard=standard.to_dict(),
        material_unit_id=unit_id,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def assess_reference_engine(
    dataset_id: str,
    native_engine: EngineIdentity,
    native_observations: Iterable[MetricObservation],
    reference_engine: EngineIdentity,
    reference_observations: Iterable[MetricObservation],
    criteria: Iterable[AcceptanceCriterion],
    *,
    native_input_signature: str,
    reference_input_signature: str,
    acquisition: AcquisitionMetadata | None = None,
    title: str = "Native/reference-engine comparison",
) -> MetrologyAssessment:
    if not isinstance(native_engine, EngineIdentity) or not isinstance(
        reference_engine, EngineIdentity
    ):
        raise MetrologyValidationError("both engine identities and versions are required")
    if native_engine == reference_engine:
        raise MetrologyValidationError("native and reference engine identities must differ")
    native_signature = str(native_input_signature).strip()
    reference_signature = str(reference_input_signature).strip()
    if not native_signature or not reference_signature:
        raise MetrologyValidationError("both engine input signatures are required")
    if native_signature != reference_signature:
        raise MetrologyValidationError(
            "reference-engine comparison inputs differ; identical signed inputs are required"
        )
    evaluations, criterion_rows = _evaluate_sets(
        tuple(native_observations), tuple(reference_observations), criteria
    )
    status = (
        AssessmentStatus.PASS
        if all(row.passed for row in evaluations)
        else AssessmentStatus.FAIL
    )
    warnings = (
        "Reference-engine agreement is not certified experimental accuracy.",
    )
    return MetrologyAssessment(
        dataset_id=str(dataset_id),
        title=title,
        evidence_class=EvidenceClass.REFERENCE_ENGINE,
        status=status,
        evaluations=evaluations,
        criteria=criterion_rows,
        acquisition_fingerprint="" if acquisition is None else acquisition.fingerprint,
        input_signature=native_signature,
        native_engine=native_engine.to_dict(),
        reference_engine=reference_engine.to_dict(),
        warnings=warnings,
    )


def assess_reference_values(
    dataset_id: str,
    evidence_class: EvidenceClass,
    observations: Iterable[MetricObservation],
    reference_observations: Iterable[MetricObservation],
    criteria: Iterable[AcceptanceCriterion],
    *,
    acquisition: AcquisitionMetadata | None = None,
    title: str = "Scientific reference-value assessment",
) -> MetrologyAssessment:
    if evidence_class not in {
        EvidenceClass.SOFTWARE_REGRESSION,
        EvidenceClass.EXPERIMENTAL_VALIDATION,
    }:
        raise MetrologyValidationError(
            "generic reference values support software regression or experimental validation"
        )
    if evidence_class == EvidenceClass.EXPERIMENTAL_VALIDATION and acquisition is None:
        raise MetrologyValidationError(
            "experimental validation requires complete acquisition metadata"
        )
    evaluations, criterion_rows = _evaluate_sets(
        tuple(observations), tuple(reference_observations), criteria
    )
    status = (
        AssessmentStatus.PASS
        if all(row.passed for row in evaluations)
        else AssessmentStatus.FAIL
    )
    warnings = ()
    if evidence_class == EvidenceClass.SOFTWARE_REGRESSION:
        warnings = (
            "Software regression verifies deterministic behavior, not experimental accuracy.",
        )
    return MetrologyAssessment(
        dataset_id=str(dataset_id),
        title=title,
        evidence_class=evidence_class,
        status=status,
        evaluations=evaluations,
        criteria=criterion_rows,
        acquisition_fingerprint="" if acquisition is None else acquisition.fingerprint,
        input_signature=(
            ""
            if acquisition is None
            else acquisition.source_data_signature
        ),
        warnings=warnings,
    )


def authorize_claim(
    assessment: MetrologyAssessment,
    scope: ClaimScope,
    statement: str,
) -> ScientificClaim:
    if not isinstance(assessment, MetrologyAssessment):
        raise MetrologyValidationError("assessment must be a MetrologyAssessment")
    if not assessment.passed:
        raise MetrologyValidationError(
            "scientific claim cannot be marked validated because the assessment did not pass"
        )
    if not isinstance(scope, ClaimScope):
        try:
            scope = ClaimScope(str(scope))
        except ValueError as exc:
            raise MetrologyValidationError(f"unknown scientific claim scope: {scope}") from exc
    if scope not in assessment.allowed_claim_scopes:
        allowed = ", ".join(row.value for row in assessment.allowed_claim_scopes) or "none"
        raise MetrologyValidationError(
            f"{assessment.evidence_class.value} cannot validate '{scope.value}'; "
            f"allowed scopes: {allowed}"
        )
    return ScientificClaim(
        dataset_id=assessment.dataset_id,
        assessment_id=assessment.assessment_id,
        scope=scope,
        statement=str(statement),
        assessment_fingerprint=assessment.fingerprint,
    )
