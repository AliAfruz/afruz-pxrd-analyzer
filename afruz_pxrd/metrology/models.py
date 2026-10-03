from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping
import uuid

import numpy as np

from ..contracts.base import utc_now


METROLOGY_SCHEMA_VERSION = 1


class MetrologyValidationError(ValueError):
    """Raised when scientific evidence is incomplete or internally inconsistent."""


class EvidenceClass(str, Enum):
    UNCLASSIFIED = "Unclassified evidence"
    SOFTWARE_REGRESSION = "Software regression"
    CERTIFIED_REFERENCE = "Certified reference material"
    REFERENCE_ENGINE = "Reference-engine comparison"
    EXPERIMENTAL_VALIDATION = "Experimental validation"


class AssessmentStatus(str, Enum):
    PASS = "Pass"
    REVIEW = "Review required"
    FAIL = "Fail"
    INELIGIBLE = "Ineligible"


class ClaimScope(str, Enum):
    SOFTWARE_BEHAVIOR = "software_behavior"
    ENGINE_AGREEMENT = "engine_agreement"
    INSTRUMENT_PERFORMANCE = "instrument_performance"
    MEASUREMENT_ACCURACY = "measurement_accuracy"
    METHOD_VALIDATION = "method_validation"


CLAIM_SCOPES_BY_EVIDENCE = {
    EvidenceClass.SOFTWARE_REGRESSION: frozenset({ClaimScope.SOFTWARE_BEHAVIOR}),
    EvidenceClass.REFERENCE_ENGINE: frozenset({ClaimScope.ENGINE_AGREEMENT}),
    EvidenceClass.CERTIFIED_REFERENCE: frozenset(
        {ClaimScope.INSTRUMENT_PERFORMANCE, ClaimScope.MEASUREMENT_ACCURACY}
    ),
    EvidenceClass.EXPERIMENTAL_VALIDATION: frozenset(
        {ClaimScope.MEASUREMENT_ACCURACY, ClaimScope.METHOD_VALIDATION}
    ),
    EvidenceClass.UNCLASSIFIED: frozenset(),
}


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise MetrologyValidationError(f"{field_name} is required")
    return text


def _finite(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MetrologyValidationError(f"{field_name} must be numeric") from exc
    if not np.isfinite(number):
        raise MetrologyValidationError(f"{field_name} must be finite")
    return number


def _positive(value: Any, field_name: str) -> float:
    number = _finite(value, field_name)
    if number <= 0:
        raise MetrologyValidationError(f"{field_name} must be positive")
    return number


def _iso_datetime(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MetrologyValidationError(f"{field_name} must be ISO-8601 text") from exc
    return text


def stable_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AcquisitionMetadata:
    """Traceable acquisition conditions required for experimental claims."""

    instrument_id: str
    instrument_name: str
    geometry: str
    radiation: str
    wavelength_angstrom: float
    detector: str
    optics: str
    step_size_deg: float
    counting_time_seconds: float
    temperature_c: float
    specimen_preparation: str
    specimen_geometry: str
    operator: str
    acquired_at: str
    source_data_signature: str
    additional: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "instrument_id",
            "instrument_name",
            "geometry",
            "radiation",
            "detector",
            "optics",
            "specimen_preparation",
            "specimen_geometry",
            "operator",
            "source_data_signature",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "wavelength_angstrom",
            _positive(self.wavelength_angstrom, "wavelength_angstrom"),
        )
        object.__setattr__(
            self, "step_size_deg", _positive(self.step_size_deg, "step_size_deg")
        )
        object.__setattr__(
            self,
            "counting_time_seconds",
            _positive(self.counting_time_seconds, "counting_time_seconds"),
        )
        object.__setattr__(
            self, "temperature_c", _finite(self.temperature_c, "temperature_c")
        )
        object.__setattr__(
            self, "acquired_at", _iso_datetime(self.acquired_at, "acquired_at")
        )
        if not isinstance(self.additional, dict):
            raise MetrologyValidationError("additional acquisition metadata must be an object")

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "instrument_name": self.instrument_name,
            "geometry": self.geometry,
            "radiation": self.radiation,
            "wavelength_angstrom": self.wavelength_angstrom,
            "detector": self.detector,
            "optics": self.optics,
            "step_size_deg": self.step_size_deg,
            "counting_time_seconds": self.counting_time_seconds,
            "temperature_c": self.temperature_c,
            "specimen_preparation": self.specimen_preparation,
            "specimen_geometry": self.specimen_geometry,
            "operator": self.operator,
            "acquired_at": self.acquired_at,
            "source_data_signature": self.source_data_signature,
            "additional": dict(self.additional),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AcquisitionMetadata":
        if not isinstance(value, Mapping):
            raise MetrologyValidationError("acquisition metadata must be an object")
        required = {
            "instrument_id",
            "instrument_name",
            "geometry",
            "radiation",
            "wavelength_angstrom",
            "detector",
            "optics",
            "step_size_deg",
            "counting_time_seconds",
            "temperature_c",
            "specimen_preparation",
            "specimen_geometry",
            "operator",
            "acquired_at",
            "source_data_signature",
        }
        missing = required - set(value)
        if missing:
            raise MetrologyValidationError(
                f"acquisition metadata is missing: {sorted(missing)}"
            )
        return cls(
            **{name: value[name] for name in required},
            additional=dict(value.get("additional", {})),
        )


@dataclass(frozen=True)
class CertifiedProperty:
    property_id: str
    name: str
    value: float
    unit: str
    expanded_uncertainty: float
    coverage_factor: float
    reference_temperature_c: float | None = None
    value_status: str = "Certified"
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("property_id", "name", "unit", "value_status"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "value", _finite(self.value, "value"))
        object.__setattr__(
            self,
            "expanded_uncertainty",
            _positive(self.expanded_uncertainty, "expanded_uncertainty"),
        )
        object.__setattr__(
            self, "coverage_factor", _positive(self.coverage_factor, "coverage_factor")
        )
        if self.reference_temperature_c is not None:
            object.__setattr__(
                self,
                "reference_temperature_c",
                _finite(self.reference_temperature_c, "reference_temperature_c"),
            )

    @property
    def standard_uncertainty(self) -> float:
        return self.expanded_uncertainty / self.coverage_factor

    @property
    def is_certified(self) -> bool:
        return self.value_status.casefold() == "certified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "expanded_uncertainty": self.expanded_uncertainty,
            "coverage_factor": self.coverage_factor,
            "standard_uncertainty": self.standard_uncertainty,
            "reference_temperature_c": self.reference_temperature_c,
            "value_status": self.value_status,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CertifiedProperty":
        if not isinstance(value, Mapping):
            raise MetrologyValidationError("certified property must be an object")
        return cls(
            property_id=value.get("property_id", ""),
            name=value.get("name", ""),
            value=value.get("value"),
            unit=value.get("unit", ""),
            expanded_uncertainty=value.get("expanded_uncertainty"),
            coverage_factor=value.get("coverage_factor"),
            reference_temperature_c=value.get("reference_temperature_c"),
            value_status=value.get("value_status", "Certified"),
            notes=str(value.get("notes", "")),
        )


@dataclass(frozen=True)
class CertifiedStandard:
    standard_id: str
    title: str
    material: str
    provider: str
    certificate_url: str
    certificate_revision: str
    intended_uses: tuple[str, ...]
    properties: dict[str, CertifiedProperty]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "standard_id",
            "title",
            "material",
            "provider",
            "certificate_url",
            "certificate_revision",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if not isinstance(self.intended_uses, tuple) or not self.intended_uses:
            raise MetrologyValidationError("intended_uses must be a non-empty tuple")
        if not isinstance(self.properties, dict) or not all(
            isinstance(item, CertifiedProperty) for item in self.properties.values()
        ):
            raise MetrologyValidationError(
                "standard properties must map property IDs to CertifiedProperty values"
            )
        for key, item in self.properties.items():
            if key != item.property_id:
                raise MetrologyValidationError(
                    f"standard property key '{key}' does not match '{item.property_id}'"
                )

    def property(self, property_id: str, *, require_certified: bool = True) -> CertifiedProperty:
        try:
            value = self.properties[str(property_id)]
        except KeyError as exc:
            raise MetrologyValidationError(
                f"{self.standard_id} has no property '{property_id}'"
            ) from exc
        if require_certified and not value.is_certified:
            raise MetrologyValidationError(
                f"{self.standard_id} property '{property_id}' is not a certified value"
            )
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "standard_id": self.standard_id,
            "title": self.title,
            "material": self.material,
            "provider": self.provider,
            "certificate_url": self.certificate_url,
            "certificate_revision": self.certificate_revision,
            "intended_uses": list(self.intended_uses),
            "properties": {
                key: item.to_dict() for key, item in sorted(self.properties.items())
            },
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CertifiedStandard":
        if not isinstance(value, Mapping):
            raise MetrologyValidationError("certified standard must be an object")
        properties = value.get("properties", {})
        if not isinstance(properties, Mapping):
            raise MetrologyValidationError("standard properties must be an object")
        return cls(
            standard_id=value.get("standard_id", ""),
            title=value.get("title", ""),
            material=value.get("material", ""),
            provider=value.get("provider", ""),
            certificate_url=value.get("certificate_url", ""),
            certificate_revision=value.get("certificate_revision", ""),
            intended_uses=tuple(str(row) for row in value.get("intended_uses", [])),
            properties={
                str(key): CertifiedProperty.from_dict(item)
                for key, item in properties.items()
            },
            notes=tuple(str(row) for row in value.get("notes", [])),
        )


@dataclass(frozen=True)
class MetricObservation:
    metric: str
    value: float
    unit: str
    standard_uncertainty: float | None = None
    method: str = ""
    source_result_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", _required_text(self.metric, "metric"))
        object.__setattr__(self, "unit", _required_text(self.unit, "unit"))
        object.__setattr__(self, "value", _finite(self.value, "value"))
        if self.standard_uncertainty is not None:
            object.__setattr__(
                self,
                "standard_uncertainty",
                _positive(self.standard_uncertainty, "standard_uncertainty"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "standard_uncertainty": self.standard_uncertainty,
            "method": self.method,
            "source_result_id": self.source_result_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MetricObservation":
        if not isinstance(value, Mapping):
            raise MetrologyValidationError("metric observation must be an object")
        return cls(
            metric=value.get("metric", ""),
            value=value.get("value"),
            unit=value.get("unit", ""),
            standard_uncertainty=value.get("standard_uncertainty"),
            method=str(value.get("method", "")),
            source_result_id=str(value.get("source_result_id", "")),
        )


@dataclass(frozen=True)
class AcceptanceCriterion:
    metric: str
    absolute_tolerance: float | None = None
    relative_tolerance_percent: float | None = None
    maximum_en: float | None = None
    require_measurement_uncertainty: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", _required_text(self.metric, "metric"))
        configured = False
        for name in (
            "absolute_tolerance",
            "relative_tolerance_percent",
            "maximum_en",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive(value, name))
                configured = True
        if not configured:
            raise MetrologyValidationError(
                f"criterion '{self.metric}' requires at least one explicit tolerance"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance_percent": self.relative_tolerance_percent,
            "maximum_en": self.maximum_en,
            "require_measurement_uncertainty": self.require_measurement_uncertainty,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AcceptanceCriterion":
        if not isinstance(value, Mapping):
            raise MetrologyValidationError("acceptance criterion must be an object")
        return cls(
            metric=value.get("metric", ""),
            absolute_tolerance=value.get("absolute_tolerance"),
            relative_tolerance_percent=value.get("relative_tolerance_percent"),
            maximum_en=value.get("maximum_en"),
            require_measurement_uncertainty=bool(
                value.get("require_measurement_uncertainty", False)
            ),
            notes=str(value.get("notes", "")),
        )


@dataclass(frozen=True)
class MetricEvaluation:
    metric: str
    measured_value: float
    reference_value: float
    unit: str
    signed_error: float
    absolute_error: float
    relative_error_percent: float | None
    en_number: float | None
    passed: bool
    checks: tuple[dict[str, Any], ...]
    measurement_standard_uncertainty: float | None = None
    reference_standard_uncertainty: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", _required_text(self.metric, "metric"))
        object.__setattr__(self, "unit", _required_text(self.unit, "unit"))
        for name in (
            "measured_value",
            "reference_value",
            "signed_error",
            "absolute_error",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.absolute_error < 0:
            raise MetrologyValidationError("absolute_error cannot be negative")
        expected_signed = self.measured_value - self.reference_value
        tolerance = 1e-12 * max(1.0, abs(expected_signed), abs(self.signed_error))
        if abs(self.signed_error - expected_signed) > tolerance:
            raise MetrologyValidationError("stored signed_error is inconsistent")
        if abs(self.absolute_error - abs(self.signed_error)) > tolerance:
            raise MetrologyValidationError("stored absolute_error is inconsistent")
        for name in ("relative_error_percent", "en_number"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        if self.en_number is not None and self.en_number < 0:
            raise MetrologyValidationError("en_number cannot be negative")
        for name in (
            "measurement_standard_uncertainty",
            "reference_standard_uncertainty",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive(value, name))
        if not isinstance(self.checks, tuple) or not self.checks:
            raise MetrologyValidationError("metric evaluation requires checks")
        for check in self.checks:
            if not isinstance(check, dict):
                raise MetrologyValidationError("metric checks must be objects")
            missing = {"name", "value", "limit", "passed"} - set(check)
            if missing:
                raise MetrologyValidationError(
                    f"metric check is missing: {sorted(missing)}"
                )
            if not isinstance(check["passed"], bool):
                raise MetrologyValidationError("metric check passed flag must be boolean")
        if self.passed != all(check["passed"] for check in self.checks):
            raise MetrologyValidationError(
                "metric evaluation pass flag is inconsistent with its checks"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "measured_value": self.measured_value,
            "reference_value": self.reference_value,
            "unit": self.unit,
            "signed_error": self.signed_error,
            "absolute_error": self.absolute_error,
            "relative_error_percent": self.relative_error_percent,
            "en_number": self.en_number,
            "passed": self.passed,
            "checks": [dict(row) for row in self.checks],
            "measurement_standard_uncertainty": self.measurement_standard_uncertainty,
            "reference_standard_uncertainty": self.reference_standard_uncertainty,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MetricEvaluation":
        if not isinstance(value, Mapping):
            raise MetrologyValidationError("metric evaluation must be an object")
        return cls(
            metric=str(value["metric"]),
            measured_value=float(value["measured_value"]),
            reference_value=float(value["reference_value"]),
            unit=str(value["unit"]),
            signed_error=float(value["signed_error"]),
            absolute_error=float(value["absolute_error"]),
            relative_error_percent=(
                None
                if value.get("relative_error_percent") is None
                else float(value["relative_error_percent"])
            ),
            en_number=(
                None if value.get("en_number") is None else float(value["en_number"])
            ),
            passed=bool(value["passed"]),
            checks=tuple(dict(row) for row in value.get("checks", [])),
            measurement_standard_uncertainty=value.get(
                "measurement_standard_uncertainty"
            ),
            reference_standard_uncertainty=value.get("reference_standard_uncertainty"),
        )


@dataclass(frozen=True)
class MetrologyAssessment:
    dataset_id: str
    title: str
    evidence_class: EvidenceClass
    status: AssessmentStatus
    evaluations: tuple[MetricEvaluation, ...]
    criteria: tuple[AcceptanceCriterion, ...]
    acquisition_fingerprint: str = ""
    input_signature: str = ""
    standard: dict[str, Any] | None = None
    material_unit_id: str = ""
    native_engine: dict[str, str] | None = None
    reference_engine: dict[str, str] | None = None
    warnings: tuple[str, ...] = ()
    assessment_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("dataset_id", "title", "assessment_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if not isinstance(self.evidence_class, EvidenceClass):
            raise MetrologyValidationError("evidence_class must be an EvidenceClass")
        if not isinstance(self.status, AssessmentStatus):
            raise MetrologyValidationError("status must be an AssessmentStatus")
        if not self.evaluations:
            raise MetrologyValidationError("assessment requires metric evaluations")
        if not self.criteria:
            raise MetrologyValidationError("assessment requires explicit criteria")
        evaluation_metrics = [row.metric for row in self.evaluations]
        criterion_metrics = [row.metric for row in self.criteria]
        if (
            len(evaluation_metrics) != len(set(evaluation_metrics))
            or len(criterion_metrics) != len(set(criterion_metrics))
            or set(evaluation_metrics) != set(criterion_metrics)
        ):
            raise MetrologyValidationError(
                "assessment evaluations and criteria must cover identical unique metrics"
            )
        evaluations_pass = all(row.passed for row in self.evaluations)
        if self.status == AssessmentStatus.PASS and not evaluations_pass:
            raise MetrologyValidationError("passing assessment contains a failed metric")
        if self.status == AssessmentStatus.FAIL and evaluations_pass:
            raise MetrologyValidationError("failed assessment contains no failed metric")
        if self.evidence_class == EvidenceClass.REFERENCE_ENGINE and not str(
            self.input_signature
        ).strip():
            raise MetrologyValidationError(
                "reference-engine assessment requires a shared input signature"
            )
        if self.evidence_class in {
            EvidenceClass.CERTIFIED_REFERENCE,
            EvidenceClass.EXPERIMENTAL_VALIDATION,
        } and (not self.acquisition_fingerprint or not self.input_signature):
            raise MetrologyValidationError(
                "experimental evidence requires acquisition and input signatures"
            )
        if self.evidence_class == EvidenceClass.CERTIFIED_REFERENCE:
            if not isinstance(self.standard, dict) or not self.material_unit_id.strip():
                raise MetrologyValidationError(
                    "certified-reference evidence requires a standard snapshot and material unit ID"
                )
        if self.evidence_class == EvidenceClass.REFERENCE_ENGINE:
            for name, identity in (
                ("native_engine", self.native_engine),
                ("reference_engine", self.reference_engine),
            ):
                if not isinstance(identity, dict) or not {
                    "name",
                    "version",
                }.issubset(identity):
                    raise MetrologyValidationError(
                        f"{name} identity and version are required"
                    )
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(row, str) for row in self.warnings
        ):
            raise MetrologyValidationError("assessment warnings must be strings")
        _iso_datetime(self.created_at, "created_at")

    @property
    def passed(self) -> bool:
        return self.status == AssessmentStatus.PASS and all(
            row.passed for row in self.evaluations
        )

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.to_dict())

    @property
    def allowed_claim_scopes(self) -> tuple[ClaimScope, ...]:
        if not self.passed:
            return ()
        return tuple(sorted(CLAIM_SCOPES_BY_EVIDENCE[self.evidence_class], key=str))

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "dataset_id": self.dataset_id,
            "title": self.title,
            "evidence_class": self.evidence_class.value,
            "status": self.status.value,
            "passed": self.passed,
            "allowed_claim_scopes": [row.value for row in self.allowed_claim_scopes],
            "evaluations": [row.to_dict() for row in self.evaluations],
            "criteria": [row.to_dict() for row in self.criteria],
            "acquisition_fingerprint": self.acquisition_fingerprint,
            "input_signature": self.input_signature,
            "standard": None if self.standard is None else dict(self.standard),
            "material_unit_id": self.material_unit_id,
            "native_engine": None if self.native_engine is None else dict(self.native_engine),
            "reference_engine": (
                None if self.reference_engine is None else dict(self.reference_engine)
            ),
            "warnings": list(self.warnings),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MetrologyAssessment":
        if not isinstance(value, Mapping):
            raise MetrologyValidationError("metrology assessment must be an object")
        try:
            evidence_class = EvidenceClass(str(value["evidence_class"]))
            status = AssessmentStatus(str(value["status"]))
        except (KeyError, ValueError) as exc:
            raise MetrologyValidationError("unknown assessment classification/status") from exc
        return cls(
            assessment_id=str(value.get("assessment_id", "")),
            dataset_id=str(value.get("dataset_id", "")),
            title=str(value.get("title", "")),
            evidence_class=evidence_class,
            status=status,
            evaluations=tuple(
                MetricEvaluation.from_dict(row) for row in value.get("evaluations", [])
            ),
            criteria=tuple(
                AcceptanceCriterion.from_dict(row) for row in value.get("criteria", [])
            ),
            acquisition_fingerprint=str(value.get("acquisition_fingerprint", "")),
            input_signature=str(value.get("input_signature", "")),
            standard=(
                None if value.get("standard") is None else dict(value["standard"])
            ),
            material_unit_id=str(value.get("material_unit_id", "")),
            native_engine=(
                None
                if value.get("native_engine") is None
                else {str(k): str(v) for k, v in dict(value["native_engine"]).items()}
            ),
            reference_engine=(
                None
                if value.get("reference_engine") is None
                else {
                    str(k): str(v) for k, v in dict(value["reference_engine"]).items()
                }
            ),
            warnings=tuple(str(row) for row in value.get("warnings", [])),
            created_at=str(value.get("created_at", "")),
        )


@dataclass(frozen=True)
class ScientificClaim:
    dataset_id: str
    assessment_id: str
    scope: ClaimScope
    statement: str
    assessment_fingerprint: str
    claim_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    authorized_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "assessment_id",
            "statement",
            "assessment_fingerprint",
            "claim_id",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if not isinstance(self.scope, ClaimScope):
            raise MetrologyValidationError("scope must be a ClaimScope")
        _iso_datetime(self.authorized_at, "authorized_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "dataset_id": self.dataset_id,
            "assessment_id": self.assessment_id,
            "scope": self.scope.value,
            "statement": self.statement,
            "assessment_fingerprint": self.assessment_fingerprint,
            "status": "Validated",
            "authorized_at": self.authorized_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScientificClaim":
        if not isinstance(value, Mapping):
            raise MetrologyValidationError("scientific claim must be an object")
        try:
            scope = ClaimScope(str(value["scope"]))
        except (KeyError, ValueError) as exc:
            raise MetrologyValidationError("unknown scientific claim scope") from exc
        if value.get("status") != "Validated":
            raise MetrologyValidationError("scientific claim status must be Validated")
        return cls(
            claim_id=str(value.get("claim_id", "")),
            dataset_id=str(value.get("dataset_id", "")),
            assessment_id=str(value.get("assessment_id", "")),
            scope=scope,
            statement=str(value.get("statement", "")),
            assessment_fingerprint=str(value.get("assessment_fingerprint", "")),
            authorized_at=str(value.get("authorized_at", "")),
        )


@dataclass
class MetrologyRegistry:
    acquisition_by_dataset: dict[str, AcquisitionMetadata] = field(default_factory=dict)
    assessments: dict[str, MetrologyAssessment] = field(default_factory=dict)
    claims: dict[str, ScientificClaim] = field(default_factory=dict)

    def set_acquisition(self, dataset_id: str, value: AcquisitionMetadata) -> None:
        dataset_id = _required_text(dataset_id, "dataset_id")
        if not isinstance(value, AcquisitionMetadata):
            raise MetrologyValidationError("value must be AcquisitionMetadata")
        self.acquisition_by_dataset[dataset_id] = value
        self._invalidate_dataset_evidence(dataset_id)

    def record_assessment(self, value: MetrologyAssessment) -> None:
        if not isinstance(value, MetrologyAssessment):
            raise MetrologyValidationError("value must be MetrologyAssessment")
        self.assessments[value.assessment_id] = value

    def record_claim(self, value: ScientificClaim) -> None:
        if not isinstance(value, ScientificClaim):
            raise MetrologyValidationError("value must be ScientificClaim")
        assessment = self.assessments.get(value.assessment_id)
        if assessment is None or assessment.dataset_id != value.dataset_id:
            raise MetrologyValidationError("claim references an unknown assessment")
        if assessment.fingerprint != value.assessment_fingerprint:
            raise MetrologyValidationError("claim assessment fingerprint is stale")
        self.claims[value.claim_id] = value

    def for_dataset(self, dataset_id: str) -> tuple[MetrologyAssessment, ...]:
        return tuple(
            row
            for row in self.assessments.values()
            if row.dataset_id == str(dataset_id)
        )

    def claims_for_dataset(self, dataset_id: str) -> tuple[ScientificClaim, ...]:
        return tuple(
            row for row in self.claims.values() if row.dataset_id == str(dataset_id)
        )

    def remove_dataset(self, dataset_id: str) -> None:
        dataset_id = str(dataset_id)
        self.acquisition_by_dataset.pop(dataset_id, None)
        assessment_ids = {
            key for key, row in self.assessments.items() if row.dataset_id == dataset_id
        }
        for key in assessment_ids:
            self.assessments.pop(key, None)
        self.claims = {
            key: row
            for key, row in self.claims.items()
            if row.dataset_id != dataset_id and row.assessment_id not in assessment_ids
        }

    def _invalidate_dataset_evidence(self, dataset_id: str) -> None:
        assessment_ids = {
            key for key, row in self.assessments.items() if row.dataset_id == dataset_id
        }
        for key in assessment_ids:
            self.assessments.pop(key, None)
        self.claims = {
            key: row
            for key, row in self.claims.items()
            if row.assessment_id not in assessment_ids
        }

    def validate(self, dataset_ids: Iterable[str] | None = None) -> None:
        for key, row in self.assessments.items():
            if key != row.assessment_id:
                raise MetrologyValidationError(
                    f"assessment registry key '{key}' does not match its identity"
                )
        for key, row in self.claims.items():
            if key != row.claim_id:
                raise MetrologyValidationError(
                    f"claim registry key '{key}' does not match its identity"
                )
        allowed = None if dataset_ids is None else {str(row) for row in dataset_ids}
        if allowed is not None:
            referenced = {
                *self.acquisition_by_dataset,
                *(row.dataset_id for row in self.assessments.values()),
                *(row.dataset_id for row in self.claims.values()),
            }
            unknown = referenced - allowed
            if unknown:
                raise MetrologyValidationError(
                    f"metrology evidence references unknown datasets: {sorted(unknown)}"
                )
        for claim in self.claims.values():
            assessment = self.assessments.get(claim.assessment_id)
            if assessment is None:
                raise MetrologyValidationError("claim references a missing assessment")
            if not assessment.passed:
                raise MetrologyValidationError("claim references an assessment that did not pass")
            if claim.scope not in assessment.allowed_claim_scopes:
                raise MetrologyValidationError("claim scope is not supported by its evidence")
            if claim.assessment_fingerprint != assessment.fingerprint:
                raise MetrologyValidationError("claim references stale assessment evidence")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": METROLOGY_SCHEMA_VERSION,
            "acquisition_by_dataset": {
                key: row.to_dict()
                for key, row in sorted(self.acquisition_by_dataset.items())
            },
            "assessments": {
                key: row.to_dict() for key, row in sorted(self.assessments.items())
            },
            "claims": {key: row.to_dict() for key, row in sorted(self.claims.items())},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "MetrologyRegistry":
        if value in (None, {}):
            return cls()
        if not isinstance(value, Mapping):
            raise MetrologyValidationError("metrology registry must be an object")
        if int(value.get("schema_version", -1)) != METROLOGY_SCHEMA_VERSION:
            raise MetrologyValidationError(
                f"unsupported metrology schema version: {value.get('schema_version')}"
            )
        acquisition = value.get("acquisition_by_dataset", {})
        assessments = value.get("assessments", {})
        claims = value.get("claims", {})
        if not all(isinstance(row, Mapping) for row in (acquisition, assessments, claims)):
            raise MetrologyValidationError("metrology registry groups must be objects")
        registry = cls(
            acquisition_by_dataset={
                str(key): AcquisitionMetadata.from_dict(item)
                for key, item in acquisition.items()
            },
            assessments={
                str(key): MetrologyAssessment.from_dict(item)
                for key, item in assessments.items()
            },
            claims={
                str(key): ScientificClaim.from_dict(item)
                for key, item in claims.items()
            },
        )
        registry.validate()
        return registry
