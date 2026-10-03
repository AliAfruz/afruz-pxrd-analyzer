from __future__ import annotations

from typing import Iterable

import numpy as np

from ..contracts import EngineIdentity
from .evaluation import (
    assess_certified_standard,
    assess_reference_engine,
    assess_reference_values,
    authorize_claim,
    dataset_data_signature,
)
from .models import (
    AcceptanceCriterion,
    AcquisitionMetadata,
    CertifiedStandard,
    ClaimScope,
    EvidenceClass,
    MetrologyAssessment,
    MetrologyRegistry,
    MetrologyValidationError,
    MetricObservation,
    ScientificClaim,
)
from .standards import StandardCatalogue


class MetrologyService:
    """Headless command surface for traceable scientific evidence and claims."""

    def __init__(
        self,
        registry: MetrologyRegistry | None = None,
        catalogue: StandardCatalogue | None = None,
    ) -> None:
        self.registry = registry if registry is not None else MetrologyRegistry()
        self.catalogue = catalogue if catalogue is not None else StandardCatalogue()

    def bind_registry(self, registry: MetrologyRegistry) -> None:
        if not isinstance(registry, MetrologyRegistry):
            raise TypeError("registry must be a MetrologyRegistry")
        self.registry = registry

    def register_standard(
        self,
        standard: CertifiedStandard,
        *,
        replace_existing: bool = False,
    ) -> None:
        self.catalogue.register(standard, replace_existing=replace_existing)

    def set_acquisition(
        self,
        dataset_id: str,
        metadata: AcquisitionMetadata,
        *,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> AcquisitionMetadata:
        if (x is None) != (y is None):
            raise MetrologyValidationError("x and y must be supplied together")
        if x is not None and y is not None:
            current_signature = dataset_data_signature(x, y)
            if current_signature != metadata.source_data_signature:
                raise MetrologyValidationError(
                    "acquisition source_data_signature does not match the dataset arrays"
                )
        self.registry.set_acquisition(str(dataset_id), metadata)
        return metadata

    def assess_standard(
        self,
        dataset_id: str,
        standard_id: str,
        observations: Iterable[MetricObservation],
        criteria: Iterable[AcceptanceCriterion],
        *,
        material_unit_id: str,
        title: str | None = None,
    ) -> MetrologyAssessment:
        acquisition = self._acquisition(dataset_id)
        assessment = assess_certified_standard(
            dataset_id,
            self.catalogue.get(standard_id),
            observations,
            criteria,
            acquisition,
            material_unit_id=material_unit_id,
            title=title,
        )
        self.registry.record_assessment(assessment)
        return assessment

    def compare_engines(
        self,
        dataset_id: str,
        native_engine: EngineIdentity,
        native_observations: Iterable[MetricObservation],
        reference_engine: EngineIdentity,
        reference_observations: Iterable[MetricObservation],
        criteria: Iterable[AcceptanceCriterion],
        *,
        native_input_signature: str,
        reference_input_signature: str,
        title: str = "Native/reference-engine comparison",
    ) -> MetrologyAssessment:
        assessment = assess_reference_engine(
            dataset_id,
            native_engine,
            native_observations,
            reference_engine,
            reference_observations,
            criteria,
            native_input_signature=native_input_signature,
            reference_input_signature=reference_input_signature,
            acquisition=self.registry.acquisition_by_dataset.get(str(dataset_id)),
            title=title,
        )
        self.registry.record_assessment(assessment)
        return assessment

    def assess_values(
        self,
        dataset_id: str,
        evidence_class: EvidenceClass,
        observations: Iterable[MetricObservation],
        reference_observations: Iterable[MetricObservation],
        criteria: Iterable[AcceptanceCriterion],
        *,
        title: str = "Scientific reference-value assessment",
    ) -> MetrologyAssessment:
        assessment = assess_reference_values(
            dataset_id,
            evidence_class,
            observations,
            reference_observations,
            criteria,
            acquisition=self.registry.acquisition_by_dataset.get(str(dataset_id)),
            title=title,
        )
        self.registry.record_assessment(assessment)
        return assessment

    def authorize_claim(
        self,
        assessment_id: str,
        scope: ClaimScope,
        statement: str,
    ) -> ScientificClaim:
        try:
            assessment = self.registry.assessments[str(assessment_id)]
        except KeyError as exc:
            raise MetrologyValidationError(
                f"unknown metrology assessment: {assessment_id}"
            ) from exc
        claim = authorize_claim(assessment, scope, statement)
        self.registry.record_claim(claim)
        return claim

    def dataset_summary(self, dataset_id: str) -> dict:
        dataset_id = str(dataset_id)
        assessments = self.registry.for_dataset(dataset_id)
        claims = self.registry.claims_for_dataset(dataset_id)
        return {
            "dataset_id": dataset_id,
            "acquisition": (
                None
                if dataset_id not in self.registry.acquisition_by_dataset
                else self.registry.acquisition_by_dataset[dataset_id].to_dict()
            ),
            "assessments": [row.to_dict() for row in assessments],
            "claims": [row.to_dict() for row in claims],
            "validated_claim_count": len(claims),
        }

    def _acquisition(self, dataset_id: str) -> AcquisitionMetadata:
        try:
            return self.registry.acquisition_by_dataset[str(dataset_id)]
        except KeyError as exc:
            raise MetrologyValidationError(
                "complete acquisition metadata must be registered before a "
                "certified-reference or experimental assessment"
            ) from exc
