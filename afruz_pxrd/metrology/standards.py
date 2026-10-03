from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .models import CertifiedProperty, CertifiedStandard, MetrologyValidationError


NIST_SRM_640G = CertifiedStandard(
    standard_id="NIST-SRM-640g",
    title="Line Position and Line Shape Standard for Powder Diffraction",
    material="Silicon powder",
    provider="National Institute of Standards and Technology",
    certificate_url="https://tsapps.nist.gov/srmext/certificates/640g.pdf",
    certificate_revision="Certificate verified 2026-08-11",
    intended_uses=(
        "line position calibration",
        "instrument profile evaluation for laboratory X-ray sources",
    ),
    properties={
        "lattice_parameter_a": CertifiedProperty(
            property_id="lattice_parameter_a",
            name="Cubic lattice parameter a at 22.5 degC",
            value=5.431109,
            unit="angstrom",
            expanded_uncertainty=0.000080,
            coverage_factor=2.0,
            reference_temperature_c=22.5,
            notes=(
                "Converted from the certificate value 0.5431109 nm with an "
                "expanded uncertainty interval of +/-0.000008 nm."
            ),
        )
    },
    notes=(
        "The certificate's tabulated Cu Kalpha peak positions are non-certified values; "
        "they must not be represented as SI-traceable certified properties.",
        "Use the certificate applicable to the physical SRM unit and record its identity.",
    ),
)


NIST_SRM_660C = CertifiedStandard(
    standard_id="NIST-SRM-660c",
    title="Line Position and Line Shape Standard for Powder Diffraction",
    material="Lanthanum hexaboride (11B-enriched LaB6) powder",
    provider="National Institute of Standards and Technology",
    certificate_url="https://tsapps.nist.gov/srmext/certificates/660c.pdf",
    certificate_revision="Certificate verified 2026-08-11",
    intended_uses=(
        "line position calibration",
        "line shape calibration",
        "instrument profile function determination",
    ),
    properties={
        "lattice_parameter_a": CertifiedProperty(
            property_id="lattice_parameter_a",
            name="Cubic lattice parameter a at 22.5 degC",
            value=4.156826,
            unit="angstrom",
            expanded_uncertainty=0.000080,
            coverage_factor=2.0,
            reference_temperature_c=22.5,
            notes=(
                "Converted from the certificate value 0.4156826 nm with an "
                "expanded uncertainty interval of +/-0.000008 nm."
            ),
        )
    },
    notes=(
        "The material was engineered to minimize size and strain broadening for IPF work.",
        "Use the certificate applicable to the physical SRM unit and record its identity.",
    ),
)


NIST_SRM_674B = CertifiedStandard(
    standard_id="NIST-SRM-674b",
    title="X-Ray Powder Diffraction Intensity Set",
    material="ZnO, TiO2, Cr2O3, and CeO2 powder set",
    provider="National Institute of Standards and Technology",
    certificate_url="https://tsapps.nist.gov/srmext/certificates/674b.pdf",
    certificate_revision="Certificate verified 2026-08-11",
    intended_uses=(
        "quantitative powder diffraction method validation",
        "phase-purity and lattice-parameter reference work",
    ),
    properties={},
    notes=(
        "Certified values are material- and certificate-specific and are intentionally "
        "not duplicated in the built-in catalog. Register the values from the certificate "
        "that accompanies the physical unit before evaluating a claim.",
        "A catalog entry without certified properties can document intended use but cannot pass a claim gate.",
    ),
)


BUILTIN_STANDARDS = {
    row.standard_id: row
    for row in (NIST_SRM_640G, NIST_SRM_660C, NIST_SRM_674B)
}


class StandardCatalogue:
    """Versioned in-memory catalog with explicit support for lab-specific standards."""

    def __init__(self, standards: Iterable[CertifiedStandard] = ()) -> None:
        self._standards = dict(BUILTIN_STANDARDS)
        for standard in standards:
            self.register(standard)

    def register(self, standard: CertifiedStandard, *, replace_existing: bool = False) -> None:
        if not isinstance(standard, CertifiedStandard):
            raise MetrologyValidationError("standard must be a CertifiedStandard")
        if standard.standard_id in self._standards and not replace_existing:
            raise MetrologyValidationError(
                f"standard already registered: {standard.standard_id}"
            )
        self._standards[standard.standard_id] = standard

    def get(self, standard_id: str) -> CertifiedStandard:
        try:
            standard = self._standards[str(standard_id)]
        except KeyError as exc:
            raise MetrologyValidationError(f"unknown certified standard: {standard_id}") from exc
        # Return a structurally independent property mapping so callers cannot mutate
        # the process-wide built-in catalog accidentally.
        return replace(standard, properties=dict(standard.properties))

    def all(self) -> tuple[CertifiedStandard, ...]:
        return tuple(self.get(key) for key in sorted(self._standards))

    def to_dict(self) -> dict[str, dict]:
        return {row.standard_id: row.to_dict() for row in self.all()}
