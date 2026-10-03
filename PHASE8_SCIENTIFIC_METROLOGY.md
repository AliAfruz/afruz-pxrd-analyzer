# Phase 8 — Scientific Ground Truth and Metrology

Release: 22.6.0

Status: Implemented
Scientific boundary: existing preprocessing, peak fitting, calibration, cell, Rietveld, and QPA numerical algorithms are unchanged.

## Purpose

Phase 8 separates four kinds of evidence that must not be presented as equivalent:

1. **Software regression** — verifies deterministic software behavior.
2. **Reference-engine comparison** — verifies agreement with a named, versioned engine.
3. **Certified reference material** — compares a measurement with a certified property and uncertainty.
4. **Experimental validation** — evaluates a laboratory method using traceable experimental reference values and a declared design.

A passing regression or engine comparison does not demonstrate experimental accuracy. A campaign with good summary statistics does not become publication evidence unless it is linked to passing certified-reference or experimental metrology evidence.

## Public API

The GUI-independent API is exposed by `afruz_pxrd.metrology` and by `ProjectController`.

```python
from afruz_pxrd.metrology import (
    AcceptanceCriterion,
    AcquisitionMetadata,
    ClaimScope,
    MetricObservation,
    dataset_data_signature,
)

metadata = AcquisitionMetadata(
    instrument_id="LAB-XRD-01",
    instrument_name="Laboratory diffractometer",
    geometry="Bragg-Brentano symmetric",
    radiation="Cu Kalpha1",
    wavelength_angstrom=1.5405929,
    detector="Silicon strip detector",
    optics="Fixed divergence and receiving slits",
    step_size_deg=0.02,
    counting_time_seconds=1.5,
    temperature_c=22.5,
    specimen_preparation="Back-loaded and rotated powder",
    specimen_geometry="Flat plate",
    operator="Operator identity",
    acquired_at="2026-08-11T10:00:00+00:00",
    source_data_signature=dataset_data_signature(dataset.x, dataset.y_raw),
)
controller.set_acquisition_metadata(dataset.uid, metadata)

assessment = controller.assess_certified_standard(
    dataset.uid,
    "NIST-SRM-640g",
    [MetricObservation(
        metric="lattice_parameter_a",
        value=5.43112,
        unit="angstrom",
        standard_uncertainty=0.00005,
        method="Weighted cubic unit-cell refinement",
    )],
    [AcceptanceCriterion(
        metric="lattice_parameter_a",
        absolute_tolerance=0.00020,
        maximum_en=1.0,
        require_measurement_uncertainty=True,
    )],
    material_unit_id="Physical SRM unit / certificate identity",
)

claim = controller.authorize_scientific_claim(
    assessment.assessment_id,
    ClaimScope.MEASUREMENT_ACCURACY,
    "The lattice parameter agrees with the certified value under the stated rule.",
)
```

Claim authorization fails if the assessment did not pass, its evidence class cannot support the requested scope, its tolerances were not explicit, or its evidence became stale.

## Acquisition record

Experimental and certified-reference evidence requires:

- instrument ID and name;
- measurement geometry;
- radiation and wavelength;
- detector and optics;
- step size and counting time;
- acquisition temperature;
- specimen preparation and geometry;
- operator and acquisition time; and
- a SHA-256 identity calculated from the exact raw X/Y arrays.

`ProjectController.set_acquisition_metadata()` verifies the data identity before recording the metadata. Replacing the acquisition record invalidates all previous assessments and authorized claims for that dataset. Duplicating a dataset does not duplicate acquisition evidence or validation claims.

## Built-in reference-material catalog

The catalog includes the following certificate snapshots verified on 2026-08-11:

| Standard | Certified property included | Intended use |
|---|---|---|
| NIST SRM 640g silicon | `a = 5.431109 Å`, expanded uncertainty `0.000080 Å`, `k = 2`, 22.5 °C | Line position and laboratory IPF evaluation |
| NIST SRM 660c LaB6 | `a = 4.156826 Å`, expanded uncertainty `0.000080 Å`, `k = 2`, 22.5 °C | Line position, line shape, and IPF determination |
| NIST SRM 674b oxide set | No duplicated values | QPA method validation; values must be registered from the applicable physical-unit certificate |

The 640g certificate's calculated Cu Kalpha peak positions are non-certified. They are not stored as certified properties. A 674b catalog entry documents intended use but cannot pass a certified-value claim until the applicable certificate values are registered explicitly.

Primary sources:

- NIST SRM 640g certificate: <https://tsapps.nist.gov/srmext/certificates/640g.pdf>
- NIST SP 260-245 certification report: <https://doi.org/10.6028/NIST.SP.260-245>
- NIST SRM 660c certificate: <https://tsapps.nist.gov/srmext/certificates/660c.pdf>
- NIST SRM 660c certification report: <https://www.nist.gov/publications/certification-standard-reference-material-660c-powder-diffraction>
- NIST SRM 674b certificate: <https://tsapps.nist.gov/srmext/certificates/674b.pdf>
- IUCr Rietveld refinement guidelines: <https://journals.iucr.org/j/issues/1999/01/00/gl0561/>

Researchers must confirm that the built-in snapshot matches the certificate accompanying the physical material. The material-unit/certificate identity is mandatory for a certified assessment.

## Acceptance evaluation

Each measured metric must have exactly one matching reference and one pre-declared criterion. Supported criteria are:

- maximum absolute error;
- maximum relative error;
- maximum normalized error number, `En`; and
- mandatory measurement uncertainty.

`En` is calculated from the measured and reference standard uncertainties using a common coverage factor of two for the combined expanded-uncertainty denominator. If an `En` rule is requested and measurement uncertainty is absent, the check fails rather than silently falling back to an unweighted comparison.

Certified properties that specify a reference temperature enter review status when acquisition temperature differs by more than 1 °C. A documented physical correction and new assessment are then required.

## Claim scopes

Evidence can authorize only these scopes:

| Evidence class | Allowed validated claim |
|---|---|
| Software regression | Software behavior |
| Reference-engine comparison | Engine agreement |
| Certified reference material | Instrument performance; measurement accuracy |
| Experimental validation | Measurement accuracy; method validation |
| Unclassified | None |

This prevents engine agreement or deterministic examples from being presented as certified measurement accuracy.

## Persistence and export

The project-owned `MetrologyRegistry` stores:

- acquisition records;
- immutable assessment snapshots;
- certificate values and source URLs used by each assessment;
- native and reference engine names and versions;
- acceptance criteria and per-check decisions; and
- explicit authorized claims with assessment fingerprints.

The registry is saved under `analysis_state.metrology_registry` in `.afz`, included in dataset/project export snapshots, included in undo/redo snapshots, pruned when a dataset is removed, and validated before an opened project replaces live state.

## Completion evidence

Phase 8 tests verify:

- exact built-in certified values and value-status handling;
- source-array identity checks;
- pass, fail, missing-uncertainty, and temperature-review paths;
- normalized-error evaluation;
- evidence-class claim restrictions;
- rejection of stale or corrupted claims;
- acquisition-change invalidation;
- versioned native/reference-engine comparisons;
- `.afz` round-trip and duplication behavior; and
- separation of synthetic regression campaigns from publication evidence.
