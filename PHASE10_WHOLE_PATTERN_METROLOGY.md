# Phase 10 — Whole-pattern extraction metrology

Release: 22.8.0
Status: Implemented

## Goal

Make Pawley and Le Bail whole-pattern results scientifically interpretable when peaks overlap. A visually good profile fit is not treated as proof that every listed reflection intensity is independently measured.

## Implemented scientific boundary

- Pawley intensities retain the non-negative linear least-squares solution and now receive an active-set covariance calculation.
- Le Bail results receive a final-profile linearized covariance that is explicitly labelled as an approximation, not as covariance from the iterative repartition algorithm.
- Le Bail cycle history records relative intensity change and weighted residual sum of squares; the configured cycle count is preserved and a strict convergence flag is reported without silently stopping early.
- The weighted reflection basis is projected away from the polynomial background before singular-value rank and condition diagnostics are calculated.
- Basis-column correlations create reproducible overlap groups. Severe overlaps are gated as `not-individually-identifiable`; group intensity sums and their propagated standard errors remain available.
- Each reflection reports standard error, I/s.e., maximum basis correlation, overlap-group identity, boundary status, and identifiability classification.
- The independent reflection rank replaces the raw reflection count in the approximate fitted-parameter count used for Rexp/GoF degrees of freedom.
- Nonlinear lattice/profile parameters retain covariance, correlation, effective rank, condition number, and standard errors. Lattice standard errors are expanded consistently through crystal-system constraints.

## Radiation and calibrated position physics

The Phase 10 basis may be monochromatic or an explicit primary/secondary doublet. Each component is evaluated at its own Bragg position and the composite column is normalized to unit total area, so an extracted intensity means total integrated intensity across the configured spectrum.

When enabled with an active Phase 9 profile, specimen-displacement and transparency corrections are evaluated at each component angle. The calibration zero shift remains the initial/global zero parameter and is not applied a second time as a fixed geometry correction.

Legacy API calls remain monochromatic and do not apply geometry corrections unless requested.

## Export and persistence

The TXT refinement bundle now includes:

- the enriched reflection table;
- an upper-triangle reflection-intensity covariance/correlation table;
- an overlap-group table;
- the existing profile, phase, statistical-validity, plot-data, and manifest files.

Typed whole-pattern contracts retain reflection records and the metrology payload. Old project payloads continue to load with empty optional Phase 10 fields.

## Scientific interpretation

Pawley integrated intensities are correlated quantities, especially in overlapping regions. The eigenvalue/singular-value spectrum describes how many independent intensity combinations are supported by the pattern; it does not manufacture information for coincident peaks. This implementation follows that distinction and reports group-level information when individual terms are not identifiable.

Primary technical references:

- IUCr good-practice discussion of Pawley intensity extraction and reporting: https://journals.iucr.org/c/issues/2025/10/00/ky3231/index.html
- IUCr treatment of eigenvalue spectra and effective errors for overlapping powder reflections: https://journals.iucr.org/a/issues/2008/01/00/sc5011/index.html
- IUCr description of Le Bail iterative intensity evolution: https://journals.iucr.org/j/issues/2022/04/00/nb5328/index.html
- IUCr analysis of reduced intensity covariance and independently constrained combinations: https://journals.iucr.org/j/issues/2000/05/00/hw0079/index.html

## Completion criteria

- Coincident synthetic reflections are detected as rank deficient and fail the individual-identifiability gate.
- Separated synthetic reflections retain full rank and finite standard errors.
- Doublet basis columns preserve unit total integrated area.
- Geometry correction and residual/calibrated zero shift remain numerically separate.
- Covariance and overlap tables are included in the reproducible export manifest.
- Full historical tests and Phase 0 baseline artifacts remain preserved.
