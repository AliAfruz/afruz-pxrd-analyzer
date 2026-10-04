# Changelog

All notable user-visible changes are recorded here. Versions follow semantic
versioning where practical.

## [23.2.0] — 2026-10-04 (local development)

### Added

- A large central **Sequential Multiphase Refinement and Exclusion Laboratory**
  with Pawley, Le Bail and Rietveld run buttons.
- Multi-CIF import and one explicit active CIF snapshot per refinement.
- Unlimited project-unique refinement IDs, parent/child lineage, frozen result
  history, user labels and continue-from-selected controls.
- Plot-click/drag next-run exclusions with reflection-domain activity auditing,
  GPU-aware native engine execution and complete multi-run ZIP export.
- AFZ persistence for CIF models, completed/failed/cancelled run records,
  settings, exclusions, history selection and next-run counter.

### Scientific boundary

- A masked region is removed from the observed/calculated fit domain for that
  run. Matching CIF reflection centers are marked inactive for audit and plot
  ticks, but crystallographic structure factors are not deleted.
- A better masked fit can guide a phase-impurity investigation but cannot prove
  phase identity, purity or quantitative abundance without an explicit model
  and independent evidence.
- The established production refiners and the three 23.1 separate
  investigation windows remain available and unchanged.
- This local update has not been pushed to GitHub or released on Zenodo.

## [23.1.0] — 2026-10-04 (local development)

### Added

- A large central **Simulated PXRD from CIF** workspace with reflection sticks,
  a display-only broadened pseudo-Voigt preview, GPU-aware calculation, a
  reflection table, and clean TXT export.
- A separate **Phase-Exclusion Investigation** workspace that launches
  independent Pawley, Le Bail, and Rietveld windows without replacing the
  established production refinement tabs.
- Plot-selected interval exclusion and peak-center/half-width exclusion,
  merged-interval auditing, included/excluded point counts, shaded plot bands,
  persistence, and dedicated clean export.

### Scientific boundary

- Excluded points do not contribute to parameter estimation or fit statistics.
  Exclusion can test an impurity or model-deficiency hypothesis, but cannot by
  itself identify, prove, or quantify an impurity phase.
- Full-range and exclusion-aware results must be reported together with every
  excluded interval and its scientific rationale.
- This local update has not been pushed to GitHub or released on Zenodo.

## [23.0.0] — 2026-10-03

### Added

- Doping-series comparison with overlay, stacked, heatmap, difference, and
  peak-evolution views.
- Independent and reference-outward sequential Rietveld workflows with quality
  gates, failure isolation, trend export, and refined CIF copies.
- Crystal Studio rendering, GPU-assisted interactive visualization, MOF-aware
  display presets, pore visualization, structure-quality checks, and
  publication-image export.
- Typed scientific result contracts, provenance tracking, validation evidence,
  and GUI-independent engine boundaries.
- Instrument-profile physics, uncertainty propagation, Pawley/Le Bail
  identifiability diagnostics, and statistical validation fixtures.
- Clean tabular/text export and selectable/copyable scientific tables.

### Scientific scope

- Synthetic fixtures validate deterministic software behavior; they are not
  certified reference materials.
- Refinement outputs require inspection of the observed, calculated, and
  difference profiles and must not be accepted from a scalar residual alone.
- Crystal rendering is a visualization of the supplied/refined crystallographic
  model and does not independently validate chemical correctness.

### Release packaging

- Added MIT license, Citation File Format metadata, Zenodo metadata, PEP 621
  package metadata, contributor guidance, security policy, and publication
  checklist.
