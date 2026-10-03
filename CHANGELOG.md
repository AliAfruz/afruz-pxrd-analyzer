# Changelog

All notable user-visible changes are recorded here. Versions follow semantic
versioning where practical.

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
