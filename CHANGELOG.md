# Changelog

All notable user-visible changes are recorded here. Versions follow semantic
versioning where practical.

## [23.5.1] — 2026-10-16 (local development)

### Fixed

- Le Bail initial integrated intensities now use a weighted scale of the sampled
  reference profile, removing the erroneous inverse-step factor from an
  index-domain trapezoidal sum. Raw/normalized observations are not rescaled.
- Observed-area repartition no longer applies square-root least-squares weights
  or clips every negative background-subtracted point to zero. Signed net counts
  are integrated before imposing nonnegative reflection intensities; background
  and outer nonlinear fits retain their original statistical weights.
- Removed the arbitrary ratio ceiling and synchronized the final background
  with the final intensity cycle. CPU and CUDA use the same extraction routine.
- Added cooperative inner-cycle cancellation, extraction revision/initialization
  diagnostics, and an explicit warning when the requested cycles have not converged.
- Applies to regular, step-wise, exclusion and series Le Bail workflows through
  their shared engine. Pawley/Rietveld calculations and stored results are unchanged.
- Added NaCl, sampling/normalization, signed-noise, overlap, background and
  CPU/CUDA regression coverage. No manuscript edits or remote publication.

## [23.5.0] — 2026-10-07 (local development)

### Added

- Editable glTF 2.0 binary (GLB) export from GPU Interactive Crystal and main
  Crystal Studio, using the same frozen geometry for CIF, refinement and pore views.
- Individually named atom/contact/pore objects, shared compact sphere/cylinder
  meshes, editable polyhedral surfaces grouped by element, H-bond and cell guides.
- Sphere quality, geometry selection, material opacity/color, worker progress,
  occupancy/site metadata and explicit Ångström/graphical-scale conventions.
- Source/refinement provenance, reusable scene and Blender guide sidecars;
  overwritten-file backups, nonfinite geometry rejection and structure-review gate.
- Existing CPU/GPU image-rendering geometry, pore algorithms and refinement
  calculations are unchanged. No GitHub/Zenodo publication is included.

## [23.4.0] — 2026-10-05 (local development)

### Added

- Shared per-dataset background selection in the regular Pawley/Le Bail/Rietveld
  workspaces, their exclusion investigation windows, and step-wise Refinement Studio.
- A separate background editor reusing all six existing detection methods,
  their full controls, protected-peak preview, diagnostics and ensemble comparison.
  Preview preparation runs in the background with progress indication.
- Import an existing main Background preview, or choose it directly using
  **Use for all refinements** in Pattern Preparation. AFZ retains the raw-unit
  curve and source fingerprint. Changing raw data invalidates stale selections.
- Native refinement of a smooth signed correction above the chosen baseline,
  bounded so the total background remains nonnegative, using the existing CPU/CUDA
  solvers. Raw observations/weights stay unchanged; processed input is bypassed
  to prevent double subtraction. Ordinary refinement remains unchanged by default.
- Frozen background settings for queued attempts; later choices cannot alter
  pending jobs or prior results. Plots show starting and total backgrounds, and
  TXT profile exports add starting-background and correction columns.
- Adaptive-background uncertainty/effective degrees of freedom are not propagated:
  Rwp/Rp remain available, but Rexp/GoF/reduced chi-square are gated. Held-out
  diagnostics using a full-data starting curve are marked as conditional.
- Local development only; no GitHub/Zenodo update.

### Fixed

- The standalone Rietveld prediction helper now initializes its axial-divergence
  profile parameter and can reconstruct a chosen background plus signed correction.

## [23.3.2] — 2026-10-05 (local development)

### Added

- **Open Crystal Studio** in the detached Refinement Studio's History panel,
  using the selected completed run's saved fitted cell and frozen atomic model.
  Run ID, method, lineage, fit domain and diagnostics accompany scene provenance.
- Pawley and Le Bail cell-only views explicitly retain fixed CIF sites. Completed
  non-converged fits carry warnings; missing snapshots never fall back to a live CIF.
- A separate **Preview candidate crystal** action under Steps for initial or
  parent-seeded models, clearly distinguished from completed refinement results.
- Background crystal model preparation with a visible busy indicator, protecting
  the XRD workspace from long validation of large framework snapshots.
- Existing Crystal Studio CPU/GPU/pore rendering, validation and export are
  reused unchanged. Numerical engines and original refinement workspaces are
  unchanged. This is a local update only; no GitHub or Zenodo release.

## [23.3.1] — 2026-10-04 (local development)

### Changed

- Moved the step-wise exclusion laboratory out of the crowded main workspace
  into a persistent, non-modal **Step-wise Refinement Studio** window.
- Added a full-height central XRD canvas, compact Steps/CIFs/Settings/Zones
  controls on the left, and History/Reflections on the right. Side panels
  scroll independently; queue controls remain pinned and visible.
- Added maximization/restoration, splitter resizing, Reset view, direct AFZ
  saving and history export. AFZ retains panel widths and active control tabs.
- Changed the workflow and Analysis-menu actions to open the Studio, not start
  an invisible refinement. Opening it no longer needs a global phase/CIF
  selection; numerical commands still validate their own inputs normally.
- The window reuses the same project-owned laboratory. Closing/Escape hides
  it without losing, duplicating or cancelling active jobs and saved history.
- Numerical engines, established refiners and Crystal Studio are unchanged.
  The previous installed application was backed up and verified. No GitHub
  or Zenodo publication is part of this local patch.

## [23.3.0] — 2026-10-04 (local development)

### Added

- A scrollable **Step-wise queue** tab inside the existing exclusion laboratory,
  with checked CIF candidates, editable labels, explicit order and a shared
  Pawley, Le Bail or Rietveld method per queued batch.
- Serial background execution, pause-after-current, cooperative cancellation,
  audited failures and optional continuation after a failed candidate.
- History checkboxes for selecting completed, converged candidates for the
  next step. Follow-up runs preserve parent IDs and initialize from valid
  refined cells/profile parameters without altering fixed CIF atom sites.
- Frozen measurement/model/settings/mask snapshots, stable IDs, matched-domain
  signatures, AFZ pending-job resume and deduplicated large model snapshots.
- Full-history ZIP export now includes the step plan, every attempted run's
  provenance, frozen input and model, and clean completed-refinement TXT data.
- Wrapped reflection summaries and bounded progress text so long labels do
  not squeeze the history panel; history IDs remain visible beside selection.

### Scientific and release boundary

- Exactly one CIF is fitted per run. Candidate selection is not a simultaneous
  multiphase fit, quantitative phase analysis or subtraction of a fitted phase.
- Shared ranges/exclusions are the default. Parent-specific masks can be
  inherited deliberately; R factors from differing fit domains must not be
  directly ranked as if they used identical observations.
- Original numerical engines, production refiners, Crystal Studio and the
  restored overall interface are unchanged. No GitHub/Zenodo publication.

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
