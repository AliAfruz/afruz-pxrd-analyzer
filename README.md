# Afruz PXRD Analyzer 23.0.0

[![Windows tests](https://github.com/AliAfruz/afruz-pxrd-analyzer/actions/workflows/tests.yml/badge.svg)](https://github.com/AliAfruz/afruz-pxrd-analyzer/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23119992.svg)](https://doi.org/10.5281/zenodo.23119992)

Afruz PXRD Analyzer is a Windows-focused PySide6 workbench for powder X-ray diffraction import, preprocessing, peak analysis, crystallographic matching, whole-pattern refinement, quantitative phase analysis, validation, and reproducible reporting.

The application contains both established workflows and explicitly labelled experimental/prototype engines. Synthetic examples verify deterministic software behavior; they are not certified reference materials and do not make a result publication-ready automatically.

## Release and citation

This source package is prepared for a GitHub release and Zenodo archival. Cite
the exact archived software version used in an analysis. Citation metadata are
provided in [`CITATION.cff`](CITATION.cff), while the scientific references for
Pawley, Le Bail, Rietveld, profile, scattering-factor, and preferred-orientation
methods are mapped in [`SCIENTIFIC_REFERENCES.md`](SCIENTIFIC_REFERENCES.md).

Creator order is Ali Afruz (corresponding creator; University of Mohaghegh
Ardabili; ORCID [0000-0002-2969-8428](https://orcid.org/0000-0002-2969-8428)),
followed by Maryam Kaffash Jamshid. The
source repository is [AliAfruz/afruz-pxrd-analyzer](https://github.com/AliAfruz/afruz-pxrd-analyzer).
Release 23.0.0 is archived under the version DOI
[10.5281/zenodo.23119993](https://doi.org/10.5281/zenodo.23119993); the project
concept DOI is [10.5281/zenodo.23119992](https://doi.org/10.5281/zenodo.23119992).
Follow [`PUBLISHING_GUIDE.md`](PUBLISHING_GUIDE.md) for future releases.
The complete release verification is recorded in
[`RELEASE_AUDIT.md`](RELEASE_AUDIT.md), and fixture licensing/provenance is
tracked in [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

The CCDC MIL-101 reference structure CCDC 605510 is supported as a local user
reference but is deliberately not redistributed. See
[`LOCAL_REFERENCE_DATA.md`](LOCAL_REFERENCE_DATA.md) for its DOI, checksum, and
the direct-download/validation procedure.

Licensed under the [MIT License](LICENSE). Scientific output is provided without
a guarantee of fitness for a particular interpretation; users must validate the
instrument model, structural model, constraints, residual profile, uncertainty
model, and chemical plausibility for their own specimens.

## Cinematic interface

The complete desktop workbench now uses the **Aura Night** visual system by
default, with animated spectral lighting, a moving workflow indicator, an orbital
brand mark, vector action icons, restrained hover glow, smooth tab transitions and
an animated progress drawer. **Aura Pearl** provides the same hierarchy in a bright
laboratory palette. The original **Dark Gold** and **Light Gold** themes remain
available from **View → Theme** for existing preferences and saved projects.

Choose **View → Reduce Motion** to stop decorative motion while preserving every
control and status indicator. Themes and animation affect presentation only; plots,
refinement calculations, exports, provenance and saved scientific results keep their
existing numerical behavior.

The desktop starts maximized and the left project/workflow column scrolls when the
available height is smaller than its content, preventing guidance cards and project
controls from overlapping. Press **F11** or choose **View → Toggle Full Screen** for
a distraction-free canvas; press **F11** again to return to the maximized window.

Press **Ctrl+K** to open the Command Center. It searches project actions, every
scientific workspace, refinement and visualization tools, exports, themes and view
controls from one keyboard-first launcher. **Ctrl+Shift+L** toggles the workflow
sidebar and **Ctrl+Shift+R** toggles the contextual analysis sidebar when more plot
space is needed.

## 3D crystal figures

Open **3D Crystal Studio…** in the Rietveld workspace to render saved structures
with cinematic lighting, coordination polyhedra and high-resolution PNG/TIFF
exports. See [Crystal Studio](CRYSTAL_STUDIO.md) for controls, reproducible scene
files and the distinction between refined cell parameters and fixed CIF sites.
The **Scientific • polyhedra + H-bonds** preset adds conventional element colors,
multi-metal coordination polyhedra, a vertical legend, an outer cell box and dashed
geometric hydrogen bonds in the style of crystallographic structure figures. It
completes coordination across periodic boundaries and suppresses inferred metal–metal
contacts by default. **Run Crystal Chemistry QA…** reports periodic coordination,
polyhedral distortion, Fe/Ni bond-valence sums, formal charge balance and possible
hidden symmetry. Every Studio figure export includes full JSON and readable text QA
companions beside the image. Weak/invalid CIF geometry is marked review-required and
needs an explicit expert override for refinement or figure export; the warning stays attached.
The validator distinguishes bonded 1–3 and geometric hydrogen-bond contacts from true
nonbonded overlaps, and P1 exports fill missing formula metadata from occupied sites.

## Central application state

Release 22.2.0 separates project data from the Qt window. `ProjectState` owns datasets, preprocessing products, peak lists, refinements, QPA results, and unified scientific provenance. `ApplicationState` owns the theme, active workspace, workflow navigation, and UI preferences. `ProjectController` provides GUI-independent new/open/save and dataset commands, and undo/redo now snapshots this central state.

The main window retains compatibility properties for established analysis widgets, so existing calculations and the version 1/version 2 `.afz` loader remain unchanged. New integrations should use `afruz_pxrd.application` instead of mutating the window directly.

## Modular GUI architecture

Release 22.2.1 splits the former 9,409-line `app.py` into a small compatibility entry point, a construction-focused `main_window.py`, application controllers, cohesive workspace modules, and GUI-independent import/export/report services. Scientific workspaces are exposed through a common lifecycle interface and a `WorkspaceRegistry`; workflow activation, execution, dataset changes, and central refreshes route through controllers instead of hand-written widget import/refresh chains.

Release 22.3.0 adds versioned, typed contracts for every primary scientific result family. The shared envelope records result and dataset identity, engine/version, input signature, parameters, numerical outputs, uncertainties, warnings, validation status, creation time, and provenance. Live workspaces populate the contracts at the central scientific-state boundary, old `.afz` dictionaries migrate automatically during loading, and exporters consume the documented contracts while retaining backward-compatible project fields. See `PHASE5_TYPED_RESULTS.md` for the public schemas and migration policy.

## Scientific engine boundary

Release 22.4.0 exposes all ten primary native analysis families through one GUI-independent engine API. `EngineRequest` provides a versioned, signed input envelope; `EngineRegistry` owns discovery and per-result defaults; and `ScientificEngineService` supplies consistent validation, progress, cooperative cancellation, execution records, provenance chaining, and typed-result storage. `ProjectController.run_engine()` makes the same service available to desktop, batch, scripted, and future CLI callers without constructing a Qt widget. Existing numerical functions and their regression baselines are unchanged. See `PHASE6_ENGINE_BOUNDARIES.md`.

Release 22.7.0 adds geometry-aware instrument-profile physics and calibration uncertainty propagation. Bragg–Brentano zero shift, specimen displacement, and thick-specimen transparency are separately parameterized; resolved Cu Kα1/Kα2 and monochromated configurations are explicit; axial divergence is tied to the GSAS-style SH/L geometry ratio; and full position/width covariance is retained. Calibration validity ranges now gate cell, size/strain, Pawley/Le Bail, and Rietveld use, while compatible covariance blocks propagate into corrected uncertainties or Gaussian refinement priors. See `PHASE9_INSTRUMENT_PROFILE_PHYSICS.md`.

Release 22.8.0 makes Phase 10 Pawley/Le Bail extraction metrologically explicit. The weighted reflection basis is rank-tested after background projection; active-set intensity covariance, I/s.e., overlap groups, group-sum uncertainty, and individual-identifiability gates are reported and exported. Le Bail uncertainties are clearly labelled as post-extraction linearized approximations. Optional normalized spectral doublets and Phase 9 displacement/transparency corrections are now available without changing legacy monochromatic defaults, and refined cell standard errors retain the nonlinear covariance context. See `PHASE10_WHOLE_PATTERN_METROLOGY.md`.

Release 23.0.0 adds the Phase 12 Doping-Series workspace. It compares two or more ordered XRD patterns with overlay, stacked, heatmap, reference-difference, and peak-evolution views while retaining every original measured grid for refinement. Applied main-workspace baseline correction and smoothing can now feed comparison and the linked main Smart Peak Search, while refinement remains isolated on original intensities. Optional independent or reference-outward sequential Rietveld refinement carries forward only accepted initialization values, applies a separate quality gate to every pattern, isolates failures, tracks structural/profile trends, and exports refined CIF copies plus an auditable manifest. It is a screening and consistency workflow, not proof of dopant site occupancy or phase purity. See `PHASE12_DOPING_SERIES.md`.

Release 22.6.0 added the scientific ground-truth and metrology boundary. Complete acquisition records are tied to exact source-array signatures; certified-standard, reference-engine, experimental-validation, and software-regression evidence are kept distinct; every assessment uses explicit acceptance criteria; and a scientific claim can be marked validated only through a passing, scope-compatible assessment. See `PHASE8_SCIENTIFIC_METROLOGY.md`.

Existing widget APIs remain available during the transition. New workspace integrations should implement `set_project_state()`, `set_dataset()`, `validate_inputs()`, `run_analysis()`, and `refresh_results()` through `WorkspaceAdapter` or the same contract.

## Requirements

- Windows 10 or Windows 11
- 64-bit CPython 3.12
- Approximately 2 GB free disk space for the environment and generated reports
- A display supporting at least 1280 × 720; 1920 × 1080 is recommended

The checked baseline uses Python 3.12.13. Runtime and development packages are pinned in `requirements-lock.txt` and `requirements-dev-lock.txt`.

## Install and start on Windows

Open PowerShell in this directory and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1
.\run_windows.bat
```

The setup command creates `.venv`, finds a compatible Python 3.12 interpreter, and installs the locked dependencies. The launcher verifies Python and the required scientific/GUI packages before opening the application.

If Python 3.12 is installed in a custom location:

```powershell
$env:AFRUZ_PYTHON = "C:\Path\To\Python312\python.exe"
powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1
```

If startup checks fail, rerun the bootstrap command. Do not install packages manually into the locked environment unless you also update and verify the lock file.

## First NaCl project

1. Start the application with `.\run_windows.bat`.
2. Choose **Import** and open `Afruz_PXRD_Statistical_Validation_Set\01_NaCl_raw_counts_with_sigma.txt`.
3. Review the raw pattern and its imported intensity/sigma metadata.
4. In Pattern Preparation, select background and smoothing settings, then apply them.
5. Detect and review peaks before phase identification or refinement.
6. Import `NaCl_validation_P1.cif` when a crystallographic reference is required.
7. Run the relevant refinement and inspect warnings, statistical validity, and scientific-state status.
8. Save the project as an `.afz` file and export a Complete Report Package.

The supplied NaCl/CsCl datasets are small deterministic regression examples. For experimental work, verify wavelength, instrument profile, counting-statistics provenance, phase models, preferred orientation, absorption, and uncertainty assumptions.

## Project safety and recovery

Afruz projects use the `.afz` extension. Each file is a ZIP archive containing:

- `project.json`: project schema, interface state, analysis state, and provenance
- `datasets.npz`: compressed numerical arrays

Saving follows this sequence:

```text
write same-folder temporary archive
→ validate JSON, schema, arrays, and ZIP CRC
→ preserve the current project as project.afz.bak
→ atomically replace project.afz
```

The previous project remains untouched until the final atomic replacement. If the main project is damaged or a save is interrupted, keep both files, rename a copy of `project.afz.bak` to an `.afz` filename, and open that copy. Version 1 projects are migrated in memory; newly saved projects use schema version 2.

The loader rejects incomplete archives, duplicate required members, invalid JSON, unsupported versions, unsafe NPZ object arrays, missing dataset arrays, and misaligned profiles with an actionable error.

## Supported inputs

- Text patterns: TXT, CSV, XY, DAT
- Malvern Panalytical XRDML
- Binary V3 RD scans
- Jade/PDF/JCPDS-style reference-card text
- CIF structures for calculated reference patterns and structure-constrained workflows

Some import formats have vendor variations. Preserve the original input and confirm units, wavelength, intensity meaning, and metadata after import.

## Major workflows

- Background estimation and peak-protected smoothing
- Automatic/manual peak detection and multi-profile fitting
- Instrument calibration, crystallite size, microstrain, and residual stress
- CIF parsing, calculated patterns, phase identification, and unit-cell refinement
- Pawley/Le Bail and native Rietveld refinement
- Exploratory RIR and structure-constrained Hill–Howard QPA
- Optional GSAS-II integration
- Validation campaigns, batch recipes, plots, and complete report packages

Scientific-state tracking marks dependent results `Outdated` when upstream preparation, peaks, phases, or refinement inputs change. Review and accept current results before reporting them.

## GSAS-II integration

GSAS-II is optional and is not bundled by the Python bootstrap. Configure its Python executable or parent directory through the GSAS-II settings dialog. Native Afruz engines remain available without GSAS-II. Record the external engine version in validation/report artifacts when using it.

## Development and tests

Create the exact development environment and run the full suite:

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1 -Development
.\run_tests.bat
```

The suite covers module integrity, imports/exports, `.afz` migrations and corruption handling, preprocessing, fitting, crystallography, phase identification, refinement statistics, Rietveld/Pawley baselines, QPA, state invalidation, undo/redo, and GUI startup/shutdown.

Coverage and JUnit artifacts are written below the ignored `test_artifacts\` directory. GitHub Actions runs the same locked suite on Python 3.12 for pushes and pull requests.

## Repository guide

- `afruz_pxrd\`: application and scientific engines
- `afruz_pxrd\application\`: typed central state and GUI-independent project controller
- `afruz_pxrd\workspaces\`: workspace interface, registry, adapters, and cohesive GUI modules
- `afruz_pxrd\services\`: GUI-independent import, export, and report boundaries
- `tests\`: automated safety net
- `Afruz_PXRD_Synthetic_Test_Set\`: clean and mismatch synthetic examples
- `Afruz_PXRD_Statistical_Validation_Set\`: raw/normalized/artifact validation examples
- `baseline_artifacts\`: preserved Phase 0 numerical, graphical, export, and timing evidence
- `PHASE0_BASELINE.md`: baseline procedure and scientific boundary
- `PHASE1_TESTING.md`: automated test matrix and tolerances
- `PHASE2_PROJECT_SAFETY.md`: release-identity and project-safety implementation evidence
- `PHASE3_CENTRAL_STATE.md`: central-state architecture, compatibility contract, and verification evidence
- `PHASE4_MODULAR_GUI.md`: modular window/workspace architecture and verification evidence

## Reporting problems

Preserve the original input, `.afz`/`.afz.bak` pair, exact error text, Python version, and steps that caused the problem. Run `.\run_tests.bat` and include its summary. Never overwrite the only copy of a damaged project while investigating it.
