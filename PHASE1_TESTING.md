# Phase 1 — Automated Testing Safety Net

## Goal

Phase 1 restores executable regression protection around the preserved Phase 0 application. The original historical test sources were unavailable; this suite reconstructs the essential contracts from the shipped scientific engines, supplied NaCl/CsCl datasets, cached historical test names and Phase 0 numerical evidence.

## Run commands

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1 -Development
.\run_tests.bat
```

For a concise run without coverage:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Test layers

- Source/module integrity and locked-environment checks
- Dataset validation, seven supplied-pattern imports and clean failure messages
- Project `.afz` round-trip, TXT/Excel export and complete-report/archive generation
- Background, smoothing, peak detection and manual-peak identity
- CIF parsing, calculated patterns, phase identification and cell refinement
- Raw/normalized statistical invariance against the documented reference values
- Pawley and structure-constrained Rietveld numerical regressions
- Hill–Howard QPA audit and publication-readiness boundary
- Scientific-state invalidation, acceptance and guided-workflow prerequisites
- Native Qt main-window construction and shutdown regression

## Numerical tolerances

- Documented profile statistics: absolute tolerance `1e-4`
- Raw/normalized invariance: absolute tolerance `1e-6`
- Unit-cell recovery: absolute tolerance `1e-5 Å`
- Rietveld Rwp baselines: absolute tolerance `0.05 percentage points`
- Pawley Rwp baseline: absolute tolerance `0.10 percentage points`

These are deterministic software-regression thresholds, not experimental uncertainty claims.

## Verified release result

Verified on 2026-08-11 with CPython 3.12.13 on Windows:

- `51 passed in 60.04s`
- Whole-package branch-aware coverage: `35.9%`
- All 96 Python entry/package sources compiled and all package modules imported
- Seven supplied diffraction patterns imported successfully
- NaCl and NaCl/CsCl Phase 0 Rietveld Rwp/Rexp values remained within the frozen tolerances
- The locked development bootstrap completed and `pip check` reported no broken requirements
- JUnit and HTML coverage reports were generated under the ignored `test_artifacts/` directory

The suite exposed one application defect: shutdown called worker methods that the validated-QPA widget does not implement. `MainWindow.closeEvent()` now checks the optional worker interface before cancelling or waiting. No scientific calculation was changed.

## Historical boundary

The Phase 0 evidence preserves 243 historical pytest node IDs, but their source files were absent. Phase 1 does not claim line-for-line restoration of that unavailable suite. It creates an executable replacement around the highest-risk scientific and application boundaries. Additional engine-specific cases can be added incrementally without weakening these baseline contracts.
