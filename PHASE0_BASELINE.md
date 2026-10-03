# Phase 0 — Preserved Current Baseline

## Purpose

This baseline captures the current Afruz PXRD behavior before architectural or scientific changes. The original source directory remains untouched. The `Tunned` copy is maintained on the dedicated `codex/phase0-baseline` Git branch.

## Supported environment

- Python: 3.12.13
- Dependencies: exact versions in `requirements-lock.txt`
- Platform captured: Windows
- Application startup: `.\run_windows.bat`
- Baseline verification: `.\run_baseline.bat`

## Evidence

The baseline harness preserves:

- Environment and package versions
- Source hashes and syntax-check results
- Import summaries for the supplied patterns
- CIF parsing and calculated-reflection summaries
- Raw/normalized statistical reference metrics
- NaCl and NaCl/CsCl Rietveld results and runtimes
- Representative TXT and Excel exports
- `.afz` project round-trip evidence
- Main-window and refinement PNG screenshots
- JSON, Markdown and HTML baseline reports

Generated evidence is stored in `baseline_artifacts/` and is intentionally kept in Git. The local `.venv` and Python cache files are ignored.

## Change boundary

Phase 0 may add setup, launch, evidence and repository files. It must not change files under `afruz_pxrd/` or alter the supplied scientific datasets. A manifest records the hashes of the scientific source files so later phases can detect changes.

## Known limitations

The historical pytest cache lists 243 test node IDs, but the corresponding `tests/` source directory was not present in the copied project. Phase 0 therefore records repeatable smoke, numerical-reference and end-to-end refinement checks. Full unit-test restoration belongs to Phase 1.

The preserved `MainWindow.closeEvent()` calls `ValidatedQPAWidget.is_running()` and `cancel_backend()`, but those methods are not implemented by the current widget. A native smoke-window close therefore raises an `AttributeError`. The Phase 0 harness captures the startup/UI evidence and disposes its temporary window without changing this application defect. It should be corrected and regression-tested in Phase 1.
