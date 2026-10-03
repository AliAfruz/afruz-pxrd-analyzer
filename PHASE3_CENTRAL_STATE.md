# Phase 3 — Central application state

## Release

- Application: Afruz PXRD Analyzer
- Version: `22.2.0`
- Project schema written: `2`
- Project schemas readable: `1`, `2`

## Architecture

The `afruz_pxrd.application` package is independent of Qt and provides three public types:

- `ProjectState` is the single owner of datasets, background/smoothing products, peak lists, fit products, cell and phase refinement results, QPA results, residual-stress state, active instrument state, and the unified scientific-state registry.
- `ApplicationState` owns theme, active workspace, workflow navigation, and saved UI preferences.
- `ProjectController` owns new/open/save, dataset add/rename/duplicate/remove, dirty-state transitions, snapshots, and undo/redo commands.

`MainWindow` retains compatibility properties for existing analysis widgets, but those properties resolve to the central state objects. The corresponding result collections and navigation fields are not stored in `MainWindow.__dict__`.

## Project-operation contract

- Project opening fully loads and validates an archive before replacing the live state.
- Failed opens leave datasets, results, current path, and dirty state untouched.
- Saving validates central-state dataset invariants and delegates to the Phase 2 atomic `.afz` writer.
- Version 1 and version 2 `.afz` files continue to use the existing validated loader and migration path.
- Dataset duplication assigns a new UID and deep-copies all dataset-scoped central results and scientific provenance.
- Dataset deletion purges all dataset-scoped central results, provenance, and residual-stress observations.

## Undo/redo contract

Undo and redo now snapshot `ProjectState` and `ApplicationState` through `ProjectController`. The GUI adds its widget analysis/UI payload to the same snapshot, preserving the complete Phase 2 behavior while making the core history independently testable without Qt.

## Compatibility

No scientific algorithm, numerical tolerance, input parser, project schema, or report calculation changed in Phase 3. Existing widgets may continue to use the legacy `MainWindow` field names during the transition; new code should call `ProjectController` methods or work with the typed state objects directly.

## Verification result

Verified on 2026-08-11 with the locked CPython 3.12 environment on Windows:

- `69 passed in 52.36s`
- `23 passed in 39.40s` for the focused Phase 3, GUI smoke, release, module-integrity, and project-compatibility suite
- GUI-independent tests cover dataset CRUD, deep-copy/purge behavior, save/open, failed-open preservation, version 1 loading, central snapshots, and undo/redo
- GUI integration verifies legacy scientific fields resolve to central state and are absent from the window instance dictionary
- All 100 application/entry Python sources compiled and all top-level package modules imported
- Phase 0 NaCl, NaCl/CsCl, Pawley, and Rietveld regression values remained within their frozen tolerances
- Phase 2 atomic save, backup, malformed-archive handling, and version 1/version 2 compatibility tests passed
