# Phase 4 — Split the main window

## Release

- Application: Afruz PXRD Analyzer
- Version: `22.2.1`
- Project schema written: `2`
- Project schemas readable: `1`, `2`

## Result

The former 9,409-line `afruz_pxrd.app` module is now a seven-line compatibility import surface. The concrete `MainWindow` is in `main_window.py` and contains construction, connection, theme, and application-shell responsibilities in 1,039 lines. A mechanical source-preserving extraction moved 183 established methods into cohesive modules without changing their calculations.

## Architecture

```text
afruz_pxrd/
  app.py                         stable compatibility import
  main_window.py                 construction and signal connection
  main_window_dependencies.py    shared legacy dependency boundary

  application/
    state.py
    project_controller.py
    workflow_controller.py
    workflow_window.py
    project_window.py
    analysis_panels.py

  workspaces/
    base.py
    registry.py
    workspace_shell.py
    preparation_workspace.py
    peaks_workspace.py
    phase_workspace.py
    refinement_workspace.py
    validation_workspace.py

  services/
    import_service.py
    export_service.py
    report_service.py
```

Built-in widget classes are lazily imported by `WorkspaceRegistry`; neither `main_window.py`, the shared dependency module, nor the workspace shell imports each widget class manually.

## Workspace contract

Every registered workspace is exposed through the same lifecycle:

```python
set_project_state(state)
set_dataset(dataset)
validate_inputs()
run_analysis()
refresh_results()
```

`WorkspaceAdapter` provides lifecycle signals for validation, execution, completion, and result refresh. Existing tested widgets are adapted without requiring an immediate rewrite. Preparation, peaks, phase, refinement, and validation each expose an independently constructible workspace adapter type.

`WorkflowController` owns active workspace/task and mode transitions. Main-window workflow execution, dataset propagation, and central refresh paths now route through the controller and registry.

## Service boundaries

- `ImportService` loads multiple paths and returns typed successes/failures without showing dialogs.
- `ExportService` owns dataset and plot export entry points.
- `ReportService` owns complete-package construction and archiving.

The window retains file dialogs and user messages; the work itself can be tested without constructing `MainWindow`.

## Compatibility

- `from afruz_pxrd.app import MainWindow, run` remains supported.
- Established workspace attributes remain available to existing plugins and exporters.
- Project schema and v1/v2 `.afz` migration behavior are unchanged.
- Phase 3 central state and undo/redo remain the authoritative project model.
- No scientific engine, input parser, numerical tolerance, or result schema changed.

## Verification result

Verified on 2026-08-11 with the locked CPython 3.12 environment on Windows:

- `79 passed in 76.88s`
- `23 passed in 31.43s` for the focused modular GUI, GUI smoke, central-state, and module-integrity suite
- All 119 application/entry Python sources compiled
- Every package module imported without circular-import failures
- Every workflow task was present in the workspace registry
- All five workspace types passed the common-interface tests independently
- Import, export, report, registry, workflow-controller, and lifecycle-signal tests passed
- Phase 0 NaCl/NaCl–CsCl, Pawley, Rietveld, QPA, preprocessing, crystallography, fitting, project migration, and atomic-save regressions remained green
