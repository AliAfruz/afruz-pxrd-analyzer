# Phase 6 — Unified scientific engine boundaries

Release: **22.4.0**

Engine API schema: **1**

## Result

Afruz now has one GUI-independent execution boundary for the ten primary result
families introduced in Phase 5. The boundary adapts the established native
numerical functions; it does not change their algorithms, defaults, tolerances,
or numerical outputs.

## Request contract

`EngineRequest` requires:

| Field | Meaning |
|---|---|
| `api_version` | Version of the engine request API |
| `request_id` | Unique request identity |
| `result_kind` | Phase 5 result family to produce |
| `dataset_id` | Source dataset UID |
| `inputs` | Named arrays and scientific input objects |
| `parameters` | Named engine settings |
| `created_at` | ISO-8601 request time |
| `input_signature` | Stable digest of result kind, dataset, inputs, and parameters |

Requests serialize NumPy arrays using the safe Phase 5 array representation. A
missing required input, non-finite or misaligned profile, unsupported parameter,
wrong result kind, or request/result identity mismatch fails at the engine
boundary.

## Registry and native adapters

`create_default_engine_registry()` registers one default native adapter for each
result kind:

| Engine ID | Result kind |
|---|---|
| `afruz.native.preprocessing` | `preprocessing` |
| `afruz.native.peak_list` | `peak_list` |
| `afruz.native.peak_fitting` | `peak_fitting` |
| `afruz.native.instrument_calibration` | `instrument_calibration` |
| `afruz.native.unit_cell_refinement` | `unit_cell_refinement` |
| `afruz.native.phase_identification` | `phase_identification` |
| `afruz.native.whole_pattern_refinement` | `whole_pattern_refinement` |
| `afruz.native.rietveld_refinement` | `rietveld_refinement` |
| `afruz.native.qpa` | `qpa` |
| `afruz.native.validation` | `validation` |

The registry rejects duplicate engine IDs, verifies the protocol and API version,
supports multiple implementations per result kind, and has an explicit default
selection. It contains no Qt dependency.

## Execution lifecycle

`ScientificEngineService` performs the following sequence:

1. Resolve an explicit engine or the registered default.
2. Validate request identity and required inputs.
3. Check cancellation before entering the native engine.
4. Forward progress and cancellation hooks where the native function supports them.
5. Convert the native output to the Phase 5 typed result contract.
6. Verify engine, dataset, result-kind, and input-signature identities.
7. Chain the previous result ID into provenance and store the new typed result.
8. Record completion, failure, or cancellation with elapsed time and progress events.

Failures raise `EngineExecutionFailure` and retain the complete failed execution
record. Failed or cancelled calculations never replace the previous typed result.

## Application integration

Every `ProjectController` owns an engine service. Callers can use
`ProjectController.run_engine()` with a dataset UID, inputs, parameters, optional
engine ID, progress callback, and cancellation check. The controller marks the
project dirty only after a successful run. New/open/undo/redo operations rebind
the service to the current Phase 5 result store, so results cannot leak between
projects or snapshots.

Controller saves always embed the authoritative typed store, including for headless
callers that pass no GUI analysis dictionary. A service-produced contract is retained
during compatibility synchronization until it is explicitly removed or its dataset
is removed; absence of an older workspace dictionary cannot silently discard it.

The direct numerical module APIs remain supported for compatibility. New headless,
batch, GUI, and plugin integrations should enter through `afruz_pxrd.engines`.
