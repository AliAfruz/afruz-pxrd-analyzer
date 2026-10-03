# Phase 5 — Typed scientific result contracts

Release: **22.3.0**

Contract schema: **1**

Afruz now stores each primary scientific result in a versioned contract. Legacy
workspace dictionaries remain in project files for user-interface compatibility,
but they are no longer the public interchange format used by exporters.

## Shared envelope

Every serialized contract has exactly these required top-level fields:

| Field | Meaning |
|---|---|
| `kind` | Stable result-family identifier |
| `schema_version` | Version of the result contract |
| `result_id` | Unique identity of this result revision |
| `dataset_id` | UID of the source diffraction dataset |
| `engine` | Required `name` and `version` identity |
| `input_signature` | Stable digest of the inputs used to create the result |
| `parameters` | Documented engine settings |
| `numerical_outputs` | Result-family payload described below |
| `uncertainties` | Named uncertainty values or arrays |
| `warnings` | Ordered warning messages |
| `validation_status` | `Not validated`, `Valid`, `Warning`, `Invalid`, or `Experimental` |
| `created_at` | ISO-8601 creation time |
| `provenance` | Source result IDs, dependency signatures, and notes |

NumPy arrays serialize as objects containing `__ndarray__`, `dtype`, `shape`, and
`values`. The loader validates the declared shape and recreates the array without
using pickle.

## Result families

| `kind` | Required `numerical_outputs` fields |
|---|---|
| `preprocessing` | `processed_profile`, `background_profile`, `smoothed_profile`, `settings` |
| `peak_list` | `peaks`, `locked`, `metadata` |
| `peak_fitting` | `groups`, `candidates` |
| `instrument_calibration` | `profile`, `qa_result` |
| `unit_cell_refinement` | `initial_cell`, `refined_cell`, `matches`, `statistics` |
| `phase_identification` | `candidates`, `selected_candidate`, `matches` |
| `whole_pattern_refinement` | `observed_x`, `observed_y`, `calculated_y`, `background_y`, `statistics` |
| `rietveld_refinement` | whole-pattern fields plus `phases` |
| `qpa` | `mode`, `phases`, `fractions`, `statistics` |
| `validation` | `evidence`, `summary` |

Peak entries require `position_deg`, `intensity`, `fwhm_deg`, `included`,
`origin`, and `metadata`.

## Validation boundary

Contracts are created and validated at the central scientific-state synchronization
boundary. Required keys are checked when loading. Numerical profiles must be
one-dimensional, finite, and mutually aligned when populated. Store entries must
match their result kind and dataset UID, and orphaned entries are rejected.

Use `to_dict()` and `from_dict()` for public serialization. Constructing ad-hoc
nested dictionaries for exporters is not supported.

## `.afz` compatibility and migration

New projects save the store under `analysis_state.typed_result_contracts` while
retaining legacy workspace state for the current GUI. When a version 1 or version 2
project has no typed store, `ProjectController.open()` converts available legacy
preprocessing, peaks, fits, calibration, cell, phase, whole-pattern, Rietveld, QPA,
and validation dictionaries before exposing the project. Original numerical arrays
and dictionaries are not rewritten during loading.

The migration is deterministic: an unchanged legacy payload keeps its result ID and
creation time during live synchronization. A changed payload receives a new result
identity and records the previous result ID in provenance.

Unknown contract kinds, unsupported schema versions, missing required fields,
invalid array shapes, non-finite arrays, identity mismatches, and orphaned dataset
references fail immediately with `ContractValidationError`.
