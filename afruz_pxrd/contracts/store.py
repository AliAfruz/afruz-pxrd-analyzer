from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
import uuid

import numpy as np

from ..scientific_state import stable_signature
from ..version import APP_VERSION
from .base import (
    CONTRACT_SCHEMA_VERSION,
    ContractValidationError,
    EngineIdentity,
    Provenance,
    ScientificResultContract,
    ValidationStatus,
    utc_now,
)
from .models import (
    InstrumentCalibrationResult,
    PeakFittingResult,
    PeakListResult,
    PeakRecord,
    PhaseIdentificationResult,
    PreprocessingResult,
    QPAResult,
    RietveldRefinementResult,
    UnitCellRefinementResult,
    ValidationResult,
    WholePatternRefinementResult,
)


CONTRACT_TYPES: dict[str, type[ScientificResultContract]] = {
    row.KIND: row
    for row in (
        PreprocessingResult,
        PeakListResult,
        PeakFittingResult,
        InstrumentCalibrationResult,
        UnitCellRefinementResult,
        PhaseIdentificationResult,
        WholePatternRefinementResult,
        RietveldRefinementResult,
        QPAResult,
        ValidationResult,
    )
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.array([], dtype=float)
    return array if array.ndim == 1 else np.array([], dtype=float)


def _first_array(value: Mapping[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        if key in value:
            array = _array(value.get(key))
            if array.size:
                return array
    return np.array([], dtype=float)


def _warnings(payload: Mapping[str, Any]) -> tuple[str, ...]:
    value = payload.get("warnings", [])
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(row) for row in value if str(row).strip())
    warning = payload.get("warning")
    return (str(warning),) if warning else ()


def _validation_status(payload: Mapping[str, Any]) -> ValidationStatus:
    raw = str(payload.get("validation_status", payload.get("status", ""))).lower()
    if raw in {"invalid", "failed", "failure"} or payload.get("success") is False:
        return ValidationStatus.INVALID
    if raw in {"valid", "validated", "pass", "passed", "accepted"}:
        return ValidationStatus.VALID
    if payload.get("publication_ready") is False:
        return ValidationStatus.EXPERIMENTAL
    if payload.get("statistics_valid") is False or _warnings(payload):
        return ValidationStatus.WARNING
    return ValidationStatus.NOT_VALIDATED


def _engine(payload: Mapping[str, Any], kind: str) -> EngineIdentity:
    engine_value = payload.get("engine")
    engine_mapping = _mapping(engine_value)
    name = engine_mapping.get("name") or engine_value
    if isinstance(name, Mapping):
        name = None
    name = name or payload.get("backend") or payload.get("method")
    return EngineIdentity(
        str(name or f"Afruz {kind.replace('_', ' ')}"),
        str(
            engine_mapping.get("version")
            or payload.get("engine_version")
            or payload.get("software_version")
            or APP_VERSION
        ),
    )


def _parameters(payload: Mapping[str, Any]) -> dict[str, Any]:
    settings = payload.get("settings")
    if isinstance(settings, Mapping):
        return deepcopy(dict(settings))
    parameters = payload.get("parameters")
    if isinstance(parameters, Mapping):
        return deepcopy(dict(parameters))
    return {}


def _uncertainties(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        lowered = str(key).lower()
        if any(token in lowered for token in ("uncertainty", "error", "sigma", "standard_error")):
            result[str(key)] = deepcopy(value)
    return result


def _statistics(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "r_squared",
        "rmse",
        "rwp_percent",
        "rp_percent",
        "rexp_percent",
        "goodness_of_fit",
        "reduced_chi_square",
        "weighted_residual_sum_squares",
        "degrees_of_freedom",
        "condition_number",
        "match_score_percent",
        "statistics_valid",
        "statistics_reason",
    )
    result = {key: deepcopy(payload[key]) for key in keys if key in payload}
    if isinstance(payload.get("diagnostics"), Mapping):
        result["diagnostics"] = deepcopy(dict(payload["diagnostics"]))
    return result


def _peak_records(rows: Iterable[Mapping[str, Any]]) -> tuple[PeakRecord, ...]:
    results: list[PeakRecord] = []
    for row in rows:
        position = row.get("position", row.get("position_deg", row.get("two_theta")))
        intensity = row.get("intensity", row.get("height", 0.0))
        if position is None:
            continue
        known = {"position", "position_deg", "two_theta", "intensity", "height", "fwhm", "fwhm_deg", "use", "included", "origin", "method"}
        results.append(
            PeakRecord(
                position_deg=float(position),
                intensity=float(intensity),
                fwhm_deg=(
                    None
                    if row.get("fwhm", row.get("fwhm_deg")) is None
                    else float(row.get("fwhm", row.get("fwhm_deg")))
                ),
                included=bool(row.get("use", row.get("included", True))),
                origin=str(row.get("origin", row.get("method", "Unknown"))),
                metadata={key: deepcopy(value) for key, value in row.items() if key not in known},
            )
        )
    return tuple(results)


def _fraction_map(phases: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, row in enumerate(phases, start=1):
        name = str(
            row.get("phase_name")
            or row.get("reference_name")
            or row.get("name")
            or f"phase_{index}"
        )
        value = next(
            (
                row[key]
                for key in (
                    "weight_fraction_percent",
                    "corrected_fraction_percent",
                    "reported_fraction_percent",
                    "fraction_percent",
                    "fraction",
                )
                if row.get(key) is not None
            ),
            None,
        )
        if value is not None:
            result[name] = float(value)
    return result


def contract_from_legacy(
    kind: str,
    dataset_id: str,
    payload_value: Any,
    *,
    previous: ScientificResultContract | None = None,
) -> ScientificResultContract:
    if kind not in CONTRACT_TYPES:
        raise ContractValidationError(f"Unknown result kind: {kind}")
    payload = _mapping(payload_value)
    if not payload and isinstance(payload_value, (list, tuple)):
        payload = {"rows": list(payload_value)}
    signature = stable_signature({"kind": kind, "dataset_id": dataset_id, "payload": payload_value})
    if previous is not None and previous.input_signature == signature:
        return previous
    common = {
        "dataset_id": str(dataset_id),
        "engine": _engine(payload, kind),
        "input_signature": signature,
        "parameters": _parameters(payload),
        "uncertainties": _uncertainties(payload),
        "warnings": _warnings(payload),
        "validation_status": _validation_status(payload),
        "provenance": Provenance(
            source_result_ids=((previous.result_id,) if previous is not None else ()),
            notes=("Migrated from legacy Afruz result dictionary",),
        ),
    }

    if kind == PreprocessingResult.KIND:
        background = _mapping(payload.get("background"))
        smoothing = _mapping(payload.get("smoothing"))
        return PreprocessingResult(
            **common,
            processed_profile=_array(payload.get("processed")),
            background_profile=_first_array(background, "background", "display_background"),
            smoothed_profile=_first_array(smoothing, "smoothed", "y_smoothed"),
            settings={"background": _parameters(background), "smoothing": _parameters(smoothing)},
        )
    if kind == PeakListResult.KIND:
        rows = _records(payload.get("rows", payload.get("peaks", [])))
        metadata = _mapping(payload.get("meta", payload.get("metadata", {})))
        return PeakListResult(
            **common,
            peaks=_peak_records(rows),
            locked=bool(metadata.get("locked", metadata.get("frozen", False))),
            metadata=metadata,
        )
    if kind == PeakFittingResult.KIND:
        return PeakFittingResult(
            **common,
            groups=tuple(_records(payload.get("groups", []))),
            candidates=tuple(_records(payload.get("candidates", []))),
        )
    if kind == InstrumentCalibrationResult.KIND:
        return InstrumentCalibrationResult(
            **common,
            profile=_mapping(payload.get("profile", payload)),
            qa_result=_mapping(payload.get("qa_result")),
        )
    if kind == UnitCellRefinementResult.KIND:
        return UnitCellRefinementResult(
            **common,
            initial_cell={str(key): float(value) for key, value in _mapping(payload.get("initial_cell")).items()},
            refined_cell={str(key): float(value) for key, value in _mapping(payload.get("refined_cell")).items()},
            matches=tuple(_records(payload.get("matches", []))),
            statistics=_statistics(payload),
        )
    if kind == PhaseIdentificationResult.KIND:
        candidates = _records(payload.get("results", payload.get("candidates", [])))
        selected = payload.get("best", payload.get("selected_candidate"))
        matches = _records(selected.get("matches", [])) if isinstance(selected, Mapping) else []
        return PhaseIdentificationResult(
            **common,
            candidates=tuple(candidates),
            selected_candidate=dict(selected) if isinstance(selected, Mapping) else None,
            matches=tuple(matches),
        )
    if kind in {WholePatternRefinementResult.KIND, RietveldRefinementResult.KIND}:
        result_type = CONTRACT_TYPES[kind]
        kwargs = {
            **common,
            "observed_x": _first_array(payload, "observed_x", "x"),
            "observed_y": _first_array(payload, "observed_y", "observed"),
            "calculated_y": _first_array(payload, "calculated_y", "calculated"),
            "background_y": _first_array(payload, "background_y", "background", "baseline"),
            "statistics": _statistics(payload),
            "reflections": tuple(_records(payload.get("reflections", []))),
            "intensity_metrology": _mapping(
                payload.get("intensity_metrology", {})
            ),
        }
        if kind == RietveldRefinementResult.KIND:
            kwargs["phases"] = tuple(_records(payload.get("phases", [])))
        return result_type(**kwargs)
    if kind == QPAResult.KIND:
        selected = payload
        source = "combined"
        if isinstance(payload.get("validated"), Mapping) and payload["validated"]:
            selected = dict(payload["validated"])
            source = "validated"
        elif isinstance(payload.get("exploratory"), Mapping) and payload["exploratory"]:
            selected = dict(payload["exploratory"])
            source = "exploratory"
        phases = _records(selected.get("phases", []))
        return QPAResult(
            **{**common, "parameters": {**common["parameters"], "source": source}},
            mode=str(selected.get("mode", source)),
            phases=tuple(phases),
            fractions=_fraction_map(phases),
            statistics=_statistics(selected),
        )
    if kind == ValidationResult.KIND:
        evidence = _records(payload.get("rows", payload.get("evidence", [])))
        if not evidence and isinstance(payload_value, (list, tuple)):
            evidence = _records(payload_value)
        return ValidationResult(
            **common,
            evidence=tuple(evidence),
            summary=_mapping(payload.get("summary", payload.get("result", {}))),
        )
    raise ContractValidationError(f"No legacy converter for result kind: {kind}")


@dataclass
class ResultContractStore:
    """Central typed results indexed by contract kind and dataset identity."""

    by_kind: dict[str, dict[str, ScientificResultContract]] = field(default_factory=dict)

    def get(self, kind: str, dataset_id: str) -> ScientificResultContract | None:
        return self.by_kind.get(str(kind), {}).get(str(dataset_id))

    def record(self, contract: ScientificResultContract) -> ScientificResultContract:
        """Store an already typed engine result after strict identity validation."""
        if not isinstance(contract, ScientificResultContract):
            raise ContractValidationError(
                "Result store accepts only ScientificResultContract values"
            )
        kind = contract.KIND
        if kind not in CONTRACT_TYPES or not isinstance(contract, CONTRACT_TYPES[kind]):
            raise ContractValidationError(f"Unknown typed result class for kind: {kind}")
        contract.validate_result()
        self.by_kind.setdefault(kind, {})[contract.dataset_id] = contract
        return contract

    def record_legacy(self, kind: str, dataset_id: str, payload: Any) -> ScientificResultContract:
        previous = self.get(kind, dataset_id)
        contract = contract_from_legacy(kind, dataset_id, payload, previous=previous)
        self.by_kind.setdefault(kind, {})[str(dataset_id)] = contract
        return contract

    def remove(self, kind: str, dataset_id: str) -> None:
        values = self.by_kind.get(str(kind))
        if values is not None:
            values.pop(str(dataset_id), None)
            if not values:
                self.by_kind.pop(str(kind), None)

    def remove_dataset(self, dataset_id: str) -> None:
        for kind in tuple(self.by_kind):
            self.remove(kind, dataset_id)

    def duplicate_dataset(self, source_id: str, target_id: str) -> None:
        for kind, values in self.by_kind.items():
            source = values.get(str(source_id))
            if source is None:
                continue
            payload = source.to_dict()
            payload["result_id"] = uuid.uuid4().hex
            payload["dataset_id"] = str(target_id)
            payload["created_at"] = utc_now()
            provenance = dict(payload["provenance"])
            provenance["source_result_ids"] = [source.result_id]
            provenance["notes"] = [*provenance.get("notes", []), "Duplicated with dataset"]
            payload["provenance"] = provenance
            values[str(target_id)] = CONTRACT_TYPES[kind].from_dict(payload)

    def for_dataset(self, dataset_id: str) -> dict[str, ScientificResultContract]:
        return {
            kind: values[str(dataset_id)]
            for kind, values in self.by_kind.items()
            if str(dataset_id) in values
        }

    def prune(self, dataset_ids: Iterable[str]) -> None:
        allowed = {str(row) for row in dataset_ids}
        for kind in tuple(self.by_kind):
            self.by_kind[kind] = {
                uid: contract for uid, contract in self.by_kind[kind].items() if uid in allowed
            }
            if not self.by_kind[kind]:
                self.by_kind.pop(kind, None)

    def validate(self, dataset_ids: Iterable[str] | None = None) -> None:
        allowed = None if dataset_ids is None else {str(row) for row in dataset_ids}
        for kind, values in self.by_kind.items():
            if kind not in CONTRACT_TYPES:
                raise ContractValidationError(f"Unknown stored result kind: {kind}")
            for uid, contract in values.items():
                if contract.KIND != kind or contract.dataset_id != uid:
                    raise ContractValidationError(
                        f"Result-store identity mismatch for {kind}/{uid}"
                    )
                if allowed is not None and uid not in allowed:
                    raise ContractValidationError(
                        f"Typed result references unknown dataset UID: {uid}"
                    )
                contract.validate_result()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "results": {
                kind: {uid: contract.to_dict() for uid, contract in values.items()}
                for kind, values in self.by_kind.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ResultContractStore":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ContractValidationError("Typed result store must be an object")
        missing = {"schema_version", "results"} - set(value)
        if missing:
            raise ContractValidationError(f"Typed result store is missing: {sorted(missing)}")
        if int(value["schema_version"]) != CONTRACT_SCHEMA_VERSION:
            raise ContractValidationError(
                f"Unsupported typed result store version {value['schema_version']}"
            )
        results = value["results"]
        if not isinstance(results, Mapping):
            raise ContractValidationError("Typed result store results must be an object")
        store = cls()
        for kind, values in results.items():
            if kind not in CONTRACT_TYPES:
                raise ContractValidationError(f"Unknown typed result kind: {kind}")
            if not isinstance(values, Mapping):
                raise ContractValidationError(f"Typed result group '{kind}' must be an object")
            store.by_kind[str(kind)] = {
                str(uid): CONTRACT_TYPES[str(kind)].from_dict(payload)
                for uid, payload in values.items()
            }
        store.validate()
        return store


class ResultContractService:
    """Captures live results and migrates legacy `.afz` analysis dictionaries."""

    RESULT_KINDS = tuple(CONTRACT_TYPES)

    @staticmethod
    def capture_dataset(
        store: ResultContractStore,
        dataset_id: str,
        payloads: Mapping[str, Any],
    ) -> None:
        for kind in ResultContractService.RESULT_KINDS:
            payload = payloads.get(kind)
            empty_collection = isinstance(payload, (Mapping, list, tuple)) and not payload
            if payload is None or empty_collection:
                existing = store.get(kind, dataset_id)
                if (
                    existing is not None
                    and "engine_request"
                    in existing.provenance.dependency_signatures
                ):
                    # Phase 6 engine results are authoritative even when an old
                    # workspace dictionary has not yet been populated.
                    continue
                store.remove(kind, dataset_id)
            else:
                store.record_legacy(kind, dataset_id, payload)

    @staticmethod
    def migrate_analysis_state(datasets: Iterable[Any], state: Mapping[str, Any]) -> ResultContractStore:
        if not isinstance(state, Mapping):
            return ResultContractStore()
        existing = state.get("typed_result_contracts")
        if existing is not None:
            store = ResultContractStore.from_dict(existing)
            store.prune(str(dataset.uid) for dataset in datasets)
            return store

        store = ResultContractStore()
        dataset_rows = list(datasets)
        by_uid = {str(row.uid): row for row in dataset_rows}
        background = _mapping(state.get("background_results"))
        smoothing = _mapping(state.get("smoothing_results"))
        peaks = _mapping(state.get("peak_rows"))
        peak_meta = _mapping(state.get("peak_list_meta_by_uid"))
        fits = _mapping(state.get("fit_groups"))
        candidates = _mapping(state.get("fit_candidates"))
        cells = _mapping(state.get("cell_refinement_results"))
        phases = _mapping(state.get("phase_identification_results"))
        qpa = _mapping(state.get("qpa_results"))
        whole = _mapping(_mapping(state.get("whole_pattern_refinement")).get("results_by_uid"))
        rietveld = _mapping(_mapping(state.get("rietveld_refinement")).get("results_by_uid"))
        validated_qpa = _mapping(_mapping(state.get("validated_qpa")).get("native_results_by_uid"))
        instrument = _mapping(state.get("instrument_calibration"))
        validation = _mapping(state.get("validation_campaign"))

        evidence_by_uid: dict[str, list[dict[str, Any]]] = {uid: [] for uid in by_uid}
        evidence_rows = [
            *_records(validation.get("records", [])),
            *_records(validation.get("audit_results", [])),
            *_records(validation.get("robustness_results", [])),
        ]
        for row in evidence_rows:
            uid = str(row.get("dataset_uid", ""))
            if uid in evidence_by_uid:
                evidence_by_uid[uid].append(row)
                continue
            name = str(row.get("dataset_name", ""))
            match = next((key for key, dataset in by_uid.items() if dataset.name == name), None)
            if match is not None:
                evidence_by_uid[match].append(row)

        reference_uid = str(instrument.get("reference_uid") or "")
        for uid, dataset in by_uid.items():
            preparation_payload = {
                "processed": getattr(dataset, "y_processed", None),
                "background": background.get(uid),
                "smoothing": smoothing.get(uid),
            }
            payloads = {
                "preprocessing": preparation_payload if any(value is not None for value in preparation_payload.values()) else None,
                "peak_list": {"rows": peaks.get(uid, []), "meta": peak_meta.get(uid, {})} if peaks.get(uid) else None,
                "peak_fitting": {"groups": fits.get(uid, []), "candidates": candidates.get(uid, [])} if fits.get(uid) or candidates.get(uid) else None,
                "instrument_calibration": instrument if instrument and (not reference_uid or reference_uid == uid) else None,
                "unit_cell_refinement": cells.get(uid),
                "phase_identification": phases.get(uid),
                "whole_pattern_refinement": whole.get(uid),
                "rietveld_refinement": rietveld.get(uid),
                "qpa": {"exploratory": qpa.get(uid), "validated": validated_qpa.get(uid)} if qpa.get(uid) or validated_qpa.get(uid) else None,
                "validation": {"rows": evidence_by_uid.get(uid, []), "result": validation.get("result")} if evidence_by_uid.get(uid) else None,
            }
            ResultContractService.capture_dataset(store, uid, payloads)
        store.validate(by_uid)
        return store

    @staticmethod
    def export_dataset(store: ResultContractStore, dataset_id: str) -> dict[str, Any]:
        return {
            kind: contract.to_dict()
            for kind, contract in store.for_dataset(dataset_id).items()
        }
