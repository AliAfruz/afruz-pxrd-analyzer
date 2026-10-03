from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..contracts import ResultContractStore
from ..metrology import MetrologyRegistry
from ..models import Dataset
from ..scientific_state import ScientificStateRegistry


_PER_DATASET_FIELDS = (
    "backgrounds",
    "background_results",
    "smoothing_results",
    "peak_rows",
    "peak_list_meta_by_uid",
    "fit_groups",
    "fit_candidates",
    "size_strain_results",
    "cell_refinement_results",
    "phase_identification_results",
    "qpa_results",
)


@dataclass
class ProjectState:
    """All scientific state belonging to the currently open project.

    The type deliberately has no Qt dependency.  It is the single owner for
    datasets, preprocessing products, peak lists, refinements, QPA products,
    and their scientific provenance registry.
    """

    datasets: list[Dataset] = field(default_factory=list)
    current_project: Path | None = None
    dirty: bool = False
    backgrounds: dict[str, np.ndarray] = field(default_factory=dict)
    background_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    smoothing_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    peak_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    peak_list_meta_by_uid: dict[str, dict[str, Any]] = field(default_factory=dict)
    fit_groups: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    fit_candidates: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    size_strain_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    reference_structure: dict[str, Any] | None = None
    reference_pattern: list[dict[str, Any]] = field(default_factory=list)
    cell_refinement_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    phase_identification_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    qpa_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    residual_stress_observations: list[dict[str, Any]] = field(default_factory=list)
    residual_stress_result: dict[str, Any] | None = None
    active_instrument_profile: dict[str, Any] | None = None
    scientific_state: ScientificStateRegistry = field(default_factory=ScientificStateRegistry)
    result_contracts: ResultContractStore = field(default_factory=ResultContractStore)
    metrology_registry: MetrologyRegistry = field(default_factory=MetrologyRegistry)

    @property
    def preprocessing(self) -> dict[str, Any]:
        """Semantic view used by controllers and future non-GUI clients."""
        return {
            "backgrounds": self.backgrounds,
            "background_results": self.background_results,
            "smoothing_results": self.smoothing_results,
        }

    @property
    def peak_lists(self) -> dict[str, list[dict[str, Any]]]:
        return self.peak_rows

    @property
    def refinements(self) -> dict[str, dict[str, Any]]:
        return {
            "cell": self.cell_refinement_results,
            "size_strain": self.size_strain_results,
            "phase_identification": self.phase_identification_results,
        }

    def dataset(self, uid: str) -> Dataset:
        uid = str(uid)
        for dataset in self.datasets:
            if dataset.uid == uid:
                return dataset
        raise KeyError(f"Unknown dataset UID: {uid}")

    def dataset_index(self, uid: str) -> int:
        uid = str(uid)
        for index, dataset in enumerate(self.datasets):
            if dataset.uid == uid:
                return index
        raise KeyError(f"Unknown dataset UID: {uid}")

    def clear(self) -> None:
        """Reset project-owned state without touching application preferences."""
        self.datasets.clear()
        for name in _PER_DATASET_FIELDS:
            getattr(self, name).clear()
        self.reference_structure = None
        self.reference_pattern.clear()
        self.residual_stress_observations.clear()
        self.residual_stress_result = None
        self.active_instrument_profile = None
        self.scientific_state = ScientificStateRegistry()
        self.result_contracts = ResultContractStore()
        self.metrology_registry = MetrologyRegistry()
        self.current_project = None
        self.dirty = False

    def duplicate_dataset_payloads(self, source_uid: str, target_uid: str) -> None:
        source_uid, target_uid = str(source_uid), str(target_uid)
        for name in _PER_DATASET_FIELDS:
            mapping = getattr(self, name)
            if source_uid in mapping:
                mapping[target_uid] = deepcopy(mapping[source_uid])
        self.scientific_state.duplicate(source_uid, target_uid)
        self.result_contracts.duplicate_dataset(source_uid, target_uid)
        # A duplicated data object is not a new independent measurement.  Phase 8
        # therefore never transfers metrology assessments or validated claims.

    def remove_dataset_payloads(self, uid: str) -> None:
        uid = str(uid)
        for name in _PER_DATASET_FIELDS:
            getattr(self, name).pop(uid, None)
        self.scientific_state.remove(uid)
        self.result_contracts.remove_dataset(uid)
        self.metrology_registry.remove_dataset(uid)
        self.residual_stress_observations = [
            row
            for row in self.residual_stress_observations
            if str(row.get("dataset_uid", "")) != uid
        ]
        self.residual_stress_result = None

    def snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for descriptor in fields(self):
            name = descriptor.name
            if name == "scientific_state":
                payload[name] = deepcopy(self.scientific_state.to_dict())
            elif name == "result_contracts":
                payload[name] = deepcopy(self.result_contracts.to_dict())
            elif name == "metrology_registry":
                payload[name] = deepcopy(self.metrology_registry.to_dict())
            elif name == "current_project":
                payload[name] = (
                    None if self.current_project is None else str(self.current_project)
                )
            else:
                payload[name] = deepcopy(getattr(self, name))
        return payload

    def restore(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("Project state snapshot must be a mapping")
        self.clear()
        for descriptor in fields(self):
            name = descriptor.name
            if name not in payload:
                continue
            if name == "scientific_state":
                self.scientific_state = ScientificStateRegistry.from_dict(payload[name])
            elif name == "result_contracts":
                self.result_contracts = ResultContractStore.from_dict(payload[name])
            elif name == "metrology_registry":
                self.metrology_registry = MetrologyRegistry.from_dict(payload[name])
            elif name == "current_project":
                value = payload[name]
                self.current_project = Path(value) if value else None
            else:
                setattr(self, name, deepcopy(payload[name]))
        self.validate_invariants()

    def validate_invariants(self) -> None:
        uids = [str(dataset.uid) for dataset in self.datasets]
        if len(uids) != len(set(uids)):
            raise ValueError("Project contains duplicate dataset UIDs")
        allowed = set(uids)
        for name in _PER_DATASET_FIELDS:
            orphans = set(getattr(self, name)) - allowed
            if orphans:
                raise ValueError(f"{name} contains unknown dataset UIDs: {sorted(orphans)}")
        scientific_orphans = set(self.scientific_state.by_dataset) - allowed
        if scientific_orphans:
            raise ValueError(
                "scientific_state contains unknown dataset UIDs: "
                f"{sorted(scientific_orphans)}"
            )
        self.result_contracts.validate(allowed)
        self.metrology_registry.validate(allowed)


@dataclass
class ApplicationState:
    """Non-scientific state that controls the application shell."""

    theme_name: str = "Light"
    workflow_mode: str = "Guided"
    active_workflow_task: str = "project_home"
    active_workspace_key: str = "home"
    workflow_states: dict[str, Any] = field(default_factory=dict)
    workflow_task_states: dict[str, Any] = field(default_factory=dict)
    workflow_context: dict[str, Any] = field(default_factory=dict)
    ui_preferences: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            descriptor.name: deepcopy(getattr(self, descriptor.name))
            for descriptor in fields(self)
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("Application state snapshot must be a mapping")
        for descriptor in fields(self):
            if descriptor.name in payload:
                setattr(self, descriptor.name, deepcopy(payload[descriptor.name]))
