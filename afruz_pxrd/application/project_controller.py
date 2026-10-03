from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..contracts import EngineIdentity, ResultContractService
from ..engines import (
    CancellationCheck,
    EngineRegistry,
    EngineRun,
    ProgressCallback,
    ScientificEngineService,
    create_default_engine_registry,
)
from ..models import Dataset
from ..metrology import (
    AcceptanceCriterion,
    AcquisitionMetadata,
    ClaimScope,
    EvidenceClass,
    MetrologyAssessment,
    MetrologyRegistry,
    MetrologyService,
    MetricObservation,
    ScientificClaim,
)
from ..project import load_afz, save_afz
from ..undo_redo import UndoRedoHistory
from .state import ApplicationState, ProjectState


@dataclass(frozen=True)
class OpenedProject:
    path: Path
    ui_state: dict[str, Any]
    analysis_state: dict[str, Any]


class ProjectController:
    """GUI-independent command surface for project and dataset mutations."""

    SNAPSHOT_VERSION = 1

    def __init__(
        self,
        project_state: ProjectState | None = None,
        application_state: ApplicationState | None = None,
        *,
        load_function: Callable[[str | Path], tuple[list[Dataset], dict, dict]] = load_afz,
        save_function: Callable[[str | Path, list[Dataset], dict, dict], None] = save_afz,
        history_depth: int = 40,
        engine_registry: EngineRegistry | None = None,
        engine_service: ScientificEngineService | None = None,
        metrology_service: MetrologyService | None = None,
    ):
        self.state = project_state if project_state is not None else ProjectState()
        self.application_state = (
            application_state if application_state is not None else ApplicationState()
        )
        self._load = load_function
        self._save = save_function
        if engine_registry is not None and engine_service is not None:
            raise ValueError("Pass engine_registry or engine_service, not both")
        self.engine_service = engine_service or ScientificEngineService(
            engine_registry or create_default_engine_registry(),
            self.state.result_contracts,
        )
        self._bind_engine_result_store()
        self.metrology_service = metrology_service or MetrologyService(
            self.state.metrology_registry
        )
        self._bind_metrology_registry()
        self.history = UndoRedoHistory(
            max_depth=history_depth,
            snapshot_factory=self.create_snapshot,
            restore_callback=lambda snapshot: self.restore_snapshot(snapshot),
        )

    def configure_history(
        self,
        snapshot_factory: Callable[[], dict[str, Any]],
        restore_callback: Callable[[dict[str, Any]], None],
    ) -> None:
        self.history.snapshot_factory = snapshot_factory
        self.history.restore_callback = restore_callback

    def new_project(self) -> None:
        self.state.clear()
        self._bind_engine_result_store()
        self._bind_metrology_registry()
        self.engine_service.clear_history()

    def add_datasets(
        self,
        datasets: Iterable[Dataset],
        *,
        index: int | None = None,
    ) -> list[Dataset]:
        additions = list(datasets)
        existing = {dataset.uid for dataset in self.state.datasets}
        incoming: set[str] = set()
        for dataset in additions:
            if not isinstance(dataset, Dataset):
                raise TypeError("Only Dataset instances can be added to a project")
            dataset.validate()
            if dataset.uid in existing or dataset.uid in incoming:
                raise ValueError(f"Duplicate dataset UID: {dataset.uid}")
            incoming.add(dataset.uid)
        if index is None:
            self.state.datasets.extend(additions)
        else:
            insertion = max(0, min(int(index), len(self.state.datasets)))
            self.state.datasets[insertion:insertion] = additions
        if additions:
            self.mark_dirty()
        self.state.validate_invariants()
        return additions

    def rename_dataset(self, uid: str, name: str) -> Dataset:
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("Dataset name cannot be empty")
        dataset = self.state.dataset(uid)
        dataset.name = normalized
        self.mark_dirty()
        return dataset

    def duplicate_dataset(self, uid: str, *, index: int | None = None) -> Dataset:
        source = self.state.dataset(uid)
        clone = source.clone()
        clone.metadata.pop("acquisition_metadata", None)
        insertion = self.state.dataset_index(uid) + 1 if index is None else int(index)
        self.state.datasets.insert(max(0, min(insertion, len(self.state.datasets))), clone)
        self.state.duplicate_dataset_payloads(source.uid, clone.uid)
        self.mark_dirty()
        self.state.validate_invariants()
        return clone

    def remove_dataset(self, uid: str) -> Dataset:
        index = self.state.dataset_index(uid)
        removed = self.state.datasets.pop(index)
        self.state.remove_dataset_payloads(removed.uid)
        self.mark_dirty()
        self.state.validate_invariants()
        return removed

    def save(
        self,
        path: str | Path,
        ui_state: Mapping[str, Any],
        analysis_state: Mapping[str, Any],
    ) -> Path:
        destination = Path(path)
        if destination.suffix.lower() != ".afz":
            destination = destination.with_suffix(".afz")
        analysis_payload = dict(analysis_state)
        if "typed_result_contracts" not in analysis_payload:
            migrated = ResultContractService.migrate_analysis_state(
                self.state.datasets,
                analysis_payload,
            )
            for dataset in self.state.datasets:
                for contract in self.state.result_contracts.for_dataset(
                    dataset.uid
                ).values():
                    migrated.record(contract)
            self.state.result_contracts = migrated
            self._bind_engine_result_store()
        analysis_payload["typed_result_contracts"] = (
            self.state.result_contracts.to_dict()
        )
        analysis_payload["metrology_registry"] = (
            self.state.metrology_registry.to_dict()
        )
        self.state.validate_invariants()
        self._save(
            destination,
            self.state.datasets,
            dict(ui_state),
            analysis_payload,
        )
        self._apply_ui_state(ui_state)
        self.state.current_project = destination
        self.state.dirty = False
        return destination

    def open(self, path: str | Path) -> OpenedProject:
        source = Path(path)
        # Load and validate completely before touching the live state.
        datasets, ui_state, analysis_state = self._load(source)
        metrology_registry = MetrologyRegistry.from_dict(
            analysis_state.get("metrology_registry")
        )
        metrology_registry.validate(dataset.uid for dataset in datasets)
        self.state.clear()
        self.state.datasets.extend(datasets)
        self.state.result_contracts = ResultContractService.migrate_analysis_state(
            datasets,
            analysis_state,
        )
        self.state.metrology_registry = metrology_registry
        self._bind_engine_result_store()
        self._bind_metrology_registry()
        self.engine_service.clear_history()
        self.state.current_project = source
        self.state.dirty = False
        self.state.validate_invariants()
        self._apply_ui_state(ui_state)
        return OpenedProject(source, dict(ui_state), dict(analysis_state))

    def _apply_ui_state(self, ui_state: Mapping[str, Any]) -> None:
        """Keep shell preferences central while accepting legacy `.afz` keys."""
        ui = dict(ui_state)
        if ui.get("theme_name"):
            self.application_state.theme_name = str(ui["theme_name"])
        if ui.get("workflow_mode"):
            self.application_state.workflow_mode = str(ui["workflow_mode"])
        if ui.get("active_workflow_task"):
            self.application_state.active_workflow_task = str(ui["active_workflow_task"])
        if ui.get("active_workspace_key"):
            self.application_state.active_workspace_key = str(ui["active_workspace_key"])
        self.application_state.ui_preferences = deepcopy(ui)

    def mark_dirty(self) -> None:
        if not self.history.is_restoring:
            self.state.dirty = True

    def run_engine(
        self,
        result_kind: str,
        dataset_id: str,
        *,
        inputs: Mapping[str, Any],
        parameters: Mapping[str, Any] | None = None,
        engine_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
        cancellation_check: CancellationCheck | None = None,
    ) -> EngineRun:
        """Execute one registered engine through the shared headless service."""
        self.state.dataset(dataset_id)
        run = self.engine_service.execute_kind(
            result_kind,
            dataset_id,
            inputs=inputs,
            parameters=parameters,
            engine_id=engine_id,
            progress_callback=progress_callback,
            cancellation_check=cancellation_check,
        )
        self.mark_dirty()
        return run

    def _bind_engine_result_store(self) -> None:
        self.engine_service.bind_result_store(self.state.result_contracts)

    def _bind_metrology_registry(self) -> None:
        if hasattr(self, "metrology_service"):
            self.metrology_service.bind_registry(self.state.metrology_registry)

    def set_acquisition_metadata(
        self,
        dataset_id: str,
        metadata: AcquisitionMetadata,
    ) -> AcquisitionMetadata:
        dataset = self.state.dataset(dataset_id)
        recorded = self.metrology_service.set_acquisition(
            dataset.uid,
            metadata,
            x=dataset.x,
            y=dataset.y_raw,
        )
        dataset.metadata["acquisition_metadata"] = recorded.to_dict()
        self.mark_dirty()
        return recorded

    def assess_certified_standard(
        self,
        dataset_id: str,
        standard_id: str,
        observations: Iterable[MetricObservation],
        criteria: Iterable[AcceptanceCriterion],
        *,
        material_unit_id: str,
        title: str | None = None,
    ) -> MetrologyAssessment:
        self.state.dataset(dataset_id)
        assessment = self.metrology_service.assess_standard(
            dataset_id,
            standard_id,
            observations,
            criteria,
            material_unit_id=material_unit_id,
            title=title,
        )
        self.mark_dirty()
        return assessment

    def compare_reference_engine(
        self,
        dataset_id: str,
        native_engine: EngineIdentity,
        native_observations: Iterable[MetricObservation],
        reference_engine: EngineIdentity,
        reference_observations: Iterable[MetricObservation],
        criteria: Iterable[AcceptanceCriterion],
        *,
        native_input_signature: str,
        reference_input_signature: str,
        title: str = "Native/reference-engine comparison",
    ) -> MetrologyAssessment:
        self.state.dataset(dataset_id)
        assessment = self.metrology_service.compare_engines(
            dataset_id,
            native_engine,
            native_observations,
            reference_engine,
            reference_observations,
            criteria,
            native_input_signature=native_input_signature,
            reference_input_signature=reference_input_signature,
            title=title,
        )
        self.mark_dirty()
        return assessment

    def assess_reference_values(
        self,
        dataset_id: str,
        evidence_class: EvidenceClass,
        observations: Iterable[MetricObservation],
        reference_observations: Iterable[MetricObservation],
        criteria: Iterable[AcceptanceCriterion],
        *,
        title: str = "Scientific reference-value assessment",
    ) -> MetrologyAssessment:
        self.state.dataset(dataset_id)
        assessment = self.metrology_service.assess_values(
            dataset_id,
            evidence_class,
            observations,
            reference_observations,
            criteria,
            title=title,
        )
        self.mark_dirty()
        return assessment

    def authorize_scientific_claim(
        self,
        assessment_id: str,
        scope: ClaimScope,
        statement: str,
    ) -> ScientificClaim:
        claim = self.metrology_service.authorize_claim(
            assessment_id,
            scope,
            statement,
        )
        self.mark_dirty()
        return claim

    def mark_clean(self) -> None:
        self.state.dirty = False

    def create_snapshot(
        self,
        *,
        analysis_state: Mapping[str, Any] | None = None,
        ui_state: Mapping[str, Any] | None = None,
        selected_row: int = -1,
    ) -> dict[str, Any]:
        return {
            "snapshot_version": self.SNAPSHOT_VERSION,
            "project_state": self.state.snapshot(),
            "application_state": self.application_state.snapshot(),
            "analysis_state": deepcopy(dict(analysis_state or {})),
            "ui_state": deepcopy(dict(ui_state or {})),
            "selected_row": int(selected_row),
        }

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(snapshot, Mapping):
            raise TypeError("Undo snapshot must be a mapping")
        if "project_state" in snapshot:
            self.state.restore(snapshot["project_state"])
            self.application_state.restore(snapshot.get("application_state", {}))
        else:
            # Compatibility with Phase 2 snapshots held in a live history.
            self.state.clear()
            self.state.datasets = deepcopy(list(snapshot.get("datasets", [])))
            current_project = snapshot.get("current_project")
            self.state.current_project = Path(current_project) if current_project else None
            self.state.dirty = bool(snapshot.get("project_dirty", False))
        self._bind_engine_result_store()
        self._bind_metrology_registry()
        return {
            "analysis_state": deepcopy(dict(snapshot.get("analysis_state", {}))),
            "ui_state": deepcopy(dict(snapshot.get("ui_state", {}))),
            "selected_row": int(snapshot.get("selected_row", -1)),
        }

    def checkpoint(self, label: str) -> bool:
        created = self.history.checkpoint(label)
        if created:
            self.mark_dirty()
        return created

    def undo(self) -> str | None:
        return self.history.undo()

    def redo(self) -> str | None:
        return self.history.redo()

    def clear_history(self) -> None:
        self.history.clear()
