from __future__ import annotations

from typing import Any, Protocol

from ..models import Dataset
from .state import ApplicationState


class WorkspaceRegistryContract(Protocol):
    def __contains__(self, key: str) -> bool: ...
    def adapter(self, key: str) -> Any: ...
    def set_dataset(self, dataset: Dataset | None) -> None: ...
    def validate_inputs(self, key: str) -> Any: ...
    def run_analysis(self, key: str) -> Any: ...
    def refresh_results(self, keys=None) -> None: ...


class WorkflowController:
    """Coordinates active workspace state without depending on MainWindow."""

    MODES = ("Guided", "Expert")

    def __init__(
        self,
        application_state: ApplicationState,
        workspace_registry: WorkspaceRegistryContract,
    ):
        self.state = application_state
        self.registry = workspace_registry

    def set_mode(self, mode: str) -> str:
        normalized = str(mode)
        if normalized not in self.MODES:
            normalized = "Guided"
        self.state.workflow_mode = normalized
        return normalized

    def activate(self, key: str, *, workspace_key: str | None = None) -> Any:
        normalized = str(key)
        if normalized not in self.registry:
            raise KeyError(f"Unknown workspace: {normalized}")
        self.state.active_workflow_task = normalized
        adapter = self.registry.adapter(normalized)
        self.state.active_workspace_key = str(workspace_key or normalized)
        return adapter

    def set_dataset(self, dataset: Dataset | None) -> None:
        self.registry.set_dataset(dataset)

    def validate_active(self):
        return self.registry.validate_inputs(self.state.active_workflow_task)

    def run_active(self):
        return self.registry.run_analysis(self.state.active_workflow_task)

    def refresh(self, keys=None) -> None:
        self.registry.refresh_results(keys)
