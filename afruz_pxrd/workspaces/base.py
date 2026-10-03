from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from PySide6.QtCore import QObject, Signal

from ..application.state import ProjectState
from ..models import Dataset


@dataclass(frozen=True)
class WorkspaceValidation:
    valid: bool
    message: str = ""


@runtime_checkable
class WorkspaceContract(Protocol):
    """Common lifecycle implemented by every registered workspace."""

    def set_project_state(self, state: ProjectState) -> None: ...
    def set_dataset(self, dataset: Dataset | None) -> None: ...
    def validate_inputs(self) -> WorkspaceValidation: ...
    def run_analysis(self) -> Any: ...
    def refresh_results(self) -> None: ...


class WorkspaceAdapter(QObject):
    """Adapts established widgets to the Phase 4 workspace contract.

    Existing scientific widgets keep their tested APIs.  The adapter is the
    signal-based boundary used by MainWindow and future independent workspace
    implementations.
    """

    analysisRequested = Signal(str)
    analysisCompleted = Signal(str)
    validationChanged = Signal(str, bool, str)
    resultsRefreshed = Signal(str)

    def __init__(
        self,
        key: str,
        widget: Any,
        *,
        project_state: ProjectState | None = None,
        validate_callback: Callable[[], Any] | None = None,
        run_callback: Callable[[], Any] | None = None,
        refresh_callback: Callable[[], None] | None = None,
    ):
        super().__init__()
        self.key = str(key)
        self.widget = widget
        self.project_state = project_state
        self.dataset: Dataset | None = None
        self._validate_callback = validate_callback
        self._run_callback = run_callback
        self._refresh_callback = refresh_callback

    def set_project_state(self, state: ProjectState) -> None:
        self.project_state = state
        callback = getattr(self.widget, "set_project_state", None)
        if callable(callback):
            callback(state)

    def set_dataset(self, dataset: Dataset | None) -> None:
        self.dataset = dataset
        callback = getattr(self.widget, "set_dataset", None)
        if callable(callback):
            callback(dataset)
            return
        callback = getattr(self.widget, "refresh_for_selected_dataset", None)
        if callable(callback):
            callback()

    def validate_inputs(self) -> WorkspaceValidation:
        callback = self._validate_callback or getattr(
            self.widget, "validate_inputs", None
        )
        if not callable(callback):
            result = WorkspaceValidation(True, "")
        else:
            raw = callback()
            if isinstance(raw, WorkspaceValidation):
                result = raw
            elif isinstance(raw, tuple):
                result = WorkspaceValidation(bool(raw[0]), str(raw[1] or ""))
            else:
                result = WorkspaceValidation(bool(raw), "")
        self.validationChanged.emit(self.key, result.valid, result.message)
        return result

    def run_analysis(self) -> Any:
        validation = self.validate_inputs()
        if not validation.valid:
            raise ValueError(validation.message or f"{self.key} inputs are invalid")
        callback = self._run_callback or getattr(self.widget, "run_analysis", None)
        if not callable(callback):
            raise RuntimeError(f"Workspace '{self.key}' has no analysis command")
        self.analysisRequested.emit(self.key)
        result = callback()
        self.analysisCompleted.emit(self.key)
        return result

    def refresh_results(self) -> None:
        callback = self._refresh_callback or getattr(
            self.widget, "refresh_results", None
        )
        if callable(callback):
            callback()
        else:
            callback = getattr(self.widget, "refresh_for_selected_dataset", None)
            if callable(callback):
                callback()
        self.resultsRefreshed.emit(self.key)
