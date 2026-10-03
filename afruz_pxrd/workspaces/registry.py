from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..application.state import ProjectState
from ..models import Dataset
from .base import WorkspaceAdapter, WorkspaceValidation


@dataclass(frozen=True)
class WorkspaceWidgetSpec:
    key: str
    attribute: str
    title: str
    factory: Callable[[Any], Any]


def _builtin_specs() -> dict[str, WorkspaceWidgetSpec]:
    """Import established widgets only when the registry first needs them."""
    from ..batch_widget import BatchWorkflowWidget
    from ..cif_library_widget import CifLibraryManagerWidget
    from ..doping_series_widget import DopingSeriesWidget
    from ..gui_workbench_widget import FullGuiWorkbenchWidget
    from ..instrument_calibration_widget import InstrumentCalibrationWidget
    from ..multicomponent_refiner_widget import MultiComponentRefinerWidget
    from ..phase_revolution_widget import PhaseRevolutionWidget
    from ..rietveld_widget import RietveldRefinementWidget
    from ..structure_solution_widget import StructureSolutionPathwayWidget
    from ..validated_qpa_widget import ValidatedQPAWidget
    from ..validation_widget import ValidationCampaignWidget
    from ..whole_pattern_widget import WholePatternRefinementWidget

    rows = (
        WorkspaceWidgetSpec("instrument", "calibration_widget", "Instrument Calibration", InstrumentCalibrationWidget),
        WorkspaceWidgetSpec("pawley_lebail", "whole_pattern_widget", "Pawley / Le Bail", WholePatternRefinementWidget),
        WorkspaceWidgetSpec("rietveld", "rietveld_widget", "Rietveld", RietveldRefinementWidget),
        WorkspaceWidgetSpec("doping_series", "doping_series_widget", "Doping Series", DopingSeriesWidget),
        WorkspaceWidgetSpec("multicomponent_refiner", "multicomponent_refiner_widget", "Intelligent Multiphase", MultiComponentRefinerWidget),
        WorkspaceWidgetSpec("validated_qpa", "validated_qpa_widget", "Validated QPA", ValidatedQPAWidget),
        WorkspaceWidgetSpec("validation", "validation_widget", "Validation & Evidence", ValidationCampaignWidget),
        WorkspaceWidgetSpec("unknown_phase", "phase_revolution_widget", "Unknown Phase", PhaseRevolutionWidget),
        WorkspaceWidgetSpec("cif_library", "cif_library_widget", "CIF Library / Structure Match", CifLibraryManagerWidget),
        WorkspaceWidgetSpec("solve_structure", "structure_solution_widget", "Solve Structure", StructureSolutionPathwayWidget),
        WorkspaceWidgetSpec("batch_reports", "batch_widget", "Batch & Reports", BatchWorkflowWidget),
        WorkspaceWidgetSpec("full_gui", "full_gui_workbench_widget", "Full GUI Workbench", FullGuiWorkbenchWidget),
    )
    return {row.key: row for row in rows}


class WorkspaceRegistry:
    """Single lookup and lifecycle boundary for scientific workspaces."""

    def __init__(self, project_state: ProjectState):
        self.project_state = project_state
        self._adapters: dict[str, WorkspaceAdapter] = {}
        self._widget_specs: dict[str, WorkspaceWidgetSpec] | None = None

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._adapters)

    def __contains__(self, key: str) -> bool:
        return str(key) in self._adapters

    def adapter(self, key: str) -> WorkspaceAdapter:
        try:
            return self._adapters[str(key)]
        except KeyError as exc:
            raise KeyError(f"Unknown workspace: {key}") from exc

    def widget(self, key: str) -> Any:
        return self.adapter(key).widget

    def create_builtin_widget(self, key: str, owner: Any) -> Any:
        if self._widget_specs is None:
            self._widget_specs = _builtin_specs()
        try:
            specification = self._widget_specs[str(key)]
        except KeyError as exc:
            raise KeyError(f"Unknown built-in workspace: {key}") from exc
        widget = specification.factory(owner)
        setattr(owner, specification.attribute, widget)
        self.register(key, widget)
        return widget

    def builtin_spec(self, key: str) -> WorkspaceWidgetSpec:
        if self._widget_specs is None:
            self._widget_specs = _builtin_specs()
        return self._widget_specs[str(key)]

    def register(
        self,
        key: str,
        widget: Any,
        *,
        validate_callback: Callable[[], Any] | None = None,
        run_callback: Callable[[], Any] | None = None,
        refresh_callback: Callable[[], None] | None = None,
    ) -> WorkspaceAdapter:
        normalized = str(key)
        existing = self._adapters.get(normalized)
        if existing is not None and existing.widget is widget:
            if validate_callback is not None:
                existing._validate_callback = validate_callback
            if run_callback is not None:
                existing._run_callback = run_callback
            if refresh_callback is not None:
                existing._refresh_callback = refresh_callback
            return existing
        adapter = WorkspaceAdapter(
            normalized,
            widget,
            project_state=self.project_state,
            validate_callback=validate_callback,
            run_callback=run_callback,
            refresh_callback=refresh_callback,
        )
        self._adapters[normalized] = adapter
        return adapter

    def register_widgets(
        self,
        widgets: dict[str, Any],
        *,
        run_callbacks: dict[str, Callable[[], Any]] | None = None,
        refresh_callbacks: dict[str, Callable[[], None]] | None = None,
    ) -> None:
        for key, widget in widgets.items():
            self.register(
                key,
                widget,
                run_callback=(run_callbacks or {}).get(key),
                refresh_callback=(refresh_callbacks or {}).get(key),
            )

    def set_project_state(self, state: ProjectState) -> None:
        self.project_state = state
        for adapter in self._adapters.values():
            adapter.set_project_state(state)

    def set_dataset(self, dataset: Dataset | None) -> None:
        for adapter in self._adapters.values():
            adapter.set_dataset(dataset)

    def validate_inputs(self, key: str) -> WorkspaceValidation:
        return self.adapter(key).validate_inputs()

    def run_analysis(self, key: str) -> Any:
        return self.adapter(key).run_analysis()

    def refresh_results(self, keys: Iterable[str] | None = None) -> None:
        selected = tuple(keys) if keys is not None else self.keys
        for key in selected:
            if key in self._adapters:
                self._adapters[key].refresh_results()
