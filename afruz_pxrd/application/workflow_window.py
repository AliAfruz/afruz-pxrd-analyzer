from __future__ import annotations

from ..contracts import ResultContractService
from ..main_window_dependencies import *


class WorkflowWindowMixin:
    def _configure_unified_workflow(self):
        self._workflow_task_widgets = {
            "project_home": self.project_home_widget,
            "raw_pattern": self.raw_tab,
            "background": self.preprocessing_tab,
            "smoothing": self.smoothing_tab,
            "instrument": self.calibration_widget,
            "peak_list": self.peak_tab,
            "size_strain": self.size_strain_tab,
            "residual_stress": self.stress_tab,
            "phase_identification": self.phase_tab,
            "cif_cell": self.crystal_tab,
            "unknown_phase": self.phase_revolution_widget,
            "cif_library": self.cif_library_widget,
            "solve_structure": self.structure_solution_widget,
            "pawley_lebail": self.whole_pattern_widget,
            "rietveld": self.rietveld_widget,
            "doping_series": self.doping_series_widget,
            "multicomponent_refiner": self.multicomponent_refiner_widget,
            "qpa_preview": self.qpa_tab,
            "validated_qpa": self.validated_qpa_widget,
            "validation": self.validation_widget,
            "batch_reports": self.batch_widget,
            "full_gui": self.full_gui_workbench_widget,
            "advanced": self.advanced_analysis_tab,
        }
        self._workflow_widget_tasks = {
            widget: key for key, widget in self._workflow_task_widgets.items()
        }
        self.workspace_registry.register_widgets(
            self._workflow_task_widgets,
            run_callbacks=self._workspace_run_callbacks(),
            refresh_callbacks=self._workspace_refresh_callbacks(),
        )
        self.tabs.tabBar().hide()
        self.workflow_navigation.workspaceSelected.connect(
            self._select_workflow_workspace
        )
        self.workflow_navigation.taskSelected.connect(
            self._select_workflow_task
        )
        self.workflow_header.taskSelected.connect(
            self._select_workflow_task
        )
        self.workflow_header.modeChanged.connect(
            self._workflow_mode_changed
        )
        self.tabs.currentChanged.connect(self._sync_workflow_from_tab)
        self.project_home_widget.continueRequested.connect(
            self._continue_workflow
        )
        self.project_home_widget.importRequested.connect(self.import_patterns)
        self.project_home_widget.openRequested.connect(self.open_project)
        self.project_home_widget.recentProjectRequested.connect(
            self._open_recent_project
        )
        self.project_home_widget.recentProjectRemoveRequested.connect(
            self._remove_recent_project
        )
        self.project_home_widget.recentProjectsClearRequested.connect(
            self._clear_recent_projects
        )
        self._select_workflow_task("project_home")

    def _select_workflow_workspace(self, workspace_key: str):
        if self._workflow_syncing:
            return
        task_keys = [
            task.key
            for task in tasks_for_workspace(workspace_key, self.workflow_mode)
        ]
        if not task_keys:
            return
        recommended = recommended_task_key(
            self.workflow_states,
            self.workflow_context,
            self.workflow_mode,
            self.workflow_task_states,
        )
        selected = self.active_workflow_task if self.active_workflow_task in task_keys else None
        if selected is None and recommended in task_keys:
            selected = recommended
        if selected is None:
            selected = next(
                (
                    key
                    for key in task_keys
                    if getattr(self.workflow_task_states.get(key), "enabled", True)
                ),
                task_keys[0],
            )
        self._workflow_syncing = True
        try:
            self.active_workspace_key = workspace_key
            self.workflow_header.set_active_workspace(workspace_key)
            self.workflow_header.set_tasks(task_keys, selected)
            self.workflow_navigation.set_active_workspace(workspace_key)
            self.workflow_navigation.set_tasks(task_keys, selected)
            self.workflow_header.set_task_states(self.workflow_task_states)
            self.workflow_navigation.set_task_states(self.workflow_task_states)
        finally:
            self._workflow_syncing = False
        self._select_workflow_task(selected)

    def _select_workflow_task(self, task_key: str):
        widget = self._workflow_task_widgets.get(task_key)
        task = TASK_BY_KEY.get(task_key)
        if widget is None or task is None:
            return
        self._workflow_syncing = True
        try:
            self.workflow_controller.activate(
                task_key,
                workspace_key=task.workspace,
            )
            task_keys = [
                row.key
                for row in tasks_for_workspace(task.workspace, self.workflow_mode)
            ]
            if task_key not in task_keys:
                task_keys.append(task_key)
            self.workflow_header.set_active_workspace(task.workspace)
            self.workflow_header.set_tasks(task_keys, task_key)
            self.workflow_header.set_task_states(self.workflow_task_states)
            self.workflow_header.set_active_task(task_key)
            self.workflow_navigation.set_active_workspace(task.workspace)
            self.workflow_navigation.set_tasks(task_keys, task_key)
            self.workflow_navigation.set_task_states(self.workflow_task_states)
            self.workflow_navigation.set_active_task(task_key)
            self.tabs.setCurrentWidget(widget)
        finally:
            self._workflow_syncing = False
        self._update_workflow_dashboard()
        self._sync_analysis_panel_for_task()

    def _set_workflow_mode(self, mode: str):
        self.workflow_header.set_mode(mode)
        self.workflow_navigation.set_mode(mode)
        self._workflow_mode_changed(mode)

    def _workflow_mode_changed(self, mode: str):
        if self._workflow_syncing:
            return
        self.workflow_controller.set_mode(mode)
        self.workflow_header.set_mode(self.workflow_mode)
        self.workflow_navigation.set_mode(self.workflow_mode)
        self.guided_mode_action.setChecked(self.workflow_mode == "Guided")
        self.expert_mode_action.setChecked(self.workflow_mode == "Expert")
        task = TASK_BY_KEY.get(self.active_workflow_task)
        if self.workflow_mode == "Guided" and task is not None and not task.guided:
            guided = tasks_for_workspace(task.workspace, "Guided")
            self._select_workflow_task(guided[0].key if guided else "project_home")
            return
        self._select_workflow_workspace(self.active_workspace_key)

    def _sync_workflow_from_tab(self, index: int):
        if self._workflow_syncing:
            return
        widget = self.tabs.widget(index)
        task_key = self._workflow_widget_tasks.get(widget)
        if task_key:
            self._select_workflow_task(task_key)

    def _continue_workflow(self):
        self._update_workflow_dashboard()
        self._select_workflow_task(
            recommended_task_key(
                self.workflow_states,
                self.workflow_context,
                self.workflow_mode,
                self.workflow_task_states,
            )
        )

    def _scientific_stage_payloads(self, dataset):
        uid = dataset.uid
        prepared_exists = bool(
            dataset.y_processed is not None
            or bool(self.background_results.get(uid, {}).get("applied"))
            or bool(self.smoothing_results.get(uid, {}).get("applied"))
        )
        preparation = None
        if prepared_exists:
            preparation = {
                "processed": dataset.y_processed,
                "background": self.background_results.get(uid),
                "smoothing": self.smoothing_results.get(uid),
                "metadata": dataset.metadata,
            }

        peaks = None
        peak_rows = normalize_peak_rows(self.peak_rows.get(uid, []))
        if peak_rows:
            peaks = {
                "rows": peak_rows,
                "meta": self.peak_list_meta_by_uid.get(uid, {}),
            }

        phase_parts = {
            "known_phase": self.phase_identification_results.get(uid),
            "candidate_cells": getattr(
                getattr(self, "phase_revolution_widget", None),
                "candidates_by_uid",
                {},
            ).get(uid),
            "provisional_structure": getattr(
                getattr(self, "structure_solution_widget", None),
                "provisional_by_uid",
                {},
            ).get(uid),
        }
        phase = phase_parts if any(value for value in phase_parts.values()) else None

        refinement_parts = {
            "pawley_lebail": getattr(
                getattr(self, "whole_pattern_widget", None),
                "results_by_uid",
                {},
            ).get(uid),
            "rietveld": getattr(
                getattr(self, "rietveld_widget", None),
                "results_by_uid",
                {},
            ).get(uid),
        }
        refinement = (
            refinement_parts
            if any(value for value in refinement_parts.values())
            else None
        )

        qpa_parts = {
            "exploratory": self.qpa_results.get(uid),
            "validated": getattr(
                getattr(self, "validated_qpa_widget", None),
                "native_results_by_uid",
                {},
            ).get(uid),
        }
        qpa = qpa_parts if any(value for value in qpa_parts.values()) else None

        validation = None
        validation_widget = getattr(self, "validation_widget", None)
        if validation_widget is not None:
            rows = []
            for row in [
                *getattr(validation_widget, "audit_results", []),
                *getattr(validation_widget, "robustness_results", []),
            ]:
                if not isinstance(row, dict):
                    continue
                if row.get("dataset_uid") == uid or row.get("dataset_name") == dataset.name:
                    rows.append(row)
            if rows:
                validation = rows

        return {
            "import": {
                "x": dataset.x,
                "y_raw": dataset.y_raw,
                "source_path": dataset.source_path,
                "metadata": dataset.metadata,
            },
            "preparation": preparation,
            "peaks": peaks,
            "phase": phase,
            "refinement": refinement,
            "qpa": qpa,
            "validation": validation,
        }

    @staticmethod
    def _scientific_dependency_keys(stage: str, state):
        if stage == "validation" and state.node("qpa").is_current:
            return ("qpa",)
        return None

    def _typed_result_payloads(self, dataset, stage_payloads):
        """Translate live workspace state at the one authoritative boundary."""
        uid = dataset.uid
        calibration = {}
        calibration_widget = getattr(self, "calibration_widget", None)
        if calibration_widget is not None:
            candidate = calibration_widget.get_state()
            if isinstance(candidate, dict):
                reference_uid = str(candidate.get("reference_uid") or "")
                if (
                    (candidate.get("profile") or candidate.get("qa_result"))
                    and (not reference_uid or reference_uid == uid)
                ):
                    calibration = candidate

        whole_pattern = getattr(
            getattr(self, "whole_pattern_widget", None),
            "results_by_uid",
            {},
        ).get(uid)
        rietveld = getattr(
            getattr(self, "rietveld_widget", None),
            "results_by_uid",
            {},
        ).get(uid)
        validation = stage_payloads.get("validation")
        return {
            "preprocessing": stage_payloads.get("preparation"),
            "peak_list": stage_payloads.get("peaks"),
            "peak_fitting": (
                {
                    "groups": self.fit_groups.get(uid, []),
                    "candidates": self.fit_candidates.get(uid, []),
                }
                if self.fit_groups.get(uid) or self.fit_candidates.get(uid)
                else None
            ),
            "instrument_calibration": calibration or None,
            "unit_cell_refinement": self.cell_refinement_results.get(uid),
            "phase_identification": self.phase_identification_results.get(uid),
            "whole_pattern_refinement": whole_pattern,
            "rietveld_refinement": rietveld,
            "qpa": stage_payloads.get("qpa"),
            "validation": {"rows": validation} if validation else None,
        }

    def _sync_unified_scientific_state(self, dataset_uid: str | None = None):
        if self._scientific_state_syncing:
            return
        self._scientific_state_syncing = True
        try:
            self.scientific_state.prune(dataset.uid for dataset in self.datasets)
            datasets = [
                dataset
                for dataset in self.datasets
                if dataset_uid is None or dataset.uid == dataset_uid
            ]
            for dataset in datasets:
                state = self.scientific_state.ensure(dataset.uid)
                payloads = self._scientific_stage_payloads(dataset)
                state.update_authoritative(
                    "import",
                    payloads["import"],
                    metadata={"dataset_name": dataset.name, "points": len(dataset.x)},
                    reason="Raw measurement synchronized",
                )
                for key in SCIENTIFIC_STAGE_ORDER[1:]:
                    payload = payloads.get(key)
                    node = state.node(key)
                    if payload is None:
                        if node.exists and node.status != "Missing":
                            state.mark_missing(
                                key,
                                reason=f"No current {SCIENTIFIC_STAGE_LABELS[key].lower()} result exists",
                            )
                        continue
                    # Preparation and peaks are authoritative mutable dataset states.
                    # Derived stages are observed: an unchanged historical result must
                    # remain Outdated after an upstream revision changes.
                    if key in {"preparation", "peaks"}:
                        state.update_authoritative(
                            key,
                            payload,
                            metadata={"dataset_name": dataset.name},
                            reason=f"{SCIENTIFIC_STAGE_LABELS[key]} synchronized",
                            dependency_keys=self._scientific_dependency_keys(key, state),
                        )
                    else:
                        state.observe_result(
                            key,
                            payload,
                            metadata={"dataset_name": dataset.name},
                            dependency_keys=self._scientific_dependency_keys(key, state),
                        )
                state.reconcile()
                ResultContractService.capture_dataset(
                    self.project_state.result_contracts,
                    dataset.uid,
                    self._typed_result_payloads(dataset, payloads),
                )
            self.project_state.result_contracts.prune(
                dataset.uid for dataset in self.datasets
            )
        finally:
            self._scientific_state_syncing = False

    def _record_scientific_result(
        self,
        stage: str,
        dataset_uid: str,
        payload,
        *,
        reason: str,
        metadata: dict | None = None,
    ):
        dataset = next((row for row in self.datasets if row.uid == dataset_uid), None)
        if dataset is None:
            return None
        self._sync_unified_scientific_state(dataset_uid)
        state = self.scientific_state.ensure(dataset_uid)
        current_payload = self._scientific_stage_payloads(dataset).get(stage)
        node = state.record_result(
            stage,
            payload if current_payload is None else current_payload,
            metadata={"dataset_name": dataset.name, **(metadata or {})},
            reason=reason,
            produced=True,
            dependency_keys=self._scientific_dependency_keys(stage, state),
        )
        self._update_workflow_dashboard()
        return node

    def _refresh_shell_header(self, dataset=None) -> None:
        if not hasattr(self, "workflow_header"):
            return
        if dataset is None:
            dataset = self.selected_dataset() if hasattr(self, "dataset_list") else None
        project_name = self.current_project.stem if self.current_project else "Untitled project"
        self.workflow_header.set_project_name(project_name)
        self.workflow_header.set_dataset_name(dataset.name if dataset is not None else None)
        if self.current_project is None:
            save_state = "not saved"
        else:
            save_state = "modified" if self.project_dirty else "saved"
        self.workflow_header.set_save_state(save_state)

        run_labels = {
            "project_home": "Continue",
            "raw_pattern": "Inspect pattern",
            "background": "Preview background",
            "smoothing": "Preview smoothing",
            "instrument": "Run calibration QA",
            "peak_list": "Smart peak search",
            "size_strain": "Calculate size / strain",
            "residual_stress": "Calculate stress",
            "phase_identification": "Identify phases",
            "cif_cell": "Refine unit cell",
            "cif_library": "Search CIF library",
            "unknown_phase": "Run indexing",
            "solve_structure": "Test robustness",
            "pawley_lebail": "Run Pawley / Le Bail",
            "rietveld": "Run Rietveld",
            "multicomponent_refiner": "Run multiphase search",
            "qpa_preview": "Calculate QPA preview",
            "validated_qpa": "Calculate validated QPA",
            "validation": "Calculate validation",
            "batch_reports": "Start batch",
            "full_gui": "Open workbench",
            "advanced": "Open advanced tools",
        }
        task_state = self.workflow_task_states.get(self.active_workflow_task)
        requires_dataset = self.active_workflow_task not in {"project_home", "full_gui", "advanced"}
        enabled = dataset is not None or not requires_dataset
        tooltip = TASK_BY_KEY.get(self.active_workflow_task).guidance if self.active_workflow_task in TASK_BY_KEY else "Run the current analysis."
        if task_state is not None:
            enabled = bool(task_state.enabled)
            tooltip = task_state.reason
        self.workflow_header.set_run_label(
            run_labels.get(self.active_workflow_task, "Run analysis"),
            enabled=enabled,
            tooltip=tooltip,
        )

    def _mark_project_dirty(self) -> None:
        if getattr(self, "_undo_redo_restoring", False):
            return
        self.project_controller.mark_dirty()
        self._refresh_shell_header()

    def _workspace_run_callbacks(self) -> dict[str, Callable]:
        return {
            "project_home": self._run_project_home_workspace,
            "raw_pattern": self._run_raw_pattern_workspace,
            "background": self.preview_background,
            "smoothing": self.preview_smoothing,
            "instrument": lambda: self._click_workspace_button(
                self.calibration_widget, "qa_button"
            ),
            "peak_list": self.smart_find_peaks_for_selected,
            "size_strain": self.calculate_size_strain_for_selected,
            "residual_stress": self.calculate_residual_stress,
            "phase_identification": self.identify_phases_for_selected,
            "cif_cell": self.match_and_refine_cell_for_selected,
            "cif_library": lambda: self._click_workspace_button(
                self.cif_library_widget, "match_button"
            ),
            "unknown_phase": lambda: self._click_workspace_button(
                self.phase_revolution_widget, "index_button"
            ),
            "solve_structure": lambda: self._click_workspace_button(
                self.structure_solution_widget, "robustness_button"
            ),
            "pawley_lebail": lambda: self._click_workspace_button(
                self.whole_pattern_widget, "run_button"
            ),
            "rietveld": lambda: self._click_workspace_button(
                self.rietveld_widget, "run_button"
            ),
            "multicomponent_refiner": lambda: self._click_workspace_button(
                self.multicomponent_refiner_widget, "run_button"
            ),
            "qpa_preview": self.quantify_selected_phases,
            "validated_qpa": lambda: self._click_workspace_button(
                self.validated_qpa_widget, "calculate_button"
            ),
            "validation": lambda: self._click_workspace_button(
                self.validation_widget, "calculate_button"
            ),
            "batch_reports": lambda: self._click_workspace_button(
                self.batch_widget, "start_button"
            ),
            "full_gui": lambda: self.tabs.setCurrentWidget(
                self.full_gui_workbench_widget
            ),
            "advanced": lambda: self.tabs.setCurrentWidget(
                self.advanced_analysis_tab
            ),
        }

    def _workspace_refresh_callbacks(self) -> dict[str, Callable]:
        return {
            "background": self.populate_background_result,
            "smoothing": self.populate_smoothing_result,
            "peak_list": self.populate_peak_table,
            "size_strain": self.populate_size_strain_results,
            "residual_stress": self.populate_residual_stress_results,
            "phase_identification": self.populate_phase_identification_results,
            "cif_cell": self.populate_crystal_results,
            "qpa_preview": self.populate_qpa_results,
            "instrument": self.calibration_widget.refresh_references,
            "pawley_lebail": lambda: self._call_workspace_methods(
                self.whole_pattern_widget,
                "refresh_references",
                "refresh_for_selected_dataset",
            ),
            "rietveld": lambda: self._call_workspace_methods(
                self.rietveld_widget,
                "refresh_active_cif",
                "refresh_for_selected_dataset",
            ),
            "validated_qpa": self.validated_qpa_widget.refresh_from_rietveld,
            "full_gui": self.full_gui_workbench_widget.refresh_status,
            "batch_reports": lambda: self._call_workspace_methods(
                self.batch_widget,
                "refresh_queue_table",
                "capture_current_recipe",
            ),
        }

    @staticmethod
    def _call_workspace_methods(widget, *method_names: str) -> None:
        for method_name in method_names:
            callback = getattr(widget, method_name, None)
            if callable(callback):
                callback()

    def _click_workspace_button(self, owner, button_name: str) -> None:
        button = getattr(owner, button_name, None)
        if button is None or not button.isEnabled():
            raise RuntimeError(
                "The current analysis cannot start until its prerequisites are complete."
            )
        button.click()

    def _run_project_home_workspace(self) -> None:
        if self.selected_dataset() is None:
            self.import_patterns()
        else:
            self._continue_workflow()

    def _run_raw_pattern_workspace(self) -> None:
        if self.selected_dataset() is None:
            self.import_patterns()
            return
        self.redraw()
        self.statusBar().showMessage("Raw pattern ready for inspection.")

    def _run_current_workflow_action(self) -> None:
        task = self.active_workflow_task
        dataset = self.selected_dataset()
        task_state = self.workflow_task_states.get(task)
        if task_state is not None and not task_state.enabled:
            task_label = TASK_BY_KEY.get(task).label if task in TASK_BY_KEY else task
            message = f"{task_label} is blocked. {task_state.reason}"
            self.statusBar().showMessage(message, 10000)
            if hasattr(self, "progress_drawer"):
                self.progress_drawer.set_status(message)
            if hasattr(self, "workflow_navigation"):
                self.workflow_navigation.set_active_task(task)
            return
        try:
            self.workflow_controller.run_active()
        except (RuntimeError, ValueError) as exc:
            self.statusBar().showMessage(str(exc), 10000)

    def _cancel_active_operation(self) -> None:
        current = self.tabs.currentWidget() if hasattr(self, "tabs") else None
        owners = [
            self.multicomponent_refiner_widget,
            self.rietveld_widget,
            self.whole_pattern_widget,
            self.structure_solution_widget,
            self.phase_revolution_widget,
            self.batch_widget,
        ]
        ordered = ([current] if current is not None else []) + [row for row in owners if row is not current]
        for owner in ordered:
            button = getattr(owner, "cancel_button", None)
            if button is not None and button.isEnabled():
                button.click()
                self.progress_drawer.set_status("Cancellation requested…")
                return
        self.progress_drawer.set_status("No cancellable analysis is currently running.")
        self.progress_drawer.cancel_button.setEnabled(False)

    def _mirror_progress_bar(self, bar: QProgressBar, label: str) -> None:
        maximum = max(1, int(bar.maximum()))
        value = max(0, min(int(bar.value()), maximum))
        self.progress_drawer.set_progress(value, maximum, label=f"{label}: {value}/{maximum}")
        if hasattr(self, "analysis_action_bar"):
            self.analysis_action_bar.set_progress(value, maximum, label=label)

    def _bind_progress_drawer_sources(self) -> None:
        if not hasattr(self, "progress_drawer"):
            return
        sources = (
            (getattr(self, "whole_pattern_widget", None), "progress", "Pawley / Le Bail refinement"),
            (getattr(self, "rietveld_widget", None), "progress", "Rietveld refinement"),
            (getattr(self, "multicomponent_refiner_widget", None), "progress_bar", "Multiphase refinement"),
            (getattr(self, "structure_solution_widget", None), "progress", "Structure robustness"),
            (getattr(self, "batch_widget", None), "progress", "Batch analysis"),
        )
        for owner, attribute, label in sources:
            bar = getattr(owner, attribute, None) if owner is not None else None
            if isinstance(bar, QProgressBar):
                bar.valueChanged.connect(
                    lambda value, source=bar, name=label: self._mirror_progress_bar(source, name)
                )

    def show_scientific_state(self):
        dataset = self.selected_dataset()
        if dataset is None:
            QMessageBox.information(self, "No dataset", "Select a dataset first.")
            return
        self._sync_unified_scientific_state(dataset.uid)
        state = self.scientific_state.ensure(dataset.uid)
        dialog = ScientificStateDialog(state, dataset.name, self)
        dialog.acceptRequested.connect(
            lambda key: self._accept_scientific_stage(dataset.uid, key)
        )
        dialog.exec()
        self._update_workflow_dashboard()

    def _accept_scientific_stage(self, dataset_uid: str, stage: str):
        state = self.scientific_state.ensure(dataset_uid)
        try:
            node = state.accept(stage)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot accept result", str(exc))
            return
        self.statusBar().showMessage(
            f"Accepted {SCIENTIFIC_STAGE_LABELS[stage]} revision {node.revision}."
        )
        self._update_workflow_dashboard()

    def accept_current_scientific_result(self):
        dataset = self.selected_dataset()
        if dataset is None:
            QMessageBox.information(self, "No dataset", "Select a dataset first.")
            return
        self._sync_unified_scientific_state(dataset.uid)
        state = self.scientific_state.ensure(dataset.uid)
        stage = state.latest_current_key()
        if stage is None:
            QMessageBox.information(
                self,
                "Nothing to accept",
                "No current scientific result exists for the selected dataset.",
            )
            return
        self._accept_scientific_stage(dataset.uid, stage)

    def _workflow_flags(self) -> dict[str, bool]:
        dataset = self.selected_dataset()
        if dataset is None:
            return {key: False for key in SCIENTIFIC_STAGE_ORDER}
        self._sync_unified_scientific_state(dataset.uid)
        return self.scientific_state.ensure(dataset.uid).workflow_flags()

    def _workflow_task_context(self, dataset) -> dict[str, object]:
        """Collect task-specific readiness without duplicating engine validation."""
        if dataset is None:
            return {
                "has_dataset": False,
                "has_instrument_profile": False,
                "has_reference_structure": False,
                "has_candidate_cell": False,
                "has_candidate_cell_or_structure": False,
                "has_atomic_structure": False,
                "has_cif_candidates": False,
                "has_structure_refinement": False,
            }

        uid = dataset.uid
        candidate_cells = getattr(
            getattr(self, "phase_revolution_widget", None),
            "candidates_by_uid",
            {},
        ).get(uid, [])
        provisional = getattr(
            getattr(self, "structure_solution_widget", None),
            "provisional_by_uid",
            {},
        ).get(uid)
        rietveld_widget = getattr(self, "rietveld_widget", None)
        doping_series_widget = getattr(self, "doping_series_widget", None)
        whole_widget = getattr(self, "whole_pattern_widget", None)
        multiphase_widget = getattr(self, "multicomponent_refiner_widget", None)
        validated_qpa_widget = getattr(self, "validated_qpa_widget", None)
        cif_library_widget = getattr(self, "cif_library_widget", None)

        reference_structure = bool(getattr(self, "reference_structure", None))
        rietveld_structures = bool(getattr(rietveld_widget, "structures", []))
        atomic_structure = bool(reference_structure or provisional or rietveld_structures)
        phase_table = getattr(multiphase_widget, "phase_table", None)
        phase_rows = phase_table.rowCount() if phase_table is not None else 0
        has_cif_candidates = bool(phase_rows or reference_structure or rietveld_structures)
        rietveld_result = bool(getattr(rietveld_widget, "results_by_uid", {}).get(uid))
        whole_result = bool(getattr(whole_widget, "results_by_uid", {}).get(uid))
        validation_current = self.workflow_states.get("validation") == "Complete"
        stress_result = self.residual_stress_result or {}
        stress_matches = bool(stress_result) and (
            not isinstance(stress_result, dict)
            or stress_result.get("dataset_uid") in {None, uid}
        )

        return {
            "has_dataset": True,
            "has_instrument_profile": bool(self.active_instrument_profile),
            "has_reference_structure": reference_structure,
            "has_candidate_cell": bool(candidate_cells),
            "has_candidate_cell_or_structure": bool(candidate_cells or atomic_structure),
            "has_atomic_structure": atomic_structure,
            "has_cif_candidates": has_cif_candidates,
            "has_structure_refinement": rietveld_result,
            "background_complete": bool(self.background_results.get(uid, {}).get("applied")),
            "smoothing_complete": bool(self.smoothing_results.get(uid, {}).get("applied")),
            "instrument_complete": bool(self.active_instrument_profile),
            "peak_list_complete": bool(active_peak_rows(self.peak_rows.get(uid, []))),
            "size_strain_complete": bool(self.size_strain_results.get(uid)),
            "residual_stress_complete": stress_matches,
            "known_phase_complete": bool(self.phase_identification_results.get(uid)),
            "cell_refinement_complete": bool(self.cell_refinement_results.get(uid)),
            "cif_library_match_complete": bool(getattr(cif_library_widget, "last_match_result", None)),
            "unknown_phase_complete": bool(candidate_cells),
            "structure_complete": bool(provisional),
            "whole_pattern_complete": whole_result,
            "rietveld_complete": rietveld_result,
            "doping_series_complete": bool(
                getattr(doping_series_widget, "series_result", None)
            ),
            "multicomponent_complete": bool(getattr(multiphase_widget, "results_by_uid", {}).get(uid)),
            "qpa_preview_complete": bool(self.qpa_results.get(uid)),
            "validated_qpa_complete": bool(getattr(validated_qpa_widget, "native_results_by_uid", {}).get(uid)),
            "validation_complete": validation_current,
            "report_complete": False,
        }

    def _update_workflow_dashboard(self):
        if not hasattr(self, "workflow_header"):
            return
        flags = self._workflow_flags()
        self.workflow_states = build_workflow_status(flags)
        dataset = self.selected_dataset()
        if dataset is not None:
            scientific = self.scientific_state.ensure(dataset.uid)
            for key in SCIENTIFIC_STAGE_ORDER:
                node_status = scientific.node(key).status
                if node_status in {"Outdated", "Invalid"}:
                    self.workflow_states[key] = node_status
        self.workflow_context = self._workflow_task_context(dataset)
        self.workflow_task_states = build_task_states(
            self.workflow_states,
            self.workflow_context,
            mode=self.workflow_mode,
        )
        self.workflow_header.set_statuses(self.workflow_states)
        self.workflow_header.set_task_states(self.workflow_task_states)
        if hasattr(self, "workflow_navigation"):
            self.workflow_navigation.set_statuses(self.workflow_states)
            self.workflow_navigation.set_task_states(self.workflow_task_states)
        self._refresh_shell_header(dataset)
        if dataset is None:
            self.project_home_widget.update_summary(
                dataset="No dataset selected",
                pattern="Not prepared",
                peaks="No curated peaks",
                phase="No accepted phase or cell",
                refinement="No current refinement",
                validation="Not validated",
                notice="Import a powder pattern to begin the guided workflow.",
                has_dataset=False,
            )
            self._sync_analysis_panel_for_task()
            return
        uid = dataset.uid
        peaks = active_peak_rows(self.peak_rows.get(uid, []))
        candidate_count = len(
            getattr(self.phase_revolution_widget, "candidates_by_uid", {}).get(uid, [])
        )
        has_known_phase = bool(self.phase_identification_results.get(uid))
        has_whole = bool(
            getattr(self.whole_pattern_widget, "results_by_uid", {}).get(uid)
        )
        has_rietveld = bool(
            getattr(self.rietveld_widget, "results_by_uid", {}).get(uid)
        )
        has_validation = self.workflow_states.get("validation") == "Complete"
        next_key = recommended_task_key(
            self.workflow_states,
            self.workflow_context,
            self.workflow_mode,
            self.workflow_task_states,
        )
        next_label = TASK_BY_KEY[next_key].label
        scientific = self.scientific_state.ensure(uid)
        counts = scientific.counts()
        outdated_note = (
            f" {counts['outdated']} downstream result(s) are outdated and preserved for review."
            if counts["outdated"]
            else ""
        )
        accepted_note = (
            f" {counts['accepted']} current result(s) are explicitly accepted."
            if counts["accepted"]
            else ""
        )
        self.project_home_widget.update_summary(
            dataset=f"{dataset.name} — {len(dataset.x):,} points",
            pattern=(
                "Prepared pattern available"
                if flags["preparation"]
                else "Raw measurement only"
            ),
            peaks=(
                f"{len(peaks)} included reflection(s)"
                if peaks
                else "No authoritative peak list"
            ),
            phase=(
                f"{candidate_count} indexed candidate cell(s)"
                if candidate_count
                else ("Known-phase candidates available" if has_known_phase else "No accepted phase or cell")
            ),
            refinement=(
                "Rietveld result available"
                if has_rietveld
                else ("Pawley / Le Bail result available" if has_whole else "No current refinement")
            ),
            validation=("Validation evidence available" if has_validation else "Not validated"),
            notice=f"Recommended next step: {next_label}." + outdated_note + accepted_note,
            has_dataset=True,
        )
        self._sync_analysis_panel_for_task()
