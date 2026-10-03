from __future__ import annotations

from ..main_window_dependencies import *


class AnalysisPanelMixin:
    def _placeholder_tab(self, title: str, text: str):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        description = QLabel(text)
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignTop)
        layout.addWidget(heading)
        layout.addWidget(description)
        layout.addStretch(1)
        return widget

    @staticmethod
    def _analysis_panel_descriptors() -> dict[str, AnalysisPanelDescriptor]:
        return {
            "project_home": AnalysisPanelDescriptor(
                "Project overview",
                "Review the active dataset, scientific state, and recommended next action.",
                "Project",
            ),
            "raw_pattern": AnalysisPanelDescriptor(
                "Pattern inspection",
                "Inspect the imported measurement before any irreversible interpretation.",
                "Import",
            ),
            "background": AnalysisPanelDescriptor(
                "Background correction",
                "Choose a baseline model, preview it, then apply only after visual inspection.",
                "Prepare",
            ),
            "smoothing": AnalysisPanelDescriptor(
                "Smart smoothing",
                "Control noise reduction while preserving peak position, height, and width.",
                "Prepare",
            ),
            "instrument": AnalysisPanelDescriptor(
                "Instrument calibration",
                "Validate wavelength, zero shift, and instrumental broadening using the central workspace.",
                "Prepare",
            ),
            "peak_list": AnalysisPanelDescriptor(
                "Peak detection and fitting",
                "Create the authoritative peak list, then inspect every included reflection.",
                "Peaks",
            ),
            "size_strain": AnalysisPanelDescriptor(
                "Size and strain",
                "Calculate Scherrer and Williamson–Hall results from fitted peak widths.",
                "Peaks",
            ),
            "residual_stress": AnalysisPanelDescriptor(
                "Residual stress",
                "Build ψ observations and calculate stress with material-valid elastic constants.",
                "Peaks",
            ),
            "phase_identification": AnalysisPanelDescriptor(
                "Phase identification",
                "Rank local reference patterns against the authoritative observed peak list.",
                "Phase",
            ),
            "cif_cell": AnalysisPanelDescriptor(
                "CIF and unit-cell refinement",
                "Match observed reflections and refine lattice parameters against the active CIF.",
                "Phase",
            ),
            "cif_library": AnalysisPanelDescriptor(
                "CIF library search",
                "Search and inspect structure matches in the central scientific workspace.",
                "Phase",
            ),
            "unknown_phase": AnalysisPanelDescriptor(
                "Unknown-phase indexing",
                "Generate and rank candidate unit cells using the curated peak list.",
                "Phase",
            ),
            "solve_structure": AnalysisPanelDescriptor(
                "Structure solution",
                "Test provisional structures and robustness in the central workspace.",
                "Phase",
            ),
            "pawley_lebail": AnalysisPanelDescriptor(
                "Pawley / Le Bail refinement",
                "Refine the whole pattern without atomic-coordinate refinement.",
                "Refine",
            ),
            "rietveld": AnalysisPanelDescriptor(
                "Rietveld refinement",
                "Refine the structural model and inspect fit statistics, difference, and Bragg marks.",
                "Refine",
            ),
            "doping_series": AnalysisPanelDescriptor(
                "Doping-series comparison",
                "Track pattern and peak evolution, then run auditable independent or carry-forward refinements.",
                "Refine",
            ),
            "multicomponent_refiner": AnalysisPanelDescriptor(
                "Intelligent multiphase refinement",
                "Search, refine, and compare multiple phase combinations without blocking the GUI.",
                "Refine",
            ),
            "qpa_preview": AnalysisPanelDescriptor(
                "Exploratory QPA",
                "Estimate phase fractions and inspect residuals before validated quantification.",
                "Refine",
            ),
            "validated_qpa": AnalysisPanelDescriptor(
                "Validated QPA",
                "Run the validated quantification engine in the central workspace.",
                "Refine",
            ),
            "validation": AnalysisPanelDescriptor(
                "Validation evidence",
                "Audit assumptions, robustness, provenance, and result freshness.",
                "Report",
            ),
            "batch_reports": AnalysisPanelDescriptor(
                "Batch and reports",
                "Apply controlled recipes and export complete scientific evidence packages.",
                "Report",
            ),
            "full_gui": AnalysisPanelDescriptor(
                "Full GUI workbench",
                "Access specialist tools while keeping the workflow state synchronized.",
                "Expert",
            ),
            "advanced": AnalysisPanelDescriptor(
                "Advanced tools",
                "Inspect all legacy expert controls and planned specialist analyses.",
                "Expert",
            ),
        }

    @staticmethod
    def _analysis_task_groups() -> dict[str, tuple[str, ...]]:
        return {
            "project_home": ("appearance", "export"),
            "raw_pattern": ("appearance", "stacked_patterns", "export"),
            "background": ("preprocessing", "export"),
            "smoothing": ("preprocessing", "export"),
            "peak_list": ("peaks", "fitting", "export"),
            "size_strain": ("size_strain", "export"),
            "residual_stress": ("residual_stress", "export"),
            "phase_identification": ("phase_identification", "export"),
            "cif_cell": ("cif_cell", "export"),
            "qpa_preview": ("qpa_preview", "export"),
            "instrument": ("export",),
            "cif_library": ("export",),
            "unknown_phase": ("export",),
            "solve_structure": ("export",),
            "pawley_lebail": ("export",),
            "rietveld": ("export",),
            "doping_series": ("export",),
            "multicomponent_refiner": ("export",),
            "validated_qpa": ("export",),
            "validation": ("export",),
            "batch_reports": ("export",),
            "full_gui": tuple(),
            "advanced": (
                "appearance",
                "preprocessing",
                "peaks",
                "fitting",
                "size_strain",
                "cif_cell",
                "phase_identification",
                "qpa_preview",
                "residual_stress",
                "stacked_patterns",
                "export",
            ),
        }

    @staticmethod
    def _analysis_run_labels() -> dict[str, str]:
        return {
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
            "doping_series": "Compare doping series",
            "multicomponent_refiner": "Run multiphase search",
            "qpa_preview": "Calculate QPA preview",
            "validated_qpa": "Calculate validated QPA",
            "validation": "Calculate validation",
            "batch_reports": "Start batch",
            "full_gui": "Open workbench",
            "advanced": "Open advanced tools",
        }

    def _connect_analysis_panel_controls(self) -> None:
        content = getattr(self, "analysis_control_content", None)
        if content is None:
            return
        for widget in content.findChildren(QCheckBox):
            widget.toggled.connect(self._analysis_parameters_changed)
        for widget in content.findChildren(NoWheelComboBox):
            widget.currentTextChanged.connect(self._analysis_parameters_changed)
        for widget in content.findChildren(NoWheelDoubleSpinBox):
            widget.valueChanged.connect(self._analysis_parameters_changed)
        for widget in content.findChildren(NoWheelSpinBox):
            widget.valueChanged.connect(self._analysis_parameters_changed)
        for widget in content.findChildren(QLineEdit):
            widget.textChanged.connect(self._analysis_parameters_changed)

    def _analysis_parameters_changed(self, *args) -> None:
        if not getattr(self, "_workflow_syncing", False):
            self._sync_analysis_panel_for_task(reset_scroll=False)

    def _remember_analysis_inspector_preferences(self) -> None:
        toolbar = getattr(self, "analysis_inspector_toolbar", None)
        sections = getattr(self, "analysis_inspector_sections", {})
        if toolbar is None:
            return
        preferences = dict(getattr(self.application_state, "ui_preferences", {}) or {})
        preferences["analysis_inspector_mode"] = toolbar.current_mode()
        preferences["analysis_inspector_expanded"] = {
            key: section.is_expanded() for key, section in sections.items()
        }
        self.application_state.ui_preferences = preferences

    def _restore_analysis_inspector_state(self, ui_state) -> None:
        if not isinstance(ui_state, dict):
            return
        toolbar = getattr(self, "analysis_inspector_toolbar", None)
        sections = getattr(self, "analysis_inspector_sections", {})
        if toolbar is None:
            return
        toolbar.set_mode(ui_state.get("analysis_inspector_mode", "essential"))
        expanded = ui_state.get("analysis_inspector_expanded", {})
        if isinstance(expanded, dict):
            for key, value in expanded.items():
                section = sections.get(str(key))
                if section is not None:
                    section.set_expanded(bool(value))

    def _analysis_inspector_query_changed(self, *args) -> None:
        self._sync_analysis_panel_for_task(reset_scroll=False)

    def _analysis_inspector_mode_changed(self, mode: str) -> None:
        self._remember_analysis_inspector_preferences()
        self._sync_analysis_panel_for_task(reset_scroll=False)

    def _analysis_inspector_expansion_changed(self, key: str, expanded: bool) -> None:
        self._remember_analysis_inspector_preferences()

    def _set_visible_analysis_sections_expanded(self, expanded: bool) -> None:
        for section in getattr(self, "analysis_inspector_sections", {}).values():
            if section.isVisible():
                section.set_expanded(expanded, emit=True)
        self._remember_analysis_inspector_preferences()

    def _sync_preprocessing_rows(self, task: str) -> None:
        form = getattr(self, "preprocessing_form", None)
        if form is None or not hasattr(form, "setRowVisible"):
            return
        show_background = task in {"background", "advanced"}
        show_smoothing = task in {"smoothing", "advanced"}
        for widget in self.analysis_preprocessing_background_widgets:
            form.setRowVisible(widget, show_background)
        for widget in self.analysis_preprocessing_smoothing_widgets:
            form.setRowVisible(widget, show_smoothing)
        for widget in self.analysis_preprocessing_shared_widgets:
            form.setRowVisible(widget, show_background or show_smoothing)
        group = self.analysis_control_groups.get("preprocessing")
        if group is not None:
            if task == "background":
                title = "Background parameters"
            elif task == "smoothing":
                title = "Smoothing parameters"
            else:
                title = "Preprocessing parameters"
            section = getattr(self, "analysis_inspector_sections", {}).get(
                "preprocessing"
            )
            if section is not None:
                section.set_title(title)
                group.setTitle("")
            else:
                group.setTitle(title)

    def _analysis_validation_state(self) -> tuple[str, str, str, str, bool]:
        task = self.active_workflow_task
        dataset = self.selected_dataset()
        state = self.workflow_task_states.get(task)
        if state is not None and not state.enabled:
            return (
                "error",
                "Task blocked",
                state.reason,
                "Complete the stated prerequisite, then return to this task.",
                False,
            )
        if dataset is None and task not in {"project_home", "full_gui", "advanced"}:
            return (
                "warning",
                "No dataset selected",
                "This analysis requires an imported powder pattern.",
                "Import or select a dataset before running the task.",
                False,
            )
        if task == "project_home":
            if dataset is None:
                return (
                    "info",
                    "Start a project",
                    "Import a powder XRD pattern to begin the controlled workflow.",
                    "Supported text and instrument formats remain available from Import XRD.",
                    True,
                )
            return (
                "success",
                "Project ready",
                "The selected dataset is synchronized with the scientific workflow state.",
                "Continue to the recommended next task.",
                True,
            )
        if task == "smoothing" and self.smoothing_strength.value() <= 0.0:
            return (
                "warning",
                "Zero smoothing strength",
                "The current setting will not materially smooth the pattern.",
                "Increase smoothing strength or leave smoothing unapplied.",
                True,
            )
        if task == "qpa_preview" and self.qpa_internal_standard_check.isChecked():
            if self.qpa_internal_standard_combo.currentData() is None:
                return (
                    "error",
                    "Internal standard missing",
                    "Internal-standard correction is enabled, but no included standard phase is selected.",
                    "Choose an internal-standard phase or disable the correction.",
                    False,
                )
        if task == "residual_stress":
            return (
                "warning",
                "Material constants require verification",
                "The elastic constants shown are suggested starting points requiring optimization and replacement with values valid for the selected phase and hkl reflection.",
                "Verify E, ν or the diffraction elastic constant before publication use.",
                True,
            )
        groups = self._analysis_task_groups().get(task, tuple())
        if not groups or groups == ("export",):
            return (
                "info",
                "Controls in central workspace",
                "This task uses its dedicated parameter panel beside the scientific plot or table.",
                "Use the sticky Run button here or the matching controls in the central workspace.",
                True,
            )
        return (
            "success",
            "Parameters valid",
            "All required values are within their accepted input ranges.",
            "Review the preview and scientific assumptions before accepting results.",
            True,
        )

    def _analysis_result_summary(self) -> tuple[str, list[tuple[str, str]], str, str]:
        task = self.active_workflow_task
        dataset = self.selected_dataset()
        if dataset is None:
            return "No result", [], "Import a dataset to populate this summary.", "pending"
        uid = dataset.uid
        if task in {"project_home", "raw_pattern"}:
            x = np.asarray(dataset.x, dtype=float)
            metrics = [
                ("Points", f"{len(x):,}"),
                ("2θ range", f"{float(np.min(x)):.4f}–{float(np.max(x)):.4f}°" if len(x) else "—"),
                ("Active signal", "Processed" if dataset.y_processed is not None else "Raw"),
            ]
            return "Available", metrics, f"Source: {Path(dataset.source_path).name if dataset.source_path else 'project data'}", "success"
        if task == "background":
            result = self.background_results.get(uid, {})
            applied = bool(result.get("applied"))
            status = "Applied" if applied else ("Preview ready" if result else "Not run")
            metrics = [
                ("Method", str(result.get("method") or self.background_method.currentText())),
                ("Subtraction", "Current" if applied else "Not applied"),
            ]
            return status, metrics, "Inspect broad features and negative residuals before accepting the baseline.", "success" if applied else "pending"
        if task == "smoothing":
            result = self.smoothing_results.get(uid, {})
            applied = bool(result.get("applied"))
            metrics = [
                ("Method", str(result.get("method") or self.smoothing_method.currentText())),
                ("Strength", f"{self.smoothing_strength.value():.1f}%"),
                ("Applied", "Yes" if applied else "No"),
            ]
            return "Applied" if applied else ("Preview ready" if result else "Not run"), metrics, "Peak-preservation limits remain part of the exported processing record.", "success" if applied else "pending"
        if task == "peak_list":
            rows = normalize_peak_rows(self.peak_rows.get(uid, []))
            active = active_peak_rows(rows)
            manual = sum(1 for row in rows if str(row.get("source", "")).lower() == "manual")
            protected = sum(1 for row in rows if bool(row.get("protected")))
            fits = len(self.fit_groups.get(uid, []))
            metrics = [
                ("Included peaks", str(len(active))),
                ("Manual peaks", str(manual)),
                ("Protected", str(protected)),
                ("Fitted groups", str(fits)),
            ]
            return ("Current" if active else "Not run"), metrics, "Only included reflections are passed to downstream indexing and phase analysis.", "success" if active else "pending"
        if task == "size_strain":
            result = self.size_strain_results.get(uid, {})
            wh = result.get("williamson_hall", {}) if isinstance(result, dict) else {}
            metrics = [
                ("Valid peaks", str(result.get("scherrer_summary", {}).get("valid_peak_count", 0)) if result else "0"),
                ("WH size", self._format_optional(wh.get("crystallite_size_nm"), 5, " nm")),
                ("Microstrain", self._format_optional(wh.get("microstrain"), 5)),
                ("R²", self._format_optional(wh.get("r_squared"), 5)),
            ]
            return ("Available" if result else "Not run"), metrics, str(wh.get("note") or "Fit at least three valid peaks for Williamson–Hall regression."), "success" if result else "pending"
        if task == "residual_stress":
            result = self.residual_stress_result if isinstance(self.residual_stress_result, dict) else {}
            matches = bool(result) and result.get("dataset_uid") in {None, uid}
            metrics = [
                ("Observations", str(len(self.residual_stress_observations))),
                ("Stress", self._format_optional(result.get("stress_mpa"), 6, " MPa")),
                ("Uncertainty", self._format_optional(result.get("stress_error_mpa"), 4, " MPa")),
                ("R²", self._format_optional(result.get("r_squared"), 5)),
            ]
            return ("Available" if matches else "Not run"), metrics, "Positive stress is tensile; negative stress is compressive.", "success" if matches else "pending"
        if task == "phase_identification":
            result = self.phase_identification_results.get(uid, {})
            best = result.get("best") if isinstance(result, dict) else None
            metrics = [
                ("Candidates", str(len(result.get("ranked_candidates", result.get("candidates", [])))) if result else "0"),
                ("Best match", str(best.get("reference_name", "—")) if best else "—"),
                ("Score", self._format_optional(best.get("score") if best else None, 5)),
                ("Matched peaks", str(best.get("matched_count", "—")) if best else "—"),
            ]
            return ("Available" if best else "Not run"), metrics, "A ranked match is evidence, not automatic phase acceptance.", "success" if best else "pending"
        if task == "cif_cell":
            result = self.cell_refinement_results.get(uid, {})
            metrics = [
                ("Structure", str(self.reference_structure.get("name", "Active CIF")) if self.reference_structure else "None"),
                ("Matched peaks", str(result.get("matched_count", result.get("n_matches", "—"))) if result else "—"),
                ("RMS error", self._format_optional(result.get("rms_error_deg") if result else None, 5, "°")),
            ]
            return ("Available" if result else "Not run"), metrics, "Coordinate and crystallographic provenance remain attached to the active CIF.", "success" if result else "pending"
        if task == "qpa_preview":
            result = self.qpa_results.get(uid, {})
            diagnostics = result.get("diagnostics", {}) if isinstance(result, dict) else {}
            phases = result.get("phases", []) if isinstance(result, dict) else []
            metrics = [
                ("Phases", str(len(phases))),
                ("Weighted residual", self._format_optional(diagnostics.get("weighted_profile_residual_percent"), 5, "%")),
                ("R²", self._format_optional(diagnostics.get("r_squared"), 6)),
                ("Mode", str(result.get("mode", self.qpa_mode.currentText())) if result else self.qpa_mode.currentText()),
            ]
            return ("Available" if result else "Not run"), metrics, "Exploratory whole-pattern scaling is not a substitute for validated Rietveld QPA.", "success" if result else "pending"
        state = self.workflow_task_states.get(task)
        status = state.status if state is not None else "Pending"
        result_state = "success" if status == "Complete" else ("warning" if status in {"Outdated", "Invalid"} else "pending")
        note = state.reason if state is not None else "Detailed results are shown in the central workspace."
        return status, [], note, result_state

    def _sync_analysis_panel_for_task(self, reset_scroll: bool = True) -> None:
        if not hasattr(self, "analysis_panel_header"):
            return
        task = self.active_workflow_task
        descriptor = self._analysis_panel_descriptors().get(
            task,
            AnalysisPanelDescriptor("Analysis controls", "Review the active scientific task.", "Analysis"),
        )
        self.analysis_panel_header.set_descriptor(descriptor)

        requested = set(self._analysis_task_groups().get(task, ("export",)))
        sections = getattr(self, "analysis_inspector_sections", {})
        for section in sections.values():
            section.clear_filters()
        self._sync_preprocessing_rows(task)
        toolbar = getattr(self, "analysis_inspector_toolbar", None)
        inspector_mode = toolbar.current_mode() if toolbar is not None else "all"
        include_advanced = inspector_mode == "all"
        query = toolbar.search.text() if toolbar is not None else ""
        visible_roots = []
        visible_control_count = 0
        for key, group in self.analysis_control_groups.items():
            visible = key in requested
            if visible and bool(group.property("expertOnly")) and not include_advanced:
                visible = False
            section = sections.get(key)
            if visible and section is not None:
                has_match, count = section.apply_filters(
                    query,
                    include_advanced=include_advanced,
                )
                visible = has_match
                visible_control_count += count if visible else 0
                section.setVisible(visible)
            elif section is not None:
                section.set_search_override(False)
                section.setVisible(False)
            else:
                group.setVisible(visible)
            if visible and key != "export":
                visible_roots.append(group)

        if toolbar is not None:
            toolbar.set_summary(visible_control_count, query)

        if reset_scroll and getattr(self, "_analysis_panel_last_task", None) != task:
            self.analysis_control_scroll.verticalScrollBar().setValue(0)
        self._analysis_panel_last_task = task

        level, heading, message, action, local_valid = self._analysis_validation_state()
        self.analysis_validation_card.set_state(level, heading, message, action)
        status, metrics, note, result_state = self._analysis_result_summary()
        self.analysis_result_card.set_summary(
            status=status,
            metrics=metrics,
            note=note,
            state=result_state,
        )

        state = self.workflow_task_states.get(task)
        run_enabled = local_valid and (bool(state.enabled) if state is not None else True)
        preview_labels = {
            "raw_pattern": "Refresh plot",
            "background": "Preview background",
            "smoothing": "Preview smoothing",
        }
        preview_enabled = task in preview_labels and self.selected_dataset() is not None
        run_text = self._analysis_run_labels().get(task, "Run analysis")
        tooltip = state.reason if state is not None else descriptor.subtitle
        self.analysis_action_bar.configure(
            run_text=run_text,
            run_enabled=run_enabled,
            preview_text=preview_labels.get(task, "Preview"),
            preview_enabled=preview_enabled,
            reset_enabled=bool(visible_roots),
            tooltip=tooltip,
        )

    def _preview_current_analysis(self) -> None:
        callbacks = {
            "raw_pattern": self.redraw,
            "background": self.preview_background,
            "smoothing": self.preview_smoothing,
        }
        callback = callbacks.get(self.active_workflow_task)
        if callback is None:
            self.statusBar().showMessage("No separate preview is defined for this task.")
            return
        callback()
        self._sync_analysis_panel_for_task(reset_scroll=False)

    def _reset_current_analysis_parameters(self) -> None:
        requested = set(self._analysis_task_groups().get(self.active_workflow_task, ()))
        include_advanced = (
            getattr(self, "analysis_inspector_toolbar", None) is None
            or self.analysis_inspector_toolbar.current_mode() == "all"
        )
        roots = [
            group
            for key, group in self.analysis_control_groups.items()
            if key in requested
            and key != "export"
            and (include_advanced or not bool(group.property("expertOnly")))
        ]
        if not roots:
            self.statusBar().showMessage("The current task has no side-panel parameters to reset.")
            return
        for root in roots:
            restore_widget_defaults(
                self._analysis_parameter_defaults,
                root,
                visible_only=False,
            )
        self.statusBar().showMessage("Current task parameters restored to their Phase 21.1.0 defaults.")
        self._sync_analysis_panel_for_task(reset_scroll=False)
