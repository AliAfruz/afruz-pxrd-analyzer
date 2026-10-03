from __future__ import annotations

from ..contracts import ResultContractService
from ..main_window_dependencies import *


class ProjectWindowMixin:
    def _undo_snapshot(self) -> dict:
        return self.project_controller.create_snapshot(
            analysis_state=self._analysis_state(),
            ui_state=self._ui_state(),
            selected_row=(
                self.dataset_list.currentRow()
                if hasattr(self, "dataset_list")
                else -1
            ),
        )

    def _push_undo_checkpoint(self, label: str) -> None:
        if getattr(self, "_undo_redo_restoring", False):
            return
        if self.project_controller.checkpoint(label):
            self._update_undo_redo_actions()
            self._refresh_shell_header()

    def _restore_undo_snapshot(self, snapshot: dict) -> None:
        self._undo_redo_restoring = True
        try:
            restored = self.project_controller.restore_snapshot(snapshot)
            selected_row = restored["selected_row"]
            self._restore_analysis_state(restored["analysis_state"])
            ui_state = restored["ui_state"]
            if isinstance(ui_state, dict):
                self.workflow_mode = str(ui_state.get("workflow_mode", self.workflow_mode))
                self.active_workflow_task = str(ui_state.get("active_workflow_task", self.active_workflow_task))
                self._restore_analysis_inspector_state(ui_state)
                if hasattr(self, "main_splitter") and ui_state.get("splitter_sizes"):
                    self.main_splitter.setSizes(list(ui_state.get("splitter_sizes")))
            self.refresh_dataset_list(selected_row if selected_row >= 0 else None)
            self._refresh_after_undo_redo_restore()
        finally:
            self._undo_redo_restoring = False

    def _refresh_after_undo_redo_restore(self) -> None:
        self._rebuild_backgrounds_from_results()
        self._rebuild_smoothing_results()
        self.populate_background_result()
        self.populate_smoothing_result()
        self.populate_peak_table()
        self.populate_fit_table()
        self.populate_fit_candidate_table()
        self.populate_size_strain_results()
        self.populate_crystal_results()
        self.populate_phase_identification_results()
        self.populate_qpa_results()
        self.populate_residual_stress_results()
        self.workflow_controller.set_dataset(self.selected_dataset())
        self.workflow_controller.refresh(
            ("instrument", "pawley_lebail", "rietveld", "doping_series", "unknown_phase")
        )
        self.redraw()
        self._sync_unified_scientific_state()
        self._update_workflow_dashboard()

    def _update_undo_redo_actions(self) -> None:
        history = getattr(self, "undo_redo_history", None)
        if history is None or not hasattr(self, "undo_action"):
            return
        self.undo_action.setEnabled(history.can_undo)
        self.redo_action.setEnabled(history.can_redo)
        undo_label = history.undo_label
        redo_label = history.redo_label
        self.undo_action.setText(
            "Undo" if not undo_label else f"Undo {undo_label}"
        )
        self.redo_action.setText(
            "Redo" if not redo_label else f"Redo {redo_label}"
        )
        self.undo_action.setStatusTip("Undo the last applied scientific or dataset edit")
        self.redo_action.setStatusTip("Redo the last undone edit")

    def undo_last_change(self) -> None:
        label = self.project_controller.undo()
        self._update_undo_redo_actions()
        if label is None:
            self.statusBar().showMessage("Nothing to undo.")
            return
        self.statusBar().showMessage(f"Undid: {label}.")

    def redo_last_change(self) -> None:
        label = self.project_controller.redo()
        self._update_undo_redo_actions()
        if label is None:
            self.statusBar().showMessage("Nothing to redo.")
            return
        self.statusBar().showMessage(f"Redid: {label}.")

    def selected_dataset(self) -> Dataset | None:
        row = self.dataset_list.currentRow()
        if 0 <= row < len(self.datasets):
            return self.datasets[row]
        return None

    def refresh_dataset_list(self, selected_row: int | None = None):
        self.dataset_list.blockSignals(True)
        self.dataset_list.clear()
        for ds in self.datasets:
            item = QListWidgetItem(ds.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if ds.visible else Qt.Unchecked)
            self.dataset_list.addItem(item)
        self.dataset_list.blockSignals(False)
        if self.datasets:
            row = 0 if selected_row is None else max(0, min(selected_row, len(self.datasets) - 1))
            self.dataset_list.setCurrentRow(row)
        self.update_dataset_info(self.dataset_list.currentRow())
        self.populate_size_strain_results()
        self.populate_crystal_results()
        self.populate_phase_identification_results()
        self.populate_qpa_results()
        self.workflow_controller.refresh(
            (
                "instrument",
                "cif_library",
                "doping_series",
                "multicomponent_refiner",
                "full_gui",
                "batch_reports",
            )
        )
        self.redraw()

    def update_dataset_info(self, row: int):
        ds = self.selected_dataset()
        if ds is None:
            self.info_name.setText("—")
            self.info_points.setText("—")
            self.info_range.setText("—")
            self.info_source.setText("—")
            if hasattr(self, "background_plot"):
                self.populate_background_result()
            if hasattr(self, "freeze_main_peak_list"):
                self._peak_table_updating = True
                try:
                    self.freeze_main_peak_list.setChecked(False)
                    self.freeze_main_peak_list.setEnabled(False)
                finally:
                    self._peak_table_updating = False
                self._update_main_peak_status()
            self.workflow_controller.set_dataset(None)
            self._update_workflow_dashboard()
            return
        self.info_name.setText(ds.name)
        self.info_points.setText(f"{len(ds.x):,}")
        self.info_range.setText(f"{ds.x.min():.5g} – {ds.x.max():.5g}°")
        if ds.source_path:
            compact_source = self.compact_source_path(ds.source_path)
            self.info_source.setText(
                f'<a href="open-source-folder">{compact_source}</a>'
            )
            self.info_source.setToolTip(ds.source_path)
        else:
            self.info_source.setText("Embedded/generated")
            self.info_source.setToolTip("")
        if hasattr(self, "background_plot"):
            self.populate_background_result()
        if hasattr(self, "smoothing_plot"):
            self.populate_smoothing_result()
        if hasattr(self, "size_table"):
            self.populate_size_strain_results()
        if hasattr(self, "reference_table"):
            self.populate_crystal_results()
        if hasattr(self, "phase_candidate_table"):
            self.populate_phase_identification_results()
        if hasattr(self, "qpa_fraction_table"):
            self.populate_qpa_results()
        self.workflow_controller.set_dataset(ds)
        self.workflow_controller.refresh(("validated_qpa",))
        if hasattr(self, "manual_peak_position"):
            self.manual_peak_position.setRange(
                float(np.min(ds.x)),
                float(np.max(ds.x)),
            )
            if not (
                float(np.min(ds.x))
                <= self.manual_peak_position.value()
                <= float(np.max(ds.x))
            ):
                self.manual_peak_position.setValue(
                    float(0.5 * (np.min(ds.x) + np.max(ds.x)))
                )
        if hasattr(self, "freeze_main_peak_list"):
            self._peak_table_updating = True
            try:
                self.freeze_main_peak_list.setEnabled(True)
                self.freeze_main_peak_list.setChecked(
                    self._main_peak_list_is_locked(ds.uid)
                )
            finally:
                self._peak_table_updating = False
            self._update_main_peak_status()
        self._update_workflow_dashboard()

    def on_dataset_item_changed(self, item: QListWidgetItem):
        row = self.dataset_list.row(item)
        if 0 <= row < len(self.datasets):
            self._push_undo_checkpoint("Change dataset visibility")
            self.datasets[row].visible = item.checkState() == Qt.Checked
            self.redraw()

    def redraw(self):
        self.plot_widget.redraw(
            self.datasets,
            stack_enabled=self.stack_check.isChecked(),
            stack_offset=self.stack_offset.value(),
            stack_scale=self.stack_scale.value(),
            backgrounds=self.backgrounds,
            peak_rows=self.peak_rows,
            fit_groups=self.fit_groups,
            show_fits=self.show_fits_check.isChecked(),
            show_fit_components=self.show_fit_components_check.isChecked(),
            show_fit_baselines=self.show_fit_baselines_check.isChecked(),
            show_residuals=self.show_residuals_check.isChecked(),
            reference_pattern=self.reference_pattern,
            show_reference=self.show_reference_check.isChecked(),
        )
        self._redraw_peak_workspace()
        self._update_workflow_dashboard()

    @staticmethod
    def compact_source_path(source_path: str, maximum: int = 42) -> str:
        """Return a short HTML-safe display path without exposing the full path."""
        path = Path(source_path)
        filename = path.name or str(path)
        parent = path.parent.name

        if parent and parent not in (".", "/"):
            display = f"…/{parent}/{filename}"
        else:
            display = f"…/{filename}"

        if len(display) > maximum:
            suffix = path.suffix
            stem = path.stem
            available = max(10, maximum - len(parent) - len(suffix) - 7)
            shortened_name = (
                f"{stem[:available]}…{suffix}"
                if suffix
                else f"{filename[:available]}…"
            )
            display = (
                f"…/{parent}/{shortened_name}"
                if parent and parent not in (".", "/")
                else f"…/{shortened_name}"
            )

        return html.escape(display)

    def open_selected_source_folder(self, _link: str = ""):
        ds = self.selected_dataset()
        if ds is None or not ds.source_path:
            return

        source = Path(ds.source_path)
        folder = source.parent if source.suffix else source
        if not folder.exists():
            QMessageBox.warning(
                self,
                "Source path unavailable",
                f"The source folder no longer exists:\n{folder}",
            )
            return

        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        if not opened:
            QMessageBox.warning(
                self,
                "Could not open folder",
                f"The operating system could not open:\n{folder}",
            )

    def closeEvent(self, event):
        if self._deconvolution_is_running():
            self.cancel_advanced_deconvolution()
            thread = self._deconvolution_thread
            if thread is not None:
                thread.quit()
                thread.wait(2000)
        if (
            hasattr(self, "whole_pattern_widget")
            and self.whole_pattern_widget.is_running()
        ):
            self.whole_pattern_widget.cancel_refinement()
            thread = self.whole_pattern_widget._thread
            if thread is not None:
                thread.quit()
                thread.wait(3000)
        if (
            hasattr(self, "rietveld_widget")
            and self.rietveld_widget.is_running()
        ):
            self.rietveld_widget.cancel_refinement()
            thread = self.rietveld_widget._thread
            if thread is not None:
                thread.quit()
                thread.wait(3000)
        if (
            hasattr(self, "doping_series_widget")
            and self.doping_series_widget.is_running()
        ):
            self.doping_series_widget.cancel_analysis()
            thread = self.doping_series_widget._thread
            if thread is not None:
                thread.quit()
                thread.wait(3000)
        validated_qpa_widget = getattr(self, "validated_qpa_widget", None)
        validated_qpa_is_running = getattr(validated_qpa_widget, "is_running", None)
        if callable(validated_qpa_is_running) and validated_qpa_is_running():
            cancel_backend = getattr(validated_qpa_widget, "cancel_backend", None)
            if callable(cancel_backend):
                cancel_backend()
            thread = getattr(validated_qpa_widget, "_thread", None)
            if thread is not None:
                thread.quit()
                thread.wait(3000)
        if (
            hasattr(self, "phase_revolution_widget")
            and self.phase_revolution_widget.is_running()
        ):
            self.phase_revolution_widget.cancel_indexing()
            thread = self.phase_revolution_widget._thread
            if thread is not None:
                thread.quit()
                thread.wait(3000)
        if (
            hasattr(self, "structure_solution_widget")
            and self.structure_solution_widget.is_running()
        ):
            self.structure_solution_widget.cancel_robustness()
            thread = self.structure_solution_widget._thread
            if thread is not None:
                thread.quit()
                thread.wait(3000)
        if hasattr(self, "batch_widget") and self.batch_widget.is_running():
            self.batch_widget.cancel_batch()
            thread = self.batch_widget._thread
            if thread is not None:
                thread.quit()
                thread.wait(3000)
        event.accept()

    def new_project(self):
        if self.datasets:
            response = QMessageBox.question(
                self,
                "New project",
                "Clear the current project? Unsaved changes will be lost.",
            )
            if response != QMessageBox.Yes:
                return
        self._push_undo_checkpoint("New project")
        self.project_controller.new_project()
        self._qpa_setup_signature = None
        self.refresh_dataset_list()
        self.populate_peak_table()
        self.populate_fit_table()
        self.populate_fit_candidate_table()
        self.populate_size_strain_results()
        self.populate_crystal_results()
        self.populate_phase_identification_results()
        self.populate_qpa_results()
        self.populate_residual_stress_results()
        if hasattr(self, "calibration_widget"):
            self.calibration_widget.set_state({})
        if hasattr(self, "whole_pattern_widget"):
            self.whole_pattern_widget.set_state({})
        if hasattr(self, "rietveld_widget"):
            self.rietveld_widget.set_state({})
        if hasattr(self, "doping_series_widget"):
            self.doping_series_widget.set_state({})
        if hasattr(self, "validated_qpa_widget"):
            self.validated_qpa_widget.set_state({})
        if hasattr(self, "validation_widget"):
            self.validation_widget.set_state({})
        if hasattr(self, "phase_revolution_widget"):
            self.phase_revolution_widget.set_state({})
        if hasattr(self, "structure_solution_widget"):
            self.structure_solution_widget.set_state({})
        if hasattr(self, "batch_widget"):
            self.batch_widget.set_state({})
        self.setWindowTitle(APP_WINDOW_TITLE)
        self._select_workflow_task("project_home")
        self._update_workflow_dashboard()
        self.statusBar().showMessage("New project created.")

    def _refresh_recent_projects(self) -> None:
        store = getattr(self, "recent_projects_store", None)
        home = getattr(self, "project_home_widget", None)
        if store is not None and home is not None:
            home.set_recent_projects(store.entries(limit=5))

    def _record_recent_project(self, path) -> None:
        store = getattr(self, "recent_projects_store", None)
        if store is not None:
            store.record(path)
            self._refresh_recent_projects()

    def _open_recent_project(self, path: str) -> None:
        project = Path(path)
        if not project.is_file():
            self._remove_recent_project(path)
            self.statusBar().showMessage(
                f"Recent project is no longer available: {project}", 6000
            )
            return
        self._open_project_path(project)

    def _remove_recent_project(self, path: str) -> None:
        store = getattr(self, "recent_projects_store", None)
        if store is not None:
            store.remove(path)
            self._refresh_recent_projects()

    def _clear_recent_projects(self) -> None:
        store = getattr(self, "recent_projects_store", None)
        if store is not None:
            store.clear()
            self._refresh_recent_projects()
        self.statusBar().showMessage("Recent project history cleared.", 2600)

    @staticmethod
    def _local_drop_paths(mime_data) -> list[Path]:
        if mime_data is None or not mime_data.hasUrls():
            return []
        return [
            Path(url.toLocalFile())
            for url in mime_data.urls()
            if url.isLocalFile() and url.toLocalFile()
        ]

    def dragEnterEvent(self, event) -> None:
        paths = self._local_drop_paths(event.mimeData())
        if can_accept_drop_paths(paths):
            event.acceptProposedAction()
            overlay = getattr(self, "drop_import_overlay", None)
            if overlay is not None:
                overlay.show_for_count(len(paths))
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if can_accept_drop_paths(self._local_drop_paths(event.mimeData())):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        overlay = getattr(self, "drop_import_overlay", None)
        if overlay is not None:
            overlay.hide()
        event.accept()

    def dropEvent(self, event) -> None:
        overlay = getattr(self, "drop_import_overlay", None)
        if overlay is not None:
            overlay.hide()
        classified = classify_drop_paths(self._local_drop_paths(event.mimeData()))
        if not classified.accepted_count:
            event.ignore()
            return
        event.acceptProposedAction()

        if classified.project is not None:
            if not self._open_project_path(classified.project):
                return
        if classified.patterns:
            self._import_pattern_paths(classified.patterns)
        if classified.cifs:
            self._import_cif_reference_path(classified.cifs[0])
        if classified.rejected:
            names = ", ".join(path.name for path in classified.rejected[:3])
            suffix = "…" if len(classified.rejected) > 3 else ""
            self.statusBar().showMessage(
                f"Import complete. Skipped unsupported or extra files: {names}{suffix}",
                7000,
            )

    def import_patterns(self, checked=False):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import powder XRD patterns",
            "",
            (
                "Supported XRD patterns (*.xrdml *.rd *.xy *.csv *.txt *.dat *.pdfcard *.jcpds *.jade *.card *.ref);;"
                "Malvern Panalytical XRDML (*.xrdml);;Binary RD scans (*.rd *.RD);;"
                "Jade/PDF reference-card text (*.txt *.dat *.pdfcard *.jcpds *.jade *.card *.ref);;"
                "Text patterns (*.xy *.csv *.txt *.dat);;"
                "All files (*)"
            ),
        )
        if not paths:
            return

        return self._import_pattern_paths(paths)

    def _import_pattern_paths(self, paths):
        paths = [str(Path(path)) for path in paths]
        if not paths:
            return None
        self._push_undo_checkpoint("Import pattern")
        first_new = len(self.datasets)
        batch = self.import_service.load_paths(paths)
        if batch.datasets:
            self.project_controller.add_datasets(batch.datasets)
        imported_dataset_count = len(batch.datasets)

        self.refresh_dataset_list(
            first_new if len(self.datasets) > first_new else None
        )
        self.populate_phase_identification_results()
        self.statusBar().showMessage(
            f"Imported {imported_dataset_count} dataset(s) from "
            f"{batch.imported_file_count} file(s)."
        )
        if imported_dataset_count:
            self._select_workflow_task("raw_pattern")
        self._update_workflow_dashboard()
        if batch.failures:
            QMessageBox.warning(
                self,
                "Import warnings",
                "\n\n".join(row.display_text() for row in batch.failures),
            )
        return batch

    def _ui_state(self):
        inspector_toolbar = getattr(self, "analysis_inspector_toolbar", None)
        inspector_sections = getattr(self, "analysis_inspector_sections", {})
        ui_state = {
            "active_tab": self.tabs.currentIndex(),
            "active_workflow_task": self.active_workflow_task,
            "active_workspace_key": self.active_workspace_key,
            "workflow_mode": self.workflow_mode,
            "theme_name": self.current_theme_name,
            "splitter_sizes": self.main_splitter.sizes(),
            "view_range": self.plot_widget.get_view_range(),
            "peak_view_range": (
                self.peak_plot_widget.get_view_range()
                if hasattr(self, "peak_plot_widget") else None
            ),
            "peak_splitter_sizes": (
                self.peak_workspace_splitter.sizes()
                if hasattr(self, "peak_workspace_splitter") else None
            ),
            "analysis_inspector_mode": (
                inspector_toolbar.current_mode()
                if inspector_toolbar is not None else "essential"
            ),
            "analysis_inspector_expanded": {
                key: section.is_expanded()
                for key, section in inspector_sections.items()
            },
        }
        self.application_state.ui_preferences = deepcopy(ui_state)
        return ui_state

    def _analysis_state(self):
        return {
            "typed_result_contracts": self.project_state.result_contracts.to_dict(),
            "metrology_registry": self.project_state.metrology_registry.to_dict(),
            "background_check": self.background_check.isChecked(),
            "background_method": self.background_method.currentText(),
            "background_smoothness": self.background_smoothness.value(),
            "background_asymmetry": self.background_asymmetry.value(),
            "background_iterations": self.background_iterations.value(),
            "background_window_degrees": self.background_window_degrees.value(),
            "background_percentile": self.background_percentile.value(),
            "background_peak_protection": self.background_peak_protection.isChecked(),
            "background_clip_negative": self.background_clip_negative.isChecked(),
            "background_results": self.background_results,
            "smoothing_results": self.smoothing_results,
            "poly_order": self.poly_order.value(),
            "smooth_check": self.smooth_check.isChecked(),
            "smooth_window": self.smooth_window.value(),
            "smooth_order": self.smooth_order.value(),
            "smoothing_method": self.smoothing_method.currentText(),
            "smoothing_strength": self.smoothing_strength.value(),
            "smoothing_peak_protection": self.smoothing_peak_protection.isChecked(),
            "smoothing_peak_preservation": self.smoothing_peak_preservation.value(),
            "smoothing_max_position_shift": self.smoothing_max_position_shift.value(),
            "smoothing_max_height_change": self.smoothing_max_height_change.value(),
            "smoothing_max_fwhm_change": self.smoothing_max_fwhm_change.value(),
            "smoothing_source": self.smoothing_source.currentText(),
            "normalize_check": self.normalize_check.isChecked(),
            "prominence": self.prominence.value(),
            "min_distance": self.min_distance.value(),
            "smart_sensitivity": self.smart_sensitivity.currentText(),
            "manual_peak_snap_window": self.manual_peak_snap_window.value(),
            "preserve_manual_peaks": self.preserve_manual_peaks_check.isChecked(),
            "peak_rows": self.peak_rows,
            "peak_list_meta_by_uid": self.peak_list_meta_by_uid,
            "fit_groups": self.fit_groups,
            "fit_candidates": self.fit_candidates,
            "size_strain_results": self.size_strain_results,
            "reference_structure": self.reference_structure,
            "reference_pattern": self.reference_pattern,
            "cell_refinement_results": self.cell_refinement_results,
            "phase_identification_results": self.phase_identification_results,
            "qpa_results": self.qpa_results,
            "residual_stress_observations": self.residual_stress_observations,
            "residual_stress_result": self.residual_stress_result,
            "multicomponent_results": getattr(getattr(self, "multicomponent_refiner_widget", None), "results_by_uid", {}),
            "cif_library_last_match": getattr(getattr(self, "cif_library_widget", None), "last_match_result", None),
            "instrument_calibration": (
                self.calibration_widget.get_state()
                if hasattr(self, "calibration_widget")
                else {}
            ),
            "whole_pattern_refinement": (
                self.whole_pattern_widget.get_state()
                if hasattr(self, "whole_pattern_widget")
                else {}
            ),
            "rietveld_refinement": (
                self.rietveld_widget.get_state()
                if hasattr(self, "rietveld_widget")
                else {}
            ),
            "doping_series": (
                self.doping_series_widget.get_state()
                if hasattr(self, "doping_series_widget")
                else {}
            ),
            "validated_qpa": (
                self.validated_qpa_widget.get_state()
                if hasattr(self, "validated_qpa_widget")
                else {}
            ),
            "validation_campaign": (
                self.validation_widget.get_state()
                if hasattr(self, "validation_widget")
                else {}
            ),
            "phase_revolution": (
                self.phase_revolution_widget.get_state()
                if hasattr(self, "phase_revolution_widget")
                else {}
            ),
            "structure_solution_pathway": (
                self.structure_solution_widget.get_state()
                if hasattr(self, "structure_solution_widget")
                else {}
            ),
            "unified_scientific_state": self.scientific_state.to_dict(),
            "batch_workflow": (
                self.batch_widget.get_state()
                if hasattr(self, "batch_widget")
                else {}
            ),
            "stress_peak_source": self.stress_peak_source.currentText(),
            "stress_target_two_theta": self.stress_target_two_theta.value(),
            "stress_search_window": self.stress_search_window.value(),
            "stress_wavelength": self.stress_wavelength.value(),
            "stress_default_peak_error": self.stress_default_peak_error.value(),
            "stress_azimuth": self.stress_azimuth.value(),
            "stress_reference_mode": self.stress_reference_mode.currentText(),
            "stress_free_two_theta": self.stress_free_two_theta.value(),
            "stress_elastic_mode": self.stress_elastic_mode.currentText(),
            "stress_youngs_modulus": self.stress_youngs_modulus.value(),
            "stress_poisson_ratio": self.stress_poisson_ratio.value(),
            "stress_xec_half_s2": self.stress_xec_half_s2.value(),
            "stress_regression_mode": self.stress_regression_mode.currentText(),
            "qpa_mode": self.qpa_mode.currentText(),
            "qpa_use_processed": self.qpa_use_processed_check.isChecked(),
            "qpa_convert_d": self.qpa_convert_d_check.isChecked(),
            "qpa_reference_cutoff": self.qpa_reference_cutoff.value(),
            "qpa_reference_fwhm": self.qpa_reference_fwhm.value(),
            "qpa_profile_eta": self.qpa_profile_eta.value(),
            "qpa_baseline_order": self.qpa_baseline_order.value(),
            "qpa_weighting": self.qpa_weighting.currentText(),
            "qpa_bootstrap_samples": self.qpa_bootstrap_samples.value(),
            "qpa_internal_standard": self.qpa_internal_standard_check.isChecked(),
            "qpa_known_standard_percent": self.qpa_known_standard_percent.value(),
            "phase_peak_source": self.phase_peak_source.currentText(),
            "phase_tolerance": self.phase_tolerance.value(),
            "phase_max_shift": self.phase_max_shift.value(),
            "phase_reference_cutoff": self.phase_reference_cutoff.value(),
            "phase_maximum_phases": self.phase_maximum_phases.value(),
            "phase_mixture_pool": self.phase_mixture_pool.value(),
            "phase_convert_d": self.phase_convert_d_check.isChecked(),
            "reference_cutoff": self.reference_cutoff.value(),
            "match_tolerance": self.match_tolerance.value(),
            "crystal_system": self.crystal_system_selector.currentText(),
            "refine_zero_shift": self.refine_zero_shift_check.isChecked(),
            "show_reference": self.show_reference_check.isChecked(),
            "wavelength_angstrom": self.wavelength_angstrom.value(),
            "shape_factor": self.shape_factor.value(),
            "instrument_fwhm": self.instrument_fwhm.value(),
            "instrument_correction": self.instrument_correction.currentText(),
            "fit_model": self.fit_model.currentText(),
            "fit_window_multiplier": self.fit_window_multiplier.value(),
            "advanced_fit_model": self.advanced_fit_model.currentText(),
            "advanced_baseline_model": self.advanced_baseline_model.currentText(),
            "advanced_robust_loss": self.advanced_robust_loss.currentText(),
            "advanced_selection_criterion": self.advanced_selection_criterion.currentText(),
            "maximum_extra_components": self.maximum_extra_components.value(),
            "shared_width": self.shared_width_check.isChecked(),
            "show_fits": self.show_fits_check.isChecked(),
            "show_fit_components": self.show_fit_components_check.isChecked(),
            "show_fit_baselines": self.show_fit_baselines_check.isChecked(),
            "show_residuals": self.show_residuals_check.isChecked(),
            "stack_enabled": self.stack_check.isChecked(),
            "stack_offset": self.stack_offset.value(),
            "stack_scale": self.stack_scale.value(),
        }

    def save_project(self, checked=False, save_as=False):
        path = self.current_project
        if save_as or path is None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Afruz project", "", "Afruz PXRD Project (*.afz)"
            )
            if not filename:
                return
            path = Path(filename).with_suffix(".afz")

        try:
            self._sync_unified_scientific_state()
            path = self.project_controller.save(
                path,
                self._ui_state(),
                self._analysis_state(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.setWindowTitle(project_window_title(self.current_project.name))
        self._record_recent_project(self.current_project)
        self._refresh_shell_header()
        self.statusBar().showMessage(f"Saved {self.current_project}")

    def open_project(self, checked=False):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Afruz project", "", "Afruz PXRD Project (*.afz)"
        )
        if not filename:
            return
        return self._open_project_path(filename)

    def _open_project_path(self, filename) -> bool:
        filename = str(Path(filename))
        try:
            opened = self.project_controller.open(filename)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return False

        ui = opened.ui_state
        analysis = opened.analysis_state
        self.apply_theme(ui.get("theme_name", DEFAULT_THEME_NAME))
        self._qpa_setup_signature = None
        self._restore_analysis_state(analysis)
        self.refresh_dataset_list()
        self.populate_peak_table()
        self.populate_fit_table()
        self.populate_fit_candidate_table()
        self.populate_size_strain_results()
        self.populate_crystal_results()
        self.populate_phase_identification_results()
        self.populate_qpa_results()
        self.populate_residual_stress_results()
        self.workflow_mode = ui.get("workflow_mode", "Guided")
        self.workflow_header.set_mode(self.workflow_mode)
        self._restore_analysis_inspector_state(ui)
        saved_task = ui.get("active_workflow_task")
        if saved_task in self._workflow_task_widgets:
            self._select_workflow_task(saved_task)
        else:
            self.tabs.setCurrentIndex(int(ui.get("active_tab", 0)))
            self._sync_workflow_from_tab(self.tabs.currentIndex())
        self._sync_unified_scientific_state()
        self._update_workflow_dashboard()
        if ui.get("splitter_sizes"):
            self.main_splitter.setSizes(ui["splitter_sizes"])
        if ui.get("view_range"):
            self.plot_widget.set_view_range(ui["view_range"])
        if ui.get("peak_view_range") and hasattr(self, "peak_plot_widget"):
            self.peak_plot_widget.set_view_range(ui["peak_view_range"])
        if ui.get("peak_splitter_sizes") and hasattr(self, "peak_workspace_splitter"):
            self.peak_workspace_splitter.setSizes(ui["peak_splitter_sizes"])
        self.project_controller.clear_history()
        self._update_undo_redo_actions()
        self.setWindowTitle(project_window_title(self.current_project.name))
        self.statusBar().showMessage(f"Opened {self.current_project}")
        self._record_recent_project(self.current_project)
        return True

    def _restore_analysis_state(self, state):
        self.project_state.result_contracts = ResultContractService.migrate_analysis_state(
            self.datasets,
            state if isinstance(state, dict) else {},
        )
        self.scientific_state = ScientificStateRegistry.from_dict(
            state.get("unified_scientific_state", {}) if isinstance(state, dict) else {}
        )
        setters = [
            (self.background_check.setChecked, state.get("background_check", False)),
            (self.background_method.setCurrentText, state.get("background_method", "Auto ensemble")),
            (self.background_smoothness.setValue, state.get("background_smoothness", 70.0)),
            (self.background_asymmetry.setValue, state.get("background_asymmetry", 0.01)),
            (self.background_iterations.setValue, state.get("background_iterations", 50)),
            (self.background_window_degrees.setValue, state.get("background_window_degrees", 2.0)),
            (self.background_percentile.setValue, state.get("background_percentile", 20.0)),
            (self.background_peak_protection.setChecked, state.get("background_peak_protection", True)),
            (self.background_clip_negative.setChecked, state.get("background_clip_negative", False)),
            (self.poly_order.setValue, state.get("poly_order", 3)),
            (self.smooth_check.setChecked, state.get("smooth_check", False)),
            (self.smooth_window.setValue, state.get("smooth_window", 11)),
            (self.smooth_order.setValue, state.get("smooth_order", 3)),
            (self.smoothing_method.setCurrentText, state.get("smoothing_method", "Auto intelligent")),
            (self.smoothing_strength.setValue, state.get("smoothing_strength", 45.0)),
            (self.smoothing_peak_protection.setChecked, state.get("smoothing_peak_protection", True)),
            (self.smoothing_peak_preservation.setValue, state.get("smoothing_peak_preservation", 85.0)),
            (self.smoothing_max_position_shift.setValue, state.get("smoothing_max_position_shift", 0.02)),
            (self.smoothing_max_height_change.setValue, state.get("smoothing_max_height_change", 8.0)),
            (self.smoothing_max_fwhm_change.setValue, state.get("smoothing_max_fwhm_change", 10.0)),
            (self.smoothing_source.setCurrentText, state.get("smoothing_source", "Current processed pattern")),
            (self.normalize_check.setChecked, state.get("normalize_check", False)),
            (self.prominence.setValue, state.get("prominence", 0.03)),
            (self.min_distance.setValue, state.get("min_distance", 5)),
            (
                self.smart_sensitivity.setCurrentText,
                state.get("smart_sensitivity", "Balanced"),
            ),
            (
                self.manual_peak_snap_window.setValue,
                state.get("manual_peak_snap_window", 0.15),
            ),
            (
                self.preserve_manual_peaks_check.setChecked,
                state.get("preserve_manual_peaks", True),
            ),
            (
                self.phase_peak_source.setCurrentText,
                state.get("phase_peak_source", "Auto: fitted > detected > smart"),
            ),
            (
                self.phase_tolerance.setValue,
                state.get("phase_tolerance", 0.20),
            ),
            (
                self.phase_max_shift.setValue,
                state.get("phase_max_shift", 0.30),
            ),
            (
                self.phase_reference_cutoff.setValue,
                state.get("phase_reference_cutoff", 1.0),
            ),
            (
                self.phase_maximum_phases.setValue,
                state.get("phase_maximum_phases", 1),
            ),
            (
                self.phase_mixture_pool.setValue,
                state.get("phase_mixture_pool", 6),
            ),
            (
                self.phase_convert_d_check.setChecked,
                state.get("phase_convert_d", True),
            ),
            (
                self.stress_peak_source.setCurrentText,
                state.get("stress_peak_source", PEAK_SOURCE_MODES[0]),
            ),
            (
                self.stress_target_two_theta.setValue,
                state.get("stress_target_two_theta", 32.0),
            ),
            (
                self.stress_search_window.setValue,
                state.get("stress_search_window", 0.40),
            ),
            (
                self.stress_wavelength.setValue,
                state.get("stress_wavelength", 1.5406),
            ),
            (
                self.stress_default_peak_error.setValue,
                state.get("stress_default_peak_error", 0.01),
            ),
            (
                self.stress_azimuth.setValue,
                state.get("stress_azimuth", 0.0),
            ),
            (
                self.stress_reference_mode.setCurrentText,
                state.get("stress_reference_mode", REFERENCE_MODES[0]),
            ),
            (
                self.stress_free_two_theta.setValue,
                state.get("stress_free_two_theta", 32.0),
            ),
            (
                self.stress_elastic_mode.setCurrentText,
                state.get("stress_elastic_mode", ELASTIC_MODES[0]),
            ),
            (
                self.stress_youngs_modulus.setValue,
                state.get("stress_youngs_modulus", 200.0),
            ),
            (
                self.stress_poisson_ratio.setValue,
                state.get("stress_poisson_ratio", 0.30),
            ),
            (
                self.stress_xec_half_s2.setValue,
                state.get("stress_xec_half_s2", 0.0058),
            ),
            (
                self.stress_regression_mode.setCurrentText,
                state.get("stress_regression_mode", REGRESSION_MODES[0]),
            ),
            (
                self.qpa_mode.setCurrentText,
                state.get("qpa_mode", "RIR-corrected weight fractions"),
            ),
            (
                self.qpa_use_processed_check.setChecked,
                state.get("qpa_use_processed", True),
            ),
            (
                self.qpa_convert_d_check.setChecked,
                state.get("qpa_convert_d", True),
            ),
            (
                self.qpa_reference_cutoff.setValue,
                state.get("qpa_reference_cutoff", 1.0),
            ),
            (
                self.qpa_reference_fwhm.setValue,
                state.get("qpa_reference_fwhm", 0.20),
            ),
            (
                self.qpa_profile_eta.setValue,
                state.get("qpa_profile_eta", 0.5),
            ),
            (
                self.qpa_baseline_order.setValue,
                state.get("qpa_baseline_order", 1),
            ),
            (
                self.qpa_weighting.setCurrentText,
                state.get("qpa_weighting", "Balanced"),
            ),
            (
                self.qpa_bootstrap_samples.setValue,
                state.get("qpa_bootstrap_samples", 50),
            ),
            (
                self.qpa_internal_standard_check.setChecked,
                state.get("qpa_internal_standard", False),
            ),
            (
                self.qpa_known_standard_percent.setValue,
                state.get("qpa_known_standard_percent", 20.0),
            ),
            (
                self.reference_cutoff.setValue,
                state.get("reference_cutoff", 0.5),
            ),
            (
                self.match_tolerance.setValue,
                state.get("match_tolerance", 0.20),
            ),
            (
                self.crystal_system_selector.setCurrentText,
                state.get("crystal_system", "Triclinic"),
            ),
            (
                self.refine_zero_shift_check.setChecked,
                state.get("refine_zero_shift", True),
            ),
            (
                self.show_reference_check.setChecked,
                state.get("show_reference", True),
            ),
            (
                self.wavelength_angstrom.setValue,
                state.get("wavelength_angstrom", 1.5406),
            ),
            (
                self.shape_factor.setValue,
                state.get("shape_factor", 0.9),
            ),
            (
                self.instrument_fwhm.setValue,
                state.get("instrument_fwhm", 0.0),
            ),
            (
                self.instrument_correction.setCurrentText,
                state.get("instrument_correction", "None"),
            ),
            (
                self.fit_model.setCurrentText,
                state.get("fit_model", "Pseudo-Voigt"),
            ),
            (
                self.fit_window_multiplier.setValue,
                state.get("fit_window_multiplier", 5.0),
            ),
            (
                self.advanced_fit_model.setCurrentText,
                state.get("advanced_fit_model", "Auto"),
            ),
            (
                self.advanced_baseline_model.setCurrentText,
                state.get("advanced_baseline_model", "Quadratic"),
            ),
            (
                self.advanced_robust_loss.setCurrentText,
                state.get("advanced_robust_loss", "Soft L1"),
            ),
            (
                self.advanced_selection_criterion.setCurrentText,
                state.get("advanced_selection_criterion", "BIC"),
            ),
            (
                self.maximum_extra_components.setValue,
                state.get("maximum_extra_components", 2),
            ),
            (
                self.shared_width_check.setChecked,
                state.get("shared_width", False),
            ),
            (
                self.show_fits_check.setChecked,
                state.get("show_fits", True),
            ),
            (
                self.show_fit_components_check.setChecked,
                state.get("show_fit_components", True),
            ),
            (
                self.show_fit_baselines_check.setChecked,
                state.get("show_fit_baselines", False),
            ),
            (
                self.show_residuals_check.setChecked,
                state.get("show_residuals", False),
            ),
            (self.stack_check.setChecked, state.get("stack_enabled", False)),
            (self.stack_offset.setValue, state.get("stack_offset", 100.0)),
            (self.stack_scale.setValue, state.get("stack_scale", 1.0)),
        ]
        for setter, value in setters:
            setter(value)
        loaded_background_results = state.get("background_results", {})
        self.background_results = (
            loaded_background_results
            if isinstance(loaded_background_results, dict)
            else {}
        )
        self._rebuild_backgrounds_from_results()
        loaded_smoothing_results = state.get("smoothing_results", {})
        self.smoothing_results = (
            loaded_smoothing_results
            if isinstance(loaded_smoothing_results, dict)
            else {}
        )
        self._rebuild_smoothing_results()
        loaded_peak_rows = state.get("peak_rows", {})
        self.peak_rows = (
            {
                str(uid): normalize_peak_rows(rows)
                for uid, rows in loaded_peak_rows.items()
                if isinstance(rows, list)
            }
            if isinstance(loaded_peak_rows, dict)
            else {}
        )
        loaded_peak_meta = state.get("peak_list_meta_by_uid", {})
        self.peak_list_meta_by_uid = (
            loaded_peak_meta if isinstance(loaded_peak_meta, dict) else {}
        )
        loaded_fit_groups = state.get("fit_groups", {})
        self.fit_groups = (
            loaded_fit_groups if isinstance(loaded_fit_groups, dict) else {}
        )
        loaded_fit_candidates = state.get("fit_candidates", {})
        self.fit_candidates = (
            loaded_fit_candidates
            if isinstance(loaded_fit_candidates, dict)
            else {}
        )
        loaded_size_strain = state.get("size_strain_results", {})
        self.size_strain_results = (
            loaded_size_strain
            if isinstance(loaded_size_strain, dict)
            else {}
        )
        loaded_reference_structure = state.get("reference_structure")
        self.reference_structure = (
            loaded_reference_structure
            if isinstance(loaded_reference_structure, dict)
            else None
        )
        loaded_reference_pattern = state.get("reference_pattern", [])
        self.reference_pattern = (
            loaded_reference_pattern
            if isinstance(loaded_reference_pattern, list)
            else []
        )
        loaded_cell_results = state.get("cell_refinement_results", {})
        self.cell_refinement_results = (
            loaded_cell_results
            if isinstance(loaded_cell_results, dict)
            else {}
        )
        loaded_phase_results = state.get("phase_identification_results", {})
        self.phase_identification_results = (
            loaded_phase_results
            if isinstance(loaded_phase_results, dict)
            else {}
        )
        loaded_qpa_results = state.get("qpa_results", {})
        self.qpa_results = (
            loaded_qpa_results
            if isinstance(loaded_qpa_results, dict)
            else {}
        )
        loaded_stress_observations = state.get(
            "residual_stress_observations", []
        )
        self.residual_stress_observations = (
            loaded_stress_observations
            if isinstance(loaded_stress_observations, list)
            else []
        )
        loaded_stress_result = state.get("residual_stress_result")
        self.residual_stress_result = (
            loaded_stress_result
            if isinstance(loaded_stress_result, dict)
            else None
        )
        if hasattr(self, "calibration_widget"):
            self.calibration_widget.set_state(
                state.get("instrument_calibration", {})
            )
        if hasattr(self, "whole_pattern_widget"):
            self.whole_pattern_widget.set_state(
                state.get("whole_pattern_refinement", {})
            )
        if hasattr(self, "rietveld_widget"):
            self.rietveld_widget.set_state(
                state.get("rietveld_refinement", {})
            )
        if hasattr(self, "doping_series_widget"):
            self.doping_series_widget.set_state(
                state.get("doping_series", {})
            )
        if hasattr(self, "validated_qpa_widget"):
            self.validated_qpa_widget.set_state(
                state.get("validated_qpa", {})
            )
        if hasattr(self, "validation_widget"):
            self.validation_widget.set_state(
                state.get("validation_campaign", {})
            )
        if hasattr(self, "phase_revolution_widget"):
            self.phase_revolution_widget.set_state(
                state.get("phase_revolution", {})
            )
        if hasattr(self, "structure_solution_widget"):
            self.structure_solution_widget.set_state(
                state.get("structure_solution_pathway", {})
            )
        if hasattr(self, "batch_widget"):
            self.batch_widget.set_state(
                state.get("batch_workflow", {})
            )

    def rename_selected(self):
        ds = self.selected_dataset()
        if ds is None:
            return
        name, ok = QInputDialog.getText(self, "Rename dataset", "Dataset name:", text=ds.name)
        if ok and name.strip():
            self._push_undo_checkpoint("Rename dataset")
            self.project_controller.rename_dataset(ds.uid, name)
            self.refresh_dataset_list(self.dataset_list.currentRow())

    def duplicate_selected(self):
        ds = self.selected_dataset()
        if ds is None:
            return
        row = self.dataset_list.currentRow()
        self._push_undo_checkpoint("Duplicate dataset")
        duplicate = self.project_controller.duplicate_dataset(ds.uid)
        if ds.uid in self.peak_list_meta_by_uid:
            self.peak_list_meta_by_uid[duplicate.uid]["last_reason"] = (
                "Duplicated from another dataset"
            )
        for widget_name in ("whole_pattern_widget", "rietveld_widget"):
            widget = getattr(self, widget_name, None)
            guides = getattr(widget, "peak_guides_by_uid", None)
            if isinstance(guides, dict) and ds.uid in guides:
                copied_guide = deepcopy(guides[ds.uid])
                copied_guide["dataset_uid"] = duplicate.uid
                copied_guide["dataset_name"] = duplicate.name
                guides[duplicate.uid] = copied_guide
        if hasattr(self, "phase_revolution_widget"):
            self.phase_revolution_widget.duplicate_dataset(ds.uid, duplicate.uid)
        if hasattr(self, "structure_solution_widget"):
            self.structure_solution_widget.duplicate_dataset(ds.uid, duplicate.uid)
        self.refresh_dataset_list(row + 1)

    def remove_selected(self):
        row = self.dataset_list.currentRow()
        if not (0 <= row < len(self.datasets)):
            return
        ds = self.datasets[row]
        response = QMessageBox.question(
            self, "Delete dataset", f"Delete '{ds.name}' from this project?"
        )
        if response != QMessageBox.Yes:
            return
        self._push_undo_checkpoint("Delete dataset")
        self.project_controller.remove_dataset(ds.uid)
        if hasattr(self, "whole_pattern_widget"):
            self.whole_pattern_widget.results_by_uid.pop(ds.uid, None)
            self.whole_pattern_widget.peak_guides_by_uid.pop(ds.uid, None)
        if hasattr(self, "rietveld_widget"):
            self.rietveld_widget.results_by_uid.pop(ds.uid, None)
            self.rietveld_widget.peak_guides_by_uid.pop(ds.uid, None)
        if hasattr(self, "doping_series_widget"):
            self.doping_series_widget.remove_dataset(ds.uid)
        if hasattr(self, "validated_qpa_widget"):
            self.validated_qpa_widget.native_results_by_uid.pop(ds.uid, None)
        if hasattr(self, "validation_widget"):
            self.validation_widget.remove_dataset(ds.uid)
        if hasattr(self, "phase_revolution_widget"):
            self.phase_revolution_widget.remove_dataset(ds.uid)
        if hasattr(self, "structure_solution_widget"):
            self.structure_solution_widget.remove_dataset(ds.uid)
        self.refresh_dataset_list(min(row, len(self.datasets) - 1))
        self.populate_peak_table()
        self.populate_fit_table()
        self.populate_fit_candidate_table()
        self.populate_size_strain_results()
        self.populate_crystal_results()
        self.populate_phase_identification_results()
        self.populate_qpa_results()
        self.populate_residual_stress_results()
