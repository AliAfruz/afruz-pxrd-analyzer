from __future__ import annotations

from ..main_window_dependencies import *
from .base import WorkspaceAdapter


class PeaksWorkspace(WorkspaceAdapter):
    """Independently testable peak-analysis workspace adapter."""


class PeaksWorkspaceMixin:
    def _main_peak_meta(self, dataset_uid: str) -> dict:
        return self.peak_list_meta_by_uid.setdefault(
            dataset_uid,
            {
                "locked": False,
                "revision": 0,
                "last_reason": "",
            },
        )

    def _main_peak_list_is_locked(self, dataset_uid: str) -> bool:
        return bool(self._main_peak_meta(dataset_uid).get("locked"))

    def _active_peak_rows_for_uid(self, dataset_uid: str) -> list[dict]:
        return active_peak_rows(self.peak_rows.get(dataset_uid, []))

    def _invalidate_main_peak_dependents(self, dataset_uid: str):
        for mapping in (
            self.fit_groups,
            self.fit_candidates,
            self.size_strain_results,
            self.cell_refinement_results,
            self.phase_identification_results,
            self.qpa_results,
        ):
            mapping.pop(dataset_uid, None)
        self.residual_stress_result = None

    def _update_main_peak_status(self):
        dataset = self.selected_dataset()
        if dataset is None:
            self.main_peak_status.setText(
                "Select a dataset to curate its peak list."
            )
            return
        rows = normalize_peak_rows(self.peak_rows.get(dataset.uid, []))
        meta = self._main_peak_meta(dataset.uid)
        manual_count = sum(bool(row.get("manual")) for row in rows)
        included_count = sum(bool(row.get("use", True)) for row in rows)
        state = "FROZEN" if meta.get("locked") else "editable"
        self.main_peak_status.setText(
            f"{dataset.name}: revision {meta.get('revision', 0)} · {state} · "
            f"{included_count}/{len(rows)} included · {manual_count} manual/protected. "
            "Included peaks are used by all following peak-based analyses."
        )

    def _commit_main_peak_rows(
        self,
        dataset_uid: str,
        rows,
        *,
        reason: str,
        invalidate: bool = True,
        bump_revision: bool = True,
    ) -> list[dict]:
        if bump_revision:
            self._push_undo_checkpoint(reason)
        normalized = normalize_peak_rows(rows)
        self.peak_rows[dataset_uid] = normalized
        meta = self._main_peak_meta(dataset_uid)
        if bump_revision:
            meta["revision"] = int(meta.get("revision", 0)) + 1
        meta["last_reason"] = reason
        if invalidate:
            self._invalidate_main_peak_dependents(dataset_uid)
        self.populate_peak_table()
        self.populate_fit_table()
        self.populate_fit_candidate_table()
        self.populate_size_strain_results()
        self.populate_crystal_results()
        self.redraw()
        self._update_main_peak_status()
        self._record_scientific_result(
            "peaks",
            dataset_uid,
            {"rows": normalized, "meta": dict(meta)},
            reason=reason,
        )
        return normalized

    def _main_peak_list_blocks_edit(self, dataset) -> bool:
        if dataset is None:
            return True
        if self._phase_revolution_master_blocks_peak_replacement(dataset):
            return True
        if not self._main_peak_list_is_locked(dataset.uid):
            return False
        self.statusBar().showMessage(
            "The selected dataset peak list is frozen. Unfreeze it before editing or replacing peaks."
        )
        return True

    def add_manual_peak_dialog(self):
        dataset = self.selected_dataset()
        if dataset is None:
            QMessageBox.information(self, "No dataset", "Select a dataset first.")
            return
        value, accepted = QInputDialog.getDouble(
            self,
            "Add manual diffraction peak",
            "Approximate 2θ position (degrees):",
            float(self.manual_peak_position.value()),
            float(np.min(dataset.x)),
            float(np.max(dataset.x)),
            6,
        )
        if accepted:
            self.manual_peak_position.setValue(value)
            self.add_manual_peak_for_selected(
                value,
                origin="Manual dialog entry",
            )

    def add_manual_peak_for_selected(
        self,
        position_deg: float,
        *,
        origin: str = "Manual",
    ):
        dataset = self.selected_dataset()
        if dataset is None:
            QMessageBox.information(self, "No dataset", "Select a dataset first.")
            return
        if self._main_peak_list_blocks_edit(dataset):
            return
        existing = normalize_peak_rows(self.peak_rows.get(dataset.uid, []))
        step = float(np.median(np.diff(dataset.x)))
        duplicate_tolerance = max(2.0 * step, 0.02)
        nearest = min(
            existing,
            key=lambda row: abs(float(row["position"]) - float(position_deg)),
            default=None,
        )
        replace_uuid = None
        notes = ""
        if nearest is not None and abs(float(nearest["position"]) - float(position_deg)) <= duplicate_tolerance:
            replace_uuid = nearest.get("peak_uuid")
            notes = str(nearest.get("notes", ""))
            existing = [
                row for row in existing
                if row.get("peak_uuid") != replace_uuid
            ]
        try:
            row = create_manual_peak(
                dataset.x,
                dataset.y,
                float(position_deg),
                snap_window_deg=self.manual_peak_snap_window.value(),
                peak_uuid=replace_uuid,
                notes=notes,
                origin=origin,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Manual peak could not be added", str(exc))
            return
        existing.append(row)
        self._commit_main_peak_rows(
            dataset.uid,
            existing,
            reason=f"Added manual peak at {row['position']:.6g}° 2θ",
        )
        self.statusBar().showMessage(
            f"Added protected manual peak at {row['position']:.6g}° 2θ in {dataset.name}."
        )

    def _redraw_peak_workspace(self) -> None:
        if not hasattr(self, "peak_plot_widget"):
            return
        dataset = self.selected_dataset()
        if dataset is None:
            self.peak_plot_widget.clear()
            self.peak_plot_widget.coordinate_label.setText(
                "2θ: —    Intensity: —    Import or select a dataset."
            )
            return
        self.peak_plot_widget.redraw(
            [dataset],
            stack_enabled=False,
            stack_offset=0.0,
            stack_scale=1.0,
            backgrounds=(
                {dataset.uid: self.backgrounds[dataset.uid]}
                if dataset.uid in self.backgrounds else {}
            ),
            peak_rows={dataset.uid: self.peak_rows.get(dataset.uid, [])},
            fit_groups={dataset.uid: self.fit_groups.get(dataset.uid, [])},
            show_fits=self.show_fits_check.isChecked(),
            show_fit_components=self.show_fit_components_check.isChecked(),
            show_fit_baselines=self.show_fit_baselines_check.isChecked(),
            show_residuals=self.show_residuals_check.isChecked(),
            reference_pattern=[],
            show_reference=False,
            include_hidden=True,
        )
        self._highlight_peak_workspace_selection(zoom=False)

    def auto_range_peak_workspace(self) -> None:
        if not hasattr(self, "peak_plot_widget"):
            return
        self.peak_plot_widget.clear_peak_highlight()
        self.peak_plot_widget.plot.enableAutoRange()
        self.statusBar().showMessage(
            "Peak workspace restored to the complete selected-dataset range."
        )

    def _peak_workspace_table_selection_changed(self, *args) -> None:
        if getattr(self, "_peak_table_updating", False):
            return
        self.zoom_peak_workspace_selection()

    def _select_dataset_uid_for_peak_workspace(self, dataset_uid: str) -> None:
        selected = self.selected_dataset()
        if selected is not None and selected.uid == dataset_uid:
            return
        for index, dataset in enumerate(self.datasets):
            if dataset.uid == dataset_uid:
                self.dataset_list.setCurrentRow(index)
                return

    def _highlight_peak_workspace_selection(self, *, zoom: bool) -> bool:
        if not hasattr(self, "peak_plot_widget"):
            return False
        tab_index = self.peak_results_tabs.currentIndex()

        if tab_index == 0:
            selected_rows = self.peak_table.selectionModel().selectedRows()
            if not selected_rows:
                return False
            row_index = selected_rows[0].row()
            anchor = self.peak_table.item(row_index, 0)
            metadata = None if anchor is None else anchor.data(Qt.UserRole)
            if not isinstance(metadata, dict):
                return False
            dataset_uid = str(metadata.get("dataset_uid", ""))
            peak_uuid = str(metadata.get("peak_uuid", ""))
            self._select_dataset_uid_for_peak_workspace(dataset_uid)
            peak = next(
                (
                    row
                    for row in normalize_peak_rows(self.peak_rows.get(dataset_uid, []))
                    if str(row.get("peak_uuid", "")) == peak_uuid
                ),
                None,
            )
            if peak is None:
                return False
            position = float(peak["position"])
            fwhm = max(float(peak.get("fwhm", 0.0)), 0.02)
            half_window = max(4.0 * fwhm, 0.25)
            self.peak_plot_widget.highlight_peak(
                position,
                float(peak.get("intensity", 0.0)),
                window_min=position - half_window,
                window_max=position + half_window,
                zoom=zoom,
            )
            return True

        if tab_index == 1:
            selected_rows = self.fit_table.selectionModel().selectedRows()
            if not selected_rows:
                return False
            row_index = selected_rows[0].row()
            anchor = self.fit_table.item(row_index, 0)
            metadata = None if anchor is None else anchor.data(Qt.UserRole)
            if not isinstance(metadata, dict):
                return False
            dataset_uid = str(metadata.get("dataset_uid", ""))
            self._select_dataset_uid_for_peak_workspace(dataset_uid)
            center = float(metadata.get("center", 0.0))
            window_min = float(metadata.get("window_min", center - 0.75))
            window_max = float(metadata.get("window_max", center + 0.75))
            self.peak_plot_widget.highlight_peak(
                center,
                None,
                window_min=window_min,
                window_max=window_max,
                zoom=zoom,
            )
            return True

        if tab_index == 2:
            selected_rows = self.fit_candidate_table.selectionModel().selectedRows()
            if not selected_rows:
                return False
            row_index = selected_rows[0].row()
            anchor = self.fit_candidate_table.item(row_index, 0)
            metadata = None if anchor is None else anchor.data(Qt.UserRole)
            if not isinstance(metadata, dict):
                return False
            dataset_uid = str(metadata.get("dataset_uid", ""))
            self._select_dataset_uid_for_peak_workspace(dataset_uid)
            center = 0.5 * (
                float(metadata.get("window_min", 0.0))
                + float(metadata.get("window_max", 0.0))
            )
            self.peak_plot_widget.highlight_peak(
                center,
                None,
                window_min=float(metadata.get("window_min", center - 0.75)),
                window_max=float(metadata.get("window_max", center + 0.75)),
                zoom=zoom,
            )
            return True
        return False

    def zoom_peak_workspace_selection(self) -> None:
        if not self._highlight_peak_workspace_selection(zoom=True):
            self.statusBar().showMessage(
                "Select a detected peak, fitted component, or model-selection row first."
            )

    def _plot_manual_peak_requested(self, position_deg: float, _intensity: float):
        if not self.click_add_main_peak_button.isChecked():
            return
        self.manual_peak_position.setValue(float(position_deg))
        self.add_manual_peak_for_selected(
            float(position_deg),
            origin="Manual plot click",
        )

    def _context_manual_peak_requested(
        self,
        position_deg: float,
        _intensity: float = 0.0,
        *,
        origin: str = "Plot right-click",
    ):
        """Add a protected manual peak from a plot context-menu position."""
        dataset = self.selected_dataset()
        if dataset is None:
            self.statusBar().showMessage(
                "Select a dataset before adding a manual peak."
            )
            return
        self.manual_peak_position.setValue(float(position_deg))
        self.add_manual_peak_for_selected(float(position_deg), origin=origin)

    def _selected_main_peak_references(self) -> list[tuple[str, str]]:
        references: list[tuple[str, str]] = []
        for model_index in self.peak_table.selectionModel().selectedRows():
            anchor = self.peak_table.item(model_index.row(), 0)
            metadata = None if anchor is None else anchor.data(Qt.UserRole)
            if isinstance(metadata, dict):
                references.append(
                    (str(metadata.get("dataset_uid")), str(metadata.get("peak_uuid")))
                )
        return references

    def delete_selected_main_peaks(self):
        references = self._selected_main_peak_references()
        if not references:
            self.statusBar().showMessage("Select one or more peak rows to delete.")
            return
        grouped: dict[str, set[str]] = {}
        for dataset_uid, peak_uuid in references:
            grouped.setdefault(dataset_uid, set()).add(peak_uuid)
        changed = 0
        for dataset_uid, peak_uuids in grouped.items():
            if self._main_peak_list_is_locked(dataset_uid):
                continue
            rows = [
                row for row in normalize_peak_rows(self.peak_rows.get(dataset_uid, []))
                if row.get("peak_uuid") not in peak_uuids
            ]
            self._commit_main_peak_rows(
                dataset_uid,
                rows,
                reason=f"Deleted {len(peak_uuids)} selected peak(s)",
            )
            changed += len(peak_uuids)
        self.statusBar().showMessage(
            f"Deleted {changed} peak(s). Frozen dataset lists were unchanged."
        )

    def set_selected_main_peaks_use(self, included: bool):
        references = self._selected_main_peak_references()
        if not references:
            self.statusBar().showMessage("Select one or more peak rows first.")
            return
        grouped: dict[str, set[str]] = {}
        for dataset_uid, peak_uuid in references:
            grouped.setdefault(dataset_uid, set()).add(peak_uuid)
        changed = 0
        for dataset_uid, peak_uuids in grouped.items():
            if self._main_peak_list_is_locked(dataset_uid):
                continue
            rows = normalize_peak_rows(self.peak_rows.get(dataset_uid, []))
            for row in rows:
                if row.get("peak_uuid") in peak_uuids:
                    row["use"] = bool(included)
                    changed += 1
            self._commit_main_peak_rows(
                dataset_uid,
                rows,
                reason=("Included" if included else "Excluded")
                + f" {len(peak_uuids)} selected peak(s)",
            )
        self.statusBar().showMessage(
            f"{'Included' if included else 'Excluded'} {changed} peak(s)."
        )

    def send_main_peak_list_to_phase_revolution(self):
        dataset = self.selected_dataset()
        widget = getattr(self, "phase_revolution_widget", None)
        if dataset is None or widget is None:
            self.statusBar().showMessage(
                "Select a dataset before sending its curated peaks to Unknown Phase."
            )
            return
        rows = normalize_peak_rows(self.peak_rows.get(dataset.uid, []))
        if not rows:
            self.statusBar().showMessage(
                "The selected dataset has no curated peaks to send."
            )
            return
        master_rows = []
        for row in rows:
            master_rows.append(
                {
                    "peak_uuid": row.get("peak_uuid"),
                    "use": bool(row.get("use", True)),
                    "two_theta_deg": float(row["position"]),
                    "position_uncertainty_deg": (
                        row.get("position_error")
                        if row.get("position_error") is not None
                        else max(
                            float(np.median(np.diff(dataset.x))) / 2.0,
                            1e-6,
                        )
                    ),
                    "intensity": float(row.get("intensity", 0.0)),
                    "prominence": float(row.get("prominence", 0.0)),
                    "signal_to_noise": float(row.get("snr") or 0.0),
                    "quality_score": float(row.get("confidence") or 0.0),
                    "fwhm_deg": float(row.get("fwhm", 0.0)),
                    "origin": str(row.get("origin") or row.get("method", "Main peak list")),
                    "manual": bool(row.get("manual")),
                    "row_locked": bool(row.get("protected")),
                    "note": str(row.get("notes", "")),
                }
            )
        widget._commit_master_peaks(
            dataset.uid,
            master_rows,
            reason="Imported from the main curated peak list",
        )
        widget.refresh_for_selected_dataset()
        self.statusBar().showMessage(
            f"Sent {len(master_rows)} curated peak(s) from {dataset.name} to Unknown Phase."
        )

    def _main_peak_freeze_toggled(self, checked: bool):
        if self._peak_table_updating:
            return
        dataset = self.selected_dataset()
        if dataset is None:
            return
        self._push_undo_checkpoint("Freeze peak list" if checked else "Unfreeze peak list")
        self._main_peak_meta(dataset.uid)["locked"] = bool(checked)
        self.populate_peak_table()
        self._update_main_peak_status()
        self.statusBar().showMessage(
            f"{dataset.name} peak list is now {'frozen' if checked else 'editable'}."
        )

    def _main_peak_table_item_changed(self, item: QTableWidgetItem):
        if self._peak_table_updating:
            return
        anchor = self.peak_table.item(item.row(), 0)
        metadata = None if anchor is None else anchor.data(Qt.UserRole)
        if not isinstance(metadata, dict):
            return
        dataset_uid = str(metadata.get("dataset_uid"))
        peak_uuid = str(metadata.get("peak_uuid"))
        dataset = next((ds for ds in self.datasets if ds.uid == dataset_uid), None)
        if dataset is None:
            return
        if self._main_peak_list_is_locked(dataset_uid):
            self.populate_peak_table()
            self.statusBar().showMessage(
                "That dataset peak list is frozen; the edit was not applied."
            )
            return
        rows = normalize_peak_rows(self.peak_rows.get(dataset_uid, []))
        index = next(
            (i for i, row in enumerate(rows) if row.get("peak_uuid") == peak_uuid),
            None,
        )
        if index is None:
            return
        row = rows[index]
        reason = "Edited peak list"
        invalidate = True
        try:
            if item.column() == 0:
                row["use"] = item.checkState() == Qt.Checked
                reason = "Changed peak include/exclude state"
            elif item.column() == 3:
                replacement = create_manual_peak(
                    dataset.x,
                    dataset.y,
                    float(item.text()),
                    snap_window_deg=self.manual_peak_snap_window.value(),
                    peak_uuid=peak_uuid,
                    notes=str(row.get("notes", "")),
                    origin="Manual table edit",
                )
                replacement["use"] = bool(row.get("use", True))
                rows[index] = replacement
                reason = "Edited peak position"
            elif item.column() == 10:
                row["notes"] = item.text().strip()
                reason = "Edited peak notes"
                invalidate = False
            else:
                return
        except Exception as exc:
            QMessageBox.warning(self, "Peak edit rejected", str(exc))
            self.populate_peak_table()
            return
        self._commit_main_peak_rows(
            dataset_uid,
            rows,
            reason=reason,
            invalidate=invalidate,
        )

    def _phase_revolution_master_blocks_peak_replacement(self, dataset):
        widget = getattr(self, "phase_revolution_widget", None)
        if widget is None or dataset is None:
            return False
        if not widget.is_master_shared_and_locked(dataset.uid):
            return False
        widget._sync_master_to_application(dataset.uid)
        self.statusBar().showMessage(
            "The frozen Unknown Phase master reflection list is active. "
            "Unlock or stop sharing it before replacing peaks with another detector."
        )
        return True

    def find_peaks_for_selected(self):
        ds = self.selected_dataset()
        if ds is None:
            QMessageBox.information(self, "No dataset", "Select a dataset first.")
            return
        if self._main_peak_list_blocks_edit(ds):
            return
        detected = detect_peaks(
            ds.x,
            ds.y,
            prominence_fraction=self.prominence.value(),
            minimum_distance_points=self.min_distance.value(),
        )
        rows = (
            merge_detected_with_manual(
                self.peak_rows.get(ds.uid, []),
                detected,
            )
            if self.preserve_manual_peaks_check.isChecked()
            else normalize_peak_rows(detected)
        )
        self._commit_main_peak_rows(
            ds.uid,
            rows,
            reason="Regular peak search",
        )
        manual_count = sum(bool(row.get("manual")) for row in rows)
        self.statusBar().showMessage(
            f"Detected {len(detected)} automatic peak(s) in {ds.name}; "
            f"{manual_count} manual/protected peak(s) were retained."
        )

    def smart_find_peaks_for_selected(self):
        ds = self.selected_dataset()
        if ds is None:
            QMessageBox.information(self, "No dataset", "Select a dataset first.")
            return
        if self._main_peak_list_blocks_edit(ds):
            return
        try:
            detected, diagnostics = smart_detect_peaks(
                ds.x,
                ds.y,
                sensitivity=self.smart_sensitivity.currentText(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Smart peak search failed", str(exc))
            return

        rows = (
            merge_detected_with_manual(
                self.peak_rows.get(ds.uid, []),
                detected,
            )
            if self.preserve_manual_peaks_check.isChecked()
            else normalize_peak_rows(detected)
        )
        self._commit_main_peak_rows(
            ds.uid,
            rows,
            reason="Smart peak search",
        )
        manual_count = sum(bool(row.get("manual")) for row in rows)
        self.statusBar().showMessage(
            f"Smart search detected {len(detected)} automatic peak(s) in {ds.name}; "
            f"estimated noise σ={diagnostics['noise_sigma']:.4g}; "
            f"{manual_count} manual/protected peak(s) were retained."
        )

    def fit_detected_peaks_for_selected(self):
        ds = self.selected_dataset()
        if ds is None:
            self.statusBar().showMessage("Select a dataset before fitting peaks.")
            return

        stored_peaks = normalize_peak_rows(self.peak_rows.get(ds.uid, []))
        peaks = active_peak_rows(stored_peaks)
        if not peaks and stored_peaks:
            self.statusBar().showMessage(
                "The curated peak list contains no included peaks. Include at least one peak before fitting."
            )
            return
        if not peaks:
            rows, diagnostics = smart_detect_peaks(
                ds.x,
                ds.y,
                sensitivity=self.smart_sensitivity.currentText(),
            )
            self._commit_main_peak_rows(
                ds.uid,
                rows,
                reason="Automatic peak seeds for fitting",
            )
            peaks = self._active_peak_rows_for_uid(ds.uid)
            if not peaks:
                self.statusBar().showMessage(
                    "No peaks were detected automatically; fitting was not started."
                )
                return

        try:
            groups, diagnostics = fit_detected_peaks(
                ds.x,
                ds.y,
                peaks,
                model=self.fit_model.currentText(),
                window_multiplier=self.fit_window_multiplier.value(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Peak fitting failed", str(exc))
            return

        self.fit_groups[ds.uid] = groups
        self.fit_candidates[ds.uid] = diagnostics.get("candidates", [])
        self.size_strain_results.pop(ds.uid, None)
        self.cell_refinement_results.pop(ds.uid, None)
        self.phase_identification_results.pop(ds.uid, None)
        self.qpa_results.pop(ds.uid, None)
        self.populate_fit_table()
        self.populate_fit_candidate_table()
        self.populate_size_strain_results()
        self.redraw()
        self.statusBar().showMessage(
            f"Fitted {diagnostics['components']} peak(s) in "
            f"{diagnostics['groups']} group(s) using {self.fit_model.currentText()}. "
            f"Failed groups: {diagnostics['failed_groups']}."
        )

    def _deconvolution_is_running(self) -> bool:
        return bool(
            self._deconvolution_thread is not None
            and self._deconvolution_thread.isRunning()
        )

    def _set_deconvolution_busy(self, busy: bool):
        protected_widgets = (
            self.advanced_fit_button,
            self.fit_peaks_button,
            self.peak_plot_advanced_fit_button,
            self.peak_plot_fit_button,
            self.peak_plot_find_button,
            self.peak_plot_smart_button,
            self.clear_fits_button,
            self.apply_button,
            self.find_peaks_button,
            self.smart_peaks_button,
            self.dataset_list,
            self.rename_button,
            self.duplicate_button,
            self.remove_button,
            self.reset_button,
            self.import_button,
            self.new_button,
            self.open_button,
        )
        for widget in protected_widgets:
            widget.setEnabled(not busy)

        protected_actions = (
            self.new_action,
            self.open_action,
            self.import_action,
        )
        for action in protected_actions:
            action.setEnabled(not busy)

        self.cancel_deconvolution_button.setEnabled(busy)

    def advanced_fit_peaks_for_selected(self):
        if self._deconvolution_is_running():
            self.statusBar().showMessage(
                "Advanced deconvolution is already running."
            )
            return

        ds = self.selected_dataset()
        if ds is None:
            self.statusBar().showMessage(
                "Select a dataset before advanced deconvolution."
            )
            return

        stored_peaks = normalize_peak_rows(self.peak_rows.get(ds.uid, []))
        peaks = active_peak_rows(stored_peaks)
        if not peaks and stored_peaks:
            self.statusBar().showMessage(
                "The curated peak list contains no included peaks for advanced deconvolution."
            )
            return
        if not peaks:
            try:
                rows, diagnostics = smart_detect_peaks(
                    ds.x,
                    ds.y,
                    sensitivity=self.smart_sensitivity.currentText(),
                )
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Smart peak search failed",
                    str(exc),
                )
                return
            self._commit_main_peak_rows(
                ds.uid,
                rows,
                reason="Automatic peak seeds for advanced deconvolution",
            )
            peaks = self._active_peak_rows_for_uid(ds.uid)

        if not peaks:
            self.statusBar().showMessage(
                "No seed peaks were found for advanced deconvolution."
            )
            return

        options = {
            "model": self.advanced_fit_model.currentText(),
            "window_multiplier": self.fit_window_multiplier.value(),
            "baseline_model": self.advanced_baseline_model.currentText(),
            "robust_loss": self.advanced_robust_loss.currentText(),
            "maximum_extra_components": (
                self.maximum_extra_components.value()
            ),
            "selection_criterion": (
                self.advanced_selection_criterion.currentText()
            ),
            "shared_width": self.shared_width_check.isChecked(),
        }

        thread = QThread(self)
        worker = DeconvolutionWorker(
            np.asarray(ds.x, dtype=float).copy(),
            np.asarray(ds.y, dtype=float).copy(),
            [dict(peak) for peak in peaks],
            options,
        )
        worker.moveToThread(thread)

        self._deconvolution_thread = thread
        self._deconvolution_worker = worker
        self._deconvolution_dataset_uid = ds.uid
        self._deconvolution_dataset_name = ds.name

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_deconvolution_progress)
        worker.finished.connect(self._on_deconvolution_completed)
        worker.failed.connect(self._on_deconvolution_failed)
        worker.cancelled.connect(self._on_deconvolution_cancelled)

        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(
            self._on_deconvolution_thread_finished
        )
        thread.finished.connect(thread.deleteLater)

        self.deconvolution_progress.setRange(0, 1)
        self.deconvolution_progress.setValue(0)
        self.deconvolution_progress.setFormat(
            "%v / %m steps completed — %p%"
        )
        self.deconvolution_progress_label.setText(
            f"Preparing advanced deconvolution for {ds.name}…"
        )
        self._set_deconvolution_busy(True)
        self.statusBar().showMessage(
            f"Advanced deconvolution started for {ds.name}."
        )
        thread.start()

    def _on_deconvolution_progress(
        self,
        completed: int,
        total: int,
        message: str,
    ):
        total = max(1, int(total))
        completed = max(0, min(int(completed), total))
        self.deconvolution_progress.setRange(0, total)
        self.deconvolution_progress.setValue(completed)
        self.deconvolution_progress.setFormat(
            f"{completed} / {total} steps completed — %p%"
        )
        self.deconvolution_progress_label.setText(message)
        self.statusBar().showMessage(message)

    def cancel_advanced_deconvolution(self):
        worker = self._deconvolution_worker
        if worker is None or not self._deconvolution_is_running():
            return

        worker.request_cancel()
        self.cancel_deconvolution_button.setEnabled(False)
        self.deconvolution_progress_label.setText(
            "Cancellation requested. Stopping the active optimization safely…"
        )
        self.statusBar().showMessage(
            "Cancellation requested for advanced deconvolution."
        )

    def _on_deconvolution_completed(
        self,
        groups,
        diagnostics,
    ):
        dataset_uid = self._deconvolution_dataset_uid
        dataset = next(
            (
                item
                for item in self.datasets
                if item.uid == dataset_uid
            ),
            None,
        )
        if dataset is None:
            self.deconvolution_progress_label.setText(
                "Deconvolution completed, but its source dataset is no longer open."
            )
            return

        self.fit_groups[dataset.uid] = list(groups)
        self.fit_candidates[dataset.uid] = diagnostics.get(
            "candidates", []
        )
        self.size_strain_results.pop(dataset.uid, None)
        self.cell_refinement_results.pop(dataset.uid, None)

        self.populate_fit_table()
        self.populate_fit_candidate_table()
        self.populate_size_strain_results()
        self.populate_crystal_results()
        self.redraw()

        final_total = max(
            self.deconvolution_progress.maximum(),
            int(diagnostics.get("progress_steps_planned", 1)),
            1,
        )
        self.deconvolution_progress.setRange(0, final_total)
        self.deconvolution_progress.setValue(final_total)
        self.deconvolution_progress.setFormat(
            f"{final_total} / {final_total} steps completed — 100%"
        )

        chosen_models = sorted(
            {
                group.get("model", "Unknown")
                for group in groups
            }
        )
        model_text = (
            ", ".join(chosen_models)
            if chosen_models
            else "none"
        )
        message = (
            f"Completed {diagnostics['components']} component(s) in "
            f"{diagnostics['groups']} group(s); models: {model_text}; "
            f"criterion: {diagnostics.get('selection_criterion', 'BIC')}; "
            f"failed groups: {diagnostics['failed_groups']}."
        )
        self.deconvolution_progress_label.setText(message)
        self.statusBar().showMessage(message)

    def _on_deconvolution_failed(self, traceback_text: str):
        final_line = (
            traceback_text.strip().splitlines()[-1]
            if traceback_text.strip()
            else "Unknown deconvolution error."
        )
        self.deconvolution_progress_label.setText(
            f"Advanced deconvolution failed: {final_line}"
        )
        self.statusBar().showMessage(
            "Advanced deconvolution failed."
        )

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Advanced deconvolution failed")
        box.setText(final_line)
        box.setDetailedText(traceback_text)
        box.exec()

    def _on_deconvolution_cancelled(self):
        completed = self.deconvolution_progress.value()
        total = self.deconvolution_progress.maximum()
        self.deconvolution_progress.setFormat(
            f"Cancelled after {completed} / {total} steps"
        )
        self.deconvolution_progress_label.setText(
            "Advanced deconvolution was cancelled. Existing fit results were unchanged."
        )
        self.statusBar().showMessage(
            "Advanced deconvolution cancelled."
        )

    def _on_deconvolution_thread_finished(self):
        self._set_deconvolution_busy(False)
        self._deconvolution_thread = None
        self._deconvolution_worker = None
        self._deconvolution_dataset_uid = None
        self._deconvolution_dataset_name = ""

    def clear_fits_for_selected(self):
        ds = self.selected_dataset()
        if ds is None:
            return
        self.fit_groups.pop(ds.uid, None)
        self.fit_candidates.pop(ds.uid, None)
        self.size_strain_results.pop(ds.uid, None)
        self.cell_refinement_results.pop(ds.uid, None)
        self.phase_identification_results.pop(ds.uid, None)
        self.qpa_results.pop(ds.uid, None)
        self.populate_fit_table()
        self.populate_fit_candidate_table()
        self.populate_size_strain_results()
        self.redraw()
        self.statusBar().showMessage(f"Cleared fitted profiles for {ds.name}.")

    def show_peak_table(self):
        self.tabs.setCurrentWidget(self.peak_tab)
        self.peak_results_tabs.setCurrentIndex(0)

    def show_fit_table(self):
        self.tabs.setCurrentWidget(self.peak_tab)
        self.peak_results_tabs.setCurrentIndex(1)

    def show_model_selection_table(self):
        self.tabs.setCurrentWidget(self.peak_tab)
        self.peak_results_tabs.setCurrentIndex(2)

    @staticmethod
    def _format_uncertainty(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "—"
        return "—" if not np.isfinite(numeric) else f"{numeric:.3g}"

    def populate_fit_table(self):
        rows = []
        by_uid = {ds.uid: ds for ds in self.datasets}
        for uid, groups in self.fit_groups.items():
            ds = by_uid.get(uid)
            if ds is None:
                continue
            for group in groups:
                for component in group.get("components", []):
                    rows.append((uid, ds.name, group, component))

        self.fit_table.setRowCount(len(rows))
        for row_index, (dataset_uid, dataset_name, group, component) in enumerate(rows):
            eta = component.get("eta")
            shape = component.get("shape")
            if eta is not None:
                shape_text = f"η={float(eta):.4f}"
            elif shape is not None:
                shape_text = f"m={float(shape):.4g}"
            elif component.get("model") == "Voigt":
                shape_text = (
                    f"G={float(component.get('fwhm_g', 0.0)):.4g}; "
                    f"L={float(component.get('fwhm_l', 0.0)):.4g}"
                )
            elif component.get("model") == "Split Pseudo-Voigt":
                shape_text = (
                    f"L={float(component.get('fwhm_left', 0.0)):.4g}; "
                    f"R={float(component.get('fwhm_right', 0.0)):.4g}"
                )
            else:
                shape_text = "—"

            component_flags = list(component.get("quality_flags", []))
            group_flags = list(group.get("quality_flags", []))
            flags = "; ".join(dict.fromkeys(component_flags + group_flags)) or "—"

            values = [
                dataset_name,
                str(group.get("group_id", "—")),
                group.get("fit_mode", "Standard"),
                component.get("model", group.get("model", "—")),
                str(component.get("component_id", "—")),
                f"{float(component['center']):.7g}",
                self._format_uncertainty(component.get("center_error")),
                f"{float(component['amplitude']):.7g}",
                f"{float(component['fwhm']):.7g}",
                self._format_uncertainty(component.get("fwhm_error")),
                f"{float(component['area']):.7g}",
                self._format_optional(
                    component.get("area_fraction_percent"), 5, "%"
                ),
                shape_text,
                self._format_optional(component.get("snr"), 5),
                self._format_optional(
                    component.get("separation_ratio"), 5
                ),
                self._format_optional(group.get("aicc"), 7),
                self._format_optional(group.get("bic"), 7),
                self._format_optional(group.get("r_squared"), 7),
                self._format_optional(group.get("rmse"), 7),
                group.get("baseline_model", "Linear"),
                flags,
            ]
            metadata = {
                "dataset_uid": dataset_uid,
                "center": float(component.get("center", 0.0)),
                "window_min": float(group.get("window_min", component.get("center", 0.0) - 0.75)),
                "window_max": float(group.get("window_max", component.get("center", 0.0) + 0.75)),
                "group_id": group.get("group_id"),
                "component_id": component.get("component_id"),
            }
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, metadata)
                self.fit_table.setItem(row_index, col, item)

    def populate_fit_candidate_table(self):
        rows = []
        by_uid = {ds.uid: ds for ds in self.datasets}
        for uid, candidates in self.fit_candidates.items():
            ds = by_uid.get(uid)
            if ds is None:
                continue
            for candidate in candidates:
                rows.append((uid, ds.name, candidate))

        self.fit_candidate_table.setRowCount(len(rows))
        for row_index, (dataset_uid, dataset_name, candidate) in enumerate(rows):
            flags = "; ".join(candidate.get("quality_flags", [])) or "—"
            values = [
                dataset_name,
                str(candidate.get("group_id", "—")),
                str(candidate.get("candidate_id", "—")),
                "Yes" if candidate.get("chosen") else "No",
                candidate.get("model", "—"),
                candidate.get("baseline_model", "—"),
                str(candidate.get("component_count", "—")),
                str(candidate.get("parameter_count", "—")),
                str(candidate.get("point_count", "—")),
                self._format_optional(candidate.get("aic"), 8),
                self._format_optional(candidate.get("aicc"), 8),
                self._format_optional(candidate.get("bic"), 8),
                self._format_optional(candidate.get("r_squared"), 7),
                self._format_optional(candidate.get("rmse"), 7),
                flags,
            ]
            metadata = {
                "dataset_uid": dataset_uid,
                "window_min": float(candidate.get("window_min", 0.0)),
                "window_max": float(candidate.get("window_max", 0.0)),
                "group_id": candidate.get("group_id"),
                "candidate_id": candidate.get("candidate_id"),
            }
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, metadata)
                self.fit_candidate_table.setItem(row_index, col, item)

    def export_fit_table_csv(self):
        rows = []
        by_uid = {ds.uid: ds for ds in self.datasets}
        for uid, groups in self.fit_groups.items():
            ds = by_uid.get(uid)
            if ds is None:
                continue
            for group in groups:
                for component in group.get("components", []):
                    rows.append({
                        "dataset": ds.name, "dataset_uid": ds.uid,
                        "group": group.get("group_id"), "fit_mode": group.get("fit_mode"),
                        "model": component.get("model"), "component_id": component.get("component_id"),
                        "center_deg": component.get("center"), "center_error_deg": component.get("center_error"),
                        "height": component.get("amplitude"), "height_error": component.get("amplitude_error"),
                        "fwhm_deg": component.get("fwhm"), "fwhm_error_deg": component.get("fwhm_error"),
                        "fwhm_left_deg": component.get("fwhm_left"), "fwhm_right_deg": component.get("fwhm_right"),
                        "fwhm_g_deg": component.get("fwhm_g"), "fwhm_l_deg": component.get("fwhm_l"),
                        "area": component.get("area"), "area_fraction_percent": component.get("area_fraction_percent"),
                        "eta": component.get("eta"), "eta_error": component.get("eta_error"),
                        "pearson_shape": component.get("shape"), "pearson_shape_error": component.get("shape_error"),
                        "snr": component.get("snr"), "separation_ratio": component.get("separation_ratio"),
                        "component_flags": component.get("quality_flags", []),
                        "baseline_model": group.get("baseline_model"),
                        "baseline_coefficients": group.get("baseline_coefficients"),
                        "robust_loss": group.get("robust_loss"), "selection_criterion": group.get("selection_criterion"),
                        "candidate_count": group.get("candidate_count"), "shared_width": group.get("shared_width"),
                        "aic": group.get("aic"), "aicc": group.get("aicc"), "bic": group.get("bic"),
                        "r_squared": group.get("r_squared"), "rmse": group.get("rmse"),
                        "reduced_chi_square": group.get("reduced_chi_square"),
                        "durbin_watson": group.get("durbin_watson"),
                        "max_parameter_correlation": group.get("max_parameter_correlation"),
                        "group_flags": group.get("quality_flags", []),
                        "window_min_deg": group.get("window_min"), "window_max_deg": group.get("window_max"),
                    })
        if not rows:
            self.statusBar().showMessage("There are no fitted peaks to export.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export peak fitting table", "peak_fits.txt", "Text data (*.txt)")
        if not filename:
            return
        try:
            result = write_table_txt(filename, rows, title="Peak fitting results")
        except Exception as exc:
            QMessageBox.critical(self, "TXT export failed", str(exc))
            return
        self.statusBar().showMessage(f"Peak fitting TXT exported: {result['txt_path']}")

    def populate_peak_table(self):
        self._peak_table_updating = True
        try:
            rows = []
            by_uid = {ds.uid: ds for ds in self.datasets}
            for uid, peaks in list(self.peak_rows.items()):
                ds = by_uid.get(uid)
                if ds is None:
                    continue
                normalized = normalize_peak_rows(peaks)
                self.peak_rows[uid] = normalized
                for peak in normalized:
                    rows.append((uid, ds.name, peak))
            self.peak_table.setRowCount(len(rows))
            for row_index, (uid, name, peak) in enumerate(rows):
                locked = self._main_peak_list_is_locked(uid)
                snr = peak.get("snr")
                confidence = peak.get("confidence")
                confidence_label = peak.get("confidence_label", "")
                metadata = {
                    "dataset_uid": uid,
                    "peak_uuid": peak.get("peak_uuid"),
                }

                use_item = QTableWidgetItem("")
                use_item.setData(Qt.UserRole, metadata)
                use_flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
                if not locked:
                    use_flags |= Qt.ItemIsUserCheckable
                use_item.setFlags(use_flags)
                use_item.setCheckState(
                    Qt.Checked if peak.get("use", True) else Qt.Unchecked
                )
                self.peak_table.setItem(row_index, 0, use_item)

                values = [
                    name,
                    str(peak.get("origin") or peak.get("method", "Regular")),
                    f"{peak['position']:.8g}",
                    f"{peak['intensity']:.8g}",
                    f"{peak['fwhm']:.8g}",
                    f"{peak['prominence']:.8g}",
                    "—" if snr is None else f"{snr:.3g}",
                    (
                        "—"
                        if confidence is None
                        else f"{confidence:.1f}% {confidence_label}"
                    ),
                    "Yes" if peak.get("protected") else "No",
                    str(peak.get("notes", "")),
                ]
                for offset, value in enumerate(values, start=1):
                    item = QTableWidgetItem(str(value))
                    item.setData(Qt.UserRole, metadata)
                    editable = (
                        not locked
                        and offset in (3, 10)
                    )
                    if not editable:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.peak_table.setItem(row_index, offset, item)
        finally:
            self._peak_table_updating = False

        dataset = self.selected_dataset()
        self._peak_table_updating = True
        try:
            if dataset is None:
                self.freeze_main_peak_list.setChecked(False)
                self.freeze_main_peak_list.setEnabled(False)
            else:
                self.freeze_main_peak_list.setEnabled(True)
                self.freeze_main_peak_list.setChecked(
                    self._main_peak_list_is_locked(dataset.uid)
                )
        finally:
            self._peak_table_updating = False
        self._update_main_peak_status()

    def estimate_stack_offset(self):
        visible = [ds for ds in self.datasets if ds.visible]
        if not visible:
            return
        spans = [float(np.max(ds.y) - np.min(ds.y)) for ds in visible]
        positive = [span for span in spans if span > 0]
        estimate = np.median(positive) * 1.15 if positive else 100.0
        self.stack_offset.setValue(float(estimate))
        self.stack_check.setChecked(True)
        self.redraw()
