from __future__ import annotations

from ..main_window_dependencies import *
from .base import WorkspaceAdapter


class PreparationWorkspace(WorkspaceAdapter):
    """Independently testable preparation-workspace adapter."""


class PreparationWorkspaceMixin:
    def _build_preprocessing_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        heading = QLabel("Background Detection and Subtraction")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        description = QLabel(
            "Detect a smooth, peak-protected baseline using arPLS, AsLS, SNIP, "
            "rolling-percentile, polynomial, or an automatic ensemble. The raw "
            "pattern is preserved and subtraction is reversible. No automatic "
            "baseline should be accepted without reviewing broad humps, amorphous "
            "scattering, fluorescence and sample-holder contributions."
        )
        description.setWordWrap(True)
        description.setObjectName("mutedLabel")
        install_label_copy_menu(description)
        layout.addWidget(description)

        button_row = QHBoxLayout()
        self.background_preview_tab_button = QPushButton("Preview selected method")
        self.background_preview_tab_button.setObjectName("primaryButton")
        self.background_apply_tab_button = QPushButton("Apply subtraction")
        # Historical label retained for compatibility tests: Export background TXT
        self.background_export_button = QPushButton("Export background via Raptor")
        button_row.addWidget(self.background_preview_tab_button)
        button_row.addWidget(self.background_apply_tab_button)
        button_row.addWidget(self.background_export_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.background_plot = AdvancedBackgroundPlotWidget()
        self.background_plot.setMinimumHeight(360)
        layout.addWidget(self.background_plot, 1)

        diagnostics_group = QGroupBox("Background diagnostics")
        diagnostics_layout = QFormLayout(diagnostics_group)
        self.background_method_value = QLabel("—")
        self.background_quality_value = QLabel("—")
        self.background_score_value = QLabel("—")
        self.background_noise_value = QLabel("—")
        self.background_protection_value = QLabel("—")
        self.background_negative_value = QLabel("—")
        self.background_contact_value = QLabel("—")
        self.background_roughness_value = QLabel("—")
        self.background_area_value = QLabel("—")
        self.background_warning_value = QLabel("No background has been evaluated.")
        self.background_warning_value.setWordWrap(True)
        for label in (
            self.background_method_value,
            self.background_quality_value,
            self.background_score_value,
            self.background_noise_value,
            self.background_protection_value,
            self.background_negative_value,
            self.background_contact_value,
            self.background_roughness_value,
            self.background_area_value,
            self.background_warning_value,
        ):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            install_label_copy_menu(label)
        diagnostics_layout.addRow("Selected method", self.background_method_value)
        diagnostics_layout.addRow("Quality classification", self.background_quality_value)
        diagnostics_layout.addRow("Automatic score", self.background_score_value)
        diagnostics_layout.addRow("Estimated noise σ", self.background_noise_value)
        diagnostics_layout.addRow("Peak protection", self.background_protection_value)
        diagnostics_layout.addRow("Negative corrected points", self.background_negative_value)
        diagnostics_layout.addRow("Background contact", self.background_contact_value)
        diagnostics_layout.addRow("Baseline roughness", self.background_roughness_value)
        diagnostics_layout.addRow("Background area fraction", self.background_area_value)
        diagnostics_layout.addRow("Warnings", self.background_warning_value)
        layout.addWidget(diagnostics_group)

        candidate_group = QGroupBox("Automatic candidate comparison")
        candidate_layout = QVBoxLayout(candidate_group)
        self.background_candidate_table = QTableWidget(0, 8)
        self.background_candidate_table.setHorizontalHeaderLabels(
            [
                "Method",
                "Score",
                "Negative %",
                "Severe negative %",
                "Contact %",
                "Roughness",
                "Edge mismatch",
                "Background area %",
            ]
        )
        self.background_candidate_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.background_candidate_table.setMaximumHeight(190)
        install_table_copy_menu(self.background_candidate_table)
        candidate_layout.addWidget(self.background_candidate_table)
        layout.addWidget(candidate_group)
        return widget

    def _build_smoothing_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        heading = QLabel("Peak-Preserving Smoothing")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        description = QLabel(
            "Reduce counting noise while explicitly monitoring peak centroid, "
            "height, integrated area and FWHM. Auto mode compares multiple "
            "smoothers and rejects candidates that distort protected peaks. "
            "Raw data remain unchanged and every preview is reversible."
        )
        description.setWordWrap(True)
        description.setObjectName("mutedLabel")
        install_label_copy_menu(description)
        layout.addWidget(description)

        controls = QHBoxLayout()
        self.smoothing_source = NoWheelComboBox()
        self.smoothing_source.addItems([
            "Current processed pattern",
            "Raw pattern",
        ])
        self.smoothing_preview_tab_button = QPushButton("Preview smart smoothing")
        self.smoothing_preview_tab_button.setObjectName("primaryButton")
        self.smoothing_apply_tab_button = QPushButton("Apply smoothed pattern")
        self.smoothing_revert_button = QPushButton("Revert smoothing")
        self.smoothing_export_button = QPushButton("Export smoothing via Raptor")
        controls.addWidget(QLabel("Source"))
        controls.addWidget(self.smoothing_source)
        controls.addWidget(self.smoothing_preview_tab_button)
        controls.addWidget(self.smoothing_apply_tab_button)
        controls.addWidget(self.smoothing_revert_button)
        controls.addWidget(self.smoothing_export_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.smoothing_plot = AdvancedSmoothingPlotWidget()
        self.smoothing_plot.setMinimumHeight(360)
        layout.addWidget(self.smoothing_plot, 1)

        diagnostics_group = QGroupBox("Smart smoothing diagnostics")
        diagnostics_layout = QFormLayout(diagnostics_group)
        self.smoothing_method_value = QLabel("—")
        self.smoothing_quality_value = QLabel("—")
        self.smoothing_score_value = QLabel("—")
        self.smoothing_noise_value = QLabel("—")
        self.smoothing_peak_protection_value = QLabel("—")
        self.smoothing_position_value = QLabel("—")
        self.smoothing_height_value = QLabel("—")
        self.smoothing_area_value = QLabel("—")
        self.smoothing_fwhm_value = QLabel("—")
        self.smoothing_residual_value = QLabel("—")
        self.smoothing_warning_value = QLabel("No smoothing has been evaluated.")
        self.smoothing_warning_value.setWordWrap(True)
        for label in (
            self.smoothing_method_value,
            self.smoothing_quality_value,
            self.smoothing_score_value,
            self.smoothing_noise_value,
            self.smoothing_peak_protection_value,
            self.smoothing_position_value,
            self.smoothing_height_value,
            self.smoothing_area_value,
            self.smoothing_fwhm_value,
            self.smoothing_residual_value,
            self.smoothing_warning_value,
        ):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            install_label_copy_menu(label)
        diagnostics_layout.addRow("Selected method", self.smoothing_method_value)
        diagnostics_layout.addRow("Quality classification", self.smoothing_quality_value)
        diagnostics_layout.addRow("Automatic score", self.smoothing_score_value)
        diagnostics_layout.addRow("Noise reduction", self.smoothing_noise_value)
        diagnostics_layout.addRow("Peak protection", self.smoothing_peak_protection_value)
        diagnostics_layout.addRow("Median position shift", self.smoothing_position_value)
        diagnostics_layout.addRow("Median height retention", self.smoothing_height_value)
        diagnostics_layout.addRow("Median area retention", self.smoothing_area_value)
        diagnostics_layout.addRow("Median FWHM change", self.smoothing_fwhm_value)
        diagnostics_layout.addRow("Residual lag-1 correlation", self.smoothing_residual_value)
        diagnostics_layout.addRow("Warnings", self.smoothing_warning_value)
        layout.addWidget(diagnostics_group)

        candidate_group = QGroupBox("Automatic method and strength comparison")
        candidate_layout = QVBoxLayout(candidate_group)
        self.smoothing_candidate_table = QTableWidget(0, 10)
        self.smoothing_candidate_table.setHorizontalHeaderLabels([
            "Method",
            "Strength",
            "Score",
            "Noise reduction %",
            "Position shift °",
            "Height retention %",
            "Area retention %",
            "FWHM change %",
            "Residual corr.",
            "Quality",
        ])
        self.smoothing_candidate_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.smoothing_candidate_table.setMaximumHeight(220)
        install_table_copy_menu(self.smoothing_candidate_table)
        candidate_layout.addWidget(self.smoothing_candidate_table)
        layout.addWidget(candidate_group)
        return widget

    def _background_parameters_from_controls(self):
        return BackgroundParameters(
            method=self.background_method.currentText(),
            smoothness=self.background_smoothness.value(),
            asymmetry=self.background_asymmetry.value(),
            iterations=self.background_iterations.value(),
            window_degrees=self.background_window_degrees.value(),
            percentile=self.background_percentile.value(),
            polynomial_order=self.poly_order.value(),
            peak_protection=self.background_peak_protection.isChecked(),
            clip_negative=self.background_clip_negative.isChecked(),
        ).normalized()

    def preview_background(self):
        ds = self.selected_dataset()
        if ds is None:
            QMessageBox.information(self, "No dataset", "Select a dataset first.")
            return
        try:
            result = detect_background(
                ds.x,
                ds.y_raw,
                self._background_parameters_from_controls(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Background detection failed", str(exc))
            return
        self._push_undo_checkpoint("Preview background")
        self.backgrounds[ds.uid] = np.asarray(result["background"], dtype=float)
        self.background_results[ds.uid] = {
            "diagnostics": result.get("diagnostics", {}),
            "parameters": result.get("parameters", {}),
            "background": np.asarray(result["background"], dtype=float).tolist(),
            "protected_mask": np.asarray(
                result.get("protected_mask", np.zeros(len(ds.x), dtype=bool)),
                dtype=bool,
            ).tolist(),
            "applied": False,
        }
        self.populate_background_result()
        self.redraw()
        self.tabs.setCurrentWidget(self.preprocessing_tab)
        diagnostics = result.get("diagnostics", {})
        self.statusBar().showMessage(
            f"Background preview for {ds.name}: "
            f"{diagnostics.get('selected_method', 'method unavailable')} — "
            f"{diagnostics.get('quality', 'review required')}."
        )

    def _rebuild_backgrounds_from_results(self):
        """Restore raw background records and scale-safe display curves.

        ``background_results[uid]["background"]`` always remains in the
        original detector-count units used by diagnostics and TXT export.
        ``self.backgrounds`` is the curve drawn over the active pattern and may
        therefore be normalized when the processed pattern was normalized.
        """
        self.backgrounds.clear()
        datasets_by_uid = {dataset.uid: dataset for dataset in self.datasets}
        rebuilt: dict[str, dict] = {}
        for uid, record in list(self.background_results.items()):
            dataset = datasets_by_uid.get(uid)
            if dataset is None or not isinstance(record, dict):
                continue
            parameters = record.get("parameters", {})
            if not isinstance(parameters, dict):
                continue

            stored_background = np.asarray(record.get("background", []), dtype=float)
            if not (
                stored_background.shape == dataset.y_raw.shape
                and np.all(np.isfinite(stored_background))
            ):
                try:
                    result = detect_background(
                        dataset.x,
                        dataset.y_raw,
                        BackgroundParameters(**parameters),
                    )
                except Exception:
                    continue
                stored_background = np.asarray(result["background"], dtype=float)
                diagnostics = result.get("diagnostics", {})
                parameters = result.get("parameters", parameters)
                protected_mask = np.asarray(
                    result.get(
                        "protected_mask",
                        np.zeros(len(dataset.x), dtype=bool),
                    ),
                    dtype=bool,
                ).tolist()
            else:
                diagnostics = record.get("diagnostics", {})
                protected_mask = list(record.get("protected_mask", []))

            display_background = np.asarray(
                record.get("display_background", []), dtype=float
            )
            if not (
                display_background.shape == dataset.y_raw.shape
                and np.all(np.isfinite(display_background))
            ):
                try:
                    factor = float(record.get("normalization_factor", 1.0))
                except (TypeError, ValueError):
                    factor = 1.0
                if np.isfinite(factor) and factor > 0.0 and not np.isclose(factor, 1.0):
                    display_background = stored_background * factor
                else:
                    display_background = stored_background.copy()

                    # Legacy projects could contain a normalized active pattern
                    # but a raw-count background with no recorded scale factor.
                    # Drawing that curve would flatten the visible peaks, so omit
                    # only the main-plot overlay while retaining the raw scientific
                    # background in its dedicated tab and exports.
                    if dataset.y_processed is not None and bool(record.get("applied")):
                        processed_max = float(np.max(np.abs(dataset.y_processed)))
                        background_max = float(np.max(np.abs(stored_background)))
                        if (
                            processed_max > 0.0
                            and processed_max <= 100.000001
                            and background_max > 5.0 * processed_max
                        ):
                            display_background = None

            if display_background is not None:
                self.backgrounds[uid] = np.asarray(display_background, dtype=float)

            rebuilt_record = {
                **record,
                "diagnostics": diagnostics,
                "parameters": parameters,
                "background": stored_background.tolist(),
                "protected_mask": protected_mask,
                "applied": bool(record.get("applied", dataset.y_processed is not None)),
            }
            if display_background is not None:
                rebuilt_record["display_background"] = np.asarray(
                    display_background, dtype=float
                ).tolist()
            else:
                rebuilt_record.pop("display_background", None)
            rebuilt[uid] = rebuilt_record
        self.background_results = rebuilt

    def populate_background_result(self):
        if not hasattr(self, "background_plot"):
            return
        ds = self.selected_dataset()
        record = None if ds is None else self.background_results.get(ds.uid)
        background = (
            None
            if ds is None or not isinstance(record, dict)
            else np.asarray(record.get("background", []), dtype=float)
        )
        if (
            ds is None
            or record is None
            or background is None
            or background.shape != ds.y_raw.shape
            or not np.all(np.isfinite(background))
        ):
            self.background_plot.clear()
            for label in (
                self.background_method_value,
                self.background_quality_value,
                self.background_score_value,
                self.background_noise_value,
                self.background_protection_value,
                self.background_negative_value,
                self.background_contact_value,
                self.background_roughness_value,
                self.background_area_value,
            ):
                label.setText("—")
            self.background_warning_value.setText(
                "No background has been evaluated for the selected dataset."
            )
            self.background_candidate_table.setRowCount(0)
            return

        diagnostics = record.get("diagnostics", {})
        parameters = record.get("parameters", {})
        corrected = np.asarray(ds.y_raw, dtype=float) - np.asarray(background, dtype=float)
        if parameters.get("clip_negative", False):
            corrected = np.clip(corrected, 0.0, None)
        protected_mask = np.asarray(
            record.get("protected_mask", np.zeros(len(ds.x), dtype=bool)),
            dtype=bool,
        )
        self.background_plot.update_result(
            ds.x,
            ds.y_raw,
            background,
            corrected,
            protected_mask,
        )
        applied_text = "Applied" if record.get("applied") else "Preview only"
        self.background_method_value.setText(
            f"{diagnostics.get('selected_method', '—')} ({applied_text})"
        )
        self.background_quality_value.setText(str(diagnostics.get("quality", "—")))
        self.background_score_value.setText(
            self._format_background_value(diagnostics.get("score"), ".6g")
        )
        self.background_noise_value.setText(
            self._format_background_value(diagnostics.get("noise_sigma"), ".6g")
        )
        self.background_protection_value.setText(
            f"{diagnostics.get('protected_peak_count', 0)} peaks; "
            f"{100.0 * float(diagnostics.get('protected_point_fraction', 0.0)):.4g}% points"
        )
        self.background_negative_value.setText(
            f"{100.0 * float(diagnostics.get('negative_fraction', 0.0)):.4g}% total; "
            f"{100.0 * float(diagnostics.get('severe_negative_fraction', 0.0)):.4g}% below −3σ"
        )
        self.background_contact_value.setText(
            f"{100.0 * float(diagnostics.get('contact_fraction', 0.0)):.4g}%"
        )
        self.background_roughness_value.setText(
            self._format_background_value(diagnostics.get("roughness_ratio"), ".6g")
        )
        self.background_area_value.setText(
            f"{100.0 * float(diagnostics.get('background_area_fraction', 0.0)):.4g}%"
        )
        warnings_list = diagnostics.get("warnings", [])
        self.background_warning_value.setText(
            "No automatic warning. Manual scientific review remains required."
            if not warnings_list
            else " ".join(str(item) for item in warnings_list)
        )

        candidates = diagnostics.get("candidate_scores", [])
        self.background_candidate_table.setRowCount(len(candidates))
        for row_index, candidate in enumerate(candidates):
            values = [
                candidate.get("method", ""),
                self._format_background_value(candidate.get("score"), ".6g"),
                f"{100.0 * float(candidate.get('negative_fraction', 0.0)):.4g}",
                f"{100.0 * float(candidate.get('severe_negative_fraction', 0.0)):.4g}",
                f"{100.0 * float(candidate.get('contact_fraction', 0.0)):.4g}",
                self._format_background_value(candidate.get("roughness_ratio"), ".6g"),
                self._format_background_value(candidate.get("edge_mismatch_ratio"), ".6g"),
                f"{100.0 * float(candidate.get('background_area_fraction', 0.0)):.4g}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.background_candidate_table.setItem(row_index, column, item)

    @staticmethod
    def _format_background_value(value, specification):
        if value is None:
            return "—"
        try:
            return format(float(value), specification)
        except (TypeError, ValueError):
            return str(value)

    def export_background_csv(self):
        ds = self.selected_dataset()
        record = {} if ds is None else self.background_results.get(ds.uid, {})
        background = np.asarray(record.get("background", []), dtype=float)
        if (
            ds is None
            or background.shape != ds.y_raw.shape
            or not np.all(np.isfinite(background))
        ):
            QMessageBox.information(self, "No background", "Preview or apply a background model first.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export advanced background", f"{ds.name}_background.txt", "Text data (*.txt)"
        )
        if not filename:
            return
        parameters = record.get("parameters", {})
        corrected = np.asarray(ds.y_raw, dtype=float) - background
        if parameters.get("clip_negative", False):
            corrected = np.clip(corrected, 0.0, None)
        protected = np.asarray(record.get("protected_mask", np.zeros(len(ds.x), dtype=bool)), dtype=bool)
        try:
            result = write_columns_txt(
                filename,
                {
                    "two_theta_deg": ds.x,
                    "raw_intensity": ds.y_raw,
                    "detected_background": background,
                    "background_corrected": corrected,
                    "peak_protected": protected.astype(int),
                },
                title=f"Advanced background — {ds.name}",
                metadata={"dataset_uid": ds.uid, "method": parameters.get("method", "Not recorded")},
            )
        except Exception as exc:
            QMessageBox.critical(self, "TXT export failed", str(exc))
            return
        self.statusBar().showMessage(f"Advanced background TXT exported: {result['txt_path']}")

    def _smoothing_parameters_from_controls(self):
        return SmoothingParameters(
            method=self.smoothing_method.currentText(),
            strength=self.smoothing_strength.value(),
            peak_protection=self.smoothing_peak_protection.isChecked(),
            peak_preservation=self.smoothing_peak_preservation.value(),
            maximum_position_shift_deg=self.smoothing_max_position_shift.value(),
            maximum_height_change_percent=self.smoothing_max_height_change.value(),
            maximum_fwhm_change_percent=self.smoothing_max_fwhm_change.value(),
        ).normalized()

    def _smoothing_source_array(self, ds):
        record = self.smoothing_results.get(ds.uid, {})
        if record.get("applied") and record.get("source") is not None:
            stored = np.asarray(record.get("source"), dtype=float)
            if stored.shape == ds.y_raw.shape and np.all(np.isfinite(stored)):
                return stored, str(record.get("source_label", "Stored pre-smoothing source"))
        if (
            self.smoothing_source.currentText() == "Current processed pattern"
            and ds.y_processed is not None
        ):
            return np.asarray(ds.y_processed, dtype=float).copy(), "Current processed pattern"
        return np.asarray(ds.y_raw, dtype=float).copy(), "Raw pattern"

    def preview_smoothing(self):
        ds = self.selected_dataset()
        if ds is None:
            QMessageBox.information(self, "No dataset", "Select a dataset first.")
            return
        source, source_label = self._smoothing_source_array(ds)
        try:
            result = smart_smooth_pattern(
                ds.x,
                source,
                self._smoothing_parameters_from_controls(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Smart smoothing failed", str(exc))
            return
        self._push_undo_checkpoint("Preview smart smoothing")
        self.smoothing_results[ds.uid] = {
            "diagnostics": result.get("diagnostics", {}),
            "parameters": result.get("parameters", {}),
            "source": source.tolist(),
            "source_label": source_label,
            "smoothed": np.asarray(result["smoothed"], dtype=float).tolist(),
            "protected_mask": np.asarray(
                result.get("protected_mask", np.zeros(len(ds.x), dtype=bool)),
                dtype=bool,
            ).tolist(),
            "applied": False,
        }
        self.populate_smoothing_result()
        self.tabs.setCurrentWidget(self.smoothing_tab)
        diagnostics = result.get("diagnostics", {})
        self.statusBar().showMessage(
            f"Smart smoothing preview for {ds.name}: "
            f"{diagnostics.get('selected_method', 'method unavailable')} — "
            f"{diagnostics.get('quality', 'review required')}."
        )

    def apply_smoothing_preview(self):
        ds = self.selected_dataset()
        if ds is None:
            return
        record = self.smoothing_results.get(ds.uid)
        if not isinstance(record, dict) or record.get("smoothed") is None:
            self.preview_smoothing()
            record = self.smoothing_results.get(ds.uid)
        if not isinstance(record, dict) or record.get("smoothed") is None:
            return
        smoothed = np.asarray(record["smoothed"], dtype=float)
        if smoothed.shape != ds.y_raw.shape:
            QMessageBox.critical(
                self,
                "Smoothing result mismatch",
                "The smoothing result no longer matches the selected dataset.",
            )
            return
        quality = str(record.get("diagnostics", {}).get("quality", ""))
        if quality.startswith("Aggressive"):
            response = QMessageBox.warning(
                self,
                "Aggressive smoothing",
                "Diagnostics indicate significant peak distortion. Apply anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if response != QMessageBox.Yes:
                return
        self._push_undo_checkpoint("Apply smart smoothing")
        ds.y_processed = smoothed.copy()
        ds.metadata["processing_provenance"] = {
            "source": record.get("source_label", "raw pattern"),
            "normalized": record.get("normalization_target") is not None,
            "normalization_factor": record.get("normalization_factor", 1.0),
            "background_subtracted": False,
            "smoothed": True,
            "negative_values_clipped": False,
            "statistical_covariance_propagated": False,
        }
        record["applied"] = True
        self._clear_downstream_results(ds.uid)
        self.redraw()
        self.populate_smoothing_result()
        self.statusBar().showMessage(f"Applied smart smoothing to {ds.name}.")
        self._record_scientific_result(
            "preparation",
            ds.uid,
            self._scientific_stage_payloads(ds)["preparation"],
            reason="Applied smart smoothing",
        )

    def revert_smoothing(self):
        ds = self.selected_dataset()
        if ds is None:
            return
        record = self.smoothing_results.get(ds.uid)
        if not isinstance(record, dict) or record.get("source") is None:
            QMessageBox.information(
                self,
                "No smoothing source",
                "No reversible smoothing preview is stored for this dataset.",
            )
            return
        source = np.asarray(record["source"], dtype=float)
        if source.shape != ds.y_raw.shape:
            QMessageBox.critical(
                self,
                "Smoothing source mismatch",
                "The stored pre-smoothing source no longer matches the dataset.",
            )
            return
        self._push_undo_checkpoint("Revert smart smoothing")
        ds.y_processed = source.copy()
        record["applied"] = False
        self._clear_downstream_results(ds.uid)
        self.redraw()
        self.populate_smoothing_result()
        self.statusBar().showMessage(f"Reverted smart smoothing for {ds.name}.")
        self._record_scientific_result(
            "preparation",
            ds.uid,
            self._scientific_stage_payloads(ds)["preparation"],
            reason="Reverted smart smoothing",
        )

    def _refresh_protected_main_peaks(self, uid: str):
        dataset = next((ds for ds in self.datasets if ds.uid == uid), None)
        stored = normalize_peak_rows(self.peak_rows.get(uid, []))
        protected = (
            stored
            if self._main_peak_list_is_locked(uid)
            else [
                row for row in stored
                if row.get("manual") or row.get("protected")
            ]
        )
        if dataset is None or not protected:
            self.peak_rows.pop(uid, None)
            return
        refreshed = []
        for old in protected:
            try:
                row = create_manual_peak(
                    dataset.x,
                    dataset.y,
                    float(old["position"]),
                    snap_window_deg=0.0,
                    peak_uuid=old.get("peak_uuid"),
                    notes=str(old.get("notes", "")),
                    origin=str(old.get("origin", "Protected")),
                )
            except Exception:
                row = dict(old)
            row["use"] = bool(old.get("use", True))
            row["manual"] = bool(old.get("manual", False))
            row["protected"] = True
            row["method"] = str(old.get("method", row.get("method", "Manual")))
            for key in (
                "master_peak_revision",
                "master_peak_checksum",
            ):
                if key in old:
                    row[key] = old[key]
            refreshed.append(row)
        self.peak_rows[uid] = normalize_peak_rows(refreshed)

    def _clear_downstream_results(self, uid):
        if (
            hasattr(self, "preserve_manual_peaks_check")
            and self.preserve_manual_peaks_check.isChecked()
        ) or self._main_peak_list_is_locked(uid):
            self._refresh_protected_main_peaks(uid)
        else:
            self.peak_rows.pop(uid, None)
        self.fit_groups.pop(uid, None)
        self.fit_candidates.pop(uid, None)
        self.size_strain_results.pop(uid, None)
        self.cell_refinement_results.pop(uid, None)
        self.phase_identification_results.pop(uid, None)
        self.qpa_results.pop(uid, None)
        if hasattr(self, "whole_pattern_widget"):
            self.whole_pattern_widget.results_by_uid.pop(uid, None)
        if hasattr(self, "rietveld_widget"):
            self.rietveld_widget.results_by_uid.pop(uid, None)
        if hasattr(self, "validated_qpa_widget"):
            self.validated_qpa_widget.native_results_by_uid.pop(uid, None)
        self.populate_peak_table()
        self.populate_fit_table()
        self.populate_fit_candidate_table()
        self.populate_size_strain_results()

    def _rebuild_smoothing_results(self):
        datasets = {dataset.uid: dataset for dataset in self.datasets}
        rebuilt = {}
        for uid, record in list(self.smoothing_results.items()):
            dataset = datasets.get(uid)
            if dataset is None or not isinstance(record, dict):
                continue
            try:
                source = np.asarray(record.get("source"), dtype=float)
                smoothed = np.asarray(record.get("smoothed"), dtype=float)
                protected = np.asarray(
                    record.get("protected_mask", np.zeros(len(dataset.x), dtype=bool)),
                    dtype=bool,
                )
            except Exception:
                continue
            if source.shape != dataset.y_raw.shape or smoothed.shape != dataset.y_raw.shape:
                continue
            rebuilt[uid] = {
                **record,
                "source": source.tolist(),
                "smoothed": smoothed.tolist(),
                "protected_mask": protected.tolist(),
            }
        self.smoothing_results = rebuilt

    def populate_smoothing_result(self):
        if not hasattr(self, "smoothing_plot"):
            return
        ds = self.selected_dataset()
        record = None if ds is None else self.smoothing_results.get(ds.uid)
        if ds is None or not isinstance(record, dict):
            self.smoothing_plot.clear()
            for label in (
                self.smoothing_method_value,
                self.smoothing_quality_value,
                self.smoothing_score_value,
                self.smoothing_noise_value,
                self.smoothing_peak_protection_value,
                self.smoothing_position_value,
                self.smoothing_height_value,
                self.smoothing_area_value,
                self.smoothing_fwhm_value,
                self.smoothing_residual_value,
            ):
                label.setText("—")
            self.smoothing_warning_value.setText(
                "No smoothing has been evaluated for the selected dataset."
            )
            self.smoothing_candidate_table.setRowCount(0)
            return
        source = np.asarray(record.get("source"), dtype=float)
        smoothed = np.asarray(record.get("smoothed"), dtype=float)
        protected = np.asarray(
            record.get("protected_mask", np.zeros(len(ds.x), dtype=bool)),
            dtype=bool,
        )
        if source.shape != ds.y_raw.shape or smoothed.shape != ds.y_raw.shape:
            self.smoothing_plot.clear()
            return
        self.smoothing_plot.update_result(ds.x, source, smoothed, protected)
        diagnostics = record.get("diagnostics", {})
        applied = "Applied" if record.get("applied") else "Preview only"
        self.smoothing_method_value.setText(
            f"{diagnostics.get('selected_method', '—')} at "
            f"{self._format_background_value(diagnostics.get('selected_strength'), '.5g')} "
            f"({applied})"
        )
        self.smoothing_quality_value.setText(str(diagnostics.get("quality", "—")))
        self.smoothing_score_value.setText(
            self._format_background_value(diagnostics.get("score"), ".6g")
        )
        self.smoothing_noise_value.setText(
            f"{self._format_background_value(diagnostics.get('noise_sigma_before'), '.6g')} → "
            f"{self._format_background_value(diagnostics.get('noise_sigma_after'), '.6g')}; "
            f"{self._format_background_value(diagnostics.get('noise_reduction_percent'), '.5g')}%"
        )
        self.smoothing_peak_protection_value.setText(
            f"{diagnostics.get('protected_peak_count', 0)} peaks; "
            f"{100.0 * float(diagnostics.get('protected_point_fraction', 0.0)):.4g}% points"
        )
        self.smoothing_position_value.setText(
            f"maximum of peak/centroid medians = "
            f"{max(float(diagnostics.get('median_position_shift_deg', 0.0)), float(diagnostics.get('median_centroid_shift_deg', 0.0))):.6g}°"
        )
        self.smoothing_height_value.setText(
            f"{self._format_background_value(diagnostics.get('median_height_retention_percent'), '.6g')}%"
        )
        self.smoothing_area_value.setText(
            f"{self._format_background_value(diagnostics.get('median_area_retention_percent'), '.6g')}%"
        )
        self.smoothing_fwhm_value.setText(
            f"{self._format_background_value(diagnostics.get('median_fwhm_change_percent'), '.6g')}%"
        )
        self.smoothing_residual_value.setText(
            self._format_background_value(diagnostics.get("residual_lag1_correlation"), ".6g")
        )
        warnings = diagnostics.get("warnings", [])
        boundary = diagnostics.get("scientific_boundary", "")
        self.smoothing_warning_value.setText(
            ("No automatic warning. " if not warnings else " ".join(map(str, warnings)) + " ")
            + str(boundary)
        )
        candidates = diagnostics.get("candidate_scores", [])
        self.smoothing_candidate_table.setRowCount(len(candidates))
        for row_index, candidate in enumerate(candidates):
            values = [
                candidate.get("method", ""),
                self._format_background_value(candidate.get("strength"), ".5g"),
                self._format_background_value(candidate.get("score"), ".6g"),
                self._format_background_value(candidate.get("noise_reduction_percent"), ".6g"),
                self._format_background_value(max(float(candidate.get("median_position_shift_deg", 0.0)), float(candidate.get("median_centroid_shift_deg", 0.0))), ".6g"),
                self._format_background_value(candidate.get("median_height_retention_percent"), ".6g"),
                self._format_background_value(candidate.get("median_area_retention_percent"), ".6g"),
                self._format_background_value(candidate.get("median_fwhm_change_percent"), ".6g"),
                self._format_background_value(candidate.get("residual_lag1_correlation"), ".6g"),
                candidate.get("quality", ""),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.smoothing_candidate_table.setItem(row_index, column, item)

    def export_smoothing_csv(self):
        ds = self.selected_dataset()
        record = None if ds is None else self.smoothing_results.get(ds.uid)
        if ds is None or not isinstance(record, dict):
            QMessageBox.information(self, "No smoothing result", "Preview smart smoothing first.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export smart smoothing", f"{ds.name}_smart_smoothing.txt", "Text data (*.txt)"
        )
        if not filename:
            return
        source = np.asarray(record["source"], dtype=float)
        smoothed = np.asarray(record["smoothed"], dtype=float)
        protected = np.asarray(record.get("protected_mask", np.zeros(len(ds.x), dtype=bool)), dtype=bool)
        try:
            result = write_columns_txt(
                filename,
                {
                    "two_theta_deg": ds.x,
                    "raw_intensity": ds.y_raw,
                    "smoothing_source": source,
                    "smart_smoothed": smoothed,
                    "removed_component": source - smoothed,
                    "peak_protected": protected.astype(int),
                },
                title=f"Smart smoothing — {ds.name}",
                metadata={
                    "dataset_uid": ds.uid,
                    "source_label": record.get("source_label", "Smoothing source"),
                    "normalization_factor": record.get("normalization_factor", 1.0),
                    "normalization_target": record.get("normalization_target"),
                    "display_intensity_units": (
                        "normalized maximum = 100"
                        if record.get("normalization_target") is not None
                        else "input intensity units"
                    ),
                },
            )
        except Exception as exc:
            QMessageBox.critical(self, "TXT export failed", str(exc))
            return
        self.statusBar().showMessage(f"Smart smoothing TXT exported: {result['txt_path']}")

    def apply_processing(self):
        ds = self.selected_dataset()
        if ds is None:
            QMessageBox.information(self, "No dataset", "Select a dataset first.")
            return

        params = ProcessingParameters(
            polynomial_order=self.poly_order.value(),
            smoothing_window=self.smooth_window.value(),
            smoothing_order=self.smooth_order.value(),
            smoothing_method=self.smoothing_method.currentText(),
            smoothing_strength=self.smoothing_strength.value(),
            smoothing_peak_protection=self.smoothing_peak_protection.isChecked(),
            smoothing_peak_preservation=self.smoothing_peak_preservation.value(),
            smoothing_maximum_position_shift_deg=self.smoothing_max_position_shift.value(),
            smoothing_maximum_height_change_percent=self.smoothing_max_height_change.value(),
            smoothing_maximum_fwhm_change_percent=self.smoothing_max_fwhm_change.value(),
            normalize=self.normalize_check.isChecked(),
            subtract_background=self.background_check.isChecked(),
            smooth=self.smooth_check.isChecked(),
            background_method=self.background_method.currentText(),
            background_smoothness=self.background_smoothness.value(),
            background_asymmetry=self.background_asymmetry.value(),
            background_iterations=self.background_iterations.value(),
            background_window_degrees=self.background_window_degrees.value(),
            background_percentile=self.background_percentile.value(),
            background_peak_protection=self.background_peak_protection.isChecked(),
            clip_negative=self.background_clip_negative.isChecked(),
        )
        try:
            processed, aux = process_pattern(ds.x, ds.y_raw, params)
        except Exception as exc:
            QMessageBox.critical(self, "Processing failed", str(exc))
            return

        self._push_undo_checkpoint("Apply pattern preparation")
        ds.y_processed = processed
        ds.metadata["processing_provenance"] = {
            "source": "raw pattern",
            "normalized": bool(params.normalize),
            "normalization_factor": aux.get("normalization_factor", 1.0),
            "background_subtracted": bool(params.subtract_background),
            "smoothed": bool(params.smooth),
            "negative_values_clipped": bool(params.clip_negative and params.subtract_background),
            "statistical_covariance_propagated": bool(
                params.normalize and not params.subtract_background and not params.smooth
            ),
        }
        if "background" in aux:
            raw_background = np.asarray(aux["background"], dtype=float)
            display_background = np.asarray(
                aux.get("background_display", raw_background), dtype=float
            )
            self.backgrounds[ds.uid] = display_background
            self.background_results[ds.uid] = {
                "diagnostics": aux.get("background_diagnostics", {}),
                "parameters": aux.get("background_parameters", {}),
                "background": raw_background.tolist(),
                "display_background": display_background.tolist(),
                "normalization_factor": aux.get("normalization_factor", 1.0),
                "normalization_reference_maximum": aux.get(
                    "normalization_reference_maximum"
                ),
                "normalization_target": aux.get("normalization_target"),
                "protected_mask": np.asarray(
                    aux.get("background_protected_mask", np.zeros(len(ds.x), dtype=bool)),
                    dtype=bool,
                ).tolist(),
                "applied": True,
            }
        else:
            self.backgrounds.pop(ds.uid, None)
            self.background_results.pop(ds.uid, None)
        if "smoothed" in aux:
            normalized = "normalization_factor" in aux
            self.smoothing_results[ds.uid] = {
                "diagnostics": aux.get("smoothing_diagnostics", {}),
                "parameters": aux.get("smoothing_parameters", {}),
                "source": np.asarray(aux.get("smoothing_source", ds.y_raw), dtype=float).tolist(),
                "source_label": (
                    "Pre-smoothing source in normalized intensity units"
                    if normalized
                    else "Pre-smoothing source"
                ),
                "smoothed": np.asarray(aux["smoothed"], dtype=float).tolist(),
                "normalization_factor": aux.get("normalization_factor", 1.0),
                "normalization_reference_maximum": aux.get(
                    "normalization_reference_maximum"
                ),
                "normalization_target": aux.get("normalization_target"),
                "protected_mask": np.asarray(
                    aux.get("smoothing_protected_mask", np.zeros(len(ds.x), dtype=bool)),
                    dtype=bool,
                ).tolist(),
                "applied": True,
            }
        else:
            self.smoothing_results.pop(ds.uid, None)
        self._clear_downstream_results(ds.uid)
        self.redraw()
        self.populate_background_result()
        self.populate_smoothing_result()
        self.populate_peak_table()
        self.populate_fit_table()
        self.populate_fit_candidate_table()
        self.populate_size_strain_results()
        self.statusBar().showMessage(f"Processed {ds.name}; previous peaks and fits were cleared.")
        self._record_scientific_result(
            "preparation",
            ds.uid,
            self._scientific_stage_payloads(ds)["preparation"],
            reason="Applied pattern preparation",
        )

    def reset_processing(self):
        ds = self.selected_dataset()
        if ds is None:
            return
        self._push_undo_checkpoint("Reset processing")
        ds.reset_processing()
        ds.metadata.pop("processing_provenance", None)
        self.backgrounds.pop(ds.uid, None)
        self.background_results.pop(ds.uid, None)
        self.smoothing_results.pop(ds.uid, None)
        self._clear_downstream_results(ds.uid)
        if hasattr(self, "phase_revolution_widget"):
            widget = self.phase_revolution_widget
            if widget.share_master_peaks.isChecked() and widget.peaks_by_uid.get(ds.uid):
                widget._sync_master_to_application(ds.uid)
        self.redraw()
        self.populate_background_result()
        self.populate_smoothing_result()
        self.populate_peak_table()
        self.populate_fit_table()
        self.populate_fit_candidate_table()
        self.populate_size_strain_results()
        self.populate_crystal_results()
        self.statusBar().showMessage(f"Reset processing for {ds.name}.")
        self._sync_unified_scientific_state(ds.uid)
        self._update_workflow_dashboard()
