from __future__ import annotations

from ..main_window_dependencies import *
from .base import WorkspaceAdapter


class PhaseWorkspace(WorkspaceAdapter):
    """Independently testable phase/QPA workspace adapter."""


class PhaseWorkspaceMixin:
    def _phase_reference_library(self):
        selected = self.selected_dataset()
        references = []
        seen_uids = set()
        for dataset in self.datasets:
            if selected is not None and dataset.uid == selected.uid:
                continue
            reference = reference_from_dataset(dataset)
            if reference is None or reference.uid in seen_uids:
                continue
            references.append(reference)
            seen_uids.add(reference.uid)

        cif_reference = reference_from_cif_pattern(
            self.reference_structure,
            self.reference_pattern,
            self.wavelength_angstrom.value(),
        )
        if cif_reference is not None and cif_reference.uid not in seen_uids:
            references.append(cif_reference)
        return references

    def _observed_peaks_for_phase_identification(self, dataset):
        source_mode = self.phase_peak_source.currentText()
        fitted = []
        for group in self.fit_groups.get(dataset.uid, []):
            for component in group.get("components", []):
                fitted.append(
                    {
                        "position": float(component["center"]),
                        "intensity": float(component.get("amplitude", 0.0)),
                        "source": "Fitted",
                    }
                )
        fitted.sort(key=lambda row: row["position"])

        detected = [
            {
                "position": float(row["position"]),
                "intensity": float(row.get("intensity", 0.0)),
                "source": row.get("method", "Detected"),
            }
            for row in self._active_peak_rows_for_uid(dataset.uid)
        ]

        if source_mode == "Fitted peaks only":
            return fitted
        if source_mode == "Detected peaks only":
            return detected
        if fitted:
            return fitted
        if detected:
            return detected

        rows, _ = smart_detect_peaks(
            dataset.x,
            dataset.y,
            sensitivity=self.smart_sensitivity.currentText(),
        )
        self._commit_main_peak_rows(
            dataset.uid,
            rows,
            reason="Automatic peak search for phase identification",
        )
        return [
            {
                "position": float(row["position"]),
                "intensity": float(row.get("intensity", 0.0)),
                "source": row.get("method", "Smart"),
            }
            for row in rows
        ]

    def identify_phases_for_selected(self):
        dataset = self.selected_dataset()
        if dataset is None:
            self.statusBar().showMessage(
                "Select an experimental dataset before phase identification."
            )
            return
        if dataset.metadata.get("analysis_role") == "reference_pattern":
            self.statusBar().showMessage(
                "Select an experimental pattern, not a reference-card dataset."
            )
            return

        references = self._phase_reference_library()
        if not references:
            QMessageBox.information(
                self,
                "Empty reference library",
                "Import one or more Jade/PDF reference-card text files or calculate an active CIF reference first.",
            )
            return

        try:
            observed = self._observed_peaks_for_phase_identification(dataset)
        except Exception as exc:
            QMessageBox.critical(self, "Observed peak preparation failed", str(exc))
            return
        if len(observed) < 2:
            self.statusBar().showMessage(
                "At least two observed peaks are required for phase identification."
            )
            return

        try:
            result = identify_phases(
                observed,
                references,
                target_wavelength_angstrom=self.wavelength_angstrom.value(),
                tolerance_deg=self.phase_tolerance.value(),
                maximum_zero_shift_deg=self.phase_max_shift.value(),
                reference_intensity_cutoff_percent=(
                    self.phase_reference_cutoff.value()
                ),
                convert_from_d=self.phase_convert_d_check.isChecked(),
                maximum_phases=self.phase_maximum_phases.value(),
                mixture_pool_size=self.phase_mixture_pool.value(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Phase identification failed", str(exc))
            return

        result["dataset_name"] = dataset.name
        result["observed_peak_source"] = self.phase_peak_source.currentText()
        self.phase_identification_results[dataset.uid] = result
        self._record_scientific_result(
            "phase",
            dataset.uid,
            {"known_phase": result},
            reason="Completed known-phase identification",
        )
        self.qpa_results.pop(dataset.uid, None)
        self._qpa_setup_signature = None
        self.populate_phase_identification_results()
        self.populate_qpa_results()

        best = result.get("best")
        if best:
            self.statusBar().showMessage(
                f"Best local-library candidate: {best['reference_name']} — "
                f"score {best['score']:.2f}, {best['matched_count']} matched peak(s)."
            )
        else:
            self.statusBar().showMessage("No phase candidate was ranked.")

    def clear_phase_identification_for_selected(self):
        dataset = self.selected_dataset()
        if dataset is None:
            return
        self.phase_identification_results.pop(dataset.uid, None)
        self.qpa_results.pop(dataset.uid, None)
        self._qpa_setup_signature = None
        self.populate_phase_identification_results()
        self.populate_qpa_results()
        self.statusBar().showMessage(
            f"Cleared phase-identification results for {dataset.name}."
        )

    def show_phase_identification_results(self):
        self.tabs.setCurrentWidget(self.phase_tab)

    @staticmethod
    def _phase_shift_text(value):
        if isinstance(value, list):
            return ", ".join(f"{float(item):.5g}°" for item in value)
        try:
            return f"{float(value):.5g}°"
        except (TypeError, ValueError):
            return "—"

    def populate_phase_identification_results(self):
        if not hasattr(self, "phase_library_table"):
            return

        references = self._phase_reference_library()
        self.phase_library_table.setRowCount(len(references))
        for row_index, reference in enumerate(references):
            values = [
                reference.name,
                reference.formula or "—",
                reference.source,
                str(len(reference.peaks)),
                self._format_optional(reference.wavelength_angstrom, 7),
                "Active CIF" if reference.uid == "active-cif-reference" else "Imported reference",
            ]
            for column, value in enumerate(values):
                self.phase_library_table.setItem(
                    row_index, column, QTableWidgetItem(str(value))
                )

        dataset = self.selected_dataset()
        result = (
            None
            if dataset is None
            else self.phase_identification_results.get(dataset.uid)
        )
        results = [] if result is None else result.get("results", [])

        self.phase_candidate_table.blockSignals(True)
        self.phase_candidate_table.setRowCount(len(results))
        for row_index, candidate in enumerate(results):
            correlation = candidate.get("intensity_correlation")
            values = [
                str(candidate.get("rank", row_index + 1)),
                candidate.get("reference_name", "—"),
                candidate.get("formula") or "—",
                f"{float(candidate.get('score', 0.0)):.5g}",
                str(candidate.get("matched_count", 0)),
                self._format_optional(
                    candidate.get("observed_coverage_percent"), 5, "%"
                ),
                self._format_optional(
                    candidate.get("reference_coverage_percent"), 5, "%"
                ),
                self._format_optional(
                    candidate.get("mean_absolute_delta_deg"), 6, "°"
                ),
                self._phase_shift_text(candidate.get("zero_shift_deg")),
                self._format_optional(correlation, 5),
                (
                    f"{candidate.get('phase_count', 1)}-phase mixture"
                    if candidate.get("kind") == "mixture"
                    else "Single phase"
                ),
                "; ".join(candidate.get("notes", [])) or "—",
            ]
            for column, value in enumerate(values):
                self.phase_candidate_table.setItem(
                    row_index, column, QTableWidgetItem(str(value))
                )
        self.phase_candidate_table.blockSignals(False)

        if result is None:
            self.phase_summary_label.setText(
                f"Local reference library: {len(references)} entry/entries. Run Phase 5 search on an experimental dataset."
            )
            self.phase_match_table.setRowCount(0)
            self.phase_identification_plot.set_result(None)
            return

        best = result.get("best")
        if best:
            self.phase_summary_label.setText(
                f"{result.get('dataset_name', dataset.name)} — best candidate: "
                f"{best['reference_name']}; score {best['score']:.3f}; "
                f"{best['matched_count']}/{result['observed_peak_count']} observed peaks matched. "
                "Review unmatched peaks and alternative candidates before assigning phases."
            )
        else:
            self.phase_summary_label.setText("No candidate was ranked.")

        if results:
            self.phase_candidate_table.selectRow(0)
            self.update_selected_phase_candidate()
        else:
            self.phase_match_table.setRowCount(0)
            self.phase_identification_plot.set_result(None)

    def update_selected_phase_candidate(self):
        dataset = self.selected_dataset()
        if dataset is None:
            return
        result = self.phase_identification_results.get(dataset.uid)
        if not result:
            return
        row = self.phase_candidate_table.currentRow()
        candidates = result.get("results", [])
        if not 0 <= row < len(candidates):
            row = 0
        if not candidates:
            return
        candidate = candidates[row]
        matches = candidate.get("matches", [])
        self.phase_match_table.setRowCount(len(matches))
        for row_index, match in enumerate(matches):
            values = [
                self._format_optional(match.get("observed_2theta"), 7),
                self._format_optional(match.get("observed_intensity"), 6),
                match.get("reference_name") or candidate.get("reference_name", "—"),
                self._format_optional(match.get("reference_2theta"), 7),
                self._format_optional(match.get("shifted_reference_2theta"), 7),
                match.get("hkl_label") or "—",
                self._format_optional(match.get("reference_intensity"), 6),
                self._format_optional(match.get("delta_deg"), 6, "°"),
                "Matched",
            ]
            for column, value in enumerate(values):
                self.phase_match_table.setItem(
                    row_index, column, QTableWidgetItem(str(value))
                )
        self.phase_identification_plot.set_result(candidate)
        self.populate_qpa_phase_setup()
        self.populate_qpa_results(update_setup=False)

    def export_phase_identification_csv(self):
        dataset = self.selected_dataset()
        if dataset is None:
            return
        result = self.phase_identification_results.get(dataset.uid)
        if not result:
            self.statusBar().showMessage("There are no phase-identification results to export.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export phase-identification results", f"{dataset.name}_phase_identification.txt", "Text data (*.txt)"
        )
        if not filename:
            return
        candidate_rows, match_rows = [], []
        for candidate in result.get("results", []):
            candidate_rows.append({
                "dataset": dataset.name, "dataset_uid": dataset.uid,
                "rank": candidate.get("rank"), "candidate": candidate.get("reference_name"),
                "formula": candidate.get("formula"), "kind": candidate.get("kind"),
                "phase_count": candidate.get("phase_count"), "score": candidate.get("score"),
                "matched_count": candidate.get("matched_count"),
                "observed_coverage_percent": candidate.get("observed_coverage_percent"),
                "reference_coverage_percent": candidate.get("reference_coverage_percent"),
                "mean_absolute_delta_deg": candidate.get("mean_absolute_delta_deg"),
                "zero_shift_deg": candidate.get("zero_shift_deg"),
                "intensity_correlation": candidate.get("intensity_correlation"),
                "notes": candidate.get("notes", []),
            })
            for match in candidate.get("matches", []):
                match_rows.append({"dataset": dataset.name, "candidate_rank": candidate.get("rank"), "candidate": candidate.get("reference_name"), **match})
        candidate_path = Path(filename).with_suffix(".txt")
        match_path = candidate_path.with_name(candidate_path.stem + "_matched_peaks.txt")
        try:
            write_table_txt(candidate_path, candidate_rows, title=f"Phase identification ranking — {dataset.name}", metadata={"dataset_uid": dataset.uid})
            write_table_txt(match_path, match_rows, title=f"Phase identification matched peaks — {dataset.name}", metadata={"dataset_uid": dataset.uid})
        except Exception as exc:
            QMessageBox.critical(self, "TXT export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported clean TXT: {candidate_path.name}, {match_path.name}.")

    def _selected_phase_candidate_for_qpa(self):
        dataset = self.selected_dataset()
        if dataset is None:
            return None, None
        result = self.phase_identification_results.get(dataset.uid)
        if not result:
            return dataset, None
        candidates = result.get("results", [])
        if not candidates:
            return dataset, None
        row = self.phase_candidate_table.currentRow()
        if not 0 <= row < len(candidates):
            row = 0
        return dataset, candidates[row]

    def populate_qpa_phase_setup(self, force: bool = False):
        if not hasattr(self, "qpa_setup_table"):
            return
        dataset, candidate = self._selected_phase_candidate_for_qpa()
        if dataset is None or candidate is None:
            self.qpa_setup_table.blockSignals(True)
            self.qpa_setup_table.setRowCount(0)
            self.qpa_setup_table.blockSignals(False)
            self.qpa_internal_standard_combo.clear()
            self._qpa_setup_signature = None
            return

        reference_uids = list(candidate.get("reference_uids", []))
        signature = (
            dataset.uid,
            candidate.get("rank"),
            tuple(reference_uids),
        )
        if signature == self._qpa_setup_signature and not force:
            return
        if (
            self._qpa_setup_signature is not None
            and signature != self._qpa_setup_signature
        ):
            self.qpa_results.pop(dataset.uid, None)

        references = {
            reference.uid: reference
            for reference in self._phase_reference_library()
        }
        shifts = candidate.get("zero_shift_deg", 0.0)
        if not isinstance(shifts, list):
            shifts = [shifts] * len(reference_uids)
        elif len(shifts) < len(reference_uids):
            shifts = list(shifts) + [0.0] * (
                len(reference_uids) - len(shifts)
            )

        persisted_result = self.qpa_results.get(dataset.uid, {})
        persisted = {
            row.get("reference_uid"): row
            for row in persisted_result.get("phases", [])
        }

        self.qpa_setup_table.blockSignals(True)
        self.qpa_setup_table.setRowCount(len(reference_uids))
        self.qpa_internal_standard_combo.blockSignals(True)
        self.qpa_internal_standard_combo.clear()

        for row_index, reference_uid in enumerate(reference_uids):
            reference = references.get(reference_uid)
            if reference is None:
                continue
            saved = persisted.get(reference_uid, {})

            include_item = QTableWidgetItem("Use")
            include_item.setFlags(
                Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
                | Qt.ItemIsUserCheckable
            )
            include_item.setCheckState(Qt.Checked)
            include_item.setData(Qt.UserRole, reference_uid)

            phase_item = QTableWidgetItem(reference.name)
            phase_item.setData(Qt.UserRole, reference_uid)
            phase_item.setFlags(
                phase_item.flags() & ~Qt.ItemIsEditable
            )

            formula_item = QTableWidgetItem(reference.formula or "—")
            formula_item.setFlags(
                formula_item.flags() & ~Qt.ItemIsEditable
            )

            metadata_rir = reference.metadata.get("rir")
            rir_value = saved.get("rir", metadata_rir)
            rir_item = QTableWidgetItem(
                "" if rir_value is None else f"{float(rir_value):.8g}"
            )

            shift_value = saved.get(
                "zero_shift_deg",
                shifts[row_index] if row_index < len(shifts) else 0.0,
            )
            shift_item = QTableWidgetItem(f"{float(shift_value):.8g}")

            for column, item in enumerate(
                (
                    include_item,
                    phase_item,
                    formula_item,
                    rir_item,
                    shift_item,
                )
            ):
                self.qpa_setup_table.setItem(row_index, column, item)

            self.qpa_internal_standard_combo.addItem(
                reference.name,
                reference_uid,
            )

        saved_standard_uid = persisted_result.get(
            "internal_standard_uid"
        )
        if saved_standard_uid:
            index = self.qpa_internal_standard_combo.findData(
                saved_standard_uid
            )
            if index >= 0:
                self.qpa_internal_standard_combo.setCurrentIndex(index)

        self.qpa_internal_standard_combo.blockSignals(False)
        self.qpa_setup_table.blockSignals(False)
        self._qpa_setup_signature = signature

    def _qpa_setup_item_changed(self, item):
        dataset = self.selected_dataset()
        if dataset is None:
            return
        if item.column() in (0, 3, 4):
            self.qpa_results.pop(dataset.uid, None)
            self.populate_qpa_results(update_setup=False)
            self.statusBar().showMessage(
                "QPA phase settings changed; run Phase 6 again."
            )

    def _qpa_phase_specs_from_table(self):
        references = {
            reference.uid: reference
            for reference in self._phase_reference_library()
        }
        specs = []
        for row in range(self.qpa_setup_table.rowCount()):
            include_item = self.qpa_setup_table.item(row, 0)
            phase_item = self.qpa_setup_table.item(row, 1)
            if (
                include_item is None
                or phase_item is None
                or include_item.checkState() != Qt.Checked
            ):
                continue
            reference_uid = phase_item.data(Qt.UserRole)
            reference = references.get(reference_uid)
            if reference is None:
                continue

            rir_text = (
                self.qpa_setup_table.item(row, 3).text().strip()
                if self.qpa_setup_table.item(row, 3) is not None
                else ""
            )
            shift_text = (
                self.qpa_setup_table.item(row, 4).text().strip()
                if self.qpa_setup_table.item(row, 4) is not None
                else "0"
            )
            try:
                rir_override = float(rir_text) if rir_text else None
            except ValueError as exc:
                raise ValueError(
                    f"Invalid RIR value for {reference.name}: {rir_text!r}."
                ) from exc
            try:
                shift = float(shift_text) if shift_text else 0.0
            except ValueError as exc:
                raise ValueError(
                    f"Invalid zero shift for {reference.name}: {shift_text!r}."
                ) from exc
            specs.append(
                QPAPhaseSpec(
                    reference=reference,
                    zero_shift_deg=shift,
                    rir_override=rir_override,
                )
            )
        return specs

    def quantify_selected_phases(self):
        dataset = self.selected_dataset()
        if dataset is None:
            self.statusBar().showMessage(
                "Select an experimental dataset before Phase 6 QPA."
            )
            return
        if dataset.metadata.get("analysis_role") == "reference_pattern":
            self.statusBar().showMessage(
                "Select an experimental pattern, not a reference pattern."
            )
            return

        self.populate_qpa_phase_setup()
        try:
            phase_specs = self._qpa_phase_specs_from_table()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid QPA phase setup", str(exc))
            return
        if not phase_specs:
            self.statusBar().showMessage(
                "No phases are included in the QPA phase setup."
            )
            return

        internal_standard_uid = None
        if self.qpa_internal_standard_check.isChecked():
            internal_standard_uid = self.qpa_internal_standard_combo.currentData()
            if not internal_standard_uid:
                self.statusBar().showMessage(
                    "Choose an included internal-standard phase."
                )
                return

        y_values = (
            dataset.y
            if self.qpa_use_processed_check.isChecked()
            else dataset.y_raw
        )
        try:
            result = quantify_phases(
                dataset.x,
                y_values,
                phase_specs,
                target_wavelength_angstrom=self.wavelength_angstrom.value(),
                mode=self.qpa_mode.currentText(),
                convert_from_d=self.qpa_convert_d_check.isChecked(),
                reference_intensity_cutoff_percent=(
                    self.qpa_reference_cutoff.value()
                ),
                reference_fwhm_deg=self.qpa_reference_fwhm.value(),
                pseudo_voigt_eta=self.qpa_profile_eta.value(),
                baseline_order=self.qpa_baseline_order.value(),
                weighting=self.qpa_weighting.currentText(),
                bootstrap_samples=self.qpa_bootstrap_samples.value(),
                internal_standard_uid=internal_standard_uid,
                known_internal_standard_percent=(
                    self.qpa_known_standard_percent.value()
                    if internal_standard_uid
                    else None
                ),
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Quantitative phase analysis failed",
                str(exc),
            )
            return

        _, candidate = self._selected_phase_candidate_for_qpa()
        result["dataset_name"] = dataset.name
        result["used_processed_pattern"] = (
            self.qpa_use_processed_check.isChecked()
        )
        result["phase5_candidate"] = (
            None if candidate is None else candidate.get("reference_name")
        )
        self.qpa_results[dataset.uid] = result
        self._record_scientific_result(
            "qpa",
            dataset.uid,
            {"exploratory": result},
            reason="Completed exploratory quantitative analysis",
        )
        self.populate_qpa_results(update_setup=False)

        diagnostics = result["diagnostics"]
        self.statusBar().showMessage(
            f"Phase 6 completed for {dataset.name}: "
            f"{len(result['phases'])} phase(s), "
            f"weighted residual {diagnostics['weighted_profile_residual_percent']:.4g}%, "
            f"R²={diagnostics['r_squared']:.6f}."
        )

    def clear_qpa_for_selected(self):
        dataset = self.selected_dataset()
        if dataset is None:
            return
        self.qpa_results.pop(dataset.uid, None)
        self.populate_qpa_results(update_setup=False)
        self.statusBar().showMessage(
            f"Cleared QPA results for {dataset.name}."
        )

    def show_qpa_results(self):
        self.tabs.setCurrentWidget(self.qpa_tab)

    def populate_qpa_results(self, update_setup: bool = True):
        if not hasattr(self, "qpa_fraction_table"):
            return
        if update_setup:
            self.populate_qpa_phase_setup()

        dataset = self.selected_dataset()
        result = None if dataset is None else self.qpa_results.get(dataset.uid)
        if not result:
            self.qpa_fraction_table.setRowCount(0)
            self.qpa_plot.set_result(None)
            self.qpa_r2_value.setText("—")
            self.qpa_rmse_value.setText("—")
            self.qpa_profile_residual_value.setText("—")
            self.qpa_weighted_residual_value.setText("—")
            self.qpa_correlation_value.setText("—")
            self.qpa_bootstrap_value.setText("—")
            self.qpa_amorphous_value.setText("—")
            self.qpa_warning_value.setText("—")
            if dataset is None:
                self.qpa_summary_label.setText(
                    "Select an experimental dataset and a Phase 5 candidate."
                )
            else:
                self.qpa_summary_label.setText(
                    "Review the selected Phase 5 phase set and run Phase 6 quantification."
                )
            return

        phase_rows = list(result.get("phases", []))
        amorphous = result.get("amorphous_percent")
        table_rows = len(phase_rows) + (1 if amorphous is not None else 0)
        self.qpa_fraction_table.setRowCount(table_rows)
        for row_index, phase in enumerate(phase_rows):
            status = []
            if phase.get("rir") is None:
                status.append("RIR missing")
            if float(phase.get("scale_coefficient", 0.0)) <= 1e-10:
                status.append("At lower bound")
            values = [
                phase.get("reference_name", "—"),
                phase.get("formula") or "—",
                self._format_optional(phase.get("scale_coefficient"), 7),
                self._format_optional(phase.get("rir"), 7),
                self._format_optional(
                    phase.get("scale_fraction_percent"), 6, "%"
                ),
                self._format_optional(
                    phase.get("weight_fraction_percent"), 6, "%"
                ),
                self._format_optional(
                    phase.get("weight_fraction_uncertainty_percent"), 4, "%"
                ),
                self._format_optional(
                    phase.get("corrected_weight_percent"), 6, "%"
                ),
                self._format_optional(
                    phase.get("corrected_uncertainty_percent"), 4, "%"
                ),
                str(phase.get("prepared_peak_count", "—")),
                "; ".join(status) or "OK",
            ]
            for column, value in enumerate(values):
                self.qpa_fraction_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(str(value)),
                )

        if amorphous is not None:
            row_index = len(phase_rows)
            values = [
                "Amorphous / unmodelled",
                "—", "—", "—", "—", "—", "—",
                self._format_optional(amorphous, 6, "%"),
                self._format_optional(
                    result.get("amorphous_uncertainty_percent"), 4, "%"
                ),
                "—",
                "Internal-standard estimate",
            ]
            for column, value in enumerate(values):
                self.qpa_fraction_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(str(value)),
                )

        diagnostics = result.get("diagnostics", {})
        self.qpa_plot.set_result(result)
        self.qpa_r2_value.setText(
            self._format_optional(diagnostics.get("r_squared"), 8)
        )
        self.qpa_rmse_value.setText(
            self._format_optional(diagnostics.get("rmse"), 8)
        )
        self.qpa_profile_residual_value.setText(
            self._format_optional(
                diagnostics.get("profile_residual_percent"), 6, "%"
            )
        )
        self.qpa_weighted_residual_value.setText(
            self._format_optional(
                diagnostics.get("weighted_profile_residual_percent"), 6, "%"
            )
        )
        self.qpa_correlation_value.setText(
            self._format_optional(
                diagnostics.get("maximum_basis_correlation"), 7
            )
        )
        self.qpa_bootstrap_value.setText(
            f"{result.get('bootstrap_samples_successful', 0)} / "
            f"{result.get('bootstrap_samples_requested', 0)} successful"
        )
        self.qpa_amorphous_value.setText(
            self._format_optional(amorphous, 6, "%")
        )
        self.qpa_warning_value.setText(
            "; ".join(result.get("warnings", [])) or "None"
        )
        mode_label = (
            "semi-quantitative RIR weight fractions"
            if result.get("mode") == "RIR-corrected weight fractions"
            else "relative pattern scale fractions"
        )
        self.qpa_summary_label.setText(
            f"{result.get('dataset_name', dataset.name)} — {mode_label}; "
            f"{len(phase_rows)} phase(s); candidate: "
            f"{result.get('phase5_candidate') or 'manual phase set'}."
        )

    def export_qpa_csv(self):
        dataset = self.selected_dataset()
        if dataset is None:
            return
        result = self.qpa_results.get(dataset.uid)
        if not result:
            self.statusBar().showMessage("There are no QPA results to export.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export quantitative phase analysis", f"{dataset.name}_qpa.txt", "Text data (*.txt)"
        )
        if not filename:
            return
        phase_path = Path(filename).with_suffix(".txt")
        profile_path = phase_path.with_name(phase_path.stem + "_profile.txt")
        diagnostic_path = phase_path.with_name(phase_path.stem + "_diagnostics.txt")
        phase_rows = [{"dataset": dataset.name, "dataset_uid": dataset.uid, "mode": result.get("mode"), **phase} for phase in result.get("phases", [])]
        if result.get("amorphous_percent") is not None:
            phase_rows.append({
                "dataset": dataset.name, "dataset_uid": dataset.uid,
                "reference_name": "Amorphous / unmodelled",
                "corrected_weight_percent": result.get("amorphous_percent"),
                "corrected_uncertainty_percent": result.get("amorphous_uncertainty_percent"),
            })
        plot = result.get("plot", {})
        profile_data = {
            "two_theta_deg": plot.get("x", []), "observed": plot.get("observed", []),
            "calculated": plot.get("calculated", []), "baseline": plot.get("baseline", []),
            "residual": plot.get("residual", []),
        }
        used_names = set(profile_data)
        for index, contribution in enumerate(plot.get("contributions", []), start=1):
            raw_name = str(contribution.get("reference_name", f"phase_{index}"))
            key = "phase_" + "_".join(raw_name.split())
            base, suffix = key, 2
            while key in used_names:
                key = f"{base}_{suffix}"; suffix += 1
            used_names.add(key)
            profile_data[key] = contribution.get("y", [])
        diagnostics = {
            **result.get("diagnostics", {}), "mode": result.get("mode"),
            "target_wavelength_angstrom": result.get("target_wavelength_angstrom"),
            "reference_fwhm_deg": result.get("reference_fwhm_deg"),
            "baseline_order": result.get("baseline_order"), "weighting": result.get("weighting"),
            "bootstrap_samples_successful": result.get("bootstrap_samples_successful"),
            "amorphous_percent": result.get("amorphous_percent"), "warnings": result.get("warnings", []),
        }
        try:
            write_table_txt(phase_path, phase_rows, title=f"Quantitative phase analysis — {dataset.name}", metadata={"dataset_uid": dataset.uid})
            write_columns_txt(profile_path, profile_data, title=f"QPA fitted profile — {dataset.name}", metadata={"dataset_uid": dataset.uid})
            write_table_txt(diagnostic_path, [diagnostics], title=f"QPA diagnostics — {dataset.name}", metadata={"dataset_uid": dataset.uid})
        except Exception as exc:
            QMessageBox.critical(self, "TXT export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported QPA TXT files: {phase_path.name}, {profile_path.name}, {diagnostic_path.name}.")
