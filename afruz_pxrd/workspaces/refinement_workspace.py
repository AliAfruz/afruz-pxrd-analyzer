from __future__ import annotations

from ..main_window_dependencies import *
from .base import WorkspaceAdapter


class RefinementWorkspace(WorkspaceAdapter):
    """Independently testable refinement workspace adapter."""


class RefinementWorkspaceMixin:
    def import_cif_reference(self, checked=False):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import CIF reference structure",
            "",
            "Crystallographic Information File (*.cif);;All files (*)",
        )
        if not filename:
            return

        return self._import_cif_reference_path(filename)

    def _import_cif_reference_path(self, filename) -> bool:
        filename = str(Path(filename))
        try:
            structure = load_cif(filename)
        except CIFImportError as exc:
            QMessageBox.critical(self, "CIF import failed", str(exc))
            return False

        self.reference_structure = structure
        self.reference_pattern = []
        self.cell_refinement_results.clear()
        self.crystal_system_selector.setCurrentText(
            structure.get("crystal_system", "Triclinic")
        )
        self.calculate_reference_pattern()
        self.statusBar().showMessage(
            f"Imported CIF reference {Path(filename).name}: "
            f"{structure['expanded_atom_count']} expanded atom(s)."
        )
        return True

    def calculate_reference_pattern(self):
        if self.reference_structure is None:
            self.statusBar().showMessage(
                "Import a CIF reference before calculating a pattern."
            )
            return

        ds = self.selected_dataset()
        if ds is not None:
            two_theta_min = max(0.0, float(np.min(ds.x)))
            two_theta_max = min(179.0, float(np.max(ds.x)))
        else:
            two_theta_min, two_theta_max = 5.0, 90.0

        try:
            pattern = calculate_powder_pattern(
                self.reference_structure,
                wavelength_angstrom=self.wavelength_angstrom.value(),
                two_theta_min=two_theta_min,
                two_theta_max=two_theta_max,
                intensity_cutoff_percent=self.reference_cutoff.value(),
                use_gpu=len(self.reference_structure.get("atoms", [])) > 2000,
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Reference pattern calculation failed",
                str(exc),
            )
            return

        self.reference_pattern = pattern
        self.cell_refinement_results.clear()
        self.populate_crystal_results()
        self.redraw()
        self.statusBar().showMessage(
            f"Calculated {len(pattern)} CIF reference peak(s) from "
            f"{two_theta_min:.4g}° to {two_theta_max:.4g}° 2θ."
        )

    def clear_cif_reference(self):
        self.reference_structure = None
        self.reference_pattern = []
        self.cell_refinement_results.clear()
        self.populate_crystal_results()
        self.redraw()
        self.statusBar().showMessage("Cleared CIF reference and cell refinements.")

    def _observed_peaks_for_cell_refinement(self, ds):
        groups = self.fit_groups.get(ds.uid, [])
        fitted = []
        for group in groups:
            for component in group.get("components", []):
                fitted.append(
                    {
                        "position": float(component["center"]),
                        "position_error": component.get("center_error"),
                        "source": "Fitted",
                    }
                )
        if fitted:
            return sorted(fitted, key=lambda row: row["position"])

        detected = self._active_peak_rows_for_uid(ds.uid)
        return [
            {
                "position": float(peak["position"]),
                "position_error": None,
                "source": peak.get("method", "Detected"),
            }
            for peak in detected
        ]

    def match_and_refine_cell_for_selected(self):
        ds = self.selected_dataset()
        if ds is None:
            self.statusBar().showMessage(
                "Select a dataset before matching CIF peaks."
            )
            return
        if self.reference_structure is None:
            self.statusBar().showMessage(
                "Import a CIF reference before matching peaks."
            )
            return
        if not self.reference_pattern:
            self.calculate_reference_pattern()
            if not self.reference_pattern:
                return

        observed = self._observed_peaks_for_cell_refinement(ds)
        if not observed and self.peak_rows.get(ds.uid):
            self.statusBar().showMessage(
                "The curated peak list contains no included peaks for cell refinement."
            )
            return
        if not observed:
            rows, _ = smart_detect_peaks(
                ds.x,
                ds.y,
                sensitivity=self.smart_sensitivity.currentText(),
            )
            self._commit_main_peak_rows(
                ds.uid,
                rows,
                reason="Automatic peak search for cell refinement",
            )
            observed = self._observed_peaks_for_cell_refinement(ds)
        if not observed:
            self.statusBar().showMessage(
                "No observed peaks are available for CIF matching."
            )
            return

        matches = match_observed_to_reference(
            observed,
            self.reference_pattern,
            tolerance_deg=self.match_tolerance.value(),
        )
        try:
            result = refine_unit_cell(
                matches,
                initial_cell=self.reference_structure["cell"],
                crystal_system=self.crystal_system_selector.currentText(),
                wavelength_angstrom=self.wavelength_angstrom.value(),
                refine_zero_shift=self.refine_zero_shift_check.isChecked(),
                instrument_profile=deepcopy(
                    getattr(self, "active_instrument_profile", None)
                ),
                apply_instrument_position_correction=bool(
                    getattr(self, "active_instrument_profile", None)
                ),
                allow_profile_extrapolation=False,
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Cell refinement unavailable",
                f"Matched {len(matches)} peak(s).\n\n{exc}",
            )
            partial_result = {
                "success": False,
                "message": str(exc),
                "crystal_system": self.crystal_system_selector.currentText(),
                "wavelength_angstrom": self.wavelength_angstrom.value(),
                "initial_cell": self.reference_structure["cell"],
                "refined_cell": self.reference_structure["cell"],
                "cell_errors": {},
                "zero_shift_deg": 0.0,
                "zero_shift_error_deg": None,
                "match_count": len(matches),
                "weighted": False,
                "r_squared": None,
                "rmse_deg": None,
                "maximum_absolute_residual_deg": None,
                "match_score_percent": None,
                "matches": [
                    {
                        **match,
                        "initial_calculated_2theta": match["reference_2theta"],
                        "refined_calculated_2theta": match["reference_2theta"],
                        "refined_delta": match["initial_delta"],
                    }
                    for match in matches
                ],
            }
            partial_result["dataset_name"] = ds.name
            self.cell_refinement_results[ds.uid] = partial_result
            self._record_scientific_result(
                "phase",
                ds.uid,
                {"cell_refinement": partial_result},
                reason="Recorded partial unit-cell matching result",
            )
            self.populate_crystal_results()
            return

        result["dataset_name"] = ds.name
        self.cell_refinement_results[ds.uid] = result
        self._record_scientific_result(
            "phase",
            ds.uid,
            {"cell_refinement": result},
            reason="Completed unit-cell refinement",
        )
        self.populate_crystal_results()
        self.statusBar().showMessage(
            f"Refined {result['crystal_system']} cell using "
            f"{result['match_count']} matched peak(s): "
            f"RMSE={result['rmse_deg']:.6g}° 2θ."
        )

    def show_crystal_results(self):
        self.tabs.setCurrentWidget(self.crystal_tab)

    @staticmethod
    def _format_cell(cell: dict | None, errors: dict | None = None) -> str:
        if not cell:
            return "—"
        errors = errors or {}
        parts = []
        for name in ("a", "b", "c"):
            value = float(cell[name])
            error = errors.get(name)
            if error is None:
                parts.append(f"{name}={value:.7g} Å")
            else:
                parts.append(f"{name}={value:.7g}±{float(error):.2g} Å")
        for name in ("alpha", "beta", "gamma"):
            symbol = {"alpha": "α", "beta": "β", "gamma": "γ"}[name]
            value = float(cell[name])
            error = errors.get(name)
            if error is None:
                parts.append(f"{symbol}={value:.7g}°")
            else:
                parts.append(f"{symbol}={value:.7g}±{float(error):.2g}°")
        return "; ".join(parts)

    def populate_crystal_results(self):
        structure = self.reference_structure
        if structure is None:
            self.crystal_summary_label.setText(
                "Import a CIF reference from the Phase 4 controls."
            )
            self.cif_name_value.setText("—")
            self.cif_formula_value.setText("—")
            self.cif_space_group_value.setText("—")
            self.cif_system_value.setText("—")
            self.cif_cell_value.setText("—")
            self.cif_atoms_value.setText("—")
            self.reference_table.setRowCount(0)
        else:
            self.crystal_summary_label.setText(
                f"{structure.get('data_name', 'CIF structure')} — "
                f"{len(self.reference_pattern)} calculated reference peak(s); "
                f"λ={self.wavelength_angstrom.value():.6g} Å. "
                + (
                    "CIF Cromer–Mann X-ray factors active"
                    + (
                        " (" + ", ".join(sorted(structure["xray_scattering_factors"])) + ")."
                        if structure.get("xray_scattering_factors")
                        else "."
                    )
                    if structure.get("xray_scattering_factors")
                    else "CIF has no Cromer–Mann table; atomic-number approximation active."
                )
                + (
                    f" CIF refinement metadata reports λ="
                    f"{float(structure['reported_wavelength_angstrom']):.7g} Å; "
                    "peak positions only match external software when wavelengths match."
                    if structure.get("reported_wavelength_angstrom") is not None
                    and abs(
                        float(structure["reported_wavelength_angstrom"])
                        - self.wavelength_angstrom.value()
                    ) > 1e-7
                    else ""
                )
            )
            self.cif_name_value.setText(
                structure.get("data_name", "CIF structure")
            )
            self.cif_formula_value.setText(
                structure.get("formula") or "—"
            )
            self.cif_space_group_value.setText(
                structure.get("space_group", "Unknown")
            )
            self.cif_system_value.setText(
                structure.get("crystal_system", "Unknown")
            )
            self.cif_cell_value.setText(
                self._format_cell(structure.get("cell"))
            )
            self.cif_atoms_value.setText(
                f"{structure.get('expanded_atom_count', 0)} "
                f"(asymmetric: {structure.get('asymmetric_atom_count', 0)})"
            )

            self.reference_table.setRowCount(len(self.reference_pattern))
            for row_index, peak in enumerate(self.reference_pattern):
                equivalent = ", ".join(
                    f"({hkl[0]} {hkl[1]} {hkl[2]})"
                    for hkl in peak.get("equivalent_hkls", [])[:8]
                )
                values = [
                    f"{float(peak['two_theta']):.7g}",
                    f"{float(peak['d_spacing']):.7g}",
                    peak.get("hkl_label", "—"),
                    f"{float(peak['intensity']):.5g}",
                    str(peak.get("multiplicity_count", "—")),
                    equivalent,
                ]
                for column, value in enumerate(values):
                    self.reference_table.setItem(
                        row_index,
                        column,
                        QTableWidgetItem(str(value)),
                    )

        ds = self.selected_dataset()
        result = (
            None if ds is None
            else self.cell_refinement_results.get(ds.uid)
        )
        matches = [] if result is None else result.get("matches", [])
        self.cell_match_table.setRowCount(len(matches))
        for row_index, match in enumerate(matches):
            values = [
                f"{float(match['observed_2theta']):.7g}",
                self._format_optional(match.get("observed_error"), 4),
                match.get("hkl_label", "—"),
                self._format_optional(
                    match.get("initial_calculated_2theta"), 7
                ),
                self._format_optional(match.get("initial_delta"), 6),
                self._format_optional(
                    match.get("refined_calculated_2theta"), 7
                ),
                self._format_optional(match.get("refined_delta"), 6),
                self._format_optional(
                    match.get("reference_intensity"), 5
                ),
                "—" if result is None else result.get("dataset_name", "—"),
                "Refined" if result and result.get("success") else "Matched only",
            ]
            for column, value in enumerate(values):
                self.cell_match_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(str(value)),
                )

        self.cell_refinement_plot.set_result(result)
        if result is None:
            self.refined_cell_value.setText("—")
            self.cell_zero_shift_value.setText("—")
            self.cell_rmse_value.setText("—")
            self.cell_r2_value.setText("—")
            self.cell_match_score_value.setText("—")
            self.cell_weighting_value.setText("—")
            return

        self.refined_cell_value.setText(
            self._format_cell(
                result.get("refined_cell"),
                result.get("cell_errors"),
            )
        )
        self.cell_zero_shift_value.setText(
            self._format_optional(
                result.get("zero_shift_deg"), 7, "°"
            )
        )
        self.cell_rmse_value.setText(
            self._format_optional(result.get("rmse_deg"), 7, "°")
        )
        self.cell_r2_value.setText(
            self._format_optional(result.get("r_squared"), 7)
        )
        self.cell_match_score_value.setText(
            self._format_optional(
                result.get("match_score_percent"), 6, "%"
            )
        )
        self.cell_weighting_value.setText(
            "Fitted peak-center uncertainty"
            if result.get("weighted")
            else "Unweighted"
        )

    def export_reference_peaks_csv(self):
        if not self.reference_pattern:
            self.statusBar().showMessage("There are no CIF reference peaks to export.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export CIF reference peaks", "cif_reference_peaks.txt", "Text data (*.txt)")
        if not filename:
            return
        rows = []
        for peak in self.reference_pattern:
            row = dict(peak)
            row["hkl"] = peak.get("hkl_label")
            row["equivalent_hkls"] = [list(hkl) for hkl in peak.get("equivalent_hkls", [])]
            rows.append(row)
        try:
            result = write_table_txt(filename, rows, title="CIF reference Bragg positions")
        except Exception as exc:
            QMessageBox.critical(self, "TXT export failed", str(exc))
            return
        self.statusBar().showMessage(f"CIF reference peaks TXT exported: {result['txt_path']}")

    def export_cell_refinement_csv(self):
        ds = self.selected_dataset()
        if ds is None:
            return
        result = self.cell_refinement_results.get(ds.uid)
        if not result:
            self.statusBar().showMessage("There is no cell refinement result to export.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export matched and refined peaks", f"{ds.name}_cell_refinement.txt", "Text data (*.txt)")
        if not filename:
            return
        rows = []
        for match in result.get("matches", []):
            rows.append({
                **dict(match), "dataset": ds.name, "dataset_uid": ds.uid,
                "crystal_system": result.get("crystal_system"),
                "wavelength_angstrom": result.get("wavelength_angstrom"),
                "zero_shift_deg": result.get("zero_shift_deg"), "rmse_deg": result.get("rmse_deg"),
                "r_squared": result.get("r_squared"), "match_score_percent": result.get("match_score_percent"),
                **{f"refined_{name}": value for name, value in result.get("refined_cell", {}).items()},
            })
        try:
            exported = write_table_txt(filename, rows, title=f"Cell refinement matched peaks — {ds.name}", metadata={"dataset_uid": ds.uid})
        except Exception as exc:
            QMessageBox.critical(self, "TXT export failed", str(exc))
            return
        self.statusBar().showMessage(f"Cell-refinement TXT exported: {exported['txt_path']}")

    def use_stress_dataset_wavelength(self):
        dataset = self.selected_dataset()
        if dataset is None:
            self.statusBar().showMessage(
                "Select a dataset before reading wavelength metadata."
            )
            return
        value = dataset.metadata.get("wavelength_k_alpha1")
        if value is None:
            self.statusBar().showMessage(
                "The selected dataset has no Kα1 wavelength metadata."
            )
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            self.statusBar().showMessage(
                "The selected dataset wavelength is not numeric."
            )
            return
        self.stress_wavelength.setValue(value)
        self.statusBar().showMessage(
            f"Loaded residual-stress wavelength {value:.6f} Å."
        )

    def build_residual_stress_observations(self):
        observations = []
        missing = []
        target = self.stress_target_two_theta.value()
        half_window = self.stress_search_window.value()
        default_error = self.stress_default_peak_error.value()
        source_mode = self.stress_peak_source.currentText()

        for dataset in self.datasets:
            if not dataset.visible:
                continue
            if dataset.metadata.get("analysis_role") == "reference_pattern":
                continue
            result = extract_peak_observation(
                dataset.x,
                dataset.y,
                target_two_theta_deg=target,
                half_window_deg=half_window,
                source_mode=source_mode,
                fit_groups=self.fit_groups.get(dataset.uid, []),
                detected_peaks=self._active_peak_rows_for_uid(dataset.uid),
                default_error_deg=default_error,
            )
            if result is None:
                missing.append(dataset.name)
                continue
            psi, psi_source = infer_psi_deg(
                dataset.metadata,
                dataset.name,
            )
            observations.append(
                {
                    "observation_id": dataset.uid,
                    "dataset_uid": dataset.uid,
                    "dataset_name": dataset.name,
                    "included": True,
                    "psi_deg": float(psi),
                    "psi_source": psi_source,
                    "two_theta_deg": float(result["two_theta_deg"]),
                    "two_theta_error_deg": float(
                        result["two_theta_error_deg"]
                    ),
                    "source": result["source"],
                    "target_delta_deg": result.get("target_delta_deg"),
                }
            )

        self.residual_stress_observations = observations
        self.residual_stress_result = None
        self.populate_residual_stress_results()
        message = (
            f"Built {len(observations)} ψ observation(s) from visible experimental datasets."
        )
        if missing:
            message += f" No peak found in {len(missing)} dataset(s)."
        self.statusBar().showMessage(message)
        if missing:
            QMessageBox.warning(
                self,
                "Residual-stress peak warnings",
                "No peak was found inside the selected window for:\n\n"
                + "\n".join(missing),
            )

    def _sync_stress_observations_from_table(self):
        observations = []
        for row in range(self.stress_observation_table.rowCount()):
            include_item = self.stress_observation_table.item(row, 0)
            dataset_item = self.stress_observation_table.item(row, 1)
            if dataset_item is None:
                continue
            try:
                psi = float(self.stress_observation_table.item(row, 2).text())
                two_theta = float(
                    self.stress_observation_table.item(row, 4).text()
                )
                error = float(
                    self.stress_observation_table.item(row, 5).text()
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid ψ, peak 2θ, or uncertainty in row {row + 1}."
                ) from exc
            observations.append(
                {
                    "observation_id": dataset_item.data(Qt.UserRole),
                    "dataset_uid": dataset_item.data(Qt.UserRole),
                    "dataset_name": dataset_item.text(),
                    "included": (
                        include_item is not None
                        and include_item.checkState() == Qt.Checked
                    ),
                    "psi_deg": psi,
                    "psi_source": dataset_item.data(Qt.UserRole + 1) or "table",
                    "two_theta_deg": two_theta,
                    "two_theta_error_deg": error,
                    "source": (
                        self.stress_observation_table.item(row, 8).text()
                        if self.stress_observation_table.item(row, 8) is not None
                        else "Manual"
                    ),
                }
            )
        self.residual_stress_observations = observations
        return observations

    def _stress_observation_item_changed(self, item):
        if self._stress_table_updating:
            return
        if item.column() not in (0, 2, 4, 5):
            return
        try:
            self._sync_stress_observations_from_table()
        except ValueError:
            return
        self.residual_stress_result = None
        self.populate_residual_stress_results()
        self.statusBar().showMessage(
            "Residual-stress observations changed; calculate Phase 7 again."
        )

    def calculate_residual_stress(self):
        try:
            observations = self._sync_stress_observations_from_table()
            result = analyze_sin2psi(
                observations,
                wavelength_angstrom=self.stress_wavelength.value(),
                reference_mode=self.stress_reference_mode.currentText(),
                stress_free_two_theta_deg=(
                    self.stress_free_two_theta.value()
                    if self.stress_reference_mode.currentText()
                    == "Use stress-free 2θ"
                    else None
                ),
                elastic_mode=self.stress_elastic_mode.currentText(),
                youngs_modulus_gpa=self.stress_youngs_modulus.value(),
                poisson_ratio=self.stress_poisson_ratio.value(),
                xec_half_s2_per_gpa=self.stress_xec_half_s2.value(),
                regression_mode=self.stress_regression_mode.currentText(),
                azimuth_deg=self.stress_azimuth.value(),
            )
        except (ResidualStressError, ValueError) as exc:
            QMessageBox.critical(
                self,
                "Residual stress analysis failed",
                str(exc),
            )
            return

        self.residual_stress_result = result
        self.populate_residual_stress_results()
        sign = "tensile" if result["stress_mpa"] >= 0 else "compressive"
        self.statusBar().showMessage(
            f"Residual stress: {result['stress_mpa']:.6g} ± "
            f"{result['stress_error_mpa']:.3g} MPa ({sign}); "
            f"R²={result['r_squared']:.6f}."
        )

    def clear_residual_stress(self):
        self.residual_stress_observations = []
        self.residual_stress_result = None
        self.populate_residual_stress_results()
        self.statusBar().showMessage("Cleared Phase 7 residual-stress analysis.")

    def show_residual_stress_results(self):
        self.tabs.setCurrentWidget(self.stress_tab)

    def populate_residual_stress_results(self):
        result = self.residual_stress_result
        result_rows = {
            row.get("observation_id") or row.get("dataset_uid"): row
            for row in (result.get("observations", []) if result else [])
        }

        self._stress_table_updating = True
        self.stress_observation_table.blockSignals(True)
        self.stress_observation_table.setRowCount(
            len(self.residual_stress_observations)
        )
        wavelength = self.stress_wavelength.value()
        for row_index, observation in enumerate(
            self.residual_stress_observations
        ):
            observation_id = observation.get("observation_id") or observation.get(
                "dataset_uid"
            )
            calculated = result_rows.get(observation_id, {})

            include_item = QTableWidgetItem("Use")
            include_item.setFlags(
                Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
                | Qt.ItemIsUserCheckable
            )
            include_item.setCheckState(
                Qt.Checked if observation.get("included", True) else Qt.Unchecked
            )

            dataset_item = QTableWidgetItem(
                str(observation.get("dataset_name", "—"))
            )
            dataset_item.setData(Qt.UserRole, observation_id)
            dataset_item.setData(
                Qt.UserRole + 1,
                observation.get("psi_source", "table"),
            )
            dataset_item.setFlags(dataset_item.flags() & ~Qt.ItemIsEditable)

            psi = float(observation.get("psi_deg", 0.0))
            two_theta = float(observation.get("two_theta_deg", 0.0))
            error = float(observation.get("two_theta_error_deg", 0.01))
            try:
                d_value = float(
                    calculated.get(
                        "d_angstrom",
                        two_theta_to_d(two_theta, wavelength),
                    )
                )
            except Exception:
                d_value = float("nan")
            sin2psi = float(
                calculated.get(
                    "sin2psi",
                    np.sin(np.radians(psi)) ** 2,
                )
            )
            strain = calculated.get("strain")
            residual = calculated.get("residual_microstrain")
            status = calculated.get(
                "status",
                (
                    "ψ inferred: " + observation.get("psi_source", "unknown")
                    if observation.get("psi_source") != "metadata:psi_deg"
                    else "Ready"
                ),
            )

            values = [
                include_item,
                dataset_item,
                QTableWidgetItem(f"{psi:.8g}"),
                QTableWidgetItem(f"{sin2psi:.8g}"),
                QTableWidgetItem(f"{two_theta:.9g}"),
                QTableWidgetItem(f"{error:.6g}"),
                QTableWidgetItem("—" if not np.isfinite(d_value) else f"{d_value:.9g}"),
                QTableWidgetItem(
                    "—" if strain is None else f"{float(strain) * 1e6:.8g}"
                ),
                QTableWidgetItem(str(observation.get("source", "—"))),
                QTableWidgetItem(
                    "—" if residual is None else f"{float(residual):.8g}"
                ),
                QTableWidgetItem(str(status)),
            ]
            for column, table_item in enumerate(values):
                if column not in (0, 2, 4, 5):
                    table_item.setFlags(
                        table_item.flags() & ~Qt.ItemIsEditable
                    )
                self.stress_observation_table.setItem(
                    row_index,
                    column,
                    table_item,
                )

        self.stress_observation_table.blockSignals(False)
        self._stress_table_updating = False
        self.residual_stress_plot.set_result(result)

        if not result:
            self.stress_summary_label.setText(
                "Build observations from multiple ψ-tilt scans, verify every angle and peak center, then calculate stress."
            )
            for label in (
                self.stress_value,
                self.stress_uncertainty_value,
                self.stress_sign_value,
                self.stress_d0_value,
                self.stress_slope_value,
                self.stress_intercept_value,
                self.stress_r2_value,
                self.stress_rmse_value,
                self.stress_points_value,
                self.stress_geometry_value,
                self.stress_warning_value,
            ):
                label.setText("—")
            return

        sign = "Tensile" if result["stress_mpa"] >= 0 else "Compressive"
        self.stress_summary_label.setText(
            f"sin²ψ at φ={result['azimuth_deg']:.6g}°; "
            f"λ={result['wavelength_angstrom']:.6g} Å; "
            f"{result['elastic_mode']}; {result['regression_mode']}."
        )
        self.stress_value.setText(f"{result['stress_mpa']:.8g} MPa")
        self.stress_uncertainty_value.setText(
            f"± {result['stress_error_mpa']:.5g} MPa"
        )
        self.stress_sign_value.setText(sign)
        d0_error = result.get("d0_error_angstrom")
        self.stress_d0_value.setText(
            f"{result['d0_angstrom']:.9g} Å"
            + (
                ""
                if d0_error is None
                else f" ± {float(d0_error):.3g} Å"
            )
        )
        self.stress_slope_value.setText(
            f"{result['slope_strain']:.8g} ± "
            f"{result['slope_strain_error']:.3g} strain/sin²ψ"
        )
        self.stress_intercept_value.setText(
            f"{result['strain_intercept'] * 1e6:.8g} µε"
        )
        self.stress_r2_value.setText(f"{result['r_squared']:.8g}")
        self.stress_rmse_value.setText(
            f"{result['rmse_microstrain']:.8g} µε"
        )
        self.stress_points_value.setText(str(result["point_count"]))
        self.stress_geometry_value.setText(
            f"ψ {result['psi_min_deg']:.6g}° to {result['psi_max_deg']:.6g}°; "
            f"sin²ψ span {result['sin2psi_span']:.6g}"
        )
        self.stress_warning_value.setText(
            "; ".join(result.get("warnings", [])) or "No automatic warning."
        )

    def export_residual_stress_csv(self):
        if not self.residual_stress_result:
            self.statusBar().showMessage("There is no Phase 7 result to export.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export residual-stress analysis", "residual_stress.txt", "Text data (*.txt)")
        if not filename:
            return
        path = Path(filename).with_suffix(".txt")
        result = self.residual_stress_result
        summary = {key: value for key, value in result.items() if key not in ("observations", "pair_splitting")}
        try:
            write_table_txt(path, result.get("observations", []), title="Residual-stress observations")
            write_table_txt(path.with_name(path.stem + "_summary.txt"), [summary], title="Residual-stress summary")
            if result.get("pair_splitting"):
                write_table_txt(path.with_name(path.stem + "_psi_pairs.txt"), result["pair_splitting"], title="Residual-stress ψ pairs")
        except Exception as exc:
            QMessageBox.critical(self, "TXT export failed", str(exc))
            return
        self.statusBar().showMessage("Residual-stress TXT observations, summary and ψ pairs exported.")

    def use_dataset_wavelength(self):
        ds = self.selected_dataset()
        if ds is None:
            self.statusBar().showMessage(
                "Select a dataset before reading wavelength metadata."
            )
            return

        wavelength = ds.metadata.get("wavelength_k_alpha1")
        if wavelength is None:
            self.statusBar().showMessage(
                "The selected dataset has no Kα1 wavelength metadata."
            )
            return

        try:
            wavelength = float(wavelength)
        except (TypeError, ValueError):
            self.statusBar().showMessage(
                "The dataset wavelength metadata is not numeric."
            )
            return

        self.wavelength_angstrom.setValue(wavelength)
        self.statusBar().showMessage(
            f"Loaded Kα1 wavelength {wavelength:.6f} Å from {ds.name}."
        )

    def calculate_size_strain_for_selected(self):
        ds = self.selected_dataset()
        if ds is None:
            self.statusBar().showMessage(
                "Select a dataset before calculating size and strain."
            )
            return

        groups = self.fit_groups.get(ds.uid, [])
        if not groups:
            self.statusBar().showMessage(
                "Fit the detected peaks before calculating size and strain."
            )
            return

        try:
            result = analyze_size_strain(
                groups,
                wavelength_angstrom=self.wavelength_angstrom.value(),
                shape_factor=self.shape_factor.value(),
                instrument_fwhm_deg=self.instrument_fwhm.value(),
                correction_mode=self.instrument_correction.currentText(),
                instrument_profile=self.active_instrument_profile,
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Size and strain calculation failed",
                str(exc),
            )
            return

        result["dataset_name"] = ds.name
        self.size_strain_results[ds.uid] = result
        self.populate_size_strain_results()

        valid_count = result["scherrer_summary"]["valid_peak_count"]
        wh = result["williamson_hall"]
        if wh.get("valid"):
            message = (
                f"Size/strain analysis completed for {ds.name}: "
                f"{valid_count} valid peaks, "
                f"Williamson–Hall D={wh['crystallite_size_nm']:.4g} nm, "
                f"ε={wh['microstrain']:.4g}."
            )
        else:
            message = (
                f"Scherrer analysis completed for {ds.name} with "
                f"{valid_count} valid peaks. Williamson–Hall: {wh.get('note', 'unavailable')}"
            )
        self.statusBar().showMessage(message)

    def clear_size_strain_for_selected(self):
        ds = self.selected_dataset()
        if ds is None:
            return
        self.size_strain_results.pop(ds.uid, None)
        self.populate_size_strain_results()
        self.statusBar().showMessage(
            f"Cleared size and strain results for {ds.name}."
        )

    def show_size_strain_results(self):
        self.tabs.setCurrentWidget(self.size_strain_tab)

    @staticmethod
    def _format_optional(value, digits=6, suffix=""):
        if value is None:
            return "—"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "—"
        if not np.isfinite(numeric):
            return "—"
        return f"{numeric:.{digits}g}{suffix}"

    def populate_size_strain_results(self):
        rows = []
        by_uid = {ds.uid: ds for ds in self.datasets}
        for uid, result in self.size_strain_results.items():
            ds = by_uid.get(uid)
            if ds is None:
                continue
            for peak in result.get("peak_results", []):
                rows.append((ds.name, peak))

        self.size_table.setRowCount(len(rows))
        for row_index, (dataset_name, peak) in enumerate(rows):
            values = [
                dataset_name,
                peak.get("peak_number", "—"),
                peak.get("model", "—"),
                self._format_optional(peak.get("two_theta_deg"), 7),
                self._format_optional(
                    peak.get("observed_fwhm_deg"), 7
                ),
                self._format_optional(
                    peak.get("instrument_fwhm_deg"), 7
                ),
                self._format_optional(
                    peak.get("corrected_fwhm_deg"), 7
                ),
                self._format_optional(
                    peak.get("corrected_fwhm_error_deg"), 4
                ),
                self._format_optional(
                    peak.get("scherrer_size_nm"), 7
                ),
                self._format_optional(
                    peak.get("scherrer_size_error_nm"), 4
                ),
                "Yes" if peak.get("valid") else "No",
                peak.get("note", ""),
            ]
            for column, value in enumerate(values):
                self.size_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(str(value)),
                )

        ds = self.selected_dataset()
        selected_result = (
            None
            if ds is None
            else self.size_strain_results.get(ds.uid)
        )
        self.size_strain_plot.set_result(selected_result)

        if not selected_result:
            self.size_summary_label.setText(
                "Run the Phase 3 analysis from the right-side controls."
            )
            self.scherrer_mean_value.setText("—")
            self.scherrer_median_value.setText("—")
            self.wh_size_value.setText("—")
            self.wh_strain_value.setText("—")
            self.wh_r2_value.setText("—")
            self.wh_weighting_value.setText("—")
            return

        scherrer = selected_result["scherrer_summary"]
        wh = selected_result["williamson_hall"]
        advanced = selected_result.get("advanced_microstructure", {})
        advanced_classification = advanced.get("classification", {})
        warning_count = len(advanced_classification.get("warnings", []))
        self.size_summary_label.setText(
            f"{selected_result.get('dataset_name', ds.name)} — "
            f"λ={selected_result['wavelength_angstrom']:.6g} Å, "
            f"K={selected_result['shape_factor']:.4g}, "
            f"instrument correction: {selected_result['correction_mode']} "
            + (
                f"(profile {selected_result.get('instrument_profile_fingerprint')}). "
                if selected_result.get("instrument_profile_fingerprint")
                else f"({selected_result['instrument_fwhm_deg']:.6g}°). "
            )
            + (
                f"Microstructure status: {advanced_classification.get('status')} "
                f"with {warning_count} warning(s)."
                if advanced_classification
                else ""
            )
        )
        self.scherrer_mean_value.setText(
            self._format_optional(scherrer.get("mean_nm"), 7, " nm")
        )
        self.scherrer_median_value.setText(
            self._format_optional(scherrer.get("median_nm"), 7, " nm")
        )
        self.wh_size_value.setText(
            self._format_optional(
                wh.get("crystallite_size_nm"), 7, " nm"
            )
        )
        self.wh_strain_value.setText(
            self._format_optional(wh.get("microstrain"), 7)
        )
        self.wh_r2_value.setText(
            self._format_optional(wh.get("r_squared"), 7)
        )
        self.wh_weighting_value.setText(
            "Weighted by fitted FWHM uncertainty"
            if wh.get("weighted")
            else "Unweighted"
        )

    def export_size_strain_results_csv(self):
        ds = self.selected_dataset()
        if ds is None:
            return
        result = self.size_strain_results.get(ds.uid)
        if not result:
            self.statusBar().showMessage("There are no size and strain results to export.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export size and strain results", f"{ds.name}_size_strain.txt", "Text data (*.txt)")
        if not filename:
            return
        rows = []
        wh = result["williamson_hall"]
        advanced = result.get("advanced_microstructure", {})
        ellipsoid = advanced.get("anisotropic_size_ellipsoid", {})
        strain = advanced.get("stephens_like_strain", {})
        texture = advanced.get("march_dollase_texture", {})
        for peak in result.get("peak_results", []):
            rows.append({
                **dict(peak), "dataset": ds.name, "dataset_uid": ds.uid,
                "wavelength_angstrom": result["wavelength_angstrom"], "shape_factor": result["shape_factor"],
                "instrument_fwhm_deg": result["instrument_fwhm_deg"], "correction_mode": result["correction_mode"],
                "wh_crystallite_size_nm": wh.get("crystallite_size_nm"), "wh_microstrain": wh.get("microstrain"),
                "wh_r_squared": wh.get("r_squared"), "anisotropic_size_a_nm": ellipsoid.get("size_a_nm"),
                "anisotropic_size_b_nm": ellipsoid.get("size_b_nm"), "anisotropic_size_c_nm": ellipsoid.get("size_c_nm"),
                "anisotropic_size_ratio": ellipsoid.get("anisotropy_ratio"),
                "stephens_like_dominant_term": strain.get("dominant_term"),
                "stephens_like_anisotropy_index": strain.get("anisotropy_index"),
                "march_dollase_r": texture.get("r"), "march_dollase_texture_strength": texture.get("texture_strength"),
            })
        try:
            exported = write_table_txt(filename, rows, title=f"Size and strain analysis — {ds.name}", metadata={"dataset_uid": ds.uid})
        except Exception as exc:
            QMessageBox.critical(self, "TXT export failed", str(exc))
            return
        self.statusBar().showMessage(f"Size/strain TXT exported: {exported['txt_path']}")
