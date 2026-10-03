from __future__ import annotations

from ..contracts import ResultContractService
from ..main_window_dependencies import *
from .base import WorkspaceAdapter


class ValidationWorkspace(WorkspaceAdapter):
    """Independently testable validation/report workspace adapter."""


class ValidationWorkspaceMixin:
    def _report_project_name(self) -> str:
        if self.current_project is not None:
            return self.current_project.stem
        return "Untitled Afruz project"

    def _accepted_results_for_report(self) -> dict:
        accepted: dict[str, dict] = {}
        dataset_names = {dataset.uid: dataset.name for dataset in self.datasets}
        for dataset_uid, state in self.scientific_state.by_dataset.items():
            dataset_name = dataset_names.get(dataset_uid, dataset_uid)
            for stage_key in SCIENTIFIC_STAGE_ORDER:
                node = state.node(stage_key)
                if not node.exists:
                    continue
                if node.is_accepted_current or node.status == "Current":
                    label = f"{dataset_name} — {SCIENTIFIC_STAGE_LABELS.get(stage_key, stage_key)}"
                    accepted[label] = {
                        "status": node.status,
                        "revision": node.revision,
                        "accepted_current": node.is_accepted_current,
                        "result_uid": node.result_uid,
                        "updated_at": node.updated_at,
                        "reason": node.reason,
                        "dependencies": dict(node.dependencies),
                        "metadata": dict(node.metadata),
                    }
        return accepted

    def _warnings_for_report(self) -> list[str]:
        warnings: list[str] = []
        for dataset_uid, state in self.scientific_state.by_dataset.items():
            dataset_name = next((ds.name for ds in self.datasets if ds.uid == dataset_uid), dataset_uid)
            for stage_key in SCIENTIFIC_STAGE_ORDER:
                node = state.node(stage_key)
                if node.status in {"Outdated", "Invalid"}:
                    label = SCIENTIFIC_STAGE_LABELS.get(stage_key, stage_key)
                    warnings.append(
                        f"{dataset_name}: {label} is {node.status.lower()}"
                        + (f" — {node.reason}" if node.reason else "")
                    )
        for uid, rows in self.peak_rows.items():
            if rows and not any(row.get("use", True) for row in rows):
                dataset_name = next((ds.name for ds in self.datasets if ds.uid == uid), uid)
                warnings.append(f"{dataset_name}: peak list contains no included peaks.")
        return warnings

    def export_complete_report_package(self):
        if not self.datasets:
            QMessageBox.information(
                self,
                "Export Complete Report Package",
                "Import at least one dataset before exporting a complete report.",
            )
            return
        directory = QFileDialog.getExistingDirectory(
            self,
            "Choose complete report output folder",
            str(Path.home()),
        )
        if not directory:
            return
        output_root = Path(directory) / f"{safe_project_component(self._report_project_name())}_complete_report"
        try:
            result = self.report_service.build_complete_package(
                output_root,
                project_name=self._report_project_name(),
                datasets=self.datasets,
                scientific_state=self.scientific_state,
                analysis_state=self._analysis_state(),
                accepted_results=self._accepted_results_for_report(),
                warnings=self._warnings_for_report(),
                figures=[
                    {
                        "title": "Main diffraction plot",
                        "description": "Use Export PNG/SVG for the current view; this report records figure metadata and data tables.",
                    }
                ],
                notes="Generated from Afruz Complete Report Package exporter.",
            )
            package = result["package"]
            archive = result["archive"]
        except Exception as exc:
            QMessageBox.critical(self, "Complete report export failed", str(exc))
            return
        QMessageBox.information(
            self,
            "Complete report exported",
            "Complete report package created.\n\n"
            f"Folder: {package['output_dir']}\n"
            f"Archive: {archive['archive_path']}",
        )
        self.statusBar().showMessage("Complete report package exported.")

    def export_png(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export plot", "", "PNG image (*.png)"
        )
        if not filename:
            return
        self.export_service.export_plot_png(
            self.plot_widget.plot.plotItem,
            filename,
        )
        self.statusBar().showMessage("PNG exported.")

    def export_svg(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export plot", "", "SVG vector image (*.svg)"
        )
        if not filename:
            return
        self.export_service.export_plot_svg(
            self.plot_widget.plot.plotItem,
            filename,
        )
        self.statusBar().showMessage("SVG exported.")

    def show_export_center(self, checked=False, preferred_formats=None, preferred_categories=None):
        if not self.datasets:
            QMessageBox.information(
                self,
                "Nothing to export",
                "Import at least one diffraction dataset before opening Clean Data Export.",
            )
            return
        formats = preferred_formats if isinstance(preferred_formats, set) else None
        categories = preferred_categories if isinstance(preferred_categories, set) else None
        dialog = ExportCenterDialog(self, preferred_formats=formats, preferred_categories=categories)
        dialog.exec()

    def build_export_snapshot(self, *, all_datasets: bool = False):
        def export_value_present(value):
            if value is None:
                return False
            if isinstance(value, (dict, list, tuple, set)):
                return bool(value)
            return True

        selected = self.selected_dataset()
        datasets = list(self.datasets) if all_datasets else ([selected] if selected is not None else [])
        validation_widget = getattr(self, "validation_widget", None)
        batch_widget = getattr(self, "batch_widget", None)
        exported = []
        for ds in datasets:
            uid = ds.uid
            self._sync_unified_scientific_state(uid)
            result_contracts = ResultContractService.export_dataset(
                self.project_state.result_contracts,
                uid,
            )
            validation_rows = []
            if validation_widget is not None:
                for row in [
                    *getattr(validation_widget, "audit_results", []),
                    *getattr(validation_widget, "robustness_results", []),
                ]:
                    if not isinstance(row, dict):
                        continue
                    if row.get("dataset_uid") == uid or row.get("dataset_name") == ds.name:
                        validation_rows.append(row)
            background_record = self.background_results.get(uid) or {}
            smoothing_record = self.smoothing_results.get(uid) or {}
            has_preprocessing = bool(
                ds.y_processed is not None or background_record or smoothing_record
            )
            preprocessing = {
                "background": background_record,
                "smoothing": smoothing_record,
                "processed_profile_available": ds.y_processed is not None,
                "background_profile_available": uid in self.backgrounds,
            } if has_preprocessing else None
            preprocessing_profile = []
            background_raw = background_record.get("background") if isinstance(background_record, dict) else []
            background_display = background_record.get("display_background") if isinstance(background_record, dict) else []
            smoothing_source = smoothing_record.get("source") if isinstance(smoothing_record, dict) else []
            smoothing_smoothed = smoothing_record.get("smoothed") if isinstance(smoothing_record, dict) else []
            background_mask = background_record.get("protected_mask") if isinstance(background_record, dict) else []
            smoothing_mask = smoothing_record.get("protected_mask") if isinstance(smoothing_record, dict) else []
            background_raw = [] if background_raw is None else background_raw
            background_display = [] if background_display is None else background_display
            smoothing_source = [] if smoothing_source is None else smoothing_source
            smoothing_smoothed = [] if smoothing_smoothed is None else smoothing_smoothed
            background_mask = [] if background_mask is None else background_mask
            smoothing_mask = [] if smoothing_mask is None else smoothing_mask
            for point_index, two_theta in enumerate(ds.x) if has_preprocessing else []:
                preprocessing_profile.append({
                    "point_index": point_index + 1,
                    "two_theta_deg": float(two_theta),
                    "intensity_raw": float(ds.y_raw[point_index]),
                    "intensity_processed": (
                        float(ds.y_processed[point_index]) if ds.y_processed is not None else None
                    ),
                    "background_raw_units": (
                        background_raw[point_index] if point_index < len(background_raw) else None
                    ),
                    "background_display_units": (
                        background_display[point_index] if point_index < len(background_display) else None
                    ),
                    "smoothing_source": (
                        smoothing_source[point_index] if point_index < len(smoothing_source) else None
                    ),
                    "smoothed_intensity": (
                        smoothing_smoothed[point_index] if point_index < len(smoothing_smoothed) else None
                    ),
                    "background_peak_protected": (
                        bool(background_mask[point_index]) if point_index < len(background_mask) else None
                    ),
                    "smoothing_peak_protected": (
                        bool(smoothing_mask[point_index]) if point_index < len(smoothing_mask) else None
                    ),
                })
            phase_revolution_data = {
                "residual_analysis": getattr(getattr(self, "phase_revolution_widget", None), "residuals_by_uid", {}).get(uid),
                "master_peaks": getattr(getattr(self, "phase_revolution_widget", None), "peaks_by_uid", {}).get(uid),
                "peak_list_metadata": getattr(getattr(self, "phase_revolution_widget", None), "peak_list_meta_by_uid", {}).get(uid),
                "indexing": getattr(getattr(self, "phase_revolution_widget", None), "indexing_by_uid", {}).get(uid),
                "candidate_cells": getattr(getattr(self, "phase_revolution_widget", None), "candidates_by_uid", {}).get(uid),
            }
            if not any(export_value_present(value) for value in phase_revolution_data.values()):
                phase_revolution_data = None
            structure_solution_data = {
                "screening": getattr(getattr(self, "structure_solution_widget", None), "screening_by_uid", {}).get(uid),
                "robustness": getattr(getattr(self, "structure_solution_widget", None), "robustness_by_uid", {}).get(uid),
                "extracted_intensities": getattr(getattr(self, "structure_solution_widget", None), "intensities_by_uid", {}).get(uid),
                "provisional_structure": getattr(getattr(self, "structure_solution_widget", None), "provisional_by_uid", {}).get(uid),
                "solution_package": getattr(getattr(self, "structure_solution_widget", None), "packages_by_uid", {}).get(uid),
            }
            if not any(export_value_present(value) for value in structure_solution_data.values()):
                structure_solution_data = None
            analyses = {
                "result_contracts": result_contracts,
                "metrology": self.project_controller.metrology_service.dataset_summary(uid),
                "preprocessing": preprocessing,
                "preprocessing_profile": preprocessing_profile,
                "peaks": normalize_peak_rows(self.peak_rows.get(uid, [])),
                "peak_list_metadata": self.peak_list_meta_by_uid.get(uid, {}),
                "peak_fits": self.fit_groups.get(uid, []),
                "fit_candidates": self.fit_candidates.get(uid, []),
                "size_strain": self.size_strain_results.get(uid),
                "cell_refinement": self.cell_refinement_results.get(uid),
                "phase_identification": self.phase_identification_results.get(uid),
                "phase_revolution": phase_revolution_data,
                "structure_solution": structure_solution_data,
                "whole_pattern": getattr(getattr(self, "whole_pattern_widget", None), "results_by_uid", {}).get(uid),
                "rietveld": getattr(getattr(self, "rietveld_widget", None), "results_by_uid", {}).get(uid),
                "multicomponent": getattr(getattr(self, "multicomponent_refiner_widget", None), "results_by_uid", {}).get(uid),
                "exploratory_qpa": self.qpa_results.get(uid),
                "validated_qpa": getattr(getattr(self, "validated_qpa_widget", None), "native_results_by_uid", {}).get(uid),
                "validation": {"rows": validation_rows} if validation_rows else None,
            }
            exported.append({
                "uid": uid,
                "name": ds.name,
                "source_path": ds.source_path,
                "x": ds.x,
                "y_raw": ds.y_raw,
                "y_processed": ds.y_processed,
                "metadata": ds.metadata,
                "analyses": analyses,
            })
        return {
            "project_name": self.current_project.stem if self.current_project else "Afruz_PXRD_Project",
            "project_path": str(self.current_project) if self.current_project else "Not saved",
            "application_version": APP_VERSION,
            "datasets": exported,
            "project_analyses": {
                "cif_reference": {
                    "structure": self.reference_structure,
                    "reference_pattern": self.reference_pattern,
                } if self.reference_structure or self.reference_pattern else None,
                "residual_stress": {
                    "observations": self.residual_stress_observations,
                    "result": self.residual_stress_result,
                } if self.residual_stress_result or self.residual_stress_observations else None,
                "instrument_profile": self.active_instrument_profile,
                "metrology_registry": self.project_state.metrology_registry.to_dict(),
            },
            "batch_result": getattr(batch_widget, "batch_result", None) if batch_widget is not None else None,
            "metadata": {
                "workflow_mode": self.workflow_mode,
                "active_workflow_task": self.active_workflow_task,
                "theme": self.current_theme_name,
                "project_dirty": self.project_dirty,
                "selected_dataset": selected.name if selected is not None else "None",
                "instrument_profile_available": self.active_instrument_profile is not None,
            },
        }

    def export_selected_csv(self):
        self.show_export_center(preferred_formats={"text", "zip"})

    def export_selected_excel(self):
        self.show_export_center(preferred_formats={"excel"})

    def show_about(self):
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"{APP_WINDOW_TITLE}\n\n"
            "Powder X-ray diffraction analysis, refinement, quantitative analysis, "
            "validation, and reproducible reporting.\n\n"
            "Projects are validated before opening and saved atomically with a recoverable "
            ".afz.bak copy of the previous project. Scientific results retain provenance and "
            "outdated-result tracking.\n\n"
            "Synthetic validation data verify software behavior but are not certified "
            "experimental reference materials.",
        )
