from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .batch_engine import comparison_rows
from .batch_plot import BATCH_METRICS, BatchComparisonPlotWidget
from .batch_recipe import (
    load_recipe,
    new_recipe,
    recipe_fingerprint,
    save_recipe,
    validate_recipe,
)
from .batch_report import export_batch_bundle
from .batch_worker import BatchAnalysisWorker
from .context_copy import install_label_copy_menu, install_table_copy_menu
from .io_engine import DataImportError, load_patterns
from .phase_identification import reference_from_dataset
from .widgets import NoWheelComboBox


STAGE_LABELS = {
    "preprocessing": "Preprocessing",
    "peak_detection": "Peak detection",
    "peak_fitting": "Peak fitting",
    "size_strain": "Size and strain",
    "cif_cell": "CIF/reference cell matching",
    "phase_identification": "Phase identification",
    "whole_pattern": "Pawley / Le Bail refinement",
    "rietveld": "Rietveld refinement",
    "qpa": "Quantitative phase analysis",
    "residual_stress": "Residual-stress series",
}


class BatchWorkflowWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.recipe = new_recipe()
        self.batch_result: dict | None = None
        self.queue_uids: list[str] = []
        self._thread: QThread | None = None
        self._worker: BatchAnalysisWorker | None = None
        self._updating_tables = False
        self._build_ui()
        self._update_recipe_preview()
        self.refresh_queue_table()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        heading = QLabel("Phase 8 — Batch Processing and Automated Reports")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        self.summary_label = QLabel(
            "Build a controlled sample queue, capture one shared recipe, run "
            "the same analysis stages, compare results, and export traceable reports."
        )
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("mutedLabel")
        install_label_copy_menu(self.summary_label)
        layout.addWidget(self.summary_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self._build_queue_tab()
        self._build_comparison_tab()
        self._build_recipe_tab()
        self._build_report_tab()

    def _build_queue_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        controls = QHBoxLayout()
        self.add_current_button = QPushButton("Add current datasets")
        self.import_folder_button = QPushButton("Import folder")
        self.clear_queue_button = QPushButton("Clear queue")
        self.start_button = QPushButton("Start batch")
        self.start_button.setObjectName("primaryButton")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.retry_button = QPushButton("Retry failed")
        controls.addWidget(self.add_current_button)
        controls.addWidget(self.import_folder_button)
        controls.addWidget(self.clear_queue_button)
        controls.addStretch(1)
        controls.addWidget(self.retry_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.cancel_button)
        layout.addLayout(controls)

        stage_group = QGroupBox("Shared analysis stages")
        stage_layout = QGridLayout(stage_group)
        self.stage_checks = {}
        for index, (key, label) in enumerate(STAGE_LABELS.items()):
            check = QCheckBox(label)
            check.setChecked(bool(self.recipe["stages"].get(key, False)))
            self.stage_checks[key] = check
            stage_layout.addWidget(check, index // 4, index % 4)
        layout.addWidget(stage_group)

        self.queue_table = QTableWidget(0, 9)
        self.queue_table.setHorizontalHeaderLabels(
            [
                "Include",
                "Dataset",
                "Temperature",
                "Time",
                "Composition",
                "Source",
                "Status",
                "Current stage",
                "Message",
            ]
        )
        self.queue_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.queue_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        install_table_copy_menu(self.queue_table)
        layout.addWidget(self.queue_table, 1)

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m stages — %p%")
        self.progress_label = QLabel("Batch queue is ready.")
        self.progress_label.setWordWrap(True)
        self.progress_label.setObjectName("mutedLabel")
        install_label_copy_menu(self.progress_label)
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.progress_label, 2)
        layout.addLayout(progress_row)

        note = QLabel(
            "Reference-card datasets remain in the local library but are not "
            "treated as experimental samples. A failed file does not stop the "
            "remaining queue when continue-on-error is enabled."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        layout.addWidget(note)

        self.add_current_button.clicked.connect(self.add_current_datasets)
        self.import_folder_button.clicked.connect(self.import_folder)
        self.clear_queue_button.clicked.connect(self.clear_queue)
        self.start_button.clicked.connect(self.start_batch)
        self.cancel_button.clicked.connect(self.cancel_batch)
        self.retry_button.clicked.connect(self.retry_failed)
        self.queue_table.itemChanged.connect(self._queue_item_changed)
        self.tabs.addTab(page, "Batch Queue")

    def _build_comparison_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        top = QHBoxLayout()
        top.addWidget(QLabel("Dashboard metric"))
        self.metric_selector = NoWheelComboBox()
        self.metric_selector.addItems(list(BATCH_METRICS.keys()))
        top.addWidget(self.metric_selector)
        top.addStretch(1)
        self.refresh_comparison_button = QPushButton("Refresh comparison")
        top.addWidget(self.refresh_comparison_button)
        layout.addLayout(top)

        self.comparison_plot = BatchComparisonPlotWidget()
        layout.addWidget(self.comparison_plot, 2)

        self.comparison_table = QTableWidget(0, 17)
        self.comparison_table.setHorizontalHeaderLabels(
            [
                "Sample",
                "Status",
                "Peaks",
                "Fitted components",
                "Primary 2θ",
                "Primary FWHM",
                "Mean size (nm)",
                "WH size (nm)",
                "Microstrain",
                "Identified phase",
                "Phase score",
                "Pawley / Le Bail Rwp (%)",
                "Rietveld Rwp (%)",
                "Analysis time (s)",
                "Temperature",
                "Time",
                "Recipe deviation",
            ]
        )
        self.comparison_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.comparison_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        install_table_copy_menu(self.comparison_table)
        layout.addWidget(self.comparison_table, 1)

        self.metric_selector.currentTextChanged.connect(
            self.refresh_comparison
        )
        self.refresh_comparison_button.clicked.connect(
            self.refresh_comparison
        )
        self.tabs.addTab(page, "Comparison Dashboard")

    def _build_recipe_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        controls = QHBoxLayout()
        self.recipe_name = QLineEdit("Afruz shared batch recipe")
        self.capture_recipe_button = QPushButton("Capture current controls")
        self.save_recipe_button = QPushButton("Save recipe JSON")
        self.load_recipe_button = QPushButton("Load recipe JSON")
        controls.addWidget(QLabel("Recipe name"))
        controls.addWidget(self.recipe_name, 1)
        controls.addWidget(self.capture_recipe_button)
        controls.addWidget(self.save_recipe_button)
        controls.addWidget(self.load_recipe_button)
        layout.addLayout(controls)

        self.recipe_fingerprint_label = QLabel("Fingerprint: —")
        self.recipe_fingerprint_label.setObjectName("mutedLabel")
        install_label_copy_menu(self.recipe_fingerprint_label)
        layout.addWidget(self.recipe_fingerprint_label)

        self.recipe_preview = QTextEdit()
        self.recipe_preview.setReadOnly(True)
        layout.addWidget(self.recipe_preview, 1)

        note = QLabel(
            "The fingerprint identifies the complete recipe. A dataset or result "
            "with a different fingerprint is marked as a manual deviation and "
            "should not be compared as though it used identical conditions."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        layout.addWidget(note)

        self.capture_recipe_button.clicked.connect(
            self.capture_current_recipe
        )
        self.save_recipe_button.clicked.connect(self.save_recipe_dialog)
        self.load_recipe_button.clicked.connect(self.load_recipe_dialog)
        self.recipe_name.editingFinished.connect(self._recipe_name_changed)
        self.tabs.addTab(page, "Reproducibility Recipe")

    def _build_report_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        destination_group = QGroupBox("Report output")
        form = QFormLayout(destination_group)
        destination_row = QHBoxLayout()
        self.output_directory = QLineEdit("")
        self.browse_output_button = QPushButton("Browse")
        destination_row.addWidget(self.output_directory, 1)
        destination_row.addWidget(self.browse_output_button)
        form.addRow("Output directory", destination_row)
        self.report_base_name = QLineEdit("afruz_pxrd_batch_report")
        form.addRow("Base filename", self.report_base_name)
        layout.addWidget(destination_group)

        formats = QLabel(
            "One export creates HTML, PDF, Excel, CSV summary, warning/error "
            "CSV, JSON, PNG comparison, PNG overlay, and SVG overlay."
        )
        formats.setWordWrap(True)
        formats.setObjectName("mutedLabel")
        layout.addWidget(formats)

        buttons = QHBoxLayout()
        self.export_report_button = QPushButton("Export complete report bundle")
        self.export_report_button.setObjectName("primaryButton")
        self.export_raptor_button = QPushButton("Export clean tables via Raptor")
        self.open_report_folder_button = QPushButton("Open output folder")
        buttons.addWidget(self.export_report_button)
        buttons.addWidget(self.export_raptor_button)
        buttons.addWidget(self.open_report_folder_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.report_files_table = QTableWidget(0, 2)
        self.report_files_table.setHorizontalHeaderLabels(["Format", "File"])
        self.report_files_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.report_files_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        install_table_copy_menu(self.report_files_table)
        layout.addWidget(self.report_files_table, 1)

        self.report_status = QLabel("Run a batch before exporting reports.")
        self.report_status.setWordWrap(True)
        self.report_status.setObjectName("mutedLabel")
        install_label_copy_menu(self.report_status)
        layout.addWidget(self.report_status)

        self.browse_output_button.clicked.connect(self.browse_output)
        self.export_report_button.clicked.connect(self.export_reports)
        self.export_raptor_button.clicked.connect(
            lambda checked=False: self.main_window.show_export_center(
                preferred_formats={"excel", "text", "zip"},
                preferred_categories={"project", "pattern", "batch"},
            )
        )
        self.open_report_folder_button.clicked.connect(
            self.open_output_folder
        )
        self.tabs.addTab(page, "Automated Reports")

    def _is_experimental(self, dataset) -> bool:
        metadata = dataset.metadata or {}
        return not (
            metadata.get("analysis_role") == "reference_pattern"
            or metadata.get("plot_style") == "sticks"
        )

    def add_current_datasets(self):
        for dataset in self.main_window.datasets:
            if self._is_experimental(dataset) and dataset.uid not in self.queue_uids:
                self.queue_uids.append(dataset.uid)
        self.refresh_queue_table()

    def import_folder(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Import XRD folder for batch analysis",
            "",
        )
        if not directory:
            return
        extensions = {
            ".xrdml", ".xy", ".csv", ".txt", ".dat",
            ".pdfcard", ".jcpds", ".jade", ".card", ".ref",
        }
        failures = []
        imported = 0
        for path in sorted(Path(directory).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            try:
                datasets = load_patterns(path)
            except (DataImportError, Exception) as exc:
                failures.append(f"{path.name}: {exc}")
                continue
            for dataset in datasets:
                self.main_window.datasets.append(dataset)
                imported += 1
                if self._is_experimental(dataset):
                    self.queue_uids.append(dataset.uid)
        self.queue_uids = list(dict.fromkeys(self.queue_uids))
        self.main_window.refresh_dataset_list()
        self.refresh_queue_table()
        self.main_window.statusBar().showMessage(
            f"Batch folder import added {imported} dataset(s)."
        )
        if failures:
            QMessageBox.warning(
                self,
                "Batch import warnings",
                "\n".join(failures[:40]),
            )

    def clear_queue(self):
        if self.is_running():
            return
        self.queue_uids.clear()
        self.refresh_queue_table()

    def _dataset_by_uid(self, uid: str):
        return next(
            (
                dataset for dataset in self.main_window.datasets
                if dataset.uid == uid
            ),
            None,
        )

    def refresh_queue_table(self):
        self._updating_tables = True
        valid_uids = []
        for uid in self.queue_uids:
            dataset = self._dataset_by_uid(uid)
            if dataset is not None and self._is_experimental(dataset):
                valid_uids.append(uid)
        self.queue_uids = valid_uids

        result_by_uid = {
            row.get("dataset_uid"): row
            for row in (self.batch_result or {}).get("results", [])
        }
        self.queue_table.setRowCount(len(self.queue_uids))
        for row_index, uid in enumerate(self.queue_uids):
            dataset = self._dataset_by_uid(uid)
            result = result_by_uid.get(uid, {})
            include = QTableWidgetItem("Use")
            include.setFlags(include.flags() | Qt.ItemIsUserCheckable)
            include.setCheckState(Qt.Checked)
            include.setData(Qt.UserRole, uid)
            self.queue_table.setItem(row_index, 0, include)
            values = [
                dataset.name,
                dataset.metadata.get("temperature", ""),
                dataset.metadata.get("time", ""),
                dataset.metadata.get("composition", ""),
                dataset.source_path or "Embedded/generated",
                result.get("status", "Queued"),
                "",
                "; ".join(
                    result.get("errors", []) or result.get("warnings", [])
                ),
            ]
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(str(value))
                if column not in (2, 3, 4):
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.queue_table.setItem(row_index, column, item)
        self._updating_tables = False

    def _queue_item_changed(self, item):
        if self._updating_tables:
            return
        row = item.row()
        if not (0 <= row < len(self.queue_uids)):
            return
        dataset = self._dataset_by_uid(self.queue_uids[row])
        if dataset is None:
            return
        mapping = {2: "temperature", 3: "time", 4: "composition"}
        if item.column() in mapping:
            text = item.text().strip()
            dataset.metadata[mapping[item.column()]] = text

    def _included_datasets(self):
        included = []
        for row in range(self.queue_table.rowCount()):
            item = self.queue_table.item(row, 0)
            if item is None or item.checkState() != Qt.Checked:
                continue
            dataset = self._dataset_by_uid(str(item.data(Qt.UserRole)))
            if dataset is not None:
                included.append(dataset)
        return included

    def capture_current_recipe(self):
        window = self.main_window
        recipe = new_recipe()
        recipe["name"] = self.recipe_name.text().strip() or recipe["name"]
        recipe["stages"] = {
            key: check.isChecked()
            for key, check in self.stage_checks.items()
        }
        recipe["preprocessing"].update(
            {
                "subtract_background": window.background_check.isChecked(),
                "background_method": window.background_method.currentText(),
                "background_smoothness": window.background_smoothness.value(),
                "background_asymmetry": window.background_asymmetry.value(),
                "background_iterations": window.background_iterations.value(),
                "background_window_degrees": window.background_window_degrees.value(),
                "background_percentile": window.background_percentile.value(),
                "background_peak_protection": window.background_peak_protection.isChecked(),
                "clip_negative": window.background_clip_negative.isChecked(),
                "polynomial_order": window.poly_order.value(),
                "smooth": window.smooth_check.isChecked(),
                "smoothing_window": window.smooth_window.value(),
                "smoothing_order": window.smooth_order.value(),
                "smoothing_method": window.smoothing_method.currentText(),
                "smoothing_strength": window.smoothing_strength.value(),
                "smoothing_peak_protection": window.smoothing_peak_protection.isChecked(),
                "smoothing_peak_preservation": window.smoothing_peak_preservation.value(),
                "smoothing_maximum_position_shift_deg": window.smoothing_max_position_shift.value(),
                "smoothing_maximum_height_change_percent": window.smoothing_max_height_change.value(),
                "smoothing_maximum_fwhm_change_percent": window.smoothing_max_fwhm_change.value(),
                "normalize": window.normalize_check.isChecked(),
            }
        )
        recipe["peak_detection"].update(
            {
                "mode": "Smart",
                "smart_sensitivity": window.smart_sensitivity.currentText(),
                "prominence_fraction": window.prominence.value(),
                "minimum_distance_points": window.min_distance.value(),
            }
        )
        recipe["peak_fitting"].update(
            {
                "mode": "Standard",
                "model": window.fit_model.currentText(),
                "window_multiplier": window.fit_window_multiplier.value(),
            }
        )
        active_profile = deepcopy(
            getattr(window, "active_instrument_profile", None)
        )
        recipe["instrument_calibration"] = {
            "enabled": bool(active_profile),
            "profile": active_profile,
            "profile_fingerprint": (
                None if not active_profile
                else active_profile.get("fingerprint")
            ),
        }
        recipe["size_strain"].update(
            {
                "wavelength_angstrom": window.wavelength_angstrom.value(),
                "shape_factor": window.shape_factor.value(),
                "instrument_fwhm_deg": window.instrument_fwhm.value(),
                "correction_mode": window.instrument_correction.currentText(),
            }
        )
        recipe["cif_cell"].update(
            {
                "match_tolerance_deg": window.match_tolerance.value(),
                "crystal_system": window.crystal_system_selector.currentText(),
                "refine_zero_shift": window.refine_zero_shift_check.isChecked(),
            }
        )
        recipe["phase_identification"].update(
            {
                "peak_source": window.phase_peak_source.currentText(),
                "tolerance_deg": window.phase_tolerance.value(),
                "maximum_zero_shift_deg": window.phase_max_shift.value(),
                "reference_intensity_cutoff_percent": (
                    window.phase_reference_cutoff.value()
                ),
                "maximum_phases": window.phase_maximum_phases.value(),
                "mixture_pool_size": window.phase_mixture_pool.value(),
                "convert_from_d": window.phase_convert_d_check.isChecked(),
            }
        )
        whole_widget = getattr(
            window,
            "whole_pattern_widget",
            None,
        )
        if whole_widget is not None:
            phase_state = whole_widget._phase_state()
            recipe["whole_pattern"].update(
                {
                    "mode": whole_widget.mode.currentText(),
                    "use_gpu": whole_widget.use_cuda.isChecked(),
                    "use_processed_pattern": (
                        whole_widget.use_processed_check.isChecked()
                    ),
                    "wavelength_angstrom": whole_widget.wavelength.value(),
                    "two_theta_min_deg": whole_widget.range_min.value(),
                    "two_theta_max_deg": whole_widget.range_max.value(),
                    "reference_intensity_cutoff_percent": (
                        whole_widget.reference_cutoff.value()
                    ),
                    "background_order": (
                        whole_widget.background_order.value()
                    ),
                    "weighting": whole_widget.weighting.currentText(),
                    "refine_zero_shift": (
                        whole_widget.refine_zero_shift.isChecked()
                    ),
                    "refine_profile": (
                        whole_widget.refine_profile.isChecked()
                    ),
                    "refine_eta": whole_widget.refine_eta.isChecked(),
                    "initial_u": whole_widget.initial_u.value(),
                    "initial_v": whole_widget.initial_v.value(),
                    "initial_w": whole_widget.initial_w.value(),
                    "initial_eta": whole_widget.initial_eta.value(),
                    "cell_tolerance_percent": (
                        whole_widget.cell_tolerance.value()
                    ),
                    "extraction_cycles": (
                        whole_widget.extraction_cycles.value()
                    ),
                    "maximum_nonlinear_evaluations": (
                        whole_widget.maximum_evaluations.value()
                    ),
                    "maximum_optimization_points": (
                        whole_widget.optimization_points.value()
                    ),
                    "phase_uids": [
                        uid
                        for uid, row in phase_state.items()
                        if row.get("include")
                    ],
                    "refine_cell_uids": [
                        uid
                        for uid, row in phase_state.items()
                        if row.get("include") and row.get("refine")
                    ],
                }
            )

        rietveld_widget = getattr(window, "rietveld_widget", None)
        if rietveld_widget is not None:
            recipe["rietveld"].update(
                {
                    "use_gpu": rietveld_widget.use_cuda.isChecked(),
                    "use_processed_pattern": rietveld_widget.use_processed.isChecked(),
                    "wavelength_angstrom": rietveld_widget.wavelength.value(),
                    "two_theta_min_deg": rietveld_widget.range_min.value(),
                    "two_theta_max_deg": rietveld_widget.range_max.value(),
                    "intensity_cutoff_percent": rietveld_widget.cutoff.value(),
                    "background_order": rietveld_widget.background_order.value(),
                    "weighting": rietveld_widget.weighting.currentText(),
                    "robust_loss": rietveld_widget.loss.currentText(),
                    "refine_zero_shift": rietveld_widget.refine_zero.isChecked(),
                    "refine_profile": rietveld_widget.refine_profile.isChecked(),
                    "refine_eta": rietveld_widget.refine_eta.isChecked(),
                    "initial_u": rietveld_widget.initial_u.value(),
                    "initial_v": rietveld_widget.initial_v.value(),
                    "initial_w": rietveld_widget.initial_w.value(),
                    "initial_eta": rietveld_widget.initial_eta.value(),
                    "cell_tolerance_percent": rietveld_widget.cell_tolerance.value(),
                    "k_alpha2_enabled": rietveld_widget.kalpha2.isChecked(),
                    "k_alpha2_wavelength_angstrom": rietveld_widget.kalpha2_wavelength.value(),
                    "k_alpha2_ratio": rietveld_widget.kalpha2_ratio.value(),
                    "maximum_nonlinear_evaluations": rietveld_widget.max_evaluations.value(),
                    "maximum_optimization_points": rietveld_widget.optimization_points.value(),
                }
            )

        recipe["qpa"].update(
            {
                "mode": window.qpa_mode.currentText(),
                "use_processed_pattern": window.qpa_use_processed_check.isChecked(),
                "convert_from_d": window.qpa_convert_d_check.isChecked(),
                "reference_intensity_cutoff_percent": (
                    window.qpa_reference_cutoff.value()
                ),
                "reference_fwhm_deg": window.qpa_reference_fwhm.value(),
                "pseudo_voigt_eta": window.qpa_profile_eta.value(),
                "baseline_order": window.qpa_baseline_order.value(),
                "weighting": window.qpa_weighting.currentText(),
                "bootstrap_samples": window.qpa_bootstrap_samples.value(),
            }
        )
        recipe["residual_stress"].update(
            {
                "peak_source": window.stress_peak_source.currentText(),
                "target_two_theta_deg": window.stress_target_two_theta.value(),
                "search_half_window_deg": window.stress_search_window.value(),
                "wavelength_angstrom": window.stress_wavelength.value(),
                "default_peak_error_deg": window.stress_default_peak_error.value(),
                "azimuth_deg": window.stress_azimuth.value(),
                "reference_mode": window.stress_reference_mode.currentText(),
                "stress_free_two_theta_deg": window.stress_free_two_theta.value(),
                "elastic_mode": window.stress_elastic_mode.currentText(),
                "youngs_modulus_gpa": window.stress_youngs_modulus.value(),
                "poisson_ratio": window.stress_poisson_ratio.value(),
                "xec_half_s2_per_gpa": window.stress_xec_half_s2.value(),
                "regression_mode": window.stress_regression_mode.currentText(),
            }
        )
        self.recipe = validate_recipe(recipe)
        self._update_recipe_preview()

    def _update_recipe_preview(self):
        fingerprint = recipe_fingerprint(self.recipe)
        self.recipe_fingerprint_label.setText(
            f"Fingerprint: {fingerprint}"
        )
        self.recipe_preview.setPlainText(
            json.dumps(self.recipe, indent=2, ensure_ascii=False)
        )
        if self.batch_result:
            self.refresh_comparison()

    def _recipe_name_changed(self):
        self.recipe["name"] = (
            self.recipe_name.text().strip()
            or "Afruz shared batch recipe"
        )
        self._update_recipe_preview()

    def save_recipe_dialog(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save reproducibility recipe",
            "afruz_batch_recipe.json",
            "JSON (*.json)",
        )
        if not filename:
            return
        try:
            path = save_recipe(filename, self.recipe)
        except Exception as exc:
            QMessageBox.critical(self, "Recipe save failed", str(exc))
            return
        self.main_window.statusBar().showMessage(f"Saved recipe {path}")

    def load_recipe_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load reproducibility recipe",
            "",
            "JSON (*.json)",
        )
        if not filename:
            return
        try:
            self.recipe = load_recipe(filename)
        except Exception as exc:
            QMessageBox.critical(self, "Recipe load failed", str(exc))
            return
        self.recipe_name.setText(self.recipe.get("name", "Batch recipe"))
        for key, check in self.stage_checks.items():
            check.setChecked(bool(self.recipe["stages"].get(key, False)))
        self._update_recipe_preview()

    def is_running(self):
        return bool(self._thread is not None and self._thread.isRunning())

    def start_batch(self):
        if self.is_running():
            return
        datasets = self._included_datasets()
        if not datasets:
            QMessageBox.information(
                self,
                "Empty batch queue",
                "Add and include at least one experimental dataset.",
            )
            return
        self.capture_current_recipe()
        references = self.main_window._phase_reference_library()

        thread = QThread(self)
        worker = BatchAnalysisWorker(
            datasets,
            deepcopy(self.recipe),
            references,
            deepcopy(self.main_window.reference_pattern),
            deepcopy(self.main_window.reference_structure),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.cancelled.connect(self._on_cancelled)
        worker.failed.connect(self._on_failed)
        for signal in (worker.finished, worker.cancelled, worker.failed):
            signal.connect(thread.quit)
            signal.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.add_current_button.setEnabled(False)
        self.import_folder_button.setEnabled(False)
        self.clear_queue_button.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText("Preparing controlled batch analysis…")
        thread.start()

    def cancel_batch(self):
        if self._worker is None:
            return
        self._worker.request_cancel()
        self.cancel_button.setEnabled(False)
        self.progress_label.setText(
            "Cancellation requested. The active numerical stage will stop safely."
        )

    def _on_progress(self, completed: int, total: int, message: str):
        total = max(1, int(total))
        completed = max(0, min(int(completed), total))
        self.progress.setRange(0, total)
        self.progress.setValue(completed)
        self.progress.setFormat(
            f"{completed} / {total} stages — %p%"
        )
        self.progress_label.setText(message)
        for row in range(self.queue_table.rowCount()):
            dataset_item = self.queue_table.item(row, 1)
            if dataset_item and dataset_item.text() in message:
                stage_item = self.queue_table.item(row, 7)
                if stage_item:
                    stage_item.setText(message.split(":")[-1].strip())

    def _on_finished(self, result):
        self.batch_result = result
        total = max(1, self.progress.maximum())
        self.progress.setValue(total)
        self.progress.setFormat(
            f"{total} / {total} stages — 100%"
        )
        self.progress_label.setText(
            f"Batch completed: {result.get('completed_count', 0)} completed, "
            f"{result.get('failed_count', 0)} failed, "
            f"{result.get('elapsed_seconds', 0):.3f} s."
        )
        self.refresh_queue_table()
        self.refresh_comparison()
        self.report_status.setText(
            "Batch results are ready for HTML, PDF, Excel, CSV, JSON, PNG, and SVG export."
        )

    def _on_cancelled(self, result):
        self.batch_result = result
        self.progress_label.setText(
            "Batch cancelled. Completed dataset results were preserved."
        )
        self.refresh_queue_table()
        self.refresh_comparison()

    def _on_failed(self, traceback_text: str):
        final = traceback_text.strip().splitlines()[-1]
        self.progress_label.setText(f"Batch worker failed: {final}")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Batch processing failed")
        box.setText(final)
        box.setDetailedText(traceback_text)
        box.exec()

    def _thread_finished(self):
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.add_current_button.setEnabled(True)
        self.import_folder_button.setEnabled(True)
        self.clear_queue_button.setEnabled(True)
        self._worker = None
        self._thread = None

    def retry_failed(self):
        if not self.batch_result:
            return
        failed = {
            row.get("dataset_uid")
            for row in self.batch_result.get("results", [])
            if row.get("status") in ("Failed", "Completed with warnings")
        }
        if not failed:
            self.main_window.statusBar().showMessage(
                "There are no failed batch samples to retry."
            )
            return
        self._updating_tables = True
        for row in range(self.queue_table.rowCount()):
            item = self.queue_table.item(row, 0)
            uid = item.data(Qt.UserRole) if item else None
            if item:
                item.setCheckState(
                    Qt.Checked if uid in failed else Qt.Unchecked
                )
        self._updating_tables = False
        self.start_batch()

    @staticmethod
    def _format(value, digits=6):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "—"
        return f"{value:.{digits}g}"

    def refresh_comparison(self):
        rows = comparison_rows(self.batch_result or {})
        current_fingerprint = recipe_fingerprint(self.recipe)
        self.comparison_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            deviation = (
                row.get("recipe_fingerprint") != current_fingerprint
            )
            values = [
                row.get("dataset_name"),
                row.get("status"),
                row.get("peak_count"),
                row.get("fitted_component_count"),
                self._format(row.get("primary_peak_position_deg")),
                self._format(row.get("primary_peak_fwhm_deg")),
                self._format(row.get("scherrer_mean_nm")),
                self._format(row.get("wh_size_nm")),
                self._format(row.get("microstrain")),
                row.get("identified_phase") or "—",
                self._format(row.get("phase_score")),
                self._format(
                    row.get("whole_pattern_rwp_percent")
                ),
                self._format(row.get("rietveld_rwp_percent")),
                self._format(row.get("elapsed_seconds")),
                row.get("temperature") or "—",
                row.get("time") or "—",
                "Yes" if deviation else "No",
            ]
            row["recipe_deviation"] = deviation
            for column, value in enumerate(values):
                self.comparison_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(str(value)),
                )
        self.comparison_plot.set_results(
            rows,
            self.metric_selector.currentText(),
        )

    def browse_output(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select report output directory",
            self.output_directory.text(),
        )
        if directory:
            self.output_directory.setText(directory)

    def export_reports(self):
        if not self.batch_result:
            QMessageBox.information(
                self,
                "No batch result",
                "Run a batch before exporting reports.",
            )
            return
        directory = self.output_directory.text().strip()
        if not directory:
            directory = QFileDialog.getExistingDirectory(
                self,
                "Select report output directory",
                "",
            )
            if not directory:
                return
            self.output_directory.setText(directory)
        try:
            paths = export_batch_bundle(
                directory,
                self.batch_result,
                self.report_base_name.text().strip()
                or "afruz_pxrd_batch_report",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Report export failed",
                str(exc),
            )
            return
        self.report_files_table.setRowCount(len(paths))
        for row, (format_name, path) in enumerate(paths.items()):
            self.report_files_table.setItem(
                row, 0, QTableWidgetItem(format_name.upper())
            )
            self.report_files_table.setItem(
                row, 1, QTableWidgetItem(path)
            )
        self.report_status.setText(
            f"Exported {len(paths)} report files to {directory}."
        )
        self.main_window.statusBar().showMessage(
            "Phase 8 report bundle exported."
        )

    def open_output_folder(self):
        directory = self.output_directory.text().strip()
        if directory and Path(directory).exists():
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(Path(directory).resolve()))
            )

    def apply_theme(self, theme_name: str):
        self.comparison_plot.apply_theme(theme_name)

    def get_state(self) -> dict:
        return {
            "recipe": deepcopy(self.recipe),
            "batch_result": deepcopy(self.batch_result),
            "queue_uids": list(self.queue_uids),
            "metric": self.metric_selector.currentText(),
            "output_directory": self.output_directory.text(),
            "report_base_name": self.report_base_name.text(),
        }

    def set_state(self, state: dict | None):
        if not isinstance(state, dict):
            return
        try:
            self.recipe = validate_recipe(state.get("recipe", self.recipe))
        except Exception:
            self.recipe = new_recipe()
        result = state.get("batch_result")
        self.batch_result = result if isinstance(result, dict) else None
        self.queue_uids = [
            str(uid) for uid in state.get("queue_uids", [])
        ]
        self.recipe_name.setText(self.recipe.get("name", "Batch recipe"))
        for key, check in self.stage_checks.items():
            check.setChecked(bool(self.recipe["stages"].get(key, False)))
        self.metric_selector.setCurrentText(
            state.get("metric", "Primary peak position")
        )
        self.output_directory.setText(
            state.get("output_directory", "")
        )
        self.report_base_name.setText(
            state.get("report_base_name", "afruz_pxrd_batch_report")
        )
        self._update_recipe_preview()
        self.refresh_queue_table()
        self.refresh_comparison()
