from __future__ import annotations

from copy import deepcopy
import uuid

import numpy as np
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .context_copy import install_label_copy_menu, install_table_copy_menu
from .crystal_profiles import PROFILE_MODELS
from .crystallography import load_cif
from .doping_series import (
    NORMALIZATION_MODES,
    REFINEMENT_MODES,
    export_doping_series_bundle,
)
from .doping_series_plot import DopingSeriesPlotWidget, TREND_METRICS
from .doping_series_worker import DopingSeriesWorker
from .processing import (
    active_peak_rows,
    merge_detected_with_manual,
    normalize_peak_rows,
    smart_detect_peaks,
)
from .rietveld_refinement import (
    GENERATED_PATTERN_SCALING,
    RIETVELD_WEIGHTING,
    RietveldPhaseSpec,
)
from .version import APP_VERSION
from .widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox


PEAK_SOURCE_MODES = (
    "Main fitted/curated peaks, then Smart fallback",
    "Always use Main Smart Peak Search",
)


class DopingSeriesWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.structures: list[dict] = []
        self.series_result: dict | None = None
        self._thread: QThread | None = None
        self._worker: DopingSeriesWorker | None = None
        self._running_phase_specs: list[RietveldPhaseSpec] = []
        self._updating = False
        self._series_state: dict[str, dict] = {}
        self._build_ui()
        self.refresh_datasets()

    def _build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self.workspace_scroll = QScrollArea()
        self.workspace_scroll.setObjectName("dopingSeriesWorkspaceScroll")
        self.workspace_scroll.setWidgetResizable(True)
        self.workspace_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.workspace_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.workspace_page = QWidget()
        self.workspace_page.setObjectName("dopingSeriesWorkspacePage")
        self.workspace_page.setMinimumSize(860, 900)
        layout = QVBoxLayout(self.workspace_page)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(7)
        heading = QLabel("Phase 12 — Doping-Series Comparison and Sequential Refinement")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        description = QLabel(
            "Compare composition-ordered XRD patterns without modifying raw measurements, track peak evolution, "
            "and refine every original pattern against controlled CIF model(s). Sequential carry-forward is an "
            "initialization strategy—not proof of a dopant site, phase assemblage, or structural trend."
        )
        description.setWordWrap(True)
        description.setObjectName("mutedLabel")
        install_label_copy_menu(description)
        layout.addWidget(description)

        self.setup_tabs = QTabWidget()
        self.setup_tabs.setObjectName("dopingSeriesSetupTabs")
        self.setup_tabs.setDocumentMode(True)
        self.setup_tabs.setMinimumHeight(315)
        self.setup_tabs.setMaximumHeight(410)

        self.dataset_panel = self._build_dataset_panel()
        self.cif_panel = self._build_cif_panel()
        self.controls_panel = self._build_controls_panel()
        self.controls_panel.setMinimumWidth(620)

        self.controls_scroll = QScrollArea()
        self.controls_scroll.setObjectName("dopingSeriesControlsScroll")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.controls_scroll.setWidget(self.controls_panel)

        self.setup_tabs.addTab(self.dataset_panel, "1. Series Datasets")
        self.setup_tabs.addTab(self.cif_panel, "2. CIF Models")
        self.setup_tabs.addTab(self.controls_scroll, "3. Refinement Controls")
        layout.addWidget(self.setup_tabs)

        controls = QHBoxLayout()
        self.compare_button = QPushButton("Compare series")
        self.compare_button.setObjectName("primaryButton")
        self.refine_button = QPushButton("Compare + refine all")
        self.refine_button.setObjectName("primaryButton")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.export_button = QPushButton("Export complete series")
        self.export_button.setEnabled(False)
        controls.addWidget(self.compare_button)
        controls.addWidget(self.refine_button)
        controls.addWidget(self.cancel_button)
        controls.addStretch(1)
        controls.addWidget(self.export_button)
        layout.addLayout(controls)

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, DopingSeriesWorker.PROGRESS_MAXIMUM)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Ready — %p%")
        self.progress_label = QLabel("Ready — include at least two datasets and enter their concentrations.")
        self.progress_label.setObjectName("mutedLabel")
        self.progress_label.setWordWrap(False)
        install_label_copy_menu(self.progress_label)
        progress_row.addWidget(self.progress, 2)
        progress_row.addWidget(self.progress_label, 4)
        layout.addLayout(progress_row)

        self.result_tabs = QTabWidget()
        self.result_tabs.setMinimumHeight(390)
        self.plot_widget = DopingSeriesPlotWidget()
        self.result_tabs.addTab(self.plot_widget, "Series Visualizations")
        self.result_tabs.addTab(self._table_page(self._build_peak_table()), "Peak Evolution")
        self.result_tabs.addTab(self._table_page(self._build_trend_table()), "Refinement Trends")
        self.result_tabs.addTab(self._table_page(self._build_run_table()), "Run Diagnostics")
        layout.addWidget(self.result_tabs, 1)

        boundary = QLabel(
            "Peak shifts can also arise from zero error or specimen displacement. Intensity and width changes can "
            "also arise from preferred orientation, absorption, thickness, crystallite statistics, or preparation. "
            "Use calibrated acquisition and independent evidence before assigning a change specifically to doping."
        )
        boundary.setObjectName("mutedLabel")
        boundary.setWordWrap(True)
        layout.addWidget(boundary)

        self.workspace_scroll.setWidget(self.workspace_page)
        outer_layout.addWidget(self.workspace_scroll)

        self.compare_button.clicked.connect(lambda: self._start(include_refinement=False))
        self.refine_button.clicked.connect(lambda: self._start(include_refinement=True))
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.export_button.clicked.connect(self.export_result)
        self.refresh_button.clicked.connect(self.refresh_datasets)
        self.review_preparation_button.clicked.connect(self.open_main_preparation)
        self.smart_search_button.clicked.connect(self.run_linked_smart_peak_search)
        self.add_active_cif_button.clicked.connect(self.add_active_cif)
        self.add_phase11_button.clicked.connect(self.add_phase11_cifs)
        self.import_cif_button.clicked.connect(self.import_cif)
        self.remove_cif_button.clicked.connect(self.remove_selected_cif)
        self.normalization.currentTextChanged.connect(self._normalization_changed)
        self.trend_metric.currentTextChanged.connect(self.plot_widget.set_trend_metric)
        self.dataset_table.itemChanged.connect(self._dataset_item_changed)
        self._normalization_changed(self.normalization.currentText())

    def _build_dataset_panel(self):
        group = QGroupBox("Series datasets and composition metadata")
        layout = QVBoxLayout(group)
        row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh project datasets")
        self.review_preparation_button = QPushButton("Review baseline / smoothing")
        self.smart_search_button = QPushButton("Run linked Smart peak search")
        self.reference_dataset = NoWheelComboBox()
        row.addWidget(self.refresh_button)
        row.addWidget(self.review_preparation_button)
        row.addWidget(self.smart_search_button)
        row.addWidget(QLabel("Reference:"))
        row.addWidget(self.reference_dataset, 1)
        layout.addLayout(row)

        peak_row = QHBoxLayout()
        self.peak_source_mode = NoWheelComboBox()
        self.peak_source_mode.addItems(list(PEAK_SOURCE_MODES))
        self.smart_sensitivity = NoWheelComboBox()
        self.smart_sensitivity.addItems(["Conservative", "Balanced", "Sensitive"])
        main_sensitivity = getattr(self.main_window, "smart_sensitivity", None)
        if main_sensitivity is not None and hasattr(main_sensitivity, "currentText"):
            self.smart_sensitivity.setCurrentText(main_sensitivity.currentText())
        else:
            self.smart_sensitivity.setCurrentText("Balanced")
        peak_row.addWidget(QLabel("Peak source:"))
        peak_row.addWidget(self.peak_source_mode, 1)
        peak_row.addWidget(QLabel("Smart sensitivity:"))
        peak_row.addWidget(self.smart_sensitivity)
        layout.addLayout(peak_row)

        preparation_note = QLabel(
            "Prepared signal = the main workspace's applied baseline correction/smoothing. It is used for "
            "comparison and Smart peak search only; CIF refinement keeps the original measured intensities."
        )
        preparation_note.setObjectName("mutedLabel")
        preparation_note.setWordWrap(True)
        layout.addWidget(preparation_note)

        self.dataset_table = QTableWidget(0, 9)
        self.dataset_table.setHorizontalHeaderLabels(
            [
                "Use", "Dataset", "Dopant", "Concentration", "Unit",
                "Prepared signal", "Preparation", "Main peaks", "Raw provenance",
            ]
        )
        self.dataset_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.dataset_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.dataset_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.dataset_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        install_table_copy_menu(self.dataset_table)
        layout.addWidget(self.dataset_table)
        return group

    def _build_cif_panel(self):
        group = QGroupBox("CIF phase model(s)")
        layout = QVBoxLayout(group)
        buttons = QHBoxLayout()
        self.add_active_cif_button = QPushButton("Add active CIF")
        self.add_phase11_button = QPushButton("Copy Phase 11 model")
        self.import_cif_button = QPushButton("Import CIF…")
        self.remove_cif_button = QPushButton("Remove")
        for button in (
            self.add_active_cif_button,
            self.add_phase11_button,
            self.import_cif_button,
            self.remove_cif_button,
        ):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.cif_table = QTableWidget(0, 6)
        self.cif_table.setHorizontalHeaderLabels(
            ["Use", "Phase", "Formula", "Refine cell", "Refine Biso", "Source"]
        )
        self.cif_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cif_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.cif_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        install_table_copy_menu(self.cif_table)
        layout.addWidget(self.cif_table)
        return group

    def _build_controls_panel(self):
        group = QGroupBox("Comparison and refinement controls")
        form = QFormLayout(group)
        self.normalization = NoWheelComboBox()
        self.normalization.addItems(list(NORMALIZATION_MODES))
        self.window_min = NoWheelDoubleSpinBox()
        self.window_min.setRange(0.0, 179.0)
        self.window_min.setDecimals(4)
        self.window_min.setValue(20.0)
        self.window_max = NoWheelDoubleSpinBox()
        self.window_max.setRange(0.1, 179.9)
        self.window_max.setDecimals(4)
        self.window_max.setValue(30.0)
        self.wavelength = NoWheelDoubleSpinBox()
        self.wavelength.setRange(0.1, 10.0)
        self.wavelength.setDecimals(7)
        self.wavelength.setValue(1.5406)
        self.peak_tolerance = NoWheelDoubleSpinBox()
        self.peak_tolerance.setRange(0.005, 2.0)
        self.peak_tolerance.setDecimals(4)
        self.peak_tolerance.setValue(0.15)
        self.refinement_mode = NoWheelComboBox()
        self.refinement_mode.addItems(list(REFINEMENT_MODES))
        self.refinement_mode.setCurrentText("Sequential from reference")
        self.weighting = NoWheelComboBox()
        self.weighting.addItems(list(RIETVELD_WEIGHTING))
        self.profile_model = NoWheelComboBox()
        self.profile_model.addItems(list(PROFILE_MODELS))
        self.profile_model.setCurrentText("TCH pseudo-Voigt")
        self.background_order = NoWheelSpinBox()
        self.background_order.setRange(0, 6)
        self.background_order.setValue(3)
        self.refine_zero = QCheckBox("Refine zero shift")
        self.refine_zero.setChecked(True)
        self.refine_profile = QCheckBox("Refine shared profile terms per pattern")
        self.refine_profile.setChecked(True)
        self.staged = QCheckBox("Use staged Rietveld refinement")
        self.staged.setChecked(True)
        self.use_calibration = QCheckBox("Use active Phase 9 instrument profile")
        self.use_calibration.setChecked(True)
        self.cell_tolerance = NoWheelDoubleSpinBox()
        self.cell_tolerance.setRange(0.1, 15.0)
        self.cell_tolerance.setDecimals(2)
        self.cell_tolerance.setValue(3.0)
        self.max_evaluations = NoWheelSpinBox()
        self.max_evaluations.setRange(10, 1000)
        self.max_evaluations.setValue(100)
        self.optimization_points = NoWheelSpinBox()
        self.optimization_points.setRange(300, 6000)
        self.optimization_points.setValue(1500)
        self.maximum_rwp = NoWheelDoubleSpinBox()
        self.maximum_rwp.setRange(1.0, 100.0)
        self.maximum_rwp.setDecimals(2)
        self.maximum_rwp.setValue(30.0)
        self.maximum_correlation = NoWheelDoubleSpinBox()
        self.maximum_correlation.setRange(0.5, 0.99999)
        self.maximum_correlation.setDecimals(5)
        self.maximum_correlation.setValue(0.98)
        self.trend_metric = NoWheelComboBox()
        self.trend_metric.addItems(list(TREND_METRICS))
        self.trend_metric.setCurrentText("Cell volume")
        for label, widget in (
            ("Normalization", self.normalization),
            ("Window minimum 2θ", self.window_min),
            ("Window maximum 2θ", self.window_max),
            ("Wavelength (Å)", self.wavelength),
            ("Peak match tolerance (°)", self.peak_tolerance),
            ("Refinement mode", self.refinement_mode),
            ("Weighting", self.weighting),
            ("Profile model", self.profile_model),
            ("Background order", self.background_order),
            ("Cell bounds (±%)", self.cell_tolerance),
            ("Maximum evaluations", self.max_evaluations),
            ("Optimization points", self.optimization_points),
            ("Rwp carry gate (%)", self.maximum_rwp),
            ("Correlation carry gate", self.maximum_correlation),
            ("Trend plot", self.trend_metric),
        ):
            form.addRow(label, widget)
        form.addRow(self.refine_zero)
        form.addRow(self.refine_profile)
        form.addRow(self.staged)
        form.addRow(self.use_calibration)
        return group

    @staticmethod
    def _table_page(table):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(table)
        return page

    def _build_peak_table(self):
        self.peak_table = QTableWidget(0, 12)
        self.peak_table.setHorizontalHeaderLabels(
            ["Track", "Dataset", "Concentration", "2θ", "2θ s.e.", "Shift", "d", "Δd", "FWHM", "Intensity", "Class", "Source"]
        )
        self.peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        install_table_copy_menu(self.peak_table)
        return self.peak_table

    def _build_trend_table(self):
        self.trend_table = QTableWidget(0, 14)
        self.trend_table.setHorizontalHeaderLabels(
            ["Dataset", "Dopant", "Concentration", "Status", "Rwp", "χ²red", "a", "b", "c", "Volume", "Zero", "Phase fraction", "Max |corr|", "Time (s)"]
        )
        self.trend_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        install_table_copy_menu(self.trend_table)
        return self.trend_table

    def _build_run_table(self):
        self.run_table = QTableWidget(0, 9)
        self.run_table.setHorizontalHeaderLabels(
            ["Dataset", "Concentration", "Result", "Initialization", "Rwp", "Gate", "Gate reasons", "Warnings", "Error"]
        )
        self.run_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.run_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        install_table_copy_menu(self.run_table)
        return self.run_table

    @staticmethod
    def _check_item(text: str, checked: bool, uid: str | None = None):
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        if uid is not None:
            item.setData(Qt.UserRole, uid)
        return item

    def _capture_series_table(self):
        if self._updating:
            return
        for row in range(self.dataset_table.rowCount()):
            include = self.dataset_table.item(row, 0)
            if include is None:
                continue
            uid = str(include.data(Qt.UserRole))
            self._series_state[uid] = {
                "include": include.checkState() == Qt.Checked,
                "dopant": self.dataset_table.item(row, 2).text().strip(),
                "series_value": self.dataset_table.item(row, 3).text().strip(),
                "series_unit": self.dataset_table.item(row, 4).text().strip(),
                "use_processed": self.dataset_table.item(row, 5).checkState() == Qt.Checked,
            }

    def _dataset_item_changed(self, *_args):
        if self._updating:
            return
        self._capture_series_table()
        self._refresh_reference_combo()

    def refresh_datasets(self):
        self._capture_series_table()
        datasets = list(getattr(self.main_window, "datasets", []))
        self._updating = True
        self.dataset_table.setRowCount(len(datasets))
        for row, dataset in enumerate(datasets):
            metadata = dataset.metadata if isinstance(dataset.metadata, dict) else {}
            previous = self._series_state.get(dataset.uid, {})
            default_value = metadata.get(
                "dopant_concentration",
                metadata.get("series_value", row),
            )
            values = [
                self._check_item("", previous.get("include", True), dataset.uid),
                QTableWidgetItem(dataset.name),
                QTableWidgetItem(str(previous.get("dopant", metadata.get("dopant_element", "")))),
                QTableWidgetItem(str(previous.get("series_value", default_value))),
                QTableWidgetItem(str(previous.get("series_unit", metadata.get("dopant_unit", "at.%")))),
                self._check_item(
                    "",
                    previous.get("use_processed", dataset.y_processed is not None)
                    and dataset.y_processed is not None,
                ),
                QTableWidgetItem(self._preparation_status(dataset)),
                QTableWidgetItem(self._peak_status(dataset.uid)),
                QTableWidgetItem(str(metadata.get("intensity_provenance", "raw_counts"))),
            ]
            values[1].setFlags(values[1].flags() & ~Qt.ItemIsEditable)
            for column in (6, 7, 8):
                values[column].setFlags(values[column].flags() & ~Qt.ItemIsEditable)
            if dataset.y_processed is None:
                values[5].setFlags(values[5].flags() & ~Qt.ItemIsEnabled)
            for column, item in enumerate(values):
                self.dataset_table.setItem(row, column, item)
        self._updating = False
        self._capture_series_table()
        self._refresh_reference_combo()

    @staticmethod
    def _preparation_status(dataset) -> str:
        if dataset.y_processed is None:
            return "Raw only — prepare in main workspace"
        metadata = dataset.metadata if isinstance(dataset.metadata, dict) else {}
        provenance = metadata.get("processing_provenance", {})
        if not isinstance(provenance, dict):
            return "Applied prepared pattern"
        steps = []
        if provenance.get("background_subtracted"):
            steps.append("baseline corrected")
        if provenance.get("smoothed"):
            steps.append("smoothed")
        if provenance.get("normalized"):
            steps.append("normalized")
        return "Applied: " + (" + ".join(steps) if steps else "prepared pattern")

    def _peak_status(self, dataset_uid: str) -> str:
        fitted = sum(
            len(group.get("components", []))
            for group in getattr(self.main_window, "fit_groups", {}).get(dataset_uid, [])
        )
        curated = len(
            active_peak_rows(
                getattr(self.main_window, "peak_rows", {}).get(dataset_uid, [])
            )
        )
        if fitted:
            return f"{fitted} fitted · {curated} curated"
        if curated:
            return f"{curated} curated"
        return "None — Smart fallback will run"

    def _selected_or_first_included_uid(self) -> str | None:
        selected = self.dataset_table.selectionModel().selectedRows()
        rows = [selected[0].row()] if selected else range(self.dataset_table.rowCount())
        for row in rows:
            item = self.dataset_table.item(row, 0)
            if item is not None and (selected or item.checkState() == Qt.Checked):
                return str(item.data(Qt.UserRole))
        return None

    def open_main_preparation(self):
        uid = self._selected_or_first_included_uid()
        if not uid:
            QMessageBox.information(self, "No dataset", "Select or include a dataset first.")
            return
        selector = getattr(self.main_window, "_select_dataset_uid_for_peak_workspace", None)
        if callable(selector):
            selector(uid)
        else:
            dataset_list = getattr(self.main_window, "dataset_list", None)
            for index, dataset in enumerate(getattr(self.main_window, "datasets", [])):
                if dataset.uid == uid and dataset_list is not None:
                    dataset_list.setCurrentRow(index)
                    break
        task_selector = getattr(self.main_window, "_select_workflow_task", None)
        if callable(task_selector):
            task_selector("background")

    def _comparison_signal(self, dataset, state: dict) -> tuple[np.ndarray, bool]:
        use_processed = bool(
            state.get("use_processed") and dataset.y_processed is not None
        )
        signal = dataset.y_processed if use_processed else dataset.y_raw
        return np.asarray(signal, dtype=float), use_processed

    def _smart_peak_rows(self, dataset, signal: np.ndarray) -> tuple[list[dict], dict]:
        sensitivity = self.smart_sensitivity.currentText()
        detected, diagnostics = smart_detect_peaks(
            dataset.x,
            signal,
            sensitivity=sensitivity,
        )
        source = f"Main Smart peak search ({sensitivity})"
        for peak in detected:
            peak["source"] = source
            peak["origin"] = source
        return detected, diagnostics

    def run_linked_smart_peak_search(self):
        """Run the main Smart detector for every included series pattern."""

        if self.is_running():
            return
        self._capture_series_table()
        datasets = {dataset.uid: dataset for dataset in self.main_window.datasets}
        included = [
            datasets[uid]
            for uid, state in self._series_state.items()
            if state.get("include") and uid in datasets
        ]
        if not included:
            QMessageBox.information(self, "No datasets", "Include at least one series dataset first.")
            return

        main_sensitivity = getattr(self.main_window, "smart_sensitivity", None)
        if main_sensitivity is not None and hasattr(main_sensitivity, "setCurrentText"):
            main_sensitivity.setCurrentText(self.smart_sensitivity.currentText())

        detected_total = 0
        completed = 0
        blocked_names = []
        failures = []
        self.progress.setRange(0, len(included))
        self.progress.setValue(0)
        self.progress.setFormat("Smart peak search — %p%")
        for index, dataset in enumerate(included, start=1):
            state = self._series_state[dataset.uid]
            signal, use_processed = self._comparison_signal(dataset, state)
            message = (
                f"Smart peak search: {dataset.name} ({index}/{len(included)}) — "
                f"{'prepared' if use_processed else 'raw'} signal"
            )
            self.progress_label.setText(message)
            QApplication.processEvents()
            blocker = getattr(self.main_window, "_main_peak_list_blocks_edit", None)
            if callable(blocker) and blocker(dataset):
                blocked_names.append(dataset.name)
                self.progress.setValue(index)
                continue
            try:
                detected, _diagnostics = self._smart_peak_rows(dataset, signal)
                preserve_control = getattr(
                    self.main_window,
                    "preserve_manual_peaks_check",
                    None,
                )
                preserve_manual = (
                    True
                    if preserve_control is None
                    else bool(preserve_control.isChecked())
                )
                rows = (
                    merge_detected_with_manual(
                        getattr(self.main_window, "peak_rows", {}).get(dataset.uid, []),
                        detected,
                    )
                    if preserve_manual
                    else normalize_peak_rows(detected)
                )
                commit = getattr(self.main_window, "_commit_main_peak_rows", None)
                if callable(commit):
                    commit(
                        dataset.uid,
                        rows,
                        reason="Phase 12 linked Smart peak search",
                    )
                else:
                    self.main_window.peak_rows[dataset.uid] = rows
                detected_total += len(detected)
                completed += 1
            except Exception as exc:
                failures.append(f"{dataset.name}: {exc}")
            self.progress.setValue(index)
            QApplication.processEvents()

        self.refresh_datasets()
        self.progress.setRange(0, DopingSeriesWorker.PROGRESS_MAXIMUM)
        self.progress.setValue(DopingSeriesWorker.PROGRESS_MAXIMUM)
        self.progress.setFormat("Peak search complete — %p%")
        summary = (
            f"Main Smart peak search updated {completed} pattern(s) with "
            f"{detected_total} automatic peak(s)."
        )
        if blocked_names:
            summary += f" Locked/skipped: {', '.join(blocked_names)}."
        if failures:
            summary += f" Failed: {'; '.join(failures)}"
        self.progress_label.setText(summary)
        self.progress_label.setToolTip(summary)
        if failures:
            QMessageBox.warning(self, "Some peak searches failed", "\n".join(failures))

    def _refresh_reference_combo(self):
        current_uid = self.reference_dataset.currentData()
        self.reference_dataset.blockSignals(True)
        self.reference_dataset.clear()
        for row in range(self.dataset_table.rowCount()):
            include = self.dataset_table.item(row, 0)
            if include is None or include.checkState() != Qt.Checked:
                continue
            uid = str(include.data(Qt.UserRole))
            name = self.dataset_table.item(row, 1).text()
            value = self.dataset_table.item(row, 3).text()
            self.reference_dataset.addItem(f"{value} — {name}", uid)
        index = self.reference_dataset.findData(current_uid)
        self.reference_dataset.setCurrentIndex(index if index >= 0 else 0)
        self.reference_dataset.blockSignals(False)

    def _records(self):
        self._capture_series_table()
        datasets = {dataset.uid: dataset for dataset in self.main_window.datasets}
        records = []
        for row in range(self.dataset_table.rowCount()):
            include = self.dataset_table.item(row, 0)
            if include is None or include.checkState() != Qt.Checked:
                continue
            uid = str(include.data(Qt.UserRole))
            dataset = datasets.get(uid)
            if dataset is None:
                continue
            state = self._series_state[uid]
            try:
                value = float(state["series_value"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{dataset.name}: concentration must be numeric.") from exc
            y, use_processed = self._comparison_signal(dataset, state)
            fit_peaks = []
            for group in getattr(self.main_window, "fit_groups", {}).get(uid, []):
                for component in group.get("components", []):
                    fit_peaks.append(
                        {
                            "position": component.get("center"),
                            "position_error": component.get("center_error"),
                            "intensity": component.get("height", 0.0),
                            "fwhm": component.get("fwhm"),
                            "fwhm_error": component.get("fwhm_error"),
                            "included": True,
                            "method": "Fitted",
                            "source": "Main fitted peak model",
                        }
                    )
            curated_peaks = active_peak_rows(
                deepcopy(getattr(self.main_window, "peak_rows", {}).get(uid, []))
            )
            for peak in curated_peaks:
                peak["source"] = str(
                    peak.get("source")
                    or peak.get("origin")
                    or peak.get("method")
                    or "Main curated peak list"
                )
            force_smart = self.peak_source_mode.currentText() == PEAK_SOURCE_MODES[1]
            if force_smart:
                peaks, peak_diagnostics = self._smart_peak_rows(dataset, y)
            else:
                peaks = fit_peaks or curated_peaks
                peak_diagnostics = None
                if not peaks:
                    peaks, peak_diagnostics = self._smart_peak_rows(dataset, y)
            metadata = dataset.metadata if isinstance(dataset.metadata, dict) else {}
            records.append(
                {
                    "dataset_uid": uid,
                    "dataset_name": dataset.name,
                    "x": np.asarray(dataset.x, dtype=float).copy(),
                    "y": y.copy(),
                    "raw_y": np.asarray(dataset.y_raw, dtype=float).copy(),
                    "refinement_y": np.asarray(dataset.y_raw, dtype=float).copy(),
                    "use_processed": use_processed,
                    "comparison_signal": "main prepared pattern" if use_processed else "raw pattern",
                    "intensity_provenance": "processed_for_comparison" if use_processed else str(
                        metadata.get("intensity_provenance", "raw_counts")
                    ),
                    "refinement_intensity_provenance": str(
                        metadata.get("intensity_provenance", "raw_counts")
                    ),
                    "dopant": state["dopant"],
                    "series_value": value,
                    "series_unit": state["series_unit"],
                    "peaks": peaks,
                    "peak_search_diagnostics": peak_diagnostics,
                }
            )
        return records

    def _normalization_changed(self, name: str):
        enabled = name == "Reference-window intensity"
        self.window_min.setEnabled(enabled)
        self.window_max.setEnabled(enabled)

    def _add_structure(self, structure: dict):
        signature = str(structure.get("source_path") or structure.get("data_name") or "")
        if any(str(row.get("source_path") or row.get("data_name") or "") == signature for row in self.structures):
            return
        copied = deepcopy(structure)
        copied["_series_uid"] = uuid.uuid4().hex
        self.structures.append(copied)
        self._populate_cif_table()

    def add_active_cif(self):
        structure = getattr(self.main_window, "reference_structure", None)
        if not structure:
            QMessageBox.information(self, "No active CIF", "Import or select a CIF structure first.")
            return
        self._add_structure(structure)

    def add_phase11_cifs(self):
        widget = getattr(self.main_window, "rietveld_widget", None)
        structures = widget.included_structures() if widget is not None else []
        if not structures:
            QMessageBox.information(self, "No Phase 11 model", "Include one or more CIF phases in Phase 11 first.")
            return
        for structure in structures:
            self._add_structure(structure)

    def import_cif(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Phase 12 CIF model", "", "Crystallographic Information File (*.cif)"
        )
        if filename:
            try:
                self._add_structure(load_cif(filename))
            except Exception as exc:
                QMessageBox.critical(self, "CIF import failed", str(exc))

    def remove_selected_cif(self):
        selected = sorted({index.row() for index in self.cif_table.selectionModel().selectedRows()}, reverse=True)
        for row in selected:
            if 0 <= row < len(self.structures):
                self.structures.pop(row)
        self._populate_cif_table()

    def _cif_state(self):
        state = {}
        for row in range(self.cif_table.rowCount()):
            use = self.cif_table.item(row, 0)
            if use is None:
                continue
            uid = str(use.data(Qt.UserRole))
            state[uid] = {
                "include": use.checkState() == Qt.Checked,
                "refine_cell": self.cif_table.item(row, 3).checkState() == Qt.Checked,
                "refine_biso": self.cif_table.item(row, 4).checkState() == Qt.Checked,
            }
        return state

    def _populate_cif_table(self, saved_state=None):
        previous = (
            deepcopy(saved_state)
            if isinstance(saved_state, dict)
            else self._cif_state()
        )
        self.cif_table.setRowCount(len(self.structures))
        for row, structure in enumerate(self.structures):
            uid = structure["_series_uid"]
            state = previous.get(uid, {})
            values = [
                self._check_item("", state.get("include", True), uid),
                QTableWidgetItem(str(structure.get("data_name", f"Phase {row + 1}"))),
                QTableWidgetItem(str(structure.get("formula", ""))),
                self._check_item("", state.get("refine_cell", True)),
                self._check_item("", state.get("refine_biso", False)),
                QTableWidgetItem(str(structure.get("source_path", structure.get("_afruz_origin", "Embedded")))),
            ]
            for column, item in enumerate(values):
                if column in {1, 2, 5}:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.cif_table.setItem(row, column, item)

    def _phase_specs(self):
        state = self._cif_state()
        specs = []
        for structure in self.structures:
            row = state.get(structure["_series_uid"], {})
            if not row.get("include"):
                continue
            specs.append(
                RietveldPhaseSpec(
                    structure={key: deepcopy(value) for key, value in structure.items() if key != "_series_uid"},
                    name=structure.get("data_name"),
                    refine_cell=bool(row.get("refine_cell", True)),
                    refine_biso=bool(row.get("refine_biso", False)),
                )
            )
        return specs

    def validate_inputs(self, *, require_cif=False):
        try:
            records = self._records()
        except Exception as exc:
            return False, str(exc)
        if len(records) < 2:
            return False, "Include at least two datasets."
        if len({row["series_value"] for row in records}) != len(records):
            return False, "Dopant concentrations/series values must be unique."
        if self.reference_dataset.currentData() not in {row["dataset_uid"] for row in records}:
            return False, "Select an included reference dataset."
        if self.normalization.currentText() == "Reference-window intensity" and self.window_min.value() >= self.window_max.value():
            return False, "The normalization window minimum must be below its maximum."
        if require_cif and not self._phase_specs():
            return False, "Add and include at least one CIF phase model."
        return True, ""

    def run_analysis(self):
        return self._start(include_refinement=False)

    def _start(self, *, include_refinement: bool):
        if self.is_running():
            return None
        valid, message = self.validate_inputs(require_cif=include_refinement)
        if not valid:
            QMessageBox.warning(self, "Invalid Phase 12 setup", message)
            return None
        records = self._records()
        reference_uid = str(self.reference_dataset.currentData())
        comparison_settings = {
            "reference_uid": reference_uid,
            "normalization": self.normalization.currentText(),
            "reference_window": (
                (self.window_min.value(), self.window_max.value())
                if self.normalization.currentText() == "Reference-window intensity"
                else None
            ),
            "wavelength_angstrom": self.wavelength.value(),
            "peak_match_tolerance_deg": self.peak_tolerance.value(),
            "peak_detection_sensitivity": self.smart_sensitivity.currentText(),
        }
        phase_specs = self._phase_specs() if include_refinement else []
        profile = deepcopy(
            getattr(self.main_window, "active_instrument_profile", None)
            if self.use_calibration.isChecked()
            else None
        )
        refinement_settings = {
            "wavelength_angstrom": self.wavelength.value(),
            "background_order": self.background_order.value(),
            "weighting": self.weighting.currentText(),
            "refine_zero_shift": self.refine_zero.isChecked(),
            "refine_profile": self.refine_profile.isChecked(),
            "refine_eta": False,
            "profile_model": self.profile_model.currentText(),
            "refine_lorentzian_width": self.profile_model.currentText() in {"TCH pseudo-Voigt", "Lorentzian X-Y"},
            "refine_asymmetry": self.profile_model.currentText() == "Split pseudo-Voigt",
            "initial_u": float((profile or {}).get("caglioti_u", 0.005)),
            "initial_v": float((profile or {}).get("caglioti_v", 0.0)),
            "initial_w": float((profile or {}).get("caglioti_w", 0.02)),
            "initial_eta": float((profile or {}).get("eta", 0.5)),
            "initial_x": float((profile or {}).get("lorentzian_x", 0.0)),
            "initial_y": float((profile or {}).get("lorentzian_y", 0.0)),
            "initial_asymmetry": float((profile or {}).get("axial_asymmetry", 0.0)),
            "initial_axial_sh_over_l": float((profile or {}).get("axial_sh_over_l", 0.0)),
            "cell_tolerance_percent": self.cell_tolerance.value(),
            "maximum_nonlinear_evaluations": self.max_evaluations.value(),
            "maximum_optimization_points": self.optimization_points.value(),
            "robust_loss": "soft_l1",
            "use_staged_refinement": self.staged.isChecked(),
            "intensity_provenance": "raw_counts",
            "generated_pattern_scaling": GENERATED_PATTERN_SCALING[0],
            "instrument_profile": profile,
            "allow_profile_extrapolation": False,
        }
        series_settings = {
            "mode": self.refinement_mode.currentText(),
            "reference_uid": reference_uid,
            "maximum_rwp_percent": self.maximum_rwp.value(),
            "maximum_parameter_correlation": self.maximum_correlation.value(),
            "continue_on_error": True,
        }
        worker = DopingSeriesWorker(
            records,
            comparison_settings,
            phase_specs,
            refinement_settings,
            series_settings,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(lambda *_: thread.quit())
        worker.failed.connect(lambda *_: thread.quit())
        worker.cancelled.connect(lambda: thread.quit())
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        self._running_phase_specs = deepcopy(phase_specs)
        self.compare_button.setEnabled(False)
        self.refine_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.progress.setRange(0, DopingSeriesWorker.PROGRESS_MAXIMUM)
        self.progress.setValue(0)
        self.progress.setFormat("Working — %p%")
        thread.start()
        return None

    def is_running(self):
        return self._thread is not None and self._thread.isRunning()

    def cancel_analysis(self):
        if self._worker is not None:
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.progress_label.setText("Cancellation requested; stopping at a safe dataset boundary.")

    def _on_progress(self, done, total, message):
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        percent = 100.0 * done / total
        scaled = int(round(percent * DopingSeriesWorker.PROGRESS_MAXIMUM / 100.0))
        self.progress.setRange(0, DopingSeriesWorker.PROGRESS_MAXIMUM)
        self.progress.setValue(scaled)
        self.progress.setFormat("Working — %p%")
        status = f"{percent:.1f}% — {message}"
        self.progress_label.setText(status)
        self.progress_label.setToolTip(status)

    def _on_finished(self, result):
        self.progress.setValue(DopingSeriesWorker.FINALIZING)
        self.progress.setFormat("Finalizing — %p%")
        finalizing = "99.0% — Finalizing result tables, plots, and project records"
        self.progress_label.setText(finalizing)
        self.progress_label.setToolTip(finalizing)
        self.progress.repaint()
        self.progress_label.repaint()
        result["phase_specifications"] = [
            {
                "structure": deepcopy(spec.structure),
                "name": spec.name,
                "refine_cell": spec.refine_cell,
                "refine_biso": spec.refine_biso,
            }
            for spec in self._running_phase_specs
        ]
        self.series_result = result
        self.populate_result()
        for uid, refinement in result.get("refinement", {}).get("results_by_uid", {}).items():
            if hasattr(self.main_window, "_record_scientific_result"):
                self.main_window._record_scientific_result(
                    "refinement",
                    uid,
                    {"rietveld": refinement, "doping_series": {"series_value": refinement.get("series_value")}},
                    reason="Completed Phase 12 series refinement",
                )
        comparison = result.get("comparison", {})
        refinement = result.get("refinement", {})
        self.progress.setValue(self.progress.maximum())
        self.progress.setFormat("Complete — %p%")
        self.progress_label.setText(
            f"Compared {comparison.get('dataset_count', 0)} pattern(s), tracked {comparison.get('peak_track_count', 0)} peak family/families, "
            f"and completed {refinement.get('completed_count', 0)} refinement(s)."
        )
        self.export_button.setEnabled(True)

    def _on_failed(self, traceback_text: str):
        final = traceback_text.strip().splitlines()[-1]
        self.progress.setFormat("Failed — %p%")
        self.progress_label.setText(final)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Phase 12 analysis failed")
        box.setText(final)
        box.setDetailedText(traceback_text)
        box.exec()

    def _on_cancelled(self):
        self.progress.setFormat("Cancelled — %p%")
        self.progress_label.setText("Phase 12 analysis cancelled; previous result retained.")

    def _thread_finished(self):
        self.compare_button.setEnabled(True)
        self.refine_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._worker = None
        self._thread = None

    @staticmethod
    def _fmt(value, digits=7):
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}g}"
        except (TypeError, ValueError):
            return str(value)

    def populate_result(self):
        result = self.series_result
        self.plot_widget.set_result(result)
        observations = [] if not result else result.get("comparison", {}).get("peak_observations", [])
        self.peak_table.setRowCount(len(observations))
        for row, observation in enumerate(observations):
            values = [
                observation.get("track_id"), observation.get("dataset_name"), self._fmt(observation.get("series_value")),
                self._fmt(observation.get("position_deg")), self._fmt(observation.get("position_standard_error_deg")),
                self._fmt(observation.get("position_shift_deg")), self._fmt(observation.get("d_spacing_angstrom")),
                self._fmt(observation.get("d_spacing_shift_angstrom")), self._fmt(observation.get("fwhm_deg")),
                self._fmt(observation.get("intensity")), observation.get("reference_classification"), observation.get("source"),
            ]
            for column, value in enumerate(values):
                self.peak_table.setItem(row, column, QTableWidgetItem(str(value)))
        trends = [] if not result else result.get("refinement", {}).get("trends", [])
        self.trend_table.setRowCount(len(trends))
        for row, trend in enumerate(trends):
            values = [
                trend.get("dataset_name"), trend.get("dopant"), self._fmt(trend.get("series_value")), trend.get("status"),
                self._fmt(trend.get("rwp_percent")), self._fmt(trend.get("reduced_chi_square")),
                self._fmt(trend.get("cell_a_angstrom")), self._fmt(trend.get("cell_b_angstrom")),
                self._fmt(trend.get("cell_c_angstrom")), self._fmt(trend.get("cell_volume_angstrom3")),
                self._fmt(trend.get("zero_shift_deg")), self._fmt(trend.get("primary_phase_fraction_percent")),
                self._fmt(trend.get("maximum_absolute_correlation")), self._fmt(trend.get("elapsed_seconds")),
            ]
            for column, value in enumerate(values):
                self.trend_table.setItem(row, column, QTableWidgetItem(str(value)))
        refinement = {} if not result else result.get("refinement", {})
        result_by_uid = refinement.get("results_by_uid", {})
        failures = {row["dataset_uid"]: row for row in refinement.get("failures", [])}
        dataset_profiles = [] if not result else result.get("comparison", {}).get("profiles", [])
        self.run_table.setRowCount(len(dataset_profiles))
        for row, profile in enumerate(dataset_profiles):
            uid = profile["dataset_uid"]
            fit = result_by_uid.get(uid)
            failure = failures.get(uid, {})
            gate = {} if fit is None else fit.get("series_quality_gate", {})
            values = [
                profile.get("dataset_name"), self._fmt(profile.get("series_value")),
                "Completed" if fit else ("Comparison only" if not refinement else "Failed"),
                "—" if fit is None else fit.get("series_initialization"),
                "—" if fit is None else self._fmt(fit.get("rwp_percent")),
                "—" if fit is None else gate.get("status"),
                "; ".join(gate.get("reasons", [])),
                "—" if fit is None else "; ".join(fit.get("warnings", [])),
                failure.get("error", ""),
            ]
            for column, value in enumerate(values):
                self.run_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _saved_phase_specs(self):
        if self.series_result:
            values = self.series_result.get("phase_specifications", [])
            if values:
                return [
                    RietveldPhaseSpec(
                        structure=deepcopy(row["structure"]),
                        name=row.get("name"),
                        refine_cell=bool(row.get("refine_cell", True)),
                        refine_biso=bool(row.get("refine_biso", False)),
                    )
                    for row in values
                ]
        return self._phase_specs()

    def export_result(self):
        if not self.series_result:
            return
        directory = QFileDialog.getExistingDirectory(self, "Export Phase 12 series bundle")
        if not directory:
            return
        try:
            exported = export_doping_series_bundle(
                directory,
                self.series_result,
                self._saved_phase_specs(),
                software_version=APP_VERSION,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Phase 12 export failed", str(exc))
            return
        self.main_window.statusBar().showMessage(
            f"Phase 12 export complete: {exported['file_count']} files; manifest {exported['manifest_txt']}"
        )

    def get_state(self):
        self._capture_series_table()
        return {
            "series_state": deepcopy(self._series_state),
            "structures": deepcopy(self.structures),
            "cif_state": deepcopy(self._cif_state()),
            "series_result": deepcopy(self.series_result),
            "settings": {
                "reference_uid": self.reference_dataset.currentData(),
                "normalization": self.normalization.currentText(),
                "window_min": self.window_min.value(), "window_max": self.window_max.value(),
                "wavelength": self.wavelength.value(), "peak_tolerance": self.peak_tolerance.value(),
                "peak_source_mode": self.peak_source_mode.currentText(),
                "smart_sensitivity": self.smart_sensitivity.currentText(),
                "refinement_mode": self.refinement_mode.currentText(),
                "weighting": self.weighting.currentText(), "profile_model": self.profile_model.currentText(),
                "background_order": self.background_order.value(), "refine_zero": self.refine_zero.isChecked(),
                "refine_profile": self.refine_profile.isChecked(), "staged": self.staged.isChecked(),
                "use_calibration": self.use_calibration.isChecked(), "cell_tolerance": self.cell_tolerance.value(),
                "max_evaluations": self.max_evaluations.value(), "optimization_points": self.optimization_points.value(),
                "maximum_rwp": self.maximum_rwp.value(), "maximum_correlation": self.maximum_correlation.value(),
                "trend_metric": self.trend_metric.currentText(),
            },
        }

    def set_state(self, state):
        state = state if isinstance(state, dict) else {}
        self._series_state = deepcopy(state.get("series_state", {}))
        self.structures = deepcopy(state.get("structures", []))
        self.series_result = deepcopy(state.get("series_result"))
        self._populate_cif_table(state.get("cif_state", {}))
        self.refresh_datasets()
        settings = state.get("settings", {})
        self.normalization.setCurrentText(settings.get("normalization", NORMALIZATION_MODES[0]))
        self.window_min.setValue(settings.get("window_min", 20.0)); self.window_max.setValue(settings.get("window_max", 30.0))
        self.wavelength.setValue(settings.get("wavelength", 1.5406)); self.peak_tolerance.setValue(settings.get("peak_tolerance", 0.15))
        self.peak_source_mode.setCurrentText(settings.get("peak_source_mode", PEAK_SOURCE_MODES[0]))
        self.smart_sensitivity.setCurrentText(settings.get("smart_sensitivity", "Balanced"))
        self.refinement_mode.setCurrentText(settings.get("refinement_mode", "Sequential from reference"))
        self.weighting.setCurrentText(settings.get("weighting", RIETVELD_WEIGHTING[0])); self.profile_model.setCurrentText(settings.get("profile_model", "TCH pseudo-Voigt"))
        self.background_order.setValue(settings.get("background_order", 3)); self.refine_zero.setChecked(settings.get("refine_zero", True))
        self.refine_profile.setChecked(settings.get("refine_profile", True)); self.staged.setChecked(settings.get("staged", True))
        self.use_calibration.setChecked(settings.get("use_calibration", True)); self.cell_tolerance.setValue(settings.get("cell_tolerance", 3.0))
        self.max_evaluations.setValue(settings.get("max_evaluations", 100)); self.optimization_points.setValue(settings.get("optimization_points", 1500))
        self.maximum_rwp.setValue(settings.get("maximum_rwp", 30.0)); self.maximum_correlation.setValue(settings.get("maximum_correlation", 0.98))
        self.trend_metric.setCurrentText(settings.get("trend_metric", "Cell volume"))
        reference_uid = settings.get("reference_uid")
        index = self.reference_dataset.findData(reference_uid)
        if index >= 0:
            self.reference_dataset.setCurrentIndex(index)
        self.populate_result()
        self.export_button.setEnabled(bool(self.series_result))

    def set_project_state(self, _state):
        self.refresh_datasets()

    def set_dataset(self, _dataset):
        self.refresh_datasets()

    def remove_dataset(self, dataset_uid: str):
        uid = str(dataset_uid)
        self._series_state.pop(uid, None)
        if self.series_result:
            used = {
                str(row.get("dataset_uid"))
                for row in self.series_result.get("comparison", {}).get("profiles", [])
            }
            if uid in used:
                self.series_result = None
                self.export_button.setEnabled(False)
        self.refresh_datasets()
        self.populate_result()

    def refresh_results(self):
        self.refresh_datasets()
        self.populate_result()

    def apply_theme(self, theme_name: str):
        self.plot_widget.apply_theme(theme_name)
