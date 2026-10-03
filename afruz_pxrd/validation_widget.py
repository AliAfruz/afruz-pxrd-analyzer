from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .context_copy import install_label_copy_menu, install_table_copy_menu
from .metrology import EvidenceClass
from .validation_campaign import (
    REPLICA_TYPES,
    ValidationCampaignError,
    analyze_robustness_runs,
    analyze_validation_campaign,
    audit_qpa_result,
    export_reproducibility_package,
    generate_synthetic_campaign,
    normalize_record,
)
from .validation_plot import ValidationPlotWidget
from .widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox


RECORD_COLUMNS = [
    ("sample_id", "Sample"),
    ("dataset_name", "Dataset"),
    ("phase_name", "Phase"),
    ("known_wt_percent", "Known wt%"),
    ("measured_wt_percent", "Measured wt%"),
    ("measured_error_percent", "σ wt%"),
    ("engine", "Engine"),
    ("replica_type", "Replica type"),
    ("preparation_id", "Preparation"),
    ("specimen_id", "Specimen/repack"),
    ("scan_id", "Scan"),
    ("run_id", "Robustness run"),
    ("classification", "Classification"),
    ("notes", "Notes"),
]


class ValidationCampaignWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.records: list[dict] = []
        self.result: dict | None = None
        self.audit_results: list[dict] = []
        self.robustness_runs: list[dict] = []
        self.robustness_results: list[dict] = []
        self._updating = False
        self._build_ui()
        self.refresh_for_selected_dataset()

    def _build_ui(self):
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        heading = QLabel("Phase 8 — Scientific Ground Truth and Metrology")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        intro = QLabel(
            "Compare known and measured phase fractions across independent preparations, "
            "repacked specimens and repeat scans. Evidence is classified as software "
            "regression, certified-reference, reference-engine, or experimental validation. "
            "Passing campaign statistics alone never authorize a scientific claim."
        )
        intro.setWordWrap(True)
        intro.setObjectName("mutedLabel")
        install_label_copy_menu(intro)
        layout.addWidget(intro)

        self.setup_splitter = QSplitter(Qt.Horizontal)
        self.setup_splitter.setChildrenCollapsible(False)
        self.setup_splitter.setMaximumHeight(320)
        self.setup_splitter.addWidget(self._build_record_panel())
        self.setup_splitter.addWidget(self._build_control_scroll())
        self.setup_splitter.setSizes([950, 390])
        self.setup_splitter.setStretchFactor(0, 3)
        self.setup_splitter.setStretchFactor(1, 1)
        layout.addWidget(self.setup_splitter)

        action_row = QHBoxLayout()
        self.add_current_button = QPushButton("Add current QPA result")
        self.audit_current_button = QPushButton("Audit current QPA")
        self.capture_robustness_button = QPushButton("Capture robustness run")
        self.calculate_button = QPushButton("Calculate validation")
        self.calculate_button.setObjectName("primaryButton")
        self.export_button = QPushButton("Export reproducibility package")
        action_row.addWidget(self.add_current_button)
        action_row.addWidget(self.audit_current_button)
        action_row.addWidget(self.capture_robustness_button)
        action_row.addStretch(1)
        action_row.addWidget(self.calculate_button)
        action_row.addWidget(self.export_button)
        layout.addLayout(action_row)

        self.status_label = QLabel("Add validation records or load the synthetic demonstration.")
        self.status_label.setObjectName("mutedLabel")
        self.status_label.setWordWrap(False)
        self.status_label.setMinimumWidth(0)
        self.status_label.setMaximumHeight(24)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        install_label_copy_menu(self.status_label)
        layout.addWidget(self.status_label)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setMinimumSize(0, 220)
        self.plot = ValidationPlotWidget()
        self.tabs.addTab(self.plot, "Known vs Measured")
        self.error_plot = ValidationPlotWidget()
        self.tabs.addTab(self.error_plot, "Error Plot")
        self.tabs.addTab(self._summary_page(), "Campaign Summary")
        self.tabs.addTab(self._table_page(self._make_table([
            "Phase", "N", "Known mean", "Measured mean", "Bias", "MAE", "RMSE",
            "Maximum error", "Mean recovery", "Bias CI95 lower", "Bias CI95 upper",
        ], "phase_metrics_table")), "Phase Metrics")
        self.tabs.addTab(self._table_page(self._make_table([
            "Level", "Phase", "Group", "N", "Mean wt%", "SD wt%", "RSD %",
        ], "replica_table")), "Replica Precision")
        self.tabs.addTab(self._table_page(self._make_table([
            "Dataset", "Engine", "Status", "Phases", "Rwp %", "Max correlation",
            "Condition number", "Errors", "Warnings",
        ], "audit_table")), "Refinement Audits")
        self.tabs.addTab(self._table_page(self._make_table([
            "Phase", "Runs", "Mean wt%", "SD wt%", "Minimum", "Maximum", "Spread",
            "Relative spread %", "Threshold", "Status",
        ], "robustness_table")), "Robustness")
        self.tabs.addTab(self._table_page(self._make_table([
            "Phase", "Blank N", "Blank mean", "Blank SD", "Empirical LOD", "Empirical LOQ", "Method",
        ], "detection_table")), "LOD / LOQ")
        layout.addWidget(self.tabs, 1)

        note = QLabel(
            "Accuracy thresholds in this workspace are suggested starting points requiring "
            "optimization. Publication claims require traceable reference mixtures, suitable "
            "concentration ranges, independent preparations and review of the underlying GPX/CIF/instrument files."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        layout.addWidget(note)

        self.add_current_button.clicked.connect(self.add_current_qpa)
        self.audit_current_button.clicked.connect(self.audit_current_qpa)
        self.capture_robustness_button.clicked.connect(self.capture_robustness_run)
        self.calculate_button.clicked.connect(self.calculate_validation)
        self.export_button.clicked.connect(self.export_package)
        self.import_button.clicked.connect(self.import_csv)
        self.synthetic_button.clicked.connect(self.load_synthetic)
        self.remove_button.clicked.connect(self.remove_selected_records)
        self.clear_button.clicked.connect(self.clear_campaign)
        self.record_table.itemChanged.connect(self._table_changed)

    def _build_record_panel(self):
        group = QGroupBox("Validation campaign records")
        group.setMinimumWidth(0)
        layout = QVBoxLayout(group)
        self.record_table = QTableWidget(0, len(RECORD_COLUMNS))
        self.record_table.setHorizontalHeaderLabels([label for _, label in RECORD_COLUMNS])
        self._stabilize_table(self.record_table)
        header = self.record_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column, width in {
            1: 150, 2: 120, 3: 90, 4: 95, 5: 75, 6: 150, 7: 135,
            8: 100, 9: 110, 10: 65, 11: 100, 12: 180, 13: 220,
        }.items():
            self.record_table.setColumnWidth(column, width)
        self.record_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        install_table_copy_menu(self.record_table)
        layout.addWidget(self.record_table)
        return group

    def _build_control_scroll(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(560)
        content = QWidget()
        form = QFormLayout(content)
        form.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        self.campaign_name = QLineEdit("Afruz QPA validation campaign")
        self.minimum_records = NoWheelSpinBox()
        self.minimum_records.setRange(2, 10000)
        self.minimum_records.setValue(6)
        self.maximum_mae = NoWheelDoubleSpinBox()
        self.maximum_mae.setRange(0.01, 100.0)
        self.maximum_mae.setDecimals(4)
        self.maximum_mae.setValue(3.0)
        self.maximum_rmse = NoWheelDoubleSpinBox()
        self.maximum_rmse.setRange(0.01, 100.0)
        self.maximum_rmse.setDecimals(4)
        self.maximum_rmse.setValue(4.0)
        self.maximum_bias = NoWheelDoubleSpinBox()
        self.maximum_bias.setRange(0.01, 100.0)
        self.maximum_bias.setDecimals(4)
        self.maximum_bias.setValue(2.0)
        self.maximum_single_error = NoWheelDoubleSpinBox()
        self.maximum_single_error.setRange(0.01, 100.0)
        self.maximum_single_error.setDecimals(4)
        self.maximum_single_error.setValue(8.0)
        self.maximum_robustness_spread = NoWheelDoubleSpinBox()
        self.maximum_robustness_spread.setRange(0.01, 100.0)
        self.maximum_robustness_spread.setDecimals(4)
        self.maximum_robustness_spread.setValue(2.0)
        self.evidence_classification = NoWheelComboBox()
        self.evidence_classification.addItems([row.value for row in EvidenceClass])
        self.evidence_classification.setCurrentText(EvidenceClass.UNCLASSIFIED.value)
        self.replica_type = NoWheelComboBox()
        self.replica_type.addItems(list(REPLICA_TYPES))
        self.replica_type.setCurrentText("Independent preparation")
        self.preparation_id = QLineEdit("P1")
        self.specimen_id = QLineEdit("P1-S1")
        self.scan_id = QLineEdit("1")
        self.run_id = QLineEdit("default")
        self.include_inputs = QCheckBox("Include available project, patterns and CIF files")
        self.include_inputs.setChecked(True)

        self.import_button = QPushButton("Import validation CSV")
        self.synthetic_button = QPushButton("Load synthetic demonstration")
        self.remove_button = QPushButton("Remove selected rows")
        self.clear_button = QPushButton("Clear campaign")

        form.addRow("Campaign name", self.campaign_name)
        form.addRow("Minimum complete records", self.minimum_records)
        form.addRow("Maximum MAE (wt%)", self.maximum_mae)
        form.addRow("Maximum RMSE (wt%)", self.maximum_rmse)
        form.addRow("Maximum |bias| (wt%)", self.maximum_bias)
        form.addRow("Maximum single error (wt%)", self.maximum_single_error)
        form.addRow("Robustness spread (wt%)", self.maximum_robustness_spread)
        form.addRow("Evidence classification", self.evidence_classification)
        form.addRow("New-record replica type", self.replica_type)
        form.addRow("Preparation ID", self.preparation_id)
        form.addRow("Specimen/repack ID", self.specimen_id)
        form.addRow("Scan ID", self.scan_id)
        form.addRow("Robustness run ID", self.run_id)
        form.addRow(self.include_inputs)
        form.addRow(self.import_button)
        form.addRow(self.synthetic_button)
        form.addRow(self.remove_button)
        form.addRow(self.clear_button)
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _stabilize_table(table):
        table.setMinimumSize(0, 0)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.horizontalHeader().setMinimumSectionSize(48)
        table.horizontalHeader().setStretchLastSection(False)

    def _make_table(self, headers, attribute_name):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        self._stabilize_table(table)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        install_table_copy_menu(table)
        setattr(self, attribute_name, table)
        return table

    @staticmethod
    def _table_page(table):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(table)
        return page

    def _summary_page(self):
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        form = QFormLayout(content)
        self.summary_status = QLabel("—")
        self.summary_classification = QLabel("—")
        self.summary_classification.setWordWrap(True)
        self.summary_claim_status = QLabel("—")
        self.summary_claim_status.setWordWrap(True)
        self.summary_records = QLabel("—")
        self.summary_bias = QLabel("—")
        self.summary_mae = QLabel("—")
        self.summary_rmse = QLabel("—")
        self.summary_max_error = QLabel("—")
        self.summary_recovery = QLabel("—")
        self.summary_bland_altman = QLabel("—")
        self.summary_warnings = QLabel("—")
        self.summary_warnings.setWordWrap(True)
        for label in (
            self.summary_status, self.summary_classification, self.summary_claim_status,
            self.summary_records,
            self.summary_bias, self.summary_mae, self.summary_rmse,
            self.summary_max_error, self.summary_recovery,
            self.summary_bland_altman, self.summary_warnings,
        ):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setMinimumWidth(0)
            install_label_copy_menu(label)
        form.addRow("Campaign status", self.summary_status)
        form.addRow("Classification", self.summary_classification)
        form.addRow("Scientific claim", self.summary_claim_status)
        form.addRow("Complete records", self.summary_records)
        form.addRow("Bias", self.summary_bias)
        form.addRow("MAE", self.summary_mae)
        form.addRow("RMSE", self.summary_rmse)
        form.addRow("Maximum absolute error", self.summary_max_error)
        form.addRow("Mean recovery", self.summary_recovery)
        form.addRow("Bland–Altman", self.summary_bland_altman)
        form.addRow("Warnings", self.summary_warnings)
        page.setWidget(content)
        return page

    @staticmethod
    def _fmt(value, digits=7):
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}g}"
        except (TypeError, ValueError):
            return str(value)

    def _current_qpa_result(self):
        dataset = self.main_window.selected_dataset()
        widget = getattr(self.main_window, "validated_qpa_widget", None)
        if dataset is None or widget is None:
            return None, None
        result = widget.native_results_by_uid.get(dataset.uid)
        return dataset, result

    @staticmethod
    def _phase_fraction_rows(result):
        rows = []
        engine = str(result.get("engine", ""))
        for phase in result.get("phases", []):
            measured = phase.get("absolute_weight_percent")
            if measured is None:
                measured = phase.get("crystalline_weight_percent")
            if measured is None and phase.get("mass_fraction") is not None:
                measured = 100.0 * float(phase["mass_fraction"])
            uncertainty = phase.get("crystalline_weight_percent_error")
            if uncertainty is None and phase.get("mass_fraction_error") is not None:
                uncertainty = 100.0 * float(phase["mass_fraction_error"])
            rows.append({
                "phase_name": phase.get("phase_name") or phase.get("name"),
                "measured_wt_percent": measured,
                "measured_error_percent": uncertainty,
                "engine": engine,
                "classification": result.get("classification", ""),
            })
        return rows

    def add_current_qpa(self):
        dataset, result = self._current_qpa_result()
        if dataset is None or not result:
            QMessageBox.information(self, "No QPA result", "Run Phase 12 QPA for the selected dataset first.")
            return
        added = 0
        for phase in self._phase_fraction_rows(result):
            if phase["measured_wt_percent"] is None:
                continue
            self.records.append(normalize_record({
                "sample_id": dataset.name,
                "dataset_uid": dataset.uid,
                "dataset_name": dataset.name,
                "phase_name": phase["phase_name"],
                "known_wt_percent": None,
                "measured_wt_percent": phase["measured_wt_percent"],
                "measured_error_percent": phase["measured_error_percent"],
                "engine": phase["engine"],
                "classification": phase["classification"],
                "replica_type": self.replica_type.currentText(),
                "preparation_id": self.preparation_id.text().strip() or dataset.name,
                "specimen_id": self.specimen_id.text().strip() or dataset.name,
                "scan_id": self.scan_id.text().strip() or "1",
                "run_id": self.run_id.text().strip() or "default",
                "notes": "Enter the independently known gravimetric/reference wt% before calculation.",
            }, len(self.records)))
            added += 1
        self.populate_records()
        self._set_status(f"Added {added} phase records. Enter the known wt% values in the table.")

    def audit_current_qpa(self):
        dataset, result = self._current_qpa_result()
        if dataset is None or not result:
            QMessageBox.information(self, "No QPA result", "Run Phase 12 QPA for the selected dataset first.")
            return
        audit = audit_qpa_result(result, dataset_name=dataset.name)
        self.audit_results = [row for row in self.audit_results if row.get("dataset_name") != dataset.name]
        self.audit_results.append(audit)
        if hasattr(self.main_window, "_record_scientific_result"):
            self.main_window._record_scientific_result(
                "validation",
                dataset.uid,
                {"audit_results": [audit]},
                reason="Completed validation audit",
            )
        self.populate_audits()
        self._set_status(f"Current result audit: {audit['status']}.")
        self.tabs.setCurrentIndex(5)

    def capture_robustness_run(self):
        dataset, result = self._current_qpa_result()
        if dataset is None or not result:
            QMessageBox.information(self, "No QPA result", "Run Phase 12 QPA first, then capture each starting-condition result.")
            return
        run_id = self.run_id.text().strip() or f"run-{len(self.robustness_runs)+1}"
        self.robustness_runs.append({
            "run_id": run_id,
            "dataset_uid": dataset.uid,
            "dataset_name": dataset.name,
            "result": deepcopy(result),
        })
        self.robustness_results = analyze_robustness_runs(
            self.robustness_runs,
            maximum_spread_wt_percent=self.maximum_robustness_spread.value(),
        )
        if hasattr(self.main_window, "_record_scientific_result"):
            self.main_window._record_scientific_result(
                "validation",
                dataset.uid,
                {"robustness_results": self.robustness_results},
                reason="Updated robustness validation",
            )
        self.populate_robustness()
        self._set_status(f"Captured robustness run '{run_id}'. Total runs: {len(self.robustness_runs)}.")
        self.tabs.setCurrentIndex(6)

    def import_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Import Phase 8 validation records", "", "CSV (*.csv);;All files (*)")
        if not filename:
            return
        try:
            with Path(filename).open("r", newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
        except Exception as exc:
            QMessageBox.critical(self, "Validation CSV import failed", str(exc))
            return
        start = len(self.records)
        self.records.extend(normalize_record(row, start + index) for index, row in enumerate(rows))
        self.populate_records()
        self._set_status(f"Imported {len(rows)} validation rows.")

    def load_synthetic(self):
        if self.records:
            response = QMessageBox.question(self, "Replace campaign", "Replace the current records with the synthetic demonstration?")
            if response != QMessageBox.Yes:
                return
        self.records = generate_synthetic_campaign()
        self.audit_results = []
        self.robustness_runs = []
        self.robustness_results = []
        self.campaign_name.setText("Phase 8 synthetic binary validation demonstration")
        self.evidence_classification.setCurrentText(EvidenceClass.SOFTWARE_REGRESSION.value)
        self.populate_records()
        self.calculate_validation()

    def remove_selected_records(self):
        selected = sorted({index.row() for index in self.record_table.selectionModel().selectedRows()}, reverse=True)
        for row in selected:
            if 0 <= row < len(self.records):
                self.records.pop(row)
        self.populate_records()

    def clear_campaign(self):
        self.records = []
        self.result = None
        self.audit_results = []
        self.robustness_runs = []
        self.robustness_results = []
        self.populate_records()
        self.populate_result(None)
        self._set_status("Validation campaign cleared.")

    def _table_changed(self, item):
        if self._updating:
            return
        row = item.row()
        column = item.column()
        if not (0 <= row < len(self.records)):
            return
        key = RECORD_COLUMNS[column][0]
        value = item.text().strip()
        if key in {"known_wt_percent", "measured_wt_percent", "measured_error_percent"}:
            try:
                self.records[row][key] = None if value == "" else float(value)
            except ValueError:
                self.records[row][key] = None
        else:
            self.records[row][key] = value
        self.result = None

    def populate_records(self):
        self._updating = True
        try:
            self.record_table.setRowCount(len(self.records))
            for row_index, raw in enumerate(self.records):
                row = normalize_record(raw, row_index)
                self.records[row_index] = row
                for column, (key, _) in enumerate(RECORD_COLUMNS):
                    value = row.get(key)
                    text = "" if value is None else self._fmt(value) if key in {
                        "known_wt_percent", "measured_wt_percent", "measured_error_percent"
                    } else str(value)
                    self.record_table.setItem(row_index, column, QTableWidgetItem(text))
        finally:
            self._updating = False

    def _thresholds(self):
        return {
            "minimum_records": self.minimum_records.value(),
            "maximum_mae_wt_percent": self.maximum_mae.value(),
            "maximum_rmse_wt_percent": self.maximum_rmse.value(),
            "maximum_absolute_bias_wt_percent": self.maximum_bias.value(),
            "maximum_single_error_wt_percent": self.maximum_single_error.value(),
            "maximum_robustness_spread_wt_percent": self.maximum_robustness_spread.value(),
            "note": "Suggested starting points requiring optimization for the intended method and concentration range.",
        }

    def calculate_validation(self):
        self.robustness_results = analyze_robustness_runs(
            self.robustness_runs,
            maximum_spread_wt_percent=self.maximum_robustness_spread.value(),
        )
        try:
            dataset_ids = {
                str(row.get("dataset_uid", ""))
                for row in self.records
                if str(row.get("dataset_uid", ""))
            }
            registry = self.main_window.project_state.metrology_registry
            assessment_ids = [
                row.assessment_id
                for row in registry.assessments.values()
                if row.dataset_id in dataset_ids and row.passed
            ]
            self.result = analyze_validation_campaign(
                self.records,
                campaign_name=self.campaign_name.text().strip() or "Afruz QPA validation campaign",
                thresholds=self._thresholds(),
                audit_results=self.audit_results,
                robustness_results=self.robustness_results,
                evidence_classification=self.evidence_classification.currentText(),
                metrology_assessment_ids=assessment_ids,
            )
        except ValidationCampaignError as exc:
            QMessageBox.critical(self, "Validation calculation failed", str(exc))
            return
        self.populate_result(self.result)
        self._set_status(
            f"Validation completed: {self.result['status']} — "
            f"MAE {self.result['overall_metrics']['mae_wt_percent']:.4g} wt%."
        )
        self.tabs.setCurrentIndex(2)

    def _set_status(self, message):
        compact = " ".join(str(message).split())
        self.status_label.setText(compact if len(compact) <= 140 else compact[:139].rstrip() + "…")
        self.status_label.setToolTip(str(message))

    def populate_result(self, result):
        self.plot.set_result(result, "Agreement")
        self.error_plot.set_result(result, "Error")
        if not result:
            for label in (
                self.summary_status, self.summary_classification, self.summary_claim_status,
                self.summary_records,
                self.summary_bias, self.summary_mae, self.summary_rmse,
                self.summary_max_error, self.summary_recovery,
                self.summary_bland_altman, self.summary_warnings,
            ):
                label.setText("—")
            for table in (
                self.phase_metrics_table, self.replica_table, self.audit_table,
                self.robustness_table, self.detection_table,
            ):
                table.setRowCount(0)
            return
        overall = result["overall_metrics"]
        bland = result["bland_altman"]
        self.summary_status.setText(result["status"])
        self.summary_classification.setText(result["classification"])
        self.summary_claim_status.setText(
            result.get("scientific_claim_status", "Not validated for a publication claim")
        )
        self.summary_records.setText(str(overall["record_count"]))
        self.summary_bias.setText(
            f"{self._fmt(overall['bias_wt_percent'])} wt% "
            f"(95% CI {self._fmt(overall['bias_ci95_lower_wt_percent'])} to "
            f"{self._fmt(overall['bias_ci95_upper_wt_percent'])})"
        )
        self.summary_mae.setText(f"{self._fmt(overall['mae_wt_percent'])} wt%")
        self.summary_rmse.setText(f"{self._fmt(overall['rmse_wt_percent'])} wt%")
        self.summary_max_error.setText(f"{self._fmt(overall['maximum_absolute_error_wt_percent'])} wt%")
        self.summary_recovery.setText(f"{self._fmt(overall['mean_recovery_percent'])}%")
        self.summary_bland_altman.setText(
            f"mean {self._fmt(bland['mean_difference_wt_percent'])} wt%; "
            f"limits {self._fmt(bland['lower_limit_wt_percent'])} to "
            f"{self._fmt(bland['upper_limit_wt_percent'])} wt%"
        )
        self.summary_warnings.setText("\n".join(result.get("warnings", [])) or "None")
        self.populate_phase_metrics(result.get("phase_metrics", []))
        self.populate_replica_metrics(result.get("replica_metrics", []))
        self.populate_audits()
        self.populate_robustness()
        self.populate_detection(result.get("detection_limits", []))

    def populate_phase_metrics(self, rows=None):
        rows = rows if rows is not None else ([] if not self.result else self.result.get("phase_metrics", []))
        self.phase_metrics_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [
                row.get("group"), row.get("record_count"), self._fmt(row.get("known_mean_wt_percent")),
                self._fmt(row.get("measured_mean_wt_percent")), self._fmt(row.get("bias_wt_percent")),
                self._fmt(row.get("mae_wt_percent")), self._fmt(row.get("rmse_wt_percent")),
                self._fmt(row.get("maximum_absolute_error_wt_percent")), self._fmt(row.get("mean_recovery_percent")),
                self._fmt(row.get("bias_ci95_lower_wt_percent")), self._fmt(row.get("bias_ci95_upper_wt_percent")),
            ]
            for j, value in enumerate(values):
                self.phase_metrics_table.setItem(i, j, QTableWidgetItem(str(value)))

    def populate_replica_metrics(self, rows=None):
        rows = rows if rows is not None else ([] if not self.result else self.result.get("replica_metrics", []))
        self.replica_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [row.get("level"), row.get("phase_name"), row.get("group_id"), row.get("replicate_count"),
                      self._fmt(row.get("mean_wt_percent")), self._fmt(row.get("sd_wt_percent")), self._fmt(row.get("rsd_percent"))]
            for j, value in enumerate(values):
                self.replica_table.setItem(i, j, QTableWidgetItem(str(value)))

    def populate_audits(self):
        rows = self.audit_results
        self.audit_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [row.get("dataset_name"), row.get("engine"), row.get("status"), row.get("phase_count"),
                      self._fmt(row.get("rwp_percent")), self._fmt(row.get("maximum_parameter_correlation")),
                      self._fmt(row.get("parameter_condition_number")), "; ".join(row.get("errors", [])) or "—",
                      "; ".join(row.get("warnings", [])) or "—"]
            for j, value in enumerate(values):
                self.audit_table.setItem(i, j, QTableWidgetItem(str(value)))

    def populate_robustness(self):
        rows = self.robustness_results
        self.robustness_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [row.get("phase_name"), row.get("run_count"), self._fmt(row.get("mean_wt_percent")),
                      self._fmt(row.get("sd_wt_percent")), self._fmt(row.get("minimum_wt_percent")),
                      self._fmt(row.get("maximum_wt_percent")), self._fmt(row.get("spread_wt_percent")),
                      self._fmt(row.get("relative_spread_percent")), self._fmt(row.get("threshold_wt_percent")), row.get("status")]
            for j, value in enumerate(values):
                self.robustness_table.setItem(i, j, QTableWidgetItem(str(value)))

    def populate_detection(self, rows):
        self.detection_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [row.get("phase_name"), row.get("blank_count"), self._fmt(row.get("blank_mean_wt_percent")),
                      self._fmt(row.get("blank_sd_wt_percent")), self._fmt(row.get("empirical_lod_wt_percent")),
                      self._fmt(row.get("empirical_loq_wt_percent")), row.get("method")]
            for j, value in enumerate(values):
                self.detection_table.setItem(i, j, QTableWidgetItem(str(value)))

    def _files_for_package(self):
        if not self.include_inputs.isChecked():
            return []
        files = []
        project = getattr(self.main_window, "current_project", None)
        if project and Path(project).exists():
            files.append(Path(project))
        for dataset in getattr(self.main_window, "datasets", []):
            if dataset.source_path and Path(dataset.source_path).exists():
                files.append(Path(dataset.source_path))
        rietveld = getattr(self.main_window, "rietveld_widget", None)
        if rietveld is not None:
            for structure in rietveld.included_structures():
                source = structure.get("source_path")
                if source and Path(source).exists():
                    files.append(Path(source))
        unique = []
        seen = set()
        for path in files:
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                unique.append(path)
        return unique

    def export_package(self):
        if not self.result:
            self.calculate_validation()
            if not self.result:
                return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Phase 8 reproducibility package",
            "afruz_phase8_validation_package.zip", "ZIP archive (*.zip)"
        )
        if not filename:
            return
        try:
            output = export_reproducibility_package(
                filename,
                self.result,
                include_files=self._files_for_package(),
                additional_metadata={
                    "selected_dataset": getattr(self.main_window.selected_dataset(), "name", None),
                    "project_file": str(getattr(self.main_window, "current_project", "") or ""),
                },
            )
        except Exception as exc:
            QMessageBox.critical(self, "Validation package export failed", str(exc))
            return
        self._set_status(f"Reproducibility package exported: {output}")
        self.main_window.statusBar().showMessage("Phase 8 reproducibility package exported.")

    def refresh_for_selected_dataset(self):
        dataset = self.main_window.selected_dataset()
        if dataset is not None:
            self.status_label.setToolTip(f"Selected dataset: {dataset.name}")

    def remove_dataset(self, dataset_uid: str):
        self.records = [row for row in self.records if str(row.get("dataset_uid", "")) != str(dataset_uid)]
        self.robustness_runs = [row for row in self.robustness_runs if str(row.get("dataset_uid", "")) != str(dataset_uid)]
        self.populate_records()

    def apply_theme(self, theme_name):
        self.plot.apply_theme(theme_name)
        self.error_plot.apply_theme(theme_name)

    def get_state(self):
        return {
            "records": deepcopy(self.records),
            "result": deepcopy(self.result),
            "audit_results": deepcopy(self.audit_results),
            "robustness_runs": deepcopy(self.robustness_runs),
            "robustness_results": deepcopy(self.robustness_results),
            "campaign_name": self.campaign_name.text(),
            "thresholds": self._thresholds(),
            "evidence_classification": self.evidence_classification.currentText(),
            "replica_type": self.replica_type.currentText(),
            "preparation_id": self.preparation_id.text(),
            "specimen_id": self.specimen_id.text(),
            "scan_id": self.scan_id.text(),
            "run_id": self.run_id.text(),
            "include_inputs": self.include_inputs.isChecked(),
            "splitter_sizes": self.setup_splitter.sizes(),
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        self.records = deepcopy(state.get("records", []))
        self.result = deepcopy(state.get("result")) if isinstance(state.get("result"), dict) else None
        self.audit_results = deepcopy(state.get("audit_results", []))
        self.robustness_runs = deepcopy(state.get("robustness_runs", []))
        self.robustness_results = deepcopy(state.get("robustness_results", []))
        self.campaign_name.setText(state.get("campaign_name", "Afruz QPA validation campaign"))
        thresholds = state.get("thresholds", {})
        self.minimum_records.setValue(thresholds.get("minimum_records", 6))
        self.maximum_mae.setValue(thresholds.get("maximum_mae_wt_percent", 3.0))
        self.maximum_rmse.setValue(thresholds.get("maximum_rmse_wt_percent", 4.0))
        self.maximum_bias.setValue(thresholds.get("maximum_absolute_bias_wt_percent", 2.0))
        self.maximum_single_error.setValue(thresholds.get("maximum_single_error_wt_percent", 8.0))
        self.maximum_robustness_spread.setValue(thresholds.get("maximum_robustness_spread_wt_percent", 2.0))
        self.evidence_classification.setCurrentText(
            state.get("evidence_classification", EvidenceClass.UNCLASSIFIED.value)
        )
        self.replica_type.setCurrentText(state.get("replica_type", "Independent preparation"))
        self.preparation_id.setText(state.get("preparation_id", "P1"))
        self.specimen_id.setText(state.get("specimen_id", "P1-S1"))
        self.scan_id.setText(state.get("scan_id", "1"))
        self.run_id.setText(state.get("run_id", "default"))
        self.include_inputs.setChecked(state.get("include_inputs", True))
        if state.get("splitter_sizes"):
            self.setup_splitter.setSizes([int(value) for value in state["splitter_sizes"]])
        self.populate_records()
        self.populate_result(self.result)
