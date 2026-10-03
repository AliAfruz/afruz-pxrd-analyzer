from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
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
from .text_export import write_mapping_txt, write_table_txt
from .validated_qpa import QPAValidationError, audit_structure, compute_hill_howard_qpa
from .widgets import NoWheelComboBox, NoWheelDoubleSpinBox


class ValidatedQPAWidget(QWidget):
    """Transparent in-process crystallographic QPA audit and calculation."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.native_results_by_uid: dict[str, dict] = {}
        self._build_ui()
        self.refresh_from_rietveld()

    def _build_ui(self):
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        heading = QLabel("Crystallographic QPA — Native Hill–Howard Audit")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        description = QLabel(
            "Audit CIF chemistry and unit-cell mass, then calculate transparent "
            "Hill–Howard crystalline fractions from the current structure-constrained "
            "refinement. The calculation runs entirely inside Afruz."
        )
        description.setWordWrap(True)
        description.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        description.setObjectName("mutedLabel")
        layout.addWidget(description)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setMaximumHeight(315)
        splitter.addWidget(self._audit_panel())
        splitter.addWidget(self._control_panel())
        splitter.setSizes([850, 380])
        self.setup_splitter = splitter
        layout.addWidget(splitter)

        actions = QHBoxLayout()
        self.calculate_button = QPushButton("Calculate native QPA")
        self.calculate_button.setObjectName("primaryButton")
        self.export_button = QPushButton("Export QPA result")
        actions.addWidget(self.calculate_button)
        actions.addStretch(1)
        actions.addWidget(self.export_button)
        layout.addLayout(actions)

        self.tabs = QTabWidget()
        self.fraction_table = QTableWidget(0, 11)
        self.fraction_table.setHorizontalHeaderLabels([
            "Phase", "Engine", "Scale", "Scale σ", "Cell mass", "Volume", "ZMV",
            "Crystalline wt%", "σ wt%", "Absolute wt%", "Classification",
        ])
        self._stabilize_table(self.fraction_table)
        self.fraction_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.fraction_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        install_table_copy_menu(self.fraction_table)
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(self.fraction_table)
        self.tabs.addTab(page, "Weight Fractions")

        self.summary = QLabel("No QPA result.")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        install_label_copy_menu(self.summary)
        summary_scroll = QScrollArea()
        summary_scroll.setWidgetResizable(True)
        summary_scroll.setWidget(self.summary)
        self.tabs.addTab(summary_scroll, "Scientific Status")
        layout.addWidget(self.tabs, 1)

        warning = QLabel(
            "This is a transparent native calculation. Publication claims still require "
            "validated structures, suitable specimen preparation, microabsorption and "
            "preferred-orientation assessment, uncertainty review, and certified-mixture recovery."
        )
        warning.setWordWrap(True)
        warning.setObjectName("mutedLabel")
        layout.addWidget(warning)

        self.calculate_button.clicked.connect(self.calculate_native_preview)
        self.export_button.clicked.connect(self.export_result)
        self.internal_standard.currentTextChanged.connect(self._refresh_known_standard_state)

    @staticmethod
    def _stabilize_table(table):
        table.setMinimumSize(0, 0)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)

    def _audit_panel(self):
        group = QGroupBox("Crystallographic QPA audit")
        layout = QVBoxLayout(group)
        self.audit_table = QTableWidget(0, 10)
        self.audit_table.setHorizontalHeaderLabels([
            "Phase", "Status", "Formula", "Atoms/cell", "Cell mass", "Volume", "ZMV",
            "Inferred Z", "Errors", "Warnings",
        ])
        self._stabilize_table(self.audit_table)
        self.audit_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.audit_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        install_table_copy_menu(self.audit_table)
        layout.addWidget(self.audit_table)
        return group

    def _control_panel(self):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        form = QFormLayout(content)
        self.internal_standard = NoWheelComboBox()
        self.internal_standard.addItem("None")
        self.known_standard_percent = NoWheelDoubleSpinBox()
        self.known_standard_percent.setRange(0.001, 99.999)
        self.known_standard_percent.setDecimals(5)
        self.known_standard_percent.setValue(10.0)
        self.known_standard_percent.setEnabled(False)
        native_note = QLabel(
            "No external refinement engine is called. Scale factors and covariance are read "
            "from the current Afruz structure-constrained refinement."
        )
        native_note.setWordWrap(True)
        native_note.setObjectName("mutedLabel")
        form.addRow(native_note)
        form.addRow("Internal standard", self.internal_standard)
        form.addRow("Known wt% in total mixture", self.known_standard_percent)
        area.setWidget(content)
        return area

    def _refresh_known_standard_state(self):
        self.known_standard_percent.setEnabled(self.internal_standard.currentText() != "None")

    def _current_dataset(self):
        return self.main_window.selected_dataset()

    def _phase_structures(self):
        widget = getattr(self.main_window, "rietveld_widget", None)
        return [] if widget is None else widget.included_structures()

    def _rietveld_result(self):
        dataset = self._current_dataset()
        widget = getattr(self.main_window, "rietveld_widget", None)
        if dataset is None or widget is None:
            return None
        return widget.results_by_uid.get(dataset.uid)

    @staticmethod
    def _fmt(value, digits=8):
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}g}"
        except (TypeError, ValueError):
            return str(value)

    def refresh_from_rietveld(self):
        structures = self._phase_structures()
        audits = [audit_structure(s, s.get("data_name", f"Phase {i+1}")) for i, s in enumerate(structures)]
        self.audit_table.setRowCount(len(audits))
        current = self.internal_standard.currentText() if hasattr(self, "internal_standard") else "None"
        self.internal_standard.blockSignals(True)
        self.internal_standard.clear()
        self.internal_standard.addItem("None")
        for row_index, audit in enumerate(audits):
            self.internal_standard.addItem(audit["phase_name"])
            values = [
                audit["phase_name"], audit["status"], audit["formula"] or "—",
                audit["expanded_atom_count"], self._fmt(audit["unit_cell_mass_g_mol"]),
                self._fmt(audit["unit_cell_volume_angstrom3"]), self._fmt(audit["zmv"]),
                audit["inferred_z"] if audit["inferred_z"] is not None else "—",
                "; ".join(audit["errors"]) or "—", "; ".join(audit["warnings"]) or "—",
            ]
            for column, value in enumerate(values):
                self.audit_table.setItem(row_index, column, QTableWidgetItem(str(value)))
        index = self.internal_standard.findText(current)
        self.internal_standard.setCurrentIndex(max(0, index))
        self.internal_standard.blockSignals(False)
        self._refresh_known_standard_state()
        dataset = self._current_dataset()
        self.populate_result(None if dataset is None else self.native_results_by_uid.get(dataset.uid))

    def calculate_native_preview(self):
        dataset = self._current_dataset()
        result = self._rietveld_result()
        structures = self._phase_structures()
        if dataset is None or not result:
            QMessageBox.information(self, "No structure refinement", "Run the structure-constrained refinement first.")
            return
        try:
            qpa = compute_hill_howard_qpa(
                result.get("phases", []),
                structures,
                scale_covariance=result.get("scale_covariance"),
                scale_standard_errors=[row.get("scale_standard_error") for row in result.get("phases", [])],
                engine="Afruz native Hill–Howard",
                internal_standard_phase=(None if self.internal_standard.currentText() == "None" else self.internal_standard.currentText()),
                internal_standard_known_wt_percent=(None if self.internal_standard.currentText() == "None" else self.known_standard_percent.value()),
            )
        except QPAValidationError as exc:
            QMessageBox.critical(self, "Native QPA audit failed", str(exc))
            return
        self.native_results_by_uid[dataset.uid] = qpa
        if hasattr(self.main_window, "_record_scientific_result"):
            self.main_window._record_scientific_result(
                "qpa",
                dataset.uid,
                {"validated": qpa},
                reason="Completed validated quantitative analysis",
            )
        self.populate_result(qpa)

    def populate_result(self, result):
        self.fraction_table.setRowCount(0)
        if not result:
            self.summary.setText("No QPA result.")
            return
        rows = result.get("phases", [])
        engine = result.get("engine", "Afruz native Hill–Howard")
        classification = result.get("classification", "Native crystallographic QPA")
        self.fraction_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row.get("phase_name"), engine, self._fmt(row.get("scale_factor")),
                self._fmt(row.get("scale_standard_error")), self._fmt(row.get("unit_cell_mass_g_mol")),
                self._fmt(row.get("unit_cell_volume_angstrom3")), self._fmt(row.get("zmv")),
                self._fmt(row.get("crystalline_weight_percent")), self._fmt(row.get("crystalline_weight_percent_error")),
                self._fmt(row.get("absolute_weight_percent")), classification,
            ]
            for column, value in enumerate(values):
                self.fraction_table.setItem(row_index, column, QTableWidgetItem(str(value)))
        summary = [classification]
        if result.get("amorphous_or_unmodelled_percent") is not None:
            summary.append(f"Amorphous/unmodelled: {result['amorphous_or_unmodelled_percent']:.5g}%")
        summary.extend(result.get("warnings", []))
        summary.append("Publication-ready flag: No — experimental validation remains required.")
        self.summary.setText("\n\n".join(str(item) for item in summary if item))

    def export_result(self):
        dataset = self._current_dataset()
        if dataset is None:
            return
        result = self.native_results_by_uid.get(dataset.uid)
        if not result:
            return
        self.main_window.show_export_center(
            preferred_formats={"excel", "text", "zip"},
            preferred_categories={"project", "pattern", "qpa"},
        )
    def apply_theme(self, theme_name):
        _ = theme_name

    def get_state(self):
        return {
            "native_results_by_uid": deepcopy(self.native_results_by_uid),
            "internal_standard": self.internal_standard.currentText(),
            "known_standard_percent": self.known_standard_percent.value(),
            "splitter_sizes": self.setup_splitter.sizes(),
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        values = state.get("native_results_by_uid")
        self.native_results_by_uid = deepcopy(values) if isinstance(values, dict) else {}
        self.known_standard_percent.setValue(state.get("known_standard_percent", 10.0))
        self.refresh_from_rietveld()
        self.internal_standard.setCurrentText(state.get("internal_standard", "None"))
        sizes = state.get("splitter_sizes")
        if isinstance(sizes, list) and sizes:
            self.setup_splitter.setSizes([int(value) for value in sizes])
