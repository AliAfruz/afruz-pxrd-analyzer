from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .context_copy import install_label_copy_menu, install_table_copy_menu
from .structure_solution import (
    STRUCTURE_SOLUTION_INTERFACES,
    StructureSolutionError,
    audit_provisional_cif,
    build_phase11_handoff,
    build_p1_provisional_cif,
    create_structure_solution_package,
    export_extracted_intensities,
    extract_intensity_table,
    import_and_audit_provisional_cif,
    screen_space_groups,
)
from .structure_solution_worker import CandidateRobustnessWorker
from .widgets import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)


class StructureSolutionPathwayWidget(QWidget):
    """Phase Revolution continuation from indexed cell to provisional structure."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.screening_by_uid: dict[str, list[dict]] = {}
        self.robustness_by_uid: dict[str, dict] = {}
        self.intensities_by_uid: dict[str, list[dict]] = {}
        self.provisional_by_uid: dict[str, dict] = {}
        self.packages_by_uid: dict[str, dict] = {}
        self.phase11_handoffs_by_uid: dict[str, dict] = {}
        self._thread: QThread | None = None
        self._worker: CandidateRobustnessWorker | None = None
        self._active_uid: str | None = None
        self._build_ui()
        self.refresh_for_selected_dataset()

    def _build_ui(self):
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        heading = QLabel(
            "Phase 14.5 — Space-Group Screening and Structure-Solution Pathway"
        )
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        description = QLabel(
            "Continue an indexed unknown cell through candidate robustness, "
            "systematic-absence pre-screening, extracted-intensity preparation, "
            "provisional P1 CIF management, and reproducible external "
            "structure-solution handoff. Every structure remains provisional "
            "until chemical, CIF and Rietveld validation pass."
        )
        description.setWordWrap(True)
        description.setObjectName("mutedLabel")
        install_label_copy_menu(description)
        layout.addWidget(description)

        self.setup_splitter = QSplitter(Qt.Horizontal)
        self.setup_splitter.setChildrenCollapsible(False)
        self.setup_splitter.addWidget(self._build_candidate_group())
        self.setup_splitter.addWidget(self._build_scroll_controls())
        self.setup_splitter.setSizes([760, 390])
        self.setup_splitter.setMaximumHeight(320)
        layout.addWidget(self.setup_splitter)

        run_row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Phase 14 candidates")
        self.screen_button = QPushButton("Screen space groups")
        self.screen_button.setObjectName("primaryButton")
        self.robustness_button = QPushButton("Run cell robustness")
        self.cancel_button = QPushButton("Cancel robustness")
        self.cancel_button.setEnabled(False)
        run_row.addWidget(self.refresh_button)
        run_row.addWidget(self.screen_button)
        run_row.addWidget(self.robustness_button)
        run_row.addWidget(self.cancel_button)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setMaximumHeight(24)
        self.progress.setFormat("%p% · %v/%m")
        self.progress_label = QLabel("Ready.")
        self.progress_label.setWordWrap(False)
        self.progress_label.setMinimumWidth(0)
        self.progress_label.setMaximumHeight(24)
        self.progress_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Fixed,
        )
        install_label_copy_menu(self.progress_label)
        progress_row.addWidget(self.progress, 2)
        progress_row.addWidget(self.progress_label, 3)
        layout.addLayout(progress_row)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setMinimumSize(0, 250)
        self.tabs.addTab(
            self._table_page(self._build_screening_table()),
            "Space-Group Pre-screen",
        )
        self.tabs.addTab(
            self._build_robustness_page(),
            "Cell Robustness",
        )
        self.tabs.addTab(
            self._build_intensity_page(),
            "Extracted Intensities",
        )
        self.tabs.addTab(
            self._build_provisional_page(),
            "Provisional CIF",
        )
        self.tabs.addTab(
            self._build_interface_page(),
            "Solution Interfaces",
        )
        self.tabs.addTab(
            self._build_diagnostics_page(),
            "Scientific Boundary",
        )
        layout.addWidget(self.tabs, 1)

        self.refresh_button.clicked.connect(self.refresh_for_selected_dataset)
        self.screen_button.clicked.connect(self.run_space_group_screening)
        self.robustness_button.clicked.connect(self.run_robustness)
        self.cancel_button.clicked.connect(self.cancel_robustness)

    def _build_candidate_group(self):
        group = QGroupBox("Indexed candidate cells from Phase 14 indexing")
        layout = QVBoxLayout(group)
        self.candidate_table = QTableWidget(0, 10)
        self.candidate_table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Status",
                "Bravais",
                "M20",
                "X20",
                "a",
                "b",
                "c",
                "Angles",
                "Volume",
            ]
        )
        self.candidate_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.candidate_table.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self._stabilize_table(self.candidate_table)
        self.candidate_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.candidate_table.setColumnWidth(0, 55)
        self.candidate_table.setColumnWidth(1, 110)
        self.candidate_table.setColumnWidth(2, 135)
        self.candidate_table.setColumnWidth(8, 170)
        install_table_copy_menu(self.candidate_table)
        layout.addWidget(self.candidate_table)
        return group

    def _build_scroll_controls(self):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setMinimumWidth(300)
        area.setMaximumWidth(520)

        content = QWidget()
        content.setMinimumSize(0, 0)
        form = QFormLayout(content)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.wavelength = NoWheelDoubleSpinBox()
        self.wavelength.setRange(0.1, 10.0)
        self.wavelength.setDecimals(7)
        self.wavelength.setValue(1.5406)

        self.screen_tolerance = NoWheelDoubleSpinBox()
        self.screen_tolerance.setRange(0.005, 1.0)
        self.screen_tolerance.setDecimals(4)
        self.screen_tolerance.setValue(0.08)

        self.maximum_index = NoWheelSpinBox()
        self.maximum_index.setRange(4, 40)
        self.maximum_index.setValue(16)

        self.robustness_iterations = NoWheelSpinBox()
        self.robustness_iterations.setRange(10, 1000)
        self.robustness_iterations.setValue(100)

        self.match_tolerance = NoWheelDoubleSpinBox()
        self.match_tolerance.setRange(0.01, 1.0)
        self.match_tolerance.setDecimals(4)
        self.match_tolerance.setValue(0.25)

        self.jitter_sigma = NoWheelDoubleSpinBox()
        self.jitter_sigma.setRange(0.0, 0.2)
        self.jitter_sigma.setDecimals(5)
        self.jitter_sigma.setValue(0.003)

        self.drop_fraction = NoWheelDoubleSpinBox()
        self.drop_fraction.setRange(0.0, 0.45)
        self.drop_fraction.setDecimals(3)
        self.drop_fraction.setSingleStep(0.025)
        self.drop_fraction.setValue(0.15)

        self.cell_bounds = NoWheelDoubleSpinBox()
        self.cell_bounds.setRange(0.1, 20.0)
        self.cell_bounds.setDecimals(2)
        self.cell_bounds.setValue(5.0)

        self.random_seed = NoWheelSpinBox()
        self.random_seed.setRange(0, 999999999)
        self.random_seed.setValue(1405)

        form.addRow("Wavelength (Å)", self.wavelength)
        form.addRow("Space-group tolerance (°)", self.screen_tolerance)
        form.addRow("Maximum h/k/l", self.maximum_index)
        form.addRow("Robustness iterations", self.robustness_iterations)
        form.addRow("Peak-match tolerance (°)", self.match_tolerance)
        form.addRow("Position jitter σ (°)", self.jitter_sigma)
        form.addRow("Peak-drop fraction", self.drop_fraction)
        form.addRow("Cell bounds (±%)", self.cell_bounds)
        form.addRow("Random seed", self.random_seed)

        note = QLabel(
            "Space-group results are a curated extinction pre-screen. "
            "They do not replace International Tables and full symmetry-engine checks, "
            "single-crystal/electron-diffraction evidence or Rietveld validation."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        form.addRow(note)

        area.setWidget(content)
        return area

    @staticmethod
    def _table_page(table):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(table)
        return page

    @staticmethod
    def _stabilize_table(table: QTableWidget):
        table.setMinimumSize(0, 0)
        table.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        table.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        table.setHorizontalScrollMode(
            QAbstractItemView.ScrollPerPixel
        )
        table.setVerticalScrollMode(
            QAbstractItemView.ScrollPerPixel
        )
        table.horizontalHeader().setMinimumSectionSize(45)
        table.horizontalHeader().setStretchLastSection(False)

    def _build_screening_table(self):
        self.screening_table = QTableWidget(0, 11)
        self.screening_table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Space group",
                "No.",
                "Status",
                "Coverage",
                "Matched",
                "Observed",
                "Forbidden conflicts",
                "Allowed reflections",
                "Score",
                "Rule level",
            ]
        )
        self._stabilize_table(self.screening_table)
        self.screening_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.screening_table.setColumnWidth(1, 130)
        self.screening_table.setColumnWidth(3, 160)
        self.screening_table.setColumnWidth(10, 380)
        install_table_copy_menu(self.screening_table)
        return self.screening_table

    def _build_robustness_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        summary_group = QGroupBox("Robustness summary")
        form = QFormLayout(summary_group)
        self.robust_status = QLabel("—")
        self.robust_coverage = QLabel("—")
        self.robust_success = QLabel("—")
        self.robust_volume = QLabel("—")
        self.robust_cell_spread = QLabel("—")
        self.robust_rmse = QLabel("—")
        self.robust_warning = QLabel("—")
        self.robust_warning.setWordWrap(True)
        for label in (
            self.robust_status,
            self.robust_coverage,
            self.robust_success,
            self.robust_volume,
            self.robust_cell_spread,
            self.robust_rmse,
            self.robust_warning,
        ):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            install_label_copy_menu(label)
        form.addRow("Classification", self.robust_status)
        form.addRow("Assigned-peak coverage", self.robust_coverage)
        form.addRow("Successful iterations", self.robust_success)
        form.addRow("Volume stability", self.robust_volume)
        form.addRow("Maximum cell RSD", self.robust_cell_spread)
        form.addRow("Median position RMSE", self.robust_rmse)
        form.addRow("Boundary", self.robust_warning)
        layout.addWidget(summary_group)

        self.assignment_table = QTableWidget(0, 7)
        self.assignment_table.setHorizontalHeaderLabels(
            [
                "hkl",
                "Observed 2θ",
                "Initial predicted 2θ",
                "Initial Δ2θ",
                "d-spacing",
                "h",
                "k / l",
            ]
        )
        self._stabilize_table(self.assignment_table)
        self.assignment_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        install_table_copy_menu(self.assignment_table)
        layout.addWidget(self.assignment_table, 1)
        return page

    def _build_intensity_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.phase_selector = NoWheelComboBox()
        self.capture_intensity_button = QPushButton(
            "Capture selected Phase 10 intensities"
        )
        self.export_intensity_button = QPushButton("Export clean TXT / HKL")
        controls.addWidget(QLabel("Phase"))
        controls.addWidget(self.phase_selector, 1)
        controls.addWidget(self.capture_intensity_button)
        controls.addWidget(self.export_intensity_button)
        layout.addLayout(controls)

        self.intensity_note = QLabel(
            "No intensity dataset captured. Phase 10 Pawley/Le Bail reflection "
            "intensities are required."
        )
        self.intensity_note.setWordWrap(True)
        self.intensity_note.setObjectName("mutedLabel")
        install_label_copy_menu(self.intensity_note)
        layout.addWidget(self.intensity_note)

        self.intensity_table = QTableWidget(0, 12)
        self.intensity_table.setHorizontalHeaderLabels(
            [
                "Phase",
                "h",
                "k",
                "l",
                "2θ",
                "d",
                "FWHM",
                "Intensity",
                "σ(I)",
                "Uncertainty source",
                "Overlap group",
                "Overlap",
            ]
        )
        self._stabilize_table(self.intensity_table)
        self.intensity_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.intensity_table.setColumnWidth(0, 150)
        self.intensity_table.setColumnWidth(9, 180)
        install_table_copy_menu(self.intensity_table)
        layout.addWidget(self.intensity_table, 1)

        self.capture_intensity_button.clicked.connect(
            self.capture_phase10_intensities
        )
        self.export_intensity_button.clicked.connect(
            self.export_intensity_files
        )
        return page

    def _build_provisional_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QGridLayout()

        self.formula = QLineEdit("")
        self.proposed_space_group = NoWheelComboBox()
        self.proposed_space_group.setEditable(True)
        self.proposed_space_group.addItem("Unresolved")
        self.provenance = QLineEdit(
            "Indexed powder candidate; atomic model remains provisional."
        )

        controls.addWidget(QLabel("Formula"), 0, 0)
        controls.addWidget(self.formula, 0, 1)
        controls.addWidget(QLabel("Proposed space group"), 0, 2)
        controls.addWidget(self.proposed_space_group, 0, 3)
        controls.addWidget(QLabel("Provenance"), 1, 0)
        controls.addWidget(self.provenance, 1, 1, 1, 3)

        self.add_atom_button = QPushButton("Add atom")
        self.remove_atom_button = QPushButton("Remove selected")
        self.import_cif_button = QPushButton("Import solution CIF")
        self.audit_cif_button = QPushButton("Audit provisional CIF")
        self.save_cif_button = QPushButton("Save provisional P1 CIF")
        self.send_phase11_button = QPushButton(
            "Validate and send to Phase 11"
        )
        self.send_phase11_button.setObjectName("primaryButton")
        controls.addWidget(self.add_atom_button, 2, 0)
        controls.addWidget(self.remove_atom_button, 2, 1)
        controls.addWidget(self.import_cif_button, 2, 2)
        controls.addWidget(self.audit_cif_button, 2, 3)
        controls.addWidget(self.save_cif_button, 3, 2)
        controls.addWidget(self.send_phase11_button, 3, 3)
        layout.addLayout(controls)

        self.atom_table = QTableWidget(0, 7)
        self.atom_table.setHorizontalHeaderLabels(
            ["Label", "Element", "x", "y", "z", "Occupancy", "Biso"]
        )
        self._stabilize_table(self.atom_table)
        self.atom_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.atom_table.setColumnWidth(0, 100)
        self.atom_table.setColumnWidth(1, 80)
        install_table_copy_menu(self.atom_table)
        layout.addWidget(self.atom_table, 1)

        self.cif_audit_summary = QLabel("No provisional CIF has been audited.")
        self.cif_audit_summary.setWordWrap(True)
        self.cif_audit_summary.setObjectName("mutedLabel")
        install_label_copy_menu(self.cif_audit_summary)
        layout.addWidget(self.cif_audit_summary)

        self.add_atom_button.clicked.connect(self.add_atom)
        self.remove_atom_button.clicked.connect(self.remove_selected_atoms)
        self.import_cif_button.clicked.connect(self.import_solution_cif)
        self.audit_cif_button.clicked.connect(self.audit_current_cif)
        self.save_cif_button.clicked.connect(self.save_provisional_cif)
        self.send_phase11_button.clicked.connect(
            lambda: self.send_to_phase11(open_phase11=True)
        )
        return page

    def _build_interface_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        group = QGroupBox("External structure-solution handoff")
        form = QFormLayout(group)

        self.interface_selector = NoWheelComboBox()
        self.interface_selector.addItems(
            list(STRUCTURE_SOLUTION_INTERFACES)
        )
        self.z_value = NoWheelSpinBox()
        self.z_value.setRange(0, 999)
        self.z_value.setValue(0)
        self.package_directory = QLineEdit("")
        self.browse_package_button = QPushButton("Browse…")
        path_row = QHBoxLayout()
        path_row.addWidget(self.package_directory, 1)
        path_row.addWidget(self.browse_package_button)
        self.interface_notes = QTextEdit()
        self.interface_notes.setMaximumHeight(90)
        self.create_package_button = QPushButton(
            "Create reproducible solution package"
        )
        self.create_package_button.setObjectName("primaryButton")

        form.addRow("Interface", self.interface_selector)
        form.addRow("Formula units Z (0=unknown)", self.z_value)
        form.addRow("Output directory", path_row)
        form.addRow("Notes", self.interface_notes)
        form.addRow(self.create_package_button)
        layout.addWidget(group)

        self.package_summary = QLabel("No structure-solution package created.")
        self.package_summary.setWordWrap(True)
        self.package_summary.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        install_label_copy_menu(self.package_summary)
        layout.addWidget(self.package_summary)
        layout.addStretch(1)

        self.browse_package_button.clicked.connect(
            self.browse_package_directory
        )
        self.create_package_button.clicked.connect(
            self.create_solution_package
        )
        return page

    def _build_diagnostics_page(self):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        form = QFormLayout(content)
        self.diag_dataset = QLabel("—")
        self.diag_candidate = QLabel("—")
        self.diag_screen = QLabel("—")
        self.diag_robustness = QLabel("—")
        self.diag_intensities = QLabel("—")
        self.diag_cif = QLabel("—")
        self.diag_phase11 = QLabel("—")
        self.diag_package = QLabel("—")
        self.diag_boundary = QLabel(
            "An indexed cell and low residual do not prove an atomic structure. "
            "A publishable result requires competing-model tests, chemistry, "
            "independent CIF validation and structure-constrained Rietveld refinement."
        )
        self.diag_boundary.setWordWrap(True)
        for label in (
            self.diag_dataset,
            self.diag_candidate,
            self.diag_screen,
            self.diag_robustness,
            self.diag_intensities,
            self.diag_cif,
            self.diag_phase11,
            self.diag_package,
            self.diag_boundary,
        ):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            install_label_copy_menu(label)
        form.addRow("Dataset", self.diag_dataset)
        form.addRow("Selected candidate", self.diag_candidate)
        form.addRow("Space-group screening", self.diag_screen)
        form.addRow("Cell robustness", self.diag_robustness)
        form.addRow("Extracted intensities", self.diag_intensities)
        form.addRow("Provisional CIF", self.diag_cif)
        form.addRow("Phase 11 handoff", self.diag_phase11)
        form.addRow("Solution package", self.diag_package)
        form.addRow("Scientific boundary", self.diag_boundary)
        area.setWidget(content)
        return area

    @staticmethod
    def _fmt(value, digits=7):
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}g}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _stable_text(message: str, maximum=125):
        compact = " ".join(str(message).split())
        return compact if len(compact) <= maximum else compact[: maximum - 1] + "…"

    def _dataset(self):
        return self.main_window.selected_dataset()

    def _phase14_candidates(self):
        dataset = self._dataset()
        if dataset is None:
            return []
        widget = getattr(self.main_window, "phase_revolution_widget", None)
        if widget is None:
            return []
        return widget.candidates_by_uid.get(dataset.uid, [])

    def _phase14_peaks(self):
        dataset = self._dataset()
        if dataset is None:
            return []
        widget = getattr(self.main_window, "phase_revolution_widget", None)
        if widget is None:
            return []
        if hasattr(widget, "master_peak_list"):
            return widget.master_peak_list(dataset.uid)
        return widget.peaks_by_uid.get(dataset.uid, [])

    def selected_candidate(self):
        row = self.candidate_table.currentRow()
        if row < 0 and self.candidate_table.rowCount():
            row = 0
        if row < 0:
            return None
        item = self.candidate_table.item(row, 0)
        return None if item is None else deepcopy(item.data(Qt.UserRole))

    def refresh_candidates(self):
        candidates = self._phase14_candidates()
        self.candidate_table.setRowCount(len(candidates))
        for row_index, candidate in enumerate(candidates):
            angles = (
                f"{self._fmt(candidate.get('alpha_deg'))} / "
                f"{self._fmt(candidate.get('beta_deg'))} / "
                f"{self._fmt(candidate.get('gamma_deg'))}"
            )
            values = [
                candidate.get("rank", row_index + 1),
                candidate.get("status", ""),
                candidate.get("bravais_name", ""),
                self._fmt(candidate.get("m20")),
                candidate.get("x20", ""),
                self._fmt(candidate.get("a_angstrom")),
                self._fmt(candidate.get("b_angstrom")),
                self._fmt(candidate.get("c_angstrom")),
                angles,
                self._fmt(candidate.get("volume_angstrom3")),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if column == 0:
                    item.setData(Qt.UserRole, deepcopy(candidate))
                self.candidate_table.setItem(row_index, column, item)
        if candidates:
            self.candidate_table.selectRow(0)

    def run_space_group_screening(self):
        dataset = self._dataset()
        candidate = self.selected_candidate()
        peaks = self._phase14_peaks()
        if dataset is None or candidate is None:
            QMessageBox.information(
                self,
                "No candidate",
                "Create and select an indexed Phase 14 candidate first.",
            )
            return
        try:
            results = screen_space_groups(
                candidate,
                peaks,
                wavelength_angstrom=self.wavelength.value(),
                tolerance_deg=self.screen_tolerance.value(),
                maximum_index=self.maximum_index.value(),
            )
        except StructureSolutionError as exc:
            QMessageBox.critical(
                self,
                "Space-group screening failed",
                str(exc),
            )
            return
        self.screening_by_uid[dataset.uid] = results
        self.populate_screening(results)
        self._refresh_space_group_selector(results)
        self.tabs.setCurrentIndex(0)
        self._set_status(
            f"Screened {len(results)} curated space-group candidates."
        )
        self.refresh_diagnostics()

    def populate_screening(self, rows):
        self.screening_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row.get("rank"),
                row.get("name"),
                row.get("number"),
                row.get("status"),
                self._fmt(100.0 * row.get("coverage_fraction", 0.0)),
                row.get("matched_peak_count"),
                row.get("observed_peak_count"),
                row.get("forbidden_conflict_count"),
                row.get("predicted_allowed_count"),
                self._fmt(row.get("screening_score")),
                row.get("screening_level"),
            ]
            for column, value in enumerate(values):
                self.screening_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(str(value)),
                )

    def _refresh_space_group_selector(self, screening):
        current = self.proposed_space_group.currentText()
        self.proposed_space_group.clear()
        self.proposed_space_group.addItem("Unresolved")
        for row in screening:
            self.proposed_space_group.addItem(
                f"{row['name']} (No. {row['number']})"
            )
        index = self.proposed_space_group.findText(current)
        if index >= 0:
            self.proposed_space_group.setCurrentIndex(index)

    def is_running(self):
        return bool(self._thread is not None and self._thread.isRunning())

    def run_robustness(self):
        if self.is_running():
            return
        dataset = self._dataset()
        candidate = self.selected_candidate()
        peaks = self._phase14_peaks()
        if dataset is None or candidate is None:
            QMessageBox.information(
                self,
                "No candidate",
                "Create and select an indexed Phase 14 candidate first.",
            )
            return
        settings = {
            "wavelength_angstrom": self.wavelength.value(),
            "matching_tolerance_deg": self.match_tolerance.value(),
            "position_jitter_sigma_deg": self.jitter_sigma.value(),
            "drop_fraction": self.drop_fraction.value(),
            "iterations": self.robustness_iterations.value(),
            "parameter_bound_percent": self.cell_bounds.value(),
            "maximum_index": self.maximum_index.value(),
            "random_seed": self.random_seed.value(),
        }
        thread = QThread(self)
        worker = CandidateRobustnessWorker(candidate, peaks, settings)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_robustness_progress)
        worker.finished.connect(self._on_robustness_finished)
        worker.failed.connect(self._on_robustness_failed)
        worker.cancelled.connect(self._on_robustness_cancelled)
        worker.finished.connect(lambda *_: thread.quit())
        worker.failed.connect(lambda *_: thread.quit())
        worker.cancelled.connect(lambda *_: thread.quit())
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        self._active_uid = dataset.uid
        self.progress.setRange(0, self.robustness_iterations.value())
        self.progress.setValue(0)
        self.robustness_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._set_status("Starting candidate robustness test.")
        thread.start()

    def cancel_robustness(self):
        if self._worker is not None:
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self._set_status("Cancellation requested.")

    def _on_robustness_progress(self, completed, total, message):
        self.progress.setRange(0, max(1, int(total)))
        self.progress.setValue(max(0, min(int(completed), int(total))))
        self._set_status(message)

    def _on_robustness_finished(self, result):
        if self._active_uid:
            self.robustness_by_uid[self._active_uid] = result
        self.populate_robustness(result)
        self.progress.setValue(self.progress.maximum())
        self._set_status(result.get("status", "Robustness completed."))
        self.tabs.setCurrentIndex(1)
        self.refresh_diagnostics()

    def _on_robustness_failed(self, traceback_text):
        final_line = traceback_text.strip().splitlines()[-1]
        self._set_status(final_line)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Candidate robustness failed")
        box.setText(final_line)
        box.setDetailedText(traceback_text)
        box.exec()

    def _on_robustness_cancelled(self):
        self._set_status("Candidate robustness cancelled; previous result retained.")

    def _thread_finished(self):
        self.robustness_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._thread = None
        self._worker = None
        self._active_uid = None

    def populate_robustness(self, result):
        if not result:
            for label in (
                self.robust_status,
                self.robust_coverage,
                self.robust_success,
                self.robust_volume,
                self.robust_cell_spread,
                self.robust_rmse,
                self.robust_warning,
            ):
                label.setText("—")
            self.assignment_table.setRowCount(0)
            return
        self.robust_status.setText(str(result.get("status")))
        self.robust_coverage.setText(
            f"{100.0 * result.get('assignment_coverage_fraction', 0.0):.4g}% "
            f"({result.get('assigned_peak_count', 0)}/"
            f"{result.get('included_peak_count', 0)})"
        )
        self.robust_success.setText(
            f"{result.get('iterations_successful', 0)}/"
            f"{result.get('iterations_requested', 0)} "
            f"({100.0 * result.get('success_fraction', 0.0):.4g}%)"
        )
        self.robust_volume.setText(
            f"{self._fmt(result.get('volume_mean_angstrom3'))} Å³; "
            f"CV {self._fmt(result.get('volume_cv_percent'))}%"
        )
        self.robust_cell_spread.setText(
            f"{self._fmt(result.get('maximum_cell_parameter_rsd_percent'))}%"
        )
        self.robust_rmse.setText(
            f"{self._fmt(result.get('median_position_rmse_deg'))}°"
        )
        self.robust_warning.setText(str(result.get("warning", "")))

        assignments = result.get("assignments", [])
        self.assignment_table.setRowCount(len(assignments))
        for row_index, row in enumerate(assignments):
            values = [
                row.get("hkl_label"),
                self._fmt(row.get("observed_two_theta_deg")),
                self._fmt(row.get("two_theta_deg")),
                self._fmt(row.get("initial_delta_deg")),
                self._fmt(row.get("d_spacing_angstrom")),
                row.get("h"),
                f"{row.get('k')} / {row.get('l')}",
            ]
            for column, value in enumerate(values):
                self.assignment_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(str(value)),
                )

    def _phase10_result(self):
        dataset = self._dataset()
        widget = getattr(self.main_window, "whole_pattern_widget", None)
        if dataset is None or widget is None:
            return None
        return widget.results_by_uid.get(dataset.uid)

    def refresh_phase_selector(self):
        current = self.phase_selector.currentData()
        self.phase_selector.clear()
        result = self._phase10_result() or {}
        for phase in result.get("phases", []):
            name = str(phase.get("phase_name", "Phase"))
            index = phase.get("phase_index")
            self.phase_selector.addItem(name, index)
        if current is not None:
            index = self.phase_selector.findData(current)
            if index >= 0:
                self.phase_selector.setCurrentIndex(index)

    def capture_phase10_intensities(self):
        dataset = self._dataset()
        result = self._phase10_result()
        if dataset is None or not result:
            QMessageBox.information(
                self,
                "No Phase 10 result",
                "Run Pawley or Le Bail refinement for this dataset first.",
            )
            return
        phase_index = self.phase_selector.currentData()
        try:
            rows = extract_intensity_table(
                result,
                phase_index=phase_index,
            )
        except StructureSolutionError as exc:
            QMessageBox.critical(self, "Intensity extraction failed", str(exc))
            return
        if not rows:
            QMessageBox.information(
                self,
                "No indexed intensities",
                "The selected Phase 10 phase has no indexed extracted reflections.",
            )
            return
        self.intensities_by_uid[dataset.uid] = rows
        self.populate_intensities(rows)
        self.tabs.setCurrentIndex(2)
        self.refresh_diagnostics()

    def populate_intensities(self, rows):
        self.intensity_table.setRowCount(len(rows))
        missing_uncertainty = 0
        overlap_count = 0
        for row_index, row in enumerate(rows):
            if row.get("intensity_uncertainty") is None:
                missing_uncertainty += 1
            if row.get("overlap_flag"):
                overlap_count += 1
            values = [
                row.get("phase_name"),
                row.get("h"),
                row.get("k"),
                row.get("l"),
                self._fmt(row.get("two_theta_deg")),
                self._fmt(row.get("d_spacing_angstrom")),
                self._fmt(row.get("fwhm_deg")),
                self._fmt(row.get("intensity")),
                self._fmt(row.get("intensity_uncertainty")),
                row.get("uncertainty_source"),
                row.get("overlap_group"),
                "Yes" if row.get("overlap_flag") else "No",
            ]
            for column, value in enumerate(values):
                self.intensity_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(str(value)),
                )
        self.intensity_note.setText(
            f"{len(rows)} reflections; {overlap_count} overlap-flagged; "
            f"{missing_uncertainty} lack covariance-derived intensity uncertainty."
        )

    def export_intensity_files(self):
        dataset = self._dataset()
        if dataset is None:
            return
        rows = self.intensities_by_uid.get(dataset.uid, [])
        if not rows:
            return
        directory = QFileDialog.getExistingDirectory(
            self,
            "Export extracted intensities",
        )
        if not directory:
            return
        try:
            result = export_extracted_intensities(directory, rows)
        except StructureSolutionError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._set_status(
            f"Exported {result['reflection_count']} extracted reflections."
        )

    def add_atom(self):
        row = self.atom_table.rowCount()
        self.atom_table.insertRow(row)
        defaults = [
            f"C{row + 1}",
            "C",
            "0.0",
            "0.0",
            "0.0",
            "1.0",
            "1.0",
        ]
        for column, value in enumerate(defaults):
            self.atom_table.setItem(row, column, QTableWidgetItem(value))

    def remove_selected_atoms(self):
        rows = sorted(
            {index.row() for index in self.atom_table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            self.atom_table.removeRow(row)

    def _current_atoms(self):
        atoms = []
        for row in range(self.atom_table.rowCount()):
            values = [
                self.atom_table.item(row, column)
                for column in range(self.atom_table.columnCount())
            ]
            if not all(values):
                raise StructureSolutionError(
                    f"Atom row {row + 1} is incomplete."
                )
            try:
                atoms.append(
                    {
                        "label": values[0].text().strip(),
                        "element": values[1].text().strip(),
                        "x": float(values[2].text()),
                        "y": float(values[3].text()),
                        "z": float(values[4].text()),
                        "occupancy": float(values[5].text()),
                        "b_iso": float(values[6].text()),
                    }
                )
            except ValueError as exc:
                raise StructureSolutionError(
                    f"Atom row {row + 1} contains a non-numeric value."
                ) from exc
        return atoms

    def _provisional_cif_text(self):
        candidate = self.selected_candidate()
        if candidate is None:
            raise StructureSolutionError(
                "Select an indexed candidate cell first."
            )
        atoms = self._current_atoms()
        return build_p1_provisional_cif(
            candidate,
            atoms,
            title="Afruz unknown-phase provisional structure",
            formula=self.formula.text().strip(),
            proposed_space_group=self.proposed_space_group.currentText(),
            provenance=self.provenance.text().strip(),
        )

    def audit_current_cif(self):
        dataset = self._dataset()
        if dataset is None:
            return
        try:
            cif_text = self._provisional_cif_text()
            audit = audit_provisional_cif(cif_text)
        except StructureSolutionError as exc:
            QMessageBox.critical(self, "Provisional CIF failed", str(exc))
            return
        record = {
            "raw_cif_text": cif_text,
            "audit": audit,
            "formula": self.formula.text().strip(),
            "proposed_space_group": self.proposed_space_group.currentText(),
            "atoms": self._current_atoms(),
        }
        self.provisional_by_uid[dataset.uid] = record
        if hasattr(self.main_window, "_record_scientific_result"):
            self.main_window._record_scientific_result(
                "phase",
                dataset.uid,
                {"provisional_structure": record},
                reason="Audited provisional structure model",
            )
        self.populate_cif_audit(record)
        self.refresh_diagnostics()

    def populate_cif_audit(self, record):
        if not record:
            self.cif_audit_summary.setText(
                "No provisional CIF has been audited."
            )
            return
        audit = record.get("audit", {})
        messages = [str(audit.get("status", "Unknown"))]
        if audit.get("errors"):
            messages.append("Errors: " + "; ".join(audit["errors"]))
        if audit.get("warnings"):
            messages.append("Warnings: " + "; ".join(audit["warnings"]))
        shortest = audit.get("shortest_contact_angstrom")
        if shortest is not None:
            messages.append(
                f"Shortest P1 contact: {self._fmt(shortest)} Å."
            )
        self.cif_audit_summary.setText(" ".join(messages))

    def import_solution_cif(self):
        dataset = self._dataset()
        if dataset is None:
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import provisional solution CIF",
            "",
            "CIF (*.cif)",
        )
        if not filename:
            return
        try:
            record = import_and_audit_provisional_cif(filename)
        except StructureSolutionError as exc:
            QMessageBox.critical(self, "CIF import failed", str(exc))
            return
        structure = record.get("structure") or {}
        self.formula.setText(structure.get("formula", ""))
        self.proposed_space_group.setCurrentText(
            structure.get("space_group", "Unresolved")
        )
        self.atom_table.setRowCount(0)
        for atom in structure.get("atoms", []):
            row = self.atom_table.rowCount()
            self.atom_table.insertRow(row)
            values = [
                atom.get("label"),
                atom.get("element"),
                self._fmt(atom.get("x"), 10),
                self._fmt(atom.get("y"), 10),
                self._fmt(atom.get("z"), 10),
                self._fmt(atom.get("occupancy"), 7),
                self._fmt(atom.get("b_iso"), 7),
            ]
            for column, value in enumerate(values):
                self.atom_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(str(value)),
                )
        self.provisional_by_uid[dataset.uid] = {
            "raw_cif_text": record.get("raw_cif_text"),
            "audit": {
                key: value
                for key, value in record.items()
                if key not in {"raw_cif_text", "source_path"}
            },
            "formula": structure.get("formula", ""),
            "proposed_space_group": structure.get("space_group", "Unresolved"),
            "atoms": deepcopy(structure.get("atoms", [])),
            "source_path": record.get("source_path"),
        }
        self.populate_cif_audit(self.provisional_by_uid[dataset.uid])
        self.refresh_diagnostics()

    def save_provisional_cif(self):
        try:
            text = self._provisional_cif_text()
        except StructureSolutionError as exc:
            QMessageBox.critical(self, "CIF creation failed", str(exc))
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save provisional P1 CIF",
            "unknown_phase_provisional_p1.cif",
            "CIF (*.cif)",
        )
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != ".cif":
            path = path.with_suffix(".cif")
        path.write_text(text, encoding="utf-8")
        self._set_status(f"Saved provisional P1 CIF: {path}")

    def _phase11_handoff_record(self):
        dataset = self._dataset()
        if dataset is None:
            raise StructureSolutionError("Select a dataset first.")
        cif_text = self._provisional_cif_text()
        candidate = self.selected_candidate()
        peak_meta = {}
        phase_widget = getattr(
            self.main_window,
            "phase_revolution_widget",
            None,
        )
        if phase_widget is not None and hasattr(
            phase_widget,
            "master_peak_metadata",
        ):
            peak_meta = phase_widget.master_peak_metadata(dataset.uid)
        return build_phase11_handoff(
            cif_text,
            dataset_uid=dataset.uid,
            dataset_name=dataset.name,
            candidate=candidate,
            master_peak_metadata=peak_meta,
            proposed_space_group=self.proposed_space_group.currentText(),
            provenance=self.provenance.text().strip(),
        )

    def send_to_phase11(self, *, open_phase11: bool = True):
        dataset = self._dataset()
        if dataset is None:
            return None
        rietveld = getattr(self.main_window, "rietveld_widget", None)
        if rietveld is None:
            QMessageBox.critical(
                self,
                "Phase 11 unavailable",
                "The Phase 11 Rietveld workspace is not available.",
            )
            return None
        try:
            record = self._phase11_handoff_record()
        except StructureSolutionError as exc:
            QMessageBox.critical(self, "Phase 11 handoff failed", str(exc))
            return None

        audit = record.get("audit", {})
        warnings = list(audit.get("warnings") or [])
        substantive_warnings = [
            warning
            for warning in warnings
            if "stored in P1" not in str(warning)
        ]
        if substantive_warnings:
            response = QMessageBox.question(
                self,
                "Provisional structure warnings",
                "The model has review warnings:\n\n"
                + "\n".join(f"• {item}" for item in substantive_warnings)
                + "\n\nSend it to Phase 11 as a provisional model?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if response != QMessageBox.Yes:
                return None

        added = rietveld.add_unknown_discovery_structure(record)
        if not added:
            return None
        self.provisional_by_uid[dataset.uid] = {
            "raw_cif_text": record["structure"].get("raw_cif_text"),
            "audit": deepcopy(record.get("audit", {})),
            "formula": record["structure"].get("formula", ""),
            "proposed_space_group": self.proposed_space_group.currentText(),
            "atoms": deepcopy(record["structure"].get("atoms", [])),
        }
        self.phase11_handoffs_by_uid[dataset.uid] = deepcopy(
            record.get("handoff", {})
        )
        if hasattr(self.main_window, "_record_scientific_result"):
            self.main_window._record_scientific_result(
                "phase",
                dataset.uid,
                {
                    "provisional_structure": self.provisional_by_uid[dataset.uid],
                    "phase11_handoff": self.phase11_handoffs_by_uid[dataset.uid],
                },
                reason="Validated provisional structure for Phase 11",
            )
        self.populate_cif_audit(self.provisional_by_uid[dataset.uid])
        self.refresh_diagnostics()
        self._set_status(
            "Provisional unknown structure added to Phase 11 Rietveld refinement."
        )
        if open_phase11 and hasattr(self.main_window, "tabs"):
            self.main_window.tabs.setCurrentWidget(rietveld)
        return record

    def browse_package_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select structure-solution package directory",
        )
        if directory:
            self.package_directory.setText(directory)

    def create_solution_package(self):
        dataset = self._dataset()
        candidate = self.selected_candidate()
        if dataset is None or candidate is None:
            QMessageBox.information(
                self,
                "No candidate",
                "Select an indexed candidate cell first.",
            )
            return
        rows = self.intensities_by_uid.get(dataset.uid, [])
        if not rows:
            QMessageBox.information(
                self,
                "No extracted intensities",
                "Capture Phase 10 extracted intensities first.",
            )
            return
        directory = self.package_directory.text().strip()
        if not directory:
            directory = str(
                Path(tempfile.mkdtemp(prefix="afruz_structure_solution_"))
            )
            self.package_directory.setText(directory)
        try:
            atoms = self._current_atoms()
            result = create_structure_solution_package(
                directory,
                interface=self.interface_selector.currentText(),
                candidate=candidate,
                wavelength_angstrom=self.wavelength.value(),
                intensity_rows=rows,
                formula=self.formula.text().strip(),
                z_value=(
                    None if self.z_value.value() == 0
                    else self.z_value.value()
                ),
                proposed_space_group=(
                    self.proposed_space_group.currentText()
                ),
                atoms=atoms or None,
                notes=self.interface_notes.toPlainText().strip(),
            )
        except StructureSolutionError as exc:
            QMessageBox.critical(
                self,
                "Structure-solution package failed",
                str(exc),
            )
            return
        self.packages_by_uid[dataset.uid] = result
        self.package_summary.setText(
            f"{result['classification']} — {result['interface']}\n"
            f"{result['directory']}"
        )
        self.refresh_diagnostics()
        self._set_status("Structure-solution package created.")

    def _set_status(self, message):
        text = self._stable_text(message)
        self.progress_label.setText(text)
        self.progress_label.setToolTip(str(message))
        self.main_window.statusBar().showMessage(str(message))

    def refresh_diagnostics(self):
        dataset = self._dataset()
        if dataset is None:
            return
        candidate = self.selected_candidate()
        screening = self.screening_by_uid.get(dataset.uid, [])
        robustness = self.robustness_by_uid.get(dataset.uid)
        intensities = self.intensities_by_uid.get(dataset.uid, [])
        provisional = self.provisional_by_uid.get(dataset.uid)
        package = self.packages_by_uid.get(dataset.uid)
        phase11_handoff = self.phase11_handoffs_by_uid.get(dataset.uid)

        self.diag_dataset.setText(dataset.name)
        self.diag_candidate.setText(
            "—"
            if candidate is None
            else (
                f"#{candidate.get('rank')} {candidate.get('bravais_name')} "
                f"M20={self._fmt(candidate.get('m20'))}"
            )
        )
        self.diag_screen.setText(
            "Not run"
            if not screening
            else (
                f"{screening[0].get('name')} — "
                f"{screening[0].get('status')}; "
                f"coverage "
                f"{100.0 * screening[0].get('coverage_fraction', 0):.4g}%"
            )
        )
        self.diag_robustness.setText(
            "Not run"
            if not robustness
            else str(robustness.get("status"))
        )
        self.diag_intensities.setText(
            f"{len(intensities)} extracted reflections"
            if intensities
            else "Not captured"
        )
        self.diag_cif.setText(
            "Not audited"
            if not provisional
            else str(
                provisional.get("audit", {}).get("status", "Audited")
            )
        )
        self.diag_phase11.setText(
            "Not sent"
            if not phase11_handoff
            else (
                f"{phase11_handoff.get('classification', 'Provisional')} — "
                f"candidate #{phase11_handoff.get('candidate_rank') or '—'}; "
                f"peaks revision {phase11_handoff.get('master_peak_revision') or '—'}"
            )
        )
        self.diag_package.setText(
            "Not created"
            if not package
            else str(package.get("directory"))
        )

    def refresh_for_selected_dataset(self):
        dataset = self._dataset()
        if dataset is None:
            self.candidate_table.setRowCount(0)
            self.populate_screening([])
            self.populate_robustness(None)
            self.populate_intensities([])
            return
        wavelength = (
            dataset.metadata.get("wavelength_angstrom")
            or dataset.metadata.get("wavelength_k_alpha1")
        )
        if wavelength:
            try:
                self.wavelength.setValue(float(wavelength))
            except (TypeError, ValueError):
                pass
        self.refresh_candidates()
        self.refresh_phase_selector()
        self.populate_screening(
            self.screening_by_uid.get(dataset.uid, [])
        )
        self.populate_robustness(
            self.robustness_by_uid.get(dataset.uid)
        )
        self.populate_intensities(
            self.intensities_by_uid.get(dataset.uid, [])
        )
        record = self.provisional_by_uid.get(dataset.uid)
        if record:
            self.formula.setText(record.get("formula", ""))
            self.proposed_space_group.setCurrentText(
                record.get("proposed_space_group", "Unresolved")
            )
            self.atom_table.setRowCount(0)
            for atom in record.get("atoms", []):
                row = self.atom_table.rowCount()
                self.atom_table.insertRow(row)
                values = [
                    atom.get("label"),
                    atom.get("element"),
                    self._fmt(atom.get("x"), 10),
                    self._fmt(atom.get("y"), 10),
                    self._fmt(atom.get("z"), 10),
                    self._fmt(atom.get("occupancy")),
                    self._fmt(atom.get("b_iso")),
                ]
                for column, value in enumerate(values):
                    self.atom_table.setItem(
                        row,
                        column,
                        QTableWidgetItem(str(value)),
                    )
            self.populate_cif_audit(record)
        else:
            self.atom_table.setRowCount(0)
            self.cif_audit_summary.setText(
                "No provisional CIF has been audited."
            )
        package = self.packages_by_uid.get(dataset.uid)
        self.package_summary.setText(
            "No structure-solution package created."
            if not package
            else f"{package.get('classification')}\n{package.get('directory')}"
        )
        self.refresh_diagnostics()

    def duplicate_dataset(self, source_uid, target_uid):
        for mapping in (
            self.screening_by_uid,
            self.robustness_by_uid,
            self.intensities_by_uid,
            self.provisional_by_uid,
            self.packages_by_uid,
            self.phase11_handoffs_by_uid,
        ):
            if source_uid in mapping:
                mapping[target_uid] = deepcopy(mapping[source_uid])

    def remove_dataset(self, dataset_uid):
        for mapping in (
            self.screening_by_uid,
            self.robustness_by_uid,
            self.intensities_by_uid,
            self.provisional_by_uid,
            self.packages_by_uid,
            self.phase11_handoffs_by_uid,
        ):
            mapping.pop(dataset_uid, None)

    def apply_theme(self, theme_name):
        _ = theme_name

    def get_state(self):
        return {
            "screening_by_uid": deepcopy(self.screening_by_uid),
            "robustness_by_uid": deepcopy(self.robustness_by_uid),
            "intensities_by_uid": deepcopy(self.intensities_by_uid),
            "provisional_by_uid": deepcopy(self.provisional_by_uid),
            "packages_by_uid": deepcopy(self.packages_by_uid),
            "phase11_handoffs_by_uid": deepcopy(
                self.phase11_handoffs_by_uid
            ),
            "splitter_sizes": self.setup_splitter.sizes(),
            "settings": {
                "wavelength": self.wavelength.value(),
                "screen_tolerance": self.screen_tolerance.value(),
                "maximum_index": self.maximum_index.value(),
                "robustness_iterations": self.robustness_iterations.value(),
                "match_tolerance": self.match_tolerance.value(),
                "jitter_sigma": self.jitter_sigma.value(),
                "drop_fraction": self.drop_fraction.value(),
                "cell_bounds": self.cell_bounds.value(),
                "random_seed": self.random_seed.value(),
                "formula": self.formula.text(),
                "proposed_space_group": self.proposed_space_group.currentText(),
                "provenance": self.provenance.text(),
                "interface": self.interface_selector.currentText(),
                "z_value": self.z_value.value(),
                "package_directory": self.package_directory.text(),
                "interface_notes": self.interface_notes.toPlainText(),
            },
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        for attribute in (
            "screening_by_uid",
            "robustness_by_uid",
            "intensities_by_uid",
            "provisional_by_uid",
            "packages_by_uid",
            "phase11_handoffs_by_uid",
        ):
            value = state.get(attribute)
            setattr(
                self,
                attribute,
                deepcopy(value) if isinstance(value, dict) else {},
            )
        settings = state.get("settings", {})
        if isinstance(settings, dict):
            for widget, key, default in (
                (self.wavelength, "wavelength", 1.5406),
                (self.screen_tolerance, "screen_tolerance", 0.08),
                (self.maximum_index, "maximum_index", 16),
                (self.robustness_iterations, "robustness_iterations", 100),
                (self.match_tolerance, "match_tolerance", 0.25),
                (self.jitter_sigma, "jitter_sigma", 0.003),
                (self.drop_fraction, "drop_fraction", 0.15),
                (self.cell_bounds, "cell_bounds", 5.0),
                (self.random_seed, "random_seed", 1405),
                (self.z_value, "z_value", 0),
            ):
                widget.setValue(settings.get(key, default))
            self.formula.setText(settings.get("formula", ""))
            self.proposed_space_group.setCurrentText(
                settings.get("proposed_space_group", "Unresolved")
            )
            self.provenance.setText(
                settings.get(
                    "provenance",
                    "Indexed powder candidate; atomic model remains provisional.",
                )
            )
            self.interface_selector.setCurrentText(
                settings.get(
                    "interface",
                    STRUCTURE_SOLUTION_INTERFACES[0],
                )
            )
            self.package_directory.setText(
                settings.get("package_directory", "")
            )
            self.interface_notes.setPlainText(
                settings.get("interface_notes", "")
            )
        sizes = state.get("splitter_sizes")
        if isinstance(sizes, list) and sizes:
            self.setup_splitter.setSizes(
                [int(value) for value in sizes]
            )
        self.refresh_for_selected_dataset()
