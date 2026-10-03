from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from PySide6.QtCore import Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices
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
    QInputDialog,
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
from .phase_revolution import (
    BRAVAIS_NAMES,
    PhaseRevolutionError,
    candidate_reference_dataset,
    detect_unknown_peaks,
    manual_peak_from_position,
    master_peak_list_checksum,
    merge_master_peak_lists,
    normalize_master_peak_list,
    refresh_master_peak_measurements,
    enrich_candidate_chemistry,
    export_unknown_peak_csv,
    export_unknown_peak_txt,
    extract_unknown_residual,
)
from .phase_revolution_plot import PhaseRevolutionPlotWidget
from .text_export import write_columns_txt, write_mapping_txt, write_table_txt, write_manifest_txt
from .phase_revolution_worker import PhaseRevolutionIndexingWorker
from .widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox


class PhaseRevolutionWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.residuals_by_uid: dict[str, dict] = {}
        self.peaks_by_uid: dict[str, list[dict]] = {}
        self.peak_list_meta_by_uid: dict[str, dict] = {}
        self.indexing_by_uid: dict[str, dict] = {}
        self.candidates_by_uid: dict[str, list[dict]] = {}
        self._thread: QThread | None = None
        self._worker: PhaseRevolutionIndexingWorker | None = None
        self._active_dataset_uid: str | None = None
        self._active_peak_snapshot: dict | None = None
        self._updating_peaks = False
        self._build_ui()
        self.refresh_for_selected_dataset()

    def _build_ui(self):
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        heading = QLabel("Unknown Phase — Native Structure Discovery")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        intro = QLabel(
            "Subtract verified known-phase models, curate the unexplained reflections, "
            "search candidate unit cells with the built-in Afruz native indexer, test chemical plausibility, "
            "and transfer a selected indexed cell to Phase 10 for Pawley/Le Bail confirmation."
        )
        intro.setWordWrap(True)
        intro.setObjectName("mutedLabel")
        install_label_copy_menu(intro)
        layout.addWidget(intro)

        self.setup_splitter = QSplitter(Qt.Horizontal)
        self.setup_splitter.setChildrenCollapsible(False)
        self.setup_splitter.setMaximumHeight(330)
        self.setup_splitter.addWidget(self._build_peak_group())
        self.setup_splitter.addWidget(self._build_control_scroll())
        self.setup_splitter.setSizes([950, 420])
        self.setup_splitter.setStretchFactor(0, 3)
        self.setup_splitter.setStretchFactor(1, 1)
        layout.addWidget(self.setup_splitter)

        actions = QHBoxLayout()
        self.residual_button = QPushButton("Create unknown residual")
        self.detect_button = QPushButton("Detect unknown peaks")
        self.index_button = QPushButton("Run native Afruz indexing")
        self.index_button.setObjectName("primaryButton")
        self.cancel_button = QPushButton("Cancel indexing")
        self.cancel_button.setEnabled(False)
        self.handoff_button = QPushButton("Send selected cell to Phase 10")
        self.handoff_button.setEnabled(False)
        self.export_button = QPushButton("Export discovery package")
        actions.addWidget(self.residual_button)
        actions.addWidget(self.detect_button)
        actions.addWidget(self.index_button)
        actions.addWidget(self.cancel_button)
        actions.addStretch(1)
        actions.addWidget(self.handoff_button)
        actions.addWidget(self.export_button)
        layout.addLayout(actions)

        self.status_label = QLabel("Create a residual pattern to begin.")
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
        self.plot = PhaseRevolutionPlotWidget()
        self.tabs.addTab(self.plot, "Unknown Residual")
        self.tabs.addTab(self._candidate_page(), "Candidate Cells")
        self.tabs.addTab(self._diagnostics_page(), "Discovery Record")
        layout.addWidget(self.tabs, 1)

        note = QLabel(
            "A candidate cell is not a solved crystal structure. It must survive Pawley/Le Bail "
            "verification, systematic-absence analysis, chemical checks and structure solution "
            "before a provisional CIF can be treated as a physical phase model."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        layout.addWidget(note)

        self.residual_button.clicked.connect(self.create_residual)
        self.detect_button.clicked.connect(self.detect_peaks)
        self.index_button.clicked.connect(self.run_indexing)
        self.cancel_button.clicked.connect(self.cancel_indexing)
        self.handoff_button.clicked.connect(self.handoff_to_phase10)
        self.export_button.clicked.connect(self.export_package)
        self.peak_table.itemChanged.connect(self._peak_table_changed)
        self.add_peak_button.clicked.connect(self.add_manual_peak_dialog)
        self.add_plot_peak_button.toggled.connect(self._toggle_plot_peak_add)
        self.delete_peak_button.clicked.connect(self.delete_selected_peaks)
        self.freeze_peak_list.toggled.connect(self._master_lock_toggled)
        self.share_master_peaks.toggled.connect(self._share_master_toggled)
        self.import_peak_button.clicked.connect(self.import_master_peak_list)
        self.export_peak_button.clicked.connect(self.export_master_peak_list)
        self.plot.peakAddRequested.connect(self.add_manual_peak_at_position)
        self.candidate_table.itemSelectionChanged.connect(self._candidate_selected)
        self.browse_work_button.clicked.connect(self._browse_work_directory)
        self.open_work_button.clicked.connect(self._open_work_directory)

    def _build_peak_group(self):
        group = QGroupBox("Unknown reflection list")
        group.setMinimumWidth(0)
        layout = QVBoxLayout(group)
        self.peak_table = QTableWidget(0, 14)
        self.peak_table.setHorizontalHeaderLabels(
            [
                "Use",
                "Peak",
                "2θ (°)",
                "σ position (°)",
                "Intensity",
                "Prominence",
                "S/N",
                "Quality",
                "FWHM (°)",
                "Nearest known (°)",
                "Overlap",
                "Note",
                "Origin",
                "Protected",
            ]
        )
        self._stabilize_table(self.peak_table)
        header = self.peak_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(11, QHeaderView.Stretch)
        for column, width in {
            0: 58,
            1: 55,
            2: 92,
            3: 105,
            4: 100,
            5: 100,
            6: 75,
            7: 75,
            8: 85,
            9: 115,
            10: 75,
            12: 125,
            13: 85,
        }.items():
            self.peak_table.setColumnWidth(column, width)
        self.peak_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.peak_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        install_table_copy_menu(self.peak_table)
        layout.addWidget(self.peak_table)

        toolbar = QHBoxLayout()
        self.add_peak_button = QPushButton("Add peak at 2θ…")
        self.add_plot_peak_button = QPushButton("Click plot to add")
        self.add_plot_peak_button.setCheckable(True)
        self.delete_peak_button = QPushButton("Delete selected")
        self.import_peak_button = QPushButton("Import list…")
        self.export_peak_button = QPushButton("Export list…")
        self.freeze_peak_list = QCheckBox("Freeze master list")
        self.share_master_peaks = QCheckBox("Use in all peak-based analysis")
        self.share_master_peaks.setChecked(True)
        toolbar.addWidget(self.add_peak_button)
        toolbar.addWidget(self.add_plot_peak_button)
        toolbar.addWidget(self.delete_peak_button)
        toolbar.addWidget(self.import_peak_button)
        toolbar.addWidget(self.export_peak_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.share_master_peaks)
        toolbar.addWidget(self.freeze_peak_list)
        layout.addLayout(toolbar)

        self.master_peak_status = QLabel("Master list: empty")
        self.master_peak_status.setObjectName("mutedLabel")
        self.master_peak_status.setWordWrap(False)
        self.master_peak_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        install_label_copy_menu(self.master_peak_status)
        layout.addWidget(self.master_peak_status)
        return group

    def _build_control_scroll(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(350)
        scroll.setMaximumWidth(620)
        content = QWidget()
        form = QFormLayout(content)
        form.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        self.source_mode = NoWheelComboBox()
        self.source_mode.addItems(
            [
                "Auto: Rietveld → Pawley/Le Bail → observed",
                "Phase 11 Rietveld result",
                "Phase 10 Pawley/Le Bail result",
                "Observed pattern only",
            ]
        )
        self.use_processed = QCheckBox("Use processed observed pattern")
        self.use_processed.setChecked(True)
        self.subtract_background = QCheckBox("Subtract refined background")
        self.subtract_background.setChecked(True)
        self.smart_peak_search = QCheckBox("Use Smart Smoothing peak detector")
        self.smart_peak_search.setChecked(True)
        self.wavelength = NoWheelDoubleSpinBox()
        self.wavelength.setRange(0.1, 10.0)
        self.wavelength.setDecimals(7)
        self.wavelength.setValue(1.5406)
        self.range_min = NoWheelDoubleSpinBox()
        self.range_min.setRange(0.0, 179.0)
        self.range_min.setDecimals(4)
        self.range_max = NoWheelDoubleSpinBox()
        self.range_max.setRange(0.1, 179.9)
        self.range_max.setDecimals(4)
        self.prominence = NoWheelDoubleSpinBox()
        self.prominence.setRange(0.01, 100.0)
        self.prominence.setDecimals(3)
        self.prominence.setValue(3.0)
        self.minimum_snr = NoWheelDoubleSpinBox()
        self.minimum_snr.setRange(2.0, 50.0)
        self.minimum_snr.setDecimals(2)
        self.minimum_snr.setValue(4.5)
        self.minimum_quality = NoWheelDoubleSpinBox()
        self.minimum_quality.setRange(0.0, 100.0)
        self.minimum_quality.setDecimals(1)
        self.minimum_quality.setValue(35.0)
        self.minimum_distance = NoWheelDoubleSpinBox()
        self.minimum_distance.setRange(0.001, 5.0)
        self.minimum_distance.setDecimals(4)
        self.minimum_distance.setValue(0.12)
        self.overlap_exclusion = NoWheelDoubleSpinBox()
        self.overlap_exclusion.setRange(0.0, 2.0)
        self.overlap_exclusion.setDecimals(4)
        self.overlap_exclusion.setValue(0.08)
        self.maximum_peaks = NoWheelSpinBox()
        self.maximum_peaks.setRange(7, 120)
        self.maximum_peaks.setValue(40)

        form.addRow("Residual source", self.source_mode)
        form.addRow(self.use_processed)
        form.addRow(self.subtract_background)
        form.addRow(self.smart_peak_search)
        form.addRow("Wavelength (Å)", self.wavelength)
        form.addRow("2θ minimum", self.range_min)
        form.addRow("2θ maximum", self.range_max)
        form.addRow("Manual prominence (% range)", self.prominence)
        form.addRow("Smart minimum S/N", self.minimum_snr)
        form.addRow("Smart minimum quality", self.minimum_quality)
        form.addRow("Minimum spacing (°)", self.minimum_distance)
        form.addRow("Known-overlap exclusion (°)", self.overlap_exclusion)
        form.addRow("Maximum peaks", self.maximum_peaks)

        separator = QLabel("Afruz native indexing")
        separator.setObjectName("sectionTitle")
        form.addRow(separator)

        native_note = QLabel(
            "Runs entirely inside Afruz with NumPy/SciPy. No external program, "
            "Python environment, executable, instrument file, or native DLL is required."
        )
        native_note.setWordWrap(True)
        native_note.setObjectName("mutedLabel")
        form.addRow(native_note)

        self.work_directory = QLineEdit("")
        self.browse_work_button = QPushButton("Browse…")
        self.open_work_button = QPushButton("Open folder")
        work_row = QWidget()
        work_layout = QHBoxLayout(work_row)
        work_layout.setContentsMargins(0, 0, 0, 0)
        work_layout.addWidget(self.work_directory, 1)
        work_layout.addWidget(self.browse_work_button)
        work_layout.addWidget(self.open_work_button)
        form.addRow("Indexing record directory", work_row)

        self.starting_volume = NoWheelDoubleSpinBox()
        self.starting_volume.setRange(5.0, 1000000.0)
        self.starting_volume.setDecimals(3)
        self.starting_volume.setValue(200.0)
        self.timeout = NoWheelSpinBox()
        self.timeout.setRange(2, 600)
        self.timeout.setValue(35)
        self.minimum_m20 = NoWheelDoubleSpinBox()
        self.minimum_m20.setRange(0.0, 1000.0)
        self.minimum_m20.setDecimals(2)
        self.minimum_m20.setValue(2.0)
        self.maximum_x20 = NoWheelSpinBox()
        self.maximum_x20.setRange(0, 100)
        self.maximum_x20.setValue(10)
        self.zero_shift = NoWheelDoubleSpinBox()
        self.zero_shift.setRange(-2.0, 2.0)
        self.zero_shift.setDecimals(6)
        self.zero_shift.setValue(0.0)
        self.refine_zero_shift = QCheckBox("Refine zero shift")
        self.refine_zero_shift.setChecked(True)
        self.maximum_zero_shift = NoWheelDoubleSpinBox()
        self.maximum_zero_shift.setRange(0.001, 2.0)
        self.maximum_zero_shift.setDecimals(4)
        self.maximum_zero_shift.setValue(0.25)
        self.peak_tolerance = NoWheelDoubleSpinBox()
        self.peak_tolerance.setRange(0.005, 1.0)
        self.peak_tolerance.setDecimals(4)
        self.peak_tolerance.setValue(0.10)
        self.impurity_tolerance = NoWheelDoubleSpinBox()
        self.impurity_tolerance.setRange(0.0, 45.0)
        self.impurity_tolerance.setDecimals(1)
        self.impurity_tolerance.setValue(15.0)
        self.maximum_index = NoWheelSpinBox()
        self.maximum_index.setRange(4, 24)
        self.maximum_index.setValue(10)
        self.global_iterations = NoWheelSpinBox()
        self.global_iterations.setRange(5, 200)
        self.global_iterations.setValue(32)
        self.population_size = NoWheelSpinBox()
        self.population_size.setRange(3, 30)
        self.population_size.setValue(7)
        self.search_seeds = NoWheelSpinBox()
        self.search_seeds.setRange(1, 20)
        self.search_seeds.setValue(5)
        self.candidates_per_lattice = NoWheelSpinBox()
        self.candidates_per_lattice.setRange(1, 12)
        self.candidates_per_lattice.setValue(4)
        self.random_seed = NoWheelSpinBox()
        self.random_seed.setRange(0, 999999999)
        self.random_seed.setValue(1600)

        form.addRow("Starting cell volume (Å³)", self.starting_volume)
        form.addRow("Time limit per lattice (s)", self.timeout)
        form.addRow("Minimum M20 display threshold", self.minimum_m20)
        form.addRow("Maximum X20 display threshold", self.maximum_x20)
        form.addRow("Initial zero shift (°)", self.zero_shift)
        form.addRow(self.refine_zero_shift)
        form.addRow("Maximum |zero shift| (°)", self.maximum_zero_shift)
        form.addRow("Peak matching tolerance (°)", self.peak_tolerance)
        form.addRow("Allowed impurity peaks (%)", self.impurity_tolerance)
        form.addRow("Maximum h/k/l", self.maximum_index)
        form.addRow("Global-search iterations", self.global_iterations)
        form.addRow("Population size", self.population_size)
        form.addRow("Independent search seeds", self.search_seeds)
        form.addRow("Candidates per lattice", self.candidates_per_lattice)
        form.addRow("Reproducible random seed", self.random_seed)

        lattice_group = QGroupBox("Bravais lattices")
        grid = QGridLayout(lattice_group)
        self.bravais_checks: dict[str, QCheckBox] = {}
        for index, name in enumerate(BRAVAIS_NAMES):
            check = QCheckBox(name)
            check.setChecked(index <= 12)
            if name == "Triclinic":
                check.setChecked(False)
                check.setToolTip(
                    "Experimental and computationally expensive. Start with higher symmetry."
                )
            self.bravais_checks[name] = check
            grid.addWidget(check, index // 2, index % 2)
        form.addRow(lattice_group)

        chemistry = QLabel("Chemical plausibility")
        chemistry.setObjectName("sectionTitle")
        form.addRow(chemistry)
        self.formula_mass = NoWheelDoubleSpinBox()
        self.formula_mass.setRange(0.0, 100000.0)
        self.formula_mass.setDecimals(5)
        self.formula_mass.setValue(0.0)
        self.z_values = QLineEdit("1,2,4,8")
        self.density_min = NoWheelDoubleSpinBox()
        self.density_min.setRange(0.0, 100.0)
        self.density_min.setDecimals(4)
        self.density_min.setValue(0.5)
        self.density_max = NoWheelDoubleSpinBox()
        self.density_max.setRange(0.01, 100.0)
        self.density_max.setDecimals(4)
        self.density_max.setValue(10.0)
        form.addRow("Formula mass (g mol⁻¹)", self.formula_mass)
        form.addRow("Candidate Z values", self.z_values)
        form.addRow("Density minimum (g cm⁻³)", self.density_min)
        form.addRow("Density maximum (g cm⁻³)", self.density_max)

        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _line_browse(line: QLineEdit, button: QPushButton):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line, 1)
        layout.addWidget(button)
        return widget

    def _candidate_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.candidate_table = QTableWidget(0, 14)
        self.candidate_table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Status",
                "Bravais",
                "M20",
                "X20",
                "Indexed peaks",
                "a",
                "b",
                "c",
                "α",
                "β",
                "γ",
                "Volume",
                "Plausible densities",
            ]
        )
        self._stabilize_table(self.candidate_table)
        header = self.candidate_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(13, QHeaderView.Stretch)
        for column, width in {
            0: 55,
            1: 90,
            2: 150,
            3: 75,
            4: 60,
            5: 100,
            6: 85,
            7: 85,
            8: 85,
            9: 75,
            10: 75,
            11: 75,
            12: 100,
        }.items():
            self.candidate_table.setColumnWidth(column, width)
        self.candidate_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.candidate_table.setSelectionMode(QAbstractItemView.SingleSelection)
        install_table_copy_menu(self.candidate_table)
        layout.addWidget(self.candidate_table)
        return page

    def _diagnostics_page(self):
        outer = QWidget()
        layout = QVBoxLayout(outer)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        form = QFormLayout(content)
        self.diag_dataset = QLabel("—")
        self.diag_source = QLabel("—")
        self.diag_peak_count = QLabel("—")
        self.diag_peak_revision = QLabel("—")
        self.diag_peak_checksum = QLabel("—")
        self.diag_backend = QLabel("—")
        self.diag_files = QLabel("—")
        self.diag_warnings = QLabel("—")
        self.diag_warnings.setWordWrap(True)
        for label in (
            self.diag_dataset,
            self.diag_source,
            self.diag_peak_count,
            self.diag_peak_revision,
            self.diag_peak_checksum,
            self.diag_backend,
            self.diag_files,
            self.diag_warnings,
        ):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setMinimumWidth(0)
            install_label_copy_menu(label)
        form.addRow("Dataset", self.diag_dataset)
        form.addRow("Residual source", self.diag_source)
        form.addRow("Included / total peaks", self.diag_peak_count)
        form.addRow("Master-list revision", self.diag_peak_revision)
        form.addRow("Master-list checksum", self.diag_peak_checksum)
        form.addRow("Indexing backend", self.diag_backend)
        form.addRow("Backend files", self.diag_files)
        form.addRow("Warnings", self.diag_warnings)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        return outer

    @staticmethod
    def _stabilize_table(table: QTableWidget):
        table.setMinimumSize(0, 0)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.horizontalHeader().setMinimumSectionSize(45)
        table.horizontalHeader().setStretchLastSection(False)

    @staticmethod
    def _fmt(value, digits=7):
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}g}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _status_text(message: str, maximum: int = 125):
        compact = " ".join(str(message).split())
        return compact if len(compact) <= maximum else compact[: maximum - 1] + "…"

    def _set_status(self, message: str):
        self.status_label.setText(self._status_text(message))
        self.status_label.setToolTip(str(message))

    def _meta_for_uid(self, dataset_uid: str) -> dict:
        peaks = normalize_master_peak_list(self.peaks_by_uid.get(dataset_uid, []))
        self.peaks_by_uid[dataset_uid] = peaks
        checksum = master_peak_list_checksum(peaks)
        meta = self.peak_list_meta_by_uid.setdefault(
            dataset_uid,
            {
                "revision": 0,
                "locked": False,
                "checksum": checksum,
                "last_reason": "Created",
                "last_indexed_checksum": None,
                "last_indexed_revision": None,
            },
        )
        meta.setdefault("revision", 0)
        meta.setdefault("locked", False)
        meta.setdefault("last_indexed_checksum", None)
        meta.setdefault("last_indexed_revision", None)
        meta["checksum"] = checksum
        return meta

    def master_peak_list(self, dataset_uid: str) -> list[dict]:
        peaks = normalize_master_peak_list(self.peaks_by_uid.get(dataset_uid, []))
        self.peaks_by_uid[dataset_uid] = peaks
        self._meta_for_uid(dataset_uid)
        return deepcopy(peaks)

    def master_peak_metadata(self, dataset_uid: str) -> dict:
        return deepcopy(self._meta_for_uid(dataset_uid))

    def _analysis_peak_rows(self, dataset_uid: str) -> list[dict]:
        dataset = next(
            (item for item in self.main_window.datasets if item.uid == dataset_uid),
            None,
        )
        x = None if dataset is None else dataset.x
        rows = []
        for peak in self.master_peak_list(dataset_uid):
            if not peak.get("use", True):
                continue
            position = float(peak["two_theta_deg"])
            index = 0
            if x is not None and len(x):
                index = int(np.argmin(np.abs(x - position)))
            quality = float(peak.get("quality_score", 0.0))
            rows.append(
                {
                    "position": position,
                    "position_error": peak.get("position_uncertainty_deg"),
                    "intensity": float(peak.get("intensity", 0.0)),
                    "prominence": float(peak.get("prominence", 0.0)),
                    "fwhm": float(peak.get("fwhm_deg", 0.0)),
                    "index": index,
                    "method": "Phase Revolution Master",
                    "snr": float(peak.get("signal_to_noise", 0.0)),
                    "confidence": quality,
                    "confidence_label": (
                        "High" if quality >= 80.0
                        else "Medium" if quality >= 60.0
                        else "Manual/Review"
                    ),
                    "peak_uuid": peak.get("peak_uuid"),
                    "origin": peak.get("origin"),
                    "master_peak_revision": self._meta_for_uid(dataset_uid).get("revision"),
                    "master_peak_checksum": self._meta_for_uid(dataset_uid).get("checksum"),
                }
            )
        return rows

    def _sync_master_to_application(self, dataset_uid: str):
        if not self.share_master_peaks.isChecked():
            return
        master = self.master_peak_list(dataset_uid)
        if not master:
            return
        rows = self._analysis_peak_rows(dataset_uid)
        checksum = self._meta_for_uid(dataset_uid).get("checksum")
        existing = self.main_window.peak_rows.get(dataset_uid, [])
        already_synced = bool(existing) and all(
            row.get("master_peak_checksum") == checksum for row in existing
        )
        if already_synced:
            return
        self.main_window.peak_rows[dataset_uid] = rows
        for mapping_name in (
            "fit_groups",
            "fit_candidates",
            "size_strain_results",
            "cell_refinement_results",
            "phase_identification_results",
            "qpa_results",
        ):
            mapping = getattr(self.main_window, mapping_name, None)
            if isinstance(mapping, dict):
                mapping.pop(dataset_uid, None)
        if hasattr(self.main_window, "populate_peak_table"):
            self.main_window.populate_peak_table()

    def _share_master_toggled(self, checked: bool):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        if checked:
            self._sync_master_to_application(dataset.uid)
            self._set_status("The master reflection list is now shared with all peak-based analyses.")
        else:
            self._set_status("The master reflection list remains stored, but global peak analyses may use their own lists.")

    def is_master_shared_and_locked(self, dataset_uid: str) -> bool:
        return bool(
            self.share_master_peaks.isChecked()
            and self._meta_for_uid(dataset_uid).get("locked")
            and self.peaks_by_uid.get(dataset_uid)
        )

    def _invalidate_peak_dependents(self, dataset_uid: str, reason: str):
        self.indexing_by_uid.pop(dataset_uid, None)
        self.candidates_by_uid.pop(dataset_uid, None)
        structure = getattr(self.main_window, "structure_solution_widget", None)
        if structure is not None:
            for name in ("screening_by_uid", "robustness_by_uid", "packages_by_uid"):
                mapping = getattr(structure, name, None)
                if isinstance(mapping, dict):
                    mapping.pop(dataset_uid, None)
        self.handoff_button.setEnabled(False)
        if reason:
            self._set_status(
                f"Master peak list changed: {reason}. Previous indexing and downstream peak-based results were invalidated."
            )

    def _commit_master_peaks(
        self,
        dataset_uid: str,
        peaks,
        *,
        reason: str,
        invalidate: bool = True,
        bump_revision: bool = True,
    ) -> list[dict]:
        normalized = normalize_master_peak_list(peaks)
        old_checksum = self._meta_for_uid(dataset_uid).get("checksum")
        new_checksum = master_peak_list_checksum(normalized)
        meta = self._meta_for_uid(dataset_uid)
        changed = old_checksum != new_checksum
        self.peaks_by_uid[dataset_uid] = normalized
        if changed and bump_revision:
            meta["revision"] = int(meta.get("revision", 0)) + 1
        meta["checksum"] = new_checksum
        meta["last_reason"] = reason
        if changed and invalidate:
            self._invalidate_peak_dependents(dataset_uid, reason)
        self.populate_peaks(normalized)
        self.plot.set_data(self.residuals_by_uid.get(dataset_uid), normalized)
        self._sync_master_to_application(dataset_uid)
        self._refresh_master_status(dataset_uid)
        self.refresh_diagnostics()
        return normalized

    def _refresh_master_status(self, dataset_uid: str):
        peaks = self.peaks_by_uid.get(dataset_uid, [])
        meta = self._meta_for_uid(dataset_uid)
        state = "FROZEN" if meta.get("locked") else "editable"
        text = (
            f"Master list r{meta.get('revision', 0)} · {state} · "
            f"{sum(bool(row.get('use', True)) for row in peaks)}/{len(peaks)} included · "
            f"SHA-256 {meta.get('checksum', '')[:12]}…"
        )
        self.master_peak_status.setText(text)
        self.master_peak_status.setToolTip(meta.get("checksum", ""))

    def _set_master_locked(self, dataset_uid: str, locked: bool):
        meta = self._meta_for_uid(dataset_uid)
        meta["locked"] = bool(locked)
        self._updating_peaks = True
        self.freeze_peak_list.setChecked(bool(locked))
        self._updating_peaks = False
        self.populate_peaks(self.peaks_by_uid.get(dataset_uid, []))
        self.detect_button.setEnabled(not locked)
        self.add_peak_button.setEnabled(not locked)
        self.add_plot_peak_button.setEnabled(not locked)
        self.delete_peak_button.setEnabled(not locked)
        self.import_peak_button.setEnabled(not locked)
        if locked and self.add_plot_peak_button.isChecked():
            self.add_plot_peak_button.setChecked(False)
        self._sync_master_to_application(dataset_uid)
        self._refresh_master_status(dataset_uid)
        self.refresh_diagnostics()

    def _master_lock_toggled(self, checked: bool):
        if self._updating_peaks:
            return
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        self._set_master_locked(dataset.uid, checked)
        self._set_status(
            "Master peak list frozen for all downstream analysis."
            if checked
            else "Master peak list unlocked; the next edit will invalidate dependent results."
        )

    def _toggle_plot_peak_add(self, checked: bool):
        self.plot.set_manual_add_enabled(bool(checked))
        self.add_plot_peak_button.setText("Click plot: ON" if checked else "Click plot to add")

    def add_manual_peak_dialog(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        residual = self.residuals_by_uid.get(dataset.uid)
        if not residual:
            QMessageBox.information(self, "No residual", "Create the unknown residual first.")
            return
        position, accepted = QInputDialog.getDouble(
            self,
            "Add manual master peak",
            "2θ position (degrees):",
            value=float((self.range_min.value() + self.range_max.value()) / 2.0),
            min=float(residual["x"][0]),
            max=float(residual["x"][-1]),
            decimals=6,
        )
        if accepted:
            self.add_manual_peak_at_position(position)

    def add_manual_peak_at_position(self, position: float):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        meta = self._meta_for_uid(dataset.uid)
        if meta.get("locked"):
            QMessageBox.information(self, "Master list frozen", "Unlock the master list before adding a peak.")
            return
        residual = self.residuals_by_uid.get(dataset.uid)
        if not residual:
            QMessageBox.information(self, "No residual", "Create the unknown residual first.")
            return
        try:
            manual = manual_peak_from_position(
                residual["x"],
                residual["positive_residual"],
                position,
                known_reflections=self._known_reflections(dataset.uid),
                overlap_exclusion_deg=self.overlap_exclusion.value(),
                default_fwhm_deg=self.minimum_distance.value(),
            )
        except PhaseRevolutionError as exc:
            QMessageBox.critical(self, "Manual peak failed", str(exc))
            return
        peaks = self.master_peak_list(dataset.uid)
        tolerance = max(0.005, self.minimum_distance.value() / 3.0)
        nearest = min(
            range(len(peaks)),
            key=lambda index: abs(peaks[index]["two_theta_deg"] - manual["two_theta_deg"]),
            default=None,
        )
        if nearest is not None and abs(peaks[nearest]["two_theta_deg"] - manual["two_theta_deg"]) <= tolerance:
            existing = peaks[nearest]
            for key in (
                "two_theta_deg", "position_uncertainty_deg", "intensity",
                "prominence", "signal_to_noise", "quality_score", "fwhm_deg",
                "noise_sigma", "nearest_known_distance_deg", "overlap_flag",
            ):
                existing[key] = manual[key]
            existing["manual"] = True
            existing["row_locked"] = True
            existing["origin"] = "Manual"
            existing["use"] = True
            existing["note"] = "Manual peak; converted from nearby detected reflection"
            reason = f"converted peak near {manual['two_theta_deg']:.6g}° to manual"
        else:
            peaks.append(manual)
            reason = f"added manual peak at {manual['two_theta_deg']:.6g}°"
        self._commit_master_peaks(dataset.uid, peaks, reason=reason)

    def delete_selected_peaks(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        if self._meta_for_uid(dataset.uid).get("locked"):
            QMessageBox.information(self, "Master list frozen", "Unlock the master list before deleting peaks.")
            return
        selected_rows = sorted({index.row() for index in self.peak_table.selectedIndexes()}, reverse=True)
        if not selected_rows:
            return
        peaks = self._read_peaks()
        for row in selected_rows:
            if 0 <= row < len(peaks):
                peaks.pop(row)
        self._commit_master_peaks(
            dataset.uid,
            peaks,
            reason=f"deleted {len(selected_rows)} selected peak(s)",
        )

    @staticmethod
    def _truthy(value):
        return str(value).strip().lower() in {"1", "true", "yes", "y", "checked"}

    def import_master_peak_list(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        if self._meta_for_uid(dataset.uid).get("locked"):
            QMessageBox.information(self, "Master list frozen", "Unlock the master list before importing peaks.")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import master reflection list",
            "",
            "Peak lists (*.json *.csv);;JSON (*.json);;CSV (*.csv)",
        )
        if not filename:
            return
        path = Path(filename)
        try:
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows = payload.get("peaks", []) if isinstance(payload, dict) else payload
            else:
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    for key in ("use", "overlap_flag", "manual", "row_locked"):
                        if key in row:
                            row[key] = self._truthy(row[key])
            peaks = normalize_master_peak_list(rows)
        except Exception as exc:
            QMessageBox.critical(self, "Peak-list import failed", str(exc))
            return
        self._commit_master_peaks(dataset.uid, peaks, reason=f"imported master list from {path.name}")

    def export_master_peak_list(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export master reflection list",
            f"{dataset.name}_master_peaks.txt",
            "Text data (*.txt)",
        )
        if not filename:
            return
        peaks = self.master_peak_list(dataset.uid)
        meta = self.master_peak_metadata(dataset.uid)
        try:
            path = export_unknown_peak_txt(
                filename,
                peaks,
                metadata={
                    "classification": "Phase Revolution master reflection list",
                    "dataset": dataset.name,
                    "dataset_uid": dataset.uid,
                    **meta,
                },
            )
        except Exception as exc:
            QMessageBox.critical(self, "TXT export failed", str(exc))
            return
        self._set_status(f"Master reflection list TXT exported to {path}")
    def _current_results(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return None, None, None
        whole = None
        rietveld = None
        if hasattr(self.main_window, "whole_pattern_widget"):
            whole = self.main_window.whole_pattern_widget.results_by_uid.get(dataset.uid)
        if hasattr(self.main_window, "rietveld_widget"):
            rietveld = self.main_window.rietveld_widget.results_by_uid.get(dataset.uid)
        mode = self.source_mode.currentText()
        if mode.startswith("Phase 11"):
            return dataset, None, rietveld
        if mode.startswith("Phase 10"):
            return dataset, whole, None
        if mode.startswith("Observed"):
            return dataset, None, None
        return dataset, whole, rietveld

    def create_residual(self):
        dataset, whole, rietveld = self._current_results()
        if dataset is None:
            QMessageBox.information(self, "No dataset", "Select an experimental dataset first.")
            return
        observed = dataset.y if self.use_processed.isChecked() else dataset.y_raw
        try:
            residual = extract_unknown_residual(
                dataset.x,
                observed,
                whole_pattern_result=whole,
                rietveld_result=rietveld,
                subtract_background=self.subtract_background.isChecked(),
            ).as_dict()
        except PhaseRevolutionError as exc:
            QMessageBox.critical(self, "Residual extraction failed", str(exc))
            return
        self.residuals_by_uid[dataset.uid] = residual
        existing = self.master_peak_list(dataset.uid)
        meta = self._meta_for_uid(dataset.uid)
        if existing and not meta.get("locked"):
            existing = refresh_master_peak_measurements(
                existing,
                residual["x"],
                residual["positive_residual"],
                known_reflections=self._known_reflections(dataset.uid),
                overlap_exclusion_deg=self.overlap_exclusion.value(),
            )
            existing = self._commit_master_peaks(
                dataset.uid,
                existing,
                reason="residual recreated and master peak measurements refreshed",
            )
        else:
            self.populate_peaks(existing)
            self.plot.set_data(residual, existing)
            if existing:
                self._invalidate_peak_dependents(
                    dataset.uid,
                    "residual source changed while frozen peak positions were preserved",
                )
        self.populate_candidates([])
        self._set_status(
            f"Unknown residual created from {residual['source']}; "
            f"{len(existing)} master peaks preserved."
        )
        self.refresh_diagnostics()

    def _known_reflections(self, dataset_uid: str):
        mode = self.source_mode.currentText()
        result = None
        if mode.startswith("Phase 11") and hasattr(self.main_window, "rietveld_widget"):
            result = self.main_window.rietveld_widget.results_by_uid.get(dataset_uid)
        elif mode.startswith("Phase 10") and hasattr(self.main_window, "whole_pattern_widget"):
            result = self.main_window.whole_pattern_widget.results_by_uid.get(dataset_uid)
        elif mode.startswith("Auto"):
            if hasattr(self.main_window, "rietveld_widget"):
                result = self.main_window.rietveld_widget.results_by_uid.get(dataset_uid)
            if result is None and hasattr(self.main_window, "whole_pattern_widget"):
                result = self.main_window.whole_pattern_widget.results_by_uid.get(dataset_uid)
        return [
            float(row["two_theta_deg"])
            for row in (result or {}).get("reflections", [])
            if row.get("two_theta_deg") is not None
        ]

    def detect_peaks(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        if self._meta_for_uid(dataset.uid).get("locked"):
            QMessageBox.information(
                self,
                "Master list frozen",
                "Unlock the master list before running automatic peak detection.",
            )
            return
        residual = self.residuals_by_uid.get(dataset.uid)
        if residual is None:
            self.create_residual()
            residual = self.residuals_by_uid.get(dataset.uid)
        if residual is None:
            return
        try:
            peaks = detect_unknown_peaks(
                residual["x"],
                residual["positive_residual"],
                prominence_percent=self.prominence.value(),
                minimum_distance_deg=self.minimum_distance.value(),
                minimum_two_theta=self.range_min.value(),
                maximum_two_theta=self.range_max.value(),
                known_reflections=self._known_reflections(dataset.uid),
                overlap_exclusion_deg=self.overlap_exclusion.value(),
                maximum_peaks=self.maximum_peaks.value(),
                smart_search=self.smart_peak_search.isChecked(),
                minimum_signal_to_noise=self.minimum_snr.value(),
                minimum_quality_score=self.minimum_quality.value(),
            )
        except PhaseRevolutionError as exc:
            QMessageBox.critical(self, "Unknown peak detection failed", str(exc))
            return
        protected_existing = [
            row for row in self.master_peak_list(dataset.uid)
            if row.get("manual") or row.get("row_locked")
        ]
        peaks = merge_master_peak_lists(
            protected_existing,
            peaks,
            duplicate_tolerance_deg=max(0.005, self.minimum_distance.value() / 3.0),
            preserve_manual=True,
        )
        peaks = self._commit_master_peaks(
            dataset.uid,
            peaks,
            reason="automatic detection refreshed while manual/protected peaks were preserved",
        )
        included = sum(bool(row.get("use", True)) for row in peaks)
        detector = "smart noise-aware" if self.smart_peak_search.isChecked() else "manual prominence"
        self._set_status(
            f"Detected {len(peaks)} {detector} residual peaks; "
            f"{included} are included for indexing."
        )
        self.refresh_diagnostics()

    def _read_peaks(self):
        rows = []
        for row in range(self.peak_table.rowCount()):
            use_item = self.peak_table.item(row, 0)
            try:
                rows.append(
                    {
                        "peak_id": int(float(self.peak_table.item(row, 1).text())),
                        "peak_uuid": str(self.peak_table.item(row, 1).data(Qt.UserRole) or ""),
                        "use": use_item.checkState() == Qt.Checked,
                        "two_theta_deg": float(self.peak_table.item(row, 2).text()),
                        "position_uncertainty_deg": float(self.peak_table.item(row, 3).text()),
                        "intensity": float(self.peak_table.item(row, 4).text()),
                        "prominence": float(self.peak_table.item(row, 5).text()),
                        "signal_to_noise": self._optional_float(self.peak_table.item(row, 6).text()) or 0.0,
                        "quality_score": self._optional_float(self.peak_table.item(row, 7).text()) or 0.0,
                        "fwhm_deg": float(self.peak_table.item(row, 8).text()),
                        "nearest_known_distance_deg": self._optional_float(self.peak_table.item(row, 9).text()),
                        "overlap_flag": self.peak_table.item(row, 10).text() == "Yes",
                        "note": self.peak_table.item(row, 11).text(),
                        "origin": self.peak_table.item(row, 12).text(),
                        "manual": self.peak_table.item(row, 12).text().lower().startswith("manual"),
                        "row_locked": self.peak_table.item(row, 13).text() == "Yes",
                    }
                )
            except (AttributeError, ValueError):
                continue
        return normalize_master_peak_list(rows)

    @staticmethod
    def _optional_float(text):
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    def populate_peaks(self, peaks):
        dataset = self.main_window.selected_dataset()
        locked = bool(dataset is not None and self._meta_for_uid(dataset.uid).get("locked"))
        peaks = normalize_master_peak_list(peaks)
        self._updating_peaks = True
        self.peak_table.setRowCount(len(peaks))
        for row_index, peak in enumerate(peaks):
            use = QTableWidgetItem("")
            if locked:
                use.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            else:
                use.setFlags(use.flags() | Qt.ItemIsUserCheckable)
            use.setCheckState(Qt.Checked if peak.get("use", True) else Qt.Unchecked)
            self.peak_table.setItem(row_index, 0, use)
            values = [
                peak.get("peak_id", row_index + 1),
                self._fmt(peak.get("two_theta_deg")),
                self._fmt(peak.get("position_uncertainty_deg")),
                self._fmt(peak.get("intensity")),
                self._fmt(peak.get("prominence")),
                self._fmt(peak.get("signal_to_noise")),
                self._fmt(peak.get("quality_score")),
                self._fmt(peak.get("fwhm_deg")),
                self._fmt(peak.get("nearest_known_distance_deg")),
                "Yes" if peak.get("overlap_flag") else "No",
                peak.get("note", ""),
                peak.get("origin", "Imported"),
                "Yes" if peak.get("row_locked") else "No",
            ]
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(str(value))
                if column == 1:
                    item.setData(Qt.UserRole, peak.get("peak_uuid"))
                editable = column in {2, 3, 11}
                if locked or not editable:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.peak_table.setItem(row_index, column, item)
        self._updating_peaks = False
        if dataset is not None:
            self._updating_peaks = True
            self.freeze_peak_list.setChecked(locked)
            self._updating_peaks = False
            self.detect_button.setEnabled(not locked)
            self.add_peak_button.setEnabled(not locked)
            self.add_plot_peak_button.setEnabled(not locked)
            self.delete_peak_button.setEnabled(not locked)
            self.import_peak_button.setEnabled(not locked)
            self._refresh_master_status(dataset.uid)

    def _peak_table_changed(self, _item):
        if self._updating_peaks:
            return
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        if self._meta_for_uid(dataset.uid).get("locked"):
            self.populate_peaks(self.peaks_by_uid.get(dataset.uid, []))
            return
        peaks = self._read_peaks()
        self._commit_master_peaks(dataset.uid, peaks, reason="manual table edit")

    def _selected_bravais(self):
        return [name for name, check in self.bravais_checks.items() if check.isChecked()]

    def _z_list(self):
        values = []
        for token in self.z_values.text().replace(";", ",").split(","):
            try:
                value = int(token.strip())
            except ValueError:
                continue
            if value > 0 and value not in values:
                values.append(value)
        return values

    def is_running(self):
        return bool(self._thread is not None and self._thread.isRunning())

    def run_indexing(self):
        if self.is_running():
            return
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        peaks = self._read_peaks()
        peaks = self._commit_master_peaks(
            dataset.uid,
            peaks,
            reason="synchronized table before native indexing",
            invalidate=False,
        )
        included = [row for row in peaks if row.get("use", True)]
        if len(included) < 7:
            QMessageBox.warning(
                self,
                "Too few unknown reflections",
                "Include at least seven reliable unknown reflections before indexing.",
            )
            return
        selected_bravais = self._selected_bravais()
        if not selected_bravais:
            QMessageBox.warning(
                self,
                "No lattice selected",
                "Select at least one Bravais lattice for the native search.",
            )
            return
        self._set_master_locked(dataset.uid, True)
        snapshot = self.master_peak_metadata(dataset.uid)
        self._active_peak_snapshot = snapshot
        work = self.work_directory.text().strip()
        if not work:
            work = tempfile.mkdtemp(prefix="afruz_native_indexing_")
            self.work_directory.setText(work)
        settings = {
            "wavelength_angstrom": self.wavelength.value(),
            "bravais_names": selected_bravais,
            "starting_volume_angstrom3": self.starting_volume.value(),
            "zero_shift_deg": self.zero_shift.value(),
            "refine_zero_shift": self.refine_zero_shift.isChecked(),
            "maximum_zero_shift_deg": self.maximum_zero_shift.value(),
            "peak_tolerance_deg": self.peak_tolerance.value(),
            "impurity_tolerance_fraction": self.impurity_tolerance.value() / 100.0,
            "maximum_index": self.maximum_index.value(),
            "global_iterations": self.global_iterations.value(),
            "population_size": self.population_size.value(),
            "candidate_seeds_per_lattice": self.search_seeds.value(),
            "maximum_candidates_per_lattice": self.candidates_per_lattice.value(),
            "random_seed": self.random_seed.value(),
            "timeout_seconds": self.timeout.value(),
            "minimum_m20": self.minimum_m20.value(),
            "maximum_x20": self.maximum_x20.value(),
            "formula_mass_g_mol": self.formula_mass.value() or None,
            "z_values": self._z_list(),
            "density_min_g_cm3": self.density_min.value(),
            "density_max_g_cm3": self.density_max.value(),
            "master_peak_revision": snapshot.get("revision"),
            "master_peak_checksum": snapshot.get("checksum"),
        }
        thread = QThread(self)
        worker = PhaseRevolutionIndexingWorker(work, peaks, settings)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._indexing_progress)
        worker.finished.connect(self._indexing_finished)
        worker.failed.connect(self._indexing_failed)
        worker.cancelled.connect(self._indexing_cancelled)
        worker.finished.connect(lambda *_: thread.quit())
        worker.failed.connect(lambda *_: thread.quit())
        worker.cancelled.connect(lambda *_: thread.quit())
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        self._active_dataset_uid = dataset.uid
        self.index_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.handoff_button.setEnabled(False)
        self._set_status(
            f"Starting native search across {len(selected_bravais)} Bravais lattices…"
        )
        thread.start()

    def _indexing_progress(self, completed, total, message):
        self._set_status(f"{message} ({completed}/{total})")

    def cancel_indexing(self):
        if self._worker is not None:
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self._set_status("Cancellation requested; any returned result will be discarded.")

    def _indexing_finished(self, result):
        uid = self._active_dataset_uid
        if uid is None:
            return
        snapshot = deepcopy(self._active_peak_snapshot or self._meta_for_uid(uid))
        result["master_peak_list"] = snapshot
        candidates = []
        for candidate in result.get("candidates", []):
            enriched = enrich_candidate_chemistry(
                candidate,
                formula_mass_g_mol=(self.formula_mass.value() or None),
                z_values=self._z_list(),
                density_min_g_cm3=self.density_min.value(),
                density_max_g_cm3=self.density_max.value(),
            )
            enriched["master_peak_checksum"] = snapshot.get("checksum")
            enriched["master_peak_revision"] = snapshot.get("revision")
            candidates.append(enriched)
        meta = self._meta_for_uid(uid)
        meta["last_indexed_checksum"] = snapshot.get("checksum")
        meta["last_indexed_revision"] = snapshot.get("revision")
        self.indexing_by_uid[uid] = result
        self.candidates_by_uid[uid] = candidates
        if hasattr(self.main_window, "_record_scientific_result"):
            self.main_window._record_scientific_result(
                "phase",
                uid,
                {"indexing": result, "candidate_cells": candidates},
                reason="Completed native Afruz indexing",
            )
        self.populate_candidates(candidates)
        if candidates:
            self._set_status(
                f"Native Afruz indexing returned {len(candidates)} candidate cells. "
                "Verify several candidates with Pawley/Le Bail refinement."
            )
        else:
            self._set_status(
                "Native indexing completed without a candidate cell. Review the master "
                "peak list, wavelength, lattice selection, cell bounds and search effort."
            )
        self.tabs.setCurrentIndex(1)
        self.refresh_diagnostics()

    def _indexing_failed(self, traceback_text):
        work = self.work_directory.text().strip()
        summary = "Native Afruz indexing failed. Open Details for the exact numerical error."
        self._set_status(summary)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Native indexing failed")
        box.setText(summary)
        if work:
            box.setInformativeText(
                "Diagnostic files were preserved in:\n" + work
            )
        box.setDetailedText(traceback_text)
        box.exec()

    def _indexing_cancelled(self):
        self._set_status("Indexing result discarded after cancellation request.")

    def _thread_finished(self):
        self.index_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._thread = None
        self._worker = None
        self._active_dataset_uid = None
        self._active_peak_snapshot = None
        self._candidate_selected()

    def populate_candidates(self, candidates):
        self.candidate_table.setRowCount(len(candidates))
        for row_index, candidate in enumerate(candidates):
            densities = ", ".join(
                f"Z={row['z']}: {row['density_g_cm3']:.4g}"
                + (" ✓" if row["plausible"] else "")
                for row in candidate.get("density_checks", [])
            ) or "Not assessed"
            values = [
                candidate.get("rank", row_index + 1),
                candidate.get("status", "Review"),
                candidate.get("bravais_name", ""),
                self._fmt(candidate.get("m20")),
                candidate.get("x20", ""),
                candidate.get("indexed_peak_count", "—"),
                self._fmt(candidate.get("a_angstrom")),
                self._fmt(candidate.get("b_angstrom")),
                self._fmt(candidate.get("c_angstrom")),
                self._fmt(candidate.get("alpha_deg")),
                self._fmt(candidate.get("beta_deg")),
                self._fmt(candidate.get("gamma_deg")),
                self._fmt(candidate.get("volume_angstrom3")),
                densities,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if column == 0:
                    item.setData(Qt.UserRole, deepcopy(candidate))
                self.candidate_table.setItem(row_index, column, item)
        if candidates:
            self.candidate_table.selectRow(0)
        self._candidate_selected()

    def _selected_candidate(self):
        row = self.candidate_table.currentRow()
        if row < 0:
            return None
        item = self.candidate_table.item(row, 0)
        return None if item is None else item.data(Qt.UserRole)

    def _candidate_selected(self):
        dataset = self.main_window.selected_dataset()
        candidate = self._selected_candidate()
        valid = False
        if dataset is not None and candidate is not None:
            valid = (
                candidate.get("master_peak_checksum")
                == self._meta_for_uid(dataset.uid).get("checksum")
            )
        self.handoff_button.setEnabled(valid and not self.is_running())

    def handoff_to_phase10(self):
        candidate = self._selected_candidate()
        dataset = self.main_window.selected_dataset()
        if candidate is None or dataset is None:
            return
        meta = self._meta_for_uid(dataset.uid)
        if candidate.get("master_peak_checksum") != meta.get("checksum"):
            QMessageBox.warning(
                self,
                "Peak-list revision mismatch",
                "The master reflection list changed after this candidate was indexed. "
                "Run native Afruz indexing again before handoff.",
            )
            return
        master_peaks = self.master_peak_list(dataset.uid)
        try:
            reference_dataset = candidate_reference_dataset(
                candidate,
                wavelength_angstrom=self.wavelength.value(),
                two_theta_min=self.range_min.value(),
                two_theta_max=self.range_max.value(),
                observed_master_peaks=master_peaks,
                master_peak_metadata=meta,
            )
        except PhaseRevolutionError as exc:
            QMessageBox.critical(self, "Candidate handoff failed", str(exc))
            return
        selected_row = self.main_window.dataset_list.currentRow()
        self.main_window.datasets.append(reference_dataset)
        self.main_window.refresh_dataset_list(selected_row)
        if hasattr(self.main_window, "whole_pattern_widget"):
            self.main_window.whole_pattern_widget.refresh_references()
            self.main_window.tabs.setCurrentWidget(self.main_window.whole_pattern_widget)
        self.main_window.statusBar().showMessage(
            "Candidate cell added as an indexed hkl reference. Run Pawley/Le Bail verification before acceptance."
        )

    def export_package(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Export Unknown Phase discovery package")
        if not directory:
            return
        out = Path(directory) / f"{dataset.name}_phase14_discovery"
        out.mkdir(parents=True, exist_ok=True)
        residual = self.residuals_by_uid.get(dataset.uid)
        peaks = self.peaks_by_uid.get(dataset.uid, [])
        candidates = self.candidates_by_uid.get(dataset.uid, [])
        indexing = self.indexing_by_uid.get(dataset.uid)
        generated = []
        try:
            if residual:
                result = write_columns_txt(
                    out / "unknown_residual.txt",
                    {
                        "two_theta_deg": residual["x"],
                        "observed": residual["observed"],
                        "background": residual["background"],
                        "known_phase_calculated": residual["known_phase_calculated"],
                        "signed_residual": residual["signed_residual"],
                        "positive_residual": residual["positive_residual"],
                    },
                    title=f"Unknown-phase residual — {dataset.name}",
                    metadata={"dataset_uid": dataset.uid, "residual_source": residual.get("source")},
                    backup=False,
                )
                generated.append(Path(result["txt_path"]))
            peak_meta = self.master_peak_metadata(dataset.uid)
            generated.append(export_unknown_peak_txt(
                out / "master_reflection_list.txt", peaks,
                metadata={"dataset": dataset.name, "dataset_uid": dataset.uid, **peak_meta},
            ))
            result = write_table_txt(
                out / "candidate_cells.txt", candidates,
                title=f"Native indexing candidate cells — {dataset.name}",
                metadata={"dataset_uid": dataset.uid}, backup=False,
            )
            generated.append(Path(result["txt_path"]))
            record = {
                "classification": "Indexed unknown phase — structure unsolved" if candidates else "Peak list prepared",
                "dataset": dataset.name,
                "dataset_uid": dataset.uid,
                "wavelength_angstrom": self.wavelength.value(),
                "residual_source": None if not residual else residual.get("source"),
                "peak_count": len(peaks),
                "included_peak_count": sum(bool(row.get("use", True)) for row in peaks),
                "master_peak_list": peak_meta,
                "indexing": indexing,
                "scientific_warning": (
                    "Candidate cells are hypotheses. Pawley/Le Bail verification, space-group screening, "
                    "structure solution and Rietveld validation remain required."
                ),
            }
            result = write_mapping_txt(
                out / "phase_revolution_record.txt", record,
                title=f"Phase Revolution record — {dataset.name}", backup=False,
            )
            generated.append(Path(result["txt_path"]))
            work = self.work_directory.text().strip()
            if work and Path(work).is_dir():
                work_path = Path(work)
                for filename in ("native_indexing_request.json", "native_indexing_progress.json", "native_indexing_result.json"):
                    source = work_path / filename
                    if source.exists():
                        data = json.loads(source.read_text(encoding="utf-8"))
                        result = write_mapping_txt(
                            out / Path(filename).with_suffix(".txt"), data,
                            title=Path(filename).stem.replace("_", " ").title(), backup=False,
                        )
                        generated.append(Path(result["txt_path"]))
            manifest = write_manifest_txt(
                out / "manifest.txt", generated, root=out,
                title=f"Unknown Phase discovery package manifest — {dataset.name}", backup=False,
            )
        except Exception as exc:
            QMessageBox.critical(self, "TXT package export failed", str(exc))
            return
        self._set_status(f"Clean TXT discovery package exported to {out}")
    def _browse_work_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select indexing work directory")
        if directory:
            self.work_directory.setText(directory)

    def _open_work_directory(self):
        work = self.work_directory.text().strip()
        if not work:
            QMessageBox.information(
                self,
                "No indexing directory",
                "Run indexing once or select an indexing work directory first.",
            )
            return
        path = Path(work).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def refresh_diagnostics(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        residual = self.residuals_by_uid.get(dataset.uid)
        peaks = self.peaks_by_uid.get(dataset.uid, [])
        result = self.indexing_by_uid.get(dataset.uid, {})
        meta = self._meta_for_uid(dataset.uid)
        self.diag_dataset.setText(dataset.name)
        self.diag_source.setText("—" if not residual else str(residual.get("source")))
        self.diag_peak_count.setText(
            f"{sum(bool(row.get('use', True)) for row in peaks)} / {len(peaks)}"
        )
        self.diag_peak_revision.setText(
            f"r{meta.get('revision', 0)} — "
            + ("FROZEN" if meta.get("locked") else "editable")
        )
        self.diag_peak_checksum.setText(str(meta.get("checksum", "")))
        self.diag_backend.setText(str(result.get("engine", "Not run")))
        files = [
            result.get("request_file"),
            result.get("result_file"),
            str(Path(result.get("work_directory", "")) / "native_indexing_progress.json")
            if result.get("work_directory") else None,
        ]
        self.diag_files.setText("\n".join(str(path) for path in files if path) or "—")
        warnings = []
        if residual:
            warnings.extend(residual.get("warnings", []))
        warnings.extend(result.get("warnings", []))
        if self.candidates_by_uid.get(dataset.uid):
            warnings.append(
                "Candidate cells are not structure solutions and are not yet valid CIF phases."
            )
        self.diag_warnings.setText("\n".join(warnings) or "None")

    def refresh_for_selected_dataset(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            self.populate_peaks([])
            self.populate_candidates([])
            self.plot.set_data(None, [])
            self.master_peak_status.setText("Master list: no dataset selected")
            return
        self.range_min.setValue(float(dataset.x[0]))
        self.range_max.setValue(float(dataset.x[-1]))
        wavelength = dataset.metadata.get("wavelength_angstrom") or dataset.metadata.get("wavelength_k_alpha1")
        if wavelength:
            try:
                self.wavelength.setValue(float(wavelength))
            except (TypeError, ValueError):
                pass
        residual = self.residuals_by_uid.get(dataset.uid)
        peaks = self.peaks_by_uid.get(dataset.uid, [])
        candidates = self.candidates_by_uid.get(dataset.uid, [])
        self.populate_peaks(peaks)
        self.populate_candidates(candidates)
        self.plot.set_data(residual, peaks)
        self._set_master_locked(
            dataset.uid,
            self._meta_for_uid(dataset.uid).get("locked", False),
        )
        self.refresh_diagnostics()

    def duplicate_dataset(self, source_uid: str, target_uid: str):
        for mapping in (
            self.residuals_by_uid,
            self.peaks_by_uid,
            self.peak_list_meta_by_uid,
            self.indexing_by_uid,
            self.candidates_by_uid,
        ):
            if source_uid in mapping:
                mapping[target_uid] = deepcopy(mapping[source_uid])
        if target_uid in self.peak_list_meta_by_uid:
            self.peak_list_meta_by_uid[target_uid]["last_reason"] = "Duplicated from another dataset"
        self._sync_master_to_application(target_uid)

    def remove_dataset(self, dataset_uid: str):
        self.residuals_by_uid.pop(dataset_uid, None)
        self.peaks_by_uid.pop(dataset_uid, None)
        self.peak_list_meta_by_uid.pop(dataset_uid, None)
        self.indexing_by_uid.pop(dataset_uid, None)
        self.candidates_by_uid.pop(dataset_uid, None)

    def apply_theme(self, theme_name: str):
        self.plot.apply_theme(theme_name)

    def get_state(self):
        return {
            "residuals_by_uid": deepcopy(self.residuals_by_uid),
            "peaks_by_uid": deepcopy(self.peaks_by_uid),
            "peak_list_meta_by_uid": deepcopy(self.peak_list_meta_by_uid),
            "indexing_by_uid": deepcopy(self.indexing_by_uid),
            "candidates_by_uid": deepcopy(self.candidates_by_uid),
            "splitter_sizes": self.setup_splitter.sizes(),
            "settings": {
                "source_mode": self.source_mode.currentText(),
                "use_processed": self.use_processed.isChecked(),
                "subtract_background": self.subtract_background.isChecked(),
                "smart_peak_search": self.smart_peak_search.isChecked(),
                "share_master_peaks": self.share_master_peaks.isChecked(),
                "wavelength": self.wavelength.value(),
                "prominence": self.prominence.value(),
                "minimum_snr": self.minimum_snr.value(),
                "minimum_quality": self.minimum_quality.value(),
                "minimum_distance": self.minimum_distance.value(),
                "overlap_exclusion": self.overlap_exclusion.value(),
                "maximum_peaks": self.maximum_peaks.value(),
                "work_directory": self.work_directory.text(),
                "starting_volume": self.starting_volume.value(),
                "timeout": self.timeout.value(),
                "minimum_m20": self.minimum_m20.value(),
                "maximum_x20": self.maximum_x20.value(),
                "zero_shift": self.zero_shift.value(),
                "refine_zero_shift": self.refine_zero_shift.isChecked(),
                "maximum_zero_shift": self.maximum_zero_shift.value(),
                "peak_tolerance": self.peak_tolerance.value(),
                "impurity_tolerance": self.impurity_tolerance.value(),
                "maximum_index": self.maximum_index.value(),
                "global_iterations": self.global_iterations.value(),
                "population_size": self.population_size.value(),
                "search_seeds": self.search_seeds.value(),
                "candidates_per_lattice": self.candidates_per_lattice.value(),
                "random_seed": self.random_seed.value(),
                "bravais": {name: check.isChecked() for name, check in self.bravais_checks.items()},
                "formula_mass": self.formula_mass.value(),
                "z_values": self.z_values.text(),
                "density_min": self.density_min.value(),
                "density_max": self.density_max.value(),
            },
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        for attribute, key in (
            ("residuals_by_uid", "residuals_by_uid"),
            ("peaks_by_uid", "peaks_by_uid"),
            ("peak_list_meta_by_uid", "peak_list_meta_by_uid"),
            ("indexing_by_uid", "indexing_by_uid"),
            ("candidates_by_uid", "candidates_by_uid"),
        ):
            value = state.get(key)
            setattr(self, attribute, deepcopy(value) if isinstance(value, dict) else {})
        settings = state.get("settings", {})
        if isinstance(settings, dict):
            self.source_mode.setCurrentText(settings.get("source_mode", self.source_mode.itemText(0)))
            self.use_processed.setChecked(settings.get("use_processed", True))
            self.subtract_background.setChecked(settings.get("subtract_background", True))
            self.smart_peak_search.setChecked(settings.get("smart_peak_search", True))
            self.share_master_peaks.setChecked(settings.get("share_master_peaks", True))
            for widget, key, default in (
                (self.wavelength, "wavelength", 1.5406),
                (self.prominence, "prominence", 3.0),
                (self.minimum_snr, "minimum_snr", 4.5),
                (self.minimum_quality, "minimum_quality", 35.0),
                (self.minimum_distance, "minimum_distance", 0.12),
                (self.overlap_exclusion, "overlap_exclusion", 0.08),
                (self.maximum_peaks, "maximum_peaks", 40),
                (self.starting_volume, "starting_volume", 200.0),
                (self.timeout, "timeout", 35),
                (self.minimum_m20, "minimum_m20", 2.0),
                (self.maximum_x20, "maximum_x20", 10),
                (self.zero_shift, "zero_shift", 0.0),
                (self.maximum_zero_shift, "maximum_zero_shift", 0.25),
                (self.peak_tolerance, "peak_tolerance", 0.10),
                (self.impurity_tolerance, "impurity_tolerance", 15.0),
                (self.maximum_index, "maximum_index", 10),
                (self.global_iterations, "global_iterations", 32),
                (self.population_size, "population_size", 7),
                (self.search_seeds, "search_seeds", 5),
                (self.candidates_per_lattice, "candidates_per_lattice", 4),
                (self.random_seed, "random_seed", 1600),
                (self.formula_mass, "formula_mass", 0.0),
                (self.density_min, "density_min", 0.5),
                (self.density_max, "density_max", 10.0),
            ):
                widget.setValue(settings.get(key, default))
            self.refine_zero_shift.setChecked(settings.get("refine_zero_shift", True))
            self.work_directory.setText(settings.get("work_directory", ""))
            self.z_values.setText(settings.get("z_values", "1,2,4,8"))
            for name, checked in settings.get("bravais", {}).items():
                if name in self.bravais_checks:
                    self.bravais_checks[name].setChecked(bool(checked))
        sizes = state.get("splitter_sizes")
        if isinstance(sizes, list) and sizes:
            self.setup_splitter.setSizes([int(value) for value in sizes])
        self.refresh_for_selected_dataset()
