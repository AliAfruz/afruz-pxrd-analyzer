from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QThread, QTimer, QRect
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
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QLayout,
)

from .context_copy import install_label_copy_menu, install_table_copy_menu
from .crystal_profiles import PROFILE_MODELS
from .instrument_physics import RADIATION_CONFIGURATIONS
from .peak_transfer import build_peak_list_guide, compare_peak_list_guide
from .whole_pattern_plot import WholePatternPlotWidget
from .refinement_export import export_refinement_txt_bundle
from .refinement_statistics import infer_dataset_statistical_input
from .whole_pattern_refinement import (
    WHOLE_PATTERN_MODES,
    WHOLE_PATTERN_WEIGHTING,
    WholePatternPhaseSpec,
)
from .whole_pattern_worker import WholePatternWorker
from .widgets import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)


class WholePatternRefinementWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.results_by_uid: dict[str, dict] = {}
        self.peak_guides_by_uid: dict[str, dict] = {}
        self._thread: QThread | None = None
        self._worker: WholePatternWorker | None = None
        self._active_dataset_uid: str | None = None
        self._references = []
        self._updating_phase_table = False
        self._geometry_restore_generation = 0
        self._build_ui()

    def _build_ui(self):
        self.setMinimumSize(0, 0)
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        heading = QLabel(
            "Phase 10 — Pawley and Le Bail Whole-Pattern Refinement"
        )
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        description = QLabel(
            "Decompose the complete diffraction profile using reference "
            "reflection positions. Pawley mode solves independent non-negative "
            "reflection intensities; Le Bail-style mode iteratively redistributes "
            "observed intensity. Neither mode refines atomic coordinates."
        )
        description.setWordWrap(True)
        description.setObjectName("mutedLabel")
        install_label_copy_menu(description)
        layout.addWidget(description)

        peak_guide_row = QHBoxLayout()
        self.import_peak_guide_button = QPushButton("Import peaks from Peak List")
        self.clear_peak_guide_button = QPushButton("Clear imported peaks")
        self.range_from_peak_guide = QCheckBox("Limit fit range to imported peak span")
        self.range_from_peak_guide.setChecked(False)
        self.range_from_peak_guide.setToolTip(
            "Unchecked uses the complete measured 2θ range. Check only when you intentionally "
            "want to exclude data below the first imported peak or above the last imported peak."
        )
        self.full_range_button = QPushButton("Use full dataset range")
        self.full_range_button.setToolTip(
            "Restore the refinement range to the first and last measured 2θ values."
        )
        self.peak_guide_status = QLabel("No Peak List snapshot imported.")
        self.peak_guide_status.setObjectName("mutedLabel")
        self.peak_guide_status.setWordWrap(False)
        self.peak_guide_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        install_label_copy_menu(self.peak_guide_status)
        peak_guide_row.addWidget(self.import_peak_guide_button)
        peak_guide_row.addWidget(self.clear_peak_guide_button)
        peak_guide_row.addWidget(self.range_from_peak_guide)
        peak_guide_row.addWidget(self.full_range_button)
        peak_guide_row.addWidget(self.peak_guide_status, 1)
        layout.addLayout(peak_guide_row)

        phase_group = self._build_phase_group()
        settings_group = self._build_settings_group()

        phase_group.setMinimumSize(0, 0)
        phase_group.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self.refinement_controls_scroll = QScrollArea()
        self.refinement_controls_scroll.setObjectName(
            "refinementControlsScroll"
        )
        self.refinement_controls_scroll.setWidgetResizable(True)
        self.refinement_controls_scroll.setFrameShape(QFrame.NoFrame)
        self.refinement_controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self.refinement_controls_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        self.refinement_controls_scroll.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        self.refinement_controls_scroll.setMinimumSize(280, 0)
        self.refinement_controls_scroll.setMaximumWidth(430)
        self.refinement_controls_scroll.setWidget(settings_group)
        settings_group.setMinimumWidth(0)
        settings_group.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Maximum,
        )

        self.controls_splitter = QSplitter(Qt.Horizontal)
        self.controls_splitter.setChildrenCollapsible(False)
        self.controls_splitter.setMinimumSize(0, 205)
        self.controls_splitter.setMaximumHeight(300)
        self.controls_splitter.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self.controls_splitter.addWidget(phase_group)
        self.controls_splitter.addWidget(
            self.refinement_controls_scroll
        )
        self.controls_splitter.setStretchFactor(0, 4)
        self.controls_splitter.setStretchFactor(1, 0)
        self.controls_splitter.setSizes([760, 360])
        layout.addWidget(self.controls_splitter, 0)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run whole-pattern refinement")
        self.run_button.setObjectName("primaryButton")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.mof_preset_button = QPushButton("MOF refinement preset")
        self.mof_preset_button.setToolTip(
            "Le Bail starting controls for MIL-101 and related dense low-angle MOF patterns."
        )
        self.refresh_references_button = QPushButton("Refresh phase library")
        self.export_button = QPushButton("Export result")
        self.export_button.setEnabled(False)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.cancel_button)
        run_row.addWidget(self.mof_preset_button)
        run_row.addWidget(self.refresh_references_button)
        run_row.addStretch(1)
        run_row.addWidget(self.export_button)
        layout.addLayout(run_row)

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("%p%  ·  %v/%m")
        self.progress.setMinimumWidth(180)
        self.progress.setMaximumHeight(24)
        self.progress.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self.progress_label = QLabel("Ready.")
        self.progress_label.setWordWrap(False)
        self.progress_label.setMinimumWidth(0)
        self.progress_label.setMaximumHeight(24)
        self.progress_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Fixed,
        )
        self.progress_label.setObjectName("mutedLabel")
        install_label_copy_menu(self.progress_label)
        progress_row.addWidget(self.progress, 2)
        progress_row.addWidget(self.progress_label, 3)
        layout.addLayout(progress_row)

        self.result_tabs = QTabWidget()
        self.result_tabs.setUsesScrollButtons(True)
        self.result_tabs.setMinimumSize(0, 170)
        self.result_tabs.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self.plot_widget = WholePatternPlotWidget()
        self.plot_widget.peakAddRequested.connect(
            self._manual_peak_from_refinement_plot
        )
        self.result_tabs.addTab(self.plot_widget, "Profile and Difference")
        self.result_tabs.addTab(
            self._table_page(self._build_phase_result_table()),
            "Phase Summary",
        )
        self.result_tabs.addTab(
            self._table_page(self._build_reflection_table()),
            "Extracted Reflections",
        )
        self.result_tabs.addTab(
            self._build_diagnostics_page(),
            "Diagnostics",
        )
        layout.addWidget(self.result_tabs, 1)

        limitation = QLabel(
            "Pattern fractions and extracted reflection intensities are not "
            "weight fractions. Preferred orientation, absorption, fluorescence, "
            "microstructure, imperfect line-shape physics, missing phases, "
            "and incorrect reference indexing can bias the result."
        )
        limitation.setWordWrap(True)
        limitation.setObjectName("mutedLabel")
        layout.addWidget(limitation)

        self.run_button.clicked.connect(self.run_refinement)
        self.cancel_button.clicked.connect(self.cancel_refinement)
        self.refresh_references_button.clicked.connect(
            self.refresh_references
        )
        self.export_button.clicked.connect(self.export_result)
        self.mof_preset_button.clicked.connect(self.apply_mof_refinement_preset)
        self.use_calibration_check.toggled.connect(
            self._load_calibration_initial_values
        )
        self.import_peak_guide_button.clicked.connect(
            self.import_peaks_from_main_list
        )
        self.clear_peak_guide_button.clicked.connect(
            self.clear_imported_peak_guide
        )
        self.full_range_button.clicked.connect(
            self.use_full_dataset_range
        )
        self.range_from_peak_guide.toggled.connect(
            self._peak_guide_range_mode_changed
        )
        self.use_processed_check.toggled.connect(self._update_statistics_status)
        self.statistics_mode.currentTextChanged.connect(self._update_statistics_status)
        self.weighting.currentTextChanged.connect(self._update_statistics_status)
        self.radiation_configuration.currentTextChanged.connect(
            self._radiation_configuration_changed
        )
        self._radiation_configuration_changed(
            self.radiation_configuration.currentText()
        )

    def apply_mof_refinement_preset(self):
        """Use a restrained Le Bail first pass for large-cell porous frameworks."""
        le_bail = next((name for name in WHOLE_PATTERN_MODES if "Le Bail" in name), None)
        if le_bail:
            self.mode.setCurrentText(le_bail)
        self.use_processed_check.setChecked(False)
        if self.range_min.value() < 35.0 < self.range_max.value():
            self.range_max.setValue(35.0)
        self.reference_cutoff.setValue(0.05)
        self.background_order.setValue(3)
        self.refine_zero_shift.setChecked(True)
        self.profile_model.setCurrentText("TCH pseudo-Voigt")
        self.refine_profile.setChecked(True)
        self.refine_eta.setChecked(False)
        self.refine_lorentzian.setChecked(True)
        self.refine_asymmetry.setChecked(False)
        self.use_staged_refinement.setChecked(True)
        self.use_cuda.setChecked(True)
        self.cell_tolerance.setValue(2.0)
        self.extraction_cycles.setValue(20)
        self.maximum_evaluations.setValue(300)
        self.optimization_points.setValue(3000)
        for row in range(self.phase_table.rowCount()):
            item = self.phase_table.item(row, 1)
            if item is not None:
                item.setCheckState(Qt.Checked)
        self.progress_label.setText(
            "MOF preset: Le Bail extraction, low-angle first pass (up to 35° 2θ when available), "
            "weak reflections retained, staged cell/profile fit."
        )

    def _radiation_configuration_changed(self, name: str):
        is_doublet = "doublet" in str(name).lower()
        self.secondary_wavelength.setEnabled(is_doublet)
        self.secondary_ratio.setEnabled(is_doublet)

    def _build_phase_group(self):
        group = QGroupBox("Included reference phases")
        group.setMinimumSize(0, 0)
        layout = QVBoxLayout(group)
        layout.setSizeConstraint(
            QLayout.SizeConstraint.SetNoConstraint
        )
        self.phase_table = QTableWidget(0, 7)
        self.phase_table.setHorizontalHeaderLabels(
            [
                "Include",
                "Refine cell",
                "Phase",
                "Formula",
                "Crystal system",
                "Cell",
                "Source",
            ]
        )
        self.phase_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self._stabilize_table(self.phase_table)
        self.phase_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.phase_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        for column, width in {
            0: 72,
            1: 92,
            3: 110,
            4: 120,
            5: 175,
            6: 150,
        }.items():
            self.phase_table.setColumnWidth(column, width)
        install_table_copy_menu(self.phase_table)
        layout.addWidget(self.phase_table)
        return group

    def _build_settings_group(self):
        group = QGroupBox("Refinement controls")
        group.setMinimumWidth(0)
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow
        )
        form.setFormAlignment(Qt.AlignTop)

        self.mode = NoWheelComboBox()
        self.mode.addItems(list(WHOLE_PATTERN_MODES))
        self.weighting = NoWheelComboBox()
        self.weighting.addItems(list(WHOLE_PATTERN_WEIGHTING))

        self.use_processed_check = QCheckBox("Use processed pattern")
        self.use_processed_check.setChecked(False)
        self.statistics_mode = NoWheelComboBox()
        self.statistics_mode.addItems([
            "Auto from input metadata",
            "Treat as Poisson counts",
            "No valid counting statistics",
        ])
        self.statistics_status = QLabel("Statistics status will appear after a dataset is selected.")
        self.statistics_status.setWordWrap(True)
        self.statistics_status.setObjectName("mutedLabel")

        self.wavelength = NoWheelDoubleSpinBox()
        self.wavelength.setRange(0.1, 10.0)
        self.wavelength.setDecimals(7)
        self.wavelength.setValue(1.5406)
        self.radiation_configuration = NoWheelComboBox()
        self.radiation_configuration.addItems(list(RADIATION_CONFIGURATIONS))
        self.radiation_configuration.setCurrentText("Custom monochromatic")
        self.secondary_wavelength = NoWheelDoubleSpinBox()
        self.secondary_wavelength.setRange(0.1, 10.0)
        self.secondary_wavelength.setDecimals(7)
        self.secondary_wavelength.setValue(1.5444274)
        self.secondary_ratio = NoWheelDoubleSpinBox()
        self.secondary_ratio.setRange(0.0, 5.0)
        self.secondary_ratio.setDecimals(6)
        self.secondary_ratio.setValue(0.5)
        self.apply_position_correction = QCheckBox(
            "Apply Phase 9 displacement/transparency correction"
        )
        self.apply_position_correction.setChecked(True)
        self.overlap_threshold = NoWheelDoubleSpinBox()
        self.overlap_threshold.setRange(0.5, 0.9999)
        self.overlap_threshold.setDecimals(4)
        self.overlap_threshold.setValue(0.90)

        self.range_min = NoWheelDoubleSpinBox()
        self.range_min.setRange(0.0, 179.0)
        self.range_min.setDecimals(4)
        self.range_max = NoWheelDoubleSpinBox()
        self.range_max.setRange(0.1, 179.9)
        self.range_max.setDecimals(4)

        self.reference_cutoff = NoWheelDoubleSpinBox()
        self.reference_cutoff.setRange(0.0, 100.0)
        self.reference_cutoff.setDecimals(3)
        self.reference_cutoff.setValue(0.5)

        self.background_order = NoWheelSpinBox()
        self.background_order.setRange(0, 6)
        self.background_order.setValue(3)

        self.refine_zero_shift = QCheckBox("Refine global zero shift")
        self.refine_zero_shift.setChecked(True)
        self.refine_profile = QCheckBox("Refine U–V–W profile")
        self.refine_profile.setChecked(True)
        self.refine_eta = QCheckBox("Refine pseudo-Voigt η")
        self.refine_eta.setChecked(False)
        self.profile_model = NoWheelComboBox()
        self.profile_model.addItems(list(PROFILE_MODELS))
        self.profile_model.setCurrentText("TCH pseudo-Voigt")
        self.refine_lorentzian = QCheckBox("Refine Lorentzian X–Y")
        self.refine_lorentzian.setChecked(True)
        self.refine_asymmetry = QCheckBox("Refine split-profile asymmetry")
        self.refine_asymmetry.setChecked(False)
        self.use_staged_refinement = QCheckBox("Use staged profile refinement")
        self.use_staged_refinement.setChecked(True)
        self.use_cuda = QCheckBox(
            "Use NVIDIA CUDA acceleration (automatic CPU fallback)"
        )
        self.use_cuda.setChecked(True)
        self.use_cuda.setToolTip(
            "Uses CUDA for large reflection-profile matrices in Pawley and Le Bail, "
            "and for very large Le Bail repartition workloads. Scientifically constrained "
            "SciPy optimizers remain on CPU. Small problems automatically stay on CPU "
            "when that is faster."
        )
        self.use_calibration_check = QCheckBox(
            "Initialize from active Phase 9 profile"
        )
        self.use_calibration_check.setChecked(True)

        self.initial_u = NoWheelDoubleSpinBox()
        self.initial_u.setRange(0.0, 3.0)
        self.initial_u.setDecimals(8)
        self.initial_u.setValue(0.005)
        self.initial_v = NoWheelDoubleSpinBox()
        self.initial_v.setRange(-2.0, 2.0)
        self.initial_v.setDecimals(8)
        self.initial_v.setValue(0.0)
        self.initial_w = NoWheelDoubleSpinBox()
        self.initial_w.setRange(0.00000001, 3.0)
        self.initial_w.setDecimals(8)
        self.initial_w.setValue(0.02)
        self.initial_eta = NoWheelDoubleSpinBox()
        self.initial_eta.setRange(0.0, 1.0)
        self.initial_eta.setDecimals(5)
        self.initial_eta.setValue(0.5)
        self.initial_x = NoWheelDoubleSpinBox()
        self.initial_x.setRange(0.0, 2.0)
        self.initial_x.setDecimals(8)
        self.initial_x.setValue(0.02)
        self.initial_y = NoWheelDoubleSpinBox()
        self.initial_y.setRange(0.0, 2.0)
        self.initial_y.setDecimals(8)
        self.initial_y.setValue(0.001)
        self.initial_asymmetry = NoWheelDoubleSpinBox()
        self.initial_asymmetry.setRange(-0.75, 0.75)
        self.initial_asymmetry.setDecimals(5)
        self.initial_asymmetry.setValue(0.0)

        self.cell_tolerance = NoWheelDoubleSpinBox()
        self.cell_tolerance.setRange(0.1, 15.0)
        self.cell_tolerance.setDecimals(2)
        self.cell_tolerance.setValue(3.0)

        self.extraction_cycles = NoWheelSpinBox()
        self.extraction_cycles.setRange(2, 100)
        self.extraction_cycles.setValue(12)
        self.maximum_evaluations = NoWheelSpinBox()
        self.maximum_evaluations.setRange(10, 1000)
        self.maximum_evaluations.setValue(120)
        self.optimization_points = NoWheelSpinBox()
        self.optimization_points.setRange(300, 6000)
        self.optimization_points.setValue(1800)

        form.addRow("Mode", self.mode)
        form.addRow("Weighting", self.weighting)
        form.addRow(self.use_processed_check)
        form.addRow("Intensity statistics", self.statistics_mode)
        form.addRow("Statistics status", self.statistics_status)
        form.addRow("Radiation", self.radiation_configuration)
        form.addRow("Secondary wavelength", self.secondary_wavelength)
        form.addRow("Secondary / primary ratio", self.secondary_ratio)
        form.addRow(self.apply_position_correction)
        form.addRow("Overlap correlation gate", self.overlap_threshold)
        form.addRow("Wavelength (Å)", self.wavelength)
        form.addRow("2θ minimum", self.range_min)
        form.addRow("2θ maximum", self.range_max)
        form.addRow("Reference cutoff (%)", self.reference_cutoff)
        form.addRow("Background order", self.background_order)
        form.addRow(self.refine_zero_shift)
        form.addRow("Profile model", self.profile_model)
        form.addRow(self.refine_profile)
        form.addRow(self.refine_eta)
        form.addRow(self.refine_lorentzian)
        form.addRow(self.refine_asymmetry)
        form.addRow(self.use_staged_refinement)
        form.addRow(self.use_cuda)
        form.addRow(self.use_calibration_check)
        form.addRow("Initial U", self.initial_u)
        form.addRow("Initial V", self.initial_v)
        form.addRow("Initial W", self.initial_w)
        form.addRow("Initial η", self.initial_eta)
        form.addRow("Initial X", self.initial_x)
        form.addRow("Initial Y", self.initial_y)
        form.addRow("Initial asymmetry", self.initial_asymmetry)
        form.addRow("Cell bounds (±%)", self.cell_tolerance)
        form.addRow("Le Bail cycles", self.extraction_cycles)
        form.addRow("Maximum evaluations", self.maximum_evaluations)
        form.addRow("Optimization points", self.optimization_points)
        return group

    @staticmethod
    def _table_page(table):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(table)
        return page

    def _build_phase_result_table(self):
        self.phase_result_table = QTableWidget(0, 10)
        self.phase_result_table.setHorizontalHeaderLabels(
            [
                "Phase",
                "Formula",
                "Crystal system",
                "Cell refined",
                "a",
                "b",
                "c",
                "α / β / γ",
                "Reflections",
                "Pattern fraction (%)",
            ]
        )
        self._stabilize_table(self.phase_result_table)
        self.phase_result_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.phase_result_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        for column, width in {
            1: 100,
            2: 115,
            3: 90,
            4: 80,
            5: 80,
            6: 80,
            7: 175,
            8: 90,
            9: 125,
        }.items():
            self.phase_result_table.setColumnWidth(column, width)
        install_table_copy_menu(self.phase_result_table)
        return self.phase_result_table

    def _build_reflection_table(self):
        self.reflection_table = QTableWidget(0, 14)
        self.reflection_table.setHorizontalHeaderLabels(
            [
                "Phase",
                "hkl",
                "Initial 2θ",
                "Refined 2θ",
                "FWHM",
                "Extracted intensity",
                "Intensity s.e.",
                "I / s.e.",
                "Overlap group",
                "Identifiability",
                "Intensity fraction (%)",
                "Reference intensity",
                "d-spacing",
                "Index",
            ]
        )
        self._stabilize_table(self.reflection_table)
        self.reflection_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.reflection_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        for column, width in {
            1: 75,
            2: 100,
            3: 100,
            4: 85,
            5: 125,
            6: 100,
            7: 90,
            8: 90,
            9: 180,
            10: 130,
            11: 110,
            12: 95,
            13: 65,
        }.items():
            self.reflection_table.setColumnWidth(column, width)
        install_table_copy_menu(self.reflection_table)
        return self.reflection_table

    def _build_diagnostics_page(self):
        content = QWidget()
        content.setMinimumSize(0, 0)
        form = QFormLayout(content)
        form.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow
        )
        form.setFormAlignment(Qt.AlignTop)
        self.diag_mode = QLabel("—")
        self.diag_rwp = QLabel("—")
        self.diag_rp = QLabel("—")
        self.diag_rexp = QLabel("—")
        self.diag_gof_sqrt = QLabel("—")
        self.diag_gof = QLabel("—")
        self.diag_statistics = QLabel("—")
        self.diag_statistics.setWordWrap(True)
        self.diag_r2 = QLabel("—")
        self.diag_rmse = QLabel("—")
        self.diag_dw = QLabel("—")
        self.diag_zero = QLabel("—")
        self.diag_profile = QLabel("—")
        self.diag_counts = QLabel("—")
        self.diag_metrology = QLabel("—")
        self.diag_metrology.setWordWrap(True)
        self.diag_time = QLabel("—")
        self.diag_acceleration = QLabel("—")
        self.diag_acceleration.setWordWrap(True)
        self.diag_warnings = QLabel("—")
        self.diag_warnings.setWordWrap(True)
        self.diag_warnings.setMinimumWidth(0)
        self.diag_warnings.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        for label in (
            self.diag_mode,
            self.diag_rwp,
            self.diag_rp,
            self.diag_rexp,
            self.diag_gof_sqrt,
            self.diag_gof,
            self.diag_statistics,
            self.diag_r2,
            self.diag_rmse,
            self.diag_dw,
            self.diag_zero,
            self.diag_profile,
            self.diag_counts,
            self.diag_metrology,
            self.diag_time,
            self.diag_acceleration,
            self.diag_warnings,
        ):
            label.setMinimumWidth(0)
            label.setTextInteractionFlags(
                Qt.TextSelectableByMouse
            )
            install_label_copy_menu(label)
        form.addRow("Mode", self.diag_mode)
        form.addRow("Rwp", self.diag_rwp)
        form.addRow("Rp", self.diag_rp)
        form.addRow("Rexp", self.diag_rexp)
        form.addRow("GoF = Rwp/Rexp", self.diag_gof_sqrt)
        form.addRow("Reduced χ²", self.diag_gof)
        form.addRow("Statistical validity", self.diag_statistics)
        form.addRow("R²", self.diag_r2)
        form.addRow("RMSE", self.diag_rmse)
        form.addRow("Durbin–Watson", self.diag_dw)
        form.addRow("Zero shift", self.diag_zero)
        form.addRow("Profile", self.diag_profile)
        form.addRow(
            "Data / reflections / parameters",
            self.diag_counts,
        )
        form.addRow("Reflection identifiability", self.diag_metrology)
        form.addRow("Elapsed time", self.diag_time)
        form.addRow("Compute acceleration", self.diag_acceleration)
        form.addRow("Warnings", self.diag_warnings)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        scroll.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        scroll.setMinimumSize(0, 0)
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _stabilize_table(table: QTableWidget):
        """
        Prevent long table contents from resizing the application window.
        Horizontal scrolling is used instead of changing the top-level size.
        """
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
        table.horizontalHeader().setMinimumSectionSize(48)
        table.horizontalHeader().setStretchLastSection(False)

    @staticmethod
    def _stable_status_text(message: str, maximum: int = 118) -> str:
        compact = " ".join(str(message).split())
        if len(compact) <= maximum:
            return compact
        return compact[: maximum - 1].rstrip() + "…"

    def _capture_window_layout(self) -> dict:
        window = self.main_window
        splitter = getattr(window, "main_splitter", None)
        return {
            "geometry": QRect(window.geometry()),
            "maximized": bool(window.isMaximized()),
            "full_screen": bool(window.isFullScreen()),
            "minimized": bool(window.isMinimized()),
            "splitter_sizes": (
                list(splitter.sizes())
                if splitter is not None
                else None
            ),
        }

    def _restore_window_layout_later(self, snapshot: dict):
        """
        Restore the user's window placement after Qt's immediate and delayed
        layout passes. This prevents Windows from moving the application when
        dynamic result widgets update their size hints.
        """
        self._geometry_restore_generation += 1
        generation = self._geometry_restore_generation

        def restore():
            if generation != self._geometry_restore_generation:
                return

            window = self.main_window
            splitter = getattr(window, "main_splitter", None)
            splitter_sizes = snapshot.get("splitter_sizes")
            if splitter is not None and splitter_sizes:
                splitter.setSizes(splitter_sizes)

            if snapshot.get("full_screen"):
                if not window.isFullScreen():
                    window.showFullScreen()
                return
            if snapshot.get("maximized"):
                if not window.isMaximized():
                    window.showMaximized()
                return
            if snapshot.get("minimized"):
                return

            geometry = snapshot.get("geometry")
            if geometry is not None and geometry.isValid():
                screen = window.screen()
                available = (
                    screen.availableGeometry()
                    if screen is not None
                    else geometry
                )
                width = min(
                    geometry.width(),
                    available.width(),
                )
                height = min(
                    geometry.height(),
                    available.height(),
                )
                x = min(
                    max(geometry.x(), available.left()),
                    available.right() - width + 1,
                )
                y = min(
                    max(geometry.y(), available.top()),
                    available.bottom() - height + 1,
                )
                window.setGeometry(x, y, width, height)

        QTimer.singleShot(0, restore)
        QTimer.singleShot(80, restore)

    def _phase_state(self):
        state = {}
        for row in range(self.phase_table.rowCount()):
            include = self.phase_table.item(row, 0)
            refine = self.phase_table.item(row, 1)
            uid = include.data(Qt.UserRole) if include else None
            if uid:
                state[str(uid)] = {
                    "include": include.checkState() == Qt.Checked,
                    "refine": refine.checkState() == Qt.Checked if refine else False,
                }
        return state

    def refresh_references(self):
        previous = self._phase_state()
        self._references = self.main_window._phase_reference_library()
        best_uids = set()
        dataset = self.main_window.selected_dataset()
        if dataset is not None:
            result = self.main_window.phase_identification_results.get(
                dataset.uid, {}
            )
            best = result.get("best") or {}
            best_uids.update(best.get("reference_uids", []))
            uid = best.get("reference_uid")
            if uid:
                best_uids.add(uid)

        self._updating_phase_table = True
        self.phase_table.setRowCount(len(self._references))
        for row, reference in enumerate(self._references):
            metadata = reference.metadata or {}
            cell = metadata.get("cell")
            crystal_system = metadata.get("crystal_system", "—")
            has_hkl = any(
                str(peak.get("hkl_label", "")).strip()
                for peak in reference.peaks
            )
            cell_refinable = isinstance(cell, dict) and has_hkl
            saved = previous.get(reference.uid, {})
            default_include = (
                saved.get("include")
                if "include" in saved
                else (
                    reference.uid in best_uids
                    or (
                        not best_uids
                        and row == 0
                    )
                )
            )
            default_refine = saved.get(
                "refine",
                cell_refinable,
            )

            include_item = QTableWidgetItem("Use")
            include_item.setFlags(
                include_item.flags() | Qt.ItemIsUserCheckable
            )
            include_item.setCheckState(
                Qt.Checked if default_include else Qt.Unchecked
            )
            include_item.setData(Qt.UserRole, reference.uid)
            self.phase_table.setItem(row, 0, include_item)

            refine_item = QTableWidgetItem(
                "Refine" if cell_refinable else "Unavailable"
            )
            refine_item.setFlags(
                refine_item.flags() | Qt.ItemIsUserCheckable
            )
            refine_item.setCheckState(
                Qt.Checked
                if cell_refinable and default_refine
                else Qt.Unchecked
            )
            if not cell_refinable:
                refine_item.setFlags(
                    refine_item.flags() & ~Qt.ItemIsEnabled
                )
            self.phase_table.setItem(row, 1, refine_item)

            cell_text = "—"
            if isinstance(cell, dict):
                cell_text = (
                    f"{cell.get('a', 0):.5g}, "
                    f"{cell.get('b', 0):.5g}, "
                    f"{cell.get('c', 0):.5g} Å"
                )
            values = [
                reference.name,
                reference.formula or "—",
                crystal_system,
                cell_text,
                reference.source,
            ]
            for column, value in enumerate(values, start=2):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.phase_table.setItem(row, column, item)
        self._updating_phase_table = False

    def _update_peak_guide_status(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            self.peak_guide_status.setText("No dataset selected.")
            return
        guide = self.peak_guides_by_uid.get(dataset.uid)
        if not guide:
            self.peak_guide_status.setText(
                "No Peak List snapshot imported; whole-pattern refinement remains available."
            )
            return
        valid, message = compare_peak_list_guide(
            guide,
            dataset,
            self.main_window.peak_rows.get(dataset.uid, []),
            self.main_window._main_peak_meta(dataset.uid),
        )
        prefix = "Current" if valid else "Outdated"
        self.peak_guide_status.setText(
            f"{prefix}: revision {guide.get('revision', '—')} · "
            f"{guide.get('peak_count', 0)} peaks · {message}"
        )

    def use_full_dataset_range(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None or len(dataset.x) == 0:
            return
        self.range_from_peak_guide.blockSignals(True)
        self.range_from_peak_guide.setChecked(False)
        self.range_from_peak_guide.blockSignals(False)
        self.range_min.setValue(float(dataset.x[0]))
        self.range_max.setValue(float(dataset.x[-1]))
        self._update_peak_guide_status()
        self.main_window.statusBar().showMessage(
            f"Pawley / Le Bail range restored to the complete measured dataset: "
            f"{float(dataset.x[0]):.4f}–{float(dataset.x[-1]):.4f}° 2θ."
        )

    def _peak_guide_range_mode_changed(self, enabled: bool):
        dataset = self.main_window.selected_dataset()
        if dataset is None or len(dataset.x) == 0:
            return
        if not enabled:
            self.range_min.setValue(float(dataset.x[0]))
            self.range_max.setValue(float(dataset.x[-1]))
            self._update_peak_guide_status()
            return
        guide = self.peak_guides_by_uid.get(dataset.uid)
        if not guide:
            self.range_from_peak_guide.blockSignals(True)
            self.range_from_peak_guide.setChecked(False)
            self.range_from_peak_guide.blockSignals(False)
            self.main_window.statusBar().showMessage(
                "Import a Peak List before limiting the fit range to its peak span."
            )
            return
        margin = max(0.25, 3.0 * float(np.median(np.diff(dataset.x))))
        self.range_min.setValue(
            max(float(dataset.x[0]), guide["minimum_two_theta_deg"] - margin)
        )
        self.range_max.setValue(
            min(float(dataset.x[-1]), guide["maximum_two_theta_deg"] + margin)
        )
        self._update_peak_guide_status()

    def import_peaks_from_main_list(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            QMessageBox.information(self, "No dataset", "Select a dataset first.")
            return
        try:
            guide = build_peak_list_guide(
                dataset,
                self.main_window.peak_rows.get(dataset.uid, []),
                self.main_window._main_peak_meta(dataset.uid),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Peak List import unavailable", str(exc))
            return
        self.peak_guides_by_uid[dataset.uid] = guide
        if self.range_from_peak_guide.isChecked():
            self._peak_guide_range_mode_changed(True)
        self._update_peak_guide_status()
        self.main_window.statusBar().showMessage(
            f"Imported Peak List revision {guide['revision']} with {guide['peak_count']} peaks into Pawley / Le Bail."
        )

    def clear_imported_peak_guide(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        self.peak_guides_by_uid.pop(dataset.uid, None)
        self.range_from_peak_guide.blockSignals(True)
        self.range_from_peak_guide.setChecked(False)
        self.range_from_peak_guide.blockSignals(False)
        self.range_min.setValue(float(dataset.x[0]))
        self.range_max.setValue(float(dataset.x[-1]))
        self._update_peak_guide_status()
        self.main_window.statusBar().showMessage(
            "Cleared the Pawley / Le Bail Peak List guide; the full measured 2θ range was restored."
        )

    def _manual_peak_from_refinement_plot(self, position_deg: float, _intensity: float):
        self.main_window._context_manual_peak_requested(
            position_deg,
            _intensity,
            origin="Pawley / Le Bail plot right-click",
        )
        dataset = self.main_window.selected_dataset()
        if dataset is not None:
            self.import_peaks_from_main_list()

    def refresh_for_selected_dataset(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        self.range_from_peak_guide.blockSignals(True)
        self.range_from_peak_guide.setChecked(False)
        self.range_from_peak_guide.blockSignals(False)
        self.range_min.setValue(float(dataset.x[0]))
        self.range_max.setValue(float(dataset.x[-1]))
        metadata_wavelength = dataset.metadata.get("wavelength_angstrom")
        if metadata_wavelength is None:
            metadata_wavelength = dataset.metadata.get(
                "wavelength_k_alpha1"
            )
        if metadata_wavelength:
            try:
                self.wavelength.setValue(float(metadata_wavelength))
            except (TypeError, ValueError):
                pass
        self._load_calibration_initial_values()
        self.refresh_references()
        self.populate_result(
            self.results_by_uid.get(dataset.uid)
        )
        self._update_peak_guide_status()
        self._update_statistics_status()

    def _load_calibration_initial_values(self):
        if not self.use_calibration_check.isChecked():
            return
        profile = getattr(
            self.main_window,
            "active_instrument_profile",
            None,
        )
        if not profile:
            return
        self.profile_model.setCurrentText(
            str(profile.get("profile_model", "Pseudo-Voigt U-V-W"))
        )
        self.initial_u.setValue(float(profile.get("caglioti_u", 0.005)))
        self.initial_v.setValue(float(profile.get("caglioti_v", 0.0)))
        self.initial_w.setValue(float(profile.get("caglioti_w", 0.02)))
        self.initial_eta.setValue(float(profile.get("eta", 0.5)))
        self.initial_x.setValue(float(profile.get("lorentzian_x", 0.0)))
        self.initial_y.setValue(float(profile.get("lorentzian_y", 0.0)))
        self.initial_asymmetry.setValue(float(profile.get("axial_asymmetry", 0.0)))
        self.radiation_configuration.setCurrentText(
            str(profile.get("radiation_configuration", "Custom monochromatic"))
        )
        secondary = profile.get("secondary_wavelength_angstrom")
        if secondary is not None:
            self.secondary_wavelength.setValue(float(secondary))
        self.secondary_ratio.setValue(
            float(profile.get("secondary_to_primary_ratio", 0.5) or 0.0)
        )

    def _phase_specs(self):
        specs = []
        references = {reference.uid: reference for reference in self._references}
        for row in range(self.phase_table.rowCount()):
            include = self.phase_table.item(row, 0)
            refine = self.phase_table.item(row, 1)
            if include is None or include.checkState() != Qt.Checked:
                continue
            uid = str(include.data(Qt.UserRole))
            reference = references.get(uid)
            if reference is None:
                continue
            specs.append(
                WholePatternPhaseSpec(
                    reference=reference,
                    refine_cell=(
                        refine is not None
                        and refine.checkState() == Qt.Checked
                    ),
                    included=True,
                )
            )
        return specs

    def _known_processing_scale(self, dataset) -> float | None:
        for collection_name in ("smoothing_results", "background_results"):
            collection = getattr(self.main_window, collection_name, {})
            record = collection.get(dataset.uid) if isinstance(collection, dict) else None
            if isinstance(record, dict):
                try:
                    factor = float(record.get("normalization_factor"))
                except (TypeError, ValueError):
                    factor = None
                if factor is not None and np.isfinite(factor) and factor > 0:
                    return factor
        return None

    def _selected_input_and_statistics(self, dataset):
        use_processed = self.use_processed_check.isChecked() and dataset.y_processed is not None
        observed = np.asarray(dataset.y_processed if use_processed else dataset.y_raw, dtype=float)
        metadata = dataset.metadata if isinstance(dataset.metadata, dict) else {}
        statistical = infer_dataset_statistical_input(
            dataset.y_raw,
            observed,
            intensity_unit=str(metadata.get("intensity_unit", "")),
            counting_time_s=metadata.get("counting_time_s"),
            interpretation=self.statistics_mode.currentText(),
            known_processing_scale=self._known_processing_scale(dataset),
        )
        statistical["selected_input_label"] = "processed pattern" if use_processed else "raw measured pattern"
        return observed, statistical

    def _update_statistics_status(self, *_args):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            self.statistics_status.setText("Select a dataset to evaluate Rexp validity.")
            return
        try:
            _observed, statistical = self._selected_input_and_statistics(dataset)
        except Exception as exc:
            self.statistics_status.setText(f"Invalid statistics configuration: {exc}")
            return
        valid = bool(statistical.get("statistics_expected_valid")) and self.weighting.currentText() == "Poisson-like"
        if valid:
            status = "VALID: Rexp, GoF and reduced χ² will use absolute Poisson variances."
        elif self.weighting.currentText() != "Poisson-like":
            status = f"FIT-ONLY: {self.weighting.currentText()} weights are empirical; Rexp and χ² are unavailable."
        else:
            status = "FIT-ONLY: Rexp, GoF and reduced χ² are unavailable."
        self.statistics_status.setText(
            f"{status} Input: {statistical.get('selected_input_label')}. {statistical.get('statistics_note', '')}"
        )

    def is_running(self):
        return bool(self._thread is not None and self._thread.isRunning())

    def run_refinement(self):
        if self.is_running():
            return
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            QMessageBox.information(
                self,
                "No dataset",
                "Select an experimental dataset first.",
            )
            return
        metadata = dataset.metadata or {}
        if (
            metadata.get("analysis_role") == "reference_pattern"
            or metadata.get("plot_style") == "sticks"
        ):
            QMessageBox.information(
                self,
                "Reference pattern selected",
                "Select an experimental continuous-scan dataset, not a reference stick pattern.",
            )
            return
        guide = self.peak_guides_by_uid.get(dataset.uid)
        if guide is not None:
            valid, message = compare_peak_list_guide(
                guide,
                dataset,
                self.main_window.peak_rows.get(dataset.uid, []),
                self.main_window._main_peak_meta(dataset.uid),
            )
            if not valid:
                QMessageBox.warning(
                    self,
                    "Imported Peak List is outdated",
                    message,
                )
                return
        phase_specs = self._phase_specs()
        if not phase_specs:
            QMessageBox.information(
                self,
                "No included phases",
                "Include at least one reference phase.",
            )
            return
        if self.range_min.value() >= self.range_max.value():
            QMessageBox.warning(
                self,
                "Invalid range",
                "The minimum 2θ must be lower than the maximum.",
            )
            return

        try:
            y, statistical = self._selected_input_and_statistics(dataset)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid intensity statistics", str(exc))
            return
        settings = {
            "mode": self.mode.currentText(),
            "wavelength_angstrom": self.wavelength.value(),
            "two_theta_min": self.range_min.value(),
            "two_theta_max": self.range_max.value(),
            "intensity_cutoff_percent": self.reference_cutoff.value(),
            "background_order": self.background_order.value(),
            "weighting": self.weighting.currentText(),
            "refine_zero_shift": self.refine_zero_shift.isChecked(),
            "initial_zero_shift": float(
                (
                    getattr(self.main_window, "active_instrument_profile", None)
                    or {}
                ).get("zero_shift_deg", 0.0)
                if self.use_calibration_check.isChecked()
                else 0.0
            ),
            "refine_profile": self.refine_profile.isChecked(),
            "refine_eta": self.refine_eta.isChecked(),
            "profile_model": self.profile_model.currentText(),
            "refine_lorentzian_width": self.refine_lorentzian.isChecked(),
            "refine_asymmetry": self.refine_asymmetry.isChecked(),
            "use_staged_refinement": self.use_staged_refinement.isChecked(),
            "initial_u": self.initial_u.value(),
            "initial_v": self.initial_v.value(),
            "initial_w": self.initial_w.value(),
            "initial_eta": self.initial_eta.value(),
            "initial_x": self.initial_x.value(),
            "initial_y": self.initial_y.value(),
            "initial_asymmetry": self.initial_asymmetry.value(),
            "initial_axial_sh_over_l": float(
                (
                    getattr(self.main_window, "active_instrument_profile", None)
                    or {}
                ).get("axial_sh_over_l", 0.0)
                if self.use_calibration_check.isChecked()
                else 0.0
            ),
            "cell_tolerance_percent": self.cell_tolerance.value(),
            "extraction_cycles": self.extraction_cycles.value(),
            "maximum_nonlinear_evaluations": (
                self.maximum_evaluations.value()
            ),
            "maximum_optimization_points": (
                self.optimization_points.value()
            ),
            "count_reference": (
                None if statistical.get("count_reference") is None
                else np.asarray(statistical["count_reference"], dtype=float).copy()
            ),
            "intensity_scale_factor": float(statistical.get("intensity_scale_factor", 1.0)),
            "intensity_provenance": str(statistical.get("intensity_provenance", "unknown")),
            "statistics_note": str(statistical.get("statistics_note", "")),
            "instrument_profile": deepcopy(
                getattr(self.main_window, "active_instrument_profile", None)
                if self.use_calibration_check.isChecked()
                else None
            ),
            "allow_profile_extrapolation": False,
            "radiation_configuration": self.radiation_configuration.currentText(),
            "secondary_wavelength_angstrom": self.secondary_wavelength.value(),
            "secondary_to_primary_ratio": self.secondary_ratio.value(),
            "apply_instrument_position_correction": bool(
                self.apply_position_correction.isChecked()
                and self.use_calibration_check.isChecked()
            ),
            "overlap_correlation_threshold": self.overlap_threshold.value(),
            "use_gpu": self.use_cuda.isChecked(),
        }

        window_snapshot = self._capture_window_layout()

        thread = QThread(self)
        worker = WholePatternWorker(
            dataset.x.copy(),
            y.copy(),
            phase_specs,
            settings,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(lambda *_: thread.quit())
        worker.failed.connect(lambda *_: thread.quit())
        worker.cancelled.connect(lambda *_: thread.quit())
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        self._active_dataset_uid = dataset.uid
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.progress.setRange(
            0,
            max(1, self.maximum_evaluations.value()),
        )
        self.progress.setValue(0)
        start_message = (
            f"Preparing {self.mode.currentText()} for {dataset.name}."
        )
        self.progress_label.setText(
            self._stable_status_text(start_message)
        )
        self.progress_label.setToolTip(start_message)
        thread.start()
        self._restore_window_layout_later(window_snapshot)

    def cancel_refinement(self):
        if self._worker is not None:
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            message = (
                "Cancellation requested; stopping at a safe numerical checkpoint."
            )
            self.progress_label.setText(
                self._stable_status_text(message)
            )
            self.progress_label.setToolTip(message)

    def _on_progress(self, completed: int, total: int, message: str):
        total = max(1, int(total))
        completed = max(0, min(int(completed), total))
        self.progress.setRange(0, total)
        self.progress.setValue(completed)
        self.progress.setFormat("%p%  ·  %v/%m")
        self.progress_label.setText(
            self._stable_status_text(message)
        )
        self.progress_label.setToolTip(str(message))

    def _on_finished(self, result):
        window_snapshot = self._capture_window_layout()
        if self._active_dataset_uid:
            guide = self.peak_guides_by_uid.get(self._active_dataset_uid)
            if guide:
                result["peak_list_guide"] = deepcopy(guide)
                warnings = list(result.get("warnings", []))
                warnings.append(
                    "A curated Peak List was imported for range selection, diagnostics and provenance. "
                    "The refinement objective still used every measured point inside the selected 2θ range."
                )
                result["warnings"] = warnings
            self.results_by_uid[self._active_dataset_uid] = result
            if hasattr(self.main_window, "_record_scientific_result"):
                self.main_window._record_scientific_result(
                    "refinement",
                    self._active_dataset_uid,
                    {"pawley_lebail": result},
                    reason="Completed Pawley / Le Bail refinement",
                )
        self.populate_result(result)
        self.progress.setValue(self.progress.maximum())
        completion_message = (
            f"Completed in {result.get('elapsed_seconds', 0):.3f} s — "
            f"Rwp {result.get('rwp_percent', 0):.4g}%."
        )
        self.progress_label.setText(
            self._stable_status_text(completion_message)
        )
        self.progress_label.setToolTip(completion_message)
        self.export_button.setEnabled(True)
        self.main_window.statusBar().showMessage(
            "Phase 10 whole-pattern refinement completed."
        )
        self._restore_window_layout_later(window_snapshot)

    def _on_cancelled(self):
        message = (
            "Whole-pattern refinement cancelled; previous result retained."
        )
        self.progress_label.setText(
            self._stable_status_text(message)
        )
        self.progress_label.setToolTip(message)

    def _on_failed(self, traceback_text: str):
        window_snapshot = self._capture_window_layout()
        final_line = traceback_text.strip().splitlines()[-1]
        self.progress_label.setText(
            self._stable_status_text(final_line)
        )
        self.progress_label.setToolTip(final_line)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Whole-pattern refinement failed")
        box.setText(final_line)
        box.setDetailedText(traceback_text)
        box.exec()
        self._restore_window_layout_later(window_snapshot)

    def _thread_finished(self):
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._worker = None
        self._thread = None
        self._active_dataset_uid = None
        dataset = self.main_window.selected_dataset()
        self.export_button.setEnabled(
            bool(
                dataset
                and dataset.uid in self.results_by_uid
            )
        )

    @staticmethod
    def _fmt(value, digits=7):
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}g}"
        except (TypeError, ValueError):
            return str(value)

    def populate_result(self, result: dict | None):
        self.plot_widget.set_result(result)
        phases = [] if not result else result.get("phases", [])
        self.phase_result_table.setRowCount(len(phases))
        for row_index, phase in enumerate(phases):
            cell = phase.get("refined_cell") or {}
            angles = (
                f"{self._fmt(cell.get('alpha'))} / "
                f"{self._fmt(cell.get('beta'))} / "
                f"{self._fmt(cell.get('gamma'))}"
            )
            values = [
                phase.get("phase_name", ""),
                phase.get("formula", ""),
                phase.get("crystal_system", ""),
                "Yes" if phase.get("cell_refined") else "No",
                self._fmt(cell.get("a")),
                self._fmt(cell.get("b")),
                self._fmt(cell.get("c")),
                angles,
                phase.get("reflection_count", 0),
                self._fmt(phase.get("pattern_fraction_percent")),
            ]
            for column, value in enumerate(values):
                self.phase_result_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(str(value)),
                )

        reflections = [] if not result else result.get("reflections", [])
        self.reflection_table.setRowCount(len(reflections))
        for row_index, reflection in enumerate(reflections):
            values = [
                reflection.get("phase_name", ""),
                reflection.get("hkl_label", ""),
                self._fmt(reflection.get("initial_two_theta_deg")),
                self._fmt(reflection.get("two_theta_deg")),
                self._fmt(reflection.get("fwhm_deg")),
                self._fmt(reflection.get("extracted_intensity")),
                self._fmt(reflection.get("intensity_standard_error")),
                self._fmt(reflection.get("intensity_signal_to_uncertainty")),
                reflection.get("overlap_group_id", ""),
                reflection.get("intensity_identifiability", ""),
                self._fmt(
                    reflection.get("intensity_fraction_percent")
                ),
                self._fmt(reflection.get("reference_intensity")),
                self._fmt(reflection.get("d_spacing")),
                reflection.get("reflection_index", ""),
            ]
            for column, value in enumerate(values):
                self.reflection_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(str(value)),
                )

        if not result:
            for label in (
                self.diag_mode,
                self.diag_rwp,
                self.diag_rp,
                self.diag_rexp,
                self.diag_gof_sqrt,
                self.diag_gof,
                self.diag_statistics,
                self.diag_r2,
                self.diag_rmse,
                self.diag_dw,
                self.diag_zero,
                self.diag_profile,
                self.diag_counts,
                self.diag_metrology,
                self.diag_time,
                self.diag_acceleration,
                self.diag_warnings,
            ):
                label.setText("—")
            self.export_button.setEnabled(False)
            return

        profile = result.get("profile", {})
        self.diag_mode.setText(result.get("mode", "—"))
        self.diag_rwp.setText(f"{self._fmt(result.get('rwp_percent'))}%")
        self.diag_rp.setText(f"{self._fmt(result.get('rp_percent'))}%")
        statistics_valid = bool(result.get("statistics_valid"))
        self.diag_rexp.setText(
            f"{self._fmt(result.get('rexp_percent'))}%"
            if statistics_valid and result.get("rexp_percent") is not None
            else "Unavailable"
        )
        self.diag_gof_sqrt.setText(
            self._fmt(result.get("goodness_of_fit_sqrt")) if statistics_valid else "Unavailable"
        )
        self.diag_gof.setText(
            self._fmt(result.get("reduced_chi_square", result.get("goodness_of_fit")))
            if statistics_valid else "Unavailable"
        )
        self.diag_statistics.setText(
            ("Valid" if statistics_valid else "Not statistically valid")
            + f" — {result.get('weighting_model', result.get('weighting', ''))}. "
            + str(result.get("statistics_reason", ""))
        )
        self.diag_r2.setText(self._fmt(result.get("r_squared")))
        self.diag_rmse.setText(self._fmt(result.get("rmse")))
        self.diag_dw.setText(self._fmt(result.get("durbin_watson")))
        self.diag_zero.setText(
            f"{self._fmt(result.get('zero_shift_deg'))}°"
        )
        self.diag_profile.setText(
            f"{profile.get('model', 'Pseudo-Voigt U-V-W')}: "
            f"U={self._fmt(profile.get('caglioti_u'))}, "
            f"V={self._fmt(profile.get('caglioti_v'))}, "
            f"W={self._fmt(profile.get('caglioti_w'))}, "
            f"η={self._fmt(profile.get('eta'))}, "
            f"X={self._fmt(profile.get('lorentzian_x'))}, "
            f"Y={self._fmt(profile.get('lorentzian_y'))}, "
            f"asym={self._fmt(profile.get('axial_asymmetry'))}"
        )
        self.diag_counts.setText(
            f"{result.get('data_point_count', 0)} / "
            f"{result.get('reflection_count', 0)} / "
            f"{result.get('parameter_count', 0)}"
        )
        metrology = result.get("intensity_metrology", {})
        self.diag_metrology.setText(
            f"Independent rank {metrology.get('effective_independent_reflection_count', 0)}; "
            f"individually identifiable {metrology.get('individually_identifiable_count', 0)}; "
            f"unresolved groups {metrology.get('unresolved_group_count', 0)}; "
            f"condition {self._fmt(metrology.get('reflection_information_condition_number'))}"
        )
        self.diag_time.setText(
            f"{self._fmt(result.get('elapsed_seconds'))} s; "
            f"{result.get('nonlinear_evaluations', 0)} evaluations"
        )
        acceleration = result.get("acceleration") or {}
        basis_acceleration = acceleration.get("profile_basis") or {}
        extraction_acceleration = acceleration.get("intensity_extraction") or {}
        self.diag_acceleration.setText(
            f"Overall: {acceleration.get('calculation_backend', 'CPU')}; "
            f"device: {acceleration.get('device_name') or 'CPU'}. "
            f"Profile matrix: {basis_acceleration.get('backend', 'CPU')} "
            f"({'used' if basis_acceleration.get('used') else 'not used'}). "
            f"Intensity extraction: {extraction_acceleration.get('backend', 'CPU')} "
            f"({'used' if extraction_acceleration.get('used') else 'not used'}). "
            f"Nonlinear optimizer: {acceleration.get('nonlinear_optimizer_backend', 'SciPy CPU')}."
        )
        self.diag_warnings.setText(
            "\n".join(result.get("warnings", [])) or "None"
        )
        self.export_button.setEnabled(True)

    def export_result(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        result = self.results_by_uid.get(dataset.uid)
        if not result:
            return
        self.main_window.show_export_center(
            preferred_formats={"excel", "text", "zip"},
            preferred_categories={"project", "pattern", "refinement"},
        )
    def apply_theme(self, theme_name: str):
        self.plot_widget.apply_theme(theme_name)

    def get_state(self):
        return {
            "results_by_uid": deepcopy(self.results_by_uid),
            "peak_guides_by_uid": deepcopy(self.peak_guides_by_uid),
            "settings": {
                "mode": self.mode.currentText(),
                "weighting": self.weighting.currentText(),
                "use_processed": self.use_processed_check.isChecked(),
                "statistics_mode": self.statistics_mode.currentText(),
                "wavelength": self.wavelength.value(),
                "radiation_configuration": self.radiation_configuration.currentText(),
                "secondary_wavelength": self.secondary_wavelength.value(),
                "secondary_ratio": self.secondary_ratio.value(),
                "apply_position_correction": self.apply_position_correction.isChecked(),
                "overlap_threshold": self.overlap_threshold.value(),
                "range_min": self.range_min.value(),
                "range_max": self.range_max.value(),
                "reference_cutoff": self.reference_cutoff.value(),
                "background_order": self.background_order.value(),
                "refine_zero_shift": self.refine_zero_shift.isChecked(),
                "refine_profile": self.refine_profile.isChecked(),
                "refine_eta": self.refine_eta.isChecked(),
                "profile_model": self.profile_model.currentText(),
                "refine_lorentzian": self.refine_lorentzian.isChecked(),
                "refine_asymmetry": self.refine_asymmetry.isChecked(),
                "use_staged_refinement": self.use_staged_refinement.isChecked(),
                "use_cuda": self.use_cuda.isChecked(),
                "use_calibration": self.use_calibration_check.isChecked(),
                "initial_u": self.initial_u.value(),
                "initial_v": self.initial_v.value(),
                "initial_w": self.initial_w.value(),
                "initial_eta": self.initial_eta.value(),
                "initial_x": self.initial_x.value(),
                "initial_y": self.initial_y.value(),
                "initial_asymmetry": self.initial_asymmetry.value(),
                "cell_tolerance": self.cell_tolerance.value(),
                "extraction_cycles": self.extraction_cycles.value(),
                "maximum_evaluations": self.maximum_evaluations.value(),
                "optimization_points": self.optimization_points.value(),
            },
            "phase_state": self._phase_state(),
            "controls_splitter_sizes": (
                self.controls_splitter.sizes()
                if hasattr(self, "controls_splitter")
                else []
            ),
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        results = state.get("results_by_uid")
        self.results_by_uid = (
            deepcopy(results) if isinstance(results, dict) else {}
        )
        guides = state.get("peak_guides_by_uid")
        self.peak_guides_by_uid = (
            deepcopy(guides) if isinstance(guides, dict) else {}
        )
        settings = state.get("settings", {})
        if isinstance(settings, dict):
            self.mode.setCurrentText(
                settings.get("mode", WHOLE_PATTERN_MODES[0])
            )
            self.weighting.setCurrentText(
                settings.get("weighting", WHOLE_PATTERN_WEIGHTING[0])
            )
            self.use_processed_check.setChecked(
                settings.get("use_processed", False)
            )
            self.statistics_mode.setCurrentText(
                settings.get("statistics_mode", "Auto from input metadata")
            )
            for widget, key, default in (
                (self.wavelength, "wavelength", 1.5406),
                (self.secondary_wavelength, "secondary_wavelength", 1.5444274),
                (self.secondary_ratio, "secondary_ratio", 0.5),
                (self.overlap_threshold, "overlap_threshold", 0.90),
                (self.range_min, "range_min", 0.0),
                (self.range_max, "range_max", 179.0),
                (self.reference_cutoff, "reference_cutoff", 0.5),
                (self.background_order, "background_order", 3),
                (self.initial_u, "initial_u", 0.005),
                (self.initial_v, "initial_v", 0.0),
                (self.initial_w, "initial_w", 0.02),
                (self.initial_eta, "initial_eta", 0.5),
                (self.initial_x, "initial_x", 0.02),
                (self.initial_y, "initial_y", 0.001),
                (self.initial_asymmetry, "initial_asymmetry", 0.0),
                (self.cell_tolerance, "cell_tolerance", 3.0),
                (self.extraction_cycles, "extraction_cycles", 12),
                (self.maximum_evaluations, "maximum_evaluations", 120),
                (self.optimization_points, "optimization_points", 1800),
            ):
                widget.setValue(settings.get(key, default))
            self.refine_zero_shift.setChecked(
                settings.get("refine_zero_shift", True)
            )
            self.radiation_configuration.setCurrentText(
                settings.get("radiation_configuration", "Custom monochromatic")
            )
            self.apply_position_correction.setChecked(
                settings.get("apply_position_correction", True)
            )
            self.refine_profile.setChecked(
                settings.get("refine_profile", True)
            )
            self.refine_eta.setChecked(
                settings.get("refine_eta", False)
            )
            self.profile_model.setCurrentText(
                settings.get("profile_model", "TCH pseudo-Voigt")
            )
            self.refine_lorentzian.setChecked(
                settings.get("refine_lorentzian", True)
            )
            self.refine_asymmetry.setChecked(
                settings.get("refine_asymmetry", False)
            )
            self.use_staged_refinement.setChecked(
                settings.get("use_staged_refinement", True)
            )
            self.use_cuda.setChecked(settings.get("use_cuda", True))
            self.use_calibration_check.setChecked(
                settings.get("use_calibration", True)
            )
        self.refresh_references()
        phase_state = state.get("phase_state", {})
        if isinstance(phase_state, dict):
            for row in range(self.phase_table.rowCount()):
                include = self.phase_table.item(row, 0)
                refine = self.phase_table.item(row, 1)
                uid = str(include.data(Qt.UserRole)) if include else ""
                saved = phase_state.get(uid)
                if not saved:
                    continue
                include.setCheckState(
                    Qt.Checked
                    if saved.get("include", False)
                    else Qt.Unchecked
                )
                if refine and refine.flags() & Qt.ItemIsEnabled:
                    refine.setCheckState(
                        Qt.Checked
                        if saved.get("refine", False)
                        else Qt.Unchecked
                    )
        splitter_sizes = state.get(
            "controls_splitter_sizes",
            [],
        )
        if (
            hasattr(self, "controls_splitter")
            and isinstance(splitter_sizes, list)
            and len(splitter_sizes) == 2
            and sum(splitter_sizes) > 0
        ):
            self.controls_splitter.setSizes(
                [int(value) for value in splitter_sizes]
            )
        dataset = self.main_window.selected_dataset()
        if dataset is not None and len(dataset.x) > 0:
            self.range_from_peak_guide.blockSignals(True)
            self.range_from_peak_guide.setChecked(False)
            self.range_from_peak_guide.blockSignals(False)
            self.range_min.setValue(float(dataset.x[0]))
            self.range_max.setValue(float(dataset.x[-1]))
        self.populate_result(
            None if dataset is None else self.results_by_uid.get(dataset.uid)
        )
