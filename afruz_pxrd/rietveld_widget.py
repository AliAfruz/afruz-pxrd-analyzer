from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from PySide6.QtCore import Qt, QThread
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
    QProgressBar,
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
from .peak_transfer import build_peak_list_guide, compare_peak_list_guide
from .crystal_profiles import PROFILE_MODELS
from .crystallography import load_cif
from .structure_validation import (
    ensure_site_formula_metadata,
    structure_refinement_gate,
    validate_crystal_structure,
)
from .rietveld_plot import RietveldPlotWidget
from .refinement_export import export_refinement_txt_bundle
from .refinement_statistics import infer_dataset_statistical_input
from .rietveld_refinement import (
    GENERATED_PATTERN_SCALING,
    RIETVELD_LOSSES,
    RIETVELD_WEIGHTING,
    RietveldError,
    RietveldPhaseSpec,
    validate_phase_specs_for_refinement,
)
from .rietveld_worker import RietveldWorker
from .widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox


class RietveldRefinementWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.structures: list[dict] = []
        self.results_by_uid: dict[str, dict] = {}
        self.peak_guides_by_uid: dict[str, dict] = {}
        self._thread: QThread | None = None
        self._worker: RietveldWorker | None = None
        self._active_dataset_uid: str | None = None
        self._updating_phase_table = False
        self._build_ui()
        self.refresh_active_cif()
        self.refresh_for_selected_dataset()

    def _build_ui(self):
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        heading = QLabel("Phase 11 — Structure-Constrained Rietveld Refinement")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        description = QLabel(
            "A CIF contains crystal structure information, not a measured detector pattern. Afruz first "
            "calculates the powder reflections and jointly scales the generated phase profiles into the "
            "observed intensity units before refinement. Atomic coordinates and occupancies remain fixed "
            "by default; phase scales, unit cells, "
            "profile terms, background, zero shift, optional global Biso, and March–Dollase texture "
            "can be refined under explicit controls."
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

        self.unknown_handoff_notice = QLabel(
            "No provisional Unknown Discovery structure is linked to the selected dataset."
        )
        self.unknown_handoff_notice.setWordWrap(True)
        self.unknown_handoff_notice.setObjectName("mutedLabel")
        install_label_copy_menu(self.unknown_handoff_notice)
        layout.addWidget(self.unknown_handoff_notice)

        self.setup_splitter = QSplitter(Qt.Horizontal)
        self.setup_splitter.setChildrenCollapsible(False)
        self.setup_splitter.setMaximumHeight(315)
        self.setup_splitter.addWidget(self._build_phase_panel())
        self.setup_splitter.addWidget(self._build_control_scroll())
        self.setup_splitter.setSizes([850, 390])
        self.setup_splitter.setStretchFactor(0, 2)
        self.setup_splitter.setStretchFactor(1, 1)
        layout.addWidget(self.setup_splitter)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run Rietveld refinement")
        self.run_button.setObjectName("primaryButton")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.mof_preset_button = QPushButton("MIL-101(Cr) smart preset")
        self.mof_preset_button.setToolTip(
            "Restrained controls validated against the supplied MIL-101(Cr) dataset. "
            "When a Pawley/Le Bail result exists, its profile is transferred and fixed; "
            "a (111) texture candidate is tested with held-out validation."
        )
        self.export_button = QPushButton("Export result")
        self.export_button.setEnabled(False)
        self.crystal_studio_button = QPushButton("3D Crystal Studio…")
        self.crystal_studio_button.setToolTip("Render a saved Rietveld structure or open a CIF reference; export cinematic or publication figures.")
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.cancel_button)
        run_row.addWidget(self.mof_preset_button)
        run_row.addStretch(1)
        run_row.addWidget(self.crystal_studio_button)
        run_row.addWidget(self.export_button)
        layout.addLayout(run_row)

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("%p%  ·  %v/%m")
        self.progress.setMaximumHeight(24)
        self.progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.progress_label = QLabel("Ready.")
        self.progress_label.setWordWrap(False)
        self.progress_label.setMinimumWidth(0)
        self.progress_label.setMaximumHeight(24)
        self.progress_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.progress_label.setObjectName("mutedLabel")
        install_label_copy_menu(self.progress_label)
        progress_row.addWidget(self.progress, 2)
        progress_row.addWidget(self.progress_label, 3)
        layout.addLayout(progress_row)

        self.result_tabs = QTabWidget()
        self.result_tabs.setUsesScrollButtons(True)
        self.result_tabs.setMinimumSize(0, 220)
        self.plot_widget = RietveldPlotWidget()
        self.plot_widget.peakAddRequested.connect(
            self._manual_peak_from_refinement_plot
        )
        self.result_tabs.addTab(self.plot_widget, "Profile and Difference")
        self.result_tabs.addTab(self._table_page(self._build_phase_result_table()), "Phase Summary")
        self.result_tabs.addTab(self._table_page(self._build_atom_table()), "Fixed Atomic Model")
        self.result_tabs.addTab(self._table_page(self._build_reflection_table()), "Calculated Reflections")
        self.result_tabs.addTab(self._diagnostics_scroll(), "Diagnostics")
        layout.addWidget(self.result_tabs, 1)

        limitation = QLabel(
            "This engine uses CIF-supplied Cromer–Mann X-ray form factors and anomalous-dispersion "
            "terms when available; CIFs without those tables retain an explicitly reported "
            "atomic-number approximation. The pseudo-Voigt profile and specimen corrections must "
            "still be matched to the instrument. "
            "It is designed for controlled learning and screening, not as a replacement for established "
            "FullProf, TOPAS, or a validated fundamental-parameters refinement."
        )
        limitation.setWordWrap(True)
        limitation.setObjectName("mutedLabel")
        layout.addWidget(limitation)

        self.add_active_button.clicked.connect(self.add_active_cif)
        self.add_cif_button.clicked.connect(self.add_cif_dialog)
        self.remove_phase_button.clicked.connect(self.remove_selected_phase)
        self.run_button.clicked.connect(self.run_refinement)
        self.cancel_button.clicked.connect(self.cancel_refinement)
        self.export_button.clicked.connect(self.export_result)
        self.crystal_studio_button.clicked.connect(self.open_crystal_studio)
        self.mof_preset_button.clicked.connect(self.apply_mof_refinement_preset)
        self.use_calibration.toggled.connect(self._load_calibration)
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
        self.use_processed.toggled.connect(self._update_statistics_status)
        self.statistics_mode.currentTextChanged.connect(self._update_statistics_status)
        self.weighting.currentTextChanged.connect(self._update_statistics_status)

    def apply_mof_refinement_preset(self):
        """Use a restrained first-pass model for large-cell porous frameworks."""
        self.use_processed.setChecked(False)
        if self.range_min.value() < 35.0 < self.range_max.value():
            self.range_max.setValue(35.0)
        self.cutoff.setValue(0.05)
        self.background_order.setValue(2)
        self.loss.setCurrentText("soft_l1")
        self.profile_model.setCurrentText("TCH pseudo-Voigt")
        self.initial_zero.setValue(0.0)
        self.refine_zero.setChecked(False)
        profile_seeded = self._seed_profile_from_whole_pattern_result()
        self.refine_profile.setChecked(not profile_seeded)
        self.refine_eta.setChecked(False)
        self.refine_lorentzian.setChecked(not profile_seeded)
        self.refine_asymmetry.setChecked(False)
        self.use_staged_refinement.setChecked(True)
        self.cell_tolerance.setValue(1.5)
        self.max_evaluations.setValue(160)
        self.optimization_points.setValue(1500)
        self.validation_stride.setValue(5)
        self.freeze_structure_factors.setChecked(True)
        self.use_cuda.setChecked(True)
        self.allow_flagged_structures.setChecked(False)
        self.kalpha2.setChecked(True)
        self.kalpha2_wavelength.setValue(1.544426)
        self.kalpha2_ratio.setValue(0.5)
        for row in range(self.phase_table.rowCount()):
            for column, checked in ((4, True), (5, False), (7, True)):
                item = self.phase_table.item(row, column)
                if item is not None:
                    item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            axis = self.phase_table.item(row, 6)
            initial_r = self.phase_table.item(row, 8)
            if axis is not None:
                axis.setText("1 1 1")
            if initial_r is not None:
                initial_r.setText("1.0")
        seed_message = (
            "Pawley/Le Bail profile transferred and fixed; "
            if profile_seeded
            else "profile refinement enabled because no whole-pattern seed was available; "
        )
        self.progress_label.setText(
            "MOF preset: raw counts, low-angle first pass (up to 35° 2θ when available), "
            "weak reflections retained, held-out predictive check, "
            + seed_message
            + "large-MOF CIF intensities frozen; coordinates, occupancies, ΔBiso "
            "fixed; (111) March–Dollase texture enabled as a held-out candidate."
        )

    def _seed_profile_from_whole_pattern_result(self) -> bool:
        dataset = self.main_window.selected_dataset()
        whole_widget = getattr(self.main_window, "whole_pattern_widget", None)
        result = (
            getattr(whole_widget, "results_by_uid", {}).get(dataset.uid)
            if dataset is not None and whole_widget is not None
            else None
        )
        profile = result.get("profile") if isinstance(result, dict) else None
        if not isinstance(profile, dict):
            return False
        controls = (
            (self.initial_u, "caglioti_u"),
            (self.initial_v, "caglioti_v"),
            (self.initial_w, "caglioti_w"),
            (self.initial_eta, "eta"),
            (self.initial_x, "lorentzian_x"),
            (self.initial_y, "lorentzian_y"),
        )
        try:
            for control, key in controls:
                control.setValue(float(profile[key]))
            self.initial_zero.setValue(float(result.get("zero_shift_deg", 0.0)))
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def _build_phase_panel(self):
        group = QGroupBox("CIF structure phases")
        layout = QVBoxLayout(group)
        buttons = QHBoxLayout()
        self.add_active_button = QPushButton("Add active CIF")
        self.add_cif_button = QPushButton("Add CIF phase…")
        self.remove_phase_button = QPushButton("Remove selected")
        buttons.addWidget(self.add_active_button)
        buttons.addWidget(self.add_cif_button)
        buttons.addWidget(self.remove_phase_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.phase_table = QTableWidget(0, 10)
        self.phase_table.setHorizontalHeaderLabels(
            [
                "Include", "Phase", "Formula", "Cell", "Refine cell",
                "Refine ΔBiso", "PO axis hkl", "Refine PO", "Initial r", "Structure check",
            ]
        )
        self._stabilize_table(self.phase_table)
        self.phase_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.phase_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.phase_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for column, width in {0: 68, 2: 95, 3: 185, 4: 88, 5: 98, 6: 105, 7: 82, 8: 76, 9: 210}.items():
            self.phase_table.setColumnWidth(column, width)
        install_table_copy_menu(self.phase_table)
        layout.addWidget(self.phase_table)
        return group

    def _build_control_scroll(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(520)
        content = QWidget()
        content.setMinimumSize(0, 0)
        form = QFormLayout(content)
        form.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        self.use_processed = QCheckBox("Use processed pattern")
        self.use_processed.setChecked(False)
        self.statistics_mode = NoWheelComboBox()
        self.statistics_mode.addItems([
            "Auto from input metadata",
            "Treat as Poisson counts",
            "No valid counting statistics",
        ])
        self.statistics_status = QLabel("Statistics status will appear after a dataset is selected.")
        self.statistics_status.setWordWrap(True)
        self.statistics_status.setObjectName("mutedLabel")
        self.generated_scaling = NoWheelComboBox()
        self.generated_scaling.addItems(list(GENERATED_PATTERN_SCALING))
        self.generated_scaling.setCurrentText(GENERATED_PATTERN_SCALING[0])
        self.generated_scaling.setToolTip(
            "Recommended: jointly fit non-negative phase scale factors and background so the CIF-generated "
            "profiles are expressed in the same intensity units as the observed pattern. This never changes "
            "or normalizes the experimental data."
        )
        self.generated_scaling_status = QLabel(
            "CIF profiles are normalized to unit integrated area, then jointly scaled to the observed pattern. "
            "Max-to-100 normalization is not used for Rietveld scale fitting."
        )
        self.generated_scaling_status.setWordWrap(True)
        self.generated_scaling_status.setObjectName("mutedLabel")
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
        self.cutoff = NoWheelDoubleSpinBox()
        self.cutoff.setRange(0.0, 20.0)
        self.cutoff.setDecimals(3)
        self.cutoff.setValue(0.2)
        self.background_order = NoWheelSpinBox()
        self.background_order.setRange(0, 6)
        self.background_order.setValue(3)
        self.weighting = NoWheelComboBox()
        self.weighting.addItems(list(RIETVELD_WEIGHTING))
        self.loss = NoWheelComboBox()
        self.loss.addItems(list(RIETVELD_LOSSES))
        self.loss.setCurrentText("soft_l1")

        self.refine_zero = QCheckBox("Refine global zero shift")
        self.refine_zero.setChecked(True)
        self.initial_zero = NoWheelDoubleSpinBox()
        self.initial_zero.setRange(-2.0, 2.0)
        self.initial_zero.setDecimals(6)
        self.initial_zero.setValue(0.0)
        self.initial_zero.setToolTip(
            "Fixed value when zero refinement is off; starting value when it is on."
        )
        self.refine_profile = QCheckBox("Refine U–V–W")
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
        self.use_calibration = QCheckBox("Initialize from Phase 9 profile")
        self.use_calibration.setChecked(True)

        self.initial_u = NoWheelDoubleSpinBox()
        self.initial_u.setRange(0.0, 3.0)
        self.initial_u.setDecimals(8)
        self.initial_u.setValue(0.005)
        self.initial_v = NoWheelDoubleSpinBox()
        self.initial_v.setRange(-2.0, 2.0)
        self.initial_v.setDecimals(8)
        self.initial_v.setValue(0.0)
        self.initial_w = NoWheelDoubleSpinBox()
        self.initial_w.setRange(1e-8, 3.0)
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
        self.allow_flagged_structures = QCheckBox("Expert override: refine weak/invalid CIF geometry")
        self.allow_flagged_structures.setChecked(False)
        self.allow_flagged_structures.setToolTip(
            "Off by default. When enabled, refinement may use a structure that failed the geometry gate. "
            "The result remains flagged and is not marked publication-ready."
        )
        self.freeze_structure_factors = QCheckBox(
            "Freeze CIF structure factors (recommended for large MOFs)"
        )
        self.freeze_structure_factors.setChecked(False)
        self.freeze_structure_factors.setToolTip(
            "Calculates relative reflection intensities once from the CIF, then "
            "refines cell, zero, scale, background and profile without repeating "
            "the full atomic sum. Required for practical refinement of expanded "
            "MIL-101-size models. Biso cannot be refined in this mode."
        )
        self.use_cuda = QCheckBox(
            "Use NVIDIA CUDA acceleration (automatic CPU fallback)"
        )
        self.use_cuda.setChecked(True)
        self.use_cuda.setToolTip(
            "Uses CuPy on the NVIDIA GPU for large atom-by-reflection structure-factor "
            "batches and CIF powder-pattern synthesis. The SciPy nonlinear optimizer "
            "remains on CPU. If CUDA is unavailable or fails, the run safely falls back "
            "to CPU and records that fact in Diagnostics."
        )

        self.kalpha2 = QCheckBox("Include Kα2 doublet")
        self.kalpha2.setChecked(False)
        self.kalpha2_wavelength = NoWheelDoubleSpinBox()
        self.kalpha2_wavelength.setRange(0.1, 10.0)
        self.kalpha2_wavelength.setDecimals(7)
        self.kalpha2_wavelength.setValue(1.54439)
        self.kalpha2_ratio = NoWheelDoubleSpinBox()
        self.kalpha2_ratio.setRange(0.0, 2.0)
        self.kalpha2_ratio.setDecimals(4)
        self.kalpha2_ratio.setValue(0.5)

        self.max_evaluations = NoWheelSpinBox()
        self.max_evaluations.setRange(10, 1000)
        self.max_evaluations.setValue(120)
        self.optimization_points = NoWheelSpinBox()
        self.optimization_points.setRange(300, 6000)
        self.optimization_points.setValue(1800)
        self.validation_stride = NoWheelSpinBox()
        self.validation_stride.setRange(0, 20)
        self.validation_stride.setSpecialValueText("Off")
        self.validation_stride.setValue(0)
        self.validation_stride.setToolTip(
            "Every nth point is withheld from nonlinear optimization and reported as an "
            "independent predictive check. Use 5 for smart MOF model screening; use Off "
            "for the final all-point production refinement."
        )

        form.addRow(self.use_processed)
        form.addRow("Intensity statistics", self.statistics_mode)
        form.addRow("Statistics status", self.statistics_status)
        form.addRow("Generated CIF scaling", self.generated_scaling)
        form.addRow("Scaling rule", self.generated_scaling_status)
        form.addRow("Wavelength (Å)", self.wavelength)
        form.addRow("2θ minimum", self.range_min)
        form.addRow("2θ maximum", self.range_max)
        form.addRow("Reflection cutoff (%)", self.cutoff)
        form.addRow("Background order", self.background_order)
        form.addRow("Weighting", self.weighting)
        form.addRow("Robust loss", self.loss)
        form.addRow(self.refine_zero)
        form.addRow("Initial/fixed zero shift (°)", self.initial_zero)
        form.addRow("Profile model", self.profile_model)
        form.addRow(self.refine_profile)
        form.addRow(self.refine_eta)
        form.addRow(self.refine_lorentzian)
        form.addRow(self.refine_asymmetry)
        form.addRow(self.use_staged_refinement)
        form.addRow(self.use_calibration)
        form.addRow("Initial U", self.initial_u)
        form.addRow("Initial V", self.initial_v)
        form.addRow("Initial W", self.initial_w)
        form.addRow("Initial η", self.initial_eta)
        form.addRow("Initial X", self.initial_x)
        form.addRow("Initial Y", self.initial_y)
        form.addRow("Initial asymmetry", self.initial_asymmetry)
        form.addRow("Cell bounds (±%)", self.cell_tolerance)
        form.addRow(self.use_cuda)
        form.addRow(self.freeze_structure_factors)
        form.addRow(self.allow_flagged_structures)
        form.addRow(self.kalpha2)
        form.addRow("Kα2 wavelength (Å)", self.kalpha2_wavelength)
        form.addRow("Kα2/Kα1 ratio", self.kalpha2_ratio)
        form.addRow("Maximum evaluations", self.max_evaluations)
        form.addRow("Optimization points", self.optimization_points)
        form.addRow("Held-out validation stride", self.validation_stride)
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _stabilize_table(table: QTableWidget):
        table.setMinimumSize(0, 0)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.horizontalHeader().setMinimumSectionSize(48)

    @staticmethod
    def _table_page(table):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(table)
        return page

    def _build_phase_result_table(self):
        self.phase_result_table = QTableWidget(0, 15)
        self.phase_result_table.setHorizontalHeaderLabels(
            [
                "Phase", "Formula", "Scale", "Pattern fraction (%)", "a", "b", "c",
                "α", "β", "γ", "ΔBiso", "PO hkl", "March r", "Atoms fixed", "X-ray factors",
            ]
        )
        self._stabilize_table(self.phase_result_table)
        self.phase_result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.phase_result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        install_table_copy_menu(self.phase_result_table)
        return self.phase_result_table

    def _build_atom_table(self):
        self.atom_table = QTableWidget(0, 9)
        self.atom_table.setHorizontalHeaderLabels(
            ["Phase", "Label", "Element", "x", "y", "z", "Occupancy", "Biso", "Status"]
        )
        self._stabilize_table(self.atom_table)
        self.atom_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.atom_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        install_table_copy_menu(self.atom_table)
        return self.atom_table

    def _build_reflection_table(self):
        self.reflection_table = QTableWidget(0, 9)
        self.reflection_table.setHorizontalHeaderLabels(
            ["Phase", "hkl", "2θ", "d-spacing", "FWHM", "Structure I", "PO factor", "Phase index", "Reflection"]
        )
        self._stabilize_table(self.reflection_table)
        self.reflection_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.reflection_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        install_table_copy_menu(self.reflection_table)
        return self.reflection_table

    def _diagnostics_scroll(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)
        self.diag_labels = {}
        for key, title in (
            ("rwp", "Rwp"), ("rp", "Rp"),
            ("validation", "Held-out predictive check"), ("rexp", "Rexp"),
            ("gof_sqrt", "GoF = Rwp/Rexp"), ("gof", "Reduced χ²"),
            ("statistics", "Statistical validity"),
            ("normalization", "CIF-to-observed scaling"),
            ("r2", "R²"), ("rmse", "RMSE"), ("dw", "Durbin–Watson"),
            ("zero", "Zero shift"), ("profile", "Profile U, V, W, η"),
            ("correlation", "Parameter correlation"), ("counts", "Data / phases / reflections / parameters"),
            ("time", "Elapsed / evaluations"),
            ("scattering", "CIF / Rietveld scattering kernel"),
            ("acceleration", "Compute acceleration"),
            ("warnings", "Warnings"),
        ):
            label = QLabel("—")
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setWordWrap(key in {"warnings", "normalization", "statistics", "scattering", "acceleration"})
            if key in {"warnings", "normalization", "statistics", "scattering", "acceleration"}:
                label.setMinimumWidth(0)
                label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            install_label_copy_menu(label)
            self.diag_labels[key] = label
            form.addRow(title, label)
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _uid_for_structure(structure: dict) -> str:
        source = structure.get("source_path") or structure.get("raw_cif_text") or structure.get("data_name") or repr(structure.get("cell"))
        return hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]

    def refresh_active_cif(self):
        active = getattr(self.main_window, "reference_structure", None)
        if active and not self.structures:
            self._add_structure(active)

    def add_active_cif(self):
        active = getattr(self.main_window, "reference_structure", None)
        if not active:
            QMessageBox.information(self, "No active CIF", "Import a CIF reference first, or use Add CIF phase.")
            return
        self._add_structure(active)

    def add_cif_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Add Rietveld CIF phase", "", "CIF (*.cif)")
        if not filename:
            return
        try:
            structure = load_cif(filename)
        except Exception as exc:
            QMessageBox.critical(self, "CIF import failed", str(exc))
            return
        self._add_structure(structure)

    def _add_structure(self, structure: dict):
        uid = self._uid_for_structure(structure)
        if any(row.get("_rietveld_uid") == uid for row in self.structures):
            return False
        copied = deepcopy(structure)
        ensure_site_formula_metadata(copied)
        copied["_afruz_structure_plausibility"] = validate_crystal_structure(copied)
        copied["_rietveld_uid"] = uid
        self.structures.append(copied)
        self.refresh_phase_table()
        self._update_unknown_handoff_notice()
        return True

    def add_unknown_discovery_structure(self, record: dict) -> bool:
        """Add or update a validated Unknown Discovery structure in Phase 11.

        The structure remains explicitly provisional. Its source dataset,
        candidate cell, master-reflection checksum and CIF audit are retained
        inside the structure record and therefore in saved Afruz projects.
        """

        if not isinstance(record, dict):
            QMessageBox.critical(
                self,
                "Invalid Unknown Discovery handoff",
                "The Phase 11 handoff record is not valid.",
            )
            return False
        structure = record.get("structure")
        handoff = record.get("handoff")
        if not isinstance(structure, dict) or not isinstance(handoff, dict):
            QMessageBox.critical(
                self,
                "Invalid Unknown Discovery handoff",
                "The handoff does not contain both a structure and provenance record.",
            )
            return False
        if not structure.get("atoms"):
            QMessageBox.critical(
                self,
                "Atomic model required",
                "Phase 11 requires at least one atom with fractional coordinates.",
            )
            return False

        selected = self.main_window.selected_dataset()
        source_uid = str(handoff.get("source_dataset_uid") or "")
        if selected is not None and source_uid and selected.uid != source_uid:
            QMessageBox.warning(
                self,
                "Dataset mismatch",
                "The provisional structure belongs to a different source dataset. "
                "Select that dataset before transferring it to Phase 11.",
            )
            return False

        copied = deepcopy(structure)
        copied["_afruz_origin"] = "Unknown Discovery"
        copied["_afruz_phase11_handoff"] = deepcopy(handoff)
        copied["_afruz_validation_status"] = str(
            handoff.get("classification")
            or copied.get("_afruz_validation_status")
            or "Provisional — review required"
        )
        ensure_site_formula_metadata(copied)
        copied["_afruz_structure_plausibility"] = validate_crystal_structure(copied)
        uid = self._uid_for_structure(copied)
        copied["_rietveld_uid"] = uid

        replaced = False
        for index, existing in enumerate(self.structures):
            if existing.get("_rietveld_uid") == uid:
                self.structures[index] = copied
                replaced = True
                break
        if not replaced:
            self.structures.append(copied)

        self.refresh_phase_table()
        for row in range(self.phase_table.rowCount()):
            item = self.phase_table.item(row, 0)
            if item is not None and str(item.data(Qt.UserRole)) == uid:
                item.setCheckState(Qt.Checked)
                self.phase_table.selectRow(row)
                break
        self._update_unknown_handoff_notice()
        action = "Updated" if replaced else "Added"
        self.main_window.statusBar().showMessage(
            f"{action} provisional Unknown Discovery structure in Phase 11."
        )
        return True

    @staticmethod
    def _unknown_handoff_tooltip(structure: dict) -> str:
        handoff = structure.get("_afruz_phase11_handoff") or {}
        if not isinstance(handoff, dict):
            return ""
        parts = [
            f"Origin: {handoff.get('source', 'Unknown Discovery')}",
            f"Status: {handoff.get('classification', 'Provisional')}",
            f"Source dataset: {handoff.get('source_dataset_name') or handoff.get('source_dataset_uid') or '—'}",
            f"Candidate: #{handoff.get('candidate_rank') or '—'} {handoff.get('candidate_bravais') or ''}".strip(),
            f"Master peaks: revision {handoff.get('master_peak_revision') or '—'}; checksum {handoff.get('master_peak_checksum') or '—'}",
            str(handoff.get("scientific_boundary") or ""),
        ]
        return "\n".join(part for part in parts if part)

    def _update_unknown_handoff_notice(self):
        dataset = self.main_window.selected_dataset()
        uid = None if dataset is None else dataset.uid
        linked = []
        for structure in self.structures:
            handoff = structure.get("_afruz_phase11_handoff") or {}
            if isinstance(handoff, dict) and (
                not uid or str(handoff.get("source_dataset_uid") or "") == uid
            ):
                linked.append(structure)
        if not linked:
            self.unknown_handoff_notice.setText(
                "No provisional Unknown Discovery structure is linked to the selected dataset."
            )
            return
        names = ", ".join(
            str(row.get("data_name") or row.get("formula") or "Unknown phase")
            for row in linked
        )
        self.unknown_handoff_notice.setText(
            f"Provisional Unknown Discovery model linked: {names}. "
            "A successful Rietveld fit validates pattern compatibility only; "
            "it does not independently prove the atomic structure or space group."
        )

    def remove_selected_phase(self):
        rows = sorted({index.row() for index in self.phase_table.selectedIndexes()}, reverse=True)
        for row in rows:
            if 0 <= row < len(self.structures):
                self.structures.pop(row)
        self.refresh_phase_table()
        self._update_unknown_handoff_notice()

    def refresh_phase_table(self):
        state = self._phase_table_state()
        self._updating_phase_table = True
        self.phase_table.setRowCount(len(self.structures))
        for row, structure in enumerate(self.structures):
            uid = structure["_rietveld_uid"]
            saved = state.get(uid, {})
            include = QTableWidgetItem("Use")
            include.setFlags(include.flags() | Qt.ItemIsUserCheckable)
            include.setCheckState(Qt.Checked if saved.get("include", True) else Qt.Unchecked)
            include.setData(Qt.UserRole, uid)
            self.phase_table.setItem(row, 0, include)

            cell = structure.get("cell", {})
            phase_name = structure.get("data_name", f"Phase {row + 1}")
            if structure.get("_afruz_origin") == "Unknown Discovery":
                phase_name = f"{phase_name}  [Provisional]"
            values = [
                phase_name,
                structure.get("formula", "") or "—",
                f"{cell.get('a', 0):.5g}, {cell.get('b', 0):.5g}, {cell.get('c', 0):.5g} Å",
            ]
            tooltip = self._unknown_handoff_tooltip(structure)
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if tooltip:
                    item.setToolTip(tooltip)
                self.phase_table.setItem(row, column, item)

            for column, label, key, default in (
                (4, "Cell", "refine_cell", True),
                (5, "ΔB", "refine_biso", False),
                (7, "PO", "refine_po", False),
            ):
                item = QTableWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if saved.get(key, default) else Qt.Unchecked)
                self.phase_table.setItem(row, column, item)
            self.phase_table.setItem(row, 6, QTableWidgetItem(saved.get("po_axis", "")))
            self.phase_table.setItem(row, 8, QTableWidgetItem(str(saved.get("initial_r", 1.0))))
            validation = structure.get("_afruz_structure_plausibility")
            if not isinstance(validation, dict):
                validation = validate_crystal_structure(structure)
                structure["_afruz_structure_plausibility"] = validation
            gate = structure_refinement_gate(validation)
            check = QTableWidgetItem(f"{gate['status']} · {gate['score']:.0f}/100")
            check.setFlags(check.flags() & ~Qt.ItemIsEditable)
            check.setToolTip("\n".join(gate["reasons"]) or "No plausibility warnings.")
            self.phase_table.setItem(row, 9, check)
        self._updating_phase_table = False

    def _phase_table_state(self):
        state = {}
        for row in range(self.phase_table.rowCount()):
            include = self.phase_table.item(row, 0)
            if include is None:
                continue
            uid = str(include.data(Qt.UserRole))
            state[uid] = {
                "include": include.checkState() == Qt.Checked,
                "refine_cell": self.phase_table.item(row, 4).checkState() == Qt.Checked,
                "refine_biso": self.phase_table.item(row, 5).checkState() == Qt.Checked,
                "po_axis": self.phase_table.item(row, 6).text().strip(),
                "refine_po": self.phase_table.item(row, 7).checkState() == Qt.Checked,
                "initial_r": self.phase_table.item(row, 8).text().strip(),
            }
        return state

    @staticmethod
    def _parse_hkl(text: str):
        if not text.strip():
            return None
        values = re.findall(r"[+-]?\d+", text)
        if len(values) != 3:
            raise ValueError("Preferred-orientation axis must contain exactly three integers, for example 0 0 1.")
        hkl = tuple(int(value) for value in values)
        if hkl == (0, 0, 0):
            raise ValueError("Preferred-orientation hkl cannot be 0 0 0.")
        return hkl

    def included_structures(self):
        state = self._phase_table_state()
        return [
            {key: deepcopy(value) for key, value in structure.items() if key != "_rietveld_uid"}
            for structure in self.structures
            if state.get(structure.get("_rietveld_uid"), {}).get("include", False)
        ]

    def _phase_specs(self):
        state = self._phase_table_state()
        specs = []
        for structure in self.structures:
            uid = structure["_rietveld_uid"]
            row = state.get(uid, {})
            if not row.get("include", False):
                continue
            axis = self._parse_hkl(row.get("po_axis", ""))
            try:
                initial_r = float(row.get("initial_r", 1.0))
            except ValueError as exc:
                raise ValueError("Initial March–Dollase r must be numeric.") from exc
            specs.append(
                RietveldPhaseSpec(
                    structure={key: deepcopy(value) for key, value in structure.items() if key != "_rietveld_uid"},
                    name=structure.get("data_name"),
                    refine_cell=bool(row.get("refine_cell", True)),
                    refine_biso=bool(row.get("refine_biso", False)),
                    preferred_orientation_hkl=axis,
                    refine_preferred_orientation=bool(row.get("refine_po", False)),
                    initial_preferred_orientation_r=initial_r,
                    freeze_structure_factors=self.freeze_structure_factors.isChecked(),
                    allow_flagged_structure=self.allow_flagged_structures.isChecked(),
                )
            )
        return specs

    def _update_peak_guide_status(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            self.peak_guide_status.setText("No dataset selected.")
            return
        guide = self.peak_guides_by_uid.get(dataset.uid)
        if not guide:
            self.peak_guide_status.setText(
                "No Peak List snapshot imported; Rietveld still uses the complete selected range."
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
            f"Rietveld range restored to the complete measured dataset: "
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
            f"Imported Peak List revision {guide['revision']} with {guide['peak_count']} peaks into Rietveld."
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
            "Cleared the Rietveld Peak List guide; the full measured 2θ range was restored."
        )

    def _manual_peak_from_refinement_plot(self, position_deg: float, _intensity: float):
        self.main_window._context_manual_peak_requested(
            position_deg,
            _intensity,
            origin="Rietveld plot right-click",
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
        wavelength = dataset.metadata.get("wavelength_angstrom") or dataset.metadata.get("wavelength_k_alpha1")
        if wavelength:
            try:
                self.wavelength.setValue(float(wavelength))
            except (TypeError, ValueError):
                pass
        self.refresh_active_cif()
        self._load_calibration()
        self._update_unknown_handoff_notice()
        self.populate_result(self.results_by_uid.get(dataset.uid))
        self._update_peak_guide_status()
        self._update_statistics_status()

    def _known_processing_scale(self, dataset) -> float | None:
        for collection_name in ("smoothing_results", "background_results"):
            collection = getattr(self.main_window, collection_name, {})
            record = collection.get(dataset.uid) if isinstance(collection, dict) else None
            if isinstance(record, dict):
                factor = record.get("normalization_factor")
                try:
                    factor = float(factor)
                except (TypeError, ValueError):
                    factor = None
                if factor is not None and np.isfinite(factor) and factor > 0:
                    return factor
        provenance = dataset.metadata.get("processing_provenance")
        if isinstance(provenance, dict):
            try:
                factor = float(provenance.get("normalization_factor"))
            except (TypeError, ValueError):
                factor = None
            if factor is not None and np.isfinite(factor) and factor > 0:
                return factor
        return None

    def _selected_input_and_statistics(self, dataset):
        use_processed = self.use_processed.isChecked() and dataset.y_processed is not None
        observed = np.asarray(
            dataset.y_processed if use_processed else dataset.y_raw,
            dtype=float,
        )
        metadata = dataset.metadata if isinstance(dataset.metadata, dict) else {}
        counting_time = metadata.get("counting_time_s")
        statistical = infer_dataset_statistical_input(
            dataset.y_raw,
            observed,
            intensity_unit=str(metadata.get("intensity_unit", "")),
            counting_time_s=counting_time,
            interpretation=self.statistics_mode.currentText(),
            known_processing_scale=self._known_processing_scale(dataset),
        )
        statistical["selected_input_label"] = (
            "processed pattern" if use_processed else "raw measured pattern"
        )
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
        weighting = self.weighting.currentText()
        valid = bool(statistical.get("statistics_expected_valid")) and weighting == "Poisson-like"
        if valid:
            status = "VALID: Rexp, GoF and reduced χ² will use absolute Poisson variances."
        elif weighting != "Poisson-like":
            status = (
                f"FIT-ONLY: {weighting} weights are empirical. Rexp, GoF and reduced χ² will be unavailable."
            )
        else:
            status = "FIT-ONLY: Rexp, GoF and reduced χ² will be unavailable."
        self.statistics_status.setText(
            f"{status} Input: {statistical.get('selected_input_label')}. "
            f"{statistical.get('statistics_note', '')}"
        )

    def _load_calibration(self):
        if not self.use_calibration.isChecked():
            return
        profile = getattr(self.main_window, "active_instrument_profile", None)
        if profile:
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

    @staticmethod
    def _status(message: str, maximum=120):
        compact = " ".join(str(message).split())
        return compact if len(compact) <= maximum else compact[: maximum - 1] + "…"

    def is_running(self):
        return bool(self._thread is not None and self._thread.isRunning())

    def run_refinement(self):
        if self.is_running():
            return
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            QMessageBox.information(self, "No dataset", "Select an experimental dataset first.")
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
        try:
            phase_specs = self._phase_specs()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid phase controls", str(exc))
            return
        if not phase_specs:
            QMessageBox.information(self, "No CIF phases", "Add and include at least one CIF structure phase.")
            return
        large_structure = any(
            len(spec.structure.get("atoms", [])) > 2000
            for spec in phase_specs
        )
        if not large_structure:
            try:
                validate_phase_specs_for_refinement(phase_specs)
            except RietveldError as exc:
                QMessageBox.warning(self, "CIF geometry blocked", str(exc))
                return
        else:
            self.progress_label.setText(
                "Large CIF queued; periodic geometry validation will run in the background."
            )
        if self.range_min.value() >= self.range_max.value():
            QMessageBox.warning(self, "Invalid range", "The minimum 2θ must be lower than the maximum.")
            return

        try:
            y, statistical = self._selected_input_and_statistics(dataset)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid intensity statistics", str(exc))
            return
        settings = {
            "wavelength_angstrom": self.wavelength.value(),
            "two_theta_min": self.range_min.value(),
            "two_theta_max": self.range_max.value(),
            "intensity_cutoff_percent": self.cutoff.value(),
            "background_order": self.background_order.value(),
            "weighting": self.weighting.currentText(),
            "refine_zero_shift": self.refine_zero.isChecked(),
            "initial_zero_shift": self.initial_zero.value(),
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
                if self.use_calibration.isChecked()
                else 0.0
            ),
            "cell_tolerance_percent": self.cell_tolerance.value(),
            "k_alpha2_enabled": self.kalpha2.isChecked(),
            "k_alpha2_wavelength_angstrom": self.kalpha2_wavelength.value(),
            "k_alpha2_ratio": self.kalpha2_ratio.value(),
            "maximum_nonlinear_evaluations": self.max_evaluations.value(),
            "maximum_optimization_points": self.optimization_points.value(),
            "validation_stride": self.validation_stride.value(),
            "robust_loss": self.loss.currentText(),
            "count_reference": (
                None
                if statistical.get("count_reference") is None
                else np.asarray(statistical["count_reference"], dtype=float).copy()
            ),
            "intensity_scale_factor": float(statistical.get("intensity_scale_factor", 1.0)),
            "intensity_provenance": str(statistical.get("intensity_provenance", "unknown")),
            "statistics_note": str(statistical.get("statistics_note", "")),
            "generated_pattern_scaling": self.generated_scaling.currentText(),
            "instrument_profile": deepcopy(
                getattr(self.main_window, "active_instrument_profile", None)
                if self.use_calibration.isChecked()
                else None
            ),
            "allow_profile_extrapolation": False,
            "use_gpu": self.use_cuda.isChecked(),
        }
        thread = QThread(self)
        worker = RietveldWorker(dataset.x.copy(), y.copy(), phase_specs, settings)
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
        self.progress.setRange(0, max(1, self.max_evaluations.value()))
        self.progress.setValue(0)
        message = f"Preparing structure-constrained refinement for {dataset.name}."
        self.progress_label.setText(self._status(message))
        self.progress_label.setToolTip(message)
        thread.start()

    def cancel_refinement(self):
        if self._worker is not None:
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            message = "Cancellation requested; stopping at a safe numerical checkpoint."
            self.progress_label.setText(self._status(message))
            self.progress_label.setToolTip(message)

    def _on_progress(self, done, total, message):
        total = max(1, int(total))
        self.progress.setRange(0, total)
        self.progress.setValue(max(0, min(int(done), total)))
        self.progress_label.setText(self._status(message))
        self.progress_label.setToolTip(str(message))

    def _on_finished(self, result):
        if self._active_dataset_uid:
            guide = self.peak_guides_by_uid.get(self._active_dataset_uid)
            if guide:
                result["peak_list_guide"] = deepcopy(guide)
                warnings = list(result.get("warnings", []))
                warnings.append(
                    "A curated Peak List was imported for range selection, diagnostics and provenance. "
                    "Rietveld refinement still used every measured point inside the selected 2θ range."
                )
                result["warnings"] = warnings
            self.results_by_uid[self._active_dataset_uid] = result
            if hasattr(self.main_window, "_record_scientific_result"):
                self.main_window._record_scientific_result(
                    "refinement",
                    self._active_dataset_uid,
                    {"rietveld": result},
                    reason="Completed Rietveld refinement",
                )
        self.populate_result(result)
        self.progress.setValue(self.progress.maximum())
        message = f"Completed in {result.get('elapsed_seconds', 0):.3f} s — Rwp {result.get('rwp_percent', 0):.4g}%."
        self.progress_label.setText(self._status(message))
        self.progress_label.setToolTip(message)
        self.export_button.setEnabled(True)
        self.main_window.statusBar().showMessage("Phase 11 Rietveld refinement completed.")
        if hasattr(self.main_window, "validated_qpa_widget"):
            self.main_window.validated_qpa_widget.refresh_from_rietveld()

    def _on_cancelled(self):
        message = "Rietveld refinement cancelled; previous result retained."
        self.progress_label.setText(message)

    def _on_failed(self, traceback_text):
        final = traceback_text.strip().splitlines()[-1]
        self.progress_label.setText(self._status(final))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Rietveld refinement failed")
        box.setText(final)
        box.setDetailedText(traceback_text)
        box.exec()

    def _thread_finished(self):
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._worker = None
        self._thread = None
        self._active_dataset_uid = None
        dataset = self.main_window.selected_dataset()
        self.export_button.setEnabled(bool(dataset and dataset.uid in self.results_by_uid))

    @staticmethod
    def _fmt(value, digits=7):
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}g}"
        except (TypeError, ValueError):
            return str(value)

    def populate_result(self, result):
        self.setUpdatesEnabled(False)
        try:
            self.plot_widget.set_result(result)
            phases = [] if not result else result.get("phases", [])
            self.phase_result_table.setRowCount(len(phases))
            for row_index, phase in enumerate(phases):
                cell = phase.get("refined_cell") or {}
                values = [
                    phase.get("phase_name", ""), phase.get("formula", ""), self._fmt(phase.get("scale_factor")),
                    self._fmt(phase.get("pattern_scale_fraction_percent")), self._fmt(cell.get("a")), self._fmt(cell.get("b")),
                    self._fmt(cell.get("c")), self._fmt(cell.get("alpha")), self._fmt(cell.get("beta")), self._fmt(cell.get("gamma")),
                    self._fmt(phase.get("delta_biso")), phase.get("preferred_orientation_hkl") or "—",
                    self._fmt(phase.get("march_dollase_r")), "Yes" if phase.get("atoms_fixed") else "No",
                    phase.get("xray_scattering_model", "—"),
                ]
                for column, value in enumerate(values):
                    self.phase_result_table.setItem(row_index, column, QTableWidgetItem(str(value)))

            atom_rows = []
            for structure in self.structures:
                for atom in structure.get("atoms", []):
                    atom_rows.append((structure.get("data_name", "Phase"), atom))
            self.atom_table.setRowCount(len(atom_rows))
            for row_index, (phase_name, atom) in enumerate(atom_rows):
                values = [phase_name, atom.get("label", ""), atom.get("element", ""), self._fmt(atom.get("x")), self._fmt(atom.get("y")),
                          self._fmt(atom.get("z")), self._fmt(atom.get("occupancy")), self._fmt(atom.get("b_iso")), "Fixed; optional global ΔBiso only"]
                for column, value in enumerate(values):
                    self.atom_table.setItem(row_index, column, QTableWidgetItem(str(value)))

            reflections = [] if not result else result.get("reflections", [])
            self.reflection_table.setRowCount(len(reflections))
            for row_index, reflection in enumerate(reflections):
                values = [reflection.get("phase_name", ""), reflection.get("hkl_label", ""), self._fmt(reflection.get("two_theta_deg")),
                          self._fmt(reflection.get("d_spacing")), self._fmt(reflection.get("fwhm_deg")), self._fmt(reflection.get("structure_intensity")),
                          self._fmt(reflection.get("preferred_orientation_factor")), reflection.get("phase_index", ""), row_index + 1]
                for column, value in enumerate(values):
                    self.reflection_table.setItem(row_index, column, QTableWidgetItem(str(value)))

            if not result:
                for label in self.diag_labels.values():
                    label.setText("—")
                self.export_button.setEnabled(False)
                return
            profile = result.get("profile", {})
            labels = self.diag_labels
            labels["rwp"].setText(f"{self._fmt(result.get('rwp_percent'))}%")
            labels["rp"].setText(f"{self._fmt(result.get('rp_percent'))}%")
            validation = result.get("cross_validation") or {}
            labels["validation"].setText(
                (
                    f"Every {validation.get('stride')}th point: "
                    f"Rwp={self._fmt(validation.get('rwp_percent'))}%, "
                    f"Rp={self._fmt(validation.get('rp_percent'))}%, "
                    f"RMSE={self._fmt(validation.get('rmse'))}; "
                    f"{validation.get('validation_point_count', 0)} held out"
                )
                if validation.get("enabled")
                else "Off — all selected points used for optimization"
            )
            statistics_valid = bool(result.get("statistics_valid"))
            labels["rexp"].setText(
                f"{self._fmt(result.get('rexp_percent'))}%"
                if statistics_valid and result.get("rexp_percent") is not None
                else "Unavailable"
            )
            labels["gof_sqrt"].setText(
                self._fmt(result.get("goodness_of_fit_sqrt"))
                if statistics_valid
                else "Unavailable"
            )
            labels["gof"].setText(
                self._fmt(result.get("reduced_chi_square", result.get("goodness_of_fit")))
                if statistics_valid
                else "Unavailable"
            )
            labels["statistics"].setText(
                ("Valid" if statistics_valid else "Not statistically valid")
                + f" — {result.get('weighting_model', result.get('weighting', ''))}. "
                + str(result.get("statistics_reason", ""))
            )
            normalization = result.get("generated_pattern_normalization") or {}
            scales = normalization.get("initial_phase_scale_factors") or []
            fractions = normalization.get("initial_pattern_scale_fractions_percent") or []
            scale_text = ", ".join(
                f"P{index + 1}={self._fmt(value, 6)}" for index, value in enumerate(scales)
            ) or "—"
            fraction_text = ", ".join(
                f"P{index + 1}={self._fmt(value, 5)}%" for index, value in enumerate(fractions)
            ) or "—"
            labels["normalization"].setText(
                f"{normalization.get('method', result.get('generated_pattern_scaling', '—'))}; "
                f"observed unchanged; initial scales: {scale_text}; fractions: {fraction_text}; "
                f"initial Rwp={self._fmt(normalization.get('initial_rwp_percent'), 5)}%; "
                f"correlation={self._fmt(normalization.get('initial_profile_correlation'), 5)}"
            )
            labels["r2"].setText(self._fmt(result.get("r_squared")))
            labels["rmse"].setText(self._fmt(result.get("rmse")))
            labels["dw"].setText(self._fmt(result.get("durbin_watson")))
            labels["zero"].setText(f"{self._fmt(result.get('zero_shift_deg'))}°")
            labels["profile"].setText(
                f"{profile.get('model', 'Pseudo-Voigt U-V-W')}: "
                f"U={self._fmt(profile.get('caglioti_u'))}, V={self._fmt(profile.get('caglioti_v'))}, "
                f"W={self._fmt(profile.get('caglioti_w'))}, η={self._fmt(profile.get('eta'))}, "
                f"X={self._fmt(profile.get('lorentzian_x'))}, Y={self._fmt(profile.get('lorentzian_y'))}, "
                f"asym={self._fmt(profile.get('axial_asymmetry'))}"
            )
            labels["correlation"].setText(
                f"condition={self._fmt(result.get('condition_number'))}; max |corr|={self._fmt(result.get('maximum_absolute_correlation'))}"
            )
            labels["counts"].setText(
                f"{result.get('data_point_count', 0)} / {result.get('phase_count', 0)} / "
                f"{result.get('reflection_count', 0)} / {result.get('parameter_count', 0)}"
            )
            scattering_rows = [
                f"{phase.get('phase_name', 'Phase')}: {phase.get('xray_scattering_model', '—')}"
                + (
                    f" [{', '.join(phase.get('xray_scattering_elements') or [])}]"
                    if phase.get("xray_scattering_elements")
                    else ""
                )
                for phase in phases
            ]
            labels["scattering"].setText(
                "; ".join(scattering_rows)
                + f". Calculation wavelength: {self._fmt(result.get('wavelength_angstrom'), 8)} Å."
            )
            labels["time"].setText(f"{self._fmt(result.get('elapsed_seconds'))} s / {result.get('nonlinear_evaluations', 0)}")
            acceleration = result.get("acceleration") or {}
            labels["acceleration"].setText(
                f"Calculations: {acceleration.get('calculation_backend', acceleration.get('backend', 'CPU'))}; "
                f"device: {acceleration.get('device_name') or 'CPU'}; "
                f"optimizer: {acceleration.get('optimizer_backend', 'SciPy CPU')}. "
                f"{acceleration.get('reason', '')}"
            )
            labels["warnings"].setText("\n".join(result.get("warnings", [])) or "None")
            self.export_button.setEnabled(True)
        finally:
            self.setUpdatesEnabled(True)

    def open_crystal_studio(self):
        from .crystal_studio import CrystalStudioDialog

        dataset = self.main_window.selected_dataset()
        result = self.results_by_uid.get(dataset.uid) if dataset is not None else None
        studio = CrystalStudioDialog(result, self)
        cinematic_ui = getattr(self.main_window, "cinematic_ui", None)
        if cinematic_ui is not None:
            cinematic_ui.decorate(studio)
        studio.show()

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
    def apply_theme(self, theme_name):
        self.plot_widget.apply_theme(theme_name)

    def get_state(self):
        return {
            "structures": deepcopy(self.structures),
            "results_by_uid": deepcopy(self.results_by_uid),
            "peak_guides_by_uid": deepcopy(self.peak_guides_by_uid),
            "phase_state": self._phase_table_state(),
            "splitter_sizes": self.setup_splitter.sizes(),
            "settings": {
                "use_processed": self.use_processed.isChecked(), "statistics_mode": self.statistics_mode.currentText(),
                "generated_pattern_scaling": self.generated_scaling.currentText(), "wavelength": self.wavelength.value(),
                "range_min": self.range_min.value(), "range_max": self.range_max.value(), "cutoff": self.cutoff.value(),
                "background_order": self.background_order.value(), "weighting": self.weighting.currentText(), "loss": self.loss.currentText(),
                "refine_zero": self.refine_zero.isChecked(), "initial_zero": self.initial_zero.value(),
                "refine_profile": self.refine_profile.isChecked(), "refine_eta": self.refine_eta.isChecked(),
                "profile_model": self.profile_model.currentText(), "refine_lorentzian": self.refine_lorentzian.isChecked(),
                "refine_asymmetry": self.refine_asymmetry.isChecked(),
                "use_staged_refinement": self.use_staged_refinement.isChecked(),
                "use_cuda": self.use_cuda.isChecked(),
                "freeze_structure_factors": self.freeze_structure_factors.isChecked(),
                "use_calibration": self.use_calibration.isChecked(), "initial_u": self.initial_u.value(), "initial_v": self.initial_v.value(),
                "initial_w": self.initial_w.value(), "initial_eta": self.initial_eta.value(), "initial_x": self.initial_x.value(),
                "initial_y": self.initial_y.value(), "initial_asymmetry": self.initial_asymmetry.value(), "cell_tolerance": self.cell_tolerance.value(),
                "kalpha2": self.kalpha2.isChecked(), "kalpha2_wavelength": self.kalpha2_wavelength.value(), "kalpha2_ratio": self.kalpha2_ratio.value(),
                "max_evaluations": self.max_evaluations.value(), "optimization_points": self.optimization_points.value(),
                "validation_stride": self.validation_stride.value(),
            },
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        structures = state.get("structures")
        self.structures = deepcopy(structures) if isinstance(structures, list) else []
        results = state.get("results_by_uid")
        self.results_by_uid = deepcopy(results) if isinstance(results, dict) else {}
        guides = state.get("peak_guides_by_uid")
        self.peak_guides_by_uid = deepcopy(guides) if isinstance(guides, dict) else {}
        self.refresh_phase_table()
        saved_phase_state = state.get("phase_state", {})
        if isinstance(saved_phase_state, dict):
            for row in range(self.phase_table.rowCount()):
                include = self.phase_table.item(row, 0)
                uid = str(include.data(Qt.UserRole)) if include else ""
                saved = saved_phase_state.get(uid)
                if not saved:
                    continue
                include.setCheckState(Qt.Checked if saved.get("include", True) else Qt.Unchecked)
                for column, key in ((4, "refine_cell"), (5, "refine_biso"), (7, "refine_po")):
                    self.phase_table.item(row, column).setCheckState(Qt.Checked if saved.get(key, False) else Qt.Unchecked)
                self.phase_table.item(row, 6).setText(saved.get("po_axis", ""))
                self.phase_table.item(row, 8).setText(str(saved.get("initial_r", 1.0)))
        settings = state.get("settings", {})
        if isinstance(settings, dict):
            for widget, key, default in (
                (self.wavelength, "wavelength", 1.5406), (self.range_min, "range_min", 0.0), (self.range_max, "range_max", 179.0),
                (self.cutoff, "cutoff", 0.2), (self.background_order, "background_order", 3), (self.initial_u, "initial_u", 0.005),
                (self.initial_v, "initial_v", 0.0), (self.initial_w, "initial_w", 0.02), (self.initial_eta, "initial_eta", 0.5),
                (self.initial_x, "initial_x", 0.02), (self.initial_y, "initial_y", 0.001),
                (self.initial_asymmetry, "initial_asymmetry", 0.0), (self.initial_zero, "initial_zero", 0.0),
                (self.cell_tolerance, "cell_tolerance", 3.0), (self.kalpha2_wavelength, "kalpha2_wavelength", 1.54439),
                (self.kalpha2_ratio, "kalpha2_ratio", 0.5), (self.max_evaluations, "max_evaluations", 120),
                (self.optimization_points, "optimization_points", 1800),
                (self.validation_stride, "validation_stride", 0),
            ):
                widget.setValue(settings.get(key, default))
            self.weighting.setCurrentText(settings.get("weighting", RIETVELD_WEIGHTING[0]))
            self.loss.setCurrentText(settings.get("loss", "soft_l1"))
            self.use_processed.setChecked(settings.get("use_processed", False))
            self.statistics_mode.setCurrentText(settings.get("statistics_mode", "Auto from input metadata"))
            self.generated_scaling.setCurrentText(
                settings.get("generated_pattern_scaling", GENERATED_PATTERN_SCALING[0])
            )
            self.refine_zero.setChecked(settings.get("refine_zero", True))
            self.refine_profile.setChecked(settings.get("refine_profile", True))
            self.refine_eta.setChecked(settings.get("refine_eta", False))
            self.profile_model.setCurrentText(settings.get("profile_model", "TCH pseudo-Voigt"))
            self.refine_lorentzian.setChecked(settings.get("refine_lorentzian", True))
            self.refine_asymmetry.setChecked(settings.get("refine_asymmetry", False))
            self.use_staged_refinement.setChecked(settings.get("use_staged_refinement", True))
            self.use_cuda.setChecked(settings.get("use_cuda", True))
            self.freeze_structure_factors.setChecked(
                settings.get("freeze_structure_factors", False)
            )
            self.use_calibration.setChecked(settings.get("use_calibration", True))
            self.kalpha2.setChecked(settings.get("kalpha2", False))
        splitter_sizes = state.get("splitter_sizes")
        if isinstance(splitter_sizes, list) and len(splitter_sizes) == 2:
            self.setup_splitter.setSizes([int(v) for v in splitter_sizes])
        self.refresh_active_cif()
        self.refresh_for_selected_dataset()
