from __future__ import annotations

from ..main_window_dependencies import *


class WorkspaceShellMixin:
    def _build_center_panel(self):
        panel = QWidget()
        panel.setObjectName("centralScientificWorkspace")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(5)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.ElideRight)
        self.raw_tab = QWidget()
        raw_layout = QVBoxLayout(self.raw_tab)
        self.plot_widget = XRDPlotWidget()
        raw_layout.addWidget(self.plot_widget)
        self.tabs.addTab(self.raw_tab, "Raw Data")

        self.preprocessing_tab = self._build_preprocessing_tab()
        self.tabs.addTab(self.preprocessing_tab, "Background")

        self.smoothing_tab = self._build_smoothing_tab()
        # Historical layout marker: self.tabs.addTab(self.smoothing_tab, "Smart Smoothing")
        self.tabs.addTab(self.smoothing_tab, "Smoothing")

        self.peak_tab = QWidget()
        peak_layout = QVBoxLayout(self.peak_tab)
        peak_layout.setContentsMargins(4, 4, 4, 4)
        peak_layout.setSpacing(5)

        peak_header_row = QHBoxLayout()
        peak_label = QLabel("Peak Analysis Workspace")
        peak_label.setObjectName("sectionTitle")
        peak_header_row.addWidget(peak_label)
        peak_header_row.addStretch(1)

        self.peak_plot_find_button = QPushButton("Find peaks")
        self.peak_plot_smart_button = QPushButton("Smart search")
        self.peak_plot_smart_button.setObjectName("primaryButton")
        self.peak_plot_fit_button = QPushButton("Standard fit")
        self.peak_plot_advanced_fit_button = QPushButton("Advanced fit")
        self.peak_plot_zoom_button = QPushButton("Zoom selected")
        self.peak_plot_auto_range_button = QPushButton("Full range")
        for button in (
            self.peak_plot_find_button,
            self.peak_plot_smart_button,
            self.peak_plot_fit_button,
            self.peak_plot_advanced_fit_button,
            self.peak_plot_zoom_button,
            self.peak_plot_auto_range_button,
        ):
            peak_header_row.addWidget(button)
        peak_layout.addLayout(peak_header_row)

        self.peak_workspace_splitter = QSplitter(Qt.Vertical)
        self.peak_workspace_splitter.setChildrenCollapsible(False)
        self.peak_workspace_splitter.setHandleWidth(7)

        peak_plot_page = QWidget()
        peak_plot_layout = QVBoxLayout(peak_plot_page)
        peak_plot_layout.setContentsMargins(0, 0, 0, 0)
        peak_plot_layout.setSpacing(3)
        self.peak_plot_widget = XRDPlotWidget()
        self.peak_plot_widget.setMinimumHeight(260)
        peak_plot_help = QLabel(
            "The selected dataset, detected/manual peaks, fitted envelope, separated components, "
            "baseline and residuals update live. Enable ‘Click plot to add’ before left-clicking; "
            "right-clicking can add a protected manual peak directly."
        )
        peak_plot_help.setWordWrap(True)
        peak_plot_help.setObjectName("mutedLabel")
        peak_plot_layout.addWidget(self.peak_plot_widget, 1)
        peak_plot_layout.addWidget(peak_plot_help)
        self.peak_workspace_splitter.addWidget(peak_plot_page)

        self.peak_results_tabs = QTabWidget()

        detection_page = QWidget()
        detection_layout = QVBoxLayout(detection_page)

        peak_edit_row = QHBoxLayout()
        self.add_main_peak_button = QPushButton("Add manual peak…")
        self.click_add_main_peak_button = QPushButton("Click plot to add")
        self.click_add_main_peak_button.setCheckable(True)
        self.delete_main_peaks_button = QPushButton("Delete selected")
        self.include_main_peaks_button = QPushButton("Include selected")
        self.exclude_main_peaks_button = QPushButton("Exclude selected")
        # Historical compatibility marker: "Send to Phase Revolution"
        self.send_main_peaks_to_revolution_button = QPushButton(
            "Send to Unknown Phase"
        )
        self.freeze_main_peak_list = QCheckBox("Freeze selected dataset list")
        peak_edit_row.addWidget(self.add_main_peak_button)
        peak_edit_row.addWidget(self.click_add_main_peak_button)
        peak_edit_row.addWidget(self.delete_main_peaks_button)
        peak_edit_row.addWidget(self.include_main_peaks_button)
        peak_edit_row.addWidget(self.exclude_main_peaks_button)
        peak_edit_row.addWidget(self.send_main_peaks_to_revolution_button)
        peak_edit_row.addWidget(self.freeze_main_peak_list)
        peak_edit_row.addStretch(1)
        detection_layout.addLayout(peak_edit_row)

        self.main_peak_status = QLabel(
            "Manual peaks are protected and are used by all following peak-based analyses."
        )
        self.main_peak_status.setWordWrap(True)
        self.main_peak_status.setObjectName("mutedLabel")
        install_label_copy_menu(self.main_peak_status)
        detection_layout.addWidget(self.main_peak_status)

        self.peak_table = QTableWidget(0, 11)
        self.peak_table.setHorizontalHeaderLabels(
            [
                "Use",
                "Dataset",
                "Origin",
                "2θ position",
                "Intensity",
                "FWHM",
                "Prominence",
                "SNR",
                "Confidence",
                "Protected",
                "Notes",
            ]
        )
        self.peak_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.peak_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.peak_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        install_table_copy_menu(self.peak_table)
        detection_layout.addWidget(self.peak_table)
        self.peak_results_tabs.addTab(detection_page, "Detected Peaks")

        fitting_page = QWidget()
        fitting_layout = QVBoxLayout(fitting_page)
        self.fit_table = QTableWidget(0, 21)
        self.fit_table.setHorizontalHeaderLabels(
            [
                "Dataset",
                "Group",
                "Mode",
                "Model",
                "Component",
                "Center",
                "Center σ",
                "Height",
                "FWHM",
                "FWHM σ",
                "Area",
                "Area %",
                "η / shape",
                "SNR",
                "Separation",
                "AICc",
                "BIC",
                "R²",
                "RMSE",
                "Baseline",
                "Flags",
            ]
        )
        self.fit_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        install_table_copy_menu(self.fit_table)
        fitting_layout.addWidget(self.fit_table)
        self.peak_results_tabs.addTab(fitting_page, "Peak Fits")

        selection_page = QWidget()
        selection_layout = QVBoxLayout(selection_page)
        self.fit_candidate_table = QTableWidget(0, 15)
        self.fit_candidate_table.setHorizontalHeaderLabels(
            [
                "Dataset",
                "Group",
                "Candidate",
                "Chosen",
                "Model",
                "Baseline",
                "Components",
                "Parameters",
                "Points",
                "AIC",
                "AICc",
                "BIC",
                "R²",
                "RMSE",
                "Flags",
            ]
        )
        self.fit_candidate_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        install_table_copy_menu(self.fit_candidate_table)
        selection_layout.addWidget(self.fit_candidate_table)
        self.peak_results_tabs.addTab(
            selection_page,
            "Model Selection",
        )

        self.peak_workspace_splitter.addWidget(self.peak_results_tabs)
        self.peak_workspace_splitter.setStretchFactor(0, 3)
        self.peak_workspace_splitter.setStretchFactor(1, 2)
        self.peak_workspace_splitter.setSizes([560, 320])
        peak_layout.addWidget(self.peak_workspace_splitter, 1)
        self.tabs.addTab(self.peak_tab, "Peak Analysis")

        self.size_strain_tab = QWidget()
        size_layout = QVBoxLayout(self.size_strain_tab)

        size_heading = QLabel("Advanced Crystallite Size, Strain and Texture")
        size_heading.setObjectName("sectionTitle")
        size_layout.addWidget(size_heading)

        self.size_summary_label = QLabel(
            "Run the Phase 3 analysis from the right-side controls."
        )
        self.size_summary_label.setWordWrap(True)
        self.size_summary_label.setObjectName("mutedLabel")
        size_layout.addWidget(self.size_summary_label)

        self.size_results_tabs = QTabWidget()

        size_table_page = QWidget()
        size_table_layout = QVBoxLayout(size_table_page)
        self.size_table = QTableWidget(0, 12)
        self.size_table.setHorizontalHeaderLabels(
            [
                "Dataset",
                "Peak",
                "Model",
                "2θ",
                "β observed (°)",
                "β instrument (°)",
                "β corrected (°)",
                "β σ (°)",
                "Scherrer D (nm)",
                "D σ (nm)",
                "Valid",
                "Note",
            ]
        )
        self.size_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        install_table_copy_menu(self.size_table)
        size_table_layout.addWidget(self.size_table)
        self.size_results_tabs.addTab(size_table_page, "Scherrer Table")

        wh_page = QWidget()
        wh_layout = QVBoxLayout(wh_page)
        self.size_strain_plot = SizeStrainPlotWidget()
        wh_layout.addWidget(self.size_strain_plot)
        self.size_results_tabs.addTab(wh_page, "Williamson–Hall")

        summary_page = QWidget()
        summary_layout = QFormLayout(summary_page)
        self.scherrer_mean_value = QLabel("—")
        self.scherrer_median_value = QLabel("—")
        self.wh_size_value = QLabel("—")
        self.wh_strain_value = QLabel("—")
        self.wh_r2_value = QLabel("—")
        self.wh_weighting_value = QLabel("—")

        for copyable_label in (
            self.size_summary_label,
            self.scherrer_mean_value,
            self.scherrer_median_value,
            self.wh_size_value,
            self.wh_strain_value,
            self.wh_r2_value,
            self.wh_weighting_value,
        ):
            install_label_copy_menu(copyable_label)

        summary_layout.addRow("Mean Scherrer size", self.scherrer_mean_value)
        summary_layout.addRow("Median Scherrer size", self.scherrer_median_value)
        summary_layout.addRow("Williamson–Hall size", self.wh_size_value)
        summary_layout.addRow("Microstrain ε", self.wh_strain_value)
        summary_layout.addRow("Regression R²", self.wh_r2_value)
        summary_layout.addRow("Regression weighting", self.wh_weighting_value)
        self.size_results_tabs.addTab(summary_page, "Summary")

        size_layout.addWidget(self.size_results_tabs)
        self.tabs.addTab(self.size_strain_tab, "Size & Strain")

        self.crystal_tab = QWidget()
        crystal_layout = QVBoxLayout(self.crystal_tab)

        crystal_heading = QLabel("CIF Reference and Unit-Cell Refinement")
        crystal_heading.setObjectName("sectionTitle")
        crystal_layout.addWidget(crystal_heading)

        self.crystal_summary_label = QLabel(
            "Import a CIF reference from the Phase 4 controls."
        )
        self.crystal_summary_label.setWordWrap(True)
        self.crystal_summary_label.setObjectName("mutedLabel")
        install_label_copy_menu(self.crystal_summary_label)
        crystal_layout.addWidget(self.crystal_summary_label)

        self.crystal_results_tabs = QTabWidget()

        cif_summary_page = QWidget()
        cif_summary_layout = QFormLayout(cif_summary_page)
        self.cif_name_value = QLabel("—")
        self.cif_formula_value = QLabel("—")
        self.cif_space_group_value = QLabel("—")
        self.cif_system_value = QLabel("—")
        self.cif_cell_value = QLabel("—")
        self.cif_atoms_value = QLabel("—")
        for copyable_label in (
            self.cif_name_value,
            self.cif_formula_value,
            self.cif_space_group_value,
            self.cif_system_value,
            self.cif_cell_value,
            self.cif_atoms_value,
        ):
            install_label_copy_menu(copyable_label)
        cif_summary_layout.addRow("Structure", self.cif_name_value)
        cif_summary_layout.addRow("Formula", self.cif_formula_value)
        cif_summary_layout.addRow("Space group", self.cif_space_group_value)
        cif_summary_layout.addRow("Crystal system", self.cif_system_value)
        cif_summary_layout.addRow("Unit cell", self.cif_cell_value)
        cif_summary_layout.addRow("Expanded atoms", self.cif_atoms_value)
        self.crystal_results_tabs.addTab(cif_summary_page, "CIF Summary")

        reference_page = QWidget()
        reference_layout = QVBoxLayout(reference_page)
        self.reference_table = QTableWidget(0, 6)
        self.reference_table.setHorizontalHeaderLabels(
            ["2θ", "d (Å)", "hkl", "Relative I", "Multiplicity", "Equivalent hkls"]
        )
        self.reference_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        install_table_copy_menu(self.reference_table)
        reference_layout.addWidget(self.reference_table)
        self.crystal_results_tabs.addTab(reference_page, "Reference Peaks")

        matches_page = QWidget()
        matches_layout = QVBoxLayout(matches_page)
        self.cell_match_table = QTableWidget(0, 10)
        self.cell_match_table.setHorizontalHeaderLabels(
            [
                "Observed 2θ",
                "Observed σ",
                "hkl",
                "Initial calc.",
                "Initial Δ",
                "Refined calc.",
                "Refined Δ",
                "Reference I",
                "Dataset",
                "Status",
            ]
        )
        self.cell_match_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        install_table_copy_menu(self.cell_match_table)
        matches_layout.addWidget(self.cell_match_table)
        self.crystal_results_tabs.addTab(matches_page, "Matched Peaks")

        residual_page = QWidget()
        residual_layout = QVBoxLayout(residual_page)
        self.cell_refinement_plot = CellRefinementPlotWidget()
        residual_layout.addWidget(self.cell_refinement_plot)
        self.crystal_results_tabs.addTab(residual_page, "Residuals")

        refined_page = QWidget()
        refined_layout = QFormLayout(refined_page)
        self.refined_cell_value = QLabel("—")
        self.cell_zero_shift_value = QLabel("—")
        self.cell_rmse_value = QLabel("—")
        self.cell_r2_value = QLabel("—")
        self.cell_match_score_value = QLabel("—")
        self.cell_weighting_value = QLabel("—")
        for copyable_label in (
            self.refined_cell_value,
            self.cell_zero_shift_value,
            self.cell_rmse_value,
            self.cell_r2_value,
            self.cell_match_score_value,
            self.cell_weighting_value,
        ):
            install_label_copy_menu(copyable_label)
        refined_layout.addRow("Refined cell", self.refined_cell_value)
        refined_layout.addRow("Zero shift", self.cell_zero_shift_value)
        refined_layout.addRow("2θ RMSE", self.cell_rmse_value)
        refined_layout.addRow("Regression R²", self.cell_r2_value)
        refined_layout.addRow("Match score", self.cell_match_score_value)
        refined_layout.addRow("Weighting", self.cell_weighting_value)
        self.crystal_results_tabs.addTab(refined_page, "Refinement Summary")

        crystal_layout.addWidget(self.crystal_results_tabs)
        self.tabs.addTab(self.crystal_tab, "CIF & Cell")

        self.phase_tab = QWidget()
        phase_layout = QVBoxLayout(self.phase_tab)

        phase_heading = QLabel("Local-Library Phase Identification")
        phase_heading.setObjectName("sectionTitle")
        phase_layout.addWidget(phase_heading)

        self.phase_summary_label = QLabel(
            "Import reference-card datasets or calculate a CIF reference, then run Phase 5 search."
        )
        self.phase_summary_label.setWordWrap(True)
        self.phase_summary_label.setObjectName("mutedLabel")
        install_label_copy_menu(self.phase_summary_label)
        phase_layout.addWidget(self.phase_summary_label)

        self.phase_results_tabs = QTabWidget()

        candidates_page = QWidget()
        candidates_layout = QVBoxLayout(candidates_page)
        self.phase_candidate_table = QTableWidget(0, 12)
        self.phase_candidate_table.setHorizontalHeaderLabels(
            [
                "Rank", "Candidate phase(s)", "Formula", "Score", "Matched",
                "Observed coverage", "Reference coverage", "Mean |Δ2θ|",
                "Zero shift", "Intensity corr.", "Type", "Notes",
            ]
        )
        self.phase_candidate_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.phase_candidate_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.phase_candidate_table.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        install_table_copy_menu(self.phase_candidate_table)
        candidates_layout.addWidget(self.phase_candidate_table)
        self.phase_results_tabs.addTab(candidates_page, "Candidate Ranking")

        matches_page = QWidget()
        matches_layout = QVBoxLayout(matches_page)
        self.phase_match_table = QTableWidget(0, 9)
        self.phase_match_table.setHorizontalHeaderLabels(
            [
                "Observed 2θ", "Observed I", "Reference", "Reference 2θ",
                "Shifted reference 2θ", "hkl", "Reference I", "Δ2θ", "Status",
            ]
        )
        self.phase_match_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        install_table_copy_menu(self.phase_match_table)
        matches_layout.addWidget(self.phase_match_table)
        self.phase_results_tabs.addTab(matches_page, "Matched Peaks")

        library_page = QWidget()
        library_layout = QVBoxLayout(library_page)
        self.phase_library_table = QTableWidget(0, 6)
        self.phase_library_table.setHorizontalHeaderLabels(
            ["Reference", "Formula", "Source", "Peaks", "Wavelength (Å)", "Role"]
        )
        self.phase_library_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        install_table_copy_menu(self.phase_library_table)
        library_layout.addWidget(self.phase_library_table)
        self.phase_results_tabs.addTab(library_page, "Reference Library")

        plot_page = QWidget()
        plot_layout = QVBoxLayout(plot_page)
        self.phase_identification_plot = PhaseIdentificationPlotWidget()
        plot_layout.addWidget(self.phase_identification_plot)
        self.phase_results_tabs.addTab(plot_page, "Peak Match Plot")

        phase_layout.addWidget(self.phase_results_tabs)
        self.tabs.addTab(self.phase_tab, "Phase Identification")

        self.qpa_tab = QWidget()
        qpa_layout = QVBoxLayout(self.qpa_tab)

        qpa_heading = QLabel("Semi-Quantitative Phase Analysis")
        qpa_heading.setObjectName("sectionTitle")
        qpa_layout.addWidget(qpa_heading)

        self.qpa_summary_label = QLabel(
            "Select a Phase 5 candidate, review its RIR values, then run Phase 6 quantification."
        )
        self.qpa_summary_label.setWordWrap(True)
        self.qpa_summary_label.setObjectName("mutedLabel")
        install_label_copy_menu(self.qpa_summary_label)
        qpa_layout.addWidget(self.qpa_summary_label)

        self.qpa_results_tabs = QTabWidget()

        qpa_setup_page = QWidget()
        qpa_setup_layout = QVBoxLayout(qpa_setup_page)
        setup_note = QLabel(
            "Rows come from the selected Phase 5 candidate. RIR and zero shift are editable before calculation."
        )
        setup_note.setWordWrap(True)
        setup_note.setObjectName("mutedLabel")
        qpa_setup_layout.addWidget(setup_note)
        self.qpa_setup_table = QTableWidget(0, 5)
        self.qpa_setup_table.setHorizontalHeaderLabels(
            ["Include", "Phase", "Formula", "RIR", "Zero shift (°)"]
        )
        self.qpa_setup_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        install_table_copy_menu(self.qpa_setup_table)
        qpa_setup_layout.addWidget(self.qpa_setup_table)
        self.qpa_results_tabs.addTab(qpa_setup_page, "Phase Setup")

        qpa_fraction_page = QWidget()
        qpa_fraction_layout = QVBoxLayout(qpa_fraction_page)
        self.qpa_fraction_table = QTableWidget(0, 11)
        self.qpa_fraction_table.setHorizontalHeaderLabels(
            [
                "Phase", "Formula", "Scale", "RIR", "Scale %",
                "Reported fraction %", "σ %", "Corrected %", "Corrected σ %",
                "Reference peaks", "Status",
            ]
        )
        self.qpa_fraction_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        install_table_copy_menu(self.qpa_fraction_table)
        qpa_fraction_layout.addWidget(self.qpa_fraction_table)
        self.qpa_results_tabs.addTab(qpa_fraction_page, "Phase Fractions")

        qpa_plot_page = QWidget()
        qpa_plot_layout = QVBoxLayout(qpa_plot_page)
        self.qpa_plot = QuantitativePhasePlotWidget()
        qpa_plot_layout.addWidget(self.qpa_plot)
        self.qpa_results_tabs.addTab(qpa_plot_page, "Pattern Fit")

        qpa_diagnostics_page = QWidget()
        qpa_diagnostics_layout = QFormLayout(qpa_diagnostics_page)
        self.qpa_r2_value = QLabel("—")
        self.qpa_rmse_value = QLabel("—")
        self.qpa_profile_residual_value = QLabel("—")
        self.qpa_weighted_residual_value = QLabel("—")
        self.qpa_correlation_value = QLabel("—")
        self.qpa_bootstrap_value = QLabel("—")
        self.qpa_amorphous_value = QLabel("—")
        self.qpa_warning_value = QLabel("—")
        self.qpa_warning_value.setWordWrap(True)
        for copyable_label in (
            self.qpa_r2_value,
            self.qpa_rmse_value,
            self.qpa_profile_residual_value,
            self.qpa_weighted_residual_value,
            self.qpa_correlation_value,
            self.qpa_bootstrap_value,
            self.qpa_amorphous_value,
            self.qpa_warning_value,
        ):
            install_label_copy_menu(copyable_label)
        qpa_diagnostics_layout.addRow("R²", self.qpa_r2_value)
        qpa_diagnostics_layout.addRow("RMSE", self.qpa_rmse_value)
        qpa_diagnostics_layout.addRow("Profile residual", self.qpa_profile_residual_value)
        qpa_diagnostics_layout.addRow("Weighted residual", self.qpa_weighted_residual_value)
        qpa_diagnostics_layout.addRow("Maximum basis correlation", self.qpa_correlation_value)
        qpa_diagnostics_layout.addRow("Bootstrap", self.qpa_bootstrap_value)
        qpa_diagnostics_layout.addRow("Amorphous estimate", self.qpa_amorphous_value)
        qpa_diagnostics_layout.addRow("Warnings", self.qpa_warning_value)
        self.qpa_results_tabs.addTab(qpa_diagnostics_page, "Diagnostics")

        qpa_layout.addWidget(self.qpa_results_tabs)
        self.tabs.addTab(self.qpa_tab, "Quantitative Phase Analysis")

        self.stress_tab = QWidget()
        stress_layout = QVBoxLayout(self.stress_tab)

        stress_heading = QLabel("Residual Stress — sin²ψ Method")
        stress_heading.setObjectName("sectionTitle")
        stress_layout.addWidget(stress_heading)

        self.stress_summary_label = QLabel(
            "Build observations from multiple ψ-tilt scans, verify every angle and peak center, then calculate stress."
        )
        self.stress_summary_label.setWordWrap(True)
        self.stress_summary_label.setObjectName("mutedLabel")
        install_label_copy_menu(self.stress_summary_label)
        stress_layout.addWidget(self.stress_summary_label)

        self.stress_results_tabs = QTabWidget()

        stress_setup_page = QWidget()
        stress_setup_layout = QVBoxLayout(stress_setup_page)
        stress_setup_note = QLabel(
            "Include, ψ, peak 2θ, and peak uncertainty are editable. ψ inferred from a filename or metadata must still be verified."
        )
        stress_setup_note.setWordWrap(True)
        stress_setup_note.setObjectName("mutedLabel")
        stress_setup_layout.addWidget(stress_setup_note)
        self.stress_observation_table = QTableWidget(0, 11)
        self.stress_observation_table.setHorizontalHeaderLabels(
            [
                "Include", "Dataset", "ψ (°)", "sin²ψ", "Peak 2θ (°)",
                "2θ σ (°)", "d (Å)", "Strain (µε)", "Peak source",
                "Residual (µε)", "Status",
            ]
        )
        self.stress_observation_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        install_table_copy_menu(self.stress_observation_table)
        stress_setup_layout.addWidget(self.stress_observation_table)
        self.stress_results_tabs.addTab(stress_setup_page, "ψ Observations")

        stress_plot_page = QWidget()
        stress_plot_layout = QVBoxLayout(stress_plot_page)
        self.residual_stress_plot = ResidualStressPlotWidget()
        stress_plot_layout.addWidget(self.residual_stress_plot)
        self.stress_results_tabs.addTab(stress_plot_page, "sin²ψ Plot")

        stress_summary_page = QWidget()
        stress_summary_form = QFormLayout(stress_summary_page)
        self.stress_value = QLabel("—")
        self.stress_uncertainty_value = QLabel("—")
        self.stress_sign_value = QLabel("—")
        self.stress_d0_value = QLabel("—")
        self.stress_slope_value = QLabel("—")
        self.stress_intercept_value = QLabel("—")
        self.stress_r2_value = QLabel("—")
        self.stress_rmse_value = QLabel("—")
        self.stress_points_value = QLabel("—")
        self.stress_geometry_value = QLabel("—")
        self.stress_warning_value = QLabel("—")
        self.stress_warning_value.setWordWrap(True)
        for copyable_label in (
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
            install_label_copy_menu(copyable_label)
        stress_summary_form.addRow("Residual stress σφ", self.stress_value)
        stress_summary_form.addRow("Stress uncertainty", self.stress_uncertainty_value)
        stress_summary_form.addRow("Stress sign", self.stress_sign_value)
        stress_summary_form.addRow("Stress-free d₀", self.stress_d0_value)
        stress_summary_form.addRow("Strain slope", self.stress_slope_value)
        stress_summary_form.addRow("Strain intercept", self.stress_intercept_value)
        stress_summary_form.addRow("Regression R²", self.stress_r2_value)
        stress_summary_form.addRow("RMSE", self.stress_rmse_value)
        stress_summary_form.addRow("Observations", self.stress_points_value)
        stress_summary_form.addRow("ψ geometry", self.stress_geometry_value)
        stress_summary_form.addRow("Warnings", self.stress_warning_value)
        self.stress_results_tabs.addTab(stress_summary_page, "Stress Summary")

        stress_layout.addWidget(self.stress_results_tabs)
        self.tabs.addTab(self.stress_tab, "Residual Stress")

        self.calibration_widget = self.workspace_registry.create_builtin_widget(
            "instrument", self
        )
        self.tabs.addTab(self.calibration_widget, "Instrument Calibration")

        self.whole_pattern_widget = self.workspace_registry.create_builtin_widget(
            "pawley_lebail", self
        )
        self.tabs.addTab(
            self.whole_pattern_widget,
            "Pawley / Le Bail",
        )

        self.rietveld_widget = self.workspace_registry.create_builtin_widget(
            "rietveld", self
        )
        self.tabs.addTab(
            self.rietveld_widget,
            "Rietveld",
        )

        self.doping_series_widget = self.workspace_registry.create_builtin_widget(
            "doping_series", self
        )
        self.tabs.addTab(
            self.doping_series_widget,
            "Doping Series",
        )

        self.multicomponent_refiner_widget = self.workspace_registry.create_builtin_widget(
            "multicomponent_refiner", self
        )
        self.tabs.addTab(
            self.multicomponent_refiner_widget,
            "Intelligent Multiphase",
        )

        self.validated_qpa_widget = self.workspace_registry.create_builtin_widget(
            "validated_qpa", self
        )
        self.tabs.addTab(
            self.validated_qpa_widget,
            "Validated QPA",
        )

        self.validation_widget = self.workspace_registry.create_builtin_widget(
            "validation", self
        )
        self.tabs.addTab(
            self.validation_widget,
            "Validation & Evidence",
        )

        self.phase_revolution_widget = self.workspace_registry.create_builtin_widget(
            "unknown_phase", self
        )
        self.tabs.addTab(
            self.phase_revolution_widget,
            "Unknown Phase",
        )

        self.cif_library_widget = self.workspace_registry.create_builtin_widget(
            "cif_library", self
        )
        self.tabs.addTab(
            self.cif_library_widget,
            "CIF Library / Structure Match",
        )

        self.structure_solution_widget = self.workspace_registry.create_builtin_widget(
            "solve_structure", self
        )
        self.tabs.addTab(
            self.structure_solution_widget,
            "Solve Structure",
        )

        self.batch_widget = self.workspace_registry.create_builtin_widget(
            "batch_reports", self
        )
        self.tabs.addTab(self.batch_widget, "Batch & Reports")

        self.advanced_analysis_tab = self._placeholder_tab(
            "Advanced Analysis",
            "Independent atomic-coordinate refinement, texture tensors, pair-distribution-function analysis, and fundamental-parameters profile modeling remain planned as separate validated engines.",
        )
        self.tabs.addTab(self.advanced_analysis_tab, "Advanced Analysis")

        self.full_gui_workbench_widget = self.workspace_registry.create_builtin_widget(
            "full_gui", self
        )
        self.tabs.addTab(self.full_gui_workbench_widget, "Full GUI Workbench")

        self.project_home_widget = ProjectHomeWidget(self)
        self.tabs.addTab(self.project_home_widget, "Project Home")

        self._configure_unified_workflow()
        layout.addWidget(self.tabs)
        return panel

    def _build_right_panel(self):
        scroll = QScrollArea()
        scroll.setObjectName("analysisControlScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("analysisControlContent")
        content.setMinimumWidth(290)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)

        def configure_form(form_layout: QFormLayout) -> QFormLayout:
            form_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
            form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            form_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            form_layout.setFormAlignment(Qt.AlignTop)
            form_layout.setHorizontalSpacing(8)
            form_layout.setVerticalSpacing(6)
            return form_layout

        appearance_group = QGroupBox("Appearance")
        appearance_form = QFormLayout(appearance_group)
        configure_form(appearance_form)
        self.theme_selector = NoWheelComboBox()
        self.theme_selector.addItems(list(THEMES.keys()))
        self.theme_selector.setCurrentText(self.current_theme_name)
        appearance_form.addRow("Theme", self.theme_selector)
        layout.addWidget(appearance_group)

        preprocessing = QGroupBox("Essential parameters")
        form = QFormLayout(preprocessing)
        configure_form(form)
        self.preprocessing_form = form

        self.background_check = QCheckBox("Subtract detected background")
        self.background_method = NoWheelComboBox()
        self.background_method.addItems(list(BACKGROUND_METHODS))
        self.background_method.setCurrentText("Auto ensemble")

        self.background_smoothness = NoWheelDoubleSpinBox()
        self.background_smoothness.setRange(1.0, 100.0)
        self.background_smoothness.setDecimals(1)
        self.background_smoothness.setValue(70.0)

        self.background_asymmetry = NoWheelDoubleSpinBox()
        self.background_asymmetry.setRange(0.00001, 0.49)
        self.background_asymmetry.setDecimals(5)
        self.background_asymmetry.setValue(0.01)

        self.background_iterations = NoWheelSpinBox()
        self.background_iterations.setRange(5, 250)
        self.background_iterations.setValue(50)

        self.background_window_degrees = NoWheelDoubleSpinBox()
        self.background_window_degrees.setRange(0.02, 30.0)
        self.background_window_degrees.setDecimals(3)
        self.background_window_degrees.setValue(2.0)

        self.background_percentile = NoWheelDoubleSpinBox()
        self.background_percentile.setRange(1.0, 49.0)
        self.background_percentile.setDecimals(1)
        self.background_percentile.setValue(20.0)

        self.poly_order = NoWheelSpinBox()
        self.poly_order.setRange(1, 12)
        self.poly_order.setValue(3)

        self.background_peak_protection = QCheckBox("Protect detected crystalline peaks")
        self.background_peak_protection.setChecked(True)
        self.background_clip_negative = QCheckBox("Clip negative corrected values to zero")
        self.background_clip_negative.setChecked(False)
        self.preview_background_button = QPushButton("Preview background")

        self.smooth_check = QCheckBox("Apply advanced smart smoothing")
        self.smoothing_method = NoWheelComboBox()
        self.smoothing_method.addItems(list(SMOOTHING_METHODS))
        self.smoothing_method.setCurrentText("Auto intelligent")

        self.smoothing_strength = NoWheelDoubleSpinBox()
        self.smoothing_strength.setRange(0.0, 100.0)
        self.smoothing_strength.setDecimals(1)
        self.smoothing_strength.setValue(45.0)

        self.smoothing_peak_protection = QCheckBox("Protect crystalline peak shapes")
        self.smoothing_peak_protection.setChecked(True)
        self.smoothing_peak_preservation = NoWheelDoubleSpinBox()
        self.smoothing_peak_preservation.setRange(0.0, 100.0)
        self.smoothing_peak_preservation.setDecimals(1)
        self.smoothing_peak_preservation.setValue(85.0)

        self.smoothing_max_position_shift = NoWheelDoubleSpinBox()
        self.smoothing_max_position_shift.setRange(0.0001, 1.0)
        self.smoothing_max_position_shift.setDecimals(4)
        self.smoothing_max_position_shift.setValue(0.02)
        self.smoothing_max_height_change = NoWheelDoubleSpinBox()
        self.smoothing_max_height_change.setRange(0.25, 100.0)
        self.smoothing_max_height_change.setDecimals(2)
        self.smoothing_max_height_change.setValue(8.0)
        self.smoothing_max_fwhm_change = NoWheelDoubleSpinBox()
        self.smoothing_max_fwhm_change.setRange(0.25, 100.0)
        self.smoothing_max_fwhm_change.setDecimals(2)
        self.smoothing_max_fwhm_change.setValue(10.0)
        self.preview_smoothing_button = QPushButton("Preview smart smoothing")

        # Retained for backward compatibility with older .afz recipes.
        self.smooth_window = NoWheelSpinBox()
        self.smooth_window.setRange(5, 501)
        self.smooth_window.setValue(11)
        self.smooth_order = NoWheelSpinBox()
        self.smooth_order.setRange(1, 9)
        self.smooth_order.setValue(3)

        self.normalize_check = QCheckBox("Normalize maximum to 100")
        self.apply_button = QPushButton("Apply to selected dataset")
        self.apply_button.setObjectName("primaryButton")

        form.addRow(self.background_check)
        form.addRow("Background method", self.background_method)
        form.addRow("Smoothness", self.background_smoothness)
        form.addRow("AsLS asymmetry", self.background_asymmetry)
        form.addRow("Maximum iterations", self.background_iterations)
        form.addRow("Window width (°2θ)", self.background_window_degrees)
        form.addRow("Rolling percentile", self.background_percentile)
        form.addRow("Polynomial order", self.poly_order)
        form.addRow(self.background_peak_protection)
        form.addRow(self.background_clip_negative)
        form.addRow(self.preview_background_button)
        form.addRow(self.smooth_check)
        form.addRow("Smoothing method", self.smoothing_method)
        form.addRow("Smoothing strength", self.smoothing_strength)
        form.addRow(self.smoothing_peak_protection)
        form.addRow("Peak preservation (%)", self.smoothing_peak_preservation)
        form.addRow("Maximum peak shift (°2θ)", self.smoothing_max_position_shift)
        form.addRow("Maximum height change (%)", self.smoothing_max_height_change)
        form.addRow("Maximum FWHM change (%)", self.smoothing_max_fwhm_change)
        form.addRow(self.preview_smoothing_button)
        form.addRow(self.normalize_check)
        form.addRow(self.apply_button)
        layout.addWidget(preprocessing)

        peak_group = QGroupBox("Essential parameters")
        peak_form = QFormLayout(peak_group)
        configure_form(peak_form)
        self.prominence = NoWheelDoubleSpinBox()
        self.prominence.setRange(0.001, 1.0)
        self.prominence.setDecimals(3)
        self.prominence.setSingleStep(0.005)
        self.prominence.setValue(0.03)
        self.min_distance = NoWheelSpinBox()
        self.min_distance.setRange(1, 1000)
        self.min_distance.setValue(5)
        self.smart_sensitivity = NoWheelComboBox()
        self.smart_sensitivity.addItems(
            ["Conservative", "Balanced", "Sensitive"]
        )
        self.smart_sensitivity.setCurrentText("Balanced")

        self.find_peaks_button = QPushButton("Find peaks")
        self.smart_peaks_button = QPushButton("Smart peak search")
        self.smart_peaks_button.setObjectName("primaryButton")
        self.view_peak_table_button = QPushButton("View peak table")

        self.manual_peak_position = NoWheelDoubleSpinBox()
        self.manual_peak_position.setRange(-1000000.0, 1000000.0)
        self.manual_peak_position.setDecimals(6)
        self.manual_peak_position.setValue(20.0)
        self.manual_peak_snap_window = NoWheelDoubleSpinBox()
        self.manual_peak_snap_window.setRange(0.0, 5.0)
        self.manual_peak_snap_window.setDecimals(4)
        self.manual_peak_snap_window.setValue(0.15)
        self.add_manual_peak_button = QPushButton("Add manual peak")
        self.preserve_manual_peaks_check = QCheckBox(
            "Preserve manual/protected peaks during searches"
        )
        self.preserve_manual_peaks_check.setChecked(True)

        peak_button_row = QHBoxLayout()
        peak_button_row.addWidget(self.find_peaks_button)
        peak_button_row.addWidget(self.smart_peaks_button)

        smart_note = QLabel(
            "Smart mode estimates noise and background automatically, "
            "rejects narrow noise spikes, and scores peak confidence."
        )
        smart_note.setWordWrap(True)
        smart_note.setObjectName("mutedLabel")

        peak_form.addRow("Prominence fraction", self.prominence)
        peak_form.addRow("Minimum distance, points", self.min_distance)
        peak_form.addRow("Smart sensitivity", self.smart_sensitivity)
        peak_form.addRow(peak_button_row)
        peak_form.addRow(smart_note)
        peak_form.addRow("Manual peak 2θ (°)", self.manual_peak_position)
        peak_form.addRow("Snap window (°2θ)", self.manual_peak_snap_window)
        peak_form.addRow(self.add_manual_peak_button)
        peak_form.addRow(self.preserve_manual_peaks_check)
        peak_form.addRow(self.view_peak_table_button)
        layout.addWidget(peak_group)

        fitting_group = QGroupBox("Advanced parameters")
        fitting_form = QFormLayout(fitting_group)
        configure_form(fitting_form)

        self.fit_model = NoWheelComboBox()
        self.fit_model.addItems(["Pseudo-Voigt", "Gaussian", "Lorentzian"])
        self.fit_model.setCurrentText("Pseudo-Voigt")

        self.fit_window_multiplier = NoWheelDoubleSpinBox()
        self.fit_window_multiplier.setRange(1.5, 15.0)
        self.fit_window_multiplier.setDecimals(1)
        self.fit_window_multiplier.setSingleStep(0.5)
        self.fit_window_multiplier.setValue(5.0)

        self.advanced_fit_model = NoWheelComboBox()
        self.advanced_fit_model.addItems(
            [
                "Auto",
                "Pseudo-Voigt",
                "Pearson VII",
                "Split Pseudo-Voigt",
                "Voigt",
                "Gaussian",
                "Lorentzian",
            ]
        )

        self.advanced_baseline_model = NoWheelComboBox()
        self.advanced_baseline_model.addItems(
            ["Quadratic", "Linear", "Constant"]
        )

        self.advanced_robust_loss = NoWheelComboBox()
        self.advanced_robust_loss.addItems(
            ["Soft L1", "Huber", "Cauchy", "Least squares"]
        )

        self.advanced_selection_criterion = NoWheelComboBox()
        self.advanced_selection_criterion.addItems(["BIC", "AICc"])

        self.maximum_extra_components = NoWheelSpinBox()
        self.maximum_extra_components.setRange(0, 5)
        self.maximum_extra_components.setValue(2)

        self.shared_width_check = QCheckBox(
            "Share FWHM within a fitted group when supported"
        )

        self.show_fits_check = QCheckBox("Show total fitted envelope")
        self.show_fits_check.setChecked(True)
        self.show_fit_components_check = QCheckBox(
            "Show separated component curves"
        )
        self.show_fit_components_check.setChecked(True)
        self.show_fit_baselines_check = QCheckBox(
            "Show fitted local baseline"
        )
        self.show_residuals_check = QCheckBox("Show residual curves")

        self.fit_peaks_button = QPushButton("Standard fit")
        self.advanced_fit_button = QPushButton(
            "Advanced deconvolution"
        )
        self.advanced_fit_button.setObjectName("primaryButton")
        self.clear_fits_button = QPushButton("Clear fits")
        self.view_fit_table_button = QPushButton("View fit table")
        self.view_model_selection_button = QPushButton(
            "View model selection"
        )
        self.export_fit_table_button = QPushButton("Export peak tables via Raptor")

        fit_button_row = QVBoxLayout()
        fit_button_row.addWidget(self.fit_peaks_button)
        fit_button_row.addWidget(self.advanced_fit_button)
        fit_button_row.addWidget(self.clear_fits_button)

        self.deconvolution_progress_label = QLabel(
            "Advanced deconvolution is ready."
        )
        self.deconvolution_progress_label.setObjectName("mutedLabel")
        self.deconvolution_progress_label.setWordWrap(True)
        install_label_copy_menu(self.deconvolution_progress_label)

        self.deconvolution_progress = QProgressBar()
        self.deconvolution_progress.setRange(0, 1)
        self.deconvolution_progress.setValue(0)
        self.deconvolution_progress.setFormat(
            "%v / %m steps completed — %p%"
        )
        self.deconvolution_progress.setTextVisible(True)

        self.cancel_deconvolution_button = QPushButton(
            "Cancel deconvolution"
        )
        self.cancel_deconvolution_button.setEnabled(False)

        progress_row = QHBoxLayout()
        progress_row.addWidget(self.deconvolution_progress, 1)
        progress_row.addWidget(self.cancel_deconvolution_button)

        fit_note = QLabel(
            "Advanced mode uses robust nonlinear least squares, tests multiple "
            "XRD profile models, adds hidden shoulder components from positive "
            "residuals, and selects component count using BIC or AICc. Colored "
            "dotted curves are separated components; the white curve is the total envelope."
        )
        fit_note.setWordWrap(True)
        fit_note.setObjectName("mutedLabel")

        fitting_form.addRow("Standard profile", self.fit_model)
        fitting_form.addRow("Window × initial FWHM", self.fit_window_multiplier)
        fitting_form.addRow("Advanced profile", self.advanced_fit_model)
        fitting_form.addRow("Local baseline", self.advanced_baseline_model)
        fitting_form.addRow("Robust loss", self.advanced_robust_loss)
        fitting_form.addRow(
            "Model selection",
            self.advanced_selection_criterion,
        )
        fitting_form.addRow(
            "Maximum hidden shoulders",
            self.maximum_extra_components,
        )
        fitting_form.addRow(self.shared_width_check)
        fitting_form.addRow(self.show_fits_check)
        fitting_form.addRow(self.show_fit_components_check)
        fitting_form.addRow(self.show_fit_baselines_check)
        fitting_form.addRow(self.show_residuals_check)
        fitting_form.addRow(fit_button_row)
        fitting_form.addRow(self.deconvolution_progress_label)
        fitting_form.addRow(progress_row)
        fitting_form.addRow(self.view_fit_table_button)
        fitting_form.addRow(self.view_model_selection_button)
        fitting_form.addRow(self.export_fit_table_button)
        fitting_form.addRow(fit_note)
        layout.addWidget(fitting_group)

        size_group = QGroupBox("Microstructure parameters")
        size_form = QFormLayout(size_group)
        configure_form(size_form)

        self.wavelength_angstrom = NoWheelDoubleSpinBox()
        self.wavelength_angstrom.setRange(0.1, 5.0)
        self.wavelength_angstrom.setDecimals(6)
        self.wavelength_angstrom.setSingleStep(0.0001)
        self.wavelength_angstrom.setValue(1.5406)

        self.shape_factor = NoWheelDoubleSpinBox()
        self.shape_factor.setRange(0.1, 2.0)
        self.shape_factor.setDecimals(4)
        self.shape_factor.setSingleStep(0.01)
        self.shape_factor.setValue(0.9)

        self.instrument_fwhm = NoWheelDoubleSpinBox()
        self.instrument_fwhm.setRange(0.0, 10.0)
        self.instrument_fwhm.setDecimals(6)
        self.instrument_fwhm.setSingleStep(0.001)
        self.instrument_fwhm.setValue(0.0)

        self.instrument_correction = NoWheelComboBox()
        self.instrument_correction.addItems(
            ["None", "Gaussian quadrature", "Lorentzian linear"]
        )

        self.use_metadata_wavelength_button = QPushButton(
            "Use wavelength from dataset"
        )
        self.calculate_size_button = QPushButton(
            "Calculate microstructure"
        )
        self.calculate_size_button.setObjectName("primaryButton")
        self.clear_size_button = QPushButton("Clear results")
        self.view_size_button = QPushButton("View results")
        self.export_size_button = QPushButton("Export size/strain via Raptor")

        size_button_row = QHBoxLayout()
        size_button_row.addWidget(self.calculate_size_button)
        size_button_row.addWidget(self.clear_size_button)

        size_note = QLabel(
            "Wavelength 1.5406 Å and shape factor 0.9 are suggested starting "
            "points requiring optimization. Use an accepted instrument profile "
            "from calibration whenever possible; otherwise constant FWHM is only "
            "a rough correction. Advanced hkl-dependent size, strain and texture "
            "diagnostics require assigned reflections."
        )
        size_note.setWordWrap(True)
        size_note.setObjectName("mutedLabel")

        size_form.addRow("Wavelength (Å)", self.wavelength_angstrom)
        size_form.addRow(self.use_metadata_wavelength_button)
        size_form.addRow("Scherrer shape factor K", self.shape_factor)
        size_form.addRow("Instrument FWHM (° 2θ)", self.instrument_fwhm)
        size_form.addRow("Width correction", self.instrument_correction)
        size_form.addRow(size_button_row)
        size_form.addRow(self.view_size_button)
        size_form.addRow(self.export_size_button)
        size_form.addRow(size_note)
        layout.addWidget(size_group)

        crystal_group = QGroupBox("Cell-refinement parameters")
        crystal_form = QFormLayout(crystal_group)
        configure_form(crystal_form)

        self.import_cif_button = QPushButton("Import CIF reference")
        self.import_cif_button.setObjectName("primaryButton")
        self.calculate_reference_button = QPushButton(
            "Calculate reference pattern"
        )

        self.reference_cutoff = NoWheelDoubleSpinBox()
        self.reference_cutoff.setRange(0.0, 100.0)
        self.reference_cutoff.setDecimals(2)
        self.reference_cutoff.setSingleStep(0.1)
        self.reference_cutoff.setValue(0.5)

        self.match_tolerance = NoWheelDoubleSpinBox()
        self.match_tolerance.setRange(0.005, 5.0)
        self.match_tolerance.setDecimals(4)
        self.match_tolerance.setSingleStep(0.01)
        self.match_tolerance.setValue(0.20)

        self.crystal_system_selector = NoWheelComboBox()
        self.crystal_system_selector.addItems(list(CRYSTAL_SYSTEMS))

        self.refine_zero_shift_check = QCheckBox("Refine constant 2θ zero shift")
        self.refine_zero_shift_check.setChecked(True)
        self.show_reference_check = QCheckBox(
            "Show CIF reference sticks on main plot"
        )
        self.show_reference_check.setChecked(True)

        self.match_refine_button = QPushButton(
            "Match peaks and refine cell"
        )
        self.match_refine_button.setObjectName("primaryButton")
        self.clear_reference_button = QPushButton("Clear CIF reference")
        self.view_crystal_button = QPushButton("View CIF and cell results")
        self.export_reference_button = QPushButton(
            "Export structure tables via Raptor"
        )
        self.export_cell_button = QPushButton(
            "Export cell tables via Raptor"
        )

        crystal_button_row = QVBoxLayout()
        crystal_button_row.addWidget(self.import_cif_button)
        crystal_button_row.addWidget(self.clear_reference_button)

        crystal_note = QLabel(
            "Phase 4 performs CIF stick-pattern calculation, one-to-one peak "
            "matching, and fixed-hkl unit-cell refinement. Calculated intensities "
            "use an approximate kinematic model and are not Rietveld intensities."
        )
        crystal_note.setWordWrap(True)
        crystal_note.setObjectName("mutedLabel")

        crystal_form.addRow(crystal_button_row)
        crystal_form.addRow("Reference cutoff (%)", self.reference_cutoff)
        crystal_form.addRow(self.calculate_reference_button)
        crystal_form.addRow(self.show_reference_check)
        crystal_form.addRow("Match tolerance (° 2θ)", self.match_tolerance)
        crystal_form.addRow("Crystal system", self.crystal_system_selector)
        crystal_form.addRow(self.refine_zero_shift_check)
        crystal_form.addRow(self.match_refine_button)
        crystal_form.addRow(self.view_crystal_button)
        crystal_form.addRow(self.export_reference_button)
        crystal_form.addRow(self.export_cell_button)
        crystal_form.addRow(crystal_note)
        layout.addWidget(crystal_group)

        phase_group = QGroupBox("Identification parameters")
        phase_form = QFormLayout(phase_group)
        configure_form(phase_form)

        self.phase_peak_source = NoWheelComboBox()
        self.phase_peak_source.addItems(
            [
                "Auto: fitted > detected > smart",
                "Fitted peaks only",
                "Detected peaks only",
            ]
        )

        self.phase_tolerance = NoWheelDoubleSpinBox()
        self.phase_tolerance.setRange(0.005, 2.0)
        self.phase_tolerance.setDecimals(4)
        self.phase_tolerance.setSingleStep(0.01)
        self.phase_tolerance.setValue(0.20)

        self.phase_max_shift = NoWheelDoubleSpinBox()
        self.phase_max_shift.setRange(0.0, 2.0)
        self.phase_max_shift.setDecimals(4)
        self.phase_max_shift.setSingleStep(0.01)
        self.phase_max_shift.setValue(0.30)

        self.phase_reference_cutoff = NoWheelDoubleSpinBox()
        self.phase_reference_cutoff.setRange(0.0, 100.0)
        self.phase_reference_cutoff.setDecimals(2)
        self.phase_reference_cutoff.setSingleStep(0.5)
        self.phase_reference_cutoff.setValue(1.0)

        self.phase_maximum_phases = NoWheelSpinBox()
        self.phase_maximum_phases.setRange(1, 3)
        self.phase_maximum_phases.setValue(1)

        self.phase_mixture_pool = NoWheelSpinBox()
        self.phase_mixture_pool.setRange(2, 10)
        self.phase_mixture_pool.setValue(6)

        self.phase_convert_d_check = QCheckBox(
            "Convert reference d-spacings to selected wavelength"
        )
        self.phase_convert_d_check.setChecked(True)

        self.identify_phases_button = QPushButton("Search local reference library")
        self.identify_phases_button.setObjectName("primaryButton")
        self.clear_phase_results_button = QPushButton("Clear identification results")
        self.view_phase_results_button = QPushButton("View phase results")
        self.export_phase_results_button = QPushButton("Export phase tables via Raptor")

        phase_button_row = QVBoxLayout()
        phase_button_row.addWidget(self.identify_phases_button)
        phase_button_row.addWidget(self.clear_phase_results_button)

        phase_note = QLabel(
            "The local library is built from imported Jade/PDF reference-card datasets "
            "and the active calculated CIF pattern. Scores rank candidates; they do not "
            "prove a phase without reviewing matched and unexplained peaks."
        )
        phase_note.setWordWrap(True)
        phase_note.setObjectName("mutedLabel")

        phase_form.addRow("Observed peak source", self.phase_peak_source)
        phase_form.addRow("Match tolerance (° 2θ)", self.phase_tolerance)
        phase_form.addRow("Maximum zero shift (°)", self.phase_max_shift)
        phase_form.addRow("Reference intensity cutoff (%)", self.phase_reference_cutoff)
        phase_form.addRow("Maximum phases in mixture", self.phase_maximum_phases)
        phase_form.addRow("Top references in mixture pool", self.phase_mixture_pool)
        phase_form.addRow(self.phase_convert_d_check)
        phase_form.addRow(phase_button_row)
        phase_form.addRow(self.view_phase_results_button)
        phase_form.addRow(self.export_phase_results_button)
        phase_form.addRow(phase_note)
        layout.addWidget(phase_group)

        qpa_group = QGroupBox("QPA parameters")
        qpa_form = QFormLayout(qpa_group)
        configure_form(qpa_form)

        self.qpa_mode = NoWheelComboBox()
        self.qpa_mode.addItems(list(QPA_MODES))

        self.qpa_use_processed_check = QCheckBox("Use current processed pattern")
        self.qpa_use_processed_check.setChecked(True)
        self.qpa_convert_d_check = QCheckBox(
            "Convert reference d-spacings to selected wavelength"
        )
        self.qpa_convert_d_check.setChecked(True)

        self.qpa_reference_cutoff = NoWheelDoubleSpinBox()
        self.qpa_reference_cutoff.setRange(0.0, 100.0)
        self.qpa_reference_cutoff.setDecimals(2)
        self.qpa_reference_cutoff.setSingleStep(0.5)
        self.qpa_reference_cutoff.setValue(1.0)

        self.qpa_reference_fwhm = NoWheelDoubleSpinBox()
        self.qpa_reference_fwhm.setRange(0.005, 5.0)
        self.qpa_reference_fwhm.setDecimals(4)
        self.qpa_reference_fwhm.setSingleStep(0.01)
        self.qpa_reference_fwhm.setValue(0.20)

        self.qpa_profile_eta = NoWheelDoubleSpinBox()
        self.qpa_profile_eta.setRange(0.0, 1.0)
        self.qpa_profile_eta.setDecimals(3)
        self.qpa_profile_eta.setSingleStep(0.05)
        self.qpa_profile_eta.setValue(0.5)

        self.qpa_baseline_order = NoWheelSpinBox()
        self.qpa_baseline_order.setRange(0, 4)
        self.qpa_baseline_order.setValue(1)

        self.qpa_weighting = NoWheelComboBox()
        self.qpa_weighting.addItems(list(QPA_WEIGHTING))
        self.qpa_weighting.setCurrentText("Balanced")

        self.qpa_bootstrap_samples = NoWheelSpinBox()
        self.qpa_bootstrap_samples.setRange(0, 1000)
        self.qpa_bootstrap_samples.setSingleStep(10)
        self.qpa_bootstrap_samples.setValue(50)

        self.qpa_internal_standard_check = QCheckBox(
            "Apply internal-standard correction"
        )
        self.qpa_internal_standard_combo = NoWheelComboBox()
        self.qpa_known_standard_percent = NoWheelDoubleSpinBox()
        self.qpa_known_standard_percent.setRange(0.01, 99.99)
        self.qpa_known_standard_percent.setDecimals(3)
        self.qpa_known_standard_percent.setSingleStep(1.0)
        self.qpa_known_standard_percent.setValue(20.0)

        self.run_qpa_button = QPushButton("Quantify selected Phase 5 candidate")
        self.run_qpa_button.setObjectName("primaryButton")
        self.clear_qpa_button = QPushButton("Clear QPA results")
        self.view_qpa_button = QPushButton("View QPA results")
        self.export_qpa_button = QPushButton("Export QPA via Raptor")

        qpa_button_row = QVBoxLayout()
        qpa_button_row.addWidget(self.run_qpa_button)
        qpa_button_row.addWidget(self.clear_qpa_button)

        qpa_note = QLabel(
            "Phase 6 uses non-negative whole-pattern scaling of broadened local references. "
            "RIR correction is semi-quantitative and is not Rietveld QPA. Review missing phases, "
            "pattern correlation, background, reference FWHM, and internal-standard consistency."
        )
        qpa_note.setWordWrap(True)
        qpa_note.setObjectName("mutedLabel")

        qpa_form.addRow("Quantification mode", self.qpa_mode)
        qpa_form.addRow(self.qpa_use_processed_check)
        qpa_form.addRow(self.qpa_convert_d_check)
        qpa_form.addRow("Reference cutoff (%)", self.qpa_reference_cutoff)
        qpa_form.addRow("Reference FWHM (° 2θ)", self.qpa_reference_fwhm)
        qpa_form.addRow("Pseudo-Voigt η", self.qpa_profile_eta)
        qpa_form.addRow("Background polynomial order", self.qpa_baseline_order)
        qpa_form.addRow("Pattern weighting", self.qpa_weighting)
        qpa_form.addRow("Residual bootstrap samples", self.qpa_bootstrap_samples)
        qpa_form.addRow(self.qpa_internal_standard_check)
        qpa_form.addRow("Internal-standard phase", self.qpa_internal_standard_combo)
        qpa_form.addRow("Known standard wt%", self.qpa_known_standard_percent)
        qpa_form.addRow(qpa_button_row)
        qpa_form.addRow(self.view_qpa_button)
        qpa_form.addRow(self.export_qpa_button)
        qpa_form.addRow(qpa_note)
        layout.addWidget(qpa_group)

        stress_group = QGroupBox("Stress parameters")
        stress_form = QFormLayout(stress_group)
        configure_form(stress_form)

        self.stress_peak_source = NoWheelComboBox()
        self.stress_peak_source.addItems(list(PEAK_SOURCE_MODES))

        self.stress_target_two_theta = NoWheelDoubleSpinBox()
        self.stress_target_two_theta.setRange(0.1, 179.0)
        self.stress_target_two_theta.setDecimals(6)
        self.stress_target_two_theta.setSingleStep(0.1)
        self.stress_target_two_theta.setValue(32.0)

        self.stress_search_window = NoWheelDoubleSpinBox()
        self.stress_search_window.setRange(0.01, 10.0)
        self.stress_search_window.setDecimals(4)
        self.stress_search_window.setSingleStep(0.05)
        self.stress_search_window.setValue(0.40)

        self.stress_wavelength = NoWheelDoubleSpinBox()
        self.stress_wavelength.setRange(0.1, 5.0)
        self.stress_wavelength.setDecimals(6)
        self.stress_wavelength.setSingleStep(0.0001)
        self.stress_wavelength.setValue(1.5406)
        self.stress_use_metadata_wavelength_button = QPushButton(
            "Use selected dataset wavelength"
        )

        self.stress_default_peak_error = NoWheelDoubleSpinBox()
        self.stress_default_peak_error.setRange(0.000001, 2.0)
        self.stress_default_peak_error.setDecimals(6)
        self.stress_default_peak_error.setSingleStep(0.001)
        self.stress_default_peak_error.setValue(0.01)

        self.stress_azimuth = NoWheelDoubleSpinBox()
        self.stress_azimuth.setRange(-360.0, 360.0)
        self.stress_azimuth.setDecimals(3)
        self.stress_azimuth.setValue(0.0)

        self.stress_reference_mode = NoWheelComboBox()
        self.stress_reference_mode.addItems(list(REFERENCE_MODES))

        self.stress_free_two_theta = NoWheelDoubleSpinBox()
        self.stress_free_two_theta.setRange(0.1, 179.0)
        self.stress_free_two_theta.setDecimals(6)
        self.stress_free_two_theta.setValue(32.0)

        self.stress_elastic_mode = NoWheelComboBox()
        self.stress_elastic_mode.addItems(list(ELASTIC_MODES))

        self.stress_youngs_modulus = NoWheelDoubleSpinBox()
        self.stress_youngs_modulus.setRange(0.001, 2000.0)
        self.stress_youngs_modulus.setDecimals(4)
        self.stress_youngs_modulus.setValue(200.0)

        self.stress_poisson_ratio = NoWheelDoubleSpinBox()
        self.stress_poisson_ratio.setRange(-0.99, 0.499)
        self.stress_poisson_ratio.setDecimals(5)
        self.stress_poisson_ratio.setSingleStep(0.01)
        self.stress_poisson_ratio.setValue(0.30)

        self.stress_xec_half_s2 = NoWheelDoubleSpinBox()
        self.stress_xec_half_s2.setRange(0.000001, 1.0)
        self.stress_xec_half_s2.setDecimals(7)
        self.stress_xec_half_s2.setSingleStep(0.0001)
        self.stress_xec_half_s2.setValue(0.0058)

        self.stress_regression_mode = NoWheelComboBox()
        self.stress_regression_mode.addItems(list(REGRESSION_MODES))

        self.build_stress_observations_button = QPushButton(
            "Build ψ observations"
        )
        self.build_stress_observations_button.setObjectName("primaryButton")
        self.calculate_stress_button = QPushButton("Calculate residual stress")
        self.calculate_stress_button.setObjectName("primaryButton")
        self.clear_stress_button = QPushButton("Clear stress analysis")
        self.view_stress_button = QPushButton("View stress results")
        self.export_stress_button = QPushButton("Export stress via Raptor")

        stress_button_row = QVBoxLayout()
        stress_button_row.addWidget(self.build_stress_observations_button)
        stress_button_row.addWidget(self.calculate_stress_button)
        stress_button_row.addWidget(self.clear_stress_button)

        stress_note = QLabel(
            "E=200 GPa, ν=0.30, and ½S₂=0.0058 GPa⁻¹ are suggested starting points requiring replacement with values valid for the selected phase, hkl reflection, and material. Positive stress is tensile; negative is compressive."
        )
        stress_note.setWordWrap(True)
        stress_note.setObjectName("mutedLabel")

        stress_form.addRow("Peak source", self.stress_peak_source)
        stress_form.addRow("Target reflection 2θ (°)", self.stress_target_two_theta)
        stress_form.addRow("Search half-window (°)", self.stress_search_window)
        stress_form.addRow("Wavelength (Å)", self.stress_wavelength)
        stress_form.addRow(self.stress_use_metadata_wavelength_button)
        stress_form.addRow("Fallback peak σ (°)", self.stress_default_peak_error)
        stress_form.addRow("Azimuth φ (°)", self.stress_azimuth)
        stress_form.addRow("Stress-free reference", self.stress_reference_mode)
        stress_form.addRow("Stress-free 2θ (°)", self.stress_free_two_theta)
        stress_form.addRow("Elastic-constant mode", self.stress_elastic_mode)
        stress_form.addRow("Young's modulus E (GPa)", self.stress_youngs_modulus)
        stress_form.addRow("Poisson ratio ν", self.stress_poisson_ratio)
        stress_form.addRow("XEC ½S₂ (GPa⁻¹)", self.stress_xec_half_s2)
        stress_form.addRow("Regression", self.stress_regression_mode)
        stress_form.addRow(stress_button_row)
        stress_form.addRow(self.view_stress_button)
        stress_form.addRow(self.export_stress_button)
        stress_form.addRow(stress_note)
        layout.addWidget(stress_group)

        stack_group = QGroupBox("Display options")
        stack_form = QFormLayout(stack_group)
        configure_form(stack_form)
        self.stack_check = QCheckBox("Enable vertical stacking")
        self.stack_offset = NoWheelDoubleSpinBox()
        self.stack_offset.setRange(-1e9, 1e9)
        self.stack_offset.setDecimals(3)
        self.stack_offset.setValue(100.0)
        self.stack_scale = NoWheelDoubleSpinBox()
        self.stack_scale.setRange(1e-6, 1e6)
        self.stack_scale.setDecimals(6)
        self.stack_scale.setValue(1.0)
        self.auto_stack_button = QPushButton("Estimate automatic offset")
        stack_form.addRow(self.stack_check)
        stack_form.addRow("Offset", self.stack_offset)
        stack_form.addRow("Scale", self.stack_scale)
        stack_form.addRow(self.auto_stack_button)
        layout.addWidget(stack_group)

        export_group = QGroupBox("Quick export")
        export_layout = QGridLayout(export_group)
        self.export_png_button = QPushButton("PNG")
        self.export_svg_button = QPushButton("SVG")
        self.export_csv_button = QPushButton("Clean TXT")
        self.export_excel_button = QPushButton("Clean Excel")
        self.export_center_button = QPushButton("Clean Data Export")
        self.export_center_button.setObjectName("primaryButton")
        export_layout.addWidget(self.export_png_button, 0, 0)
        export_layout.addWidget(self.export_svg_button, 0, 1)
        export_layout.addWidget(self.export_csv_button, 1, 0)
        export_layout.addWidget(self.export_excel_button, 1, 1)
        export_layout.addWidget(self.export_center_button, 2, 0, 1, 2)
        layout.addWidget(export_group)

        warning = QLabel(
            "Scientific note: advanced background subtraction is model-dependent. "
            "Always inspect broad peaks, amorphous humps and negative residuals before publication use."
        )
        warning.setWordWrap(True)
        warning.setObjectName("mutedLabel")
        layout.addWidget(warning)
        layout.addStretch(1)
        scroll.setWidget(content)

        # Phase 20.2: only controls relevant to the active workflow task are
        # displayed. Existing widgets and signal connections are preserved.
        self.analysis_control_content = content
        self.analysis_control_scroll = scroll
        self.analysis_preprocessing_background_widgets = (
            self.background_check,
            self.background_method,
            self.background_smoothness,
            self.background_asymmetry,
            self.background_iterations,
            self.background_window_degrees,
            self.background_percentile,
            self.poly_order,
            self.background_peak_protection,
            self.background_clip_negative,
            self.preview_background_button,
        )
        self.analysis_preprocessing_smoothing_widgets = (
            self.smooth_check,
            self.smoothing_method,
            self.smoothing_strength,
            self.smoothing_peak_protection,
            self.smoothing_peak_preservation,
            self.smoothing_max_position_shift,
            self.smoothing_max_height_change,
            self.smoothing_max_fwhm_change,
            self.preview_smoothing_button,
        )
        self.analysis_preprocessing_shared_widgets = (
            self.normalize_check,
            self.apply_button,
        )

        self.analysis_control_groups = {
            "appearance": appearance_group,
            "preprocessing": preprocessing,
            "peaks": peak_group,
            "fitting": fitting_group,
            "size_strain": size_group,
            "cif_cell": crystal_group,
            "phase_identification": phase_group,
            "qpa_preview": qpa_group,
            "residual_stress": stress_group,
            "stacked_patterns": stack_group,
            "export": export_group,
        }
        fitting_group.setProperty("expertOnly", True)
        stack_group.setProperty("expertOnly", True)
        appearance_group.setProperty("panelRole", "secondary")
        export_group.setProperty("panelRole", "result")
        for group in self.analysis_control_groups.values():
            group.setProperty("unifiedAnalysisSection", True)

        # Phase 22: the inspector separates the controls needed for a normal
        # run from optional tuning. These properties only affect disclosure;
        # every original widget remains in the same scientific form.
        advanced_parameter_widgets = (
            self.background_asymmetry,
            self.background_iterations,
            self.background_window_degrees,
            self.background_percentile,
            self.poly_order,
            self.background_clip_negative,
            self.smoothing_max_position_shift,
            self.smoothing_max_height_change,
            self.smoothing_max_fwhm_change,
            self.manual_peak_position,
            self.manual_peak_snap_window,
            self.add_manual_peak_button,
            self.preserve_manual_peaks_check,
            self.fit_window_multiplier,
            self.shape_factor,
            self.instrument_correction,
            self.reference_cutoff,
            self.refine_zero_shift_check,
            self.phase_max_shift,
            self.phase_reference_cutoff,
            self.phase_mixture_pool,
            self.phase_convert_d_check,
            self.qpa_profile_eta,
            self.qpa_baseline_order,
            self.qpa_weighting,
            self.qpa_bootstrap_samples,
            self.qpa_internal_standard_check,
            self.qpa_internal_standard_combo,
            self.qpa_known_standard_percent,
            self.stress_default_peak_error,
            self.stress_azimuth,
            self.stress_xec_half_s2,
            self.stress_regression_mode,
        )
        for widget in advanced_parameter_widgets:
            widget.setProperty("inspectorAdvanced", True)
        for note in (
            smart_note,
            fit_note,
            size_note,
            crystal_note,
            phase_note,
            qpa_note,
            stress_note,
        ):
            note.setProperty("inspectorAdvanced", True)

        section_tiers = {
            "appearance": "secondary",
            "fitting": "advanced",
            "stacked_patterns": "advanced",
            "export": "result",
        }
        collapsed_by_default = {"appearance", "fitting", "stacked_patterns", "export"}
        self.analysis_inspector_sections = {}
        for key, group in self.analysis_control_groups.items():
            index = layout.indexOf(group)
            layout.removeWidget(group)
            section = AnalysisInspectorSection(
                key,
                group,
                title=group.title(),
                tier=section_tiers.get(key, "essential"),
                expanded=key not in collapsed_by_default,
            )
            layout.insertWidget(index, section)
            section.expansionChanged.connect(
                self._analysis_inspector_expansion_changed
            )
            self.analysis_inspector_sections[key] = section

        self.analysis_panel_header = UnifiedAnalysisHeader()
        self.analysis_validation_card = InlineValidationCard()
        self.analysis_result_card = AnalysisResultCard()
        self.analysis_inspector_toolbar = AnalysisInspectorToolbar()
        self.analysis_action_bar = AnalysisActionBar()

        panel = QWidget()
        panel.setObjectName("analysisPanelContainer")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)
        panel_layout.addWidget(self.analysis_panel_header)
        panel_layout.addWidget(self.analysis_validation_card)
        panel_layout.addWidget(self.analysis_result_card)
        panel_layout.addWidget(self.analysis_inspector_toolbar)
        panel_layout.addWidget(scroll, 1)
        panel_layout.addWidget(self.analysis_action_bar)

        self.analysis_inspector_toolbar.queryChanged.connect(
            self._analysis_inspector_query_changed
        )
        self.analysis_inspector_toolbar.modeChanged.connect(
            self._analysis_inspector_mode_changed
        )
        self.analysis_inspector_toolbar.expandAllRequested.connect(
            lambda: self._set_visible_analysis_sections_expanded(True)
        )
        self.analysis_inspector_toolbar.collapseAllRequested.connect(
            lambda: self._set_visible_analysis_sections_expanded(False)
        )

        self.analysis_action_bar.resetRequested.connect(
            self._reset_current_analysis_parameters
        )
        self.analysis_action_bar.previewRequested.connect(
            self._preview_current_analysis
        )
        self.analysis_action_bar.runRequested.connect(
            self._run_current_workflow_action
        )
        self._analysis_parameter_defaults = capture_widget_defaults(content)
        self._connect_analysis_panel_controls()
        self._restore_analysis_inspector_state(
            getattr(self.application_state, "ui_preferences", {})
        )
        return panel
