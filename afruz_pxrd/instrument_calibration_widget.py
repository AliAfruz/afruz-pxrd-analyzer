from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .context_copy import install_label_copy_menu, install_table_copy_menu
from .text_export import write_mapping_txt
from .instrument_calibration import (
    GEOMETRIES,
    RADIATION_CONFIGURATIONS,
    CalibrationError,
    calibrate_instrument,
    load_profile,
    quality_check,
    save_profile,
)
from .instrument_calibration_plot import InstrumentCalibrationPlotWidget
from .phase_identification import prepare_reference_peaks
from .widgets import NoWheelComboBox, NoWheelDoubleSpinBox


class InstrumentCalibrationWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.profile: dict | None = None
        self.qa_result: dict | None = None
        self._references = []
        self._build_ui()
        self.refresh_results()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        heading = QLabel("Phase 9 — Instrument Calibration and Quality Control")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        note = QLabel(
            "Fit the peaks of a measured standard, select its verified reference "
            "pattern, then calibrate zero shift and angle-dependent U–V–W "
            "instrument broadening. Specimen displacement is valid only for "
            "symmetric Bragg–Brentano geometry."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        install_label_copy_menu(note)
        layout.addWidget(note)

        top = QHBoxLayout()
        setup_group = QGroupBox("Calibration setup")
        setup_form = QFormLayout(setup_group)

        self.reference_selector = NoWheelComboBox()
        self.refresh_reference_button = QPushButton("Refresh references")
        reference_row = QHBoxLayout()
        reference_row.addWidget(self.reference_selector, 1)
        reference_row.addWidget(self.refresh_reference_button)
        setup_form.addRow("Standard reference", reference_row)

        self.instrument_name = QLineEdit("")
        self.operator_name = QLineEdit("")
        self.geometry = NoWheelComboBox()
        self.geometry.addItems(list(GEOMETRIES))

        self.radiation_configuration = NoWheelComboBox()
        self.radiation_configuration.addItems(list(RADIATION_CONFIGURATIONS))

        self.wavelength = NoWheelDoubleSpinBox()
        self.wavelength.setRange(0.1, 10.0)
        self.wavelength.setDecimals(7)
        self.wavelength.setValue(1.5406)

        self.secondary_wavelength = NoWheelDoubleSpinBox()
        self.secondary_wavelength.setRange(0.1, 10.0)
        self.secondary_wavelength.setDecimals(7)
        self.secondary_wavelength.setValue(1.5444274)

        self.secondary_ratio = NoWheelDoubleSpinBox()
        self.secondary_ratio.setRange(0.0, 2.0)
        self.secondary_ratio.setDecimals(5)
        self.secondary_ratio.setValue(0.5)

        self.radius = NoWheelDoubleSpinBox()
        self.radius.setRange(1.0, 2000.0)
        self.radius.setDecimals(3)
        self.radius.setValue(240.0)

        self.match_tolerance = NoWheelDoubleSpinBox()
        self.match_tolerance.setRange(0.005, 2.0)
        self.match_tolerance.setDecimals(4)
        self.match_tolerance.setValue(0.25)

        self.intensity_cutoff = NoWheelDoubleSpinBox()
        self.intensity_cutoff.setRange(0.0, 100.0)
        self.intensity_cutoff.setDecimals(2)
        self.intensity_cutoff.setValue(1.0)

        self.fit_zero_shift = QCheckBox("Fit constant 2θ zero shift")
        self.fit_zero_shift.setChecked(True)
        self.fit_displacement = QCheckBox(
            "Fit specimen displacement"
        )
        self.fit_displacement.setChecked(False)
        self.fit_transparency = QCheckBox(
            "Fit thick-specimen transparency (1/μ)"
        )
        self.fit_transparency.setChecked(False)

        self.sample_length = NoWheelDoubleSpinBox()
        self.sample_length.setRange(0.0, 100.0)
        self.sample_length.setDecimals(3)
        self.sample_length.setValue(0.0)

        self.receiving_slit_length = NoWheelDoubleSpinBox()
        self.receiving_slit_length.setRange(0.0, 100.0)
        self.receiving_slit_length.setDecimals(3)
        self.receiving_slit_length.setValue(0.0)
        self.use_active_profile = QCheckBox(
            "Use this profile for size/strain correction"
        )
        self.use_active_profile.setChecked(True)

        setup_form.addRow("Instrument", self.instrument_name)
        setup_form.addRow("Operator", self.operator_name)
        setup_form.addRow("Geometry", self.geometry)
        setup_form.addRow("Radiation configuration", self.radiation_configuration)
        setup_form.addRow("Wavelength (Å)", self.wavelength)
        setup_form.addRow("Secondary wavelength (Å)", self.secondary_wavelength)
        setup_form.addRow("Secondary/primary ratio", self.secondary_ratio)
        setup_form.addRow("Goniometer radius (mm)", self.radius)
        setup_form.addRow("Axial sample length (mm)", self.sample_length)
        setup_form.addRow("Receiving slit length (mm)", self.receiving_slit_length)
        setup_form.addRow("Match tolerance (°)", self.match_tolerance)
        setup_form.addRow("Reference cutoff (%)", self.intensity_cutoff)
        setup_form.addRow(self.fit_zero_shift)
        setup_form.addRow(self.fit_displacement)
        setup_form.addRow(self.fit_transparency)
        setup_form.addRow(self.use_active_profile)

        button_row = QHBoxLayout()
        self.calibrate_button = QPushButton("Calibrate selected standard")
        self.calibrate_button.setObjectName("primaryButton")
        self.qa_button = QPushButton("Run QA against active profile")
        button_row.addWidget(self.calibrate_button)
        button_row.addWidget(self.qa_button)
        setup_form.addRow(button_row)

        profile_group = QGroupBox("Calibration profile")
        profile_form = QFormLayout(profile_group)
        self.profile_name = QLabel("—")
        self.profile_fingerprint = QLabel("—")
        self.profile_valid_range = QLabel("—")
        self.profile_zero_shift = QLabel("—")
        self.profile_displacement = QLabel("—")
        self.profile_transparency = QLabel("—")
        self.profile_axial_geometry = QLabel("—")
        self.profile_model = QLabel("—")
        self.profile_uvw = QLabel("—")
        self.profile_quality = QLabel("—")
        for label in (
            self.profile_name,
            self.profile_fingerprint,
            self.profile_valid_range,
            self.profile_zero_shift,
            self.profile_displacement,
            self.profile_transparency,
            self.profile_axial_geometry,
            self.profile_model,
            self.profile_uvw,
            self.profile_quality,
        ):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            install_label_copy_menu(label)
        profile_form.addRow("Standard", self.profile_name)
        profile_form.addRow("Fingerprint", self.profile_fingerprint)
        profile_form.addRow("Valid 2θ range", self.profile_valid_range)
        profile_form.addRow("Zero shift", self.profile_zero_shift)
        profile_form.addRow("Displacement", self.profile_displacement)
        profile_form.addRow("Transparency 1/μ", self.profile_transparency)
        profile_form.addRow("Axial geometry", self.profile_axial_geometry)
        profile_form.addRow("Selected profile model", self.profile_model)
        profile_form.addRow("Profile parameters", self.profile_uvw)
        profile_form.addRow("Diagnostics", self.profile_quality)

        profile_buttons = QHBoxLayout()
        self.save_button = QPushButton("Save profile JSON")
        self.load_button = QPushButton("Load profile JSON")
        self.clear_button = QPushButton("Clear profile")
        profile_buttons.addWidget(self.save_button)
        profile_buttons.addWidget(self.load_button)
        profile_buttons.addWidget(self.clear_button)
        profile_form.addRow(profile_buttons)

        top.addWidget(setup_group, 1)
        top.addWidget(profile_group, 1)
        layout.addLayout(top)

        self.results_tabs = QTabWidget()
        table_page = QWidget()
        table_layout = QVBoxLayout(table_page)
        self.observation_table = QTableWidget(0, 12)
        self.observation_table.setHorizontalHeaderLabels(
            [
                "hkl",
                "Reference 2θ",
                "Observed 2θ",
                "Calculated 2θ",
                "Position residual",
                "Observed FWHM",
                "Calculated FWHM",
                "Width residual",
                "Center σ",
                "FWHM σ",
                "Model",
                "Group",
            ]
        )
        self.observation_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        install_table_copy_menu(self.observation_table)
        table_layout.addWidget(self.observation_table)
        self.results_tabs.addTab(table_page, "Matched Standard Peaks")

        width_page = QWidget()
        width_layout = QVBoxLayout(width_page)
        self.width_plot = InstrumentCalibrationPlotWidget()
        width_layout.addWidget(self.width_plot)
        self.results_tabs.addTab(width_page, "Resolution Profile")

        position_page = QWidget()
        position_layout = QVBoxLayout(position_page)
        self.position_plot = InstrumentCalibrationPlotWidget()
        position_layout.addWidget(self.position_plot)
        self.results_tabs.addTab(position_page, "Position Calibration")

        comparison_page = QWidget()
        comparison_layout = QVBoxLayout(comparison_page)
        self.profile_comparison_details = QTextEdit()
        self.profile_comparison_details.setReadOnly(True)
        comparison_layout.addWidget(self.profile_comparison_details)
        self.results_tabs.addTab(comparison_page, "Profile Model Comparison")

        qa_page = QWidget()
        qa_layout = QVBoxLayout(qa_page)
        self.qa_summary = QLabel("Run a QA check after loading or creating a profile.")
        self.qa_summary.setWordWrap(True)
        self.qa_summary.setObjectName("mutedLabel")
        install_label_copy_menu(self.qa_summary)
        qa_layout.addWidget(self.qa_summary)
        self.qa_details = QTextEdit()
        self.qa_details.setReadOnly(True)
        qa_layout.addWidget(self.qa_details, 1)
        self.results_tabs.addTab(qa_page, "Quality Assurance")

        layout.addWidget(self.results_tabs, 1)

        warning = QLabel(
            "Use certified standard data and a verified instrument geometry. "
            "The selected model is chosen from Caglioti pseudo-Voigt, TCH pseudo-Voigt "
            "and split pseudo-Voigt candidates using BIC. Gaussian/Lorentzian separation "
            "and split asymmetry remain model-dependent and must be validated across a broad 2θ range."
        )
        warning.setWordWrap(True)
        warning.setObjectName("mutedLabel")
        layout.addWidget(warning)

        self.refresh_reference_button.clicked.connect(self.refresh_references)
        self.calibrate_button.clicked.connect(self.calibrate)
        self.qa_button.clicked.connect(self.run_qa)
        self.save_button.clicked.connect(self.save_dialog)
        self.load_button.clicked.connect(self.load_dialog)
        self.clear_button.clicked.connect(self.clear_profile)
        self.use_active_profile.toggled.connect(self._sync_active_profile)
        self.radiation_configuration.currentTextChanged.connect(
            self._radiation_configuration_changed
        )
        self._radiation_configuration_changed(
            self.radiation_configuration.currentText()
        )

    def _radiation_configuration_changed(self, name: str):
        is_doublet = "doublet" in str(name)
        self.secondary_wavelength.setEnabled(is_doublet)
        self.secondary_ratio.setEnabled(is_doublet)
        if str(name).startswith("Cu Kα"):
            self.wavelength.setValue(1.5405929)
            self.secondary_wavelength.setValue(1.5444274)
            self.secondary_ratio.setValue(0.5)

    def refresh_references(self):
        current_uid = self.reference_selector.currentData()
        self._references = self.main_window._phase_reference_library()
        self.reference_selector.clear()
        for reference in self._references:
            self.reference_selector.addItem(reference.name, reference.uid)
        if current_uid:
            index = self.reference_selector.findData(current_uid)
            if index >= 0:
                self.reference_selector.setCurrentIndex(index)

    def _selected_reference(self):
        uid = self.reference_selector.currentData()
        return next(
            (reference for reference in self._references if reference.uid == uid),
            None,
        )

    def _prepared_reference_peaks(self, reference, dataset):
        return prepare_reference_peaks(
            reference,
            target_wavelength_angstrom=self.wavelength.value(),
            convert_from_d=True,
            intensity_cutoff_percent=self.intensity_cutoff.value(),
            two_theta_min=float(dataset.x[0]),
            two_theta_max=float(dataset.x[-1]),
        )

    def calibrate(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            QMessageBox.information(
                self,
                "No standard dataset",
                "Select the measured standard dataset.",
            )
            return
        fit_groups = self.main_window.fit_groups.get(dataset.uid, [])
        if not fit_groups:
            QMessageBox.information(
                self,
                "Standard peaks are not fitted",
                "Run peak detection and peak fitting on the standard first.",
            )
            return
        reference = self._selected_reference()
        if reference is None:
            QMessageBox.information(
                self,
                "No standard reference",
                "Import or calculate a verified standard reference pattern.",
            )
            return
        peaks = self._prepared_reference_peaks(reference, dataset)
        try:
            profile = calibrate_instrument(
                fit_groups,
                peaks,
                standard_name=reference.name,
                standard_formula=reference.formula,
                radiation=str(
                    reference.metadata.get("radiation") or "Cu Kα1"
                ),
                wavelength_angstrom=self.wavelength.value(),
                radiation_configuration=self.radiation_configuration.currentText(),
                secondary_wavelength_angstrom=self.secondary_wavelength.value(),
                secondary_to_primary_ratio=self.secondary_ratio.value(),
                geometry=self.geometry.currentText(),
                goniometer_radius_mm=self.radius.value(),
                tolerance_deg=self.match_tolerance.value(),
                intensity_cutoff_percent=0.0,
                fit_zero_shift=self.fit_zero_shift.isChecked(),
                fit_displacement=self.fit_displacement.isChecked(),
                fit_transparency=self.fit_transparency.isChecked(),
                sample_length_mm=self.sample_length.value(),
                receiving_slit_length_mm=self.receiving_slit_length.value(),
                instrument_name=self.instrument_name.text().strip(),
                operator=self.operator_name.text().strip(),
                notes=f"Measured dataset: {dataset.name}",
            )
        except (CalibrationError, ValueError) as exc:
            QMessageBox.critical(
                self,
                "Instrument calibration failed",
                str(exc),
            )
            return
        self.profile = profile
        self.qa_result = None
        self._sync_active_profile()
        self.refresh_results()
        self.main_window.statusBar().showMessage(
            f"Instrument profile calibrated from {profile['matched_reflection_count']} reflections."
        )

    def run_qa(self):
        if not self.profile:
            QMessageBox.information(
                self,
                "No active calibration",
                "Create or load a calibration profile first.",
            )
            return
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return
        fit_groups = self.main_window.fit_groups.get(dataset.uid, [])
        reference = self._selected_reference()
        if not fit_groups or reference is None:
            QMessageBox.information(
                self,
                "QA input incomplete",
                "Select a fitted standard dataset and its reference.",
            )
            return
        peaks = self._prepared_reference_peaks(reference, dataset)
        try:
            self.qa_result = quality_check(
                self.profile,
                fit_groups,
                peaks,
                tolerance_deg=self.match_tolerance.value(),
            )
        except (CalibrationError, ValueError) as exc:
            QMessageBox.critical(self, "QA check failed", str(exc))
            return
        self.refresh_results()
        self.results_tabs.setCurrentIndex(3)

    def _sync_active_profile(self):
        active = bool(
            self.profile and self.use_active_profile.isChecked()
        )
        self.main_window.active_instrument_profile = (
            deepcopy(self.profile) if active else None
        )
        if (
            active
            and hasattr(self.main_window, "instrument_correction")
            and self.main_window.instrument_correction.currentText()
            == "None"
        ):
            self.main_window.instrument_correction.setCurrentText(
                "Gaussian quadrature"
            )
            self.main_window.statusBar().showMessage(
                "Active U–V–W profile selected; width correction set to "
                "Gaussian quadrature. Verify that this approximation is "
                "appropriate for the profile model."
            )

    @staticmethod
    def _fmt(value, digits=7):
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}g}"
        except (TypeError, ValueError):
            return str(value)

    def refresh_results(self):
        profile = self.profile
        self.width_plot.set_profile(profile, "Width")
        self.position_plot.set_profile(profile, "Position")
        rows = [] if not profile else profile.get("observations", [])
        self.observation_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row.get("hkl_label", ""),
                self._fmt(row.get("reference_two_theta_deg")),
                self._fmt(row.get("observed_two_theta_deg")),
                self._fmt(row.get("calculated_two_theta_deg")),
                self._fmt(row.get("position_residual_deg")),
                self._fmt(row.get("observed_fwhm_deg")),
                self._fmt(row.get("calculated_fwhm_deg")),
                self._fmt(row.get("fwhm_residual_deg")),
                self._fmt(row.get("center_error_deg")),
                self._fmt(row.get("fwhm_error_deg")),
                row.get("model", ""),
                row.get("group_id", ""),
            ]
            for column, value in enumerate(values):
                self.observation_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(str(value)),
                )

        if not profile:
            self.profile_name.setText("—")
            self.profile_fingerprint.setText("—")
            self.profile_valid_range.setText("—")
            self.profile_zero_shift.setText("—")
            self.profile_displacement.setText("—")
            self.profile_transparency.setText("—")
            self.profile_axial_geometry.setText("—")
            self.profile_model.setText("—")
            self.profile_uvw.setText("—")
            self.profile_quality.setText("—")
            self.profile_comparison_details.clear()
        else:
            self.profile_name.setText(profile.get("standard_name", "—"))
            self.profile_fingerprint.setText(profile.get("fingerprint", "—"))
            self.profile_valid_range.setText(
                f"{self._fmt(profile.get('valid_two_theta_min_deg'))}–"
                f"{self._fmt(profile.get('valid_two_theta_max_deg'))}°"
            )
            self.profile_zero_shift.setText(
                f"{self._fmt(profile.get('zero_shift_deg'))} ± "
                f"{self._fmt(profile.get('zero_shift_error_deg'))}°"
            )
            self.profile_displacement.setText(
                f"{self._fmt(profile.get('specimen_displacement_mm'))} ± "
                f"{self._fmt(profile.get('specimen_displacement_error_mm'))} mm"
            )
            self.profile_transparency.setText(
                f"{self._fmt(profile.get('inverse_linear_absorption_mm'))} ± "
                f"{self._fmt(profile.get('inverse_linear_absorption_error_mm'))} mm"
            )
            self.profile_axial_geometry.setText(
                f"SH/L={self._fmt(profile.get('axial_sh_over_l'))}; "
                f"sample={self._fmt(profile.get('sample_length_mm'))} mm; "
                f"slit={self._fmt(profile.get('receiving_slit_length_mm'))} mm"
            )
            self.profile_model.setText(
                str(profile.get("profile_model", "Pseudo-Voigt U-V-W"))
            )
            self.profile_uvw.setText(
                f"U={self._fmt(profile.get('caglioti_u'))}, "
                f"V={self._fmt(profile.get('caglioti_v'))}, "
                f"W={self._fmt(profile.get('caglioti_w'))}, "
                f"η={self._fmt(profile.get('eta'))}, "
                f"X={self._fmt(profile.get('lorentzian_x'))}, "
                f"Y={self._fmt(profile.get('lorentzian_y'))}, "
                f"asym={self._fmt(profile.get('axial_asymmetry'))}"
            )
            self.profile_quality.setText(
                f"{profile.get('matched_reflection_count', 0)} peaks; "
                f"position MAE {self._fmt(profile.get('position_mae_deg'))}°; "
                f"width R² {self._fmt(profile.get('width_r_squared'))}; "
                f"selected BIC {self._fmt(profile.get('selected_profile_bic'))}"
            )
            self.profile_comparison_details.setPlainText(
                json.dumps(
                    profile.get("profile_model_comparison", {}),
                    indent=2,
                    ensure_ascii=False,
                )
            )

        if self.qa_result:
            warnings = self.qa_result.get("warnings", [])
            self.qa_summary.setText(
                f"QA status: {self.qa_result.get('status')} — "
                f"{self.qa_result.get('matched_reflection_count')} reflections, "
                f"position MAE {self._fmt(self.qa_result.get('position_mae_deg'))}°, "
                f"width RMSE {self._fmt(self.qa_result.get('width_rmse_deg'))}°."
            )
            self.qa_details.setPlainText(
                json.dumps(self.qa_result, indent=2, ensure_ascii=False)
            )
        else:
            self.qa_summary.setText(
                "Run a QA check after loading or creating a profile."
            )
            self.qa_details.clear()

    def save_dialog(self):
        if not self.profile:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save instrument calibration profile",
            "afruz_instrument_calibration.json",
            "JSON (*.json)",
        )
        if not filename:
            return
        try:
            saved = save_profile(filename, self.profile)
            txt = write_mapping_txt(
                Path(saved).with_suffix(".txt"),
                self.profile,
                title="Instrument calibration profile",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Profile save failed", str(exc))
            return
        self.main_window.statusBar().showMessage(
            f"Instrument profile saved with clean TXT snapshot: {txt['txt_path']}"
        )

    def load_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load instrument calibration profile",
            "",
            "JSON (*.json)",
        )
        if not filename:
            return
        try:
            self.profile = load_profile(filename)
        except Exception as exc:
            QMessageBox.critical(self, "Profile load failed", str(exc))
            return
        self.qa_result = None
        self._sync_active_profile()
        self.refresh_results()

    def clear_profile(self):
        self.profile = None
        self.qa_result = None
        self._sync_active_profile()
        self.refresh_results()

    def apply_theme(self, theme_name: str):
        self.width_plot.apply_theme(theme_name)
        self.position_plot.apply_theme(theme_name)

    def get_state(self) -> dict:
        return {
            "profile": deepcopy(self.profile),
            "qa_result": deepcopy(self.qa_result),
            "reference_uid": self.reference_selector.currentData(),
            "instrument_name": self.instrument_name.text(),
            "operator_name": self.operator_name.text(),
            "geometry": self.geometry.currentText(),
            "radiation_configuration": self.radiation_configuration.currentText(),
            "wavelength": self.wavelength.value(),
            "secondary_wavelength": self.secondary_wavelength.value(),
            "secondary_ratio": self.secondary_ratio.value(),
            "radius": self.radius.value(),
            "sample_length": self.sample_length.value(),
            "receiving_slit_length": self.receiving_slit_length.value(),
            "match_tolerance": self.match_tolerance.value(),
            "intensity_cutoff": self.intensity_cutoff.value(),
            "fit_zero_shift": self.fit_zero_shift.isChecked(),
            "fit_displacement": self.fit_displacement.isChecked(),
            "fit_transparency": self.fit_transparency.isChecked(),
            "use_active_profile": self.use_active_profile.isChecked(),
        }

    def set_state(self, state: dict | None):
        if not isinstance(state, dict):
            return
        self.profile = (
            deepcopy(state.get("profile"))
            if isinstance(state.get("profile"), dict)
            else None
        )
        self.qa_result = (
            deepcopy(state.get("qa_result"))
            if isinstance(state.get("qa_result"), dict)
            else None
        )
        self.instrument_name.setText(state.get("instrument_name", ""))
        self.operator_name.setText(state.get("operator_name", ""))
        self.geometry.setCurrentText(
            state.get("geometry", GEOMETRIES[0])
        )
        self.radiation_configuration.setCurrentText(
            state.get("radiation_configuration", RADIATION_CONFIGURATIONS[0])
        )
        self.wavelength.setValue(state.get("wavelength", 1.5406))
        self.secondary_wavelength.setValue(
            state.get("secondary_wavelength", 1.5444274)
        )
        self.secondary_ratio.setValue(state.get("secondary_ratio", 0.5))
        self.radius.setValue(state.get("radius", 240.0))
        self.sample_length.setValue(state.get("sample_length", 0.0))
        self.receiving_slit_length.setValue(
            state.get("receiving_slit_length", 0.0)
        )
        self.match_tolerance.setValue(
            state.get("match_tolerance", 0.25)
        )
        self.intensity_cutoff.setValue(
            state.get("intensity_cutoff", 1.0)
        )
        self.fit_zero_shift.setChecked(
            state.get("fit_zero_shift", True)
        )
        self.fit_displacement.setChecked(
            state.get("fit_displacement", False)
        )
        self.fit_transparency.setChecked(
            state.get("fit_transparency", False)
        )
        self.use_active_profile.setChecked(
            state.get("use_active_profile", True)
        )
        self.refresh_references()
        uid = state.get("reference_uid")
        if uid:
            index = self.reference_selector.findData(uid)
            if index >= 0:
                self.reference_selector.setCurrentIndex(index)
        self._sync_active_profile()
        self.refresh_results()
