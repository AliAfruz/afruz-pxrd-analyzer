from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import traceback

from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .context_copy import install_label_copy_menu, install_table_copy_menu
from .multicomponent_refiner import (
    CandidatePhase,
    MultiComponentRefinementError,
    MultiComponentSearchSettings,
    export_multicomponent_report_bundle,
    import_candidate_phase,
    run_multicomponent_refinement,
)
from .multicomponent_refiner_worker import MultiComponentRefinerWorker, MultiComponentSecondPassWorker
from .widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox


class MultiComponentRefinerWidget(QWidget):
    """GUI for Phase 19.0 multi-CIF model search and safe refinement."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.results_by_uid: dict[str, dict] = {}
        self.last_export_by_uid: dict[str, dict] = {}
        self._worker_thread: QThread | None = None
        self._worker: MultiComponentRefinerWorker | None = None
        self._active_dataset_uid: str | None = None
        self._active_phases: list[CandidatePhase] = []
        self._worker_mode = "pass1"
        self._build_ui()
        self.refresh_for_selected_dataset()

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(6, 6, 6, 6)

        self.main_scroll = QScrollArea()
        self.main_scroll.setWidgetResizable(True)
        self.main_scroll.setFrameShape(QScrollArea.NoFrame)
        self.main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.main_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        heading = QLabel("Phase 21.0 — Two-Pass Intelligent Multiphase Refiner")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        description = QLabel(
            "Import multiple candidate CIF files, mark phases as required/optional, run a conservative "
            "multiphase Pawley/Rietveld-style search in a non-blocking background worker, inspect "
            "supported/rejected phases, review residual peak mismatches, and run a protected second-pass refinement "
            "without deleting the original measured data."
        )
        description.setWordWrap(True)
        description.setObjectName("mutedLabel")
        install_label_copy_menu(description)
        layout.addWidget(description)

        top = QHBoxLayout()
        top.addWidget(self._build_phase_group(), 3)
        top.addWidget(self._build_settings_group(), 1)
        layout.addLayout(top)

        run_row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh selected pattern")
        self.run_button = QPushButton("Run Pass 1 + detect mismatches")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setToolTip("Run intelligent multiphase search, then classify residual peak mismatches.")
        self.second_pass_button = QPushButton("Run Pass 2 with unchecked windows")
        self.second_pass_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.export_button = QPushButton("Export Phase 19 report bundle")
        self.export_button.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimumWidth(160)
        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("mutedLabel")
        install_label_copy_menu(self.status_label)
        run_row.addWidget(self.refresh_button)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.second_pass_button)
        run_row.addWidget(self.cancel_button)
        run_row.addWidget(self.export_button)
        run_row.addWidget(self.progress_bar)
        run_row.addWidget(self.status_label, 1)
        layout.addLayout(run_row)

        self.result_tabs = QTabWidget()
        self.result_tabs.addTab(self._table_page(self._build_ranked_table()), "Ranked Models")
        self.result_tabs.addTab(self._table_page(self._build_fraction_table()), "Phase Fractions")
        self.result_tabs.addTab(self._table_page(self._build_rejected_table()), "Rejected / Weak")
        self.result_tabs.addTab(self._table_page(self._build_candidate_table()), "Candidate CIFs")
        self.result_tabs.addTab(self._build_mismatch_page(), "Mismatch Review / Pass 2")
        self.result_tabs.addTab(self._build_warnings_page(), "Warnings")
        self.result_tabs.setMinimumHeight(540)
        self.result_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.result_tabs, 3)

        self.main_scroll.setWidget(content)
        root_layout.addWidget(self.main_scroll, 1)

        self.add_button.clicked.connect(self.add_cif_phases)
        self.remove_button.clicked.connect(self.remove_selected_phase)
        self.clear_button.clicked.connect(self.clear_phases)
        self.required_all_button.clicked.connect(lambda: self._set_all_required(True))
        self.optional_all_button.clicked.connect(lambda: self._set_all_required(False))
        self.refresh_button.clicked.connect(self.refresh_for_selected_dataset)
        self.run_button.clicked.connect(self.run_refinement)
        self.second_pass_button.clicked.connect(self.run_second_pass)
        self.cancel_button.clicked.connect(self.cancel_refinement)
        self.export_button.clicked.connect(self.export_result)

    def _build_phase_group(self):
        group = QGroupBox("Candidate CIF phases")
        layout = QVBoxLayout(group)
        button_row = QHBoxLayout()
        self.add_button = QPushButton("Add CIF phase(s)…")
        self.add_button.setObjectName("primaryButton")
        self.remove_button = QPushButton("Remove selected")
        self.clear_button = QPushButton("Clear")
        self.required_all_button = QPushButton("Mark all required")
        self.optional_all_button = QPushButton("Mark all optional")
        button_row.addWidget(self.add_button)
        button_row.addWidget(self.remove_button)
        button_row.addWidget(self.clear_button)
        button_row.addStretch(1)
        button_row.addWidget(self.required_all_button)
        button_row.addWidget(self.optional_all_button)
        layout.addLayout(button_row)

        self.phase_table = QTableWidget(0, 8)
        self.phase_table.setHorizontalHeaderLabels(
            ["Use", "Label", "Required", "Refine cell", "Formula", "Space group", "Atoms", "Source CIF"]
        )
        self.phase_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.phase_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.phase_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.phase_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        self.phase_table.setMinimumHeight(180)
        self.phase_table.setMaximumHeight(260)
        self.phase_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        install_table_copy_menu(self.phase_table)
        layout.addWidget(self.phase_table)
        return group

    def _build_settings_group(self):
        group = QGroupBox("Safe model-search settings")
        form = QFormLayout(group)
        self.dataset_label = QLabel("No pattern selected")
        self.dataset_label.setWordWrap(True)
        install_label_copy_menu(self.dataset_label)
        self.wavelength = NoWheelDoubleSpinBox()
        self.wavelength.setRange(0.1, 5.0)
        self.wavelength.setDecimals(5)
        self.wavelength.setValue(1.5406)
        self.two_theta_min = NoWheelDoubleSpinBox()
        self.two_theta_min.setRange(0.0, 180.0)
        self.two_theta_min.setDecimals(3)
        self.two_theta_min.setValue(5.0)
        self.two_theta_max = NoWheelDoubleSpinBox()
        self.two_theta_max.setRange(0.0, 180.0)
        self.two_theta_max.setDecimals(3)
        self.two_theta_max.setValue(80.0)
        self.max_phases = NoWheelSpinBox()
        self.max_phases.setRange(1, 8)
        self.max_phases.setValue(3)
        self.max_models = NoWheelSpinBox()
        self.max_models.setRange(1, 512)
        self.max_models.setValue(64)
        self.background_order = NoWheelSpinBox()
        self.background_order.setRange(0, 8)
        self.background_order.setValue(3)
        self.weighting = NoWheelComboBox()
        self.weighting.addItems(["Balanced", "Uniform", "Poisson"])
        self.intensity_cutoff = NoWheelDoubleSpinBox()
        self.intensity_cutoff.setRange(0.0, 20.0)
        self.intensity_cutoff.setDecimals(3)
        self.intensity_cutoff.setValue(0.5)
        self.peak_tolerance = NoWheelDoubleSpinBox()
        self.peak_tolerance.setRange(0.01, 2.0)
        self.peak_tolerance.setDecimals(3)
        self.peak_tolerance.setValue(0.18)
        self.min_fraction = NoWheelDoubleSpinBox()
        self.min_fraction.setRange(0.0, 25.0)
        self.min_fraction.setDecimals(3)
        self.min_fraction.setValue(0.5)
        self.mismatch_z = NoWheelDoubleSpinBox()
        self.mismatch_z.setRange(2.0, 50.0)
        self.mismatch_z.setDecimals(1)
        self.mismatch_z.setValue(6.0)
        self.artifact_z = NoWheelDoubleSpinBox()
        self.artifact_z.setRange(3.0, 100.0)
        self.artifact_z.setDecimals(1)
        self.artifact_z.setValue(10.0)
        self.mask_half_width = NoWheelDoubleSpinBox()
        self.mask_half_width.setRange(0.01, 1.0)
        self.mask_half_width.setDecimals(3)
        self.mask_half_width.setValue(0.10)
        self.max_mask_percent = NoWheelDoubleSpinBox()
        self.max_mask_percent.setRange(0.1, 20.0)
        self.max_mask_percent.setDecimals(2)
        self.max_mask_percent.setValue(3.0)
        form.addRow("Selected pattern", self.dataset_label)
        form.addRow("Wavelength (Å)", self.wavelength)
        form.addRow("2θ min", self.two_theta_min)
        form.addRow("2θ max", self.two_theta_max)
        form.addRow("Max phases/model", self.max_phases)
        form.addRow("Max models", self.max_models)
        form.addRow("Background order", self.background_order)
        form.addRow("Weighting", self.weighting)
        form.addRow("Intensity cutoff %", self.intensity_cutoff)
        form.addRow("Peak tolerance (°)", self.peak_tolerance)
        form.addRow("Weak phase cutoff %", self.min_fraction)
        form.addRow("Mismatch residual z", self.mismatch_z)
        form.addRow("Artifact residual z", self.artifact_z)
        form.addRow("Pass 2 half-window (°)", self.mask_half_width)
        form.addRow("Maximum masked points %", self.max_mask_percent)
        boundary = QLabel(
            "Default mode refines scale/background only. Cell/profile/texture refinement must be validated later."
        )
        boundary.setWordWrap(True)
        boundary.setObjectName("mutedLabel")
        install_label_copy_menu(boundary)
        form.addRow("Boundary", boundary)
        return group

    def _build_mismatch_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "Pass 1 finds large positive residual peaks. Uncheck only confirmed artifacts. "
            "Unexplained crystalline peaks stay checked because they may indicate a missing phase."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        layout.addWidget(note)

        buttons = QHBoxLayout()
        self.auto_uncheck_button = QPushButton("Auto-uncheck artifact candidates")
        self.keep_all_mismatch_button = QPushButton("Keep all windows")
        self.mismatch_summary_label = QLabel("Run Pass 1 to generate mismatch review.")
        self.mismatch_summary_label.setObjectName("mutedLabel")
        buttons.addWidget(self.auto_uncheck_button)
        buttons.addWidget(self.keep_all_mismatch_button)
        buttons.addWidget(self.mismatch_summary_label, 1)
        layout.addLayout(buttons)

        self.mismatch_table = QTableWidget(0, 12)
        self.mismatch_table.setHorizontalHeaderLabels(
            [
                "Use Pass 2",
                "2θ observed",
                "Residual",
                "z-score",
                "Width °",
                "Nearest phase",
                "Nearest 2θ",
                "Δ2θ",
                "Classification",
                "Recommended action",
                "Window min",
                "Window max",
            ]
        )
        self._finish_table(self.mismatch_table)
        self.mismatch_table.setMinimumHeight(420)
        layout.addWidget(self.mismatch_table, 1)

        self.second_pass_summary = QTextEdit()
        self.second_pass_summary.setReadOnly(True)
        self.second_pass_summary.setMaximumHeight(130)
        self.second_pass_summary.setPlainText("No Pass 2 result yet.")
        layout.addWidget(self.second_pass_summary)

        self.auto_uncheck_button.clicked.connect(self._auto_uncheck_artifacts)
        self.keep_all_mismatch_button.clicked.connect(self._keep_all_mismatch_windows)
        return page

    def _build_ranked_table(self):
        self.ranked_table = QTableWidget(0, 10)
        self.ranked_table.setHorizontalHeaderLabels(
            ["Rank", "Model", "Status", "Phases", "Rwp %", "Rp %", "R²", "BIC", "BIC gain", "Unexplained peaks"]
        )
        self._finish_table(self.ranked_table)
        return self.ranked_table

    def _build_fraction_table(self):
        self.fraction_table = QTableWidget(0, 7)
        self.fraction_table.setHorizontalHeaderLabels(
            ["Phase", "Pattern fraction %", "Scale", "Status", "Formula", "Required", "Notes"]
        )
        self._finish_table(self.fraction_table)
        return self.fraction_table

    def _build_rejected_table(self):
        self.rejected_table = QTableWidget(0, 3)
        self.rejected_table.setHorizontalHeaderLabels(["Phase", "Status", "Reason"])
        self._finish_table(self.rejected_table)
        return self.rejected_table

    def _build_candidate_table(self):
        self.candidate_result_table = QTableWidget(0, 8)
        self.candidate_result_table.setHorizontalHeaderLabels(
            ["Label", "Formula", "Space group", "Crystal system", "Atoms", "Required", "Plausibility", "Source"]
        )
        self._finish_table(self.candidate_result_table)
        return self.candidate_result_table

    def _build_warnings_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.warnings_box = QTextEdit()
        self.warnings_box.setReadOnly(True)
        self.warnings_box.setText("No result yet.")
        self.warnings_box.setMinimumHeight(420)
        self.warnings_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.warnings_box, 1)
        return page

    def _table_page(self, table):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        table.setMinimumHeight(420)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(table, 1)
        return page

    def _finish_table(self, table):
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setDefaultSectionSize(24)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        install_table_copy_menu(table)

    def refresh_for_selected_dataset(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            self.dataset_label.setText("No pattern selected")
            self.status_label.setText("Import/select a pattern first.")
            return
        self.dataset_label.setText(f"{dataset.name} · {len(dataset.x)} points · {float(dataset.x[0]):.3f}–{float(dataset.x[-1]):.3f}° 2θ")
        if self.two_theta_min.value() <= 0.001:
            self.two_theta_min.setValue(float(dataset.x[0]))
        self.two_theta_max.setValue(float(dataset.x[-1]))
        if dataset.uid in self.results_by_uid:
            self._populate_result(self.results_by_uid[dataset.uid])
        self.status_label.setText("Ready.")

    def add_cif_phases(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add candidate CIF phases",
            "",
            "CIF files (*.cif *.CIF);;All files (*)",
        )
        if not paths:
            return
        failures = []
        for path in paths:
            try:
                phase = import_candidate_phase(path, label=Path(path).stem, required=False, refine_cell=False)
                self._append_phase_row(phase)
            except Exception as exc:
                failures.append(f"{Path(path).name}: {exc}")
        if failures:
            QMessageBox.warning(self, "CIF import warnings", "\n\n".join(failures))
        self.status_label.setText(f"Candidate CIF phases: {self.phase_table.rowCount()}")

    def _append_phase_row(self, phase: CandidatePhase):
        row = self.phase_table.rowCount()
        self.phase_table.insertRow(row)
        use_item = QTableWidgetItem("")
        use_item.setFlags(use_item.flags() | Qt.ItemIsUserCheckable)
        use_item.setCheckState(Qt.Checked if phase.enabled else Qt.Unchecked)
        use_item.setData(Qt.UserRole, phase)
        required_item = QTableWidgetItem("")
        required_item.setFlags(required_item.flags() | Qt.ItemIsUserCheckable)
        required_item.setCheckState(Qt.Checked if phase.required else Qt.Unchecked)
        refine_item = QTableWidgetItem("")
        refine_item.setFlags(refine_item.flags() | Qt.ItemIsUserCheckable)
        refine_item.setCheckState(Qt.Checked if phase.refine_cell else Qt.Unchecked)
        structure = phase.structure
        values = [
            use_item,
            QTableWidgetItem(phase.label),
            required_item,
            refine_item,
            QTableWidgetItem(str(structure.get("formula") or "")),
            QTableWidgetItem(str(structure.get("space_group") or "")),
            QTableWidgetItem(str(len(structure.get("atoms", [])))),
            QTableWidgetItem(str(phase.source)),
        ]
        for column, item in enumerate(values):
            self.phase_table.setItem(row, column, item)

    def remove_selected_phase(self):
        rows = sorted({index.row() for index in self.phase_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.phase_table.removeRow(row)
        self.status_label.setText(f"Candidate CIF phases: {self.phase_table.rowCount()}")

    def clear_phases(self):
        self.phase_table.setRowCount(0)
        self.status_label.setText("Candidate CIF phases cleared.")

    def _set_all_required(self, required: bool):
        state = Qt.Checked if required else Qt.Unchecked
        for row in range(self.phase_table.rowCount()):
            item = self.phase_table.item(row, 2)
            if item is not None:
                item.setCheckState(state)

    def _collect_phases(self) -> list[CandidatePhase]:
        phases = []
        for row in range(self.phase_table.rowCount()):
            base_item = self.phase_table.item(row, 0)
            if base_item is None:
                continue
            phase = base_item.data(Qt.UserRole)
            if not isinstance(phase, CandidatePhase):
                continue
            label_item = self.phase_table.item(row, 1)
            label = label_item.text().strip() if label_item is not None else phase.label
            enabled = base_item.checkState() == Qt.Checked
            required_item = self.phase_table.item(row, 2)
            refine_item = self.phase_table.item(row, 3)
            phases.append(
                CandidatePhase(
                    label=label or phase.label,
                    structure=phase.structure,
                    source=phase.source,
                    role=phase.role,
                    required=required_item is not None and required_item.checkState() == Qt.Checked,
                    enabled=enabled,
                    refine_cell=refine_item is not None and refine_item.checkState() == Qt.Checked,
                    refine_biso=phase.refine_biso,
                    preferred_orientation_hkl=phase.preferred_orientation_hkl,
                    refine_preferred_orientation=phase.refine_preferred_orientation,
                )
            )
        return phases

    def _settings(self) -> MultiComponentSearchSettings:
        return MultiComponentSearchSettings(
            wavelength_angstrom=float(self.wavelength.value()),
            two_theta_min=float(self.two_theta_min.value()),
            two_theta_max=float(self.two_theta_max.value()),
            intensity_cutoff_percent=float(self.intensity_cutoff.value()),
            background_order=int(self.background_order.value()),
            weighting=self.weighting.currentText(),
            max_phases=int(self.max_phases.value()),
            max_models=int(self.max_models.value()),
            refinement_mode="safe_rietveld",
            refine_zero_shift=False,
            refine_profile=False,
            peak_match_tolerance_deg=float(self.peak_tolerance.value()),
            minimum_phase_fraction_percent=float(self.min_fraction.value()),
            mismatch_residual_z_threshold=float(self.mismatch_z.value()),
            artifact_residual_z_threshold=float(self.artifact_z.value()),
            mismatch_window_half_width_deg=float(self.mask_half_width.value()),
            maximum_excluded_point_fraction=float(self.max_mask_percent.value()) / 100.0,
        )

    def run_refinement(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            QMessageBox.information(self, "No pattern", "Import/select a pattern first.")
            return
        if self._worker_thread is not None:
            QMessageBox.information(self, "Already running", "A multiphase search is already running.")
            return
        phases = self._collect_phases()
        if not phases:
            QMessageBox.information(self, "No CIF phases", "Add one or more candidate CIF phases first.")
            return
        self._set_busy(True)
        self._worker_mode = "pass1"
        self._active_dataset_uid = dataset.uid
        self._active_phases = list(phases)
        self.progress_bar.setValue(0)
        self.status_label.setText("0% · Starting background multiphase worker…")

        self._worker_thread = QThread(self)
        self._worker = MultiComponentRefinerWorker(
            list(dataset.x),
            list(dataset.y),
            phases,
            self._settings(),
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_refinement_progress)
        self._worker.finished.connect(self._on_refinement_finished)
        self._worker.failed.connect(self._on_refinement_failed)
        self._worker.cancelled.connect(self._on_refinement_cancelled)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker.cancelled.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._worker.cancelled.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.finished.connect(self._clear_worker_handles)
        self._worker_thread.start()

    def run_second_pass(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            QMessageBox.information(self, "No pattern", "Import/select a pattern first.")
            return
        if self._worker_thread is not None:
            QMessageBox.information(self, "Already running", "A refinement is already running.")
            return
        first_pass = self.results_by_uid.get(dataset.uid)
        if not first_pass or not first_pass.get("best_model"):
            QMessageBox.information(self, "No Pass 1 result", "Run Pass 1 before starting Pass 2.")
            return
        phases = self._collect_phases()
        review = self._collect_mismatch_review()
        excluded = [row for row in review if not row.get("use_in_second_pass", True)]
        if not excluded:
            QMessageBox.information(
                self,
                "Nothing unchecked",
                "No mismatch windows are unchecked. Review the table before Pass 2.",
            )
            return
        self._set_busy(True)
        self._worker_mode = "pass2"
        self._active_dataset_uid = dataset.uid
        self._active_phases = list(phases)
        self.progress_bar.setValue(0)
        self.status_label.setText("0% · Starting reviewed Pass 2 refinement…")

        self._worker_thread = QThread(self)
        self._worker = MultiComponentSecondPassWorker(
            list(dataset.x),
            list(dataset.y),
            phases,
            first_pass,
            review,
            self._settings(),
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_refinement_progress)
        self._worker.finished.connect(self._on_second_pass_finished)
        self._worker.failed.connect(self._on_refinement_failed)
        self._worker.cancelled.connect(self._on_refinement_cancelled)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker.cancelled.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._worker.cancelled.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.finished.connect(self._clear_worker_handles)
        self._worker_thread.start()

    def cancel_refinement(self):
        if self._worker is None:
            return
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancellation requested. Waiting for current model to finish…")
        self._worker.cancel()

    def _set_busy(self, busy: bool):
        self.run_button.setEnabled(not busy)
        self.second_pass_button.setEnabled((not busy) and self._has_reviewable_result())
        self.cancel_button.setEnabled(busy)
        self.add_button.setEnabled(not busy)
        self.remove_button.setEnabled(not busy)
        self.clear_button.setEnabled(not busy)
        self.required_all_button.setEnabled(not busy)
        self.optional_all_button.setEnabled(not busy)
        self.refresh_button.setEnabled(not busy)
        self.phase_table.setEnabled(not busy)
        self.export_button.setEnabled((not busy) and bool(self.results_by_uid))

    def _on_refinement_progress(self, percent: int, message: str, event: object):
        percent = int(max(0, min(100, percent)))
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"{percent}% · {message}")

    def _on_refinement_finished(self, result: object):
        result = dict(result or {})
        dataset_uid = self._active_dataset_uid
        if dataset_uid:
            self.results_by_uid[dataset_uid] = result
        self._populate_result(result)
        self._set_busy(False)
        self.export_button.setEnabled(True)
        self.progress_bar.setValue(100)
        best = result.get("best_model") or {}
        self.status_label.setText(f"100% · Done. Best model: {best.get('model_id', 'none')} · {best.get('status_label', '')}")

    def _on_second_pass_finished(self, result: object):
        second_pass = dict(result or {})
        dataset_uid = self._active_dataset_uid
        if dataset_uid and dataset_uid in self.results_by_uid:
            combined = self.results_by_uid[dataset_uid]
            combined["mismatch_review"] = self._collect_mismatch_review()
            combined["second_pass_refinement"] = second_pass
            combined.setdefault("report_tables", {})["phase21_mismatch_review"] = combined["mismatch_review"]
            combined["report_tables"]["phase21_second_pass_excluded_windows"] = second_pass.get("excluded_windows", [])
            self.results_by_uid[dataset_uid] = combined
            self._populate_result(combined)
        self._set_busy(False)
        self.export_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText(
            "100% · Pass 2 complete. "
            f"Excluded {second_pass.get('excluded_point_count', 0)} points; "
            f"fit-domain Rwp {self._fmt(second_pass.get('second_pass_fit_rwp_percent'))}%."
        )

    def _on_refinement_failed(self, message: str, traceback_text: str):
        self._set_busy(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Failed. Review CIFs/settings.")
        QMessageBox.critical(self, "Multiphase search failed", message)

    def _on_refinement_cancelled(self, message: str):
        self._set_busy(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Cancelled. Previous results were preserved.")

    def _clear_worker_handles(self):
        self._worker_thread = None
        self._worker = None
        self._active_dataset_uid = None
        self._active_phases = []
        self._worker_mode = "pass1"

    def _populate_result(self, result: dict):
        self._fill_ranked(result.get("ranked_models", []))
        best = result.get("best_model") or {}
        self._fill_fractions(best.get("phase_fractions", []), result)
        self._fill_rejected(result.get("rejected_phases", []))
        self._fill_candidates(result.get("candidate_phases", []))
        self._fill_mismatch_review(result.get("mismatch_review", []))
        self._fill_second_pass_summary(result.get("second_pass_refinement"))
        self.second_pass_button.setEnabled(self._has_reviewable_result())
        warnings = list(result.get("scientific_warnings", []))
        if best.get("warnings"):
            warnings.extend(best.get("warnings") or [])
        self.warnings_box.setPlainText("\n\n".join(str(item) for item in warnings) if warnings else "No warnings.")

    def _has_reviewable_result(self) -> bool:
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            return False
        result = self.results_by_uid.get(dataset.uid) or {}
        return bool(result.get("best_model")) and self.mismatch_table.rowCount() > 0

    def _fill_mismatch_review(self, rows):
        self.mismatch_table.setRowCount(0)
        artifact_count = 0
        for row_data in rows or []:
            row = self.mismatch_table.rowCount()
            self.mismatch_table.insertRow(row)
            use_item = QTableWidgetItem("")
            use_item.setFlags(use_item.flags() | Qt.ItemIsUserCheckable)
            use_item.setCheckState(Qt.Checked if row_data.get("use_in_second_pass", True) else Qt.Unchecked)
            use_item.setData(Qt.UserRole, dict(row_data))
            self.mismatch_table.setItem(row, 0, use_item)
            values = [
                self._fmt(row_data.get("two_theta_deg")),
                self._fmt(row_data.get("positive_residual")),
                self._fmt(row_data.get("residual_z_score")),
                self._fmt(row_data.get("estimated_half_height_width_deg")),
                row_data.get("nearest_phase", ""),
                self._fmt(row_data.get("nearest_reflection_two_theta_deg")),
                self._fmt(row_data.get("delta_two_theta_deg")),
                row_data.get("classification", ""),
                row_data.get("recommended_action", ""),
                self._fmt(row_data.get("window_min_deg")),
                self._fmt(row_data.get("window_max_deg")),
            ]
            for column, value in enumerate(values, start=1):
                self.mismatch_table.setItem(row, column, QTableWidgetItem(str(value)))
            if row_data.get("auto_exclude_recommended"):
                artifact_count += 1
        self.mismatch_summary_label.setText(
            f"Review windows: {self.mismatch_table.rowCount()} · artifact candidates: {artifact_count}"
        )

    def _collect_mismatch_review(self) -> list[dict]:
        rows = []
        for row in range(self.mismatch_table.rowCount()):
            item = self.mismatch_table.item(row, 0)
            if item is None:
                continue
            data = dict(item.data(Qt.UserRole) or {})
            data["use_in_second_pass"] = item.checkState() == Qt.Checked
            rows.append(data)
        return rows

    def _auto_uncheck_artifacts(self):
        for row in range(self.mismatch_table.rowCount()):
            item = self.mismatch_table.item(row, 0)
            if item is None:
                continue
            data = dict(item.data(Qt.UserRole) or {})
            item.setCheckState(Qt.Unchecked if data.get("auto_exclude_recommended") else Qt.Checked)
        self.status_label.setText("Only isolated artifact candidates were unchecked. Real unexplained peaks remain enabled.")

    def _keep_all_mismatch_windows(self):
        for row in range(self.mismatch_table.rowCount()):
            item = self.mismatch_table.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.Checked)
        self.status_label.setText("All mismatch windows are included in Pass 2.")

    def _fill_second_pass_summary(self, result):
        if not isinstance(result, dict):
            self.second_pass_summary.setPlainText("No Pass 2 result yet.")
            return
        self.second_pass_summary.setPlainText(
            "Pass 2 user-reviewed refinement\n"
            f"Phases: {', '.join(result.get('phase_labels', []))}\n"
            f"Retained points: {result.get('retained_point_count', '—')} / {result.get('input_point_count', '—')}\n"
            f"Excluded points: {result.get('excluded_point_count', '—')} "
            f"({self._fmt(result.get('excluded_point_fraction_percent'))}%)\n"
            f"Fit-domain Rwp: {self._fmt(result.get('second_pass_fit_rwp_percent'))}%\n"
            f"Warning: {result.get('comparison_warning', '')}"
        )

    def _fill_ranked(self, rows):
        self.ranked_table.setRowCount(0)
        for index, row_data in enumerate(rows, start=1):
            row = self.ranked_table.rowCount()
            self.ranked_table.insertRow(row)
            values = [
                index,
                row_data.get("model_id", ""),
                row_data.get("status_label", ""),
                row_data.get("phase_count", ""),
                self._fmt(row_data.get("rwp_percent")),
                self._fmt(row_data.get("rp_percent")),
                self._fmt(row_data.get("r_squared")),
                self._fmt(row_data.get("bic")),
                self._fmt(row_data.get("bic_improvement_vs_background")),
                row_data.get("unexplained_peak_count", ""),
            ]
            self._set_row(self.ranked_table, row, values)

    def _fill_fractions(self, rows, result):
        candidate_by_label = {row.get("label"): row for row in result.get("candidate_phases", [])}
        self.fraction_table.setRowCount(0)
        for row_data in rows:
            row = self.fraction_table.rowCount()
            self.fraction_table.insertRow(row)
            label = row_data.get("phase_name", "")
            candidate = candidate_by_label.get(label, {})
            values = [
                label,
                self._fmt(row_data.get("pattern_fraction_percent")),
                self._fmt(row_data.get("scale")),
                row_data.get("status_label", ""),
                candidate.get("formula", ""),
                candidate.get("required", ""),
                row_data.get("warning", ""),
            ]
            self._set_row(self.fraction_table, row, values)

    def _fill_rejected(self, rows):
        self.rejected_table.setRowCount(0)
        for row_data in rows:
            row = self.rejected_table.rowCount()
            self.rejected_table.insertRow(row)
            self._set_row(
                self.rejected_table,
                row,
                [row_data.get("phase_label", ""), row_data.get("status_label", ""), row_data.get("reason", "")],
            )

    def _fill_candidates(self, rows):
        self.candidate_result_table.setRowCount(0)
        for row_data in rows:
            row = self.candidate_result_table.rowCount()
            self.candidate_result_table.insertRow(row)
            plaus = row_data.get("structure_plausibility") or {}
            values = [
                row_data.get("label", ""),
                row_data.get("formula", ""),
                row_data.get("space_group", ""),
                row_data.get("crystal_system", ""),
                row_data.get("atom_count", ""),
                row_data.get("required", ""),
                plaus.get("status", "") if isinstance(plaus, dict) else "",
                row_data.get("source", ""),
            ]
            self._set_row(self.candidate_result_table, row, values)

    def _set_row(self, table, row, values):
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if isinstance(value, (int, float)):
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, column, item)

    def _fmt(self, value):
        if value is None:
            return "—"
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return str(value)

    def export_result(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None or dataset.uid not in self.results_by_uid:
            QMessageBox.information(self, "No result", "Run the multiphase search first.")
            return
        self.main_window.show_export_center(
            preferred_formats={"excel", "text", "zip"},
            preferred_categories={"project", "pattern", "structure", "refinement", "qpa"},
        )
    def apply_theme(self, theme_name: str):
        return None
