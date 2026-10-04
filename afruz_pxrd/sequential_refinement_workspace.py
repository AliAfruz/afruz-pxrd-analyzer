from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import traceback
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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

from .context_copy import CopyablePlotWidget, install_label_copy_menu, install_table_copy_menu
from .candidate_queue import can_advance, candidate_run_label, comparison_domain_signature, pack_snapshots, parent_seed, unpack_snapshots, warm_start_settings
from .crystallography import calculate_powder_pattern, load_cif
from .exclusion_regions import ExclusionRegion, attach_exclusion_audit, filter_arrays, inclusion_mask, merge_regions
from .gpu_backend import cuda_status
from .phase_identification import reference_from_cif_pattern
from .refinement_export import export_refinement_txt_bundle
from .rietveld_refinement import RietveldPhaseSpec, refine_rietveld
from .text_export import safe_filename, write_columns_txt, write_mapping_txt, write_table_txt
from .theme import DEFAULT_THEME_NAME, THEMES
from .whole_pattern_refinement import WholePatternPhaseSpec, refine_whole_pattern
from .widgets import NoWheelDoubleSpinBox, NoWheelSpinBox


METHOD_LABELS = {
    "pawley": "Pawley decomposition",
    "lebail": "Le Bail extraction",
    "rietveld": "Rietveld refinement",
}


def refinement_run_id(counter: int, method: str) -> str:
    """Return a compact, human-readable ID that stays unique in one project."""

    code = {"pawley": "PW", "lebail": "LB", "rietveld": "RT"}.get(str(method), "RF")
    return f"AFR-{code}-{max(1, int(counter)):05d}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _position_of_reflection(row: dict) -> float | None:
    for key in ("two_theta_deg", "two_theta"):
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def annotate_reflection_activity(reflections: list[dict], regions) -> tuple[list[dict], dict]:
    """Mark CIF reflections inside excluded fit windows without deleting hkl physics."""

    merged = merge_regions(regions or [])
    output = []
    inactive = 0
    for source in reflections or []:
        row = dict(source)
        position = _position_of_reflection(row)
        matching = [region for region in merged if position is not None and region.start_deg <= position <= region.end_deg]
        row["fit_active"] = not bool(matching)
        row["exclusion_label"] = " | ".join(region.label for region in matching)
        if matching:
            inactive += 1
        output.append(row)
    return output, {
        "total_reflection_count": len(output),
        "fit_active_reflection_count": len(output) - inactive,
        "fit_inactive_reflection_count": inactive,
        "rule": (
            "Reflections whose centers lie inside excluded 2theta windows are marked inactive for fit-domain "
            "display and audit. Structure factors are not physically deleted; measured and calculated profile "
            "points in those windows are omitted from the objective."
        ),
    }


class SequentialRefinementWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, method: str, x, y, structure: dict, settings: dict, regions, parent=None):
        super().__init__(parent)
        self.method = str(method)
        self.x = np.asarray(x, dtype=float).copy()
        self.y = np.asarray(y, dtype=float).copy()
        self.structure = deepcopy(structure)
        self.settings = deepcopy(settings)
        self.regions = [row.to_dict() for row in merge_regions(regions or [])]
        self._cancel_event = threading.Event()

    def request_cancel(self):
        self._cancel_event.set()

    def _emit_progress(self, done, total, message):
        self.progress.emit(int(done), max(1, int(total)), str(message))

    @Slot()
    def run(self):
        try:
            minimum = 80 if self.method == "rietveld" else 50
            x_fit, arrays, audit = filter_arrays(
                self.x,
                (self.y,),
                self.regions,
                minimum_points=minimum,
            )
            y_fit = arrays[0]
            if self._cancel_event.is_set():
                self.cancelled.emit()
                return
            common = {
                "wavelength_angstrom": float(self.settings["wavelength_angstrom"]),
                "two_theta_min": float(self.settings["two_theta_min"]),
                "two_theta_max": float(self.settings["two_theta_max"]),
                "intensity_cutoff_percent": float(self.settings["intensity_cutoff_percent"]),
                "background_order": int(self.settings["background_order"]),
                "weighting": str(self.settings["weighting"]),
                "refine_zero_shift": bool(self.settings["refine_zero_shift"]),
                "initial_zero_shift": float(self.settings.get("initial_zero_shift", 0.0)),
                "refine_profile": bool(self.settings["refine_profile"]),
                "refine_eta": False,
                "initial_u": float(self.settings.get("initial_u", 0.005)),
                "initial_v": float(self.settings.get("initial_v", 0.0)),
                "initial_w": float(self.settings.get("initial_w", 0.02)),
                "initial_eta": float(self.settings.get("initial_eta", 0.5)),
                "profile_model": "Pseudo-Voigt U-V-W",
                "refine_lorentzian_width": False,
                "refine_asymmetry": False,
                "cell_tolerance_percent": float(self.settings["cell_tolerance_percent"]),
                "maximum_nonlinear_evaluations": int(self.settings["maximum_evaluations"]),
                "maximum_optimization_points": int(self.settings["optimization_points"]),
                "use_staged_refinement": bool(self.settings["use_staged_refinement"]),
                "use_gpu": bool(self.settings["use_gpu"]),
                "progress_callback": self._emit_progress,
                "cancel_check": self._cancel_event.is_set,
            }
            if self.method in {"pawley", "lebail"}:
                self._emit_progress(0, 1, "Calculating the selected CIF reflection set...")
                pattern = calculate_powder_pattern(
                    self.structure,
                    wavelength_angstrom=common["wavelength_angstrom"],
                    two_theta_min=common["two_theta_min"],
                    two_theta_max=common["two_theta_max"],
                    intensity_cutoff_percent=common["intensity_cutoff_percent"],
                    use_gpu=common["use_gpu"],
                )
                reference = reference_from_cif_pattern(
                    self.structure,
                    pattern,
                    common["wavelength_angstrom"],
                )
                if reference is None:
                    raise ValueError("The selected CIF generated no usable reflections in the chosen range.")
                result = refine_whole_pattern(
                    x_fit,
                    y_fit,
                    [WholePatternPhaseSpec(reference, refine_cell=bool(self.settings["refine_cell"]))],
                    mode=("Pawley decomposition" if self.method == "pawley" else "Le Bail-style extraction"),
                    extraction_cycles=int(self.settings["extraction_cycles"]),
                    robust_loss="soft_l1",
                    **common,
                )
            elif self.method == "rietveld":
                result = refine_rietveld(
                    x_fit,
                    y_fit,
                    [
                        RietveldPhaseSpec(
                            deepcopy(self.structure),
                            name=str(self.structure.get("data_name") or self.structure.get("formula") or "CIF phase"),
                            refine_cell=bool(self.settings["refine_cell"]),
                            refine_biso=False,
                            freeze_structure_factors=bool(self.settings["freeze_structure_factors"]),
                            included=True,
                            allow_flagged_structure=bool(self.settings.get("allow_flagged_structure", False)),
                        )
                    ],
                    robust_loss="soft_l1",
                    generated_pattern_scaling="Joint weighted scale + background",
                    validation_stride=int(self.settings.get("validation_stride", 0)),
                    **common,
                )
            else:
                raise ValueError(f"Unsupported sequential-refinement method: {self.method}")

            audited = attach_exclusion_audit(result, audit, method=METHOD_LABELS[self.method])
            reflections, reflection_audit = annotate_reflection_activity(
                list(audited.get("reflections", [])),
                self.regions,
            )
            audited["reflections"] = reflections
            audited["cif_reflection_fit_domain"] = reflection_audit
            audited["sequential_refinement"] = {
                "method_key": self.method,
                "input_kind": "raw measured pattern" if not self.settings.get("use_processed") else "processed pattern",
                "objective_domain_rule": (
                    "Only measured/calculated profile points outside the recorded closed exclusion intervals "
                    "contribute to the objective and reported fit-domain statistics."
                ),
            }
        except Exception:
            if self._cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(audited)


class SequentialExclusionEditor(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Next refinement: labelled exclusions", parent)
        self._regions_by_uid: dict[str, list[dict]] = {}
        self._uid: str | None = None
        self._plot = None
        self._selector = pg.LinearRegionItem(
            values=(20.0, 20.3),
            movable=True,
            brush=pg.mkBrush(242, 183, 70, 48),
            pen=pg.mkPen(242, 183, 70, 225, width=1.5),
        )
        self._selector.setZValue(40)
        self._bands: list[object] = []
        self._build_ui()

    @staticmethod
    def _spin(value: float):
        spin = NoWheelDoubleSpinBox()
        spin.setRange(0.0, 180.0)
        spin.setDecimals(5)
        spin.setSingleStep(0.02)
        spin.setValue(value)
        return spin

    def _build_ui(self):
        layout = QVBoxLayout(self)
        controls = QGridLayout()
        self.start = self._spin(20.0)
        self.end = self._spin(20.3)
        self.half_width = self._spin(0.12)
        self.half_width.setRange(0.001, 10.0)
        self.label = QLineEdit("Suspected impurity peak")
        self.add_region = QPushButton("Exclude labelled region in next run")
        self.add_peak = QPushButton("Exclude peak at selector center")
        self.remove = QPushButton("Remove selected")
        self.clear = QPushButton("Clear next-run exclusions")
        controls.addWidget(QLabel("Start 2theta"), 0, 0)
        controls.addWidget(self.start, 0, 1)
        controls.addWidget(QLabel("End 2theta"), 0, 2)
        controls.addWidget(self.end, 0, 3)
        controls.addWidget(QLabel("Peak half-width"), 1, 0)
        controls.addWidget(self.half_width, 1, 1)
        controls.addWidget(QLabel("Scientific label"), 1, 2)
        controls.addWidget(self.label, 1, 3)
        controls.addWidget(self.add_region, 2, 0, 1, 2)
        controls.addWidget(self.add_peak, 2, 2, 1, 2)
        controls.addWidget(self.remove, 3, 2)
        controls.addWidget(self.clear, 3, 3)
        layout.addLayout(controls)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Start", "End", "Width", "Label / hypothesis"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setMaximumHeight(120)
        install_table_copy_menu(self.table)
        layout.addWidget(self.table)
        self.note = QLabel(
            "Drag the amber selector or click the main plot to center it. These exclusions apply only to the next run."
        )
        self.note.setObjectName("mutedLabel")
        self.note.setWordWrap(True)
        install_label_copy_menu(self.note)
        layout.addWidget(self.note)
        self.start.valueChanged.connect(self._spins_changed)
        self.end.valueChanged.connect(self._spins_changed)
        self._selector.sigRegionChanged.connect(self._selector_changed)
        self.add_region.clicked.connect(self.add_selected_region)
        self.add_peak.clicked.connect(self.add_peak_region)
        self.remove.clicked.connect(self.remove_selected)
        self.clear.clicked.connect(self.clear_regions)

    def bind_plot(self, plot):
        self._plot = plot
        try:
            plot.removeItem(self._selector)
        except Exception:
            pass
        plot.addItem(self._selector, ignoreBounds=True)
        self._redraw_bands()

    def set_dataset(self, uid: str | None, x=None):
        self._uid = None if uid is None else str(uid)
        values = np.asarray([] if x is None else x, dtype=float)
        if values.size:
            low, high = float(np.min(values)), float(np.max(values))
            self.start.setRange(low, high)
            self.end.setRange(low, high)
            width = max(0.05, 0.004 * (high - low))
            center = low + 0.35 * (high - low)
            self.set_selector(center - width, center + width)
        self._populate()

    def set_selector(self, start: float, end: float):
        start, end = sorted((float(start), float(end)))
        self.start.blockSignals(True)
        self.end.blockSignals(True)
        self.start.setValue(start)
        self.end.setValue(end)
        self.start.blockSignals(False)
        self.end.blockSignals(False)
        self._selector.setRegion((start, end))

    def center_selector(self, center: float):
        width = max(0.002, float(self.end.value() - self.start.value()))
        self.set_selector(float(center) - width / 2.0, float(center) + width / 2.0)

    def _spins_changed(self, *_args):
        self._selector.setRegion(tuple(sorted((self.start.value(), self.end.value()))))

    def _selector_changed(self):
        start, end = self._selector.getRegion()
        self.start.blockSignals(True)
        self.end.blockSignals(True)
        self.start.setValue(float(start))
        self.end.setValue(float(end))
        self.start.blockSignals(False)
        self.end.blockSignals(False)

    def regions(self) -> list[ExclusionRegion]:
        if self._uid is None:
            return []
        return merge_regions(self._regions_by_uid.get(self._uid, []))

    def set_regions(self, rows):
        if self._uid is None:
            return
        self._regions_by_uid[self._uid] = [row.to_dict() for row in merge_regions(rows or [])]
        self._populate()
        self.changed.emit()

    def _append(self, start: float, end: float):
        if self._uid is None:
            return
        try:
            rows = merge_regions(
                [*self.regions(), ExclusionRegion(start, end, self.label.text().strip() or "Excluded feature")]
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid exclusion", str(exc))
            return
        self._regions_by_uid[self._uid] = [row.to_dict() for row in rows]
        self._populate()
        self.changed.emit()

    def add_selected_region(self):
        self._append(self.start.value(), self.end.value())

    def add_peak_region(self):
        center = 0.5 * (self.start.value() + self.end.value())
        half = self.half_width.value()
        self._append(center - half, center + half)

    def remove_selected(self):
        if self._uid is None:
            return
        rows = [row.to_dict() for row in self.regions()]
        index = self.table.currentRow()
        if 0 <= index < len(rows):
            rows.pop(index)
            self._regions_by_uid[self._uid] = rows
            self._populate()
            self.changed.emit()

    def clear_regions(self):
        if self._uid is not None:
            self._regions_by_uid[self._uid] = []
            self._populate()
            self.changed.emit()

    def _populate(self):
        rows = self.regions()
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = (row.start_deg, row.end_deg, row.end_deg - row.start_deg, row.label)
            for column, value in enumerate(values):
                text = f"{value:.6g}" if isinstance(value, float) else str(value)
                self.table.setItem(index, column, QTableWidgetItem(text))
        self.note.setText(
            f"{len(rows)} merged interval(s) prepared for the next run. Drag the amber selector or click the main plot to reposition it."
        )
        self._redraw_bands()

    def _redraw_bands(self):
        if self._plot is None:
            return
        for item in self._bands:
            try:
                self._plot.removeItem(item)
            except Exception:
                pass
        self._bands = []
        for region in self.regions():
            item = pg.LinearRegionItem(
                values=(region.start_deg, region.end_deg),
                movable=False,
                brush=pg.mkBrush(235, 87, 87, 38),
                pen=pg.mkPen(235, 87, 87, 180, width=1.0),
            )
            item.setZValue(-12)
            self._plot.addItem(item)
            self._bands.append(item)
        try:
            self._plot.removeItem(self._selector)
        except Exception:
            pass
        self._plot.addItem(self._selector, ignoreBounds=True)

    def get_state(self):
        return {"regions_by_uid": deepcopy(self._regions_by_uid)}

    def set_state(self, state):
        rows = (state or {}).get("regions_by_uid", {})
        self._regions_by_uid = {}
        if isinstance(rows, dict):
            for uid, values in rows.items():
                try:
                    self._regions_by_uid[str(uid)] = [row.to_dict() for row in merge_regions(values or [])]
                except (TypeError, ValueError):
                    self._regions_by_uid[str(uid)] = []
        self._populate()


class SequentialRefinementWorkspace(QWidget):
    """Central, lineage-aware refinement laboratory built on validated engines."""

    runCompleted = Signal(str)

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.structures: list[dict] = []
        self.history_by_uid: dict[str, list[dict]] = {}
        self.next_run_counter = 1
        self.steps_by_uid: dict[str, list[dict]] = {}
        self.selected_step_by_uid: dict[str, str] = {}
        self.checked_runs_by_uid: dict[str, list[str]] = {}
        self._queue_running = False
        self._queue_scope: tuple[str, str] | None = None
        self._step_switching = False
        self._thread: QThread | None = None
        self._worker: SequentialRefinementWorker | None = None
        self._active_run_id: str | None = None
        self._active_dataset_uid: str | None = None
        self._selected_history_run_id: str | None = None
        self._last_dataset_uid: str | None = None
        self._plot_items: list[object] = []
        self.current_theme_name = DEFAULT_THEME_NAME
        self._build_ui()
        self.refresh_for_selected_dataset()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.dataset_label = QLabel("No experimental pattern selected")
        self.dataset_label.setObjectName("mutedLabel")
        self.dataset_label.setMinimumWidth(0)
        self.dataset_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        install_label_copy_menu(self.dataset_label)
        self.cif_combo = QComboBox()
        self.cif_combo.setMinimumWidth(260)
        self.add_cif = QPushButton("Add CIF(s)...")
        self.add_active_cif = QPushButton("Add active CIF")
        self.remove_cif = QPushButton("Remove CIF")
        self.use_processed = QCheckBox("Use processed pattern")
        self.use_processed.setChecked(False)
        self.use_gpu = QCheckBox("Use GPU where supported")
        status = cuda_status()
        self.use_gpu.setChecked(bool(status.get("available")))
        self.use_gpu.setToolTip(str(status.get("reason") or status.get("device_name") or "GPU status unavailable"))
        top.addWidget(QLabel("Dataset"))
        top.addWidget(self.dataset_label, 1)
        top.addWidget(QLabel("Active CIF"))
        top.addWidget(self.cif_combo)
        top.addWidget(self.add_cif)
        top.addWidget(self.add_active_cif)
        top.addWidget(self.remove_cif)
        top.addWidget(self.use_processed)
        top.addWidget(self.use_gpu)
        layout.addLayout(top)

        run_row = QHBoxLayout()
        self.pawley_button = QPushButton("1  Run Pawley")
        self.lebail_button = QPushButton("2  Run Le Bail")
        self.rietveld_button = QPushButton("3  Run Rietveld")
        for button in (self.pawley_button, self.lebail_button, self.rietveld_button):
            button.setMinimumHeight(46)
            button.setObjectName("primaryButton")
            run_row.addWidget(button)
        self.cancel_button = QPushButton("Cancel current run")
        self.cancel_button.setEnabled(False)
        self.export_button = QPushButton("Full history export (.zip)")
        self.run_label = QLineEdit()
        self.run_label.setPlaceholderText("Optional run label / hypothesis")
        run_row.addWidget(self.cancel_button)
        run_row.addWidget(self.export_button)
        run_row.addWidget(self.run_label, 1)
        layout.addLayout(run_row)

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label = QLabel("Ready. Raw measured intensity is the default input.")
        self.progress_label.setObjectName("mutedLabel")
        # Long user labels must not enlarge the entire central workspace.
        self.progress_label.setMinimumWidth(0)
        self.progress_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        install_label_copy_menu(self.progress_label)
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.progress_label, 2)
        layout.addLayout(progress_row)

        splitter = QSplitter(Qt.Horizontal)
        center = QSplitter(Qt.Vertical)
        self.plot = CopyablePlotWidget(background=THEMES[DEFAULT_THEME_NAME]["plot_background"], copy_title="Sequential refinement history")
        self.plot.showGrid(x=True, y=True, alpha=THEMES[DEFAULT_THEME_NAME]["grid_alpha"])
        self.plot.setLabel("bottom", "2theta", units="degrees")
        self.plot.setLabel("left", "Intensity")
        self.legend = self.plot.addLegend()
        center.addWidget(self.plot)
        self.detail_tabs = QTabWidget()
        self.exclusion_editor = SequentialExclusionEditor()
        self.detail_tabs.addTab(self.exclusion_editor, "Next-run exclusions")
        self.detail_tabs.addTab(self._settings_panel(), "Refinement controls")
        self.detail_tabs.addTab(self._cif_library_panel(), "CIF library")
        self.detail_tabs.addTab(self._reflection_panel(), "Active/inactive CIF reflections")
        self.detail_tabs.addTab(self._queue_panel(), "Step-wise queue")
        center.addWidget(self.detail_tabs)
        center.setSizes([610, 260])
        center.setStretchFactor(0, 4)
        center.setStretchFactor(1, 2)
        splitter.addWidget(center)
        splitter.addWidget(self._history_panel())
        splitter.setSizes([1150, 540])
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        self.exclusion_editor.bind_plot(self.plot)

        boundary = QLabel(
            "Scientific boundary: masking is a fit-domain decision, not deletion of crystallographic hkl reflections. "
            "Each run permanently records its CIF, method, parameters, parent ID, excluded regions and inactive-reflection audit."
        )
        boundary.setWordWrap(True)
        boundary.setStyleSheet("color: #e7b84b; font-weight: 600;")
        install_label_copy_menu(boundary)
        layout.addWidget(boundary)

        self.add_cif.clicked.connect(self.add_cif_files)
        self.add_active_cif.clicked.connect(self.add_active_reference_cif)
        self.remove_cif.clicked.connect(self.remove_active_cif)
        self.cif_combo.currentIndexChanged.connect(self._active_cif_changed)
        self.cif_table.itemSelectionChanged.connect(self._cif_table_selected)
        self.pawley_button.clicked.connect(lambda: self.run_method("pawley"))
        self.lebail_button.clicked.connect(lambda: self.run_method("lebail"))
        self.rietveld_button.clicked.connect(lambda: self.run_method("rietveld"))
        self.cancel_button.clicked.connect(self.cancel_current_run)
        self.export_button.clicked.connect(self.export_full_history)
        self.history_table.itemSelectionChanged.connect(self._history_selection_changed)
        self.history_table.itemChanged.connect(self._history_item_changed)
        self.continue_button.clicked.connect(self.continue_from_selected)
        self.show_latest_button.clicked.connect(self.select_latest_run)
        self.exclusion_editor.changed.connect(self._draft_exclusions_changed)
        self.plot.scene().sigMouseClicked.connect(self._plot_clicked)

    def _queue_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.step_selector = QComboBox()
        self.step_name = QLineEdit()
        self.queue_method = QComboBox()
        for method, label in METHOD_LABELS.items():
            self.queue_method.addItem(label, method)
        self.queue_method.setCurrentIndex(2)
        form.addRow("Review / run step", self.step_selector)
        form.addRow("Step label", self.step_name)
        form.addRow("Queue method", self.queue_method)
        layout.addLayout(form)
        self.common_domain = QCheckBox("Same range and exclusions for all candidates")
        self.common_domain.setChecked(True)
        self.common_domain.setToolTip(
            "Checked: every run gets the current range and labelled exclusions at queue creation. "
            "Unchecked: each child inherits its own parent's range and exclusions. Other settings are shared."
        )
        self.continue_after_failure = QCheckBox("Continue queue after a failed candidate")
        self.continue_after_failure.setChecked(True)
        layout.addWidget(self.common_domain)
        layout.addWidget(self.continue_after_failure)
        self.candidate_table = QTableWidget(0, 6)
        self.candidate_table.setHorizontalHeaderLabels(["Run", "Order", "CIF", "Candidate label", "Parent ID", "Latest status"])
        self.candidate_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.candidate_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.candidate_table.setMinimumHeight(160)
        self.candidate_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.candidate_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        install_table_copy_menu(self.candidate_table)
        layout.addWidget(self.candidate_table, 1)
        buttons = QGridLayout()
        self.queue_checked_button = QPushButton("Queue checked candidates")
        self.start_queue_button = QPushButton("Start / resume this step")
        self.start_queue_button.setObjectName("primaryButton")
        self.pause_queue_button = QPushButton("Pause after current run")
        self.next_step_button = QPushButton("Next step from checked results")
        self.move_candidate_up = QPushButton("Move candidate up")
        self.move_candidate_down = QPushButton("Move candidate down")
        self.discard_queue_button = QPushButton("Discard unstarted runs in this step")
        for index, button in enumerate((self.queue_checked_button, self.start_queue_button,
                                        self.pause_queue_button, self.next_step_button,
                                        self.move_candidate_up, self.move_candidate_down)):
            buttons.addWidget(button, index // 2, index % 2)
        buttons.addWidget(self.discard_queue_button, 3, 0, 1, 2)
        layout.addLayout(buttons)
        self.queue_summary = QLabel("Step 1: check the CIF candidates, queue them, then start.")
        self.queue_summary.setWordWrap(True)
        self.queue_summary.setObjectName("mutedLabel")
        install_label_copy_menu(self.queue_summary)
        layout.addWidget(self.queue_summary)
        note = QLabel(
            "Each run fits ONE CIF against the original measurement, not a subtracted residual. "
            "Use the Next checkboxes in history to choose completed results for Step 2, Step 3, etc. "
            "Only compare R factors directly when the input and fit domains match."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        layout.addWidget(note)
        scroll.setWidget(panel)
        self.step_selector.currentIndexChanged.connect(self._step_selected)
        self.step_name.editingFinished.connect(self._step_name_changed)
        self.candidate_table.itemChanged.connect(self._candidate_item_changed)
        self.queue_checked_button.clicked.connect(self.queue_checked_candidates)
        self.start_queue_button.clicked.connect(self.start_queue)
        self.pause_queue_button.clicked.connect(self.pause_queue)
        self.next_step_button.clicked.connect(self.prepare_next_step)
        self.move_candidate_up.clicked.connect(lambda: self._move_candidate(-1))
        self.move_candidate_down.clicked.connect(lambda: self._move_candidate(1))
        self.discard_queue_button.clicked.connect(self.discard_pending_queue)
        for control in (self.common_domain, self.continue_after_failure):
            control.toggled.connect(self._queue_preferences_changed)
        self.queue_method.currentIndexChanged.connect(self._queue_preferences_changed)
        return scroll

    def _current_step(self, uid: str | None = None):
        dataset = self._current_dataset()
        uid = str(uid or (dataset.uid if dataset is not None else ""))
        selected = self.selected_step_by_uid.get(uid)
        return next((row for row in self.steps_by_uid.get(uid, []) if row.get("step_id") == selected), None)

    @staticmethod
    def _candidate(structure: dict, parent=None):
        return {
            "candidate_uid": uuid4().hex[:20],
            "cif_uid": str(structure.get("_sequential_refinement_uid") or ""),
            "cif_name": str(structure.get("data_name") or structure.get("formula") or "CIF candidate"),
            "label": str((parent or {}).get("candidate_label") or structure.get("data_name") or "CIF candidate"),
            "parent_run_id": (parent or {}).get("run_id"), "enabled": True,
            "structure_snapshot": deepcopy(structure) if parent else None,
        }

    def _ensure_steps(self, uid: str):
        rows = self.steps_by_uid.setdefault(str(uid), [])
        if not rows:
            rows.append({"step_id": "S01", "number": 1, "name": "Compare CIF candidates",
                         "candidates": [], "method_key": "rietveld", "common_domain": True,
                         "continue_after_failure": True})
        first = rows[0]
        known = {candidate.get("cif_uid") for candidate in first.get("candidates", [])}
        for structure in self.structures:
            if structure.get("_sequential_refinement_uid") not in known:
                first.setdefault("candidates", []).append(self._candidate(structure))
        if self.selected_step_by_uid.get(str(uid)) not in {row["step_id"] for row in rows}:
            self.selected_step_by_uid[str(uid)] = rows[0]["step_id"]

    def _capture_step_draft(self, uid: str | None = None):
        if self._step_switching:
            return
        step = self._current_step(uid)
        if step is None:
            return
        step["draft_settings"] = deepcopy(self._settings())
        step["draft_exclusions"] = [row.to_dict() for row in self.exclusion_editor.regions()]
        step["method_key"] = self.queue_method.currentData() or "rietveld"
        step["common_domain"] = self.common_domain.isChecked()
        step["continue_after_failure"] = self.continue_after_failure.isChecked()

    def _populate_steps(self):
        dataset = self._current_dataset()
        self.step_selector.blockSignals(True)
        self.step_selector.clear()
        if dataset is not None:
            self._ensure_steps(str(dataset.uid))
            for step in self.steps_by_uid[str(dataset.uid)]:
                self.step_selector.addItem(f"Step {step['number']} — {step['name']}", step["step_id"])
            self.step_selector.setCurrentIndex(self.step_selector.findData(self.selected_step_by_uid[str(dataset.uid)]))
        self.step_selector.blockSignals(False)
        self._populate_candidates()

    def _populate_candidates(self):
        step = self._current_step()
        self.candidate_table.blockSignals(True)
        candidates = (step or {}).get("candidates", [])
        self.candidate_table.setRowCount(len(candidates))
        for index, candidate in enumerate(candidates):
            records = [row for row in self._history() if row.get("candidate_uid") == candidate["candidate_uid"]]
            values = ("", index + 1, candidate.get("cif_name"), candidate.get("label"),
                      candidate.get("parent_run_id") or "original CIF", records[-1]["status"] if records else "Not queued")
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value or ""))
                item.setData(Qt.UserRole, candidate["candidate_uid"])
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if column == 0:
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked if candidate.get("enabled", True) else Qt.Unchecked)
                if column == 3:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.candidate_table.setItem(index, column, item)
        self.candidate_table.blockSignals(False)
        if step:
            self.step_name.setText(str(step["name"]))
        pending = sum(row.get("status") == "Queued" and row.get("step_id") == (step or {}).get("step_id") for row in self._history())
        self.queue_summary.setText(
            f"Step {(step or {}).get('number', 1)} · {len(candidates)} candidate(s) · {pending} queued. "
            + ("Queue running: only one numerical worker runs at a time." if self._queue_running else "Queue paused / ready; start explicitly to run.")
        )
        editable = not self.is_running() and pending == 0
        self.candidate_table.setEnabled(editable)
        self.move_candidate_up.setEnabled(editable)
        self.move_candidate_down.setEnabled(editable)
        self.discard_queue_button.setEnabled(pending > 0 and not self.is_running())

    def _step_selected(self, *_args):
        dataset = self._current_dataset()
        if dataset is None or self._step_switching:
            return
        self._capture_step_draft()
        self.selected_step_by_uid[str(dataset.uid)] = str(self.step_selector.currentData())
        self._apply_step_controls()
        self._populate_candidates()
        self._mark_dirty()

    def _apply_step_controls(self):
        step = self._current_step()
        if step is None:
            return
        was_switching = self._step_switching
        self._step_switching = True
        try:
            self._set_settings(step.get("draft_settings") or {})
            if "draft_exclusions" in step:
                self.exclusion_editor.set_regions(step["draft_exclusions"])
            self.common_domain.setChecked(bool(step.get("common_domain", True)))
            self.continue_after_failure.setChecked(bool(step.get("continue_after_failure", True)))
            self.queue_method.setCurrentIndex(max(0, self.queue_method.findData(step.get("method_key", "rietveld"))))
        finally:
            self._step_switching = was_switching

    def _step_name_changed(self):
        step = self._current_step()
        if step:
            step["name"] = self.step_name.text().strip() or f"Step {step['number']}"
            self._populate_steps()
            self._mark_dirty()

    def _queue_preferences_changed(self, *_args):
        if not self._step_switching:
            self._capture_step_draft()
            self._mark_dirty()

    def _candidate_item_changed(self, item):
        step = self._current_step()
        if not step or not 0 <= item.row() < len(step["candidates"]):
            return
        candidate = step["candidates"][item.row()]
        if item.column() == 0:
            candidate["enabled"] = item.checkState() == Qt.Checked
        elif item.column() == 3:
            candidate["label"] = item.text().strip() or candidate["cif_name"]
        self._mark_dirty()

    def _move_candidate(self, delta):
        step = self._current_step()
        index = self.candidate_table.currentRow()
        target = index + delta
        if step and 0 <= index < len(step["candidates"]) and 0 <= target < len(step["candidates"]):
            step["candidates"][index], step["candidates"][target] = step["candidates"][target], step["candidates"][index]
            self._populate_candidates()
            self.candidate_table.selectRow(target)
            self._mark_dirty()

    def prepare_next_step(self):
        dataset = self._current_dataset()
        if dataset is None or self.is_running():
            return
        selected = set(self.checked_runs_by_uid.get(str(dataset.uid), []))
        parents = [row for row in self._history() if row.get("run_id") in selected and can_advance(row)]
        if not parents:
            QMessageBox.information(self, "Choose results", "Check Next beside completed runs in the history table first.")
            return
        candidates = []
        try:
            for parent in parents:
                structure = parent.get("structure_snapshot") or next(
                    (row for row in self.structures if row.get("_sequential_refinement_uid") == parent.get("cif_uid")), None
                )
                if not structure:
                    raise ValueError(f"Reimport the CIF for {parent['run_id']}; this legacy result has no model snapshot.")
                parent_seed(structure, parent)  # Validate before making any new step.
                candidates.append(self._candidate(structure, parent))
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot advance candidate", str(exc))
            return
        self._capture_step_draft()
        number = max(row["number"] for row in self.steps_by_uid[str(dataset.uid)]) + 1
        step = {"step_id": f"S{number:02d}", "number": number, "name": "Selected candidate follow-up",
                "candidates": candidates, "draft_settings": deepcopy(self._settings()),
                "draft_exclusions": [row.to_dict() for row in self.exclusion_editor.regions()],
                "method_key": self.queue_method.currentData(), "common_domain": True,
                "continue_after_failure": self.continue_after_failure.isChecked()}
        self.steps_by_uid[str(dataset.uid)].append(step)
        self.selected_step_by_uid[str(dataset.uid)] = step["step_id"]
        self.checked_runs_by_uid[str(dataset.uid)] = []
        self.common_domain.setChecked(True)
        self._populate_steps()
        self._populate_history()
        self._mark_dirty()
        return step

    def queue_checked_candidates(self):
        dataset, step = self._current_dataset(), self._current_step()
        if dataset is None or step is None or self.is_running():
            return []
        settings = self._settings()
        if settings["two_theta_min"] >= settings["two_theta_max"]:
            QMessageBox.warning(self, "Invalid range", "The minimum 2theta must be lower than the maximum.")
            return []
        method = str(self.queue_method.currentData())
        prepared = []
        already_pending = {row.get("candidate_uid") for row in self._history() if row.get("status") in {"Queued", "Running"}}
        try:
            for index, candidate in enumerate(step["candidates"]):
                if not candidate.get("enabled", True) or candidate["candidate_uid"] in already_pending:
                    continue
                parent = self._record_by_id(candidate.get("parent_run_id"))
                if candidate.get("parent_run_id") and parent is None:
                    raise ValueError(f"Parent record is missing for {candidate['label']}; choose an available completed result.")
                structure = candidate.get("structure_snapshot") or next(
                    (row for row in self.structures if row.get("_sequential_refinement_uid") == candidate.get("cif_uid")), None
                )
                if structure is None:
                    raise ValueError(f"CIF not available: {candidate['cif_name']}. Reimport it before queuing.")
                seed, seed_audit = parent_seed(structure, parent)
                current_settings = warm_start_settings(settings, parent)
                regions = [row.to_dict() for row in self.exclusion_editor.regions()]
                if parent and not self.common_domain.isChecked():
                    for key in ("two_theta_min", "two_theta_max"):
                        current_settings[key] = parent["settings"][key]
                    regions = deepcopy(parent.get("excluded_regions") or [])
                label = candidate_run_label(step["number"], index + 1, candidate["label"], method)
                if self.run_label.text().strip():
                    label += f" · {self.run_label.text().strip()}"
                metadata = {"step_id": step["step_id"], "step_number": step["number"], "step_name": step["name"],
                            "candidate_uid": candidate["candidate_uid"], "candidate_order": index + 1,
                            "candidate_label": candidate["label"], "common_domain": self.common_domain.isChecked(),
                            "continue_after_failure": self.continue_after_failure.isChecked(), "structure_seed_audit": seed_audit}
                prepared.append(self._make_record(method, dataset, seed, current_settings, regions, label,
                                                  candidate.get("parent_run_id"), metadata))
        except (ValueError, TypeError, KeyError) as exc:
            QMessageBox.warning(self, "Queue preparation failed", str(exc))
            return []
        if not prepared:
            QMessageBox.information(self, "No new candidates", "Check candidate rows. Already queued candidates are not duplicated.")
            return []
        self._history().extend(prepared)
        self._capture_step_draft()
        self._populate_history()
        self._populate_candidates()
        self._mark_dirty()
        return prepared

    def start_queue(self):
        dataset, step = self._current_dataset(), self._current_step()
        if dataset is None or step is None or self.is_running():
            return
        self._queue_scope = (str(dataset.uid), str(step["step_id"]))
        if not self._pending_queue():
            QMessageBox.information(self, "No queued runs", "Queue checked candidates for this step first.")
            return
        self._queue_running = True
        self._set_busy(True)
        self._dispatch_next_queued()

    def _pending_queue(self):
        if not self._queue_scope:
            return []
        uid, step_id = self._queue_scope
        return sorted(
            [row for row in self.history_by_uid.get(uid, []) if row.get("step_id") == step_id and row.get("status") == "Queued"],
            key=lambda row: row.get("candidate_order", 0),
        )

    def _dispatch_next_queued(self):
        if not self._queue_running or self._thread is not None:
            return
        pending = self._pending_queue()
        if not pending:
            self._queue_running = False
            self._set_busy(False)
            self._populate_candidates()
            self.progress_label.setText("Step queue finished. Check completed results to prepare the next step.")
            self._mark_dirty()
            return
        try:
            self._launch_record(pending[0])
        except (ValueError, TypeError, KeyError) as exc:
            record = pending[0]
            record.update(status="Failed", error=f"Saved queue could not start: {exc}", completed_utc=_utc_now())
            if not record.get("continue_after_failure", True):
                self._queue_running = False
            self._populate_history()
            self._populate_candidates()
            self._mark_dirty()
            if self._queue_running:
                QTimer.singleShot(0, self._dispatch_next_queued)
            else:
                self._set_busy(False)

    def discard_pending_queue(self):
        if self.is_running():
            return
        step = self._current_step() or {}
        for record in self._history():
            if record.get("step_id") == step.get("step_id") and record.get("status") == "Queued":
                record.update(status="Discarded before start", completed_utc=_utc_now())
        self._populate_history()
        self._populate_candidates()
        self._mark_dirty()

    def pause_queue(self):
        self._queue_running = False
        self._set_busy(self._thread is not None)
        self._populate_candidates()
        self.progress_label.setText("Queue paused: the current run can finish; unstarted runs remain queued.")
        self._mark_dirty()

    def _history_item_changed(self, item):
        if item.column() != 0:
            return
        dataset = self._current_dataset()
        run_id = str(item.data(Qt.UserRole))
        record = self._record_by_id(run_id)
        if dataset is None or record is None or not can_advance(record):
            return
        selected = set(self.checked_runs_by_uid.get(str(dataset.uid), []))
        selected.add(run_id) if item.checkState() == Qt.Checked else selected.discard(run_id)
        self.checked_runs_by_uid[str(dataset.uid)] = sorted(selected)
        self._mark_dirty()

    def _settings_panel(self):
        panel = QWidget()
        form = QFormLayout(panel)
        self.wavelength = NoWheelDoubleSpinBox()
        self.wavelength.setRange(0.1, 5.0)
        self.wavelength.setDecimals(6)
        self.wavelength.setValue(1.5406)
        self.range_min = NoWheelDoubleSpinBox()
        self.range_min.setRange(0.0, 179.0)
        self.range_min.setDecimals(4)
        self.range_min.setValue(5.0)
        self.range_max = NoWheelDoubleSpinBox()
        self.range_max.setRange(0.1, 179.9)
        self.range_max.setDecimals(4)
        self.range_max.setValue(80.0)
        self.cutoff = NoWheelDoubleSpinBox()
        self.cutoff.setRange(0.0, 20.0)
        self.cutoff.setDecimals(3)
        self.cutoff.setValue(0.2)
        self.background_order = NoWheelSpinBox()
        self.background_order.setRange(0, 6)
        self.background_order.setValue(3)
        self.max_evaluations = NoWheelSpinBox()
        self.max_evaluations.setRange(10, 4000)
        self.max_evaluations.setValue(160)
        self.optimization_points = NoWheelSpinBox()
        self.optimization_points.setRange(300, 6000)
        self.optimization_points.setValue(1800)
        self.extraction_cycles = NoWheelSpinBox()
        self.extraction_cycles.setRange(1, 100)
        self.extraction_cycles.setValue(14)
        self.cell_tolerance = NoWheelDoubleSpinBox()
        self.cell_tolerance.setRange(0.01, 20.0)
        self.cell_tolerance.setDecimals(3)
        self.cell_tolerance.setValue(3.0)
        self.refine_cell = QCheckBox("Refine unit cell")
        self.refine_cell.setChecked(True)
        self.refine_profile = QCheckBox("Refine shared profile")
        self.refine_profile.setChecked(True)
        self.refine_zero = QCheckBox("Refine zero shift")
        self.refine_zero.setChecked(True)
        self.staged = QCheckBox("Use staged refinement")
        self.staged.setChecked(True)
        self.freeze_structure = QCheckBox("Freeze CIF structure factors in Rietveld")
        self.freeze_structure.setChecked(False)
        form.addRow("Wavelength (angstrom)", self.wavelength)
        form.addRow("2theta minimum", self.range_min)
        form.addRow("2theta maximum", self.range_max)
        form.addRow("CIF intensity cutoff (%)", self.cutoff)
        form.addRow("Background order", self.background_order)
        form.addRow("Maximum evaluations", self.max_evaluations)
        form.addRow("Optimization points", self.optimization_points)
        form.addRow("Le Bail extraction cycles", self.extraction_cycles)
        form.addRow("Cell bounds (+/- %)", self.cell_tolerance)
        form.addRow(self.refine_cell)
        form.addRow(self.refine_profile)
        form.addRow(self.refine_zero)
        form.addRow(self.staged)
        form.addRow(self.freeze_structure)
        return panel

    def _cif_library_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.cif_table = QTableWidget(0, 5)
        self.cif_table.setHorizontalHeaderLabels(["Active", "CIF / phase", "Formula", "Atoms", "Source"])
        self.cif_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cif_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.cif_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.cif_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        install_table_copy_menu(self.cif_table)
        layout.addWidget(self.cif_table)
        return panel

    def _reflection_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.reflection_summary = QLabel("Select a completed run to inspect its reflection-domain audit.")
        self.reflection_summary.setObjectName("mutedLabel")
        self.reflection_summary.setWordWrap(True)
        install_label_copy_menu(self.reflection_summary)
        layout.addWidget(self.reflection_summary)
        self.reflection_table = QTableWidget(0, 5)
        self.reflection_table.setHorizontalHeaderLabels(["Fit active", "2theta", "hkl", "Intensity", "Exclusion label"])
        self.reflection_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.reflection_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        install_table_copy_menu(self.reflection_table)
        layout.addWidget(self.reflection_table)
        return panel

    def _history_panel(self):
        group = QGroupBox("Refinement lineage / history")
        group.setMinimumWidth(520)
        layout = QVBoxLayout(group)
        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels(["Next", "Step", "Run ID / label", "Type", "CIF", "Rwp %", "Status"])
        self.history_table.setWordWrap(False)
        # CIF is already in the default candidate label and the row tooltip.
        # Retain the full column for data copying, without repeating it on screen.
        self.history_table.setColumnHidden(4, True)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        install_table_copy_menu(self.history_table)
        layout.addWidget(self.history_table, 1)
        self.history_summary = QLabel("No refinement history for this dataset.")
        self.history_summary.setWordWrap(True)
        self.history_summary.setObjectName("mutedLabel")
        install_label_copy_menu(self.history_summary)
        layout.addWidget(self.history_summary)
        self.continue_button = QPushButton("Continue from selected run")
        self.show_latest_button = QPushButton("Show latest completed run")
        layout.addWidget(self.continue_button)
        layout.addWidget(self.show_latest_button)
        return group

    def _settings(self) -> dict:
        return {
            "wavelength_angstrom": self.wavelength.value(),
            "two_theta_min": self.range_min.value(),
            "two_theta_max": self.range_max.value(),
            "intensity_cutoff_percent": self.cutoff.value(),
            "background_order": self.background_order.value(),
            "weighting": "Poisson-like",
            "refine_zero_shift": self.refine_zero.isChecked(),
            "initial_zero_shift": 0.0,
            "refine_profile": self.refine_profile.isChecked(),
            "initial_u": 0.005,
            "initial_v": 0.0,
            "initial_w": 0.02,
            "initial_eta": 0.5,
            "cell_tolerance_percent": self.cell_tolerance.value(),
            "maximum_evaluations": self.max_evaluations.value(),
            "optimization_points": self.optimization_points.value(),
            "extraction_cycles": self.extraction_cycles.value(),
            "use_staged_refinement": self.staged.isChecked(),
            "refine_cell": self.refine_cell.isChecked(),
            "freeze_structure_factors": self.freeze_structure.isChecked(),
            "validation_stride": 0,
            "use_gpu": self.use_gpu.isChecked(),
            "use_processed": self.use_processed.isChecked(),
        }

    def _set_settings(self, settings: dict):
        if not isinstance(settings, dict):
            return
        controls = (
            (self.wavelength, "wavelength_angstrom"),
            (self.range_min, "two_theta_min"),
            (self.range_max, "two_theta_max"),
            (self.cutoff, "intensity_cutoff_percent"),
            (self.background_order, "background_order"),
            (self.max_evaluations, "maximum_evaluations"),
            (self.optimization_points, "optimization_points"),
            (self.extraction_cycles, "extraction_cycles"),
            (self.cell_tolerance, "cell_tolerance_percent"),
        )
        for control, key in controls:
            if key in settings:
                control.setValue(float(settings[key]) if isinstance(control, NoWheelDoubleSpinBox) else int(settings[key]))
        checks = (
            (self.refine_zero, "refine_zero_shift"),
            (self.refine_profile, "refine_profile"),
            (self.staged, "use_staged_refinement"),
            (self.refine_cell, "refine_cell"),
            (self.freeze_structure, "freeze_structure_factors"),
            (self.use_gpu, "use_gpu"),
            (self.use_processed, "use_processed"),
        )
        for control, key in checks:
            if key in settings:
                control.setChecked(bool(settings[key]))

    @staticmethod
    def _structure_uid(structure: dict, source: str = "") -> str:
        existing = structure.get("_sequential_refinement_uid")
        if existing:
            return str(existing)
        cell = structure.get("cell") or {}
        seed = "|".join(
            [
                str(source),
                str(structure.get("data_name") or ""),
                str(structure.get("formula") or ""),
                str(len(structure.get("atoms") or [])),
                *(str(cell.get(key, "")) for key in ("a", "b", "c", "alpha", "beta", "gamma")),
                uuid4().hex,
            ]
        )
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]

    def _add_structure(self, structure: dict, source: str):
        copied = deepcopy(structure)
        uid = self._structure_uid(copied, source)
        copied["_sequential_refinement_uid"] = uid
        copied["_sequential_refinement_source"] = str(source)
        if any(str(row.get("_sequential_refinement_uid")) == uid for row in self.structures):
            return uid
        self.structures.append(copied)
        self._refresh_cif_library(select_uid=uid)
        self._populate_steps()
        self._mark_dirty()
        return uid

    def add_cif_files(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, "Add CIF models", "", "CIF files (*.cif);;All files (*.*)")
        errors = []
        for filename in filenames:
            try:
                self._add_structure(load_cif(filename), filename)
            except Exception as exc:
                errors.append(f"{Path(filename).name}: {exc}")
        if errors:
            QMessageBox.warning(self, "Some CIF files were not added", "\n".join(errors))

    def add_active_reference_cif(self):
        structure = getattr(self.main_window, "reference_structure", None)
        if not isinstance(structure, dict):
            QMessageBox.information(self, "No active CIF", "Import a CIF in Phase & Structure first, or use Add CIF(s).")
            return
        source = str(structure.get("source_path") or structure.get("data_name") or "Active Phase & Structure CIF")
        self._add_structure(structure, source)

    def remove_active_cif(self):
        uid = self._active_structure_uid()
        if not uid:
            return
        self.structures = [row for row in self.structures if str(row.get("_sequential_refinement_uid")) != uid]
        self._refresh_cif_library()
        self._mark_dirty()

    def _active_structure_uid(self) -> str | None:
        value = self.cif_combo.currentData()
        return None if value is None else str(value)

    def _active_structure(self) -> dict | None:
        uid = self._active_structure_uid()
        return next((row for row in self.structures if str(row.get("_sequential_refinement_uid")) == uid), None)

    def _select_structure_uid(self, uid: str | None):
        if uid is None:
            return
        index = self.cif_combo.findData(str(uid))
        if index >= 0:
            self.cif_combo.setCurrentIndex(index)

    def _refresh_cif_library(self, select_uid: str | None = None):
        active = str(select_uid or self._active_structure_uid() or "")
        self.cif_combo.blockSignals(True)
        self.cif_combo.clear()
        for structure in self.structures:
            uid = str(structure.get("_sequential_refinement_uid"))
            label = str(structure.get("data_name") or structure.get("formula") or f"CIF {len(self.cif_combo) + 1}")
            self.cif_combo.addItem(label, uid)
        index = self.cif_combo.findData(active)
        self.cif_combo.setCurrentIndex(index if index >= 0 else (0 if self.cif_combo.count() else -1))
        self.cif_combo.blockSignals(False)
        active = str(self._active_structure_uid() or "")
        self.cif_table.setRowCount(len(self.structures))
        for row, structure in enumerate(self.structures):
            uid = str(structure.get("_sequential_refinement_uid"))
            values = (
                "Active" if uid == active else "",
                structure.get("data_name") or f"CIF {row + 1}",
                structure.get("formula") or "-",
                len(structure.get("atoms") or []),
                structure.get("_sequential_refinement_source") or "",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, uid)
                self.cif_table.setItem(row, column, item)
            if uid == active:
                self.cif_table.selectRow(row)

    def _active_cif_changed(self, *_args):
        self._refresh_cif_library(select_uid=self._active_structure_uid())
        self._mark_dirty()

    def _cif_table_selected(self):
        row = self.cif_table.currentRow()
        item = self.cif_table.item(row, 0) if row >= 0 else None
        if item is not None:
            self._select_structure_uid(str(item.data(Qt.UserRole)))

    def _current_dataset(self):
        return self.main_window.selected_dataset()

    def refresh_for_selected_dataset(self):
        dataset = self._current_dataset()
        if self._last_dataset_uid and (dataset is None or str(dataset.uid) != self._last_dataset_uid):
            self._capture_step_draft(self._last_dataset_uid)
        if dataset is None:
            self._last_dataset_uid = None
            self.dataset_label.setText("No experimental pattern selected")
            self.exclusion_editor.set_dataset(None)
            self.history_table.setRowCount(0)
            self._populate_steps()
            self._draw_preview(None)
            return
        dataset_changed = str(dataset.uid) != str(self._last_dataset_uid or "")
        self._last_dataset_uid = str(dataset.uid)
        self.dataset_label.setText(f"{dataset.name} · {len(dataset.x):,} points")
        if dataset_changed and len(dataset.x):
            self.range_min.setValue(float(np.min(dataset.x)))
            self.range_max.setValue(float(np.max(dataset.x)))
            self.exclusion_editor.set_dataset(dataset.uid, dataset.x)
        elif self.exclusion_editor._uid != str(dataset.uid):
            self.exclusion_editor.set_dataset(dataset.uid, dataset.x)
        self._populate_steps()
        if dataset_changed:
            self._apply_step_controls()
        self._populate_history()
        selected = self._record_by_id(self._selected_history_run_id)
        if selected is None and dataset_changed:
            self._selected_history_run_id = None
        self._draw_record(selected) if selected else self._draw_preview(dataset)

    def _history(self, uid: str | None = None) -> list[dict]:
        dataset = self._current_dataset()
        key = str(uid or (dataset.uid if dataset is not None else ""))
        return self.history_by_uid.setdefault(key, []) if key else []

    def _record_by_id(self, run_id: str | None) -> dict | None:
        if not run_id:
            return None
        return next((row for row in self._history() if row.get("run_id") == run_id), None)

    def _selected_parent_id(self) -> str | None:
        record = self._record_by_id(self._selected_history_run_id)
        return None if record is None else str(record.get("run_id"))

    def _reserve_run_id(self, method: str) -> str:
        existing = {str(row.get("run_id")) for rows in self.history_by_uid.values() for row in rows}
        while True:
            candidate = refinement_run_id(self.next_run_counter, method)
            self.next_run_counter += 1
            if candidate not in existing:
                return candidate

    def run_method(self, method: str):
        if self.is_running():
            QMessageBox.information(self, "Refinement running", "Cancel or wait for the current refinement before starting another.")
            return
        dataset = self._current_dataset()
        structure = self._active_structure()
        if dataset is None:
            QMessageBox.information(self, "No dataset", "Select an experimental diffraction dataset first.")
            return
        if structure is None:
            QMessageBox.information(self, "No CIF", "Add one or more CIF files and select the active model.")
            return
        settings = self._settings()
        if settings["two_theta_min"] >= settings["two_theta_max"]:
            QMessageBox.warning(self, "Invalid range", "The minimum 2theta must be lower than the maximum.")
            return
        regions = self.exclusion_editor.regions()
        try:
            record = self._make_record(method, dataset, structure, settings, regions,
                                       self.run_label.text().strip(), self._selected_parent_id())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid refinement input", str(exc))
            return
        self._history(dataset.uid).append(record)
        self._launch_record(record)

    def _make_record(self, method, dataset, structure, settings, regions, label, parent_id, metadata=None):
        observed = dataset.y_processed if settings["use_processed"] and dataset.y_processed is not None else dataset.y_raw
        x, y = np.asarray(dataset.x, dtype=float), np.asarray(observed, dtype=float)
        if x.shape != y.shape:
            raise ValueError("Observed coordinates and intensity lengths differ.")
        signature = comparison_domain_signature(x, y, settings, regions)
        included = inclusion_mask(x, regions) & (x >= settings["two_theta_min"]) & (x <= settings["two_theta_max"])
        minimum = 80 if method == "rietveld" else 50
        if np.count_nonzero(included) < minimum:
            raise ValueError(f"At least {minimum} measured points must remain inside the fit range after exclusions.")
        run_id = self._reserve_run_id(method)
        record = {
            "run_id": run_id,
            "parent_run_id": parent_id,
            "label": label,
            "method_key": method,
            "method": METHOD_LABELS[method],
            "status": "Queued",
            "queued_utc": _utc_now(),
            "dataset_uid": dataset.uid,
            "dataset_name": dataset.name,
            "cif_uid": structure.get("_sequential_refinement_uid"),
            "cif_name": str(structure.get("data_name") or structure.get("formula") or "CIF phase"),
            "cif_source": str(structure.get("_sequential_refinement_source") or ""),
            "settings": deepcopy(settings),
            "excluded_regions": [row.to_dict() for row in merge_regions(regions)],
            "structure_snapshot": deepcopy(structure),
            "input_snapshot": {"x": x.tolist(), "observed_y": y.tolist()},
            "comparison_domain_signature": signature,
            "result": None,
        }
        record.update(deepcopy(metadata or {}))
        return record

    def _launch_record(self, record: dict):
        if self._thread is not None:
            raise RuntimeError("Only one candidate worker may run at a time.")
        snapshot = record.get("input_snapshot") or {}
        if not record.get("structure_snapshot") or not snapshot.get("x") or not snapshot.get("observed_y"):
            raise ValueError("The queued run is missing its frozen model or measurement snapshot; queue a new attempt.")
        run_id, method = str(record["run_id"]), str(record["method_key"])
        settings = record["settings"]
        record["status"] = "Running"
        record["started_utc"] = _utc_now()
        self._active_run_id = run_id
        self._active_dataset_uid = str(record["dataset_uid"])
        dataset = self._current_dataset()
        if dataset is not None and str(dataset.uid) == self._active_dataset_uid:
            self._selected_history_run_id = run_id
        self._populate_history(select_id=run_id)
        self._populate_candidates()
        self._set_busy(True)
        self.progress.setRange(0, max(1, settings["maximum_evaluations"]))
        self.progress.setValue(0)
        self.progress_label.setText(f"{run_id} · preparing {METHOD_LABELS[method]}...")
        thread = QThread(self)
        worker = SequentialRefinementWorker(
            method,
            record["input_snapshot"]["x"],
            record["input_snapshot"]["observed_y"],
            record["structure_snapshot"],
            settings,
            record["excluded_regions"],
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
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        self._mark_dirty()
        thread.start()

    def cancel_current_run(self):
        self._queue_running = False
        if self._worker is not None:
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.progress_label.setText("Cancellation requested; waiting for a safe numerical checkpoint...")

    @Slot(int, int, str)
    def _on_progress(self, done: int, total: int, message: str):
        self.progress.setRange(0, max(1, int(total)))
        self.progress.setValue(max(0, min(int(done), max(1, int(total)))))
        record = self._active_record() or {}
        title = str(record.get("label") or self._active_run_id)
        self.progress_label.setText(f"{title} [{self._active_run_id}] · {' '.join(str(message).split())}")
        self.progress_label.setToolTip(self.progress_label.text())

    def _active_record(self) -> dict | None:
        if not self._active_run_id or not self._active_dataset_uid:
            return None
        return next(
            (row for row in self.history_by_uid.get(self._active_dataset_uid, []) if row.get("run_id") == self._active_run_id),
            None,
        )

    @Slot(object)
    def _on_finished(self, result):
        record = self._active_record()
        if record is not None:
            result = dict(result or {})
            result["refinement_id"] = record["run_id"]
            result["parent_refinement_id"] = record.get("parent_run_id")
            result["user_label"] = record.get("label")
            result["cif_uid"] = record.get("cif_uid")
            result["cif_name"] = record.get("cif_name")
            result["cif_source"] = record.get("cif_source")
            for key in ("step_id", "step_number", "step_name", "candidate_uid", "candidate_order", "candidate_label", "comparison_domain_signature"):
                if key in record:
                    result[key] = record[key]
            record["status"] = "Completed"
            record["completed_utc"] = _utc_now()
            record["result"] = result
            if result.get("success") is False:
                record["convergence_warning"] = "Numerical run returned without convergence; rerun before advancing."
            self.progress_label.setText(
                f"{record['run_id']} completed · Rwp {float(result.get('rwp_percent', 0.0)):.5g}% · "
                f"{result.get('excluded_data_point_count', 0)} excluded point(s)."
            )
            dataset = self._current_dataset()
            if dataset is not None and str(dataset.uid) == self._active_dataset_uid:
                self._selected_history_run_id = record["run_id"]
                self._draw_record(record)
        self.progress.setValue(self.progress.maximum())
        self._populate_history(select_id=self._selected_history_run_id)
        self._populate_candidates()
        self._mark_dirty()
        if record is not None:
            self.runCompleted.emit(str(record["run_id"]))

    @Slot(str)
    def _on_failed(self, traceback_text: str):
        record = self._active_record()
        final = str(traceback_text).strip().splitlines()[-1] if traceback_text else "Unknown refinement failure"
        if record is not None:
            record["status"] = "Failed"
            record["completed_utc"] = _utc_now()
            record["error"] = final
            record["error_details"] = str(traceback_text)
            if record.get("candidate_uid") and not record.get("continue_after_failure", True):
                self._queue_running = False
        self.progress_label.setText(f"{self._active_run_id} failed · {final}")
        self._populate_history(select_id=self._active_run_id)
        self._populate_candidates()
        self._mark_dirty()
        if record is not None and record.get("candidate_uid"):
            return  # Failed candidate is audited; no modal dialog blocks later jobs.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Sequential refinement failed")
        box.setText(final)
        box.setDetailedText(str(traceback_text))
        box.exec()

    @Slot()
    def _on_cancelled(self):
        self._queue_running = False
        record = self._active_record()
        if record is not None:
            record["status"] = "Cancelled"
            record["completed_utc"] = _utc_now()
        self.progress_label.setText(f"{self._active_run_id} cancelled; earlier history is unchanged.")
        self._populate_history(select_id=self._active_run_id)
        self._populate_candidates()
        self._mark_dirty()

    @Slot()
    def _thread_finished(self):
        self._worker = None
        self._thread = None
        self._active_run_id = None
        self._active_dataset_uid = None
        self._set_busy(self._queue_running)
        self._populate_candidates()
        if self._queue_running:
            QTimer.singleShot(0, self._dispatch_next_queued)

    def _set_busy(self, busy: bool):
        for control in (
            self.pawley_button,
            self.lebail_button,
            self.rietveld_button,
            self.add_cif,
            self.add_active_cif,
            self.remove_cif,
            self.cif_combo,
            self.use_processed,
            self.use_gpu,
            self.step_selector, self.step_name, self.queue_method, self.common_domain,
            self.continue_after_failure, self.candidate_table, self.queue_checked_button,
            self.start_queue_button, self.next_step_button, self.move_candidate_up, self.move_candidate_down,
            self.discard_queue_button,
        ):
            control.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.pause_queue_button.setEnabled(self._queue_running)

    def _populate_history(self, select_id: str | None = None):
        rows = self._history()
        self.history_table.blockSignals(True)
        self.history_table.setRowCount(len(rows))
        selected_row = -1
        for index, record in enumerate(rows):
            result = record.get("result") or {}
            values = (
                "",
                f"{record.get('step_id', 'Manual')} / {record.get('candidate_order', '-')}",
                f"{record.get('run_id', '')}\n{record.get('label') or record.get('cif_name', '')}",
                {"pawley": "PW", "lebail": "LB", "rietveld": "RT"}.get(record.get("method_key"), record.get("method", "")),
                record.get("cif_name", ""),
                "-" if result.get("rwp_percent") is None else f"{float(result['rwp_percent']):.5g}",
                "Completed · not converged" if record.get("convergence_warning") else record.get("status", ""),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, str(record.get("run_id")))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setToolTip(f"{record.get('run_id')} · Parent: {record.get('parent_run_id') or 'original CIF'}\n{record.get('label') or ''}\n{record.get('method')} · CIF: {record.get('cif_name')}")
                if column == 0 and can_advance(record):
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked if record.get("run_id") in self.checked_runs_by_uid.get(str(record.get("dataset_uid")), []) else Qt.Unchecked)
                self.history_table.setItem(index, column, item)
            if str(record.get("run_id")) == str(select_id or self._selected_history_run_id):
                selected_row = index
        self.history_table.resizeRowsToContents()
        if selected_row >= 0:
            self.history_table.selectRow(selected_row)
        self.history_table.blockSignals(False)
        completed = sum(1 for row in rows if row.get("status") == "Completed")
        self.history_summary.setText(
            f"{len(rows)} run attempt(s); {completed} completed. Selecting a record restores its frozen plot and audit."
            if rows
            else "No refinement history for this dataset. The first run becomes the root of a reproducible lineage."
        )

    def _history_selection_changed(self):
        row = self.history_table.currentRow()
        item = self.history_table.item(row, 0) if row >= 0 else None
        if item is None:
            return
        self._selected_history_run_id = str(item.data(Qt.UserRole))
        record = self._record_by_id(self._selected_history_run_id)
        if record is not None:
            self._draw_record(record)

    def continue_from_selected(self):
        record = self._record_by_id(self._selected_history_run_id)
        if record is None:
            return
        self._select_structure_uid(record.get("cif_uid"))
        self._set_settings(record.get("settings") or {})
        self.exclusion_editor.set_regions(record.get("excluded_regions") or [])
        self.run_label.setText(f"Continuation of {record.get('run_id')}")
        self.progress_label.setText(
            f"Prepared next run from {record.get('run_id')}. Add/remove labelled exclusions, choose a method, then run."
        )

    def select_latest_run(self):
        rows = [row for row in self._history() if row.get("status") == "Completed"]
        if not rows:
            return
        self._selected_history_run_id = str(rows[-1].get("run_id"))
        self._populate_history(select_id=self._selected_history_run_id)
        self._draw_record(rows[-1])

    def _plot_clicked(self, event):
        if event.button() != Qt.LeftButton or not self.plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.plot.getPlotItem().vb.mapSceneToView(event.scenePos())
        self.exclusion_editor.center_selector(float(point.x()))

    @staticmethod
    def _finite_segments(x: np.ndarray, y: np.ndarray):
        if len(x) < 2:
            return x, y
        steps = np.diff(x)
        positive = steps[steps > 0]
        threshold = (float(np.median(positive)) * 3.5) if positive.size else float("inf")
        out_x, out_y = [], []
        for index, (x_value, y_value) in enumerate(zip(x, y)):
            if index and x_value - x[index - 1] > threshold:
                out_x.append(np.nan)
                out_y.append(np.nan)
            out_x.append(float(x_value))
            out_y.append(float(y_value))
        return np.asarray(out_x), np.asarray(out_y)

    def _clear_plot(self):
        self.plot.clear()
        self.legend = self.plot.addLegend()
        self._plot_items = []

    def _draw_preview(self, dataset):
        self._clear_plot()
        if dataset is not None:
            y = dataset.y_processed if self.use_processed.isChecked() and dataset.y_processed is not None else dataset.y_raw
            self.plot.plot(dataset.x, y, pen=pg.mkPen(174, 190, 221, 220, width=1.25), name="Observed preview")
            self.plot.enableAutoRange()
        self.exclusion_editor.bind_plot(self.plot)
        self._fill_reflections(None)

    def _draw_record(self, record: dict | None):
        result = (record or {}).get("result") if isinstance(record, dict) else None
        if not isinstance(result, dict):
            self._draw_preview(self._current_dataset())
            return
        self._clear_plot()
        theme = THEMES.get(self.current_theme_name, THEMES[DEFAULT_THEME_NAME])
        x = np.asarray(result.get("observed_x", []), dtype=float)
        observed = np.asarray(result.get("observed_y", []), dtype=float)
        calculated = np.asarray(result.get("calculated_y", []), dtype=float)
        background = np.asarray(result.get("background_y", []), dtype=float)
        difference = np.asarray(result.get("difference_y", []), dtype=float)
        self.plot.plot(x, observed, pen=None, symbol="o", symbolSize=3.0, symbolPen=pg.mkPen(theme["plot_axis"]), name="Observed / included")
        for values, pen, name in (
            (calculated, pg.mkPen(theme["gold"], width=2.0), "Calculated / included"),
            (background, pg.mkPen(theme["muted"], width=1.1, style=Qt.DashLine), "Background"),
        ):
            sx, sy = self._finite_segments(x, values)
            self.plot.plot(sx, sy, connect="finite", pen=pen, name=name)
        if len(x) and len(difference):
            span = max(float(np.ptp(observed)), 1.0)
            offset = float(np.min(observed)) - 0.18 * span
            sx, sy = self._finite_segments(x, difference + offset)
            self.plot.plot(sx, sy, connect="finite", pen=pg.mkPen(theme["gold_bright"], width=1.0), name="Difference")
        regions = merge_regions(record.get("excluded_regions") or [])
        source = record.get("input_snapshot") or {}
        dataset = self._current_dataset()
        if source:
            source_x = np.asarray(source["x"], dtype=float)
            source_y = np.asarray(source["observed_y"], dtype=float)
        elif dataset is not None:  # Older project records predate frozen inputs.
            source_x = np.asarray(dataset.x)
            source_y = dataset.y_processed if record.get("settings", {}).get("use_processed") and dataset.y_processed is not None else dataset.y_raw
        else:
            source_x, source_y = np.array([]), np.array([])
        if regions and len(source_x):
            excluded = ~inclusion_mask(source_x, regions)
            if np.any(excluded):
                self.plot.plot(source_x[excluded], np.asarray(source_y)[excluded], pen=None, symbol="x", symbolSize=7, symbolPen=pg.mkPen(235, 87, 87, 230, width=1.4), name="Excluded measured points")
        reflections = [row for row in result.get("reflections", []) if row.get("fit_active", True)]
        centers = [position for row in reflections if (position := _position_of_reflection(row)) is not None]
        if centers and len(observed):
            span = max(float(np.ptp(observed)), 1.0)
            base = float(np.min(observed)) - 0.27 * span
            tick_x, tick_y = [], []
            for center in centers:
                tick_x.extend([center, center, np.nan])
                tick_y.extend([base, base + 0.035 * span, np.nan])
            self.plot.plot(tick_x, tick_y, connect="finite", pen=pg.mkPen(theme["gold"], width=1.2), name="Fit-active CIF reflections")
        self.plot.enableAutoRange()
        self.exclusion_editor.bind_plot(self.plot)
        self._fill_reflections(result)
        label = str(record.get("label") or "").strip()
        self.progress_label.setText(
            f"Viewing {record.get('run_id')} · {record.get('method')} · {record.get('cif_name')} · "
            f"Rwp {float(result.get('rwp_percent', 0.0)):.5g}%" + (f" · {label}" if label else "")
        )
        self.progress_label.setToolTip(self.progress_label.text())

    def _fill_reflections(self, result: dict | None):
        rows = list((result or {}).get("reflections", []))
        self.reflection_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            intensity = row.get("structure_intensity", row.get("reference_intensity", row.get("intensity", "")))
            values = (
                "Yes" if row.get("fit_active", True) else "No",
                _position_of_reflection(row),
                row.get("hkl_label", ""),
                intensity,
                row.get("exclusion_label", ""),
            )
            for column, value in enumerate(values):
                text = "-" if value is None else (f"{value:.7g}" if isinstance(value, float) else str(value))
                self.reflection_table.setItem(index, column, QTableWidgetItem(text))
        audit = (result or {}).get("cif_reflection_fit_domain") or {}
        self.reflection_summary.setText(
            f"{audit.get('fit_active_reflection_count', 0)} active; {audit.get('fit_inactive_reflection_count', 0)} inactive reflection center(s). "
            + str(audit.get("rule") or "Select a completed run to inspect its reflection-domain audit.")
        )

    def _draft_exclusions_changed(self):
        if self._step_switching:
            return
        self._mark_dirty()
        record = self._record_by_id(self._selected_history_run_id)
        self._draw_record(record) if record else self._draw_preview(self._current_dataset())

    def export_full_history(self, *, target_path=None):
        dataset = self._current_dataset()
        self._capture_step_draft()
        rows = deepcopy(self._history())
        completed = [row for row in rows if row.get("status") == "Completed" and isinstance(row.get("result"), dict)]
        if dataset is None or not rows:
            QMessageBox.information(self, "No history", "Queue or complete a refinement before exporting the history bundle.")
            return
        default = f"{safe_filename(dataset.name)}_sequential_refinement_history.zip"
        filename = str(target_path) if target_path is not None else QFileDialog.getSaveFileName(
            self, "Export complete sequential-refinement history", default, "ZIP archive (*.zip)"
        )[0]
        if not filename:
            return
        target = Path(filename).with_suffix(".zip")
        try:
            with tempfile.TemporaryDirectory(prefix="afruz-sequential-export-") as temp_name:
                root = Path(temp_name)
                history_rows = []
                for record in rows:
                    result = record.get("result") or {}
                    history_rows.append(
                        {
                            "run_id": record.get("run_id"),
                            "parent_run_id": record.get("parent_run_id"),
                            "label": record.get("label"),
                            "step_id": record.get("step_id"),
                            "step_name": record.get("step_name"),
                            "candidate_order": record.get("candidate_order"),
                            "candidate_label": record.get("candidate_label"),
                            "selected_for_next_step": record.get("run_id") in self.checked_runs_by_uid.get(str(dataset.uid), []),
                            "comparison_domain_signature": record.get("comparison_domain_signature"),
                            "method": record.get("method"),
                            "status": record.get("status"),
                            "cif_name": record.get("cif_name"),
                            "cif_source": record.get("cif_source"),
                            "started_utc": record.get("started_utc"),
                            "completed_utc": record.get("completed_utc"),
                            "rwp_percent": result.get("rwp_percent"),
                            "rp_percent": result.get("rp_percent"),
                            "excluded_point_count": result.get("excluded_data_point_count"),
                            "gpu_backend": (result.get("acceleration") or {}).get("calculation_backend"),
                        }
                    )
                write_table_txt(root / "refinement_history.txt", history_rows, title=f"Sequential refinement history - {dataset.name}")
                plan_rows = []
                for step in self.steps_by_uid.get(str(dataset.uid), []):
                    for index, candidate in enumerate(step.get("candidates", [])):
                        plan_rows.append({"step_id": step["step_id"], "step_name": step["name"],
                                          "order": index + 1, "label": candidate["label"], "cif": candidate["cif_name"],
                                          "enabled": candidate.get("enabled", True), "parent_run_id": candidate.get("parent_run_id")})
                write_table_txt(root / "step_candidate_plan.txt", plan_rows, title="Step-wise candidate plan")
                write_mapping_txt(
                    root / "workspace_manifest.txt",
                    {
                        "dataset_name": dataset.name,
                        "dataset_uid": dataset.uid,
                        "run_attempt_count": len(rows),
                        "completed_run_count": len(completed),
                        "pending_run_count": sum(record.get("status") == "Queued" for record in rows),
                        "candidate_fit_convention": "One CIF per run; candidates are not fitted together and no fitted phase is subtracted.",
                        "comparison_rule": "Direct R-factor comparison requires the same observation and fit-domain signature.",
                        "exported_utc": _utc_now(),
                        "scientific_boundary": (
                            "Excluded intervals are fit-domain masks. Inactive hkl rows are audit labels; "
                            "the crystal structure factor model was not altered by deleting reflections."
                        ),
                    },
                    title="Sequential refinement export manifest",
                )
                cif_dir = root / "cif_models"
                cif_dir.mkdir()
                copied_sources = set()
                for structure in self.structures:
                    source = Path(str(structure.get("_sequential_refinement_source") or ""))
                    if source.is_file() and source.suffix.lower() == ".cif" and str(source.resolve()) not in copied_sources:
                        copied_sources.add(str(source.resolve()))
                        model_id = str(structure.get("_sequential_refinement_uid") or "CIF")
                        (cif_dir / f"{safe_filename(model_id)}__{safe_filename(source.name)}").write_bytes(source.read_bytes())
                for record in rows:
                    run_dir = root / safe_filename(str(record["run_id"]))
                    run_dir.mkdir()
                    if record.get("status") == "Completed" and isinstance(record.get("result"), dict):
                        export_refinement_txt_bundle(
                            run_dir / f"{safe_filename(str(record['run_id']))}_{safe_filename(str(record['method']))}.txt",
                            record["result"], dataset_name=dataset.name, dataset_uid=dataset.uid,
                            refinement_label=f"{record['method']} [{record['run_id']}]",
                        )
                    input_snapshot = record.get("input_snapshot")
                    if isinstance(input_snapshot, dict):
                        write_columns_txt(run_dir / "frozen_input_pattern.txt", {
                            "two_theta_deg": input_snapshot["x"], "intensity": input_snapshot["observed_y"]
                        }, title=f"Frozen measurement input - {record['run_id']}")
                    structure_snapshot = record.get("structure_snapshot")
                    if isinstance(structure_snapshot, dict):
                        from .project import _json_default
                        (run_dir / "frozen_model_snapshot.json").write_text(
                            json.dumps(structure_snapshot, default=_json_default, indent=2), encoding="utf-8"
                        )
                    write_table_txt(
                        run_dir / "excluded_regions.txt",
                        record.get("excluded_regions") or [],
                        title=f"Applied closed exclusion intervals - {record['run_id']}",
                        metadata={"parent_run_id": record.get("parent_run_id") or "root"},
                    )
                    write_mapping_txt(
                        run_dir / "run_provenance.txt",
                        {key: value for key, value in record.items() if key not in {"result", "input_snapshot", "structure_snapshot"}},
                        title=f"Run provenance - {record['run_id']}",
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=7) as archive:
                    for path in sorted(root.rglob("*")):
                        if path.is_file():
                            archive.write(path, path.relative_to(root).as_posix())
        except Exception as exc:
            QMessageBox.critical(self, "Full history export failed", str(exc))
            return
        self.progress_label.setText(f"Exported complete refinement history: {target}")
        return target

    def _mark_dirty(self):
        callback = getattr(self.main_window, "_mark_project_dirty", None)
        if callable(callback):
            callback()

    def is_running(self) -> bool:
        return self._thread is not None or self._queue_running

    def apply_theme(self, theme_name: str):
        theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME_NAME])
        self.current_theme_name = theme_name if theme_name in THEMES else DEFAULT_THEME_NAME
        self.plot.setBackground(theme["plot_background"])
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))

    def get_state(self):
        self._capture_step_draft()
        histories = deepcopy(self.history_by_uid)
        for rows in histories.values():
            for record in rows:
                if record.get("status") == "Running":
                    record["status"] = "Interrupted snapshot"
                    record["error"] = "The project was saved while this numerical run was still active. Start a new child run after reopening."
        histories, steps, snapshot_pool = pack_snapshots(histories, self.steps_by_uid)
        return {
            "schema_version": 2,
            "structures": deepcopy(self.structures),
            "history_by_uid": histories,
            "next_run_counter": int(self.next_run_counter),
            "active_cif_uid": self._active_structure_uid(),
            "selected_history_run_id": self._selected_history_run_id,
            "draft_exclusions": self.exclusion_editor.get_state(),
            "settings": self._settings(),
            "run_label": self.run_label.text(),
            "steps_by_uid": steps,
            "queue_snapshot_pool": snapshot_pool,
            "selected_step_by_uid": deepcopy(self.selected_step_by_uid),
            "checked_runs_by_uid": deepcopy(self.checked_runs_by_uid),
            "queue_scope": list(self._queue_scope) if self._queue_scope else None,
            "queue_resume_policy": "Paused on open; pending frozen jobs resume only after explicit Start.",
        }

    def set_state(self, state):
        state = state or {}
        self._queue_running = False
        if self._worker is not None:
            self._worker.request_cancel()
        structures = state.get("structures")
        self.structures = deepcopy(structures) if isinstance(structures, list) else []
        histories = state.get("history_by_uid")
        self.history_by_uid = deepcopy(histories) if isinstance(histories, dict) else {}
        self.steps_by_uid = deepcopy(state.get("steps_by_uid")) if isinstance(state.get("steps_by_uid"), dict) else {}
        unpack_snapshots(self.history_by_uid, self.steps_by_uid, state.get("queue_snapshot_pool") or {})
        self.selected_step_by_uid = deepcopy(state.get("selected_step_by_uid")) if isinstance(state.get("selected_step_by_uid"), dict) else {}
        self.checked_runs_by_uid = deepcopy(state.get("checked_runs_by_uid")) if isinstance(state.get("checked_runs_by_uid"), dict) else {}
        scope = state.get("queue_scope")
        self._queue_scope = tuple(str(value) for value in scope) if isinstance(scope, (list, tuple)) and len(scope) == 2 else None
        for rows in self.history_by_uid.values():
            for record in rows:
                if record.get("status") == "Running":
                    record["status"] = "Interrupted"
        try:
            self.next_run_counter = max(1, int(state.get("next_run_counter", 1)))
        except (TypeError, ValueError):
            self.next_run_counter = 1
        self._selected_history_run_id = state.get("selected_history_run_id")
        # Loading a project is not a user edit to the restored step draft.
        # In particular, do not capture the previous dataset's controls into
        # the just-restored step when the active dataset changes on open.
        self._step_switching = True
        try:
            self.exclusion_editor.set_state(state.get("draft_exclusions"))
            self._refresh_cif_library(select_uid=state.get("active_cif_uid"))
            self.refresh_for_selected_dataset()
        finally:
            self._step_switching = False
        self._apply_step_controls()
        # Dataset refresh establishes valid control ranges; saved values are
        # restored afterwards so reopening an AFZ does not silently expand a
        # deliberately restricted refinement interval.
        self._set_settings(state.get("settings") or {})
        self.run_label.setText(str(state.get("run_label") or ""))
        self._set_busy(self._thread is not None)
        self._populate_candidates()
