from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile
import threading
import traceback
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
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
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .context_copy import CopyablePlotWidget, install_label_copy_menu, install_table_copy_menu
from .crystallography import calculate_powder_pattern, load_cif
from .exclusion_regions import ExclusionRegion, attach_exclusion_audit, filter_arrays, inclusion_mask, merge_regions
from .gpu_backend import cuda_status
from .phase_identification import reference_from_cif_pattern
from .refinement_export import export_refinement_txt_bundle
from .rietveld_refinement import RietveldPhaseSpec, refine_rietveld
from .text_export import safe_filename, write_mapping_txt, write_table_txt
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

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.structures: list[dict] = []
        self.history_by_uid: dict[str, list[dict]] = {}
        self.next_run_counter = 1
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
        center.addWidget(self.detail_tabs)
        center.setSizes([610, 260])
        center.setStretchFactor(0, 4)
        center.setStretchFactor(1, 2)
        splitter.addWidget(center)
        splitter.addWidget(self._history_panel())
        splitter.setSizes([1150, 330])
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
        self.continue_button.clicked.connect(self.continue_from_selected)
        self.show_latest_button.clicked.connect(self.select_latest_run)
        self.exclusion_editor.changed.connect(self._draft_exclusions_changed)
        self.plot.scene().sigMouseClicked.connect(self._plot_clicked)

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
        layout = QVBoxLayout(group)
        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels(["Unique ID", "Method", "CIF", "Rwp %", "Status", "Parent"])
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
        if dataset is None:
            self._last_dataset_uid = None
            self.dataset_label.setText("No experimental pattern selected")
            self.exclusion_editor.set_dataset(None)
            self.history_table.setRowCount(0)
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
        if self._thread is not None:
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
        observed = dataset.y_processed if settings["use_processed"] and dataset.y_processed is not None else dataset.y_raw
        regions = self.exclusion_editor.regions()
        run_id = self._reserve_run_id(method)
        record = {
            "run_id": run_id,
            "parent_run_id": self._selected_parent_id(),
            "label": self.run_label.text().strip(),
            "method_key": method,
            "method": METHOD_LABELS[method],
            "status": "Running",
            "started_utc": _utc_now(),
            "dataset_uid": dataset.uid,
            "dataset_name": dataset.name,
            "cif_uid": self._active_structure_uid(),
            "cif_name": str(structure.get("data_name") or structure.get("formula") or "CIF phase"),
            "cif_source": str(structure.get("_sequential_refinement_source") or ""),
            "settings": deepcopy(settings),
            "excluded_regions": [row.to_dict() for row in regions],
            "result": None,
        }
        self._history(dataset.uid).append(record)
        self._active_run_id = run_id
        self._active_dataset_uid = dataset.uid
        self._selected_history_run_id = run_id
        self._populate_history(select_id=run_id)
        self._set_busy(True)
        self.progress.setRange(0, max(1, settings["maximum_evaluations"]))
        self.progress.setValue(0)
        self.progress_label.setText(f"{run_id} · preparing {METHOD_LABELS[method]}...")
        thread = QThread(self)
        worker = SequentialRefinementWorker(
            method,
            dataset.x,
            observed,
            structure,
            settings,
            regions,
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
        if self._worker is not None:
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.progress_label.setText("Cancellation requested; waiting for a safe numerical checkpoint...")

    def _on_progress(self, done: int, total: int, message: str):
        self.progress.setRange(0, max(1, int(total)))
        self.progress.setValue(max(0, min(int(done), max(1, int(total)))))
        self.progress_label.setText(f"{self._active_run_id} · {' '.join(str(message).split())}")

    def _active_record(self) -> dict | None:
        if not self._active_run_id or not self._active_dataset_uid:
            return None
        return next(
            (row for row in self.history_by_uid.get(self._active_dataset_uid, []) if row.get("run_id") == self._active_run_id),
            None,
        )

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
            record["status"] = "Completed"
            record["completed_utc"] = _utc_now()
            record["result"] = result
            self._selected_history_run_id = record["run_id"]
            self.progress_label.setText(
                f"{record['run_id']} completed · Rwp {float(result.get('rwp_percent', 0.0)):.5g}% · "
                f"{result.get('excluded_data_point_count', 0)} excluded point(s)."
            )
            self._draw_record(record)
        self.progress.setValue(self.progress.maximum())
        self._populate_history(select_id=self._selected_history_run_id)
        self._set_busy(False)
        self._mark_dirty()

    def _on_failed(self, traceback_text: str):
        record = self._active_record()
        final = str(traceback_text).strip().splitlines()[-1] if traceback_text else "Unknown refinement failure"
        if record is not None:
            record["status"] = "Failed"
            record["completed_utc"] = _utc_now()
            record["error"] = final
        self.progress_label.setText(f"{self._active_run_id} failed · {final}")
        self._populate_history(select_id=self._active_run_id)
        self._set_busy(False)
        self._mark_dirty()
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Sequential refinement failed")
        box.setText(final)
        box.setDetailedText(str(traceback_text))
        box.exec()

    def _on_cancelled(self):
        record = self._active_record()
        if record is not None:
            record["status"] = "Cancelled"
            record["completed_utc"] = _utc_now()
        self.progress_label.setText(f"{self._active_run_id} cancelled; earlier history is unchanged.")
        self._populate_history(select_id=self._active_run_id)
        self._set_busy(False)
        self._mark_dirty()

    def _thread_finished(self):
        self._worker = None
        self._thread = None
        self._active_run_id = None
        self._active_dataset_uid = None
        self._set_busy(False)

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
        ):
            control.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def _populate_history(self, select_id: str | None = None):
        rows = self._history()
        self.history_table.blockSignals(True)
        self.history_table.setRowCount(len(rows))
        selected_row = -1
        for index, record in enumerate(rows):
            result = record.get("result") or {}
            values = (
                record.get("run_id", ""),
                record.get("method", ""),
                record.get("cif_name", ""),
                "-" if result.get("rwp_percent") is None else f"{float(result['rwp_percent']):.5g}",
                record.get("status", ""),
                record.get("parent_run_id") or "root",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, str(record.get("run_id")))
                self.history_table.setItem(index, column, item)
            if str(record.get("run_id")) == str(select_id or self._selected_history_run_id):
                selected_row = index
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
        dataset = self._current_dataset()
        regions = merge_regions(record.get("excluded_regions") or [])
        if dataset is not None and regions:
            source_y = dataset.y_processed if record.get("settings", {}).get("use_processed") and dataset.y_processed is not None else dataset.y_raw
            excluded = ~inclusion_mask(dataset.x, regions)
            if np.any(excluded):
                self.plot.plot(dataset.x[excluded], np.asarray(source_y)[excluded], pen=None, symbol="x", symbolSize=7, symbolPen=pg.mkPen(235, 87, 87, 230, width=1.4), name="Excluded measured points")
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
        self._mark_dirty()
        record = self._record_by_id(self._selected_history_run_id)
        self._draw_record(record) if record else self._draw_preview(self._current_dataset())

    def export_full_history(self):
        dataset = self._current_dataset()
        rows = self._history()
        completed = [row for row in rows if row.get("status") == "Completed" and isinstance(row.get("result"), dict)]
        if dataset is None or not completed:
            QMessageBox.information(self, "No completed history", "Complete at least one refinement before exporting the history bundle.")
            return
        default = f"{safe_filename(dataset.name)}_sequential_refinement_history.zip"
        filename, _ = QFileDialog.getSaveFileName(self, "Export complete sequential-refinement history", default, "ZIP archive (*.zip)")
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
                write_mapping_txt(
                    root / "workspace_manifest.txt",
                    {
                        "dataset_name": dataset.name,
                        "dataset_uid": dataset.uid,
                        "run_attempt_count": len(rows),
                        "completed_run_count": len(completed),
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
                        (cif_dir / safe_filename(source.name)).write_bytes(source.read_bytes())
                for record in completed:
                    run_dir = root / safe_filename(str(record["run_id"]))
                    run_dir.mkdir()
                    export_refinement_txt_bundle(
                        run_dir / f"{safe_filename(str(record['run_id']))}_{safe_filename(str(record['method']))}.txt",
                        record["result"],
                        dataset_name=dataset.name,
                        dataset_uid=dataset.uid,
                        refinement_label=f"{record['method']} [{record['run_id']}]",
                    )
                    write_table_txt(
                        run_dir / "excluded_regions.txt",
                        record.get("excluded_regions") or [],
                        title=f"Applied closed exclusion intervals - {record['run_id']}",
                        metadata={"parent_run_id": record.get("parent_run_id") or "root"},
                    )
                    write_mapping_txt(
                        run_dir / "run_provenance.txt",
                        {key: value for key, value in record.items() if key != "result"},
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

    def _mark_dirty(self):
        callback = getattr(self.main_window, "_mark_project_dirty", None)
        if callable(callback):
            callback()

    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.isRunning())

    def apply_theme(self, theme_name: str):
        theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME_NAME])
        self.current_theme_name = theme_name if theme_name in THEMES else DEFAULT_THEME_NAME
        self.plot.setBackground(theme["plot_background"])
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))

    def get_state(self):
        histories = deepcopy(self.history_by_uid)
        for rows in histories.values():
            for record in rows:
                if record.get("status") == "Running":
                    record["status"] = "Interrupted snapshot"
                    record["error"] = "The project was saved while this numerical run was still active. Start a new child run after reopening."
        return {
            "schema_version": 1,
            "structures": deepcopy(self.structures),
            "history_by_uid": histories,
            "next_run_counter": int(self.next_run_counter),
            "active_cif_uid": self._active_structure_uid(),
            "selected_history_run_id": self._selected_history_run_id,
            "draft_exclusions": self.exclusion_editor.get_state(),
            "settings": self._settings(),
            "run_label": self.run_label.text(),
        }

    def set_state(self, state):
        state = state or {}
        structures = state.get("structures")
        self.structures = deepcopy(structures) if isinstance(structures, list) else []
        histories = state.get("history_by_uid")
        self.history_by_uid = deepcopy(histories) if isinstance(histories, dict) else {}
        for rows in self.history_by_uid.values():
            for record in rows:
                if record.get("status") == "Running":
                    record["status"] = "Interrupted"
        try:
            self.next_run_counter = max(1, int(state.get("next_run_counter", 1)))
        except (TypeError, ValueError):
            self.next_run_counter = 1
        self._selected_history_run_id = state.get("selected_history_run_id")
        self.exclusion_editor.set_state(state.get("draft_exclusions"))
        self._refresh_cif_library(select_uid=state.get("active_cif_uid"))
        self.refresh_for_selected_dataset()
        # Dataset refresh establishes valid control ranges; saved values are
        # restored afterwards so reopening an AFZ does not silently expand a
        # deliberately restricted refinement interval.
        self._set_settings(state.get("settings") or {})
        self.run_label.setText(str(state.get("run_label") or ""))
