from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import traceback

import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .context_copy import CopyablePlotWidget, install_label_copy_menu, install_table_copy_menu
from .crystallography import calculate_powder_pattern, load_cif
from .text_export import write_columns_txt, write_table_txt
from .theme import DEFAULT_THEME_NAME, THEMES
from .widgets import NoWheelDoubleSpinBox


def simulated_profile(
    pattern: list[dict],
    two_theta_min: float,
    two_theta_max: float,
    *,
    fwhm_deg: float,
    eta: float,
    point_count: int = 6000,
) -> tuple[np.ndarray, np.ndarray]:
    """Broaden CIF reflection sticks for visualization, without refinement."""

    x = np.linspace(float(two_theta_min), float(two_theta_max), max(500, int(point_count)))
    y = np.zeros_like(x)
    width = max(float(fwhm_deg), 1e-5)
    mixing = float(np.clip(eta, 0.0, 1.0))
    for peak in pattern:
        center = float(peak["two_theta"])
        intensity = max(0.0, float(peak.get("intensity", 0.0)))
        delta = (x - center) / width
        gaussian = np.exp(-4.0 * np.log(2.0) * delta * delta)
        lorentzian = 1.0 / (1.0 + 4.0 * delta * delta)
        y += intensity * (mixing * lorentzian + (1.0 - mixing) * gaussian)
    maximum = float(np.max(y)) if y.size else 0.0
    if maximum > 0:
        y = 100.0 * y / maximum
    return x, y


class _CifSimulationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, structure: dict, settings: dict):
        super().__init__()
        self.structure = structure
        self.settings = settings

    @Slot()
    def run(self):
        try:
            pattern = calculate_powder_pattern(
                self.structure,
                wavelength_angstrom=self.settings["wavelength_angstrom"],
                two_theta_min=self.settings["two_theta_min"],
                two_theta_max=self.settings["two_theta_max"],
                intensity_cutoff_percent=self.settings["cutoff_percent"],
                use_gpu=self.settings["use_gpu"],
            )
            x, y = simulated_profile(
                pattern,
                self.settings["two_theta_min"],
                self.settings["two_theta_max"],
                fwhm_deg=self.settings["fwhm_deg"],
                eta=self.settings["eta"],
            )
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit({"pattern": pattern, "profile_x": x, "profile_y": y})


class CifSimulationWidget(QWidget):
    """Large central CIF-to-PXRD calculation workspace."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.structure: dict | None = None
        self.pattern: list[dict] = []
        self.profile_x = np.asarray([], dtype=float)
        self.profile_y = np.asarray([], dtype=float)
        self._thread: QThread | None = None
        self._worker: _CifSimulationWorker | None = None
        self.plot: CopyablePlotWidget | None = None
        self.table: QTableWidget | None = None
        self._visualization_layout: QVBoxLayout | None = None
        self._theme_name = DEFAULT_THEME_NAME
        self._build_ui()
        self.use_active_cif(silent=True)

    @staticmethod
    def _spin(low, high, value, decimals=5, step=0.01):
        spin = NoWheelDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _build_ui(self):
        layout = QVBoxLayout(self)
        heading = QLabel("Simulated PXRD from CIF")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        description = QLabel(
            "Calculate crystallographic reflection positions and relative kinematic intensities directly from a CIF, "
            "then inspect both the reflection sticks and a display-only broadened profile in a large central plot. "
            "The broadened curve is a simulation preview, not a fitted Rietveld profile and not measured data."
        )
        description.setWordWrap(True)
        description.setObjectName("mutedLabel")
        install_label_copy_menu(description)
        layout.addWidget(description)

        controls = QGroupBox("CIF and simulation controls")
        form = QFormLayout(controls)
        source_row = QHBoxLayout()
        self.active_button = QPushButton("Use active CIF")
        self.import_button = QPushButton("Import another CIF...")
        self.calculate_button = QPushButton("Calculate simulated PXRD")
        self.calculate_button.setObjectName("primaryButton")
        source_row.addWidget(self.active_button)
        source_row.addWidget(self.import_button)
        source_row.addStretch(1)
        source_row.addWidget(self.calculate_button)
        form.addRow(source_row)

        parameter_row = QHBoxLayout()
        self.wavelength = self._spin(0.1, 5.0, 1.5406, 7, 0.0001)
        self.minimum = self._spin(0.1, 175.0, 3.0, 4, 0.5)
        self.maximum = self._spin(1.0, 179.9, 80.0, 4, 0.5)
        self.cutoff = self._spin(0.0, 100.0, 0.1, 4, 0.1)
        self.fwhm = self._spin(0.005, 5.0, 0.12, 5, 0.01)
        self.eta = self._spin(0.0, 1.0, 0.5, 4, 0.05)
        self.use_gpu = QCheckBox("Use GPU when available")
        self.use_gpu.setChecked(True)
        for label, widget in (
            ("Wavelength (A)", self.wavelength),
            ("2theta min", self.minimum),
            ("2theta max", self.maximum),
            ("Cutoff (%)", self.cutoff),
            ("Display FWHM", self.fwhm),
            ("Pseudo-Voigt eta", self.eta),
        ):
            parameter_row.addWidget(QLabel(label))
            parameter_row.addWidget(widget)
        parameter_row.addWidget(self.use_gpu)
        form.addRow(parameter_row)
        layout.addWidget(controls)

        self.structure_status = QLabel("No CIF selected.")
        self.structure_status.setObjectName("mutedLabel")
        self.structure_status.setWordWrap(True)
        install_label_copy_menu(self.structure_status)
        layout.addWidget(self.structure_status)

        visualization_host = QWidget()
        self._visualization_layout = QVBoxLayout(visualization_host)
        self._visualization_layout.setContentsMargins(0, 0, 0, 0)
        placeholder = QLabel("Open this task to initialize the large CIF diffraction canvas.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setObjectName("mutedLabel")
        placeholder.setProperty("cifSimulationPlaceholder", True)
        self._visualization_layout.addWidget(placeholder, 1)
        layout.addWidget(visualization_host, 1)

        footer = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Ready")
        self.export_button = QPushButton("Export simulated profile and reflections")
        self.export_button.setEnabled(False)
        footer.addWidget(self.progress, 1)
        footer.addWidget(self.export_button)
        layout.addLayout(footer)

        self.active_button.clicked.connect(self.use_active_cif)
        self.import_button.clicked.connect(self.import_cif)
        self.calculate_button.clicked.connect(self.calculate)
        self.export_button.clicked.connect(self.export)

    def _ensure_visualization(self):
        if self.plot is not None and self.table is not None:
            return
        if self._visualization_layout is None:
            return
        while self._visualization_layout.count():
            item = self._visualization_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        splitter = QSplitter(Qt.Vertical)
        theme = THEMES.get(self._theme_name, THEMES[DEFAULT_THEME_NAME])
        self.plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="Simulated PXRD from CIF",
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.plot.setLabel("bottom", "2theta", units="degrees")
        self.plot.setLabel("left", "Relative intensity", units="%")
        self.plot.addLegend()
        splitter.addWidget(self.plot)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["2theta (deg)", "d (A)", "hkl", "Relative I", "Multiplicity", "Equivalent hkls", "Source"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        install_table_copy_menu(self.table)
        splitter.addWidget(self.table)
        splitter.setSizes([650, 210])
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        self._visualization_layout.addWidget(splitter, 1)
        if self.pattern:
            self._redraw()
            self._populate_table()

    def showEvent(self, event):
        self._ensure_visualization()
        super().showEvent(event)

    def use_active_cif(self, checked=False, *, silent=False):
        structure = getattr(self.main_window, "reference_structure", None)
        if not isinstance(structure, dict):
            if not silent:
                QMessageBox.information(
                    self,
                    "No active CIF",
                    "Import a CIF in the CIF & unit-cell task, or use Import another CIF here.",
                )
            return
        self.structure = deepcopy(structure)
        self._update_structure_status("Active CIF")

    def import_cif(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Import CIF for simulation", "", "CIF files (*.cif)")
        if not filename:
            return
        try:
            self.structure = load_cif(filename)
        except Exception as exc:
            QMessageBox.critical(self, "CIF import failed", str(exc))
            return
        self._update_structure_status(str(Path(filename)))

    def _update_structure_status(self, source: str):
        structure = self.structure or {}
        self.structure_status.setText(
            f"{structure.get('data_name', 'CIF structure')} | {structure.get('formula', 'formula not reported')} | "
            f"{structure.get('space_group_name', structure.get('space_group', 'space group not reported'))} | "
            f"{len(structure.get('atoms', []))} expanded atom(s) | Source: {source}"
        )

    def is_running(self):
        return bool(self._thread is not None and self._thread.isRunning())

    def calculate(self):
        if self.is_running():
            return
        if not self.structure:
            QMessageBox.information(self, "No CIF", "Select an active CIF or import a CIF first.")
            return
        self._ensure_visualization()
        if self.minimum.value() >= self.maximum.value():
            QMessageBox.warning(self, "Invalid range", "2theta minimum must be below the maximum.")
            return
        settings = {
            "wavelength_angstrom": self.wavelength.value(),
            "two_theta_min": self.minimum.value(),
            "two_theta_max": self.maximum.value(),
            "cutoff_percent": self.cutoff.value(),
            "fwhm_deg": self.fwhm.value(),
            "eta": self.eta.value(),
            "use_gpu": self.use_gpu.isChecked(),
        }
        thread = QThread(self)
        worker = _CifSimulationWorker(deepcopy(self.structure), settings)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._calculation_finished)
        worker.failed.connect(self._calculation_failed)
        worker.finished.connect(lambda *_: thread.quit())
        worker.failed.connect(lambda *_: thread.quit())
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        self.calculate_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Calculating crystallographic reflections...")
        thread.start()

    def _calculation_finished(self, result: dict):
        self.pattern = list(result.get("pattern", []))
        self.profile_x = np.asarray(result.get("profile_x", []), dtype=float)
        self.profile_y = np.asarray(result.get("profile_y", []), dtype=float)
        self._redraw()
        self._populate_table()
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat(f"Calculated {len(self.pattern)} reflection group(s)")
        self.export_button.setEnabled(bool(self.pattern))
        self.main_window.statusBar().showMessage(
            f"Simulated PXRD calculated from CIF: {len(self.pattern)} reflection group(s)."
        )

    def _calculation_failed(self, trace: str):
        message = trace.strip().splitlines()[-1] if trace.strip() else "Unknown CIF simulation error"
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Calculation failed")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("CIF PXRD simulation failed")
        box.setText(message)
        box.setDetailedText(trace)
        box.exec()

    def _thread_finished(self):
        self._thread = None
        self._worker = None
        self.calculate_button.setEnabled(True)

    def _redraw(self):
        self._ensure_visualization()
        if self.plot is None:
            return
        self.plot.clear()
        if not self.pattern:
            return
        theme = THEMES[DEFAULT_THEME_NAME]
        self.plot.plot(
            self.profile_x,
            self.profile_y,
            pen=pg.mkPen(theme["gold"], width=2.2),
            name="Broadened simulation preview",
        )
        stick_x, stick_y = [], []
        for peak in self.pattern:
            position = float(peak["two_theta"])
            intensity = float(peak.get("intensity", 0.0))
            stick_x.extend((position, position, np.nan))
            stick_y.extend((0.0, intensity, np.nan))
        self.plot.plot(
            stick_x,
            stick_y,
            connect="finite",
            pen=pg.mkPen(theme["plot_axis"], width=1.0),
            name="CIF reflection sticks",
        )
        self.plot.enableAutoRange()

    def _populate_table(self):
        self._ensure_visualization()
        if self.table is None:
            return
        self.table.setRowCount(len(self.pattern))
        structure_name = (self.structure or {}).get("data_name", "CIF")
        for row_index, peak in enumerate(self.pattern):
            equivalent = peak.get("equivalent_hkls") or []
            values = (
                f"{float(peak.get('two_theta', 0.0)):.8g}",
                f"{float(peak.get('d_spacing', 0.0)):.8g}",
                peak.get("hkl_label") or str(peak.get("hkl", "")),
                f"{float(peak.get('intensity', 0.0)):.8g}",
                str(peak.get("multiplicity", "")),
                "; ".join(str(tuple(row)) for row in equivalent),
                structure_name,
            )
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(str(value)))

    def export(self):
        if not self.pattern:
            return
        name = (self.structure or {}).get("data_name", "cif").replace(" ", "_")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export simulated PXRD",
            f"{name}_simulated_pxrd.txt",
            "Text data (*.txt)",
        )
        if not filename:
            return
        base = Path(filename).with_suffix(".txt")
        reflection_path = base.with_name(base.stem + "_reflections.txt")
        try:
            write_columns_txt(
                base,
                {
                    "two_theta_deg": self.profile_x,
                    "simulated_relative_intensity_percent": self.profile_y,
                },
                title="CIF simulated PXRD broadened preview",
                metadata={
                    "wavelength_angstrom": self.wavelength.value(),
                    "display_fwhm_deg": self.fwhm.value(),
                    "pseudo_voigt_eta": self.eta.value(),
                    "scientific_boundary": "Display broadening only; no measured data were fitted.",
                },
            )
            write_table_txt(
                reflection_path,
                self.pattern,
                title="CIF calculated powder reflections",
                metadata={
                    "wavelength_angstrom": self.wavelength.value(),
                    "two_theta_min_deg": self.minimum.value(),
                    "two_theta_max_deg": self.maximum.value(),
                    "intensity_cutoff_percent": self.cutoff.value(),
                },
            )
        except Exception as exc:
            QMessageBox.critical(self, "Simulation export failed", str(exc))
            return
        self.main_window.statusBar().showMessage(
            f"Exported simulated profile and reflection table: {base.name}, {reflection_path.name}."
        )

    def refresh_for_selected_dataset(self):
        active = getattr(self.main_window, "reference_structure", None)
        if self.structure is None and isinstance(active, dict):
            self.use_active_cif(silent=True)

    def run_analysis(self):
        self.calculate()

    def validate_inputs(self):
        return (self.structure is not None, "Select or import a CIF first." if self.structure is None else "")

    def apply_theme(self, theme_name):
        self._theme_name = theme_name
        if self.plot is None:
            return
        theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME_NAME])
        self.plot.setBackground(theme["plot_background"])
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def get_state(self):
        return {
            "structure": deepcopy(self.structure),
            "pattern": deepcopy(self.pattern),
            "profile_x": self.profile_x.tolist(),
            "profile_y": self.profile_y.tolist(),
            "settings": {
                "wavelength": self.wavelength.value(),
                "minimum": self.minimum.value(),
                "maximum": self.maximum.value(),
                "cutoff": self.cutoff.value(),
                "fwhm": self.fwhm.value(),
                "eta": self.eta.value(),
                "use_gpu": self.use_gpu.isChecked(),
            },
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        structure = state.get("structure")
        self.structure = deepcopy(structure) if isinstance(structure, dict) else None
        self.pattern = deepcopy(state.get("pattern")) if isinstance(state.get("pattern"), list) else []
        self.profile_x = np.asarray(state.get("profile_x", []), dtype=float)
        self.profile_y = np.asarray(state.get("profile_y", []), dtype=float)
        settings = state.get("settings") or {}
        for widget, key, default in (
            (self.wavelength, "wavelength", 1.5406),
            (self.minimum, "minimum", 3.0),
            (self.maximum, "maximum", 80.0),
            (self.cutoff, "cutoff", 0.1),
            (self.fwhm, "fwhm", 0.12),
            (self.eta, "eta", 0.5),
        ):
            widget.setValue(float(settings.get(key, default)))
        self.use_gpu.setChecked(bool(settings.get("use_gpu", True)))
        if self.structure:
            self._update_structure_status("Saved project")
        self._redraw()
        self._populate_table()
        self.export_button.setEnabled(bool(self.pattern))
