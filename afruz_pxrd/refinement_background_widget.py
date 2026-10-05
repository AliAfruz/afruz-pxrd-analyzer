"""One shared full background engine for regular and step-wise refiners."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QCheckBox, QHeaderView,
)

from .advanced_background import BACKGROUND_METHODS, BackgroundParameters, detect_background
from .background_plot import AdvancedBackgroundPlotWidget
from .context_copy import install_label_copy_menu, install_table_copy_menu
from .refinement_background import BACKGROUND_KEY, array_fingerprint, background_selection, selected_background
from .widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox
from .text_export import ensure_txt_path, write_columns_txt, write_mapping_txt


class _BackgroundSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object, str)


class _BackgroundJob(QRunnable):
    def __init__(self, x, y, parameters):
        super().__init__()
        self.signals = _BackgroundSignals()
        self.x, self.y = np.asarray(x).copy(), np.asarray(y).copy()
        self.parameters = deepcopy(parameters)

    def run(self):
        try:
            result = detect_background(self.x, self.y, self.parameters, progress_callback=self.signals.progress.emit)
            self.signals.finished.emit(result, "")
        except Exception as exc:
            self.signals.finished.emit(None, str(exc))


class RefinementBackgroundDialog(QDialog):
    backgroundChanged = Signal()

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = getattr(main_window, "_target", main_window)
        self.dataset = self.main_window.selected_dataset()
        self.x, self.raw = np.asarray(self.dataset.x).copy(), np.asarray(self.dataset.y_raw).copy()
        self.source_hash = array_fingerprint(self.x, self.raw)
        self._job = None
        self._result = None
        self.setWindowTitle(f"Refinement background engine — {self.dataset.name}")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(1320, 850)
        self.setMinimumSize(960, 620)
        root = QVBoxLayout(self)
        note = QLabel("Shared for this dataset: Pawley, Le Bail and Rietveld, including step-wise runs. "
                      "Raw observations stay unchanged; refinement fits a smooth correction on top of your chosen baseline.")
        note.setWordWrap(True)
        install_label_copy_menu(note)
        root.addWidget(note)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.controls = QWidget()
        form = QFormLayout(self.controls)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.method = NoWheelComboBox()
        self.method.addItems(BACKGROUND_METHODS)
        form.addRow("Method", self.method)
        self.values = {}
        defaults = BackgroundParameters()
        fields = (
            ("smoothness", "Smoothness", 1, 100, 2),
            ("asymmetry", "Asymmetry", 0.00001, 0.49, 5),
            ("iterations", "Iterations", 5, 250, None),
            ("window_degrees", "Window (degrees)", 0.02, 80, 3),
            ("percentile", "Percentile", 1, 49, 2),
            ("polynomial_order", "Polynomial order", 1, 12, None),
            ("protection_sigma", "Protection sigma", 1.5, 20, 2),
            ("protection_width_multiplier", "Protection width multiplier", 0.5, 10, 2),
        )
        for key, label, low, high, decimals in fields:
            control = NoWheelSpinBox() if decimals is None else NoWheelDoubleSpinBox()
            control.setRange(low, high)
            if decimals is not None:
                control.setDecimals(decimals)
            control.setValue(getattr(defaults, key))
            self.values[key] = control
            form.addRow(label, control)
            control.valueChanged.connect(self._invalidate_preview)
        self.protect = QCheckBox("Protect detected peaks")
        self.protect.setChecked(True)
        self.clip = QCheckBox("Clip negatives in preview only")
        self.clip.setToolTip("This affects the corrected preview only. Refinement always fits unchanged raw measurements, without clipping.")
        form.addRow(self.protect)
        form.addRow(self.clip)
        self.method.currentIndexChanged.connect(self._invalidate_preview)
        self.protect.toggled.connect(self._invalidate_preview)
        self.clip.toggled.connect(self._invalidate_preview)
        self.preview_button = QPushButton("Preview background")
        self.preview_button.setObjectName("primaryButton")
        self.import_button = QPushButton("Use main Background preview")
        self.import_button.setToolTip("Load the raw-unit curve and diagnostics already previewed in Pattern Preparation.")
        form.addRow(self.preview_button)
        form.addRow(self.import_button)
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setMinimumHeight(160)
        form.addRow("Diagnostics", self.diagnostics)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls)
        scroll.setMinimumWidth(310)
        splitter.addWidget(scroll)
        self.plot = AdvancedBackgroundPlotWidget()
        self.plot.peakAddRequested.connect(self._add_main_peak)
        self.plot.setMinimumWidth(360)
        splitter.addWidget(self.plot)
        splitter.setSizes([370, 950])
        root.addWidget(splitter, 1)
        self.candidate_table = QTableWidget(0, 8)
        self.candidate_table.setHorizontalHeaderLabels(["Method", "Score", "Negative %", "Severe negative %", "Contact %", "Roughness", "Edge mismatch", "Background area %"])
        self.candidate_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.candidate_table.setMaximumHeight(150)
        install_table_copy_menu(self.candidate_table)
        root.addWidget(self.candidate_table)
        footer = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.status = QLabel("Preview a method, review it, then use it for future refinements.")
        self.status.setWordWrap(True)
        install_label_copy_menu(self.status)
        self.use_button = QPushButton("Use for all refinements")
        self.use_button.setObjectName("primaryButton")
        self.use_button.setEnabled(False)
        self.clear_button = QPushButton("Restore ordinary fitted background")
        self.export_button = QPushButton("Export background TXT…")
        self.export_button.setEnabled(False)
        close = QPushButton("Close")
        footer.addWidget(self.progress)
        footer.addWidget(self.status, 1)
        footer.addWidget(self.use_button)
        root.addLayout(footer)
        bottom = QHBoxLayout()
        bottom.addWidget(self.clear_button)
        bottom.addWidget(self.export_button)
        bottom.addStretch(1)
        bottom.addWidget(close)
        root.addLayout(bottom)
        self.preview_button.clicked.connect(self.preview_background)
        self.import_button.clicked.connect(self.import_main_preview)
        self.use_button.clicked.connect(self.use_for_refinements)
        self.clear_button.clicked.connect(self.clear_background)
        self.export_button.clicked.connect(self.export_background)
        close.clicked.connect(self.close)
        saved = self.dataset.metadata.get(BACKGROUND_KEY)
        if isinstance(saved, dict):
            try:
                selected_background(self.dataset)
                self._accept_result(saved)
            except (ValueError, TypeError, KeyError) as exc:
                self.status.setText(str(exc))

    def parameters(self):
        return BackgroundParameters(method=self.method.currentText(),
            **{key: value.value() for key, value in self.values.items()},
            peak_protection=self.protect.isChecked(), clip_negative=self.clip.isChecked()).normalized()

    def _invalidate_preview(self, *_):
        self._result = None
        if hasattr(self, "use_button"):
            self.use_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.status.setText("Parameters changed. Preview again before applying.")

    def preview_background(self):
        if self._job is not None:
            return
        self._job = _BackgroundJob(self.x, self.raw, self.parameters())
        self.controls.setEnabled(False)
        self.use_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.status.setText("Preparing peak-protected background…")
        self._job.signals.progress.connect(self._progress)
        self._job.signals.finished.connect(self._ready)
        QThreadPool.globalInstance().start(self._job)

    @Slot(int, int, str)
    def _progress(self, done, total, text):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.status.setText(text)

    @Slot(object, str)
    def _ready(self, result, error):
        self._job = None
        self.controls.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        if error:
            self.status.setText(error)
            self._result = None
            return
        try:
            self._accept_result(result)
        except (ValueError, TypeError, KeyError) as exc:
            self._result = None
            self.use_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.status.setText(f"Invalid background preview: {exc}")

    def _accept_result(self, result):
        params = BackgroundParameters(**(result.get("parameters") or {})).normalized()
        controls = [self.method, *self.values.values(), self.protect, self.clip]
        for control in controls:
            control.blockSignals(True)
        self.method.setCurrentText(params.method)
        for key, control in self.values.items():
            control.setValue(getattr(params, key))
        self.protect.setChecked(params.peak_protection)
        self.clip.setChecked(params.clip_negative)
        for control in controls:
            control.blockSignals(False)
        self._result = deepcopy(result)
        base = np.asarray(result["background"], dtype=float)
        corrected = self.raw - base
        if params.clip_negative:
            corrected = np.maximum(corrected, 0)
        self.plot.update_result(self.x, self.raw, base, corrected, np.asarray(result.get("protected_mask", np.zeros(len(base))), dtype=bool))
        diagnostics = result.get("diagnostics") or {}
        self.diagnostics.setPlainText("\n".join(f"{key}: {value}" for key, value in diagnostics.items() if key != "candidate_scores"))
        scores = diagnostics.get("candidate_scores") or []
        self.candidate_table.setRowCount(len(scores))
        keys = ("method", "score", "negative_fraction", "severe_negative_fraction", "contact_fraction", "roughness_ratio", "edge_mismatch_ratio", "background_area_fraction")
        for row, values in enumerate(scores):
            for column, key in enumerate(keys):
                value = values.get(key, "")
                if key.endswith("fraction") and isinstance(value, (int, float)):
                    value *= 100
                self.candidate_table.setItem(row, column, QTableWidgetItem(f"{value:.5g}" if isinstance(value, (float, int)) else str(value)))
        self.use_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.status.setText(f"{diagnostics.get('selected_method', params.method)} · {diagnostics.get('quality', 'Review the baseline')}. Ready to use for future runs.")

    def import_main_preview(self):
        record = getattr(self.main_window, "background_results", {}).get(self.dataset.uid)
        try:
            if not isinstance(record, dict):
                raise ValueError("Preview a background in Pattern Preparation first, or preview a method here.")
            if record.get("source_pattern_sha256", self.source_hash) != self.source_hash:
                raise ValueError("The main background preview belongs to an older raw pattern. Preview it again.")
            background_selection(self.dataset, record)
            self._accept_result(record)
        except (ValueError, TypeError, KeyError) as exc:
            QMessageBox.warning(self, "Background preview unavailable", str(exc))

    def _current_dataset(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None or dataset.uid != self.dataset.uid or array_fingerprint(dataset.x, dataset.y_raw) != self.source_hash:
            raise ValueError("The selected raw dataset changed while this window was open. Reopen the background engine for the current dataset.")
        return dataset

    def _changed(self):
        callback = getattr(self.main_window, "_mark_project_dirty", None)
        if callable(callback):
            callback()
        self.backgroundChanged.emit()

    def use_for_refinements(self):
        if self._result is None:
            return
        try:
            dataset = self._current_dataset()
            dataset.metadata[BACKGROUND_KEY] = background_selection(dataset, self._result)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot choose background", str(exc))
            return
        self._changed()
        self.status.setText("Chosen for all future refinements of this dataset. Queued runs and earlier results keep their frozen backgrounds. Save .AFZ to retain it.")

    def clear_background(self):
        try:
            self._current_dataset().metadata.pop(BACKGROUND_KEY, None)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot change background", str(exc))
            return
        self._changed()
        self.status.setText("Ordinary polynomial background restored for future runs. Existing queued jobs/results are unchanged.")

    def _add_main_peak(self, x, y):
        callback = getattr(self.main_window, "_context_manual_peak_requested", None)
        if not callable(callback):
            return
        try:
            self._current_dataset()
        except ValueError as exc:
            QMessageBox.warning(self, "Dataset changed", str(exc))
            return
        callback(x, y, origin="Refinement background plot right-click")

    def export_background(self):
        if self._result is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export chosen starting background", "refinement_background.txt", "Text files (*.txt)")
        if not filename:
            return
        try:
            path = ensure_txt_path(filename)
            base = np.asarray(self._result["background"], dtype=float)
            corrected = self.raw - base
            if self.parameters().clip_negative:
                corrected = np.maximum(corrected, 0)
            write_columns_txt(path, {"two_theta_deg": self.x, "raw_intensity": self.raw,
                "starting_background_intensity": base, "corrected_preview_intensity": corrected,
                "protected_peak_region": np.asarray(self._result.get("protected_mask", np.zeros(len(base))), dtype=int)},
                title=f"Refinement starting background — {self.dataset.name}",
                metadata={"source_pattern_sha256": self.source_hash, "method": self.method.currentText(),
                          "preview_negative_clipping": self.parameters().clip_negative,
                          "refinement_observation_clipping": False,
                          "refinement_rule": "calculated = phase profiles + chosen background + fitted correction; raw observations unchanged"})
            write_mapping_txt(path.with_name(path.stem + "_diagnostics.txt"),
                {"parameters": self._result.get("parameters"), "diagnostics": self._result.get("diagnostics")},
                title="Starting background parameters and diagnostics")
            self.status.setText(f"Exported {path.name} and {path.stem}_diagnostics.txt.")
        except Exception as exc:
            QMessageBox.warning(self, "Background export failed", str(exc))


class RefinementBackgroundControl(QWidget):
    changed = Signal()

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = getattr(main_window, "_target", main_window)
        self.dialogs = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton("Background engine…")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setObjectName("mutedLabel")
        install_label_copy_menu(self.status)
        layout.addWidget(self.button)
        layout.addWidget(self.status)
        self.button.clicked.connect(self.open_engine)

    def refresh(self):
        dataset = self.main_window.selected_dataset()
        model = None if dataset is None else dataset.metadata.get(BACKGROUND_KEY)
        try:
            if dataset is not None:
                selected_background(dataset)
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            self.button.setEnabled(True)
            return
        method = ((model or {}).get("diagnostics") or {}).get("selected_method") or ((model or {}).get("parameters") or {}).get("method")
        self.status.setText(f"Shared: {method or 'chosen baseline'} + fitted correction · raw input" if model else "Ordinary fitted background; no chosen baseline.")
        self.button.setEnabled(dataset is not None)

    def open_engine(self):
        if self.main_window.selected_dataset() is None:
            QMessageBox.information(self, "No dataset", "Select an experimental dataset first.")
            return None
        dialog = RefinementBackgroundDialog(self.main_window, self)
        self.dialogs.append(dialog)
        dialog.destroyed.connect(lambda *_args, owned=dialog: self.dialogs.remove(owned) if owned in self.dialogs else None)
        dialog.backgroundChanged.connect(lambda: refresh_background_controls(self.main_window, self))
        decoration = getattr(self.main_window, "cinematic_ui", None)
        if decoration is not None:
            decoration.decorate(dialog)
        dialog.show()
        return dialog


def refresh_background_controls(main_window, source=None):
    """Synchronize every open regular/Studio panel without changing frozen results."""
    controls = [source] if source is not None else []
    finder = getattr(main_window, "findChildren", None)
    if callable(finder):
        controls.extend(finder(RefinementBackgroundControl))
    seen = set()
    for control in controls:
        if id(control) in seen:
            continue
        seen.add(id(control))
        control.refresh()
        control.changed.emit()
