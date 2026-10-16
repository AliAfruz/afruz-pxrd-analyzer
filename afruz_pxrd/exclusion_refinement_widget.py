from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .context_copy import install_label_copy_menu, install_table_copy_menu
from .exclusion_regions import (
    ExclusionRegion,
    attach_exclusion_audit,
    filter_arrays,
    inclusion_mask,
    merge_regions,
)
from .models import Dataset
from .refinement_export import export_refinement_txt_bundle
from .refinement_background import BACKGROUND_KEY, subset_background
from .rietveld_widget import RietveldRefinementWidget
from .sequential_refinement_workspace import SequentialRefinementWorkspace
from .text_export import write_table_txt
from .whole_pattern_widget import WholePatternRefinementWidget
from .widgets import NoWheelDoubleSpinBox
from .version import APP_RELEASE, APP_NAME


class _InvestigationMainWindowProxy:
    """Delegate to MainWindow while isolating investigation datasets/results."""

    def __init__(self, target, record_callback: Callable):
        self._target = target
        self._record_callback = record_callback
        self.filtered_dataset: Dataset | None = None

    def selected_dataset(self):
        return self.filtered_dataset or self._target.selected_dataset()

    def _record_scientific_result(self, stage, dataset_uid, payload, *, reason=""):
        self._record_callback(stage, dataset_uid, payload, reason)

    def __getattr__(self, name):
        return getattr(self._target, name)


class ExclusionRegionEditor(QGroupBox):
    regionsChanged = Signal(object)

    def __init__(self, dataset_provider: Callable[[], Dataset | None], parent=None):
        super().__init__("Excluded peaks and 2theta regions", parent)
        self._dataset_provider = dataset_provider
        self._regions_by_uid: dict[str, list[dict]] = {}
        self._active_uid: str | None = None
        self._plot = None
        self._selector = pg.LinearRegionItem(
            values=(20.0, 21.0),
            movable=True,
            brush=pg.mkBrush(242, 183, 70, 45),
            pen=pg.mkPen(242, 183, 70, 210, width=1.4),
        )
        self._selector.setZValue(30)
        self._build_ui()

    def _spin(self, value: float) -> NoWheelDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(-360.0, 360.0)
        spin.setDecimals(5)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        return spin

    def _build_ui(self):
        layout = QVBoxLayout(self)
        note = QLabel(
            "Move the amber selector on the profile, then exclude that interval. "
            "For a single unwanted peak, specify its center and half-width. Excluded points receive no "
            "influence in the investigation fit; the original dataset and production refinements are unchanged."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        install_label_copy_menu(note)
        layout.addWidget(note)

        controls = QGridLayout()
        self.start = self._spin(20.0)
        self.end = self._spin(21.0)
        self.center = self._spin(20.5)
        self.half_width = self._spin(0.15)
        self.half_width.setRange(0.0001, 30.0)
        self.half_width.setValue(0.15)
        self.label = QLineEdit("Suspected impurity / excluded feature")
        self.add_region_button = QPushButton("Exclude selected region")
        self.add_region_button.setObjectName("primaryButton")
        self.add_peak_button = QPushButton("Exclude peak +/- half-width")
        self.use_view_button = QPushButton("Selector from visible plot range")
        self.remove_button = QPushButton("Remove selected exclusion")
        self.clear_button = QPushButton("Clear all exclusions")
        controls.addWidget(QLabel("Start (deg 2theta)"), 0, 0)
        controls.addWidget(self.start, 0, 1)
        controls.addWidget(QLabel("End (deg 2theta)"), 0, 2)
        controls.addWidget(self.end, 0, 3)
        controls.addWidget(self.add_region_button, 0, 4)
        controls.addWidget(QLabel("Peak center"), 1, 0)
        controls.addWidget(self.center, 1, 1)
        controls.addWidget(QLabel("Half-width"), 1, 2)
        controls.addWidget(self.half_width, 1, 3)
        controls.addWidget(self.add_peak_button, 1, 4)
        controls.addWidget(QLabel("Audit label"), 2, 0)
        controls.addWidget(self.label, 2, 1, 1, 3)
        controls.addWidget(self.use_view_button, 2, 4)
        controls.addWidget(self.remove_button, 3, 3)
        controls.addWidget(self.clear_button, 3, 4)
        layout.addLayout(controls)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Start (deg)", "End (deg)", "Width (deg)", "Scientific label"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setMaximumHeight(125)
        install_table_copy_menu(self.table)
        layout.addWidget(self.table)

        self.status = QLabel("No regions excluded.")
        self.status.setObjectName("mutedLabel")
        install_label_copy_menu(self.status)
        layout.addWidget(self.status)

        self.start.valueChanged.connect(self._spin_region_changed)
        self.end.valueChanged.connect(self._spin_region_changed)
        self._selector.sigRegionChanged.connect(self._selector_changed)
        self.add_region_button.clicked.connect(self.add_selected_region)
        self.add_peak_button.clicked.connect(self.add_peak_region)
        self.use_view_button.clicked.connect(self.use_visible_range)
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button.clicked.connect(self.clear_regions)

    def bind_plot(self, plot):
        if plot is None:
            return
        self._plot = plot
        try:
            plot.removeItem(self._selector)
        except Exception:
            pass
        plot.addItem(self._selector, ignoreBounds=True)

    def set_dataset(self, dataset: Dataset | None):
        self._active_uid = None if dataset is None else dataset.uid
        if dataset is not None and len(dataset.x):
            low, high = float(dataset.x[0]), float(dataset.x[-1])
            width = max(0.1, 0.05 * (high - low))
            center = low + 0.35 * (high - low)
            self.start.setRange(low, high)
            self.end.setRange(low, high)
            self.center.setRange(low, high)
            self._set_selector(center - width / 2.0, center + width / 2.0)
        self._populate()

    def _set_selector(self, start: float, end: float):
        start, end = sorted((float(start), float(end)))
        self.start.blockSignals(True)
        self.end.blockSignals(True)
        self.start.setValue(start)
        self.end.setValue(end)
        self.start.blockSignals(False)
        self.end.blockSignals(False)
        self.center.setValue((start + end) / 2.0)
        self._selector.setRegion((start, end))

    def _spin_region_changed(self, *_args):
        start, end = sorted((self.start.value(), self.end.value()))
        self.center.setValue((start + end) / 2.0)
        self._selector.setRegion((start, end))

    def _selector_changed(self):
        start, end = self._selector.getRegion()
        self.start.blockSignals(True)
        self.end.blockSignals(True)
        self.center.blockSignals(True)
        self.start.setValue(float(start))
        self.end.setValue(float(end))
        self.center.setValue((float(start) + float(end)) / 2.0)
        self.start.blockSignals(False)
        self.end.blockSignals(False)
        self.center.blockSignals(False)

    def _current_rows(self) -> list[dict]:
        if self._active_uid is None:
            return []
        return list(self._regions_by_uid.get(self._active_uid, []))

    def regions(self) -> list[ExclusionRegion]:
        return merge_regions(self._current_rows())

    def add_selected_region(self):
        self._append(self.start.value(), self.end.value(), self.label.text())

    def add_peak_region(self):
        center = self.center.value()
        half = self.half_width.value()
        self._append(center - half, center + half, self.label.text() or "Excluded peak")

    def _append(self, start: float, end: float, label: str):
        if self._active_uid is None:
            QMessageBox.information(self, "No dataset", "Select a diffraction dataset first.")
            return
        try:
            rows = merge_regions(
                [*self._current_rows(), ExclusionRegion(start, end, label or "Excluded region")]
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid exclusion", str(exc))
            return
        self._regions_by_uid[self._active_uid] = [row.to_dict() for row in rows]
        self._populate()
        self.regionsChanged.emit(self._regions_by_uid[self._active_uid])

    def remove_selected(self):
        if self._active_uid is None:
            return
        row = self.table.currentRow()
        rows = self._current_rows()
        if 0 <= row < len(rows):
            rows.pop(row)
            self._regions_by_uid[self._active_uid] = rows
            self._populate()
            self.regionsChanged.emit(rows)

    def clear_regions(self):
        if self._active_uid is None:
            return
        self._regions_by_uid[self._active_uid] = []
        self._populate()
        self.regionsChanged.emit([])

    def use_visible_range(self):
        if self._plot is None:
            return
        x_range = self._plot.getViewBox().viewRange()[0]
        self._set_selector(float(x_range[0]), float(x_range[1]))

    def _populate(self):
        rows = self._current_rows()
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                f"{float(row['start_deg']):.6g}",
                f"{float(row['end_deg']):.6g}",
                f"{float(row.get('width_deg', float(row['end_deg']) - float(row['start_deg']))):.6g}",
                row.get("label", "Excluded region"),
            )
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(str(value)))
        dataset = self._dataset_provider()
        if dataset is None or not len(dataset.x):
            self.status.setText("No dataset selected.")
            return
        mask = inclusion_mask(dataset.x, rows)
        excluded = int(len(mask) - np.count_nonzero(mask))
        self.status.setText(
            f"{len(rows)} merged exclusion interval(s); {excluded}/{len(mask)} measured point(s) omitted."
        )

    def get_state(self) -> dict:
        return {"regions_by_uid": deepcopy(self._regions_by_uid)}

    def set_state(self, state: dict | None):
        state = state or {}
        rows = state.get("regions_by_uid")
        self._regions_by_uid = {}
        if isinstance(rows, dict):
            for uid, values in rows.items():
                try:
                    self._regions_by_uid[str(uid)] = [
                        row.to_dict() for row in merge_regions(values or [])
                    ]
                except (TypeError, ValueError):
                    self._regions_by_uid[str(uid)] = []
        self._populate()


class _ExclusionInvestigationMixin:
    investigation_label = "Exclusion-aware refinement"

    def _initialize_investigation(self, real_main_window):
        self.real_main_window = real_main_window
        self._audit_by_uid: dict[str, dict] = {}
        self._source_snapshot_by_uid: dict[str, dict] = {}
        self._exclusion_plot_items: list[object] = []
        self.investigation_records: dict[str, dict] = {}
        self.exclusion_editor = ExclusionRegionEditor(
            real_main_window.selected_dataset,
            self,
        )
        self.layout().insertWidget(2, self.exclusion_editor)
        self.exclusion_editor.regionsChanged.connect(lambda _rows: self._decorate_plot())
        self.exclusion_editor.set_dataset(real_main_window.selected_dataset())
        self.exclusion_editor.bind_plot(self.plot_widget.plot)
        self._preview_source_pattern()

    def _capture_record(self, stage, dataset_uid, payload, reason):
        self.investigation_records[str(dataset_uid)] = {
            "stage": stage,
            "payload": deepcopy(payload),
            "reason": reason,
        }

    def _filtered_dataset(self, dataset: Dataset) -> tuple[Dataset, dict]:
        x, arrays, audit = filter_arrays(
            dataset.x,
            (dataset.y_raw, dataset.y_processed),
            self.exclusion_editor.regions(),
            minimum_points=50,
        )
        raw, processed = arrays
        filtered = Dataset(
            uid=dataset.uid,
            name=dataset.name,
            x=x,
            y_raw=raw,
            y_processed=processed,
            source_path=dataset.source_path,
            visible=dataset.visible,
            metadata=deepcopy(dataset.metadata),
        )
        subset_background(dataset, filtered)
        selected_y = dataset.y_processed if self._uses_processed() and dataset.y_processed is not None else dataset.y_raw
        self._source_snapshot_by_uid[dataset.uid] = {
            "x": np.asarray(dataset.x, dtype=float).copy(),
            "y": np.asarray(selected_y, dtype=float).copy(),
        }
        return filtered, audit

    def _uses_processed(self) -> bool:
        dataset = self.real_main_window.selected_dataset()
        if dataset is not None and (dataset.metadata or {}).get(BACKGROUND_KEY) is not None:
            return False
        checkbox = getattr(self, "use_processed_check", None)
        if checkbox is None:
            checkbox = getattr(self, "use_processed", None)
        return bool(checkbox is not None and checkbox.isChecked())

    def _preview_source_pattern(self):
        if not hasattr(self, "real_main_window"):
            return
        dataset = self.real_main_window.selected_dataset()
        if dataset is None:
            return
        selected_y = (
            dataset.y_processed
            if self._uses_processed() and dataset.y_processed is not None
            else dataset.y_raw
        )
        self._source_snapshot_by_uid[dataset.uid] = {
            "x": np.asarray(dataset.x, dtype=float).copy(),
            "y": np.asarray(selected_y, dtype=float).copy(),
        }
        if dataset.uid not in self.results_by_uid:
            self.plot_widget.plot.clear()
            self.plot_widget.plot.plot(
                dataset.x,
                selected_y,
                pen=pg.mkPen(172, 186, 214, 210, width=1.25),
                name="Observed preview",
            )
            self.plot_widget.plot.enableAutoRange()
            self.plot_widget.message.setText(
                "Observed preview - choose exclusion regions, then run the investigation fit."
            )
        self._decorate_plot()

    def _run_with_exclusions(self, parent_run: Callable):
        dataset = self.real_main_window.selected_dataset()
        if dataset is None:
            QMessageBox.information(self, "No dataset", "Select an experimental dataset first.")
            return
        try:
            filtered, audit = self._filtered_dataset(dataset)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid exclusion set", str(exc))
            return
        self._audit_by_uid[dataset.uid] = audit
        saved_guide = self.peak_guides_by_uid.pop(dataset.uid, None)
        self.main_window.filtered_dataset = filtered
        try:
            parent_run()
        finally:
            self.main_window.filtered_dataset = None
            if saved_guide is not None:
                self.peak_guides_by_uid[dataset.uid] = saved_guide

    def _audit_result(self, result: dict) -> dict:
        uid = self._active_dataset_uid
        audit = self._audit_by_uid.get(uid or "", {})
        return attach_exclusion_audit(
            result,
            audit,
            method=self.investigation_label,
        )

    def _decorate_plot(self):
        if not hasattr(self, "exclusion_editor"):
            return
        plot = self.plot_widget.plot
        for item in self._exclusion_plot_items:
            try:
                plot.removeItem(item)
            except Exception:
                pass
        self._exclusion_plot_items = []
        dataset = self.real_main_window.selected_dataset()
        if dataset is None:
            self.exclusion_editor.bind_plot(plot)
            return
        rows = self.exclusion_editor.regions()
        snapshot = self._source_snapshot_by_uid.get(dataset.uid)
        if snapshot is not None and rows:
            excluded = ~inclusion_mask(snapshot["x"], rows)
            if np.any(excluded):
                excluded_points = plot.plot(
                    snapshot["x"][excluded],
                    snapshot["y"][excluded],
                    pen=None,
                    symbol="x",
                    symbolSize=7,
                    symbolPen=pg.mkPen(235, 87, 87, 230, width=1.5),
                    name="Excluded measured points",
                )
                self._exclusion_plot_items.append(excluded_points)
        for row in rows:
            band = pg.LinearRegionItem(
                values=(row.start_deg, row.end_deg),
                movable=False,
                brush=pg.mkBrush(235, 87, 87, 38),
                pen=pg.mkPen(235, 87, 87, 160, width=1.0),
            )
            band.setZValue(-15)
            plot.addItem(band)
            self._exclusion_plot_items.append(band)
        self.exclusion_editor.bind_plot(plot)

    def refresh_for_selected_dataset(self):
        super().refresh_for_selected_dataset()
        if hasattr(self, "exclusion_editor"):
            self.exclusion_editor.set_dataset(self.real_main_window.selected_dataset())
            self._preview_source_pattern()

    def export_result(self):
        dataset = self.real_main_window.selected_dataset()
        if dataset is None:
            return
        result = self.results_by_uid.get(dataset.uid)
        if not result:
            QMessageBox.information(self, "No investigation result", "Run this investigation refinement first.")
            return
        default = f"{dataset.name}_{self.investigation_label.lower().replace(' ', '_')}.txt"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export exclusion-aware refinement",
            default,
            "Text data (*.txt)",
        )
        if not filename:
            return
        try:
            exported = export_refinement_txt_bundle(
                filename,
                result,
                dataset_name=dataset.name,
                dataset_uid=dataset.uid,
                refinement_label=self.investigation_label,
            )
            region_path = Path(filename).with_suffix("").with_name(
                Path(filename).with_suffix("").name + "_excluded_regions.txt"
            )
            write_table_txt(
                region_path,
                result.get("excluded_regions", []),
                title=f"{self.investigation_label} exclusions - {dataset.name}",
                metadata={
                    "dataset_uid": dataset.uid,
                    "scientific_rule": "Intervals are closed and were omitted from optimization and reported fit statistics.",
                },
            )
        except Exception as exc:
            QMessageBox.critical(self, "Investigation export failed", str(exc))
            return
        self.real_main_window.statusBar().showMessage(
            f"Exported {exported['file_count']} refinement files plus excluded-region audit."
        )

    def _investigation_state(self, base_state: dict) -> dict:
        state = dict(base_state)
        state["exclusion_editor"] = self.exclusion_editor.get_state()
        state["investigation_records"] = deepcopy(self.investigation_records)
        return state

    def _restore_investigation_state(self, state: dict | None):
        state = state or {}
        self.exclusion_editor.set_state(state.get("exclusion_editor"))
        records = state.get("investigation_records")
        self.investigation_records = deepcopy(records) if isinstance(records, dict) else {}
        self.exclusion_editor.set_dataset(self.real_main_window.selected_dataset())


class InvestigationWholePatternWidget(_ExclusionInvestigationMixin, WholePatternRefinementWidget):
    def __init__(self, real_main_window, mode: str, parent=None):
        self.fixed_mode = str(mode)
        self.investigation_label = (
            "Exclusion-aware Pawley refinement"
            if "Pawley" in self.fixed_mode
            else "Exclusion-aware Le Bail refinement"
        )
        proxy = _InvestigationMainWindowProxy(real_main_window, self._capture_proxy_record)
        super().__init__(proxy, parent)
        proxy._record_callback = self._capture_record
        self.mode.setCurrentText(self.fixed_mode)
        self.mode.setEnabled(False)
        self.run_button.setText(f"Run {self.investigation_label}")
        headings = self.findChildren(QLabel)
        for label in headings:
            if label.objectName() == "sectionTitle":
                label.setText(self.investigation_label.title())
                break
        self._initialize_investigation(real_main_window)

    @staticmethod
    def _capture_proxy_record(*_args, **_kwargs):
        pass

    def run_refinement(self):
        self.mode.setCurrentText(self.fixed_mode)
        self._run_with_exclusions(lambda: super(InvestigationWholePatternWidget, self).run_refinement())

    def _on_finished(self, result):
        uid = self._active_dataset_uid
        saved_guide = self.peak_guides_by_uid.pop(uid, None) if uid else None
        try:
            super()._on_finished(self._audit_result(result))
        finally:
            if uid and saved_guide is not None:
                self.peak_guides_by_uid[uid] = saved_guide
        self.real_main_window.statusBar().showMessage(
            "Exclusion-aware Pawley / Le Bail investigation completed; production results were not replaced."
        )
        self._decorate_plot()

    def populate_result(self, result):
        super().populate_result(result)
        self._decorate_plot()

    def get_state(self):
        return self._investigation_state(super().get_state())

    def set_state(self, state):
        super().set_state(state)
        self.mode.setCurrentText(self.fixed_mode)
        self.mode.setEnabled(False)
        if hasattr(self, "exclusion_editor"):
            self._restore_investigation_state(state)


class InvestigationRietveldWidget(_ExclusionInvestigationMixin, RietveldRefinementWidget):
    investigation_label = "Exclusion-aware Rietveld refinement"

    def __init__(self, real_main_window, parent=None):
        proxy = _InvestigationMainWindowProxy(real_main_window, self._capture_proxy_record)
        super().__init__(proxy, parent)
        proxy._record_callback = self._capture_record
        self.run_button.setText("Run exclusion-aware Rietveld")
        for label in self.findChildren(QLabel):
            if label.objectName() == "sectionTitle":
                label.setText("Exclusion-Aware Rietveld Investigation")
                break
        self._initialize_investigation(real_main_window)

    @staticmethod
    def _capture_proxy_record(*_args, **_kwargs):
        pass

    def run_refinement(self):
        self._run_with_exclusions(lambda: super(InvestigationRietveldWidget, self).run_refinement())

    def _on_finished(self, result):
        uid = self._active_dataset_uid
        saved_guide = self.peak_guides_by_uid.pop(uid, None) if uid else None
        try:
            super()._on_finished(self._audit_result(result))
        finally:
            if uid and saved_guide is not None:
                self.peak_guides_by_uid[uid] = saved_guide
        self.real_main_window.statusBar().showMessage(
            "Exclusion-aware Rietveld investigation completed; production results were not replaced."
        )
        self._decorate_plot()

    def populate_result(self, result):
        super().populate_result(result)
        self._decorate_plot()

    def get_state(self):
        return self._investigation_state(super().get_state())

    def set_state(self, state):
        super().set_state(state)
        if hasattr(self, "exclusion_editor"):
            self._restore_investigation_state(state)


class ExclusionRefinementDialog(QDialog):
    def __init__(self, widget: QWidget, title: str, parent=None):
        super().__init__(parent)
        self.refinement_widget = widget
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.Window, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1420, 920)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(widget)

    def closeEvent(self, event):
        if getattr(self.refinement_widget, "is_running", lambda: False)():
            QMessageBox.information(
                self,
                "Refinement is running",
                "Cancel the investigation refinement and wait for it to stop before closing this window.",
            )
            event.ignore()
            return
        super().closeEvent(event)


class SequentialRefinementStudioDialog(QDialog):
    """Persistent non-modal window for the single project-owned laboratory.

    Closing only hides it: queued jobs and scientific state are never discarded
    or copied into an independent model. MainWindow remains the save/lifecycle
    owner, and its existing shutdown guard cancels the same worker.
    """

    def __init__(self, workspace, parent=None):
        super().__init__(parent if isinstance(parent, QWidget) else None)
        self.refinement_widget = workspace
        self.has_been_shown = False
        self.setWindowTitle(f"Step-wise Refinement Studio — {APP_RELEASE}")
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowSystemMenuHint
                            | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.setModal(False)
        self.setSizeGripEnabled(True)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(1680, 960)
        self.setMinimumSize(1040, 640)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(workspace)
        workspace.show()

    def reject(self):
        # Escape is equivalent to the title-bar close, not scientific reset.
        self.hide()

    def closeEvent(self, event):
        self.hide()
        event.accept()


class ExclusionRefinementHubWidget(QWidget):
    """Central sequential lab plus the three original isolated investigations."""

    METHODS = {
        "pawley": "Pawley decomposition",
        "lebail": "Le Bail-style extraction",
        "rietveld": "Rietveld",
    }

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._dialogs: dict[str, ExclusionRefinementDialog] = {}
        self._widgets: dict[str, QWidget] = {}
        self._pending_state: dict[str, dict] = {}
        self.studio_dialog = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        heading = QLabel("Step-wise Refinement Studio")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        description = QLabel(
            "Open the laboratory in its own resizable window. Compare CIF candidates one by one, "
            "retain selected results for later steps, and label excluded regions on a large XRD canvas. "
            "The established refinement workspaces remain unchanged."
        )
        description.setWordWrap(True)
        description.setObjectName("mutedLabel")
        install_label_copy_menu(description)
        layout.addWidget(description)

        self.sequential_workspace = SequentialRefinementWorkspace(self.main_window, self)
        self.sequential_workspace.hide()
        self.open_studio_button = QPushButton("Open Refinement Studio — separate window")
        self.open_studio_button.setObjectName("primaryButton")
        self.open_studio_button.setMinimumHeight(52)
        layout.addWidget(self.open_studio_button)
        self.studio_status = QLabel("Your CIF library, steps and run history remain part of the current .AFZ project.")
        self.studio_status.setWordWrap(True)
        self.studio_status.setObjectName("mutedLabel")
        layout.addWidget(self.studio_status)
        window_note = QLabel("Closing the Studio hides it; active and queued runs can continue while the main application stays open. Reopen it here to review progress.")
        window_note.setWordWrap(True)
        window_note.setObjectName("mutedLabel")
        layout.addWidget(window_note)
        layout.addStretch(1)

        legacy = QGroupBox("Original separate investigation windows (preserved)")
        button_row = QHBoxLayout(legacy)
        self.pawley_button = QPushButton("Open Pawley investigation window")
        self.lebail_button = QPushButton("Open Le Bail investigation window")
        self.rietveld_button = QPushButton("Open Rietveld investigation window")
        for button in (self.pawley_button, self.lebail_button, self.rietveld_button):
            button_row.addWidget(button)
        layout.addWidget(legacy)

        warning = QLabel(
            "Scientific reporting rule: publish the full-range fit together with every excluded interval and its "
            "reason. Never hide unexplained peaks only to improve R factors. Confirm a proposed impurity with an "
            "explicit phase model, complementary measurements, or both."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #e7b84b; font-weight: 600;")
        install_label_copy_menu(warning)
        layout.addWidget(warning)

        self.pawley_button.clicked.connect(lambda: self.open_window("pawley"))
        self.lebail_button.clicked.connect(lambda: self.open_window("lebail"))
        self.rietveld_button.clicked.connect(lambda: self.open_window("rietveld"))
        self.open_studio_button.clicked.connect(self.open_studio)

    def open_studio(self):
        if self.studio_dialog is None:
            self.studio_dialog = SequentialRefinementStudioDialog(self.sequential_workspace, self.main_window)
        self.sequential_workspace.refresh_for_selected_dataset()
        if not self.studio_dialog.has_been_shown:
            self.studio_dialog.has_been_shown = True
            self.studio_dialog.showMaximized()
        else:
            self.studio_dialog.show()
        self.studio_dialog.raise_()
        self.studio_dialog.activateWindow()
        return self.studio_dialog

    def _create_widget(self, key: str):
        if key == "pawley":
            widget = InvestigationWholePatternWidget(
                self.main_window,
                "Pawley decomposition",
            )
        elif key == "lebail":
            widget = InvestigationWholePatternWidget(
                self.main_window,
                "Le Bail-style extraction",
            )
        elif key == "rietveld":
            widget = InvestigationRietveldWidget(self.main_window)
        else:
            raise KeyError(key)
        pending = self._pending_state.get(key)
        if pending:
            widget.set_state(pending)
        return widget

    def open_window(self, key: str):
        dialog = self._dialogs.get(key)
        if dialog is None:
            widget = self._create_widget(key)
            title = f"{self.METHODS[key]} - Phase-Exclusion Investigation - {APP_NAME}"
            dialog = ExclusionRefinementDialog(widget, title, self.main_window)
            cinematic_ui = getattr(self.main_window, "cinematic_ui", None)
            if cinematic_ui is not None:
                cinematic_ui.decorate(dialog)
            self._widgets[key] = widget
            self._dialogs[key] = dialog
            dialog.finished.connect(lambda _code, name=key: self._dialog_closed(name))
        widget = self._widgets[key]
        widget.refresh_for_selected_dataset()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _dialog_closed(self, key: str):
        widget = self._widgets.pop(key, None)
        if widget is not None:
            self._pending_state[key] = widget.get_state()
        self._dialogs.pop(key, None)

    def refresh_for_selected_dataset(self):
        self.sequential_workspace.refresh_for_selected_dataset()
        for widget in self._widgets.values():
            widget.refresh_for_selected_dataset()

    def run_analysis(self):
        self.open_studio()

    def validate_inputs(self):
        # This workspace action opens a window; numerical commands inside it
        # still validate the selected dataset, CIF and fit-domain masks.
        return True, ""

    def apply_theme(self, theme_name):
        self.sequential_workspace.apply_theme(theme_name)
        for widget in self._widgets.values():
            widget.apply_theme(theme_name)

    def get_state(self):
        state = deepcopy(self._pending_state)
        for key, widget in self._widgets.items():
            state[key] = widget.get_state()
        return {
            "methods": state,
            "sequential_workspace": self.sequential_workspace.get_state(),
        }

    def set_state(self, state):
        methods = (state or {}).get("methods") if isinstance(state, dict) else None
        self._pending_state = deepcopy(methods) if isinstance(methods, dict) else {}
        self.sequential_workspace.set_state(
            (state or {}).get("sequential_workspace", {})
            if isinstance(state, dict)
            else {}
        )
        for key, widget in self._widgets.items():
            if key in self._pending_state:
                widget.set_state(self._pending_state[key])
