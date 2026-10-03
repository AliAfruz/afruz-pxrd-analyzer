from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from .context_copy import CopyablePlotWidget, install_label_copy_menu
from .theme import DEFAULT_THEME_NAME, THEMES


class PhaseRevolutionPlotWidget(QWidget):
    peakAddRequested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        theme = THEMES[DEFAULT_THEME_NAME]
        self.plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="Unknown Phase residual pattern",
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.plot.setLabel("bottom", "2θ", units="degrees")
        self.plot.setLabel("left", "Intensity")
        self.message = QLabel("Create an unknown residual pattern to begin.")
        self.message.setObjectName("mutedLabel")
        self.message.setWordWrap(False)
        self.message.setMinimumWidth(0)
        self.message.setMaximumHeight(24)
        self.message.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        install_label_copy_menu(self.message)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.message)
        self._manual_add_enabled = False
        self.plot.enable_manual_peak_context(True, label="Add manual master peak here")
        self.plot.manualPeakRequested.connect(
            lambda x, _y: self.peakAddRequested.emit(float(x))
        )
        self.plot.scene().sigMouseClicked.connect(self._scene_clicked)

    def set_manual_add_enabled(self, enabled: bool):
        self._manual_add_enabled = bool(enabled)
        self.plot.setCursor(
            QCursor(Qt.CrossCursor if self._manual_add_enabled else Qt.ArrowCursor)
        )
        if self._manual_add_enabled:
            self.message.setText("Click the residual plot to add a manual master peak.")

    def _scene_clicked(self, event):
        if not self._manual_add_enabled or event.button() != Qt.LeftButton:
            return
        view_box = self.plot.getPlotItem().vb
        if not self.plot.sceneBoundingRect().contains(event.scenePos()):
            return
        position = view_box.mapSceneToView(event.scenePos())
        self.peakAddRequested.emit(float(position.x()))

    def apply_theme(self, theme_name: str):
        theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME_NAME])
        self.plot.setBackground(theme["plot_background"])
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def set_data(self, residual: dict | None, peaks: list[dict] | None = None):
        self.plot.clear()
        if not residual:
            self.message.setText("Create an unknown residual pattern to begin.")
            return
        theme = THEMES[DEFAULT_THEME_NAME]
        x = np.asarray(residual.get("x", []), dtype=float)
        observed = np.asarray(residual.get("observed", []), dtype=float)
        known = np.asarray(residual.get("known_phase_calculated", []), dtype=float)
        background = np.asarray(residual.get("background", []), dtype=float)
        signed = np.asarray(residual.get("signed_residual", []), dtype=float)
        if not len(x):
            return
        self.plot.plot(x, observed, pen=pg.mkPen(theme["plot_axis"], width=1), name="Observed")
        if len(known) == len(x) and np.any(known):
            self.plot.plot(
                x,
                known + background,
                pen=pg.mkPen(theme["gold"], width=1.6),
                name="Known phases + background",
            )
        span = max(float(np.ptp(observed)), 1.0)
        offset = float(np.min(observed)) - 0.18 * span
        self.plot.plot(
            x,
            signed + offset,
            pen=pg.mkPen(theme["gold_bright"], width=1.2),
            name="Signed unknown residual",
        )
        for peak in peaks or []:
            position = float(peak.get("two_theta_deg", 0.0))
            if not peak.get("use", True):
                continue
            marker_pen = pg.mkPen(
                theme["gold_bright"] if peak.get("manual") else theme["gold"],
                width=2.0 if peak.get("row_locked") else 1.0,
            )
            self.plot.plot(
                [position, position],
                [offset, offset + (0.055 if peak.get("manual") else 0.035) * span],
                pen=marker_pen,
            )
        self.plot.addLegend()
        self.plot.enableAutoRange()
        message = (
            f"Residual source: {residual.get('source', 'unknown')}; "
            f"{sum(bool(row.get('use', True)) for row in (peaks or []))} included unknown peaks."
        )
        self.message.setText(message if len(message) <= 120 else message[:119] + "…")
        self.message.setToolTip(message)
