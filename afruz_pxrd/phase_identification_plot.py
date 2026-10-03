from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .context_copy import CopyablePlotWidget, install_label_copy_menu
from .theme import DEFAULT_THEME_NAME, THEMES


class PhaseIdentificationPlotWidget(QWidget):
    peakAddRequested = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        theme = THEMES[DEFAULT_THEME_NAME]
        self.plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="Phase-identification peak match",
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.plot.setLabel("bottom", "2θ", units="degrees", color=theme["gold_bright"])
        self.plot.setLabel("left", "Observed (+) / reference (−)", color=theme["gold_bright"])
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.enable_manual_peak_context(True)
        self.plot.manualPeakRequested.connect(
            lambda x, y: self.peakAddRequested.emit(float(x), float(y))
        )

        self.message = QLabel("Run a local-library phase search to display peak matching.")
        self.message.setObjectName("mutedLabel")
        self.message.setWordWrap(True)
        install_label_copy_menu(self.message)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.message)

    def apply_theme(self, theme_name: str):
        theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME_NAME])
        self.plot.setBackground(theme["plot_background"])
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.setLabel("bottom", "2θ", units="degrees", color=theme["gold_bright"])
        self.plot.setLabel("left", "Observed (+) / reference (−)", color=theme["gold_bright"])
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def set_result(self, result: dict | None):
        self.plot.clear()
        if not result:
            self.message.setText("Run a local-library phase search to display peak matching.")
            return

        matches = result.get("matches", [])
        if not matches:
            self.message.setText("The selected candidate has no matched peaks.")
            return

        obs_x = np.asarray([row["observed_2theta"] for row in matches], dtype=float)
        obs_y = np.asarray([row["observed_intensity"] for row in matches], dtype=float)
        ref_x = np.asarray([row["shifted_reference_2theta"] for row in matches], dtype=float)
        ref_y = -np.asarray([row["reference_intensity"] for row in matches], dtype=float)

        self.plot.addItem(pg.ScatterPlotItem(obs_x, obs_y, symbol="o", size=9, brush="#f2cc70", pen=pg.mkPen("#ffffff", width=0.7)))
        stick_x, stick_y = [], []
        connector_x, connector_y = [], []
        for ox, oy, rx, ry in zip(obs_x, obs_y, ref_x, ref_y):
            stick_x.extend([rx, rx, np.nan])
            stick_y.extend([0.0, ry, np.nan])
            connector_x.extend([rx, ox, np.nan])
            connector_y.extend([ry, oy, np.nan])
        self.plot.plot(np.asarray(stick_x), np.asarray(stick_y), pen=pg.mkPen("#74c0fc", width=1.5), connect="finite")
        self.plot.plot(np.asarray(connector_x), np.asarray(connector_y), pen=pg.mkPen("#9aa0aa", width=0.8, style=pg.QtCore.Qt.DashLine), connect="finite")
        self.plot.plot([float(min(np.min(obs_x), np.min(ref_x))), float(max(np.max(obs_x), np.max(ref_x)))], [0.0, 0.0], pen=pg.mkPen("#ffffff", width=1.0))
        self.plot.enableAutoRange()
        self.message.setText(
            f"{result.get('reference_name', 'Candidate')} — score {result.get('score', 0.0):.2f}; "
            f"{result.get('matched_count', 0)} matched peaks; mean |Δ2θ| "
            f"{result.get('mean_absolute_delta_deg', float('nan')):.5g}°."
        )
