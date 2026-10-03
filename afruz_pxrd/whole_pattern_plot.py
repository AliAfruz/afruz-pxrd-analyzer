from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from .context_copy import CopyablePlotWidget, install_label_copy_menu
from .theme import DEFAULT_THEME_NAME, THEMES


class WholePatternPlotWidget(QWidget):
    peakAddRequested = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        theme = THEMES[DEFAULT_THEME_NAME]
        self.plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="Whole-pattern refinement",
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.plot.setLabel("bottom", "2θ", units="degrees")
        self.plot.setLabel("left", "Intensity")
        self.plot.enable_manual_peak_context(True, label="Add manual peak to Peak List here")
        self.plot.manualPeakRequested.connect(
            lambda x, y: self.peakAddRequested.emit(float(x), float(y))
        )
        self.legend = self.plot.addLegend()
        self.message = QLabel(
            "Run Pawley or Le Bail-style refinement to display the profile."
        )
        self.message.setObjectName("mutedLabel")
        self.message.setWordWrap(False)
        self.message.setMinimumWidth(0)
        self.message.setMaximumHeight(24)
        self.message.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Fixed,
        )
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
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def set_result(self, result: dict | None):
        self.plot.clear()
        if not result:
            self.message.setText(
                "Run Pawley or Le Bail-style refinement to display the profile."
            )
            return

        x = np.asarray(result.get("observed_x", []), dtype=float)
        observed = np.asarray(result.get("observed_y", []), dtype=float)
        calculated = np.asarray(result.get("calculated_y", []), dtype=float)
        background = np.asarray(result.get("background_y", []), dtype=float)
        difference = np.asarray(result.get("difference_y", []), dtype=float)
        if not len(x):
            return

        theme = THEMES[DEFAULT_THEME_NAME]
        self.plot.plot(
            x,
            observed,
            pen=None,
            symbol="o",
            symbolSize=3.0,
            symbolPen=pg.mkPen(theme["plot_axis"], width=0.8),
            symbolBrush=pg.mkBrush(theme["plot_axis"]),
            name="Observed",
        )
        self.plot.plot(
            x,
            calculated,
            pen=pg.mkPen(theme["gold"], width=2.0),
            name="Calculated",
        )
        self.plot.plot(
            x,
            background,
            pen=pg.mkPen(theme["muted"], width=1.2, style=Qt.DashLine),
            name="Background",
        )

        span = max(float(np.ptp(observed)), 1.0)
        difference_offset = float(np.min(observed)) - 0.18 * span
        self.plot.plot(
            x,
            difference + difference_offset,
            pen=pg.mkPen(theme["gold_bright"], width=1.0),
            name="Difference",
        )

        phase_rows = list(result.get("phases", []))
        reflections = list(result.get("reflections", []))
        for phase_order, phase in enumerate(phase_rows):
            phase_id = int(phase.get("phase_index", phase_order))
            centers = [
                float(row["two_theta_deg"]) for row in reflections
                if int(row.get("phase_index", -999)) == phase_id and row.get("two_theta_deg") is not None
            ]
            if not centers:
                continue
            baseline = difference_offset - span * (0.055 + phase_order * 0.035)
            tick_x, tick_y = [], []
            for center in centers:
                tick_x.extend([center, center, np.nan])
                tick_y.extend([baseline, baseline + span * 0.032, np.nan])
            phase_name = phase.get("phase_name") or phase.get("formula") or f"Phase {phase_order + 1}"
            tick_color = pg.intColor(phase_order, hues=max(1, len(phase_rows)), values=235, maxValue=255)
            self.plot.plot(
                tick_x, tick_y, connect="finite",
                pen=pg.mkPen(tick_color, width=1.4),
                name=f"Bragg — {phase_name}",
            )

        self.plot.enableAutoRange()
        message = (
            f"{result.get('mode')} — "
            f"Rwp {result.get('rwp_percent', 0):.4g}%, "
            f"Rp {result.get('rp_percent', 0):.4g}%, "
            f"R² {result.get('r_squared', 0):.6g}; "
            f"{result.get('reflection_count', 0)} reflections."
        )
        compact = " ".join(message.split())
        self.message.setText(
            compact
            if len(compact) <= 120
            else compact[:119].rstrip() + "…"
        )
        self.message.setToolTip(message)
