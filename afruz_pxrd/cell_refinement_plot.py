from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from .context_copy import CopyablePlotWidget, install_label_copy_menu
from .theme import DEFAULT_THEME_NAME, THEMES


class CellRefinementPlotWidget(QWidget):
    peakAddRequested = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        theme = THEMES[DEFAULT_THEME_NAME]
        self.plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="Unit-cell refinement residuals",
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.plot.setLabel(
            "bottom",
            "Observed 2θ",
            units="degrees",
            color=theme["gold_bright"],
        )
        self.plot.setLabel(
            "left",
            "Observed − calculated",
            units="degrees",
            color=theme["gold_bright"],
        )
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.addLegend(offset=(10, 10))
        self.plot.enable_manual_peak_context(
            True, label="Add observed peak to Peak List here"
        )
        self.plot.manualPeakRequested.connect(
            lambda x, y: self.peakAddRequested.emit(float(x), float(y))
        )

        self.message = QLabel(
            "Import a CIF, match peaks, and refine the unit cell."
        )
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
        self.plot.setLabel(
            "bottom",
            "Observed 2θ",
            units="degrees",
            color=theme["gold_bright"],
        )
        self.plot.setLabel(
            "left",
            "Observed − calculated",
            units="degrees",
            color=theme["gold_bright"],
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def set_result(self, result: dict | None):
        self.plot.clear()
        self.plot.addLegend(offset=(10, 10))
        if not result or not result.get("matches"):
            self.message.setText(
                "Import a CIF, match peaks, and refine the unit cell."
            )
            return

        matches = result["matches"]
        observed = np.asarray(
            [row["observed_2theta"] for row in matches],
            dtype=float,
        )
        initial = np.asarray(
            [row["initial_delta"] for row in matches],
            dtype=float,
        )
        refined = np.asarray(
            [row["refined_delta"] for row in matches],
            dtype=float,
        )

        self.plot.addItem(
            pg.ScatterPlotItem(
                observed,
                initial,
                symbol="o",
                size=8,
                brush="#9aa0aa",
                pen=None,
                name="Initial residual",
            )
        )
        self.plot.addItem(
            pg.ScatterPlotItem(
                observed,
                refined,
                symbol="d",
                size=10,
                brush="#d6ad55",
                pen=pg.mkPen("#ffffff", width=0.8),
                name="Refined residual",
            )
        )
        x_line = np.asarray(
            [float(np.min(observed)), float(np.max(observed))],
            dtype=float,
        )
        self.plot.plot(
            x_line,
            np.zeros_like(x_line),
            pen=pg.mkPen("#ffffff", width=1.2),
        )
        self.plot.enableAutoRange()
        weighting = "weighted" if result.get("weighted") else "unweighted"
        self.message.setText(
            f"{result['match_count']} matched peaks; "
            f"RMSE={result['rmse_deg']:.6g}°; "
            f"R²={result['r_squared']:.6f}; {weighting}."
        )
