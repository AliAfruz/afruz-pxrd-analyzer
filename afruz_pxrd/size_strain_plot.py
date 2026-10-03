from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from .context_copy import CopyablePlotWidget, install_label_copy_menu
from .theme import DEFAULT_THEME_NAME, THEMES


class SizeStrainPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        theme = THEMES[DEFAULT_THEME_NAME]
        self.plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="Williamson–Hall plot",
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.plot.setLabel(
            "bottom",
            "4 sin(θ)",
            color=theme["gold_bright"],
        )
        self.plot.setLabel(
            "left",
            "β cos(θ)",
            color=theme["gold_bright"],
        )
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))

        self.message = QLabel(
            "Calculate size and strain to display the Williamson–Hall plot."
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
            "4 sin(θ)",
            color=theme["gold_bright"],
        )
        self.plot.setLabel(
            "left",
            "β cos(θ)",
            color=theme["gold_bright"],
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def set_result(self, result: dict | None):
        self.plot.clear()
        if not result:
            self.message.setText(
                "Calculate size and strain to display the Williamson–Hall plot."
            )
            return

        wh = result.get("williamson_hall", {})
        x = np.asarray(wh.get("x", []), dtype=float)
        y = np.asarray(wh.get("y", []), dtype=float)
        if len(x):
            self.plot.addItem(
                pg.ScatterPlotItem(
                    x,
                    y,
                    symbol="o",
                    size=9,
                    brush="#d6ad55",
                    pen=pg.mkPen("#ffffff", width=0.8),
                )
            )

        if wh.get("valid") and len(x):
            x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            y_line = (
                float(wh["slope"]) * x_line
                + float(wh["intercept"])
            )
            self.plot.plot(
                x_line,
                y_line,
                pen=pg.mkPen("#ffffff", width=2.0),
            )
            weighting = "weighted" if wh.get("weighted") else "unweighted"
            self.message.setText(
                f"Williamson–Hall regression: R²={wh['r_squared']:.6f}; "
                f"{weighting}; {wh['point_count']} peaks."
            )
        else:
            self.message.setText(
                wh.get("note")
                or "Williamson–Hall regression is unavailable."
            )
        self.plot.enableAutoRange()
