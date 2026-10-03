from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from .context_copy import CopyablePlotWidget, install_label_copy_menu
from .theme import DEFAULT_THEME_NAME, THEMES


class ResidualStressPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        theme = THEMES[DEFAULT_THEME_NAME]
        self.plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="sin²ψ residual stress plot",
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.plot.setLabel(
            "bottom",
            "sin²ψ",
            color=theme["gold_bright"],
        )
        self.plot.setLabel(
            "left",
            "Lattice strain",
            units="µε",
            color=theme["gold_bright"],
        )
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))

        self.message = QLabel(
            "Build ψ observations and calculate residual stress."
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
            "sin²ψ",
            color=theme["gold_bright"],
        )
        self.plot.setLabel(
            "left",
            "Lattice strain",
            units="µε",
            color=theme["gold_bright"],
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def set_result(self, result: dict | None):
        self.plot.clear()
        if not result or not result.get("observations"):
            self.message.setText(
                "Build ψ observations and calculate residual stress."
            )
            return

        observations = result["observations"]
        x = np.asarray([row["sin2psi"] for row in observations], dtype=float)
        y = np.asarray([row["strain"] * 1e6 for row in observations], dtype=float)
        y_error = np.asarray(
            [row["strain_error"] * 1e6 for row in observations], dtype=float
        )

        self.plot.addItem(
            pg.ErrorBarItem(
                x=x,
                y=y,
                height=2.0 * y_error,
                beam=0.012,
                pen=pg.mkPen("#a6abb4", width=1.0),
            )
        )
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

        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 250)
        y_line = (
            float(result["strain_intercept"])
            + float(result["slope_strain"]) * x_line
        ) * 1e6
        self.plot.plot(
            x_line,
            y_line,
            pen=pg.mkPen("#ffffff", width=2.0),
        )
        self.plot.enableAutoRange()

        sign = "tensile" if result["stress_mpa"] >= 0 else "compressive"
        weighting = "weighted" if result.get("weighted") else "unweighted"
        self.message.setText(
            f"σφ={result['stress_mpa']:.6g} ± "
            f"{result['stress_error_mpa']:.3g} MPa ({sign}); "
            f"R²={result['r_squared']:.6f}; {weighting}; "
            f"{result['point_count']} observations."
        )
