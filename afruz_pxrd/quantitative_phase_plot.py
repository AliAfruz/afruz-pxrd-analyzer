from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from .context_copy import CopyablePlotWidget, install_label_copy_menu
from .theme import DEFAULT_THEME_NAME, THEMES


class QuantitativePhasePlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        theme = THEMES[DEFAULT_THEME_NAME]
        self.plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="Quantitative phase analysis fit",
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.plot.setLabel(
            "bottom",
            "2θ",
            units="degrees",
            color=theme["gold_bright"],
        )
        self.plot.setLabel(
            "left",
            "Intensity",
            units="a.u.",
            color=theme["gold_bright"],
        )
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.addLegend(offset=(10, 10))

        self.message = QLabel(
            "Select a Phase 5 candidate and run Phase 6 quantification."
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
            "2θ",
            units="degrees",
            color=theme["gold_bright"],
        )
        self.plot.setLabel(
            "left",
            "Intensity",
            units="a.u.",
            color=theme["gold_bright"],
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def set_result(self, result: dict | None):
        self.plot.clear()
        self.plot.addLegend(offset=(10, 10))
        if not result or not result.get("plot"):
            self.message.setText(
                "Select a Phase 5 candidate and run Phase 6 quantification."
            )
            return

        data = result["plot"]
        x = np.asarray(data.get("x", []), dtype=float)
        observed = np.asarray(data.get("observed", []), dtype=float)
        calculated = np.asarray(data.get("calculated", []), dtype=float)
        baseline = np.asarray(data.get("baseline", []), dtype=float)
        residual = np.asarray(data.get("residual", []), dtype=float)
        if not len(x):
            self.message.setText("The QPA result contains no plot data.")
            return

        palette = [
            "#ff8787",
            "#74c0fc",
            "#8ce99a",
            "#b197fc",
            "#ffa94d",
            "#63e6be",
            "#e599f7",
            "#ffd43b",
        ]
        self.plot.plot(
            x,
            observed,
            pen=pg.mkPen("#aeb4bd", width=1.1),
            name="Observed",
        )
        for index, contribution in enumerate(data.get("contributions", [])):
            values = np.asarray(contribution.get("y", []), dtype=float)
            if len(values) != len(x):
                continue
            self.plot.plot(
                x,
                values,
                pen=pg.mkPen(
                    palette[index % len(palette)],
                    width=1.2,
                    style=pg.QtCore.Qt.DotLine,
                ),
                name=contribution.get("reference_name", f"Phase {index + 1}"),
            )

        self.plot.plot(
            x,
            baseline,
            pen=pg.mkPen(
                "#7f8792",
                width=1.0,
                style=pg.QtCore.Qt.DashLine,
            ),
            name="Background",
        )
        self.plot.plot(
            x,
            calculated,
            pen=pg.mkPen("#ffffff", width=2.0),
            name="Calculated",
        )

        span = max(
            float(np.max(observed) - np.min(observed)),
            float(np.max(calculated) - np.min(calculated)),
            1.0,
        )
        residual_offset = min(float(np.min(observed)), float(np.min(calculated))) - 0.18 * span
        self.plot.plot(
            x,
            residual + residual_offset,
            pen=pg.mkPen(
                "#9aa0aa",
                width=1.0,
                style=pg.QtCore.Qt.DashLine,
            ),
            name="Difference",
        )
        self.plot.plot(
            np.asarray([float(x[0]), float(x[-1])]),
            np.asarray([residual_offset, residual_offset]),
            pen=pg.mkPen("#666c75", width=0.8),
        )
        self.plot.enableAutoRange()

        diagnostics = result.get("diagnostics", {})
        self.message.setText(
            f"{diagnostics.get('phase_count', 0)} phase(s); "
            f"R²={diagnostics.get('r_squared', float('nan')):.6f}; "
            f"weighted residual={diagnostics.get('weighted_profile_residual_percent', float('nan')):.4g}%; "
            f"{result.get('bootstrap_samples_successful', 0)} successful bootstrap sample(s)."
        )
