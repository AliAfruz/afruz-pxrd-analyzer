from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .context_copy import CopyablePlotWidget
from .theme import DEFAULT_THEME_NAME, THEMES


class ValidationPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        theme = THEMES[DEFAULT_THEME_NAME]
        self.plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="QPA validation",
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.plot.setLabel("bottom", "Known phase fraction", units="wt%")
        self.plot.setLabel("left", "Measured phase fraction", units="wt%")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

    def apply_theme(self, theme_name: str):
        theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME_NAME])
        self.plot.setBackground(theme["plot_background"])
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def set_result(self, result: dict | None, mode: str = "Agreement"):
        self.plot.clear()
        if not result:
            return
        rows = result.get("records", [])
        if not rows:
            return
        theme = THEMES[DEFAULT_THEME_NAME]
        known = np.asarray([row["known_wt_percent"] for row in rows], dtype=float)
        measured = np.asarray([row["measured_wt_percent"] for row in rows], dtype=float)
        phases = sorted({str(row.get("phase_name", "Phase")) for row in rows})

        if mode == "Error":
            error = measured - known
            self.plot.setLabel("bottom", "Known phase fraction", units="wt%")
            self.plot.setLabel("left", "Measured − known", units="wt%")
            self.plot.plot(
                [float(np.min(known)), float(np.max(known))],
                [0.0, 0.0],
                pen=pg.mkPen(theme["muted"], width=1.2),
            )
            for phase in phases:
                indices = [i for i, row in enumerate(rows) if str(row.get("phase_name", "Phase")) == phase]
                self.plot.plot(
                    known[indices],
                    error[indices],
                    pen=None,
                    symbol="o",
                    symbolSize=8,
                    name=phase,
                )
        else:
            self.plot.setLabel("bottom", "Known phase fraction", units="wt%")
            self.plot.setLabel("left", "Measured phase fraction", units="wt%")
            minimum = float(min(np.min(known), np.min(measured)))
            maximum = float(max(np.max(known), np.max(measured)))
            margin = max(1.0, 0.05 * (maximum - minimum or 1.0))
            self.plot.plot(
                [minimum - margin, maximum + margin],
                [minimum - margin, maximum + margin],
                pen=pg.mkPen(theme["muted"], width=1.2),
                name="Ideal 1:1",
            )
            for phase in phases:
                indices = [i for i, row in enumerate(rows) if str(row.get("phase_name", "Phase")) == phase]
                errors = np.asarray([
                    row.get("measured_error_percent") or 0.0
                    for i, row in enumerate(rows)
                    if i in indices
                ], dtype=float)
                self.plot.plot(
                    known[indices],
                    measured[indices],
                    pen=None,
                    symbol="o",
                    symbolSize=8,
                    name=phase,
                )
                for x, y, err in zip(known[indices], measured[indices], errors):
                    if err > 0:
                        self.plot.plot(
                            [x, x], [y - err, y + err],
                            pen=pg.mkPen(theme["gold"], width=1.0),
                        )
        self.plot.addLegend()
        self.plot.enableAutoRange()
