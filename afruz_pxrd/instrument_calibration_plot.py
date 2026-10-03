from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout

from .context_copy import CopyablePlotWidget
from .theme import DEFAULT_THEME_NAME, THEMES


class InstrumentCalibrationPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = CopyablePlotWidget(
            background=THEMES[DEFAULT_THEME_NAME]["plot_background"],
            copy_title="Instrument calibration",
        )
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

    def apply_theme(self, theme_name: str):
        theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME_NAME])
        self.plot.setBackground(theme["plot_background"])
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))

    def set_profile(self, profile: dict | None, mode: str = "Width"):
        self.plot.clear()
        if not profile:
            return
        rows = profile.get("observations", [])
        if not rows:
            return
        theme = THEMES[DEFAULT_THEME_NAME]
        x = np.asarray(
            [row["reference_two_theta_deg"] for row in rows],
            dtype=float,
        )
        if mode == "Position":
            observed = np.asarray(
                [row["observed_two_theta_deg"] for row in rows],
                dtype=float,
            )
            calculated = np.asarray(
                [row["calculated_two_theta_deg"] for row in rows],
                dtype=float,
            )
            self.plot.setLabel("bottom", "Reference 2θ", units="degrees")
            self.plot.setLabel("left", "Observed / calculated 2θ", units="degrees")
        else:
            observed = np.asarray(
                [row["observed_fwhm_deg"] for row in rows],
                dtype=float,
            )
            calculated = np.asarray(
                [row["calculated_fwhm_deg"] for row in rows],
                dtype=float,
            )
            self.plot.setLabel("bottom", "2θ", units="degrees")
            self.plot.setLabel("left", "FWHM", units="degrees")

        order = np.argsort(x)
        self.plot.plot(
            x,
            observed,
            pen=None,
            symbol="o",
            symbolSize=8,
            symbolBrush=theme["gold_bright"],
            name="Observed",
        )
        self.plot.plot(
            x[order],
            calculated[order],
            pen=pg.mkPen(theme["gold"], width=2),
            name="Calculated",
        )
        self.plot.addLegend()
        self.plot.enableAutoRange()
