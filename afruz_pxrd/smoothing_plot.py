from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from .context_copy import CopyablePlotWidget
from .theme import DEFAULT_THEME_NAME, THEMES


class AdvancedSmoothingPlotWidget(QWidget):
    """Source/smoothed and smoothing-residual diagnostic plots."""

    peakAddRequested = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_theme_name = DEFAULT_THEME_NAME
        theme = THEMES[self.current_theme_name]

        self.pattern_plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="Advanced smart smoothing",
        )
        self.pattern_plot.setLabel("bottom", "2θ", units="degrees")
        self.pattern_plot.setLabel("left", "Intensity", units="a.u.")
        self.pattern_plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.pattern_plot.addLegend(offset=(10, 10))
        self.pattern_plot.setDownsampling(auto=True, mode="peak")
        self.pattern_plot.setClipToView(True)
        self.pattern_plot.enable_manual_peak_context(True)
        self.pattern_plot.manualPeakRequested.connect(
            lambda x, y: self.peakAddRequested.emit(float(x), float(y))
        )

        self.residual_plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="Smoothing residual",
        )
        self.residual_plot.setLabel("bottom", "2θ", units="degrees")
        self.residual_plot.setLabel("left", "Source − smoothed", units="a.u.")
        self.residual_plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.residual_plot.addLegend(offset=(10, 10))
        self.residual_plot.setDownsampling(auto=True, mode="peak")
        self.residual_plot.setClipToView(True)
        self.residual_plot.enable_manual_peak_context(True)
        self.residual_plot.manualPeakRequested.connect(
            lambda x, y: self.peakAddRequested.emit(float(x), float(y))
        )

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.pattern_plot)
        splitter.addWidget(self.residual_plot)
        splitter.setSizes([350, 220])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.apply_theme(self.current_theme_name)

    def apply_theme(self, theme_name: str):
        theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME_NAME])
        self.current_theme_name = theme["name"]
        for plot in (self.pattern_plot, self.residual_plot):
            plot.setBackground(theme["plot_background"])
            plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
            plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))
            plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def clear(self):
        self.pattern_plot.clear()
        self.residual_plot.clear()
        self.pattern_plot.addLegend(offset=(10, 10))
        self.residual_plot.addLegend(offset=(10, 10))

    def update_result(
        self,
        x: np.ndarray,
        source: np.ndarray,
        smoothed: np.ndarray,
        protected_mask: np.ndarray | None = None,
    ):
        self.clear()
        theme = THEMES.get(self.current_theme_name, THEMES[DEFAULT_THEME_NAME])
        x = np.asarray(x, dtype=float)
        source = np.asarray(source, dtype=float)
        smoothed = np.asarray(smoothed, dtype=float)
        residual = source - smoothed

        self.pattern_plot.plot(
            x,
            source,
            pen=pg.mkPen(theme["plot_axis"], width=1.0),
            name="Smoothing source",
        )
        self.pattern_plot.plot(
            x,
            smoothed,
            pen=pg.mkPen(theme["gold_bright"], width=1.8),
            name="Smart-smoothed pattern",
        )

        if protected_mask is not None:
            protected_mask = np.asarray(protected_mask, dtype=bool)
            if protected_mask.shape == source.shape and np.any(protected_mask):
                indices = np.flatnonzero(protected_mask)
                step = max(1, len(indices) // 1600)
                self.pattern_plot.plot(
                    x[indices[::step]],
                    smoothed[indices[::step]],
                    pen=None,
                    symbol="o",
                    symbolSize=3,
                    symbolBrush=pg.mkBrush("#ff8a80"),
                    name="Peak-protected points",
                )

        self.residual_plot.plot(
            x,
            residual,
            pen=pg.mkPen("#7ad3ff", width=1.2),
            name="Removed component",
        )
        self.residual_plot.addLine(
            y=0.0,
            pen=pg.mkPen(theme["plot_axis"], width=1.0, style=Qt.DashLine),
        )
        self.pattern_plot.enableAutoRange()
        self.residual_plot.enableAutoRange()
