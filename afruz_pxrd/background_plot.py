from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from .context_copy import CopyablePlotWidget
from .theme import DEFAULT_THEME_NAME, THEMES


class AdvancedBackgroundPlotWidget(QWidget):
    """Raw/background and corrected-pattern diagnostic plots."""

    peakAddRequested = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_theme_name = DEFAULT_THEME_NAME
        theme = THEMES[self.current_theme_name]

        self.raw_plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="Advanced background model",
        )
        self.raw_plot.setLabel("bottom", "2θ", units="degrees")
        self.raw_plot.setLabel("left", "Intensity", units="a.u.")
        self.raw_plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.raw_plot.addLegend(offset=(10, 10))
        self.raw_plot.setDownsampling(auto=True, mode="peak")
        self.raw_plot.setClipToView(True)
        self.raw_plot.enable_manual_peak_context(True)
        self.raw_plot.manualPeakRequested.connect(
            lambda x, y: self.peakAddRequested.emit(float(x), float(y))
        )

        self.corrected_plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="Background-corrected pattern",
        )
        self.corrected_plot.setLabel("bottom", "2θ", units="degrees")
        self.corrected_plot.setLabel("left", "Corrected intensity", units="a.u.")
        self.corrected_plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.corrected_plot.addLegend(offset=(10, 10))
        self.corrected_plot.setDownsampling(auto=True, mode="peak")
        self.corrected_plot.setClipToView(True)
        self.corrected_plot.enable_manual_peak_context(True)
        self.corrected_plot.manualPeakRequested.connect(
            lambda x, y: self.peakAddRequested.emit(float(x), float(y))
        )

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.raw_plot)
        splitter.addWidget(self.corrected_plot)
        splitter.setSizes([320, 260])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.apply_theme(self.current_theme_name)

    def apply_theme(self, theme_name: str):
        theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME_NAME])
        self.current_theme_name = theme["name"]
        for plot in (self.raw_plot, self.corrected_plot):
            plot.setBackground(theme["plot_background"])
            plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
            plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))
            plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def clear(self):
        self.raw_plot.clear()
        self.corrected_plot.clear()
        self.raw_plot.addLegend(offset=(10, 10))
        self.corrected_plot.addLegend(offset=(10, 10))

    def update_result(
        self,
        x: np.ndarray,
        raw: np.ndarray,
        background: np.ndarray,
        corrected: np.ndarray,
        protected_mask: np.ndarray | None = None,
    ):
        self.clear()
        theme = THEMES.get(
            self.current_theme_name,
            THEMES[DEFAULT_THEME_NAME],
        )
        x = np.asarray(x, dtype=float)
        raw = np.asarray(raw, dtype=float)
        background = np.asarray(background, dtype=float)
        corrected = np.asarray(corrected, dtype=float)

        self.raw_plot.plot(
            x,
            raw,
            pen=pg.mkPen(theme["gold_bright"], width=1.4),
            name="Raw pattern",
        )
        self.raw_plot.plot(
            x,
            background,
            pen=pg.mkPen("#7ad3ff", width=2.0),
            name="Detected background",
        )

        if protected_mask is not None:
            protected_mask = np.asarray(protected_mask, dtype=bool)
            if protected_mask.shape == raw.shape and np.any(protected_mask):
                indices = np.flatnonzero(protected_mask)
                step = max(1, len(indices) // 1500)
                self.raw_plot.plot(
                    x[indices[::step]],
                    raw[indices[::step]],
                    pen=None,
                    symbol="o",
                    symbolSize=3,
                    symbolBrush=pg.mkBrush("#ff8a80"),
                    name="Peak-protected points",
                )

        self.corrected_plot.plot(
            x,
            corrected,
            pen=pg.mkPen(theme["gold_bright"], width=1.5),
            name="Corrected pattern",
        )
        self.corrected_plot.addLine(
            y=0.0,
            pen=pg.mkPen(theme["plot_axis"], width=1.0, style=Qt.DashLine),
        )
        self.raw_plot.enableAutoRange()
        self.corrected_plot.enableAutoRange()
