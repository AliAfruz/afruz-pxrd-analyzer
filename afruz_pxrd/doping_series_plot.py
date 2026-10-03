from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .context_copy import CopyablePlotWidget
from .theme import DEFAULT_THEME_NAME, THEMES


TREND_METRICS = {
    "Cell a": ("cell_a_angstrom", "a", "Å"),
    "Cell b": ("cell_b_angstrom", "b", "Å"),
    "Cell c": ("cell_c_angstrom", "c", "Å"),
    "Cell volume": ("cell_volume_angstrom3", "Cell volume", "Å³"),
    "Rwp": ("rwp_percent", "Rwp", "%"),
    "Zero shift": ("zero_shift_deg", "Zero shift", "degrees"),
    "Primary phase fraction": ("primary_phase_fraction_percent", "Pattern scale fraction", "%"),
    "Profile W": ("caglioti_w", "Caglioti W", "degrees²"),
}


class DopingSeriesPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabs = QTabWidget()
        self.overlay = self._plot("Series overlay", "Normalized intensity")
        self.overlay.addLegend(offset=(8, 8))
        self.stacked = self._plot("Stacked series", "Normalized intensity + offset")
        self.difference = self._plot("Difference from reference", "Δ intensity")
        self.heatmap = self._plot("Series heatmap", "Series")
        self.trend = self._plot("Refinement trend", "Value")
        for widget, label in (
            (self.overlay, "Overlay"),
            (self.stacked, "Stacked"),
            (self.heatmap, "Heatmap"),
            (self.difference, "Difference"),
            (self.trend, "Refinement Trends"),
        ):
            self.tabs.addTab(widget, label)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self._result = None
        self._theme_name = DEFAULT_THEME_NAME
        self._trend_metric = "Cell volume"

    def _plot(self, title: str, left_label: str):
        theme = THEMES[DEFAULT_THEME_NAME]
        plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title=title,
        )
        plot.setLabel("bottom", "2θ", units="degrees", color=theme["gold_bright"])
        plot.setLabel("left", left_label, color=theme["gold_bright"])
        plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))
        return plot

    def apply_theme(self, theme_name: str):
        self._theme_name = theme_name
        theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME_NAME])
        for plot in (self.overlay, self.stacked, self.difference, self.heatmap, self.trend):
            plot.setBackground(theme["plot_background"])
            plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
            plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))
            plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.set_result(self._result)

    def set_trend_metric(self, metric_name: str):
        self._trend_metric = metric_name
        self._draw_trend()

    def set_result(self, result: dict | None):
        self._result = result
        for plot in (self.overlay, self.stacked, self.difference, self.heatmap, self.trend):
            plot.clear()
        if not result:
            return
        comparison = result.get("comparison", {})
        x = np.asarray(comparison.get("common_two_theta_deg", []), dtype=float)
        profiles = comparison.get("profiles", [])
        if not len(x) or not profiles:
            return
        theme = THEMES.get(self._theme_name, THEMES[DEFAULT_THEME_NAME])
        reference_uid = comparison.get("reference_dataset_uid")
        maximum = max(
            max(np.max(np.abs(np.asarray(profile.get("common_y", [0.0]), dtype=float))), 1e-12)
            for profile in profiles
        )
        offset = maximum * 0.85
        for index, profile in enumerate(profiles):
            color = pg.intColor(index, hues=max(2, len(profiles)), values=1, maxValue=230)
            y = np.asarray(profile.get("common_y", []), dtype=float)
            difference = np.asarray(profile.get("difference_y", []), dtype=float)
            label = f"{profile.get('series_value'):g} {profile.get('series_unit', '')} — {profile.get('dataset_name')}"
            width = 2.4 if profile.get("dataset_uid") == reference_uid else 1.3
            self.overlay.plot(x, y, pen=pg.mkPen(color, width=width), name=label)
            self.stacked.plot(x, y + index * offset, pen=pg.mkPen(color, width=1.3))
            self.difference.plot(x, difference, pen=pg.mkPen(color, width=1.3))
        if profiles:
            ticks = [
                (index * offset, f"{profile.get('series_value'):g}")
                for index, profile in enumerate(profiles)
            ]
            self.stacked.getAxis("left").setTicks([ticks])
        matrix = np.asarray(comparison.get("heatmap_matrix", []), dtype=float)
        if matrix.shape == (len(profiles), len(x)):
            image = pg.ImageItem(matrix.T)
            try:
                image.setColorMap(pg.colormap.get("viridis"))
            except Exception:
                pass
            image.setRect(
                QRectF(
                    float(x[0]),
                    -0.5,
                    float(x[-1] - x[0]),
                    float(len(profiles)),
                )
            )
            self.heatmap.addItem(image)
            self.heatmap.getAxis("left").setTicks(
                [[(index, f"{profile.get('series_value'):g}") for index, profile in enumerate(profiles)]]
            )
        self.overlay.setLabel("left", comparison.get("normalization", "Intensity"))
        self.stacked.setLabel("left", f"Series value ({profiles[0].get('series_unit', '')})")
        self.difference.addLine(y=0.0, pen=pg.mkPen(theme["muted"], width=1, style=Qt.DashLine))
        for plot in (self.overlay, self.stacked, self.difference, self.heatmap):
            plot.enableAutoRange()
        self._draw_trend()

    def _draw_trend(self):
        self.trend.clear()
        if not self._result:
            return
        trends = self._result.get("refinement", {}).get("trends", [])
        key, label, unit = TREND_METRICS.get(
            self._trend_metric, TREND_METRICS["Cell volume"]
        )
        x_values = []
        y_values = []
        for row in trends:
            try:
                x_value = float(row.get("series_value"))
                y_value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(x_value) and np.isfinite(y_value):
                x_values.append(x_value)
                y_values.append(y_value)
        theme = THEMES.get(self._theme_name, THEMES[DEFAULT_THEME_NAME])
        if y_values:
            order = np.argsort(x_values)
            x = np.asarray(x_values)[order]
            y = np.asarray(y_values)[order]
            self.trend.plot(
                x,
                y,
                pen=pg.mkPen(theme["gold"], width=2.0),
                symbol="o",
                symbolSize=8,
                symbolBrush=theme["gold_bright"],
            )
            self.trend.setLabel("bottom", "Dopant concentration / series value")
            self.trend.setLabel("left", label, units=unit)
            self.trend.enableAutoRange()
