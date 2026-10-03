from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from .context_copy import CopyablePlotWidget, install_label_copy_menu
from .theme import DEFAULT_THEME_NAME, THEMES


BATCH_METRICS = {
    "Primary peak position": ("primary_peak_position_deg", "2θ", "degrees"),
    "Primary peak FWHM": ("primary_peak_fwhm_deg", "FWHM", "degrees"),
    "Detected peak count": ("peak_count", "Peak count", ""),
    "Mean Scherrer size": ("scherrer_mean_nm", "Size", "nm"),
    "Williamson–Hall size": ("wh_size_nm", "Size", "nm"),
    "Microstrain": ("microstrain", "Microstrain", ""),
    "Phase score": ("phase_score", "Identification score", ""),
    "Pawley / Le Bail Rwp": (
        "whole_pattern_rwp_percent",
        "Rwp",
        "%",
    ),
    "Rietveld Rwp": (
        "rietveld_rwp_percent",
        "Rwp",
        "%",
    ),
    "Analysis time": ("elapsed_seconds", "Time", "s"),
}


class BatchComparisonPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        theme = THEMES[DEFAULT_THEME_NAME]
        self.plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="Batch comparison dashboard",
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.plot.setLabel("bottom", "Sample", color=theme["gold_bright"])
        self.plot.setLabel("left", "Value", color=theme["gold_bright"])
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))

        self.message = QLabel("Run a batch to populate the comparison dashboard.")
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
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def set_results(self, rows: list[dict], metric_name: str):
        self.plot.clear()
        if not rows:
            self.message.setText("Run a batch to populate the comparison dashboard.")
            return

        metric_key, axis_label, unit = BATCH_METRICS.get(
            metric_name,
            BATCH_METRICS["Primary peak position"],
        )
        x_values = []
        y_values = []
        labels = []
        for index, row in enumerate(rows, start=1):
            value = row.get(metric_key)
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            x_values.append(index)
            y_values.append(value)
            labels.append(str(row.get("dataset_name", index)))

        if not y_values:
            self.message.setText(
                f"No numeric {metric_name.lower()} results are available."
            )
            return

        theme = THEMES[DEFAULT_THEME_NAME]
        self.plot.plot(
            np.asarray(x_values, dtype=float),
            np.asarray(y_values, dtype=float),
            pen=pg.mkPen(theme["gold"], width=2.0),
            symbol="o",
            symbolSize=8,
            symbolBrush=theme["gold_bright"],
        )
        self.plot.getAxis("bottom").setTicks(
            [list(zip(x_values, labels))]
        )
        self.plot.setLabel("left", axis_label, units=unit)
        self.plot.enableAutoRange()
        self.message.setText(
            f"{metric_name}: {len(y_values)} comparable sample(s). "
            "Missing or failed values are omitted."
        )
