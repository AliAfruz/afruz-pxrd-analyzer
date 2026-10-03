from __future__ import annotations

from typing import Iterable
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from .context_copy import CopyablePlotWidget, install_label_copy_menu
from .fitting import (
    evaluate_component,
    evaluate_fit_baseline,
    evaluate_fit_group,
)
from .models import Dataset
from .theme import DEFAULT_THEME_NAME, THEMES


class XRDPlotWidget(QWidget):
    coordinateChanged = Signal(str)
    peakAddRequested = Signal(float, float)
    contextPeakAddRequested = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_theme_name = DEFAULT_THEME_NAME
        theme = THEMES[self.current_theme_name]
        self.plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="XRD pattern",
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.plot.setLabel("bottom", "2θ", units="degrees", color=theme["gold_bright"])
        self.plot.setLabel("left", "Intensity", units="a.u.", color=theme["gold_bright"])
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.addLegend(offset=(10, 10))
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.setClipToView(True)
        self.plot.enable_manual_peak_context(True)
        self.plot.manualPeakRequested.connect(
            lambda x, y: self.contextPeakAddRequested.emit(float(x), float(y))
        )

        self.coordinate_label = QLabel("2θ: —    Intensity: —")
        self.coordinate_label.setObjectName("mutedLabel")
        install_label_copy_menu(self.coordinate_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.coordinate_label)

        self._proxy = pg.SignalProxy(
            self.plot.scene().sigMouseMoved,
            rateLimit=30,
            slot=self._mouse_moved,
        )
        self.plot.scene().sigMouseClicked.connect(self._mouse_clicked)
        self._curves = []
        self._peak_items = []
        self._background_items = []
        self._fit_items = []
        self._component_items = []
        self._fit_baseline_items = []
        self._residual_items = []
        self._reference_items = []
        self._selection_items = []

    def apply_theme(self, theme_name: str):
        theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME_NAME])
        self.current_theme_name = theme["name"]
        self.plot.setBackground(theme["plot_background"])
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
        self.plot.getAxis("bottom").setTextPen(
            pg.mkPen(theme["plot_axis"])
        )
        self.plot.getAxis("left").setTextPen(
            pg.mkPen(theme["plot_axis"])
        )
        self.plot.showGrid(
            x=True,
            y=True,
            alpha=theme["grid_alpha"],
        )

    def _mouse_moved(self, event):
        pos = event[0]
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        point = self.plot.plotItem.vb.mapSceneToView(pos)
        text = f"2θ: {point.x():.5f}°    Intensity: {point.y():.5g}"
        self.coordinate_label.setText(text)
        self.coordinateChanged.emit(text)

    def _mouse_clicked(self, event):
        if event.button() != Qt.LeftButton:
            return
        scene_position = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(scene_position):
            return
        point = self.plot.plotItem.vb.mapSceneToView(scene_position)
        self.peakAddRequested.emit(float(point.x()), float(point.y()))

    def clear(self):
        self.plot.clear()
        self.plot.addLegend(offset=(10, 10))
        self._curves.clear()
        self._peak_items.clear()
        self._background_items.clear()
        self._fit_items.clear()
        self._component_items.clear()
        self._fit_baseline_items.clear()
        self._residual_items.clear()
        self._reference_items.clear()
        self._selection_items.clear()

    def redraw(
        self,
        datasets: Iterable[Dataset],
        stack_enabled: bool = False,
        stack_offset: float = 100.0,
        stack_scale: float = 1.0,
        backgrounds: dict[str, np.ndarray] | None = None,
        peak_rows: dict[str, list[dict]] | None = None,
        fit_groups: dict[str, list[dict]] | None = None,
        show_fits: bool = True,
        show_fit_components: bool = False,
        show_fit_baselines: bool = False,
        show_residuals: bool = False,
        reference_pattern: list[dict] | None = None,
        show_reference: bool = False,
        include_hidden: bool = False,
    ):
        self.clear()
        visible_index = 0
        palette = [
            "#f2cc70",
            "#74c0fc",
            "#ff8787",
            "#8ce99a",
            "#b197fc",
            "#ffa94d",
            "#63e6be",
            "#e599f7",
        ]
        backgrounds = backgrounds or {}
        peak_rows = peak_rows or {}
        fit_groups = fit_groups or {}
        reference_pattern = reference_pattern or []
        displayed_y_values = []

        for ds in datasets:
            if not ds.visible and not include_hidden:
                continue

            source_y = np.asarray(ds.y, dtype=float)
            y = source_y * stack_scale
            offset = visible_index * stack_offset if stack_enabled else 0.0
            color = palette[visible_index % len(palette)]
            displayed_y_values.append(y + offset)
            is_stick_pattern = (
                ds.metadata.get("plot_style") == "sticks"
            )

            if is_stick_pattern:
                stick_x = []
                stick_y = []
                for position, intensity in zip(ds.x, y):
                    stick_x.extend(
                        [float(position), float(position), np.nan]
                    )
                    stick_y.extend(
                        [float(offset), float(offset + intensity), np.nan]
                    )
                curve = self.plot.plot(
                    np.asarray(stick_x, dtype=float),
                    np.asarray(stick_y, dtype=float),
                    pen=pg.mkPen(color, width=1.6),
                    name=ds.name,
                    connect="finite",
                )
                self._curves.append(curve)
                visible_index += 1
                continue

            curve = self.plot.plot(
                ds.x,
                y + offset,
                pen=pg.mkPen(color, width=1.6),
                name=ds.name,
            )
            self._curves.append(curve)

            if ds.uid in backgrounds:
                bg = backgrounds[ds.uid] * stack_scale + offset
                item = self.plot.plot(
                    ds.x,
                    bg,
                    pen=pg.mkPen("#7f858e", width=1, style=pg.QtCore.Qt.DashLine),
                )
                self._background_items.append(item)

            for peak in peak_rows.get(ds.uid, []):
                px = peak["position"]
                py = peak["intensity"] * stack_scale + offset
                method = str(peak.get("method", ""))
                is_smart = method == "Smart"
                is_manual = bool(peak.get("manual")) or method == "Manual"
                is_included = bool(peak.get("use", True))
                confidence = peak.get("confidence")
                marker_size = (
                    10
                    if confidence is None
                    else 9 + 4 * float(confidence) / 100.0
                )
                marker = pg.ScatterPlotItem(
                    [px],
                    [py],
                    symbol="o" if is_manual else ("d" if is_smart else "t"),
                    size=marker_size + (2 if is_manual else 0),
                    brush=(color if is_included else pg.mkBrush(128, 128, 128, 90)),
                    pen=(
                        pg.mkPen("#ffffff", width=1.2)
                        if is_manual
                        else pg.mkPen("#ffffff", width=0.8)
                        if is_smart
                        else None
                    ),
                )
                self.plot.addItem(marker)
                self._peak_items.append(marker)

            groups = fit_groups.get(ds.uid, [])
            if groups and (
                show_fits
                or show_fit_components
                or show_fit_baselines
                or show_residuals
            ):
                data_span = max(
                    float(np.max(source_y) - np.min(source_y)),
                    np.finfo(float).eps,
                )
                residual_baseline = (
                    float(np.min(source_y)) * stack_scale
                    + offset
                    - 0.12 * data_span * abs(stack_scale)
                )

                for group in groups:
                    mask = (
                        (ds.x >= float(group["window_min"]))
                        & (ds.x <= float(group["window_max"]))
                    )
                    if not np.any(mask):
                        continue
                    fit_x = ds.x[mask]
                    fit_y_raw = evaluate_fit_group(fit_x, group)
                    baseline_raw = evaluate_fit_baseline(fit_x, group)

                    if show_fit_baselines:
                        baseline_item = self.plot.plot(
                            fit_x,
                            baseline_raw * stack_scale + offset,
                            pen=pg.mkPen(
                                "#a6abb4",
                                width=1.1,
                                style=pg.QtCore.Qt.DashLine,
                            ),
                        )
                        self._fit_baseline_items.append(baseline_item)

                    if show_fit_components:
                        component_palette = [
                            "#ff8787",
                            "#74c0fc",
                            "#8ce99a",
                            "#b197fc",
                            "#ffa94d",
                            "#63e6be",
                            "#e599f7",
                            "#ffd43b",
                        ]
                        for component_index, component in enumerate(
                            group.get("components", [])
                        ):
                            component_curve = (
                                baseline_raw
                                + evaluate_component(fit_x, component)
                            )
                            component_item = self.plot.plot(
                                fit_x,
                                component_curve * stack_scale + offset,
                                pen=pg.mkPen(
                                    component_palette[
                                        component_index
                                        % len(component_palette)
                                    ],
                                    width=1.35,
                                    style=pg.QtCore.Qt.DotLine,
                                ),
                            )
                            self._component_items.append(component_item)

                    if show_fits:
                        fit_item = self.plot.plot(
                            fit_x,
                            fit_y_raw * stack_scale + offset,
                            pen=pg.mkPen("#ffffff", width=2.2),
                        )
                        self._fit_items.append(fit_item)

                    if show_residuals:
                        residual = (
                            source_y[mask] - fit_y_raw
                        ) * stack_scale + residual_baseline
                        residual_item = self.plot.plot(
                            fit_x,
                            residual,
                            pen=pg.mkPen(
                                "#9aa0aa",
                                width=1.1,
                                style=pg.QtCore.Qt.DashLine,
                            ),
                        )
                        self._residual_items.append(residual_item)

            visible_index += 1

        if show_reference and reference_pattern and displayed_y_values:
            combined_min = min(
                float(np.min(values)) for values in displayed_y_values
            )
            combined_max = max(
                float(np.max(values)) for values in displayed_y_values
            )
            data_span = max(combined_max - combined_min, 1.0)
            baseline = combined_min - 0.16 * data_span
            maximum_height = 0.14 * data_span
            x_values = []
            y_values = []
            for peak in reference_pattern:
                position = float(peak["two_theta"])
                height = maximum_height * float(peak["intensity"]) / 100.0
                x_values.extend([position, position, np.nan])
                y_values.extend([baseline, baseline + height, np.nan])
            theme = THEMES.get(
                self.current_theme_name,
                THEMES[DEFAULT_THEME_NAME],
            )
            reference_item = self.plot.plot(
                np.asarray(x_values, dtype=float),
                np.asarray(y_values, dtype=float),
                pen=pg.mkPen(theme["gold"], width=1.5),
                name="CIF reference",
                connect="finite",
            )
            self._reference_items.append(reference_item)

        self.plot.enableAutoRange()

    def highlight_peak(
        self,
        position: float,
        intensity: float | None = None,
        *,
        window_min: float | None = None,
        window_max: float | None = None,
        zoom: bool = True,
    ) -> None:
        """Highlight one detected or fitted reflection without changing data."""
        for item in list(self._selection_items):
            try:
                self.plot.removeItem(item)
            except Exception:
                pass
        self._selection_items.clear()

        theme = THEMES.get(self.current_theme_name, THEMES[DEFAULT_THEME_NAME])
        line = pg.InfiniteLine(
            pos=float(position),
            angle=90,
            movable=False,
            pen=pg.mkPen(theme["gold_bright"], width=2.0),
        )
        self.plot.addItem(line)
        self._selection_items.append(line)

        if intensity is not None and np.isfinite(float(intensity)):
            marker = pg.ScatterPlotItem(
                [float(position)],
                [float(intensity)],
                symbol="o",
                size=17,
                brush=pg.mkBrush(0, 0, 0, 0),
                pen=pg.mkPen(theme["gold_bright"], width=2.2),
            )
            self.plot.addItem(marker)
            self._selection_items.append(marker)

        valid_window = (
            window_min is not None
            and window_max is not None
            and np.isfinite(float(window_min))
            and np.isfinite(float(window_max))
            and float(window_max) > float(window_min)
        )
        if valid_window:
            lower = float(window_min)
            upper = float(window_max)
            left = pg.InfiniteLine(
                pos=lower, angle=90, movable=False,
                pen=pg.mkPen(theme["plot_axis"], width=1.0, style=pg.QtCore.Qt.DashLine),
            )
            right = pg.InfiniteLine(
                pos=upper, angle=90, movable=False,
                pen=pg.mkPen(theme["plot_axis"], width=1.0, style=pg.QtCore.Qt.DashLine),
            )
            self.plot.addItem(left)
            self.plot.addItem(right)
            self._selection_items.extend([left, right])
            if zoom:
                padding = max(0.04 * (upper - lower), 0.02)
                self.plot.setXRange(lower - padding, upper + padding, padding=0)
        elif zoom:
            half_width = 0.75
            self.plot.setXRange(
                float(position) - half_width,
                float(position) + half_width,
                padding=0,
            )

    def clear_peak_highlight(self) -> None:
        for item in list(self._selection_items):
            try:
                self.plot.removeItem(item)
            except Exception:
                pass
        self._selection_items.clear()

    def get_view_range(self) -> list[list[float]]:
        ranges = self.plot.viewRange()
        return [[float(v) for v in ranges[0]], [float(v) for v in ranges[1]]]

    def set_view_range(self, ranges):
        try:
            self.plot.setXRange(*ranges[0], padding=0)
            self.plot.setYRange(*ranges[1], padding=0)
        except Exception:
            pass
