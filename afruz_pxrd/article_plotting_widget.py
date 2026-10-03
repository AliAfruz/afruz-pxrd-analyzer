from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .article_plotting import (
    DEFAULT_EXPORT_DPI,
    WHITE_TEMPLATE_NAME,
    ArticlePlotExportSettings,
    ArticlePlotStyle,
    bragg_ticks_from_profile,
    export_article_plot_package,
    normalize_article_style,
    profile_arrays,
    profile_from_multicomponent_result,
    render_article_refinement_plot,
)
from .context_copy import install_label_copy_menu
from .widgets import NoWheelDoubleSpinBox, NoWheelSpinBox


class ArticlePlottingWidget(QWidget):
    """Publication plotting workspace with a white central article canvas."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._last_profile: dict | None = None
        self._last_ticks: list[dict] = []
        self._build_ui()
        self.refresh_for_selected_dataset()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        heading = QLabel("Phase 19.4 — Article Plotting Engine")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        desc = QLabel(
            "Create journal-style PXRD figures with full control of colors, fonts, line widths, labels, legend and Bragg tick markers. "
            "The central plot canvas and all exported plots use a white article template; GUI controls may follow the dark Afruz theme. "
            "Exports are forced to 1200 DPI."
        )
        desc.setWordWrap(True)
        desc.setObjectName("mutedLabel")
        install_label_copy_menu(desc)
        layout.addWidget(desc)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_canvas_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 980])
        layout.addWidget(splitter, 1)

    def _build_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        source_group = QGroupBox("Source")
        source_layout = QFormLayout(source_group)
        self.dataset_label = QLabel("No pattern selected")
        self.dataset_label.setWordWrap(True)
        self.source_label = QLabel("—")
        self.source_label.setWordWrap(True)
        install_label_copy_menu(self.dataset_label)
        install_label_copy_menu(self.source_label)
        source_layout.addRow("Dataset", self.dataset_label)
        source_layout.addRow("Profile", self.source_label)
        layout.addWidget(source_group)

        text_group = QGroupBox("Text and font")
        text_layout = QFormLayout(text_group)
        self.title_edit = self._line_edit("PXRD refinement profile")
        self.x_label_edit = self._line_edit("2θ (degrees)")
        self.y_label_edit = self._line_edit("Intensity (a.u.)")
        self.title_size = self._double_spin(6, 48, 12, 1)
        self.axis_size = self._double_spin(6, 36, 10, 1)
        self.tick_size = self._double_spin(4, 28, 8, 1)
        self.legend_size = self._double_spin(4, 28, 8, 1)
        text_layout.addRow("Title", self.title_edit)
        text_layout.addRow("X label", self.x_label_edit)
        text_layout.addRow("Y label", self.y_label_edit)
        text_layout.addRow("Title font", self.title_size)
        text_layout.addRow("Axis font", self.axis_size)
        text_layout.addRow("Tick font", self.tick_size)
        text_layout.addRow("Legend font", self.legend_size)
        layout.addWidget(text_group)

        color_group = QGroupBox("Colors")
        color_layout = QFormLayout(color_group)
        self.obs_color = self._line_edit("#111111")
        self.calc_color = self._line_edit("#b8860b")
        self.diff_color = self._line_edit("#1f77b4")
        self.bkg_color = self._line_edit("#777777")
        self.tick_color = self._line_edit("#111111")
        self.grid_color = self._line_edit("#d8d8d8")
        color_layout.addRow("Observed", self.obs_color)
        color_layout.addRow("Calculated", self.calc_color)
        color_layout.addRow("Obs−Calc", self.diff_color)
        color_layout.addRow("Background", self.bkg_color)
        color_layout.addRow("Bragg ticks", self.tick_color)
        color_layout.addRow("Grid", self.grid_color)
        layout.addWidget(color_group)

        line_group = QGroupBox("Lines, layout and export")
        line_layout = QFormLayout(line_group)
        self.obs_width = self._double_spin(0.1, 8, 0.75, 2)
        self.calc_width = self._double_spin(0.1, 8, 1.25, 2)
        self.diff_width = self._double_spin(0.1, 8, 0.85, 2)
        self.tick_width = self._double_spin(0.1, 6, 0.70, 2)
        self.fig_width = self._double_spin(2.0, 20.0, 7.2, 2)
        self.fig_height = self._double_spin(2.0, 20.0, 4.8, 2)
        self.export_dpi = NoWheelSpinBox()
        self.export_dpi.setRange(DEFAULT_EXPORT_DPI, DEFAULT_EXPORT_DPI)
        self.export_dpi.setValue(DEFAULT_EXPORT_DPI)
        self.show_background = QCheckBox("Show background")
        self.show_background.setChecked(True)
        self.show_difference = QCheckBox("Show Obs−Calc")
        self.show_difference.setChecked(True)
        self.show_legend = QCheckBox("Show legend")
        self.show_legend.setChecked(True)
        self.show_grid = QCheckBox("Show grid")
        self.show_grid.setChecked(True)
        line_layout.addRow("Observed width", self.obs_width)
        line_layout.addRow("Calculated width", self.calc_width)
        line_layout.addRow("Difference width", self.diff_width)
        line_layout.addRow("Tick width", self.tick_width)
        line_layout.addRow("Figure width in", self.fig_width)
        line_layout.addRow("Figure height in", self.fig_height)
        line_layout.addRow("Export DPI", self.export_dpi)
        line_layout.addRow("Background", self.show_background)
        line_layout.addRow("Difference", self.show_difference)
        line_layout.addRow("Legend", self.show_legend)
        line_layout.addRow("Grid", self.show_grid)
        layout.addWidget(line_group)

        actions = QGridLayout()
        self.refresh_button = QPushButton("Refresh source")
        self.preview_button = QPushButton("Apply preview")
        self.preview_button.setObjectName("primaryButton")
        self.export_button = QPushButton("Export 1200 DPI article package…")
        self.export_button.setObjectName("primaryButton")
        actions.addWidget(self.refresh_button, 0, 0)
        actions.addWidget(self.preview_button, 0, 1)
        actions.addWidget(self.export_button, 1, 0, 1, 2)
        layout.addLayout(actions)
        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("mutedLabel")
        install_label_copy_menu(self.status_label)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        self.refresh_button.clicked.connect(self.refresh_for_selected_dataset)
        self.preview_button.clicked.connect(self.update_preview)
        self.export_button.clicked.connect(self.export_article_package)
        return panel

    def _build_canvas_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        canvas_note = QLabel(
            f"Central plot canvas: {WHITE_TEMPLATE_NAME}. Export: fixed {DEFAULT_EXPORT_DPI} DPI."
        )
        canvas_note.setObjectName("mutedLabel")
        install_label_copy_menu(canvas_note)
        layout.addWidget(canvas_note)
        self.figure = Figure(figsize=(7.2, 4.8), facecolor="white")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background: white;")
        layout.addWidget(self.canvas, 1)
        return panel

    def _line_edit(self, text: str):
        from PySide6.QtWidgets import QLineEdit

        edit = QLineEdit(text)
        return edit

    def _double_spin(self, minimum, maximum, value, decimals):
        spin = NoWheelDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(int(decimals))
        spin.setValue(float(value))
        return spin

    def current_style(self) -> ArticlePlotStyle:
        return normalize_article_style(
            {
                "title": self.title_edit.text(),
                "x_label": self.x_label_edit.text(),
                "y_label": self.y_label_edit.text(),
                "title_font_size": self.title_size.value(),
                "axis_font_size": self.axis_size.value(),
                "tick_font_size": self.tick_size.value(),
                "legend_font_size": self.legend_size.value(),
                "observed_color": self.obs_color.text(),
                "calculated_color": self.calc_color.text(),
                "difference_color": self.diff_color.text(),
                "background_color": self.bkg_color.text(),
                "bragg_tick_color": self.tick_color.text(),
                "grid_color": self.grid_color.text(),
                "observed_line_width": self.obs_width.value(),
                "calculated_line_width": self.calc_width.value(),
                "difference_line_width": self.diff_width.value(),
                "bragg_tick_line_width": self.tick_width.value(),
                "figure_width_in": self.fig_width.value(),
                "figure_height_in": self.fig_height.value(),
                "show_background": self.show_background.isChecked(),
                "show_difference": self.show_difference.isChecked(),
                "show_legend": self.show_legend.isChecked(),
                "show_grid": self.show_grid.isChecked(),
            }
        )

    def refresh_for_selected_dataset(self):
        dataset = self.main_window.selected_dataset() if hasattr(self.main_window, "selected_dataset") else None
        if dataset is None:
            self.dataset_label.setText("No pattern selected")
            self.source_label.setText("No profile loaded")
            self._last_profile = None
            self._last_ticks = []
            self.update_preview()
            return
        self.dataset_label.setText(dataset.name)
        result = None
        multicomponent = getattr(self.main_window, "multicomponent_refiner_widget", None)
        if multicomponent is not None:
            result = getattr(multicomponent, "results_by_uid", {}).get(dataset.uid)
        profile = profile_from_multicomponent_result(result)
        if profile is None:
            y = np.asarray(dataset.y, dtype=float)
            self._last_profile = {
                "observed_x": np.asarray(dataset.x, dtype=float).tolist(),
                "observed_y": y.tolist(),
                "calculated_y": y.tolist(),
                "background_y": np.zeros(len(y), dtype=float).tolist(),
                "difference_y": np.zeros(len(y), dtype=float).tolist(),
                "phases": [],
                "reflections": [],
            }
            self._last_ticks = []
            self.source_label.setText("Observed-only article plot. Run Intelligent Multiphase to add calculated curve and Bragg ticks.")
        else:
            self._last_profile = dict(profile)
            self._last_ticks = bragg_ticks_from_profile(profile)
            self.source_label.setText(
                f"Best refinement profile loaded with {len(self._last_ticks)} Bragg tick marker(s)."
            )
        self.update_preview()

    def update_preview(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.set_facecolor("white")
        ax.set_facecolor("white")
        if self._last_profile is None:
            ax.text(0.5, 0.5, "No pattern selected", ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw_idle()
            return
        style = self.current_style()
        try:
            arrays = profile_arrays(self._last_profile)
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw_idle()
            return
        x = arrays["x"]
        observed = arrays["observed"]
        calculated = arrays["calculated"]
        difference = arrays["difference"]
        background = arrays["background"]
        span = max(float(np.nanmax(observed) - np.nanmin(observed)), 1.0)
        lower = float(np.nanmin(observed))
        diff_offset = lower - span * style.difference_offset_fraction
        ax.plot(x, observed, color=style.observed_color, lw=style.observed_line_width, label="Observed")
        ax.plot(x, calculated, color=style.calculated_color, lw=style.calculated_line_width, label="Calculated")
        if style.show_background and np.any(background):
            ax.plot(x, background, color=style.background_color, lw=style.background_line_width, ls="--", label="Background")
        if style.show_difference:
            ax.plot(x, difference + diff_offset, color=style.difference_color, lw=style.difference_line_width, label="Obs − Calc")
        phase_rows: dict[int, list[Mapping]] = {}
        for tick in self._last_ticks:
            try:
                phase_rows.setdefault(int(tick.get("phase_index", 0)), []).append(tick)
            except Exception:
                continue
        gap = span * style.tick_row_gap_fraction
        height = span * style.tick_height_fraction
        for row, phase_index in enumerate(sorted(phase_rows), start=1):
            y0 = diff_offset - row * gap
            for tick in phase_rows[phase_index][: style.max_bragg_ticks_per_phase]:
                try:
                    theta = float(tick.get("two_theta_deg"))
                except Exception:
                    continue
                ax.vlines(theta, y0, y0 - height, color=style.bragg_tick_color, lw=style.bragg_tick_line_width)
        ax.set_title(style.title, fontsize=style.title_font_size)
        ax.set_xlabel(style.x_label, fontsize=style.axis_font_size)
        ax.set_ylabel(style.y_label, fontsize=style.axis_font_size)
        ax.tick_params(labelsize=style.tick_font_size, colors="#111111")
        if style.show_grid:
            ax.grid(True, color=style.grid_color, alpha=0.6, lw=0.35)
        if style.show_legend:
            ax.legend(frameon=False, fontsize=style.legend_font_size)
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status_label.setText("Preview updated on white article canvas.")

    def export_article_package(self):
        if self._last_profile is None:
            QMessageBox.warning(self, "No pattern", "Select a pattern before exporting an article plot.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose article plot export folder")
        if not directory:
            return
        try:
            result = export_article_plot_package(
                Path(directory),
                self._last_profile,
                self._last_ticks,
                style=self.current_style(),
                settings=ArticlePlotExportSettings(dpi=DEFAULT_EXPORT_DPI),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.status_label.setText(
            f"Exported {len(result['files'])} file(s) at {result['dpi']} DPI using white article template."
        )
        QMessageBox.information(
            self,
            "Article plot exported",
            f"Exported article plot package at {DEFAULT_EXPORT_DPI} DPI.\n\n{directory}",
        )

    def apply_theme(self, theme_name: str):
        # The GUI controls follow the application stylesheet. The plot canvas is
        # intentionally not themed: it remains white for article/export fidelity.
        if hasattr(self, "canvas"):
            self.canvas.setStyleSheet("background: white;")
