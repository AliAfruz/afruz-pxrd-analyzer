from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from afruz_pxrd.app import MainWindow
from afruz_pxrd.context_copy import CopyablePlotWidget, PlotNavigationToolbar


@pytest.mark.gui
def test_plot_toolbar_supports_zoom_history_modes_and_compact_layout(qtbot):
    plot = CopyablePlotWidget(copy_title="Toolbar QA")
    qtbot.addWidget(plot)
    plot.resize(700, 420)
    plot.plot([10, 20, 30, 40], [2, 8, 4, 7])
    plot.show()
    qtbot.wait(240)

    toolbar = plot.navigation_toolbar
    assert toolbar.isVisible()
    initial = toolbar._current_range()
    toolbar.zoom_in_button.click()
    qtbot.wait(220)
    zoomed = toolbar._current_range()
    assert zoomed[0][1] - zoomed[0][0] < initial[0][1] - initial[0][0]
    assert toolbar.back_button.isEnabled()

    toolbar.back_button.click()
    restored = toolbar._current_range()
    assert toolbar._ranges_close(restored, initial)
    assert toolbar.forward_button.isEnabled()

    toolbar.box_zoom_button.click()
    assert toolbar.box_zoom_button.isChecked()
    toolbar.pan_button.click()
    assert toolbar.pan_button.isChecked()

    plot.resize(340, 260)
    qtbot.wait(20)
    assert not toolbar.copy_button.isVisible()
    assert toolbar.home_button.isVisible()
    assert toolbar.export_button.isVisible()


@pytest.mark.gui
def test_plot_toolbar_copies_clean_figure_and_exports_png_svg(
    qtbot, monkeypatch, tmp_path
):
    plot = CopyablePlotWidget(copy_title="Publication Figure")
    qtbot.addWidget(plot)
    plot.resize(520, 320)
    plot.plot([1, 2, 3], [4, 9, 5])
    plot.show()
    qtbot.wait(80)
    export_errors = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args: export_errors.append(str(args[-1])),
    )

    plot.navigation_toolbar.copy_image()
    assert not plot.navigation_toolbar.isHidden()
    assert not QApplication.clipboard().pixmap().isNull()

    png_base = tmp_path / "plot-image"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(png_base), "PNG image (*.png)"),
    )
    png = plot.navigation_toolbar.export_png()
    assert png == png_base.with_suffix(".png")
    assert png.is_file() and png.stat().st_size > 0

    svg_base = tmp_path / "plot-vector"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(svg_base), "SVG vector image (*.svg)"),
    )
    svg = plot.navigation_toolbar.export_svg()
    assert svg == svg_base.with_suffix(".svg"), export_errors
    assert svg.is_file() and "<svg" in svg.read_text(encoding="utf-8")
    assert not export_errors


@pytest.mark.gui
def test_main_window_scientific_plots_share_modern_toolbar(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    toolbars = window.findChildren(PlotNavigationToolbar)
    assert len(toolbars) >= 10
    assert window.plot_widget.plot.navigation_toolbar in toolbars
    assert window.background_plot.raw_plot.navigation_toolbar in toolbars
    assert window.smoothing_plot.residual_plot.navigation_toolbar in toolbars
    assert window.qpa_plot.plot.navigation_toolbar in toolbars
