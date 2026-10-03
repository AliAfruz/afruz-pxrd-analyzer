from __future__ import annotations

import html
import re
from collections.abc import Callable
from pathlib import Path

import pyqtgraph as pg
import pyqtgraph.exporters
from PySide6.QtCore import QEvent, QItemSelectionModel, QMimeData, QObject, QRect, Qt, QSize, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence, QPainter, QPixmap
from PySide6.QtSvg import QSvgGenerator
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QTableView,
    QTableWidget,
    QTextEdit,
    QToolButton,
    QWidget,
)

from .cinematic_ui import cinematic_icon
from .theme import DEFAULT_THEME_NAME, THEMES


def _safe_cell_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _matrix_to_tsv(matrix: list[list[str]]) -> str:
    return "\n".join(
        "\t".join(_safe_cell_text(value) for value in row)
        for row in matrix
    )


def _matrix_to_html(matrix: list[list[str]], header_rows: int = 0) -> str:
    rows = []
    for row_index, row in enumerate(matrix):
        cell_tag = "th" if row_index < header_rows else "td"
        cells = "".join(
            f"<{cell_tag}>{html.escape(_safe_cell_text(value))}</{cell_tag}>"
            for value in row
        )
        rows.append(f"<tr>{cells}</tr>")
    return (
        "<table style='border-collapse:collapse'>"
        + "".join(rows)
        + "</table>"
    )


def _copy_matrix_to_clipboard(
    matrix: list[list[str]],
    header_rows: int = 0,
) -> None:
    if not matrix:
        return

    mime = QMimeData()
    mime.setText(_matrix_to_tsv(matrix))
    mime.setHtml(_matrix_to_html(matrix, header_rows=header_rows))
    QGuiApplication.clipboard().setMimeData(mime)


def _table_headers(table: QTableView, columns: list[int]) -> list[str]:
    model = table.model()
    if model is None:
        return [""] * len(columns)
    return [
        _safe_cell_text(model.headerData(column, Qt.Horizontal, Qt.DisplayRole))
        for column in columns
    ]


def _table_dimensions(table: QTableView) -> tuple[int, int]:
    model = table.model()
    return (0, 0) if model is None else (model.rowCount(), model.columnCount())


def _table_cell_text(table: QTableView, row: int, column: int) -> str:
    model = table.model()
    if model is None:
        return ""
    return _safe_cell_text(model.data(model.index(row, column), Qt.DisplayRole))


def _entire_table_matrix(table: QTableView) -> list[list[str]]:
    row_count, column_count = _table_dimensions(table)
    columns = list(range(column_count))
    matrix = [_table_headers(table, columns)]
    for row in range(row_count):
        matrix.append(
            [_table_cell_text(table, row, column) for column in columns]
        )
    return matrix


def _selected_table_matrix(
    table: QTableView,
    include_headers: bool,
) -> list[list[str]]:
    indexes = table.selectedIndexes()
    if not indexes:
        return []

    selected_positions = {
        (index.row(), index.column())
        for index in indexes
    }
    rows = sorted({row for row, _ in selected_positions})
    columns = sorted({column for _, column in selected_positions})

    matrix: list[list[str]] = []
    if include_headers:
        matrix.append(_table_headers(table, columns))

    for row in rows:
        matrix.append(
            [
                (
                    ""
                    if (row, column) not in selected_positions
                    else _table_cell_text(table, row, column)
                )
                for column in columns
            ]
        )
    return matrix


def install_table_copy_menu(table: QTableView) -> None:
    """
    Add right-click copy actions to a table.

    Clipboard content is written as both TSV and HTML so it pastes cleanly
    into spreadsheets, documents, and email clients.
    """
    if table.property("afruzCopySupportInstalled"):
        return
    table.setProperty("afruzCopySupportInstalled", True)
    table.setContextMenuPolicy(Qt.CustomContextMenu)
    table.setSelectionMode(QAbstractItemView.ExtendedSelection)

    copy_shortcut = QAction("Copy selected cells", table)
    copy_shortcut.setShortcut(QKeySequence.Copy)
    copy_shortcut.setShortcutContext(Qt.WidgetWithChildrenShortcut)
    copy_shortcut.triggered.connect(
        lambda: _copy_matrix_to_clipboard(
            _selected_table_matrix(table, include_headers=False)
        )
    )
    table.addAction(copy_shortcut)

    select_all_shortcut = QAction("Select all", table)
    select_all_shortcut.setShortcut(QKeySequence.SelectAll)
    select_all_shortcut.setShortcutContext(Qt.WidgetWithChildrenShortcut)
    select_all_shortcut.triggered.connect(table.selectAll)
    table.addAction(select_all_shortcut)

    def show_menu(position):
        clicked_index = table.indexAt(position)
        if clicked_index.isValid() and not table.selectionModel().isSelected(clicked_index):
            table.clearSelection()
            table.selectionModel().select(clicked_index, QItemSelectionModel.Select)

        has_selection = bool(table.selectedIndexes())
        row_count, column_count = _table_dimensions(table)
        has_data = row_count > 0 and column_count > 0

        menu = QMenu(table)
        copy_selected = menu.addAction("Copy selected cells")
        copy_selected_headers = menu.addAction(
            "Copy selected cells with headers"
        )
        menu.addSeparator()
        select_all = menu.addAction("Select all")
        copy_all = menu.addAction("Copy entire table")

        copy_selected.setEnabled(has_selection)
        copy_selected_headers.setEnabled(has_selection)
        copy_all.setEnabled(has_data)
        select_all.setEnabled(has_data)

        chosen = menu.exec(table.viewport().mapToGlobal(position))
        if chosen is copy_selected:
            _copy_matrix_to_clipboard(
                _selected_table_matrix(table, include_headers=False)
            )
        elif chosen is copy_selected_headers:
            _copy_matrix_to_clipboard(
                _selected_table_matrix(table, include_headers=True),
                header_rows=1,
            )
        elif chosen is select_all:
            table.selectAll()
        elif chosen is copy_all:
            _copy_matrix_to_clipboard(
                _entire_table_matrix(table),
                header_rows=1,
            )

    table.customContextMenuRequested.connect(show_menu)


def install_label_copy_menu(
    label: QLabel,
    text_provider: Callable[[], str] | None = None,
) -> None:
    """Add a small right-click Copy action to a label."""
    if label.property("afruzCopySupportInstalled"):
        return
    label.setProperty("afruzCopySupportInstalled", True)
    label.setContextMenuPolicy(Qt.CustomContextMenu)
    label.setTextInteractionFlags(
        label.textInteractionFlags()
        | Qt.TextSelectableByMouse
        | Qt.TextSelectableByKeyboard
    )

    def show_menu(position):
        menu = QMenu(label)
        selected_text = label.selectedText()
        copy_action = menu.addAction("Copy selected text")
        copy_action.setEnabled(bool(selected_text))
        copy_all_action = menu.addAction("Copy all text")
        chosen = menu.exec(label.mapToGlobal(position))
        if chosen is copy_action:
            QGuiApplication.clipboard().setText(selected_text)
        elif chosen is copy_all_action:
            text = (
                text_provider()
                if text_provider is not None
                else label.text()
            )
            QGuiApplication.clipboard().setText(text or "")

    label.customContextMenuRequested.connect(show_menu)


def _install_copy_support_in_tree(widget: QWidget) -> None:
    """Enhance existing and newly-created report controls in one pass."""
    if isinstance(widget, QTableView):
        install_table_copy_menu(widget)
    elif isinstance(widget, QLabel):
        install_label_copy_menu(widget)
    elif isinstance(widget, (QTextEdit, QPlainTextEdit)):
        # These controls already implement Ctrl+C/Ctrl+A and selection.  This
        # property documents that global copy support has inspected them.
        widget.setProperty("afruzCopySupportInstalled", True)
    for child in widget.findChildren(QWidget, "", Qt.FindDirectChildrenOnly):
        _install_copy_support_in_tree(child)


class _GlobalCopySupport(QObject):
    """Enhance dialogs that are created after the main window is built."""

    def eventFilter(self, watched, event):
        try:
            is_show = event.type() == QEvent.Show
        except RuntimeError:
            return False
        if is_show and isinstance(watched, QWidget):
            try:
                _install_copy_support_in_tree(watched)
            except RuntimeError:
                pass
        return False


def install_global_copy_support(app: QApplication, root: QWidget | None = None) -> None:
    """Make every label and table in an application window copyable.

    A Show-event hook covers dialogs constructed later without touching Qt's
    fragile ChildAdded construction events.
    """
    support = getattr(app, "_afruz_global_copy_support", None)
    if not isinstance(support, _GlobalCopySupport):
        support = _GlobalCopySupport(app)
        app.installEventFilter(support)
        app._afruz_global_copy_support = support
    if root is not None:
        _install_copy_support_in_tree(root)


class PlotNavigationToolbar(QFrame):
    """Floating, consistent navigation and export controls for every plot."""

    statusChanged = Signal(str)

    def __init__(self, plot_widget: "CopyablePlotWidget"):
        super().__init__(plot_widget)
        self.plot_widget = plot_widget
        self.view_box = plot_widget.getPlotItem().getViewBox()
        self._history: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._history_index = -1
        self._restoring_view = False
        self.setObjectName("plotNavigationToolbar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 4, 5, 4)
        layout.setSpacing(2)

        self.home_button = self._button("home", "Show all data")
        self.back_button = self._button("undo", "Previous view")
        self.forward_button = self._button("redo", "Next view")
        self.zoom_in_button = self._button("zoom_in", "Zoom in")
        self.zoom_out_button = self._button("zoom_out", "Zoom out")
        self.pan_button = self._button("pan", "Pan with mouse drag", checkable=True)
        self.box_zoom_button = self._button(
            "box_zoom", "Zoom to a dragged rectangle", checkable=True
        )
        self.copy_button = self._button("copy", "Copy plot image")
        self.export_button = self._button("download", "Export this plot")

        for button in (
            self.home_button,
            self.back_button,
            self.forward_button,
            self.zoom_in_button,
            self.zoom_out_button,
            self.pan_button,
            self.box_zoom_button,
            self.copy_button,
            self.export_button,
        ):
            layout.addWidget(button)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.pan_button)
        self.mode_group.addButton(self.box_zoom_button)
        self.pan_button.setChecked(True)

        export_menu = QMenu(self.export_button)
        self.copy_2x_action = export_menu.addAction("Copy image at 2×")
        export_menu.addSeparator()
        self.export_png_action = export_menu.addAction("Save PNG…")
        self.export_svg_action = export_menu.addAction("Save SVG…")
        self.export_button.setMenu(export_menu)
        self.export_button.setPopupMode(QToolButton.InstantPopup)

        self.home_button.clicked.connect(self.auto_range)
        self.back_button.clicked.connect(self.previous_view)
        self.forward_button.clicked.connect(self.next_view)
        self.zoom_in_button.clicked.connect(lambda: self.zoom(0.72))
        self.zoom_out_button.clicked.connect(lambda: self.zoom(1.38))
        self.pan_button.clicked.connect(lambda: self.set_mouse_mode("pan"))
        self.box_zoom_button.clicked.connect(lambda: self.set_mouse_mode("box"))
        self.copy_button.clicked.connect(self.copy_image)
        self.copy_2x_action.triggered.connect(lambda: self.copy_image(scale=2))
        self.export_png_action.triggered.connect(self.export_png)
        self.export_svg_action.triggered.connect(self.export_svg)

        self._history_timer = QTimer(self)
        self._history_timer.setSingleShot(True)
        self._history_timer.setInterval(180)
        self._history_timer.timeout.connect(self._commit_history)
        self.view_box.sigRangeChanged.connect(self._view_range_changed)
        QTimer.singleShot(0, self._commit_history)
        self._update_history_buttons()
        self.adjustSize()

    def _button(self, icon_name: str, tooltip: str, *, checkable: bool = False):
        button = QToolButton(self)
        button.setObjectName("plotToolButton")
        button.setProperty("cinematicIconName", icon_name)
        button.setIcon(cinematic_icon(icon_name, THEMES[DEFAULT_THEME_NAME]["icon"]))
        button.setIconSize(QSize(17, 17))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setCheckable(checkable)
        button.setFixedSize(29, 29)
        return button

    def reposition(self) -> None:
        self.adjustSize()
        compact = self.plot_widget.width() < 390
        for button in (self.back_button, self.forward_button, self.copy_button):
            button.setVisible(not compact)
        self.adjustSize()
        x = max(8, self.plot_widget.width() - self.width() - 10)
        self.move(x, 8)
        self.raise_()

    def _current_range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        ranges = self.view_box.viewRange()
        return (
            (float(ranges[0][0]), float(ranges[0][1])),
            (float(ranges[1][0]), float(ranges[1][1])),
        )

    @staticmethod
    def _ranges_close(left, right) -> bool:
        values = [*left[0], *left[1], *right[0], *right[1]]
        scale = max(1.0, *(abs(value) for value in values))
        return all(
            abs(a - b) <= 1e-8 * scale
            for a, b in zip((*left[0], *left[1]), (*right[0], *right[1]))
        )

    def _view_range_changed(self, *args) -> None:
        if not self._restoring_view:
            self._history_timer.start()

    def _commit_history(self) -> None:
        if self._restoring_view:
            return
        current = self._current_range()
        empty_view = ((0.0, 1.0), (0.0, 1.0))
        if (
            len(self._history) == 1
            and self._ranges_close(self._history[0], empty_view)
            and not self._ranges_close(current, empty_view)
        ):
            # Replace PyQtGraph's pre-data startup range. Treating it as a
            # real history entry makes Back appear to erase the plotted data.
            self._history[0] = current
            self._history_index = 0
            self._update_history_buttons()
            return
        if self._history_index >= 0 and self._ranges_close(
            self._history[self._history_index], current
        ):
            return
        if self._history_index + 1 < len(self._history):
            self._history = self._history[: self._history_index + 1]
        self._history.append(current)
        self._history = self._history[-40:]
        self._history_index = len(self._history) - 1
        self._update_history_buttons()

    def _restore_history_index(self, index: int) -> None:
        if not 0 <= index < len(self._history):
            return
        self._history_timer.stop()
        x_range, y_range = self._history[index]
        self._restoring_view = True
        try:
            self.view_box.setRange(xRange=x_range, yRange=y_range, padding=0)
            self._history_index = index
        finally:
            self._restoring_view = False
        self._update_history_buttons()

    def _update_history_buttons(self) -> None:
        self.back_button.setEnabled(self._history_index > 0)
        self.forward_button.setEnabled(
            0 <= self._history_index < len(self._history) - 1
        )

    def previous_view(self) -> None:
        self._restore_history_index(self._history_index - 1)

    def next_view(self) -> None:
        self._restore_history_index(self._history_index + 1)

    def auto_range(self) -> None:
        self.plot_widget.enableAutoRange()
        self.view_box.autoRange()
        self._history_timer.start()
        self.statusChanged.emit("Plot restored to the complete data range.")

    def zoom(self, factor: float) -> None:
        self._history_timer.stop()
        self._commit_history()
        self.view_box.scaleBy((float(factor), float(factor)))
        self._history_timer.start()
        self.statusChanged.emit("Plot view zoomed.")

    def set_mouse_mode(self, mode: str) -> None:
        if mode == "box":
            self.view_box.setMouseMode(pg.ViewBox.RectMode)
            self.box_zoom_button.setChecked(True)
            self.statusChanged.emit("Box zoom enabled — drag over the region to inspect.")
        else:
            self.view_box.setMouseMode(pg.ViewBox.PanMode)
            self.pan_button.setChecked(True)
            self.statusChanged.emit("Pan mode enabled — drag the plot to move the view.")

    def copy_image(self, checked=False, *, scale: int = 1) -> None:
        self.plot_widget.copy_figure_to_clipboard(scale=scale)
        self.statusChanged.emit(f"Plot copied to the clipboard at {scale}×.")

    def _suggested_name(self, suffix: str) -> str:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", self.plot_widget.copy_title).strip("._")
        return f"{stem or 'figure'}{suffix}"

    def export_png(self) -> Path | None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export this plot",
            self._suggested_name(".png"),
            "PNG image (*.png)",
        )
        if not filename:
            return None
        destination = Path(filename).with_suffix(".png")
        try:
            exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
            exporter.parameters()["width"] = 2400
            exporter.export(str(destination))
        except Exception as exc:
            QMessageBox.warning(self, "Plot export failed", str(exc))
            return None
        self.statusChanged.emit(f"Plot exported to {destination.name}.")
        return destination

    def export_svg(self) -> Path | None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export this plot",
            self._suggested_name(".svg"),
            "SVG vector image (*.svg)",
        )
        if not filename:
            return None
        destination = Path(filename).with_suffix(".svg")
        try:
            exporter = pg.exporters.SVGExporter(self.plot_widget.plotItem)
            exporter.export(str(destination))
        except Exception:
            # Some PyQtGraph/PySide combinations fail while parsing generated
            # SVG transforms. Qt's own SVG paint engine is a reliable vector
            # fallback and preserves the plot exactly as the user sees it.
            toolbar_was_visible = self.isVisible()
            self.hide()
            try:
                width = max(1, self.plot_widget.width())
                height = max(1, self.plot_widget.height())
                generator = QSvgGenerator()
                generator.setFileName(str(destination))
                generator.setSize(QSize(width, height))
                generator.setViewBox(QRect(0, 0, width, height))
                generator.setTitle(self.plot_widget.copy_title)
                painter = QPainter(generator)
                try:
                    self.plot_widget.render(painter)
                finally:
                    painter.end()
            except Exception as exc:
                QMessageBox.warning(self, "Plot export failed", str(exc))
                return None
            finally:
                self.setVisible(toolbar_was_visible)
                self.reposition()
        self.statusChanged.emit(f"Plot exported to {destination.name}.")
        return destination


class CopyablePlotWidget(pg.PlotWidget):
    """PyQtGraph plot with clipboard and optional manual-peak actions."""

    manualPeakRequested = Signal(float, float)

    def __init__(self, *args, copy_title: str = "figure", **kwargs):
        super().__init__(*args, **kwargs)
        self.copy_title = copy_title
        self._manual_peak_context_enabled = False
        self._manual_peak_context_label = "Add manual peak here"
        self.getPlotItem().getViewBox().setMenuEnabled(False)
        self.navigation_toolbar = PlotNavigationToolbar(self)
        self.navigation_toolbar.statusChanged.connect(self._publish_plot_status)


    def enable_manual_peak_context(
        self,
        enabled: bool = True,
        *,
        label: str = "Add manual peak here",
    ) -> None:
        """Show a right-click action that emits the clicked plot coordinates."""
        self._manual_peak_context_enabled = bool(enabled)
        self._manual_peak_context_label = str(label or "Add manual peak here")

    def _plot_coordinates_for_context_event(self, event) -> tuple[float, float] | None:
        scene_position = self.mapToScene(event.pos())
        view_box = self.getPlotItem().getViewBox()
        if not view_box.sceneBoundingRect().contains(scene_position):
            return None
        point = view_box.mapSceneToView(scene_position)
        return float(point.x()), float(point.y())

    def _render_pixmap(self, scale: int = 1) -> QPixmap:
        scale = max(1, int(scale))
        toolbar_was_visible = self.navigation_toolbar.isVisible()
        self.navigation_toolbar.hide()
        try:
            if scale == 1:
                return self.grab()

            target_size = QSize(
                max(1, self.width() * scale),
                max(1, self.height() * scale),
            )
            pixmap = QPixmap(target_size)
            pixmap.fill(Qt.transparent)

            painter = QPainter(pixmap)
            try:
                painter.scale(scale, scale)
                self.render(painter)
            finally:
                painter.end()
            return pixmap
        finally:
            self.navigation_toolbar.setVisible(toolbar_was_visible)
            self.navigation_toolbar.reposition()

    def copy_figure_to_clipboard(self, scale: int = 1) -> None:
        QApplication.clipboard().setPixmap(
            self._render_pixmap(scale=scale)
        )

    def _publish_plot_status(self, message: str) -> None:
        window = self.window()
        status_bar = getattr(window, "statusBar", None)
        if callable(status_bar):
            status_bar().showMessage(str(message), 3200)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "navigation_toolbar"):
            self.navigation_toolbar.reposition()

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "navigation_toolbar"):
            self.navigation_toolbar.reposition()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        coordinates = self._plot_coordinates_for_context_event(event)
        add_peak_action = None
        if self._manual_peak_context_enabled and coordinates is not None:
            add_peak_action = menu.addAction(self._manual_peak_context_label)
            menu.addSeparator()
        copy_action = menu.addAction("Copy figure image")
        copy_2x_action = menu.addAction("Copy figure image (2×)")
        menu.addSeparator()
        auto_range_action = menu.addAction("Auto-range view")
        box_zoom_action = menu.addAction("Box zoom mode")
        menu.addSeparator()
        export_png_action = menu.addAction("Save plot as PNG…")
        export_svg_action = menu.addAction("Save plot as SVG…")

        chosen = menu.exec(event.globalPos())
        if add_peak_action is not None and chosen is add_peak_action:
            self.manualPeakRequested.emit(*coordinates)
        elif chosen is copy_action:
            self.copy_figure_to_clipboard(scale=1)
        elif chosen is copy_2x_action:
            self.copy_figure_to_clipboard(scale=2)
        elif chosen is auto_range_action:
            self.navigation_toolbar.auto_range()
        elif chosen is box_zoom_action:
            self.navigation_toolbar.set_mouse_mode("box")
        elif chosen is export_png_action:
            self.navigation_toolbar.export_png()
        elif chosen is export_svg_action:
            self.navigation_toolbar.export_svg()

        event.accept()
