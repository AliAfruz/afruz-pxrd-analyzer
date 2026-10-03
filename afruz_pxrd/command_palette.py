"""Fast searchable command launcher for the desktop workbench."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QEvent, QRect, QSize, Qt, QPropertyAnimation, Signal
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from .cinematic_ui import AuraBrandMark, cinematic_icon
from .theme import DEFAULT_THEME_NAME, THEMES


COMMAND_ROLE = int(Qt.UserRole) + 1
DETAIL_ROLE = int(Qt.UserRole) + 2
SHORTCUT_ROLE = int(Qt.UserRole) + 3
ENABLED_ROLE = int(Qt.UserRole) + 4


@dataclass(frozen=True)
class CommandItem:
    label: str
    category: str
    handler: Callable[[], None]
    icon: str = "sparkles"
    shortcut: str = ""
    detail: str = ""
    keywords: tuple[str, ...] = ()
    enabled: bool | Callable[[], bool] = True

    def is_enabled(self) -> bool:
        return bool(self.enabled() if callable(self.enabled) else self.enabled)


class _CommandDelegate(QStyledItemDelegate):
    def __init__(self, theme_name: str, parent=None):
        super().__init__(parent)
        self.theme_name = theme_name

    def set_theme(self, theme_name: str) -> None:
        self.theme_name = theme_name if theme_name in THEMES else DEFAULT_THEME_NAME

    def sizeHint(self, option, index):
        return QSize(max(420, option.rect.width()), 62)

    def paint(self, painter: QPainter, option, index) -> None:
        theme = THEMES.get(self.theme_name, THEMES[DEFAULT_THEME_NAME])
        rect = option.rect.adjusted(5, 3, -5, -3)
        selected = bool(option.state & QStyle.State_Selected)
        enabled = bool(index.data(ENABLED_ROLE))

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        background = QColor(theme["panel_2"] if selected else theme["input"])
        background.setAlpha(245 if selected else 175)
        painter.setBrush(background)
        border = QColor(theme["cyan"] if selected else theme["border_soft"])
        painter.setPen(QPen(border, 1.1))
        painter.drawRoundedRect(rect, 10, 10)

        icon_rect = QRect(rect.left() + 13, rect.top() + 16, 26, 26)
        index.data(Qt.DecorationRole).paint(painter, icon_rect)

        shortcut = str(index.data(SHORTCUT_ROLE) or "")
        shortcut_width = 0
        if shortcut:
            shortcut_width = max(54, painter.fontMetrics().horizontalAdvance(shortcut) + 20)
            shortcut_rect = QRect(
                rect.right() - shortcut_width - 12,
                rect.top() + 17,
                shortcut_width,
                25,
            )
            painter.setBrush(QColor(theme["glass_2"]))
            painter.setPen(QPen(QColor(theme["border"]), 1))
            painter.drawRoundedRect(shortcut_rect, 7, 7)
            painter.setPen(QColor(theme["cyan"] if enabled else theme["muted"]))
            painter.setFont(QFont(painter.font().family(), 8, QFont.DemiBold))
            painter.drawText(shortcut_rect, Qt.AlignCenter, shortcut)

        text_left = icon_rect.right() + 13
        text_right = rect.right() - shortcut_width - (28 if shortcut else 12)
        label_rect = QRect(text_left, rect.top() + 9, max(20, text_right - text_left), 22)
        detail_rect = QRect(text_left, rect.top() + 31, max(20, text_right - text_left), 19)
        painter.setFont(QFont(painter.font().family(), 10, QFont.DemiBold))
        painter.setPen(QColor(theme["text"] if enabled else theme["muted"]))
        painter.drawText(label_rect, Qt.AlignLeft | Qt.AlignVCenter, str(index.data(Qt.DisplayRole)))
        painter.setFont(QFont(painter.font().family(), 8))
        painter.setPen(QColor(theme["cyan"] if selected and enabled else theme["muted"]))
        detail = str(index.data(DETAIL_ROLE) or "")
        painter.drawText(detail_rect, Qt.AlignLeft | Qt.AlignVCenter, detail)
        painter.restore()


class CommandPalette(QDialog):
    """A keyboard-first launcher for project, workflow and view commands."""

    commandTriggered = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("commandPalette")
        self.setWindowTitle("Afruz Command Center")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumSize(650, 500)
        self.resize(720, 560)
        self._theme_name = DEFAULT_THEME_NAME
        self._commands: list[CommandItem] = []
        self._visible_commands: list[CommandItem] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(22, 22, 22, 22)
        self.frame = QFrame()
        self.frame.setObjectName("commandPaletteFrame")
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(18, 16, 18, 14)
        frame_layout.setSpacing(11)

        header = QHBoxLayout()
        mark = AuraBrandMark(self.frame)
        mark.setFixedSize(34, 34)
        header.addWidget(mark)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel("Command Center")
        title.setObjectName("commandPaletteTitle")
        subtitle = QLabel("Jump anywhere without leaving your scientific context")
        subtitle.setObjectName("mutedLabel")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)
        keycap = QLabel("CTRL  K")
        keycap.setObjectName("commandKeycap")
        header.addWidget(keycap)
        frame_layout.addLayout(header)

        self.search = QLineEdit()
        self.search.setObjectName("commandSearch")
        self.search.setPlaceholderText("Search actions, analyses, workspaces or themes…")
        self.search.setClearButtonEnabled(True)
        self.search.addAction(cinematic_icon("search", THEMES[DEFAULT_THEME_NAME]["icon"]), QLineEdit.LeadingPosition)
        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._run_current)
        self.search.installEventFilter(self)
        frame_layout.addWidget(self.search)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("commandList")
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.itemActivated.connect(lambda _item: self._run_current())
        self.delegate = _CommandDelegate(self._theme_name, self.list_widget)
        self.list_widget.setItemDelegate(self.delegate)
        frame_layout.addWidget(self.list_widget, 1)

        footer = QHBoxLayout()
        self.count_label = QLabel("0 commands")
        self.count_label.setObjectName("mutedLabel")
        footer.addWidget(self.count_label)
        footer.addStretch(1)
        hint = QLabel("↑↓ navigate   Enter run   Esc close")
        hint.setObjectName("commandPaletteHint")
        footer.addWidget(hint)
        frame_layout.addLayout(footer)
        outer.addWidget(self.frame)

        shadow = QGraphicsDropShadowEffect(self.frame)
        shadow.setOffset(0, 10)
        shadow.setBlurRadius(44)
        shadow.setColor(QColor(0, 0, 0, 170))
        self.frame.setGraphicsEffect(shadow)
        self._appearance = QPropertyAnimation(self, b"windowOpacity", self)
        self._appearance.setDuration(150)
        self._appearance.setStartValue(0.55)
        self._appearance.setEndValue(1.0)

    def set_theme(self, theme_name: str) -> None:
        self._theme_name = theme_name if theme_name in THEMES else DEFAULT_THEME_NAME
        self.delegate.set_theme(self._theme_name)
        color = THEMES[self._theme_name]["icon"]
        actions = self.search.actions()
        if actions:
            actions[0].setIcon(cinematic_icon("search", color))
        for mark in self.findChildren(AuraBrandMark):
            mark.set_theme(self._theme_name)
        self._filter(self.search.text())

    def set_commands(self, commands: list[CommandItem]) -> None:
        self._commands = list(commands)
        self._filter(self.search.text())

    def open_centered(self, *, reduced_motion: bool = False) -> None:
        self.search.clear()
        parent = self.parentWidget()
        if parent is not None:
            width = min(760, max(650, parent.width() - 80))
            height = min(620, max(500, parent.height() - 100))
            self.resize(width, height)
            center = parent.frameGeometry().center()
            self.move(center.x() - width // 2, center.y() - height // 2)
        self.setWindowOpacity(1.0 if reduced_motion else 0.55)
        self.show()
        self.raise_()
        self.activateWindow()
        self.search.setFocus(Qt.ShortcutFocusReason)
        if not reduced_motion:
            self._appearance.stop()
            self._appearance.start()

    def _score(self, command: CommandItem, query: str) -> int:
        label = command.label.casefold()
        category = command.category.casefold()
        detail = command.detail.casefold()
        haystack = " ".join((label, category, detail, *(word.casefold() for word in command.keywords)))
        tokens = [token for token in query.casefold().split() if token]
        if not all(token in haystack for token in tokens):
            return -1
        if not tokens:
            return 10
        score = sum(20 for token in tokens if token in label)
        if label.startswith(tokens[0]):
            score += 80
        elif tokens[0] in label:
            score += 45
        if tokens[0] in category:
            score += 15
        return score

    def _filter(self, query: str) -> None:
        ranked = [
            (self._score(command, query), position, command)
            for position, command in enumerate(self._commands)
        ]
        ranked = [row for row in ranked if row[0] >= 0]
        ranked.sort(key=lambda row: (-row[0], row[1]))
        self._visible_commands = [row[2] for row in ranked]
        self.list_widget.clear()
        color = THEMES[self._theme_name]["icon"]
        for position, command in enumerate(self._visible_commands):
            enabled = command.is_enabled()
            item = QListWidgetItem(command.label)
            item.setData(COMMAND_ROLE, position)
            item.setData(DETAIL_ROLE, f"{command.category}  ·  {command.detail}".rstrip("  ·  "))
            item.setData(SHORTCUT_ROLE, command.shortcut)
            item.setData(ENABLED_ROLE, enabled)
            item.setIcon(cinematic_icon(command.icon, color if enabled else THEMES[self._theme_name]["muted"]))
            item.setFlags(item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.list_widget.addItem(item)
        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)
        self.count_label.setText(f"{len(self._visible_commands)} command{'s' if len(self._visible_commands) != 1 else ''}")

    def _run_current(self) -> None:
        item = self.list_widget.currentItem()
        if item is None:
            return
        position = int(item.data(COMMAND_ROLE))
        if not 0 <= position < len(self._visible_commands):
            return
        command = self._visible_commands[position]
        if not command.is_enabled():
            return
        self.hide()
        self.commandTriggered.emit(command.label)
        command.handler()

    def eventFilter(self, watched, event):
        if watched is self.search and event.type() == QEvent.KeyPress:
            key_event = event if isinstance(event, QKeyEvent) else None
            if key_event is not None and key_event.key() in (Qt.Key_Down, Qt.Key_Up):
                count = self.list_widget.count()
                if count:
                    delta = 1 if key_event.key() == Qt.Key_Down else -1
                    self.list_widget.setCurrentRow((self.list_widget.currentRow() + delta) % count)
                    self.list_widget.scrollToItem(self.list_widget.currentItem())
                return True
            if key_event is not None and key_event.key() == Qt.Key_Escape:
                self.hide()
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self.hide()
            return
        super().keyPressEvent(event)
