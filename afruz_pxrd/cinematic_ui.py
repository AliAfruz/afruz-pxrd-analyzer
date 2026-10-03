"""Shared cinematic visual language, vector icons, and restrained UI motion."""
from __future__ import annotations

from functools import lru_cache
import math
import os

from PySide6.QtCore import (
    QEasingCurve, QEvent, QObject, QPoint, QPropertyAnimation, QRectF, QSize,
    Qt, QTimer, QVariantAnimation,
)
from PySide6.QtGui import (
    QAction, QColor, QIcon, QLinearGradient, QPainter, QPen, QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QAbstractButton, QGraphicsDropShadowEffect, QGraphicsOpacityEffect,
    QTabWidget, QWidget,
)

from .theme import DEFAULT_THEME_NAME, THEMES


_ICON_BODY = {
    "new": '<path d="M12 5v14M5 12h14"/><path d="M5 3h8l6 6v12H5z" opacity=".45"/>',
    "open": '<path d="M3 7h7l2 2h9l-2 10H5z"/><path d="M3 7V5h7l2 2h7v2"/>',
    "save": '<path d="M4 4h14l2 2v14H4z"/><path d="M8 4v6h8V4M8 20v-6h8v6"/>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M5 15v5h14v-5"/>',
    "download": '<path d="M12 4v12M7 11l5 5 5-5"/><path d="M5 19h14"/>',
    "undo": '<path d="M9 7 4 12l5 5"/><path d="M5 12h8a6 6 0 0 1 6 6"/>',
    "redo": '<path d="m15 7 5 5-5 5"/><path d="M19 12h-8a6 6 0 0 0-6 6"/>',
    "home": '<path d="m3 11 9-8 9 8"/><path d="M5 10v11h14V10M9 21v-7h6v7"/>',
    "grid": '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/>',
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/>',
    "next": '<path d="M5 12h14M14 7l5 5-5 5"/>',
    "status": '<path d="M12 3 4 6v6c0 5 3.4 8 8 9 4.6-1 8-4 8-9V6z"/><path d="m8 12 2.5 2.5L16 9"/>',
    "play": '<path d="M8 5v14l11-7z"/>',
    "sparkles": '<path d="m12 3 1.3 4.2L17 9l-3.7 1.8L12 15l-1.3-4.2L7 9l3.7-1.8z"/><path d="m5 15 .7 2.3L8 18l-2.3.7L5 21l-.7-2.3L2 18l2.3-.7zM19 3l.6 1.9L21.5 5l-1.9.6L19 7.5l-.6-1.9-1.9-.6 1.9-.1z"/>',
    "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5"/>',
    "zoom_in": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5M7.5 10.5h6M10.5 7.5v6"/>',
    "zoom_out": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5M7.5 10.5h6"/>',
    "pan": '<path d="M12 3v18M3 12h18M12 3 9 6M12 3l3 3M21 12l-3-3M21 12l-3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3"/>',
    "box_zoom": '<rect x="3" y="3" width="14" height="14" rx="2" stroke-dasharray="2 2"/><path d="m15 15 6 6M7 10h6M10 7v6"/>',
    "eye": '<path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6z"/><circle cx="12" cy="12" r="2.7"/>',
    "edit": '<path d="m4 20 4.5-1 10-10-3.5-3.5-10 10zM13.8 6.7l3.5 3.5"/>',
    "copy": '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V4H4v12h4"/>',
    "trash": '<path d="M4 7h16M9 7V4h6v3M7 7l1 14h8l1-14M10 11v6M14 11v6"/>',
    "reset": '<path d="M4 7v6h6"/><path d="M5.5 17a8 8 0 1 0 0-10L4 9"/>',
    "x": '<path d="m5 5 14 14M19 5 5 19"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "report": '<path d="M5 3h10l4 4v14H5zM14 3v5h5"/><path d="M8 12h8M8 16h8"/>',
    "crystal": '<path d="m12 2 8 5v10l-8 5-8-5V7zM4 7l8 5 8-5M12 12v10"/>',
    "sliders": '<path d="M4 6h6M14 6h6M4 12h11M19 12h1M4 18h2M10 18h10"/><circle cx="12" cy="6" r="2"/><circle cx="17" cy="12" r="2"/><circle cx="8" cy="18" r="2"/>',
    "details": '<path d="m6 9 6 6 6-6"/>',
    "fullscreen": '<path d="M8 3H3v5M16 3h5v5M21 16v5h-5M3 16v5h5"/>',
    "check": '<path d="m4 12 5 5L20 6"/>',
}


@lru_cache(maxsize=256)
def cinematic_icon(name: str, color: str = "#c4b5fd") -> QIcon:
    body = _ICON_BODY.get(name, _ICON_BODY["sparkles"])
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        f"{body}</svg>"
    )
    pixmap = QPixmap()
    if not pixmap.loadFromData(svg.encode("utf-8"), "SVG"):
        return QIcon()
    return QIcon(pixmap)


def set_cinematic_icon(widget, name: str, color: str | None = None) -> None:
    widget.setProperty("cinematicIconName", name)
    chosen = color or THEMES[DEFAULT_THEME_NAME].get("icon", THEMES[DEFAULT_THEME_NAME]["gold_bright"])
    widget.setIcon(cinematic_icon(name, chosen))
    if hasattr(widget, "setIconSize"):
        widget.setIconSize(QSize(17, 17))


def _icon_name(text: str, object_name: str = "") -> str | None:
    value = f"{object_name} {text}".lower().replace("&", "")
    rules = (
        (("cancel", "close"), "x"),
        (("delete", "remove", "clear"), "trash"),
        (("duplicate", "copy"), "copy"),
        (("rename", "edit"), "edit"),
        (("undo",), "undo"),
        (("redo",), "redo"),
        (("new project", " new"), "new"),
        (("open", "choose folder", "browse"), "open"),
        (("import", "upload", "add current", "add active"), "upload"),
        (("save",), "save"),
        (("export", "download"), "download"),
        (("home",), "home"),
        (("workbench",), "grid"),
        (("library", "database"), "database"),
        (("phase", "multicomponent", "layer"), "layers"),
        (("crystal", "cif", "structure"), "crystal"),
        (("status", "validate", "qa", "audit"), "status"),
        (("preview", "view", "show"), "eye"),
        (("search", "find", "match", "identify"), "search"),
        (("reset", "refresh", "retry", "recalculate"), "reset"),
        (("report",), "report"),
        (("plot", "chart", "graph"), "chart"),
        (("next", "continue", "send"), "next"),
        (("run", "start", "apply", "calculate", "fit", "refine", "quantify"), "play"),
        (("setting", "parameter", "option"), "sliders"),
        (("full screen", "fullscreen"), "fullscreen"),
        (("detail",), "details"),
    )
    for needles, icon_name in rules:
        if any(needle in value for needle in needles):
            return icon_name
    return None


class AuraBackdrop(QWidget):
    """Low-cost animated backdrop with drifting spectral light and a fine grid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("auraBackdrop")
        self.setAttribute(Qt.WA_StyledBackground, False)
        self._phase = 0.0
        self._theme_name = DEFAULT_THEME_NAME
        self._reduced_motion = os.environ.get("AFRUZ_REDUCE_MOTION", "").strip() == "1"
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._advance)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._reduced_motion:
            self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def set_theme(self, theme_name: str) -> None:
        self._theme_name = theme_name if theme_name in THEMES else DEFAULT_THEME_NAME
        self.update()

    def set_reduced_motion(self, reduced: bool) -> None:
        self._reduced_motion = bool(reduced)
        if self._reduced_motion:
            self._timer.stop()
            self._phase = 0.0
            self.update()
        elif self.isVisible():
            self._timer.start()

    def _advance(self) -> None:
        self._phase = (self._phase + 0.006) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        theme = THEMES.get(self._theme_name, THEMES[DEFAULT_THEME_NAME])
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme["graphite"]))
        width, height = max(1, self.width()), max(1, self.height())
        phase = self._phase
        lights = (
            (0.18 + 0.05 * math.sin(phase), 0.10 + 0.04 * math.cos(phase), theme.get("accent", theme["gold"]), 0.48),
            (0.83 + 0.04 * math.cos(phase * .8), 0.18 + 0.05 * math.sin(phase * .7), theme.get("cyan", theme["gold_bright"]), 0.42),
            (0.62 + 0.06 * math.sin(phase * .55), 0.92 + 0.03 * math.cos(phase), theme.get("blue", theme["gold"]), 0.46),
        )
        for x_ratio, y_ratio, color_value, radius_ratio in lights:
            color = QColor(color_value)
            color.setAlpha(40 if theme.get("dark", True) else 22)
            radius = max(width, height) * radius_ratio
            gradient = QRadialGradient(width * x_ratio, height * y_ratio, radius)
            gradient.setColorAt(0.0, color)
            faded = QColor(color)
            faded.setAlpha(0)
            gradient.setColorAt(1.0, faded)
            painter.fillRect(self.rect(), gradient)
        grid_color = QColor(theme.get("grid", theme["border"]))
        grid_color.setAlpha(18 if theme.get("dark", True) else 15)
        painter.setPen(QPen(grid_color, 1))
        spacing = 42
        offset_x = int((phase * 18) % spacing)
        offset_y = int((phase * 11) % spacing)
        for x in range(-spacing + offset_x, width, spacing):
            painter.drawLine(x, 0, x, height)
        for y in range(-spacing + offset_y, height, spacing):
            painter.drawLine(0, y, width, y)


class AuraBrandMark(QWidget):
    """Animated atom/orbit mark used in the application header."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self._phase = 0.0
        self._theme_name = DEFAULT_THEME_NAME
        self._reduced_motion = False
        self._timer = QTimer(self)
        self._timer.setInterval(45)
        self._timer.timeout.connect(self._advance)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._reduced_motion:
            self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def set_theme(self, name: str) -> None:
        self._theme_name = name
        self.update()

    def set_reduced_motion(self, reduced: bool) -> None:
        self._reduced_motion = bool(reduced)
        if reduced:
            self._timer.stop()
        elif self.isVisible():
            self._timer.start()

    def _advance(self):
        self._phase = (self._phase + .035) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        theme = THEMES.get(self._theme_name, THEMES[DEFAULT_THEME_NAME])
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = self.rect().center()
        halo = QRadialGradient(center, 19)
        first = QColor(theme.get("accent", theme["gold"]))
        first.setAlpha(105)
        halo.setColorAt(0, first)
        halo.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setBrush(halo)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.rect().adjusted(1, 1, -1, -1))
        accent = QColor(theme.get("accent", theme["gold_bright"]))
        cyan = QColor(theme.get("cyan", theme["gold_bright"]))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(accent, 1.6))
        painter.save()
        painter.translate(center)
        painter.rotate(math.degrees(self._phase))
        painter.drawEllipse(QRectF(-15, -7, 30, 14))
        painter.rotate(64)
        painter.setPen(QPen(cyan, 1.4))
        painter.drawEllipse(QRectF(-15, -7, 30, 14))
        painter.restore()
        core = QRadialGradient(center - QPoint(2, 2), 8)
        core.setColorAt(0, QColor("#ffffff"))
        core.setColorAt(.35, cyan)
        core.setColorAt(1, accent)
        painter.setBrush(core)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, 6, 6)


class NavigationIndicator(QWidget):
    """Sliding spectral rail for the active workflow stage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(4)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._theme_name = DEFAULT_THEME_NAME
        self._animation = QPropertyAnimation(self, b"pos", self)
        self._animation.setDuration(230)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self.hide()

    def set_theme(self, name: str) -> None:
        self._theme_name = name
        self.update()

    def set_reduced_motion(self, reduced: bool) -> None:
        self._animation.setDuration(0 if reduced else 230)

    def move_to(self, button: QWidget, *, immediate: bool = False) -> None:
        if button is None or not button.isVisible():
            return
        target = button.mapTo(self.parentWidget(), QPoint(0, 0))
        y = target.y() + max(0, (button.height() - 30) // 2)
        self.setFixedHeight(min(30, max(18, button.height() - 8)))
        end = QPoint(2, y)
        self.show()
        self.raise_()
        if immediate or not self.isVisible():
            self.move(end)
            return
        self._animation.stop()
        self._animation.setStartValue(self.pos())
        self._animation.setEndValue(end)
        self._animation.start()

    def paintEvent(self, event):
        theme = THEMES.get(self._theme_name, THEMES[DEFAULT_THEME_NAME])
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(theme.get("cyan", theme["gold_bright"])))
        gradient.setColorAt(1, QColor(theme.get("accent", theme["gold"])))
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(self.rect(), 2, 2)


class AnimatedStatusOrb(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self._phase = 0.0
        self._busy = False
        self._theme_name = DEFAULT_THEME_NAME
        self._reduced_motion = False
        self._timer = QTimer(self)
        self._timer.setInterval(55)
        self._timer.timeout.connect(self._advance)

    def set_theme(self, name: str) -> None:
        self._theme_name = name
        self.update()

    def set_reduced_motion(self, reduced: bool) -> None:
        self._reduced_motion = bool(reduced)
        if reduced:
            self._timer.stop()
        elif self._busy:
            self._timer.start()

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        if self._busy and not self._reduced_motion:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _advance(self):
        self._phase = (self._phase + .12) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        theme = THEMES.get(self._theme_name, THEMES[DEFAULT_THEME_NAME])
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(theme.get("cyan" if self._busy else "success", theme["gold_bright"]))
        pulse = 1.0 + (.16 * math.sin(self._phase) if self._busy else 0.0)
        radius = 4.5 * pulse
        halo = QColor(color)
        halo.setAlpha(55)
        painter.setBrush(halo)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.rect().center(), 7, 7)
        painter.setBrush(color)
        center = self.rect().center()
        painter.drawEllipse(
            QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
        )


class _GlowFilter(QObject):
    def __init__(self, button: QAbstractButton, color: QColor, reduced: bool, parent=None):
        super().__init__(parent)
        self.button = button
        self.effect = QGraphicsDropShadowEffect(button)
        self.effect.setOffset(0, 0)
        self.effect.setBlurRadius(5)
        glow = QColor(color)
        glow.setAlpha(150)
        self.effect.setColor(glow)
        button.setGraphicsEffect(self.effect)
        self.animation = QPropertyAnimation(self.effect, b"blurRadius", self)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.setDuration(0 if reduced else 170)

    def set_theme(self, color: QColor) -> None:
        glow = QColor(color)
        glow.setAlpha(150)
        self.effect.setColor(glow)

    def set_reduced_motion(self, reduced: bool) -> None:
        self.animation.setDuration(0 if reduced else 170)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Enter, QEvent.Leave):
            self.animation.stop()
            self.animation.setStartValue(self.effect.blurRadius())
            self.animation.setEndValue(24.0 if event.type() == QEvent.Enter else 5.0)
            self.animation.start()
        return False


class CinematicUiController(QObject):
    """Decorate existing widgets without changing scientific widget contracts."""

    def __init__(self, root: QWidget):
        super().__init__(root)
        self.root = root
        self.theme_name = DEFAULT_THEME_NAME
        self.reduced_motion = os.environ.get("AFRUZ_REDUCE_MOTION", "").strip() == "1"
        self._tab_animations = {}
        self._glows: dict[int, _GlowFilter] = {}
        self._installed_tabs: set[int] = set()

    def install(self) -> None:
        self._decorate_tree(self.root)
        self.apply_theme(self.theme_name)

    def decorate(self, widget: QWidget) -> None:
        """Decorate a fully constructed dialog or late-created panel."""
        self._decorate_tree(widget)
        self.apply_theme(self.theme_name)

    def _decorate_tree(self, root: QWidget) -> None:
        buttons = ([root] if isinstance(root, QAbstractButton) else []) + root.findChildren(QAbstractButton)
        for button in buttons:
            if button.property("cinematicIconName") is None and button.icon().isNull():
                name = _icon_name(button.text(), button.objectName())
                if name:
                    button.setProperty("cinematicIconName", name)
            if button.property("cinematicIconName") and (
                button.objectName() == "primaryButton" or button.objectName() in {"headerRunButton", "headerStatusButton"}
            ):
                self._install_glow(button)
        tabs = ([root] if isinstance(root, QTabWidget) else []) + root.findChildren(QTabWidget)
        for tabs_widget in tabs:
            identity = id(tabs_widget)
            if identity not in self._installed_tabs:
                tabs_widget.currentChanged.connect(lambda index, tabs=tabs_widget: self._animate_tab(tabs, index))
                self._installed_tabs.add(identity)
            for index in range(tabs_widget.count()):
                if tabs_widget.tabIcon(index).isNull():
                    name = _icon_name(tabs_widget.tabText(index)) or "chart"
                    tabs_widget.setTabIcon(index, cinematic_icon(name, self._icon_color()))
        for action in root.findChildren(QAction):
            if action.property("cinematicIconName") is None and action.icon().isNull():
                name = _icon_name(action.text(), action.objectName())
                if name:
                    action.setProperty("cinematicIconName", name)

    def _install_glow(self, button: QAbstractButton) -> None:
        identity = id(button)
        if identity in self._glows or button.graphicsEffect() is not None:
            return
        glow = _GlowFilter(button, QColor(self._accent_color()), self.reduced_motion, self)
        button.installEventFilter(glow)
        self._glows[identity] = glow
        button.destroyed.connect(lambda *_args, key=identity: self._glows.pop(key, None))

    def _animate_tab(self, tabs: QTabWidget, index: int) -> None:
        if self.reduced_motion or not 0 <= index < tabs.count():
            return
        page = tabs.widget(index)
        if page is None:
            return
        effect = page.graphicsEffect()
        if effect is not None:
            page.setGraphicsEffect(None)
        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        animation = QVariantAnimation(page)
        animation.setDuration(190)
        animation.setStartValue(0.18)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.valueChanged.connect(lambda value, effect=effect: effect.setOpacity(float(value)))
        animation.finished.connect(lambda page=page: page.setGraphicsEffect(None))
        self._tab_animations[id(tabs)] = animation
        animation.start()

    def _theme(self) -> dict:
        return THEMES.get(self.theme_name, THEMES[DEFAULT_THEME_NAME])

    def _icon_color(self) -> str:
        return self._theme().get("icon", self._theme()["gold_bright"])

    def _accent_color(self) -> str:
        return self._theme().get("accent", self._theme()["gold"])

    def apply_theme(self, theme_name: str) -> None:
        self.theme_name = theme_name if theme_name in THEMES else DEFAULT_THEME_NAME
        icon_color = self._icon_color()
        for button in self.root.findChildren(QAbstractButton):
            name = button.property("cinematicIconName")
            if name:
                button.setIcon(cinematic_icon(str(name), icon_color))
                button.setIconSize(QSize(17, 17))
        for action in self.root.findChildren(QAction):
            name = action.property("cinematicIconName")
            if name:
                action.setIcon(cinematic_icon(str(name), icon_color))
        for tabs in self.root.findChildren(QTabWidget):
            for index in range(tabs.count()):
                name = _icon_name(tabs.tabText(index)) or "chart"
                tabs.setTabIcon(index, cinematic_icon(name, icon_color))
        for component_type in (AuraBackdrop, AuraBrandMark, NavigationIndicator, AnimatedStatusOrb):
            for component in self.root.findChildren(component_type):
                component.set_theme(self.theme_name)
        for glow in list(self._glows.values()):
            glow.set_theme(QColor(self._accent_color()))

    def set_reduced_motion(self, reduced: bool) -> None:
        self.reduced_motion = bool(reduced)
        for component_type in (AuraBackdrop, AuraBrandMark, NavigationIndicator, AnimatedStatusOrb):
            for component in self.root.findChildren(component_type):
                component.set_reduced_motion(self.reduced_motion)
        for glow in list(self._glows.values()):
            glow.set_reduced_motion(self.reduced_motion)
