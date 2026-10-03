from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PySide6.QtCore import QEvent, QSize, QStandardPaths, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .cinematic_ui import cinematic_icon


PROJECT_SUFFIX = ".afz"
CIF_SUFFIX = ".cif"
PATTERN_SUFFIXES = frozenset(
    {
        ".xrdml",
        ".rd",
        ".xy",
        ".csv",
        ".txt",
        ".dat",
        ".pdfcard",
        ".jcpds",
        ".jade",
        ".card",
        ".ref",
    }
)
SUPPORTED_DROP_SUFFIXES = PATTERN_SUFFIXES | {PROJECT_SUFFIX, CIF_SUFFIX}
MAX_DROP_FILES = 500


def can_accept_drop_paths(paths: Iterable[str | Path]) -> bool:
    """Fast drag-hover check; folder expansion is deferred until the drop."""
    for path_value in paths:
        path = Path(path_value).expanduser().absolute()
        if path.is_dir() or (
            path.is_file() and path.suffix.casefold() in SUPPORTED_DROP_SUFFIXES
        ):
            return True
    return False


@dataclass(frozen=True)
class RecentProjectEntry:
    path: Path
    last_opened: float

    @property
    def name(self) -> str:
        return self.path.stem or self.path.name

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def age_label(self, now: float | None = None) -> str:
        seconds = max(0, int((time.time() if now is None else now) - self.last_opened))
        if seconds < 60:
            return "just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago" if days < 30 else "older"


class RecentProjectsStore:
    """Small, failure-tolerant JSON store for recently used AFZ projects."""

    def __init__(self, path: str | Path | None = None, *, maximum: int = 10):
        if path is None:
            config_root = QStandardPaths.writableLocation(
                QStandardPaths.AppConfigLocation
            )
            path = Path(config_root) / "recent-projects.json"
        self.path = Path(path)
        self.maximum = max(1, int(maximum))

    @staticmethod
    def _normalized(path: str | Path) -> Path:
        return Path(path).expanduser().absolute()

    @staticmethod
    def _identity(path: Path) -> str:
        return str(path).replace("/", "\\").casefold()

    def entries(self, *, limit: int | None = None) -> list[RecentProjectEntry]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return []
        rows = payload.get("projects", []) if isinstance(payload, dict) else []
        entries: list[RecentProjectEntry] = []
        seen: set[str] = set()
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or not row.get("path"):
                continue
            path = self._normalized(str(row["path"]))
            identity = self._identity(path)
            if identity in seen or path.suffix.casefold() != PROJECT_SUFFIX:
                continue
            seen.add(identity)
            try:
                last_opened = float(row.get("last_opened", 0.0))
            except (TypeError, ValueError):
                last_opened = 0.0
            entries.append(RecentProjectEntry(path, last_opened))
        entries.sort(key=lambda entry: entry.last_opened, reverse=True)
        cap = self.maximum if limit is None else max(0, min(int(limit), self.maximum))
        return entries[:cap]

    def _write(self, entries: Iterable[RecentProjectEntry]) -> None:
        payload = {
            "version": 1,
            "projects": [
                {"path": str(entry.path), "last_opened": entry.last_opened}
                for entry in list(entries)[: self.maximum]
            ],
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        except OSError:
            # Recent history must never block project saving or opening.
            return

    def record(self, path: str | Path, *, when: float | None = None) -> None:
        project = self._normalized(path)
        if project.suffix.casefold() != PROJECT_SUFFIX:
            return
        identity = self._identity(project)
        retained = [
            entry
            for entry in self.entries()
            if self._identity(entry.path) != identity
        ]
        retained.insert(0, RecentProjectEntry(project, time.time() if when is None else float(when)))
        self._write(retained)

    def remove(self, path: str | Path) -> None:
        identity = self._identity(self._normalized(path))
        self._write(
            entry for entry in self.entries() if self._identity(entry.path) != identity
        )

    def clear(self) -> None:
        self._write([])


@dataclass(frozen=True)
class DropClassification:
    project: Path | None
    cifs: tuple[Path, ...]
    patterns: tuple[Path, ...]
    rejected: tuple[Path, ...]

    @property
    def accepted_count(self) -> int:
        return int(self.project is not None) + len(self.cifs) + len(self.patterns)


def classify_drop_paths(paths: Iterable[str | Path]) -> DropClassification:
    """Classify local files and safely expand dropped folders."""
    expanded: list[Path] = []
    for path_value in paths:
        path = Path(path_value).expanduser().absolute()
        if path.is_dir():
            try:
                for child in path.rglob("*"):
                    if child.is_file() and child.suffix.casefold() in SUPPORTED_DROP_SUFFIXES:
                        expanded.append(child)
                        if len(expanded) >= MAX_DROP_FILES:
                            break
            except OSError:
                expanded.append(path)
        else:
            expanded.append(path)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in expanded:
        identity = str(path).replace("/", "\\").casefold()
        if identity not in seen:
            seen.add(identity)
            unique.append(path)

    projects = [path for path in unique if path.is_file() and path.suffix.casefold() == PROJECT_SUFFIX]
    cifs = [path for path in unique if path.is_file() and path.suffix.casefold() == CIF_SUFFIX]
    patterns = [path for path in unique if path.is_file() and path.suffix.casefold() in PATTERN_SUFFIXES]
    accepted = set(projects[:1]) | set(cifs[:1]) | set(patterns)
    rejected = [path for path in unique if path not in accepted]
    return DropClassification(
        project=projects[0] if projects else None,
        cifs=tuple(cifs[:1]),
        patterns=tuple(patterns),
        rejected=tuple(rejected),
    )


class RecentProjectsPanel(QFrame):
    openRequested = Signal(str)
    removeRequested = Signal(str)
    clearRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("recentProjectsPanel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 9, 12, 10)
        outer.setSpacing(6)

        heading = QHBoxLayout()
        title = QLabel("Recent projects")
        title.setObjectName("homeQuickTitle")
        self.count_label = QLabel("0")
        self.count_label.setObjectName("recentProjectCount")
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("recentProjectClear")
        self.clear_button.clicked.connect(self.clearRequested.emit)
        heading.addWidget(title)
        heading.addWidget(self.count_label)
        heading.addStretch(1)
        heading.addWidget(self.clear_button)
        outer.addLayout(heading)

        self.rows_widget = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(4)
        outer.addWidget(self.rows_widget)

        self.empty_label = QLabel("Projects you open or save will appear here.")
        self.empty_label.setObjectName("mutedLabel")
        self.empty_label.setWordWrap(True)
        self.rows_layout.addWidget(self.empty_label)

    def _clear_rows(self) -> None:
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def set_entries(self, entries: Iterable[RecentProjectEntry]) -> None:
        rows = list(entries)
        self._clear_rows()
        self.count_label.setText(str(len(rows)))
        self.clear_button.setEnabled(bool(rows))
        if not rows:
            self.empty_label = QLabel("Projects you open or save will appear here.")
            self.empty_label.setObjectName("mutedLabel")
            self.empty_label.setWordWrap(True)
            self.rows_layout.addWidget(self.empty_label)
            return

        for entry in rows:
            row = QFrame()
            row.setObjectName("recentProjectRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(5, 3, 4, 3)
            row_layout.setSpacing(5)
            button = QPushButton(
                f"{entry.name}\n{entry.path.parent.name}  ·  {entry.age_label()}"
            )
            button.setObjectName("recentProjectButton")
            button.setToolTip(str(entry.path))
            button.setEnabled(entry.exists)
            button.clicked.connect(
                lambda checked=False, path=str(entry.path): self.openRequested.emit(path)
            )
            remove = QPushButton("×")
            remove.setObjectName("recentProjectRemove")
            remove.setToolTip("Remove from recent projects")
            remove.setFixedSize(28, 28)
            remove.clicked.connect(
                lambda checked=False, path=str(entry.path): self.removeRequested.emit(path)
            )
            row_layout.addWidget(button, 1)
            row_layout.addWidget(remove)
            self.rows_layout.addWidget(row)


class DropImportOverlay(QFrame):
    """Fast, non-interactive overlay shown while supported files are dragged in."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("dropImportOverlay")
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        parent.installEventFilter(self)

        outer = QVBoxLayout(self)
        outer.addStretch(1)
        card = QFrame()
        card.setObjectName("dropImportCard")
        card.setMaximumWidth(560)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 26, 30, 26)
        card_layout.setSpacing(8)
        icon = QLabel()
        icon.setObjectName("dropImportIcon")
        icon.setAlignment(Qt.AlignCenter)
        icon.setPixmap(cinematic_icon("upload", "#42dcff").pixmap(QSize(52, 52)))
        self.title = QLabel("Drop to import")
        self.title.setObjectName("dropImportTitle")
        self.title.setAlignment(Qt.AlignCenter)
        detail = QLabel("AFZ project · XRD patterns · CIF structure · supported folders")
        detail.setObjectName("dropImportDetail")
        detail.setAlignment(Qt.AlignCenter)
        detail.setWordWrap(True)
        card_layout.addWidget(icon)
        card_layout.addWidget(self.title)
        card_layout.addWidget(detail)
        outer.addWidget(card, 0, Qt.AlignHCenter)
        outer.addStretch(1)
        self.setGeometry(parent.rect())
        self.hide()

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() == QEvent.Resize:
            self.setGeometry(watched.rect())
        return super().eventFilter(watched, event)

    def show_for_count(self, count: int) -> None:
        self.title.setText("Drop to import" if count != 1 else "Drop file to import")
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
