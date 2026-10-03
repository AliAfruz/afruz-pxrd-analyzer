from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class AnalysisPanelDescriptor:
    title: str
    subtitle: str
    stage: str


class UnifiedAnalysisHeader(QFrame):
    """Compact heading used by every right-side analysis panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("unifiedAnalysisHeader")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(1)
        self.title_label = QLabel("Analysis controls")
        self.title_label.setObjectName("analysisPanelTitle")
        self.subtitle_label = QLabel("Select a workflow task to show its parameters.")
        self.subtitle_label.setObjectName("mutedLabel")
        self.subtitle_label.setWordWrap(True)
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.subtitle_label)
        layout.addLayout(text_layout, 1)

        self.stage_badge = QLabel("PROJECT")
        self.stage_badge.setObjectName("analysisStageBadge")
        self.stage_badge.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.stage_badge)

    def set_descriptor(self, descriptor: AnalysisPanelDescriptor) -> None:
        self.title_label.setText(descriptor.title)
        self.subtitle_label.setText(descriptor.subtitle)
        self.stage_badge.setText(descriptor.stage.upper())


class InlineValidationCard(QFrame):
    """Non-blocking prerequisite and parameter validation message."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("analysisValidationInfo")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        self.heading = QLabel("Ready")
        self.heading.setObjectName("analysisValidationHeading")
        self.message = QLabel("Parameters are ready for review.")
        self.message.setWordWrap(True)
        self.message.setObjectName("analysisValidationMessage")
        self.action = QLabel("")
        self.action.setWordWrap(True)
        self.action.setObjectName("mutedLabel")
        self.action.setVisible(False)
        layout.addWidget(self.heading)
        layout.addWidget(self.message)
        layout.addWidget(self.action)

    def set_state(
        self,
        level: str,
        heading: str,
        message: str,
        action: str = "",
    ) -> None:
        normalized = level if level in {"success", "info", "warning", "error"} else "info"
        self.setObjectName(f"analysisValidation{normalized.title()}")
        self.heading.setText(heading)
        self.message.setText(message)
        self.action.setText(action)
        self.action.setVisible(bool(action))
        self.style().unpolish(self)
        self.style().polish(self)


class AnalysisResultCard(QFrame):
    """Small scientific result summary shown before detailed controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("analysisResultCard")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)

        heading_row = QHBoxLayout()
        heading_row.setSpacing(6)
        self.title_label = QLabel("Result summary")
        self.title_label.setObjectName("analysisResultTitle")
        self.status_label = QLabel("No result")
        self.status_label.setObjectName("analysisResultPending")
        self.status_label.setAlignment(Qt.AlignCenter)
        heading_row.addWidget(self.title_label)
        heading_row.addStretch(1)
        heading_row.addWidget(self.status_label)
        outer.addLayout(heading_row)

        self.metrics_widget = QWidget()
        self.metrics_layout = QGridLayout(self.metrics_widget)
        self.metrics_layout.setContentsMargins(0, 0, 0, 0)
        self.metrics_layout.setHorizontalSpacing(8)
        self.metrics_layout.setVerticalSpacing(4)
        outer.addWidget(self.metrics_widget)

        self.note_label = QLabel("Results will appear here after the current analysis completes.")
        self.note_label.setObjectName("mutedLabel")
        self.note_label.setWordWrap(True)
        outer.addWidget(self.note_label)

    def _clear_metrics(self) -> None:
        while self.metrics_layout.count():
            item = self.metrics_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def set_summary(
        self,
        *,
        status: str,
        metrics: Iterable[tuple[str, str]] = (),
        note: str = "",
        state: str = "pending",
    ) -> None:
        object_names = {
            "success": "analysisResultSuccess",
            "warning": "analysisResultWarning",
            "error": "analysisResultError",
            "pending": "analysisResultPending",
        }
        self.status_label.setText(status)
        self.status_label.setObjectName(object_names.get(state, "analysisResultPending"))
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

        self._clear_metrics()
        rows = list(metrics)
        for row, (name, value) in enumerate(rows):
            name_label = QLabel(str(name))
            name_label.setObjectName("analysisMetricName")
            value_label = QLabel(str(value))
            value_label.setObjectName("analysisMetricValue")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.metrics_layout.addWidget(name_label, row, 0)
            self.metrics_layout.addWidget(value_label, row, 1)
        self.note_label.setText(note or "No additional result details are available yet.")


class AnalysisActionBar(QFrame):
    """Sticky Reset / Preview / Run footer with compact progress feedback."""

    resetRequested = Signal()
    previewRequested = Signal()
    runRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("analysisActionBar")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 7, 8, 7)
        outer.setSpacing(5)

        progress_row = QHBoxLayout()
        progress_row.setSpacing(7)
        self.progress_label = QLabel("Ready")
        self.progress_label.setObjectName("mutedLabel")
        self.progress_label.setMinimumWidth(62)
        self.progress_label.setMaximumWidth(125)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        progress_row.addWidget(self.progress_label)
        progress_row.addWidget(self.progress, 1)
        outer.addLayout(progress_row)

        button_row = QGridLayout()
        button_row.setSpacing(6)
        self.reset_button = QPushButton("Reset")
        self.preview_button = QPushButton("Preview")
        self.run_button = QPushButton("Run analysis")
        self.run_button.setObjectName("primaryButton")
        self.reset_button.clicked.connect(self.resetRequested.emit)
        self.preview_button.clicked.connect(self.previewRequested.emit)
        self.run_button.clicked.connect(self.runRequested.emit)
        button_row.addWidget(self.reset_button, 0, 0)
        button_row.addWidget(self.preview_button, 0, 1)
        button_row.addWidget(self.run_button, 1, 0, 1, 2)
        button_row.setColumnStretch(0, 1)
        button_row.setColumnStretch(1, 1)
        outer.addLayout(button_row)

    def configure(
        self,
        *,
        run_text: str,
        run_enabled: bool,
        preview_text: str = "Preview",
        preview_enabled: bool = False,
        reset_enabled: bool = True,
        tooltip: str = "",
    ) -> None:
        self.run_button.setText(run_text)
        self.run_button.setEnabled(run_enabled)
        self.run_button.setToolTip(tooltip)
        self.preview_button.setText(preview_text)
        self.preview_button.setEnabled(preview_enabled)
        self.reset_button.setEnabled(reset_enabled)

    def set_progress(self, value: int, maximum: int, label: str = "") -> None:
        maximum = max(1, int(maximum))
        value = max(0, min(int(value), maximum))
        self.progress.setRange(0, maximum)
        self.progress.setValue(value)
        if label:
            self._set_compact_status(label)
        elif value >= maximum:
            self._set_compact_status("Complete")
        elif value > 0:
            self._set_compact_status("Running")
        else:
            self._set_compact_status("Ready")

    def _set_compact_status(self, label: str) -> None:
        text = str(label).strip() or "Ready"
        display = text if len(text) <= 24 else text[:21].rstrip() + "…"
        self.progress_label.setText(display)
        self.progress_label.setToolTip(text)

    def set_status(self, label: str) -> None:
        self._set_compact_status(label)


class AnalysisInspectorToolbar(QFrame):
    """Search and disclosure controls for the analysis parameter inspector."""

    queryChanged = Signal(str)
    modeChanged = Signal(str)
    expandAllRequested = Signal()
    collapseAllRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("analysisInspectorToolbar")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 7, 8, 7)
        outer.setSpacing(6)

        search_row = QHBoxLayout()
        search_row.setSpacing(6)
        self.search = QLineEdit()
        self.search.setObjectName("analysisParameterSearch")
        self.search.setPlaceholderText("Find a parameter…")
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Find an analysis parameter")
        self.summary_label = QLabel("Essential controls")
        self.summary_label.setObjectName("analysisInspectorSummary")
        self.summary_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        search_row.addWidget(self.search, 1)
        search_row.addWidget(self.summary_label)
        outer.addLayout(search_row)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(5)
        self.essential_button = QPushButton("Essential")
        self.all_button = QPushButton("All controls")
        for button in (self.essential_button, self.all_button):
            button.setObjectName("analysisModeButton")
            button.setCheckable(True)
        self.essential_button.setMinimumWidth(76)
        self.all_button.setMinimumWidth(100)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.essential_button)
        self.mode_group.addButton(self.all_button)
        self.essential_button.setChecked(True)

        self.expand_button = QPushButton("▾")
        self.expand_button.setObjectName("analysisInspectorUtilityButton")
        self.expand_button.setToolTip("Expand every visible section")
        self.expand_button.setAccessibleName("Expand all visible sections")
        self.expand_button.setFixedWidth(28)
        self.collapse_button = QPushButton("›")
        self.collapse_button.setObjectName("analysisInspectorUtilityButton")
        self.collapse_button.setToolTip("Collapse every visible section")
        self.collapse_button.setAccessibleName("Collapse all visible sections")
        self.collapse_button.setFixedWidth(28)

        mode_row.addWidget(self.essential_button)
        mode_row.addWidget(self.all_button)
        mode_row.addStretch(1)
        mode_row.addWidget(self.expand_button)
        mode_row.addWidget(self.collapse_button)
        outer.addLayout(mode_row)

        self.search.textChanged.connect(self.queryChanged.emit)
        self.essential_button.clicked.connect(
            lambda: self.modeChanged.emit("essential")
        )
        self.all_button.clicked.connect(lambda: self.modeChanged.emit("all"))
        self.expand_button.clicked.connect(self.expandAllRequested.emit)
        self.collapse_button.clicked.connect(self.collapseAllRequested.emit)

    def current_mode(self) -> str:
        return "all" if self.all_button.isChecked() else "essential"

    def set_mode(self, mode: str, *, emit: bool = False) -> None:
        normalized = "all" if str(mode).lower() == "all" else "essential"
        changed = normalized != self.current_mode()
        self.all_button.setChecked(normalized == "all")
        self.essential_button.setChecked(normalized == "essential")
        if emit and changed:
            self.modeChanged.emit(normalized)

    def set_summary(self, visible_controls: int, query: str = "") -> None:
        count = max(0, int(visible_controls))
        if str(query).strip():
            text = f"{count} found"
        elif self.current_mode() == "all":
            text = f"{count} controls"
        else:
            text = f"{count} essential"
        self.summary_label.setText(text)


class AnalysisInspectorSection(QFrame):
    """Collapsible wrapper that preserves the original scientific widget tree."""

    expansionChanged = Signal(str, bool)

    def __init__(
        self,
        key: str,
        content: QGroupBox,
        *,
        title: str | None = None,
        tier: str = "essential",
        expanded: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.key = str(key)
        self.content = content
        self.tier = str(tier)
        self._expanded = bool(expanded)
        self._search_override = False
        self._row_visibility: dict[tuple[QFormLayout, int], bool] = {}
        self.setObjectName("analysisInspectorSection")
        self.setProperty("inspectorTier", self.tier)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setObjectName("analysisSectionHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(7, 5, 6, 5)
        header_layout.setSpacing(5)

        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("analysisSectionToggle")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(self._expanded)
        self.toggle_button.setAccessibleName(f"Toggle {title or content.title()} section")
        self.badge = QLabel()
        self.badge.setObjectName("analysisSectionBadge")
        self.badge.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.toggle_button, 1)
        header_layout.addWidget(self.badge)
        outer.addWidget(header)

        content.setParent(self)
        content.setProperty("inspectorEmbedded", True)
        self._title = str(title or content.title() or "Parameters")
        content.setTitle("")
        outer.addWidget(content)

        self.toggle_button.clicked.connect(self._toggle_clicked)
        self._refresh_header()
        self._refresh_content_visibility()

    def _toggle_clicked(self, checked: bool) -> None:
        self.set_expanded(bool(checked), emit=True)

    def _refresh_header(self) -> None:
        chevron = "▾" if self._expanded else "›"
        self.toggle_button.setText(f"{chevron}  {self._title}")
        labels = {
            "advanced": "ADVANCED",
            "secondary": "VIEW",
            "result": "EXPORT",
        }
        badge_text = labels.get(self.tier, "")
        self.badge.setText(badge_text)
        self.badge.setVisible(bool(badge_text))

    def _refresh_content_visibility(self) -> None:
        self.content.setVisible(self._expanded or self._search_override)

    def set_title(self, title: str) -> None:
        self._title = str(title).strip() or "Parameters"
        self.toggle_button.setAccessibleName(f"Toggle {self._title} section")
        self._refresh_header()

    def title(self) -> str:
        return self._title

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool, *, emit: bool = False) -> None:
        expanded = bool(expanded)
        changed = expanded != self._expanded
        self._expanded = expanded
        self.toggle_button.setChecked(expanded)
        self._refresh_header()
        self._refresh_content_visibility()
        if emit and changed:
            self.expansionChanged.emit(self.key, expanded)

    def set_search_override(self, active: bool) -> None:
        self._search_override = bool(active)
        self._refresh_content_visibility()

    @staticmethod
    def _layout_widgets(layout: QLayout | None) -> list[QWidget]:
        widgets: list[QWidget] = []
        if layout is None:
            return widgets
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget()
            if widget is not None:
                widgets.append(widget)
                widgets.extend(widget.findChildren(QWidget))
            elif item.layout() is not None:
                widgets.extend(AnalysisInspectorSection._layout_widgets(item.layout()))
        return widgets

    @classmethod
    def _form_row_widgets(cls, form: QFormLayout, row: int) -> list[QWidget]:
        widgets: list[QWidget] = []
        for role in (
            QFormLayout.LabelRole,
            QFormLayout.FieldRole,
            QFormLayout.SpanningRole,
        ):
            item = form.itemAt(row, role)
            if item is None:
                continue
            if item.widget() is not None:
                widget = item.widget()
                widgets.append(widget)
                widgets.extend(widget.findChildren(QWidget))
            elif item.layout() is not None:
                widgets.extend(cls._layout_widgets(item.layout()))
        return list(dict.fromkeys(widgets))

    @staticmethod
    def _widget_search_text(widget: QWidget) -> str:
        parts = [
            widget.objectName(),
            widget.toolTip(),
            widget.accessibleName(),
        ]
        if isinstance(widget, (QLabel, QAbstractButton)):
            parts.append(widget.text())
        if isinstance(widget, QLineEdit):
            parts.extend((widget.text(), widget.placeholderText()))
        if isinstance(widget, QComboBox):
            parts.append(widget.currentText())
            parts.extend(widget.itemText(index) for index in range(widget.count()))
        return " ".join(str(part) for part in parts if part).casefold()

    def clear_filters(self) -> None:
        for (form, row), visible in tuple(self._row_visibility.items()):
            try:
                form.setRowVisible(row, visible)
            except RuntimeError:
                pass
        self._row_visibility.clear()
        self.set_search_override(False)

    def apply_filters(self, query: str, *, include_advanced: bool) -> tuple[bool, int]:
        """Apply row-level filters and return (has_match, visible_control_count)."""
        normalized_query = str(query).strip().casefold()
        title_matches = bool(normalized_query and normalized_query in self._title.casefold())
        forms = self.content.findChildren(QFormLayout)
        visible_count = 0
        has_match = not normalized_query

        for form in forms:
            for row in range(form.rowCount()):
                base_visible = bool(form.isRowVisible(row))
                self._row_visibility[(form, row)] = base_visible
                widgets = self._form_row_widgets(form, row)
                is_advanced = any(
                    bool(widget.property("inspectorAdvanced")) for widget in widgets
                )
                searchable = " ".join(
                    self._widget_search_text(widget) for widget in widgets
                )
                query_matches = (
                    not normalized_query
                    or title_matches
                    or normalized_query in searchable
                )
                visible = base_visible and (include_advanced or not is_advanced) and query_matches
                form.setRowVisible(row, visible)
                if visible:
                    visible_count += 1
                    has_match = True

        if not forms:
            searchable = " ".join(
                self._widget_search_text(widget)
                for widget in [self.content, *self.content.findChildren(QWidget)]
            )
            has_match = not normalized_query or title_matches or normalized_query in searchable
            visible_count = 1 if has_match else 0

        self.set_search_override(bool(normalized_query and has_match))
        return has_match, visible_count


def capture_widget_defaults(root: QWidget) -> dict[QWidget, object]:
    """Capture editable widget values without depending on object names."""
    defaults: dict[QWidget, object] = {}
    for widget in root.findChildren(QWidget):
        if isinstance(widget, QCheckBox):
            defaults[widget] = widget.isChecked()
        elif isinstance(widget, QComboBox):
            defaults[widget] = widget.currentIndex()
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            defaults[widget] = widget.value()
        elif isinstance(widget, QLineEdit) and not isinstance(
            widget.parentWidget(), (QComboBox, QSpinBox, QDoubleSpinBox)
        ):
            defaults[widget] = widget.text()
    return defaults


def restore_widget_defaults(
    defaults: dict[QWidget, object],
    visible_root: QWidget | None = None,
    *,
    visible_only: bool = False,
) -> None:
    """Restore captured values, optionally only for visible widgets under a root."""
    for widget, value in defaults.items():
        if visible_root is not None and not visible_root.isAncestorOf(widget):
            continue
        if visible_only and widget.isHidden():
            continue
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            widget.setCurrentIndex(int(value))
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setValue(value)
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))
