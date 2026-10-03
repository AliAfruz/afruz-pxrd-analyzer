from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .workflow import (
    STATUS_LABELS,
    STATUS_ORDER,
    TASK_BY_KEY,
    WORKSPACES,
    WorkflowTaskState,
)
from .version import APP_NAME, APP_VERSION
from .project_access import RecentProjectsPanel
from .cinematic_ui import (
    AnimatedStatusOrb, AuraBrandMark, NavigationIndicator, set_cinematic_icon,
)


class UnifiedWorkflowHeader(QWidget):
    workspaceSelected = Signal(str)
    taskSelected = Signal(str)
    modeChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._task_keys: list[str] = []
        self._updating = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 4)
        outer.setSpacing(4)

        navigation = QFrame()
        navigation.setObjectName("workflowNavigation")
        nav_layout = QHBoxLayout(navigation)
        nav_layout.setContentsMargins(8, 6, 8, 6)
        nav_layout.setSpacing(4)

        self.workspace_group = QButtonGroup(self)
        self.workspace_group.setExclusive(True)
        self.workspace_buttons: dict[str, QPushButton] = {}
        for workspace in WORKSPACES:
            button = QPushButton(workspace.short_label)
            button.setCheckable(True)
            button.setObjectName("workflowWorkspaceButton")
            button.setToolTip(f"{workspace.label}\n{workspace.description}")
            button.clicked.connect(
                lambda checked=False, key=workspace.key: self.workspaceSelected.emit(key)
            )
            self.workspace_group.addButton(button)
            self.workspace_buttons[workspace.key] = button
            nav_layout.addWidget(button)

        nav_layout.addStretch(1)
        nav_layout.addWidget(QLabel("Mode"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Guided", "Expert"])
        self.mode_selector.setToolTip(
            "Guided mode shows the essential scientific route. Expert mode exposes every specialist workspace."
        )
        self.mode_selector.currentTextChanged.connect(self._mode_text_changed)
        nav_layout.addWidget(self.mode_selector)
        outer.addWidget(navigation)

        task_frame = QFrame()
        task_frame.setObjectName("workflowTaskBar")
        task_layout = QHBoxLayout(task_frame)
        task_layout.setContentsMargins(8, 5, 8, 5)
        task_layout.addWidget(QLabel("Current task"))
        self.task_selector = QComboBox()
        self.task_selector.setMinimumWidth(230)
        self.task_selector.currentIndexChanged.connect(self._task_index_changed)
        task_layout.addWidget(self.task_selector)
        self.guidance_label = QLabel("Select a workflow task.")
        self.guidance_label.setObjectName("mutedLabel")
        self.guidance_label.setWordWrap(True)
        task_layout.addWidget(self.guidance_label, 1)
        outer.addWidget(task_frame)

        status_frame = QFrame()
        status_frame.setObjectName("workflowStatusBar")
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(8, 4, 8, 4)
        status_layout.setSpacing(5)
        self.status_labels: dict[str, QLabel] = {}
        for key in STATUS_ORDER:
            label = QLabel(f"{STATUS_LABELS[key]} —")
            label.setAlignment(Qt.AlignCenter)
            label.setObjectName("workflowStatusPending")
            label.setMinimumWidth(88)
            status_layout.addWidget(label)
            self.status_labels[key] = label
        status_layout.addStretch(1)
        outer.addWidget(status_frame)

    def _mode_text_changed(self, mode: str) -> None:
        if not self._updating:
            self.modeChanged.emit(mode)

    def _task_index_changed(self, index: int) -> None:
        if self._updating or not 0 <= index < len(self._task_keys):
            return
        self.taskSelected.emit(self._task_keys[index])

    def set_mode(self, mode: str) -> None:
        index = self.mode_selector.findText(mode)
        if index >= 0 and index != self.mode_selector.currentIndex():
            self._updating = True
            try:
                self.mode_selector.setCurrentIndex(index)
            finally:
                self._updating = False

    def set_active_workspace(self, workspace_key: str) -> None:
        button = self.workspace_buttons.get(workspace_key)
        if button is not None and not button.isChecked():
            button.setChecked(True)

    def set_tasks(self, task_keys: list[str], active_task_key: str | None = None) -> None:
        self._updating = True
        try:
            self._task_keys = list(task_keys)
            self.task_selector.clear()
            for key in self._task_keys:
                task = TASK_BY_KEY[key]
                self.task_selector.addItem(task.label)
            if active_task_key in self._task_keys:
                self.task_selector.setCurrentIndex(self._task_keys.index(active_task_key))
            elif self._task_keys:
                self.task_selector.setCurrentIndex(0)
        finally:
            self._updating = False
        self.set_active_task(active_task_key or (self._task_keys[0] if self._task_keys else None))

    def set_active_task(self, task_key: str | None) -> None:
        if task_key in self._task_keys:
            index = self._task_keys.index(task_key)
            if index != self.task_selector.currentIndex():
                self._updating = True
                try:
                    self.task_selector.setCurrentIndex(index)
                finally:
                    self._updating = False
            self.guidance_label.setText(TASK_BY_KEY[task_key].guidance)

    def set_statuses(self, states: dict[str, str]) -> None:
        for key, label in self.status_labels.items():
            state = states.get(key, "Not ready")
            label.setText(f"{STATUS_LABELS[key]}  {self._symbol(state)}")
            label.setToolTip(state)
            if state == "Complete":
                label.setObjectName("workflowStatusComplete")
            elif state == "Next":
                label.setObjectName("workflowStatusNext")
            elif state in {"Outdated", "Invalid"}:
                label.setObjectName("workflowStatusOutdated")
            else:
                label.setObjectName("workflowStatusPending")
            label.style().unpolish(label)
            label.style().polish(label)

    @staticmethod
    def _symbol(state: str) -> str:
        if state == "Complete":
            return "✓"
        if state == "Next":
            return "→"
        if state in {"Outdated", "Invalid"}:
            return "⚠"
        return "—"


class CleanApplicationHeader(QFrame):
    """Compact Phase 20 application header.

    It intentionally preserves the workflow-header API used by older code while
    moving stage navigation into the left sidebar.
    """

    workspaceSelected = Signal(str)
    taskSelected = Signal(str)
    modeChanged = Signal(str)
    runRequested = Signal()
    saveRequested = Signal()
    statusRequested = Signal()
    commandRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cleanApplicationHeader")
        self._task_keys: list[str] = []
        self._updating = False
        self._active_workspace = "home"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(8)

        brand_box = QVBoxLayout()
        brand_box.setSpacing(0)
        self.brand_mark = AuraBrandMark(self)
        layout.addWidget(self.brand_mark)
        self.brand_label = QLabel(APP_NAME)
        self.brand_label.setObjectName("appBrandLabel")
        self.version_label = QLabel(f"Version {APP_VERSION}")
        self.version_label.setObjectName("appVersionLabel")
        brand_box.addWidget(self.brand_label)
        brand_box.addWidget(self.version_label)
        layout.addLayout(brand_box)

        layout.addWidget(self._divider())

        project_box = QVBoxLayout()
        project_box.setSpacing(0)
        project_caption = QLabel("Project")
        project_caption.setObjectName("headerCaption")
        self.project_label = QLabel("Untitled project")
        self.project_label.setObjectName("headerValue")
        self.project_label.setMinimumWidth(125)
        project_box.addWidget(project_caption)
        project_box.addWidget(self.project_label)
        layout.addLayout(project_box)

        dataset_box = QVBoxLayout()
        dataset_box.setSpacing(0)
        dataset_caption = QLabel("Dataset")
        dataset_caption.setObjectName("headerCaption")
        self.dataset_label = QLabel("No dataset")
        self.dataset_label.setObjectName("headerValue")
        self.dataset_label.setMinimumWidth(110)
        dataset_box.addWidget(dataset_caption)
        dataset_box.addWidget(self.dataset_label)
        layout.addLayout(dataset_box)

        task_box = QVBoxLayout()
        task_box.setSpacing(1)
        task_caption = QLabel("Current task")
        task_caption.setObjectName("headerCaption")
        self.task_selector = QComboBox()
        self.task_selector.setObjectName("headerTaskSelector")
        self.task_selector.setMinimumWidth(190)
        self.task_selector.currentIndexChanged.connect(self._task_index_changed)
        task_box.addWidget(task_caption)
        task_box.addWidget(self.task_selector)
        layout.addLayout(task_box, 1)

        mode_box = QVBoxLayout()
        mode_box.setSpacing(1)
        mode_caption = QLabel("Mode")
        mode_caption.setObjectName("headerCaption")
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Guided", "Expert"])
        self.mode_selector.setMinimumWidth(96)
        self.mode_selector.currentTextChanged.connect(self._mode_text_changed)
        mode_box.addWidget(mode_caption)
        mode_box.addWidget(self.mode_selector)
        layout.addLayout(mode_box)

        self.save_state_label = QLabel("Not saved")
        self.save_state_label.setObjectName("saveStateUnsaved")
        self.save_state_label.setAlignment(Qt.AlignCenter)
        self.save_state_label.setMinimumWidth(76)
        layout.addWidget(self.save_state_label)

        self.command_button = QPushButton("")
        self.command_button.setObjectName("headerCommandButton")
        self.command_button.setFixedSize(36, 34)
        set_cinematic_icon(self.command_button, "search")
        self.command_button.setToolTip("Open Command Center (Ctrl+K)")
        self.command_button.clicked.connect(self.commandRequested.emit)
        layout.addWidget(self.command_button)

        self.status_button = QPushButton("Status")
        self.status_button.setObjectName("headerStatusButton")
        set_cinematic_icon(self.status_button, "status")
        self.status_button.setToolTip("Open the scientific-state audit for the selected dataset")
        self.status_button.clicked.connect(self.statusRequested.emit)
        layout.addWidget(self.status_button)

        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("headerSaveButton")
        set_cinematic_icon(self.save_button, "save")
        self.save_button.clicked.connect(self.saveRequested.emit)
        layout.addWidget(self.save_button)

        self.run_button = QPushButton("Run analysis")
        self.run_button.setObjectName("headerRunButton")
        set_cinematic_icon(self.run_button, "play")
        self.run_button.clicked.connect(self.runRequested.emit)
        layout.addWidget(self.run_button)

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setObjectName("headerDivider")
        return line

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.command_button.setVisible(event.size().width() >= 1650)

    def _mode_text_changed(self, mode: str) -> None:
        if not self._updating:
            self.modeChanged.emit(mode)

    def _task_index_changed(self, index: int) -> None:
        if self._updating or not 0 <= index < len(self._task_keys):
            return
        self.taskSelected.emit(self._task_keys[index])

    def set_project_name(self, name: str) -> None:
        value = str(name).strip() or "Untitled project"
        self.project_label.setText(value)
        self.project_label.setToolTip(value)

    def set_dataset_name(self, name: str | None) -> None:
        value = str(name).strip() if name else "No dataset"
        self.dataset_label.setText(value)
        self.dataset_label.setToolTip(value)
        self.run_button.setEnabled(bool(name))

    def set_save_state(self, state: str) -> None:
        normalized = str(state).strip().lower()
        if normalized == "saved":
            text, object_name = "Saved", "saveStateSaved"
        elif normalized == "modified":
            text, object_name = "Modified", "saveStateModified"
        else:
            text, object_name = "Not saved", "saveStateUnsaved"
        self.save_state_label.setText(text)
        self.save_state_label.setObjectName(object_name)
        self.save_state_label.style().unpolish(self.save_state_label)
        self.save_state_label.style().polish(self.save_state_label)

    def set_mode(self, mode: str) -> None:
        index = self.mode_selector.findText(mode)
        if index >= 0 and index != self.mode_selector.currentIndex():
            self._updating = True
            try:
                self.mode_selector.setCurrentIndex(index)
            finally:
                self._updating = False

    def set_active_workspace(self, workspace_key: str) -> None:
        self._active_workspace = str(workspace_key)

    def set_tasks(self, task_keys: list[str], active_task_key: str | None = None) -> None:
        self._updating = True
        try:
            self._task_keys = list(task_keys)
            self.task_selector.clear()
            for key in self._task_keys:
                task = TASK_BY_KEY[key]
                self.task_selector.addItem(task.label)
            if active_task_key in self._task_keys:
                self.task_selector.setCurrentIndex(self._task_keys.index(active_task_key))
            elif self._task_keys:
                self.task_selector.setCurrentIndex(0)
        finally:
            self._updating = False
        self.set_active_task(active_task_key or (self._task_keys[0] if self._task_keys else None))

    def set_task_states(self, states: dict[str, WorkflowTaskState]) -> None:
        """Decorate the task selector without changing its selected task."""
        self._updating = True
        try:
            for index, key in enumerate(self._task_keys):
                state = states.get(key)
                task = TASK_BY_KEY[key]
                text = task.label if state is None else f"{state.symbol}  {task.label}"
                self.task_selector.setItemText(index, text)
                if state is not None:
                    self.task_selector.setItemData(index, state.reason, Qt.ToolTipRole)
        finally:
            self._updating = False

    def set_active_task(self, task_key: str | None) -> None:
        if task_key in self._task_keys:
            index = self._task_keys.index(task_key)
            if index != self.task_selector.currentIndex():
                self._updating = True
                try:
                    self.task_selector.setCurrentIndex(index)
                finally:
                    self._updating = False
            task = TASK_BY_KEY[task_key]
            self.task_selector.setToolTip(task.guidance)
            self.run_button.setToolTip(f"Run or open the primary action for: {task.label}")

    def set_run_label(
        self,
        text: str,
        *,
        enabled: bool | None = None,
        tooltip: str | None = None,
    ) -> None:
        self.run_button.setText(str(text))
        if enabled is not None:
            self.run_button.setEnabled(bool(enabled))
        if tooltip is not None:
            self.run_button.setToolTip(str(tooltip))

    def set_statuses(self, states: dict[str, str]) -> None:
        completed = sum(1 for state in states.values() if state == "Complete")
        total = max(1, len(states))
        warnings = sum(1 for state in states.values() if state in {"Outdated", "Invalid"})
        if warnings:
            self.status_button.setText(f"Status  ⚠ {warnings}")
        else:
            self.status_button.setText(f"Status  {completed}/{total}")
        details = "\n".join(
            f"{STATUS_LABELS.get(key, key.title())}: {value}"
            for key, value in states.items()
        )
        self.status_button.setToolTip(details or "No workflow status available")


class WorkflowNavigationPanel(QFrame):
    """Persistent left-side workflow navigator for Phase 20."""

    workspaceSelected = Signal(str)
    taskSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workflowSidebarNavigation")
        self._updating = False
        self._task_keys: list[str] = []
        self._task_states: dict[str, WorkflowTaskState] = {}
        self._active_workspace = "home"

        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        title_row = QHBoxLayout()
        title = QLabel("Workflow")
        title.setObjectName("sectionTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)
        self.mode_badge = QLabel("Guided")
        self.mode_badge.setObjectName("workflowModeBadge")
        title_row.addWidget(self.mode_badge)
        layout.addLayout(title_row)

        self.workspace_group = QButtonGroup(self)
        self.workspace_group.setExclusive(True)
        self.workspace_buttons: dict[str, QPushButton] = {}
        workspace_icons = {
            "home": "home", "preparation": "sliders", "peaks": "chart",
            "phase_structure": "crystal", "refinement_qpa": "layers",
            "validation_reports": "status",
        }
        for number, workspace in enumerate(WORKSPACES, start=1):
            display_label = workspace.label.replace("&", "&&")
            button = QPushButton(f"{number}.  {display_label}")
            button.setCheckable(True)
            button.setObjectName("workflowSidebarButton")
            button.setToolTip(workspace.description)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setMinimumHeight(34)
            set_cinematic_icon(button, workspace_icons.get(workspace.key, "sparkles"))
            button.clicked.connect(
                lambda checked=False, key=workspace.key: self._workspace_clicked(key)
            )
            self.workspace_group.addButton(button)
            self.workspace_buttons[workspace.key] = button
            layout.addWidget(button)

        self.selection_indicator = NavigationIndicator(self)

        self.stage_description = QLabel("Review the project and choose the next scientific step.")
        self.stage_description.setWordWrap(True)
        self.stage_description.setObjectName("mutedLabel")
        layout.addWidget(self.stage_description)

        self.task_caption = QLabel("Tasks in this stage")
        self.task_caption.setObjectName("headerCaption")
        layout.addWidget(self.task_caption)

        self.task_list = QListWidget()
        self.task_list.setObjectName("workflowTaskList")
        self.task_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.task_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.task_list.setMinimumHeight(38)
        self.task_list.setMaximumHeight(112)
        self.task_list.currentRowChanged.connect(self._task_row_changed)
        layout.addWidget(self.task_list)

        self.guidance_label = QLabel("Select a task.")
        self.guidance_label.setObjectName("workflowGuidance")
        self.guidance_label.setWordWrap(True)
        layout.addWidget(self.guidance_label)

        self.task_state_label = QLabel("")
        self.task_state_label.setObjectName("workflowTaskNotice")
        self.task_state_label.setWordWrap(True)
        self.task_state_label.setVisible(False)
        layout.addWidget(self.task_state_label)

    def _workspace_clicked(self, workspace_key: str) -> None:
        if not self._updating:
            self.workspaceSelected.emit(workspace_key)

    def _task_row_changed(self, row: int) -> None:
        if self._updating or not 0 <= row < len(self._task_keys):
            return
        self.taskSelected.emit(self._task_keys[row])

    def set_mode(self, mode: str) -> None:
        self.mode_badge.setText(mode)

    def set_active_workspace(self, workspace_key: str) -> None:
        self._active_workspace = str(workspace_key)
        button = self.workspace_buttons.get(workspace_key)
        if button is not None and not button.isChecked():
            self._updating = True
            try:
                button.setChecked(True)
            finally:
                self._updating = False
        workspace = next((row for row in WORKSPACES if row.key == workspace_key), None)
        if workspace is not None:
            self.stage_description.setText(workspace.description)
        if button is not None:
            QTimer.singleShot(0, lambda button=button: self.selection_indicator.move_to(button))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        button = self.workspace_buttons.get(self._active_workspace)
        if button is not None:
            QTimer.singleShot(0, lambda button=button: self.selection_indicator.move_to(button, immediate=True))

    def set_tasks(self, task_keys: list[str], active_task_key: str | None = None) -> None:
        self._updating = True
        try:
            self._task_keys = list(task_keys)
            self.task_list.clear()
            for key in self._task_keys:
                task = TASK_BY_KEY[key]
                state = self._task_states.get(key)
                text = task.label if state is None else f"{state.symbol}  {task.label}"
                item = QListWidgetItem(text)
                item.setToolTip(task.guidance if state is None else state.reason)
                if state is not None:
                    item.setData(Qt.UserRole, state.status)
                self.task_list.addItem(item)
            if active_task_key in self._task_keys:
                self.task_list.setCurrentRow(self._task_keys.index(active_task_key))
            elif self._task_keys:
                self.task_list.setCurrentRow(0)
            visible_rows = max(1, min(len(self._task_keys), 4))
            row_height = self.task_list.sizeHintForRow(0)
            if row_height <= 0:
                row_height = 26
            self.task_list.setFixedHeight(min(112, max(38, visible_rows * row_height + 8)))
        finally:
            self._updating = False
        self.set_active_task(active_task_key or (self._task_keys[0] if self._task_keys else None))

    def set_task_states(self, states: dict[str, WorkflowTaskState]) -> None:
        self._task_states = dict(states)
        current_key = (
            self._task_keys[self.task_list.currentRow()]
            if 0 <= self.task_list.currentRow() < len(self._task_keys)
            else None
        )
        self._updating = True
        try:
            for row, key in enumerate(self._task_keys):
                item = self.task_list.item(row)
                if item is None:
                    continue
                task = TASK_BY_KEY[key]
                state = self._task_states.get(key)
                item.setText(task.label if state is None else f"{state.symbol}  {task.label}")
                item.setToolTip(task.guidance if state is None else state.reason)
                if state is not None:
                    item.setData(Qt.UserRole, state.status)
        finally:
            self._updating = False
        self.set_active_task(current_key)

    def set_active_task(self, task_key: str | None) -> None:
        if task_key in self._task_keys:
            row = self._task_keys.index(task_key)
            if row != self.task_list.currentRow():
                self._updating = True
                try:
                    self.task_list.setCurrentRow(row)
                finally:
                    self._updating = False
            self.guidance_label.setText(TASK_BY_KEY[task_key].guidance)
            state = self._task_states.get(task_key)
            if state is None:
                self.task_state_label.setVisible(False)
            else:
                self.task_state_label.setText(f"{state.status}: {state.reason}")
                self.task_state_label.setProperty("taskState", state.status)
                self.task_state_label.style().unpolish(self.task_state_label)
                self.task_state_label.style().polish(self.task_state_label)
                self.task_state_label.setVisible(True)

    def set_statuses(self, states: dict[str, str]) -> None:
        status_key_by_workspace = {
            "home": "import",
            "preparation": "preparation",
            "peaks": "peaks",
            "phase_structure": "phase",
            "validation_reports": "validation",
        }
        for number, workspace in enumerate(WORKSPACES, start=1):
            if workspace.key == "refinement_qpa":
                pair = (states.get("refinement", "Not ready"), states.get("qpa", "Not ready"))
                if any(value == "Invalid" for value in pair):
                    state = "Invalid"
                elif any(value == "Outdated" for value in pair):
                    state = "Outdated"
                elif all(value == "Complete" for value in pair):
                    state = "Complete"
                elif any(value == "Next" for value in pair):
                    state = "Next"
                else:
                    state = "Not ready"
            else:
                state = states.get(status_key_by_workspace.get(workspace.key, ""), "Not ready")
            symbol = UnifiedWorkflowHeader._symbol(state)
            button = self.workspace_buttons[workspace.key]
            display_label = workspace.label.replace("&", "&&")
            button.setText(f"{number}.  {display_label}  {symbol}")
            button.setProperty("workflowState", state.replace(" ", "_"))
            button.style().unpolish(button)
            button.style().polish(button)


class ProgressDrawer(QFrame):
    """Compact bottom drawer that mirrors application progress and messages."""

    cancellationRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("progressDrawer")
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._expanded = False
        self._last_message = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(5)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.status_orb = AnimatedStatusOrb(self)
        row.addWidget(self.status_orb)
        self.activity_label = QLabel("Ready")
        self.activity_label.setObjectName("progressDrawerStatus")
        self.activity_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        row.addWidget(self.activity_label, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(220)
        row.addWidget(self.progress_bar)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancellationRequested.emit)
        row.addWidget(self.cancel_button)

        self.toggle_button = QPushButton("Details ▾")
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self.set_expanded)
        row.addWidget(self.toggle_button)
        layout.addLayout(row)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("progressLog")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setMaximumHeight(0)
        self.log_view.setVisible(False)
        layout.addWidget(self.log_view)
        self._reduced_motion = False
        self._drawer_animation = QPropertyAnimation(self.log_view, b"maximumHeight", self)
        self._drawer_animation.setDuration(220)
        self._drawer_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._drawer_animation.finished.connect(self._drawer_animation_finished)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self.toggle_button.setText("Details ▴" if self._expanded else "Details ▾")
        self._drawer_animation.stop()
        if self._expanded:
            self.log_view.setVisible(True)
            self._drawer_animation.setStartValue(self.log_view.maximumHeight())
            self._drawer_animation.setEndValue(140)
        else:
            self._drawer_animation.setStartValue(max(0, self.log_view.height()))
            self._drawer_animation.setEndValue(0)
        self._drawer_animation.setDuration(0 if self._reduced_motion else 220)
        self._drawer_animation.start()

    def _drawer_animation_finished(self) -> None:
        if not self._expanded:
            self.log_view.setVisible(False)

    def set_reduced_motion(self, reduced: bool) -> None:
        self._reduced_motion = bool(reduced)
        self._drawer_animation.setDuration(0 if reduced else 220)

    def set_status(self, message: str) -> None:
        cleaned = " ".join(str(message).split())
        if not cleaned:
            return
        self.activity_label.setText(cleaned)
        self.activity_label.setToolTip(cleaned)
        if cleaned != self._last_message:
            self.log_view.appendPlainText(cleaned)
            self._last_message = cleaned

    def set_progress(self, value: int, maximum: int = 100, *, label: str | None = None) -> None:
        maximum = max(1, int(maximum))
        value = max(0, min(int(value), maximum))
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(value)
        busy = 0 < value < maximum
        self.status_orb.set_busy(busy)
        self.cancel_button.setEnabled(busy)
        if label:
            self.set_status(label)

    def set_busy(self, busy: bool, *, label: str | None = None) -> None:
        if busy:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        self.cancel_button.setEnabled(bool(busy))
        self.status_orb.set_busy(bool(busy))
        if label:
            self.set_status(label)


class ProjectHomeWidget(QWidget):
    continueRequested = Signal()
    importRequested = Signal()
    openRequested = Signal()
    recentProjectRequested = Signal(str)
    recentProjectRemoveRequested = Signal(str)
    recentProjectsClearRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Project Home")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "One dataset, one prepared pattern, one authoritative peak list and traceable downstream results."
        )
        subtitle.setObjectName("mutedLabel")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        cards = QFrame()
        cards.setObjectName("workflowHomeCards")
        grid = QGridLayout(cards)
        grid.setContentsMargins(10, 10, 10, 10)
        self.dataset_value = QLabel("No dataset selected")
        self.pattern_value = QLabel("Not prepared")
        self.peaks_value = QLabel("No curated peaks")
        self.phase_value = QLabel("No accepted phase or cell")
        self.refinement_value = QLabel("No current refinement")
        self.validation_value = QLabel("Not validated")
        values = (
            ("Dataset", self.dataset_value),
            ("Prepared pattern", self.pattern_value),
            ("Master peaks", self.peaks_value),
            ("Phase / structure", self.phase_value),
            ("Refinement", self.refinement_value),
            ("Validation", self.validation_value),
        )
        for row, (name, value) in enumerate(values):
            name_label = QLabel(name)
            name_label.setObjectName("mutedLabel")
            value.setWordWrap(True)
            grid.addWidget(name_label, row, 0)
            grid.addWidget(value, row, 1)
        layout.addWidget(cards)

        self.warning_label = QLabel("Import a powder pattern to begin.")
        self.warning_label.setWordWrap(True)
        self.warning_label.setObjectName("workflowHomeNotice")
        layout.addWidget(self.warning_label)

        quick_actions = QFrame()
        quick_actions.setObjectName("homeQuickActions")
        quick_layout = QVBoxLayout(quick_actions)
        quick_layout.setContentsMargins(12, 10, 12, 12)
        quick_layout.setSpacing(8)
        quick_heading = QHBoxLayout()
        quick_title = QLabel("Quick start")
        quick_title.setObjectName("homeQuickTitle")
        quick_heading.addWidget(quick_title)
        quick_heading.addStretch(1)
        command_hint = QLabel("Ctrl+K  Command Center")
        command_hint.setObjectName("commandKeycap")
        quick_heading.addWidget(command_hint)
        quick_layout.addLayout(quick_heading)
        quick_note = QLabel("Begin with measured data or continue an existing project. Afruz keeps the recommended scientific route visible as you work.")
        quick_note.setObjectName("mutedLabel")
        quick_note.setWordWrap(True)
        quick_layout.addWidget(quick_note)

        action_row = QHBoxLayout()
        self.import_button = QPushButton("Import first pattern")
        self.import_button.setObjectName("primaryButton")
        self.import_button.clicked.connect(self.importRequested.emit)
        action_row.addWidget(self.import_button)
        self.open_button = QPushButton("Open project")
        self.open_button.clicked.connect(self.openRequested.emit)
        action_row.addWidget(self.open_button)
        self.continue_button = QPushButton("Recommended next step")
        self.continue_button.clicked.connect(self.continueRequested.emit)
        action_row.addWidget(self.continue_button)
        action_row.addStretch(1)
        quick_layout.addLayout(action_row)
        layout.addWidget(quick_actions)

        self.recent_projects_panel = RecentProjectsPanel()
        self.recent_projects_panel.openRequested.connect(
            self.recentProjectRequested.emit
        )
        self.recent_projects_panel.removeRequested.connect(
            self.recentProjectRemoveRequested.emit
        )
        self.recent_projects_panel.clearRequested.connect(
            self.recentProjectsClearRequested.emit
        )
        layout.addWidget(self.recent_projects_panel)
        layout.addStretch(1)

    def set_recent_projects(self, entries) -> None:
        self.recent_projects_panel.set_entries(entries)

    def update_summary(
        self,
        *,
        dataset: str,
        pattern: str,
        peaks: str,
        phase: str,
        refinement: str,
        validation: str,
        notice: str,
        has_dataset: bool,
    ) -> None:
        self.dataset_value.setText(dataset)
        self.pattern_value.setText(pattern)
        self.peaks_value.setText(peaks)
        self.phase_value.setText(phase)
        self.refinement_value.setText(refinement)
        self.validation_value.setText(validation)
        self.warning_label.setText(notice)
        self.import_button.setVisible(not has_dataset)
        self.continue_button.setEnabled(has_dataset)
        self.continue_button.setObjectName("primaryButton" if has_dataset else "")
        self.continue_button.style().unpolish(self.continue_button)
        self.continue_button.style().polish(self.continue_button)
