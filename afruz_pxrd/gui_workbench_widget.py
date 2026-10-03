from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from .context_copy import install_label_copy_menu, install_table_copy_menu
from .gui_integration import GUI_FEATURES, validate_full_gui_coverage


class FullGuiWorkbenchWidget(QWidget):
    """One launch surface for the whole Afruz workflow.

    This widget does not duplicate the scientific engines. It makes every
    major engine reachable from the GUI and gives the user a single guided
    place to start, continue, refine and export.
    """

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._build_ui()
        self.refresh_status()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        heading = QLabel("Phase 19.2 — Full GUI Workbench")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        description = QLabel(
            "All major Afruz engines are now reachable from the GUI: import, preparation, "
            "peak curation, unknown indexing, CIF library matching, structure solving, "
            "Pawley/Le Bail, Rietveld, intelligent multiphase refinement, QPA, validation, "
            "batch processing and complete reports."
        )
        description.setWordWrap(True)
        description.setObjectName("mutedLabel")
        install_label_copy_menu(description)
        layout.addWidget(description)

        status_group = QGroupBox("Current project status")
        status_layout = QGridLayout(status_group)
        self.dataset_count_label = QLabel("—")
        self.selected_dataset_label = QLabel("—")
        self.feature_count_label = QLabel("—")
        self.coverage_label = QLabel("—")
        for label in (
            self.dataset_count_label,
            self.selected_dataset_label,
            self.feature_count_label,
            self.coverage_label,
        ):
            install_label_copy_menu(label)
        status_layout.addWidget(QLabel("Datasets"), 0, 0)
        status_layout.addWidget(self.dataset_count_label, 0, 1)
        status_layout.addWidget(QLabel("Selected pattern"), 1, 0)
        status_layout.addWidget(self.selected_dataset_label, 1, 1)
        status_layout.addWidget(QLabel("GUI features"), 0, 2)
        status_layout.addWidget(self.feature_count_label, 0, 3)
        status_layout.addWidget(QLabel("Coverage audit"), 1, 2)
        status_layout.addWidget(self.coverage_label, 1, 3)
        layout.addWidget(status_group)

        action_group = QGroupBox("Fast actions")
        action_layout = QGridLayout(action_group)
        actions = [
            ("Import XRD / RD", self.main_window.import_patterns),
            ("Background", lambda: self.main_window.tabs.setCurrentWidget(self.main_window.preprocessing_tab)),
            ("Smoothing", lambda: self.main_window.tabs.setCurrentWidget(self.main_window.smoothing_tab)),
            ("Peak List", lambda: self.main_window.tabs.setCurrentWidget(self.main_window.peak_tab)),
            ("Unknown Phase", lambda: self.main_window.tabs.setCurrentWidget(self.main_window.phase_revolution_widget)),
            ("CIF & Cell", lambda: self.main_window.tabs.setCurrentWidget(self.main_window.crystal_tab)),
            ("CIF Library / Match", lambda: self.main_window.tabs.setCurrentWidget(self.main_window.cif_library_widget)),
            ("Solve Structure", lambda: self.main_window.tabs.setCurrentWidget(self.main_window.structure_solution_widget)),
            ("Pawley / Le Bail", lambda: self.main_window.tabs.setCurrentWidget(self.main_window.whole_pattern_widget)),
            ("Rietveld", lambda: self.main_window.tabs.setCurrentWidget(self.main_window.rietveld_widget)),
            ("Multiphase Refiner", lambda: self.main_window.tabs.setCurrentWidget(self.main_window.multicomponent_refiner_widget)),
            ("Validated QPA", lambda: self.main_window.tabs.setCurrentWidget(self.main_window.validated_qpa_widget)),
            ("Validation", lambda: self.main_window.tabs.setCurrentWidget(self.main_window.validation_widget)),
            ("Complete Report", self.main_window.export_complete_report_package),
        ]
        for index, (text, callback) in enumerate(actions):
            button = QPushButton(text)
            if text in {"Import XRD / RD", "Multiphase Refiner", "Complete Report"}:
                button.setObjectName("primaryButton")
            button.clicked.connect(callback)
            action_layout.addWidget(button, index // 4, index % 4)
        layout.addWidget(action_group)

        self.feature_table = QTableWidget(0, 4)
        self.feature_table.setHorizontalHeaderLabels(["Workspace", "Feature", "GUI tab", "Status"])
        self.feature_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.feature_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.feature_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        install_table_copy_menu(self.feature_table)
        layout.addWidget(self.feature_table, 1)

        note = QLabel(
            "Scientific boundary: this page launches engines and records GUI coverage. "
            "Final scientific claims still require accepted results, validation and report export."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        install_label_copy_menu(note)
        layout.addWidget(note)

    def refresh_status(self):
        datasets = getattr(self.main_window, "datasets", [])
        selected = self.main_window.selected_dataset() if hasattr(self.main_window, "selected_dataset") else None
        audit = validate_full_gui_coverage()
        self.dataset_count_label.setText(str(len(datasets)))
        self.selected_dataset_label.setText(selected.name if selected is not None else "No pattern selected")
        self.feature_count_label.setText(str(audit["feature_count"]))
        self.coverage_label.setText("Complete" if audit["success"] else "Missing: " + ", ".join(audit["missing_keys"]))

        self.feature_table.setRowCount(0)
        for feature in GUI_FEATURES:
            row = self.feature_table.rowCount()
            self.feature_table.insertRow(row)
            values = [feature.workspace, feature.label, feature.tab_title, "Available in GUI"]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 3:
                    item.setTextAlignment(Qt.AlignCenter)
                self.feature_table.setItem(row, column, item)

    def apply_theme(self, theme_name: str):
        return None
