from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .scientific_state import DatasetScientificState, STAGE_LABELS, STAGE_ORDER


class ScientificStateDialog(QDialog):
    acceptRequested = Signal(str)

    def __init__(self, state: DatasetScientificState, dataset_name: str, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Unified Scientific State")
        self.resize(940, 460)

        layout = QVBoxLayout(self)
        title = QLabel(f"Scientific state — {dataset_name}")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        explanation = QLabel(
            "Each result stores the exact upstream revisions that produced it. "
            "Upstream changes preserve older results but mark them Outdated."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("mutedLabel")
        layout.addWidget(explanation)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Stage",
                "Status",
                "Revision",
                "Accepted",
                "Inputs",
                "Updated",
                "Reason",
                "Result ID",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        action_row = QHBoxLayout()
        self.accept_button = QPushButton("Accept selected current result")
        self.accept_button.setObjectName("primaryButton")
        self.accept_button.clicked.connect(self._accept_selected)
        action_row.addWidget(self.accept_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()

    def refresh(self) -> None:
        self.state.reconcile()
        self.table.setRowCount(0)
        for key in STAGE_ORDER:
            node = self.state.node(key)
            row = self.table.rowCount()
            self.table.insertRow(row)
            inputs = ", ".join(
                f"{STAGE_LABELS.get(dep, dep)} r{revision}"
                for dep, revision in node.dependencies.items()
            ) or "—"
            values = (
                STAGE_LABELS[key],
                node.status,
                str(node.revision) if node.revision else "—",
                "Yes" if node.is_accepted_current else (
                    f"Historical r{node.accepted_revision}" if node.accepted_revision else "No"
                ),
                inputs,
                node.updated_at or "—",
                node.reason or "—",
                node.result_uid[:12] if node.result_uid else "—",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, key)
                if column in (1, 2, 3):
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        if self.table.rowCount():
            self.table.selectRow(0)

    def _accept_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        item = self.table.item(row, 0)
        if item is None:
            return
        key = item.data(Qt.UserRole)
        self.acceptRequested.emit(str(key))
        self.refresh()
