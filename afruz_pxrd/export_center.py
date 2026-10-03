from __future__ import annotations

"""Phase 22 GUI for previewing and running clean scientific exports."""

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .export_engine import (
    CATEGORY_LABELS,
    CLEAN_CONTENT_LABELS,
    ExportPackage,
    build_clean_export_package,
    build_export_package,
    export_package_bundle,
    preview_package,
)
from .context_copy import install_label_copy_menu, install_table_copy_menu
from .text_export import safe_filename


class ExportWorker(QObject):
    progress = Signal(int, str)
    completed = Signal(dict)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        package: ExportPackage,
        output_directory: str,
        base_name: str,
        excel: bool,
        text: bool,
        zip_text: bool,
        compact_text: bool,
    ):
        super().__init__()
        self.package = package
        self.output_directory = output_directory
        self.base_name = base_name
        self.excel = excel
        self.text = text
        self.zip_text = zip_text
        self.compact_text = compact_text
        self._cancelled = False

    @Slot()
    def cancel(self):
        self._cancelled = True

    @Slot()
    def run(self):
        try:
            result = export_package_bundle(
                self.package,
                self.output_directory,
                base_name=self.base_name,
                excel=self.excel,
                text=self.text,
                zip_text=self.zip_text,
                compact_text=self.compact_text,
                progress=lambda value, text: self.progress.emit(value, text),
                cancelled=lambda: self._cancelled,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class ExportCenterDialog(QDialog):
    """Preview table shapes before writing Excel or clean TXT packages."""

    def __init__(self, main_window, *, preferred_formats: set[str] | None = None, preferred_categories: set[str] | None = None):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Clean Data Export")
        self.resize(1040, 720)
        self.package: ExportPackage | None = None
        self.thread: QThread | None = None
        self.worker: ExportWorker | None = None
        self.preferred_formats = {"text"} if preferred_formats is None else set(preferred_formats)
        # Old workflow buttons requested every format.  Keep them useful, but
        # start with the simplest reproducible output instead of three copies.
        if self.preferred_formats == {"excel", "text", "zip"}:
            self.preferred_formats = {"text"}
        self.preferred_categories = preferred_categories
        self._build_ui()
        self._connect()
        self.refresh_preview()

    def _build_ui(self):
        root = QVBoxLayout(self)
        intro = QLabel(
            "Export only the data you need: raw, treated, peaks and refinement. "
            "Every TXT file is a clean, plot-ready table with units in the headings."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        options = QGroupBox("Export plan")
        options_layout = QGridLayout(options)
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("Selected dataset only", "selected")
        self.scope_combo.addItem("All project datasets", "all")
        self.excel_check = QCheckBox("Also create Excel workbook (.xlsx)")
        self.text_check = QCheckBox("Clean TXT data folder (recommended)")
        self.zip_check = QCheckBox("Also create ZIP")
        self.excel_check.setChecked("excel" in self.preferred_formats)
        self.text_check.setChecked("text" in self.preferred_formats or "zip" in self.preferred_formats)
        self.zip_check.setChecked("zip" in self.preferred_formats)
        self.destination_edit = QLineEdit()
        default_dir = (
            str(self.main_window.current_project.parent)
            if getattr(self.main_window, "current_project", None)
            else str(Path.home())
        )
        self.destination_edit.setText(default_dir)
        self.browse_button = QPushButton("Browse…")
        self.base_name_edit = QLineEdit()
        project_name = (
            self.main_window.current_project.stem
            if getattr(self.main_window, "current_project", None)
            else "Afruz_PXRD_Data"
        )
        self.base_name_edit.setText(safe_filename(project_name))
        options_layout.addWidget(QLabel("Scope"), 0, 0)
        options_layout.addWidget(self.scope_combo, 0, 1, 1, 2)
        options_layout.addWidget(self.excel_check, 1, 0)
        options_layout.addWidget(self.text_check, 1, 1)
        options_layout.addWidget(self.zip_check, 1, 2)
        options_layout.addWidget(QLabel("Output folder"), 2, 0)
        options_layout.addWidget(self.destination_edit, 2, 1)
        options_layout.addWidget(self.browse_button, 2, 2)
        options_layout.addWidget(QLabel("Package name"), 3, 0)
        options_layout.addWidget(self.base_name_edit, 3, 1, 1, 2)
        root.addWidget(options)

        categories = QGroupBox("Data to export")
        category_layout = QGridLayout(categories)
        self.category_checks: dict[str, QCheckBox] = {}
        preferred_clean = self._preferred_clean_content()
        for index, (key, label) in enumerate(CLEAN_CONTENT_LABELS.items()):
            check = QCheckBox(label)
            check.setChecked(key in preferred_clean)
            self.category_checks[key] = check
            category_layout.addWidget(check, index // 2, index % 2)
        self.advanced_check = QCheckBox("Advanced audit export (all internal scientific tables)")
        self.advanced_check.setToolTip(
            "Use the original comprehensive export for validation archives and software auditing."
        )
        category_layout.addWidget(self.advanced_check, 2, 0, 1, 2)
        root.addWidget(categories)

        preview_group = QGroupBox("Validated table preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_table = QTableWidget(0, 7)
        self.preview_table.setHorizontalHeaderLabels(
            ["Table", "Category", "Dataset", "Rows", "Columns", "Status", "Warnings"]
        )
        header = self.preview_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        preview_layout.addWidget(self.preview_table)
        self.preview_summary = QLabel()
        self.preview_summary.setWordWrap(True)
        preview_layout.addWidget(self.preview_summary)
        root.addWidget(preview_group, 1)

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_label = QLabel("Ready")
        self.cancel_button = QPushButton("Cancel export")
        self.cancel_button.setEnabled(False)
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.progress_label, 2)
        progress_row.addWidget(self.cancel_button)
        root.addLayout(progress_row)

        buttons = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh preview")
        self.refresh_button.setObjectName("primaryButton")
        self.export_button = QPushButton("Export package")
        self.export_button.setObjectName("primaryButton")
        self.close_button = QPushButton("Close")
        buttons.addWidget(self.refresh_button)
        buttons.addStretch(1)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.close_button)
        root.addLayout(buttons)
        install_table_copy_menu(self.preview_table)
        install_label_copy_menu(intro)
        install_label_copy_menu(self.preview_summary)
        install_label_copy_menu(self.progress_label)

    def _connect(self):
        self.scope_combo.currentIndexChanged.connect(self.refresh_preview)
        for check in self.category_checks.values():
            check.toggled.connect(self.refresh_preview)
        self.advanced_check.toggled.connect(self._update_advanced_state)
        self.advanced_check.toggled.connect(self.refresh_preview)
        self.browse_button.clicked.connect(self.choose_destination)
        self.refresh_button.clicked.connect(self.refresh_preview)
        self.export_button.clicked.connect(self.start_export)
        self.cancel_button.clicked.connect(self.cancel_export)
        self.close_button.clicked.connect(self.reject)
        self.text_check.toggled.connect(self._update_format_state)
        self._update_format_state()
        self._update_advanced_state()

    def _update_format_state(self):
        self.zip_check.setEnabled(self.text_check.isChecked())
        if not self.text_check.isChecked():
            self.zip_check.setChecked(False)

    def _update_advanced_state(self):
        clean_enabled = not self.advanced_check.isChecked()
        for check in self.category_checks.values():
            check.setEnabled(clean_enabled)

    def choose_destination(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Choose clean data output folder", self.destination_edit.text()
        )
        if directory:
            self.destination_edit.setText(directory)

    def _categories(self) -> set[str]:
        return {key for key, check in self.category_checks.items() if check.isChecked()}

    def _preferred_clean_content(self) -> set[str]:
        if self.preferred_categories is None:
            return set(CLEAN_CONTENT_LABELS)
        mapping = {
            "project": {"raw"},
            "pattern": {"raw"},
            "preprocessing": {"treated"},
            "peaks": {"peaks"},
            "refinement": {"refinement"},
        }
        selected: set[str] = set()
        for category in self.preferred_categories:
            selected.update(mapping.get(category, set()))
        return selected or set(CLEAN_CONTENT_LABELS)

    def refresh_preview(self):
        try:
            snapshot = self.main_window.build_export_snapshot(
                all_datasets=self.scope_combo.currentData() == "all"
            )
            if self.advanced_check.isChecked():
                self.package = build_export_package(snapshot, include_categories=set(CATEGORY_LABELS))
            else:
                self.package = build_clean_export_package(snapshot, include_content=self._categories())
            rows = preview_package(self.package)
        except Exception as exc:
            self.package = None
            self.preview_table.setRowCount(0)
            self.preview_summary.setText(f"Preview failed: {exc}")
            self.export_button.setEnabled(False)
            return

        self.preview_table.setRowCount(len(rows))
        total_rows = 0
        total_columns = 0
        for row_index, row in enumerate(rows):
            values = [
                row["table_title"], row["category"], row["dataset"],
                row["rows"], row["columns"], row["status"], row["warnings"],
            ]
            total_rows += int(row["rows"])
            total_columns += int(row["columns"])
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                self.preview_table.setItem(row_index, column_index, item)
        if self.advanced_check.isChecked():
            summary = (
                f"Advanced audit export: {len(rows)} table(s), {total_rows:,} data row(s), "
                f"{total_columns:,} columns, data dictionary and manifest."
            )
        else:
            summary = (
                f"Clean export: {len(rows)} file(s), {total_rows:,} data row(s), "
                f"{total_columns:,} columns. Only a short README and checksum manifest are added."
            )
        self.preview_summary.setText(summary)
        self.export_button.setEnabled(bool(rows))

    def start_export(self):
        if self.package is None:
            self.refresh_preview()
        if self.package is None:
            return
        if not self.excel_check.isChecked() and not self.text_check.isChecked():
            QMessageBox.warning(self, "No format selected", "Select Excel, clean TXT, or both.")
            return
        destination = Path(self.destination_edit.text()).expanduser()
        if not str(destination).strip():
            QMessageBox.warning(self, "Output folder required", "Choose an output folder.")
            return
        try:
            destination.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "Output folder unavailable", str(exc))
            return
        base_name = safe_filename(self.base_name_edit.text(), "Afruz_PXRD_Data")

        self.thread = QThread(self)
        self.worker = ExportWorker(
            self.package,
            str(destination),
            base_name,
            self.excel_check.isChecked(),
            self.text_check.isChecked(),
            self.zip_check.isChecked(),
            not self.advanced_check.isChecked(),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._export_finished)

        self.export_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.progress_label.setText("Starting export…")
        self.thread.start()

    def cancel_export(self):
        if self.worker is not None:
            self.worker.cancel()
            self.progress_label.setText("Cancellation requested…")
            self.cancel_button.setEnabled(False)

    @Slot(int, str)
    def _on_progress(self, value: int, message: str):
        self.progress.setValue(max(0, min(100, value)))
        self.progress_label.setText(message)

    @Slot(dict)
    def _on_completed(self, result: dict[str, Any]):
        paths = []
        for label, key in (
            ("Excel", "excel_path"),
            ("TXT folder", "text_directory"),
            ("ZIP", "zip_path"),
            ("Checksums", "checksums_path"),
        ):
            if result.get(key):
                paths.append(f"{label}: {result[key]}")
        QMessageBox.information(
            self,
            "Data export completed",
            f"Exported {result.get('table_count', 0)} validated tables.\n\n" + "\n".join(paths),
        )
        self.main_window.statusBar().showMessage(
            f"Data export completed: {result.get('table_count', 0)} clean table(s)."
        )

    @Slot(str)
    def _on_failed(self, message: str):
        QMessageBox.critical(self, "Data export failed", message)
        self.progress_label.setText(f"Failed: {message}")

    def _export_finished(self):
        self.export_button.setEnabled(self.package is not None)
        self.refresh_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None
        self.thread = None

    def reject(self):
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "Export running", "Cancel or wait for the current export before closing.")
            return
        super().reject()
