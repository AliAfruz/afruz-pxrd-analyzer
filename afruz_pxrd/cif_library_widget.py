from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .cif_library import (
    CifLibrarySettings,
    IncrementalScanOptions,
    build_cif_library_scalable,
    grouped_structure_families,
    library_health_report,
)
from .context_copy import install_label_copy_menu, install_table_copy_menu
from .multicomponent_refiner import import_candidate_phase
from .processing import active_peak_rows
from .structure_matching import intelligent_match_structure_candidates
from .widgets import NoWheelDoubleSpinBox, NoWheelSpinBox


class CifLibraryManagerWidget(QWidget):
    """GUI manager for Phase 18.3 scalable CIF library and structure matching."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.last_match_result: dict | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        heading = QLabel("Phase 19.2 — CIF Library and Structure Matching GUI")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        description = QLabel(
            "Build/update a scalable local CIF database, search candidates with element/material filters, "
            "match against the current Peak List, and send selected CIF hits directly to the Intelligent Multiphase Refiner."
        )
        description.setObjectName("mutedLabel")
        description.setWordWrap(True)
        install_label_copy_menu(description)
        layout.addWidget(description)

        top = QHBoxLayout()
        top.addWidget(self._build_database_group(), 1)
        top.addWidget(self._build_search_group(), 1)
        layout.addLayout(top)

        buttons = QHBoxLayout()
        self.scan_button = QPushButton("Build / Update CIF index")
        self.scan_button.setObjectName("primaryButton")
        self.health_button = QPushButton("Library health")
        self.match_button = QPushButton("Search current Peak List")
        self.send_button = QPushButton("Send selected CIFs to Multiphase Refiner")
        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("mutedLabel")
        install_label_copy_menu(self.status_label)
        buttons.addWidget(self.scan_button)
        buttons.addWidget(self.health_button)
        buttons.addWidget(self.match_button)
        buttons.addWidget(self.send_button)
        buttons.addWidget(self.status_label, 1)
        layout.addLayout(buttons)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._table_page(self._build_match_table()), "Structure Matches")
        self.tabs.addTab(self._table_page(self._build_health_table()), "Library Health")
        self.tabs.addTab(self._table_page(self._build_family_table()), "Duplicate Families")
        layout.addWidget(self.tabs, 1)

        self.choose_root_button.clicked.connect(self.choose_library_root)
        self.choose_db_button.clicked.connect(self.choose_db_path)
        self.scan_button.clicked.connect(self.build_index)
        self.health_button.clicked.connect(self.refresh_health)
        self.match_button.clicked.connect(self.search_current_peak_list)
        self.send_button.clicked.connect(self.send_selected_to_multicomponent)

    def _build_database_group(self):
        group = QGroupBox("Library database")
        form = QFormLayout(group)
        root_row = QHBoxLayout()
        self.root_label = QLabel(str(Path.home() / "CIF_DATABASES" / "COD" / "cif"))
        self.root_label.setWordWrap(True)
        self.choose_root_button = QPushButton("Choose folder…")
        root_row.addWidget(self.root_label, 1)
        root_row.addWidget(self.choose_root_button)
        db_row = QHBoxLayout()
        self.db_label = QLabel(str(Path.home() / "CIF_DATABASES" / "afruz_cif_library.sqlite"))
        self.db_label.setWordWrap(True)
        self.choose_db_button = QPushButton("Choose DB…")
        db_row.addWidget(self.db_label, 1)
        db_row.addWidget(self.choose_db_button)
        self.max_files = NoWheelSpinBox()
        self.max_files.setRange(0, 10000000)
        self.max_files.setValue(5000)
        self.wavelength = NoWheelDoubleSpinBox()
        self.wavelength.setRange(0.1, 5.0)
        self.wavelength.setDecimals(5)
        self.wavelength.setValue(1.5406)
        self.two_theta_min = NoWheelDoubleSpinBox()
        self.two_theta_min.setRange(0, 180)
        self.two_theta_min.setDecimals(3)
        self.two_theta_min.setValue(5.0)
        self.two_theta_max = NoWheelDoubleSpinBox()
        self.two_theta_max.setRange(0, 180)
        self.two_theta_max.setDecimals(3)
        self.two_theta_max.setValue(90.0)
        form.addRow("CIF folder", root_row)
        form.addRow("SQLite DB", db_row)
        form.addRow("Max files (0 = all)", self.max_files)
        form.addRow("Wavelength (Å)", self.wavelength)
        form.addRow("2θ min", self.two_theta_min)
        form.addRow("2θ max", self.two_theta_max)
        note = QLabel("Start with 1,000–5,000 CIFs before scanning a full COD-size library.")
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        install_label_copy_menu(note)
        form.addRow("Safe scale-up", note)
        return group

    def _build_search_group(self):
        group = QGroupBox("Search filters")
        form = QFormLayout(group)
        self.required_elements = QLineEdit("Ni Fe O")
        self.excluded_elements = QLineEdit("")
        self.material_types = QLineEdit("")
        self.peak_tolerance = NoWheelDoubleSpinBox()
        self.peak_tolerance.setRange(0.01, 2.0)
        self.peak_tolerance.setDecimals(3)
        self.peak_tolerance.setValue(0.15)
        self.result_limit = NoWheelSpinBox()
        self.result_limit.setRange(1, 1000)
        self.result_limit.setValue(50)
        form.addRow("Required elements", self.required_elements)
        form.addRow("Excluded elements", self.excluded_elements)
        form.addRow("Material type contains", self.material_types)
        form.addRow("Peak tolerance (°)", self.peak_tolerance)
        form.addRow("Result limit", self.result_limit)
        note = QLabel("Use spaces/commas, for example: Ni Fe O. Material filter examples: oxide, MOF, organic.")
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        install_label_copy_menu(note)
        form.addRow("Hint", note)
        return group

    def _build_match_table(self):
        self.match_table = QTableWidget(0, 13)
        self.match_table.setHorizontalHeaderLabels(
            ["Use", "Score", "Class", "Formula", "Space group", "System", "Material", "Peak", "Cell", "Chem", "Family", "Atoms", "CIF path"]
        )
        self._finish_table(self.match_table)
        return self.match_table

    def _build_health_table(self):
        self.health_table = QTableWidget(0, 2)
        self.health_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._finish_table(self.health_table)
        return self.health_table

    def _build_family_table(self):
        self.family_table = QTableWidget(0, 6)
        self.family_table.setHorizontalHeaderLabels(["Family key", "Entries", "Formula", "Space group", "System", "Representative"])
        self._finish_table(self.family_table)
        return self.family_table

    def _table_page(self, table):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(table)
        return page

    def _finish_table(self, table):
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        install_table_copy_menu(table)

    def choose_library_root(self):
        directory = QFileDialog.getExistingDirectory(self, "Choose CIF library folder", self.root_label.text())
        if directory:
            self.root_label.setText(directory)

    def choose_db_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Choose SQLite CIF database", self.db_label.text(), "SQLite DB (*.sqlite *.db);;All files (*)")
        if path:
            self.db_label.setText(str(Path(path)))

    def _db_path(self) -> Path:
        return Path(self.db_label.text()).expanduser()

    def _root_path(self) -> Path:
        return Path(self.root_label.text()).expanduser()

    def _tokens(self, label: QLabel) -> list[str]:
        return [token.strip() for token in label.text().replace(",", " ").split() if token.strip()]

    def build_index(self):
        root = self._root_path()
        if not root.exists():
            QMessageBox.warning(self, "CIF folder missing", f"Folder does not exist:\n{root}")
            return
        max_files = int(self.max_files.value()) or None
        self.status_label.setText("Building/updating CIF index…")
        try:
            summary = build_cif_library_scalable(
                root,
                self._db_path(),
                CifLibrarySettings(
                    wavelength_angstrom=float(self.wavelength.value()),
                    two_theta_min=float(self.two_theta_min.value()),
                    two_theta_max=float(self.two_theta_max.value()),
                    maximum_files=max_files,
                ),
                IncrementalScanOptions(incremental=True, retry_failed=False, checkpoint_every=250),
            )
        except Exception as exc:
            QMessageBox.critical(self, "CIF indexing failed", str(exc))
            self.status_label.setText("Indexing failed.")
            return
        self.status_label.setText(
            f"Index updated: {summary['ok']} ok, {summary['failed']} failed, {summary['skipped_unchanged']} skipped."
        )
        self.refresh_health()

    def refresh_health(self):
        try:
            health = library_health_report(self._db_path())
            families = grouped_structure_families(self._db_path(), limit=100)
        except Exception as exc:
            QMessageBox.warning(self, "Library health unavailable", str(exc))
            return
        rows = []
        rows.append(("Total entries", health.get("total_entries")))
        rows.append(("Structure families", health.get("structure_family_count")))
        for key, value in (health.get("status_counts") or {}).items():
            rows.append((f"Status: {key}", value))
        for key, value in (health.get("material_type_counts") or {}).items():
            rows.append((f"Material: {key}", value))
        self.health_table.setRowCount(0)
        for metric, value in rows:
            row = self.health_table.rowCount()
            self.health_table.insertRow(row)
            self.health_table.setItem(row, 0, QTableWidgetItem(str(metric)))
            self.health_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.family_table.setRowCount(0)
        for data in families:
            row = self.family_table.rowCount()
            self.family_table.insertRow(row)
            for column, key in enumerate(["family_key", "n", "formula", "space_group", "crystal_system", "representative_path"]):
                self.family_table.setItem(row, column, QTableWidgetItem(str(data.get(key, ""))))
        self.status_label.setText("Library health refreshed.")

    def search_current_peak_list(self):
        dataset = self.main_window.selected_dataset()
        if dataset is None:
            QMessageBox.information(self, "No pattern", "Import/select a pattern first.")
            return
        peaks = active_peak_rows(self.main_window.peak_rows.get(dataset.uid, []))
        if not peaks:
            QMessageBox.information(self, "No Peak List", "Find or import peaks before peak-list CIF matching.")
            return
        try:
            result = intelligent_match_structure_candidates(
                self._db_path(),
                observed_peaks=peaks,
                required_elements=self._tokens(self.required_elements),
                excluded_elements=self._tokens(self.excluded_elements),
                material_types=self._tokens(self.material_types),
                peak_tolerance_deg=float(self.peak_tolerance.value()),
                limit=int(self.result_limit.value()),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Structure match failed", str(exc))
            return
        self.last_match_result = result
        self._populate_matches(result.get("candidates", []))
        self.status_label.setText(f"Found {result.get('candidate_count', 0)} structure candidate(s).")

    def _populate_matches(self, rows):
        self.match_table.setRowCount(0)
        for candidate in rows:
            row = self.match_table.rowCount()
            self.match_table.insertRow(row)
            use_item = QTableWidgetItem("")
            use_item.setFlags(use_item.flags() | Qt.ItemIsUserCheckable)
            use_item.setCheckState(Qt.Unchecked)
            use_item.setData(Qt.UserRole, candidate)
            context = candidate.get("match_context") or {}
            scores = candidate.get("component_scores") or {}
            values = [
                use_item,
                self._fmt(candidate.get("overall_score")),
                candidate.get("classification", ""),
                candidate.get("formula", ""),
                candidate.get("space_group", ""),
                candidate.get("crystal_system", ""),
                candidate.get("material_type", ""),
                self._fmt(scores.get("peak")),
                self._fmt(scores.get("cell")),
                self._fmt(scores.get("chemistry")),
                candidate.get("structure_family_key", ""),
                candidate.get("atom_count", ""),
                candidate.get("absolute_path", ""),
            ]
            for column, value in enumerate(values):
                if isinstance(value, QTableWidgetItem):
                    self.match_table.setItem(row, column, value)
                else:
                    self.match_table.setItem(row, column, QTableWidgetItem(str(value)))

    def send_selected_to_multicomponent(self):
        if not hasattr(self.main_window, "multicomponent_refiner_widget"):
            QMessageBox.warning(self, "Multiphase GUI missing", "The multiphase refiner widget is not available.")
            return
        count = 0
        for row in range(self.match_table.rowCount()):
            item = self.match_table.item(row, 0)
            if item is None or item.checkState() != Qt.Checked:
                continue
            candidate = item.data(Qt.UserRole)
            path = (candidate or {}).get("absolute_path") if isinstance(candidate, dict) else None
            if not path:
                continue
            try:
                phase = import_candidate_phase(path, label=Path(path).stem, required=False, refine_cell=False)
                self.main_window.multicomponent_refiner_widget._append_phase_row(phase)
                count += 1
            except Exception:
                continue
        if count:
            self.main_window.tabs.setCurrentWidget(self.main_window.multicomponent_refiner_widget)
            self.main_window.multicomponent_refiner_widget.status_label.setText(f"Added {count} CIF candidate(s) from library match.")
        self.status_label.setText(f"Sent {count} candidate(s) to Multiphase Refiner.")

    def _fmt(self, value):
        if value is None:
            return "—"
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return str(value)

    def refresh_for_selected_dataset(self):
        return None

    def apply_theme(self, theme_name: str):
        return None
