from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .gsasii_core import (
    GSASIIProfile,
    first_existing_profile,
    normalise_profile,
    profile_from_installation_root,
    test_profile_connection,
)


SETTINGS_ORGANIZATION = "AfruzMD"
SETTINGS_APPLICATION = "AfruzPXRDAnalyzer"
SETTINGS_GROUP = "integrations/gsasii"


class GSASIISettingsStore:
    def __init__(self, settings: QSettings | None = None):
        self.settings = settings or QSettings(
            SETTINGS_ORGANIZATION,
            SETTINGS_APPLICATION,
        )

    def load(self) -> GSASIIProfile:
        self.settings.beginGroup(SETTINGS_GROUP)
        try:
            profile = GSASIIProfile(
                installation_root=str(
                    self.settings.value("installation_root", "") or ""
                ),
                python_executable=str(
                    self.settings.value("python_executable", "") or ""
                ),
                source_parent=str(
                    self.settings.value("source_parent", "") or ""
                ),
                default_instrument_file=str(
                    self.settings.value("default_instrument_file", "") or ""
                ),
                timeout_seconds=int(
                    self.settings.value("timeout_seconds", 600) or 600
                ),
            )
        finally:
            self.settings.endGroup()
        return normalise_profile(profile)

    def save(self, profile: GSASIIProfile | dict) -> GSASIIProfile:
        profile = normalise_profile(profile)
        self.settings.beginGroup(SETTINGS_GROUP)
        try:
            for key, value in profile.to_dict().items():
                self.settings.setValue(key, value)
            self.settings.setValue("configured", bool(profile.python_executable))
        finally:
            self.settings.endGroup()
        self.settings.sync()
        return profile

    def clear(self) -> None:
        self.settings.beginGroup(SETTINGS_GROUP)
        try:
            self.settings.remove("")
        finally:
            self.settings.endGroup()
        self.settings.sync()

    def import_legacy(
        self,
        python_executable: str = "",
        source_parent: str = "",
    ) -> GSASIIProfile:
        current = self.load()
        if current.python_executable and current.source_parent:
            return current
        profile = normalise_profile(
            {
                "python_executable": python_executable,
                "source_parent": source_parent,
                "timeout_seconds": current.timeout_seconds,
                "default_instrument_file": current.default_instrument_file,
            }
        )
        if profile.python_executable or profile.source_parent:
            return self.save(profile)
        return current


class GSASIISettingsDialog(QDialog):
    settings_saved = Signal(object)

    def __init__(
        self,
        store: GSASIISettingsStore,
        parent=None,
    ):
        super().__init__(parent)
        self.store = store
        self.setWindowTitle("GSAS-II Integration Settings")
        self.resize(760, 570)
        self.setMinimumSize(640, 460)
        self._build_ui()
        self.set_profile(self.store.load())

    def _build_ui(self):
        layout = QVBoxLayout(self)

        heading = QLabel("Application-wide GSAS-II connection")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        description = QLabel(
            "Configure GSAS-II once. Phase Revolution, Validated QPA, and future "
            "GSAS-II tools will use this profile automatically. For the standard "
            "Windows installation, select the gsas2main directory."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        self.root_edit = QLineEdit()
        self.python_edit = QLineEdit()
        self.parent_edit = QLineEdit()
        self.instrument_edit = QLineEdit()
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(30, 7200)
        self.timeout_spin.setValue(600)

        self.root_browse = QPushButton("Browse…")
        self.root_apply = QPushButton("Derive paths")
        root_row = self._path_row(
            self.root_edit,
            self.root_browse,
            self.root_apply,
        )

        self.python_browse = QPushButton("Browse…")
        python_row = self._path_row(self.python_edit, self.python_browse)

        self.parent_browse = QPushButton("Browse…")
        parent_row = self._path_row(self.parent_edit, self.parent_browse)

        self.instrument_browse = QPushButton("Browse…")
        instrument_row = self._path_row(
            self.instrument_edit,
            self.instrument_browse,
        )

        form.addRow("GSAS-II installation root", root_row)
        form.addRow("GSAS-II Python executable", python_row)
        form.addRow("GSAS-II source parent", parent_row)
        form.addRow("Default instrument file", instrument_row)
        form.addRow("Backend timeout (s)", self.timeout_spin)
        layout.addLayout(form)

        suggested_root = Path.home() / "gsas2main"
        example = QLabel(
            f"Typical installation for this account: root = {suggested_root}; "
            "Python = root\\python.exe; source parent = root\\GSAS-II."
        )
        example.setWordWrap(True)
        example.setObjectName("mutedLabel")
        layout.addWidget(example)

        action_row = QHBoxLayout()
        self.auto_detect_button = QPushButton("Auto-detect")
        self.test_button = QPushButton("Test connection")
        self.clear_button = QPushButton("Clear")
        action_row.addWidget(self.auto_detect_button)
        action_row.addWidget(self.test_button)
        action_row.addWidget(self.clear_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.status_label = QLabel("Not tested.")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(150)
        layout.addWidget(self.details, 1)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        layout.addWidget(self.buttons)

        self.root_browse.clicked.connect(self._browse_root)
        self.root_apply.clicked.connect(self._derive_paths)
        self.python_browse.clicked.connect(self._browse_python)
        self.parent_browse.clicked.connect(self._browse_parent)
        self.instrument_browse.clicked.connect(self._browse_instrument)
        self.auto_detect_button.clicked.connect(self._auto_detect)
        self.test_button.clicked.connect(self._test_connection)
        self.clear_button.clicked.connect(self._clear_fields)
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)

    @staticmethod
    def _path_row(edit, *buttons):
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        for button in buttons:
            row.addWidget(button)
        return container

    def profile(self) -> GSASIIProfile:
        return normalise_profile(
            {
                "installation_root": self.root_edit.text(),
                "python_executable": self.python_edit.text(),
                "source_parent": self.parent_edit.text(),
                "default_instrument_file": self.instrument_edit.text(),
                "timeout_seconds": self.timeout_spin.value(),
            }
        )

    def set_profile(self, profile: GSASIIProfile | dict):
        profile = normalise_profile(profile)
        self.root_edit.setText(profile.installation_root)
        self.python_edit.setText(profile.python_executable)
        self.parent_edit.setText(profile.source_parent)
        self.instrument_edit.setText(profile.default_instrument_file)
        self.timeout_spin.setValue(profile.timeout_seconds)

    def _browse_root(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select gsas2main installation root",
            self.root_edit.text(),
        )
        if directory:
            self.root_edit.setText(directory)
            self._derive_paths()

    def _derive_paths(self):
        root = self.root_edit.text().strip()
        if not root:
            return
        inferred = profile_from_installation_root(
            root,
            default_instrument_file=self.instrument_edit.text(),
            timeout_seconds=self.timeout_spin.value(),
        )
        self.set_profile(inferred)
        self.status_label.setText("Paths derived. Test the connection before saving.")

    def _browse_python(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select GSAS-II Python executable",
            self.python_edit.text(),
            "Python executable (python.exe python);;All files (*)",
        )
        if filename:
            self.python_edit.setText(filename)

    def _browse_parent(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select directory containing the GSASII package",
            self.parent_edit.text(),
        )
        if directory:
            self.parent_edit.setText(directory)

    def _browse_instrument(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select default GSAS-II instrument file",
            self.instrument_edit.text(),
            "GSAS-II instrument (*.instprm *.prm);;All files (*)",
        )
        if filename:
            self.instrument_edit.setText(filename)

    def _auto_detect(self):
        profile = first_existing_profile()
        if profile is None:
            self.status_label.setText(
                "No standard GSAS-II installation was detected automatically."
            )
            return
        profile = GSASIIProfile(
            **{
                **profile.to_dict(),
                "default_instrument_file": self.instrument_edit.text(),
                "timeout_seconds": self.timeout_spin.value(),
            }
        )
        self.set_profile(profile)
        self.status_label.setText(
            f"Detected candidate installation: {profile.installation_root}"
        )

    def _test_connection(self):
        self.status_label.setText("Testing GSAS-II imports…")
        result = test_profile_connection(self.profile())
        self.details.setPlainText(json.dumps(result, indent=2, ensure_ascii=False))
        if result.get("available"):
            self.status_label.setText(
                "Connected: GSASIIscriptable and GSASIIindex are available."
            )
        else:
            self.status_label.setText(
                "Connection failed. Review the detailed import error below."
            )

    def _clear_fields(self):
        self.set_profile(GSASIIProfile())
        self.status_label.setText("Configuration cleared in the dialog.")
        self.details.clear()

    def _save(self):
        result = test_profile_connection(self.profile())
        self.details.setPlainText(json.dumps(result, indent=2, ensure_ascii=False))
        if not result.get("available"):
            answer = QMessageBox.question(
                self,
                "Save unverified configuration?",
                "The GSAS-II connection test failed. Save these paths anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        profile = self.store.save(self.profile())
        self.settings_saved.emit(profile)
        self.accept()
