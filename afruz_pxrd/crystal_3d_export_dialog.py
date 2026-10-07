"""Responsive, non-modal options for an editable Blender-compatible export."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFormLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from .context_copy import install_label_copy_menu
from .crystal_3d_export import ModelExportOptions, export_editable_crystal
from .widgets import NoWheelComboBox


class _ExportSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(object, str)


class _ExportJob(QRunnable):
    def __init__(self, scene, path, options, camera):
        super().__init__()
        self.signals = _ExportSignals()
        self.scene, self.path, self.options, self.camera = scene, path, options, camera

    def run(self):
        result, error = None, ""
        try:
            result = export_editable_crystal(self.scene, self.path, self.options, camera=self.camera, progress=self.signals.progress.emit)
        except Exception as exc:
            error = str(exc)
        try:
            self.signals.finished.emit(result, error)
        except RuntimeError:
            pass


class EditableCrystalExportDialog(QDialog):
    def __init__(self, scene, parent=None, *, background_style="Journal white", camera=None):
        super().__init__(parent)
        self.scene = deepcopy(scene)
        self.camera = deepcopy(camera)
        self.background_style = background_style
        self._job = None
        self.last_result = None
        self.setWindowTitle("Editable 3D crystal — Blender GLB export")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(640, 590)
        layout = QVBoxLayout(self)
        note = QLabel(f"Frozen scene: {scene.model.get('data_name', 'Crystal')} · {len(scene.positions):,} atoms · {len(scene.bonds):,} contacts. "
                      "Export true editable geometry, not a screenshot. 1 Blender unit represents 1 Å at graphical-abstract scale.")
        note.setWordWrap(True)
        install_label_copy_menu(note)
        layout.addWidget(note)
        self.controls = QWidget(self)
        form = QFormLayout(self.controls)
        self.quality = NoWheelComboBox()
        for label, value in (("Compact — 80 faces per sphere", 1), ("Balanced — 320 faces per sphere", 2), ("High — 1,280 faces per sphere", 3)):
            self.quality.addItem(label, value)
        self.quality.setCurrentIndex(2)
        form.addRow("Sphere mesh quality", self.quality)
        self.parts = {}
        available = {"atoms": len(scene.positions), "bonds": len(scene.bonds), "polyhedra": len(scene.triangles),
                     "pores": len(scene.pore_centers), "hydrogen_bonds": len(scene.hydrogen_bonds), "cell_edges": len(scene.edges)}
        for key, label in (("atoms", "Atoms — separate named objects"), ("bonds", "Contacts — separate bond cylinders"),
                           ("polyhedra", "Polyhedral surfaces — editable meshes by element"), ("pores", "Orange pore envelopes — separate spheres"),
                           ("hydrogen_bonds", "Hydrogen-bond contacts — dashed cylinders"), ("cell_edges", "Unit-cell guides")):
            checkbox = QCheckBox(f"{label} ({available[key]:,})")
            checkbox.setChecked(bool(available[key]))
            checkbox.setEnabled(bool(available[key]))
            self.parts[key] = checkbox
            form.addRow(checkbox)
        self.override = QCheckBox("Expert override: export a structure requiring review")
        self.flagged = scene.model.get("provenance", {}).get("publication_ready") is False
        self.override.setVisible(self.flagged)
        form.addRow(self.override)
        layout.addWidget(self.controls)
        scientific = QLabel("The pore sphere is a grid-derived maximal-empty-sphere estimate, not an atom or an adsorption-accessible surface. "
                            "Atom sizes/contact cylinders and coordination hulls are display conventions. Coordinates, occupancies and source provenance are retained. "
                            "Shared meshes keep large MOFs compact; make single-user in Blender when editing just one atom's mesh/material.")
        scientific.setWordWrap(True)
        install_label_copy_menu(scientific)
        layout.addWidget(scientific)
        if self.flagged:
            warning = QLabel("STRUCTURE REVIEW REQUIRED. This export is not proof of a validated crystal structure. Inspect the CIF and its scientific warnings.")
            warning.setWordWrap(True)
            install_label_copy_menu(warning)
            layout.addWidget(warning)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.status = QLabel("Exports one .glb model, a reusable .crystal.json source and a short Blender guide.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.export_button = QPushButton("Export editable 3D (.glb)…")
        self.export_button.setObjectName("primaryButton")
        self.export_button.setEnabled(not self.flagged)
        self.override.toggled.connect(lambda checked: self.export_button.setEnabled(checked and self._job is None))
        self.export_button.clicked.connect(self.export_model)
        layout.addWidget(self.export_button)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        layout.addWidget(close)

    def export_model(self):
        if self._job is not None:
            return
        options = ModelExportOptions(sphere_subdivisions=int(self.quality.currentData()),
            **{key: control.isChecked() for key, control in self.parts.items()},
            background_style=self.background_style, allow_flagged_structure=self.override.isChecked())
        filename, _ = QFileDialog.getSaveFileName(self, "Export editable Blender model", "crystal-model.glb", "Binary glTF 2.0 (*.glb)")
        if not filename:
            return
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(".glb")
        self.export_button.setEnabled(False)
        self.controls.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Preparing the frozen editable scene…")
        self._job = _ExportJob(self.scene, path, options, self.camera)
        self._job.signals.progress.connect(self._progress)
        self._job.signals.finished.connect(self._ready)
        QThreadPool.globalInstance().start(self._job)

    @Slot(int, str)
    def _progress(self, value, message):
        self.progress.setValue(value)
        self.status.setText(message)

    @Slot(object, str)
    def _ready(self, result, error):
        self._job = None
        self.controls.setEnabled(True)
        self.export_button.setEnabled(not self.flagged or self.override.isChecked())
        if error:
            self.status.setText(error)
            QMessageBox.warning(self, "Editable 3D export failed", error)
            return
        self.last_result = result
        self.progress.setValue(100)
        self.status.setText(f"Saved {result['model_path']}\nBlender: File → Import → glTF 2.0. "
                            f"Source: {result['scene_path'].name} · Guide: {result['guide_path'].name}. "
                            + (f"Preserved {len(result['backups'])} previous file(s) as backups." if result['backups'] else ""))
