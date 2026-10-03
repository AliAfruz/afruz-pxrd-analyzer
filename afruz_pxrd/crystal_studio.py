"""Crystal Studio: a non-modal figure editor for frozen Rietveld structures."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSize, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from .crystallography import load_cif
from .crystal_chemistry import (
    analyze_crystal_chemistry, export_crystal_chemistry_bundle,
    export_crystal_chemistry_report, format_crystal_chemistry_report,
)
from .crystal_scene import (
    METAL_ELEMENTS, MOF_PORE_STYLE, MOF_STYLE, SCIENTIFIC_STYLE,
    RenderSettings, STYLES, build_scene, load_scene_document,
    model_from_structure, models_from_result,
)
from .crystal_render import export_crystal, render_crystal
from .crystal_intelligence import analyze_crystal_model, format_smart_crystal_report
from .widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox


class _JobSignals(QObject):
    scene_ready = Signal(int, object)
    finished = Signal(int, object, str)


class _RenderJob(QRunnable):
    def __init__(self, revision, model, settings, output=None, skip_cpu_preview=False):
        super().__init__()
        self.signals = _JobSignals()
        self.revision, self.model, self.settings, self.output = revision, model, settings, output
        self.skip_cpu_preview = bool(skip_cpu_preview)

    @Slot()
    def run(self):
        value = None
        error = ""
        try:
            scene = build_scene(self.model, self.settings)
            self.signals.scene_ready.emit(self.revision, scene)
            if self.output:
                image_path, scene_path = export_crystal(scene, **self.output)
                qa_json, qa_text = export_crystal_chemistry_bundle(self.model, image_path)
                value = (image_path, scene_path, qa_json, qa_text)
            elif self.skip_cpu_preview:
                value = (
                    None, scene.warnings, len(scene.positions),
                    len(scene.bonds), len(scene.hydrogen_bonds),
                    len(scene.pore_centers),
                )
            else:
                picture = render_crystal(scene, 1000, 750)
                data = picture.tobytes()
                # copy() detaches from the Python bytes before they leave scope.
                qimage = QImage(data, picture.width, picture.height, picture.width * 4, QImage.Format_RGBA8888).copy()
                value = (
                    qimage, scene.warnings, len(scene.positions),
                    len(scene.bonds), len(scene.hydrogen_bonds),
                    len(scene.pore_centers),
                )
        except Exception as exc:
            error = str(exc)
        try:
            self.signals.finished.emit(self.revision, value, error)
        except RuntimeError:
            # The window may be closed while an owned background render finishes.
            pass


class _QaSignals(QObject):
    finished = Signal(object, str)


class _QaJob(QRunnable):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.signals = _QaSignals()

    @Slot()
    def run(self):
        report = None
        error = ""
        try:
            report = analyze_crystal_chemistry(self.model)
        except Exception as exc:
            error = str(exc)
        try:
            self.signals.finished.emit(report, error)
        except RuntimeError:
            pass


class _SmartSignals(QObject):
    finished = Signal(int, object, str)


class _SmartViewJob(QRunnable):
    """Classify periodic topology and plan a view away from the GUI thread."""

    def __init__(self, model_index, model):
        super().__init__()
        self.model_index = int(model_index)
        self.model = model
        self.signals = _SmartSignals()

    @Slot()
    def run(self):
        report = None
        error = ""
        try:
            report = analyze_crystal_model(self.model)
        except Exception as exc:
            error = str(exc)
        try:
            self.signals.finished.emit(self.model_index, report, error)
        except RuntimeError:
            pass


class _StructureLoadSignals(QObject):
    finished = Signal(object, str)


class _StructureLoadJob(QRunnable):
    """Parse and validate large CIFs without blocking the Qt event loop."""

    def __init__(self, path):
        super().__init__()
        self.path = str(path)
        self.signals = _StructureLoadSignals()

    @Slot()
    def run(self):
        value = None
        error = ""
        try:
            if Path(self.path).suffix.lower() == ".json":
                model, settings = load_scene_document(self.path)
            else:
                model = model_from_structure(load_cif(self.path))
                settings = None
            value = (model, settings)
        except Exception as exc:
            error = str(exc)
        try:
            self.signals.finished.emit(value, error)
        except RuntimeError:
            pass


class CrystalChemistryReportDialog(QDialog):
    """Readable, exportable view of one frozen Crystal Chemistry QA report."""

    def __init__(self, report, parent=None):
        super().__init__(parent)
        self.report = report
        self.setWindowTitle("Crystal Chemistry QA — Afruz PXRD")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(920, 720)
        layout = QVBoxLayout(self)
        summary = report["summary"]
        heading = QLabel(
            f"{summary['structure_name']}  ·  {summary['status'].upper()}  ·  "
            f"{summary['occupied_cell_composition']}"
        )
        heading.setWordWrap(True)
        heading.setStyleSheet("font-size:16px; font-weight:600; padding:4px;")
        layout.addWidget(heading)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        text.setPlainText(format_crystal_chemistry_report(report))
        layout.addWidget(text, 1)
        row = QHBoxLayout()
        export_button = QPushButton("Export QA report…")
        export_button.clicked.connect(self.export_report)
        row.addWidget(export_button)
        row.addStretch()
        close_buttons = QDialogButtonBox(QDialogButtonBox.Close)
        close_buttons.rejected.connect(self.close)
        row.addWidget(close_buttons)
        layout.addLayout(row)

    def export_report(self):
        path, selected = QFileDialog.getSaveFileName(
            self, "Export Crystal Chemistry QA", "crystal-chemistry-qa.json",
            "Full JSON report (*.json);;Readable text report (*.txt);;Markdown report (*.md)",
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".txt" if selected.startswith("Readable") else ".md" if selected.startswith("Markdown") else ".json"
        try:
            saved = export_crystal_chemistry_report(self.report, path)
            QMessageBox.information(self, "QA report saved", f"Saved {saved}")
        except Exception as exc:
            QMessageBox.warning(self, "QA export failed", str(exc))


class _CrystalPreview(QLabel):
    rotated = Signal(float, float)
    zoomed = Signal(float)

    def __init__(self):
        super().__init__("Run a Rietveld refinement or open a CIF to preview a structure.")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setWordWrap(True)
        self.setStyleSheet("background:#0b1523; color:#a8bfd3; border-radius:12px; padding:8px;")
        self.setCursor(Qt.OpenHandCursor)
        self._image = None
        self._last = None

    def set_image(self, image):
        self._image = image
        self._rescale()

    def _rescale(self):
        if self._image is not None:
            self.setPixmap(QPixmap.fromImage(self._image).scaled(self.size() - QSize(16, 16), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._last is not None:
            delta = event.position() - self._last
            self._last = event.position()
            self.rotated.emit(delta.x() * .45, delta.y() * .35)

    def mouseReleaseEvent(self, event):
        self._last = None
        self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, event):
        self.zoomed.emit(.08 if event.angleDelta().y() > 0 else -.08)
        event.accept()


class CrystalStudioDialog(QDialog):
    def __init__(self, result=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crystal Studio — Afruz PXRD")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(1260, 860)
        self.setMinimumSize(780, 560)
        self.models = []
        self._revision = 0
        self._preview_job = None
        self._export_job = None
        self._qa_job = None
        self._smart_job = None
        self._smart_report = None
        self._load_job = None
        self._qa_dialogs = []
        self._gpu_dialogs = []
        self._last_scene = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(130)
        self._timer.timeout.connect(self._render)
        self._build_ui()
        notice = ""
        try:
            self.models = models_from_result(result)
        except ValueError as exc:
            notice = str(exc)
        self._populate_models()
        if self.models:
            # Rietveld snapshots enter through the constructor rather than the
            # Open CIF action, so protect large framework results here too.
            self._apply_safe_large_structure_preset(
                self.models[self.phase_combo.currentIndex()]
            )
        if notice:
            self.provenance.setText(notice)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        heading = QLabel("Crystal Studio")
        heading.setStyleSheet("font-size:25px; font-weight:600;")
        layout.addWidget(heading)
        subtitle = QLabel("Shape the view. Keep the structure.   •   Drag to rotate · scroll to zoom")
        subtitle.setObjectName("mutedLabel")
        layout.addWidget(subtitle)
        self.provenance = QLabel()
        self.provenance.setWordWrap(True)
        self.provenance.setObjectName("mutedLabel")
        layout.addWidget(self.provenance)
        self.validation_banner = QLabel()
        self.validation_banner.setWordWrap(True)
        self.validation_banner.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.validation_banner.hide()
        layout.addWidget(self.validation_banner)
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)
        self.preview = _CrystalPreview()
        split.addWidget(self.preview)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(400)
        controls = QWidget()
        column = QVBoxLayout(controls)
        scroll.setWidget(controls)
        split.addWidget(scroll)
        split.setStretchFactor(0, 1)
        split.setSizes([850, 330])

        structure_form = self._group(column, "Structure and style")
        self.phase_combo = NoWheelComboBox()
        self.phase_combo.currentIndexChanged.connect(self._phase_changed)
        structure_form.addRow("Phase", self.phase_combo)
        self.open_button = QPushButton("Open CIF / saved scene…")
        self.open_button.clicked.connect(self.open_structure)
        structure_form.addRow(self.open_button)
        self.title_edit = QLineEdit()
        structure_form.addRow("Figure title", self.title_edit)
        self.style_combo = NoWheelComboBox()
        self.style_combo.addItems(STYLES)
        self.style_combo.setToolTip(
            "MOF isolated pore/cage builds a periodic neighbourhood around the "
            "detected void, then keeps the surrounding connected framework with "
            "coordination polyhedra on a white publication background."
        )
        structure_form.addRow("Figure template", self.style_combo)
        self.repeats = []
        row = QWidget()
        repeats_layout = QHBoxLayout(row)
        repeats_layout.setContentsMargins(0, 0, 0, 0)
        for name in "abc":
            box = NoWheelSpinBox()
            box.setRange(1, 4)
            box.setValue(2)
            repeats_layout.addWidget(QLabel(name))
            repeats_layout.addWidget(box)
            self.repeats.append(box)
        structure_form.addRow("Repeat cells", row)

        smart = self._group(column, "Smart view — explainable and opt-in")
        smart_note = QLabel(
            "Classifies composition and periodic topology, then recommends a camera, "
            "level of detail and scientific template. It never changes the CIF, atomic "
            "coordinates or refinement result."
        )
        smart_note.setWordWrap(True)
        smart_note.setObjectName("mutedLabel")
        smart.addRow(smart_note)
        self.smart_button = QPushButton("Analyze + apply smart view")
        self.smart_button.clicked.connect(self.run_smart_analysis)
        smart.addRow(self.smart_button)
        self.smart_summary = QPlainTextEdit()
        self.smart_summary.setReadOnly(True)
        self.smart_summary.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.smart_summary.setMinimumHeight(150)
        self.smart_summary.setPlainText(
            "Open a structure, then run the deterministic view planner. Manual controls remain authoritative."
        )
        smart.addRow(self.smart_summary)
        self.copy_smart_button = QPushButton("Copy smart report")
        self.copy_smart_button.setEnabled(False)
        self.copy_smart_button.clicked.connect(self.copy_smart_report)
        smart.addRow(self.copy_smart_button)

        geometry = self._group(column, "Atoms and coordination")
        self.atom_scale = self._number(.35, 2, 1, .05)
        geometry.addRow("Sphere size", self.atom_scale)
        self.atom_opacity = self._number(5, 100, 100, 5, 0)
        self.atom_opacity.setSuffix(" %")
        self.atom_opacity.setToolTip(
            "Controls atom transparency in the CPU preview/export and GPU 3D view."
        )
        geometry.addRow("Atom opacity", self.atom_opacity)
        self.bond_radius = self._number(.01, .2, .065, .005, 3)
        geometry.addRow("Bond thickness (Å)", self.bond_radius)
        self.cutoff = self._number(0, 12, 0, .1)
        self.cutoff.setSpecialValueText("Auto nearest shell")
        self.cutoff.setToolTip("Distance-only display contacts. Auto: mutual nearest shells +16%, at most 6 Å. This does not determine chemical bonding.")
        geometry.addRow("Contact cutoff (Å)", self.cutoff)
        self.bonds = self._check(geometry, "Show bonds", True)
        self.unlike = self._check(geometry, "Unlike elements only", True)
        self.unlike.setToolTip(
            "Normally suppresses same-element display contacts. MOF presets turn "
            "this off so covalent C–C linker bonds remain visible; inferred "
            "metal–metal contacts stay controlled separately."
        )
        self.polyhedra = self._check(geometry, "Coordination polyhedra", True)
        self.center_combo = NoWheelComboBox()
        self.center_combo.addItem("Auto")
        geometry.addRow("Center element", self.center_combo)
        self.opacity = self._number(0, .5, .13, .01)
        geometry.addRow("Polyhedron opacity", self.opacity)
        self.polyhedron_edges = self._check(geometry, "Polyhedron outlines", False)
        self.hydrogen_bonds = self._check(geometry, "Dashed hydrogen bonds", False)
        self.hydrogen_bonds.setToolTip("Shows H···N/O/F/S contacts from geometric distance and angle criteria; these are display assignments.")
        self.complete_boundaries = self._check(geometry, "Complete periodic boundary coordination", True)
        self.complete_boundaries.setToolTip("Adds only nearby periodic images outside the cell so boundary atoms keep complete bonds and polyhedra.")
        self.metal_metal_bonds = self._check(geometry, "Show inferred metal–metal contacts", False)
        self.metal_metal_bonds.setToolTip("Off by default for the scientific style. Enable only when the metal–metal connection is chemically intended.")
        self.cell_edges = self._check(geometry, "Unit-cell edges", True)
        self.cell_grid = self._check(geometry, "Internal repeat-cell grid", True)
        self.labels = self._check(geometry, "Element labels", False)
        self.hide_hydrogen = self._check(geometry, "Hide hydrogen for framework view", False)
        self.hide_hydrogen.setToolTip(
            "Useful for large MOFs and pore/cage views. Atomic coordinates remain in the saved model; only display is affected."
        )
        self.axes = self._check(geometry, "Crystallographic axes", True)
        self.caption = self._check(geometry, "Caption & element key", True)

        pore = self._group(column, "Pore-volume envelope")
        self.pore_volumes = self._check(
            pore, "Show yellow pore envelopes", False
        )
        self.pore_volumes.setToolTip(
            "Shows periodic grid-derived maximal empty spheres. This is an "
            "illustrative void envelope, not a gas-accessible surface result."
        )
        self.pore_probe = self._number(0, 3, 0, .1, 2)
        self.pore_probe.setSpecialValueText("Geometric void")
        pore.addRow("Probe subtraction (Å)", self.pore_probe)
        self.pore_opacity = self._number(.05, 1.0, .88, .05, 2)
        self.pore_opacity.setToolTip(
            "Higher opacity hides rear framework fragments and gives a clean "
            "space-filling pore figure; lower it to inspect atoms behind the envelope."
        )
        pore.addRow("Yellow volume opacity", self.pore_opacity)
        self.pore_count = NoWheelSpinBox()
        self.pore_count.setRange(1, 24)
        self.pore_count.setValue(2)
        pore.addRow("Maximum cavities / cell", self.pore_count)
        self.isolate_pore = self._check(
            pore, "Isolate one pore / cage", False
        )
        self.isolate_pore.setToolTip(
            "Keeps one calculated pore envelope and the surrounding framework shell. "
            "This is a visualization crop, not an extracted molecular species."
        )
        self.smart_pore = self._check(
            pore, "Smart framework-only cage", True
        )
        self.smart_pore.setEnabled(False)
        self.smart_pore.setToolTip(
            "Recommended: detect the void from bonded framework fragments, exclude "
            "disconnected guests/solvent/disorder, and retain metal-connected cage walls."
        )
        self.pore_index = NoWheelSpinBox()
        self.pore_index.setRange(1, 24)
        self.pore_index.setValue(1)
        self.pore_index.setEnabled(False)
        pore.addRow("Pore number", self.pore_index)
        self.pore_shell = self._number(.5, 12, 8, .5, 1)
        self.pore_shell.setSuffix(" Å")
        self.pore_shell.setEnabled(False)
        self.pore_shell.setToolTip(
            "Radial framework thickness outside the yellow maximal-empty-sphere estimate."
        )
        pore.addRow("Framework shell", self.pore_shell)
        pore_note = QLabel(
            "Radii are estimated from periodic clearance to approximate atomic "
            "van der Waals surfaces. Exported scenes retain the values and warning."
        )
        pore_note.setWordWrap(True)
        pore_note.setObjectName("mutedLabel")
        pore.addRow(pore_note)

        camera = self._group(column, "Camera")
        self.azimuth = self._number(-360, 360, 32, 5, 1)
        self.elevation = self._number(-89, 89, 19, 5, 1)
        self.zoom = self._number(.35, 2.5, 1, .05)
        camera.addRow("Azimuth (°)", self.azimuth)
        camera.addRow("Elevation (°)", self.elevation)
        camera.addRow("Zoom", self.zoom)
        reset_button = QPushButton("Reset camera")
        reset_button.clicked.connect(self.reset_camera)
        camera.addRow(reset_button)
        self.gpu_first = QCheckBox("GPU-first preview for large structures")
        self.gpu_first.setChecked(False)
        self.gpu_first.setToolTip(
            "Skips the slower CPU raster preview after geometry is built. Open the "
            "interactive OpenGL window for a shaded 3D view and GPU PNG/TIFF export."
        )
        camera.addRow(self.gpu_first)
        self.gpu_view_button = QPushButton("Open GPU 3D + fast export…")
        self.gpu_view_button.setEnabled(False)
        self.gpu_view_button.setToolTip(
            "Opens a hardware-accelerated OpenGL view as soon as scene geometry is ready. "
            "The window can export its current camera through a tiled GPU framebuffer."
        )
        self.gpu_view_button.clicked.connect(self.open_gpu_view)
        camera.addRow(self.gpu_view_button)

        qa = self._group(column, "Crystal Chemistry QA")
        self.qa_summary = QLabel("Analyze coordination, bond valence, formal charge and translational symmetry.")
        self.qa_summary.setWordWrap(True)
        self.qa_summary.setObjectName("mutedLabel")
        qa.addRow(self.qa_summary)
        self.qa_button = QPushButton("Run Crystal Chemistry QA…")
        self.qa_button.clicked.connect(self.run_chemistry_qa)
        qa.addRow(self.qa_button)

        export = self._group(column, "Figure export")
        self.resolution = NoWheelComboBox()
        for width in (1600, 2400, 3200, 4000):
            self.resolution.addItem(f"{width:,} × {width * 3 // 4:,} px", width)
        self.resolution.setCurrentIndex(2)
        export.addRow("Resolution", self.resolution)
        self.dpi = NoWheelSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setValue(600)
        export.addRow("DPI", self.dpi)
        self.transparent = self._check(export, "Transparent background", False)
        self.allow_flagged_export = self._check(export, "Expert override: export flagged structure", False)
        self.allow_flagged_export.setToolTip(
            "Allows an image of a weak or review-required model. The exported figure and scene remain visibly flagged."
        )
        self.export_button = QPushButton("Export PNG / TIFF…")
        self.export_button.setObjectName("primaryButton")
        self.export_button.clicked.connect(self.export_figure)
        export.addRow(self.export_button)
        hint = QLabel("Includes a reusable .crystal.json scene with the structure, camera and display settings.")
        hint.setWordWrap(True)
        hint.setObjectName("mutedLabel")
        export.addRow(hint)
        column.addStretch()
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.style_combo.currentIndexChanged.connect(self._style_changed)
        self.center_combo.currentIndexChanged.connect(self._schedule)
        for control in [*self.repeats, self.atom_scale, self.atom_opacity,
                        self.bond_radius, self.cutoff, self.opacity,
                        self.pore_probe, self.pore_opacity, self.pore_count,
                        self.azimuth, self.elevation, self.zoom]:
            control.valueChanged.connect(self._schedule)
        for control in [self.bonds, self.unlike, self.polyhedra, self.polyhedron_edges, self.hydrogen_bonds, self.hide_hydrogen,
                        self.complete_boundaries, self.metal_metal_bonds,
                        self.cell_edges, self.cell_grid, self.labels, self.axes, self.caption,
                        self.pore_volumes, self.transparent]:
            control.toggled.connect(self._schedule)
        self.gpu_first.toggled.connect(self._schedule)
        self.isolate_pore.toggled.connect(self._isolate_pore_changed)
        self.smart_pore.toggled.connect(self._schedule)
        self.pore_index.valueChanged.connect(self._pore_index_changed)
        self.pore_shell.valueChanged.connect(self._schedule)
        self.allow_flagged_export.toggled.connect(self._update_export_availability)
        self.title_edit.textChanged.connect(self._schedule)
        self.preview.rotated.connect(self._rotate)
        self.preview.zoomed.connect(lambda change: self.zoom.setValue(self.zoom.value() + change))

    @staticmethod
    def _group(layout, title):
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.addWidget(group)
        return form

    @staticmethod
    def _number(low, high, value, step, decimals=2):
        spin = NoWheelDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    @staticmethod
    def _check(form, title, checked):
        check = QCheckBox(title)
        check.setChecked(checked)
        form.addRow(check)
        return check

    def _populate_models(self):
        self.phase_combo.blockSignals(True)
        self.phase_combo.clear()
        self.phase_combo.addItems([model.get("data_name", "Phase") for model in self.models])
        self.phase_combo.blockSignals(False)
        self._update_export_availability()
        self.qa_button.setEnabled(bool(self.models) and self._qa_job is None)
        self.smart_button.setEnabled(bool(self.models) and self._smart_job is None)
        if self.models:
            self._phase_changed()
        else:
            self.provenance.setText("No saved refinement selected. Open a CIF for a reference preview, or run a Rietveld refinement first.")

    def _phase_changed(self, *_):
        index = self.phase_combo.currentIndex()
        if not 0 <= index < len(self.models):
            return
        model = self.models[index]
        self._smart_report = None
        self.smart_summary.setPlainText(
            "Ready for an explainable smart-view analysis. No structure data will be modified."
        )
        self.copy_smart_button.setEnabled(False)
        self.smart_button.setEnabled(self._smart_job is None)
        self.title_edit.setText(model.get("data_name", "Crystal structure"))
        provenance = model.get("provenance", {})
        status = str(provenance.get("validation_status") or "Validation unavailable")
        score = provenance.get("validation_score")
        score_text = "" if score is None else f" · plausibility {float(score):.1f}/100"
        self.provenance.setText(f"{provenance.get('note', '')}  {status}{score_text}".strip())
        if provenance.get("publication_ready") is False:
            counts = provenance.get("structure_validation_counts") or {}
            overlap_text = f" · {counts.get('vdw_overlaps')} van der Waals overlaps" if counts.get("vdw_overlaps") else ""
            self.validation_banner.setText(
                f"STRUCTURE REVIEW REQUIRED — {status}{score_text}{overlap_text}. "
                "Preview is allowed; publication export stays locked until the expert override is checked."
            )
            self.validation_banner.setStyleSheet(
                "background:#fff2dd; color:#8a3400; border:1px solid #c56200; border-radius:7px; padding:8px; font-weight:600;"
            )
            self.validation_banner.show()
        else:
            self.validation_banner.hide()
        self.center_combo.clear()
        elements = sorted({atom["element"] for atom in model["atoms"]})
        centers = ["Auto"]
        if any(element in METAL_ELEMENTS for element in elements):
            centers.append("All metals")
        self.center_combo.addItems([*centers, *elements])
        self.qa_summary.setText(
            "Ready to analyze the frozen coordinates and refined cell. The QA report does not alter the structure."
        )
        self.qa_button.setEnabled(self._qa_job is None)
        self._update_export_availability()
        self._schedule()

    def run_smart_analysis(self):
        if not self.models or self._smart_job is not None:
            return
        model_index = self.phase_combo.currentIndex()
        self.smart_button.setEnabled(False)
        self.smart_button.setText("Analyzing periodic topology…")
        self.smart_summary.setPlainText(
            "Building an element-aware periodic contact graph in the background…"
        )
        self.status.setText(
            "Smart Crystal Studio is analyzing composition, topology, camera and display complexity…"
        )
        job = _SmartViewJob(model_index, self._current_model())
        self._smart_job = job
        job.signals.finished.connect(self._smart_ready)
        QThreadPool.globalInstance().start(job)

    @Slot(int, object, str)
    def _smart_ready(self, model_index, report, error):
        self._smart_job = None
        self.smart_button.setText("Analyze + apply smart view")
        self.smart_button.setEnabled(bool(self.models))
        if model_index != self.phase_combo.currentIndex():
            return
        if error:
            self.smart_summary.setPlainText(error)
            self.copy_smart_button.setEnabled(False)
            self.status.setText(f"Smart-view analysis failed: {error}")
            return
        self._smart_report = report
        self.smart_summary.setPlainText(format_smart_crystal_report(report))
        self.copy_smart_button.setEnabled(True)
        self._apply_smart_plan(report["recommendation"])
        family = report["classification"]["family"]
        confidence = float(report["classification"]["confidence"]) * 100.0
        self.status.setText(
            f"Applied smart view: {family} ({confidence:.0f}% advisory confidence). "
            "Only display controls changed; the structure and refinement are untouched."
        )

    def copy_smart_report(self):
        if self._smart_report is not None:
            QApplication.clipboard().setText(
                format_smart_crystal_report(self._smart_report)
            )
            self.status.setText("Copied the explainable smart-view report.")

    def _apply_smart_plan(self, plan: dict):
        """Apply only display controls from a deterministic advisory plan."""
        self.style_combo.setCurrentText(str(plan["style"]))
        # Ensure the preset is initialized when the requested style was already
        # selected, then override it with the explicit plan below.
        self._style_changed()
        for spin, value in zip(self.repeats, plan["repeats"]):
            spin.setValue(int(value))
        for name, control in {
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "zoom": self.zoom,
            "atom_scale": self.atom_scale,
            "bond_radius": self.bond_radius,
            "polyhedron_opacity": self.opacity,
        }.items():
            control.setValue(float(plan[name]))
        for name, control in {
            "bonds": self.bonds,
            "unlike_only": self.unlike,
            "polyhedra": self.polyhedra,
            "polyhedron_edges": self.polyhedron_edges,
            "hydrogen_bonds": self.hydrogen_bonds,
            "complete_boundaries": self.complete_boundaries,
            "metal_metal_bonds": self.metal_metal_bonds,
            "cell_edges": self.cell_edges,
            "cell_grid": self.cell_grid,
            "axes": self.axes,
            "labels": self.labels,
            "caption": self.caption,
            "hide_hydrogen": self.hide_hydrogen,
            "pore_volumes": self.pore_volumes,
            "gpu_first": self.gpu_first,
        }.items():
            control.setChecked(bool(plan[name]))
        requested_center = str(plan.get("center_element") or "Auto")
        self.center_combo.setCurrentText(
            requested_center if self.center_combo.findText(requested_center) >= 0 else "Auto"
        )
        self.isolate_pore.blockSignals(True)
        self.isolate_pore.setChecked(bool(plan.get("isolate_pore", False)))
        self.isolate_pore.blockSignals(False)
        self.pore_index.setEnabled(self.isolate_pore.isChecked())
        self.pore_shell.setEnabled(self.isolate_pore.isChecked())
        self.smart_pore.setChecked(bool(plan.get("smart_pore_isolation", True)))
        self.smart_pore.setEnabled(self.isolate_pore.isChecked())
        self._schedule()

    def _update_export_availability(self, *_):
        ready = bool(self.models) and self._export_job is None
        if ready:
            index = self.phase_combo.currentIndex()
            model = self.models[index] if 0 <= index < len(self.models) else None
            flagged = bool(model and model.get("provenance", {}).get("publication_ready") is False)
            ready = not flagged or self.allow_flagged_export.isChecked()
        self.export_button.setEnabled(ready)

    def settings(self):
        return RenderSettings(style=self.style_combo.currentText(), repeats=tuple(spin.value() for spin in self.repeats),
            azimuth=self.azimuth.value(), elevation=self.elevation.value(), zoom=self.zoom.value(),
            atom_scale=self.atom_scale.value(), atom_opacity=self.atom_opacity.value() / 100.0,
            bond_radius=self.bond_radius.value(), contact_cutoff=self.cutoff.value(),
            bonds=self.bonds.isChecked(), unlike_only=self.unlike.isChecked(), polyhedra=self.polyhedra.isChecked(),
            center_element=self.center_combo.currentText() or "Auto", polyhedron_opacity=self.opacity.value(),
            polyhedron_edges=self.polyhedron_edges.isChecked(), hydrogen_bonds=self.hydrogen_bonds.isChecked(),
            complete_boundaries=self.complete_boundaries.isChecked(), metal_metal_bonds=self.metal_metal_bonds.isChecked(),
            cell_edges=self.cell_edges.isChecked(), cell_grid=self.cell_grid.isChecked(), axes=self.axes.isChecked(), labels=self.labels.isChecked(),
            caption=self.caption.isChecked(), transparent=self.transparent.isChecked(),
            hide_hydrogen=self.hide_hydrogen.isChecked(),
            pore_volumes=self.pore_volumes.isChecked(),
            pore_probe_radius=self.pore_probe.value(),
            pore_opacity=self.pore_opacity.value(),
            pore_max_count=self.pore_count.value(),
            isolate_pore=self.isolate_pore.isChecked(),
            smart_pore_isolation=self.smart_pore.isChecked(),
            isolated_pore_index=self.pore_index.value() - 1,
            pore_shell_thickness=self.pore_shell.value())

    def _isolate_pore_changed(self, checked):
        self.pore_index.setEnabled(bool(checked))
        self.pore_shell.setEnabled(bool(checked))
        self.smart_pore.setEnabled(bool(checked))
        if checked:
            self.style_combo.setCurrentText(MOF_PORE_STYLE)
            self.pore_volumes.setChecked(True)
            for spin in self.repeats:
                spin.setValue(1)
            self.cell_edges.setChecked(False)
            self.cell_grid.setChecked(False)
            self.axes.setChecked(False)
        self._schedule()

    def _pore_index_changed(self, value):
        if self.pore_count.value() < value:
            self.pore_count.setValue(value)
        self._schedule()

    def _style_changed(self, *_):
        style = self.style_combo.currentText()
        if style in {SCIENTIFIC_STYLE, MOF_STYLE, MOF_PORE_STYLE}:
            for spin in self.repeats:
                spin.setValue(1)
            if style == MOF_PORE_STYLE:
                self.atom_scale.setValue(.55)
                self.bond_radius.setValue(.022)
            else:
                self.atom_scale.setValue(.55 if style == MOF_STYLE else .78)
                self.bond_radius.setValue(.028 if style == MOF_STYLE else .035)
            self.cutoff.setValue(0)
            # Organic MOF linkers require C–C contacts. Metal–metal display
            # contacts remain independently disabled below.
            self.unlike.setChecked(style not in {MOF_STYLE, MOF_PORE_STYLE})
            self.polyhedra.setChecked(True)
            if self.center_combo.findText("All metals") >= 0:
                self.center_combo.setCurrentText("All metals")
            self.opacity.setValue(
                .50 if style == MOF_PORE_STYLE else (.30 if style == MOF_STYLE else .40)
            )
            self.polyhedron_edges.setChecked(True)
            self.hydrogen_bonds.setChecked(style == SCIENTIFIC_STYLE)
            self.hide_hydrogen.setChecked(style in {MOF_STYLE, MOF_PORE_STYLE})
            self.complete_boundaries.setChecked(style != MOF_PORE_STYLE)
            self.metal_metal_bonds.setChecked(False)
            self.cell_edges.setChecked(style != MOF_PORE_STYLE)
            self.cell_grid.setChecked(False)
            self.labels.setChecked(False)
            self.axes.setChecked(style != MOF_PORE_STYLE)
            self.caption.setChecked(style != MOF_PORE_STYLE)
            self.pore_volumes.setChecked(style == MOF_PORE_STYLE)
            self.transparent.setChecked(False)
            if style == MOF_PORE_STYLE:
                # A neutral isometric Cartesian starting view works for cubic,
                # primitive and non-orthogonal cells; the user can still orbit
                # the GPU view to a crystallographically chosen direction.
                self.azimuth.setValue(45.0)
                self.elevation.setValue(35.3)
                self.zoom.setValue(1.02)
                self.pore_probe.setValue(0.0)
                self.pore_opacity.setValue(.88)
                self.pore_count.setValue(2)
            else:
                self.zoom.setValue(.86 if style == MOF_STYLE else .94)
        self._schedule()

    def _current_model(self):
        model = deepcopy(self.models[self.phase_combo.currentIndex()])
        model["data_name"] = self.title_edit.text().strip() or model.get("data_name", "Crystal structure")
        return model

    def _schedule(self, *_):
        self._revision += 1
        self._last_scene = None
        if hasattr(self, "gpu_view_button"):
            self.gpu_view_button.setEnabled(False)
        self._timer.start()

    def _render(self):
        if not self.models or self._preview_job is not None:
            return
        self.status.setText("Rendering crystal…")
        job = _RenderJob(
            self._revision,
            self._current_model(),
            self.settings(),
            skip_cpu_preview=self.gpu_first.isChecked(),
        )
        self._preview_job = job
        job.signals.scene_ready.connect(self._gpu_scene_ready)
        job.signals.finished.connect(self._preview_ready)
        QThreadPool.globalInstance().start(job)

    @Slot(int, object)
    def _gpu_scene_ready(self, revision, scene):
        if revision != self._revision:
            return
        self._last_scene = scene
        self.gpu_view_button.setEnabled(True)
        if self._preview_job is not None:
            if self.gpu_first.isChecked():
                self.status.setText(
                    "GPU geometry is ready. Open GPU 3D for shaded viewing or fast PNG/TIFF export."
                )
            else:
                self.status.setText(
                    "GPU interactive geometry is ready; deterministic publication preview is still rendering…"
                )

    @Slot(int, object, str)
    def _preview_ready(self, revision, value, error):
        self._preview_job = None
        if revision != self._revision:
            self._timer.start()
            return
        if error:
            self.status.setText(error)
            self.export_button.setEnabled(False)
            self.gpu_view_button.setEnabled(False)
            self.preview.clear()
            self.preview._image = None
            self.preview.setText(error)
            return
        qimage, warnings, atom_count, bond_count, hydrogen_bond_count, pore_count = value
        if qimage is None:
            self.preview.clear()
            self.preview._image = None
            self.preview.setText(
                "GPU-first mode\n\nOpen GPU 3D + fast export for the live shaded structure.\n"
                "Uncheck GPU-first mode to also build the deterministic CPU preview."
            )
        else:
            self.preview.set_image(qimage)
        self.status.setText(
            f"{atom_count} displayed atoms · {bond_count} display contacts · "
            f"{hydrogen_bond_count} hydrogen bonds · {pore_count} pore envelopes. "
            + " ".join(warnings)
        )
        self._update_export_availability()

    def open_gpu_view(self):
        if self._last_scene is None:
            QMessageBox.information(
                self,
                "GPU scene is not ready",
                "Wait for the scene geometry to finish building, then try again.",
            )
            return
        try:
            from .crystal_gpu import GpuCrystalViewDialog, gpu_renderer_status

            available, message = gpu_renderer_status()
            if not available:
                raise RuntimeError(message)
            dialog = GpuCrystalViewDialog(self._last_scene, self)
            self._gpu_dialogs.append(dialog)
            dialog.destroyed.connect(
                lambda *_args, owned=dialog: (
                    self._gpu_dialogs.remove(owned)
                    if owned in self._gpu_dialogs else None
                )
            )
            dialog.show()
        except Exception as exc:
            QMessageBox.warning(self, "GPU 3D view unavailable", str(exc))

    def _rotate(self, da, de):
        self.azimuth.setValue((self.azimuth.value() + da + 180) % 360 - 180)
        self.elevation.setValue(self.elevation.value() + de)

    def reset_camera(self):
        self.azimuth.setValue(32)
        self.elevation.setValue(19)
        self.zoom.setValue(1)

    def apply_settings(self, settings):
        self.style_combo.setCurrentText(settings.style)
        self.center_combo.setCurrentText(settings.center_element)
        for spin, value in zip(self.repeats, settings.repeats):
            spin.setValue(value)
        for name, control in {"azimuth": self.azimuth, "elevation": self.elevation, "zoom": self.zoom,
                              "atom_scale": self.atom_scale, "bond_radius": self.bond_radius,
                              "contact_cutoff": self.cutoff, "polyhedron_opacity": self.opacity,
                              "pore_probe_radius": self.pore_probe,
                              "pore_opacity": self.pore_opacity,
                              "pore_max_count": self.pore_count,
                              "pore_shell_thickness": self.pore_shell}.items():
            control.setValue(getattr(settings, name))
        self.atom_opacity.setValue(settings.atom_opacity * 100.0)
        self.pore_index.setValue(settings.isolated_pore_index + 1)
        for name, control in {"bonds": self.bonds, "unlike_only": self.unlike, "polyhedra": self.polyhedra,
                              "polyhedron_edges": self.polyhedron_edges, "hydrogen_bonds": self.hydrogen_bonds,
                              "hide_hydrogen": self.hide_hydrogen,
                              "complete_boundaries": self.complete_boundaries, "metal_metal_bonds": self.metal_metal_bonds,
                              "cell_edges": self.cell_edges, "cell_grid": self.cell_grid,
                              "axes": self.axes, "labels": self.labels,
                              "caption": self.caption, "pore_volumes": self.pore_volumes,
                              "transparent": self.transparent}.items():
            control.setChecked(getattr(settings, name))
        self.isolate_pore.blockSignals(True)
        self.isolate_pore.setChecked(settings.isolate_pore)
        self.isolate_pore.blockSignals(False)
        self.pore_index.setEnabled(settings.isolate_pore)
        self.pore_shell.setEnabled(settings.isolate_pore)
        self.smart_pore.setChecked(settings.smart_pore_isolation)
        self.smart_pore.setEnabled(settings.isolate_pore)
        self._schedule()

    def open_structure(self):
        if self._load_job is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open crystal structure", "", "Crystal structures (*.cif *.json)")
        if not path:
            return
        self.open_button.setEnabled(False)
        self.open_button.setText("Loading and validating…")
        self.status.setText(
            f"Loading {Path(path).name} in the background; large MOF validation can take 10–30 seconds…"
        )
        self.preview.clear()
        self.preview._image = None
        self.preview.setText("Loading CIF and checking periodic geometry…")
        job = _StructureLoadJob(path)
        self._load_job = job
        job.signals.finished.connect(self._structure_loaded)
        QThreadPool.globalInstance().start(job)

    def _apply_safe_large_structure_preset(self, model: dict) -> bool:
        atom_count = len(model.get("atoms") or [])
        if atom_count <= 1800:
            return False
        elements = {
            str(atom.get("element") or "X")
            for atom in (model.get("atoms") or [])
            if float(atom.get("occupancy", 1.0) or 0.0) > 0
        }
        probable_mof = bool(
            "C" in elements
            and (elements & METAL_ELEMENTS)
            and (elements & {"N", "O", "F", "P", "S", "Cl", "Br", "I"})
        )
        self.style_combo.setCurrentText(MOF_STYLE if probable_mof else SCIENTIFIC_STYLE)
        # Apply even when MOF style was already selected and therefore emitted
        # no currentIndexChanged signal.
        self._style_changed()
        for spin in self.repeats:
            spin.setValue(1)
        self.hide_hydrogen.setChecked(True)
        self.gpu_first.setChecked(True)
        if atom_count > 5000:
            # Boundary completion tests many periodic images for every site. A
            # 16k-site conventional MIL-101 cell is already chemically complete
            # enough for an overview and remains available as frozen model data.
            self.complete_boundaries.setChecked(False)
            if not probable_mof:
                # A huge unknown/general structure should first open reliably;
                # the user can enable inferred contacts after reviewing the
                # explainable smart report.
                self.bonds.setChecked(False)
                self.polyhedra.setChecked(False)
        return True

    @Slot(object, str)
    def _structure_loaded(self, value, error):
        self._load_job = None
        self.open_button.setEnabled(True)
        self.open_button.setText("Open CIF / saved scene…")
        if error:
            self.status.setText(error)
            QMessageBox.warning(self, "Cannot open structure", error)
            return
        model, settings = value
        self.models.append(model)
        self._populate_models()
        self.phase_combo.setCurrentIndex(len(self.models) - 1)
        if settings:
            self.apply_settings(settings)
        elif self._apply_safe_large_structure_preset(model):
            self.status.setText(
                f"Loaded {len(model.get('atoms') or []):,} sites. Applied a safe "
                "single-cell MOF view; rendering continues in the background."
            )

    def run_chemistry_qa(self):
        if not self.models or self._qa_job is not None:
            return
        self.qa_button.setEnabled(False)
        self.qa_button.setText("Analyzing…")
        self.qa_summary.setText("Calculating periodic coordination, bond valence, charge and symmetry evidence…")
        job = _QaJob(self._current_model())
        self._qa_job = job
        job.signals.finished.connect(self._qa_ready)
        QThreadPool.globalInstance().start(job)

    @Slot(object, str)
    def _qa_ready(self, report, error):
        self._qa_job = None
        self.qa_button.setEnabled(bool(self.models))
        self.qa_button.setText("Run Crystal Chemistry QA…")
        if error:
            self.qa_summary.setText(error)
            QMessageBox.warning(self, "Crystal Chemistry QA failed", error)
            return
        summary = report["summary"]
        self.qa_summary.setText(
            f"{summary['status'].capitalize()} · {summary['metal_site_count']} metal sites · "
            f"{summary['review_finding_count']} review findings"
        )
        dialog = CrystalChemistryReportDialog(report, self)
        self._qa_dialogs.append(dialog)
        dialog.destroyed.connect(lambda *_: self._qa_dialogs.remove(dialog) if dialog in self._qa_dialogs else None)
        dialog.show()

    def export_figure(self):
        if not self.models or self._export_job is not None:
            return
        model = self._current_model()
        if model.get("provenance", {}).get("publication_ready") is False:
            if not self.allow_flagged_export.isChecked():
                QMessageBox.warning(
                    self,
                    "Structure review required",
                    "Publication export is locked because this structure did not pass the plausibility check. "
                    "Review the CIF, or deliberately enable the expert override to export a visibly flagged review figure.",
                )
                return
            model["provenance"]["expert_override_export"] = True
        path, selected = QFileDialog.getSaveFileName(self, "Export crystal figure", "crystal-figure.png", "PNG image (*.png);;TIFF image (*.tif *.tiff)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".tif" if selected.startswith("TIFF") else ".png"
        width = int(self.resolution.currentData())
        job = _RenderJob(self._revision, model, self.settings(),
                         {"path": path, "width": width, "height": width * 3 // 4, "dpi": self.dpi.value()})
        self._export_job = job
        self.export_button.setEnabled(False)
        self.export_button.setText("Exporting…")
        job.signals.finished.connect(self._export_ready)
        QThreadPool.globalInstance().start(job)

    @Slot(int, object, str)
    def _export_ready(self, revision, value, error):
        self._export_job = None
        self._update_export_availability()
        self.export_button.setText("Export PNG / TIFF…")
        if error:
            QMessageBox.warning(self, "Crystal export failed", error)
        else:
            image_path, scene_path, qa_json, qa_text = value
            self.status.setText(
                f"Saved {image_path}, {scene_path.name}, {qa_json.name} and {qa_text.name}"
            )

    def closeEvent(self, event):
        self._timer.stop()
        # QRunnable jobs own their frozen inputs; QObject slots disconnect on deletion.
        super().closeEvent(event)
