"""Hardware-accelerated interactive OpenGL view for Crystal Studio scenes."""
from __future__ import annotations

import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, PngImagePlugin
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout,
)

from .crystal_scene import (
    ELEMENT_COLORS,
    MOF_PORE_STYLE,
    MOF_STYLE,
    PORE_ORANGE_RGB,
    SCIENTIFIC_ELEMENT_COLORS,
    SCIENTIFIC_STYLE,
    display_radius,
    element_color,
    scene_document,
)


def gpu_renderer_status() -> tuple[bool, str]:
    try:
        import OpenGL  # noqa: F401
        import pyqtgraph.opengl  # noqa: F401

        return True, "OpenGL GPU renderer ready."
    except Exception as exc:
        return False, f"OpenGL renderer unavailable: {exc}"


GPU_BACKGROUND_STYLES = ("Journal white", "Dark studio")

# PyQtGraph's face-expanded GLMeshItem can become corrupted on a long-lived
# Windows OpenGL context well below the nominal 16-bit/32-bit vertex limits.
# Rietveld MIL-101 scenes are especially good at exposing it because the smart
# pore crop still contains roughly 1,000–2,000 atoms.  Small VBOs cost a few
# extra draw calls but remain fast on a discrete GPU and deterministic on the
# affected NVIDIA compatibility path.
GPU_SAFE_TRIANGLE_VERTEX_LIMIT = 12_000


def _gpu_theme(name: str) -> dict:
    if name == "Journal white":
        return {
            "background": (255, 255, 255, 255),
            "bond": (0.24, 0.28, 0.31, 0.82),
            "hydrogen_bond": (0.10, 0.40, 0.72, 0.82),
            "cell_edge": (0.20, 0.23, 0.27, 0.78),
            "polyhedron_edge": (0.24, 0.29, 0.52, 0.78),
        }
    return {
        "background": (8, 15, 26, 255),
        "bond": (0.62, 0.70, 0.78, 0.70),
        "hydrogen_bond": (0.35, 0.76, 0.96, 0.72),
        "cell_edge": (0.93, 0.95, 0.98, 0.78),
        "polyhedron_edge": (0.52, 0.62, 0.96, 0.68),
    }


def _rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    value = hex_color.lstrip("#")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
        float(alpha),
    )


def _unit_icosphere(subdivisions: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return a compact triangular unit sphere without one mesh per atom."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=np.float32,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.asarray(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int32,
    )
    for _ in range(max(0, int(subdivisions))):
        midpoint_cache: dict[tuple[int, int], int] = {}
        expanded = vertices.tolist()

        def midpoint(first: int, second: int) -> int:
            key = tuple(sorted((int(first), int(second))))
            if key not in midpoint_cache:
                point = (vertices[key[0]] + vertices[key[1]]) * 0.5
                point /= np.linalg.norm(point)
                midpoint_cache[key] = len(expanded)
                expanded.append(point.tolist())
            return midpoint_cache[key]

        refined = []
        for first, second, third in faces:
            ab = midpoint(first, second)
            bc = midpoint(second, third)
            ca = midpoint(third, first)
            refined.extend(
                ((first, ab, ca), (second, bc, ab), (third, ca, bc), (ab, bc, ca))
            )
        vertices = np.asarray(expanded, dtype=np.float32)
        faces = np.asarray(refined, dtype=np.int32)
    return vertices, faces


def _batched_atom_mesh(
    positions: np.ndarray,
    radii: np.ndarray,
    colors: np.ndarray,
    subdivisions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one GPU mesh for all atoms, avoiding thousands of draw calls."""
    vertices, faces = _unit_icosphere(subdivisions)
    template = vertices[faces]
    atom_faces = (
        positions[:, None, None, :]
        + template[None, :, :, :] * radii[:, None, None, None]
    ).reshape(-1, 3, 3)
    # Pre-light the compact mesh. Qt's unshaded color program is exceptionally
    # stable across Windows OpenGL drivers, while the facets still supply a
    # clear volumetric cue and the depth buffer handles true 3D occlusion.
    normals = template.mean(axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    light = np.asarray((0.65, -0.55, 0.85), dtype=np.float32)
    light /= np.linalg.norm(light)
    brightness = 0.56 + 0.44 * np.clip(normals @ light, 0.0, 1.0)
    shaded = np.repeat(colors[:, None, :], len(faces), axis=1)
    shaded[:, :, :3] *= brightness[None, :, None]
    face_colors = shaded.reshape(-1, 4)
    return atom_faces.astype(np.float32, copy=False), face_colors.astype(np.float32, copy=False)


def _atom_batch_ranges(atom_count: int, subdivisions: int) -> list[tuple[int, int]]:
    """Keep each unindexed OpenGL atom VBO below a conservative vertex limit.

    Some Windows OpenGL paths intermittently corrupt moderately large
    face-expanded arrays in a long-lived application context even though the
    hardware advertises support for much larger draw calls.  Keep each upload
    far below that observed failure range.
    """
    faces_per_atom = len(_unit_icosphere(subdivisions)[1])
    atoms_per_batch = max(
        1,
        GPU_SAFE_TRIANGLE_VERTEX_LIMIT // (faces_per_atom * 3),
    )
    return [
        (start, min(int(atom_count), start + atoms_per_batch))
        for start in range(0, int(atom_count), atoms_per_batch)
    ]


def _pore_sphere_mesh(
    center: np.ndarray,
    radius: float,
    opacity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a high-resolution, warmly pre-lit gold pore sphere."""
    vertices, faces = _unit_icosphere(3)
    template = vertices[faces]
    triangles = np.asarray(center, dtype=np.float32)[None, None, :] + template * float(radius)
    normals = template.mean(axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    light = np.asarray((0.55, -0.45, 0.80), dtype=np.float32)
    light /= np.linalg.norm(light)
    brightness = 0.78 + 0.22 * np.clip(normals @ light, 0.0, 1.0)
    base = np.asarray(PORE_ORANGE_RGB, dtype=np.float32)
    colors = np.empty((len(faces), 4), dtype=np.float32)
    colors[:, :3] = base[None, :] * brightness[:, None]
    colors[:, 3] = float(opacity)
    return triangles.astype(np.float32, copy=False), colors


def _gpu_camera_document(view) -> dict:
    values = view.cameraParams()
    center = values.get("center")
    result = {
        "distance": float(values.get("distance", 0.0)),
        "field_of_view_deg": float(values.get("fov", 60.0)),
    }
    if center is not None:
        result["center"] = [float(center.x()), float(center.y()), float(center.z())]
    for name in ("elevation", "azimuth"):
        if name in values:
            result[name] = float(values[name])
    return result


def export_gpu_crystal(view, scene, path: str | Path, width=2400, height=1800, dpi=300):
    """Export the current OpenGL camera through a tiled GPU framebuffer."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".png", ".tif", ".tiff"}:
        raise ValueError("GPU export supports PNG or TIFF.")
    width, height, dpi = int(width), int(height), int(dpi)
    if width < 320 or height < 240:
        raise ValueError("GPU export must be at least 320 × 240 pixels.")
    if not 72 <= dpi <= 1200:
        raise ValueError("DPI must be between 72 and 1,200.")

    # PyQtGraph renders overlapping framebuffer tiles, so exports can exceed
    # the maximum texture size without allocating one giant GPU render target.
    bgra = np.asarray(
        view.renderToArray((width, height), textureSize=2048, padding=64),
        dtype=np.uint8,
    )
    if bgra.shape != (height, width, 4):
        raise RuntimeError(f"OpenGL returned an unexpected image shape: {bgra.shape}")
    rgba = np.ascontiguousarray(bgra[..., [2, 1, 0, 3]])
    picture = Image.fromarray(rgba, "RGBA")

    document = scene_document(scene, width, height, dpi)
    document["export"].update(
        backend="OpenGL GPU tiled framebuffer",
        reproducibility="Appearance can vary slightly with the GPU and graphics driver.",
        camera=_gpu_camera_document(view),
        background_style=getattr(view, "_afruz_background_style", "Unspecified"),
    )
    document["conventions"] = document["conventions"].replace(
        "Orthographic projection", "OpenGL perspective projection"
    )
    serialized = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
    scene_path = path.with_suffix(".crystal.json")
    token = uuid4().hex
    temp_image = path.with_name(f".{path.stem}.{token}{path.suffix}")
    temp_scene = scene_path.with_name(f".{scene_path.stem}.{token}{scene_path.suffix}")
    try:
        if suffix == ".png":
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Description", document["conventions"])
            metadata.add_text("AfruzCrystalScene", serialized)
            picture.save(temp_image, dpi=(dpi, dpi), pnginfo=metadata)
        else:
            picture.save(
                temp_image,
                dpi=(dpi, dpi),
                compression="tiff_lzw",
                tiffinfo={270: serialized},
            )
        temp_scene.write_text(serialized, encoding="utf-8")
        temp_scene.replace(scene_path)
        temp_image.replace(path)
    finally:
        temp_image.unlink(missing_ok=True)
        temp_scene.unlink(missing_ok=True)
    return path, scene_path


class GpuCrystalViewDialog(QDialog):
    """Rotatable GPU view with driver-accelerated high-resolution export."""

    def __init__(self, scene, parent=None):
        super().__init__(parent)
        import pyqtgraph.opengl as gl

        self._gl = gl
        self._scene = scene
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("GPU Interactive Crystal — Afruz PXRD")
        self.resize(1180, 820)
        self.setMinimumSize(680, 500)
        layout = QVBoxLayout(self)
        title = QLabel(
            f"GPU Interactive 3D · {len(scene.positions):,} atoms · "
            f"{len(scene.bonds):,} contacts"
        )
        title.setStyleSheet("font-size:18px; font-weight:600; padding:3px;")
        layout.addWidget(title)
        note = QLabel(
            "Drag to orbit · mouse wheel to zoom · middle/right drag to pan. "
            "Atoms, polyhedra, bonds and pore volumes are drawn by OpenGL. "
            "GPU export is fast; the main Crystal Studio exporter remains the "
            "deterministic publication reference."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        layout.addWidget(note)
        if scene.pore_radii:
            pore_note = QLabel(
                "Scientific pore convention: the gold sphere is a grid-derived "
                f"maximal-empty-sphere estimate (radius {scene.pore_radii[0]:.2f} Å), "
                "not an adsorption-accessible surface or a refined atom."
            )
            pore_note.setWordWrap(True)
            pore_note.setObjectName("mutedLabel")
            layout.addWidget(pore_note)

        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("GPU background"))
        self.background_combo = QComboBox()
        self.background_combo.addItems(GPU_BACKGROUND_STYLES)
        scientific_scene = scene.settings.style in {
            SCIENTIFIC_STYLE, MOF_STYLE, MOF_PORE_STYLE
        }
        self.background_combo.setCurrentText(
            "Journal white" if scientific_scene else "Dark studio"
        )
        self.background_combo.setToolTip(
            "Journal white uses a pure white framebuffer and darker structural "
            "lines for figures; Dark studio is optimized for interactive viewing."
        )
        theme_row.addWidget(self.background_combo)
        theme_row.addStretch()
        layout.addLayout(theme_row)

        class SafeCrystalView(gl.GLViewWidget):
            """Safe camera plus per-context pyqtgraph shader ownership.

            Pyqtgraph 0.14 caches mesh and line shader program IDs globally.
            Those IDs are only valid in the OpenGL context that created them;
            opening a second Crystal Studio window after the first context is
            destroyed can therefore reuse an invalid program and draw giant
            black/white triangles. Restore this view's programs before every
            paint and capture newly compiled IDs afterward.
            """

            def __init__(widget, *args, **kwargs):
                super().__init__(*args, **kwargs)
                widget._afruz_mesh_programs = {}
                widget._afruz_line_program = None

            @staticmethod
            def _valid_program(program):
                if program in (None, -1, 0):
                    return False
                try:
                    from OpenGL import GL

                    return bool(GL.glIsProgram(int(program)))
                except Exception:
                    return False

            @staticmethod
            def _clear_gl_errors():
                try:
                    from OpenGL import GL

                    for _ in range(16):
                        if GL.glGetError() == GL.GL_NO_ERROR:
                            break
                except Exception:
                    pass

            def _restore_shader_cache(widget):
                from pyqtgraph.opengl import shaders as pg_shaders
                from pyqtgraph.opengl.items.GLLinePlotItem import GLLinePlotItem

                widget._clear_gl_errors()
                for name, shader in pg_shaders.ShaderProgram.names.items():
                    program = widget._afruz_mesh_programs.get(name)
                    shader.prog = program if widget._valid_program(program) else None
                line_program = widget._afruz_line_program
                GLLinePlotItem._shaderProgram = (
                    line_program if widget._valid_program(line_program) else None
                )

            def _capture_shader_cache(widget):
                from pyqtgraph.opengl import shaders as pg_shaders
                from pyqtgraph.opengl.items.GLLinePlotItem import GLLinePlotItem

                widget._afruz_mesh_programs = {
                    name: shader.prog
                    for name, shader in pg_shaders.ShaderProgram.names.items()
                    if widget._valid_program(shader.prog)
                }
                line_program = GLLinePlotItem._shaderProgram
                widget._afruz_line_program = (
                    line_program if widget._valid_program(line_program) else None
                )

            def initializeGL(widget):
                # A QOpenGLWidget may receive a replacement context after being
                # reparented or recreated. Never carry program IDs across it.
                widget._afruz_mesh_programs = {}
                widget._afruz_line_program = None
                widget._restore_shader_cache()
                super().initializeGL()

            def paintGL(widget):
                widget._restore_shader_cache()
                try:
                    super().paintGL()
                finally:
                    widget._capture_shader_cache()

            def setCameraPosition(widget, pos=None, distance=None, **kwargs):
                if distance is not None:
                    distance = float(np.clip(
                        distance,
                        getattr(widget, "_afruz_min_distance", 0.05),
                        getattr(widget, "_afruz_max_distance", 1.0e7),
                    ))
                return super().setCameraPosition(pos=pos, distance=distance, **kwargs)

            def wheelEvent(widget, event):
                super().wheelEvent(event)
                widget.opts["distance"] = float(np.clip(
                    widget.opts["distance"],
                    getattr(widget, "_afruz_min_distance", 0.05),
                    getattr(widget, "_afruz_max_distance", 1.0e7),
                ))
                widget.opts["fov"] = float(np.clip(widget.opts["fov"], 12.0, 90.0))
                widget.update()

        self.view = SafeCrystalView()
        # Apply the background before uploading geometry so the first visible
        # frame cannot expose Qt's default black framebuffer.
        self.view.setBackgroundColor(
            _gpu_theme(self.background_combo.currentText())["background"]
        )
        self._atom_items = []
        self._bond_item = None
        self._hydrogen_item = None
        self._polyhedron_item = None
        self._edge_item = None
        layout.addWidget(self.view, 1)

        self.renderer_label = QLabel("Initializing OpenGL context…")
        self.renderer_label.setObjectName("mutedLabel")
        layout.addWidget(self.renderer_label)

        row = QHBoxLayout()
        reset = QPushButton("Reset camera")
        reset.clicked.connect(self.reset_camera)
        row.addWidget(reset)
        self.resolution = QComboBox()
        for width in (1200, 1600, 2400, 3200, 4000):
            self.resolution.addItem(f"{width:,} × {width * 3 // 4:,}", width)
        self.resolution.setCurrentIndex(2)
        row.addWidget(self.resolution)
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setValue(300)
        self.dpi.setSuffix(" dpi")
        row.addWidget(self.dpi)
        self.export_button = QPushButton("Export GPU PNG / TIFF…")
        self.export_button.setObjectName("primaryButton")
        self.export_button.clicked.connect(self.export_figure)
        row.addWidget(self.export_button)
        row.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        row.addWidget(buttons)
        layout.addLayout(row)

        self._populate()
        self.background_combo.currentTextChanged.connect(self._background_changed)
        self._background_changed(self.background_combo.currentText())
        self.reset_camera()
        QTimer.singleShot(0, self._update_renderer_label)

    def _populate(self):
        gl = self._gl
        scene = self._scene
        positions = np.asarray(scene.positions, dtype=np.float32)
        if not len(positions):
            return
        primary_positions = positions[np.asarray(scene.primary, dtype=bool)]
        bounds_source = primary_positions if len(primary_positions) else positions
        self._center = (bounds_source.min(axis=0) + bounds_source.max(axis=0)) / 2.0
        centered = positions - self._center
        extent = np.ptp(bounds_source, axis=0)
        pore_guard = max(
            (
                float(np.linalg.norm(np.asarray(center, dtype=float) - self._center))
                + float(radius)
                for center, radius in zip(scene.pore_centers, scene.pore_radii)
            ),
            default=0.0,
        )
        bounding_radius = float(np.max(np.linalg.norm(bounds_source - self._center, axis=1)))
        safe_minimum = max(1.5, bounding_radius * 0.06)
        if pore_guard:
            # The black-polygon failure occurs when the camera crosses a large
            # translucent pore sphere and looks at its clipped inner faces.
            safe_minimum = max(safe_minimum, pore_guard * 1.10)
        self._distance = max(
            8.0,
            float(np.linalg.norm(extent)) * 1.15,
            safe_minimum * 1.8,
        )
        if scene.settings.isolate_pore:
            # A single cage is the subject, so compose it more tightly than a
            # complete periodic unit cell while retaining comfortable margins.
            self._distance = max(safe_minimum * 1.8, self._distance * 0.75)
        self.view._afruz_min_distance = safe_minimum
        self.view._afruz_max_distance = self._distance * 20.0

        # Draw the closed pore envelope first and let it write the depth
        # buffer. Framework atoms on the camera-facing cage wall then remain
        # visible, while rear atoms are correctly occluded instead of looking
        # as though they lie inside the void.
        for center, radius in zip(scene.pore_centers, scene.pore_radii):
            from OpenGL import GL

            pore_opacity = float(scene.settings.pore_opacity)
            translated = np.asarray(center, dtype=float) - self._center
            pore_faces, pore_colors = _pore_sphere_mesh(
                translated, float(radius), pore_opacity
            )
            pore = gl.GLMeshItem(
                vertexes=pore_faces,
                vertexColors=np.repeat(pore_colors[:, None, :], 3, axis=1),
                drawEdges=False,
                smooth=False,
                shader=None,
                glOptions="opaque" if pore_opacity >= 0.999 else "translucent",
            )
            pore.updateGLOptions({
                GL.GL_CULL_FACE: True,
                "glCullFace": (GL.GL_BACK,),
            })
            self.view.addItem(pore)

        scientific = scene.settings.style in {
            SCIENTIFIC_STYLE, MOF_STYLE, MOF_PORE_STYLE
        }
        atom_opacity = float(scene.settings.atom_opacity)
        colors = np.asarray(
            [
                _rgba(element_color(element, scientific), atom_opacity)
                for element in scene.elements
            ],
            dtype=np.float32,
        )
        radii = np.asarray(
            [
                max(
                    0.08,
                    2.0 * display_radius(element, scientific)
                    * float(scene.settings.atom_scale),
                )
                for element in scene.elements
            ],
            dtype=np.float32,
        )
        if len(positions) <= 50000:
            # Compact icosphere batches give real depth and lighting while
            # staying below conservative Windows OpenGL VBO limits.
            subdivisions = 1 if len(positions) <= 3000 else 0
            for start, stop in _atom_batch_ranges(len(positions), subdivisions):
                atom_faces, atom_colors = _batched_atom_mesh(
                    centered[start:stop],
                    radii[start:stop],
                    colors[start:stop],
                    subdivisions,
                )
                if not np.isfinite(atom_faces).all() or not np.isfinite(atom_colors).all():
                    raise RuntimeError(
                        "GPU atom geometry contains a non-finite value; use the deterministic CPU export."
                    )
                atom_faces = np.ascontiguousarray(atom_faces, dtype=np.float32)
                vertex_colors = np.ascontiguousarray(
                    np.repeat(atom_colors[:, None, :], 3, axis=1),
                    dtype=np.float32,
                )
                atoms = gl.GLMeshItem(
                    vertexes=atom_faces,
                    # PyQtGraph's face-indexed VBO needs one color per triangle
                    # vertex; a single face color otherwise leaves two vertices
                    # reading undefined GPU memory on some drivers.
                    vertexColors=vertex_colors,
                    drawEdges=False,
                    smooth=False,
                    shader=None,
                    glOptions="translucent" if atom_opacity < 0.999 else "opaque",
                )
                self.view.addItem(atoms)
                self._atom_items.append(atoms)
        else:
            sizes = np.maximum(2.0, radii * 9.0)
            atoms = gl.GLScatterPlotItem(
                pos=centered, color=colors, size=sizes, pxMode=True
            )
            atoms.setGLOptions(
                "translucent" if atom_opacity < 0.999 else "opaque"
            )
            self.view.addItem(atoms)
            self._atom_items.append(atoms)

        if scene.bonds:
            theme = _gpu_theme(self.background_combo.currentText())
            line_positions = np.asarray(
                [centered[index] for pair in scene.bonds for index in pair],
                dtype=np.float32,
            )
            bonds = gl.GLLinePlotItem(
                pos=line_positions,
                color=theme["bond"],
                width=max(1.0, float(scene.settings.bond_radius) * 24.0),
                antialias=True,
                mode="lines",
            )
            bonds.setGLOptions("translucent")
            self.view.addItem(bonds)
            self._bond_item = bonds

        if scene.hydrogen_bonds:
            hydrogen_positions = np.asarray(
                [centered[index] for pair in scene.hydrogen_bonds for index in pair],
                dtype=np.float32,
            )
            hydrogen = gl.GLLinePlotItem(
                pos=hydrogen_positions,
                color=_gpu_theme(self.background_combo.currentText())["hydrogen_bond"],
                width=1.0,
                antialias=True,
                mode="lines",
            )
            hydrogen.setGLOptions("translucent")
            self.view.addItem(hydrogen)
            self._hydrogen_item = hydrogen

        if scene.triangles:
            vertices = np.asarray(scene.triangles, dtype=np.float32) - self._center
            face_colors = np.asarray(
                [
                    _rgba(
                        element_color(element, scientific),
                        max(0.08, float(scene.settings.polyhedron_opacity)),
                    )
                    for element in scene.triangle_elements
                ],
                dtype=np.float32,
            )
            mesh = gl.GLMeshItem(
                vertexes=vertices,
                vertexColors=np.repeat(face_colors[:, None, :], 3, axis=1),
                drawEdges=bool(scene.settings.polyhedron_edges),
                edgeColor=_gpu_theme(self.background_combo.currentText())["polyhedron_edge"],
                smooth=False,
                shader="shaded",
                glOptions="translucent",
            )
            self.view.addItem(mesh)
            self._polyhedron_item = mesh

        if scene.edges:
            edge_positions = np.asarray(
                [point - self._center for edge in scene.edges for point in edge],
                dtype=np.float32,
            )
            edges = gl.GLLinePlotItem(
                pos=edge_positions,
                color=_gpu_theme(self.background_combo.currentText())["cell_edge"],
                width=1.3,
                antialias=True,
                mode="lines",
            )
            edges.setGLOptions("translucent")
            self.view.addItem(edges)
            self._edge_item = edges

    def _background_changed(self, name):
        theme = _gpu_theme(str(name))
        self.view.setBackgroundColor(theme["background"])
        self.view._afruz_background_style = str(name)
        if self._bond_item is not None:
            self._bond_item.setData(color=theme["bond"])
        if self._hydrogen_item is not None:
            self._hydrogen_item.setData(color=theme["hydrogen_bond"])
        if self._edge_item is not None:
            self._edge_item.setData(color=theme["cell_edge"])
        if self._polyhedron_item is not None:
            self._polyhedron_item.opts["edgeColor"] = theme["polyhedron_edge"]
            self._polyhedron_item.update()
        self.view.update()

    def reset_camera(self):
        settings = self._scene.settings
        distance = self._distance / max(0.35, float(settings.zoom))
        self.view.setCameraPosition(
            pos=QVector3D(0.0, 0.0, 0.0),
            distance=distance,
            elevation=float(settings.elevation),
            azimuth=float(settings.azimuth),
        )

    def _update_renderer_label(self):
        try:
            from OpenGL import GL

            self.view.makeCurrent()
            vendor = GL.glGetString(GL.GL_VENDOR)
            renderer = GL.glGetString(GL.GL_RENDERER)
            version = GL.glGetString(GL.GL_VERSION)
            decode = lambda value: value.decode("utf-8", "replace") if value else "unknown"
            self.renderer_label.setText(
                f"OpenGL GPU · {decode(vendor)} · {decode(renderer)} · {decode(version)}"
            )
        except Exception as exc:
            self.renderer_label.setText(f"OpenGL context ready; renderer details unavailable: {exc}")

    def export_figure(self):
        width = int(self.resolution.currentData())
        height = width * 3 // 4
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Export GPU crystal figure",
            "crystal-gpu.png",
            "PNG image (*.png);;TIFF image (*.tif *.tiff)",
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".tif" if selected.startswith("TIFF") else ".png"
        progress = QProgressBar(self)
        progress.setRange(0, 0)
        progress.setFormat("Rendering tiled OpenGL image…")
        progress.setTextVisible(True)
        progress.setMinimumHeight(24)
        self.layout().insertWidget(self.layout().count() - 1, progress)
        self.export_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            image_path, scene_path = export_gpu_crystal(
                self.view,
                self._scene,
                path,
                width=width,
                height=height,
                dpi=self.dpi.value(),
            )
            QMessageBox.information(
                self,
                "GPU crystal export saved",
                f"Saved {image_path}\nScene: {scene_path}\n\n"
                "The current interactive camera was preserved. For bitwise-stable "
                "publication output across computers, use Crystal Studio's CPU export.",
            )
        except Exception as exc:
            QMessageBox.warning(self, "GPU export failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
            self.export_button.setEnabled(True)
            progress.deleteLater()
