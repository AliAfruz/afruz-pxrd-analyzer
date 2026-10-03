from __future__ import annotations

import json

import numpy as np
from PIL import Image
import pytest

from afruz_pxrd.crystallography import _structure_factor_intensities_batch
from afruz_pxrd.gpu_backend import cuda_status
from afruz_pxrd.crystal_gpu import (
    GPU_SAFE_TRIANGLE_VERTEX_LIMIT,
    _atom_batch_ranges,
    _batched_atom_mesh,
    _gpu_theme,
    _pore_sphere_mesh,
    export_gpu_crystal,
)
from afruz_pxrd.crystal_scene import RenderSettings, build_scene, model_from_structure
from afruz_pxrd.rietveld_refinement import RietveldPhaseSpec, refine_rietveld
from afruz_pxrd.whole_pattern_refinement import (
    _build_peak_basis,
    _build_peak_basis_cuda,
    _positive_background_basis,
    _solve_le_bail,
    _solve_le_bail_cuda,
)


def test_cuda_status_is_safe_and_json_ready():
    status = cuda_status()
    assert status["backend"] in {"CPU", "CUDA (CuPy)"}
    assert isinstance(status["available"], bool)
    assert isinstance(status["reason"], str)


def test_cuda_structure_factor_batch_matches_cpu_formula():
    atoms = [
        {"element": "Na", "x": 0.0, "y": 0.0, "z": 0.0, "occupancy": 1.0, "b_iso": 0.2},
        {"element": "Cl", "x": 0.5, "y": 0.5, "z": 0.5, "occupancy": 1.0, "b_iso": 0.3},
        {"element": "O", "x": 0.17, "y": 0.29, "z": 0.41, "occupancy": 0.75, "b_iso": 1.1},
    ]
    hkls = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 1)]
    d_values = [4.1, 2.9, 2.35, 2.05, 1.67]
    two_theta = [10.8, 15.3, 18.9, 21.7, 26.6]
    cpu, _ = _structure_factor_intensities_batch(
        atoms, hkls, d_values, two_theta, use_gpu=False
    )
    accelerated, status = _structure_factor_intensities_batch(
        atoms, hkls, d_values, two_theta, use_gpu=True
    )
    assert accelerated == pytest.approx(cpu, rel=2e-12, abs=2e-9)
    assert status["requested"] is True
    assert np.isfinite(accelerated).all()


def test_rietveld_records_hybrid_acceleration_metadata(
    nacl_raw_frame, nacl_structure
):
    x = nacl_raw_frame["two_theta_deg"].to_numpy(dtype=float)
    y = nacl_raw_frame["intensity_counts"].to_numpy(dtype=float)
    sigma = nacl_raw_frame["sigma_counts"].to_numpy(dtype=float)
    result = refine_rietveld(
        x,
        y,
        [
            RietveldPhaseSpec(
                nacl_structure,
                name="NaCl",
                refine_cell=False,
                freeze_structure_factors=True,
            )
        ],
        wavelength_angstrom=1.5406,
        observed_sigma=sigma,
        intensity_provenance="raw_counts",
        refine_zero_shift=False,
        refine_profile=False,
        maximum_nonlinear_evaluations=2,
        use_gpu=True,
    )
    assert result["success"]
    assert result["acceleration"]["requested"] is True
    assert result["acceleration"]["optimizer_backend"] == "SciPy CPU"
    assert result["acceleration"]["publication_renderer_backend"] == "CPU deterministic"


def test_cuda_whole_pattern_profile_matrix_matches_cpu():
    x = np.linspace(4.0, 70.0, 1600)
    reflections = [
        {
            "two_theta_deg": 6.0 + index * 0.62,
            "reference_intensity": 10.0 + index,
            "spectral_components": [
                {"two_theta_deg": 6.0 + index * 0.62, "relative_intensity": 1.0},
                {"two_theta_deg": 6.018 + index * 0.62, "relative_intensity": 0.5},
            ],
        }
        for index in range(90)
    ]
    options = dict(
        u=0.006,
        v=-0.0002,
        w=0.018,
        eta=0.47,
        profile_model="TCH pseudo-Voigt",
        lorentzian_x=0.018,
        lorentzian_y=0.001,
    )
    cpu, cpu_widths = _build_peak_basis(x, reflections, **options)
    accelerated, accelerated_widths, status = _build_peak_basis_cuda(
        x, reflections, **options
    )
    assert accelerated == pytest.approx(cpu, rel=3e-13, abs=3e-13)
    assert accelerated_widths == pytest.approx(cpu_widths, rel=0, abs=0)
    assert status["requested"] is True


def test_cuda_le_bail_repartition_matches_cpu():
    x = np.linspace(2.0, 80.0, 3000)
    reflections = [
        {
            "two_theta_deg": 2.5 + index * 0.082,
            "reference_intensity": 20.0 + 80.0 * ((index % 19) / 18.0),
        }
        for index in range(900)
    ]
    basis, _ = _build_peak_basis(
        x, reflections, u=0.005, v=0.0, w=0.02, eta=0.5
    )
    background = _positive_background_basis(x, 3)
    true_intensity = np.linspace(8.0, 95.0, len(reflections))
    observed = basis @ true_intensity + 25.0
    weights = np.ones_like(observed)
    cpu = _solve_le_bail(
        basis, background, reflections, observed, weights, 20
    )
    accelerated = _solve_le_bail_cuda(
        basis, background, reflections, observed, weights, 20
    )
    assert accelerated[0] == pytest.approx(cpu[0], rel=3e-12, abs=3e-10)
    assert accelerated[2] == pytest.approx(cpu[2], rel=3e-12, abs=3e-10)
    assert accelerated[3]["acceleration"]["requested"] is True


def test_gpu_atom_spheres_are_one_batched_lit_mesh():
    positions = np.asarray([[0.0, 0.0, 0.0], [3.0, 1.0, -2.0]], dtype=np.float32)
    radii = np.asarray([0.5, 0.8], dtype=np.float32)
    colors = np.asarray([[1.0, 0.0, 0.0, 0.35], [0.0, 0.5, 1.0, 0.7]], dtype=np.float32)
    faces, face_colors = _batched_atom_mesh(positions, radii, colors, subdivisions=1)
    assert faces.shape == (2 * 80, 3, 3)
    assert face_colors.shape == (2 * 80, 4)
    first_distances = np.linalg.norm(faces[:80].reshape(-1, 3), axis=1)
    second_distances = np.linalg.norm(
        faces[80:].reshape(-1, 3) - positions[1], axis=1
    )
    assert first_distances == pytest.approx(0.5, abs=2e-7)
    assert second_distances == pytest.approx(0.8, abs=2e-7)
    assert face_colors[:80, 3] == pytest.approx(0.35)
    assert face_colors[80:, 3] == pytest.approx(0.7)


def test_gpu_large_single_pore_atoms_use_driver_safe_batches():
    batches = _atom_batch_ranges(1014, subdivisions=1)
    assert batches[0][0] == 0
    assert batches[-1][1] == 1014
    assert all(stop > start for start, stop in batches)
    assert len(batches) > 5
    assert all(
        (stop - start) * 80 * 3 <= GPU_SAFE_TRIANGLE_VERTEX_LIMIT
        for start, stop in batches
    )
    assert sum(stop - start for start, stop in batches) == 1014


def test_gpu_pore_sphere_is_dense_gold_and_preserves_opacity():
    faces, colors = _pore_sphere_mesh(np.asarray([1.0, 2.0, 3.0]), 15.79, 0.88)
    assert faces.shape == (1280, 3, 3)
    assert colors.shape == (1280, 4)
    assert colors[:, 0].min() > colors[:, 1].max()
    assert colors[:, 2].max() < 0.02
    assert colors[:, 3] == pytest.approx(0.88)


def test_gpu_tiled_export_writes_image_metadata_and_scene(tmp_path, cscl_structure):
    scene = build_scene(
        model_from_structure(cscl_structure),
        RenderSettings(repeats=(1, 1, 1), caption=False),
    )
    scene.model["provenance"]["condition_number"] = float("inf")

    class Point:
        def x(self): return 0.0
        def y(self): return 0.0
        def z(self): return 0.0

    class FakeGpuView:
        _afruz_background_style = "Journal white"

        def renderToArray(self, size, **kwargs):
            width, height = size
            pixels = np.zeros((height, width, 4), dtype=np.uint8)
            pixels[:] = [10, 20, 30, 255]  # BGRA from OpenGL
            return pixels

        def cameraParams(self):
            return {
                "center": Point(), "distance": 15.0, "fov": 60.0,
                "elevation": 19.0, "azimuth": 32.0,
            }

    image_path, scene_path = export_gpu_crystal(
        FakeGpuView(), scene, tmp_path / "gpu.png", width=400, height=300, dpi=300
    )
    with Image.open(image_path) as picture:
        assert picture.size == (400, 300)
        assert picture.getpixel((0, 0)) == (30, 20, 10, 255)
        embedded = json.loads(picture.info["AfruzCrystalScene"])
        assert embedded["export"]["backend"] == "OpenGL GPU tiled framebuffer"
        assert embedded["export"]["background_style"] == "Journal white"
    document = json.loads(scene_path.read_text(encoding="utf-8"))
    assert document["export"]["camera"]["azimuth"] == 32.0
    assert "perspective projection" in document["conventions"]
    assert document["model"]["provenance"]["condition_number"] == "Infinity"
    assert any(
        isinstance(note, dict)
        and note["path"] == "model.provenance.condition_number"
        for note in document["serialization_notes"]
    )


def test_gpu_journal_theme_is_white_with_contrasting_lines():
    journal = _gpu_theme("Journal white")
    dark = _gpu_theme("Dark studio")
    assert journal["background"] == (255, 255, 255, 255)
    assert max(journal["bond"][:3]) < 0.4
    assert max(journal["cell_edge"][:3]) < 0.4
    assert dark["background"] == (8, 15, 26, 255)
