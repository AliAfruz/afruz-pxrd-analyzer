from copy import deepcopy

import numpy as np
import pytest

from afruz_pxrd.advanced_background import BACKGROUND_METHODS, BackgroundParameters, detect_background
from afruz_pxrd.crystallography import calculate_powder_pattern
from afruz_pxrd.models import Dataset
from afruz_pxrd.phase_identification import reference_from_cif_pattern
from afruz_pxrd.project import save_afz, load_afz
from afruz_pxrd.refinement_background import (
    BACKGROUND_KEY, SeedBackground, background_kwargs, background_selection,
    selected_background, subset_background,
)
from afruz_pxrd.refinement_export import export_refinement_txt_bundle
from afruz_pxrd.rietveld_refinement import RietveldPhaseSpec, refine_rietveld
from afruz_pxrd.whole_pattern_refinement import WholePatternPhaseSpec, refine_whole_pattern, _positive_background_basis


@pytest.fixture
def background_dataset(nacl_raw_frame):
    return Dataset(uid="background-data", name="Raw-count fixture",
        x=nacl_raw_frame["two_theta_deg"].to_numpy().copy(),
        y_raw=nacl_raw_frame["intensity_counts"].to_numpy().copy(), metadata={"intensity_unit": "counts"})


@pytest.mark.parametrize("method", BACKGROUND_METHODS)
def test_every_main_background_engine_method_can_be_selected_without_mutating_data(background_dataset, method):
    dataset = background_dataset
    before = dataset.y_raw.copy()
    result = detect_background(dataset.x, dataset.y_raw, BackgroundParameters(method=method, clip_negative=True))
    dataset.metadata[BACKGROUND_KEY] = background_selection(dataset, result)
    kwargs = background_kwargs(dataset)
    assert kwargs["background_seed_y"] == pytest.approx(result["background"])
    assert kwargs["background_seed_provenance"]["parameters"]["method"] == method
    assert kwargs["background_seed_provenance"]["observation_clipping"] is False
    assert np.array_equal(dataset.y_raw, before)
    kwargs["background_seed_y"][0] = 999
    assert dataset.metadata[BACKGROUND_KEY]["background"][0] != 999


def test_selection_afz_roundtrip_and_stale_or_corrupt_snapshots_fail_safely(background_dataset, tmp_path):
    dataset = background_dataset
    result = {"background": np.full(len(dataset.x), 150.), "parameters": {"method": "SNIP"}}
    dataset.metadata[BACKGROUND_KEY] = background_selection(dataset, result)
    path = tmp_path / "background.afz"
    save_afz(path, [dataset], {}, {})
    restored, _, _ = load_afz(path)
    assert selected_background(restored[0]) == selected_background(dataset)
    restored[0].y_raw[0] += 1
    with pytest.raises(ValueError, match="raw pattern changed"):
        selected_background(restored[0])
    dataset.metadata[BACKGROUND_KEY]["background"][0] += 1
    with pytest.raises(ValueError, match="corrupt"):
        selected_background(dataset)


def test_corrupt_peak_protection_or_parameters_fail_with_helpful_error(background_dataset):
    dataset = background_dataset
    for field, bad_value in (("parameters", "invalid"), ("protected_mask", [False])):
        dataset.metadata[BACKGROUND_KEY] = background_selection(dataset, {"background": np.ones(len(dataset.x))})
        dataset.metadata[BACKGROUND_KEY][field] = bad_value
        with pytest.raises(ValueError, match="invalid"):
            selected_background(dataset)


@pytest.mark.parametrize("seed", [[1, 2], [1, float("inf"), 3], [-1, 2, 3]])
def test_invalid_seed_never_silently_enters_the_solver(seed):
    with pytest.raises(ValueError, match="finite, non-negative"):
        SeedBackground(np.arange(3.), seed)


def test_exclusion_subsetting_preserves_raw_units_and_provenance(background_dataset):
    dataset = background_dataset
    curve = np.linspace(100, 130, len(dataset.x))
    dataset.metadata[BACKGROUND_KEY] = background_selection(dataset, {"background": curve})
    mask = (dataset.x < 30) | (dataset.x > 31)
    filtered = Dataset(uid=dataset.uid, name=dataset.name, x=dataset.x[mask], y_raw=dataset.y_raw[mask], metadata=deepcopy(dataset.metadata))
    subset_background(dataset, filtered)
    assert background_kwargs(filtered)["background_seed_y"] == pytest.approx(curve[mask])
    assert filtered.metadata[BACKGROUND_KEY]["source_background_sha256"] == dataset.metadata[BACKGROUND_KEY]["seed_sha256"]


@pytest.mark.parametrize("method", ["rietveld", "pawley", "lebail"])
@pytest.mark.parametrize("use_gpu", [False, True])
def test_native_fit_uses_seed_plus_two_way_correction_and_original_raw_weights(background_dataset, nacl_structure, method, tmp_path, use_gpu):
    dataset = background_dataset
    x, y = dataset.x, dataset.y_raw
    seed = 300 + 20 * np.sin(x / 11)
    dataset.metadata[BACKGROUND_KEY] = background_selection(dataset, {"background": seed, "parameters": {"method": "arPLS"}})
    kwargs = background_kwargs(dataset)
    common = dict(refine_zero_shift=False, refine_profile=False, background_order=3,
                  maximum_nonlinear_evaluations=10, intensity_provenance="raw_counts", intensity_cutoff_percent=0.5, use_gpu=use_gpu)
    if method == "rietveld":
        phase = RietveldPhaseSpec(nacl_structure, refine_cell=False, freeze_structure_factors=True)
        common["validation_stride"] = 6
        run = lambda extra: refine_rietveld(x, y, [phase], **common, **extra)
    else:
        pattern = calculate_powder_pattern(nacl_structure, 1.5406, float(x[0]), float(x[-1]))
        phase = WholePatternPhaseSpec(reference_from_cif_pattern(nacl_structure, pattern, 1.5406), refine_cell=False)
        run = lambda extra: refine_whole_pattern(x, y, [phase], mode="Pawley decomposition" if method == "pawley" else "Le Bail-style extraction", extraction_cycles=4, **common, **extra)
    legacy = run({})
    result = run(kwargs)
    assert result["observed_y"] == y.tolist()
    assert result["statistical_weights"] == legacy["statistical_weights"]
    assert result["background_seed_y"] == pytest.approx(seed)
    assert np.all(np.asarray(result["background_y"]) >= -1e-8)
    assert np.any(np.asarray(result["background_correction_y"]) < 0)
    correction = _positive_background_basis(x, 3) @ np.asarray(result["background_coefficients"])
    assert correction == pytest.approx(result["background_correction_y"], abs=1e-8)
    assert seed + correction == pytest.approx(result["background_y"], abs=1e-8)
    assert np.asarray(result["observed_y"]) - result["calculated_y"] == pytest.approx(result["difference_y"])
    assert result["background_model"]["parameters"]["method"] == "arPLS"
    assert not result["statistics_valid"] and result["rexp_percent"] is None
    assert result["goodness_of_fit"] is None and result["reduced_chi_square"] is None
    if method == "rietveld":
        from afruz_pxrd.rietveld_refinement import simulate_rietveld_pattern
        background_only = simulate_rietveld_pattern(x, [phase], [0.], background_coefficients=result["background_coefficients"], background_seed_y=seed)
        assert background_only == pytest.approx(result["background_y"], abs=1e-8)
        assert result["cross_validation"]["background_seed_uses_validation_observations"]
        assert not result["cross_validation"]["is_independent_validation"]
    bundle = export_refinement_txt_bundle(tmp_path / f"{method}.txt", result, dataset_name=dataset.name)
    profile = (tmp_path / f"{method}_profile.txt").read_text(encoding="utf-8-sig")
    assert "chosen_background_intensity" in profile and "fitted_background_correction" in profile
