from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .version import APP_VERSION


RECIPE_VERSION = 1

DEFAULT_BATCH_STAGES = {
    "preprocessing": True,
    "peak_detection": True,
    "peak_fitting": True,
    "size_strain": True,
    "cif_cell": False,
    "phase_identification": True,
    "whole_pattern": False,
    "rietveld": False,
    "qpa": False,
    "residual_stress": False,
}

DEFAULT_RECIPE = {
    "recipe_version": RECIPE_VERSION,
    "name": "Afruz shared batch recipe",
    "created_utc": "",
    "software_version": APP_VERSION,
    "stages": DEFAULT_BATCH_STAGES,
    "preprocessing": {
        "subtract_background": False,
        "background_method": "Auto ensemble",
        "background_smoothness": 70.0,
        "background_asymmetry": 0.01,
        "background_iterations": 50,
        "background_window_degrees": 2.0,
        "background_percentile": 20.0,
        "background_peak_protection": True,
        "clip_negative": False,
        "polynomial_order": 3,
        "smooth": False,
        "smoothing_window": 11,
        "smoothing_order": 3,
        "smoothing_method": "Auto intelligent",
        "smoothing_strength": 45.0,
        "smoothing_peak_protection": True,
        "smoothing_peak_preservation": 85.0,
        "smoothing_maximum_position_shift_deg": 0.02,
        "smoothing_maximum_height_change_percent": 8.0,
        "smoothing_maximum_fwhm_change_percent": 10.0,
        "normalize": False,
    },
    "peak_detection": {
        "mode": "Smart",
        "smart_sensitivity": "Balanced",
        "prominence_fraction": 0.03,
        "minimum_distance_points": 5,
    },
    "peak_fitting": {
        "mode": "Standard",
        "model": "Pseudo-Voigt",
        "window_multiplier": 5.0,
    },
    "instrument_calibration": {
        "enabled": False,
        "profile": None,
        "profile_fingerprint": None,
    },
    "size_strain": {
        "wavelength_angstrom": 1.5406,
        "shape_factor": 0.9,
        "instrument_fwhm_deg": 0.0,
        "correction_mode": "None",
    },
    "cif_cell": {
        "match_tolerance_deg": 0.20,
        "crystal_system": "Triclinic",
        "refine_zero_shift": True,
    },
    "phase_identification": {
        "peak_source": "Auto: fitted > detected > smart",
        "tolerance_deg": 0.20,
        "maximum_zero_shift_deg": 0.30,
        "reference_intensity_cutoff_percent": 1.0,
        "maximum_phases": 1,
        "mixture_pool_size": 6,
        "convert_from_d": True,
    },
    "whole_pattern": {
        "mode": "Pawley decomposition",
        "use_gpu": True,
        "use_processed_pattern": True,
        "wavelength_angstrom": 1.5406,
        "two_theta_min_deg": None,
        "two_theta_max_deg": None,
        "reference_intensity_cutoff_percent": 0.5,
        "background_order": 3,
        "weighting": "Poisson-like",
        "refine_zero_shift": True,
        "refine_profile": True,
        "refine_eta": False,
        "initial_u": 0.005,
        "initial_v": 0.0,
        "initial_w": 0.02,
        "initial_eta": 0.5,
        "cell_tolerance_percent": 3.0,
        "extraction_cycles": 12,
        "maximum_nonlinear_evaluations": 120,
        "maximum_optimization_points": 1800,
        "phase_uids": [],
        "refine_cell_uids": [],
    },
    "rietveld": {
        "use_gpu": True,
        "use_processed_pattern": True,
        "wavelength_angstrom": 1.5406,
        "two_theta_min_deg": None,
        "two_theta_max_deg": None,
        "intensity_cutoff_percent": 0.2,
        "background_order": 3,
        "weighting": "Poisson-like",
        "robust_loss": "soft_l1",
        "refine_zero_shift": True,
        "refine_profile": True,
        "refine_eta": False,
        "initial_u": 0.005,
        "initial_v": 0.0,
        "initial_w": 0.02,
        "initial_eta": 0.5,
        "cell_tolerance_percent": 3.0,
        "k_alpha2_enabled": False,
        "k_alpha2_wavelength_angstrom": 1.54439,
        "k_alpha2_ratio": 0.5,
        "maximum_nonlinear_evaluations": 120,
        "maximum_optimization_points": 1800,
    },
    "qpa": {
        "mode": "Pattern scale fractions",
        "use_processed_pattern": True,
        "convert_from_d": True,
        "reference_intensity_cutoff_percent": 1.0,
        "reference_fwhm_deg": 0.20,
        "pseudo_voigt_eta": 0.5,
        "baseline_order": 1,
        "weighting": "Balanced",
        "bootstrap_samples": 0,
    },
    "residual_stress": {
        "peak_source": "Auto: fitted > detected > raw maximum",
        "target_two_theta_deg": 45.0,
        "search_half_window_deg": 1.0,
        "wavelength_angstrom": 1.5406,
        "default_peak_error_deg": 0.01,
        "azimuth_deg": 0.0,
        "reference_mode": "Fit d0 from sin²ψ intercept",
        "stress_free_two_theta_deg": 45.0,
        "elastic_mode": "Young's modulus and Poisson ratio",
        "youngs_modulus_gpa": 200.0,
        "poisson_ratio": 0.30,
        "xec_half_s2_per_gpa": 0.0058,
        "regression_mode": "Weighted least squares",
    },
    "scientific_controls": {
        "strict_common_recipe": True,
        "record_manual_deviations": True,
        "continue_on_error": True,
    },
}


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_recipe() -> dict[str, Any]:
    recipe = deepcopy(DEFAULT_RECIPE)
    recipe["created_utc"] = utc_now_text()
    return recipe


def canonical_recipe_json(recipe: dict[str, Any]) -> str:
    return json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def recipe_fingerprint(recipe: dict[str, Any]) -> str:
    payload = canonical_recipe_json(recipe).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def validate_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(recipe, dict):
        raise ValueError("Batch recipe must be a JSON object.")
    merged = new_recipe()

    def merge(target: dict, incoming: dict):
        for key, value in incoming.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                merge(target[key], value)
            else:
                target[key] = deepcopy(value)

    merge(merged, recipe)
    merged["recipe_version"] = RECIPE_VERSION

    stages = merged.get("stages", {})
    if not any(bool(value) for value in stages.values()):
        raise ValueError("At least one batch stage must be enabled.")

    pre = merged["preprocessing"]
    if str(pre["background_method"]) not in {
        "Auto ensemble", "arPLS", "AsLS", "SNIP", "Rolling percentile", "Polynomial"
    }:
        raise ValueError("Unsupported advanced background method.")
    if not 1.0 <= float(pre["background_smoothness"]) <= 100.0:
        raise ValueError("Background smoothness must be between 1 and 100.")
    if not 0.00001 <= float(pre["background_asymmetry"]) < 0.5:
        raise ValueError("AsLS asymmetry must be between 0.00001 and 0.5.")
    if int(pre["background_iterations"]) < 5:
        raise ValueError("Background iterations must be at least 5.")
    if float(pre["background_window_degrees"]) <= 0:
        raise ValueError("Background window width must be positive.")
    if int(pre["polynomial_order"]) < 1 or int(pre["polynomial_order"]) > 12:
        raise ValueError("Polynomial background order must be between 1 and 12.")
    if int(pre["smoothing_window"]) < 5:
        raise ValueError("Smoothing window must be at least 5.")
    if str(pre["smoothing_method"]) not in {
        "Auto intelligent",
        "Savitzky–Golay",
        "Gaussian",
        "Whittaker–Eilers",
        "Adaptive bilateral",
        "Median–Gaussian hybrid",
    }:
        raise ValueError("Unsupported advanced smoothing method.")
    if not 0.0 <= float(pre["smoothing_strength"]) <= 100.0:
        raise ValueError("Smoothing strength must be between 0 and 100.")
    if not 0.0 <= float(pre["smoothing_peak_preservation"]) <= 100.0:
        raise ValueError("Peak preservation must be between 0 and 100%.")
    if float(pre["smoothing_maximum_position_shift_deg"]) <= 0:
        raise ValueError("Maximum smoothing position shift must be positive.")

    size = merged["size_strain"]
    if float(size["wavelength_angstrom"]) <= 0:
        raise ValueError("Wavelength must be positive.")
    if float(size["shape_factor"]) <= 0:
        raise ValueError("Scherrer shape factor must be positive.")

    phase = merged["phase_identification"]
    if float(phase["tolerance_deg"]) <= 0:
        raise ValueError("Phase-identification tolerance must be positive.")
    if int(phase["maximum_phases"]) not in (1, 2, 3):
        raise ValueError("Maximum phase mixture size must be 1, 2, or 3.")

    whole = merged["whole_pattern"]
    if int(whole["background_order"]) < 0 or int(whole["background_order"]) > 6:
        raise ValueError("Whole-pattern background order must be between 0 and 6.")
    if float(whole["wavelength_angstrom"]) <= 0:
        raise ValueError("Whole-pattern wavelength must be positive.")

    rietveld = merged["rietveld"]
    if int(rietveld["background_order"]) < 0 or int(rietveld["background_order"]) > 6:
        raise ValueError("Rietveld background order must be between 0 and 6.")
    if float(rietveld["wavelength_angstrom"]) <= 0:
        raise ValueError("Rietveld wavelength must be positive.")

    qpa = merged["qpa"]
    if int(qpa["baseline_order"]) < 0 or int(qpa["baseline_order"]) > 4:
        raise ValueError("QPA baseline order must be between 0 and 4.")

    stress = merged["residual_stress"]
    if float(stress["wavelength_angstrom"]) <= 0:
        raise ValueError("Residual-stress wavelength must be positive.")

    return merged


def save_recipe(path: str | Path, recipe: dict[str, Any]) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    validated = validate_recipe(recipe)
    path.write_text(json.dumps(validated, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_recipe(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    recipe = json.loads(path.read_text(encoding="utf-8"))
    return validate_recipe(recipe)


def compare_recipe_fingerprints(
    recipe: dict[str, Any],
    dataset_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    fingerprint = recipe_fingerprint(recipe)
    metadata = dataset_metadata or {}
    stored = metadata.get("batch_recipe_fingerprint")
    return {
        "recipe_fingerprint": fingerprint,
        "dataset_fingerprint": stored,
        "deviates": bool(stored and stored != fingerprint),
    }
