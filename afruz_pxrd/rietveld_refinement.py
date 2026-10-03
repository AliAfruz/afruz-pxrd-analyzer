from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from .crystal_profiles import (
    PROFILE_MODELS,
    information_criteria,
    profile_characteristics,
    profile_unit_area,
    residual_diagnostics,
    staged_parameter_groups,
    staged_profile_refinement_plan,
)
from .crystallography import (
    _structure_factor_intensity,
    _structure_factor_intensities_batch,
    calculate_powder_pattern,
    d_spacing,
    direct_metric_tensor,
)
from .gpu_backend import cuda_status, release_gpu_memory
from .structure_validation import (
    ensure_site_formula_metadata,
    structure_refinement_gate,
    validate_crystal_structure,
)
from .crystal_scene import structure_snapshot
from .instrument_physics import (
    CalibrationRangeError,
    calibration_prior_residual,
    prepare_calibration_prior,
    validate_calibration_range,
    validate_profile_compatibility,
)
from .refinement_statistics import (
    build_refinement_weight_model,
    calculate_profile_statistics,
)


RIETVELD_WEIGHTING = (
    "Poisson-like",
    "Balanced",
    "Uniform",
)

RIETVELD_LOSSES = (
    "linear",
    "soft_l1",
    "huber",
)

GENERATED_PATTERN_SCALING = (
    "Joint weighted scale + background",
    "Joint unweighted scale + background",
)


class RietveldError(ValueError):
    pass


class RietveldCancelled(RuntimeError):
    pass


@dataclass
class RietveldPhaseSpec:
    structure: dict
    name: str | None = None
    refine_cell: bool = True
    refine_biso: bool = False
    preferred_orientation_hkl: tuple[int, int, int] | None = None
    refine_preferred_orientation: bool = False
    initial_preferred_orientation_r: float = 1.0
    freeze_structure_factors: bool = False
    included: bool = True
    allow_flagged_structure: bool = False


def validate_phase_specs_for_refinement(phase_specs: list[RietveldPhaseSpec]) -> list[dict]:
    """Validate every included atomic model and enforce the expert-override gate."""
    diagnostics = []
    for spec in phase_specs:
        if not spec.included:
            continue
        structure = deepcopy(spec.structure)
        ensure_site_formula_metadata(structure)
        # Recompute against a changed metric, but reuse an exact-cell validation
        # for immutable expanded MOF models. A 10k+ site periodic geometry audit
        # is intentionally thorough and should not be repeated for every smart
        # model candidate.
        cell_signature = tuple(
            float((structure.get("cell") or {}).get(key, float("nan")))
            for key in ("a", "b", "c", "alpha", "beta", "gamma")
        )
        atom_count = len(structure.get("atoms") or [])
        cached = structure.get("_afruz_structure_plausibility")
        cache_matches = (
            isinstance(cached, dict)
            and tuple(structure.get("_afruz_validation_cell_signature") or ())
            == cell_signature
            and int(structure.get("_afruz_validation_atom_count", -1))
            == atom_count
        )
        validation = deepcopy(cached) if cache_matches else validate_crystal_structure(structure)
        structure["_afruz_structure_plausibility"] = deepcopy(validation)
        structure["_afruz_validation_cell_signature"] = list(cell_signature)
        structure["_afruz_validation_atom_count"] = atom_count
        spec.structure = structure
        gate = structure_refinement_gate(validation)
        name = str(spec.name or spec.structure.get("data_name") or "CIF phase")
        diagnostics.append({"phase": name, "validation": validation, "gate": gate})
        if not gate["passed"] and not spec.allow_flagged_structure:
            reasons = "; ".join(gate["reasons"][:4]) or gate["status"]
            raise RietveldError(
                f"{name} is blocked before refinement: {gate['status']} "
                f"(score {gate['score']:.1f}). {reasons} "
                "Review/repair the CIF, or deliberately enable the expert override."
            )
    return diagnostics


def _background_basis(x: np.ndarray, order: int) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    span = max(float(np.ptp(values)), np.finfo(float).eps)
    t = np.clip((values - float(np.min(values))) / span, 0.0, 1.0)
    return np.column_stack(
        [
            math.comb(order, degree)
            * t**degree
            * (1.0 - t) ** (order - degree)
            for degree in range(order + 1)
        ]
    )


def _weight_vector(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if mode == "Uniform":
        return np.ones_like(y)
    shifted = y - float(np.min(y))
    positive = shifted[shifted > 0]
    floor = float(np.percentile(positive, 10.0)) if positive.size else 1.0
    floor = max(floor, np.finfo(float).eps)
    if mode == "Poisson-like":
        weights = 1.0 / np.sqrt(np.maximum(shifted, floor))
    elif mode == "Balanced":
        scale = max(float(np.percentile(shifted, 95.0)), floor)
        weights = 1.0 / np.sqrt(0.08 + shifted / scale)
    else:
        raise RietveldError(f"Unsupported weighting mode: {mode}")
    mean = float(np.mean(weights))
    return weights / mean if mean > 0 else np.ones_like(y)


def _pseudo_voigt_unit_area(
    x: np.ndarray,
    center: float,
    fwhm: float,
    eta: float,
) -> np.ndarray:
    width = max(float(fwhm), np.finfo(float).eps)
    delta = (np.asarray(x, dtype=float) - float(center)) / width
    gaussian = np.exp(-4.0 * np.log(2.0) * delta * delta)
    lorentzian = 1.0 / (1.0 + 4.0 * delta * delta)
    mixing = float(np.clip(eta, 0.0, 1.0))
    profile = mixing * lorentzian + (1.0 - mixing) * gaussian
    # Analytic area avoids dependence on cropped integration windows.
    gaussian_area = (
        width * math.sqrt(math.pi) / (2.0 * math.sqrt(math.log(2.0)))
    )
    lorentzian_area = math.pi * width / 2.0
    area = mixing * lorentzian_area + (1.0 - mixing) * gaussian_area
    return profile / max(area, np.finfo(float).eps)


def _profile_width_deg(
    two_theta_deg: float,
    u: float,
    v: float,
    w: float,
    minimum_width: float,
) -> float:
    theta = math.radians(float(two_theta_deg) / 2.0)
    tangent = math.tan(theta)
    squared = u * tangent * tangent + v * tangent + w
    return max(float(minimum_width), math.sqrt(max(squared, 1e-12)))


def _cell_parameter_names(crystal_system: str) -> list[str]:
    system = str(crystal_system or "Triclinic")
    if system == "Cubic":
        return ["a"]
    if system in {"Tetragonal", "Hexagonal"}:
        return ["a", "c"]
    if system == "Orthorhombic":
        return ["a", "b", "c"]
    if system == "Rhombohedral":
        return ["a", "alpha"]
    if system == "Monoclinic":
        return ["a", "b", "c", "beta"]
    return ["a", "b", "c", "alpha", "beta", "gamma"]


def _cell_from_values(
    names: list[str],
    values: np.ndarray,
    crystal_system: str,
) -> dict:
    p = {name: float(value) for name, value in zip(names, values)}
    system = str(crystal_system or "Triclinic")
    if system == "Cubic":
        a = p["a"]
        return {"a": a, "b": a, "c": a, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    if system == "Tetragonal":
        return {"a": p["a"], "b": p["a"], "c": p["c"], "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    if system == "Hexagonal":
        return {"a": p["a"], "b": p["a"], "c": p["c"], "alpha": 90.0, "beta": 90.0, "gamma": 120.0}
    if system == "Orthorhombic":
        return {"a": p["a"], "b": p["b"], "c": p["c"], "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    if system == "Rhombohedral":
        a, alpha = p["a"], p["alpha"]
        return {"a": a, "b": a, "c": a, "alpha": alpha, "beta": alpha, "gamma": alpha}
    if system == "Monoclinic":
        return {"a": p["a"], "b": p["b"], "c": p["c"], "alpha": 90.0, "beta": p["beta"], "gamma": 90.0}
    return {name: p[name] for name in ("a", "b", "c", "alpha", "beta", "gamma")}


def _cell_values(cell: dict, crystal_system: str) -> list[float]:
    return [float(cell[name]) for name in _cell_parameter_names(crystal_system)]


def _two_theta(cell: dict, hkl: tuple[int, int, int], wavelength: float) -> float | None:
    try:
        spacing = d_spacing(cell, hkl)
    except (ValueError, np.linalg.LinAlgError):
        return None
    argument = float(wavelength) / (2.0 * spacing)
    if not 0.0 < argument < 1.0:
        return None
    return float(2.0 * math.degrees(math.asin(argument)))


def _reciprocal_angle_cosine(
    cell: dict,
    hkl: tuple[int, int, int],
    axis: tuple[int, int, int],
) -> float:
    reciprocal = np.linalg.inv(direct_metric_tensor(cell))
    a = np.asarray(hkl, dtype=float)
    b = np.asarray(axis, dtype=float)
    denominator = math.sqrt(float(a @ reciprocal @ a) * float(b @ reciprocal @ b))
    if denominator <= 0:
        return 0.0
    return float(np.clip((a @ reciprocal @ b) / denominator, -1.0, 1.0))


def _march_dollase_factor(
    cell: dict,
    hkl: tuple[int, int, int],
    axis: tuple[int, int, int] | None,
    r_value: float,
) -> float:
    if axis is None or axis == (0, 0, 0):
        return 1.0
    r = max(0.2, min(5.0, float(r_value)))
    cosine = _reciprocal_angle_cosine(cell, hkl, axis)
    cosine_squared = cosine * cosine
    sine_squared = max(0.0, 1.0 - cosine_squared)
    denominator = r * r * cosine_squared + sine_squared / r
    return float(max(denominator, 1e-12) ** -1.5)


def _prepare_phase_templates(
    specs: list[RietveldPhaseSpec],
    wavelength: float,
    two_theta_min: float,
    two_theta_max: float,
    cutoff: float,
    use_gpu: bool = False,
) -> list[dict]:
    phases = []
    for phase_index, spec in enumerate(specs, start=1):
        if not spec.included:
            continue
        structure = deepcopy(spec.structure)
        atoms = structure.get("atoms", [])
        if not atoms:
            raise RietveldError(
                f"{spec.name or structure.get('data_name', 'Phase')} has no atomic sites. "
                "Rietveld structure intensities require a CIF with atom coordinates."
            )
        if spec.freeze_structure_factors and spec.refine_biso:
            raise RietveldError(
                f"{spec.name or structure.get('data_name', 'Phase')}: Biso cannot be "
                "refined while structure factors are frozen."
            )
        peaks = calculate_powder_pattern(
            structure,
            wavelength_angstrom=wavelength,
            two_theta_min=two_theta_min,
            two_theta_max=two_theta_max,
            intensity_cutoff_percent=cutoff,
            use_gpu=use_gpu,
        )
        if len(peaks) < 2:
            continue
        reflections = []
        for index, peak in enumerate(peaks, start=1):
            equivalent = peak.get("equivalent_hkls") or [peak.get("hkl")]
            equivalent_hkls = [tuple(int(v) for v in hkl) for hkl in equivalent if hkl]
            equivalent_multiplicities = [
                int(value)
                for value in (
                    peak.get("equivalent_multiplicities")
                    or [1] * len(equivalent_hkls)
                )
            ]
            if len(equivalent_multiplicities) != len(equivalent_hkls):
                equivalent_multiplicities = [1] * len(equivalent_hkls)
            primary = tuple(int(v) for v in peak["hkl"])
            reflections.append(
                {
                    "reflection_index": index,
                    "hkl": primary,
                    "hkl_label": peak.get("hkl_label", ""),
                    "equivalent_hkls": equivalent_hkls,
                    "equivalent_multiplicities": equivalent_multiplicities,
                    "initial_two_theta_deg": float(peak["two_theta"]),
                    "initial_intensity": float(peak["intensity"]),
                }
            )
        phases.append(
            {
                "phase_index": phase_index,
                "name": spec.name or structure.get("data_name", f"Phase {phase_index}"),
                "formula": structure.get("formula", ""),
                "structure": structure,
                "initial_cell": deepcopy(structure["cell"]),
                "crystal_system": structure.get("crystal_system", "Triclinic"),
                "refine_cell": bool(spec.refine_cell),
                "refine_biso": bool(spec.refine_biso),
                "preferred_hkl": spec.preferred_orientation_hkl,
                "refine_preferred": bool(spec.refine_preferred_orientation),
                "freeze_structure_factors": bool(spec.freeze_structure_factors),
                "structure_override": bool(spec.allow_flagged_structure),
                "initial_preferred_r": float(spec.initial_preferred_orientation_r),
                "reflections": reflections,
            }
        )
    if not phases:
        raise RietveldError("No included CIF phase produced at least two calculated reflections.")
    return phases


def _parameterization(
    phases: list[dict],
    *,
    refine_zero_shift: bool,
    refine_profile: bool,
    refine_eta: bool,
    initial_zero_shift: float,
    initial_u: float,
    initial_v: float,
    initial_w: float,
    initial_eta: float,
    profile_model: str,
    refine_lorentzian_width: bool,
    refine_asymmetry: bool,
    initial_x: float,
    initial_y: float,
    initial_asymmetry: float,
    cell_tolerance_percent: float,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict]:
    names: list[str] = []
    initial: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    mapping = {"cells": {}, "biso": {}, "preferred": {}}

    if refine_zero_shift:
        names.append("zero_shift_deg")
        initial.append(float(np.clip(initial_zero_shift, -1.0, 1.0)))
        lower.append(-1.0)
        upper.append(1.0)
    if refine_profile:
        for name, value, lo, hi in (
            ("caglioti_u", max(float(initial_u), 0.0), 0.0, 3.0),
            ("caglioti_v", float(initial_v), -2.0, 2.0),
            ("caglioti_w", max(float(initial_w), 1e-8), 1e-10, 3.0),
        ):
            names.append(name)
            initial.append(value)
            lower.append(lo)
            upper.append(hi)
    if refine_eta and profile_model != "TCH pseudo-Voigt":
        names.append("eta")
        initial.append(float(np.clip(initial_eta, 0.0, 1.0)))
        lower.append(0.0)
        upper.append(1.0)
    if refine_lorentzian_width and profile_model in {"TCH pseudo-Voigt", "Lorentzian X-Y"}:
        for name, value in (("lorentzian_x", initial_x), ("lorentzian_y", initial_y)):
            names.append(name)
            initial.append(max(float(value), 0.0))
            lower.append(0.0)
            upper.append(2.0)
    if refine_asymmetry and profile_model == "Split pseudo-Voigt":
        names.append("axial_asymmetry")
        initial.append(float(np.clip(initial_asymmetry, -0.75, 0.75)))
        lower.append(-0.75)
        upper.append(0.75)

    relative = max(0.001, float(cell_tolerance_percent) / 100.0)
    for phase in phases:
        index = phase["phase_index"]
        if phase["refine_cell"]:
            cell_names = _cell_parameter_names(phase["crystal_system"])
            values = _cell_values(phase["initial_cell"], phase["crystal_system"])
            indices = []
            for cell_name, value in zip(cell_names, values):
                indices.append(len(names))
                names.append(f"phase{index}_{cell_name}")
                initial.append(value)
                if cell_name in {"a", "b", "c"}:
                    lower.append(max(0.05, value * (1.0 - relative)))
                    upper.append(value * (1.0 + relative))
                else:
                    angle_span = max(0.2, 180.0 * relative)
                    lower.append(max(35.0, value - angle_span))
                    upper.append(min(145.0, value + angle_span))
            mapping["cells"][index] = {"indices": indices, "names": cell_names}
        if phase["refine_biso"]:
            mapping["biso"][index] = len(names)
            names.append(f"phase{index}_delta_biso")
            initial.append(0.0)
            lower.append(-5.0)
            upper.append(20.0)
        if phase["refine_preferred"] and phase["preferred_hkl"]:
            mapping["preferred"][index] = len(names)
            names.append(f"phase{index}_march_r")
            initial.append(float(np.clip(phase["initial_preferred_r"], 0.2, 5.0)))
            lower.append(0.2)
            upper.append(5.0)

    return (
        names,
        np.asarray(initial, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        mapping,
    )


def _unpack(
    values: np.ndarray,
    names: list[str],
    phases: list[dict],
    mapping: dict,
    fixed: dict,
) -> dict:
    values_by_name = {name: float(value) for name, value in zip(names, values)}
    state = {
        "zero_shift_deg": values_by_name.get("zero_shift_deg", fixed["zero_shift_deg"]),
        "u": values_by_name.get("caglioti_u", fixed["u"]),
        "v": values_by_name.get("caglioti_v", fixed["v"]),
        "w": values_by_name.get("caglioti_w", fixed["w"]),
        "eta": values_by_name.get("eta", fixed["eta"]),
        "profile_model": fixed["profile_model"],
        "x": values_by_name.get("lorentzian_x", fixed["x"]),
        "y": values_by_name.get("lorentzian_y", fixed["y"]),
        "axial_asymmetry": values_by_name.get(
            "axial_asymmetry", fixed["axial_asymmetry"]
        ),
        "axial_sh_over_l": float(fixed.get("axial_sh_over_l", 0.0)),
        "cells": {},
        "delta_biso": {},
        "preferred_r": {},
    }
    for phase in phases:
        index = phase["phase_index"]
        cell_info = mapping["cells"].get(index)
        if cell_info:
            state["cells"][index] = _cell_from_values(
                cell_info["names"],
                values[np.asarray(cell_info["indices"], dtype=int)],
                phase["crystal_system"],
            )
        else:
            state["cells"][index] = deepcopy(phase["initial_cell"])
        biso_index = mapping["biso"].get(index)
        state["delta_biso"][index] = float(values[biso_index]) if biso_index is not None else 0.0
        preferred_index = mapping["preferred"].get(index)
        state["preferred_r"][index] = (
            float(values[preferred_index])
            if preferred_index is not None
            else float(phase["initial_preferred_r"])
        )
    return state


def _phase_basis(
    x: np.ndarray,
    phase: dict,
    state: dict,
    wavelength: float,
    k_alpha2_enabled: bool,
    k_alpha2_wavelength: float,
    k_alpha2_ratio: float,
    use_gpu: bool = False,
) -> tuple[np.ndarray, list[dict], list[str]]:
    index = phase["phase_index"]
    cell = state["cells"][index]
    delta_biso = state["delta_biso"][index]
    preferred_r = state["preferred_r"][index]
    atoms = []
    if not phase.get("freeze_structure_factors"):
        for atom in phase["structure"].get("atoms", []):
            copied = dict(atom)
            copied["b_iso"] = max(0.0, float(copied.get("b_iso", 0.0)) + delta_biso)
            atoms.append(copied)

    step = float(np.median(np.diff(x)))
    minimum_width = max(step * 1.15, 0.003)
    basis = np.zeros_like(x, dtype=float)
    rows = []
    warnings = []

    intensity_lookup: dict[tuple[int, int, int], float] = {}
    if atoms and use_gpu:
        batch_hkls = []
        batch_d = []
        batch_tt = []
        seen_hkls = set()
        for reflection in phase["reflections"]:
            for equivalent in reflection["equivalent_hkls"]:
                equivalent = tuple(int(value) for value in equivalent)
                if equivalent in seen_hkls:
                    continue
                try:
                    d_value = d_spacing(cell, equivalent)
                    tt = _two_theta(cell, equivalent, wavelength)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if tt is None:
                    continue
                seen_hkls.add(equivalent)
                batch_hkls.append(equivalent)
                batch_d.append(d_value)
                batch_tt.append(tt)
        if batch_hkls:
            intensities, acceleration = _structure_factor_intensities_batch(
                atoms,
                batch_hkls,
                batch_d,
                batch_tt,
                use_gpu=use_gpu,
                scattering_factors=phase["structure"].get(
                    "xray_scattering_factors"
                ),
            )
            intensity_lookup = {
                hkl: float(value) for hkl, value in zip(batch_hkls, intensities)
            }
            if use_gpu:
                if acceleration.get("used"):
                    warnings.append(
                        "CUDA structure-factor batching used on "
                        + str(acceleration.get("device_name") or "an NVIDIA GPU")
                        + ". The nonlinear SciPy optimizer remains on CPU."
                    )
                elif acceleration.get("reason"):
                    warnings.append(str(acceleration["reason"]))

    for reflection in phase["reflections"]:
        hkl = reflection["hkl"]
        center = _two_theta(cell, hkl, wavelength)
        if center is None or center < x[0] - 1.0 or center > x[-1] + 1.0:
            continue
        center += state["zero_shift_deg"]
        try:
            spacing = d_spacing(cell, hkl)
        except ValueError:
            continue
        if phase.get("freeze_structure_factors"):
            intensity = float(reflection["initial_intensity"])
        else:
            intensity = 0.0
            for equivalent, multiplicity in zip(
                reflection["equivalent_hkls"],
                reflection["equivalent_multiplicities"],
            ):
                equivalent_key = tuple(int(value) for value in equivalent)
                if use_gpu:
                    intensity += float(multiplicity) * intensity_lookup.get(
                        equivalent_key, 0.0
                    )
                    continue
                try:
                    d_value = d_spacing(cell, equivalent_key)
                    tt = _two_theta(cell, equivalent_key, wavelength)
                    if tt is None:
                        continue
                    intensity += float(multiplicity) * _structure_factor_intensity(
                        atoms,
                        equivalent_key,
                        d_value,
                        tt,
                        wavelength,
                        phase["structure"].get("xray_scattering_factors"),
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
        intensity *= _march_dollase_factor(
            cell,
            hkl,
            phase["preferred_hkl"],
            preferred_r,
        )
        if not np.isfinite(intensity) or intensity <= 1e-14:
            continue
        characteristics = profile_characteristics(
            center,
            model=state["profile_model"],
            u=state["u"],
            v=state["v"],
            w=state["w"],
            eta=state["eta"],
            x=state["x"],
            y=state["y"],
            axial_asymmetry=state["axial_asymmetry"],
            axial_sh_over_l=state["axial_sh_over_l"],
            minimum_width=minimum_width,
        )
        width = characteristics.fwhm_deg
        half_window = max(10.0 * max(characteristics.left_fwhm_deg, characteristics.right_fwhm_deg), 5.0 * step)
        left = int(np.searchsorted(x, center - half_window, side="left"))
        right = int(np.searchsorted(x, center + half_window, side="right"))
        if right - left >= 3:
            basis[left:right] += intensity * profile_unit_area(
                x[left:right],
                center,
                model=state["profile_model"],
                u=state["u"],
                v=state["v"],
                w=state["w"],
                eta=state["eta"],
                x=state["x"],
                y=state["y"],
                axial_asymmetry=state["axial_asymmetry"],
                axial_sh_over_l=state["axial_sh_over_l"],
                minimum_width=minimum_width,
            )
        if k_alpha2_enabled and k_alpha2_ratio > 0:
            center2 = _two_theta(cell, hkl, k_alpha2_wavelength)
            if center2 is not None:
                center2 += state["zero_shift_deg"]
                characteristics2 = profile_characteristics(
                    center2,
                    model=state["profile_model"],
                    u=state["u"],
                    v=state["v"],
                    w=state["w"],
                    eta=state["eta"],
                    x=state["x"],
                    y=state["y"],
                    axial_asymmetry=state["axial_asymmetry"],
                    axial_sh_over_l=state["axial_sh_over_l"],
                    minimum_width=minimum_width,
                )
                width2 = characteristics2.fwhm_deg
                window2 = max(10.0 * max(characteristics2.left_fwhm_deg, characteristics2.right_fwhm_deg), 5.0 * step)
                left2 = int(np.searchsorted(x, center2 - window2, side="left"))
                right2 = int(np.searchsorted(x, center2 + window2, side="right"))
                if right2 - left2 >= 3:
                    basis[left2:right2] += (
                        intensity
                        * float(k_alpha2_ratio)
                        * profile_unit_area(
                            x[left2:right2],
                            center2,
                            model=state["profile_model"],
                            u=state["u"],
                            v=state["v"],
                            w=state["w"],
                            eta=state["eta"],
                            x=state["x"],
                            y=state["y"],
                            axial_asymmetry=state["axial_asymmetry"],
                            axial_sh_over_l=state["axial_sh_over_l"],
                            minimum_width=minimum_width,
                        )
                    )
        rows.append(
            {
                "phase_index": index,
                "phase_name": phase["name"],
                "hkl_label": reflection["hkl_label"],
                "hkl": list(hkl),
                "two_theta_deg": float(center),
                "d_spacing": float(spacing),
                "fwhm_deg": float(width),
                "gaussian_fwhm_deg": float(characteristics.gaussian_fwhm_deg),
                "lorentzian_fwhm_deg": float(characteristics.lorentzian_fwhm_deg),
                "profile_eta": float(characteristics.eta),
                "left_fwhm_deg": float(characteristics.left_fwhm_deg),
                "right_fwhm_deg": float(characteristics.right_fwhm_deg),
                "structure_intensity": float(intensity),
                "preferred_orientation_factor": float(
                    _march_dollase_factor(cell, hkl, phase["preferred_hkl"], preferred_r)
                ),
            }
        )

    area = float(np.trapezoid(basis, x))
    if area <= 1e-20:
        warnings.append(f"{phase['name']}: calculated profile has negligible area.")
        return basis, rows, warnings
    return basis / area, rows, warnings


def _solve_linear(
    phase_bases: list[np.ndarray],
    background_basis: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack([*phase_bases, background_basis])
    phase_count = len(phase_bases)
    lower = np.concatenate([
        np.zeros(phase_count, dtype=float),
        np.zeros(background_basis.shape[1], dtype=float),
    ])
    upper = np.full(design.shape[1], np.inf)
    solution = lsq_linear(
        design * weights[:, None],
        y * weights,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
        max_iter=2500,
    )
    if not solution.success:
        raise RietveldError(f"Linear scale/background solution failed: {solution.message}")
    scales = np.asarray(solution.x[:phase_count], dtype=float)
    background_coefficients = np.asarray(solution.x[phase_count:], dtype=float)
    phase_profiles = np.column_stack(phase_bases) * scales[None, :]
    calculated = design @ solution.x
    weighted_design = design * weights[:, None]
    weighted_residual = (y - calculated) * weights
    dof = max(1, len(y) - design.shape[1])
    try:
        linear_covariance = np.linalg.pinv(weighted_design.T @ weighted_design)
        linear_covariance *= float(np.sum(weighted_residual ** 2) / dof)
    except np.linalg.LinAlgError:
        linear_covariance = np.full((design.shape[1], design.shape[1]), np.nan)
    return scales, background_coefficients, phase_profiles, calculated, linear_covariance


def _generated_pattern_normalization_diagnostics(
    solution: dict,
    x: np.ndarray,
    observed: np.ndarray,
    refinement_sqrt_weights: np.ndarray,
    *,
    method: str,
    background_order: int,
) -> dict:
    """Describe the CIF-to-observed intensity scaling used before refinement.

    CIF files contain structural information, not measured detector counts. The
    calculated phase profiles are first normalized to unit integrated area and
    then jointly fitted to the observed intensity scale with non-negative phase
    coefficients and a polynomial background. The observed pattern is never
    rescaled or overwritten.
    """

    x = np.asarray(x, dtype=float)
    observed = np.asarray(observed, dtype=float)
    calculated = np.asarray(solution.get("calculated", []), dtype=float)
    background = np.asarray(solution.get("background", []), dtype=float)
    weights = np.asarray(refinement_sqrt_weights, dtype=float)
    scales = np.asarray(solution.get("scales", []), dtype=float)
    phase_profiles = np.asarray(solution.get("phase_profiles", []), dtype=float)

    residual = observed - calculated
    weighted_denominator = float(np.sum((observed * weights) ** 2))
    weighted_numerator = float(np.sum((residual * weights) ** 2))
    initial_rwp = (
        100.0 * math.sqrt(weighted_numerator / weighted_denominator)
        if weighted_denominator > 0.0
        else None
    )
    absolute_denominator = float(np.sum(np.abs(observed)))
    initial_rp = (
        100.0 * float(np.sum(np.abs(residual))) / absolute_denominator
        if absolute_denominator > 0.0
        else None
    )
    correlation = None
    if observed.size > 1 and calculated.size == observed.size:
        observed_std = float(np.std(observed))
        calculated_std = float(np.std(calculated))
        if observed_std > 0.0 and calculated_std > 0.0:
            correlation = float(np.corrcoef(observed, calculated)[0, 1])

    scale_sum = float(np.sum(scales))
    fractions = (
        (100.0 * scales / scale_sum).tolist()
        if scale_sum > 0.0
        else [0.0 for _ in scales]
    )
    phase_maxima = []
    if phase_profiles.ndim == 2 and phase_profiles.shape[0] == observed.size:
        phase_maxima = [
            float(np.max(phase_profiles[:, index]))
            for index in range(phase_profiles.shape[1])
        ]

    return {
        "method": str(method),
        "scientific_role": "CIF-generated profile intensity initialization",
        "cif_contains_measured_pattern": False,
        "observed_pattern_modified": False,
        "phase_basis_normalization": "unit integrated area before scale fitting",
        "phase_scales_fitted_jointly": True,
        "phase_scale_constraint": "non-negative",
        "background_fitted_simultaneously": True,
        "background_order": int(background_order),
        "two_theta_min_deg": float(x[0]),
        "two_theta_max_deg": float(x[-1]),
        "point_count": int(observed.size),
        "initial_phase_scale_factors": scales.tolist(),
        "initial_pattern_scale_fractions_percent": fractions,
        "initial_phase_profile_maxima_in_observed_units": phase_maxima,
        "initial_background_min": float(np.min(background)) if background.size else None,
        "initial_background_max": float(np.max(background)) if background.size else None,
        "initial_calculated_min": float(np.min(calculated)) if calculated.size else None,
        "initial_calculated_max": float(np.max(calculated)) if calculated.size else None,
        "initial_rwp_percent": initial_rwp,
        "initial_rp_percent": initial_rp,
        "initial_profile_correlation": correlation,
        "note": (
            "The generated CIF pattern is not normalized by forcing both maxima to 100. "
            "Phase profiles are converted to the observed intensity units by a joint "
            "least-squares scale/background fit. Final phase scales are re-solved at "
            "every nonlinear refinement step."
        ),
    }


def _correlation_diagnostics(result, parameter_names: list[str], residuals: np.ndarray) -> dict:
    if result is None or not parameter_names or not getattr(result, "jac", None).size:
        return {"condition_number": None, "maximum_absolute_correlation": None, "strong_pairs": []}
    jacobian = np.asarray(result.jac, dtype=float)
    normal = jacobian.T @ jacobian
    condition = float(np.linalg.cond(normal)) if normal.size else None
    try:
        covariance = np.linalg.pinv(normal)
        covariance *= float(np.sum(residuals ** 2) / max(1, len(residuals) - len(parameter_names)))
        standard = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = np.outer(standard, standard)
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0,
        )
        np.fill_diagonal(correlation, 0.0)
        maximum = float(np.max(np.abs(correlation))) if correlation.size else None
        pairs = []
        for i in range(len(parameter_names)):
            for j in range(i + 1, len(parameter_names)):
                value = float(correlation[i, j])
                if abs(value) >= 0.95:
                    pairs.append({"parameter_1": parameter_names[i], "parameter_2": parameter_names[j], "correlation": value})
        return {"condition_number": condition, "maximum_absolute_correlation": maximum, "strong_pairs": pairs}
    except np.linalg.LinAlgError:
        return {"condition_number": condition, "maximum_absolute_correlation": None, "strong_pairs": []}



def _select_optimization_indices(
    x: np.ndarray,
    y: np.ndarray,
    fit_pool: np.ndarray,
    maximum_points: int,
) -> tuple[np.ndarray, dict]:
    """Select nonlinear-optimization points without throwing away narrow Bragg peaks.

    Uniform thinning alone is unsafe for powder diffraction because a narrow reflection
    can fall almost completely between retained samples.  Keep broad uniform coverage,
    then spend the remaining point budget on high-intensity and high-gradient regions.
    The final full-pattern linear solve and reported statistics still use every point.
    """
    pool = np.asarray(fit_pool, dtype=int)
    budget = int(maximum_points)
    if pool.size <= budget:
        return pool.copy(), {
            "method": "all fit points",
            "pool_point_count": int(pool.size),
            "selected_point_count": int(pool.size),
            "uniform_point_count": int(pool.size),
            "feature_point_count": 0,
        }

    x_pool = np.asarray(x, dtype=float)[pool]
    y_pool = np.asarray(y, dtype=float)[pool]

    # Reserve enough points for global baseline/background coverage while ensuring
    # the nonlinear optimizer sees peak tops and flanks.
    uniform_budget = int(np.clip(round(0.45 * budget), 96, budget - 1))
    uniform_positions = np.linspace(0, pool.size - 1, uniform_budget, dtype=int)
    uniform_indices = pool[uniform_positions]

    baseline = float(np.percentile(y_pool, 5.0))
    signal = np.maximum(y_pool - baseline, 0.0)
    signal_scale = max(float(np.percentile(signal, 99.0)), np.finfo(float).eps)
    signal_score = np.clip(signal / signal_scale, 0.0, 4.0)

    # Index-space gradient is intentional: it remains stable even if the 2theta grid
    # contains tiny local spacing irregularities.
    gradient = np.abs(np.gradient(y_pool))
    gradient_scale = max(float(np.percentile(gradient, 99.0)), np.finfo(float).eps)
    gradient_score = np.clip(gradient / gradient_scale, 0.0, 4.0)

    # Peak centers are intensity-rich; peak flanks are gradient-rich.  Combining both
    # avoids the common failure where a uniformly thinned pattern fits background but
    # misses narrow reflections.
    feature_score = 0.68 * signal_score + 0.32 * gradient_score
    already = np.zeros(pool.size, dtype=bool)
    already[uniform_positions] = True
    candidate_positions = np.flatnonzero(~already)
    feature_budget = min(budget - uniform_indices.size, candidate_positions.size)

    if feature_budget > 0:
        candidate_scores = feature_score[candidate_positions]
        if feature_budget < candidate_positions.size:
            chosen_local = np.argpartition(candidate_scores, -feature_budget)[-feature_budget:]
            feature_positions = candidate_positions[chosen_local]
        else:
            feature_positions = candidate_positions
        feature_indices = pool[feature_positions]
    else:
        feature_indices = np.asarray([], dtype=int)

    selected = np.unique(np.concatenate([uniform_indices, feature_indices]))

    # Rounding/duplicate protection: fill any remaining budget with evenly distributed
    # unused points so the requested point count is honored as closely as possible.
    if selected.size < budget:
        remaining = np.setdiff1d(pool, selected, assume_unique=True)
        need = min(budget - selected.size, remaining.size)
        if need > 0:
            fill_positions = np.linspace(0, remaining.size - 1, need, dtype=int)
            selected = np.unique(np.concatenate([selected, remaining[fill_positions]]))

    selected.sort()
    return selected, {
        "method": "peak-aware hybrid thinning",
        "pool_point_count": int(pool.size),
        "selected_point_count": int(selected.size),
        "uniform_point_count": int(uniform_indices.size),
        "feature_point_count": int(feature_indices.size),
        "uniform_fraction": float(uniform_indices.size / max(1, selected.size)),
        "note": (
            "Nonlinear optimization retains uniform coverage plus high-intensity/high-gradient "
            "Bragg-peak regions. Final profile/statistics are evaluated on the full selected range."
        ),
    }


def _rietveld_plot_payload(
    x: np.ndarray,
    observed: np.ndarray,
    calculated: np.ndarray,
    background: np.ndarray,
    residual: np.ndarray,
) -> dict:
    """Return non-destructive display aids for a conventional Rietveld plot.

    ``difference_y`` remains the true observed-minus-calculated residual.  This helper
    adds a vertically offset copy for plotting, so the difference trace does not sit on
    top of the diffraction pattern.  It also supplies full and robust y-range hints;
    consumers may ignore them without changing any refinement result.
    """
    x = np.asarray(x, dtype=float)
    observed = np.asarray(observed, dtype=float)
    calculated = np.asarray(calculated, dtype=float)
    background = np.asarray(background, dtype=float)
    residual = np.asarray(residual, dtype=float)

    finite = (
        np.isfinite(x)
        & np.isfinite(observed)
        & np.isfinite(calculated)
        & np.isfinite(background)
        & np.isfinite(residual)
    )
    if not np.any(finite):
        return {
            "difference_offset": 0.0,
            "difference_plot_y": residual.tolist(),
            "full_y_range": None,
            "robust_y_range": None,
        }

    obs_f = observed[finite]
    calc_f = calculated[finite]
    bkg_f = background[finite]
    res_f = residual[finite]

    curve_low = float(min(np.min(obs_f), np.min(calc_f), np.min(bkg_f)))
    curve_high = float(max(np.max(obs_f), np.max(calc_f), np.max(bkg_f)))
    full_span = max(curve_high - curve_low, np.finfo(float).eps)

    robust_low = float(min(
        np.percentile(obs_f, 0.5),
        np.percentile(calc_f, 0.5),
        np.percentile(bkg_f, 0.5),
    ))
    robust_high = float(max(
        np.percentile(obs_f, 99.7),
        np.percentile(calc_f, 99.7),
        np.percentile(bkg_f, 99.7),
    ))
    robust_span = max(robust_high - robust_low, 0.10 * full_span, np.finfo(float).eps)

    residual_half_range = float(np.percentile(np.abs(res_f), 99.0))
    residual_half_range = max(residual_half_range, 0.04 * robust_span)
    difference_offset = robust_low - 0.16 * robust_span - residual_half_range
    difference_plot = residual + difference_offset

    full_y_min = float(min(np.min(difference_plot[finite]), curve_low) - 0.03 * full_span)
    full_y_max = float(curve_high + 0.05 * full_span)
    robust_y_min = float(min(np.percentile(difference_plot[finite], 0.3), robust_low - 0.03 * robust_span))
    robust_y_max = float(robust_high + 0.08 * robust_span)

    return {
        "difference_offset": float(difference_offset),
        "difference_plot_y": difference_plot.tolist(),
        "full_y_range": [full_y_min, full_y_max],
        "robust_y_range": [robust_y_min, robust_y_max],
        "recommended_difference_label": "Observed - calculated (offset)",
        "recommended_legend_orientation": "horizontal",
        "note": (
            "Use difference_plot_y for display only. Keep difference_y for residual analysis. "
            "robust_y_range is an optional visualization hint and may clip isolated outliers."
        ),
    }


def simulate_rietveld_pattern(
    x: np.ndarray,
    phase_specs: list[RietveldPhaseSpec],
    scales: list[float],
    *,
    wavelength_angstrom: float = 1.5406,
    background_coefficients: list[float] | None = None,
    zero_shift_deg: float = 0.0,
    caglioti_u: float = 0.005,
    caglioti_v: float = 0.0,
    caglioti_w: float = 0.02,
    eta: float = 0.5,
    profile_model: str = "Pseudo-Voigt U-V-W",
    lorentzian_x: float = 0.0,
    lorentzian_y: float = 0.0,
    axial_asymmetry: float = 0.0,
    intensity_cutoff_percent: float = 0.2,
    k_alpha2_enabled: bool = False,
    k_alpha2_wavelength_angstrom: float = 1.54439,
    k_alpha2_ratio: float = 0.5,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    phases = _prepare_phase_templates(
        phase_specs,
        wavelength_angstrom,
        float(x[0]),
        float(x[-1]),
        intensity_cutoff_percent,
    )
    if len(scales) != len(phases):
        raise RietveldError("Scale count must equal the included phase count.")
    state = {
        "zero_shift_deg": float(zero_shift_deg),
        "u": float(caglioti_u),
        "v": float(caglioti_v),
        "w": float(caglioti_w),
        "eta": float(eta),
        "profile_model": str(profile_model),
        "x": max(float(lorentzian_x), 0.0),
        "y": max(float(lorentzian_y), 0.0),
        "axial_asymmetry": float(np.clip(axial_asymmetry, -0.75, 0.75)),
        "cells": {phase["phase_index"]: deepcopy(phase["initial_cell"]) for phase in phases},
        "delta_biso": {phase["phase_index"]: 0.0 for phase in phases},
        "preferred_r": {phase["phase_index"]: phase["initial_preferred_r"] for phase in phases},
    }
    calculated = np.zeros_like(x)
    for phase, scale in zip(phases, scales):
        basis, _, _ = _phase_basis(
            x,
            phase,
            state,
            wavelength_angstrom,
            k_alpha2_enabled,
            k_alpha2_wavelength_angstrom,
            k_alpha2_ratio,
        )
        calculated += float(scale) * basis
    coefficients = list(background_coefficients or [0.0])
    calculated += _background_basis(x, len(coefficients) - 1) @ np.asarray(coefficients, dtype=float)
    return calculated


def refine_rietveld(
    x: np.ndarray,
    y: np.ndarray,
    phase_specs: list[RietveldPhaseSpec],
    *,
    wavelength_angstrom: float = 1.5406,
    two_theta_min: float | None = None,
    two_theta_max: float | None = None,
    intensity_cutoff_percent: float = 0.2,
    background_order: int = 3,
    weighting: str = "Poisson-like",
    refine_zero_shift: bool = True,
    initial_zero_shift: float = 0.0,
    refine_profile: bool = True,
    refine_eta: bool = False,
    initial_u: float = 0.005,
    initial_v: float = 0.0,
    initial_w: float = 0.02,
    initial_eta: float = 0.5,
    profile_model: str = "Pseudo-Voigt U-V-W",
    refine_lorentzian_width: bool = False,
    refine_asymmetry: bool = False,
    initial_x: float = 0.0,
    initial_y: float = 0.0,
    initial_asymmetry: float = 0.0,
    initial_axial_sh_over_l: float = 0.0,
    cell_tolerance_percent: float = 3.0,
    k_alpha2_enabled: bool = False,
    k_alpha2_wavelength_angstrom: float = 1.54439,
    k_alpha2_ratio: float = 0.5,
    maximum_nonlinear_evaluations: int = 120,
    maximum_optimization_points: int = 1800,
    validation_stride: int = 0,
    robust_loss: str = "soft_l1",
    use_staged_refinement: bool = False,
    observed_sigma: np.ndarray | None = None,
    count_reference: np.ndarray | None = None,
    intensity_scale_factor: float = 1.0,
    intensity_provenance: str = "raw_counts",
    statistics_note: str | None = None,
    generated_pattern_scaling: str = GENERATED_PATTERN_SCALING[0],
    instrument_profile: dict | None = None,
    allow_profile_extrapolation: bool = False,
    use_gpu: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    started = time.perf_counter()
    acceleration = (
        dict(cuda_status())
        if use_gpu
        else {
            "available": False,
            "backend": "CPU",
            "device_name": "",
            "reason": "GPU acceleration is disabled for this run.",
        }
    )
    acceleration.update(
        requested=bool(use_gpu),
        calculation_backend=(
            "CUDA (CuPy)" if use_gpu and acceleration.get("available") else "CPU"
        ),
        optimizer_backend="SciPy CPU",
        publication_renderer_backend="CPU deterministic",
    )
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise RietveldError("X and Y must be one-dimensional arrays of equal length.")
    if len(x) < 80:
        raise RietveldError("At least eighty measured points are required.")
    if wavelength_angstrom <= 0:
        raise RietveldError("Wavelength must be positive.")
    if instrument_profile:
        try:
            validate_profile_compatibility(
                instrument_profile,
                wavelength_angstrom=wavelength_angstrom,
            )
        except ValueError as exc:
            raise RietveldError(str(exc)) from exc
    if weighting not in RIETVELD_WEIGHTING:
        raise RietveldError(f"Unsupported weighting: {weighting}")
    if robust_loss not in RIETVELD_LOSSES:
        raise RietveldError(f"Unsupported robust loss: {robust_loss}")
    if generated_pattern_scaling not in GENERATED_PATTERN_SCALING:
        raise RietveldError(
            f"Unsupported generated-pattern scaling mode: {generated_pattern_scaling}"
        )
    if profile_model not in PROFILE_MODELS:
        raise RietveldError(f"Unsupported profile model: {profile_model}")

    if progress_callback is not None:
        progress_callback(
            0,
            1,
            (
                "CUDA ready on " + str(acceleration.get("device_name"))
                + "; validating CIF geometry."
                if use_gpu and acceleration.get("available")
                else "Validating CIF geometry (large expanded structures can take several seconds)."
            ),
        )
    validate_phase_specs_for_refinement(phase_specs)
    if cancel_check is not None and cancel_check():
        raise RietveldCancelled("Rietveld refinement was cancelled.")

    lower_range = float(x[0] if two_theta_min is None else two_theta_min)
    upper_range = float(x[-1] if two_theta_max is None else two_theta_max)
    mask = (x >= lower_range) & (x <= upper_range)
    x_full = x[mask]
    y_full = y[mask]
    sigma_full = None
    if observed_sigma is not None:
        sigma_array = np.asarray(observed_sigma, dtype=float)
        if sigma_array.shape != y.shape:
            raise RietveldError("Observed sigma must match the complete observed pattern.")
        sigma_full = sigma_array[mask]
    count_full = None
    if count_reference is not None:
        count_array = np.asarray(count_reference, dtype=float)
        if count_array.shape != y.shape:
            raise RietveldError("Count reference must match the complete observed pattern.")
        count_full = count_array[mask]
    if len(x_full) < 80:
        raise RietveldError("The selected range contains fewer than eighty data points.")
    if instrument_profile:
        try:
            calibration_coverage = validate_calibration_range(
                instrument_profile,
                np.asarray([x_full[0], x_full[-1]]),
                allow_extrapolation=allow_profile_extrapolation,
            )
        except CalibrationRangeError as exc:
            raise RietveldError(str(exc)) from exc
    else:
        calibration_coverage = None

    phases = _prepare_phase_templates(
        phase_specs,
        wavelength_angstrom,
        float(x_full[0]),
        float(x_full[-1]),
        intensity_cutoff_percent,
        use_gpu=use_gpu,
    )
    background_order = int(np.clip(background_order, 0, 6))
    maximum_optimization_points = int(np.clip(maximum_optimization_points, 300, 6000))
    try:
        full_weight_model = build_refinement_weight_model(
            y_full,
            weighting,
            observed_sigma=sigma_full,
            count_reference=count_full,
            intensity_scale_factor=intensity_scale_factor,
            provenance=intensity_provenance,
            statistics_note=statistics_note,
        )
    except ValueError as exc:
        raise RietveldError(str(exc)) from exc

    validation_stride = int(validation_stride)
    if validation_stride and validation_stride < 3:
        raise RietveldError("Validation stride must be zero (disabled) or at least three.")
    all_indices = np.arange(len(x_full), dtype=int)
    if validation_stride:
        validation_indices = np.arange(
            validation_stride // 2,
            len(x_full),
            validation_stride,
            dtype=int,
        )
        validation_indices = validation_indices[
            (validation_indices > 0) & (validation_indices < len(x_full) - 1)
        ]
        fit_pool = np.setdiff1d(all_indices, validation_indices, assume_unique=True)
    else:
        validation_indices = np.asarray([], dtype=int)
        fit_pool = all_indices
    fit_indices, optimization_sampling = _select_optimization_indices(
        x_full,
        y_full,
        fit_pool,
        maximum_optimization_points,
    )
    x_fit, y_fit = x_full[fit_indices], y_full[fit_indices]
    fit_sqrt_weights = full_weight_model.fit_sqrt_weights[fit_indices]

    fixed = {
        "zero_shift_deg": float(initial_zero_shift),
        "u": max(float(initial_u), 0.0),
        "v": float(initial_v),
        "w": max(float(initial_w), 1e-8),
        "eta": float(np.clip(initial_eta, 0.0, 1.0)),
        "profile_model": str(profile_model),
        "x": max(float(initial_x), 0.0),
        "y": max(float(initial_y), 0.0),
        "axial_asymmetry": float(np.clip(initial_asymmetry, -0.75, 0.75)),
        "axial_sh_over_l": max(float(initial_axial_sh_over_l), 0.0),
    }
    names, initial, lower, upper, mapping = _parameterization(
        phases,
        refine_zero_shift=refine_zero_shift,
        refine_profile=refine_profile,
        refine_eta=refine_eta,
        initial_zero_shift=fixed["zero_shift_deg"],
        initial_u=fixed["u"],
        initial_v=fixed["v"],
        initial_w=fixed["w"],
        initial_eta=fixed["eta"],
        profile_model=profile_model,
        refine_lorentzian_width=refine_lorentzian_width,
        refine_asymmetry=refine_asymmetry,
        initial_x=fixed["x"],
        initial_y=fixed["y"],
        initial_asymmetry=fixed["axial_asymmetry"],
        cell_tolerance_percent=cell_tolerance_percent,
    )
    calibration_prior = prepare_calibration_prior(instrument_profile, names)
    calibration_prior_size = (
        0 if calibration_prior is None else int(calibration_prior["effective_rank"])
    )
    per_stage_evaluations = max(1, int(maximum_nonlinear_evaluations) if len(initial) else 1)
    stage_groups = (
        staged_parameter_groups(names)
        if use_staged_refinement and len(initial)
        else []
    )
    optimization_stage_count = (len(stage_groups) + 1) if stage_groups else 1
    total_evaluations = per_stage_evaluations * optimization_stage_count
    evaluation_counter = 0
    current_stage_label = "Joint refinement"
    stage_history: list[dict] = []

    def solve(
        values: np.ndarray,
        x_values: np.ndarray,
        y_values: np.ndarray,
        sqrt_weights: np.ndarray,
    ) -> dict:
        state = _unpack(values, names, phases, mapping, fixed)
        phase_bases = []
        reflection_rows = []
        warnings = []
        for phase in phases:
            basis, rows, local_warnings = _phase_basis(
                x_values,
                phase,
                state,
                wavelength_angstrom,
                k_alpha2_enabled,
                k_alpha2_wavelength_angstrom,
                k_alpha2_ratio,
                use_gpu=use_gpu,
            )
            phase_bases.append(basis)
            reflection_rows.extend(rows)
            warnings.extend(local_warnings)
        background = _background_basis(x_values, background_order)
        weights = np.asarray(sqrt_weights, dtype=float)
        if weights.shape != y_values.shape:
            raise RietveldError("Refinement weights no longer match the observed pattern.")
        scale_weights = (
            weights
            if generated_pattern_scaling == GENERATED_PATTERN_SCALING[0]
            else np.ones_like(weights)
        )
        scales, background_coefficients, phase_profiles, calculated, linear_covariance = _solve_linear(
            phase_bases, background, y_values, scale_weights
        )
        return {
            "state": state,
            "scales": scales,
            "background_coefficients": background_coefficients,
            "phase_profiles": phase_profiles,
            "background": background @ background_coefficients,
            "calculated": calculated,
            "weights": weights,
            "scale_weights": scale_weights,
            "linear_covariance": linear_covariance,
            "reflections": reflection_rows,
            "warnings": warnings,
        }

    def residual_function(values: np.ndarray) -> np.ndarray:
        nonlocal evaluation_counter
        if cancel_check is not None and cancel_check():
            raise RietveldCancelled("Rietveld refinement was cancelled.")
        evaluation_counter += 1
        if progress_callback is not None:
            progress_callback(
                min(evaluation_counter, total_evaluations),
                total_evaluations,
                f"{current_stage_label}: Rietveld evaluation {evaluation_counter}/{total_evaluations}",
            )
        try:
            solution = solve(values, x_fit, y_fit, fit_sqrt_weights)
            pattern_residual = (
                (y_fit - solution["calculated"]) * solution["weights"]
            )
            prior_residual = calibration_prior_residual(values, calibration_prior)
            return np.concatenate([pattern_residual, prior_residual])
        except (RietveldError, ValueError, np.linalg.LinAlgError):
            return np.full(len(y_fit) + calibration_prior_size, 1e6, dtype=float)

    if progress_callback is not None:
        progress_callback(0, total_evaluations, "Scaling CIF-generated pattern to observed intensity units.")
    initial_normalization_solution = solve(
        initial,
        x_full,
        y_full,
        full_weight_model.fit_sqrt_weights,
    )
    generated_pattern_normalization = _generated_pattern_normalization_diagnostics(
        initial_normalization_solution,
        x_full,
        y_full,
        full_weight_model.fit_sqrt_weights,
        method=generated_pattern_scaling,
        background_order=background_order,
    )
    stage_history.append({
        "stage": 0,
        "name": "CIF-generated pattern scaling",
        "parameters": [
            "joint non-negative phase scales",
            f"background order {background_order}",
        ],
        "success": True,
        "message": (
            "Generated phase profiles were converted from relative/unit-area intensity "
            "to the observed pattern intensity units without modifying the observed data."
        ),
        "evaluations": 1,
        "initial_rwp_percent": generated_pattern_normalization.get("initial_rwp_percent"),
        "initial_phase_scale_factors": generated_pattern_normalization.get("initial_phase_scale_factors"),
    })

    optimization = None
    if len(initial):
        if stage_groups:
            refined_values = initial.copy()
            for stage_number, stage in enumerate(stage_groups, start=1):
                indices = np.asarray(stage["indices"], dtype=int)
                current_stage_label = (
                    f"Stage {stage_number}/{len(stage_groups)} — {stage['name']}"
                )

                def stage_residual(stage_values, *, _indices=indices):
                    full_values = refined_values.copy()
                    full_values[_indices] = stage_values
                    return residual_function(full_values)

                stage_result = least_squares(
                    stage_residual,
                    refined_values[indices],
                    bounds=(lower[indices], upper[indices]),
                    loss=robust_loss,
                    max_nfev=per_stage_evaluations,
                    ftol=1e-8,
                    xtol=1e-8,
                    gtol=1e-8,
                    x_scale="jac",
                )
                refined_values[indices] = stage_result.x
                stage_history.append({
                    "stage": stage_number,
                    "name": stage["name"],
                    "parameters": [names[index] for index in indices],
                    "success": bool(stage_result.success),
                    "message": str(stage_result.message),
                    "evaluations": int(stage_result.nfev),
                    "cost": float(stage_result.cost),
                })
                optimization = stage_result

            # A staged sequence is a conditioning aid, not a substitute for the final
            # simultaneous Rietveld fit.  Polish all nonlinear parameters together so
            # earlier groups can readjust after later profile/cell terms move.  This also
            # leaves a full-size Jacobian for covariance/correlation diagnostics.
            current_stage_label = (
                f"Stage {len(stage_groups) + 1}/{len(stage_groups) + 1} — Final joint polish"
            )
            optimization = least_squares(
                residual_function,
                refined_values,
                bounds=(lower, upper),
                loss=robust_loss,
                max_nfev=per_stage_evaluations,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                x_scale="jac",
            )
            refined_values = optimization.x
            stage_history.append({
                "stage": len(stage_groups) + 1,
                "name": "Final joint polish",
                "parameters": list(names),
                "success": bool(optimization.success),
                "message": str(optimization.message),
                "evaluations": int(optimization.nfev),
                "cost": float(optimization.cost),
            })
        else:
            optimization = least_squares(
                residual_function,
                initial,
                bounds=(lower, upper),
                loss=robust_loss,
                max_nfev=per_stage_evaluations,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                x_scale="jac",
            )
            refined_values = optimization.x
    else:
        refined_values = initial
        residual_function(refined_values)

    if cancel_check is not None and cancel_check():
        raise RietveldCancelled("Rietveld refinement was cancelled.")
    cross_validation = {
        "enabled": bool(validation_indices.size),
        "stride": validation_stride,
        "training_point_count": int(len(fit_indices)),
        "validation_point_count": int(len(validation_indices)),
        "rwp_percent": None,
        "rp_percent": None,
        "rmse": None,
    }
    if validation_indices.size:
        training_solution = solve(
            refined_values,
            x_fit,
            y_fit,
            fit_sqrt_weights,
        )
        validation_phase_profiles = []
        for phase in phases:
            profile, _, _ = _phase_basis(
                x_full,
                phase,
                training_solution["state"],
                wavelength_angstrom,
                k_alpha2_enabled,
                k_alpha2_wavelength_angstrom,
                k_alpha2_ratio,
            )
            validation_phase_profiles.append(profile)
        validation_calculated = np.zeros_like(x_full)
        for profile, scale in zip(
            validation_phase_profiles,
            training_solution["scales"],
        ):
            validation_calculated += float(scale) * profile
        validation_calculated += (
            _background_basis(x_full, background_order)
            @ training_solution["background_coefficients"]
        )
        held_observed = y_full[validation_indices]
        held_calculated = validation_calculated[validation_indices]
        held_weights = full_weight_model.fit_sqrt_weights[validation_indices]
        held_residual = held_observed - held_calculated
        weighted_denominator = float(
            np.sum((held_observed * held_weights) ** 2)
        )
        cross_validation.update(
            rwp_percent=(
                None
                if weighted_denominator <= 0.0
                else 100.0
                * math.sqrt(
                    float(np.sum((held_residual * held_weights) ** 2))
                    / weighted_denominator
                )
            ),
            rp_percent=(
                100.0
                * float(np.sum(np.abs(held_residual)))
                / max(float(np.sum(np.abs(held_observed))), 1e-30)
            ),
            rmse=float(np.sqrt(np.mean(held_residual**2))),
        )

    final = solve(refined_values, x_full, y_full, full_weight_model.fit_sqrt_weights)
    residual = y_full - final["calculated"]
    parameter_count = len(names) + len(phases) + len(final["background_coefficients"])
    profile_statistics = calculate_profile_statistics(
        y_full,
        final["calculated"],
        full_weight_model,
        parameter_count,
    )
    rwp = profile_statistics["rwp_percent"]
    rp = profile_statistics["rp_percent"]
    rexp = profile_statistics["rexp_percent"]
    gof = profile_statistics["goodness_of_fit"]
    dof = profile_statistics["degrees_of_freedom"]
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_full - np.mean(y_full)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    durbin_watson = float(np.sum(np.diff(residual) ** 2) / max(np.sum(residual ** 2), 1e-30))

    weighted_residual = residual * final["weights"]
    correlations = _correlation_diagnostics(optimization, names, weighted_residual)
    parameter_errors = [None] * len(names)
    if optimization is not None and optimization.jac.size:
        try:
            covariance = np.linalg.pinv(optimization.jac.T @ optimization.jac)
            covariance_residual = (
                np.asarray(optimization.fun, dtype=float)
                if optimization is not None
                else weighted_residual
            )
            covariance *= float(
                np.sum(covariance_residual ** 2)
                / max(1, len(covariance_residual) - len(names))
            )
            parameter_errors = [
                None if value < 0 or not np.isfinite(value) else float(math.sqrt(value))
                for value in np.diag(covariance)
            ]
        except np.linalg.LinAlgError:
            pass

    scale_sum = float(np.sum(final["scales"]))
    scale_covariance = np.asarray(final.get("linear_covariance"), dtype=float)[:len(phases), :len(phases)]
    scale_errors = np.sqrt(np.maximum(np.diag(scale_covariance), 0.0)) if scale_covariance.size else np.full(len(phases), np.nan)
    phase_rows = []
    for phase, scale, profile in zip(phases, final["scales"], final["phase_profiles"].T):
        index = phase["phase_index"]
        handoff = phase["structure"].get("_afruz_phase11_handoff") or {}
        plausibility = phase["structure"].get("_afruz_structure_plausibility")
        if not isinstance(plausibility, dict):
            plausibility = validate_crystal_structure(phase["structure"])
        phase_rows.append(
            {
                "phase_index": index,
                "phase_name": phase["name"],
                "formula": phase["formula"],
                "scale_factor": float(scale),
                "scale_standard_error": (
                    None if not np.isfinite(scale_errors[len(phase_rows)])
                    else float(scale_errors[len(phase_rows)])
                ),
                "pattern_scale_fraction_percent": 100.0 * float(scale) / scale_sum if scale_sum > 0 else 0.0,
                "initial_cell": phase["initial_cell"],
                "refined_cell": final["state"]["cells"][index],
                "cell_refined": phase["refine_cell"],
                "delta_biso": float(final["state"]["delta_biso"][index]),
                "biso_refined": phase["refine_biso"],
                "preferred_orientation_hkl": list(phase["preferred_hkl"]) if phase["preferred_hkl"] else None,
                "march_dollase_r": float(final["state"]["preferred_r"][index]),
                "preferred_orientation_refined": phase["refine_preferred"],
                "profile_y": profile.tolist(),
                "atom_count": len(phase["structure"].get("atoms", [])),
                "xray_scattering_model": (
                    "CIF Cromer–Mann + anomalous dispersion"
                    if phase["structure"].get("xray_scattering_factors")
                    else "Atomic-number approximation"
                ),
                "xray_scattering_elements": sorted(
                    (phase["structure"].get("xray_scattering_factors") or {}).keys()
                ),
                "cif_reported_wavelength_angstrom": phase["structure"].get(
                    "reported_wavelength_angstrom"
                ),
                "atoms_fixed": True,
                "structure_factors_frozen": bool(
                    phase.get("freeze_structure_factors", False)
                ),
                "structure_refinement_override": bool(phase.get("structure_override", False)),
                "structure_snapshot": structure_snapshot(phase["structure"]),
                "origin": phase["structure"].get("_afruz_origin"),
                "validation_status": phase["structure"].get(
                    "_afruz_validation_status"
                ),
                "source_dataset_uid": handoff.get("source_dataset_uid"),
                "master_peak_revision": handoff.get("master_peak_revision"),
                "master_peak_checksum": handoff.get("master_peak_checksum"),
                "candidate_rank": handoff.get("candidate_rank"),
                "candidate_bravais": handoff.get("candidate_bravais"),
                "proposed_space_group": handoff.get("proposed_space_group"),
                "raw_cif_sha256": handoff.get("raw_cif_sha256"),
                "structure_plausibility_status": plausibility.get("status"),
                "structure_plausibility_score": (
                    plausibility.get("scores", {}).get("overall")
                    if isinstance(plausibility.get("scores"), dict)
                    else None
                ),
                "structure_plausibility_warnings": list(plausibility.get("warnings") or []),
                "density_g_cm3": plausibility.get("density_g_cm3"),
                "shortest_contact_angstrom": plausibility.get("shortest_contact_angstrom"),
                "metal_coordination_summary": plausibility.get("metal_coordination_summary", []),
            }
        )

    warnings = list(dict.fromkeys(final["warnings"]))
    if use_gpu and not acceleration.get("available"):
        warnings.append(str(acceleration.get("reason") or "CUDA was unavailable; CPU fallback used."))
    warnings.extend(full_weight_model.warnings)
    normalization_correlation = generated_pattern_normalization.get("initial_profile_correlation")
    if normalization_correlation is not None and normalization_correlation < 0.25:
        warnings.append(
            "The initially scaled CIF-generated profile has low correlation with the observed pattern; "
            "check wavelength, phase identity, missing phases, cell parameters, and profile width."
        )
    if not any(float(value) > 0.0 for value in generated_pattern_normalization.get("initial_phase_scale_factors", [])):
        warnings.append(
            "All initial CIF phase scale factors are zero after normalization; the selected structures do not explain the observed profile."
        )
    if not full_weight_model.statistics_valid:
        warnings.append(
            "Rexp, GoF and reduced chi-square are not reported because the selected intensity data do not have a valid absolute variance model. "
            + full_weight_model.reason
        )
    if rwp > 15.0:
        warnings.append("Rwp exceeds 15%; inspect missing phases, background, profile model, wavelength, calibration, and specimen effects.")
    if correlations["maximum_absolute_correlation"] is not None and correlations["maximum_absolute_correlation"] > 0.95:
        warnings.append("At least one nonlinear parameter pair has |correlation| above 0.95.")
    if correlations["condition_number"] is not None and correlations["condition_number"] > 1e10:
        warnings.append("The nonlinear normal matrix is ill-conditioned.")
    if not 1.0 <= durbin_watson <= 3.0:
        warnings.append("The difference curve is strongly serially correlated.")
    warnings.append(
        "Atomic coordinates and occupancies are fixed. Pattern scale fractions are not weight fractions without a validated Z-M-V/mass model."
    )
    tabulated_phase_count = sum(
        1 for phase in phases if phase["structure"].get("xray_scattering_factors")
    )
    if tabulated_phase_count == len(phases):
        warnings.append(
            "All phases use CIF-supplied Cromer–Mann X-ray form factors and "
            "CIF anomalous-dispersion terms. Verify that the selected wavelength "
            "and instrument geometry match the experiment."
        )
    elif tabulated_phase_count:
        warnings.append(
            f"{len(phases) - tabulated_phase_count} phase(s) lack CIF Cromer–Mann "
            "coefficients and use the documented atomic-number approximation; "
            "relative intensities for those phases are provisional."
        )
    else:
        warnings.append(
            "The CIF files supply no Cromer–Mann coefficients, so approximate "
            "atomic-number scattering factors are used; relative intensities are provisional."
        )
    frozen_phase_count = sum(
        1 for phase in phases if phase.get("freeze_structure_factors")
    )
    if frozen_phase_count:
        warnings.append(
            f"{frozen_phase_count} phase(s) used frozen CIF-derived structure factors. "
            "Cell, zero, scale, background, and profile terms may be refined, but "
            "atomic coordinates, occupancies, Biso, and relative reflection "
            "intensities remain fixed to the starting CIF calculation."
        )
    provisional_phase_count = sum(
        1 for row in phase_rows if row.get("origin") == "Unknown Discovery"
    )
    if provisional_phase_count:
        warnings.append(
            f"{provisional_phase_count} provisional Unknown Discovery phase(s) were included. "
            "Fit agreement tests compatibility with the powder pattern but does not independently prove "
            "the atomic model or proposed space group."
        )
    for row in phase_rows:
        if str(row.get("structure_plausibility_status") or "").lower().startswith("invalid"):
            warnings.append(
                f"{row['phase_name']}: structure plausibility contains invalid geometry; "
                "do not interpret the Rietveld fit as validation until the CIF is corrected."
            )
        elif row.get("structure_plausibility_warnings"):
            warnings.append(
                f"{row['phase_name']}: structure plausibility review is required; see phase diagnostics."
            )

    profile_residual_diagnostics = residual_diagnostics(residual)
    profile_information_criteria = information_criteria(residual, parameter_count)
    plot_payload = _rietveld_plot_payload(
        x_full,
        y_full,
        final["calculated"],
        final["background"],
        residual,
    )
    refinement_plan = staged_profile_refinement_plan(
        profile_model=profile_model,
        has_instrument_profile=bool(initial_u or initial_v or initial_w),
        refine_cell=any(phase["refine_cell"] for phase in phases),
        refine_texture=any(phase["refine_preferred"] for phase in phases),
    )
    if profile_model == "Split pseudo-Voigt":
        warnings.append(
            "Split pseudo-Voigt axial asymmetry is an empirical approximation, not a full fundamental-parameters convolution."
        )
    if calibration_prior is not None:
        warnings.append(
            "The active instrument-calibration covariance was applied as a "
            "Gaussian prior to shared position/profile parameters."
        )
    if cross_validation["enabled"]:
        warnings.append(
            f"Predictive validation withheld every {validation_stride}th point "
            f"({cross_validation['validation_point_count']} points) from nonlinear "
            "optimization. Use held-out Rwp to compare candidate models, then rerun "
            "the selected model with validation off for final all-point parameters."
        )
    if progress_callback is not None:
        progress_callback(total_evaluations, total_evaluations, "Rietveld refinement completed.")

    if use_gpu:
        release_gpu_memory()

    return {
        "success": bool(optimization.success if optimization is not None else True),
        "message": optimization.message if optimization is not None else "Linear scale/background solution completed.",
        "method": "Structure-constrained Rietveld profile refinement",
        "acceleration": acceleration,
        "wavelength_angstrom": float(wavelength_angstrom),
        "two_theta_min_deg": float(x_full[0]),
        "two_theta_max_deg": float(x_full[-1]),
        "weighting": weighting,
        "weighting_model": full_weight_model.model_name,
        "intensity_provenance": full_weight_model.provenance,
        "statistics_valid": bool(full_weight_model.statistics_valid),
        "statistics_reason": full_weight_model.reason,
        "intensity_scale_factor": float(full_weight_model.intensity_scale_factor),
        "robust_loss": robust_loss,
        "staged_refinement_enabled": bool(use_staged_refinement),
        "staged_refinement_history": stage_history,
        "generated_pattern_scaling": generated_pattern_scaling,
        "generated_pattern_normalization": generated_pattern_normalization,
        "background_order": background_order,
        "zero_shift_deg": float(final["state"]["zero_shift_deg"]),
        "profile": {
            "model": str(final["state"]["profile_model"]),
            "caglioti_u": float(final["state"]["u"]),
            "caglioti_v": float(final["state"]["v"]),
            "caglioti_w": float(final["state"]["w"]),
            "eta": float(final["state"]["eta"]),
            "lorentzian_x": float(final["state"]["x"]),
            "lorentzian_y": float(final["state"]["y"]),
            "axial_asymmetry": float(final["state"]["axial_asymmetry"]),
            "axial_sh_over_l": float(final["state"]["axial_sh_over_l"]),
            "diagnostics": profile_residual_diagnostics,
            "information_criteria": profile_information_criteria,
            "staged_refinement_plan": refinement_plan,
            "instrument_calibration": {
                "profile_fingerprint": (
                    None if not instrument_profile else instrument_profile.get("fingerprint")
                ),
                "coverage": calibration_coverage,
                "covariance_prior_applied": calibration_prior is not None,
                "prior_parameter_names": (
                    [] if calibration_prior is None else calibration_prior["parameter_names"]
                ),
                "prior_effective_rank": calibration_prior_size,
            },
        },
        "k_alpha2": {
            "enabled": bool(k_alpha2_enabled),
            "wavelength_angstrom": float(k_alpha2_wavelength_angstrom),
            "intensity_ratio": float(k_alpha2_ratio),
        },
        "parameters": [
            {"parameter": name, "value": float(value), "standard_error": error}
            for name, value, error in zip(names, refined_values, parameter_errors)
        ],
        "phases": phase_rows,
        "provisional_phase_count": provisional_phase_count,
        "scale_covariance": scale_covariance.tolist(),
        "reflections": final["reflections"],
        "background_coefficients": final["background_coefficients"].tolist(),
        "observed_x": x_full.tolist(),
        "observed_y": y_full.tolist(),
        "calculated_y": final["calculated"].tolist(),
        "background_y": final["background"].tolist(),
        "difference_y": residual.tolist(),
        "plot_payload": plot_payload,
        "observed_sigma": (
            None
            if full_weight_model.sigma is None
            else full_weight_model.sigma.tolist()
        ),
        "statistical_weights": full_weight_model.reporting_weights.tolist(),
        "rwp_percent": float(rwp),
        "rp_percent": float(rp),
        "rexp_percent": None if rexp is None else float(rexp),
        "goodness_of_fit": None if gof is None else float(gof),
        "goodness_of_fit_sqrt": profile_statistics["goodness_of_fit_sqrt"],
        "reduced_chi_square": profile_statistics["reduced_chi_square"],
        "weighted_residual_sum_squares": profile_statistics["weighted_residual_sum_squares"],
        "weighted_observed_sum_squares": profile_statistics["weighted_observed_sum_squares"],
        "degrees_of_freedom": int(dof),
        "r_squared": float(r_squared),
        "rmse": rmse,
        "durbin_watson": durbin_watson,
        "condition_number": correlations["condition_number"],
        "maximum_absolute_correlation": correlations["maximum_absolute_correlation"],
        "strong_correlation_pairs": correlations["strong_pairs"],
        "data_point_count": len(y_full),
        "optimization_point_count": len(y_fit),
        "optimization_sampling": optimization_sampling,
        "cross_validation": cross_validation,
        "phase_count": len(phases),
        "reflection_count": len(final["reflections"]),
        "parameter_count": parameter_count,
        "nonlinear_parameter_count": len(names),
        "nonlinear_evaluations": int(evaluation_counter),
        "elapsed_seconds": float(time.perf_counter() - started),
        "warnings": warnings,
    }
