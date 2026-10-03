from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import re
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
from .crystallography import d_spacing
from .gpu_backend import compute_backend, release_gpu_memory
from .instrument_physics import (
    CalibrationRangeError,
    calibration_prior_residual,
    prepare_calibration_prior,
    radiation_components,
    spectral_positions_deg,
    validate_calibration_range,
    validate_profile_compatibility,
)
from .instrument_calibration import position_correction_deg
from .phase_identification import ReferenceEntry, prepare_reference_peaks
from .refinement_statistics import (
    build_refinement_weight_model,
    calculate_profile_statistics,
)
from .whole_pattern_metrology import (
    intensity_metrology,
    nonlinear_parameter_metrology,
)


WHOLE_PATTERN_MODES = (
    "Pawley decomposition",
    "Le Bail-style extraction",
)

WHOLE_PATTERN_WEIGHTING = (
    "Poisson-like",
    "Balanced",
    "Uniform",
)

PROFILE_OPTIONS = PROFILE_MODELS


class WholePatternError(ValueError):
    pass


class WholePatternCancelled(RuntimeError):
    pass


@dataclass
class WholePatternPhaseSpec:
    reference: ReferenceEntry
    refine_cell: bool = True
    included: bool = True


def _parse_hkl(label: str) -> tuple[int, int, int] | None:
    values = re.findall(r"[+-]?\d+", str(label))
    if len(values) < 3:
        return None
    hkl = tuple(int(value) for value in values[:3])
    return None if hkl == (0, 0, 0) else hkl


def _normalised_profile_x(x: np.ndarray) -> np.ndarray:
    center = float(np.mean(x))
    half_span = max(float(np.ptp(x)) / 2.0, np.finfo(float).eps)
    return (x - center) / half_span


def _background_basis(x: np.ndarray, order: int) -> np.ndarray:
    xn = _normalised_profile_x(x)
    return np.column_stack([xn ** degree for degree in range(order + 1)])


def _positive_background_basis(x: np.ndarray, order: int) -> np.ndarray:
    """Return a Bernstein basis whose non-negative coefficients stay physical.

    Le Bail extraction alternates peak repartition with a background fit.  An
    unconstrained power-series background can become strongly negative and be
    cancelled by inflated peak intensities, producing a deceptively good total
    profile.  Bernstein polynomials form a partition of unity on the fitted
    interval, so constraining their coefficients to be non-negative guarantees
    a non-negative background without preventing smooth slopes or curvature.
    """
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
        raise WholePatternError(
            f"Unsupported whole-pattern weighting mode: {mode}"
        )
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
    gaussian_area = width * math.sqrt(math.pi) / (2.0 * math.sqrt(math.log(2.0)))
    lorentzian_area = math.pi * width / 2.0
    area = mixing * lorentzian_area + (1.0 - mixing) * gaussian_area
    return profile / area if area > np.finfo(float).eps else profile


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


def _cell_parameter_names(cell: dict, crystal_system: str) -> list[str]:
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


def _cell_values(cell: dict, crystal_system: str) -> list[float]:
    return [float(cell[name]) for name in _cell_parameter_names(cell, crystal_system)]


def _cell_from_values(
    names: list[str],
    values: np.ndarray,
    crystal_system: str,
) -> dict:
    parameters = {name: float(value) for name, value in zip(names, values)}
    system = str(crystal_system or "Triclinic")
    if system == "Cubic":
        a = parameters["a"]
        return {
            "a": a, "b": a, "c": a,
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        }
    if system == "Tetragonal":
        return {
            "a": parameters["a"], "b": parameters["a"], "c": parameters["c"],
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        }
    if system == "Hexagonal":
        return {
            "a": parameters["a"], "b": parameters["a"], "c": parameters["c"],
            "alpha": 90.0, "beta": 90.0, "gamma": 120.0,
        }
    if system == "Orthorhombic":
        return {
            "a": parameters["a"], "b": parameters["b"], "c": parameters["c"],
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        }
    if system == "Rhombohedral":
        a = parameters["a"]
        alpha = parameters["alpha"]
        return {
            "a": a, "b": a, "c": a,
            "alpha": alpha, "beta": alpha, "gamma": alpha,
        }
    if system == "Monoclinic":
        return {
            "a": parameters["a"], "b": parameters["b"], "c": parameters["c"],
            "alpha": 90.0, "beta": parameters["beta"], "gamma": 90.0,
        }
    return {
        "a": parameters["a"],
        "b": parameters["b"],
        "c": parameters["c"],
        "alpha": parameters["alpha"],
        "beta": parameters["beta"],
        "gamma": parameters["gamma"],
    }


def _cell_standard_errors(
    independent_errors: dict[str, float | None],
    crystal_system: str,
) -> dict[str, float | None]:
    """Expand independent lattice-parameter errors through symmetry constraints."""

    result = {
        name: None for name in ("a", "b", "c", "alpha", "beta", "gamma")
    }
    result.update(independent_errors)
    system = str(crystal_system or "Triclinic")
    if system == "Cubic":
        result.update(
            a=independent_errors.get("a"),
            b=independent_errors.get("a"),
            c=independent_errors.get("a"),
        )
    elif system in {"Tetragonal", "Hexagonal"}:
        result["b"] = independent_errors.get("a")
    elif system == "Rhombohedral":
        result.update(
            a=independent_errors.get("a"),
            b=independent_errors.get("a"),
            c=independent_errors.get("a"),
            alpha=independent_errors.get("alpha"),
            beta=independent_errors.get("alpha"),
            gamma=independent_errors.get("alpha"),
        )
    return result


def _two_theta_from_cell(
    cell: dict,
    hkl: tuple[int, int, int],
    wavelength_angstrom: float,
) -> float | None:
    try:
        spacing = d_spacing(cell, hkl)
    except (ValueError, np.linalg.LinAlgError):
        return None
    argument = float(wavelength_angstrom) / (2.0 * float(spacing))
    if not 0.0 < argument < 1.0:
        return None
    return float(2.0 * math.degrees(math.asin(argument)))


def _prepared_phases(
    phase_specs: list[WholePatternPhaseSpec],
    wavelength_angstrom: float,
    x_min: float,
    x_max: float,
    intensity_cutoff_percent: float,
) -> list[dict]:
    phases = []
    for phase_index, spec in enumerate(phase_specs, start=1):
        if not spec.included:
            continue
        reference = spec.reference
        peaks = prepare_reference_peaks(
            reference,
            target_wavelength_angstrom=wavelength_angstrom,
            convert_from_d=True,
            intensity_cutoff_percent=intensity_cutoff_percent,
            two_theta_min=x_min,
            two_theta_max=x_max,
        )
        if len(peaks) < 2:
            continue

        metadata = dict(reference.metadata or {})
        cell = metadata.get("cell")
        crystal_system = metadata.get("crystal_system") or "Triclinic"
        valid_cell = (
            isinstance(cell, dict)
            and all(
                key in cell
                for key in ("a", "b", "c", "alpha", "beta", "gamma")
            )
        )

        reflection_rows = []
        for peak_index, peak in enumerate(peaks, start=1):
            hkl = _parse_hkl(peak.get("hkl_label", ""))
            reflection_rows.append(
                {
                    "phase_index": phase_index,
                    "phase_uid": reference.uid,
                    "phase_name": reference.name,
                    "formula": reference.formula,
                    "reflection_index": peak_index,
                    "initial_two_theta_deg": float(peak["two_theta"]),
                    "reference_intensity": max(
                        1e-8,
                        float(peak.get("intensity", 0.0)) / 100.0,
                    ),
                    "hkl": hkl,
                    "hkl_label": peak.get("hkl_label", ""),
                    "d_spacing": peak.get("d_spacing"),
                }
            )

        cell_refinable = bool(
            spec.refine_cell
            and valid_cell
            and sum(row["hkl"] is not None for row in reflection_rows) >= 2
        )
        phases.append(
            {
                "phase_index": phase_index,
                "uid": reference.uid,
                "name": reference.name,
                "formula": reference.formula,
                "source": reference.source,
                "metadata": metadata,
                "initial_cell": deepcopy(cell) if valid_cell else None,
                "crystal_system": crystal_system,
                "refine_cell": cell_refinable,
                "reflections": reflection_rows,
            }
        )
    if not phases:
        raise WholePatternError(
            "No included phase has at least two reference reflections in the "
            "selected 2θ range."
        )
    return phases


def _build_parameterization(
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
    mapping: dict = {"phase_cells": {}}

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
        if not phase["refine_cell"]:
            continue
        cell = phase["initial_cell"]
        system = phase["crystal_system"]
        cell_names = _cell_parameter_names(cell, system)
        values = _cell_values(cell, system)
        indices = []
        for cell_name, value in zip(cell_names, values):
            full_name = f"phase{phase['phase_index']}_{cell_name}"
            indices.append(len(names))
            names.append(full_name)
            initial.append(float(value))
            if cell_name in {"a", "b", "c"}:
                lower.append(max(0.05, value * (1.0 - relative)))
                upper.append(value * (1.0 + relative))
            else:
                angle_span = max(0.2, 180.0 * relative)
                lower.append(max(35.0, value - angle_span))
                upper.append(min(145.0, value + angle_span))
        mapping["phase_cells"][phase["phase_index"]] = {
            "indices": indices,
            "names": cell_names,
        }

    return (
        names,
        np.asarray(initial, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        mapping,
    )


def _unpack_parameters(
    values: np.ndarray,
    names: list[str],
    phases: list[dict],
    mapping: dict,
    fixed: dict,
) -> dict:
    parameters = {name: float(value) for name, value in zip(names, values)}
    result = {
        "zero_shift_deg": parameters.get(
            "zero_shift_deg",
            float(fixed["zero_shift_deg"]),
        ),
        "caglioti_u": parameters.get(
            "caglioti_u",
            float(fixed["caglioti_u"]),
        ),
        "caglioti_v": parameters.get(
            "caglioti_v",
            float(fixed["caglioti_v"]),
        ),
        "caglioti_w": parameters.get(
            "caglioti_w",
            float(fixed["caglioti_w"]),
        ),
        "eta": parameters.get("eta", float(fixed["eta"])),
        "profile_model": str(fixed["profile_model"]),
        "lorentzian_x": parameters.get("lorentzian_x", float(fixed["lorentzian_x"])),
        "lorentzian_y": parameters.get("lorentzian_y", float(fixed["lorentzian_y"])),
        "axial_asymmetry": parameters.get(
            "axial_asymmetry", float(fixed["axial_asymmetry"])
        ),
        "axial_sh_over_l": float(fixed.get("axial_sh_over_l", 0.0)),
        "phase_cells": {},
    }
    for phase in phases:
        phase_index = phase["phase_index"]
        if phase_index in mapping["phase_cells"]:
            info = mapping["phase_cells"][phase_index]
            cell_values = values[np.asarray(info["indices"], dtype=int)]
            result["phase_cells"][phase_index] = _cell_from_values(
                info["names"],
                cell_values,
                phase["crystal_system"],
            )
        else:
            result["phase_cells"][phase_index] = deepcopy(
                phase["initial_cell"]
            )
    return result


def _reflection_positions(
    phases: list[dict],
    parameter_state: dict,
    wavelength_angstrom: float,
    *,
    spectral_components: list[dict],
    instrument_profile: dict | None = None,
    apply_instrument_position_correction: bool = False,
) -> tuple[list[dict], list[str]]:
    rows = []
    warnings = []
    shift = float(parameter_state["zero_shift_deg"])
    for phase in phases:
        cell = parameter_state["phase_cells"].get(phase["phase_index"])
        for reflection in phase["reflections"]:
            center = float(reflection["initial_two_theta_deg"])
            if cell is not None and reflection["hkl"] is not None:
                calculated = _two_theta_from_cell(
                    cell,
                    reflection["hkl"],
                    wavelength_angstrom,
                )
                if calculated is not None:
                    center = calculated
                else:
                    warnings.append(
                        f"{phase['name']} {reflection['hkl_label']}: "
                        "refined cell moved the reflection outside the Bragg condition."
                    )
                    continue
            theoretical_center = float(center)
            try:
                components = spectral_positions_deg(
                    theoretical_center,
                    spectral_components,
                )
            except ValueError as exc:
                warnings.append(
                    f"{phase['name']} {reflection['hkl_label']}: {exc}"
                )
                continue
            corrected_components = []
            for component in components:
                component_center = float(component["two_theta_deg"])
                correction = 0.0
                if apply_instrument_position_correction and instrument_profile:
                    geometry_profile = deepcopy(instrument_profile)
                    geometry_profile["zero_shift_deg"] = 0.0
                    correction = float(
                        position_correction_deg(component_center, geometry_profile)
                    )
                corrected_components.append(
                    {
                        **component,
                        "theoretical_two_theta_deg": component_center,
                        "instrument_position_correction_deg": correction,
                        "two_theta_deg": component_center + correction + shift,
                    }
                )
            primary = corrected_components[0]
            rows.append(
                {
                    **reflection,
                    "theoretical_two_theta_deg": theoretical_center,
                    "instrument_position_correction_deg": float(
                        primary["instrument_position_correction_deg"]
                    ),
                    "two_theta_deg": float(primary["two_theta_deg"]),
                    "spectral_components": corrected_components,
                }
            )
    return rows, warnings


def _build_peak_basis(
    x: np.ndarray,
    reflections: list[dict],
    *,
    u: float,
    v: float,
    w: float,
    eta: float,
    profile_model: str = "Pseudo-Voigt U-V-W",
    lorentzian_x: float = 0.0,
    lorentzian_y: float = 0.0,
    axial_asymmetry: float = 0.0,
    axial_sh_over_l: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not reflections:
        return np.empty((len(x), 0)), np.empty(0)
    step = float(np.median(np.diff(x)))
    minimum_width = max(step * 1.15, 0.003)
    basis = np.zeros((len(x), len(reflections)), dtype=float)
    widths = np.zeros(len(reflections), dtype=float)

    for column, reflection in enumerate(reflections):
        center = float(reflection["two_theta_deg"])
        characteristics = profile_characteristics(
            center,
            model=profile_model,
            u=u,
            v=v,
            w=w,
            eta=eta,
            x=lorentzian_x,
            y=lorentzian_y,
            axial_asymmetry=axial_asymmetry,
            axial_sh_over_l=axial_sh_over_l,
            minimum_width=minimum_width,
        )
        widths[column] = characteristics.fwhm_deg
        components = reflection.get("spectral_components") or [
            {"two_theta_deg": center, "relative_intensity": 1.0}
        ]
        total_component_weight = sum(
            max(float(component.get("relative_intensity", 0.0)), 0.0)
            for component in components
        )
        if total_component_weight <= 0.0:
            raise WholePatternError("Radiation component weights must sum to a positive value.")
        composite = np.zeros(len(x), dtype=float)
        for component in components:
            component_weight = max(
                float(component.get("relative_intensity", 0.0)), 0.0
            )
            composite += component_weight * profile_unit_area(
                x,
                float(component["two_theta_deg"]),
                model=profile_model,
                u=u,
                v=v,
                w=w,
                eta=eta,
                x=lorentzian_x,
                y=lorentzian_y,
                axial_asymmetry=axial_asymmetry,
                axial_sh_over_l=axial_sh_over_l,
                minimum_width=minimum_width,
            )
        basis[:, column] = composite / total_component_weight
    return basis, widths


def _build_peak_basis_cuda(
    x: np.ndarray,
    reflections: list[dict],
    *,
    u: float,
    v: float,
    w: float,
    eta: float,
    profile_model: str = "Pseudo-Voigt U-V-W",
    lorentzian_x: float = 0.0,
    lorentzian_y: float = 0.0,
    axial_asymmetry: float = 0.0,
    axial_sh_over_l: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the unit-area reflection matrix in CUDA-sized vector batches.

    Scalar profile characteristics are still evaluated by the authoritative
    CPU model. CUDA only evaluates the large point-by-component arrays, using
    the same Gaussian/Lorentzian mixture and analytic unit-area normalization.
    """
    component_count = sum(
        len(reflection.get("spectral_components") or [None])
        for reflection in reflections
    )
    workload = int(len(x) * component_count)
    if workload < 50_000:
        basis, widths = _build_peak_basis(
            x,
            reflections,
            u=u,
            v=v,
            w=w,
            eta=eta,
            profile_model=profile_model,
            lorentzian_x=lorentzian_x,
            lorentzian_y=lorentzian_y,
            axial_asymmetry=axial_asymmetry,
            axial_sh_over_l=axial_sh_over_l,
        )
        return basis, widths, {
            "requested": True,
            "used": False,
            "backend": "CPU",
            "device_name": "",
            "reason": "The reflection profile matrix is too small for CUDA to be beneficial.",
        }

    xp, status = compute_backend(True)
    if xp is np:
        basis, widths = _build_peak_basis(
            x,
            reflections,
            u=u,
            v=v,
            w=w,
            eta=eta,
            profile_model=profile_model,
            lorentzian_x=lorentzian_x,
            lorentzian_y=lorentzian_y,
            axial_asymmetry=axial_asymmetry,
            axial_sh_over_l=axial_sh_over_l,
        )
        return basis, widths, dict(status) | {"requested": True, "used": False}

    if not reflections:
        return np.empty((len(x), 0)), np.empty(0), dict(status) | {
            "requested": True, "used": False,
        }

    try:
        step = float(np.median(np.diff(x)))
        minimum_width = max(step * 1.15, 0.003)
        widths = np.zeros(len(reflections), dtype=float)
        centers: list[float] = []
        left_widths: list[float] = []
        right_widths: list[float] = []
        mixings: list[float] = []
        areas: list[float] = []
        weights: list[float] = []
        columns: list[int] = []

        for column, reflection in enumerate(reflections):
            center = float(reflection["two_theta_deg"])
            primary = profile_characteristics(
                center,
                model=profile_model,
                u=u,
                v=v,
                w=w,
                eta=eta,
                x=lorentzian_x,
                y=lorentzian_y,
                axial_asymmetry=axial_asymmetry,
                axial_sh_over_l=axial_sh_over_l,
                minimum_width=minimum_width,
            )
            widths[column] = primary.fwhm_deg
            components = reflection.get("spectral_components") or [
                {"two_theta_deg": center, "relative_intensity": 1.0}
            ]
            total = sum(
                max(float(component.get("relative_intensity", 0.0)), 0.0)
                for component in components
            )
            if total <= 0.0:
                raise WholePatternError(
                    "Radiation component weights must sum to a positive value."
                )
            for component in components:
                component_center = float(component["two_theta_deg"])
                characteristics = profile_characteristics(
                    component_center,
                    model=profile_model,
                    u=u,
                    v=v,
                    w=w,
                    eta=eta,
                    x=lorentzian_x,
                    y=lorentzian_y,
                    axial_asymmetry=axial_asymmetry,
                    axial_sh_over_l=axial_sh_over_l,
                    minimum_width=minimum_width,
                )
                width_sum = (
                    characteristics.left_fwhm_deg
                    + characteristics.right_fwhm_deg
                )
                gaussian_area = (
                    width_sum * math.sqrt(math.pi)
                    / (4.0 * math.sqrt(math.log(2.0)))
                )
                lorentzian_area = math.pi * width_sum / 4.0
                area = (
                    characteristics.eta * lorentzian_area
                    + (1.0 - characteristics.eta) * gaussian_area
                )
                centers.append(component_center)
                left_widths.append(characteristics.left_fwhm_deg)
                right_widths.append(characteristics.right_fwhm_deg)
                mixings.append(characteristics.eta)
                areas.append(area)
                weights.append(
                    max(float(component.get("relative_intensity", 0.0)), 0.0)
                    / total
                )
                columns.append(column)

        x_gpu = xp.asarray(np.asarray(x, dtype=float), dtype=xp.float64)
        basis_gpu = xp.zeros((len(x), len(reflections)), dtype=xp.float64)
        chunk_size = 192
        for start in range(0, len(centers), chunk_size):
            stop = min(len(centers), start + chunk_size)
            local_centers = xp.asarray(centers[start:stop], dtype=xp.float64)
            delta = x_gpu[:, None] - local_centers[None, :]
            local_left = xp.asarray(left_widths[start:stop], dtype=xp.float64)
            local_right = xp.asarray(right_widths[start:stop], dtype=xp.float64)
            local_width = xp.where(delta < 0.0, local_left[None, :], local_right[None, :])
            scaled = delta / xp.maximum(local_width, 1e-12)
            gaussian = xp.exp(-4.0 * math.log(2.0) * scaled * scaled)
            lorentzian = 1.0 / (1.0 + 4.0 * scaled * scaled)
            local_eta = xp.asarray(mixings[start:stop], dtype=xp.float64)
            values = (
                local_eta[None, :] * lorentzian
                + (1.0 - local_eta[None, :]) * gaussian
            )
            local_area = xp.asarray(areas[start:stop], dtype=xp.float64)
            local_weight = xp.asarray(weights[start:stop], dtype=xp.float64)
            values = values / local_area[None, :] * local_weight[None, :]
            local_columns = xp.asarray(columns[start:stop], dtype=xp.int64)
            xp.add.at(basis_gpu, (slice(None), local_columns), values)

        basis = xp.asnumpy(basis_gpu)
        status = dict(status)
        status.update(requested=True, used=True, workload=workload)
        return basis, widths, status
    except Exception as exc:
        basis, widths = _build_peak_basis(
            x,
            reflections,
            u=u,
            v=v,
            w=w,
            eta=eta,
            profile_model=profile_model,
            lorentzian_x=lorentzian_x,
            lorentzian_y=lorentzian_y,
            axial_asymmetry=axial_asymmetry,
            axial_sh_over_l=axial_sh_over_l,
        )
        return basis, widths, {
            "requested": True,
            "used": False,
            "available": False,
            "backend": "CPU",
            "device_name": status.get("device_name", ""),
            "reason": f"CUDA profile-matrix calculation failed; CPU fallback used: {exc}",
        }


def _solve_pawley(
    peak_basis: np.ndarray,
    background_basis: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    phase_count = peak_basis.shape[1]
    design = np.column_stack([peak_basis, background_basis])
    weighted_design = design * weights[:, None]
    weighted_y = y * weights
    lower = np.concatenate(
        [
            np.zeros(phase_count, dtype=float),
            np.zeros(background_basis.shape[1], dtype=float),
        ]
    )
    upper = np.full(design.shape[1], np.inf)
    solution = lsq_linear(
        weighted_design,
        weighted_y,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
        max_iter=2500,
    )
    if not solution.success:
        raise WholePatternError(
            f"Pawley intensity solution failed: {solution.message}"
        )
    intensities = np.asarray(solution.x[:phase_count], dtype=float)
    background_coefficients = np.asarray(
        solution.x[phase_count:],
        dtype=float,
    )
    calculated = design @ solution.x
    return intensities, background_coefficients, calculated, {
        "algorithm": "bounded linear least squares",
        "success": bool(solution.success),
        "iterations": int(solution.nit),
        "weighted_cost": float(solution.cost),
        "optimality": float(solution.optimality),
    }


def _solve_le_bail(
    peak_basis: np.ndarray,
    background_basis: np.ndarray,
    reflections: list[dict],
    y: np.ndarray,
    weights: np.ndarray,
    cycles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if peak_basis.shape[1] == 0:
        raise WholePatternError("No reflections remain in the selected range.")
    reference = np.asarray(
        [row["reference_intensity"] for row in reflections],
        dtype=float,
    )
    reference = np.maximum(reference, 1e-8)
    scale = max(
        float(np.trapezoid(np.maximum(y - np.percentile(y, 5), 0.0))),
        1.0,
    )
    intensities = reference / float(np.sum(reference)) * scale
    background_coefficients = np.zeros(
        background_basis.shape[1],
        dtype=float,
    )
    weighted_background = background_basis * weights[:, None]

    history = []
    for cycle in range(max(2, int(cycles))):
        peak_calculated = peak_basis @ intensities
        target_background = y - peak_calculated
        background_solution = lsq_linear(
            weighted_background,
            target_background * weights,
            bounds=(0.0, np.inf),
            method="trf",
            lsmr_tol="auto",
            max_iter=250,
        )
        if not background_solution.success:
            raise WholePatternError(
                "Le Bail background solution failed: "
                f"{background_solution.message}"
            )
        background_coefficients = np.asarray(
            background_solution.x,
            dtype=float,
        )
        background = background_basis @ background_coefficients
        net_observed = np.maximum(y - background, 0.0)
        peak_calculated = np.maximum(peak_basis @ intensities, 1e-12)
        ratio = np.clip(net_observed / peak_calculated, 0.0, 50.0)
        numerator = peak_basis.T @ (weights * ratio)
        denominator = np.maximum(
            peak_basis.T @ weights,
            1e-12,
        )
        updated = intensities * numerator / denominator
        next_intensities = 0.45 * intensities + 0.55 * np.maximum(updated, 0.0)
        relative_change = float(
            np.linalg.norm(next_intensities - intensities)
            / max(np.linalg.norm(intensities), 1e-12)
        )
        intensities = next_intensities
        cycle_calculated = (
            peak_basis @ intensities
            + background_basis @ background_coefficients
        )
        history.append(
            {
                "cycle": cycle + 1,
                "relative_intensity_change": relative_change,
                "weighted_residual_sum_squares": float(
                    np.sum(((y - cycle_calculated) * weights) ** 2)
                ),
            }
        )

    calculated = (
        peak_basis @ intensities
        + background_basis @ background_coefficients
    )
    return intensities, background_coefficients, calculated, {
        "algorithm": "damped Le Bail profile repartition",
        "cycles": len(history),
        "final_relative_intensity_change": history[-1][
            "relative_intensity_change"
        ],
        "history": history,
        "converged_at_1e-5": bool(
            history[-1]["relative_intensity_change"] <= 1e-5
        ),
    }


def _solve_le_bail_cuda(
    peak_basis: np.ndarray,
    background_basis: np.ndarray,
    reflections: list[dict],
    y: np.ndarray,
    weights: np.ndarray,
    cycles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Le Bail repartition with CUDA matrix products and CPU bounded background."""
    workload = int(peak_basis.size * max(2, int(cycles)))
    # NumPy/BLAS is exceptionally efficient for medium matrix-vector products.
    # CUDA wins only once repeated repartition work is large enough to amortize
    # transfers and kernel dispatch; profile-basis construction is accelerated
    # independently at a much lower threshold.
    if workload < 50_000_000:
        values = _solve_le_bail(
            peak_basis, background_basis, reflections, y, weights, cycles
        )
        diagnostics = dict(values[3])
        diagnostics["acceleration"] = {
            "requested": True,
            "used": False,
            "backend": "CPU",
            "device_name": "",
            "reason": "The Le Bail profile-repartition problem is too small for CUDA to be beneficial.",
        }
        return values[0], values[1], values[2], diagnostics

    xp, status = compute_backend(True)
    if xp is np:
        values = _solve_le_bail(
            peak_basis, background_basis, reflections, y, weights, cycles
        )
        diagnostics = dict(values[3])
        diagnostics["acceleration"] = dict(status) | {
            "requested": True, "used": False
        }
        return values[0], values[1], values[2], diagnostics

    try:
        if peak_basis.shape[1] == 0:
            raise WholePatternError("No reflections remain in the selected range.")
        reference = np.asarray(
            [row["reference_intensity"] for row in reflections], dtype=float
        )
        reference = np.maximum(reference, 1e-8)
        scale = max(
            float(np.trapezoid(np.maximum(y - np.percentile(y, 5), 0.0))),
            1.0,
        )
        initial_intensities = reference / float(np.sum(reference)) * scale
        peak_gpu = xp.asarray(peak_basis, dtype=xp.float64)
        background_gpu = xp.asarray(background_basis, dtype=xp.float64)
        y_gpu = xp.asarray(y, dtype=xp.float64)
        weights_gpu = xp.asarray(weights, dtype=xp.float64)
        intensities_gpu = xp.asarray(initial_intensities, dtype=xp.float64)
        background_coefficients = np.zeros(background_basis.shape[1], dtype=float)
        weighted_background = background_basis * weights[:, None]
        denominator = xp.maximum(peak_gpu.T @ weights_gpu, 1e-12)

        history = []
        for cycle in range(max(2, int(cycles))):
            peak_calculated_gpu = peak_gpu @ intensities_gpu
            target_background = xp.asnumpy(y_gpu - peak_calculated_gpu)
            background_solution = lsq_linear(
                weighted_background,
                target_background * weights,
                bounds=(0.0, np.inf),
                method="trf",
                lsmr_tol="auto",
                max_iter=250,
            )
            if not background_solution.success:
                raise WholePatternError(
                    "Le Bail background solution failed: "
                    f"{background_solution.message}"
                )
            background_coefficients = np.asarray(background_solution.x, dtype=float)
            coefficients_gpu = xp.asarray(background_coefficients, dtype=xp.float64)
            background_values_gpu = background_gpu @ coefficients_gpu
            net_observed_gpu = xp.maximum(y_gpu - background_values_gpu, 0.0)
            peak_calculated_gpu = xp.maximum(peak_gpu @ intensities_gpu, 1e-12)
            ratio_gpu = xp.clip(net_observed_gpu / peak_calculated_gpu, 0.0, 50.0)
            numerator = peak_gpu.T @ (weights_gpu * ratio_gpu)
            updated = intensities_gpu * numerator / denominator
            next_intensities = (
                0.45 * intensities_gpu + 0.55 * xp.maximum(updated, 0.0)
            )
            relative_change = float(
                xp.asnumpy(
                    xp.linalg.norm(next_intensities - intensities_gpu)
                    / xp.maximum(xp.linalg.norm(intensities_gpu), 1e-12)
                )
            )
            intensities_gpu = next_intensities
            cycle_calculated_gpu = (
                peak_gpu @ intensities_gpu
                + background_gpu @ coefficients_gpu
            )
            weighted_rss = float(
                xp.asnumpy(
                    xp.sum(((y_gpu - cycle_calculated_gpu) * weights_gpu) ** 2)
                )
            )
            history.append(
                {
                    "cycle": cycle + 1,
                    "relative_intensity_change": relative_change,
                    "weighted_residual_sum_squares": weighted_rss,
                }
            )

        calculated_gpu = (
            peak_gpu @ intensities_gpu
            + background_gpu @ xp.asarray(background_coefficients, dtype=xp.float64)
        )
        acceleration = dict(status)
        acceleration.update(requested=True, used=True, workload=workload)
        diagnostics = {
            "algorithm": "damped Le Bail profile repartition",
            "cycles": len(history),
            "final_relative_intensity_change": history[-1]["relative_intensity_change"],
            "history": history,
            "converged_at_1e-5": bool(
                history[-1]["relative_intensity_change"] <= 1e-5
            ),
            "acceleration": acceleration,
        }
        return (
            xp.asnumpy(intensities_gpu),
            background_coefficients,
            xp.asnumpy(calculated_gpu),
            diagnostics,
        )
    except WholePatternError:
        raise
    except Exception as exc:
        values = _solve_le_bail(
            peak_basis, background_basis, reflections, y, weights, cycles
        )
        diagnostics = dict(values[3])
        diagnostics["acceleration"] = {
            "requested": True,
            "used": False,
            "available": False,
            "backend": "CPU",
            "device_name": status.get("device_name", ""),
            "reason": f"CUDA Le Bail repartition failed; CPU fallback used: {exc}",
        }
        return values[0], values[1], values[2], diagnostics


def refine_whole_pattern(
    x: np.ndarray,
    y: np.ndarray,
    phase_specs: list[WholePatternPhaseSpec],
    *,
    mode: str = "Pawley decomposition",
    wavelength_angstrom: float = 1.5406,
    two_theta_min: float | None = None,
    two_theta_max: float | None = None,
    intensity_cutoff_percent: float = 0.5,
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
    extraction_cycles: int = 12,
    maximum_nonlinear_evaluations: int = 120,
    maximum_optimization_points: int = 1800,
    robust_loss: str = "soft_l1",
    use_staged_refinement: bool = False,
    observed_sigma: np.ndarray | None = None,
    count_reference: np.ndarray | None = None,
    intensity_scale_factor: float = 1.0,
    intensity_provenance: str = "raw_counts",
    statistics_note: str | None = None,
    instrument_profile: dict | None = None,
    allow_profile_extrapolation: bool = False,
    radiation_configuration: str = "Custom monochromatic",
    secondary_wavelength_angstrom: float = 1.5444274,
    secondary_to_primary_ratio: float = 0.5,
    apply_instrument_position_correction: bool = False,
    overlap_correlation_threshold: float = 0.90,
    use_gpu: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    started_at = time.perf_counter()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if mode not in WHOLE_PATTERN_MODES:
        raise WholePatternError(f"Unsupported whole-pattern mode: {mode}")
    if weighting not in WHOLE_PATTERN_WEIGHTING:
        raise WholePatternError(f"Unsupported weighting mode: {weighting}")
    if profile_model not in PROFILE_MODELS:
        raise WholePatternError(f"Unsupported profile model: {profile_model}")
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise WholePatternError(
            "X and Y must be one-dimensional arrays of equal length."
        )
    if len(x) < 50:
        raise WholePatternError(
            "At least fifty measured points are required."
        )
    if wavelength_angstrom <= 0:
        raise WholePatternError("Wavelength must be positive.")
    try:
        configured_radiation = radiation_components(
            radiation_configuration,
            primary_wavelength_angstrom=wavelength_angstrom,
            secondary_wavelength_angstrom=secondary_wavelength_angstrom,
            secondary_to_primary_ratio=secondary_to_primary_ratio,
        )
    except ValueError as exc:
        raise WholePatternError(str(exc)) from exc
    if instrument_profile:
        try:
            validate_profile_compatibility(
                instrument_profile,
                wavelength_angstrom=wavelength_angstrom,
            )
        except ValueError as exc:
            raise WholePatternError(str(exc)) from exc
    background_order = int(np.clip(background_order, 0, 6))

    range_min = float(x[0] if two_theta_min is None else two_theta_min)
    range_max = float(x[-1] if two_theta_max is None else two_theta_max)
    mask = (x >= range_min) & (x <= range_max)
    x_full = x[mask]
    y_full = y[mask]
    sigma_full = None
    if observed_sigma is not None:
        sigma_array = np.asarray(observed_sigma, dtype=float)
        if sigma_array.shape != y.shape:
            raise WholePatternError("Observed sigma must match the complete observed pattern.")
        sigma_full = sigma_array[mask]
    count_full = None
    if count_reference is not None:
        count_array = np.asarray(count_reference, dtype=float)
        if count_array.shape != y.shape:
            raise WholePatternError("Count reference must match the complete observed pattern.")
        count_full = count_array[mask]
    if len(x_full) < 50:
        raise WholePatternError(
            "The selected refinement range contains fewer than fifty points."
        )
    if instrument_profile:
        try:
            calibration_coverage = validate_calibration_range(
                instrument_profile,
                np.asarray([x_full[0], x_full[-1]]),
                allow_extrapolation=allow_profile_extrapolation,
            )
        except CalibrationRangeError as exc:
            raise WholePatternError(str(exc)) from exc
    else:
        calibration_coverage = None

    phases = _prepared_phases(
        phase_specs,
        wavelength_angstrom,
        float(x_full[0]),
        float(x_full[-1]),
        intensity_cutoff_percent,
    )

    if cancel_check is not None and cancel_check():
        raise WholePatternCancelled("Whole-pattern refinement was cancelled.")

    maximum_optimization_points = int(
        np.clip(maximum_optimization_points, 300, 6000)
    )
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
        raise WholePatternError(str(exc)) from exc

    if len(x_full) > maximum_optimization_points:
        indices = np.linspace(
            0,
            len(x_full) - 1,
            maximum_optimization_points,
            dtype=int,
        )
        x_fit = x_full[indices]
        y_fit = y_full[indices]
        weights_fit = full_weight_model.fit_sqrt_weights[indices]
    else:
        x_fit = x_full
        y_fit = y_full
        weights_fit = full_weight_model.fit_sqrt_weights
    background_fit = _background_basis(x_fit, background_order)
    fixed = {
        "zero_shift_deg": float(initial_zero_shift),
        "caglioti_u": max(float(initial_u), 0.0),
        "caglioti_v": float(initial_v),
        "caglioti_w": max(float(initial_w), 1e-8),
        "eta": float(np.clip(initial_eta, 0.0, 1.0)),
        "profile_model": str(profile_model),
        "lorentzian_x": max(float(initial_x), 0.0),
        "lorentzian_y": max(float(initial_y), 0.0),
        "axial_asymmetry": float(np.clip(initial_asymmetry, -0.75, 0.75)),
        "axial_sh_over_l": max(float(initial_axial_sh_over_l), 0.0),
    }
    (
        parameter_names,
        initial_values,
        lower_bounds,
        upper_bounds,
        mapping,
    ) = _build_parameterization(
        phases,
        refine_zero_shift=refine_zero_shift,
        refine_profile=refine_profile,
        refine_eta=refine_eta,
        initial_zero_shift=fixed["zero_shift_deg"],
        initial_u=fixed["caglioti_u"],
        initial_v=fixed["caglioti_v"],
        initial_w=fixed["caglioti_w"],
        initial_eta=fixed["eta"],
        profile_model=profile_model,
        refine_lorentzian_width=refine_lorentzian_width,
        refine_asymmetry=refine_asymmetry,
        initial_x=fixed["lorentzian_x"],
        initial_y=fixed["lorentzian_y"],
        initial_asymmetry=fixed["axial_asymmetry"],
        cell_tolerance_percent=cell_tolerance_percent,
    )
    calibration_prior = prepare_calibration_prior(
        instrument_profile,
        parameter_names,
    )
    calibration_prior_size = (
        0 if calibration_prior is None else int(calibration_prior["effective_rank"])
    )

    per_stage_evaluations = max(
        1,
        int(maximum_nonlinear_evaluations)
        if len(initial_values)
        else 1,
    )
    stage_groups = (
        staged_parameter_groups(parameter_names)
        if use_staged_refinement and len(initial_values)
        else []
    )
    total_evaluations = per_stage_evaluations * max(1, len(stage_groups))
    evaluation_counter = 0
    current_stage_label = "Joint refinement"
    stage_history: list[dict] = []
    last_solution: dict = {}

    def solve_at(
        values: np.ndarray,
        x_values: np.ndarray,
        y_values: np.ndarray,
        sqrt_weights: np.ndarray,
    ):
        parameter_state = _unpack_parameters(
            values,
            parameter_names,
            phases,
            mapping,
            fixed,
        )
        tangent = np.tan(np.radians(x_values / 2.0))
        profile_squared = (
            parameter_state["caglioti_u"] * tangent * tangent
            + parameter_state["caglioti_v"] * tangent
            + parameter_state["caglioti_w"]
        )
        if np.any(profile_squared <= 1e-10):
            raise WholePatternError(
                "U–V–W gives a non-positive width inside the refinement range."
            )
        reflections, warnings = _reflection_positions(
            phases,
            parameter_state,
            wavelength_angstrom,
            spectral_components=configured_radiation,
            instrument_profile=instrument_profile,
            apply_instrument_position_correction=apply_instrument_position_correction,
        )
        basis_arguments = {
            "u": parameter_state["caglioti_u"],
            "v": parameter_state["caglioti_v"],
            "w": parameter_state["caglioti_w"],
            "eta": parameter_state["eta"],
            "profile_model": parameter_state["profile_model"],
            "lorentzian_x": parameter_state["lorentzian_x"],
            "lorentzian_y": parameter_state["lorentzian_y"],
            "axial_asymmetry": parameter_state["axial_asymmetry"],
            "axial_sh_over_l": parameter_state["axial_sh_over_l"],
        }
        if use_gpu:
            basis, widths, basis_acceleration = _build_peak_basis_cuda(
                x_values, reflections, **basis_arguments
            )
        else:
            basis, widths = _build_peak_basis(
                x_values, reflections, **basis_arguments
            )
            basis_acceleration = {
                "requested": False,
                "used": False,
                "backend": "CPU",
                "device_name": "",
                "reason": "GPU acceleration is disabled for this run.",
            }
        background = _positive_background_basis(x_values, background_order)
        weights = np.asarray(sqrt_weights, dtype=float)
        if weights.shape != y_values.shape:
            raise WholePatternError("Refinement weights no longer match the observed pattern.")
        if mode == "Pawley decomposition":
            intensities, background_coefficients, calculated, extraction = _solve_pawley(
                basis,
                background,
                y_values,
                weights,
            )
            extraction = dict(extraction)
            extraction["acceleration"] = {
                "requested": bool(use_gpu),
                "used": False,
                "backend": "SciPy CPU",
                "device_name": "",
                "reason": (
                    "Pawley non-negative bounded least squares remains on CPU; "
                    "the reflection profile matrix can use CUDA."
                    if use_gpu else "GPU acceleration is disabled for this run."
                ),
            }
        else:
            if use_gpu:
                intensities, background_coefficients, calculated, extraction = _solve_le_bail_cuda(
                    basis,
                    background,
                    reflections,
                    y_values,
                    weights,
                    extraction_cycles,
                )
            else:
                intensities, background_coefficients, calculated, extraction = _solve_le_bail(
                    basis,
                    background,
                    reflections,
                    y_values,
                    weights,
                    extraction_cycles,
                )
                extraction = dict(extraction)
                extraction["acceleration"] = {
                    "requested": False,
                    "used": False,
                    "backend": "CPU",
                    "device_name": "",
                    "reason": "GPU acceleration is disabled for this run.",
                }
        return {
            "parameter_state": parameter_state,
            "reflections": reflections,
            "warnings": warnings,
            "basis": basis,
            "widths": widths,
            "intensities": intensities,
            "background_coefficients": background_coefficients,
            "background": background @ background_coefficients,
            "calculated": calculated,
            "weights": weights,
            "extraction_diagnostics": extraction,
            "acceleration": {
                "profile_basis": basis_acceleration,
                "intensity_extraction": extraction.get("acceleration", {}),
            },
        }

    def residual_function(values: np.ndarray) -> np.ndarray:
        nonlocal evaluation_counter, last_solution
        if cancel_check is not None and cancel_check():
            raise WholePatternCancelled(
                "Whole-pattern refinement was cancelled."
            )
        evaluation_counter += 1
        if progress_callback is not None:
            progress_callback(
                min(evaluation_counter, total_evaluations),
                total_evaluations,
                (
                    f"{current_stage_label}: {mode} evaluation "
                    f"{evaluation_counter}/{total_evaluations}"
                ),
            )
        try:
            solution = solve_at(values, x_fit, y_fit, weights_fit)
        except (WholePatternError, ValueError, np.linalg.LinAlgError):
            return np.full(len(y_fit) + calibration_prior_size, 1e6, dtype=float)
        last_solution = solution
        pattern_residual = (
            (y_fit - solution["calculated"]) * solution["weights"]
        )
        prior_residual = calibration_prior_residual(values, calibration_prior)
        return np.concatenate([pattern_residual, prior_residual])

    optimization_result = None
    if len(initial_values):
        try:
            if stage_groups:
                refined_values = initial_values.copy()
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
                        bounds=(lower_bounds[indices], upper_bounds[indices]),
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
                        "parameters": [parameter_names[index] for index in indices],
                        "success": bool(stage_result.success),
                        "message": str(stage_result.message),
                        "evaluations": int(stage_result.nfev),
                        "cost": float(stage_result.cost),
                    })
                    optimization_result = stage_result
            else:
                optimization_result = least_squares(
                    residual_function,
                    initial_values,
                    bounds=(lower_bounds, upper_bounds),
                    loss=robust_loss,
                    max_nfev=per_stage_evaluations,
                    ftol=1e-8,
                    xtol=1e-8,
                    gtol=1e-8,
                    x_scale="jac",
                )
                refined_values = optimization_result.x
        except WholePatternCancelled:
            raise
    else:
        refined_values = initial_values
        residual_function(refined_values)

    if cancel_check is not None and cancel_check():
        raise WholePatternCancelled("Whole-pattern refinement was cancelled.")

    final = solve_at(refined_values, x_full, y_full, full_weight_model.fit_sqrt_weights)
    observed = y_full
    calculated = final["calculated"]
    residual = observed - calculated
    background = final["background"]
    intensity_diagnostics = intensity_metrology(
        final["basis"],
        _positive_background_basis(x_full, background_order),
        observed,
        final["weights"],
        final["intensities"],
        final["background_coefficients"],
        statistics_valid=full_weight_model.statistics_valid,
        method=mode,
        overlap_correlation_threshold=overlap_correlation_threshold,
    )
    parameter_count = (
        len(parameter_names)
        + intensity_diagnostics["effective_independent_reflection_count"]
        + len(final["background_coefficients"])
    )
    profile_statistics = calculate_profile_statistics(
        observed, calculated, full_weight_model, parameter_count
    )
    rwp = profile_statistics["rwp_percent"]
    rp = profile_statistics["rp_percent"]
    rexp = profile_statistics["rexp_percent"]
    goodness_of_fit = profile_statistics["goodness_of_fit"]
    degrees_of_freedom = profile_statistics["degrees_of_freedom"]
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    durbin_watson = (
        float(np.sum(np.diff(residual) ** 2) / max(np.sum(residual ** 2), 1e-30))
        if len(residual) > 1
        else None
    )

    nonlinear_metrology = nonlinear_parameter_metrology(
        optimization_result,
        (
            np.asarray(optimization_result.fun, dtype=float)
            if optimization_result is not None
            else residual * final["weights"]
        ),
        parameter_names,
    )
    parameter_errors = nonlinear_metrology["standard_errors"]
    parameter_table = [
        {
            "parameter": name,
            "value": float(value),
            "standard_error": error,
        }
        for name, value, error in zip(
            parameter_names,
            refined_values,
            parameter_errors,
        )
    ]

    reflection_rows = []
    phase_totals: dict[int, float] = {}
    total_intensity = float(np.sum(final["intensities"]))
    for reflection, width, intensity, metrology_row in zip(
        final["reflections"],
        final["widths"],
        final["intensities"],
        intensity_diagnostics["reflection_diagnostics"],
    ):
        phase_index = int(reflection["phase_index"])
        phase_totals[phase_index] = (
            phase_totals.get(phase_index, 0.0) + float(intensity)
        )
        characteristics = profile_characteristics(
            float(reflection["two_theta_deg"]),
            model=final["parameter_state"]["profile_model"],
            u=final["parameter_state"]["caglioti_u"],
            v=final["parameter_state"]["caglioti_v"],
            w=final["parameter_state"]["caglioti_w"],
            eta=final["parameter_state"]["eta"],
            x=final["parameter_state"]["lorentzian_x"],
            y=final["parameter_state"]["lorentzian_y"],
            axial_asymmetry=final["parameter_state"]["axial_asymmetry"],
            axial_sh_over_l=final["parameter_state"]["axial_sh_over_l"],
            minimum_width=0.003,
        )
        reflection_rows.append(
            {
                **reflection,
                "fwhm_deg": float(width),
                "gaussian_fwhm_deg": float(characteristics.gaussian_fwhm_deg),
                "lorentzian_fwhm_deg": float(characteristics.lorentzian_fwhm_deg),
                "profile_eta": float(characteristics.eta),
                "left_fwhm_deg": float(characteristics.left_fwhm_deg),
                "right_fwhm_deg": float(characteristics.right_fwhm_deg),
                "extracted_intensity": float(intensity),
                "intensity_standard_error": metrology_row["standard_error"],
                "intensity_signal_to_uncertainty": metrology_row[
                    "signal_to_uncertainty"
                ],
                "intensity_identifiability": metrology_row["identifiability"],
                "individually_identifiable": metrology_row[
                    "individually_identifiable"
                ],
                "overlap_group_id": metrology_row["overlap_group_id"],
                "maximum_basis_correlation": metrology_row[
                    "maximum_basis_correlation"
                ],
                "intensity_fraction_percent": (
                    100.0 * float(intensity) / total_intensity
                    if total_intensity > 0
                    else 0.0
                ),
            }
        )

    phase_rows = []
    parameter_error_map = dict(zip(parameter_names, parameter_errors))
    for phase in phases:
        phase_index = phase["phase_index"]
        extracted = float(phase_totals.get(phase_index, 0.0))
        independent_cell_errors = {
            name: parameter_error_map.get(f"phase{phase_index}_{name}")
            for name in _cell_parameter_names(
                phase["initial_cell"], phase["crystal_system"]
            )
        }
        phase_rows.append(
            {
                "phase_index": phase_index,
                "reference_uid": phase["uid"],
                "phase_name": phase["name"],
                "formula": phase["formula"],
                "source": phase["source"],
                "refined_cell": final["parameter_state"]["phase_cells"].get(
                    phase_index
                ),
                "initial_cell": phase["initial_cell"],
                "crystal_system": phase["crystal_system"],
                "cell_refined": phase["refine_cell"],
                "refined_cell_standard_errors": _cell_standard_errors(
                    independent_cell_errors,
                    phase["crystal_system"],
                ),
                "reflection_count": sum(
                    1
                    for row in reflection_rows
                    if row["phase_index"] == phase_index
                ),
                "extracted_intensity_sum": extracted,
                "pattern_fraction_percent": (
                    100.0 * extracted / total_intensity
                    if total_intensity > 0
                    else 0.0
                ),
            }
        )

    basis_acceleration = dict(final.get("acceleration", {}).get("profile_basis") or {})
    extraction_acceleration = dict(
        final.get("acceleration", {}).get("intensity_extraction") or {}
    )
    gpu_used = bool(
        basis_acceleration.get("used") or extraction_acceleration.get("used")
    )
    acceleration = {
        "requested": bool(use_gpu),
        "calculation_backend": "CUDA (CuPy)" if gpu_used else "CPU",
        "device_name": (
            basis_acceleration.get("device_name")
            or extraction_acceleration.get("device_name")
            or ""
        ),
        "profile_basis": basis_acceleration,
        "intensity_extraction": extraction_acceleration,
        "nonlinear_optimizer_backend": "SciPy CPU",
        "pawley_intensity_solver_backend": "SciPy CPU bounded least squares",
        "le_bail_background_solver_backend": "SciPy CPU bounded least squares",
    }

    warnings = list(dict.fromkeys(final["warnings"]))
    warnings.extend(full_weight_model.warnings)
    for acceleration_part in (basis_acceleration, extraction_acceleration):
        reason = str(acceleration_part.get("reason") or "")
        if use_gpu and any(word in reason.lower() for word in ("failed", "unavailable")):
            warnings.append(reason)
    if not full_weight_model.statistics_valid:
        warnings.append(
            "Rexp, GoF and reduced chi-square are not reported because the selected intensity data do not have a valid absolute variance model. "
            + full_weight_model.reason
        )
    if len(reflection_rows) > max(300, len(x_fit) // 3):
        warnings.append(
            "The reflection count is high relative to the independent data "
            "points; individual extracted intensities may be strongly correlated."
        )
    if (
        intensity_diagnostics["effective_independent_reflection_count"]
        < len(reflection_rows)
    ):
        warnings.append(
            "The weighted reflection basis is rank deficient: not every listed "
            "reflection intensity is independently determined. Use overlap-group "
            "sums and the covariance diagnostics."
        )
    if intensity_diagnostics["unresolved_group_count"]:
        warnings.append(
            f"{intensity_diagnostics['unresolved_group_count']} reflection overlap "
            "group(s) are unresolved at the configured correlation gate; individual "
            "intensities in those groups are not identifiable."
        )
    if mode == "Le Bail-style extraction":
        warnings.append(
            "Le Bail intensity uncertainties are post-extraction linearized "
            "approximations; they are not a covariance from the iterative "
            "repartition itself."
        )
    if rwp > 20.0:
        warnings.append(
            "Rwp exceeds 20%; inspect missing phases, background, profile, "
            "wavelength, calibration, and preferred orientation."
        )
    if durbin_watson is not None and not 1.0 <= durbin_watson <= 3.0:
        warnings.append(
            "The difference curve is strongly serially correlated."
        )
    for phase in phase_rows:
        cell = phase.get("refined_cell")
        if phase["cell_refined"] and cell is None:
            warnings.append(
                f"{phase['phase_name']}: lattice refinement could not be evaluated."
            )
    if refine_zero_shift and any(phase["cell_refined"] for phase in phase_rows):
        warnings.append(
            "Global zero shift and lattice parameters can be correlated; inspect "
            "the refined cell and difference curve across the full angular range."
        )
    warnings.append(
        "Rexp and goodness of fit depend on the selected statistical weighting "
        "and the approximate independent-parameter count."
    )
    warnings.append(
        "Extracted pattern fractions are not weight fractions. Pawley/Le Bail "
        "decomposition does not refine atomic coordinates, occupancies, or "
        "thermal parameters."
    )

    profile_residual_diagnostics = residual_diagnostics(residual)
    profile_information_criteria = information_criteria(residual, parameter_count)
    refinement_plan = staged_profile_refinement_plan(
        profile_model=profile_model,
        has_instrument_profile=bool(initial_u or initial_v or initial_w),
        refine_cell=any(phase["refine_cell"] for phase in phases),
        refine_texture=False,
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
    if progress_callback is not None:
        progress_callback(
            total_evaluations,
            total_evaluations,
            "Whole-pattern refinement completed.",
        )

    if use_gpu:
        release_gpu_memory()

    return {
        "success": bool(
            optimization_result.success
            if optimization_result is not None
            else True
        ),
        "message": (
            optimization_result.message
            if optimization_result is not None
            else "Linear intensity/background solution completed."
        ),
        "mode": mode,
        "acceleration": acceleration,
        "wavelength_angstrom": float(wavelength_angstrom),
        "radiation": {
            "configuration": str(radiation_configuration),
            "components": configured_radiation,
            "composite_intensity_definition": (
                "Extracted intensity is the unit-area sum across normalized "
                "spectral components."
            ),
        },
        "apply_instrument_position_correction": bool(
            apply_instrument_position_correction and instrument_profile
        ),
        "two_theta_min_deg": float(x_full[0]),
        "two_theta_max_deg": float(x_full[-1]),
        "weighting": weighting,
        "weighting_model": full_weight_model.model_name,
        "intensity_provenance": full_weight_model.provenance,
        "statistics_valid": bool(full_weight_model.statistics_valid),
        "statistics_reason": full_weight_model.reason,
        "intensity_scale_factor": float(full_weight_model.intensity_scale_factor),
        "background_order": background_order,
        "staged_refinement_enabled": bool(use_staged_refinement),
        "staged_refinement_history": stage_history,
        "refine_zero_shift": bool(refine_zero_shift),
        "refine_profile": bool(refine_profile),
        "refine_eta": bool(refine_eta),
        "profile": {
            "model": str(final["parameter_state"]["profile_model"]),
            "caglioti_u": float(final["parameter_state"]["caglioti_u"]),
            "caglioti_v": float(final["parameter_state"]["caglioti_v"]),
            "caglioti_w": float(final["parameter_state"]["caglioti_w"]),
            "eta": float(final["parameter_state"]["eta"]),
            "lorentzian_x": float(final["parameter_state"]["lorentzian_x"]),
            "lorentzian_y": float(final["parameter_state"]["lorentzian_y"]),
            "axial_asymmetry": float(final["parameter_state"]["axial_asymmetry"]),
            "axial_sh_over_l": float(final["parameter_state"]["axial_sh_over_l"]),
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
        "zero_shift_deg": float(
            final["parameter_state"]["zero_shift_deg"]
        ),
        "parameters": parameter_table,
        "nonlinear_parameter_metrology": nonlinear_metrology,
        "intensity_metrology": intensity_diagnostics,
        "extraction_diagnostics": final["extraction_diagnostics"],
        "phases": phase_rows,
        "reflections": reflection_rows,
        "background_coefficients": final[
            "background_coefficients"
        ].tolist(),
        "observed_x": x_full.tolist(),
        "observed_y": observed.tolist(),
        "calculated_y": calculated.tolist(),
        "background_y": background.tolist(),
        "difference_y": residual.tolist(),
        "observed_sigma": (
            None if full_weight_model.sigma is None else full_weight_model.sigma.tolist()
        ),
        "statistical_weights": full_weight_model.reporting_weights.tolist(),
        "rwp_percent": float(rwp),
        "rp_percent": float(rp),
        "rexp_percent": None if rexp is None else float(rexp),
        "goodness_of_fit": (
            None if goodness_of_fit is None else float(goodness_of_fit)
        ),
        "goodness_of_fit_sqrt": profile_statistics["goodness_of_fit_sqrt"],
        "reduced_chi_square": profile_statistics["reduced_chi_square"],
        "weighted_residual_sum_squares": profile_statistics["weighted_residual_sum_squares"],
        "weighted_observed_sum_squares": profile_statistics["weighted_observed_sum_squares"],
        "degrees_of_freedom": int(degrees_of_freedom),
        "r_squared": float(r_squared),
        "rmse": rmse,
        "durbin_watson": durbin_watson,
        "data_point_count": len(observed),
        "optimization_point_count": len(x_fit),
        "reflection_count": len(reflection_rows),
        "effective_independent_reflection_count": intensity_diagnostics[
            "effective_independent_reflection_count"
        ],
        "individually_identifiable_reflection_count": intensity_diagnostics[
            "individually_identifiable_count"
        ],
        "unresolved_overlap_group_count": intensity_diagnostics[
            "unresolved_group_count"
        ],
        "parameter_count": parameter_count,
        "nonlinear_parameter_count": len(parameter_names),
        "nonlinear_evaluations": int(evaluation_counter),
        "elapsed_seconds": float(time.perf_counter() - started_at),
        "warnings": warnings,
    }
