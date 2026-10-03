from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from .crystallography import direct_metric_tensor, d_spacing
from .deep_unknown_indexing import (
    assign_peaks as _deep_assign_peaks,
    cluster_predicted_reflections as _deep_cluster_reflections,
    penalized_assignment_score as _deep_penalized_assignment_score,
    q_to_two_theta as _deep_q_to_two_theta,
    refine_reciprocal_metric as _deep_refine_reciprocal_metric,
    sigma_two_theta_to_sigma_q as _deep_sigma_two_theta_to_sigma_q,
)
from .indexing_reliability import (
    BootstrapReliabilitySettings,
    evaluate_candidate_reliability,
    summarize_candidate_reliability,
)
from .multiphase_unknown import (
    MultiphaseDiscoverySettings,
    classify_residual_peaks,
    joint_two_phase_assignment,
    stable_residual_peak_rows,
    summarize_multiphase_discovery,
)


class NativeIndexingError(RuntimeError):
    pass


class NativeIndexingCancelled(RuntimeError):
    pass


BRAVAIS_NAMES = (
    "Cubic-F",
    "Cubic-I",
    "Cubic-P",
    "Trigonal-R",
    "Trigonal/Hexagonal-P",
    "Tetragonal-I",
    "Tetragonal-P",
    "Orthorhombic-F",
    "Orthorhombic-I",
    "Orthorhombic-A",
    "Orthorhombic-B",
    "Orthorhombic-C",
    "Orthorhombic-P",
    "Monoclinic-I",
    "Monoclinic-A",
    "Monoclinic-C",
    "Monoclinic-P",
    "Triclinic",
)

CRYSTAL_SYSTEM_BY_BRAVAIS = {
    "Cubic-F": "Cubic",
    "Cubic-I": "Cubic",
    "Cubic-P": "Cubic",
    "Trigonal-R": "Rhombohedral",
    "Trigonal/Hexagonal-P": "Hexagonal",
    "Tetragonal-I": "Tetragonal",
    "Tetragonal-P": "Tetragonal",
    "Orthorhombic-F": "Orthorhombic",
    "Orthorhombic-I": "Orthorhombic",
    "Orthorhombic-A": "Orthorhombic",
    "Orthorhombic-B": "Orthorhombic",
    "Orthorhombic-C": "Orthorhombic",
    "Orthorhombic-P": "Orthorhombic",
    "Monoclinic-I": "Monoclinic",
    "Monoclinic-A": "Monoclinic",
    "Monoclinic-C": "Monoclinic",
    "Monoclinic-P": "Monoclinic",
    "Triclinic": "Triclinic",
}


@dataclass(frozen=True)
class NativeIndexingSettings:
    wavelength_angstrom: float = 1.5406
    bravais_names: tuple[str, ...] = BRAVAIS_NAMES[:13]
    starting_volume_angstrom3: float = 200.0
    minimum_cell_length_angstrom: float = 2.5
    maximum_cell_length_angstrom: float = 40.0
    minimum_angle_deg: float = 55.0
    maximum_angle_deg: float = 125.0
    zero_shift_deg: float = 0.0
    refine_zero_shift: bool = True
    maximum_zero_shift_deg: float = 0.25
    peak_tolerance_deg: float = 0.10
    impurity_tolerance_fraction: float = 0.15
    maximum_index: int = 10
    global_iterations: int = 32
    population_size: int = 7
    candidate_seeds_per_lattice: int = 5
    maximum_candidates_per_lattice: int = 4
    random_seed: int = 1600
    time_limit_seconds_per_lattice: float = 35.0
    minimum_indexed_fraction: float = 0.55
    minimum_indexed_peaks: int = 7
    leave_one_out_checks: int = 8

    def validate(self) -> "NativeIndexingSettings":
        if not np.isfinite(self.wavelength_angstrom) or self.wavelength_angstrom <= 0:
            raise NativeIndexingError("Wavelength must be finite and positive.")
        unknown = sorted(set(self.bravais_names) - set(BRAVAIS_NAMES))
        if unknown:
            raise NativeIndexingError("Unsupported Bravais lattices: " + ", ".join(unknown))
        if not self.bravais_names:
            raise NativeIndexingError("Select at least one Bravais lattice.")
        if self.minimum_cell_length_angstrom <= 0:
            raise NativeIndexingError("Minimum cell length must be positive.")
        if self.maximum_cell_length_angstrom <= self.minimum_cell_length_angstrom:
            raise NativeIndexingError("Maximum cell length must exceed the minimum.")
        if not 0 <= self.impurity_tolerance_fraction < 0.5:
            raise NativeIndexingError("Impurity tolerance must be between 0 and 0.5.")
        if self.minimum_indexed_peaks < 5:
            raise NativeIndexingError("At least five indexed peaks are required.")
        return self


@dataclass
class PeakObservation:
    peak_uuid: str
    two_theta_deg: float
    uncertainty_deg: float
    intensity: float
    quality_score: float
    signal_to_noise: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    return target


def _centering_allowed(bravais_name: str, h: int, k: int, l: int) -> bool:
    if bravais_name.endswith("-P") or bravais_name in {"Triclinic", "Trigonal/Hexagonal-P"}:
        return True
    if bravais_name.endswith("-I"):
        return (h + k + l) % 2 == 0
    if bravais_name.endswith("-F"):
        parity = (h & 1, k & 1, l & 1)
        return parity == (0, 0, 0) or parity == (1, 1, 1)
    if bravais_name.endswith("-A"):
        return (k + l) % 2 == 0
    if bravais_name.endswith("-B"):
        return (h + l) % 2 == 0
    if bravais_name.endswith("-C"):
        return (h + k) % 2 == 0
    if bravais_name == "Trigonal-R":
        return (-h + k + l) % 3 == 0
    return True


def _peak_observations(peaks: Iterable[dict]) -> list[PeakObservation]:
    rows = []
    for index, row in enumerate(peaks):
        if not bool(row.get("use", True)):
            continue
        position = float(row.get("two_theta_deg", math.nan))
        if not np.isfinite(position) or not 0.0 < position < 180.0:
            continue
        uncertainty = float(row.get("position_uncertainty_deg", 0.02) or 0.02)
        uncertainty = float(np.clip(uncertainty, 0.001, 0.5))
        rows.append(
            PeakObservation(
                peak_uuid=str(row.get("peak_uuid") or f"peak-{index + 1}"),
                two_theta_deg=position,
                uncertainty_deg=uncertainty,
                intensity=max(0.0, float(row.get("intensity", 1.0) or 1.0)),
                quality_score=float(row.get("quality_score", 50.0) or 50.0),
                signal_to_noise=float(row.get("signal_to_noise", 1.0) or 1.0),
            )
        )
    rows.sort(key=lambda item: item.two_theta_deg)
    if len(rows) < 7:
        raise NativeIndexingError("At least seven enabled, finite reflections are required.")
    return rows


def _q_from_two_theta(two_theta_deg: np.ndarray, wavelength: float) -> np.ndarray:
    theta = np.radians(np.asarray(two_theta_deg, dtype=float) / 2.0)
    return np.square(2.0 * np.sin(theta) / float(wavelength))


def _two_theta_from_q(q: np.ndarray, wavelength: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    argument = 0.5 * float(wavelength) * np.sqrt(np.maximum(q, 0.0))
    result = np.full_like(argument, np.nan)
    valid = (argument > 0.0) & (argument < 1.0)
    result[valid] = np.degrees(2.0 * np.arcsin(argument[valid]))
    return result


def _cell_volume(cell: dict) -> float:
    metric = direct_metric_tensor(cell)
    determinant = float(np.linalg.det(metric))
    if not np.isfinite(determinant) or determinant <= 0:
        return math.nan
    return float(math.sqrt(determinant))


def _parameter_names(system: str) -> tuple[str, ...]:
    if system == "Cubic":
        return ("a",)
    if system in {"Tetragonal", "Hexagonal"}:
        return ("a", "c")
    if system == "Orthorhombic":
        return ("a", "b", "c")
    if system == "Rhombohedral":
        return ("a", "alpha")
    if system == "Monoclinic":
        return ("a", "b", "c", "beta")
    return ("a", "b", "c", "alpha", "beta", "gamma")


def _cell_from_parameters(system: str, values: Iterable[float]) -> dict:
    values = [float(value) for value in values]
    if system == "Cubic":
        a = values[0]
        return {"a": a, "b": a, "c": a, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    if system == "Tetragonal":
        a, c = values
        return {"a": a, "b": a, "c": c, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    if system == "Hexagonal":
        a, c = values
        return {"a": a, "b": a, "c": c, "alpha": 90.0, "beta": 90.0, "gamma": 120.0}
    if system == "Orthorhombic":
        a, b, c = values
        return {"a": a, "b": b, "c": c, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    if system == "Rhombohedral":
        a, alpha = values
        return {"a": a, "b": a, "c": a, "alpha": alpha, "beta": alpha, "gamma": alpha}
    if system == "Monoclinic":
        a, b, c, beta = values
        return {"a": a, "b": b, "c": c, "alpha": 90.0, "beta": beta, "gamma": 90.0}
    a, b, c, alpha, beta, gamma = values
    return {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}




def _values_from_cell(system: str, cell: dict, zero_shift: float | None = None) -> np.ndarray:
    names = _parameter_names(system)
    values = []
    for name in names:
        if name in {"a", "b", "c"}:
            values.append(float(cell[name]))
        else:
            values.append(float(cell[name]))
    if zero_shift is not None:
        values.append(float(zero_shift))
    return np.asarray(values, dtype=float)


def _deep_system_name(system: str) -> str | None:
    if system in {"Cubic", "Tetragonal", "Hexagonal", "Orthorhombic", "Triclinic"}:
        return system.lower()
    if system == "Monoclinic":
        return "monoclinic_b"
    return None

def _parameter_bounds(system: str, settings: NativeIndexingSettings) -> list[tuple[float, float]]:
    length = (settings.minimum_cell_length_angstrom, settings.maximum_cell_length_angstrom)
    angle = (settings.minimum_angle_deg, settings.maximum_angle_deg)
    if system == "Cubic":
        return [length]
    if system in {"Tetragonal", "Hexagonal"}:
        return [length, length]
    if system == "Orthorhombic":
        return [length, length, length]
    if system == "Rhombohedral":
        return [length, angle]
    if system == "Monoclinic":
        return [length, length, length, angle]
    return [length, length, length, angle, angle, angle]


def _predicted_reflections(
    cell: dict,
    bravais_name: str,
    q_max: float,
    maximum_index: int,
) -> list[dict]:
    rows: list[dict] = []
    for h in range(maximum_index + 1):
        for k in range(maximum_index + 1):
            for l in range(maximum_index + 1):
                if h == k == l == 0 or not _centering_allowed(bravais_name, h, k, l):
                    continue
                try:
                    spacing = float(d_spacing(cell, (h, k, l)))
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if not np.isfinite(spacing) or spacing <= 0:
                    continue
                q = 1.0 / (spacing * spacing)
                if q <= q_max * 1.08:
                    rows.append({"h": h, "k": k, "l": l, "q": q, "d_spacing_angstrom": spacing})
    rows.sort(key=lambda row: row["q"])
    # Merge exact degeneracies, retaining a representative and multiplicity count.
    merged: list[dict] = []
    for row in rows:
        if merged and abs(row["q"] - merged[-1]["q"]) <= max(1e-10, 1e-7 * row["q"]):
            merged[-1]["multiplicity"] += 1
            merged[-1]["equivalent_hkls"].append([row["h"], row["k"], row["l"]])
        else:
            merged.append({**row, "multiplicity": 1, "equivalent_hkls": [[row["h"], row["k"], row["l"]]]})
    return merged


def _greedy_match(observed_q: np.ndarray, predicted: list[dict], tolerance_q: np.ndarray) -> dict:
    """Assign observed peaks to calculated reflection clusters globally.

    The historical name is retained for compatibility with older code paths, but
    Phase 17.7 uses a Hungarian one-to-one assignment rather than a local
    nearest-neighbour greedy match. This prevents one crowded low-symmetry cell
    from repeatedly consuming the same local observations.
    """
    if not predicted:
        return {"matches": [], "unmatched_observed": list(range(len(observed_q))), "unmatched_predicted": []}
    clustered = _deep_cluster_reflections(predicted, np.asarray(tolerance_q, dtype=float))
    predicted_q = np.asarray([row["q"] for row in clustered], dtype=float)
    assignment = _deep_assign_peaks(
        observed_q,
        tolerance_q,
        predicted_q,
        outlier_cost=8.0,
        model_sigma_q=float(np.nanmedian(tolerance_q)) * 0.25 if len(tolerance_q) else 0.0,
    )
    matches = []
    for row in assignment["matches"]:
        matches.append(
            {
                "observed_index": int(row["observed_index"]),
                "predicted_index": int(row["predicted_index"]),
                "delta_q": float(row["delta_q"]),
                "normalized_delta": float(abs(row["normalized_delta"])),
                "assignment_cost": float(row.get("assignment_cost", 0.0)),
            }
        )
    matches.sort(key=lambda row: row["observed_index"])
    return {
        "matches": matches,
        "unmatched_observed": assignment["unindexed"],
        "unmatched_predicted": assignment["unmatched_calculated"],
        "assignment_cost": assignment["assignment_cost"],
        "predicted_clusters": clustered,
    }


def _observed_q_and_tolerance(
    observations: list[PeakObservation],
    wavelength: float,
    zero_shift_deg: float,
    base_tolerance_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray([row.two_theta_deg - zero_shift_deg for row in observations], dtype=float)
    positions = np.clip(positions, 0.001, 179.0)
    q = _q_from_two_theta(positions, wavelength)
    uncertainties = np.asarray(
        [max(base_tolerance_deg, row.uncertainty_deg) for row in observations],
        dtype=float,
    )
    q_plus = _q_from_two_theta(positions + uncertainties, wavelength)
    q_minus = _q_from_two_theta(np.maximum(0.001, positions - uncertainties), wavelength)
    tolerance_q = np.maximum(0.5 * np.abs(q_plus - q_minus), np.maximum(1e-7, q * 1e-4))
    return positions, q, tolerance_q


def _evaluate_parameters(
    parameter_vector: np.ndarray,
    system: str,
    bravais_name: str,
    observations: list[PeakObservation],
    settings: NativeIndexingSettings,
) -> tuple[float, dict]:
    parameter_count = len(_parameter_names(system))
    cell = _cell_from_parameters(system, parameter_vector[:parameter_count])
    volume = _cell_volume(cell)
    if not np.isfinite(volume) or volume <= 1.0:
        return 1e9, {}
    zero_shift = (
        float(parameter_vector[-1])
        if settings.refine_zero_shift
        else float(settings.zero_shift_deg)
    )
    positions, observed_q, tolerance_q = _observed_q_and_tolerance(
        observations,
        settings.wavelength_angstrom,
        zero_shift,
        settings.peak_tolerance_deg,
    )
    predicted = _predicted_reflections(
        cell,
        bravais_name,
        float(np.max(observed_q)),
        settings.maximum_index,
    )
    if len(predicted) < settings.minimum_indexed_peaks:
        return 1e8, {}
    matching = _greedy_match(observed_q, predicted, tolerance_q)
    predicted_clusters = matching.get("predicted_clusters", predicted)
    normalized = np.asarray([row["normalized_delta"] for row in matching["matches"]], dtype=float)
    impurity_allowance = int(math.floor(len(observations) * settings.impurity_tolerance_fraction))
    required = max(settings.minimum_indexed_peaks, len(observations) - impurity_allowance)
    indexed = len(normalized)
    if indexed:
        robust_residual = float(np.mean(np.square(np.minimum(normalized, 6.0))))
    else:
        robust_residual = 36.0
    missing_penalty = 18.0 * max(0, required - indexed) / max(1, required)
    density_penalty = 0.0
    if settings.starting_volume_angstrom3 > 0:
        ratio = max(volume, settings.starting_volume_angstrom3) / min(volume, settings.starting_volume_angstrom3)
        density_penalty = 0.30 * math.log(ratio) ** 2
    deep_score = _deep_penalized_assignment_score(
        {"matches": matching["matches"], "unindexed": matching["unmatched_observed"]},
        observed_count=len(observations),
        calculated_count=len(predicted_clusters),
        free_parameter_count=len(_parameter_names(system)) + int(settings.refine_zero_shift),
    )
    # The historical score is retained but combined with the Phase 17.7
    # line-density and model-complexity score so oversized low-symmetry cells
    # cannot win merely by generating many calculated lines.
    objective = robust_residual + missing_penalty + density_penalty + deep_score["deep_objective"]
    details = {
        "cell": cell,
        "volume_angstrom3": volume,
        "zero_shift_deg": zero_shift,
        "positions": positions,
        "observed_q": observed_q,
        "tolerance_q": tolerance_q,
        "predicted": predicted_clusters,
        "raw_predicted_reflections": len(predicted),
        "matching": matching,
        "deep_score": deep_score,
        "objective": float(objective),
    }
    return float(objective), details


def _local_refine(
    values: np.ndarray,
    bounds: list[tuple[float, float]],
    system: str,
    bravais_name: str,
    observations: list[PeakObservation],
    settings: NativeIndexingSettings,
) -> tuple[np.ndarray, dict]:
    objective, initial = _evaluate_parameters(values, system, bravais_name, observations, settings)
    if not initial:
        return values, {}
    initial_matches = initial["matching"]["matches"]
    if len(initial_matches) < settings.minimum_indexed_peaks:
        return values, initial

    # Phase 17.7 weighted reciprocal-metric refinement for fixed hkl
    # assignments. For supported crystal systems, this supplies a stable linear
    # SVD update before the nonlinear angular refinement.
    deep_system = _deep_system_name(system)
    if deep_system is not None:
        try:
            observed_indices_for_metric = [row["observed_index"] for row in initial_matches]
            metric_positions = np.asarray([observations[index].two_theta_deg - initial["zero_shift_deg"] for index in observed_indices_for_metric], dtype=float)
            metric_sigmas = np.asarray([max(0.001, observations[index].uncertainty_deg) for index in observed_indices_for_metric], dtype=float)
            metric_q = _q_from_two_theta(metric_positions, settings.wavelength_angstrom)
            metric_sigma_q = _deep_sigma_two_theta_to_sigma_q(metric_positions, metric_sigmas, settings.wavelength_angstrom)
            metric_hkls = np.asarray([[initial["predicted"][row["predicted_index"]][key] for key in ("h", "k", "l")] for row in initial_matches], dtype=int)
            metric = _deep_refine_reciprocal_metric(metric_q, metric_sigma_q, metric_hkls, deep_system)
            metric_cell = metric.get("cell")
            if metric_cell and metric["summary"].get("positive_definite") and metric["summary"].get("rank", 0) >= metric["summary"].get("parameter_count", 999):
                metric_vector = _values_from_cell(system, metric_cell, initial["zero_shift_deg"] if settings.refine_zero_shift else None)
                if len(metric_vector) == len(values):
                    metric_vector = np.asarray([np.clip(metric_vector[i], bounds[i][0], bounds[i][1]) for i in range(len(metric_vector))], dtype=float)
                    _, metric_details = _evaluate_parameters(metric_vector, system, bravais_name, observations, settings)
                    if metric_details and metric_details.get("objective", math.inf) <= initial.get("objective", math.inf) * 1.20:
                        metric_details["metric_refinement"] = metric["summary"]
                        values = metric_vector
                        initial = metric_details
                        initial_matches = initial["matching"]["matches"]
        except Exception:
            pass

    hkls = [
        initial["predicted"][row["predicted_index"]]
        for row in initial_matches
    ]
    observed_indices = [row["observed_index"] for row in initial_matches]
    observed_positions = np.asarray([observations[index].two_theta_deg for index in observed_indices], dtype=float)
    sigma = np.asarray(
        [max(settings.peak_tolerance_deg, observations[index].uncertainty_deg) for index in observed_indices],
        dtype=float,
    )
    parameter_count = len(_parameter_names(system))

    def residual(vector: np.ndarray) -> np.ndarray:
        cell = _cell_from_parameters(system, vector[:parameter_count])
        zero = float(vector[-1]) if settings.refine_zero_shift else settings.zero_shift_deg
        calculated = []
        for reflection in hkls:
            try:
                spacing = d_spacing(cell, (reflection["h"], reflection["k"], reflection["l"]))
            except (ValueError, np.linalg.LinAlgError):
                return np.full(len(hkls), 1e3)
            argument = settings.wavelength_angstrom / (2.0 * spacing)
            if not 0.0 < argument < 1.0:
                return np.full(len(hkls), 1e3)
            calculated.append(math.degrees(2.0 * math.asin(argument)) + zero)
        return (observed_positions - np.asarray(calculated, dtype=float)) / sigma

    lower = np.asarray([row[0] for row in bounds], dtype=float)
    upper = np.asarray([row[1] for row in bounds], dtype=float)
    try:
        result = least_squares(
            residual,
            np.clip(values, lower, upper),
            bounds=(lower, upper),
            loss="soft_l1",
            max_nfev=1800,
        )
    except (ValueError, np.linalg.LinAlgError):
        return values, initial
    refined_values = result.x if result.success else values
    _, details = _evaluate_parameters(refined_values, system, bravais_name, observations, settings)
    return refined_values, details or initial


def _figures_of_merit(details: dict, observation_count: int, wavelength: float) -> dict:
    matches = details["matching"]["matches"]
    predicted = details["predicted"]
    observed_q = details["observed_q"]
    indexed = len(matches)
    if not matches:
        return {"m20": 0.0, "f20": 0.0, "x20": min(20, observation_count)}
    first_twenty = [row for row in matches if row["observed_index"] < min(20, observation_count)]
    deltas_q = np.asarray([abs(row["delta_q"]) for row in first_twenty or matches], dtype=float)
    q_limit_index = min(19, observation_count - 1)
    q_limit = float(observed_q[q_limit_index])
    predicted_count = max(1, sum(row["q"] <= q_limit for row in predicted))
    mean_delta_q = max(float(np.mean(deltas_q)), 1e-12)
    m20 = q_limit / (2.0 * mean_delta_q * predicted_count)
    # Smith-Snyder-style diagnostic. This is deliberately labelled approximate.
    obs_positions = _two_theta_from_q(observed_q, wavelength)
    position_deltas = []
    for row in first_twenty or matches:
        predicted_position = _two_theta_from_q(
            np.asarray([predicted[row["predicted_index"]]["q"]]), wavelength
        )[0]
        position_deltas.append(abs(obs_positions[row["observed_index"]] - predicted_position))
    mean_position_delta = max(float(np.mean(position_deltas)), 1e-6)
    f20 = indexed / (mean_position_delta * predicted_count)
    matched_first_twenty = {row["observed_index"] for row in first_twenty}
    x20 = sum(index not in matched_first_twenty for index in range(min(20, observation_count)))
    return {"m20": float(m20), "f20": float(f20), "x20": int(x20)}


def _reduced_cell_fingerprint(cell: dict) -> tuple[float, ...]:
    metric = direct_metric_tensor(cell)
    eigenvalues = np.sort(np.linalg.eigvalsh(metric))
    volume = _cell_volume(cell)
    normalized = eigenvalues / max(float(np.cbrt(volume * volume)), 1e-12)
    off_diagonal = np.sort(np.abs(metric[np.triu_indices(3, 1)])) / max(float(np.mean(np.diag(metric))), 1e-12)
    return tuple(np.round(np.concatenate([normalized, off_diagonal, [math.log(max(volume, 1e-12))]]), 4))


def _candidate_from_details(
    details: dict,
    bravais_name: str,
    observations: list[PeakObservation],
    settings: NativeIndexingSettings,
) -> dict:
    cell = details["cell"]
    figures = _figures_of_merit(details, len(observations), settings.wavelength_angstrom)
    matches = details["matching"]["matches"]
    assignments = []
    predicted = details["predicted"]
    positions = details["positions"]
    for row in matches:
        observation = observations[row["observed_index"]]
        reflection = predicted[row["predicted_index"]]
        predicted_position = float(
            _two_theta_from_q(np.asarray([reflection["q"]]), settings.wavelength_angstrom)[0]
            + details["zero_shift_deg"]
        )
        assignments.append(
            {
                "peak_uuid": observation.peak_uuid,
                "observed_two_theta_deg": observation.two_theta_deg,
                "corrected_two_theta_deg": float(positions[row["observed_index"]]),
                "predicted_two_theta_deg": predicted_position,
                "delta_two_theta_deg": float(observation.two_theta_deg - predicted_position),
                "h": int(reflection["h"]),
                "k": int(reflection["k"]),
                "l": int(reflection["l"]),
                "hkl_label": f"({reflection['h']} {reflection['k']} {reflection['l']})",
                "d_spacing_angstrom": float(reflection["d_spacing_angstrom"]),
                "normalized_delta": float(row["normalized_delta"]),
            }
        )
    indexed = len(assignments)
    coverage = indexed / len(observations)
    rms_position = float(
        math.sqrt(np.mean([row["delta_two_theta_deg"] ** 2 for row in assignments]))
    ) if assignments else math.inf
    candidate = {
        "engine": "Afruz native powder indexer",
        "algorithm": "deep Q-space assignment + weighted SVD reciprocal metric + robust angular refinement + bootstrap reliability screen",
        "bravais_name": bravais_name,
        "crystal_system": CRYSTAL_SYSTEM_BY_BRAVAIS[bravais_name],
        "a_angstrom": float(cell["a"]),
        "b_angstrom": float(cell["b"]),
        "c_angstrom": float(cell["c"]),
        "alpha_deg": float(cell["alpha"]),
        "beta_deg": float(cell["beta"]),
        "gamma_deg": float(cell["gamma"]),
        "volume_angstrom3": float(details["volume_angstrom3"]),
        "zero_shift_deg": float(details["zero_shift_deg"]),
        "objective": float(details["objective"]),
        "deep_objective": float(details.get("deep_score", {}).get("deep_objective", details["objective"])),
        "bic_like": float(details.get("deep_score", {}).get("bic_like", details["objective"])),
        "line_density_ratio": float(details.get("deep_score", {}).get("line_density_ratio", 0.0)),
        "unindexed_observed_fraction": float(details.get("deep_score", {}).get("unindexed_observed_fraction", 0.0)),
        "assignment_method": "Hungarian one-to-one Q-space assignment",
        "metric_refinement": details.get("metric_refinement", {}),
        "m20": figures["m20"],
        "f20": figures["f20"],
        "x20": figures["x20"],
        "indexed_peak_count": indexed,
        "observed_peak_count": len(observations),
        "indexed_fraction": float(coverage),
        "position_rmse_deg": rms_position,
        "unindexed_peak_uuids": [
            observations[index].peak_uuid
            for index in details["matching"]["unmatched_observed"]
        ],
        "assignments": assignments,
        "reduced_cell_fingerprint": list(_reduced_cell_fingerprint(cell)),
        "figure_of_merit_warning": (
            "M20 and F20 are Afruz diagnostic implementations for candidate ranking. "
            "Final indexing must be verified by whole-pattern decomposition and competing-cell tests."
        ),
    }
    return candidate


def _leave_one_out_stability(
    candidate: dict,
    observations: list[PeakObservation],
    settings: NativeIndexingSettings,
) -> dict:
    assignments = candidate.get("assignments", [])
    if len(assignments) < settings.minimum_indexed_peaks + 1:
        return {"status": "Insufficient assigned peaks", "checks": 0, "success_fraction": 0.0}
    system = candidate["crystal_system"]
    bravais = candidate["bravais_name"]
    names = _parameter_names(system)
    initial = np.asarray([candidate[f"{name}_angstrom"] if name in {"a", "b", "c"} else candidate[f"{name}_deg"] for name in names], dtype=float)
    if settings.refine_zero_shift:
        initial = np.append(initial, candidate["zero_shift_deg"])
    bounds = _parameter_bounds(system, settings)
    if settings.refine_zero_shift:
        bounds.append((-settings.maximum_zero_shift_deg, settings.maximum_zero_shift_deg))
    checks = min(settings.leave_one_out_checks, len(observations))
    selected = np.linspace(0, len(observations) - 1, checks, dtype=int)
    volumes = []
    rms_values = []
    for omitted in selected:
        subset = [row for index, row in enumerate(observations) if index != int(omitted)]
        _, details = _local_refine(initial, bounds, system, bravais, subset, settings)
        if not details or len(details["matching"]["matches"]) < settings.minimum_indexed_peaks:
            continue
        volumes.append(details["volume_angstrom3"])
        trial = _candidate_from_details(details, bravais, subset, settings)
        rms_values.append(trial["position_rmse_deg"])
    success_fraction = len(volumes) / max(1, checks)
    volume_cv = (
        100.0 * float(np.std(volumes, ddof=1)) / float(np.mean(volumes))
        if len(volumes) > 1 and np.mean(volumes) > 0
        else 0.0
    )
    if success_fraction >= 0.8 and volume_cv <= 1.0:
        status = "Stable"
    elif success_fraction >= 0.5 and volume_cv <= 3.0:
        status = "Review"
    else:
        status = "Unstable"
    return {
        "status": status,
        "checks": checks,
        "successful_checks": len(volumes),
        "success_fraction": float(success_fraction),
        "volume_cv_percent": float(volume_cv),
        "median_position_rmse_deg": float(np.median(rms_values)) if rms_values else None,
    }


def _quick_linear_seed_score(
    observed_q: np.ndarray,
    predicted_q: np.ndarray,
    impurity_fraction: float,
) -> float:
    predicted_q = np.unique(np.round(predicted_q[predicted_q > 0], 12))
    if len(predicted_q) < 5:
        return 1e9
    pairs = []
    for i, q_value in enumerate(observed_q):
        j = int(np.argmin(np.abs(predicted_q - q_value)))
        pairs.append((abs(float(predicted_q[j] - q_value)), i, j))
    pairs.sort()
    used_i = set()
    used_j = set()
    deltas = []
    for delta, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        deltas.append(delta / max(float(observed_q[i]), 1e-8))
    required = max(5, len(observed_q) - int(math.floor(len(observed_q) * impurity_fraction)))
    indexed = len(deltas)
    residual = float(np.mean(np.square(np.minimum(np.asarray(deltas), 0.03) / 0.002))) if deltas else 100.0
    return residual + 15.0 * max(0, required - indexed) / required + 0.05 * max(0, len(predicted_q) - indexed)


def _deterministic_seed_vectors(
    system: str,
    bravais_name: str,
    observations: list[PeakObservation],
    settings: NativeIndexingSettings,
    bounds: list[tuple[float, float]],
) -> list[np.ndarray]:
    _, observed_q, _ = _observed_q_and_tolerance(
        observations,
        settings.wavelength_angstrom,
        settings.zero_shift_deg,
        settings.peak_tolerance_deg,
    )
    vectors: list[tuple[float, np.ndarray]] = []
    q_max = float(np.max(observed_q)) * 1.08

    if system == "Cubic":
        allowed_n = sorted({
            h * h + k * k + l * l
            for h in range(settings.maximum_index + 1)
            for k in range(settings.maximum_index + 1)
            for l in range(settings.maximum_index + 1)
            if (h or k or l) and _centering_allowed(bravais_name, h, k, l)
        })
        for q_value in observed_q[: min(8, len(observed_q))]:
            for n_value in allowed_n[: min(45, len(allowed_n))]:
                a_value = math.sqrt(float(n_value) / float(q_value))
                if not bounds[0][0] <= a_value <= bounds[0][1]:
                    continue
                predicted_q = np.asarray([n / (a_value * a_value) for n in allowed_n], dtype=float)
                predicted_q = predicted_q[predicted_q <= q_max]
                score = _quick_linear_seed_score(observed_q, predicted_q, settings.impurity_tolerance_fraction)
                vector = [a_value]
                if settings.refine_zero_shift:
                    vector.append(settings.zero_shift_deg)
                vectors.append((score, np.asarray(vector, dtype=float)))

    elif system in {"Tetragonal", "Hexagonal"}:
        coefficients = set()
        for h in range(settings.maximum_index + 1):
            for k in range(settings.maximum_index + 1):
                for l in range(settings.maximum_index + 1):
                    if h == k == l == 0 or not _centering_allowed(bravais_name, h, k, l):
                        continue
                    if system == "Tetragonal":
                        xy = float(h * h + k * k)
                    else:
                        xy = float((4.0 / 3.0) * (h * h + h * k + k * k))
                    coefficients.add((xy, float(l * l)))
        coefficient_rows = sorted(coefficients, key=lambda row: (row[0] + row[1], row[1], row[0]))[:24]
        obs_limit = min(7, len(observed_q))
        seen = set()
        for first_obs in range(obs_limit - 1):
            for second_obs in range(first_obs + 1, obs_limit):
                target = np.asarray([observed_q[first_obs], observed_q[second_obs]], dtype=float)
                for first_coeff in range(len(coefficient_rows) - 1):
                    for second_coeff in range(first_coeff + 1, len(coefficient_rows)):
                        matrix = np.asarray([coefficient_rows[first_coeff], coefficient_rows[second_coeff]], dtype=float)
                        determinant = float(np.linalg.det(matrix))
                        if abs(determinant) < 1e-10:
                            continue
                        reciprocal = np.linalg.solve(matrix, target)
                        if np.any(reciprocal <= 0):
                            continue
                        a_value = 1.0 / math.sqrt(float(reciprocal[0]))
                        c_value = 1.0 / math.sqrt(float(reciprocal[1]))
                        if not (bounds[0][0] <= a_value <= bounds[0][1] and bounds[1][0] <= c_value <= bounds[1][1]):
                            continue
                        key = (round(a_value, 4), round(c_value, 4))
                        if key in seen:
                            continue
                        seen.add(key)
                        predicted_q = np.asarray([
                            reciprocal[0] * xy + reciprocal[1] * zz
                            for xy, zz in coefficient_rows
                        ], dtype=float)
                        predicted_q = predicted_q[predicted_q <= q_max]
                        score = _quick_linear_seed_score(observed_q, predicted_q, settings.impurity_tolerance_fraction)
                        vector = [a_value, c_value]
                        if settings.refine_zero_shift:
                            vector.append(settings.zero_shift_deg)
                        vectors.append((score, np.asarray(vector, dtype=float)))
    vectors.sort(key=lambda row: row[0])
    unique = []
    seen_vectors = set()
    for _, vector in vectors:
        key = tuple(np.round(vector, 5))
        if key in seen_vectors:
            continue
        seen_vectors.add(key)
        unique.append(vector)
        if len(unique) >= 12:
            break
    return unique


def _search_lattice(
    bravais_name: str,
    observations: list[PeakObservation],
    settings: NativeIndexingSettings,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    lattice_index: int = 0,
    lattice_total: int = 1,
) -> dict:
    if cancel_check and cancel_check():
        raise NativeIndexingCancelled("Native indexing cancelled.")
    system = CRYSTAL_SYSTEM_BY_BRAVAIS[bravais_name]
    if progress_callback:
        progress_callback(lattice_index, lattice_total, f"Searching {bravais_name}")
    bounds = _parameter_bounds(system, settings)
    if settings.refine_zero_shift:
        bounds.append((-settings.maximum_zero_shift_deg, settings.maximum_zero_shift_deg))
    start_time = time.monotonic()
    candidates: list[dict] = []
    fingerprints: set[tuple[float, ...]] = set()

    # Deterministic reciprocal-metric seeds prevent primitive high-symmetry
    # cells from being missed or replaced by compatible supercells.
    seed_vectors = _deterministic_seed_vectors(
        system, bravais_name, observations, settings, bounds
    )
    for vector in seed_vectors:
        refined_values, details = _local_refine(
            vector, bounds, system, bravais_name, observations, settings
        )
        if not details:
            continue
        candidate = _candidate_from_details(
            details, bravais_name, observations, settings
        )
        if candidate["indexed_peak_count"] < settings.minimum_indexed_peaks:
            continue
        fingerprint = tuple(candidate["reduced_cell_fingerprint"])
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        candidates.append(candidate)
    candidates.sort(
        key=lambda row: (row["objective"], -row["indexed_fraction"], -row["m20"])
    )
    candidates = candidates[: settings.maximum_candidates_per_lattice]

    def objective(vector: np.ndarray) -> float:
        if cancel_check and cancel_check():
            raise NativeIndexingCancelled("Native indexing cancelled.")
        if time.monotonic() - start_time > settings.time_limit_seconds_per_lattice:
            return 1e6
        value, _ = _evaluate_parameters(vector, system, bravais_name, observations, settings)
        return value

    deterministic_solution_is_strong = bool(
        candidates
        and candidates[0].get("indexed_fraction", 0.0)
            >= max(settings.minimum_indexed_fraction, 1.0 - settings.impurity_tolerance_fraction)
        and candidates[0].get("position_rmse_deg", math.inf)
            <= max(0.01, settings.peak_tolerance_deg * 0.35)
        and candidates[0].get("objective", math.inf) <= 2.5
    )

    for seed_offset in range(
        0 if deterministic_solution_is_strong else settings.candidate_seeds_per_lattice
    ):
        if cancel_check and cancel_check():
            raise NativeIndexingCancelled("Native indexing cancelled.")
        if time.monotonic() - start_time > settings.time_limit_seconds_per_lattice:
            break
        try:
            result = differential_evolution(
                objective,
                bounds,
                seed=settings.random_seed + 101 * lattice_index + seed_offset,
                maxiter=settings.global_iterations,
                popsize=settings.population_size,
                polish=False,
                updating="immediate",
                workers=1,
                tol=1e-6,
            )
        except NativeIndexingCancelled:
            raise
        except Exception as exc:
            return {
                "bravais_name": bravais_name,
                "status": "Failed",
                "error": f"Global search failed: {exc}",
                "candidates": [],
                "elapsed_seconds": time.monotonic() - start_time,
            }
        refined_values, details = _local_refine(result.x, bounds, system, bravais_name, observations, settings)
        if not details:
            continue
        candidate = _candidate_from_details(details, bravais_name, observations, settings)
        if candidate["indexed_peak_count"] < settings.minimum_indexed_peaks:
            continue
        if candidate["indexed_fraction"] < settings.minimum_indexed_fraction:
            continue
        fingerprint = tuple(candidate["reduced_cell_fingerprint"])
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        candidates.append(candidate)
        candidates.sort(key=lambda row: (row["objective"], -row["indexed_fraction"], -row["m20"]))
        candidates = candidates[: settings.maximum_candidates_per_lattice]
    for candidate_index, candidate in enumerate(candidates):
        if candidate_index < 2:
            candidate["leave_one_out_stability"] = _leave_one_out_stability(
                candidate, observations, settings
            )
        else:
            candidate["leave_one_out_stability"] = {
                "status": "Not run",
                "checks": 0,
                "success_fraction": None,
                "reason": "Only the two highest-ranked lattice candidates receive the automatic stability screen.",
            }
    return {
        "bravais_name": bravais_name,
        "status": "Completed" if candidates else "Completed without candidate",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "elapsed_seconds": float(time.monotonic() - start_time),
    }


def _candidate_deduplicate(candidates: list[dict]) -> list[dict]:
    retained: list[dict] = []
    for candidate in sorted(candidates, key=lambda row: (row["objective"], -row["indexed_fraction"])):
        fingerprint = np.asarray(candidate["reduced_cell_fingerprint"], dtype=float)
        duplicate = False
        for existing in retained:
            other = np.asarray(existing["reduced_cell_fingerprint"], dtype=float)
            if fingerprint.shape == other.shape and float(np.max(np.abs(fingerprint - other))) <= 0.015:
                duplicate = True
                if candidate["objective"] < existing["objective"]:
                    existing.update(candidate)
                break
        if not duplicate:
            retained.append(candidate)
    return retained


def _rank_candidates(candidates: list[dict]) -> list[dict]:
    ranked = []
    for candidate in candidates:
        row = dict(candidate)
        stability = row.get("leave_one_out_stability", {})
        stability_bonus = {"Stable": 0.5, "Review": 0.1}.get(stability.get("status"), -0.2)
        score = (
            math.log1p(max(0.0, row.get("m20", 0.0)))
            + 2.2 * row.get("indexed_fraction", 0.0)
            - 0.07 * row.get("x20", 0)
            - 0.25 * math.log1p(max(0.0, row.get("position_rmse_deg", 0.0)) / 0.01)
            + stability_bonus
        )
        row["ranking_score"] = float(score)
        if (
            row.get("indexed_fraction", 0.0) >= 0.8
            and row.get("m20", 0.0) >= 8.0
            and row.get("x20", 99) <= 2
            and stability.get("status") == "Stable"
        ):
            row["status"] = "Promising"
        elif row.get("indexed_fraction", 0.0) >= 0.6:
            row["status"] = "Ambiguous"
        else:
            row["status"] = "Review"
        ranked.append(row)
    ranked.sort(key=lambda row: (row["ranking_score"], -row.get("bic_like", row.get("objective", 0.0)), row["m20"]), reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def run_native_indexing(
    work_directory: str | Path,
    peaks: Iterable[dict],
    *,
    wavelength_angstrom: float,
    bravais_names: Iterable[str],
    starting_volume_angstrom3: float = 200.0,
    maximum_hkl_to_observed_ratio: int = 4,
    zero_shift_deg: float = 0.0,
    refine_zero_shift: bool = True,
    maximum_zero_shift_deg: float = 0.25,
    peak_tolerance_deg: float = 0.10,
    impurity_tolerance_fraction: float = 0.15,
    maximum_index: int = 10,
    global_iterations: int = 32,
    population_size: int = 7,
    candidate_seeds_per_lattice: int = 5,
    maximum_candidates_per_lattice: int = 4,
    random_seed: int = 1600,
    timeout_seconds: float = 35.0,
    minimum_m20: float = 2.0,
    maximum_x20: int = 10,
    formula_mass_g_mol: float | None = None,
    z_values: Iterable[int] = (),
    density_min_g_cm3: float | None = None,
    density_max_g_cm3: float | None = None,
    master_peak_revision: int | None = None,
    master_peak_checksum: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    bootstrap_replicates: int = 24,
    bootstrap_omit_fraction: float = 0.10,
    bootstrap_position_sigma_scale: float = 1.0,
    bootstrap_top_candidates: int = 3,
    enable_multiphase_discovery: bool = True,
    multiphase_top_primary_candidates: int = 1,
    multiphase_minimum_residual_peaks: int = 7,
    multiphase_minimum_residual_signal_to_noise: float = 3.5,
    multiphase_minimum_residual_quality_score: float = 25.0,
    multiphase_overlap_tolerance_deg: float = 0.08,
    multiphase_timeout_seconds: float = 8.0,
    multiphase_global_iterations: int = 10,
    multiphase_candidate_seeds_per_lattice: int = 2,
    multiphase_maximum_candidates_per_lattice: int = 2,
    **_ignored,
) -> dict:
    del maximum_hkl_to_observed_ratio  # retained for project compatibility
    settings = NativeIndexingSettings(
        wavelength_angstrom=float(wavelength_angstrom),
        bravais_names=tuple(bravais_names),
        starting_volume_angstrom3=float(starting_volume_angstrom3),
        zero_shift_deg=float(zero_shift_deg),
        refine_zero_shift=bool(refine_zero_shift),
        maximum_zero_shift_deg=float(maximum_zero_shift_deg),
        peak_tolerance_deg=float(peak_tolerance_deg),
        impurity_tolerance_fraction=float(impurity_tolerance_fraction),
        maximum_index=int(maximum_index),
        global_iterations=int(global_iterations),
        population_size=int(population_size),
        candidate_seeds_per_lattice=int(candidate_seeds_per_lattice),
        maximum_candidates_per_lattice=int(maximum_candidates_per_lattice),
        random_seed=int(random_seed),
        time_limit_seconds_per_lattice=float(timeout_seconds),
    ).validate()
    observations = _peak_observations(peaks)
    work = Path(work_directory).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    request_path = work / "native_indexing_request.json"
    result_path = work / "native_indexing_result.json"
    request = {
        "engine": "Afruz native powder indexer",
        "created_utc": _utc_now(),
        "settings": asdict(settings),
        "peaks": [asdict(row) for row in observations],
        "master_peak_revision": master_peak_revision,
        "master_peak_checksum": master_peak_checksum,
    }
    _atomic_write_json(request_path, request)
    attempts = []
    all_candidates = []
    start = time.monotonic()
    for lattice_index, bravais_name in enumerate(settings.bravais_names, start=1):
        if cancel_check and cancel_check():
            raise NativeIndexingCancelled("Native indexing cancelled.")
        attempt = _search_lattice(
            bravais_name,
            observations,
            settings,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            lattice_index=lattice_index,
            lattice_total=len(settings.bravais_names),
        )
        attempts.append(attempt)
        all_candidates.extend(attempt.get("candidates", []))
        _atomic_write_json(work / "native_indexing_progress.json", {
            "engine": "Afruz native powder indexer",
            "completed_lattices": lattice_index,
            "total_lattices": len(settings.bravais_names),
            "attempts": attempts,
        })
    candidates = _rank_candidates(_candidate_deduplicate(all_candidates))
    # Compatibility filters remain visible but do not hide every result.
    passing = [
        row for row in candidates
        if row.get("m20", 0.0) >= float(minimum_m20)
        and row.get("x20", 999) <= int(maximum_x20)
    ]
    warnings = []
    if candidates and not passing:
        warnings.append(
            "No candidate met the configured M20/X20 display thresholds; all native candidates are retained for review."
        )
    bootstrap_settings = BootstrapReliabilitySettings(
        replicates=max(0, int(bootstrap_replicates)),
        omit_fraction=float(bootstrap_omit_fraction),
        position_sigma_scale=float(bootstrap_position_sigma_scale),
        random_seed=settings.random_seed + 1780,
    ).validate()
    for candidate in candidates[: max(0, int(bootstrap_top_candidates))]:
        candidate["bootstrap_reliability"] = evaluate_candidate_reliability(
            candidate,
            [asdict(row) | {"use": True} for row in observations],
            wavelength_angstrom=settings.wavelength_angstrom,
            settings=bootstrap_settings,
        )
        candidate["bootstrap_summary"] = summarize_candidate_reliability(candidate)
    for candidate in candidates[max(0, int(bootstrap_top_candidates)):]:
        candidate["bootstrap_reliability"] = {
            "method": "Bootstrap perturbation/omission screen",
            "status": "Not run",
            "reason": "Only the configured top-ranked candidates receive the automatic Phase 17.8 reliability screen.",
            "replicates_requested": int(bootstrap_settings.replicates),
        }
        candidate["bootstrap_summary"] = "Not run"

    multiphase_settings = MultiphaseDiscoverySettings(
        enabled=bool(enable_multiphase_discovery),
        top_primary_candidates=max(0, int(multiphase_top_primary_candidates)),
        minimum_residual_peaks=max(3, int(multiphase_minimum_residual_peaks)),
        minimum_residual_signal_to_noise=float(multiphase_minimum_residual_signal_to_noise),
        minimum_residual_quality_score=float(multiphase_minimum_residual_quality_score),
        overlap_tolerance_deg=float(multiphase_overlap_tolerance_deg),
        maximum_second_phase_candidates=max(1, int(multiphase_maximum_candidates_per_lattice)),
    ).validate()
    multiphase_discovery = {
        "method": "Phase 17.9 multiphase unknown discovery",
        "enabled": bool(multiphase_settings.enabled),
        "settings": asdict(multiphase_settings),
        "classification": "Not run",
        "records": [],
        "warnings": [],
    }
    if multiphase_settings.enabled and candidates and multiphase_settings.top_primary_candidates > 0:
        for primary_candidate in candidates[: multiphase_settings.top_primary_candidates]:
            residual_record = classify_residual_peaks(
                primary_candidate,
                [asdict(row) | {"use": True} for row in observations],
                settings=multiphase_settings,
            )
            phase_record = {
                "primary_candidate_rank": primary_candidate.get("rank"),
                "primary_bravais": primary_candidate.get("bravais_name"),
                "residual_peak_grouping": residual_record,
                "secondary_search": {"status": "Not run"},
                "joint_assignment": None,
                "classification": residual_record.get("classification", "Review"),
            }
            primary_candidate["multiphase_residual_grouping"] = residual_record
            residual_rows = stable_residual_peak_rows(residual_record)
            if len(residual_rows) >= multiphase_settings.minimum_residual_peaks:
                residual_observations = [
                    PeakObservation(
                        peak_uuid=str(row["peak_uuid"]),
                        two_theta_deg=float(row["two_theta_deg"]),
                        intensity=1.0,
                        position_uncertainty_deg=float(row.get("position_uncertainty_deg", 0.02)),
                    )
                    for row in residual_rows
                ]
                residual_settings = replace(
                    settings,
                    time_limit_seconds_per_lattice=min(
                        settings.time_limit_seconds_per_lattice,
                        float(multiphase_timeout_seconds),
                    ),
                    global_iterations=min(settings.global_iterations, max(1, int(multiphase_global_iterations))),
                    candidate_seeds_per_lattice=min(
                        settings.candidate_seeds_per_lattice,
                        max(1, int(multiphase_candidate_seeds_per_lattice)),
                    ),
                    maximum_candidates_per_lattice=max(
                        1,
                        int(multiphase_maximum_candidates_per_lattice),
                    ),
                    minimum_indexed_peaks=max(3, min(settings.minimum_indexed_peaks, len(residual_observations))),
                    random_seed=settings.random_seed + 1790 + int(primary_candidate.get("rank", 1)),
                )
                residual_candidates = []
                residual_attempts = []
                for lattice_index, bravais_name in enumerate(residual_settings.bravais_names, start=1):
                    if cancel_check and cancel_check():
                        raise NativeIndexingCancelled("Native indexing cancelled.")
                    attempt = _search_lattice(
                        bravais_name,
                        residual_observations,
                        residual_settings,
                        cancel_check=cancel_check,
                        lattice_index=lattice_index,
                        lattice_total=len(residual_settings.bravais_names),
                    )
                    residual_attempts.append({
                        "bravais_name": attempt.get("bravais_name", bravais_name),
                        "status": attempt.get("status"),
                        "candidate_count": len(attempt.get("candidates", [])),
                        "elapsed_seconds": attempt.get("elapsed_seconds"),
                        "error": attempt.get("error"),
                    })
                    residual_candidates.extend(attempt.get("candidates", []))
                residual_candidates = _rank_candidates(_candidate_deduplicate(residual_candidates))
                residual_candidates = residual_candidates[: multiphase_settings.maximum_second_phase_candidates]
                phase_record["secondary_search"] = {
                    "status": "Complete" if residual_candidates else "No candidate",
                    "residual_peak_count": len(residual_observations),
                    "attempts": residual_attempts,
                    "candidates": residual_candidates,
                }
                if residual_candidates:
                    joint = joint_two_phase_assignment(
                        primary_candidate,
                        residual_candidates[0],
                        [asdict(row) | {"use": True} for row in observations],
                        wavelength_angstrom=settings.wavelength_angstrom,
                        settings=multiphase_settings,
                    )
                    phase_record["joint_assignment"] = joint
                    phase_record["classification"] = joint.get("classification", phase_record["classification"])
                    primary_candidate["second_phase_candidate"] = residual_candidates[0]
                    primary_candidate["multiphase_joint_assignment"] = joint
                    primary_candidate["multiphase_summary"] = summarize_multiphase_discovery(phase_record)
                else:
                    primary_candidate["multiphase_summary"] = summarize_multiphase_discovery(phase_record)
            else:
                primary_candidate["multiphase_summary"] = summarize_multiphase_discovery(phase_record)
            multiphase_discovery["records"].append(phase_record)
        if multiphase_discovery["records"]:
            best_classification = multiphase_discovery["records"][0].get("classification", "Review")
            multiphase_discovery["classification"] = best_classification
            if best_classification == "Second phase justified":
                multiphase_discovery["warnings"].append(
                    "A possible second unknown phase was found; validate the combined model with Pawley/Le Bail before accepting it."
                )
    elif not multiphase_settings.enabled:
        multiphase_discovery["classification"] = "Disabled"
    else:
        multiphase_discovery["classification"] = "No primary candidate"

    for candidate in candidates:
        candidate["master_peak_revision"] = master_peak_revision
        candidate["master_peak_checksum"] = master_peak_checksum
        if formula_mass_g_mol and candidate.get("volume_angstrom3"):
            checks = []
            for z in z_values:
                if int(z) <= 0:
                    continue
                density = 1.66053906660 * float(formula_mass_g_mol) * int(z) / candidate["volume_angstrom3"]
                plausible = True
                if density_min_g_cm3 is not None:
                    plausible &= density >= float(density_min_g_cm3)
                if density_max_g_cm3 is not None:
                    plausible &= density <= float(density_max_g_cm3)
                checks.append({"z": int(z), "density_g_cm3": float(density), "plausible": bool(plausible)})
            candidate["density_checks"] = checks
            candidate["chemistry_status"] = "Plausible" if any(row["plausible"] for row in checks) else "Review"
        else:
            candidate["density_checks"] = []
            candidate["chemistry_status"] = "Not assessed"
    result = {
        "success": bool(candidates),
        "engine": "Afruz native powder indexer",
        "classification": "Native candidate-cell search",
        "created_utc": _utc_now(),
        "elapsed_seconds": float(time.monotonic() - start),
        "request_file": str(request_path),
        "result_file": str(result_path),
        "work_directory": str(work),
        "observed_peak_count": len(observations),
        "bootstrap_reliability_settings": asdict(bootstrap_settings),
        "multiphase_unknown_discovery": multiphase_discovery,
        "multiphase_summary": summarize_multiphase_discovery(multiphase_discovery),
        "master_peak_revision": master_peak_revision,
        "master_peak_checksum": master_peak_checksum,
        "attempts": attempts,
        "candidates": candidates,
        "warnings": warnings,
        "scientific_warning": (
            "A candidate cell is a hypothesis. Verify it with Pawley/Le Bail decomposition, "
            "systematic-absence analysis, competing cells, chemistry and structure solution."
        ),
    }
    _atomic_write_json(result_path, result)
    return result


def native_indexer_fingerprint() -> str:
    payload = {
        "engine": "Afruz native powder indexer",
        "version": 3,
        "bravais": BRAVAIS_NAMES,
        "algorithm": "deep Q-space assignment + weighted SVD reciprocal metric + robust angular refinement + bootstrap reliability screen",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
