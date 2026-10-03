from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


class DeepIndexingError(RuntimeError):
    pass


@dataclass(frozen=True)
class AssignmentSettings:
    outlier_cost: float = 8.0
    huber_delta: float = 2.5
    model_sigma_q: float = 0.0
    line_density_reference: float = 4.0
    line_density_penalty: float = 3.0
    unindexed_observed_penalty: float = 12.0
    complexity_penalty: float = 0.40
    conditioning_penalty: float = 0.15


@dataclass(frozen=True)
class MetricRefinementSummary:
    reduced_chi_squared: float
    rank: int
    condition_number: float
    parameter_count: int
    assignment_count: int
    positive_definite: bool

    def to_dict(self) -> dict:
        return asdict(self)


def two_theta_to_q(two_theta_deg: np.ndarray | Sequence[float], wavelength: float) -> np.ndarray:
    t = np.deg2rad(np.asarray(two_theta_deg, dtype=float))
    return 4.0 * np.sin(t / 2.0) ** 2 / float(wavelength) ** 2


def q_to_two_theta(q: np.ndarray | Sequence[float], wavelength: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    argument = 0.5 * float(wavelength) * np.sqrt(np.maximum(q, 0.0))
    result = np.full_like(argument, np.nan, dtype=float)
    valid = (argument >= 0.0) & (argument <= 1.0)
    result[valid] = np.rad2deg(2.0 * np.arcsin(argument[valid]))
    return result


def sigma_two_theta_to_sigma_q(
    two_theta_deg: np.ndarray | Sequence[float],
    sigma_two_theta_deg: np.ndarray | Sequence[float],
    wavelength: float,
) -> np.ndarray:
    t = np.deg2rad(np.asarray(two_theta_deg, dtype=float))
    sigma_t = np.deg2rad(np.asarray(sigma_two_theta_deg, dtype=float))
    derivative = 2.0 * np.sin(t) / float(wavelength) ** 2
    sigma_q = np.abs(derivative) * sigma_t
    return np.maximum(sigma_q, 1e-10)


def metric_design_matrix(hkls: np.ndarray | Sequence[Sequence[int]], crystal_system: str) -> np.ndarray:
    hkls = np.asarray(hkls, dtype=float)
    if hkls.ndim != 2 or hkls.shape[1] != 3:
        raise DeepIndexingError("HKLs must be an N x 3 array.")
    h, k, l = hkls.T
    system = crystal_system.lower().replace(" ", "_").replace("-", "_")
    if system == "cubic":
        return (h**2 + k**2 + l**2)[:, None]
    if system == "tetragonal":
        return np.column_stack([h**2 + k**2, l**2])
    if system in {"hexagonal", "trigonal_hex"}:
        return np.column_stack([h**2 + h * k + k**2, l**2])
    if system == "orthorhombic":
        return np.column_stack([h**2, k**2, l**2])
    if system in {"monoclinic", "monoclinic_b"}:
        return np.column_stack([h**2, k**2, l**2, 2.0 * h * l])
    if system == "triclinic":
        return np.column_stack([h**2, k**2, l**2, 2.0 * k * l, 2.0 * h * l, 2.0 * h * k])
    raise DeepIndexingError(f"Unsupported crystal system: {crystal_system}")


def centering_allowed(hkls: np.ndarray | Sequence[Sequence[int]], centering: str) -> np.ndarray:
    hkls = np.asarray(hkls, dtype=int)
    if hkls.ndim != 2 or hkls.shape[1] != 3:
        raise DeepIndexingError("HKLs must be an N x 3 array.")
    h, k, l = hkls.T
    centering = centering.upper()
    if centering == "P":
        return np.ones(len(hkls), dtype=bool)
    if centering == "I":
        return (h + k + l) % 2 == 0
    if centering == "F":
        parity = np.column_stack([h % 2, k % 2, l % 2])
        return np.all(parity == parity[:, [0]], axis=1)
    if centering == "A":
        return (k + l) % 2 == 0
    if centering == "B":
        return (h + l) % 2 == 0
    if centering == "C":
        return (h + k) % 2 == 0
    if centering == "R":
        return (-h + k + l) % 3 == 0
    raise DeepIndexingError(f"Unknown centering: {centering}")


def huber_cost(residual: np.ndarray | Sequence[float], delta: float = 2.5) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    absolute = np.abs(residual)
    return np.where(absolute <= delta, 0.5 * residual**2, delta * (absolute - 0.5 * delta))


def refine_reciprocal_metric(
    q_observed: np.ndarray | Sequence[float],
    sigma_q: np.ndarray | Sequence[float],
    hkls: np.ndarray | Sequence[Sequence[int]],
    crystal_system: str,
) -> dict:
    q_observed = np.asarray(q_observed, dtype=float)
    sigma_q = np.asarray(sigma_q, dtype=float)
    if len(q_observed) != len(sigma_q):
        raise DeepIndexingError("Q values and uncertainties must have equal length.")
    if np.any(~np.isfinite(q_observed)) or np.any(~np.isfinite(sigma_q)):
        raise DeepIndexingError("Q values and uncertainties must be finite.")
    if np.any(sigma_q <= 0):
        raise DeepIndexingError("All Q uncertainties must be positive.")
    x = metric_design_matrix(hkls, crystal_system)
    if len(q_observed) < x.shape[1]:
        raise DeepIndexingError("Not enough assigned reflections for metric refinement.")
    x_weighted = x / sigma_q[:, None]
    q_weighted = q_observed / sigma_q
    parameters, _, rank, singular_values = np.linalg.lstsq(x_weighted, q_weighted, rcond=None)
    q_calculated = x @ parameters
    normalized_residuals = (q_observed - q_calculated) / sigma_q
    degrees_of_freedom = max(len(q_observed) - len(parameters), 1)
    reduced_chi_squared = float(np.sum(normalized_residuals**2) / degrees_of_freedom)
    information = x_weighted.T @ x_weighted
    covariance = np.linalg.pinv(information) * reduced_chi_squared
    condition_number = float(singular_values[0] / singular_values[-1]) if len(singular_values) and singular_values[-1] > 0 else math.inf
    cell = None
    positive_definite = False
    try:
        cell = reciprocal_parameters_to_cell(parameters, crystal_system)
        positive_definite = True
    except Exception:
        cell = None
        positive_definite = False
    return {
        "parameters": parameters,
        "covariance": covariance,
        "q_calculated": q_calculated,
        "normalized_residuals": normalized_residuals,
        "reduced_chi_squared": reduced_chi_squared,
        "rank": int(rank),
        "condition_number": condition_number,
        "cell": cell,
        "summary": MetricRefinementSummary(
            reduced_chi_squared=reduced_chi_squared,
            rank=int(rank),
            condition_number=condition_number,
            parameter_count=int(x.shape[1]),
            assignment_count=int(len(q_observed)),
            positive_definite=positive_definite,
        ).to_dict(),
    }


def reciprocal_parameters_to_cell(parameters: np.ndarray | Sequence[float], crystal_system: str) -> dict:
    p = np.asarray(parameters, dtype=float)
    if np.any(~np.isfinite(p)):
        raise DeepIndexingError("Metric parameters are not finite.")
    system = crystal_system.lower().replace(" ", "_").replace("-", "_")
    if system == "cubic":
        if p[0] <= 0:
            raise DeepIndexingError("Cubic reciprocal coefficient must be positive.")
        a = 1.0 / math.sqrt(float(p[0]))
        return {"a": a, "b": a, "c": a, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    if system == "tetragonal":
        if np.any(p[:2] <= 0):
            raise DeepIndexingError("Tetragonal reciprocal coefficients must be positive.")
        a = 1.0 / math.sqrt(float(p[0]))
        c = 1.0 / math.sqrt(float(p[1]))
        return {"a": a, "b": a, "c": c, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    if system in {"hexagonal", "trigonal_hex"}:
        if np.any(p[:2] <= 0):
            raise DeepIndexingError("Hexagonal reciprocal coefficients must be positive.")
        a = math.sqrt(4.0 / (3.0 * float(p[0])))
        c = 1.0 / math.sqrt(float(p[1]))
        return {"a": a, "b": a, "c": c, "alpha": 90.0, "beta": 90.0, "gamma": 120.0}
    if system == "orthorhombic":
        if np.any(p[:3] <= 0):
            raise DeepIndexingError("Orthorhombic reciprocal coefficients must be positive.")
        a, b, c = 1.0 / np.sqrt(p[:3])
        return {"a": float(a), "b": float(b), "c": float(c), "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    if system in {"monoclinic", "monoclinic_b"}:
        if len(p) < 4:
            raise DeepIndexingError("Monoclinic refinement requires four reciprocal coefficients.")
        g_star = np.array([[p[0], 0.0, p[3]], [0.0, p[1], 0.0], [p[3], 0.0, p[2]]], dtype=float)
    elif system == "triclinic":
        if len(p) < 6:
            raise DeepIndexingError("Triclinic refinement requires six reciprocal coefficients.")
        g_star = np.array([[p[0], p[5], p[4]], [p[5], p[1], p[3]], [p[4], p[3], p[2]]], dtype=float)
    else:
        raise DeepIndexingError(f"Unsupported crystal system: {crystal_system}")
    eigenvalues = np.linalg.eigvalsh(g_star)
    if np.any(eigenvalues <= 0):
        raise DeepIndexingError("Reciprocal metric is not positive definite.")
    g_real = np.linalg.inv(g_star)
    a = math.sqrt(float(g_real[0, 0]))
    b = math.sqrt(float(g_real[1, 1]))
    c = math.sqrt(float(g_real[2, 2]))
    alpha = math.degrees(math.acos(float(np.clip(g_real[1, 2] / (b * c), -1.0, 1.0))))
    beta = math.degrees(math.acos(float(np.clip(g_real[0, 2] / (a * c), -1.0, 1.0))))
    gamma = math.degrees(math.acos(float(np.clip(g_real[0, 1] / (a * b), -1.0, 1.0))))
    return {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}


def assign_peaks(
    q_observed: np.ndarray | Sequence[float],
    sigma_q: np.ndarray | Sequence[float],
    q_calculated: np.ndarray | Sequence[float],
    *,
    outlier_cost: float = 8.0,
    model_sigma_q: float = 0.0,
    huber_delta: float = 2.5,
) -> dict:
    q_observed = np.asarray(q_observed, dtype=float)
    sigma_q = np.asarray(sigma_q, dtype=float)
    q_calculated = np.asarray(q_calculated, dtype=float)
    if len(q_observed) == 0:
        return {"matches": [], "unindexed": [], "unmatched_calculated": list(range(len(q_calculated))), "assignment_cost": 0.0}
    if len(q_calculated) == 0:
        return {"matches": [], "unindexed": list(range(len(q_observed))), "unmatched_calculated": [], "assignment_cost": float(outlier_cost * len(q_observed))}
    denominator = np.sqrt(np.maximum(sigma_q[:, None] ** 2 + float(model_sigma_q) ** 2, 1e-20))
    standardized = (q_observed[:, None] - q_calculated[None, :]) / denominator
    pair_cost = huber_cost(standardized, delta=huber_delta)
    n_observed, n_calculated = pair_cost.shape
    cost = np.full((n_observed, n_calculated + n_observed), float(outlier_cost), dtype=float)
    cost[:, :n_calculated] = pair_cost
    rows, columns = linear_sum_assignment(cost)
    matched = []
    unindexed = []
    used_calculated = set()
    for row, column in zip(rows, columns):
        if column < n_calculated and cost[row, column] < outlier_cost:
            used_calculated.add(int(column))
            matched.append({
                "observed_index": int(row),
                "predicted_index": int(column),
                "delta_q": float(q_observed[row] - q_calculated[column]),
                "normalized_delta": float(standardized[row, column]),
                "assignment_cost": float(cost[row, column]),
            })
        else:
            unindexed.append(int(row))
    matched.sort(key=lambda row: row["observed_index"])
    return {
        "matches": matched,
        "unindexed": sorted(unindexed),
        "unmatched_calculated": [index for index in range(n_calculated) if index not in used_calculated],
        "assignment_cost": float(cost[rows, columns].sum()),
    }


def cluster_predicted_reflections(predicted: list[dict], resolution_q: float | np.ndarray) -> list[dict]:
    if not predicted:
        return []
    rows = sorted(predicted, key=lambda row: float(row["q"]))
    clustered: list[dict] = []
    for row in rows:
        q = float(row["q"])
        if isinstance(resolution_q, np.ndarray):
            threshold = float(np.nanmedian(resolution_q))
        else:
            threshold = float(resolution_q)
        threshold = max(threshold, 1e-9, abs(q) * 1e-5)
        if clustered and abs(q - float(clustered[-1]["q"])) <= threshold:
            previous = clustered[-1]
            n_old = int(previous.get("cluster_size", 1))
            new_q = (float(previous["q"]) * n_old + q) / (n_old + 1)
            previous["q"] = new_q
            previous["cluster_size"] = n_old + 1
            previous.setdefault("equivalent_hkls", [])
            previous["equivalent_hkls"].extend(row.get("equivalent_hkls") or [[row.get("h"), row.get("k"), row.get("l")]])
            previous["multiplicity"] = int(previous.get("multiplicity", 1)) + int(row.get("multiplicity", 1))
        else:
            clone = dict(row)
            clone["cluster_size"] = 1
            clone.setdefault("equivalent_hkls", row.get("equivalent_hkls") or [[row.get("h"), row.get("k"), row.get("l")]])
            clustered.append(clone)
    return clustered


def penalized_assignment_score(
    assignment: dict,
    *,
    observed_count: int,
    calculated_count: int,
    free_parameter_count: int,
    condition_number: float | None = None,
    settings: AssignmentSettings = AssignmentSettings(),
) -> dict:
    observed_count = max(int(observed_count), 1)
    matched_count = len(assignment.get("matches", []))
    unindexed_count = len(assignment.get("unindexed", []))
    pair_cost = float(sum(row.get("assignment_cost", 0.0) for row in assignment.get("matches", [])))
    mean_match_cost = pair_cost / max(matched_count, 1)
    line_ratio = float(calculated_count) / float(observed_count)
    line_density_excess = max(0.0, line_ratio - settings.line_density_reference)
    unindexed_fraction = unindexed_count / observed_count
    bic_like = pair_cost + settings.complexity_penalty * free_parameter_count * math.log(observed_count + 1.0)
    score = (
        mean_match_cost
        + settings.unindexed_observed_penalty * unindexed_fraction
        + settings.line_density_penalty * line_density_excess**2
        + settings.complexity_penalty * free_parameter_count
    )
    if condition_number is not None and np.isfinite(condition_number) and condition_number > 0:
        score += settings.conditioning_penalty * math.log10(max(condition_number, 1.0))
    elif condition_number is not None:
        score += 10.0
    return {
        "deep_objective": float(score),
        "assignment_pair_cost": float(pair_cost),
        "mean_assignment_cost": float(mean_match_cost),
        "bic_like": float(bic_like),
        "unindexed_observed_fraction": float(unindexed_fraction),
        "line_density_ratio": float(line_ratio),
        "line_density_excess": float(line_density_excess),
        "matched_count": int(matched_count),
        "unindexed_count": int(unindexed_count),
        "calculated_count": int(calculated_count),
        "free_parameter_count": int(free_parameter_count),
    }


def clean_intensity(intensity: np.ndarray | Sequence[float], relative_threshold: float = 1e-6) -> np.ndarray:
    intensity = np.asarray(intensity, dtype=float)
    if intensity.size == 0:
        return intensity.copy()
    maximum = float(np.nanmax(intensity))
    if not np.isfinite(maximum) or maximum <= 0:
        return np.zeros_like(intensity)
    cleaned = intensity.copy()
    cleaned[cleaned < maximum * float(relative_threshold)] = 0.0
    return cleaned
