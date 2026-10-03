from __future__ import annotations

"""Metrology helpers for Pawley and Le Bail whole-pattern extraction.

The routines in this module deliberately separate numerical fitting from the
question that follows it: which extracted intensities are independently
supported by the measured pattern?  All public payloads are JSON-safe.
"""

import math

import numpy as np


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(diagonal, diagonal)
    correlation = np.zeros_like(covariance, dtype=float)
    np.divide(covariance, denominator, out=correlation, where=denominator > 0)
    np.fill_diagonal(correlation, np.where(diagonal > 0, 1.0, 0.0))
    return np.clip(correlation, -1.0, 1.0)


def _column_cosines(matrix: np.ndarray) -> np.ndarray:
    gram = matrix.T @ matrix
    norm = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = np.outer(norm, norm)
    cosines = np.zeros_like(gram, dtype=float)
    np.divide(gram, denominator, out=cosines, where=denominator > 0)
    np.fill_diagonal(cosines, np.where(norm > 0, 1.0, 0.0))
    return np.clip(cosines, -1.0, 1.0)


def _rank_diagnostics(matrix: np.ndarray) -> dict:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if not singular_values.size:
        return {
            "rank": 0,
            "condition_number": None,
            "singular_values": [],
            "rank_tolerance": 0.0,
        }
    tolerance = (
        max(matrix.shape)
        * np.finfo(float).eps
        * float(singular_values[0])
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    smallest = float(singular_values[rank - 1]) if rank else 0.0
    condition = (
        float(singular_values[0] / smallest)
        if smallest > 0.0
        else None
    )
    return {
        "rank": rank,
        "condition_number": condition,
        "singular_values": singular_values.tolist(),
        "rank_tolerance": float(tolerance),
    }


def _overlap_groups(cosines: np.ndarray, threshold: float) -> list[list[int]]:
    count = cosines.shape[0]
    parents = list(range(count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(count):
        for right in range(left + 1, count):
            if abs(float(cosines[left, right])) >= threshold:
                union(left, right)

    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(find(index), []).append(index)
    return list(grouped.values())


def intensity_metrology(
    peak_basis: np.ndarray,
    background_basis: np.ndarray,
    observed_y: np.ndarray,
    sqrt_weights: np.ndarray,
    intensities: np.ndarray,
    background_coefficients: np.ndarray,
    *,
    statistics_valid: bool,
    method: str,
    overlap_correlation_threshold: float = 0.90,
    severe_correlation_threshold: float = 0.98,
) -> dict:
    """Return active-set covariance and overlap/identifiability diagnostics.

    Pawley covariance is the local covariance of the non-negative linear
    least-squares solution.  For Le Bail extraction the same final-profile
    linearization is explicitly labelled as an approximation.
    """

    peak_basis = np.asarray(peak_basis, dtype=float)
    background_basis = np.asarray(background_basis, dtype=float)
    observed_y = np.asarray(observed_y, dtype=float)
    sqrt_weights = np.asarray(sqrt_weights, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    background_coefficients = np.asarray(background_coefficients, dtype=float)
    reflection_count = peak_basis.shape[1]
    if intensities.shape != (reflection_count,):
        raise ValueError("Intensity vector does not match the peak basis.")

    weighted_peaks = peak_basis * sqrt_weights[:, None]
    weighted_background = background_basis * sqrt_weights[:, None]
    if weighted_background.size:
        background_projection = weighted_background @ np.linalg.lstsq(
            weighted_background, weighted_peaks, rcond=None
        )[0]
        information_peaks = weighted_peaks - background_projection
    else:
        information_peaks = weighted_peaks

    peak_rank = _rank_diagnostics(information_peaks)
    cosines = _column_cosines(information_peaks)
    maximum_intensity = max(float(np.max(intensities)), 1.0) if intensities.size else 1.0
    active_tolerance = max(maximum_intensity * 1e-10, 1e-12)
    active = intensities > active_tolerance
    active_indices = np.flatnonzero(active)

    covariance = np.zeros((reflection_count, reflection_count), dtype=float)
    standard_errors: list[float | None] = [None] * reflection_count
    correlation = np.zeros_like(covariance)
    covariance_rank = 0
    variance_scale = None
    residual = observed_y - (
        peak_basis @ intensities
        + background_basis @ background_coefficients
    )
    weighted_residual = residual * sqrt_weights
    free_design = np.column_stack(
        [peak_basis[:, active], background_basis]
    )
    weighted_free_design = free_design * sqrt_weights[:, None]
    free_rank = _rank_diagnostics(weighted_free_design)
    degrees_of_freedom = max(1, len(observed_y) - free_rank["rank"])
    if weighted_free_design.shape[1]:
        normal_inverse = np.linalg.pinv(
            weighted_free_design.T @ weighted_free_design,
            rcond=1e-12,
        )
        variance_scale = (
            1.0
            if statistics_valid
            else float(np.sum(weighted_residual ** 2) / degrees_of_freedom)
        )
        free_covariance = normal_inverse * variance_scale
        covariance_rank = int(free_rank["rank"])
        intensity_covariance = free_covariance[
            : len(active_indices), : len(active_indices)
        ]
        covariance[np.ix_(active_indices, active_indices)] = intensity_covariance
        correlation = _correlation_from_covariance(covariance)
        for local_index, reflection_index in enumerate(active_indices):
            variance = float(intensity_covariance[local_index, local_index])
            if np.isfinite(variance) and variance >= 0.0:
                standard_errors[int(reflection_index)] = math.sqrt(variance)

    threshold = float(np.clip(overlap_correlation_threshold, 0.0, 0.999999))
    severe = float(np.clip(severe_correlation_threshold, threshold, 0.9999999))
    groups = _overlap_groups(cosines, threshold)
    group_rows = []
    reflection_group_ids = [0] * reflection_count
    max_basis_correlations = [0.0] * reflection_count
    for group_id, indices in enumerate(groups, start=1):
        for index in indices:
            reflection_group_ids[index] = group_id
            other = [abs(float(cosines[index, j])) for j in indices if j != index]
            max_basis_correlations[index] = max(other, default=0.0)
        group_covariance = covariance[np.ix_(indices, indices)]
        group_variance = float(np.sum(group_covariance))
        group_cosines = [
            abs(float(cosines[left, right]))
            for offset, left in enumerate(indices)
            for right in indices[offset + 1 :]
        ]
        group_rows.append(
            {
                "group_id": group_id,
                "reflection_indices": indices,
                "reflection_count": len(indices),
                "intensity_sum": float(np.sum(intensities[indices])),
                "intensity_sum_standard_error": (
                    math.sqrt(max(group_variance, 0.0))
                    if any(active[index] for index in indices)
                    else None
                ),
                "maximum_basis_correlation": max(group_cosines, default=0.0),
                "classification": (
                    "unresolved"
                    if max(group_cosines, default=0.0) >= severe
                    else "overlapped"
                    if len(indices) > 1
                    else "isolated"
                ),
            }
        )

    reflection_rows = []
    for index, intensity in enumerate(intensities):
        error = standard_errors[index]
        maximum_correlation = max_basis_correlations[index]
        estimable = bool(active[index] and maximum_correlation < severe)
        if not active[index]:
            classification = "boundary-constrained"
        elif maximum_correlation >= severe:
            classification = "not-individually-identifiable"
        elif maximum_correlation >= threshold:
            classification = "correlated"
        else:
            classification = "individually-identifiable"
        reflection_rows.append(
            {
                "reflection_index": index,
                "standard_error": error,
                "signal_to_uncertainty": (
                    float(intensity / error)
                    if error is not None and error > 0.0
                    else None
                ),
                "active_nonnegative_solution": bool(active[index]),
                "individually_identifiable": estimable,
                "identifiability": classification,
                "overlap_group_id": reflection_group_ids[index],
                "maximum_basis_correlation": maximum_correlation,
            }
        )

    high_correlation_pairs = []
    for left in range(reflection_count):
        for right in range(left + 1, reflection_count):
            basis_correlation = abs(float(cosines[left, right]))
            if basis_correlation >= threshold:
                high_correlation_pairs.append(
                    {
                        "reflection_i": left,
                        "reflection_j": right,
                        "basis_correlation": basis_correlation,
                        "intensity_correlation": float(correlation[left, right]),
                    }
                )

    return {
        "method": method,
        "covariance_interpretation": (
            "active-set linear least-squares covariance"
            if method == "Pawley decomposition"
            else "post-extraction linearized covariance approximation"
        ),
        "absolute_variance_model": bool(statistics_valid),
        "variance_scale": variance_scale,
        "degrees_of_freedom": int(degrees_of_freedom),
        "reflection_count": reflection_count,
        "active_reflection_count": int(np.count_nonzero(active)),
        "effective_independent_reflection_count": int(peak_rank["rank"]),
        "reflection_information_condition_number": peak_rank["condition_number"],
        "reflection_information_singular_values": peak_rank["singular_values"],
        "free_design_rank": covariance_rank,
        "overlap_correlation_threshold": threshold,
        "severe_correlation_threshold": severe,
        "overlap_group_count": sum(len(row["reflection_indices"]) > 1 for row in group_rows),
        "unresolved_group_count": sum(row["classification"] == "unresolved" for row in group_rows),
        "individually_identifiable_count": sum(row["individually_identifiable"] for row in reflection_rows),
        "reflection_diagnostics": reflection_rows,
        "overlap_groups": group_rows,
        "high_correlation_pairs": high_correlation_pairs,
        "intensity_covariance": covariance.tolist(),
        "intensity_correlation": correlation.tolist(),
    }


def nonlinear_parameter_metrology(
    optimization_result,
    residuals: np.ndarray,
    parameter_names: list[str],
) -> dict:
    """Return covariance, correlations and standard errors for nonlinear terms."""

    count = len(parameter_names)
    empty = {
        "parameter_names": list(parameter_names),
        "standard_errors": [None] * count,
        "covariance": None,
        "correlation": None,
        "effective_rank": 0,
        "condition_number": None,
    }
    if optimization_result is None or count == 0:
        return empty
    jacobian = np.asarray(getattr(optimization_result, "jac", []), dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[1] != count or not jacobian.size:
        return empty
    try:
        rank = _rank_diagnostics(jacobian)
        degrees_of_freedom = max(1, jacobian.shape[0] - rank["rank"])
        covariance = np.linalg.pinv(jacobian.T @ jacobian, rcond=1e-12)
        covariance *= float(np.sum(np.asarray(residuals, dtype=float) ** 2) / degrees_of_freedom)
        errors = [
            math.sqrt(float(value)) if np.isfinite(value) and value >= 0.0 else None
            for value in np.diag(covariance)
        ]
        correlation = _correlation_from_covariance(covariance)
        return {
            "parameter_names": list(parameter_names),
            "standard_errors": errors,
            "covariance": covariance.tolist(),
            "correlation": correlation.tolist(),
            "effective_rank": rank["rank"],
            "condition_number": rank["condition_number"],
        }
    except np.linalg.LinAlgError:
        return empty
