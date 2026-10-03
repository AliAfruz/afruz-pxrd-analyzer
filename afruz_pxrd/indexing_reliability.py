from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .deep_unknown_indexing import (
    assign_peaks,
    refine_reciprocal_metric,
    sigma_two_theta_to_sigma_q,
    two_theta_to_q,
)


@dataclass(frozen=True)
class BootstrapReliabilitySettings:
    """Settings for the Phase 17.8 fast indexing-reliability screen.

    The default screen perturbs/omits the experimental peak list and tests
    whether a candidate's hkl assignment remains self-consistent after weighted
    reciprocal-metric refinement. It is deliberately cheaper than a full global
    re-indexing bootstrap so it can run automatically during candidate ranking.
    """

    replicates: int = 24
    omit_fraction: float = 0.10
    position_sigma_scale: float = 1.0
    minimum_recovery_fraction: float = 0.65
    assignment_outlier_cost: float = 8.0
    random_seed: int = 1780

    def validate(self) -> "BootstrapReliabilitySettings":
        if self.replicates < 0:
            raise ValueError("Bootstrap replicate count cannot be negative.")
        if not 0.0 <= self.omit_fraction < 0.5:
            raise ValueError("Bootstrap omit fraction must be between 0 and 0.5.")
        if self.position_sigma_scale < 0:
            raise ValueError("Position sigma scale cannot be negative.")
        if not 0.0 < self.minimum_recovery_fraction <= 1.0:
            raise ValueError("Minimum recovery fraction must be in (0, 1].")
        return self


def _deep_system_name(crystal_system: str) -> str | None:
    system = str(crystal_system).strip().lower().replace(" ", "_").replace("-", "_")
    if system in {"cubic", "tetragonal", "hexagonal", "orthorhombic", "triclinic"}:
        return system
    if system in {"monoclinic", "monoclinic_b"}:
        return "monoclinic_b"
    return None


def _enabled_observations(peaks: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for index, row in enumerate(peaks):
        if not bool(row.get("use", True)):
            continue
        position = float(row.get("two_theta_deg", math.nan))
        if not np.isfinite(position) or not 0.0 < position < 180.0:
            continue
        uncertainty = float(row.get("position_uncertainty_deg", 0.02) or 0.02)
        uncertainty = float(np.clip(uncertainty, 0.001, 0.5))
        rows.append(
            {
                "peak_uuid": str(row.get("peak_uuid") or f"peak-{index + 1}"),
                "two_theta_deg": position,
                "uncertainty_deg": uncertainty,
                "quality_score": float(row.get("quality_score", 50.0) or 50.0),
                "signal_to_noise": float(row.get("signal_to_noise", 1.0) or 1.0),
            }
        )
    rows.sort(key=lambda item: item["two_theta_deg"])
    return rows


def _candidate_assignment_reflections(candidate: dict) -> list[dict]:
    reflections = []
    seen = set()
    for row in candidate.get("assignments", []):
        try:
            hkl = (int(row["h"]), int(row["k"]), int(row["l"]))
            position = float(row["predicted_two_theta_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (hkl, round(position, 6))
        if key in seen:
            continue
        seen.add(key)
        reflections.append(
            {
                "h": hkl[0],
                "k": hkl[1],
                "l": hkl[2],
                "hkl_label": f"({hkl[0]} {hkl[1]} {hkl[2]})",
                "predicted_two_theta_deg": position,
            }
        )
    reflections.sort(key=lambda row: row["predicted_two_theta_deg"])
    return reflections


def _cell_stat_summary(cells: list[dict]) -> dict:
    if not cells:
        return {"available": False, "reason": "No successful reciprocal-metric refinements."}
    keys = ["a", "b", "c", "alpha", "beta", "gamma"]
    summary: dict[str, object] = {"available": True, "sample_count": len(cells)}
    for key in keys:
        values = np.asarray([float(cell[key]) for cell in cells if key in cell and np.isfinite(float(cell[key]))], dtype=float)
        if values.size == 0:
            continue
        suffix = "angstrom" if key in {"a", "b", "c"} else "deg"
        summary[f"{key}_{suffix}_median"] = float(np.median(values))
        summary[f"{key}_{suffix}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary[f"{key}_{suffix}_p05"] = float(np.percentile(values, 5.0))
        summary[f"{key}_{suffix}_p95"] = float(np.percentile(values, 95.0))
    return summary


def evaluate_candidate_reliability(
    candidate: dict,
    peaks: Iterable[dict],
    *,
    wavelength_angstrom: float,
    settings: BootstrapReliabilitySettings | None = None,
) -> dict:
    """Evaluate candidate robustness under peak-position perturbation and omission.

    Returns a JSON-serialisable reliability record. This does not prove a
    structure; it tests whether the candidate's hkl assignment and reciprocal
    metric remain stable when the curated peak list is realistically disturbed.
    """

    cfg = (settings or BootstrapReliabilitySettings()).validate()
    observations = _enabled_observations(peaks)
    reflections = _candidate_assignment_reflections(candidate)
    if cfg.replicates == 0:
        return {
            "method": "Bootstrap perturbation/omission screen",
            "status": "Not run",
            "replicates_requested": 0,
            "recovery_percent": None,
            "warning": "Bootstrap replicate count is zero.",
        }
    if len(observations) < 5 or len(reflections) < 5:
        return {
            "method": "Bootstrap perturbation/omission screen",
            "status": "Insufficient data",
            "replicates_requested": int(cfg.replicates),
            "replicates_successful": 0,
            "recovery_percent": 0.0,
            "warning": "At least five enabled observations and assigned reflections are required.",
        }

    rng = np.random.default_rng(int(cfg.random_seed))
    predicted_positions = np.asarray([row["predicted_two_theta_deg"] for row in reflections], dtype=float)
    predicted_q = two_theta_to_q(predicted_positions, wavelength_angstrom)
    deep_system = _deep_system_name(candidate.get("crystal_system", ""))
    successes = 0
    matched_fractions = []
    assignment_costs = []
    refined_cells: list[dict] = []
    peak_match_counts = {row["peak_uuid"]: 0 for row in observations}
    hkl_counts = {row["hkl_label"]: 0 for row in reflections}
    replicate_records = []

    keep_count = max(5, int(round(len(observations) * (1.0 - cfg.omit_fraction))))
    keep_count = min(len(observations), keep_count)

    for replicate in range(int(cfg.replicates)):
        kept_indices = np.sort(rng.choice(len(observations), size=keep_count, replace=False))
        kept = [observations[int(index)] for index in kept_indices]
        base_positions = np.asarray([row["two_theta_deg"] for row in kept], dtype=float)
        sigmas = np.asarray([row["uncertainty_deg"] for row in kept], dtype=float)
        perturbed = base_positions + rng.normal(0.0, sigmas * cfg.position_sigma_scale)
        observed_q = two_theta_to_q(perturbed, wavelength_angstrom)
        sigma_q = sigma_two_theta_to_sigma_q(perturbed, sigmas, wavelength_angstrom)
        assignment = assign_peaks(
            observed_q,
            sigma_q,
            predicted_q,
            outlier_cost=cfg.assignment_outlier_cost,
            model_sigma_q=float(np.nanmedian(sigma_q)) * 0.25 if len(sigma_q) else 0.0,
        )
        matched = assignment.get("matches", [])
        matched_fraction = len(matched) / max(1, len(kept))
        matched_fractions.append(float(matched_fraction))
        assignment_costs.append(float(assignment.get("assignment_cost", 0.0)))
        recovered = matched_fraction >= cfg.minimum_recovery_fraction

        if deep_system and len(matched) >= 5:
            try:
                obs_indices = [int(row["observed_index"]) for row in matched]
                pred_indices = [int(row["predicted_index"]) for row in matched]
                metric_q = observed_q[obs_indices]
                metric_sigma_q = sigma_q[obs_indices]
                hkls = np.asarray(
                    [[reflections[j]["h"], reflections[j]["k"], reflections[j]["l"]] for j in pred_indices],
                    dtype=int,
                )
                metric = refine_reciprocal_metric(metric_q, metric_sigma_q, hkls, deep_system)
                if metric.get("cell") and metric.get("summary", {}).get("positive_definite"):
                    refined_cells.append(metric["cell"])
            except Exception:
                pass

        if recovered:
            successes += 1
            for row in matched:
                obs = kept[int(row["observed_index"])]
                ref = reflections[int(row["predicted_index"])]
                peak_match_counts[obs["peak_uuid"]] = peak_match_counts.get(obs["peak_uuid"], 0) + 1
                hkl_counts[ref["hkl_label"]] = hkl_counts.get(ref["hkl_label"], 0) + 1
        replicate_records.append(
            {
                "replicate": replicate + 1,
                "kept_peak_count": len(kept),
                "matched_peak_count": len(matched),
                "matched_fraction": float(matched_fraction),
                "recovered": bool(recovered),
                "assignment_cost": float(assignment.get("assignment_cost", 0.0)),
            }
        )

    recovery_fraction = successes / max(1, int(cfg.replicates))
    recovery_percent = 100.0 * recovery_fraction
    if recovery_percent >= 85.0:
        status = "Strong"
    elif recovery_percent >= 65.0:
        status = "Review"
    else:
        status = "Fragile"

    fragile_peaks = []
    for uuid, count in peak_match_counts.items():
        frequency = count / max(1, successes)
        if successes and frequency < 0.50:
            fragile_peaks.append({"peak_uuid": uuid, "assignment_frequency_percent": 100.0 * frequency})
    fragile_peaks.sort(key=lambda row: row["assignment_frequency_percent"])

    hkl_frequency = [
        {"hkl_label": label, "assignment_frequency_percent": 100.0 * count / max(1, successes)}
        for label, count in hkl_counts.items()
        if successes
    ]
    hkl_frequency.sort(key=lambda row: row["assignment_frequency_percent"], reverse=True)

    warnings = []
    if status == "Fragile":
        warnings.append("Candidate is fragile under peak-position perturbation or random peak omission.")
    if fragile_peaks:
        warnings.append(f"{len(fragile_peaks)} peak(s) have low bootstrap assignment frequency.")
    if len(refined_cells) < max(3, int(0.25 * cfg.replicates)):
        warnings.append("Few bootstrap trials produced a positive-definite metric refinement; review hkl assignments.")

    return {
        "method": "Bootstrap perturbation/omission screen",
        "status": status,
        "settings": asdict(cfg),
        "replicates_requested": int(cfg.replicates),
        "replicates_successful": int(successes),
        "recovery_fraction": float(recovery_fraction),
        "recovery_percent": float(recovery_percent),
        "median_matched_fraction": float(np.median(matched_fractions)) if matched_fractions else 0.0,
        "median_assignment_cost": float(np.median(assignment_costs)) if assignment_costs else 0.0,
        "cell_parameter_intervals": _cell_stat_summary(refined_cells),
        "fragile_peaks": fragile_peaks[:20],
        "hkl_assignment_frequency": hkl_frequency[:40],
        "replicate_records": replicate_records,
        "warnings": warnings,
        "scientific_warning": (
            "Bootstrap recovery is a reliability diagnostic for a candidate cell and assignment set. "
            "It does not prove the atomic structure or space group."
        ),
    }


def summarize_candidate_reliability(candidate: dict) -> str:
    reliability = candidate.get("bootstrap_reliability") or {}
    status = reliability.get("status", "Not run")
    recovery = reliability.get("recovery_percent")
    if recovery is None:
        return str(status)
    return f"{status} ({float(recovery):.1f}% recovery)"
