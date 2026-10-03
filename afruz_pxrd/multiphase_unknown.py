from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .deep_unknown_indexing import assign_peaks, sigma_two_theta_to_sigma_q, two_theta_to_q


@dataclass(frozen=True)
class MultiphaseDiscoverySettings:
    """Settings for Phase 17.9 residual unknown-phase discovery.

    This is deliberately conservative: it proposes a second unknown phase only
    from stable residual peaks and requires the joint two-phase assignment to
    improve a BIC-like score enough to overcome an added-phase penalty.
    """

    enabled: bool = True
    top_primary_candidates: int = 1
    minimum_residual_peaks: int = 7
    minimum_residual_signal_to_noise: float = 3.5
    minimum_residual_quality_score: float = 25.0
    overlap_tolerance_deg: float = 0.08
    assignment_outlier_cost: float = 8.0
    model_sigma_q_scale: float = 0.25
    added_phase_penalty: float = 8.0
    bic_improvement_threshold: float = 6.0
    maximum_second_phase_candidates: int = 3
    phase_overlap_warning_fraction: float = 0.20

    def validate(self) -> "MultiphaseDiscoverySettings":
        if self.top_primary_candidates < 0:
            raise ValueError("Top primary candidate count cannot be negative.")
        if self.minimum_residual_peaks < 3:
            raise ValueError("At least three residual peaks are required for screening.")
        if self.minimum_residual_signal_to_noise < 0:
            raise ValueError("Minimum residual S/N cannot be negative.")
        if self.minimum_residual_quality_score < 0:
            raise ValueError("Minimum residual quality cannot be negative.")
        if self.overlap_tolerance_deg < 0:
            raise ValueError("Overlap tolerance cannot be negative.")
        if self.assignment_outlier_cost <= 0:
            raise ValueError("Assignment outlier cost must be positive.")
        if self.model_sigma_q_scale < 0:
            raise ValueError("Model sigma scale cannot be negative.")
        if not 0.0 <= self.phase_overlap_warning_fraction <= 1.0:
            raise ValueError("Phase-overlap warning fraction must be within [0, 1].")
        return self


def _peak_uuid(row: dict, index: int) -> str:
    return str(row.get("peak_uuid") or row.get("uuid") or f"peak-{index + 1}")


def _peak_position(row: dict) -> float:
    for key in ("two_theta_deg", "observed_two_theta_deg", "position_deg"):
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return math.nan


def _peak_uncertainty(row: dict) -> float:
    value = row.get("position_uncertainty_deg", row.get("uncertainty_deg", 0.02))
    try:
        return float(np.clip(float(value), 0.001, 0.5))
    except (TypeError, ValueError):
        return 0.02


def _peak_snr(row: dict) -> float:
    try:
        return float(row.get("signal_to_noise", row.get("snr", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _peak_quality(row: dict) -> float:
    try:
        return float(row.get("quality_score", row.get("confidence", 50.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _enabled_peaks(peaks: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for index, row in enumerate(peaks):
        if not bool(row.get("use", True)):
            continue
        position = _peak_position(row)
        if not np.isfinite(position) or not 0.0 < position < 180.0:
            continue
        rows.append(
            {
                "peak_uuid": _peak_uuid(row, index),
                "two_theta_deg": float(position),
                "position_uncertainty_deg": _peak_uncertainty(row),
                "signal_to_noise": _peak_snr(row),
                "quality_score": _peak_quality(row),
                "source_index": index,
            }
        )
    rows.sort(key=lambda row: row["two_theta_deg"])
    return rows


def _candidate_reflections(candidate: dict, *, phase_label: str) -> list[dict]:
    reflections: list[dict] = []
    seen = set()
    for row in candidate.get("assignments", []) or []:
        try:
            h = int(row["h"])
            k = int(row["k"])
            l = int(row["l"])
            position = float(row["predicted_two_theta_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (phase_label, h, k, l, round(position, 6))
        if key in seen:
            continue
        seen.add(key)
        reflections.append(
            {
                "phase_label": phase_label,
                "h": h,
                "k": k,
                "l": l,
                "hkl_label": f"({h} {k} {l})",
                "predicted_two_theta_deg": float(position),
            }
        )
    reflections.sort(key=lambda row: row["predicted_two_theta_deg"])
    return reflections


def classify_residual_peaks(
    primary_candidate: dict,
    peaks: Iterable[dict],
    *,
    settings: MultiphaseDiscoverySettings | None = None,
) -> dict:
    """Classify observed peaks after a primary candidate cell assignment.

    Peaks are not removed from the scientific record. This simply labels which
    peaks are indexed by phase 1, near phase-1 calculated lines, stable residual
    unknown peaks, weak/noisy peaks, or excluded from the curated list.
    """

    cfg = (settings or MultiphaseDiscoverySettings()).validate()
    enabled = _enabled_peaks(peaks)
    assigned_uuids = {str(row.get("peak_uuid")) for row in primary_candidate.get("assignments", []) if row.get("peak_uuid")}
    primary_positions = np.asarray(
        [float(row["predicted_two_theta_deg"]) for row in primary_candidate.get("assignments", []) if "predicted_two_theta_deg" in row],
        dtype=float,
    )
    rows: list[dict] = []
    for row in enabled:
        uuid = row["peak_uuid"]
        position = float(row["two_theta_deg"])
        nearest_primary_delta = None
        if primary_positions.size:
            nearest_primary_delta = float(np.min(np.abs(primary_positions - position)))
        if uuid in assigned_uuids:
            category = "indexed_by_primary"
        elif nearest_primary_delta is not None and nearest_primary_delta <= cfg.overlap_tolerance_deg:
            category = "possible_primary_overlap"
        elif (
            row["signal_to_noise"] >= cfg.minimum_residual_signal_to_noise
            and row["quality_score"] >= cfg.minimum_residual_quality_score
        ):
            category = "stable_unindexed_residual"
        elif row["signal_to_noise"] < cfg.minimum_residual_signal_to_noise:
            category = "weak_or_noisy_unindexed"
        else:
            category = "review_unindexed"
        rows.append(
            {
                **row,
                "category": category,
                "nearest_primary_delta_deg": nearest_primary_delta,
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    residual = [row for row in rows if row["category"] == "stable_unindexed_residual"]
    return {
        "method": "Phase 17.9 residual-peak grouping",
        "settings": asdict(cfg),
        "primary_candidate_rank": primary_candidate.get("rank"),
        "primary_bravais": primary_candidate.get("bravais_name"),
        "observed_peak_count": len(enabled),
        "counts": counts,
        "stable_residual_peak_count": len(residual),
        "stable_residual_peak_uuids": [row["peak_uuid"] for row in residual],
        "rows": rows,
        "classification": (
            "Second-phase search ready"
            if len(residual) >= cfg.minimum_residual_peaks
            else "Insufficient stable residual peaks"
        ),
    }


def stable_residual_peak_rows(residual_record: dict) -> list[dict]:
    return [row for row in residual_record.get("rows", []) if row.get("category") == "stable_unindexed_residual"]


def joint_two_phase_assignment(
    primary_candidate: dict,
    secondary_candidate: dict,
    peaks: Iterable[dict],
    *,
    wavelength_angstrom: float,
    settings: MultiphaseDiscoverySettings | None = None,
) -> dict:
    """Assign observed peaks against a combined primary+secondary reflection list."""

    cfg = (settings or MultiphaseDiscoverySettings()).validate()
    observations = _enabled_peaks(peaks)
    primary = _candidate_reflections(primary_candidate, phase_label="phase_1")
    secondary = _candidate_reflections(secondary_candidate, phase_label="phase_2")
    reflections = primary + secondary
    reflections.sort(key=lambda row: row["predicted_two_theta_deg"])
    if not observations or not reflections:
        return {
            "method": "Joint two-phase Q-space assignment",
            "classification": "Insufficient data",
            "matched_peak_count": 0,
            "unindexed_peak_count": len(observations),
            "phase_1_matched_count": 0,
            "phase_2_matched_count": 0,
            "warnings": ["No observations or predicted reflections were available for joint assignment."],
        }
    observed_positions = np.asarray([row["two_theta_deg"] for row in observations], dtype=float)
    observed_sigmas = np.asarray([row["position_uncertainty_deg"] for row in observations], dtype=float)
    predicted_positions = np.asarray([row["predicted_two_theta_deg"] for row in reflections], dtype=float)
    observed_q = two_theta_to_q(observed_positions, wavelength_angstrom)
    sigma_q = sigma_two_theta_to_sigma_q(observed_positions, observed_sigmas, wavelength_angstrom)
    predicted_q = two_theta_to_q(predicted_positions, wavelength_angstrom)
    assignment = assign_peaks(
        observed_q,
        sigma_q,
        predicted_q,
        outlier_cost=cfg.assignment_outlier_cost,
        model_sigma_q=float(np.nanmedian(sigma_q)) * cfg.model_sigma_q_scale if sigma_q.size else 0.0,
    )
    phase_counts = {"phase_1": 0, "phase_2": 0}
    matched_rows = []
    for match in assignment.get("matches", []):
        obs = observations[int(match["observed_index"])]
        ref = reflections[int(match["predicted_index"])]
        phase_counts[ref["phase_label"]] = phase_counts.get(ref["phase_label"], 0) + 1
        matched_rows.append(
            {
                "peak_uuid": obs["peak_uuid"],
                "observed_two_theta_deg": obs["two_theta_deg"],
                "assigned_phase": ref["phase_label"],
                "hkl_label": ref["hkl_label"],
                "predicted_two_theta_deg": ref["predicted_two_theta_deg"],
                "normalized_delta": float(match.get("normalized_delta", math.nan)),
                "assignment_cost": float(match.get("assignment_cost", 0.0)),
            }
        )
    matched_count = len(matched_rows)
    unindexed_count = len(assignment.get("unindexed", []))
    observed_count = max(1, len(observations))
    pair_cost = float(sum(row["assignment_cost"] for row in matched_rows))
    # BIC-like score: pair assignment cost + unindexed penalty + phase penalty.
    joint_bic_like = (
        pair_cost
        + cfg.assignment_outlier_cost * unindexed_count
        + 2.0 * cfg.added_phase_penalty * math.log(observed_count + 1.0)
    )
    primary_bic = float(primary_candidate.get("bic_like", primary_candidate.get("objective", joint_bic_like)))
    bic_improvement = primary_bic - joint_bic_like
    phase_2_residual_matches = phase_counts.get("phase_2", 0)
    if phase_2_residual_matches >= cfg.minimum_residual_peaks and bic_improvement >= cfg.bic_improvement_threshold:
        classification = "Second phase justified"
    elif phase_2_residual_matches >= max(3, cfg.minimum_residual_peaks // 2):
        classification = "Possible second phase"
    else:
        classification = "Single phase sufficient or second phase unstable"
    warnings = []
    if classification != "Second phase justified":
        warnings.append("Second unknown phase did not pass the conservative BIC/residual-coverage threshold.")
    if unindexed_count / observed_count > 0.25:
        warnings.append("More than 25% of observed peaks remain unassigned after the two-phase assignment.")
    return {
        "method": "Joint two-phase Q-space assignment",
        "classification": classification,
        "primary_candidate_rank": primary_candidate.get("rank"),
        "secondary_candidate_rank": secondary_candidate.get("rank"),
        "primary_bravais": primary_candidate.get("bravais_name"),
        "secondary_bravais": secondary_candidate.get("bravais_name"),
        "matched_peak_count": matched_count,
        "unindexed_peak_count": unindexed_count,
        "unindexed_fraction": unindexed_count / observed_count,
        "phase_1_matched_count": phase_counts.get("phase_1", 0),
        "phase_2_matched_count": phase_counts.get("phase_2", 0),
        "assignment_cost": pair_cost,
        "joint_bic_like": float(joint_bic_like),
        "primary_bic_like": float(primary_bic),
        "bic_improvement": float(bic_improvement),
        "matched_assignments": matched_rows,
        "warnings": warnings,
        "scientific_warning": (
            "A second unknown phase is a hypothesis. It should be retained only after Pawley/Le Bail or "
            "Rietveld-compatible validation and chemical review. Overlapped powder peaks can still belong to "
            "more than one physical phase."
        ),
    }


def summarize_multiphase_discovery(record: dict) -> str:
    status = record.get("classification") or record.get("status") or "Not run"
    residual_count = record.get("residual_peak_grouping", {}).get("stable_residual_peak_count")
    joint = record.get("joint_assignment") or {}
    if joint.get("classification"):
        return f"{joint['classification']} ({residual_count or 0} stable residual peaks)"
    if residual_count is not None:
        return f"{status} ({residual_count} stable residual peaks)"
    return str(status)
