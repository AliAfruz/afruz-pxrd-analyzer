from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterable

import numpy as np


@dataclass
class ReferenceEntry:
    uid: str
    name: str
    formula: str
    source: str
    wavelength_angstrom: float | None
    peaks: list[dict]
    metadata: dict


def _normalise(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return array
    maximum = float(np.max(array))
    return array / maximum * 100.0 if maximum > 0 else np.zeros_like(array)


def _two_theta_from_d(d_spacing: float, wavelength: float) -> float | None:
    if d_spacing <= 0 or wavelength <= 0:
        return None
    argument = wavelength / (2.0 * d_spacing)
    if not 0.0 < argument < 1.0:
        return None
    return float(2.0 * math.degrees(math.asin(argument)))


def reference_from_dataset(dataset) -> ReferenceEntry | None:
    metadata = dict(getattr(dataset, "metadata", {}) or {})
    if metadata.get("analysis_role") != "reference_pattern" and metadata.get(
        "plot_style"
    ) != "sticks":
        return None

    reflections = metadata.get("reflections")
    peaks: list[dict] = []
    if isinstance(reflections, list) and reflections:
        for row in reflections:
            try:
                position = float(row.get("two_theta"))
                intensity = float(row.get("intensity", 0.0))
            except (TypeError, ValueError):
                continue
            peaks.append(
                {
                    "two_theta": position,
                    "d_spacing": row.get("d_spacing_angstrom"),
                    "intensity": intensity,
                    "hkl_label": row.get("hkl_label", ""),
                    "intensity_text": row.get("intensity_text", str(intensity)),
                }
            )
    else:
        for position, intensity in zip(dataset.x, dataset.y_raw):
            peaks.append(
                {
                    "two_theta": float(position),
                    "d_spacing": None,
                    "intensity": float(intensity),
                    "hkl_label": "",
                    "intensity_text": str(float(intensity)),
                }
            )

    if len(peaks) < 3:
        return None

    return ReferenceEntry(
        uid=str(dataset.uid),
        name=str(dataset.name),
        formula=str(metadata.get("formula") or ""),
        source=str(metadata.get("source_format") or "Imported reference"),
        wavelength_angstrom=(
            None
            if metadata.get("wavelength_k_alpha1") is None
            else float(metadata["wavelength_k_alpha1"])
        ),
        peaks=sorted(peaks, key=lambda row: row["two_theta"]),
        metadata=metadata,
    )


def reference_from_cif_pattern(
    structure: dict | None,
    pattern: list[dict],
    wavelength_angstrom: float,
) -> ReferenceEntry | None:
    if not structure or not pattern:
        return None
    peaks = []
    for row in pattern:
        peaks.append(
            {
                "two_theta": float(row["two_theta"]),
                "d_spacing": float(row["d_spacing"]),
                "intensity": float(row["intensity"]),
                "hkl_label": row.get("hkl_label", ""),
                "intensity_text": str(row.get("intensity", "")),
            }
        )
    return ReferenceEntry(
        uid="active-cif-reference",
        name=f"CIF — {structure.get('data_name', 'active structure')}",
        formula=str(structure.get("formula") or ""),
        source="Calculated from active CIF",
        wavelength_angstrom=float(wavelength_angstrom),
        peaks=peaks,
        metadata={
            "space_group": structure.get("space_group"),
            "crystal_system": structure.get("crystal_system"),
            "cell": structure.get("cell"),
        },
    )


def prepare_reference_peaks(
    reference: ReferenceEntry,
    target_wavelength_angstrom: float,
    convert_from_d: bool,
    intensity_cutoff_percent: float,
    two_theta_min: float,
    two_theta_max: float,
) -> list[dict]:
    cutoff = max(0.0, float(intensity_cutoff_percent))
    prepared = []
    intensities = _normalise(row["intensity"] for row in reference.peaks)
    for row, normalised_intensity in zip(reference.peaks, intensities):
        position = float(row["two_theta"])
        if convert_from_d and row.get("d_spacing") is not None:
            converted = _two_theta_from_d(
                float(row["d_spacing"]),
                float(target_wavelength_angstrom),
            )
            if converted is not None:
                position = converted
        if not two_theta_min <= position <= two_theta_max:
            continue
        if normalised_intensity < cutoff:
            continue
        prepared.append(
            {
                **row,
                "two_theta": float(position),
                "intensity": float(normalised_intensity),
                "reference_uid": reference.uid,
                "reference_name": reference.name,
            }
        )
    return sorted(prepared, key=lambda row: row["two_theta"])


def _one_to_one_matches(
    observed: list[dict],
    reference_peaks: list[dict],
    shift_deg: float,
    tolerance_deg: float,
) -> list[dict]:
    candidates = []
    for observed_index, obs in enumerate(observed):
        for reference_index, ref in enumerate(reference_peaks):
            shifted = float(ref["two_theta"]) + shift_deg
            delta = float(obs["position"]) - shifted
            if abs(delta) <= tolerance_deg:
                candidates.append(
                    (abs(delta), observed_index, reference_index, delta, shifted)
                )

    used_observed = set()
    used_reference = set()
    matches = []
    for _, observed_index, reference_index, delta, shifted in sorted(candidates):
        if observed_index in used_observed or reference_index in used_reference:
            continue
        used_observed.add(observed_index)
        used_reference.add(reference_index)
        obs = observed[observed_index]
        ref = reference_peaks[reference_index]
        matches.append(
            {
                "observed_index": observed_index,
                "reference_index": reference_index,
                "observed_2theta": float(obs["position"]),
                "observed_intensity": float(obs.get("intensity", 0.0)),
                "reference_2theta": float(ref.get("original_two_theta", ref["two_theta"])),
                "shifted_reference_2theta": float(shifted),
                "delta_deg": float(delta),
                "reference_intensity": float(ref["intensity"]),
                "hkl_label": ref.get("hkl_label", ""),
                "reference_uid": ref.get("reference_uid", ""),
                "reference_name": ref.get("reference_name", ""),
            }
        )
    return sorted(matches, key=lambda row: row["observed_2theta"])


def _intensity_correlation(matches: list[dict]) -> float | None:
    if len(matches) < 3:
        return None
    observed = np.asarray([row["observed_intensity"] for row in matches], dtype=float)
    reference = np.asarray([row["reference_intensity"] for row in matches], dtype=float)
    if np.std(observed) <= 0 or np.std(reference) <= 0:
        return None
    value = float(np.corrcoef(np.log1p(observed), np.log1p(reference))[0, 1])
    return value if np.isfinite(value) else None


def _score_matches(
    observed: list[dict],
    reference_peaks: list[dict],
    matches: list[dict],
    tolerance_deg: float,
) -> dict:
    observed_weights = np.sqrt(
        np.maximum(0.0, np.asarray([row.get("intensity", 0.0) for row in observed]))
    )
    reference_weights = np.sqrt(
        np.maximum(0.0, np.asarray([row.get("intensity", 0.0) for row in reference_peaks]))
    )
    matched_observed = {row["observed_index"] for row in matches}
    matched_reference = {row["reference_index"] for row in matches}

    observed_coverage = (
        float(np.sum(observed_weights[list(matched_observed)]) / np.sum(observed_weights))
        if len(observed_weights) and np.sum(observed_weights) > 0 and matched_observed
        else 0.0
    )
    reference_coverage = (
        float(np.sum(reference_weights[list(matched_reference)]) / np.sum(reference_weights))
        if len(reference_weights) and np.sum(reference_weights) > 0 and matched_reference
        else 0.0
    )
    mean_absolute_delta = (
        float(np.mean([abs(row["delta_deg"]) for row in matches]))
        if matches
        else float("inf")
    )
    position_score = (
        math.exp(-((mean_absolute_delta / max(tolerance_deg, 1e-9)) ** 2))
        if matches
        else 0.0
    )
    correlation = _intensity_correlation(matches)
    correlation_score = 0.5 if correlation is None else (correlation + 1.0) / 2.0
    count_factor = min(1.0, len(matches) / 4.0)

    score = 100.0 * count_factor * (
        0.50 * reference_coverage
        + 0.25 * observed_coverage
        + 0.20 * position_score
        + 0.05 * correlation_score
    )
    return {
        "score": float(score),
        "observed_coverage_percent": float(100.0 * observed_coverage),
        "reference_coverage_percent": float(100.0 * reference_coverage),
        "mean_absolute_delta_deg": mean_absolute_delta,
        "intensity_correlation": correlation,
        "matched_count": len(matches),
    }


def identify_single_reference(
    observed: list[dict],
    reference: ReferenceEntry,
    prepared_peaks: list[dict],
    tolerance_deg: float,
    maximum_zero_shift_deg: float,
) -> dict:
    if len(prepared_peaks) < 2:
        return {
            "reference_uid": reference.uid,
            "reference_name": reference.name,
            "formula": reference.formula,
            "source": reference.source,
            "score": 0.0,
            "matched_count": 0,
            "matches": [],
            "zero_shift_deg": 0.0,
            "notes": ["Too few reference peaks in the selected range."],
        }

    candidate_shifts = {0.0}
    maximum_shift = max(0.0, float(maximum_zero_shift_deg))
    for obs in observed:
        for ref in prepared_peaks:
            delta = float(obs["position"]) - float(ref["two_theta"])
            if abs(delta) <= maximum_shift + tolerance_deg:
                candidate_shifts.add(round(float(np.clip(delta, -maximum_shift, maximum_shift)), 5))

    best = None
    for shift in sorted(candidate_shifts):
        matches = _one_to_one_matches(observed, prepared_peaks, shift, tolerance_deg)
        metrics = _score_matches(observed, prepared_peaks, matches, tolerance_deg)
        candidate = {
            "reference_uid": reference.uid,
            "reference_name": reference.name,
            "formula": reference.formula,
            "source": reference.source,
            "zero_shift_deg": float(shift),
            "reference_peak_count": len(prepared_peaks),
            "matches": matches,
            "notes": [],
            **metrics,
        }
        if best is None or (
            candidate["score"], candidate["matched_count"], -abs(candidate["zero_shift_deg"])
        ) > (
            best["score"], best["matched_count"], -abs(best["zero_shift_deg"])
        ):
            best = candidate

    if best is None:
        raise RuntimeError("No identification candidate was evaluated.")
    if best["matched_count"] < 3:
        best["notes"].append("Fewer than three peaks matched.")
    if best["reference_coverage_percent"] < 35.0:
        best["notes"].append("Low weighted reference-peak coverage.")
    if abs(best["zero_shift_deg"]) > 0.25:
        best["notes"].append("Large fitted zero shift; verify calibration and wavelength.")
    return best


def _evaluate_mixture(
    observed: list[dict],
    members: list[dict],
    tolerance_deg: float,
) -> dict | None:
    combined_peaks = []
    for member_index, member in enumerate(members):
        for peak_index, peak in enumerate(member["prepared_peaks"]):
            combined_peaks.append(
                {
                    **peak,
                    "original_two_theta": float(peak["two_theta"]),
                    "two_theta": float(peak["two_theta"]) + float(member["zero_shift_deg"]),
                    "member_index": member_index,
                    "original_reference_index": peak_index,
                }
            )

    matches = _one_to_one_matches(observed, combined_peaks, 0.0, tolerance_deg)
    if not matches:
        return None
    matched_by_member = {index: [] for index in range(len(members))}
    for row in matches:
        member_index = combined_peaks[row["reference_index"]]["member_index"]
        matched_by_member[member_index].append(row)
    if any(len(rows) < 2 for rows in matched_by_member.values()):
        return None

    metrics = _score_matches(observed, combined_peaks, matches, tolerance_deg)
    complexity_penalty = 4.0 * (len(members) - 1)
    score = max(0.0, metrics["score"] - complexity_penalty)
    phase_names = [member["reference_name"] for member in members]
    zero_shifts = [member["zero_shift_deg"] for member in members]

    return {
        "kind": "mixture",
        "phase_count": len(members),
        "reference_uids": [member["reference_uid"] for member in members],
        "reference_name": " + ".join(phase_names),
        "formula": " + ".join(member.get("formula") or "?" for member in members),
        "source": "Local reference-library mixture search",
        "zero_shift_deg": zero_shifts,
        "reference_peak_count": len(combined_peaks),
        "matches": matches,
        "notes": [f"Complexity penalty: {complexity_penalty:.1f} score points."],
        **metrics,
        "score": float(score),
    }


def identify_phases(
    observed_peaks: list[dict],
    references: list[ReferenceEntry],
    target_wavelength_angstrom: float,
    tolerance_deg: float = 0.20,
    maximum_zero_shift_deg: float = 0.30,
    reference_intensity_cutoff_percent: float = 1.0,
    convert_from_d: bool = True,
    maximum_phases: int = 1,
    mixture_pool_size: int = 6,
) -> dict:
    if len(observed_peaks) < 2:
        raise ValueError("At least two observed peaks are required for phase identification.")
    if not references:
        raise ValueError("The local reference library is empty.")

    observed = sorted(
        [
            {
                "position": float(row["position"]),
                "intensity": float(row.get("intensity", 0.0)),
                "source": row.get("source", "Observed"),
            }
            for row in observed_peaks
        ],
        key=lambda row: row["position"],
    )
    normalised_observed = _normalise(row["intensity"] for row in observed)
    for row, intensity in zip(observed, normalised_observed):
        row["intensity"] = float(intensity)

    two_theta_min = min(row["position"] for row in observed) - tolerance_deg
    two_theta_max = max(row["position"] for row in observed) + tolerance_deg

    singles = []
    for reference in references:
        prepared = prepare_reference_peaks(
            reference,
            target_wavelength_angstrom=target_wavelength_angstrom,
            convert_from_d=convert_from_d,
            intensity_cutoff_percent=reference_intensity_cutoff_percent,
            two_theta_min=two_theta_min,
            two_theta_max=two_theta_max,
        )
        result = identify_single_reference(
            observed,
            reference,
            prepared,
            tolerance_deg=tolerance_deg,
            maximum_zero_shift_deg=maximum_zero_shift_deg,
        )
        result.update(
            {
                "kind": "single",
                "phase_count": 1,
                "reference_uids": [reference.uid],
                "prepared_peaks": prepared,
            }
        )
        singles.append(result)

    singles.sort(key=lambda row: (row["score"], row["matched_count"]), reverse=True)
    results = list(singles)
    maximum_phases = int(np.clip(maximum_phases, 1, 3))
    pool = singles[: max(2, int(mixture_pool_size))]
    if maximum_phases >= 2:
        for phase_count in range(2, maximum_phases + 1):
            for members in combinations(pool, phase_count):
                mixture = _evaluate_mixture(observed, list(members), tolerance_deg)
                if mixture is not None:
                    results.append(mixture)

    results.sort(key=lambda row: (row["score"], row["matched_count"]), reverse=True)
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
        result.pop("prepared_peaks", None)

    best = results[0] if results else None
    return {
        "observed_peak_count": len(observed),
        "reference_count": len(references),
        "target_wavelength_angstrom": float(target_wavelength_angstrom),
        "tolerance_deg": float(tolerance_deg),
        "maximum_zero_shift_deg": float(maximum_zero_shift_deg),
        "reference_intensity_cutoff_percent": float(reference_intensity_cutoff_percent),
        "convert_from_d": bool(convert_from_d),
        "maximum_phases": maximum_phases,
        "results": results,
        "best": best,
    }
