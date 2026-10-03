from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .text_export import write_table_txt
from scipy.signal import find_peaks, peak_widths

from .crystallography import d_spacing
from .models import Dataset
from .native_indexing import (
    NativeIndexingCancelled,
    NativeIndexingError,
    run_native_indexing,
)
from .smart_smoothing import detect_smart_peak_regions


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


class PhaseRevolutionError(ValueError):
    pass


class IndexingBackendError(NativeIndexingError):
    """Compatibility name for native indexing failures."""



@dataclass
class ResidualPattern:
    x: np.ndarray
    observed: np.ndarray
    background: np.ndarray
    known_phase_calculated: np.ndarray
    signed_residual: np.ndarray
    positive_residual: np.ndarray
    source: str
    warnings: list[str]

    def as_dict(self) -> dict:
        return {
            "x": self.x.tolist(),
            "observed": self.observed.tolist(),
            "background": self.background.tolist(),
            "known_phase_calculated": self.known_phase_calculated.tolist(),
            "signed_residual": self.signed_residual.tolist(),
            "positive_residual": self.positive_residual.tolist(),
            "source": self.source,
            "warnings": list(self.warnings),
        }


def _aligned_array(values, target_x: np.ndarray, source_x=None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if source_x is None:
        if array.shape != target_x.shape:
            raise PhaseRevolutionError(
                "The refinement result is not aligned with the selected dataset."
            )
        return array
    source_x = np.asarray(source_x, dtype=float)
    if source_x.shape == target_x.shape and np.allclose(source_x, target_x):
        return array
    if len(source_x) < 2 or len(array) != len(source_x):
        raise PhaseRevolutionError("Invalid refinement x/y arrays.")
    return np.interp(target_x, source_x, array, left=np.nan, right=np.nan)


def extract_unknown_residual(
    x: np.ndarray,
    observed: np.ndarray,
    *,
    whole_pattern_result: dict | None = None,
    rietveld_result: dict | None = None,
    subtract_background: bool = True,
) -> ResidualPattern:
    """Create a residual pattern after subtracting the selected known-phase model.

    Priority is given to a Rietveld result, then to a Pawley/Le Bail result. The
    returned ``positive_residual`` is clipped only for peak detection; the signed
    residual is retained for scientific inspection.
    """
    x = np.asarray(x, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if x.ndim != 1 or observed.ndim != 1 or x.shape != observed.shape:
        raise PhaseRevolutionError("X and observed intensity must be aligned 1-D arrays.")
    if len(x) < 20:
        raise PhaseRevolutionError("At least twenty data points are required.")

    result = rietveld_result or whole_pattern_result
    source = "Observed pattern only"
    warnings: list[str] = []
    background = np.zeros_like(observed)
    known = np.zeros_like(observed)

    if result:
        source = str(result.get("mode") or result.get("engine") or "Refined known phases")
        source_x = result.get("observed_x")
        calculated = _aligned_array(result.get("calculated_y", []), x, source_x)
        background = _aligned_array(result.get("background_y", np.zeros_like(calculated)), x, source_x)
        known = calculated - background
        invalid = ~np.isfinite(calculated) | ~np.isfinite(background)
        if np.any(invalid):
            warnings.append(
                "The selected refinement does not cover the complete dataset range; "
                "out-of-range points were excluded from peak detection."
            )
            calculated = np.where(invalid, observed, calculated)
            background = np.where(invalid, 0.0, background)
            known = np.where(invalid, 0.0, known)
        signed = observed - known - (background if subtract_background else 0.0)
    else:
        warnings.append(
            "No Phase 10 or Phase 11 result was available. Peak detection uses the "
            "selected observed pattern without known-phase subtraction."
        )
        signed = observed.copy()

    positive = np.maximum(signed, 0.0)
    return ResidualPattern(
        x=x.copy(),
        observed=observed.copy(),
        background=background,
        known_phase_calculated=known,
        signed_residual=signed,
        positive_residual=positive,
        source=source,
        warnings=warnings,
    )


def detect_unknown_peaks(
    x: np.ndarray,
    residual: np.ndarray,
    *,
    prominence_percent: float = 3.0,
    minimum_distance_deg: float = 0.12,
    minimum_two_theta: float | None = None,
    maximum_two_theta: float | None = None,
    known_reflections: Iterable[float] | None = None,
    overlap_exclusion_deg: float = 0.08,
    maximum_peaks: int = 40,
    smart_search: bool = True,
    minimum_signal_to_noise: float = 4.0,
    minimum_quality_score: float = 35.0,
) -> list[dict]:
    """Detect reliable residual reflections for indexing.

    Smart mode reuses the peak-aware detector from the smoothing engine. It
    estimates noise robustly, detects peaks on a mildly filtered copy, ranks
    candidates by signal-to-noise and prominence, and keeps the original
    residual intensity/position for the exported peak list. Manual mode retains
    the original range-percent prominence search for compatibility.
    """
    x = np.asarray(x, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if x.shape != residual.shape or x.ndim != 1:
        raise PhaseRevolutionError("Residual x/y arrays are not aligned.")
    if np.any(np.diff(x) <= 0):
        raise PhaseRevolutionError("2θ values must be strictly increasing.")

    minimum = float(x[0] if minimum_two_theta is None else minimum_two_theta)
    maximum = float(x[-1] if maximum_two_theta is None else maximum_two_theta)
    mask = (x >= minimum) & (x <= maximum) & np.isfinite(residual)
    if np.count_nonzero(mask) < 10:
        raise PhaseRevolutionError("The selected peak-search range is too small.")
    xx = x[mask]
    yy = np.maximum(residual[mask], 0.0)
    dynamic = float(np.max(yy) - np.min(yy))
    if dynamic <= np.finfo(float).eps:
        return []

    step = float(np.median(np.diff(xx)))
    known = np.asarray(list(known_reflections or []), dtype=float)
    raw_rows: list[dict] = []

    if smart_search:
        detection = detect_smart_peak_regions(
            xx,
            yy,
            minimum_signal_to_noise=max(2.0, float(minimum_signal_to_noise)),
            width_multiplier=1.5,
        )
        noise_sigma = float(detection["noise_sigma"])
        for detected in detection["peaks"]:
            index = int(detected["index"])
            position = float(xx[index])
            # Three-point parabolic interpolation improves the position without
            # replacing the original measured residual used for intensity.
            if 0 < index < len(xx) - 1:
                y0, y1, y2 = yy[index - 1 : index + 2]
                denominator = float(y0 - 2.0 * y1 + y2)
                if abs(denominator) > np.finfo(float).eps:
                    offset = float(np.clip(0.5 * (y0 - y2) / denominator, -0.5, 0.5))
                    position = float(np.interp(index + offset, np.arange(len(xx)), xx))
            left_x = float(np.interp(detected["left_ip"], np.arange(len(xx)), xx))
            right_x = float(np.interp(detected["right_ip"], np.arange(len(xx)), xx))
            raw_rows.append(
                {
                    "two_theta_deg": position,
                    "intensity": float(yy[index]),
                    "prominence": float(detected["prominence"]),
                    "fwhm_deg": max(0.0, right_x - left_x),
                    "signal_to_noise": float(detected["signal_to_noise"]),
                    "quality_score": float(detected["quality_score"]),
                    "noise_sigma": noise_sigma,
                    "detection_method": detection["method"],
                    "origin": "Smart detected",
                    "manual": False,
                    "row_locked": False,
                }
            )
        raw_rows = [
            row for row in raw_rows
            if row["signal_to_noise"] >= max(2.0, float(minimum_signal_to_noise))
            and row["quality_score"] >= max(0.0, float(minimum_quality_score))
        ]
    else:
        distance_points = max(
            1,
            int(round(max(0.0, minimum_distance_deg) / step)),
        )
        prominence = dynamic * max(0.0, float(prominence_percent)) / 100.0
        indices, properties = find_peaks(
            yy,
            prominence=max(prominence, np.finfo(float).eps),
            distance=distance_points,
        )
        if not len(indices):
            return []
        widths, _, left_ips, right_ips = peak_widths(
            yy,
            indices,
            rel_height=0.5,
        )
        noise_sigma = max(
            float(np.median(np.abs(np.diff(yy) - np.median(np.diff(yy)))) / 0.6744897501960817),
            np.finfo(float).eps,
        )
        for local_index, index in enumerate(indices):
            left_x = float(np.interp(left_ips[local_index], np.arange(len(xx)), xx))
            right_x = float(np.interp(right_ips[local_index], np.arange(len(xx)), xx))
            prominence_value = float(properties["prominences"][local_index])
            signal_to_noise = prominence_value / noise_sigma
            quality = 100.0 * min(1.0, prominence_value / max(dynamic * 0.15, noise_sigma))
            raw_rows.append(
                {
                    "two_theta_deg": float(xx[index]),
                    "intensity": float(yy[index]),
                    "prominence": prominence_value,
                    "fwhm_deg": max(0.0, right_x - left_x),
                    "signal_to_noise": float(signal_to_noise),
                    "quality_score": float(quality),
                    "noise_sigma": float(noise_sigma),
                    "detection_method": "Manual prominence detector",
                    "origin": "Prominence detected",
                    "manual": False,
                    "row_locked": False,
                }
            )

    # Enforce the physical minimum separation after smart detection, retaining
    # the higher-quality candidate in each local cluster.
    ranked = sorted(
        raw_rows,
        key=lambda row: (
            row.get("quality_score", 0.0),
            row.get("signal_to_noise", 0.0),
            row.get("prominence", 0.0),
        ),
        reverse=True,
    )
    accepted: list[dict] = []
    for row in ranked:
        if any(
            abs(row["two_theta_deg"] - kept["two_theta_deg"])
            < max(0.0, float(minimum_distance_deg))
            for kept in accepted
        ):
            continue
        accepted.append(row)

    maximum_peaks = max(7, min(int(maximum_peaks), 120))
    accepted = accepted[:maximum_peaks]
    rows = []
    for row in accepted:
        position = float(row["two_theta_deg"])
        nearest_known = (
            float(np.min(np.abs(known - position))) if known.size else None
        )
        excluded_overlap = bool(
            nearest_known is not None
            and nearest_known <= max(0.0, float(overlap_exclusion_deg))
        )
        use = not excluded_overlap
        rows.append(
            {
                "peak_id": len(rows) + 1,
                "use": use,
                **row,
                "position_uncertainty_deg": max(
                    step / math.sqrt(12.0),
                    min(max(row.get("fwhm_deg", step) / max(2.0, row.get("signal_to_noise", 2.0)), step), 0.05),
                    1e-5,
                ),
                "nearest_known_distance_deg": nearest_known,
                "overlap_flag": excluded_overlap,
                "note": (
                    "Near a known-phase reflection"
                    if excluded_overlap
                    else "Smart noise-aware peak"
                    if smart_search
                    else ""
                ),
            }
        )
    rows.sort(key=lambda row: row["two_theta_deg"])
    for index, row in enumerate(rows, start=1):
        row["peak_id"] = index
    return normalize_master_peak_list(rows)



def normalize_master_peak_list(peaks: Iterable[dict]) -> list[dict]:
    """Return a stable, position-sorted master reflection list.

    Unknown extra fields are retained so future phases can add annotations
    without breaking older project files. Peak IDs are reassigned only after
    sorting; ``peak_uuid`` remains stable and is the cross-analysis identity.
    """
    rows: list[dict] = []
    seen_uuids: set[str] = set()
    for index, source in enumerate(peaks, start=1):
        if not isinstance(source, dict):
            continue
        row = deepcopy(source)
        try:
            position = float(row.get("two_theta_deg"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(position):
            continue
        row["two_theta_deg"] = position
        for key, default in (
            ("position_uncertainty_deg", 0.01),
            ("intensity", 0.0),
            ("prominence", 0.0),
            ("signal_to_noise", 0.0),
            ("quality_score", 0.0),
            ("fwhm_deg", 0.0),
        ):
            try:
                value = float(row.get(key, default))
            except (TypeError, ValueError):
                value = float(default)
            row[key] = value if np.isfinite(value) else float(default)
        nearest = row.get("nearest_known_distance_deg")
        try:
            nearest = None if nearest in (None, "", "—") else float(nearest)
        except (TypeError, ValueError):
            nearest = None
        row["nearest_known_distance_deg"] = nearest
        row["use"] = bool(row.get("use", True))
        row["overlap_flag"] = bool(row.get("overlap_flag", False))
        row["note"] = str(row.get("note", ""))
        row["origin"] = str(row.get("origin") or row.get("detection_method") or "Imported")
        row["manual"] = bool(row.get("manual", row["origin"].lower().startswith("manual")))
        row["row_locked"] = bool(row.get("row_locked", row["manual"]))
        peak_uuid = str(row.get("peak_uuid") or "").strip()
        if not peak_uuid or peak_uuid in seen_uuids:
            seed = (
                f"{position:.10f}|{row['origin']}|{index}|{row.get('note', '')}"
            )
            peak_uuid = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        seen_uuids.add(peak_uuid)
        row["peak_uuid"] = peak_uuid
        row.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
        row["modified_utc"] = str(row.get("modified_utc") or row["created_utc"])
        rows.append(row)
    rows.sort(key=lambda row: (row["two_theta_deg"], row["peak_uuid"]))
    for peak_id, row in enumerate(rows, start=1):
        row["peak_id"] = peak_id
    return rows


def master_peak_list_checksum(peaks: Iterable[dict]) -> str:
    rows = normalize_master_peak_list(peaks)
    canonical = []
    for row in rows:
        canonical.append(
            {
                "peak_uuid": row["peak_uuid"],
                "use": row["use"],
                "two_theta_deg": round(row["two_theta_deg"], 10),
                "position_uncertainty_deg": round(row["position_uncertainty_deg"], 10),
                "intensity": round(row["intensity"], 8),
                "prominence": round(row["prominence"], 8),
                "signal_to_noise": round(row["signal_to_noise"], 8),
                "quality_score": round(row["quality_score"], 8),
                "fwhm_deg": round(row["fwhm_deg"], 10),
                "nearest_known_distance_deg": (
                    None
                    if row["nearest_known_distance_deg"] is None
                    else round(row["nearest_known_distance_deg"], 10)
                ),
                "overlap_flag": row["overlap_flag"],
                "origin": row["origin"],
                "manual": row["manual"],
                "row_locked": row["row_locked"],
                "note": row["note"],
            }
        )
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manual_peak_from_position(
    x: np.ndarray,
    residual: np.ndarray,
    two_theta_deg: float,
    *,
    known_reflections: Iterable[float] | None = None,
    overlap_exclusion_deg: float = 0.08,
    default_fwhm_deg: float = 0.12,
) -> dict:
    """Create a measured manual reflection at the nearest residual maximum."""
    x = np.asarray(x, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if x.ndim != 1 or x.shape != residual.shape or len(x) < 5:
        raise PhaseRevolutionError("A valid residual pattern is required for manual peak insertion.")
    if np.any(np.diff(x) <= 0):
        raise PhaseRevolutionError("2θ values must be strictly increasing.")
    target = float(two_theta_deg)
    if not float(x[0]) <= target <= float(x[-1]):
        raise PhaseRevolutionError(
            f"Manual peak position must be inside {x[0]:.6g}–{x[-1]:.6g}° 2θ."
        )
    step = float(np.median(np.diff(x)))
    center = int(np.argmin(np.abs(x - target)))
    search_half = max(1, int(round(max(default_fwhm_deg, 4.0 * step) / step)))
    left = max(0, center - search_half)
    right = min(len(x), center + search_half + 1)
    local = np.maximum(residual[left:right], 0.0)
    index = left + int(np.argmax(local)) if len(local) else center
    position = float(x[index])
    intensity = float(max(residual[index], 0.0))

    baseline_half = max(search_half * 4, 4)
    bleft = max(0, index - baseline_half)
    bright = min(len(x), index + baseline_half + 1)
    local_values = np.maximum(residual[bleft:bright], 0.0)
    local_baseline = float(np.percentile(local_values, 20.0)) if len(local_values) else 0.0
    prominence = max(0.0, intensity - local_baseline)
    differences = np.diff(residual[max(0, bleft): min(len(residual), bright)])
    if len(differences):
        median = float(np.median(differences))
        noise_sigma = float(np.median(np.abs(differences - median)) / 0.6744897501960817)
    else:
        noise_sigma = 0.0
    noise_sigma = max(noise_sigma, np.finfo(float).eps)
    signal_to_noise = prominence / noise_sigma

    half_height = local_baseline + 0.5 * prominence
    left_index = index
    while left_index > 0 and residual[left_index] > half_height:
        left_index -= 1
    right_index = index
    while right_index < len(residual) - 1 and residual[right_index] > half_height:
        right_index += 1
    fwhm = max(step, float(x[right_index] - x[left_index]))
    if prominence <= 0:
        fwhm = max(step, float(default_fwhm_deg))
    quality = float(np.clip(25.0 + 15.0 * math.log10(max(signal_to_noise, 1.0)), 0.0, 100.0))

    known = np.asarray(list(known_reflections or []), dtype=float)
    nearest_known = float(np.min(np.abs(known - position))) if known.size else None
    overlap = bool(
        nearest_known is not None
        and nearest_known <= max(0.0, float(overlap_exclusion_deg))
    )
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "peak_id": 1,
        "peak_uuid": hashlib.sha256(f"manual|{position:.10f}|{now}".encode()).hexdigest()[:16],
        "use": True,
        "two_theta_deg": position,
        "position_uncertainty_deg": max(step / math.sqrt(12.0), min(fwhm / max(signal_to_noise, 2.0), 0.05)),
        "intensity": intensity,
        "prominence": prominence,
        "signal_to_noise": signal_to_noise,
        "quality_score": quality,
        "fwhm_deg": fwhm,
        "noise_sigma": noise_sigma,
        "nearest_known_distance_deg": nearest_known,
        "overlap_flag": overlap,
        "origin": "Manual",
        "detection_method": "Manual peak insertion",
        "manual": True,
        "row_locked": True,
        "created_utc": now,
        "modified_utc": now,
        "note": "Manual peak" + ("; near a known-phase reflection" if overlap else ""),
    }
    return normalize_master_peak_list([row])[0]


def merge_master_peak_lists(
    existing: Iterable[dict],
    incoming: Iterable[dict],
    *,
    duplicate_tolerance_deg: float = 0.04,
    preserve_manual: bool = True,
) -> list[dict]:
    """Merge detected/imported peaks while preserving manual identities."""
    current = normalize_master_peak_list(existing)
    new_rows = normalize_master_peak_list(incoming)
    merged = deepcopy(current)
    for new in new_rows:
        nearest_index = None
        nearest_distance = math.inf
        for index, old in enumerate(merged):
            distance = abs(old["two_theta_deg"] - new["two_theta_deg"])
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index
        if nearest_index is not None and nearest_distance <= max(0.0, duplicate_tolerance_deg):
            old = merged[nearest_index]
            if preserve_manual and (old.get("manual") or old.get("row_locked")):
                old["intensity"] = new.get("intensity", old["intensity"])
                old["prominence"] = new.get("prominence", old["prominence"])
                old["signal_to_noise"] = new.get("signal_to_noise", old["signal_to_noise"])
                old["quality_score"] = new.get("quality_score", old["quality_score"])
                old["fwhm_deg"] = new.get("fwhm_deg", old["fwhm_deg"])
                old["modified_utc"] = datetime.now(timezone.utc).isoformat()
                if "smart-confirmed" not in old.get("note", "").lower():
                    old["note"] = (old.get("note", "") + "; smart-confirmed").strip("; ")
            else:
                new["peak_uuid"] = old["peak_uuid"]
                new["created_utc"] = old.get("created_utc", new.get("created_utc"))
                new["modified_utc"] = datetime.now(timezone.utc).isoformat()
                merged[nearest_index] = new
        else:
            merged.append(new)
    return normalize_master_peak_list(merged)


def refresh_master_peak_measurements(
    peaks: Iterable[dict],
    x: np.ndarray,
    residual: np.ndarray,
    *,
    known_reflections: Iterable[float] | None = None,
    overlap_exclusion_deg: float = 0.08,
) -> list[dict]:
    """Refresh measured fields while preserving peak UUIDs and manual notes."""
    refreshed = []
    for old in normalize_master_peak_list(peaks):
        try:
            measured = manual_peak_from_position(
                x,
                residual,
                old["two_theta_deg"],
                known_reflections=known_reflections,
                overlap_exclusion_deg=overlap_exclusion_deg,
                default_fwhm_deg=max(old.get("fwhm_deg", 0.12), 0.02),
            )
        except PhaseRevolutionError:
            refreshed.append(old)
            continue
        for key in (
            "two_theta_deg", "position_uncertainty_deg", "intensity",
            "prominence", "signal_to_noise", "quality_score", "fwhm_deg",
            "noise_sigma", "nearest_known_distance_deg", "overlap_flag",
        ):
            old[key] = measured[key]
        old["modified_utc"] = datetime.now(timezone.utc).isoformat()
        refreshed.append(old)
    return normalize_master_peak_list(refreshed)

def two_theta_to_d(two_theta_deg: float, wavelength_angstrom: float) -> float:
    theta = math.radians(float(two_theta_deg) / 2.0)
    sine = math.sin(theta)
    if wavelength_angstrom <= 0 or sine <= 0:
        raise PhaseRevolutionError("Invalid wavelength or diffraction angle.")
    return float(wavelength_angstrom / (2.0 * sine))


def density_from_formula_mass(
    volume_angstrom3: float,
    formula_mass_g_mol: float,
    z_value: float,
) -> float:
    if volume_angstrom3 <= 0 or formula_mass_g_mol <= 0 or z_value <= 0:
        raise PhaseRevolutionError("Volume, formula mass and Z must be positive.")
    return float(1.66053906660 * formula_mass_g_mol * z_value / volume_angstrom3)


def enrich_candidate_chemistry(
    candidate: dict,
    *,
    formula_mass_g_mol: float | None = None,
    z_values: Iterable[int] = (),
    density_min_g_cm3: float | None = None,
    density_max_g_cm3: float | None = None,
) -> dict:
    enriched = deepcopy(candidate)
    checks = []
    if formula_mass_g_mol and formula_mass_g_mol > 0:
        for z in z_values:
            if int(z) <= 0:
                continue
            density = density_from_formula_mass(
                float(enriched["volume_angstrom3"]),
                float(formula_mass_g_mol),
                int(z),
            )
            plausible = True
            if density_min_g_cm3 is not None:
                plausible &= density >= float(density_min_g_cm3)
            if density_max_g_cm3 is not None:
                plausible &= density <= float(density_max_g_cm3)
            checks.append(
                {"z": int(z), "density_g_cm3": density, "plausible": bool(plausible)}
            )
    enriched["density_checks"] = checks
    enriched["chemistry_status"] = (
        "Plausible" if any(row["plausible"] for row in checks)
        else "Review" if checks
        else "Not assessed"
    )
    return enriched


def rank_indexing_candidates(candidates: list[dict]) -> list[dict]:
    rows = []
    for candidate in candidates:
        row = deepcopy(candidate)
        m20 = max(0.0, float(row.get("m20", 0.0)))
        x20 = max(0, int(row.get("x20", 0)))
        indexed = max(0, int(row.get("indexed_peak_count", 0)))
        observed = max(1, int(row.get("observed_peak_count", indexed or 1)))
        coverage = indexed / observed
        chemistry_bonus = 0.15 if row.get("chemistry_status") == "Plausible" else 0.0
        row["ranking_score"] = float(
            math.log1p(m20) + 1.5 * coverage - 0.08 * x20 + chemistry_bonus
        )
        if m20 >= 10 and x20 <= 1 and coverage >= 0.8:
            row["status"] = "Promising"
        elif m20 >= 5 and coverage >= 0.6:
            row["status"] = "Ambiguous"
        else:
            row["status"] = "Review"
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row.get("ranking_score", 0.0),
            row.get("m20", 0.0),
            -row.get("x20", 0),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def build_indexing_request(
    work_directory: str | Path,
    peaks: Iterable[dict],
    **settings,
) -> tuple[dict, Path]:
    """Materialize a reproducible native-indexing request.

    The native engine executes in-process; no external interpreter, executable,
    environment variable or native-library path is required.
    """
    directory = Path(work_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    request = {
        "engine": "Afruz native powder indexer",
        "peaks": normalize_master_peak_list(peaks),
        "settings": deepcopy(settings),
    }
    request_path = directory / "native_indexing_request.json"
    request_path.write_text(
        json.dumps(request, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return request, request_path



def _centering_allowed(bravais_name: str, h: int, k: int, l: int) -> bool:
    if bravais_name.endswith("-P") or bravais_name in {"Triclinic", "Trigonal/Hexagonal-P"}:
        return True
    if bravais_name.endswith("-I"):
        return (h + k + l) % 2 == 0
    if bravais_name.endswith("-F"):
        parity = (h % 2, k % 2, l % 2)
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


def candidate_cell(candidate: dict) -> dict:
    return {
        "a": float(candidate["a_angstrom"]),
        "b": float(candidate["b_angstrom"]),
        "c": float(candidate["c_angstrom"]),
        "alpha": float(candidate["alpha_deg"]),
        "beta": float(candidate["beta_deg"]),
        "gamma": float(candidate["gamma_deg"]),
    }


def generate_candidate_reflections(
    candidate: dict,
    *,
    wavelength_angstrom: float,
    two_theta_min: float,
    two_theta_max: float,
    maximum_index: int = 18,
    merge_tolerance_deg: float = 0.002,
) -> list[dict]:
    cell = candidate_cell(candidate)
    bravais = str(candidate["bravais_name"])
    rows = []
    for h in range(0, maximum_index + 1):
        for k in range(0, maximum_index + 1):
            for l in range(0, maximum_index + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                if not _centering_allowed(bravais, h, k, l):
                    continue
                try:
                    spacing = float(d_spacing(cell, (h, k, l)))
                except (ValueError, np.linalg.LinAlgError):
                    continue
                argument = wavelength_angstrom / (2.0 * spacing)
                if not 0.0 < argument < 1.0:
                    continue
                position = float(2.0 * math.degrees(math.asin(argument)))
                if not two_theta_min <= position <= two_theta_max:
                    continue
                rows.append(
                    {
                        "two_theta": position,
                        "d_spacing_angstrom": spacing,
                        "intensity": 100.0,
                        "hkl_label": f"({h} {k} {l})",
                        "h": h,
                        "k": k,
                        "l": l,
                    }
                )
    rows.sort(key=lambda row: row["two_theta"])
    merged = []
    for row in rows:
        if merged and abs(row["two_theta"] - merged[-1]["two_theta"]) <= merge_tolerance_deg:
            merged[-1]["hkl_label"] += ", " + row["hkl_label"]
            continue
        merged.append(row)
    return merged


def candidate_reference_dataset(
    candidate: dict,
    *,
    wavelength_angstrom: float,
    two_theta_min: float,
    two_theta_max: float,
    name: str | None = None,
    observed_master_peaks: Iterable[dict] | None = None,
    master_peak_metadata: dict | None = None,
) -> Dataset:
    reflections = generate_candidate_reflections(
        candidate,
        wavelength_angstrom=wavelength_angstrom,
        two_theta_min=two_theta_min,
        two_theta_max=two_theta_max,
    )
    if len(reflections) < 3:
        raise PhaseRevolutionError(
            "The candidate cell generated fewer than three reflections in the selected range."
        )
    x = np.asarray([row["two_theta"] for row in reflections], dtype=float)
    y = np.asarray([row["intensity"] for row in reflections], dtype=float)
    cell = candidate_cell(candidate)
    label = name or (
        f"Indexed unknown — {candidate['bravais_name']} "
        f"M20={float(candidate.get('m20', 0.0)):.3g}"
    )
    return Dataset(
        name=label,
        x=x,
        y_raw=y,
        metadata={
            "analysis_role": "reference_pattern",
            "plot_style": "sticks",
            "source_format": "Phase 14 candidate cell",
            "wavelength_k_alpha1": float(wavelength_angstrom),
            "reflections": reflections,
            "cell": cell,
            "crystal_system": CRYSTAL_SYSTEM_BY_BRAVAIS.get(
                str(candidate["bravais_name"]), "Triclinic"
            ),
            "space_group": "Unresolved",
            "phase_revolution_status": "Indexed unknown phase — structure unsolved",
            "candidate_cell": deepcopy(candidate),
            "observed_master_peaks": normalize_master_peak_list(
                observed_master_peaks or []
            ),
            "master_peak_metadata": deepcopy(master_peak_metadata or {}),
            "master_peak_checksum": (master_peak_metadata or {}).get("checksum"),
            "master_peak_revision": (master_peak_metadata or {}).get("revision"),
        },
    )


def export_unknown_peak_txt(path: str | Path, peaks: list[dict], *, metadata: dict | None = None) -> Path:
    fieldnames = [
        "peak_id", "peak_uuid", "use", "origin", "manual", "row_locked",
        "two_theta_deg", "position_uncertainty_deg", "intensity", "prominence",
        "signal_to_noise", "quality_score", "fwhm_deg",
        "nearest_known_distance_deg", "overlap_flag", "note",
    ]
    result = write_table_txt(
        path,
        peaks,
        columns=fieldnames,
        title="Phase Revolution master reflection list",
        metadata=metadata or {},
    )
    return Path(result["txt_path"])


def export_unknown_peak_csv(path: str | Path, peaks: list[dict]) -> Path:
    import csv

    path = Path(path)
    fieldnames = [
        "peak_id",
        "peak_uuid",
        "use",
        "origin",
        "manual",
        "row_locked",
        "two_theta_deg",
        "position_uncertainty_deg",
        "intensity",
        "prominence",
        "signal_to_noise",
        "quality_score",
        "fwhm_deg",
        "nearest_known_distance_deg",
        "overlap_flag",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in peaks:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path
