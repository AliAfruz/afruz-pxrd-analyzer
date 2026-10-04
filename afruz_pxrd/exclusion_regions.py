from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ExclusionRegion:
    """Closed 2theta interval deliberately omitted from an investigation fit."""

    start_deg: float
    end_deg: float
    label: str = "Excluded region"

    def normalized(self) -> "ExclusionRegion":
        start = float(min(self.start_deg, self.end_deg))
        end = float(max(self.start_deg, self.end_deg))
        return ExclusionRegion(start, end, str(self.label or "Excluded region"))

    def to_dict(self) -> dict:
        row = self.normalized()
        return {
            "start_deg": row.start_deg,
            "end_deg": row.end_deg,
            "width_deg": row.end_deg - row.start_deg,
            "label": row.label,
        }


def coerce_regions(
    regions: Iterable[ExclusionRegion | Mapping | Sequence[float]],
) -> list[ExclusionRegion]:
    result: list[ExclusionRegion] = []
    for raw in regions or ():
        if isinstance(raw, ExclusionRegion):
            region = raw
        elif isinstance(raw, Mapping):
            region = ExclusionRegion(
                float(raw.get("start_deg", raw.get("start", 0.0))),
                float(raw.get("end_deg", raw.get("end", 0.0))),
                str(raw.get("label") or "Excluded region"),
            )
        else:
            values = list(raw)
            if len(values) < 2:
                raise ValueError("An exclusion interval requires start and end 2theta values.")
            region = ExclusionRegion(
                float(values[0]),
                float(values[1]),
                str(values[2]) if len(values) > 2 else "Excluded region",
            )
        normalized = region.normalized()
        if not np.isfinite(normalized.start_deg) or not np.isfinite(normalized.end_deg):
            raise ValueError("Exclusion interval limits must be finite.")
        if normalized.end_deg <= normalized.start_deg:
            raise ValueError("An exclusion interval must have a positive width.")
        result.append(normalized)
    return result


def merge_regions(
    regions: Iterable[ExclusionRegion | Mapping | Sequence[float]],
    *,
    tolerance_deg: float = 1e-9,
) -> list[ExclusionRegion]:
    """Sort and merge overlapping/touching intervals into an auditable set."""

    ordered = sorted(coerce_regions(regions), key=lambda row: (row.start_deg, row.end_deg))
    merged: list[ExclusionRegion] = []
    for row in ordered:
        if not merged or row.start_deg > merged[-1].end_deg + float(tolerance_deg):
            merged.append(row)
            continue
        previous = merged[-1]
        labels = [value for value in (previous.label, row.label) if value]
        label = labels[0] if len(set(labels)) == 1 else " / ".join(dict.fromkeys(labels))
        merged[-1] = ExclusionRegion(
            previous.start_deg,
            max(previous.end_deg, row.end_deg),
            label,
        )
    return merged


def inclusion_mask(
    x: Iterable[float] | np.ndarray,
    regions: Iterable[ExclusionRegion | Mapping | Sequence[float]],
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("Diffraction positions must be one-dimensional.")
    mask = np.ones(len(values), dtype=bool)
    for row in merge_regions(regions):
        mask &= ~((values >= row.start_deg) & (values <= row.end_deg))
    return mask


def filter_arrays(
    x: Iterable[float] | np.ndarray,
    arrays: Iterable[Iterable[float] | np.ndarray | None],
    regions: Iterable[ExclusionRegion | Mapping | Sequence[float]],
    *,
    minimum_points: int = 50,
) -> tuple[np.ndarray, list[np.ndarray | None], dict]:
    positions = np.asarray(x, dtype=float)
    merged = merge_regions(regions)
    mask = inclusion_mask(positions, merged)
    included = int(np.count_nonzero(mask))
    if included < int(minimum_points):
        raise ValueError(
            f"Only {included} points remain after exclusions; at least {minimum_points} are required."
        )
    filtered: list[np.ndarray | None] = []
    for values in arrays:
        if values is None:
            filtered.append(None)
            continue
        array = np.asarray(values, dtype=float)
        if array.shape != positions.shape:
            raise ValueError("Every filtered data column must match the diffraction-position array.")
        filtered.append(array[mask].copy())
    audit = {
        "excluded_regions": [row.to_dict() for row in merged],
        "original_point_count": int(len(positions)),
        "included_point_count": included,
        "excluded_point_count": int(len(positions) - included),
        "included_fraction": float(included / len(positions)) if len(positions) else 0.0,
    }
    return positions[mask].copy(), filtered, audit


def attach_exclusion_audit(result: Mapping, audit: Mapping, *, method: str) -> dict:
    updated = dict(result)
    record = {
        **dict(audit),
        "method": str(method),
        "objective_scope": (
            "Only included measured points contribute to parameter estimation and fit statistics. "
            "Calculated reflections inside excluded intervals are not deleted from the structural model."
        ),
    }
    updated["exclusion_investigation"] = record
    updated["excluded_regions"] = list(record.get("excluded_regions", []))
    updated["original_data_point_count"] = int(record.get("original_point_count", 0))
    updated["included_data_point_count"] = int(record.get("included_point_count", 0))
    updated["excluded_data_point_count"] = int(record.get("excluded_point_count", 0))
    warnings = list(updated.get("warnings") or [])
    warnings.append(
        f"Investigation fit excluded {updated['excluded_data_point_count']} measured point(s) "
        f"in {len(updated['excluded_regions'])} merged interval(s). Report every interval when comparing fits."
    )
    warnings.append(
        "Improvement after excluding a peak or region supports a phase-impurity or model-deficiency hypothesis; "
        "it does not by itself identify, quantify, or prove an impurity phase."
    )
    updated["warnings"] = list(dict.fromkeys(warnings))
    return updated
