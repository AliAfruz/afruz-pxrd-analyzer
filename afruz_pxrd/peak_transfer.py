from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Iterable

from .processing import active_peak_rows, normalize_peak_rows


def _scientific_peak_payload(rows: Iterable[dict]) -> list[dict]:
    normalized = normalize_peak_rows(list(rows))
    payload = []
    for row in active_peak_rows(normalized):
        payload.append(
            {
                "peak_uuid": str(row.get("peak_uuid", "")),
                "position": float(row["position"]),
                "intensity": float(row.get("intensity", 0.0)),
                "prominence": float(row.get("prominence", 0.0)),
                "fwhm": (
                    None
                    if row.get("fwhm") is None
                    else float(row.get("fwhm"))
                ),
                "position_error": (
                    None
                    if row.get("position_error") is None
                    else float(row.get("position_error"))
                ),
                "signal_to_noise": (
                    None
                    if row.get("signal_to_noise") is None
                    else float(row.get("signal_to_noise"))
                ),
                "confidence": (
                    None
                    if row.get("confidence") is None
                    else float(row.get("confidence"))
                ),
                "manual": bool(row.get("manual", False)),
                "protected": bool(
                    row.get("protected", False) or row.get("manual", False)
                ),
                "method": str(row.get("method", "")),
                "origin": str(row.get("origin", "")),
                "notes": str(row.get("notes", "")),
            }
        )
    payload.sort(key=lambda row: (row["position"], row["peak_uuid"]))
    return payload


def peak_payload_checksum(rows: Iterable[dict]) -> str:
    payload = _scientific_peak_payload(rows)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_peak_list_guide(dataset, rows: Iterable[dict], meta: dict | None) -> dict:
    peaks = _scientific_peak_payload(rows)
    if not peaks:
        raise ValueError("The selected dataset has no included curated peaks.")
    positions = [float(row["position"]) for row in peaks]
    metadata = dict(meta or {})
    return {
        "dataset_uid": str(dataset.uid),
        "dataset_name": str(dataset.name),
        "revision": int(metadata.get("revision", 0)),
        "locked": bool(metadata.get("locked", False)),
        "checksum": peak_payload_checksum(peaks),
        "peak_count": len(peaks),
        "manual_peak_count": sum(bool(row.get("manual")) for row in peaks),
        "minimum_two_theta_deg": min(positions),
        "maximum_two_theta_deg": max(positions),
        "peaks": deepcopy(peaks),
        "scientific_role": (
            "Refinement guide and provenance snapshot. Whole-pattern methods "
            "continue to use every measured point inside the selected 2θ range."
        ),
    }


def compare_peak_list_guide(
    guide: dict | None,
    dataset,
    rows: Iterable[dict],
    meta: dict | None,
) -> tuple[bool, str]:
    if not isinstance(guide, dict):
        return False, "No curated Peak List snapshot has been imported."
    if str(guide.get("dataset_uid", "")) != str(dataset.uid):
        return False, "The imported Peak List belongs to a different dataset."
    current_rows = active_peak_rows(normalize_peak_rows(list(rows)))
    if not current_rows:
        return False, "The current Peak List has no included peaks."
    current_revision = int(dict(meta or {}).get("revision", 0))
    current_checksum = peak_payload_checksum(current_rows)
    if int(guide.get("revision", -1)) != current_revision:
        return False, (
            f"Peak List revision changed from {guide.get('revision', '—')} "
            f"to {current_revision}. Import it again before refinement."
        )
    if str(guide.get("checksum", "")) != current_checksum:
        return False, "The Peak List contents changed. Import it again before refinement."
    return True, (
        f"Current Peak List revision {current_revision}; "
        f"{len(current_rows)} included peaks."
    )
