from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Iterable
import zipfile

import numpy as np

from .metrology import EvidenceClass
from .text_export import write_mapping_txt, write_table_txt, write_manifest_txt
from .version import APP_RELEASE, APP_VERSION
from scipy import stats


VALIDATION_VERSION = 2
REPLICA_TYPES = (
    "Synthetic benchmark",
    "Independent preparation",
    "Repacked specimen",
    "Repeat scan",
    "Single measurement",
)
VALIDATION_STATUSES = (
    "Pass",
    "Pass with warnings",
    "Review required",
    "Invalid for QPA",
)


class ValidationCampaignError(ValueError):
    pass


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _safe_mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return None if not len(array) else float(np.mean(array))


def _sample_sd(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return None if len(array) < 2 else float(np.std(array, ddof=1))


def _confidence_interval(values: Iterable[float], confidence: float = 0.95) -> dict:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {"mean": None, "lower": None, "upper": None, "n": 0}
    mean = float(np.mean(array))
    if len(array) < 2:
        return {"mean": mean, "lower": None, "upper": None, "n": 1}
    sd = float(np.std(array, ddof=1))
    sem = sd / math.sqrt(len(array))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(array) - 1))
    half = critical * sem
    return {
        "mean": mean,
        "lower": mean - half,
        "upper": mean + half,
        "n": len(array),
    }


def normalize_record(record: dict, index: int = 0) -> dict:
    result = deepcopy(record)
    result.setdefault("record_id", f"record-{index + 1:04d}")
    result.setdefault("sample_id", result.get("dataset_name") or f"Sample {index + 1}")
    result.setdefault("dataset_uid", "")
    result.setdefault("dataset_name", result.get("sample_id", ""))
    result.setdefault("phase_name", "")
    result.setdefault("engine", "")
    result.setdefault("classification", "")
    result.setdefault("replica_type", "Single measurement")
    result.setdefault("preparation_id", result.get("sample_id", ""))
    result.setdefault("specimen_id", result.get("preparation_id", ""))
    result.setdefault("scan_id", "1")
    result.setdefault("run_id", "1")
    result.setdefault("notes", "")
    result["known_wt_percent"] = _finite(result.get("known_wt_percent"))
    result["measured_wt_percent"] = _finite(result.get("measured_wt_percent"))
    result["measured_error_percent"] = _finite(result.get("measured_error_percent"))
    return result


def _validated_records(records: list[dict]) -> tuple[list[dict], list[str]]:
    valid: list[dict] = []
    errors: list[str] = []
    for index, raw in enumerate(records):
        row = normalize_record(raw, index)
        label = f"{row['sample_id']} / {row['phase_name'] or 'unnamed phase'}"
        known = row["known_wt_percent"]
        measured = row["measured_wt_percent"]
        if not row["phase_name"]:
            errors.append(f"{label}: phase name is missing.")
            continue
        if known is None:
            errors.append(f"{label}: known wt% is missing.")
            continue
        if measured is None:
            errors.append(f"{label}: measured wt% is missing.")
            continue
        if not 0.0 <= known <= 100.0:
            errors.append(f"{label}: known wt% is outside 0–100.")
            continue
        if not -0.001 <= measured <= 105.0:
            errors.append(f"{label}: measured wt% is outside the accepted audit range.")
            continue
        row["absolute_error_wt_percent"] = measured - known
        row["absolute_error_magnitude_wt_percent"] = abs(measured - known)
        row["relative_error_percent"] = (
            None if abs(known) < 1e-12
            else 100.0 * (measured - known) / known
        )
        row["recovery_percent"] = (
            None if abs(known) < 1e-12
            else 100.0 * measured / known
        )
        valid.append(row)
    return valid, errors


def _metric_row(name: str, rows: list[dict]) -> dict:
    errors = np.asarray([row["absolute_error_wt_percent"] for row in rows], dtype=float)
    absolute = np.abs(errors)
    relative = np.asarray(
        [
            row["relative_error_percent"]
            for row in rows
            if row["relative_error_percent"] is not None
        ],
        dtype=float,
    )
    recoveries = np.asarray(
        [row["recovery_percent"] for row in rows if row["recovery_percent"] is not None],
        dtype=float,
    )
    ci = _confidence_interval(errors)
    return {
        "group": name,
        "record_count": len(rows),
        "known_mean_wt_percent": _safe_mean(row["known_wt_percent"] for row in rows),
        "measured_mean_wt_percent": _safe_mean(row["measured_wt_percent"] for row in rows),
        "bias_wt_percent": float(np.mean(errors)),
        "bias_ci95_lower_wt_percent": ci["lower"],
        "bias_ci95_upper_wt_percent": ci["upper"],
        "mae_wt_percent": float(np.mean(absolute)),
        "rmse_wt_percent": float(np.sqrt(np.mean(errors * errors))),
        "maximum_absolute_error_wt_percent": float(np.max(absolute)),
        "error_sd_wt_percent": _sample_sd(errors),
        "mean_relative_error_percent": None if not len(relative) else float(np.mean(relative)),
        "mean_absolute_relative_error_percent": None if not len(relative) else float(np.mean(np.abs(relative))),
        "mean_recovery_percent": None if not len(recoveries) else float(np.mean(recoveries)),
        "recovery_sd_percent": _sample_sd(recoveries),
    }


def _group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in keys)
        groups.setdefault(key, []).append(row)
    return groups


def _replica_statistics(rows: list[dict]) -> list[dict]:
    output: list[dict] = []

    # Repeat scans: variation within a specimen/preparation for the same phase.
    scan_groups = _group_rows(
        rows,
        ("phase_name", "preparation_id", "specimen_id"),
    )
    for key, group in scan_groups.items():
        values = [row["measured_wt_percent"] for row in group]
        if len(values) < 2:
            continue
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        output.append({
            "level": "Repeat scan",
            "phase_name": key[0],
            "group_id": " / ".join(key[1:]),
            "replicate_count": len(values),
            "mean_wt_percent": mean,
            "sd_wt_percent": sd,
            "rsd_percent": None if abs(mean) < 1e-12 else 100.0 * sd / abs(mean),
        })

    # Repacked specimens: variation between specimen means within preparation.
    specimen_means: list[dict] = []
    for key, group in scan_groups.items():
        specimen_means.append({
            "phase_name": key[0],
            "preparation_id": key[1],
            "specimen_id": key[2],
            "value": float(np.mean([row["measured_wt_percent"] for row in group])),
        })
    repack_groups = _group_rows(specimen_means, ("phase_name", "preparation_id"))
    for key, group in repack_groups.items():
        values = [row["value"] for row in group]
        if len(values) < 2:
            continue
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        output.append({
            "level": "Repacked specimen",
            "phase_name": key[0],
            "group_id": key[1],
            "replicate_count": len(values),
            "mean_wt_percent": mean,
            "sd_wt_percent": sd,
            "rsd_percent": None if abs(mean) < 1e-12 else 100.0 * sd / abs(mean),
        })

    # Independent preparations: variation between preparation means.
    preparation_means: list[dict] = []
    preparation_groups = _group_rows(specimen_means, ("phase_name", "preparation_id"))
    for key, group in preparation_groups.items():
        preparation_means.append({
            "phase_name": key[0],
            "preparation_id": key[1],
            "value": float(np.mean([row["value"] for row in group])),
        })
    phase_groups = _group_rows(preparation_means, ("phase_name",))
    for key, group in phase_groups.items():
        values = [row["value"] for row in group]
        if len(values) < 2:
            continue
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        output.append({
            "level": "Independent preparation",
            "phase_name": key[0],
            "group_id": "all preparations",
            "replicate_count": len(values),
            "mean_wt_percent": mean,
            "sd_wt_percent": sd,
            "rsd_percent": None if abs(mean) < 1e-12 else 100.0 * sd / abs(mean),
        })
    return output


def _empirical_detection_limits(rows: list[dict]) -> list[dict]:
    output = []
    by_phase = _group_rows(rows, ("phase_name",))
    for key, group in by_phase.items():
        blanks = [
            row["measured_wt_percent"]
            for row in group
            if abs(row["known_wt_percent"]) < 1e-12
        ]
        if len(blanks) < 3:
            continue
        blank_mean = float(np.mean(blanks))
        blank_sd = float(np.std(blanks, ddof=1))
        output.append({
            "phase_name": key[0],
            "blank_count": len(blanks),
            "blank_mean_wt_percent": blank_mean,
            "blank_sd_wt_percent": blank_sd,
            "empirical_lod_wt_percent": max(0.0, blank_mean + 3.3 * blank_sd),
            "empirical_loq_wt_percent": max(0.0, blank_mean + 10.0 * blank_sd),
            "method": "Blank mean + 3.3σ / 10σ; requires laboratory validation",
        })
    return output


def _campaign_status(overall: dict, thresholds: dict, record_errors: list[str]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if record_errors:
        return "Invalid for QPA", list(record_errors)
    count = int(overall.get("record_count", 0))
    if count < int(thresholds["minimum_records"]):
        warnings.append(
            f"Only {count} complete records are available; at least "
            f"{thresholds['minimum_records']} are required by this campaign rule."
        )
    if overall["mae_wt_percent"] > float(thresholds["maximum_mae_wt_percent"]):
        warnings.append("Campaign MAE exceeds the configured threshold.")
    if overall["rmse_wt_percent"] > float(thresholds["maximum_rmse_wt_percent"]):
        warnings.append("Campaign RMSE exceeds the configured threshold.")
    if abs(overall["bias_wt_percent"]) > float(thresholds["maximum_absolute_bias_wt_percent"]):
        warnings.append("Absolute campaign bias exceeds the configured threshold.")
    if overall["maximum_absolute_error_wt_percent"] > float(thresholds["maximum_single_error_wt_percent"]):
        warnings.append("At least one phase result exceeds the single-result error threshold.")

    severe = any(
        phrase in warning
        for warning in warnings
        for phrase in ("RMSE", "single-result", "Only")
    )
    if not warnings:
        return "Pass", warnings
    return ("Review required" if severe else "Pass with warnings"), warnings


def analyze_validation_campaign(
    records: list[dict],
    *,
    campaign_name: str = "Afruz QPA validation campaign",
    thresholds: dict | None = None,
    audit_results: list[dict] | None = None,
    robustness_results: list[dict] | None = None,
    evidence_classification: str = EvidenceClass.UNCLASSIFIED.value,
    metrology_assessment_ids: list[str] | tuple[str, ...] | None = None,
) -> dict:
    default_thresholds = {
        "minimum_records": 6,
        "maximum_mae_wt_percent": 3.0,
        "maximum_rmse_wt_percent": 4.0,
        "maximum_absolute_bias_wt_percent": 2.0,
        "maximum_single_error_wt_percent": 8.0,
        "maximum_robustness_spread_wt_percent": 2.0,
        "note": "Suggested starting points requiring optimization for the intended method and concentration range.",
    }
    if thresholds:
        default_thresholds.update(deepcopy(thresholds))

    valid, errors = _validated_records(records)
    if not valid:
        raise ValidationCampaignError(
            "No complete validation records are available. "
            + (" ".join(errors[:4]) if errors else "")
        )

    overall = _metric_row("All phases", valid)
    phase_metrics = [
        _metric_row(key[0], group)
        for key, group in sorted(_group_rows(valid, ("phase_name",)).items())
    ]
    sample_metrics = [
        _metric_row(key[0], group)
        for key, group in sorted(_group_rows(valid, ("sample_id",)).items())
    ]
    differences = np.asarray([row["absolute_error_wt_percent"] for row in valid], dtype=float)
    difference_sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else None
    bland_altman = {
        "mean_difference_wt_percent": float(np.mean(differences)),
        "difference_sd_wt_percent": difference_sd,
        "lower_limit_wt_percent": None if difference_sd is None else float(np.mean(differences) - 1.96 * difference_sd),
        "upper_limit_wt_percent": None if difference_sd is None else float(np.mean(differences) + 1.96 * difference_sd),
        "record_count": len(valid),
    }
    replica_metrics = _replica_statistics(valid)
    detection_limits = _empirical_detection_limits(valid)
    status, warnings = _campaign_status(overall, default_thresholds, errors)

    audits = deepcopy(audit_results or [])
    robustness = deepcopy(robustness_results or [])
    if any(row.get("status") == "Invalid for QPA" for row in audits):
        status = "Invalid for QPA"
        warnings.append("At least one included refinement audit is invalid for QPA.")
    elif any(row.get("status") == "Review required" for row in audits) and status == "Pass":
        status = "Review required"
        warnings.append("At least one included refinement audit requires review.")
    if any(row.get("status") == "Unstable" for row in robustness):
        if status == "Pass":
            status = "Review required"
        warnings.append("At least one phase is unstable across robustness runs.")

    synthetic_only = all(
        row.get("replica_type") == "Synthetic benchmark"
        or "demonstration" in str(row.get("classification", "")).casefold()
        for row in valid
    )
    try:
        evidence_class = (
            EvidenceClass.SOFTWARE_REGRESSION
            if synthetic_only
            else EvidenceClass(str(evidence_classification))
        )
    except ValueError as exc:
        raise ValidationCampaignError(
            f"Unknown evidence classification: {evidence_classification}"
        ) from exc
    assessment_ids = tuple(
        dict.fromkeys(
            str(row).strip()
            for row in (metrology_assessment_ids or [])
            if str(row).strip()
        )
    )
    publication_evidence = evidence_class in {
        EvidenceClass.CERTIFIED_REFERENCE,
        EvidenceClass.EXPERIMENTAL_VALIDATION,
    }
    claim_eligible = bool(status == "Pass" and publication_evidence and assessment_ids)
    if status == "Pass" and not claim_eligible:
        warnings.append(
            "Campaign statistics passed, but no validated publication claim is authorized: "
            "certified/experimental evidence and a passing metrology assessment are required."
        )

    return {
        "validation_version": VALIDATION_VERSION,
        "created_utc": utc_now_text(),
        "campaign_name": str(campaign_name),
        "classification": evidence_class.value,
        "evidence_classification": evidence_class.value,
        "metrology_assessment_ids": list(assessment_ids),
        "claim_eligible": claim_eligible,
        "scientific_claim_status": (
            "Eligible for explicit claim authorization"
            if claim_eligible
            else "Not validated for a publication claim"
        ),
        "status": status,
        "thresholds": default_thresholds,
        "records": valid,
        "excluded_record_errors": errors,
        "overall_metrics": overall,
        "phase_metrics": phase_metrics,
        "sample_metrics": sample_metrics,
        "replica_metrics": replica_metrics,
        "bland_altman": bland_altman,
        "detection_limits": detection_limits,
        "refinement_audits": audits,
        "robustness_results": robustness,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _fraction_rows_from_result(result: dict) -> list[dict]:
    engine = str(result.get("engine", ""))
    rows = []
    for phase in result.get("phases", []):
        measured = phase.get("absolute_weight_percent")
        if measured is None:
            measured = phase.get("crystalline_weight_percent")
        if measured is None and phase.get("mass_fraction") is not None:
            measured = 100.0 * float(phase["mass_fraction"])
        uncertainty = phase.get("crystalline_weight_percent_error")
        if uncertainty is None and phase.get("mass_fraction_error") is not None:
            uncertainty = 100.0 * float(phase["mass_fraction_error"])
        rows.append({
            "phase_name": phase.get("phase_name") or phase.get("name"),
            "measured_wt_percent": measured,
            "measured_error_percent": uncertainty,
            "engine": engine,
            "classification": result.get("classification", ""),
        })
    return rows


def audit_qpa_result(
    result: dict,
    *,
    dataset_name: str = "Dataset",
    maximum_rwp_percent: float = 15.0,
    maximum_parameter_correlation: float = 0.95,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    phases = _fraction_rows_from_result(result)
    engine = str(result.get("engine", "Unknown engine"))
    classification = str(result.get("classification", ""))

    if len(phases) < 2:
        errors.append("Fewer than two quantified crystalline phases are present.")
    fractions = [_finite(row.get("measured_wt_percent")) for row in phases]
    if any(value is None for value in fractions):
        errors.append("At least one phase has no reportable mass fraction.")
    else:
        values = np.asarray(fractions, dtype=float)
        if np.any(values < -1e-6):
            errors.append("A negative phase mass fraction is present.")
        total = float(np.sum(values))
        if not 98.0 <= total <= 102.0 and result.get("amorphous_or_unmodelled_percent") is None:
            warnings.append(f"Reported phase fractions sum to {total:.5g}%, not approximately 100%.")

    if "native" in engine.lower() or "preview" in classification.lower():
        warnings.append("The result is a native preview rather than an independently validated production-backend result.")
    if result.get("publication_ready") is False:
        warnings.append("The source result explicitly reports publication_ready=False.")

    rwp = _finite(result.get("rwp_percent"))
    if rwp is not None and rwp > maximum_rwp_percent:
        warnings.append(f"Rwp {rwp:.5g}% exceeds the configured audit threshold.")

    max_corr = _finite(result.get("maximum_parameter_correlation"))
    if max_corr is not None and max_corr > maximum_parameter_correlation:
        warnings.append("Maximum parameter correlation exceeds the configured threshold.")
    condition = _finite(result.get("parameter_condition_number"))
    if condition is not None and condition > 1e10:
        warnings.append("The parameter condition number indicates a poorly conditioned refinement.")

    uncertainties = [row.get("measured_error_percent") for row in phases]
    if not any(_finite(value) is not None for value in uncertainties):
        warnings.append("No phase-fraction uncertainties are available.")

    source_warnings = result.get("warnings", [])
    for warning in source_warnings:
        warning_text = str(warning)
        if warning_text and warning_text not in warnings:
            warnings.append(warning_text)

    if errors:
        status = "Invalid for QPA"
    elif any(
        token in warning.lower()
        for warning in warnings
        for token in ("native preview", "poorly conditioned", "exceeds", "no phase-fraction")
    ):
        status = "Review required"
    elif warnings:
        status = "Pass with warnings"
    else:
        status = "Pass"

    return {
        "dataset_name": dataset_name,
        "engine": engine,
        "classification": classification,
        "status": status,
        "phase_count": len(phases),
        "rwp_percent": rwp,
        "maximum_parameter_correlation": max_corr,
        "parameter_condition_number": condition,
        "errors": errors,
        "warnings": warnings,
    }


def analyze_robustness_runs(
    runs: list[dict],
    *,
    maximum_spread_wt_percent: float = 2.0,
) -> list[dict]:
    values_by_phase: dict[str, list[float]] = {}
    run_ids_by_phase: dict[str, list[str]] = {}
    for index, run in enumerate(runs):
        run_id = str(run.get("run_id", f"run-{index + 1}"))
        result = run.get("result", run)
        for row in _fraction_rows_from_result(result):
            value = _finite(row.get("measured_wt_percent"))
            phase = str(row.get("phase_name") or "")
            if value is None or not phase:
                continue
            values_by_phase.setdefault(phase, []).append(value)
            run_ids_by_phase.setdefault(phase, []).append(run_id)

    output = []
    for phase, values in sorted(values_by_phase.items()):
        array = np.asarray(values, dtype=float)
        spread = float(np.ptp(array)) if len(array) else 0.0
        mean = float(np.mean(array)) if len(array) else None
        sd = float(np.std(array, ddof=1)) if len(array) > 1 else None
        output.append({
            "phase_name": phase,
            "run_count": len(array),
            "run_ids": run_ids_by_phase[phase],
            "mean_wt_percent": mean,
            "sd_wt_percent": sd,
            "minimum_wt_percent": float(np.min(array)),
            "maximum_wt_percent": float(np.max(array)),
            "spread_wt_percent": spread,
            "relative_spread_percent": None if mean is None or abs(mean) < 1e-12 else 100.0 * spread / abs(mean),
            "threshold_wt_percent": float(maximum_spread_wt_percent),
            "status": "Stable" if len(array) >= 2 and spread <= maximum_spread_wt_percent else ("Insufficient runs" if len(array) < 2 else "Unstable"),
        })
    return output


def generate_synthetic_campaign(
    *,
    seed: int = 1300,
    preparations: int = 3,
    specimens_per_preparation: int = 2,
    scans_per_specimen: int = 2,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    known = {"Phase A": 65.0, "Phase B": 35.0}
    records = []
    index = 0
    for prep in range(1, preparations + 1):
        preparation_bias = rng.normal(0.0, 0.55)
        for specimen in range(1, specimens_per_preparation + 1):
            specimen_bias = rng.normal(0.0, 0.35)
            phase_a_measurements = []
            for scan in range(1, scans_per_specimen + 1):
                scan_noise = rng.normal(0.0, 0.18)
                phase_a = known["Phase A"] + preparation_bias + specimen_bias + scan_noise
                phase_a_measurements.append(phase_a)
                values = {"Phase A": phase_a, "Phase B": 100.0 - phase_a}
                for phase_name, measured in values.items():
                    index += 1
                    records.append({
                        "record_id": f"synthetic-{index:04d}",
                        "sample_id": "Synthetic binary 65/35",
                        "dataset_name": f"P{prep}-S{specimen}-scan{scan}",
                        "phase_name": phase_name,
                        "known_wt_percent": known[phase_name],
                        "measured_wt_percent": float(measured),
                        "measured_error_percent": 0.25,
                        "engine": "Synthetic Phase 13 demonstration",
                        "classification": "Demonstration only",
                        "replica_type": "Synthetic benchmark",
                        "preparation_id": f"P{prep}",
                        "specimen_id": f"P{prep}-S{specimen}",
                        "scan_id": str(scan),
                        "run_id": "default",
                        "notes": "Software-validation example; not experimental evidence.",
                    })
    return records


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), ensure_ascii=False)
                if isinstance(row.get(key), (dict, list, tuple))
                else row.get(key)
                for key in fieldnames
            })


def _html_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "<p>No records.</p>"
    columns = columns or list(rows[0].keys())
    head = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(column, '')))}</td>"
            for column in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _render_html_report(result: dict) -> str:
    overall = result.get("overall_metrics", {})
    warnings = "".join(f"<li>{html.escape(str(value))}</li>" for value in result.get("warnings", []))
    style = """
    body{font-family:Arial,sans-serif;margin:36px;color:#222}h1,h2{color:#6b4b13}
    table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border:1px solid #bbb;padding:6px;text-align:left;font-size:12px}
    th{background:#efe4cf}.status{font-size:18px;font-weight:bold}.muted{color:#666}
    """
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(result.get('campaign_name','Validation campaign'))}</title><style>{style}</style></head><body>
    <h1>{html.escape(result.get('campaign_name','Validation campaign'))}</h1>
    <p class='status'>Status: {html.escape(str(result.get('status')))}</p>
    <p>{html.escape(str(result.get('classification','')))}</p>
    <h2>Overall accuracy</h2>
    {_html_table([overall])}
    <h2>Phase metrics</h2>{_html_table(result.get('phase_metrics', []))}
    <h2>Replica statistics</h2>{_html_table(result.get('replica_metrics', []))}
    <h2>Bland–Altman summary</h2>{_html_table([result.get('bland_altman', {})])}
    <h2>Refinement audits</h2>{_html_table(result.get('refinement_audits', []), ['dataset_name','engine','status','rwp_percent','errors','warnings'])}
    <h2>Robustness</h2>{_html_table(result.get('robustness_results', []))}
    <h2>Warnings</h2><ul>{warnings or '<li>None</li>'}</ul>
    <p class='muted'>Generated {html.escape(str(result.get('created_utc')))} by {html.escape(APP_RELEASE)}.</p>
    </body></html>"""


def export_reproducibility_package(
    output_zip: str | Path,
    campaign_result: dict,
    *,
    include_files: list[str | Path] | None = None,
    software_version: str = APP_VERSION,
    additional_metadata: dict | None = None,
) -> Path:
    output_zip = Path(output_zip)
    if output_zip.suffix.lower() != ".zip":
        output_zip = output_zip.with_suffix(".zip")
    work = output_zip.with_name(output_zip.stem + "_contents")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    result_path = work / "validation_campaign.json"
    result_path.write_text(
        json.dumps(campaign_result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_mapping_txt(
        work / "validation_campaign.txt",
        campaign_result,
        title="Afruz validation campaign complete result",
        backup=False,
    )
    _write_csv(work / "validation_records.csv", campaign_result.get("records", []))
    _write_csv(work / "phase_metrics.csv", campaign_result.get("phase_metrics", []))
    _write_csv(work / "sample_metrics.csv", campaign_result.get("sample_metrics", []))
    _write_csv(work / "replica_metrics.csv", campaign_result.get("replica_metrics", []))
    _write_csv(work / "refinement_audits.csv", campaign_result.get("refinement_audits", []))
    _write_csv(work / "robustness_results.csv", campaign_result.get("robustness_results", []))
    for filename, rows, title in (
        ("validation_records.txt", campaign_result.get("records", []), "Validation records"),
        ("phase_metrics.txt", campaign_result.get("phase_metrics", []), "Validation phase metrics"),
        ("sample_metrics.txt", campaign_result.get("sample_metrics", []), "Validation sample metrics"),
        ("replica_metrics.txt", campaign_result.get("replica_metrics", []), "Validation replica metrics"),
        ("refinement_audits.txt", campaign_result.get("refinement_audits", []), "Validation refinement audits"),
        ("robustness_results.txt", campaign_result.get("robustness_results", []), "Validation robustness results"),
    ):
        write_table_txt(work / filename, rows, title=title, backup=False)
    (work / "validation_report.html").write_text(_render_html_report(campaign_result), encoding="utf-8")

    environment = {
        "software": "Afruz PXRD Analyzer",
        "software_version": software_version,
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "generated_utc": utc_now_text(),
        "additional_metadata": deepcopy(additional_metadata or {}),
    }
    (work / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_mapping_txt(work / "environment.txt", environment, title="Validation environment", backup=False)

    inputs = work / "inputs"
    copied = []
    for source_value in include_files or []:
        source = Path(source_value)
        if not source.exists() or not source.is_file():
            continue
        inputs.mkdir(exist_ok=True)
        destination = inputs / source.name
        counter = 2
        while destination.exists():
            destination = inputs / f"{source.stem}_{counter}{source.suffix}"
            counter += 1
        shutil.copy2(source, destination)
        copied.append(destination)

    manifest_rows = []
    for path in sorted(work.rglob("*")):
        if not path.is_file() or path.name == "manifest_sha256.csv":
            continue
        manifest_rows.append({
            "relative_path": str(path.relative_to(work)).replace(os.sep, "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    _write_csv(work / "manifest_sha256.csv", manifest_rows)
    write_table_txt(
        work / "manifest_sha256.txt",
        manifest_rows,
        columns=["relative_path", "size_bytes", "sha256"],
        title="Validation reproducibility package manifest",
        backup=False,
    )

    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(work.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(work))
    shutil.rmtree(work)
    return output_zip
