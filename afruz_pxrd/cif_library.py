from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import sqlite3
import time
from typing import Iterable

import numpy as np

from .crystallography import (
    CIFImportError,
    calculate_powder_pattern,
    direct_metric_tensor,
    load_cif,
)
from .structure_validation import parse_formula_counts, validate_crystal_structure


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CifLibrarySettings:
    wavelength_angstrom: float = 1.5406
    two_theta_min: float = 5.0
    two_theta_max: float = 90.0
    intensity_cutoff_percent: float = 1.0
    merge_tolerance_deg: float = 0.02
    maximum_files: int | None = None


@dataclass(frozen=True)
class CellSearchTolerance:
    length_fraction: float = 0.025
    angle_degrees: float = 1.0
    volume_fraction: float = 0.05


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_volume(cell: dict) -> float:
    return float(math.sqrt(float(np.linalg.det(direct_metric_tensor(cell)))))


def normalize_element_symbol(symbol: str) -> str:
    text = str(symbol or "").strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:2].lower()


def formula_elements(formula: str, fallback_atoms: Iterable[dict] | None = None) -> list[str]:
    counts = parse_formula_counts(formula or "")
    if not counts and fallback_atoms:
        for atom in fallback_atoms:
            element = normalize_element_symbol(atom.get("element", ""))
            if element:
                counts[element] = counts.get(element, 0.0) + 1.0
    return sorted(counts)


def q_from_two_theta(two_theta_deg: float, wavelength_angstrom: float) -> float:
    theta = math.radians(float(two_theta_deg) / 2.0)
    return float(4.0 * math.sin(theta) ** 2 / float(wavelength_angstrom) ** 2)


def normalize_formula_key(formula: str, fallback_atoms: Iterable[dict] | None = None) -> str:
    counts = parse_formula_counts(formula or "")
    if not counts and fallback_atoms:
        for atom in fallback_atoms:
            element = normalize_element_symbol(atom.get("element", ""))
            if element:
                counts[element] = counts.get(element, 0.0) + float(atom.get("occupancy", 1.0) or 1.0)
    parts = []
    for element in sorted(counts):
        value = float(counts[element])
        if abs(value - round(value)) < 1e-6:
            rendered = str(int(round(value)))
        else:
            rendered = f"{value:.3f}".rstrip("0").rstrip(".")
        parts.append(f"{element}{rendered}")
    return " ".join(parts)


def infer_material_type(elements: Iterable[str], formula: str = "") -> str:
    element_set = {normalize_element_symbol(el) for el in elements if normalize_element_symbol(el)}
    metals = {
        "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba", "Sc", "Y", "Ti", "Zr", "Hf",
        "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Al", "Ga", "In", "Sn",
        "Pb", "Bi", "Ag", "Au", "Pt", "Pd", "Cd", "Hg", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb",
        "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    }
    has_metal = bool(element_set & metals)
    has_c = "C" in element_set
    has_h = "H" in element_set
    has_o = "O" in element_set
    has_n = "N" in element_set
    if has_metal and has_c and has_o:
        return "MOF-like / metal-organic candidate"
    if has_metal and has_o and not has_c:
        return "Metal oxide / inorganic oxide"
    if has_c and has_h and not has_metal:
        return "Organic crystal"
    if has_metal and not has_c:
        return "Inorganic crystal"
    if has_c and (has_o or has_n):
        return "Molecular / covalent crystal"
    return "Unknown"


def compact_q_fingerprint(fingerprint: dict, *, minimum_intensity: float = 1.0, precision: int = 5) -> list[float]:
    values = []
    for peak in (fingerprint or {}).get("peaks", []):
        try:
            if float(peak.get("relative_intensity", 0.0)) < minimum_intensity:
                continue
            values.append(round(float(peak["q"]), precision))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(set(values))


def structure_family_key(structure: dict, fingerprint: dict | None = None) -> str:
    cell = structure.get("cell", {}) or {}
    elements = formula_elements(structure.get("formula", ""), structure.get("atoms", []))
    composition = normalize_formula_key(structure.get("formula", ""), structure.get("atoms", []))
    # Rounded cell/fingerprint key: intended for grouping likely duplicates, not proof of identity.
    lengths = [round(float(cell.get(key, 0.0)), 2) for key in ("a", "b", "c")]
    angles = [round(float(cell.get(key, 0.0)), 1) for key in ("alpha", "beta", "gamma")]
    volume = round(cell_volume(cell), 1) if cell else 0.0
    q_head = compact_q_fingerprint(fingerprint or {}, minimum_intensity=10.0, precision=4)[:12]
    payload = json.dumps(
        {
            "composition": composition,
            "elements": elements,
            "cell": lengths + angles + [volume],
            "q": q_head,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class IncrementalScanOptions:
    incremental: bool = True
    retry_failed: bool = False
    worker_count: int = 1
    checkpoint_every: int = 250
    scan_id: str | None = None


def powder_fingerprint(
    structure: dict,
    settings: CifLibrarySettings | None = None,
) -> dict:
    settings = settings or CifLibrarySettings()
    peaks = calculate_powder_pattern(
        structure,
        wavelength_angstrom=settings.wavelength_angstrom,
        two_theta_min=settings.two_theta_min,
        two_theta_max=settings.two_theta_max,
        intensity_cutoff_percent=settings.intensity_cutoff_percent,
        merge_tolerance_deg=settings.merge_tolerance_deg,
    )
    rows = []
    for peak in peaks:
        two_theta = float(peak["two_theta"])
        rows.append(
            {
                "two_theta_deg": two_theta,
                "q": q_from_two_theta(two_theta, settings.wavelength_angstrom),
                "d_spacing_angstrom": float(peak["d_spacing"]),
                "relative_intensity": float(peak["intensity"]),
                "hkl": list(peak.get("hkl", [])),
                "hkl_label": peak.get("hkl_label", ""),
            }
        )
    rows.sort(key=lambda row: row["q"])
    return {
        "wavelength_angstrom": settings.wavelength_angstrom,
        "two_theta_min": settings.two_theta_min,
        "two_theta_max": settings.two_theta_max,
        "intensity_cutoff_percent": settings.intensity_cutoff_percent,
        "peaks": rows,
        "peak_count": len(rows),
    }


def _connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def initialize_library_database(db_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cif_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                library_root TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                absolute_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                modified_ns INTEGER NOT NULL,
                parse_status TEXT NOT NULL,
                error_message TEXT NOT NULL DEFAULT '',
                source_id TEXT NOT NULL DEFAULT '',
                formula TEXT NOT NULL DEFAULT '',
                elements_json TEXT NOT NULL DEFAULT '[]',
                element_count INTEGER NOT NULL DEFAULT 0,
                space_group TEXT NOT NULL DEFAULT 'Unknown',
                crystal_system TEXT NOT NULL DEFAULT 'Unknown',
                a REAL, b REAL, c REAL,
                alpha REAL, beta REAL, gamma REAL,
                volume REAL,
                atom_count INTEGER NOT NULL DEFAULT 0,
                density_g_cm3 REAL,
                plausibility_status TEXT NOT NULL DEFAULT '',
                plausibility_score REAL,
                fingerprint_json TEXT NOT NULL DEFAULT '{}',
                family_key TEXT NOT NULL DEFAULT '',
                composition_key TEXT NOT NULL DEFAULT '',
                material_type TEXT NOT NULL DEFAULT 'Unknown',
                q_fingerprint_json TEXT NOT NULL DEFAULT '[]',
                strong_q_fingerprint_json TEXT NOT NULL DEFAULT '[]',
                last_seen_scan_id TEXT NOT NULL DEFAULT '',
                indexed_at REAL NOT NULL,
                UNIQUE(library_root, relative_path)
            )
            """
        )
        # Upgrade older Phase 18.0 library databases in-place.
        for column, definition in {
            "family_key": "TEXT NOT NULL DEFAULT ''",
            "composition_key": "TEXT NOT NULL DEFAULT ''",
            "material_type": "TEXT NOT NULL DEFAULT 'Unknown'",
            "q_fingerprint_json": "TEXT NOT NULL DEFAULT '[]'",
            "strong_q_fingerprint_json": "TEXT NOT NULL DEFAULT '[]'",
            "last_seen_scan_id": "TEXT NOT NULL DEFAULT ''",
        }.items():
            existing = {row[1] for row in connection.execute("PRAGMA table_info(cif_entries)").fetchall()}
            if column not in existing:
                connection.execute(f"ALTER TABLE cif_entries ADD COLUMN {column} {definition}")

        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cif_elements ON cif_entries(elements_json)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cif_cell ON cif_entries(crystal_system, volume)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cif_family ON cif_entries(family_key)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cif_material ON cif_entries(material_type)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cif_composition ON cif_entries(composition_key)"
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        connection.commit()


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def parse_cif_for_library(
    path: str | Path,
    library_root: str | Path,
    settings: CifLibrarySettings | None = None,
) -> dict:
    path = Path(path)
    root = Path(library_root)
    settings = settings or CifLibrarySettings()
    stat = path.stat()
    digest = sha256_file(path)
    base = {
        "library_root": str(root.resolve()),
        "relative_path": _relative_path(root, path),
        "absolute_path": str(path.resolve()),
        "sha256": digest,
        "file_size": int(stat.st_size),
        "modified_ns": int(stat.st_mtime_ns),
        "indexed_at": time.time(),
    }
    try:
        structure = load_cif(path)
        volume = cell_volume(structure["cell"])
        validation = validate_crystal_structure(structure)
        elements = formula_elements(structure.get("formula", ""), structure.get("atoms", []))
        fingerprint = powder_fingerprint(structure, settings)
        composition_key = normalize_formula_key(structure.get("formula", ""), structure.get("atoms", []))
        material_type = infer_material_type(elements, structure.get("formula", ""))
        family_key = structure_family_key(structure, fingerprint)
        q_fingerprint = compact_q_fingerprint(fingerprint, minimum_intensity=1.0)
        strong_q_fingerprint = compact_q_fingerprint(fingerprint, minimum_intensity=15.0)
        scores = validation.get("scores", {}) or {}
        return {
            **base,
            "parse_status": "ok",
            "error_message": "",
            "source_id": str(structure.get("data_name") or path.stem),
            "formula": str(structure.get("formula") or ""),
            "elements_json": json.dumps(elements, sort_keys=True),
            "element_count": len(elements),
            "space_group": str(structure.get("space_group") or "Unknown"),
            "crystal_system": str(structure.get("crystal_system") or "Unknown"),
            "a": float(structure["cell"]["a"]),
            "b": float(structure["cell"]["b"]),
            "c": float(structure["cell"]["c"]),
            "alpha": float(structure["cell"]["alpha"]),
            "beta": float(structure["cell"]["beta"]),
            "gamma": float(structure["cell"]["gamma"]),
            "volume": float(volume),
            "atom_count": int(structure.get("expanded_atom_count") or len(structure.get("atoms", []))),
            "density_g_cm3": validation.get("density_g_cm3"),
            "plausibility_status": str(validation.get("status") or ""),
            "plausibility_score": scores.get("overall"),
            "fingerprint_json": json.dumps(fingerprint, sort_keys=True),
            "family_key": family_key,
            "composition_key": composition_key,
            "material_type": material_type,
            "q_fingerprint_json": json.dumps(q_fingerprint),
            "strong_q_fingerprint_json": json.dumps(strong_q_fingerprint),
            "last_seen_scan_id": "",
        }
    except Exception as exc:
        return {
            **base,
            "parse_status": "error",
            "error_message": str(exc),
            "source_id": path.stem,
            "formula": "",
            "elements_json": "[]",
            "element_count": 0,
            "space_group": "Unknown",
            "crystal_system": "Unknown",
            "a": None,
            "b": None,
            "c": None,
            "alpha": None,
            "beta": None,
            "gamma": None,
            "volume": None,
            "atom_count": 0,
            "density_g_cm3": None,
            "plausibility_status": "Parse failed",
            "plausibility_score": 0.0,
            "fingerprint_json": "{}",
            "family_key": "",
            "composition_key": "",
            "material_type": "Unknown",
            "q_fingerprint_json": "[]",
            "strong_q_fingerprint_json": "[]",
            "last_seen_scan_id": "",
        }


_ENTRY_COLUMNS = (
    "library_root", "relative_path", "absolute_path", "sha256", "file_size",
    "modified_ns", "parse_status", "error_message", "source_id", "formula",
    "elements_json", "element_count", "space_group", "crystal_system", "a", "b",
    "c", "alpha", "beta", "gamma", "volume", "atom_count", "density_g_cm3",
    "plausibility_status", "plausibility_score", "fingerprint_json",
    "family_key", "composition_key", "material_type", "q_fingerprint_json",
    "strong_q_fingerprint_json", "last_seen_scan_id", "indexed_at",
)


def upsert_library_entry(db_path: str | Path, row: dict) -> None:
    initialize_library_database(db_path)
    placeholders = ", ".join("?" for _ in _ENTRY_COLUMNS)
    updates = ", ".join(f"{column}=excluded.{column}" for column in _ENTRY_COLUMNS if column not in {"library_root", "relative_path"})
    with _connect(db_path) as connection:
        connection.execute(
            f"""
            INSERT INTO cif_entries ({', '.join(_ENTRY_COLUMNS)})
            VALUES ({placeholders})
            ON CONFLICT(library_root, relative_path) DO UPDATE SET {updates}
            """,
            [row.get(column) for column in _ENTRY_COLUMNS],
        )
        connection.commit()


def scan_cif_files(library_root: str | Path, maximum_files: int | None = None) -> list[Path]:
    root = Path(library_root)
    files = sorted(path for path in root.rglob("*.cif") if path.is_file())
    if maximum_files is not None:
        files = files[: int(maximum_files)]
    return files


def build_cif_library(
    library_root: str | Path,
    db_path: str | Path,
    settings: CifLibrarySettings | None = None,
) -> dict:
    settings = settings or CifLibrarySettings()
    initialize_library_database(db_path)
    root = Path(library_root)
    files = scan_cif_files(root, settings.maximum_files)
    ok = 0
    failed = 0
    skipped_unchanged = 0
    for path in files:
        row = parse_cif_for_library(path, root, settings)
        upsert_library_entry(db_path, row)
        if row["parse_status"] == "ok":
            ok += 1
        else:
            failed += 1
    return {
        "library_root": str(root.resolve()),
        "db_path": str(Path(db_path).resolve()),
        "files_seen": len(files),
        "ok": ok,
        "failed": failed,
        "skipped_unchanged": skipped_unchanged,
        "settings": asdict(settings),
    }


def _row_to_dict(row: sqlite3.Row | dict) -> dict:
    result = dict(row)
    for key in ("elements_json", "fingerprint_json", "q_fingerprint_json", "strong_q_fingerprint_json"):
        if isinstance(result.get(key), str):
            try:
                parsed = json.loads(result[key])
            except json.JSONDecodeError:
                parsed = [] if key != "fingerprint_json" else {}
            if key == "elements_json":
                result["elements"] = parsed
            elif key == "fingerprint_json":
                result["fingerprint"] = parsed
            elif key == "q_fingerprint_json":
                result["q_fingerprint"] = parsed
            elif key == "strong_q_fingerprint_json":
                result["strong_q_fingerprint"] = parsed
    return result


def list_library_entries(db_path: str | Path, status: str | None = "ok") -> list[dict]:
    initialize_library_database(db_path)
    with _connect(db_path) as connection:
        if status is None:
            rows = connection.execute("SELECT * FROM cif_entries ORDER BY relative_path").fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM cif_entries WHERE parse_status=? ORDER BY relative_path",
                (status,),
            ).fetchall()
    return [_row_to_dict(row) for row in rows]


def search_by_elements(
    db_path: str | Path,
    *,
    required: Iterable[str] = (),
    excluded: Iterable[str] = (),
    limit: int = 100,
) -> list[dict]:
    required_set = {normalize_element_symbol(el) for el in required if normalize_element_symbol(el)}
    excluded_set = {normalize_element_symbol(el) for el in excluded if normalize_element_symbol(el)}
    rows = []
    for row in list_library_entries(db_path, status="ok"):
        elements = set(row.get("elements", []))
        if required_set and not required_set.issubset(elements):
            continue
        if excluded_set and excluded_set.intersection(elements):
            continue
        rows.append(row)
    return rows[: int(limit)]


def _cell_distance_score(target: dict, row: dict, tolerance: CellSearchTolerance) -> float:
    if row.get("volume") is None:
        return math.inf
    length_terms = []
    for key in ("a", "b", "c"):
        target_value = float(target[key])
        candidate_value = float(row[key])
        scale = max(abs(target_value) * tolerance.length_fraction, 1e-8)
        length_terms.append(((candidate_value - target_value) / scale) ** 2)
    angle_terms = []
    for key in ("alpha", "beta", "gamma"):
        target_value = float(target[key])
        candidate_value = float(row[key])
        scale = max(tolerance.angle_degrees, 1e-8)
        angle_terms.append(((candidate_value - target_value) / scale) ** 2)
    target_volume = cell_volume(target)
    volume_scale = max(abs(target_volume) * tolerance.volume_fraction, 1e-8)
    volume_term = ((float(row["volume"]) - target_volume) / volume_scale) ** 2
    return float(math.sqrt(np.mean(length_terms + angle_terms + [volume_term])))


def search_by_cell(
    db_path: str | Path,
    target_cell: dict,
    *,
    tolerance: CellSearchTolerance | None = None,
    required_elements: Iterable[str] = (),
    limit: int = 25,
) -> list[dict]:
    tolerance = tolerance or CellSearchTolerance()
    candidates = search_by_elements(db_path, required=required_elements, limit=100000)
    scored = []
    for row in candidates:
        score = _cell_distance_score(target_cell, row, tolerance)
        if not math.isfinite(score):
            continue
        result = dict(row)
        result["cell_match_score"] = score
        result["overall_score"] = max(0.0, 100.0 - 12.0 * score)
        scored.append(result)
    scored.sort(key=lambda row: (-row["overall_score"], row["cell_match_score"]))
    return scored[: int(limit)]


def _observed_positions(peaks: Iterable[dict]) -> list[float]:
    positions = []
    for peak in peaks:
        if peak.get("use", True) is False:
            continue
        value = peak.get("two_theta_deg", peak.get("position", peak.get("two_theta")))
        if value is None:
            continue
        try:
            positions.append(float(value))
        except (TypeError, ValueError):
            continue
    return sorted(positions)


def fingerprint_peak_match(
    observed_peaks: Iterable[dict],
    fingerprint: dict,
    *,
    tolerance_deg: float = 0.15,
    minimum_reference_intensity: float = 1.0,
) -> dict:
    observed = _observed_positions(observed_peaks)
    reference = [
        peak for peak in fingerprint.get("peaks", [])
        if float(peak.get("relative_intensity", 0.0)) >= minimum_reference_intensity
    ]
    reference.sort(key=lambda peak: float(peak["two_theta_deg"]))
    used_reference = set()
    matches = []
    for observed_index, position in enumerate(observed):
        best = None
        for reference_index, peak in enumerate(reference):
            if reference_index in used_reference:
                continue
            delta = position - float(peak["two_theta_deg"])
            if abs(delta) <= tolerance_deg:
                score = abs(delta) / max(tolerance_deg, 1e-8)
                candidate = (score, reference_index, peak, delta)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is not None:
            _, reference_index, peak, delta = best
            used_reference.add(reference_index)
            matches.append(
                {
                    "observed_index": observed_index,
                    "observed_two_theta_deg": position,
                    "reference_two_theta_deg": float(peak["two_theta_deg"]),
                    "delta_two_theta_deg": float(delta),
                    "reference_intensity": float(peak.get("relative_intensity", 0.0)),
                    "hkl_label": peak.get("hkl_label", ""),
                }
            )
    observed_count = len(observed)
    reference_count = len(reference)
    coverage = len(matches) / observed_count if observed_count else 0.0
    reference_coverage = len(matches) / reference_count if reference_count else 0.0
    mean_abs_delta = float(np.mean([abs(row["delta_two_theta_deg"]) for row in matches])) if matches else None
    position_score = max(0.0, 100.0 * (1.0 - (mean_abs_delta or tolerance_deg) / max(tolerance_deg, 1e-8)))
    overall = 70.0 * coverage + 20.0 * reference_coverage + 10.0 * position_score / 100.0
    return {
        "observed_count": observed_count,
        "reference_count": reference_count,
        "matched_count": len(matches),
        "observed_coverage": coverage,
        "reference_coverage": reference_coverage,
        "mean_abs_delta_two_theta_deg": mean_abs_delta,
        "position_score": position_score,
        "overall_score": float(overall),
        "matches": matches,
    }


def search_by_peak_list(
    db_path: str | Path,
    observed_peaks: Iterable[dict],
    *,
    tolerance_deg: float = 0.15,
    required_elements: Iterable[str] = (),
    limit: int = 25,
) -> list[dict]:
    rows = search_by_elements(db_path, required=required_elements, limit=100000)
    scored = []
    for row in rows:
        fingerprint = row.get("fingerprint", {}) or {}
        match = fingerprint_peak_match(observed_peaks, fingerprint, tolerance_deg=tolerance_deg)
        if match["matched_count"] == 0:
            continue
        result = dict(row)
        result["peak_match"] = match
        result["overall_score"] = match["overall_score"]
        scored.append(result)
    scored.sort(key=lambda row: (-row["overall_score"], -row["peak_match"]["matched_count"]))
    return scored[: int(limit)]


def candidate_record_from_library_row(row: dict, match_context: dict | None = None) -> dict:
    return {
        "origin": "Local CIF library",
        "source_id": row.get("source_id"),
        "relative_path": row.get("relative_path"),
        "absolute_path": row.get("absolute_path"),
        "sha256": row.get("sha256"),
        "formula": row.get("formula"),
        "elements": row.get("elements", []),
        "space_group": row.get("space_group"),
        "crystal_system": row.get("crystal_system"),
        "cell": {
            "a": row.get("a"), "b": row.get("b"), "c": row.get("c"),
            "alpha": row.get("alpha"), "beta": row.get("beta"), "gamma": row.get("gamma"),
        },
        "volume": row.get("volume"),
        "atom_count": row.get("atom_count"),
        "plausibility_status": row.get("plausibility_status"),
        "plausibility_score": row.get("plausibility_score"),
        "match_context": match_context or {},
        "scientific_status": "Possible CIF match",
        "warning": (
            "A local CIF-library match is a candidate only. It requires visual comparison, "
            "Pawley/Le Bail or Rietveld validation, chemical plausibility review and competing-candidate checks."
        ),
    }


def _existing_entry_by_relative_path(
    db_path: str | Path,
    library_root: str | Path,
    relative_path: str,
) -> dict | None:
    initialize_library_database(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT * FROM cif_entries
            WHERE library_root=? AND relative_path=?
            """,
            (str(Path(library_root).resolve()), relative_path),
        ).fetchone()
    return _row_to_dict(row) if row is not None else None


def _should_rescan_file(
    db_path: str | Path,
    library_root: str | Path,
    path: Path,
    *,
    options: IncrementalScanOptions,
) -> bool:
    if not options.incremental:
        return True
    root = Path(library_root)
    relative = _relative_path(root, path)
    existing = _existing_entry_by_relative_path(db_path, root, relative)
    if existing is None:
        return True
    stat = path.stat()
    if int(existing.get("file_size") or -1) != int(stat.st_size):
        return True
    if int(existing.get("modified_ns") or -1) != int(stat.st_mtime_ns):
        return True
    if options.retry_failed and existing.get("parse_status") != "ok":
        return True
    return False


def _write_scan_checkpoint(
    checkpoint_path: Path,
    payload: dict,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(checkpoint_path)


def build_cif_library_scalable(
    library_root: str | Path,
    db_path: str | Path,
    settings: CifLibrarySettings | None = None,
    options: IncrementalScanOptions | None = None,
    *,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Build or update a local CIF library using an incremental, resumable scan.

    This Stage-1 scalable implementation is intentionally conservative: it uses
    sequential parsing for deterministic behavior but records all state needed
    for pause/resume and later parallel execution. Unchanged files are skipped by
    size + modification time; changed rows are fully reparsed and checksummed.
    """
    settings = settings or CifLibrarySettings()
    options = options or IncrementalScanOptions()
    initialize_library_database(db_path)
    root = Path(library_root)
    files = scan_cif_files(root, settings.maximum_files)
    scan_id = options.scan_id or f"scan-{int(time.time())}"
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else Path(db_path).with_suffix(".scan_checkpoint.json")
    ok = 0
    failed = 0
    rescanned = 0
    skipped_unchanged = 0
    seen_relative_paths: set[str] = set()
    started_at = time.time()
    for index, path in enumerate(files, start=1):
        relative = _relative_path(root, path)
        seen_relative_paths.add(relative)
        if not _should_rescan_file(db_path, root, path, options=options):
            skipped_unchanged += 1
            with _connect(db_path) as connection:
                connection.execute(
                    """
                    UPDATE cif_entries SET last_seen_scan_id=?
                    WHERE library_root=? AND relative_path=?
                    """,
                    (scan_id, str(root.resolve()), relative),
                )
                connection.commit()
            continue
        row = parse_cif_for_library(path, root, settings)
        row["last_seen_scan_id"] = scan_id
        upsert_library_entry(db_path, row)
        rescanned += 1
        if row["parse_status"] == "ok":
            ok += 1
        else:
            failed += 1
        if options.checkpoint_every and index % max(1, int(options.checkpoint_every)) == 0:
            _write_scan_checkpoint(
                checkpoint,
                {
                    "scan_id": scan_id,
                    "library_root": str(root.resolve()),
                    "db_path": str(Path(db_path).resolve()),
                    "files_seen_so_far": index,
                    "total_files": len(files),
                    "rescanned": rescanned,
                    "skipped_unchanged": skipped_unchanged,
                    "ok_in_this_scan": ok,
                    "failed_in_this_scan": failed,
                    "updated_at": time.time(),
                },
            )
    missing = mark_missing_entries(db_path, root, seen_relative_paths, scan_id=scan_id)
    summary = {
        "library_root": str(root.resolve()),
        "db_path": str(Path(db_path).resolve()),
        "scan_id": scan_id,
        "files_seen": len(files),
        "rescanned": rescanned,
        "skipped_unchanged": skipped_unchanged,
        "ok": ok,
        "failed": failed,
        "missing_entries": missing,
        "elapsed_seconds": time.time() - started_at,
        "settings": asdict(settings),
        "options": asdict(options),
        "checkpoint_path": str(checkpoint.resolve()),
    }
    _write_scan_checkpoint(checkpoint, {**summary, "completed": True, "updated_at": time.time()})
    return summary


def mark_missing_entries(
    db_path: str | Path,
    library_root: str | Path,
    seen_relative_paths: set[str],
    *,
    scan_id: str,
) -> int:
    initialize_library_database(db_path)
    root_text = str(Path(library_root).resolve())
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT relative_path FROM cif_entries WHERE library_root=?",
            (root_text,),
        ).fetchall()
        missing = [row["relative_path"] for row in rows if row["relative_path"] not in seen_relative_paths]
        for relative in missing:
            connection.execute(
                """
                UPDATE cif_entries
                SET parse_status='missing', error_message='File missing during latest scan', last_seen_scan_id=?
                WHERE library_root=? AND relative_path=?
                """,
                (scan_id, root_text, relative),
            )
        connection.commit()
    return len(missing)


def library_health_report(db_path: str | Path) -> dict:
    initialize_library_database(db_path)
    with _connect(db_path) as connection:
        status_rows = connection.execute(
            "SELECT parse_status, COUNT(*) AS n FROM cif_entries GROUP BY parse_status"
        ).fetchall()
        material_rows = connection.execute(
            "SELECT material_type, COUNT(*) AS n FROM cif_entries WHERE parse_status='ok' GROUP BY material_type ORDER BY n DESC"
        ).fetchall()
        family_count = connection.execute(
            "SELECT COUNT(DISTINCT family_key) AS n FROM cif_entries WHERE parse_status='ok' AND family_key != ''"
        ).fetchone()["n"]
        total = connection.execute("SELECT COUNT(*) AS n FROM cif_entries").fetchone()["n"]
    return {
        "total_entries": int(total),
        "status_counts": {row["parse_status"]: int(row["n"]) for row in status_rows},
        "material_type_counts": {row["material_type"]: int(row["n"]) for row in material_rows},
        "structure_family_count": int(family_count or 0),
    }


def grouped_structure_families(
    db_path: str | Path,
    *,
    minimum_family_size: int = 2,
    limit: int = 50,
) -> list[dict]:
    initialize_library_database(db_path)
    with _connect(db_path) as connection:
        groups = connection.execute(
            """
            SELECT family_key, COUNT(*) AS n, MIN(formula) AS formula,
                   MIN(space_group) AS space_group, MIN(crystal_system) AS crystal_system,
                   MIN(relative_path) AS representative_path
            FROM cif_entries
            WHERE parse_status='ok' AND family_key != ''
            GROUP BY family_key
            HAVING COUNT(*) >= ?
            ORDER BY n DESC, formula
            LIMIT ?
            """,
            (int(minimum_family_size), int(limit)),
        ).fetchall()
    return [dict(row) for row in groups]


def _passes_material_filter(row: dict, material_types: Iterable[str] = ()) -> bool:
    wanted = {str(item).strip().lower() for item in material_types if str(item).strip()}
    if not wanted:
        return True
    material = str(row.get("material_type") or "").lower()
    return any(token in material for token in wanted)


def query_library_candidates(
    db_path: str | Path,
    *,
    required_elements: Iterable[str] = (),
    excluded_elements: Iterable[str] = (),
    material_types: Iterable[str] = (),
    formula_contains: str = "",
    limit: int = 1000,
) -> list[dict]:
    required_set = {normalize_element_symbol(el) for el in required_elements if normalize_element_symbol(el)}
    excluded_set = {normalize_element_symbol(el) for el in excluded_elements if normalize_element_symbol(el)}
    formula_query = str(formula_contains or "").strip().lower()
    rows = []
    for row in list_library_entries(db_path, status="ok"):
        elements = set(row.get("elements", []))
        if required_set and not required_set.issubset(elements):
            continue
        if excluded_set and excluded_set.intersection(elements):
            continue
        if formula_query and formula_query not in str(row.get("formula") or row.get("composition_key") or "").lower():
            continue
        if not _passes_material_filter(row, material_types):
            continue
        rows.append(row)
        if len(rows) >= int(limit):
            break
    return rows
