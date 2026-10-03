"""Validate a user-supplied CIF without copying its atomic coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from afruz_pxrd.crystal_intelligence import analyze_crystal_model
from afruz_pxrd.crystallography import load_cif


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path: Path, expected_sha256: str | None = None) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    fingerprint = _sha256(resolved)
    expected = str(expected_sha256 or "").strip().lower()
    if expected and fingerprint.lower() != expected:
        raise ValueError(
            f"SHA-256 mismatch: expected {expected}, calculated {fingerprint}"
        )

    model = load_cif(resolved)
    report = analyze_crystal_model(model)
    atoms = list(model.get("atoms") or [])
    elements = Counter(str(atom.get("element") or "X") for atom in atoms)
    return {
        "schema": "afruz.local-cif-validation.v1",
        "filename": resolved.name,
        "sha256": fingerprint,
        "data_name": model.get("data_name"),
        "cell": model.get("cell"),
        "expanded_atom_site_count": len(atoms),
        "element_counts": dict(sorted(elements.items())),
        "classification": report.get("classification"),
        "topology_analysis_limitations": report.get("limitations"),
        "coordinates_exported": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a local CIF and print a non-coordinate JSON summary."
    )
    parser.add_argument("cif", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    print(json.dumps(validate(args.cif, args.expected_sha256), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
