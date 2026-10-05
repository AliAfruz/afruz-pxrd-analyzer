"""Shared raw-unit background snapshots and bounded additive corrections."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib

import numpy as np

BACKGROUND_KEY = "refinement_background"
SEED_STATISTICS_NOTE = (
    "The starting background was estimated from the observations; its uncertainty and effective degrees of freedom "
    "are not propagated. Rwp/Rp remain fit measures; Rexp, GoF and reduced chi-square are unavailable."
)


def array_fingerprint(*arrays) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        array = np.asarray(values, dtype="<f8")
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError("Background snapshots require finite one-dimensional arrays.")
        digest.update(str(len(array)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def background_selection(dataset, result: dict) -> dict:
    seed = np.asarray(result.get("background", []), dtype=float)
    if seed.shape != np.asarray(dataset.x).shape or not np.all(np.isfinite(seed)) or np.any(seed < 0):
        raise ValueError("The chosen background must match the raw pattern and be finite and non-negative.")
    protected = np.asarray(result.get("protected_mask", np.zeros(len(seed))), dtype=bool)
    if protected.shape != seed.shape or any(not isinstance(result.get(key) or {}, dict) for key in ("parameters", "diagnostics")):
        raise ValueError("The background parameters, diagnostics or peak-protection mask are invalid.")
    return {"schema": 1, "mode": "seed_plus_correction", "source_dataset_uid": str(dataset.uid),
            "source_dataset_name": str(dataset.name), "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_pattern_sha256": array_fingerprint(dataset.x, dataset.y_raw),
            "seed_sha256": array_fingerprint(dataset.x, seed), "x": np.asarray(dataset.x).tolist(),
            "background": seed.tolist(), "parameters": deepcopy(result.get("parameters") or {}),
            "diagnostics": deepcopy(result.get("diagnostics") or {}),
            "protected_mask": protected.tolist(),
            "estimated_from_observed": True, "intensity_units": "raw measurement units", "observation_clipping": False}


def selected_background(dataset) -> dict | None:
    model = (dataset.metadata or {}).get(BACKGROUND_KEY)
    if model is None:
        return None
    if not isinstance(model, dict) or model.get("schema") != 1 or model.get("mode") != "seed_plus_correction":
        raise ValueError("The selected refinement background is invalid; preview and choose it again.")
    if model.get("source_pattern_sha256") != array_fingerprint(dataset.x, dataset.y_raw):
        raise ValueError("The raw pattern changed after choosing its background. Preview and choose a new background.")
    seed = np.asarray(model.get("background", []), dtype=float)
    axis = np.asarray(model.get("x", []), dtype=float)
    if seed.shape != np.asarray(dataset.x).shape or not np.array_equal(axis, dataset.x):
        raise ValueError("The chosen background does not match this raw pattern's grid.")
    if not np.all(np.isfinite(seed)) or np.any(seed < 0) or model.get("seed_sha256") != array_fingerprint(axis, seed):
        raise ValueError("The chosen background snapshot is corrupt. Preview and choose it again.")
    if any(not isinstance(model.get(key) or {}, dict) for key in ("parameters", "diagnostics")) or np.asarray(model.get("protected_mask", [])).shape != seed.shape:
        raise ValueError("The background parameters, diagnostics or peak-protection mask are invalid.")
    return deepcopy(model)


def background_kwargs(dataset) -> dict:
    model = selected_background(dataset)
    if model is None:
        return {}
    return {"background_seed_y": np.asarray(model["background"], dtype=float).copy(),
            "background_seed_provenance": {key: deepcopy(value) for key, value in model.items()
                if key not in {"x", "background", "protected_mask"}}}


def subset_background(source_dataset, filtered_dataset) -> None:
    model = selected_background(source_dataset)
    if model is None:
        return
    indices = np.searchsorted(source_dataset.x, filtered_dataset.x)
    if np.any(indices >= len(source_dataset.x)) or not np.array_equal(np.asarray(source_dataset.x)[indices], filtered_dataset.x):
        raise ValueError("The exclusion grid does not match the chosen background.")
    model["source_background_sha256"] = model["seed_sha256"]
    model["background"] = np.asarray(model["background"])[indices].tolist()
    model["x"] = np.asarray(filtered_dataset.x).tolist()
    model["protected_mask"] = np.asarray(model["protected_mask"])[indices].tolist()
    model["source_pattern_sha256"] = array_fingerprint(filtered_dataset.x, filtered_dataset.y_raw)
    model["seed_sha256"] = array_fingerprint(filtered_dataset.x, model["background"])
    filtered_dataset.metadata[BACKGROUND_KEY] = model


class SeedBackground:
    """B = B0 + correction, with correction >= -min(B0).

    Bernstein polynomials sum to one. Fit nonnegative coefficients against
    y - (B0 - min(B0)), then subtract min(B0) from the reported coefficients.
    This permits positive and negative corrections while guaranteeing B >= 0,
    and reuses the existing CPU/CUDA bounded solvers without changing weights.
    """
    def __init__(self, x, seed=None, provenance=None):
        self.x = np.asarray(x, dtype=float)
        self.seed = None if seed is None else np.asarray(seed, dtype=float).copy()
        self.provenance = deepcopy(provenance or {})
        self.floor = 0.0
        if self.seed is not None:
            if self.seed.shape != self.x.shape or not np.all(np.isfinite(self.seed)) or np.any(self.seed < 0):
                raise ValueError("Starting background must be finite, non-negative and match the complete observed grid.")
            if not np.all(np.isfinite(self.x)) or np.any(np.diff(self.x) <= 0):
                raise ValueError("Starting background needs a finite, strictly increasing observed grid.")
            self.floor = float(np.min(self.seed))

    def values(self, x):
        x = np.asarray(x, dtype=float)
        return np.zeros_like(x) if self.seed is None else np.interp(x, self.x, self.seed)

    def offset(self, x):
        return self.values(x) - self.floor

    def weight_model(self, model):
        if self.seed is None or self.provenance.get("independent_known_background") is True:
            return model
        return replace(model, statistics_valid=False, rexp_available=False, reason=SEED_STATISTICS_NOTE,
                       warnings=(*model.warnings, SEED_STATISTICS_NOTE))

    def result_fields(self, x, coefficients, total_background):
        if self.seed is None:
            return {}
        base = self.values(x)
        return {"background_seed_y": base.tolist(),
                "background_correction_y": (np.asarray(total_background) - base).tolist(),
                "background_coefficients": (np.asarray(coefficients) - self.floor).tolist(),
                "background_model": {**deepcopy(self.provenance), "mode": "seed_plus_correction",
                    "correction_basis": "Bernstein polynomial", "correction_coefficient_lower_bound": -self.floor,
                    "coefficient_convention": "signed correction coefficients relative to starting background",
                    "observation_clipping": False, "statistics_note": (
                        "Starting background treated as an independent exact curve; its uncertainty is not propagated."
                        if self.provenance.get("independent_known_background") is True else SEED_STATISTICS_NOTE),
                    "rule": "total background = chosen starting background + fitted Bernstein correction; total >= 0"}}
