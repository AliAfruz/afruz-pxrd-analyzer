from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

import numpy as np

from .base import (
    ContractValidationError,
    ScientificResultContract,
    numeric_array,
    require_aligned_arrays,
)


def _records(value: Any, field_name: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"{field_name} must be a list")
    if not all(isinstance(row, Mapping) for row in value):
        raise ContractValidationError(f"{field_name} must contain objects")
    return tuple(dict(row) for row in value)


@dataclass(kw_only=True)
class PreprocessingResult(ScientificResultContract):
    KIND: ClassVar[str] = "preprocessing"
    processed_profile: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    background_profile: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    smoothed_profile: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    settings: dict[str, Any] = field(default_factory=dict)

    def validate_result(self) -> None:
        if not isinstance(self.settings, dict):
            raise ContractValidationError("settings must be a dictionary")
        self.processed_profile = numeric_array(self.processed_profile, field_name="processed_profile")
        self.background_profile = numeric_array(self.background_profile, field_name="background_profile")
        self.smoothed_profile = numeric_array(self.smoothed_profile, field_name="smoothed_profile")
        require_aligned_arrays(
            processed_profile=self.processed_profile,
            background_profile=self.background_profile,
            smoothed_profile=self.smoothed_profile,
        )

    def numerical_payload(self) -> dict[str, Any]:
        return {
            "processed_profile": self.processed_profile,
            "background_profile": self.background_profile,
            "smoothed_profile": self.smoothed_profile,
            "settings": self.settings,
        }

    @classmethod
    def payload_kwargs(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        required = {"processed_profile", "background_profile", "smoothed_profile", "settings"}
        missing = required - set(value)
        if missing:
            raise ContractValidationError(f"Preprocessing outputs are missing: {sorted(missing)}")
        return dict(value)


@dataclass(frozen=True)
class PeakRecord:
    position_deg: float
    intensity: float
    fwhm_deg: float | None = None
    included: bool = True
    origin: str = "Unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (("position_deg", self.position_deg), ("intensity", self.intensity)):
            if not np.isfinite(float(value)):
                raise ContractValidationError(f"Peak {name} must be finite")
        if self.fwhm_deg is not None and not np.isfinite(float(self.fwhm_deg)):
            raise ContractValidationError("Peak fwhm_deg must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_deg": float(self.position_deg),
            "intensity": float(self.intensity),
            "fwhm_deg": None if self.fwhm_deg is None else float(self.fwhm_deg),
            "included": bool(self.included),
            "origin": self.origin,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PeakRecord":
        if not isinstance(value, Mapping):
            raise ContractValidationError("Peak record must be an object")
        missing = {"position_deg", "intensity", "fwhm_deg", "included", "origin", "metadata"} - set(value)
        if missing:
            raise ContractValidationError(f"Peak record is missing fields: {sorted(missing)}")
        return cls(
            position_deg=float(value["position_deg"]),
            intensity=float(value["intensity"]),
            fwhm_deg=None if value["fwhm_deg"] is None else float(value["fwhm_deg"]),
            included=bool(value["included"]),
            origin=str(value["origin"]),
            metadata=dict(value["metadata"]),
        )


@dataclass(kw_only=True)
class PeakListResult(ScientificResultContract):
    KIND: ClassVar[str] = "peak_list"
    peaks: tuple[PeakRecord, ...] = ()
    locked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_result(self) -> None:
        if not isinstance(self.locked, bool):
            raise ContractValidationError("locked must be a boolean")
        if not isinstance(self.metadata, dict):
            raise ContractValidationError("metadata must be a dictionary")
        if not all(isinstance(row, PeakRecord) for row in self.peaks):
            raise ContractValidationError("peaks must contain PeakRecord values")

    def numerical_payload(self) -> dict[str, Any]:
        return {
            "peaks": [row.to_dict() for row in self.peaks],
            "locked": self.locked,
            "metadata": self.metadata,
        }

    @classmethod
    def payload_kwargs(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        missing = {"peaks", "locked", "metadata"} - set(value)
        if missing:
            raise ContractValidationError(f"Peak-list outputs are missing: {sorted(missing)}")
        return {
            "peaks": tuple(PeakRecord.from_dict(row) for row in value["peaks"]),
            "locked": bool(value["locked"]),
            "metadata": dict(value["metadata"]),
        }


@dataclass(kw_only=True)
class PeakFittingResult(ScientificResultContract):
    KIND: ClassVar[str] = "peak_fitting"
    groups: tuple[dict[str, Any], ...] = ()
    candidates: tuple[dict[str, Any], ...] = ()

    def validate_result(self) -> None:
        self.groups = _records(self.groups, "groups")
        self.candidates = _records(self.candidates, "candidates")

    def numerical_payload(self) -> dict[str, Any]:
        return {"groups": list(self.groups), "candidates": list(self.candidates)}

    @classmethod
    def payload_kwargs(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        missing = {"groups", "candidates"} - set(value)
        if missing:
            raise ContractValidationError(f"Peak-fitting outputs are missing: {sorted(missing)}")
        return {"groups": _records(value["groups"], "groups"), "candidates": _records(value["candidates"], "candidates")}


@dataclass(kw_only=True)
class InstrumentCalibrationResult(ScientificResultContract):
    KIND: ClassVar[str] = "instrument_calibration"
    profile: dict[str, Any] = field(default_factory=dict)
    qa_result: dict[str, Any] = field(default_factory=dict)

    def validate_result(self) -> None:
        if not isinstance(self.profile, dict):
            raise ContractValidationError("profile must be a dictionary")
        if not isinstance(self.qa_result, dict):
            raise ContractValidationError("qa_result must be a dictionary")

    def numerical_payload(self) -> dict[str, Any]:
        return {"profile": self.profile, "qa_result": self.qa_result}

    @classmethod
    def payload_kwargs(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        missing = {"profile", "qa_result"} - set(value)
        if missing:
            raise ContractValidationError(f"Calibration outputs are missing: {sorted(missing)}")
        return {"profile": dict(value["profile"]), "qa_result": dict(value["qa_result"])}


@dataclass(kw_only=True)
class UnitCellRefinementResult(ScientificResultContract):
    KIND: ClassVar[str] = "unit_cell_refinement"
    initial_cell: dict[str, float] = field(default_factory=dict)
    refined_cell: dict[str, float] = field(default_factory=dict)
    matches: tuple[dict[str, Any], ...] = ()
    statistics: dict[str, Any] = field(default_factory=dict)

    def validate_result(self) -> None:
        for field_name in ("initial_cell", "refined_cell", "statistics"):
            if not isinstance(getattr(self, field_name), dict):
                raise ContractValidationError(f"{field_name} must be a dictionary")
        for field_name in ("initial_cell", "refined_cell"):
            for key, value in getattr(self, field_name).items():
                if not np.isfinite(float(value)):
                    raise ContractValidationError(
                        f"{field_name} value '{key}' must be finite"
                    )
        self.matches = _records(self.matches, "matches")

    def numerical_payload(self) -> dict[str, Any]:
        return {"initial_cell": self.initial_cell, "refined_cell": self.refined_cell, "matches": list(self.matches), "statistics": self.statistics}

    @classmethod
    def payload_kwargs(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        missing = {"initial_cell", "refined_cell", "matches", "statistics"} - set(value)
        if missing:
            raise ContractValidationError(f"Unit-cell outputs are missing: {sorted(missing)}")
        return {"initial_cell": dict(value["initial_cell"]), "refined_cell": dict(value["refined_cell"]), "matches": _records(value["matches"], "matches"), "statistics": dict(value["statistics"])}


@dataclass(kw_only=True)
class PhaseIdentificationResult(ScientificResultContract):
    KIND: ClassVar[str] = "phase_identification"
    candidates: tuple[dict[str, Any], ...] = ()
    selected_candidate: dict[str, Any] | None = None
    matches: tuple[dict[str, Any], ...] = ()

    def validate_result(self) -> None:
        if self.selected_candidate is not None and not isinstance(
            self.selected_candidate, dict
        ):
            raise ContractValidationError(
                "selected_candidate must be a dictionary or null"
            )
        self.candidates = _records(self.candidates, "candidates")
        self.matches = _records(self.matches, "matches")

    def numerical_payload(self) -> dict[str, Any]:
        return {"candidates": list(self.candidates), "selected_candidate": self.selected_candidate, "matches": list(self.matches)}

    @classmethod
    def payload_kwargs(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        missing = {"candidates", "selected_candidate", "matches"} - set(value)
        if missing:
            raise ContractValidationError(f"Phase outputs are missing: {sorted(missing)}")
        selected = value["selected_candidate"]
        return {"candidates": _records(value["candidates"], "candidates"), "selected_candidate": dict(selected) if isinstance(selected, Mapping) else None, "matches": _records(value["matches"], "matches")}


@dataclass(kw_only=True)
class WholePatternRefinementResult(ScientificResultContract):
    KIND: ClassVar[str] = "whole_pattern_refinement"
    observed_x: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    observed_y: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    calculated_y: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    background_y: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    statistics: dict[str, Any] = field(default_factory=dict)
    reflections: tuple[dict[str, Any], ...] = ()
    intensity_metrology: dict[str, Any] = field(default_factory=dict)

    def validate_result(self) -> None:
        if not isinstance(self.statistics, dict):
            raise ContractValidationError("statistics must be a dictionary")
        if not isinstance(self.intensity_metrology, dict):
            raise ContractValidationError("intensity_metrology must be a dictionary")
        self.reflections = _records(self.reflections, "reflections")
        for name in ("observed_x", "observed_y", "calculated_y", "background_y"):
            setattr(self, name, numeric_array(getattr(self, name), field_name=name))
        require_aligned_arrays(observed_x=self.observed_x, observed_y=self.observed_y, calculated_y=self.calculated_y, background_y=self.background_y)

    def numerical_payload(self) -> dict[str, Any]:
        return {
            "observed_x": self.observed_x,
            "observed_y": self.observed_y,
            "calculated_y": self.calculated_y,
            "background_y": self.background_y,
            "statistics": self.statistics,
            "reflections": list(self.reflections),
            "intensity_metrology": self.intensity_metrology,
        }

    @classmethod
    def payload_kwargs(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        missing = {"observed_x", "observed_y", "calculated_y", "background_y", "statistics"} - set(value)
        if missing:
            raise ContractValidationError(f"Whole-pattern outputs are missing: {sorted(missing)}")
        return {
            "observed_x": value["observed_x"],
            "observed_y": value["observed_y"],
            "calculated_y": value["calculated_y"],
            "background_y": value["background_y"],
            "statistics": dict(value["statistics"]),
            "reflections": _records(value.get("reflections", []), "reflections"),
            "intensity_metrology": dict(value.get("intensity_metrology", {})),
        }


@dataclass(kw_only=True)
class RietveldRefinementResult(WholePatternRefinementResult):
    KIND: ClassVar[str] = "rietveld_refinement"
    phases: tuple[dict[str, Any], ...] = ()

    def validate_result(self) -> None:
        super().validate_result()
        self.phases = _records(self.phases, "phases")

    def numerical_payload(self) -> dict[str, Any]:
        return {**super().numerical_payload(), "phases": list(self.phases)}

    @classmethod
    def payload_kwargs(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        values = WholePatternRefinementResult.payload_kwargs(value)
        if "phases" not in value:
            raise ContractValidationError("Rietveld outputs are missing: ['phases']")
        values["phases"] = _records(value["phases"], "phases")
        return values


@dataclass(kw_only=True)
class QPAResult(ScientificResultContract):
    KIND: ClassVar[str] = "qpa"
    mode: str = "Unknown"
    phases: tuple[dict[str, Any], ...] = ()
    fractions: dict[str, float] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)

    def validate_result(self) -> None:
        if not isinstance(self.mode, str) or not self.mode.strip():
            raise ContractValidationError("mode must be non-empty text")
        if not isinstance(self.fractions, dict):
            raise ContractValidationError("fractions must be a dictionary")
        if not isinstance(self.statistics, dict):
            raise ContractValidationError("statistics must be a dictionary")
        self.phases = _records(self.phases, "phases")
        for key, value in self.fractions.items():
            if not np.isfinite(float(value)):
                raise ContractValidationError(f"QPA fraction '{key}' must be finite")

    def numerical_payload(self) -> dict[str, Any]:
        return {"mode": self.mode, "phases": list(self.phases), "fractions": self.fractions, "statistics": self.statistics}

    @classmethod
    def payload_kwargs(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        missing = {"mode", "phases", "fractions", "statistics"} - set(value)
        if missing:
            raise ContractValidationError(f"QPA outputs are missing: {sorted(missing)}")
        return {"mode": str(value["mode"]), "phases": _records(value["phases"], "phases"), "fractions": {str(key): float(item) for key, item in dict(value["fractions"]).items()}, "statistics": dict(value["statistics"])}


@dataclass(kw_only=True)
class ValidationResult(ScientificResultContract):
    KIND: ClassVar[str] = "validation"
    evidence: tuple[dict[str, Any], ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)

    def validate_result(self) -> None:
        if not isinstance(self.summary, dict):
            raise ContractValidationError("summary must be a dictionary")
        self.evidence = _records(self.evidence, "evidence")

    def numerical_payload(self) -> dict[str, Any]:
        return {"evidence": list(self.evidence), "summary": self.summary}

    @classmethod
    def payload_kwargs(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        missing = {"evidence", "summary"} - set(value)
        if missing:
            raise ContractValidationError(f"Validation outputs are missing: {sorted(missing)}")
        return {"evidence": _records(value["evidence"], "evidence"), "summary": dict(value["summary"])}
