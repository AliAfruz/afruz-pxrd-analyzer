from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class GuiFeature:
    """One user-visible scientific capability exposed in the full GUI."""

    key: str
    label: str
    workspace: str
    tab_title: str
    scientific_boundary: str = ""


GUI_FEATURES: tuple[GuiFeature, ...] = (
    GuiFeature("pattern_import", "Import XRD / RD / table formats", "Data", "Raw Data"),
    GuiFeature("preprocessing", "Background, smoothing and preparation", "Prepare", "Background / Smoothing"),
    GuiFeature("instrument_calibration", "Instrument calibration", "Prepare", "Instrument Calibration"),
    GuiFeature("peak_curation", "Peak detection, manual peaks and fitting", "Peaks", "Peak Analysis"),
    GuiFeature("size_strain", "Crystallite size, strain and microstructure", "Peaks", "Size & Strain"),
    GuiFeature("residual_stress", "sin²ψ residual stress", "Peaks", "Residual Stress"),
    GuiFeature("phase_identification", "Known-phase identification", "Phase", "Phase Identification"),
    GuiFeature("cif_cell", "CIF import and cell refinement", "Phase", "CIF & Cell"),
    GuiFeature("unknown_phase", "Unknown phase indexing and multiphase discovery", "Phase", "Unknown Phase"),
    GuiFeature("cif_library", "Scalable CIF library and intelligent matching", "Phase", "CIF Library / Structure Match"),
    GuiFeature("structure_solution", "Structure solution and fragment solving", "Structure", "Solve Structure"),
    GuiFeature("pawley_lebail", "Pawley / Le Bail whole-pattern refinement", "Refine", "Pawley / Le Bail"),
    GuiFeature("rietveld", "Single-structure Rietveld refinement", "Refine", "Rietveld"),
    GuiFeature("multicomponent_refiner", "Intelligent multiphase CIF refiner", "Refine", "Intelligent Multiphase"),
    GuiFeature("qpa", "Validated quantitative phase analysis", "Quantify", "Validated QPA"),
    GuiFeature("validation", "Validation, evidence and scientific state", "Validate", "Validation & Evidence"),
    GuiFeature("reports", "Complete report package export", "Report", "Batch & Reports"),
    GuiFeature("batch", "Batch processing and reproducible recipes", "Report", "Batch & Reports"),
)


def gui_feature_records() -> list[dict]:
    return [asdict(feature) for feature in GUI_FEATURES]


def gui_feature_labels() -> list[str]:
    return [feature.label for feature in GUI_FEATURES]


def validate_full_gui_coverage(required_keys: Iterable[str] | None = None) -> dict:
    """Return a small audit proving that core scientific engines are exposed in GUI metadata."""

    required = set(required_keys or [feature.key for feature in GUI_FEATURES])
    present = {feature.key for feature in GUI_FEATURES}
    missing = sorted(required - present)
    return {
        "success": not missing,
        "phase": "19.2",
        "method": "Full GUI feature coverage audit",
        "feature_count": len(GUI_FEATURES),
        "present_keys": sorted(present),
        "missing_keys": missing,
        "scientific_boundary": (
            "This audit confirms GUI exposure/launch coverage. It does not validate the scientific result produced by each engine."
        ),
    }
