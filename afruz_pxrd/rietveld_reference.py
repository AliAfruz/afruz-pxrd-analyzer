from __future__ import annotations

"""Reference annotations transcribed from the user-supplied Rietveld image."""

from typing import Any, Mapping

RIETVELD_REFERENCE_IMAGE_DATA: dict[str, Any] = {
    "source": "User-supplied reference image; transcription only",
    "plot_title": "Rietveld Refinement",
    "x_axis_label": "2θ (°)",
    "y_axis_label": "Intensity (a.u.)",
    "x_axis_visible_range_deg": [10.0, 70.0],
    "observed_style": "black point markers",
    "calculated_style": "thin red line",
    "bragg_marker_style": "vertical blue ticks below the main profile",
    "difference_style": "black difference curve in a lower offset band",
    "legend_phase_formula": "Rb3InCl6",
    "crystal_system": "Monoclinic",
    "space_group": "C2/c",
    "unit_cell_volume_angstrom3": 2451.0,
    "citation_transcribed": (
        "J. D. Majher, M. B. Gray, T. Liu, N. P. Holzapfel, P. M. Woodward, "
        "Inorg. Chem. 59, 14478–14485 (2020)."
    ),
    "scientific_note": (
        "The phase identity and volume above belong to the reference image, "
        "not to the user's current dataset. Afruz must export measured/refined "
        "values from the active result separately."
    ),
}


def result_plot_metadata(result: Mapping[str, Any] | None) -> dict[str, Any]:
    result = result or {}
    return {
        "plot_title": "Rietveld Refinement",
        "x_axis_label": "2θ (°)",
        "y_axis_label": "Intensity (a.u.)",
        "observed_series": "observed_y",
        "calculated_series": "calculated_y",
        "background_series": "background_y",
        "difference_series": "difference_y = observed_y - calculated_y",
        "bragg_positions_source": "reflections.two_theta_deg grouped by phase_index",
        "rwp_percent": result.get("rwp_percent"),
        "rp_percent": result.get("rp_percent"),
        "r_squared": result.get("r_squared"),
        "phase_count": result.get("phase_count", len(result.get("phases", []))),
        "reflection_count": len(result.get("reflections", [])),
        "reference_image_layout": RIETVELD_REFERENCE_IMAGE_DATA,
    }
