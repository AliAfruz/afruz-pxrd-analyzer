from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .rietveld_reference import result_plot_metadata
from .manuscript_plot_export import MANUSCRIPT_COLUMN_SPECS, build_manuscript_plot_data
from .text_export import (
    ensure_txt_path,
    safe_filename,
    write_columns_txt,
    write_manifest_txt,
    write_mapping_txt,
    write_table_txt,
)


def export_refinement_txt_bundle(
    path: str | Path,
    result: Mapping[str, Any],
    *,
    dataset_name: str,
    dataset_uid: str = "",
    refinement_label: str = "Rietveld refinement",
    include_reference_image_data: bool = False,
) -> dict[str, Any]:
    """Export a complete refinement result as validated tab-delimited TXT files."""
    base = ensure_txt_path(path)
    metadata = {
        "dataset_name": dataset_name,
        "dataset_uid": dataset_uid or "Not recorded",
        "refinement_label": refinement_label,
    }
    summary = write_mapping_txt(
        base,
        result,
        title=f"{refinement_label} complete result — {dataset_name}",
        metadata=metadata,
    )
    stem = base.stem
    phase_path = base.with_name(stem + "_phases.txt")
    reflection_path = base.with_name(stem + "_reflections.txt")
    profile_path = base.with_name(stem + "_profile.txt")
    bragg_path = base.with_name(stem + "_bragg_positions.txt")
    plot_metadata_path = base.with_name(stem + "_plot_metadata.txt")
    statistics_path = base.with_name(stem + "_statistics.txt")
    covariance_path = base.with_name(stem + "_intensity_covariance.txt")
    overlap_path = base.with_name(stem + "_overlap_groups.txt")
    normalization_path = base.with_name(stem + "_cif_pattern_scaling.txt")
    manuscript_path = base.with_name(f"plot-data-for-manuscrit_{safe_filename(dataset_name)}")

    phases = [dict(row) for row in result.get("phases", [])]
    reflections = [dict(row) for row in result.get("reflections", [])]
    phase_export = write_table_txt(
        phase_path,
        phases,
        title=f"{refinement_label} phase results — {dataset_name}",
        metadata=metadata,
    )
    reflection_export = write_table_txt(
        reflection_path,
        reflections,
        title=f"{refinement_label} reflection results — {dataset_name}",
        metadata=metadata,
    )
    profile_columns = {
        "two_theta_deg": result.get("observed_x", []),
        "observed_intensity": result.get("observed_y", []),
        "calculated_intensity": result.get("calculated_y", []),
        "background_intensity": result.get("background_y", []),
        "difference_observed_minus_calculated": result.get("difference_y", []),
    }
    if result.get("observed_sigma") is not None:
        profile_columns["observed_sigma"] = result.get("observed_sigma", [])
    if result.get("statistical_weights") is not None:
        profile_columns["statistical_weight"] = result.get("statistical_weights", [])
    profile_export = write_columns_txt(
        profile_path,
        profile_columns,
        title=f"{refinement_label} observed/calculated profile — {dataset_name}",
        metadata={
            **metadata,
            "difference_definition": "observed_intensity - calculated_intensity",
        },
    )

    phases_by_index = {
        int(phase.get("phase_index", index)): phase
        for index, phase in enumerate(phases)
        if str(phase.get("phase_index", index)).lstrip("-").isdigit()
    }
    bragg_rows = []
    for reflection in reflections:
        phase_index = reflection.get("phase_index")
        try:
            phase = phases_by_index.get(int(phase_index), {})
        except Exception:
            phase = {}
        bragg_rows.append(
            {
                "phase_index": phase_index,
                "phase_name": reflection.get("phase_name") or phase.get("phase_name"),
                "formula": reflection.get("formula") or phase.get("formula"),
                "hkl_label": reflection.get("hkl_label"),
                "h": reflection.get("h"),
                "k": reflection.get("k"),
                "l": reflection.get("l"),
                "two_theta_deg": reflection.get("two_theta_deg"),
                "d_spacing_angstrom": reflection.get("d_spacing") or reflection.get("d_spacing_angstrom"),
                "relative_or_structure_intensity": (
                    reflection.get("structure_intensity")
                    if reflection.get("structure_intensity") is not None
                    else reflection.get("reference_intensity")
                ),
            }
        )
    bragg_export = write_table_txt(
        bragg_path,
        bragg_rows,
        title=f"{refinement_label} Bragg marker positions — {dataset_name}",
        metadata={**metadata, "plot_role": "vertical phase tick marks below the main profile"},
    )

    plot_metadata = result_plot_metadata(result)
    if not include_reference_image_data:
        plot_metadata.pop("reference_image_layout", None)
    plot_export = write_mapping_txt(
        plot_metadata_path,
        plot_metadata,
        title=f"{refinement_label} plot metadata — {dataset_name}",
        metadata=metadata,
    )

    statistics_export = write_mapping_txt(
        statistics_path,
        {
            "statistics_valid": result.get("statistics_valid"),
            "statistics_reason": result.get("statistics_reason"),
            "weighting": result.get("weighting"),
            "weighting_model": result.get("weighting_model"),
            "intensity_provenance": result.get("intensity_provenance"),
            "intensity_scale_factor": result.get("intensity_scale_factor"),
            "rwp_percent": result.get("rwp_percent"),
            "rp_percent": result.get("rp_percent"),
            "rexp_percent": result.get("rexp_percent"),
            "goodness_of_fit_sqrt": result.get("goodness_of_fit_sqrt"),
            "reduced_chi_square": result.get("reduced_chi_square"),
            "degrees_of_freedom": result.get("degrees_of_freedom"),
            "weighted_residual_sum_squares": result.get("weighted_residual_sum_squares"),
            "weighted_observed_sum_squares": result.get("weighted_observed_sum_squares"),
        },
        title=f"{refinement_label} statistical validity — {dataset_name}",
        metadata=metadata,
    )
    metrology = dict(result.get("intensity_metrology", {}) or {})
    covariance = metrology.get("intensity_covariance", [])
    correlation = metrology.get("intensity_correlation", [])
    covariance_rows = []
    for left, covariance_row in enumerate(covariance):
        for right in range(left, len(covariance_row)):
            left_reflection = reflections[left] if left < len(reflections) else {}
            right_reflection = reflections[right] if right < len(reflections) else {}
            correlation_value = None
            if left < len(correlation) and right < len(correlation[left]):
                correlation_value = correlation[left][right]
            covariance_rows.append(
                {
                    "reflection_i": left,
                    "phase_i": left_reflection.get("phase_name"),
                    "hkl_i": left_reflection.get("hkl_label"),
                    "reflection_j": right,
                    "phase_j": right_reflection.get("phase_name"),
                    "hkl_j": right_reflection.get("hkl_label"),
                    "covariance": covariance_row[right],
                    "correlation": correlation_value,
                }
            )
    covariance_export = write_table_txt(
        covariance_path,
        covariance_rows,
        title=f"{refinement_label} reflection-intensity covariance — {dataset_name}",
        metadata={
            **metadata,
            "covariance_interpretation": metrology.get("covariance_interpretation"),
            "matrix_storage": "upper triangle including diagonal",
        },
    )
    overlap_export = write_table_txt(
        overlap_path,
        metrology.get("overlap_groups", []),
        title=f"{refinement_label} reflection overlap groups — {dataset_name}",
        metadata={
            **metadata,
            "overlap_correlation_threshold": metrology.get("overlap_correlation_threshold"),
            "severe_correlation_threshold": metrology.get("severe_correlation_threshold"),
        },
    )
    normalization_export = write_mapping_txt(
        normalization_path,
        result.get("generated_pattern_normalization", {
            "status": "Not recorded",
            "generated_pattern_scaling": result.get("generated_pattern_scaling"),
        }),
        title=f"{refinement_label} CIF-generated pattern scaling — {dataset_name}",
        metadata={
            **metadata,
            "scientific_boundary": (
                "CIF files contain structure, not measured intensity counts. "
                "Generated phase profiles are scaled to observed units without changing the experimental pattern."
            ),
        },
    )

    manuscript_rows, manuscript_metadata = build_manuscript_plot_data(result)
    manuscript_heading_by_key = {key: f"{label} [{unit}]" if unit else label for key, label, unit, _ in MANUSCRIPT_COLUMN_SPECS}
    manuscript_headings = [manuscript_heading_by_key[key] for key, _, _, _ in MANUSCRIPT_COLUMN_SPECS]
    manuscript_export_rows = [
        {manuscript_heading_by_key[key]: row.get(key) for key, _, _, _ in MANUSCRIPT_COLUMN_SPECS}
        for row in manuscript_rows
    ]
    manuscript_export = write_table_txt(
        manuscript_path,
        manuscript_export_rows,
        columns=manuscript_headings,
        title=f"Plot data for manuscript — {dataset_name}",
        metadata={
            **metadata,
            **manuscript_metadata,
            "usage": "Filter Record type = PROFILE for curves and Record type = BRAGG for phase-resolved Bragg lines.",
        },
    )

    files = [
        Path(summary["txt_path"]),
        Path(phase_export["txt_path"]),
        Path(reflection_export["txt_path"]),
        Path(profile_export["txt_path"]),
        Path(bragg_export["txt_path"]),
        Path(plot_export["txt_path"]),
        Path(statistics_export["txt_path"]),
        Path(covariance_export["txt_path"]),
        Path(overlap_export["txt_path"]),
        Path(normalization_export["txt_path"]),
        Path(manuscript_export["txt_path"]),
    ]
    manifest_path = base.with_name(stem + "_manifest.txt")
    manifest = write_manifest_txt(
        manifest_path,
        files,
        root=base.parent,
        title=f"{refinement_label} TXT export manifest — {dataset_name}",
    )
    return {
        "summary_txt": summary["txt_path"],
        "phases_txt": phase_export["txt_path"],
        "reflections_txt": reflection_export["txt_path"],
        "profile_txt": profile_export["txt_path"],
        "bragg_positions_txt": bragg_export["txt_path"],
        "plot_metadata_txt": plot_export["txt_path"],
        "statistics_txt": statistics_export["txt_path"],
        "intensity_covariance_txt": covariance_export["txt_path"],
        "overlap_groups_txt": overlap_export["txt_path"],
        "cif_pattern_scaling_txt": normalization_export["txt_path"],
        "plot_data_for_manuscript_txt": manuscript_export["txt_path"],
        "manifest_txt": manifest["txt_path"],
        "file_count": len(files) + 1,
    }
