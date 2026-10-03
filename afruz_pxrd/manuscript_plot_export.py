from __future__ import annotations

"""Single-file, plot-ready refinement export helpers.

The exported long table deliberately keeps profile rows and Bragg-line rows in
one rectangular data set.  Users can filter ``record_type`` to ``PROFILE`` for
observed/calculated/background/difference curves and to ``BRAGG`` for phase-
resolved reflection markers.  Display-only offset coordinates are included in
separate columns and never replace the scientific raw arrays.
"""

from collections.abc import Mapping, Sequence
import math
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


MANUSCRIPT_COLUMN_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("record_type", "Record type", "", "PROFILE for numerical curves or BRAGG for reflection-marker rows."),
    ("point_index", "Point index", "", "Sequential measured-profile point number. Blank for Bragg rows."),
    ("two_theta_deg", "2θ", "°", "Profile coordinate or Bragg reflection position."),
    ("observed_intensity", "Observed intensity", "counts or input units", "Measured intensity used by refinement."),
    ("calculated_intensity", "Calculated intensity", "observed units", "Calculated full profile in observed intensity units."),
    ("background_intensity", "Background", "observed units", "Calculated or estimated background profile."),
    ("difference_observed_minus_calculated", "Difference: observed − calculated", "observed units", "Unshifted point-wise residual."),
    ("difference_plot_intensity", "Difference plot intensity", "observed units", "Display-only vertically shifted residual for manuscript plotting."),
    ("phase_tick_row", "Bragg phase row", "", "One-based vertical Bragg-marker row for the phase."),
    ("phase_index", "Phase index", "", "Stable phase index from the refinement result."),
    ("phase_name", "Phase", "", "Phase or material name."),
    ("formula", "Formula", "", "Chemical formula when available."),
    ("h", "h", "", "Miller index h."),
    ("k", "k", "", "Miller index k."),
    ("l", "l", "", "Miller index l."),
    ("hkl_label", "Reflection", "hkl", "Reflection label."),
    ("d_spacing_angstrom", "d spacing", "Å", "Interplanar spacing."),
    ("bragg_relative_intensity_percent", "Bragg relative intensity", "% within phase", "Reflection intensity normalized to the strongest exported reflection of the same phase when needed."),
    ("bragg_tick_y_start", "Bragg tick y start", "plot units", "Suggested display-only lower coordinate for the vertical Bragg tick."),
    ("bragg_tick_y_end", "Bragg tick y end", "plot units", "Suggested display-only upper coordinate for the vertical Bragg tick."),
    ("included", "Included", "yes/no", "Whether the reflection/profile point is included when that status is recorded."),
    ("rwp_percent", "Rwp", "%", "Weighted profile residual."),
    ("rp_percent", "Rp", "%", "Unweighted profile residual."),
    ("rexp_percent", "Rexp", "%", "Expected profile residual when statistical weights are valid."),
    ("rwp_over_rexp_ratio_gof", "Rwp / Rexp ratio (GoF)", "", "Goodness of fit, Rwp divided by Rexp."),
    ("reduced_chi_square", "Reduced χ²", "", "Square of the Rwp/Rexp ratio when statistically valid."),
    ("statistics_valid", "Statistics valid", "yes/no", "Whether Rexp, GoF, and reduced χ² are statistically meaningful."),
)


def _plain(value: Any) -> Any:
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def _sequence(result: Mapping[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        value = result.get(key)
        if isinstance(value, (list, tuple)) or (np is not None and isinstance(value, np.ndarray)):
            return list(_plain(value))
    return []


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _hkl_label(reflection: Mapping[str, Any]) -> str | None:
    existing = reflection.get("hkl_label") or reflection.get("hkl")
    if existing not in (None, ""):
        return str(existing)
    h, k, l = reflection.get("h"), reflection.get("k"), reflection.get("l")
    if h is None or k is None or l is None:
        return None
    return f"({h} {k} {l})"


def _phase_key(value: Any, fallback: int) -> str:
    return str(value if value not in (None, "") else fallback)


def build_manuscript_plot_data(result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one long, rectangular manuscript plotting table and metadata."""
    x = _sequence(result, "observed_x", "x", "two_theta", "two_theta_deg")
    observed = _sequence(result, "observed_y", "y_observed", "observed_intensity")
    calculated = _sequence(result, "calculated_y", "y_calculated", "calculated_intensity")
    background = _sequence(result, "background_y", "y_background", "background_intensity")
    difference = _sequence(result, "difference_y", "residual_y", "difference")

    required = {"2theta": x, "observed": observed, "calculated": calculated}
    missing = [name for name, values in required.items() if not values]
    if missing:
        return [], {"status": f"Unavailable: missing {', '.join(missing)} profile arrays."}

    lengths = {"2theta": len(x), "observed": len(observed), "calculated": len(calculated)}
    if background:
        lengths["background"] = len(background)
    if difference:
        lengths["difference"] = len(difference)
    if len(set(lengths.values())) != 1:
        detail = ", ".join(f"{key}={value}" for key, value in lengths.items())
        raise ValueError(f"Cannot create manuscript plot data from misaligned arrays: {detail}")

    n = len(x)
    if not background:
        background = [None] * n
    if not difference:
        difference = [
            (_finite_float(observed[i]) - _finite_float(calculated[i]))
            if _finite_float(observed[i]) is not None and _finite_float(calculated[i]) is not None
            else None
            for i in range(n)
        ]

    observed_numbers = [v for v in (_finite_float(value) for value in observed) if v is not None]
    calculated_numbers = [v for v in (_finite_float(value) for value in calculated) if v is not None]
    background_numbers = [v for v in (_finite_float(value) for value in background) if v is not None]
    difference_numbers = [v for v in (_finite_float(value) for value in difference) if v is not None]
    profile_values = observed_numbers + calculated_numbers + background_numbers
    profile_max = max(profile_values) if profile_values else 1.0
    profile_min = min(profile_values) if profile_values else 0.0
    profile_span = max(profile_max - profile_min, abs(profile_max), 1.0)
    difference_max = max(difference_numbers) if difference_numbers else 0.0
    difference_min = min(difference_numbers) if difference_numbers else 0.0

    # Display-only coordinates. These values are reproducible conveniences for
    # plotting and never alter observed/calculated/refined scientific arrays.
    difference_target_top = profile_min - 0.10 * profile_span
    difference_plot_offset = difference_target_top - difference_max
    difference_plot = [
        (_finite_float(value) + difference_plot_offset) if _finite_float(value) is not None else None
        for value in difference
    ]
    shifted_difference_numbers = [v for v in (_finite_float(value) for value in difference_plot) if v is not None]
    shifted_difference_min = min(shifted_difference_numbers) if shifted_difference_numbers else difference_target_top
    bragg_tick_height = max(0.025 * profile_span, 1e-12)
    bragg_row_spacing = max(0.045 * profile_span, bragg_tick_height * 1.5)
    first_bragg_row_y = shifted_difference_min - 0.07 * profile_span

    rwp = _finite_float(result.get("rwp_percent"))
    rp = _finite_float(result.get("rp_percent"))
    rexp = _finite_float(result.get("rexp_percent"))
    gof = _finite_float(result.get("goodness_of_fit_sqrt"))
    if gof is None and rwp is not None and rexp not in (None, 0.0):
        gof = rwp / rexp
    reduced_chi_square = _finite_float(result.get("reduced_chi_square"))
    if reduced_chi_square is None and gof is not None:
        reduced_chi_square = gof * gof
    statistics_valid = result.get("statistics_valid")

    common_stats = {
        "rwp_percent": rwp,
        "rp_percent": rp,
        "rexp_percent": rexp,
        "rwp_over_rexp_ratio_gof": gof,
        "reduced_chi_square": reduced_chi_square,
        "statistics_valid": statistics_valid,
    }

    rows: list[dict[str, Any]] = []
    point_inclusion = _sequence(result, "included_mask", "profile_included", "mask")
    for index in range(n):
        included = point_inclusion[index] if len(point_inclusion) == n else True
        rows.append({
            "record_type": "PROFILE",
            "point_index": index + 1,
            "two_theta_deg": x[index],
            "observed_intensity": observed[index],
            "calculated_intensity": calculated[index],
            "background_intensity": background[index],
            "difference_observed_minus_calculated": difference[index],
            "difference_plot_intensity": difference_plot[index],
            "phase_tick_row": None,
            "phase_index": None,
            "phase_name": None,
            "formula": None,
            "h": None,
            "k": None,
            "l": None,
            "hkl_label": None,
            "d_spacing_angstrom": None,
            "bragg_relative_intensity_percent": None,
            "bragg_tick_y_start": None,
            "bragg_tick_y_end": None,
            "included": included,
            **common_stats,
        })

    phases = [dict(row) for row in result.get("phases", []) if isinstance(row, Mapping)]
    phase_lookup: dict[str, dict[str, Any]] = {}
    phase_order: list[str] = []
    for index, phase in enumerate(phases, start=1):
        key = _phase_key(phase.get("phase_index"), index)
        phase_lookup[key] = phase
        if key not in phase_order:
            phase_order.append(key)

    reflections = [dict(row) for row in result.get("reflections", []) if isinstance(row, Mapping)]
    for index, reflection in enumerate(reflections, start=1):
        key = _phase_key(reflection.get("phase_index"), 1)
        if key not in phase_order:
            phase_order.append(key)
        phase_lookup.setdefault(key, {})

    phase_row = {key: index + 1 for index, key in enumerate(phase_order)}
    intensity_by_phase: dict[str, list[float]] = {}
    for reflection in reflections:
        key = _phase_key(reflection.get("phase_index"), 1)
        candidate = (
            reflection.get("relative_intensity_percent")
            if reflection.get("relative_intensity_percent") is not None
            else reflection.get("structure_intensity")
            if reflection.get("structure_intensity") is not None
            else reflection.get("reference_intensity")
        )
        value = _finite_float(candidate)
        if value is not None and value >= 0:
            intensity_by_phase.setdefault(key, []).append(value)
    max_intensity = {key: max(values) for key, values in intensity_by_phase.items() if values and max(values) > 0}

    for reflection in reflections:
        key = _phase_key(reflection.get("phase_index"), 1)
        phase = phase_lookup.get(key, {})
        row_number = phase_row.get(key, 1)
        tick_y_start = first_bragg_row_y - (row_number - 1) * bragg_row_spacing
        tick_y_end = tick_y_start + bragg_tick_height
        raw_intensity = (
            reflection.get("relative_intensity_percent")
            if reflection.get("relative_intensity_percent") is not None
            else reflection.get("structure_intensity")
            if reflection.get("structure_intensity") is not None
            else reflection.get("reference_intensity")
        )
        relative_value = _finite_float(raw_intensity)
        if relative_value is not None:
            # Preserve an explicit percentage. Otherwise normalize the arbitrary
            # structure/reference intensity within its phase.
            if reflection.get("relative_intensity_percent") is None:
                denom = max_intensity.get(key)
                relative_value = 100.0 * relative_value / denom if denom else None
        rows.append({
            "record_type": "BRAGG",
            "point_index": None,
            "two_theta_deg": reflection.get("two_theta_deg") or reflection.get("two_theta") or reflection.get("position"),
            "observed_intensity": None,
            "calculated_intensity": None,
            "background_intensity": None,
            "difference_observed_minus_calculated": None,
            "difference_plot_intensity": None,
            "phase_tick_row": row_number,
            "phase_index": reflection.get("phase_index"),
            "phase_name": reflection.get("phase_name") or phase.get("phase_name") or phase.get("name"),
            "formula": reflection.get("formula") or phase.get("formula"),
            "h": reflection.get("h"),
            "k": reflection.get("k"),
            "l": reflection.get("l"),
            "hkl_label": _hkl_label(reflection),
            "d_spacing_angstrom": reflection.get("d_spacing_angstrom") or reflection.get("d_spacing"),
            "bragg_relative_intensity_percent": relative_value,
            "bragg_tick_y_start": tick_y_start,
            "bragg_tick_y_end": tick_y_end,
            "included": reflection.get("included", True),
            **common_stats,
        })

    metadata = {
        "status": "Complete",
        "file_purpose": "Single long-format table for manuscript observed/calculated/background/difference curves and phase-resolved Bragg lines.",
        "profile_filter": "record_type = PROFILE",
        "bragg_filter": "record_type = BRAGG",
        "difference_definition": "observed_intensity - calculated_intensity",
        "difference_plot_offset": difference_plot_offset,
        "bragg_tick_height": bragg_tick_height,
        "bragg_phase_row_spacing": bragg_row_spacing,
        "plot_coordinate_warning": "difference_plot_intensity and bragg_tick_y_start/end are display-only suggested starting points requiring optimization; scientific profile arrays are unchanged.",
        "rwp_percent": rwp,
        "rp_percent": rp,
        "rexp_percent": rexp,
        "rwp_over_rexp_ratio_gof": gof,
        "reduced_chi_square": reduced_chi_square,
        "statistics_valid": statistics_valid,
        "statistics_reason": result.get("statistics_reason"),
        "profile_row_count": n,
        "bragg_row_count": len(reflections),
        "phase_count": len(phase_order),
    }
    return rows, metadata
