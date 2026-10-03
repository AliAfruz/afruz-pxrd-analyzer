from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import csv
import hashlib
import json
import math

import numpy as np


WHITE_TEMPLATE_NAME = "White article template"
DEFAULT_EXPORT_DPI = 1200


@dataclass
class ArticlePlotStyle:
    """Publication-plot controls for article-ready XRD figures.

    The exported figure and plot area are intentionally locked to a white
    template, independent of the GUI theme.  The surrounding GUI controls may
    be dark, but the saved figure stays journal-friendly and reproducible.
    """

    title: str = "PXRD refinement profile"
    x_label: str = "2θ (degrees)"
    y_label: str = "Intensity (a.u.)"
    font_family: str = "DejaVu Sans"
    title_font_size: float = 12.0
    axis_font_size: float = 10.0
    tick_font_size: float = 8.0
    legend_font_size: float = 8.0
    observed_color: str = "#111111"
    calculated_color: str = "#b8860b"
    difference_color: str = "#1f77b4"
    background_color: str = "#777777"
    bragg_tick_color: str = "#111111"
    grid_color: str = "#d8d8d8"
    observed_line_width: float = 0.75
    calculated_line_width: float = 1.25
    difference_line_width: float = 0.85
    background_line_width: float = 0.85
    bragg_tick_line_width: float = 0.7
    figure_width_in: float = 7.2
    figure_height_in: float = 4.8
    show_background: bool = True
    show_difference: bool = True
    show_legend: bool = True
    show_grid: bool = True
    difference_offset_fraction: float = 0.18
    tick_row_gap_fraction: float = 0.055
    tick_height_fraction: float = 0.032
    max_bragg_ticks_per_phase: int = 220


@dataclass
class ArticlePlotExportSettings:
    dpi: int = DEFAULT_EXPORT_DPI
    formats: tuple[str, ...] = ("png", "pdf", "svg")
    transparent: bool = False
    write_profile_csv: bool = True
    write_bragg_tick_csv: bool = True


def _as_array(values: Iterable[float] | np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return arr[np.isfinite(arr)] if arr.ndim == 1 else np.asarray([], dtype=float)


def _safe_same_length(values: Iterable[float] | np.ndarray | None, length: int) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=float)
    if arr.ndim != 1 or len(arr) != length or not np.all(np.isfinite(arr)):
        return np.zeros(length, dtype=float)
    return arr


def _hex_or_default(value: str, default: str) -> str:
    text = str(value or "").strip()
    if len(text) == 7 and text.startswith("#"):
        try:
            int(text[1:], 16)
            return text
        except ValueError:
            return default
    return default


def normalize_article_style(style: ArticlePlotStyle | Mapping | None = None) -> ArticlePlotStyle:
    if style is None:
        return ArticlePlotStyle()
    if isinstance(style, ArticlePlotStyle):
        data = asdict(style)
    elif isinstance(style, Mapping):
        data = dict(style)
    else:
        data = {}
    defaults = asdict(ArticlePlotStyle())
    merged = {**defaults, **{k: v for k, v in data.items() if k in defaults}}
    for key in [
        "observed_color",
        "calculated_color",
        "difference_color",
        "background_color",
        "bragg_tick_color",
        "grid_color",
    ]:
        merged[key] = _hex_or_default(merged.get(key), defaults[key])
    for key in [
        "title_font_size",
        "axis_font_size",
        "tick_font_size",
        "legend_font_size",
        "observed_line_width",
        "calculated_line_width",
        "difference_line_width",
        "background_line_width",
        "bragg_tick_line_width",
        "figure_width_in",
        "figure_height_in",
    ]:
        try:
            merged[key] = float(merged[key])
        except (TypeError, ValueError):
            merged[key] = float(defaults[key])
        merged[key] = max(0.1, merged[key])
    try:
        merged["max_bragg_ticks_per_phase"] = max(1, int(merged["max_bragg_ticks_per_phase"]))
    except (TypeError, ValueError):
        merged["max_bragg_ticks_per_phase"] = defaults["max_bragg_ticks_per_phase"]
    return ArticlePlotStyle(**merged)


def normalize_export_settings(settings: ArticlePlotExportSettings | Mapping | None = None) -> ArticlePlotExportSettings:
    if settings is None:
        return ArticlePlotExportSettings()
    if isinstance(settings, ArticlePlotExportSettings):
        data = asdict(settings)
    elif isinstance(settings, Mapping):
        data = dict(settings)
    else:
        data = {}
    dpi = int(data.get("dpi", DEFAULT_EXPORT_DPI) or DEFAULT_EXPORT_DPI)
    # User requirement: article exports must be 1200 DPI.  Higher values are
    # allowed, lower values are promoted to 1200.
    dpi = max(DEFAULT_EXPORT_DPI, dpi)
    formats = data.get("formats", ("png", "pdf", "svg")) or ("png",)
    formats = tuple(
        fmt.lower().lstrip(".")
        for fmt in formats
        if str(fmt).lower().lstrip(".") in {"png", "pdf", "svg", "tif", "tiff"}
    ) or ("png",)
    return ArticlePlotExportSettings(
        dpi=dpi,
        formats=formats,
        transparent=bool(data.get("transparent", False)),
        write_profile_csv=bool(data.get("write_profile_csv", True)),
        write_bragg_tick_csv=bool(data.get("write_bragg_tick_csv", True)),
    )


def profile_from_multicomponent_result(result: Mapping | None) -> dict | None:
    if not isinstance(result, Mapping):
        return None
    profile = result.get("best_refinement_profile")
    if isinstance(profile, Mapping):
        return dict(profile)
    best = result.get("best_model") or {}
    best_id = best.get("model_id")
    for row in result.get("full_refinement_results", []) or []:
        if isinstance(row, Mapping) and row.get("model_id") == best_id:
            refinement = row.get("refinement")
            if isinstance(refinement, Mapping):
                return dict(refinement)
    return None


def bragg_ticks_from_profile(profile: Mapping | None) -> list[dict]:
    if not isinstance(profile, Mapping):
        return []
    phase_names = {
        int(phase.get("phase_index")): str(phase.get("phase_name") or f"Phase {phase.get('phase_index')}")
        for phase in profile.get("phases", []) or []
        if isinstance(phase, Mapping) and phase.get("phase_index") is not None
    }
    rows: list[dict] = []
    for reflection in profile.get("reflections", []) or []:
        if not isinstance(reflection, Mapping):
            continue
        try:
            phase_index = int(reflection.get("phase_index"))
            two_theta = float(reflection.get("two_theta_deg"))
        except (TypeError, ValueError):
            continue
        hkl = reflection.get("hkl") or reflection.get("hkl_label") or ""
        if isinstance(hkl, (list, tuple)):
            hkl_label = " ".join(str(int(v)) for v in hkl)
        else:
            hkl_label = str(hkl)
        rows.append(
            {
                "phase_index": phase_index,
                "phase_name": phase_names.get(phase_index, f"Phase {phase_index}"),
                "two_theta_deg": two_theta,
                "hkl": hkl_label,
                "relative_intensity": float(reflection.get("relative_intensity", reflection.get("intensity", 0.0)) or 0.0),
            }
        )
    rows.sort(key=lambda row: (row["phase_index"], row["two_theta_deg"]))
    return rows


def profile_arrays(profile: Mapping) -> dict[str, np.ndarray]:
    x = np.asarray(profile.get("observed_x", []), dtype=float)
    observed = np.asarray(profile.get("observed_y", []), dtype=float)
    if x.ndim != 1 or observed.ndim != 1 or len(x) != len(observed) or len(x) < 3:
        raise ValueError("Profile must contain same-length observed_x and observed_y arrays with at least three points.")
    mask = np.isfinite(x) & np.isfinite(observed)
    x = x[mask]
    observed = observed[mask]
    calculated = _safe_same_length(profile.get("calculated_y"), len(mask))[mask]
    background = _safe_same_length(profile.get("background_y"), len(mask))[mask]
    difference = _safe_same_length(profile.get("difference_y"), len(mask))[mask]
    if not np.any(calculated):
        calculated = observed.copy()
        difference = np.zeros_like(observed)
    elif not np.any(difference):
        difference = observed - calculated
    return {
        "x": x,
        "observed": observed,
        "calculated": calculated,
        "background": background,
        "difference": difference,
    }


def _write_csv(path: Path, rows: Sequence[Mapping], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def render_article_refinement_plot(
    profile: Mapping,
    bragg_ticks: Sequence[Mapping] | None,
    output_path: str | Path,
    style: ArticlePlotStyle | Mapping | None = None,
    settings: ArticlePlotExportSettings | Mapping | None = None,
) -> dict:
    """Render one publication-ready refinement plot.

    The plot canvas and exported file are always white.  Default export DPI is
    1200, satisfying high-resolution article/cover/figure requirements.
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    st = normalize_article_style(style)
    ex = normalize_export_settings(settings)
    arrays = profile_arrays(profile)
    ticks = list(bragg_ticks if bragg_ticks is not None else bragg_ticks_from_profile(profile))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": st.font_family,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(st.figure_width_in, st.figure_height_in), facecolor="white")
    ax.set_facecolor("white")
    x = arrays["x"]
    observed = arrays["observed"]
    calculated = arrays["calculated"]
    background = arrays["background"]
    difference = arrays["difference"]
    span = max(float(np.nanmax(observed) - np.nanmin(observed)), 1.0)
    lower = float(np.nanmin(observed))
    difference_offset = lower - span * float(st.difference_offset_fraction)
    tick_gap = span * float(st.tick_row_gap_fraction)
    tick_height = span * float(st.tick_height_fraction)

    ax.plot(x, observed, color=st.observed_color, lw=st.observed_line_width, label="Observed")
    ax.plot(x, calculated, color=st.calculated_color, lw=st.calculated_line_width, label="Calculated")
    if st.show_background and np.any(background):
        ax.plot(x, background, color=st.background_color, lw=st.background_line_width, ls="--", label="Background")
    if st.show_difference:
        ax.plot(x, difference + difference_offset, color=st.difference_color, lw=st.difference_line_width, label="Obs − Calc")
        ax.axhline(difference_offset, color=st.difference_color, lw=0.45, alpha=0.55)

    phase_order = []
    by_phase: dict[int, list[Mapping]] = {}
    for tick in ticks:
        try:
            phase_index = int(tick.get("phase_index", 0))
            theta = float(tick.get("two_theta_deg"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(theta):
            continue
        if phase_index not in by_phase:
            phase_order.append(phase_index)
            by_phase[phase_index] = []
        by_phase[phase_index].append(tick)
    phase_names: dict[int, str] = {}
    for idx, phase_index in enumerate(phase_order):
        row_y = difference_offset - (idx + 1) * tick_gap
        row_ticks = sorted(by_phase[phase_index], key=lambda row: float(row.get("two_theta_deg", 0.0)))[: st.max_bragg_ticks_per_phase]
        if row_ticks:
            phase_names[phase_index] = str(row_ticks[0].get("phase_name") or f"Phase {phase_index}")
        for tick in row_ticks:
            theta = float(tick.get("two_theta_deg"))
            ax.vlines(theta, row_y, row_y - tick_height, colors=st.bragg_tick_color, lw=st.bragg_tick_line_width)
        if row_ticks:
            ax.text(
                float(np.nanmin(x)),
                row_y - tick_height * 0.55,
                phase_names.get(phase_index, f"Phase {phase_index}"),
                fontsize=st.tick_font_size,
                ha="left",
                va="center",
                color=st.bragg_tick_color,
            )
    ax.set_title(st.title, fontsize=st.title_font_size)
    ax.set_xlabel(st.x_label, fontsize=st.axis_font_size)
    ax.set_ylabel(st.y_label, fontsize=st.axis_font_size)
    ax.tick_params(axis="both", labelsize=st.tick_font_size, colors="#111111")
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(0.6)
    if st.show_grid:
        ax.grid(True, color=st.grid_color, alpha=0.6, lw=0.35)
    if st.show_legend:
        ax.legend(frameon=False, fontsize=st.legend_font_size, loc="best")
    ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    if phase_order:
        ymin = difference_offset - (len(phase_order) + 1) * tick_gap
        ymax = float(np.nanmax(observed)) + span * 0.05
        ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(out, dpi=ex.dpi, transparent=False, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {
        "success": True,
        "path": str(out),
        "dpi": ex.dpi,
        "template": WHITE_TEMPLATE_NAME,
        "background": "white",
        "format": out.suffix.lower().lstrip("."),
        "point_count": int(len(x)),
        "bragg_tick_count": int(sum(len(v[: st.max_bragg_ticks_per_phase]) for v in by_phase.values())),
        "phase_tick_rows": len(phase_order),
        "style": asdict(st),
    }


def export_article_plot_package(
    output_dir: str | Path,
    profile: Mapping,
    bragg_ticks: Sequence[Mapping] | None = None,
    style: ArticlePlotStyle | Mapping | None = None,
    settings: ArticlePlotExportSettings | Mapping | None = None,
) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    st = normalize_article_style(style)
    ex = normalize_export_settings(settings)
    ticks = list(bragg_ticks if bragg_ticks is not None else bragg_ticks_from_profile(profile))
    written: list[Path] = []
    exports: list[dict] = []
    for fmt in ex.formats:
        extension = "tif" if fmt == "tiff" else fmt
        plot_path = root / f"afruz_article_refinement_plot.{extension}"
        info = render_article_refinement_plot(profile, ticks, plot_path, st, ex)
        exports.append(info)
        written.append(plot_path)
    arrays = profile_arrays(profile)
    if ex.write_profile_csv:
        profile_rows = []
        for i, (x, obs, calc, bkg, diff) in enumerate(
            zip(arrays["x"], arrays["observed"], arrays["calculated"], arrays["background"], arrays["difference"]),
            start=1,
        ):
            profile_rows.append(
                {
                    "index": i,
                    "two_theta_deg": float(x),
                    "observed": float(obs),
                    "calculated": float(calc),
                    "background": float(bkg),
                    "obs_minus_calc": float(diff),
                }
            )
        profile_csv = root / "afruz_article_profile_data.csv"
        _write_csv(profile_csv, profile_rows, ["index", "two_theta_deg", "observed", "calculated", "background", "obs_minus_calc"])
        written.append(profile_csv)
    if ex.write_bragg_tick_csv:
        tick_csv = root / "afruz_article_bragg_ticks.csv"
        _write_csv(tick_csv, ticks, ["phase_index", "phase_name", "two_theta_deg", "hkl", "relative_intensity"])
        written.append(tick_csv)
    style_json = root / "afruz_article_plot_style.json"
    style_json.write_text(
        json.dumps(
            {
                "phase": "19.4",
                "engine": "Article Plotting Engine",
                "template": WHITE_TEMPLATE_NAME,
                "export_dpi": ex.dpi,
                "style": asdict(st),
                "settings": asdict(ex),
                "scientific_boundary": (
                    "Article plots are visual summaries of accepted/calculated profiles. They do not validate phase identity by themselves."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    written.append(style_json)
    manifest_rows = []
    for path in written:
        manifest_rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = root / "afruz_article_plot_manifest.json"
    manifest.write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(manifest)
    return {
        "success": True,
        "phase": "19.4",
        "engine": "Article Plotting Engine",
        "template": WHITE_TEMPLATE_NAME,
        "dpi": ex.dpi,
        "exports": exports,
        "files": [str(path) for path in written],
        "manifest": str(manifest),
        "scientific_boundary": (
            "The central plot canvas and exported figures are white by design; surrounding GUI controls may follow the dark Afruz theme."
        ),
    }
