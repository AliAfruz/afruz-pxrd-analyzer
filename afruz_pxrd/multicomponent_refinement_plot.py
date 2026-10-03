from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import numpy as np

try:  # GUI dependencies are optional during headless scientific tests.
    import pyqtgraph as pg
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget
    from .context_copy import CopyablePlotWidget, install_label_copy_menu
except Exception:  # pragma: no cover - exercised only when GUI stack is absent.
    pg = None

    class _QtFallback:
        SolidLine = 1
        DashLine = 2
        DotLine = 3
        DashDotLine = 4
        DashDotDotLine = 5

    Qt = _QtFallback()

    def Signal(*_args, **_kwargs):
        return None

    class QWidget:  # minimal fallback so pure helper imports remain possible.
        pass

    QLabel = QSizePolicy = QVBoxLayout = None
    CopyablePlotWidget = None

    def install_label_copy_menu(*_args, **_kwargs):
        return None

from .theme import DEFAULT_THEME_NAME, THEMES


PHASE_TICK_PEN_STYLES = [
    Qt.SolidLine,
    Qt.DashLine,
    Qt.DotLine,
    Qt.DashDotLine,
    Qt.DashDotDotLine,
]


def best_multicomponent_refinement_profile(result: Mapping | None) -> dict | None:
    """Return the full Rietveld-style profile for the parsimony-ranked best model.

    Phase 19 stores compact ranked-model rows separately from the heavier full
    refinement arrays.  This helper resolves the best model back to its profile
    data so plots and exports can draw observed/calculated/difference curves and
    Bragg reflection tick marks.
    """

    if not isinstance(result, Mapping):
        return None
    profile = result.get("best_refinement_profile")
    if isinstance(profile, Mapping):
        return dict(profile)
    best = result.get("best_model") or {}
    best_id = best.get("model_id")
    if not best_id:
        return None
    for row in result.get("full_refinement_results", []) or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("model_id") == best_id and isinstance(row.get("refinement"), Mapping):
            return dict(row["refinement"])
    return None


def bragg_tick_rows_from_profile(profile: Mapping | None) -> list[dict]:
    """Build a compact phase-by-phase Bragg tick table from a refinement profile."""

    if not isinstance(profile, Mapping):
        return []
    phase_lookup = {
        int(phase.get("phase_index")): str(phase.get("phase_name") or f"Phase {phase.get('phase_index')}")
        for phase in profile.get("phases", []) or []
        if isinstance(phase, Mapping) and phase.get("phase_index") is not None
    }
    rows = []
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
                "phase_name": phase_lookup.get(phase_index, f"Phase {phase_index}"),
                "two_theta_deg": two_theta,
                "hkl": hkl_label,
                "relative_intensity": float(reflection.get("relative_intensity", reflection.get("intensity", 0.0)) or 0.0),
            }
        )
    rows.sort(key=lambda row: (row["phase_index"], row["two_theta_deg"]))
    return rows


class MultiComponentRefinementPlotWidget(QWidget):
    """Observed/calculated/difference plot with phase-by-phase Bragg ticks."""

    peakAddRequested = Signal(float, float)

    def __init__(self, parent=None):
        if pg is None or CopyablePlotWidget is None:
            raise RuntimeError("PySide6 and pyqtgraph are required for the GUI refinement plot widget.")
        super().__init__(parent)
        self._theme_name = DEFAULT_THEME_NAME
        theme = THEMES[DEFAULT_THEME_NAME]
        self.plot = CopyablePlotWidget(
            background=theme["plot_background"],
            copy_title="Phase 19 multiphase refinement profile",
        )
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.plot.setLabel("bottom", "2θ", units="degrees")
        self.plot.setLabel("left", "Intensity / offset rows")
        self.plot.enable_manual_peak_context(True, label="Add manual peak to Peak List here")
        self.plot.manualPeakRequested.connect(
            lambda x, y: self.peakAddRequested.emit(float(x), float(y))
        )
        self.message = QLabel(
            "Run Intelligent Multiphase to display observed/calculated/difference curves and Bragg tick marks."
        )
        self.message.setObjectName("mutedLabel")
        self.message.setWordWrap(False)
        self.message.setMinimumWidth(0)
        self.message.setMaximumHeight(24)
        self.message.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        install_label_copy_menu(self.message)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.message)

    def apply_theme(self, theme_name: str):
        self._theme_name = theme_name
        theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME_NAME])
        self.plot.setBackground(theme["plot_background"])
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme["plot_axis"]))
        self.plot.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def clear(self):
        self.plot.clear()
        self.message.setText(
            "Run Intelligent Multiphase to display observed/calculated/difference curves and Bragg tick marks."
        )

    def set_result(self, result: Mapping | None):
        self.plot.clear()
        profile = best_multicomponent_refinement_profile(result)
        if not profile:
            self.message.setText("No best full refinement profile is available yet.")
            return
        x = np.asarray(profile.get("observed_x", []), dtype=float)
        observed = np.asarray(profile.get("observed_y", []), dtype=float)
        calculated = np.asarray(profile.get("calculated_y", []), dtype=float)
        background = np.asarray(profile.get("background_y", []), dtype=float)
        difference = np.asarray(profile.get("difference_y", []), dtype=float)
        if not len(x) or not len(observed) or len(x) != len(observed):
            self.message.setText("The best model does not contain plottable observed intensity arrays.")
            return
        calculated = self._safe_array(calculated, len(x))
        background = self._safe_array(background, len(x))
        difference = self._safe_array(difference, len(x))
        theme = THEMES.get(self._theme_name, THEMES[DEFAULT_THEME_NAME])
        self.plot.plot(x, observed, pen=pg.mkPen(theme["plot_axis"], width=1.05), name="Observed")
        self.plot.plot(x, calculated, pen=pg.mkPen(theme["gold"], width=2.0), name="Calculated")
        if len(background):
            self.plot.plot(
                x,
                background,
                pen=pg.mkPen(theme["muted"], width=1.1, style=Qt.DashLine),
                name="Background",
            )
        span = max(float(np.nanmax(observed) - np.nanmin(observed)), 1.0)
        lower = float(np.nanmin(observed))
        difference_offset = lower - 0.18 * span
        self.plot.plot(
            x,
            difference + difference_offset,
            pen=pg.mkPen(theme["gold_bright"], width=1.0),
            name="Obs − Calc",
        )
        phases = [phase for phase in profile.get("phases", []) or [] if isinstance(phase, Mapping)]
        phase_order = OrderedDict()
        for index, phase in enumerate(phases):
            try:
                phase_index = int(phase.get("phase_index"))
            except (TypeError, ValueError):
                continue
            phase_order[phase_index] = {
                "row": index,
                "name": str(phase.get("phase_name") or f"Phase {phase_index}"),
            }
            profile_y = np.asarray(phase.get("profile_y", []), dtype=float)
            if len(profile_y) == len(x):
                profile_offset = difference_offset - span * (0.075 + index * 0.045)
                scaled = profile_y / max(float(np.nanmax(profile_y)), 1.0) * span * 0.055
                self.plot.plot(
                    x,
                    scaled + profile_offset,
                    pen=pg.mkPen(theme["muted"], width=0.8, style=PHASE_TICK_PEN_STYLES[index % len(PHASE_TICK_PEN_STYLES)]),
                    name=f"{phase_order[phase_index]['name']} profile row",
                )
        for tick in bragg_tick_rows_from_profile(profile):
            phase_index = int(tick["phase_index"])
            info = phase_order.get(phase_index)
            if info is None:
                continue
            row = int(info["row"])
            baseline = difference_offset - span * (0.11 + row * 0.045)
            top = baseline + span * 0.028
            style = PHASE_TICK_PEN_STYLES[row % len(PHASE_TICK_PEN_STYLES)]
            self.plot.plot(
                [float(tick["two_theta_deg"]), float(tick["two_theta_deg"])],
                [baseline, top],
                pen=pg.mkPen(theme["gold"], width=1.15, style=style),
            )
        self.plot.enableAutoRange()
        best = (result or {}).get("best_model") or {}
        message = (
            f"Bragg ticks drawn under difference curve for {len(phases)} phase row(s). "
            f"Best model: {best.get('model_id', 'not available')} · "
            f"Rwp {best.get('rwp_percent', profile.get('rwp_percent', 0)):.4g}% · "
            f"reflections {profile.get('reflection_count', len(profile.get('reflections', []) or []))}."
        )
        self.message.setText(message if len(message) <= 160 else message[:159] + "…")
        self.message.setToolTip(message)

    @staticmethod
    def _safe_array(values: np.ndarray, length: int) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if len(values) == length:
            return values
        return np.zeros(length, dtype=float)
