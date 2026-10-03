from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Signal, Slot

from .doping_series import (
    DopingSeriesCancelled,
    compare_doping_series,
    refine_doping_series,
)


class DopingSeriesWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    PROGRESS_MAXIMUM = 1000
    COMPARISON_END = 50
    REFINEMENT_END = 980
    FINALIZING = 990

    def __init__(
        self,
        records,
        comparison_settings,
        phase_specs=None,
        refinement_settings=None,
        series_refinement_settings=None,
        parent=None,
    ):
        super().__init__(parent)
        self.records = records
        self.comparison_settings = comparison_settings
        self.phase_specs = phase_specs or []
        self.refinement_settings = refinement_settings or {}
        self.series_refinement_settings = series_refinement_settings or {}
        self._cancel_event = threading.Event()

    def request_cancel(self):
        self._cancel_event.set()

    @Slot()
    def run(self):
        try:
            self.progress.emit(
                0,
                self.PROGRESS_MAXIMUM,
                "Comparing original XRD patterns",
            )
            comparison = compare_doping_series(
                self.records,
                **self.comparison_settings,
            )
            refinement = None
            if self.phase_specs:
                self.progress.emit(
                    self.COMPARISON_END,
                    self.PROGRESS_MAXIMUM,
                    "Pattern comparison complete; starting series refinements",
                )

                def report_refinement_progress(done, total, message):
                    total = max(1, int(total))
                    fraction = max(0.0, min(float(done) / total, 1.0))
                    overall_done = self.COMPARISON_END + int(
                        round(
                            fraction
                            * (self.REFINEMENT_END - self.COMPARISON_END)
                        )
                    )
                    self.progress.emit(
                        overall_done,
                        self.PROGRESS_MAXIMUM,
                        str(message),
                    )

                refinement = refine_doping_series(
                    self.records,
                    self.phase_specs,
                    self.refinement_settings,
                    progress_callback=report_refinement_progress,
                    cancel_check=self._cancel_event.is_set,
                    **self.series_refinement_settings,
                )
            if self._cancel_event.is_set():
                raise DopingSeriesCancelled("Doping-series analysis was cancelled.")
            self.progress.emit(
                self.FINALIZING,
                self.PROGRESS_MAXIMUM,
                "Finalizing result tables, plots, and project records",
            )
            self.finished.emit(
                {
                    "comparison": comparison,
                    "refinement": refinement or {},
                }
            )
        except DopingSeriesCancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())
