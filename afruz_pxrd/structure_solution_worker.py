from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Signal, Slot

from .structure_solution import (
    StructureSolutionCancelled,
    evaluate_candidate_robustness,
)


class CandidateRobustnessWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        candidate,
        peaks,
        settings,
        parent=None,
    ):
        super().__init__(parent)
        self.candidate = candidate
        self.peaks = peaks
        self.settings = settings
        self._cancel_event = threading.Event()

    def request_cancel(self):
        self._cancel_event.set()

    @Slot()
    def run(self):
        try:
            result = evaluate_candidate_robustness(
                self.candidate,
                self.peaks,
                progress_callback=self.progress.emit,
                cancel_check=self._cancel_event.is_set,
                **self.settings,
            )
        except StructureSolutionCancelled:
            self.cancelled.emit()
            return
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(result)
