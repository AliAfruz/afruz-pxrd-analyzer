from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Signal, Slot

from .whole_pattern_refinement import (
    WholePatternCancelled,
    refine_whole_pattern,
)


class WholePatternWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        x,
        y,
        phase_specs,
        settings,
        parent=None,
    ):
        super().__init__(parent)
        self.x = x
        self.y = y
        self.phase_specs = phase_specs
        self.settings = settings
        self._cancel_event = threading.Event()

    def request_cancel(self):
        self._cancel_event.set()

    @Slot()
    def run(self):
        try:
            result = refine_whole_pattern(
                self.x,
                self.y,
                self.phase_specs,
                progress_callback=self.progress.emit,
                cancel_check=self._cancel_event.is_set,
                **self.settings,
            )
        except WholePatternCancelled:
            self.cancelled.emit()
            return
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(result)
