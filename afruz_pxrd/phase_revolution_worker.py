from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Signal, Slot

from .native_indexing import NativeIndexingCancelled, run_native_indexing


class PhaseRevolutionIndexingWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, work_directory, peaks, settings, parent=None):
        super().__init__(parent)
        self.work_directory = work_directory
        self.peaks = peaks
        self.settings = settings
        self._cancel_event = threading.Event()

    def request_cancel(self):
        self._cancel_event.set()

    @Slot()
    def run(self):
        if self._cancel_event.is_set():
            self.cancelled.emit()
            return
        try:
            result = run_native_indexing(
                self.work_directory,
                self.peaks,
                progress_callback=self.progress.emit,
                cancel_check=self._cancel_event.is_set,
                **self.settings,
            )
        except NativeIndexingCancelled:
            self.cancelled.emit()
            return
        except Exception:
            if self._cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(traceback.format_exc())
            return
        if self._cancel_event.is_set():
            self.cancelled.emit()
            return
        self.finished.emit(result)
