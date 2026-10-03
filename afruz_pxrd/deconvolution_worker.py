from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Signal, Slot

from .fitting import (
    DeconvolutionCancelled,
    advanced_deconvolve_peaks,
)


class DeconvolutionWorker(QObject):
    """Run advanced peak deconvolution outside the GUI thread."""

    progress = Signal(int, int, str)
    finished = Signal(object, object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        x,
        y,
        peaks,
        options: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.x = x
        self.y = y
        self.peaks = peaks
        self.options = dict(options)
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        """Thread-safe cancellation request callable from the GUI thread."""
        self._cancel_event.set()

    @Slot()
    def run(self):
        try:
            groups, diagnostics = advanced_deconvolve_peaks(
                self.x,
                self.y,
                self.peaks,
                progress_callback=self.progress.emit,
                cancel_check=self._cancel_event.is_set,
                **self.options,
            )
        except DeconvolutionCancelled:
            self.cancelled.emit()
            return
        except Exception:
            self.failed.emit(traceback.format_exc())
            return

        if self._cancel_event.is_set():
            self.cancelled.emit()
            return

        self.finished.emit(groups, diagnostics)
