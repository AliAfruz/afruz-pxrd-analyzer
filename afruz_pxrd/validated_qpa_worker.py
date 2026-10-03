from __future__ import annotations

import threading
import traceback
from PySide6.QtCore import QObject, Signal, Slot

from .gsasii_qpa_backend import run_gsasii_qpa


class ValidatedQPAWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        python_executable,
        request,
        runner_path,
        timeout_seconds=None,
        parent=None,
    ):
        super().__init__(parent)
        self.python_executable = python_executable
        self.request = request
        self.runner_path = runner_path
        self.timeout_seconds = timeout_seconds
        self._cancel = threading.Event()

    def request_cancel(self):
        self._cancel.set()

    @Slot()
    def run(self):
        try:
            result = run_gsasii_qpa(
                self.python_executable,
                self.request,
                self.runner_path,
                progress_callback=self.progress.emit,
                cancel_check=self._cancel.is_set,
                timeout_seconds=self.timeout_seconds,
            )
        except Exception as exc:
            if self._cancel.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(result)
