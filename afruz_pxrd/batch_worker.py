from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Signal, Slot

from .batch_engine import BatchCancelled, analyze_batch


class BatchAnalysisWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal(object)

    def __init__(
        self,
        datasets,
        recipe,
        references,
        reference_pattern,
        reference_structure,
        parent=None,
    ):
        super().__init__(parent)
        self.datasets = datasets
        self.recipe = recipe
        self.references = references
        self.reference_pattern = reference_pattern
        self.reference_structure = reference_structure
        self._cancel_event = threading.Event()

    def request_cancel(self):
        self._cancel_event.set()

    @Slot()
    def run(self):
        try:
            result = analyze_batch(
                self.datasets,
                self.recipe,
                references=self.references,
                reference_pattern=self.reference_pattern,
                reference_structure=self.reference_structure,
                progress_callback=self.progress.emit,
                cancel_check=self._cancel_event.is_set,
            )
        except BatchCancelled:
            self.cancelled.emit({"cancelled": True, "results": []})
            return
        except Exception:
            self.failed.emit(traceback.format_exc())
            return

        if result.get("cancelled") or self._cancel_event.is_set():
            self.cancelled.emit(result)
        else:
            self.finished.emit(result)
