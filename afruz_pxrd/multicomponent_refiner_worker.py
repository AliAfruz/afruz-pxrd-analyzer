from __future__ import annotations

from copy import deepcopy
import traceback
from typing import Sequence

from PySide6.QtCore import QObject, Signal, Slot

from .multicomponent_refiner import (
    CandidatePhase,
    MultiComponentRefinementError,
    MultiComponentSearchSettings,
    run_multicomponent_refinement,
    run_multicomponent_second_pass,
)


class MultiComponentRefinerWorker(QObject):
    """Background worker for the Intelligent Multiphase Refiner GUI.

    The worker keeps the expensive phase-combination search off the Qt main
    thread and forwards model-by-model progress events to the widget.  It is
    intentionally small so the scientific core remains in
    ``multicomponent_refiner.run_multicomponent_refinement``.
    """

    progress = Signal(int, str, object)
    finished = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal(str)

    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        phases: Sequence[CandidatePhase],
        settings: MultiComponentSearchSettings,
        parent=None,
    ):
        super().__init__(parent)
        self.x = [float(value) for value in x]
        self.y = [float(value) for value in y]
        self.phases = deepcopy(list(phases))
        self.settings = settings
        self._cancel_requested = False

    @Slot()
    def run(self):
        try:
            result = run_multicomponent_refinement(
                self.x,
                self.y,
                self.phases,
                settings=self.settings,
                progress_callback=self._handle_progress,
                cancel_checker=self.is_cancel_requested,
            )
        except MultiComponentRefinementError as exc:
            if self._cancel_requested or "cancelled" in str(exc).lower():
                self.cancelled.emit(str(exc))
            else:
                self.failed.emit(str(exc), traceback.format_exc())
            return
        except Exception as exc:  # GUI safety net; the core should raise cleaner errors.
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self.finished.emit(result)

    @Slot()
    def cancel(self):
        self._cancel_requested = True
        self.progress.emit(99, "Cancellation requested. Waiting for current model to finish…", {})

    def is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested)

    def _handle_progress(self, event: dict):
        percent = int(event.get("percent", 0))
        message = str(event.get("message", "Working…"))
        self.progress.emit(percent, message, dict(event))


class MultiComponentSecondPassWorker(QObject):
    """Background worker for the reviewed mismatch-window Pass 2 refinement."""

    progress = Signal(int, str, object)
    finished = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal(str)

    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        phases: Sequence[CandidatePhase],
        first_pass_result: dict,
        mismatch_review: Sequence[dict],
        settings: MultiComponentSearchSettings,
        parent=None,
    ):
        super().__init__(parent)
        self.x = [float(value) for value in x]
        self.y = [float(value) for value in y]
        self.phases = deepcopy(list(phases))
        self.first_pass_result = deepcopy(dict(first_pass_result))
        self.mismatch_review = deepcopy(list(mismatch_review))
        self.settings = settings
        self._cancel_requested = False

    @Slot()
    def run(self):
        try:
            if self._cancel_requested:
                self.cancelled.emit("Pass 2 cancelled by user.")
                return
            self.progress.emit(10, "Applying reviewed mismatch-window mask…", {})
            result = run_multicomponent_second_pass(
                self.x,
                self.y,
                self.phases,
                self.first_pass_result,
                self.mismatch_review,
                settings=self.settings,
            )
        except MultiComponentRefinementError as exc:
            if self._cancel_requested or "cancelled" in str(exc).lower():
                self.cancelled.emit(str(exc))
            else:
                self.failed.emit(str(exc), traceback.format_exc())
            return
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self.progress.emit(100, "Pass 2 refinement complete.", {})
        self.finished.emit(result)

    @Slot()
    def cancel(self):
        self._cancel_requested = True
        self.progress.emit(99, "Cancellation requested…", {})
