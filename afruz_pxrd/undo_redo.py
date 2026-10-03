from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import copy


@dataclass(frozen=True)
class HistoryEntry:
    """One reversible application-state checkpoint."""

    label: str
    snapshot: dict[str, Any]


class UndoRedoHistory:
    """Bounded undo/redo history for scientific GUI state.

    The manager is intentionally independent of Qt.  The application supplies
    snapshot and restore callbacks so the same class can be tested without a
    desktop session and can later be reused by non-GUI batch editors.
    """

    def __init__(
        self,
        *,
        max_depth: int = 40,
        snapshot_factory: Callable[[], dict[str, Any]] | None = None,
        restore_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        self.max_depth = int(max_depth)
        self.snapshot_factory = snapshot_factory
        self.restore_callback = restore_callback
        self._undo_stack: list[HistoryEntry] = []
        self._redo_stack: list[HistoryEntry] = []
        self._restoring = False
        self.last_action_label = ""

    @property
    def is_restoring(self) -> bool:
        return self._restoring

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    @property
    def undo_label(self) -> str:
        return self._undo_stack[-1].label if self._undo_stack else ""

    @property
    def redo_label(self) -> str:
        return self._redo_stack[-1].label if self._redo_stack else ""

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.last_action_label = ""

    def checkpoint(
        self,
        label: str,
        snapshot: dict[str, Any] | None = None,
    ) -> bool:
        """Push the current state before a user-visible mutation.

        Calling checkpoint while restoring is ignored.  Creating a new edit
        after undo clears the redo stack, matching standard desktop behavior.
        """
        if self._restoring:
            return False
        if snapshot is None:
            if self.snapshot_factory is None:
                raise RuntimeError("No snapshot factory is configured")
            snapshot = self.snapshot_factory()
        entry = HistoryEntry(str(label or "Change"), copy.deepcopy(snapshot))
        self._undo_stack.append(entry)
        if len(self._undo_stack) > self.max_depth:
            self._undo_stack = self._undo_stack[-self.max_depth :]
        self._redo_stack.clear()
        return True

    def undo(self) -> str | None:
        if not self.can_undo:
            return None
        if self.snapshot_factory is None or self.restore_callback is None:
            raise RuntimeError("Undo/redo callbacks are not configured")
        current = copy.deepcopy(self.snapshot_factory())
        entry = self._undo_stack.pop()
        self._redo_stack.append(HistoryEntry(entry.label, current))
        self._restoring = True
        try:
            self.restore_callback(copy.deepcopy(entry.snapshot))
        finally:
            self._restoring = False
        self.last_action_label = entry.label
        return entry.label

    def redo(self) -> str | None:
        if not self.can_redo:
            return None
        if self.snapshot_factory is None or self.restore_callback is None:
            raise RuntimeError("Undo/redo callbacks are not configured")
        current = copy.deepcopy(self.snapshot_factory())
        entry = self._redo_stack.pop()
        self._undo_stack.append(HistoryEntry(entry.label, current))
        self._restoring = True
        try:
            self.restore_callback(copy.deepcopy(entry.snapshot))
        finally:
            self._restoring = False
        self.last_action_label = entry.label
        return entry.label

    def to_status(self) -> dict[str, Any]:
        return {
            "can_undo": self.can_undo,
            "can_redo": self.can_redo,
            "undo_label": self.undo_label,
            "redo_label": self.redo_label,
            "depth": len(self._undo_stack),
            "redo_depth": len(self._redo_stack),
            "max_depth": self.max_depth,
        }
