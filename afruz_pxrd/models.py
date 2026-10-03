from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import uuid
import numpy as np


@dataclass
class Dataset:
    name: str
    x: np.ndarray
    y_raw: np.ndarray
    source_path: str = ""
    uid: str = field(default_factory=lambda: uuid.uuid4().hex)
    visible: bool = True
    y_processed: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def y(self) -> np.ndarray:
        return self.y_processed if self.y_processed is not None else self.y_raw

    def clone(self) -> "Dataset":
        return Dataset(
            name=f"{self.name} copy",
            x=self.x.copy(),
            y_raw=self.y_raw.copy(),
            source_path=self.source_path,
            visible=self.visible,
            y_processed=None if self.y_processed is None else self.y_processed.copy(),
            metadata=dict(self.metadata),
        )

    def reset_processing(self) -> None:
        self.y_processed = None

    def validate(self) -> None:
        if self.x.ndim != 1 or self.y_raw.ndim != 1:
            raise ValueError("X and Y arrays must be one-dimensional.")
        if len(self.x) != len(self.y_raw):
            raise ValueError("X and Y arrays must have equal lengths.")
        if len(self.x) < 3:
            raise ValueError("Dataset must contain at least three points.")
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.y_raw)):
            raise ValueError("Dataset contains non-finite values.")
        if np.any(np.diff(self.x) <= 0):
            order = np.argsort(self.x)
            self.x = self.x[order]
            self.y_raw = self.y_raw[order]
            if self.y_processed is not None:
                self.y_processed = self.y_processed[order]
        if np.any(np.diff(self.x) == 0):
            raise ValueError("Duplicate X values remain after sorting.")
