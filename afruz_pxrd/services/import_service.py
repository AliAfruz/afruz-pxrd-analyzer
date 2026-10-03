from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..io_engine import DataImportError, load_patterns
from ..models import Dataset


@dataclass(frozen=True)
class ImportFailure:
    path: Path
    message: str

    def display_text(self) -> str:
        return f"{self.path.name}: {self.message}"


@dataclass(frozen=True)
class ImportBatch:
    datasets: tuple[Dataset, ...]
    imported_file_count: int
    failures: tuple[ImportFailure, ...]


class ImportService:
    def __init__(self, loader: Callable[[str | Path], list[Dataset]] = load_patterns):
        self._loader = loader

    def load_paths(self, paths: Iterable[str | Path]) -> ImportBatch:
        datasets: list[Dataset] = []
        failures: list[ImportFailure] = []
        imported_file_count = 0
        for path_value in paths:
            path = Path(path_value)
            try:
                loaded = list(self._loader(path_value))
            except DataImportError as exc:
                failures.append(ImportFailure(path, str(exc)))
                continue
            datasets.extend(loaded)
            imported_file_count += 1
        return ImportBatch(tuple(datasets), imported_file_count, tuple(failures))
