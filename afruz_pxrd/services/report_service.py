from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import re

from ..report_builder import build_complete_report_package, create_report_archive


def safe_project_component(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._")
    return cleaned or "afruz_project"


class ReportService:
    def __init__(
        self,
        *,
        builder: Callable[..., dict] = build_complete_report_package,
        archiver: Callable[[str | Path], dict] = create_report_archive,
    ):
        self._builder = builder
        self._archiver = archiver

    def build_complete_package(self, output_root: str | Path, **payload: Any) -> dict:
        package = self._builder(Path(output_root), **payload)
        archive = self._archiver(package["output_dir"])
        return {"package": package, "archive": archive}
