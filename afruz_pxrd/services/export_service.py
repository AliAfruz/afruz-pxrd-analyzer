from __future__ import annotations

from pathlib import Path
from typing import Any

import pyqtgraph as pg
import pyqtgraph.exporters

from ..io_engine import export_dataset_excel, export_dataset_txt
from ..models import Dataset


class ExportService:
    """Export boundary used by the window and independently testable clients."""

    def export_dataset_text(self, dataset: Dataset, path: str | Path) -> Path:
        destination = Path(path)
        export_dataset_txt(dataset, destination)
        return destination

    def export_dataset_excel(self, dataset: Dataset, path: str | Path) -> Path:
        destination = Path(path)
        export_dataset_excel(dataset, destination)
        return destination

    def export_plot_png(
        self,
        plot_item: Any,
        path: str | Path,
        *,
        width: int = 2400,
    ) -> Path:
        destination = Path(path).with_suffix(".png")
        exporter = pg.exporters.ImageExporter(plot_item)
        exporter.parameters()["width"] = int(width)
        exporter.export(str(destination))
        return destination

    def export_plot_svg(self, plot_item: Any, path: str | Path) -> Path:
        destination = Path(path).with_suffix(".svg")
        exporter = pg.exporters.SVGExporter(plot_item)
        exporter.export(str(destination))
        return destination
