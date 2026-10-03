from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from afruz_pxrd.context_copy import install_global_copy_support
from afruz_pxrd.contracts import ResultContractService, ResultContractStore
from afruz_pxrd.export_center import ExportCenterDialog
from afruz_pxrd.export_engine import build_clean_export_package, write_text_package


def _snapshot() -> dict:
    x = np.array([10.0, 11.0, 12.0])
    observed = np.array([5.0, 10.0, 6.0])
    calculated = np.array([4.5, 9.5, 6.5])
    common = {
        "observed_x": x,
        "observed_y": observed,
        "calculated_y": calculated,
        "background_y": np.array([1.0, 1.0, 1.0]),
        "rwp_percent": 4.25,
        "rp_percent": 3.1,
        "statistics_valid": True,
        "parameters": [
            {"parameter": "zero_shift_deg", "value": 0.012, "standard_error": 0.003, "unit": "deg"}
        ],
        "reflections": [
            {"h": 1, "k": 0, "l": 0, "two_theta_deg": 11.0, "d_spacing": 8.0, "extracted_intensity": 50.0}
        ],
    }
    return {
        "project_name": "Clean export test",
        "application_version": "test",
        "datasets": [{
            "uid": "sample-1",
            "name": "MIL 101 Cr",
            "x": x,
            "y_raw": np.array([8.0, 14.0, 9.0]),
            "y_processed": observed,
            "analyses": {
                "preprocessing_profile": [
                    {
                        "two_theta_deg": xx,
                        "intensity_processed": yy,
                        "background_raw_units": 1.0,
                    }
                    for xx, yy in zip(x, observed)
                ],
                "peaks": [{"position": 11.0, "intensity": 10.0, "fwhm": 0.2, "use": True}],
                "whole_pattern": {**common, "mode": "Le Bail extraction"},
                "rietveld": {
                    **common,
                    "method": "Structure-constrained Rietveld profile refinement",
                    "phases": [{
                        "phase_name": "MIL-101(Cr)",
                        "formula": "Cr3O",
                        "scale_factor": 1.2,
                        "refined_cell": {"a": 88.7, "b": 88.7, "c": 88.7},
                    }],
                },
            },
        }],
    }


def test_clean_export_separates_scientific_products_and_refinement_methods(tmp_path):
    package = build_clean_export_package(_snapshot())
    keys = {table.key for table in package.tables}
    assert "sample-1_raw_data" in keys
    assert "sample-1_treated_data" in keys
    assert "sample-1_peaks" in keys
    assert "sample-1_le_bail_profile" in keys
    assert "sample-1_rietveld_profile" in keys
    assert "sample-1_rietveld_phases" in keys

    raw = next(table for table in package.tables if table.key.endswith("_raw_data"))
    treated = next(table for table in package.tables if table.key.endswith("_treated_data"))
    assert [column.key for column in raw.columns] == ["point_index", "two_theta_deg", "intensity_raw"]
    assert "intensity_raw" not in {column.key for column in treated.columns}
    assert "intensity_treated" in {column.key for column in treated.columns}

    profile = next(table for table in package.tables if table.key.endswith("_le_bail_profile"))
    assert profile.rows[0]["difference_observed_minus_calculated"] == pytest.approx(0.5)

    output = write_text_package(tmp_path / "clean", package, compact=True)
    names = {path.name for path in Path(output).iterdir()}
    assert "MIL_101_Cr_01_raw_data.txt" in names
    assert "MIL_101_Cr_02_treated_data.txt" in names
    assert "MIL_101_Cr_04_le_bail_profile.txt" in names
    assert "MIL_101_Cr_05_rietveld_profile.txt" in names
    assert {"README.txt", "manifest.txt"}.issubset(names)
    assert "table_inventory.txt" not in names
    assert "data_dictionary.txt" not in names
    assert "package_metadata.txt" not in names

    text = (Path(output) / "MIL_101_Cr_05_rietveld_profile.txt").read_text(encoding="utf-8-sig")
    assert "Observed intensity" in text
    assert "Calculated intensity" in text
    assert "Difference: observed" in text


def test_clean_export_uses_contract_profile_but_keeps_human_readable_mode():
    snapshot = _snapshot()
    dataset = snapshot["datasets"][0]
    uid = dataset["uid"]
    authoritative = dict(dataset["analyses"]["whole_pattern"])
    authoritative["observed_y"] = np.array([7.0, 8.0, 9.0])
    authoritative["calculated_y"] = np.array([6.0, 7.0, 8.0])
    store = ResultContractStore()
    ResultContractService.capture_dataset(
        store, uid, {"whole_pattern_refinement": authoritative}
    )
    dataset["analyses"]["result_contracts"] = ResultContractService.export_dataset(store, uid)
    # Simulate an older live array; the typed contract must win, while this
    # result still contributes the Le Bail mode label.
    dataset["analyses"]["whole_pattern"]["observed_y"] = np.array([100.0, 100.0, 100.0])

    package = build_clean_export_package(snapshot, include_content={"refinement"})
    profile = next(table for table in package.tables if table.key.endswith("_le_bail_profile"))
    assert [row["observed_intensity"] for row in profile.rows] == [7.0, 8.0, 9.0]


@pytest.mark.gui
def test_global_copy_support_makes_labels_and_tables_selectable(qtbot, qapp):
    root = QWidget()
    qtbot.addWidget(root)
    layout = QVBoxLayout(root)
    label = QLabel("Selectable scientific result")
    table = QTableWidget(1, 2)
    table.setHorizontalHeaderLabels(["A", "B"])
    table.setItem(0, 0, QTableWidgetItem("alpha"))
    table.setItem(0, 1, QTableWidgetItem("beta"))
    layout.addWidget(label)
    layout.addWidget(table)

    install_global_copy_support(qapp, root)
    assert label.textInteractionFlags() & Qt.TextSelectableByMouse
    assert label.textInteractionFlags() & Qt.TextSelectableByKeyboard
    assert table.property("afruzCopySupportInstalled") is True

    table.selectAll()
    copy_action = next(action for action in table.actions() if action.text() == "Copy selected cells")
    assert copy_action.shortcut().toString() == "Ctrl+C"


@pytest.mark.gui
def test_clean_export_dialog_defaults_to_txt_and_four_clear_products(qtbot):
    class FakeMainWindow(QWidget):
        current_project = None

        def build_export_snapshot(self, *, all_datasets=False):
            return _snapshot()

    parent = FakeMainWindow()
    qtbot.addWidget(parent)
    dialog = ExportCenterDialog(parent)
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "Clean Data Export"
    assert set(dialog.category_checks) == {"raw", "treated", "peaks", "refinement"}
    assert dialog.text_check.isChecked()
    assert not dialog.excel_check.isChecked()
    assert not dialog.zip_check.isChecked()
    assert dialog.package is not None
    assert all(table.metadata.get("clean_filename") for table in dialog.package.tables)
