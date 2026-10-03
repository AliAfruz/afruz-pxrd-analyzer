from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from afruz_pxrd.io_engine import (
    DataImportError,
    export_dataset_excel,
    export_dataset_txt,
    load_patterns,
)
from afruz_pxrd.models import Dataset
from afruz_pxrd.project import load_afz, save_afz
from afruz_pxrd.report_builder import build_complete_report_package, create_report_archive


SUPPLIED_PATTERNS = (
    ("synthetic", "01_NaCl_single_clean.txt", 3801, 4.0, 80.0),
    ("synthetic", "02_NaCl_CsCl_mixed_clean.txt", 3801, 4.0, 80.0),
    ("synthetic", "03_NaCl_CsCl_mixed_with_mismatches.txt", 3801, 4.0, 80.0),
    ("statistical", "01_NaCl_raw_counts_with_sigma.txt", 3801, 4.0, 80.0),
    ("statistical", "02_NaCl_normalized_100_with_sigma.txt", 3801, 4.0, 80.0),
    ("statistical", "04_NaCl_CsCl_mixed_raw_with_sigma.txt", 3801, 4.0, 80.0),
    ("statistical", "05_NaCl_CsCl_mixed_with_artifacts.txt", 3801, 4.0, 80.0),
)


@pytest.mark.parametrize("family,filename,points,x_min,x_max", SUPPLIED_PATTERNS)
def test_supplied_pattern_imports(
    family,
    filename,
    points,
    x_min,
    x_max,
    synthetic_data_dir,
    statistical_data_dir,
):
    directory = synthetic_data_dir if family == "synthetic" else statistical_data_dir
    loaded = load_patterns(directory / filename)
    assert len(loaded) == 1
    dataset = loaded[0]
    assert len(dataset.x) == points
    assert dataset.x[0] == pytest.approx(x_min)
    assert dataset.x[-1] == pytest.approx(x_max)
    assert np.all(np.diff(dataset.x) > 0)
    assert np.all(np.isfinite(dataset.y_raw))


def test_dataset_validation_sorts_coordinates_and_rejects_duplicates():
    dataset = Dataset(
        name="unsorted",
        x=np.array([2.0, 1.0, 3.0]),
        y_raw=np.array([20.0, 10.0, 30.0]),
    )
    dataset.validate()
    assert dataset.x.tolist() == [1.0, 2.0, 3.0]
    assert dataset.y_raw.tolist() == [10.0, 20.0, 30.0]

    duplicate = Dataset(
        name="duplicate",
        x=np.array([1.0, 1.0, 2.0]),
        y_raw=np.array([10.0, 11.0, 20.0]),
    )
    with pytest.raises(ValueError, match="Duplicate X values"):
        duplicate.validate()


def test_unsupported_import_has_actionable_error(tmp_path):
    path = tmp_path / "pattern.unsupported"
    path.write_text("10 100\n11 120\n12 90\n", encoding="utf-8")
    with pytest.raises(DataImportError, match="Unsupported format"):
        load_patterns(path)


@pytest.mark.integration
def test_project_round_trip_preserves_arrays_and_state(tmp_path, statistical_data_dir):
    datasets = [
        load_patterns(statistical_data_dir / "01_NaCl_raw_counts_with_sigma.txt")[0],
        load_patterns(statistical_data_dir / "04_NaCl_CsCl_mixed_raw_with_sigma.txt")[0],
    ]
    datasets[0].y_processed = datasets[0].y_raw / datasets[0].y_raw.max() * 100.0
    path = tmp_path / "roundtrip.afz"
    ui_state = {"theme": "Dark Gold", "selected_uid": datasets[0].uid}
    analysis_state = {"phase": {"status": "baseline"}}
    save_afz(path, datasets, ui_state, analysis_state)
    restored, restored_ui, restored_analysis = load_afz(path)

    assert len(restored) == 2
    assert restored_ui == ui_state
    assert restored_analysis == analysis_state
    for original, loaded in zip(datasets, restored):
        assert original.uid == loaded.uid
        assert np.array_equal(original.x, loaded.x)
        assert np.array_equal(original.y_raw, loaded.y_raw)
    assert np.array_equal(restored[0].y_processed, datasets[0].y_processed)


@pytest.mark.integration
def test_txt_excel_and_complete_report_exports(tmp_path, statistical_data_dir):
    dataset = load_patterns(statistical_data_dir / "01_NaCl_raw_counts_with_sigma.txt")[0]
    txt_path = tmp_path / "pattern.txt"
    xlsx_path = tmp_path / "pattern.xlsx"
    export_dataset_txt(dataset, txt_path)
    export_dataset_excel(dataset, xlsx_path)

    text = txt_path.read_text(encoding="utf-8-sig")
    workbook = pd.ExcelFile(xlsx_path)
    assert "two_theta" in text.lower()
    profile_sheet = next(name for name in workbook.sheet_names if name.startswith("Diffraction profile"))
    profile = pd.read_excel(xlsx_path, sheet_name=profile_sheet, header=None)
    point_indices = pd.to_numeric(profile.iloc[:, 0], errors="coerce").dropna()
    assert len(point_indices) == len(dataset.x)

    report_dir = tmp_path / "complete_report"
    report = build_complete_report_package(
        report_dir,
        project_name="Phase1_Test",
        datasets=[dataset],
        accepted_results={"import": {"status": "Accepted"}},
        warnings=["Synthetic software-validation dataset"],
        notes="Automated Phase 1 integration test",
    )
    for key in ("json_path", "markdown_path", "html_path", "manifest_path"):
        assert Path(report[key]).is_file(), key
    manifest = json.loads(Path(report["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["files"]

    archive = create_report_archive(report_dir)
    archive_path = Path(archive["archive_path"])
    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path) as handle:
        assert any(name.endswith("/manifest.json") for name in handle.namelist())
