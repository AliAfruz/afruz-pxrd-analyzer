from __future__ import annotations

import io
import inspect
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import openpyxl
import pytest

import afruz_pxrd
import afruz_pxrd.project as project_module
from afruz_pxrd.app import MainWindow
from afruz_pxrd.batch_report import write_html_report
from afruz_pxrd.batch_recipe import DEFAULT_RECIPE
from afruz_pxrd.io_engine import export_dataset_excel, export_dataset_txt
from afruz_pxrd.models import Dataset
from afruz_pxrd.project import (
    PROJECT_VERSION,
    ProjectFormatError,
    ProjectSaveError,
    ProjectVersionError,
    load_afz,
    project_backup_path,
    save_afz,
    validate_afz,
)
from afruz_pxrd.report_builder import build_complete_report_package
from afruz_pxrd.validation_campaign import export_reproducibility_package
from afruz_pxrd.version import (
    APP_NAME,
    APP_RELEASE,
    APP_VERSION,
    APP_WINDOW_TITLE,
    project_window_title,
)


def _dataset(name="sample", uid="dataset-1", offset=0.0):
    x = np.array([10.0, 11.0, 12.0, 13.0])
    return Dataset(
        uid=uid,
        name=name,
        x=x,
        y_raw=np.array([100.0, 120.0, 90.0, 80.0]) + offset,
        y_processed=np.array([95.0, 115.0, 85.0, 75.0]) + offset,
        metadata={"wavelength_angstrom": 1.5406},
    )


def _write_archive(path: Path, manifest, arrays=None, *, manifest_bytes=None):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **(arrays or {}))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "project.json",
            manifest_bytes
            if manifest_bytes is not None
            else json.dumps(manifest).encode("utf-8"),
        )
        archive.writestr("datasets.npz", buffer.getvalue())


def test_authoritative_release_identity_is_consistent():
    assert APP_NAME == "Afruz PXRD Analyzer"
    assert APP_VERSION == "23.0.0"
    assert afruz_pxrd.__version__ == APP_VERSION
    assert APP_RELEASE == f"{APP_NAME} {APP_VERSION}"
    assert APP_WINDOW_TITLE == f"{APP_NAME} — {APP_VERSION}"
    assert project_window_title("example.afz") == f"example.afz — {APP_WINDOW_TITLE}"
    assert DEFAULT_RECIPE["software_version"] == APP_VERSION
    default = inspect.signature(export_reproducibility_package).parameters[
        "software_version"
    ].default
    assert default == APP_VERSION


@pytest.mark.gui
def test_window_header_and_about_use_authoritative_release(qtbot, monkeypatch):
    about_call = {}

    def capture_about(parent, title, message):
        about_call.update(title=title, message=message)

    monkeypatch.setattr("afruz_pxrd.app.QMessageBox.about", capture_about)
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == APP_WINDOW_TITLE
    assert window.workflow_header.brand_label.text() == APP_NAME
    assert window.workflow_header.version_label.text() == f"Version {APP_VERSION}"
    assert window.build_export_snapshot(all_datasets=True)["application_version"] == APP_VERSION
    window.show_about()
    assert about_call["title"] == f"About {APP_NAME}"
    assert APP_WINDOW_TITLE in about_call["message"]
    window.close()


@pytest.mark.integration
def test_exports_and_reports_record_authoritative_release(tmp_path):
    dataset = _dataset()
    txt_path = tmp_path / "dataset.txt"
    xlsx_path = tmp_path / "dataset.xlsx"
    export_dataset_txt(dataset, txt_path)
    export_dataset_excel(dataset, xlsx_path)
    assert f"# application_version: {APP_VERSION}" in txt_path.read_text(
        encoding="utf-8-sig"
    )

    workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    values = {
        str(cell)
        for sheet in workbook.worksheets
        for row in sheet.iter_rows(values_only=True)
        for cell in row
        if cell is not None
    }
    workbook.close()
    assert APP_VERSION in values

    package = build_complete_report_package(
        tmp_path / "report", project_name="Phase2", datasets=[dataset]
    )
    payload = json.loads(Path(package["json_path"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(package["manifest_path"]).read_text(encoding="utf-8"))
    markdown = Path(package["markdown_path"]).read_text(encoding="utf-8")
    html = Path(package["html_path"]).read_text(encoding="utf-8")
    assert payload["application_name"] == APP_NAME
    assert payload["application_version"] == APP_VERSION
    assert manifest["application_version"] == APP_VERSION
    assert f"{APP_NAME} {APP_VERSION}" in markdown
    assert f"{APP_NAME} {APP_VERSION}" in html

    batch_html_path = write_html_report(
        tmp_path / "batch.html",
        {
            "created_utc": "2026-08-11T00:00:00+00:00",
            "recipe": DEFAULT_RECIPE,
            "recipe_fingerprint": "test",
            "dataset_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "elapsed_seconds": 0.0,
            "results": [],
            "errors": [],
        },
    )
    assert APP_RELEASE in batch_html_path.read_text(encoding="utf-8")


def test_atomic_save_creates_valid_backup_of_previous_project(tmp_path):
    path = tmp_path / "project.afz"
    save_afz(path, [_dataset(offset=0.0)], {"theme": "Light"}, {"revision": 1})
    assert validate_afz(path)["project_version"] == PROJECT_VERSION
    save_afz(path, [_dataset(offset=25.0)], {"theme": "Dark"}, {"revision": 2})

    backup = project_backup_path(path)
    assert backup.is_file()
    current, current_ui, current_analysis = load_afz(path)
    previous, previous_ui, previous_analysis = load_afz(backup)
    assert current[0].y_raw[0] == pytest.approx(125.0)
    assert current_ui == {"theme": "Dark"}
    assert current_analysis == {"revision": 2}
    assert previous[0].y_raw[0] == pytest.approx(100.0)
    assert previous_ui == {"theme": "Light"}
    assert previous_analysis == {"revision": 1}


def test_failed_final_replace_preserves_original_project(tmp_path, monkeypatch):
    path = tmp_path / "protected.afz"
    save_afz(path, [_dataset(offset=0.0)], {}, {"revision": 1})
    original_bytes = path.read_bytes()
    real_replace = os.replace

    def fail_destination_replace(source, destination):
        if Path(destination) == path:
            raise OSError("simulated final replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(project_module.os, "replace", fail_destination_replace)
    with pytest.raises(ProjectSaveError, match="previous project was left unchanged"):
        save_afz(path, [_dataset(offset=50.0)], {}, {"revision": 2})

    assert path.read_bytes() == original_bytes
    restored, _, analysis = load_afz(path)
    assert restored[0].y_raw[0] == pytest.approx(100.0)
    assert analysis == {"revision": 1}
    assert not list(tmp_path.glob(".*.tmp"))


def test_version_one_project_migrates_with_processed_array_inference(tmp_path):
    dataset = _dataset()
    manifest = {
        "project_version": 1,
        "datasets": [{"uid": dataset.uid, "name": dataset.name}],
    }
    arrays = {
        f"{dataset.uid}_x": dataset.x,
        f"{dataset.uid}_raw": dataset.y_raw,
        f"{dataset.uid}_processed": dataset.y_processed,
    }
    path = tmp_path / "version1.afz"
    _write_archive(path, manifest, arrays)

    restored, ui, analysis = load_afz(path)
    assert ui == {}
    assert analysis == {}
    assert np.array_equal(restored[0].y_processed, dataset.y_processed)
    assert validate_afz(path)["project_version"] == 2


def test_version_two_project_opens_without_migration_loss(tmp_path):
    path = tmp_path / "version2.afz"
    source = _dataset()
    save_afz(path, [source], {"workflow_mode": "Guided"}, {"accepted": True})
    restored, ui, analysis = load_afz(path)
    assert restored[0].uid == source.uid
    assert np.array_equal(restored[0].x, source.x)
    assert ui == {"workflow_mode": "Guided"}
    assert analysis == {"accepted": True}


def test_invalid_zip_and_missing_members_have_actionable_errors(tmp_path):
    invalid = tmp_path / "not-a-project.afz"
    invalid.write_bytes(b"not a zip archive")
    with pytest.raises(ProjectFormatError, match="not a valid .afz ZIP archive"):
        load_afz(invalid)

    incomplete = tmp_path / "incomplete.afz"
    with zipfile.ZipFile(incomplete, "w") as archive:
        archive.writestr("project.json", "{}")
    with pytest.raises(ProjectFormatError, match="missing required member.*datasets.npz"):
        load_afz(incomplete)


def test_invalid_json_unsupported_version_and_missing_array_are_rejected(tmp_path):
    invalid_json = tmp_path / "invalid-json.afz"
    _write_archive(invalid_json, {}, {}, manifest_bytes=b"{broken")
    with pytest.raises(ProjectFormatError, match="invalid JSON"):
        load_afz(invalid_json)

    unsupported = tmp_path / "future.afz"
    _write_archive(
        unsupported,
        {"project_version": 999, "datasets": [], "ui_state": {}, "analysis_state": {}},
    )
    with pytest.raises(ProjectVersionError, match="Unsupported project version 999"):
        load_afz(unsupported)

    missing_array = tmp_path / "missing-array.afz"
    _write_archive(
        missing_array,
        {
            "project_version": 2,
            "datasets": [{"uid": "lost", "name": "Lost dataset"}],
            "ui_state": {},
            "analysis_state": {},
        },
    )
    with pytest.raises(ProjectFormatError, match="missing array"):
        load_afz(missing_array)


def test_windows_launcher_contains_dependency_and_repair_checks(project_root):
    launcher = (project_root / "run_windows.bat").read_text(encoding="utf-8")
    assert "sys.version_info[:2] == (3, 12)" in launcher
    for dependency in (
        "PySide6",
        "pyqtgraph",
        "numpy",
        "scipy",
        "pandas",
        "openpyxl",
        "matplotlib",
        "reportlab",
    ):
        assert dependency in launcher
    assert "bootstrap_windows.ps1" in launcher
