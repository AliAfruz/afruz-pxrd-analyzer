from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QUrl

from afruz_pxrd.app import MainWindow
from afruz_pxrd.project_access import (
    RecentProjectsPanel,
    RecentProjectsStore,
    classify_drop_paths,
)


class _DropEvent:
    def __init__(self, paths):
        self._mime = QMimeData()
        self._mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
        self.accepted = False
        self.ignored = False

    def mimeData(self):
        return self._mime

    def acceptProposedAction(self):
        self.accepted = True

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def test_recent_project_store_is_ordered_deduplicated_and_failure_tolerant(tmp_path):
    first = tmp_path / "first.afz"
    second = tmp_path / "second.afz"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    store = RecentProjectsStore(tmp_path / "recent.json", maximum=3)

    store.record(first, when=10)
    store.record(second, when=20)
    store.record(first, when=30)
    entries = store.entries()
    assert [entry.path for entry in entries] == [first.absolute(), second.absolute()]
    assert entries[0].age_label(now=35) == "just now"

    store.remove(first)
    assert [entry.path for entry in store.entries()] == [second.absolute()]
    store.clear()
    assert store.entries() == []
    store.path.write_text("not json", encoding="utf-8")
    assert store.entries() == []


def test_drop_classifier_expands_folders_and_limits_single_stateful_file(tmp_path):
    project_a = tmp_path / "a.afz"
    project_b = tmp_path / "b.afz"
    cif_a = tmp_path / "a.cif"
    cif_b = tmp_path / "b.cif"
    pattern = tmp_path / "scan.xy"
    unsupported = tmp_path / "figure.png"
    for path in (project_a, project_b, cif_a, cif_b, pattern, unsupported):
        path.write_text("data", encoding="utf-8")

    classified = classify_drop_paths(
        [project_a, project_b, cif_a, cif_b, pattern, unsupported]
    )
    assert classified.project == project_a.absolute()
    assert classified.cifs == (cif_a.absolute(),)
    assert classified.patterns == (pattern.absolute(),)
    assert set(classified.rejected) == {
        project_b.absolute(),
        cif_b.absolute(),
        unsupported.absolute(),
    }


@pytest.mark.gui
def test_recent_projects_panel_routes_open_remove_and_clear(qtbot, tmp_path):
    path = tmp_path / "sample.afz"
    path.write_text("project", encoding="utf-8")
    store = RecentProjectsStore(tmp_path / "recent.json")
    store.record(path, when=100)
    panel = RecentProjectsPanel()
    qtbot.addWidget(panel)
    panel.set_entries(store.entries())

    opened = []
    removed = []
    cleared = []
    panel.openRequested.connect(opened.append)
    panel.removeRequested.connect(removed.append)
    panel.clearRequested.connect(lambda: cleared.append(True))
    project_button = panel.findChild(type(panel.clear_button), "recentProjectButton")
    remove_button = panel.findChild(type(panel.clear_button), "recentProjectRemove")
    project_button.click()
    remove_button.click()
    panel.clear_button.click()

    assert opened == [str(path.absolute())]
    assert removed == [str(path.absolute())]
    assert cleared == [True]


@pytest.mark.gui
def test_main_window_accepts_pattern_drop_and_refreshes_recent_projects(
    qtbot, tmp_path, project_root: Path
):
    window = MainWindow()
    qtbot.addWidget(window)
    window.recent_projects_store = RecentProjectsStore(tmp_path / "recent.json")
    window._refresh_recent_projects()
    window.show()
    qtbot.wait(30)

    pattern = (
        project_root
        / "Afruz_PXRD_Synthetic_Test_Set"
        / "01_NaCl_single_clean.txt"
    )
    enter = _DropEvent([pattern])
    window.dragEnterEvent(enter)
    assert enter.accepted
    assert window.drop_import_overlay.isVisible()

    dropped = _DropEvent([pattern])
    window.dropEvent(dropped)
    assert dropped.accepted
    assert not window.drop_import_overlay.isVisible()
    assert len(window.datasets) >= 1

    project = tmp_path / "remembered.afz"
    project.write_text("project", encoding="utf-8")
    window._record_recent_project(project)
    assert window.project_home_widget.recent_projects_panel.count_label.text() == "1"
