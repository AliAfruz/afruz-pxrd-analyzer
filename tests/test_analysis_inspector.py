from __future__ import annotations

import pytest
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLineEdit

from afruz_pxrd.analysis_panel import (
    AnalysisInspectorSection,
    AnalysisInspectorToolbar,
)
from afruz_pxrd.app import MainWindow


@pytest.mark.gui
def test_inspector_section_collapses_and_filters_real_form_rows(qtbot):
    group = QGroupBox("Peak parameters")
    form = QFormLayout(group)
    prominence = QLineEdit()
    tolerance = QLineEdit()
    tolerance.setProperty("inspectorAdvanced", True)
    form.addRow("Prominence fraction", prominence)
    form.addRow("Match tolerance", tolerance)

    section = AnalysisInspectorSection("peaks", group, expanded=True)
    qtbot.addWidget(section)
    section.show()

    matched, count = section.apply_filters("", include_advanced=False)
    assert matched
    assert count == 1
    assert form.isRowVisible(0)
    assert not form.isRowVisible(1)

    section.clear_filters()
    matched, count = section.apply_filters("tolerance", include_advanced=True)
    assert matched
    assert count == 1
    assert not form.isRowVisible(0)
    assert form.isRowVisible(1)

    section.clear_filters()
    section.set_expanded(False)
    assert not group.isVisible()
    section.set_expanded(True)
    assert group.isVisible()


@pytest.mark.gui
def test_inspector_toolbar_exposes_search_modes_and_summary(qtbot):
    toolbar = AnalysisInspectorToolbar()
    qtbot.addWidget(toolbar)
    toolbar.show()
    modes = []
    queries = []
    toolbar.modeChanged.connect(modes.append)
    toolbar.queryChanged.connect(queries.append)

    toolbar.all_button.click()
    toolbar.search.setText("wavelength")
    toolbar.set_summary(3, "wavelength")

    assert toolbar.current_mode() == "all"
    assert modes == ["all"]
    assert queries[-1] == "wavelength"
    assert toolbar.summary_label.text() == "3 found"


@pytest.mark.gui
def test_main_window_inspector_preserves_controls_and_project_ui_state(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(30)
    window._select_workflow_task("peak_list")
    qtbot.wait(20)

    peak_section = window.analysis_inspector_sections["peaks"]
    fitting_section = window.analysis_inspector_sections["fitting"]
    assert peak_section.isVisible()
    assert not fitting_section.isVisible()

    window.analysis_inspector_toolbar.set_mode("all", emit=True)
    qtbot.wait(10)
    assert fitting_section.isVisible()

    window.analysis_inspector_toolbar.search.setText("prominence")
    qtbot.wait(10)
    assert peak_section.isVisible()
    assert not fitting_section.isVisible()
    assert window.analysis_inspector_toolbar.summary_label.text().endswith("found")

    window.analysis_inspector_toolbar.search.clear()
    peak_section.set_expanded(False, emit=True)
    ui_state = window._ui_state()
    assert ui_state["analysis_inspector_mode"] == "all"
    assert not ui_state["analysis_inspector_expanded"]["peaks"]

    window.analysis_inspector_toolbar.set_mode("essential")
    peak_section.set_expanded(True)
    window._restore_analysis_inspector_state(ui_state)
    assert window.analysis_inspector_toolbar.current_mode() == "all"
    assert not peak_section.is_expanded()
