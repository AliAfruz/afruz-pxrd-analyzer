from __future__ import annotations

import pytest

from afruz_pxrd.app import MainWindow


@pytest.mark.gui
def test_main_window_constructs_and_closes_without_worker_api_crash(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(50)

    assert window.windowTitle()
    assert window.isVisible()
    from afruz_pxrd.crystal_studio import CrystalStudioDialog
    assert window.rietveld_widget.crystal_studio_button.isEnabled()
    window.rietveld_widget.crystal_studio_button.click()
    studio = window.rietveld_widget.findChild(CrystalStudioDialog)
    assert studio is not None and studio.isVisible()
    studio_buttons = studio.findChildren(type(window.workflow_header.run_button))
    assert any(not button.icon().isNull() for button in studio_buttons)
    studio.close()
    assert window.close()
