from __future__ import annotations

import pytest
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QScrollArea

from afruz_pxrd.app import MainWindow
from afruz_pxrd.cinematic_ui import AuraBackdrop, AuraBrandMark, cinematic_icon
from afruz_pxrd.command_palette import CommandItem, CommandPalette
from afruz_pxrd.theme import DEFAULT_THEME_NAME, THEMES, build_stylesheet


def test_cinematic_themes_preserve_legacy_palettes_and_build_valid_qss():
    assert DEFAULT_THEME_NAME == "Aura Night"
    assert {"Aura Night", "Aura Pearl", "Dark Gold", "Light Gold"} <= set(THEMES)
    for name, theme in THEMES.items():
        stylesheet = build_stylesheet(name)
        assert theme["accent"] in stylesheet
        assert "QPushButton#headerRunButton" in stylesheet
        assert "qlineargradient" in stylesheet


def test_vector_icon_factory_returns_renderable_icons(qtbot):
    icon = cinematic_icon("crystal", "#42dcff")
    assert not icon.isNull()
    assert not icon.pixmap(QSize(24, 24)).isNull()


@pytest.mark.gui
def test_command_palette_filters_and_runs_keyboard_first_commands(qtbot):
    calls = []
    palette = CommandPalette()
    qtbot.addWidget(palette)
    palette.set_commands(
        [
            CommandItem("Import pattern", "Project", lambda: calls.append("import"), "upload"),
            CommandItem("3D Crystal Studio", "Visualize", lambda: calls.append("studio"), "crystal", keywords=("structure", "cif")),
        ]
    )
    palette.open_centered(reduced_motion=True)
    palette.search.setText("structure")
    assert palette.list_widget.count() == 1
    assert palette.list_widget.currentItem().text() == "3D Crystal Studio"
    qtbot.keyClick(palette.search, Qt.Key_Return)
    assert calls == ["studio"]
    assert not palette.isVisible()


@pytest.mark.gui
def test_aura_components_render_without_platform_specific_assets(qtbot):
    backdrop = AuraBackdrop()
    qtbot.addWidget(backdrop)
    backdrop.resize(640, 360)
    backdrop.show()
    qtbot.wait(30)

    image = QImage(backdrop.size(), QImage.Format_ARGB32_Premultiplied)
    image.fill(0)
    backdrop.render(image)
    assert not image.isNull()

    mark = AuraBrandMark()
    qtbot.addWidget(mark)
    mark.show()
    qtbot.wait(20)
    assert mark._timer.isActive()
    mark.set_reduced_motion(True)
    assert not mark._timer.isActive()


@pytest.mark.gui
def test_main_window_exposes_cinematic_shell_icons_and_motion_control(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(60)

    assert isinstance(window.shell_root, AuraBackdrop)
    assert window.current_theme_name == "Aura Night"
    assert not window.workflow_header.run_button.icon().isNull()
    assert not window.workflow_header.save_button.icon().isNull()
    assert all(not button.icon().isNull() for button in window.workflow_navigation.workspace_buttons.values())
    assert window.workflow_navigation.selection_indicator.isVisible()

    window.reduced_motion_action.setChecked(True)
    qtbot.wait(10)
    assert window.cinematic_ui.reduced_motion
    assert not window.shell_root._timer.isActive()
    assert not window.workflow_header.brand_mark._timer.isActive()

    window.apply_theme("Aura Pearl")
    assert window.current_theme_name == "Aura Pearl"
    assert window.shell_root._theme_name == "Aura Pearl"
    assert window.command_palette_action.shortcut().toString() == "Ctrl+K"
    window.command_palette_action.trigger()
    qtbot.wait(20)
    assert window.command_palette.isVisible()
    assert window.command_palette.list_widget.count() >= 40
    window.command_palette.hide()

    window.toggle_left_sidebar_action.setChecked(False)
    assert not window.left_panel.isVisible()
    window.toggle_left_sidebar_action.setChecked(True)
    assert window.left_panel.isVisible()


@pytest.mark.gui
def test_sidebars_remain_collision_free_across_supported_window_sizes(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert isinstance(window.left_panel, QScrollArea)
    assert window.left_panel.widget() is window.left_panel_content

    navigation = window.workflow_navigation
    ordered = [
        *navigation.workspace_buttons.values(),
        navigation.stage_description,
        navigation.task_caption,
        navigation.task_list,
        navigation.guidance_label,
    ]
    if navigation.task_state_label.isVisible():
        ordered.append(navigation.task_state_label)

    for width, height in ((1912, 1029), (1600, 900), (1280, 720)):
        window.resize(width, height)
        window.show()
        qtbot.wait(40)
        for upper, lower in zip(ordered, ordered[1:]):
            assert upper.geometry().bottom() < lower.geometry().top(), (
                f"{upper.objectName() or upper.text()} overlaps "
                f"{lower.objectName() or lower.text()} at {width}x{height}"
            )

    window.fullscreen_action.setChecked(True)
    qtbot.wait(10)
    assert window.isFullScreen()
    window.fullscreen_action.setChecked(False)
    qtbot.wait(10)
    assert window.isMaximized()
