"""Application color systems and the shared cinematic Qt stylesheet."""

AURA_NIGHT = {
    "name": "Aura Night", "dark": True, "graphite": "#070a12", "panel": "#101726",
    "panel_2": "#182238", "glass": "rgba(15, 23, 40, 226)", "glass_2": "rgba(24, 34, 56, 216)",
    "input": "#0b1220", "text": "#f5f7ff", "muted": "#9aa8bd", "border": "#293752",
    "border_soft": "rgba(117, 142, 185, 62)", "gold": "#8b5cf6", "gold_bright": "#c4b5fd",
    "accent": "#8b5cf6", "accent_2": "#d946ef", "cyan": "#42dcff", "blue": "#5b8cff",
    "success": "#55e6bd", "warning": "#f6c66b", "selection_text": "#ffffff", "danger": "#ff6b81",
    "icon": "#d8d1ff", "grid": "#344765", "plot_background": "#0a1020", "plot_axis": "#d5deed",
    "grid_alpha": 0.16,
}

AURA_PEARL = {
    "name": "Aura Pearl", "dark": False, "graphite": "#f2f5fb", "panel": "#ffffff",
    "panel_2": "#eef2fb", "glass": "rgba(255, 255, 255, 235)", "glass_2": "rgba(241, 244, 252, 232)",
    "input": "#fbfcff", "text": "#182033", "muted": "#65718a", "border": "#cad3e5",
    "border_soft": "rgba(88, 105, 145, 48)", "gold": "#7657e8", "gold_bright": "#5f43ca",
    "accent": "#7657e8", "accent_2": "#bf3fd4", "cyan": "#099ac2", "blue": "#356ee5",
    "success": "#12856c", "warning": "#ad6b00", "selection_text": "#ffffff", "danger": "#c83c59",
    "icon": "#654bd0", "grid": "#bdc9dc", "plot_background": "#ffffff", "plot_axis": "#34415b",
    "grid_alpha": 0.18,
}

# Preserved for older projects and users who prefer the original palette.
DARK_GOLD = {
    "name": "Dark Gold", "dark": True, "graphite": "#15181e", "panel": "#20242c",
    "panel_2": "#2a303a", "glass": "rgba(31, 35, 43, 232)", "glass_2": "rgba(42, 48, 58, 224)",
    "input": "#101318", "text": "#f3f4f6", "muted": "#a7adb8", "border": "#3c4451",
    "border_soft": "rgba(150, 157, 171, 52)", "gold": "#d5ad55", "gold_bright": "#f0cd78",
    "accent": "#d5ad55", "accent_2": "#f0cd78", "cyan": "#7dd3fc", "blue": "#7ea7e8",
    "success": "#72d6af", "warning": "#f0cd78", "selection_text": "#111318", "danger": "#eb7180",
    "icon": "#f0cd78", "grid": "#46505e", "plot_background": "#101318", "plot_axis": "#dadde2",
    "grid_alpha": 0.18,
}

LIGHT_GOLD = {
    "name": "Light Gold", "dark": False, "graphite": "#f4efe7", "panel": "#fffaf2",
    "panel_2": "#eadfce", "glass": "rgba(255, 250, 242, 238)", "glass_2": "rgba(234, 223, 206, 226)",
    "input": "#fffdf8", "text": "#3b3025", "muted": "#75695c", "border": "#c9b89f",
    "border_soft": "rgba(117, 91, 61, 45)", "gold": "#b58a3d", "gold_bright": "#8e672a",
    "accent": "#b58a3d", "accent_2": "#c47b48", "cyan": "#178ca0", "blue": "#3f72b5",
    "success": "#287d61", "warning": "#9c6715", "selection_text": "#fffdf8", "danger": "#b54855",
    "icon": "#8e672a", "grid": "#c8baa7", "plot_background": "#fffdf8", "plot_axis": "#4b4035",
    "grid_alpha": 0.20,
}

THEMES = {
    AURA_NIGHT["name"]: AURA_NIGHT, AURA_PEARL["name"]: AURA_PEARL,
    DARK_GOLD["name"]: DARK_GOLD, LIGHT_GOLD["name"]: LIGHT_GOLD,
}
DEFAULT_THEME_NAME = AURA_NIGHT["name"]


def build_stylesheet(theme_name: str) -> str:
    t = THEMES.get(theme_name, THEMES[DEFAULT_THEME_NAME])
    return f"""
QMainWindow, QDialog {{ background: {t["graphite"]}; color: {t["text"]}; }}
QWidget {{
    color: {t["text"]}; font-family: "Segoe UI Variable", "Segoe UI", "Inter", sans-serif;
    font-size: 10pt; selection-background-color: {t["accent"]}; selection-color: {t["selection_text"]};
}}
QWidget#auraBackdrop, QWidget#cleanApplicationShell {{ background: transparent; }}
QMenuBar {{
    background: {t["panel"]}; color: {t["muted"]}; border-bottom: 1px solid {t["border_soft"]}; padding: 2px 7px;
}}
QMenuBar::item {{ padding: 5px 9px; border-radius: 6px; }}
QMenuBar::item:selected {{ background: {t["panel_2"]}; color: {t["text"]}; }}
QMenu {{ background: {t["panel"]}; color: {t["text"]}; border: 1px solid {t["border"]}; padding: 6px; }}
QMenu::item {{ padding: 7px 28px 7px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {t["panel_2"]}; color: {t["gold_bright"]}; }}
QMenu::separator {{ height: 1px; background: {t["border_soft"]}; margin: 5px 8px; }}
QStatusBar {{ background: {t["panel"]}; color: {t["muted"]}; border-top: 1px solid {t["border_soft"]}; }}
QToolBar {{ background: {t["glass"]}; border: 0; border-bottom: 1px solid {t["border_soft"]}; spacing: 6px; padding: 5px; }}
QPushButton, QToolButton {{
    background: {t["glass_2"]}; color: {t["text"]}; border: 1px solid {t["border"]}; border-radius: 8px;
    min-height: 20px; padding: 6px 11px; font-weight: 600;
}}
QPushButton:hover, QToolButton:hover {{ background: {t["panel_2"]}; border-color: {t["cyan"]}; color: {t["cyan"]}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {t["input"]}; border-color: {t["accent"]}; padding-top: 7px; padding-bottom: 5px; }}
QPushButton:checked, QToolButton:checked {{ background: {t["panel_2"]}; border-color: {t["accent"]}; color: {t["gold_bright"]}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t["muted"]}; background: {t["glass"]}; border-color: {t["border_soft"]}; }}
QPushButton#primaryButton, QPushButton#headerRunButton {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {t["accent"]}, stop:1 {t["accent_2"]});
    color: #ffffff; font-weight: 750; border: 1px solid {t["gold_bright"]};
}}
QPushButton#primaryButton:hover, QPushButton#headerRunButton:hover {{ border-color: {t["cyan"]}; color: #ffffff; }}
QGroupBox {{
    background: {t["glass"]}; border: 1px solid {t["border_soft"]}; border-radius: 12px;
    margin-top: 14px; padding: 13px 8px 8px 8px; font-weight: 650; color: {t["gold_bright"]};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {{
    background: {t["input"]}; color: {t["text"]}; border: 1px solid {t["border"]}; border-radius: 7px; padding: 5px 7px;
}}
QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {t["blue"]}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {t["cyan"]}; }}
QComboBox::drop-down {{ border: 0; width: 24px; }}
QComboBox QAbstractItemView {{ background: {t["panel"]}; color: {t["text"]}; border: 1px solid {t["border"]}; selection-background-color: {t["accent"]}; selection-color: #ffffff; outline: 0; }}
QCheckBox, QRadioButton {{ spacing: 7px; color: {t["text"]}; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; border: 1px solid {t["border"]}; background: {t["input"]}; }}
QCheckBox::indicator {{ border-radius: 4px; }} QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {t["cyan"]}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ background: {t["accent"]}; border: 3px solid {t["gold_bright"]}; }}
QTabWidget::pane {{ border: 1px solid {t["border_soft"]}; border-radius: 10px; background: {t["glass"]}; top: -1px; }}
QTabBar::tab {{ background: transparent; color: {t["muted"]}; border: 0; border-bottom: 2px solid transparent; padding: 8px 12px; margin: 0 1px; font-weight: 600; }}
QTabBar::tab:hover {{ color: {t["cyan"]}; background: {t["glass_2"]}; border-radius: 7px; }}
QTabBar::tab:selected {{ color: {t["text"]}; border-bottom: 2px solid {t["accent"]}; }}
QListWidget, QTableWidget, QTreeWidget, QTableView, QTreeView {{
    background: {t["input"]}; alternate-background-color: {t["panel"]}; color: {t["text"]};
    border: 1px solid {t["border_soft"]}; border-radius: 9px; gridline-color: {t["border_soft"]}; outline: 0; padding: 3px;
}}
QListWidget::item, QTreeWidget::item {{ padding: 6px; border-radius: 6px; }}
QListWidget::item:hover, QTreeWidget::item:hover {{ background: {t["glass_2"]}; color: {t["cyan"]}; }}
QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected, QTableView::item:selected {{ background: {t["accent"]}; color: #ffffff; }}
QHeaderView::section {{ background: {t["panel_2"]}; color: {t["gold_bright"]}; border: 0; border-right: 1px solid {t["border_soft"]}; border-bottom: 1px solid {t["border_soft"]}; padding: 7px; font-weight: 650; }}
QScrollArea {{ background: transparent; border: 0; }} QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }} QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{ background: {t["border"]}; border-radius: 4px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: {t["accent"]}; }} QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QSplitter::handle {{ background: {t["border_soft"]}; }} QSplitter::handle:hover {{ background: {t["accent"]}; }}
QLabel#sectionTitle {{ color: {t["text"]}; font-size: 13pt; font-weight: 750; }} QLabel#mutedLabel {{ color: {t["muted"]}; }}
QLabel#headerCaption {{ color: {t["muted"]}; font-size: 8pt; font-weight: 650; letter-spacing: .5px; }} QLabel#headerValue {{ color: {t["text"]}; font-weight: 650; }}
QFrame#cleanApplicationHeader {{ background: {t["glass"]}; border: 1px solid {t["border_soft"]}; border-radius: 14px; }}
QLabel#appBrandLabel {{ color: {t["text"]}; font-size: 13pt; font-weight: 800; }} QLabel#appVersionLabel {{ color: {t["cyan"]}; font-size: 8pt; font-weight: 650; }}
QFrame#headerDivider {{ color: {t["border_soft"]}; background: {t["border_soft"]}; max-width: 1px; }}
QLabel#saveStateSaved, QLabel#saveStateModified, QLabel#saveStateUnsaved, QLabel#workflowModeBadge {{ border-radius: 9px; padding: 4px 9px; font-weight: 700; }}
QLabel#saveStateSaved {{ color: {t["success"]}; border: 1px solid {t["success"]}; background: {t["glass_2"]}; }}
QLabel#saveStateModified {{ color: {t["warning"]}; border: 1px solid {t["warning"]}; background: {t["glass_2"]}; }}
QLabel#saveStateUnsaved {{ color: {t["muted"]}; border: 1px solid {t["border"]}; background: {t["glass_2"]}; }}
QLabel#workflowModeBadge {{ color: {t["cyan"]}; border: 1px solid {t["blue"]}; background: {t["glass_2"]}; }}
QWidget#workflowSidebar, QWidget#workflowSidebarContent, QWidget#centralScientificWorkspace, QWidget#analysisControlContent, QFrame#workflowSidebarNavigation, QFrame#progressDrawer {{ background: {t["glass"]}; border: 1px solid {t["border_soft"]}; border-radius: 13px; }}
QWidget#workflowSidebarContent {{ border: 0; border-radius: 0; }}
QFrame#workflowSidebarNavigation {{ background: transparent; border: 0; }}
QPushButton#workflowSidebarButton {{ text-align: left; background: transparent; border: 1px solid transparent; border-radius: 9px; padding: 9px 10px; font-weight: 620; }}
QPushButton#workflowSidebarButton:hover {{ background: {t["glass_2"]}; border-color: {t["border_soft"]}; color: {t["cyan"]}; }}
QPushButton#workflowSidebarButton:checked {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {t["panel_2"]}, stop:1 {t["glass_2"]}); color: {t["text"]}; border-color: {t["accent"]}; }}
QPushButton#workflowSidebarButton[workflowState="Outdated"], QPushButton#workflowSidebarButton[workflowState="Invalid"] {{ color: {t["danger"]}; }}
QListWidget#workflowTaskList {{ background: {t["input"]}; border-radius: 8px; }}
QLabel#workflowGuidance {{ color: {t["muted"]}; background: {t["glass_2"]}; border-left: 3px solid {t["accent"]}; border-radius: 7px; padding: 8px; }}
QLabel#workflowTaskNotice, QLabel#workflowHomeNotice {{ color: {t["text"]}; background: {t["glass_2"]}; border: 1px solid {t["border_soft"]}; border-left: 3px solid {t["accent"]}; border-radius: 8px; padding: 8px; }}
QLabel#workflowTaskNotice[taskState="Blocked"], QLabel#workflowTaskNotice[taskState="Outdated"], QLabel#workflowTaskNotice[taskState="Invalid"] {{ color: {t["danger"]}; border-left-color: {t["danger"]}; }}
QFrame#workflowHomeCards {{ background: {t["glass_2"]}; border: 1px solid {t["border_soft"]}; border-radius: 12px; }}
QFrame#homeQuickActions {{ background: {t["glass_2"]}; border: 1px solid {t["border_soft"]}; border-radius: 12px; }}
QLabel#homeQuickTitle {{ color: {t["text"]}; font-size: 11pt; font-weight: 750; }}
QFrame#recentProjectsPanel {{ background: {t["glass_2"]}; border: 1px solid {t["border_soft"]}; border-radius: 12px; }}
QLabel#recentProjectCount {{ color: {t["cyan"]}; background: {t["input"]}; border: 1px solid {t["border"]}; border-radius: 7px; padding: 2px 7px; font-weight: 750; }}
QPushButton#recentProjectClear {{ min-height: 22px; padding: 2px 8px; background: transparent; border: 1px solid transparent; color: {t["muted"]}; }}
QPushButton#recentProjectClear:hover {{ color: {t["cyan"]}; background: {t["glass"]}; border-color: {t["border_soft"]}; }}
QFrame#recentProjectRow {{ background: {t["input"]}; border: 1px solid {t["border_soft"]}; border-radius: 9px; }}
QFrame#recentProjectRow:hover {{ border-color: {t["accent"]}; background: {t["glass"]}; }}
QPushButton#recentProjectButton {{ min-height: 38px; padding: 4px 8px; text-align: left; background: transparent; border: 0; color: {t["text"]}; font-weight: 680; }}
QPushButton#recentProjectButton:hover {{ color: {t["cyan"]}; background: transparent; }}
QPushButton#recentProjectButton:disabled {{ color: {t["muted"]}; }}
QPushButton#recentProjectRemove {{ min-height: 28px; padding: 0; background: transparent; border: 1px solid transparent; color: {t["muted"]}; font-size: 13pt; }}
QPushButton#recentProjectRemove:hover {{ color: {t["danger"]}; background: {t["glass_2"]}; border-color: {t["danger"]}; }}
QFrame#dropImportOverlay {{ background: rgba(5, 9, 24, 218); border: 2px solid {t["cyan"]}; border-radius: 16px; }}
QFrame#dropImportCard {{ background: {t["panel"]}; border: 2px dashed {t["accent"]}; border-radius: 22px; }}
QLabel#dropImportIcon {{ color: {t["cyan"]}; font-size: 34pt; font-weight: 800; }}
QLabel#dropImportTitle {{ color: {t["text"]}; font-size: 19pt; font-weight: 850; }}
QLabel#dropImportDetail {{ color: {t["muted"]}; font-size: 10pt; }}
QPushButton#headerCommandButton {{ padding: 0; border-radius: 9px; }}
QFrame#progressDrawer {{ background: {t["glass"]}; }} QLabel#progressDrawerStatus {{ color: {t["muted"]}; }}
QPlainTextEdit#progressLog {{ font-family: "Cascadia Mono", "Consolas", monospace; font-size: 9pt; }}
QProgressBar {{ background: {t["input"]}; color: {t["text"]}; border: 1px solid {t["border"]}; border-radius: 5px; text-align: center; min-height: 8px; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {t["accent"]}, stop:.55 {t["blue"]}, stop:1 {t["cyan"]}); border-radius: 4px; }}
QFrame#plotNavigationToolbar {{ background: {t["panel"]}; border: 1px solid {t["border"]}; border-radius: 11px; }}
QToolButton#plotToolButton {{ min-width: 27px; max-width: 29px; min-height: 27px; max-height: 29px; padding: 0; background: transparent; border: 1px solid transparent; border-radius: 7px; }}
QToolButton#plotToolButton:hover {{ background: {t["glass_2"]}; border-color: {t["cyan"]}; }}
QToolButton#plotToolButton:pressed {{ padding: 0; background: {t["input"]}; border-color: {t["accent"]}; }}
QToolButton#plotToolButton:checked {{ background: {t["accent"]}; border-color: {t["gold_bright"]}; }}
QToolButton#plotToolButton:disabled {{ background: transparent; border-color: transparent; }}
QToolButton#plotToolButton::menu-indicator {{ image: none; width: 0; }}
QWidget#analysisPanelContainer {{ background: transparent; }}
QFrame#unifiedAnalysisHeader, QFrame#analysisActionBar, QFrame#analysisResultCard, QFrame#analysisValidationInfo, QFrame#analysisValidationSuccess, QFrame#analysisValidationWarning, QFrame#analysisValidationError {{ background: {t["glass_2"]}; border: 1px solid {t["border_soft"]}; border-radius: 11px; }}
QFrame#analysisInspectorToolbar {{ background: {t["glass_2"]}; border: 1px solid {t["border_soft"]}; border-radius: 11px; }}
QLineEdit#analysisParameterSearch {{ min-height: 27px; border: 1px solid {t["border"]}; border-radius: 9px; padding: 3px 8px; }}
QLineEdit#analysisParameterSearch:focus {{ border-color: {t["cyan"]}; background: {t["input"]}; }}
QLabel#analysisInspectorSummary {{ color: {t["muted"]}; font-size: 8.5pt; font-weight: 650; min-width: 72px; }}
QPushButton#analysisModeButton {{ min-height: 24px; padding: 3px 9px; background: transparent; border: 1px solid {t["border_soft"]}; color: {t["muted"]}; }}
QPushButton#analysisModeButton:hover {{ color: {t["cyan"]}; border-color: {t["border"]}; }}
QPushButton#analysisModeButton:checked {{ color: #ffffff; background: {t["accent"]}; border-color: {t["gold_bright"]}; font-weight: 750; }}
QPushButton#analysisInspectorUtilityButton {{ min-height: 22px; padding: 2px 7px; background: transparent; border: 1px solid transparent; color: {t["muted"]}; font-size: 8.5pt; }}
QPushButton#analysisInspectorUtilityButton:hover {{ color: {t["cyan"]}; background: {t["glass"]}; border-color: {t["border_soft"]}; }}
QFrame#analysisInspectorSection {{ background: {t["glass_2"]}; border: 1px solid {t["border_soft"]}; border-radius: 11px; }}
QFrame#analysisInspectorSection[inspectorTier="advanced"] {{ border-color: {t["accent"]}; }}
QFrame#analysisSectionHeader {{ background: transparent; border: 0; border-radius: 10px; }}
QPushButton#analysisSectionToggle {{ min-height: 24px; padding: 3px 4px; text-align: left; background: transparent; border: 0; color: {t["text"]}; font-weight: 750; }}
QPushButton#analysisSectionToggle:hover {{ color: {t["cyan"]}; background: {t["glass"]}; }}
QLabel#analysisSectionBadge {{ color: {t["muted"]}; background: {t["input"]}; border: 1px solid {t["border"]}; border-radius: 7px; padding: 2px 6px; font-size: 7.5pt; font-weight: 750; }}
QGroupBox[inspectorEmbedded="true"] {{ background: transparent; border: 0; border-top: 1px solid {t["border_soft"]}; border-radius: 0; margin-top: 0; padding-top: 5px; }}
QGroupBox[inspectorEmbedded="true"]::title {{ subcontrol-origin: margin; padding: 0; }}
QLabel#analysisPanelTitle, QLabel#analysisResultTitle {{ color: {t["text"]}; font-size: 11.5pt; font-weight: 750; }}
QLabel#analysisStageBadge {{ color: #ffffff; background: {t["accent"]}; border: 1px solid {t["gold_bright"]}; border-radius: 9px; padding: 3px 8px; font-size: 8.5pt; font-weight: 750; }}
QFrame#analysisValidationSuccess {{ border-left: 4px solid {t["success"]}; }} QFrame#analysisValidationInfo {{ border-left: 4px solid {t["blue"]}; }}
QFrame#analysisValidationWarning {{ border-left: 4px solid {t["warning"]}; }} QFrame#analysisValidationError {{ border-left: 4px solid {t["danger"]}; }}
QLabel#analysisValidationHeading {{ font-weight: 750; }}
QLabel#analysisResultSuccess, QLabel#analysisResultWarning, QLabel#analysisResultError, QLabel#analysisResultPending {{ border-radius: 8px; padding: 4px 8px; font-weight: 700; }}
QLabel#analysisResultSuccess {{ color: {t["success"]}; border: 1px solid {t["success"]}; background: {t["glass_2"]}; }}
QLabel#analysisResultWarning {{ color: {t["warning"]}; border: 1px solid {t["warning"]}; }} QLabel#analysisResultError {{ color: {t["danger"]}; border: 1px solid {t["danger"]}; }}
QLabel#analysisResultPending, QLabel#analysisMetricName {{ color: {t["muted"]}; }} QLabel#analysisMetricValue {{ color: {t["text"]}; font-weight: 700; }}
QDialog#commandPalette {{ background: transparent; }}
QFrame#commandPaletteFrame {{ background: {t["panel"]}; border: 1px solid {t["accent"]}; border-radius: 18px; }}
QLabel#commandPaletteTitle {{ color: {t["text"]}; font-size: 15pt; font-weight: 800; }}
QLabel#commandKeycap {{ color: {t["cyan"]}; background: {t["input"]}; border: 1px solid {t["border"]}; border-radius: 7px; padding: 4px 8px; font-size: 8pt; font-weight: 700; }}
QLabel#commandPaletteHint {{ color: {t["muted"]}; font-size: 8.5pt; }}
QLineEdit#commandSearch {{ min-height: 31px; border: 1px solid {t["accent"]}; border-radius: 10px; padding: 6px 10px; font-size: 11pt; }}
QLineEdit#commandSearch:focus {{ border: 1px solid {t["cyan"]}; }}
QListWidget#commandList {{ background: transparent; border: 0; padding: 0; outline: 0; }}
QListWidget#commandList::item {{ background: transparent; border: 0; padding: 0; }}
QToolTip {{ background: {t["panel"]}; color: {t["text"]}; border: 1px solid {t["accent"]}; padding: 5px; }}
"""


GOLD = DARK_GOLD["gold"]
GOLD_BRIGHT = DARK_GOLD["gold_bright"]
GRAPHITE = DARK_GOLD["graphite"]
PANEL = DARK_GOLD["panel"]
PANEL_2 = DARK_GOLD["panel_2"]
TEXT = DARK_GOLD["text"]
MUTED = DARK_GOLD["muted"]
BORDER = DARK_GOLD["border"]
DANGER = DARK_GOLD["danger"]
APP_STYLE = build_stylesheet(DEFAULT_THEME_NAME)
