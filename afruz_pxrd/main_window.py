from __future__ import annotations

from .main_window_dependencies import *
from .main_window_dependencies import _state_attribute
from .application.workflow_window import WorkflowWindowMixin
from .application.project_window import ProjectWindowMixin
from .application.analysis_panels import AnalysisPanelMixin
from .workspaces.workspace_shell import WorkspaceShellMixin
from .workspaces.preparation_workspace import PreparationWorkspaceMixin
from .workspaces.peaks_workspace import PeaksWorkspaceMixin
from .workspaces.phase_workspace import PhaseWorkspaceMixin
from .workspaces.refinement_workspace import RefinementWorkspaceMixin
from .workspaces.validation_workspace import ValidationWorkspaceMixin


class MainWindow(
    WorkflowWindowMixin,
    ProjectWindowMixin,
    AnalysisPanelMixin,
    WorkspaceShellMixin,
    PreparationWorkspaceMixin,
    PeaksWorkspaceMixin,
    PhaseWorkspaceMixin,
    RefinementWorkspaceMixin,
    ValidationWorkspaceMixin,
    QMainWindow,
):
    datasets = _state_attribute("project_state", "datasets")
    current_project = _state_attribute("project_state", "current_project")
    project_dirty = _state_attribute("project_state", "dirty")
    backgrounds = _state_attribute("project_state", "backgrounds")
    background_results = _state_attribute("project_state", "background_results")
    smoothing_results = _state_attribute("project_state", "smoothing_results")
    peak_rows = _state_attribute("project_state", "peak_rows")
    peak_list_meta_by_uid = _state_attribute("project_state", "peak_list_meta_by_uid")
    fit_groups = _state_attribute("project_state", "fit_groups")
    fit_candidates = _state_attribute("project_state", "fit_candidates")
    size_strain_results = _state_attribute("project_state", "size_strain_results")
    reference_structure = _state_attribute("project_state", "reference_structure")
    reference_pattern = _state_attribute("project_state", "reference_pattern")
    cell_refinement_results = _state_attribute("project_state", "cell_refinement_results")
    phase_identification_results = _state_attribute("project_state", "phase_identification_results")
    qpa_results = _state_attribute("project_state", "qpa_results")
    residual_stress_observations = _state_attribute("project_state", "residual_stress_observations")
    residual_stress_result = _state_attribute("project_state", "residual_stress_result")
    active_instrument_profile = _state_attribute("project_state", "active_instrument_profile")
    scientific_state = _state_attribute("project_state", "scientific_state")

    current_theme_name = _state_attribute("application_state", "theme_name")
    workflow_mode = _state_attribute("application_state", "workflow_mode")
    active_workflow_task = _state_attribute("application_state", "active_workflow_task")
    active_workspace_key = _state_attribute("application_state", "active_workspace_key")
    workflow_states = _state_attribute("application_state", "workflow_states")
    workflow_task_states = _state_attribute("application_state", "workflow_task_states")
    workflow_context = _state_attribute("application_state", "workflow_context")

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_WINDOW_TITLE)
        self.resize(1500, 900)
        self.setMinimumSize(1200, 640)

        self.project_state = ProjectState()
        self.application_state = ApplicationState(theme_name=DEFAULT_THEME_NAME)
        self.recent_projects_store = RecentProjectsStore()
        self.workspace_registry = WorkspaceRegistry(self.project_state)
        self.workflow_controller = WorkflowController(
            self.application_state,
            self.workspace_registry,
        )
        self.import_service = ImportService()
        self.export_service = ExportService()
        self.report_service = ReportService()
        self.project_controller = ProjectController(
            self.project_state,
            self.application_state,
            history_depth=50,
        )
        self.project_controller.configure_history(
            self._undo_snapshot,
            self._restore_undo_snapshot,
        )
        self.undo_redo_history = self.project_controller.history

        self._peak_table_updating = False
        self._qpa_setup_signature: tuple | None = None
        self._stress_table_updating = False
        self._workflow_syncing = False
        self._scientific_state_syncing = False
        self._undo_redo_restoring = False

        self._deconvolution_thread: QThread | None = None
        self._deconvolution_worker: DeconvolutionWorker | None = None
        self._deconvolution_dataset_uid: str | None = None
        self._deconvolution_dataset_name: str = ""

        self._build_actions()
        self._build_menu()
        self._build_toolbar()
        self._build_ui()
        self.setAcceptDrops(True)
        self.drop_import_overlay = DropImportOverlay(self.shell_root)
        self._refresh_recent_projects()
        self.cinematic_ui = CinematicUiController(self)
        self.cinematic_ui.install()
        install_global_copy_support(QApplication.instance(), self)
        self._connect_signals()
        self._bind_progress_drawer_sources()
        self.apply_theme(self.current_theme_name)
        self._update_undo_redo_actions()
        self.statusBar().showMessage("Ready — import a powder XRD pattern.", 0)


    # ---------- UI construction ----------
    def _build_actions(self):
        self.new_action = QAction("New Project", self)
        self.open_action = QAction("Open Project…", self)
        self.save_action = QAction("Save Project", self)
        self.save_as_action = QAction("Save Project As…", self)
        self.undo_action = QAction("Undo", self)
        self.redo_action = QAction("Redo", self)
        self.import_action = QAction("Import Pattern…", self)
        self.export_png_action = QAction("Export PNG…", self)
        self.export_svg_action = QAction("Export SVG…", self)
        self.export_center_action = QAction("Clean Data Export…", self)
        self.export_complete_report_action = QAction("Export Complete Report Package…", self)
        self.quit_action = QAction("Quit", self)
        self.home_action = QAction("Project Home", self)
        self.full_gui_action = QAction("Full GUI Workbench", self)
        self.cif_library_gui_action = QAction("CIF Library / Structure Match", self)
        self.multicomponent_gui_action = QAction("Intelligent Multiphase Refiner", self)
        self.next_step_action = QAction("Next Scientific Step", self)
        self.scientific_state_action = QAction("Scientific State…", self)
        self.reduced_motion_action = QAction("Reduce Motion", self)
        self.reduced_motion_action.setCheckable(True)
        self.fullscreen_action = QAction("Toggle Full Screen", self)
        self.fullscreen_action.setCheckable(True)
        self.fullscreen_action.setShortcut("F11")
        self.command_palette_action = QAction("Command Center…", self)
        self.command_palette_action.setShortcut("Ctrl+K")
        self.command_palette_action.setProperty("cinematicIconName", "search")
        self.toggle_left_sidebar_action = QAction("Show Workflow Sidebar", self)
        self.toggle_left_sidebar_action.setCheckable(True)
        self.toggle_left_sidebar_action.setChecked(True)
        self.toggle_left_sidebar_action.setShortcut("Ctrl+Shift+L")
        self.toggle_analysis_sidebar_action = QAction("Show Analysis Sidebar", self)
        self.toggle_analysis_sidebar_action.setCheckable(True)
        self.toggle_analysis_sidebar_action.setChecked(True)
        self.toggle_analysis_sidebar_action.setShortcut("Ctrl+Shift+R")
        self.accept_result_action = QAction("Accept Current Scientific Result", self)
        self.guided_mode_action = QAction("Guided Mode", self)
        self.expert_mode_action = QAction("Expert Mode", self)
        self.guided_mode_action.setCheckable(True)
        self.expert_mode_action.setCheckable(True)
        self.guided_mode_action.setChecked(True)
        self.workflow_mode_actions = QActionGroup(self)
        self.workflow_mode_actions.setExclusive(True)
        self.workflow_mode_actions.addAction(self.guided_mode_action)
        self.workflow_mode_actions.addAction(self.expert_mode_action)
        self.new_action.setShortcut("Ctrl+N")
        self.open_action.setShortcut("Ctrl+O")
        self.save_action.setShortcut("Ctrl+S")
        self.undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        self.redo_action.setShortcuts(
            [QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")]
        )
        self.import_action.setShortcut("Ctrl+I")
        self.export_center_action.setShortcut("Ctrl+Shift+E")
        self.export_complete_report_action.setShortcut("Ctrl+Alt+R")
        self.quit_action.setShortcut("Ctrl+Q")
        self.home_action.setShortcut("Ctrl+H")
        self.full_gui_action.setShortcut("Ctrl+Alt+W")
        self.multicomponent_gui_action.setShortcut("Ctrl+Alt+M")
        self.next_step_action.setShortcut("Ctrl+Alt+Right")
        self.scientific_state_action.setShortcut("Ctrl+Alt+S")

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        for action in (
            self.new_action,
            self.open_action,
            self.save_action,
            self.save_as_action,
            self.import_action,
        ):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_png_action)
        file_menu.addAction(self.export_svg_action)
        file_menu.addAction(self.export_center_action)
        file_menu.addAction(self.export_complete_report_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

        workflow_menu = self.menuBar().addMenu("&Workflow")
        workflow_menu.addAction(self.command_palette_action)
        workflow_menu.addSeparator()
        workflow_menu.addAction(self.home_action)
        workflow_menu.addAction(self.full_gui_action)
        workflow_menu.addAction(self.next_step_action)
        workflow_menu.addAction(self.scientific_state_action)
        workflow_menu.addAction(self.accept_result_action)
        workflow_menu.addSeparator()
        workflow_menu.addAction(self.guided_mode_action)
        workflow_menu.addAction(self.expert_mode_action)

        view_menu = self.menuBar().addMenu("&View")
        theme_menu = view_menu.addMenu("Theme")
        self.theme_actions = {}
        for theme_name in THEMES:
            action = QAction(theme_name, self)
            action.setCheckable(True)
            action.setChecked(theme_name == self.current_theme_name)
            action.triggered.connect(
                lambda checked=False, name=theme_name: self.apply_theme(name)
            )
            theme_menu.addAction(action)
            self.theme_actions[theme_name] = action
        view_menu.addSeparator()
        view_menu.addAction(self.reduced_motion_action)
        view_menu.addAction(self.fullscreen_action)
        view_menu.addSeparator()
        view_menu.addAction(self.toggle_left_sidebar_action)
        view_menu.addAction(self.toggle_analysis_sidebar_action)

        analysis_menu = self.menuBar().addMenu("&Analysis")
        apply_action = QAction("Apply preprocessing", self)
        apply_action.triggered.connect(self.apply_processing)
        preview_background_action = QAction("Preview advanced background", self)
        preview_background_action.triggered.connect(self.preview_background)
        open_background_action = QAction("Open Background", self)
        open_background_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(self.preprocessing_tab)
        )
        preview_smoothing_action = QAction("Preview smart smoothing", self)
        preview_smoothing_action.triggered.connect(self.preview_smoothing)
        open_smoothing_action = QAction("Open Smoothing", self)
        open_smoothing_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(self.smoothing_tab)
        )
        reset_action = QAction("Reset processing", self)
        reset_action.triggered.connect(self.reset_processing)
        peaks_action = QAction("Find peaks", self)
        peaks_action.triggered.connect(self.find_peaks_for_selected)
        fit_action = QAction("Fit detected peaks", self)
        fit_action.triggered.connect(self.fit_detected_peaks_for_selected)
        advanced_fit_action = QAction("Advanced peak deconvolution", self)
        advanced_fit_action.triggered.connect(
            self.advanced_fit_peaks_for_selected
        )
        size_strain_action = QAction("Calculate size and strain", self)
        size_strain_action.triggered.connect(
            self.calculate_size_strain_for_selected
        )
        import_cif_action = QAction("Import CIF reference…", self)
        import_cif_action.triggered.connect(self.import_cif_reference)
        refine_cell_action = QAction("Match peaks and refine cell", self)
        refine_cell_action.triggered.connect(
            self.match_and_refine_cell_for_selected
        )
        identify_phases_action = QAction("Identify phases from local library", self)
        identify_phases_action.triggered.connect(
            self.identify_phases_for_selected
        )
        quantify_phases_action = QAction("Quantify selected phase candidate", self)
        quantify_phases_action.triggered.connect(
            self.quantify_selected_phases
        )
        residual_stress_action = QAction("Calculate sin²ψ residual stress", self)
        residual_stress_action.triggered.connect(
            self.calculate_residual_stress
        )
        calibration_action = QAction("Open instrument calibration", self)
        calibration_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(self.calibration_widget)
        )
        whole_pattern_action = QAction(
            "Open Pawley / Le Bail refinement",
            self,
        )
        whole_pattern_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(self.whole_pattern_widget)
        )
        rietveld_action = QAction(
            "Open Rietveld refinement",
            self,
        )
        rietveld_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(self.rietveld_widget)
        )
        doping_series_action = QAction(
            "Open Doping-Series Comparison",
            self,
        )
        doping_series_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(self.doping_series_widget)
        )
        validation_action = QAction(
            "Open Validation",
            self,
        )
        validation_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(self.validation_widget)
        )
        phase_revolution_action = QAction(
            "Open Unknown Phase",
            self,
        )
        phase_revolution_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(self.phase_revolution_widget)
        )
        cif_library_action = QAction(
            "Open CIF Library / Structure Match",
            self,
        )
        cif_library_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(self.cif_library_widget)
        )
        structure_solution_action = QAction(
            "Open Solve Structure",
            self,
        )
        structure_solution_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(
                self.structure_solution_widget
            )
        )
        multicomponent_refiner_action = QAction(
            "Open Intelligent Multiphase Refiner",
            self,
        )
        multicomponent_refiner_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(
                self.multicomponent_refiner_widget
            )
        )
        batch_workflow_action = QAction("Open Batch & Reports", self)
        batch_workflow_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(self.batch_widget)
        )
        preparation_menu = analysis_menu.addMenu("Pattern Preparation")
        preparation_menu.addActions(
            [
                preview_background_action,
                open_background_action,
                preview_smoothing_action,
                open_smoothing_action,
                apply_action,
                reset_action,
                calibration_action,
            ]
        )
        peaks_menu = analysis_menu.addMenu("Peaks")
        peaks_menu.addActions(
            [peaks_action, fit_action, advanced_fit_action, size_strain_action, residual_stress_action]
        )
        phase_menu = analysis_menu.addMenu("Phase & Structure")
        phase_menu.addActions(
            [identify_phases_action, import_cif_action, refine_cell_action, phase_revolution_action, cif_library_action, structure_solution_action]
        )
        refinement_menu = analysis_menu.addMenu("Refinement & Quantification")
        refinement_menu.addActions(
            [
                whole_pattern_action,
                rietveld_action,
                doping_series_action,
                multicomponent_refiner_action,
                quantify_phases_action,
            ]
        )
        evidence_menu = analysis_menu.addMenu("Validation & Reports")
        evidence_menu.addActions([validation_action, batch_workflow_action])

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("About", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)

    def _build_toolbar(self):
        # Phase 20 replaces the crowded action toolbar with the compact application
        # header. The toolbar object is retained, hidden, for shortcut/action
        # compatibility with older project builds.
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        for action in (
            self.new_action,
            self.open_action,
            self.save_action,
            self.import_action,
            self.undo_action,
            self.redo_action,
            self.home_action,
            self.full_gui_action,
            self.multicomponent_gui_action,
            self.next_step_action,
        ):
            toolbar.addAction(action)
        toolbar.setVisible(False)

    def _build_ui(self):
        self.shell_root = AuraBackdrop()
        self.shell_root.setObjectName("cleanApplicationShell")
        shell_layout = QVBoxLayout(self.shell_root)
        shell_layout.setContentsMargins(8, 7, 8, 7)
        shell_layout.setSpacing(7)
        self.setCentralWidget(self.shell_root)

        self.workflow_header = CleanApplicationHeader(self)
        shell_layout.addWidget(self.workflow_header)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setObjectName("mainWorkspaceSplitter")
        self.main_splitter.setHandleWidth(6)
        shell_layout.addWidget(self.main_splitter, 1)

        self.left_panel = self._build_left_panel()
        self.center_panel = self._build_center_panel()
        self.right_panel = self._build_right_panel()
        self.workflow_controller.set_dataset(self.selected_dataset())
        self.workflow_controller.refresh(
            (
                "instrument",
                "pawley_lebail",
                "rietveld",
                "doping_series",
                "unknown_phase",
                "multicomponent_refiner",
                "full_gui",
                "batch_reports",
            )
        )

        self.left_panel.setMinimumWidth(260)
        self.center_panel.setMinimumWidth(620)
        self.right_panel.setMinimumWidth(300)
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.center_panel)
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setSizes([290, 1000, 340])
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setCollapsible(0, True)
        self.main_splitter.setCollapsible(1, False)
        self.main_splitter.setCollapsible(2, True)

        self.progress_drawer = ProgressDrawer(self)
        self.progress_drawer.setVisible(True)
        shell_layout.addWidget(self.progress_drawer, 0)
        self._sync_analysis_panel_for_task()

    def _build_left_panel(self):
        scroll = QScrollArea()
        scroll.setObjectName("workflowSidebar")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)

        panel = QWidget()
        panel.setObjectName("workflowSidebarContent")
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout(panel)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(7)

        self.workflow_navigation = WorkflowNavigationPanel(self)
        layout.addWidget(self.workflow_navigation)

        file_group = QGroupBox("Project")
        file_layout = QGridLayout(file_group)
        self.new_button = QPushButton("New")
        self.open_button = QPushButton("Open .AFZ")
        self.save_button = QPushButton("Save .AFZ")
        self.import_button = QPushButton("Import XRD")
        self.import_button.setObjectName("primaryButton")
        file_layout.addWidget(self.new_button, 0, 0)
        file_layout.addWidget(self.open_button, 0, 1)
        file_layout.addWidget(self.save_button, 1, 0)
        file_layout.addWidget(self.import_button, 1, 1)
        layout.addWidget(file_group)

        data_group = QGroupBox("Datasets")
        data_layout = QVBoxLayout(data_group)
        self.dataset_list = QListWidget()
        self.dataset_list.setAlternatingRowColors(True)
        self.dataset_list.setSelectionMode(QAbstractItemView.SingleSelection)
        data_layout.addWidget(self.dataset_list)

        button_row = QGridLayout()
        self.rename_button = QPushButton("Rename")
        self.duplicate_button = QPushButton("Duplicate")
        self.remove_button = QPushButton("Delete")
        self.reset_button = QPushButton("Reset")
        button_row.addWidget(self.rename_button, 0, 0)
        button_row.addWidget(self.duplicate_button, 0, 1)
        button_row.addWidget(self.remove_button, 1, 0)
        button_row.addWidget(self.reset_button, 1, 1)
        data_layout.addLayout(button_row)
        self.dataset_list.setMinimumHeight(112)
        layout.addWidget(data_group)

        info_group = QGroupBox("Selected Dataset")
        info_layout = QFormLayout(info_group)
        self.info_name = QLabel("—")
        self.info_points = QLabel("—")
        self.info_range = QLabel("—")
        self.info_source = QLabel("—")
        self.info_source.setWordWrap(False)
        self.info_source.setTextFormat(Qt.RichText)
        self.info_source.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.info_source.setOpenExternalLinks(False)
        self.info_source.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        self.info_source.setMinimumWidth(0)
        self.info_source.linkActivated.connect(self.open_selected_source_folder)

        install_label_copy_menu(self.info_name)
        install_label_copy_menu(self.info_points)
        install_label_copy_menu(self.info_range)
        install_label_copy_menu(
            self.info_source,
            text_provider=lambda: (
                self.selected_dataset().source_path
                if self.selected_dataset() is not None
                else ""
            ),
        )

        info_layout.addRow("Name:", self.info_name)
        info_layout.addRow("Points:", self.info_points)
        info_layout.addRow("2θ range:", self.info_range)
        info_layout.addRow("Source:", self.info_source)
        layout.addWidget(info_group)
        layout.addStretch(1)
        scroll.setWidget(panel)
        self.left_panel_content = panel
        return scroll














    # ---------- Phase 20 clean shell ----------







    # ---------- undo / redo ----------
















    # ---------- Phase 20.2 unified analysis panels ----------












    def _connect_signals(self):
        self.theme_selector.currentTextChanged.connect(self.apply_theme)
        self.reduced_motion_action.toggled.connect(self._set_reduced_motion)
        self.fullscreen_action.toggled.connect(self._set_full_screen)
        self.command_palette_action.triggered.connect(self.show_command_palette)
        self.toggle_left_sidebar_action.toggled.connect(self._set_left_sidebar_visible)
        self.toggle_analysis_sidebar_action.toggled.connect(self._set_analysis_sidebar_visible)
        self.workflow_header.runRequested.connect(self._run_current_workflow_action)
        self.workflow_header.saveRequested.connect(self.save_project)
        self.workflow_header.statusRequested.connect(self.show_scientific_state)
        self.workflow_header.commandRequested.connect(self.show_command_palette)
        self.progress_drawer.cancellationRequested.connect(self._cancel_active_operation)
        self.statusBar().messageChanged.connect(self.progress_drawer.set_status)
        if hasattr(self, "analysis_action_bar"):
            self.statusBar().messageChanged.connect(self.analysis_action_bar.set_status)
        self.new_action.triggered.connect(self.new_project)
        self.open_action.triggered.connect(self.open_project)
        self.save_action.triggered.connect(self.save_project)
        self.save_as_action.triggered.connect(lambda: self.save_project(save_as=True))
        self.undo_action.triggered.connect(self.undo_last_change)
        self.redo_action.triggered.connect(self.redo_last_change)
        self.import_action.triggered.connect(self.import_patterns)
        self.export_png_action.triggered.connect(self.export_png)
        self.export_svg_action.triggered.connect(self.export_svg)
        self.export_center_action.triggered.connect(self.show_export_center)
        self.export_complete_report_action.triggered.connect(self.export_complete_report_package)
        self.quit_action.triggered.connect(self.close)
        self.home_action.triggered.connect(
            lambda: self._select_workflow_task("project_home")
        )
        self.full_gui_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(self.full_gui_workbench_widget)
        )
        self.cif_library_gui_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(self.cif_library_widget)
        )
        self.multicomponent_gui_action.triggered.connect(
            lambda: self.tabs.setCurrentWidget(self.multicomponent_refiner_widget)
        )
        self.next_step_action.triggered.connect(self._continue_workflow)
        self.scientific_state_action.triggered.connect(self.show_scientific_state)
        self.accept_result_action.triggered.connect(self.accept_current_scientific_result)
        self.guided_mode_action.triggered.connect(
            lambda: self._set_workflow_mode("Guided")
        )
        self.expert_mode_action.triggered.connect(
            lambda: self._set_workflow_mode("Expert")
        )
        self.new_button.clicked.connect(self.new_project)
        self.open_button.clicked.connect(self.open_project)
        self.save_button.clicked.connect(self.save_project)
        self.import_button.clicked.connect(self.import_patterns)

        self.dataset_list.currentRowChanged.connect(self.update_dataset_info)
        self.dataset_list.itemChanged.connect(self.on_dataset_item_changed)
        self.rename_button.clicked.connect(self.rename_selected)
        self.duplicate_button.clicked.connect(self.duplicate_selected)
        self.remove_button.clicked.connect(self.remove_selected)
        self.reset_button.clicked.connect(self.reset_processing)

        self.apply_button.clicked.connect(self.apply_processing)
        self.preview_background_button.clicked.connect(self.preview_background)
        self.background_preview_tab_button.clicked.connect(self.preview_background)
        self.background_apply_tab_button.clicked.connect(self.apply_processing)
        self.background_export_button.clicked.connect(
            lambda checked=False: self.show_export_center(
                preferred_formats={"excel", "text", "zip"},
                preferred_categories={"project", "pattern", "preprocessing"},
            )
        )
        self.preview_smoothing_button.clicked.connect(self.preview_smoothing)
        self.smoothing_preview_tab_button.clicked.connect(self.preview_smoothing)
        self.smoothing_apply_tab_button.clicked.connect(self.apply_smoothing_preview)
        self.smoothing_revert_button.clicked.connect(self.revert_smoothing)
        self.smoothing_export_button.clicked.connect(
            lambda checked=False: self.show_export_center(
                preferred_formats={"excel", "text", "zip"},
                preferred_categories={"project", "pattern", "preprocessing"},
            )
        )
        self.find_peaks_button.clicked.connect(self.find_peaks_for_selected)
        self.smart_peaks_button.clicked.connect(
            self.smart_find_peaks_for_selected
        )
        self.add_manual_peak_button.clicked.connect(
            lambda: self.add_manual_peak_for_selected(
                self.manual_peak_position.value(),
                origin="Manual numeric entry",
            )
        )
        self.add_main_peak_button.clicked.connect(
            self.add_manual_peak_dialog
        )
        self.peak_plot_find_button.clicked.connect(
            self.find_peaks_for_selected
        )
        self.peak_plot_smart_button.clicked.connect(
            self.smart_find_peaks_for_selected
        )
        self.peak_plot_fit_button.clicked.connect(
            self.fit_detected_peaks_for_selected
        )
        self.peak_plot_advanced_fit_button.clicked.connect(
            self.advanced_fit_peaks_for_selected
        )
        self.peak_plot_zoom_button.clicked.connect(
            self.zoom_peak_workspace_selection
        )
        self.peak_plot_auto_range_button.clicked.connect(
            self.auto_range_peak_workspace
        )
        self.delete_main_peaks_button.clicked.connect(
            self.delete_selected_main_peaks
        )
        self.include_main_peaks_button.clicked.connect(
            lambda: self.set_selected_main_peaks_use(True)
        )
        self.exclude_main_peaks_button.clicked.connect(
            lambda: self.set_selected_main_peaks_use(False)
        )
        self.send_main_peaks_to_revolution_button.clicked.connect(
            self.send_main_peak_list_to_phase_revolution
        )
        self.freeze_main_peak_list.toggled.connect(
            self._main_peak_freeze_toggled
        )
        self.peak_table.itemChanged.connect(
            self._main_peak_table_item_changed
        )
        self.peak_table.itemSelectionChanged.connect(
            self._peak_workspace_table_selection_changed
        )
        self.fit_table.itemSelectionChanged.connect(
            self._peak_workspace_table_selection_changed
        )
        self.peak_results_tabs.currentChanged.connect(
            self._peak_workspace_table_selection_changed
        )
        self.plot_widget.peakAddRequested.connect(
            self._plot_manual_peak_requested
        )
        self.plot_widget.contextPeakAddRequested.connect(
            lambda x, y: self._context_manual_peak_requested(
                x, y, origin="Main pattern right-click"
            )
        )
        self.peak_plot_widget.peakAddRequested.connect(
            self._plot_manual_peak_requested
        )
        self.peak_plot_widget.contextPeakAddRequested.connect(
            lambda x, y: self._context_manual_peak_requested(
                x, y, origin="Peak workspace right-click"
            )
        )
        self.background_plot.peakAddRequested.connect(
            lambda x, y: self._context_manual_peak_requested(
                x, y, origin="Background plot right-click"
            )
        )
        self.smoothing_plot.peakAddRequested.connect(
            lambda x, y: self._context_manual_peak_requested(
                x, y, origin="Smoothing plot right-click"
            )
        )
        self.phase_identification_plot.peakAddRequested.connect(
            lambda x, y: self._context_manual_peak_requested(
                x, y, origin="Phase-identification plot right-click"
            )
        )
        self.cell_refinement_plot.peakAddRequested.connect(
            lambda x, y: self._context_manual_peak_requested(
                x, y, origin="Cell-refinement plot right-click"
            )
        )
        self.view_peak_table_button.clicked.connect(self.show_peak_table)
        self.fit_peaks_button.clicked.connect(
            self.fit_detected_peaks_for_selected
        )
        self.advanced_fit_button.clicked.connect(
            self.advanced_fit_peaks_for_selected
        )
        self.cancel_deconvolution_button.clicked.connect(
            self.cancel_advanced_deconvolution
        )
        self.clear_fits_button.clicked.connect(self.clear_fits_for_selected)
        self.view_fit_table_button.clicked.connect(self.show_fit_table)
        self.view_model_selection_button.clicked.connect(
            self.show_model_selection_table
        )
        self.export_fit_table_button.clicked.connect(
            lambda checked=False: self.show_export_center(
                preferred_formats={"excel", "text", "zip"},
                preferred_categories={"project", "pattern", "peaks"},
            )
        )
        self.use_metadata_wavelength_button.clicked.connect(
            self.use_dataset_wavelength
        )
        self.calculate_size_button.clicked.connect(
            self.calculate_size_strain_for_selected
        )
        self.clear_size_button.clicked.connect(
            self.clear_size_strain_for_selected
        )
        self.view_size_button.clicked.connect(self.show_size_strain_results)
        self.export_size_button.clicked.connect(
            lambda checked=False: self.show_export_center(
                preferred_formats={"excel", "text", "zip"},
                preferred_categories={"project", "pattern", "peaks"},
            )
        )
        self.import_cif_button.clicked.connect(self.import_cif_reference)
        self.calculate_reference_button.clicked.connect(
            self.calculate_reference_pattern
        )
        self.clear_reference_button.clicked.connect(
            self.clear_cif_reference
        )
        self.match_refine_button.clicked.connect(
            self.match_and_refine_cell_for_selected
        )
        self.view_crystal_button.clicked.connect(
            self.show_crystal_results
        )
        self.export_reference_button.clicked.connect(
            lambda checked=False: self.show_export_center(
                preferred_formats={"excel", "text", "zip"},
                preferred_categories={"project", "pattern", "structure"},
            )
        )
        self.export_cell_button.clicked.connect(
            lambda checked=False: self.show_export_center(
                preferred_formats={"excel", "text", "zip"},
                preferred_categories={"project", "pattern", "structure"},
            )
        )
        self.identify_phases_button.clicked.connect(
            self.identify_phases_for_selected
        )
        self.clear_phase_results_button.clicked.connect(
            self.clear_phase_identification_for_selected
        )
        self.view_phase_results_button.clicked.connect(
            self.show_phase_identification_results
        )
        self.export_phase_results_button.clicked.connect(
            lambda checked=False: self.show_export_center(
                preferred_formats={"excel", "text", "zip"},
                preferred_categories={"project", "pattern", "structure"},
            )
        )
        self.phase_candidate_table.itemSelectionChanged.connect(
            self.update_selected_phase_candidate
        )
        self.run_qpa_button.clicked.connect(self.quantify_selected_phases)
        self.clear_qpa_button.clicked.connect(self.clear_qpa_for_selected)
        self.view_qpa_button.clicked.connect(self.show_qpa_results)
        self.export_qpa_button.clicked.connect(
            lambda checked=False: self.show_export_center(
                preferred_formats={"excel", "text", "zip"},
                preferred_categories={"project", "pattern", "qpa"},
            )
        )
        self.qpa_setup_table.itemChanged.connect(self._qpa_setup_item_changed)
        self.build_stress_observations_button.clicked.connect(
            self.build_residual_stress_observations
        )
        self.calculate_stress_button.clicked.connect(
            self.calculate_residual_stress
        )
        self.clear_stress_button.clicked.connect(self.clear_residual_stress)
        self.view_stress_button.clicked.connect(self.show_residual_stress_results)
        self.export_stress_button.clicked.connect(
            lambda checked=False: self.show_export_center(
                preferred_formats={"excel", "text", "zip"},
                preferred_categories={"project", "pattern", "validation"},
            )
        )
        self.stress_use_metadata_wavelength_button.clicked.connect(
            self.use_stress_dataset_wavelength
        )
        self.stress_observation_table.itemChanged.connect(
            self._stress_observation_item_changed
        )
        self.show_reference_check.toggled.connect(self.redraw)
        self.show_fits_check.toggled.connect(self.redraw)
        self.show_fit_components_check.toggled.connect(self.redraw)
        self.show_fit_baselines_check.toggled.connect(self.redraw)
        self.show_residuals_check.toggled.connect(self.redraw)
        self.stack_check.toggled.connect(self.redraw)
        self.stack_offset.valueChanged.connect(self.redraw)
        self.stack_scale.valueChanged.connect(self.redraw)
        self.auto_stack_button.clicked.connect(self.estimate_stack_offset)

        self.export_png_button.clicked.connect(self.export_png)
        self.export_svg_button.clicked.connect(self.export_svg)
        self.export_csv_button.clicked.connect(self.export_selected_csv)
        self.export_excel_button.clicked.connect(self.export_selected_excel)
        self.export_center_button.clicked.connect(self.show_export_center)

    def apply_theme(self, theme_name: str):
        if theme_name not in THEMES:
            theme_name = DEFAULT_THEME_NAME

        self.current_theme_name = theme_name
        QApplication.instance().setStyleSheet(
            build_stylesheet(theme_name)
        )
        if hasattr(self, "cinematic_ui"):
            self.cinematic_ui.apply_theme(theme_name)
        if hasattr(self, "command_palette"):
            self.command_palette.set_theme(theme_name)
        if hasattr(self, "shell_root") and hasattr(self.shell_root, "set_theme"):
            self.shell_root.set_theme(theme_name)
        self.plot_widget.apply_theme(theme_name)
        if hasattr(self, "peak_plot_widget"):
            self.peak_plot_widget.apply_theme(theme_name)
        if hasattr(self, "background_plot"):
            self.background_plot.apply_theme(theme_name)
        if hasattr(self, "smoothing_plot"):
            self.smoothing_plot.apply_theme(theme_name)
        if hasattr(self, "size_strain_plot"):
            self.size_strain_plot.apply_theme(theme_name)
        if hasattr(self, "cell_refinement_plot"):
            self.cell_refinement_plot.apply_theme(theme_name)
        if hasattr(self, "phase_identification_plot"):
            self.phase_identification_plot.apply_theme(theme_name)
        if hasattr(self, "qpa_plot"):
            self.qpa_plot.apply_theme(theme_name)
        if hasattr(self, "residual_stress_plot"):
            self.residual_stress_plot.apply_theme(theme_name)
        if hasattr(self, "calibration_widget"):
            self.calibration_widget.apply_theme(theme_name)
        if hasattr(self, "whole_pattern_widget"):
            self.whole_pattern_widget.apply_theme(theme_name)
        if hasattr(self, "rietveld_widget"):
            self.rietveld_widget.apply_theme(theme_name)
        if hasattr(self, "doping_series_widget"):
            self.doping_series_widget.apply_theme(theme_name)
        if hasattr(self, "validated_qpa_widget"):
            self.validated_qpa_widget.apply_theme(theme_name)
        if hasattr(self, "validation_widget"):
            self.validation_widget.apply_theme(theme_name)
        if hasattr(self, "phase_revolution_widget"):
            self.phase_revolution_widget.apply_theme(theme_name)
        if hasattr(self, "cif_library_widget"):
            self.cif_library_widget.apply_theme(theme_name)
        if hasattr(self, "structure_solution_widget"):
            self.structure_solution_widget.apply_theme(theme_name)
        if hasattr(self, "multicomponent_refiner_widget"):
            self.multicomponent_refiner_widget.apply_theme(theme_name)
        if hasattr(self, "full_gui_workbench_widget"):
            self.full_gui_workbench_widget.apply_theme(theme_name)
        if hasattr(self, "batch_widget"):
            self.batch_widget.apply_theme(theme_name)

        if hasattr(self, "theme_selector"):
            self.theme_selector.blockSignals(True)
            self.theme_selector.setCurrentText(theme_name)
            self.theme_selector.blockSignals(False)

        for name, action in getattr(self, "theme_actions", {}).items():
            action.blockSignals(True)
            action.setChecked(name == theme_name)
            action.blockSignals(False)

        self.statusBar().showMessage(f"Theme changed to {theme_name}.", 0)

    def _set_reduced_motion(self, reduced: bool) -> None:
        if hasattr(self, "cinematic_ui"):
            self.cinematic_ui.set_reduced_motion(bool(reduced))
        if hasattr(self, "progress_drawer") and hasattr(self.progress_drawer, "set_reduced_motion"):
            self.progress_drawer.set_reduced_motion(bool(reduced))
        self.statusBar().showMessage(
            "Reduced motion enabled." if reduced else "Cinematic motion enabled.", 0
        )

    def _set_full_screen(self, enabled: bool) -> None:
        if enabled:
            self.showFullScreen()
            message = "Full-screen workspace enabled. Press F11 to restore the maximized window."
        else:
            self.showMaximized()
            message = "Maximized workspace restored."
        self.statusBar().showMessage(message, 0)

    def _set_left_sidebar_visible(self, visible: bool) -> None:
        self.left_panel.setVisible(bool(visible))
        self.statusBar().showMessage(
            "Workflow sidebar shown." if visible else "Workflow sidebar hidden. Ctrl+Shift+L restores it.",
            2600,
        )

    def _set_analysis_sidebar_visible(self, visible: bool) -> None:
        self.right_panel.setVisible(bool(visible))
        self.statusBar().showMessage(
            "Analysis sidebar shown." if visible else "Analysis sidebar hidden. Ctrl+Shift+R restores it.",
            2600,
        )

    def _command_center_items(self) -> list[CommandItem]:
        action_specs = (
            (self.import_action, "Project", "upload", "Import one or more powder patterns"),
            (self.open_action, "Project", "open", "Open an existing .AFZ project"),
            (self.new_action, "Project", "new", "Create a clean project"),
            (self.save_action, "Project", "save", "Save the current project"),
            (self.save_as_action, "Project", "save", "Save a new project copy"),
            (self.export_center_action, "Export", "download", "Open Clean Data Export"),
            (self.export_complete_report_action, "Export", "report", "Build a reproducible report package"),
            (self.next_step_action, "Workflow", "next", "Continue to the recommended scientific step"),
            (self.scientific_state_action, "Validation", "status", "Audit scientific state and provenance"),
            (self.accept_result_action, "Validation", "check", "Accept the current scientific result"),
            (self.full_gui_action, "Workspace", "grid", "Open the complete scientific workbench"),
            (self.cif_library_gui_action, "Workspace", "database", "Search and manage CIF structures"),
            (self.multicomponent_gui_action, "Workspace", "layers", "Open intelligent multiphase refinement"),
            (self.fullscreen_action, "View", "fullscreen", "Toggle distraction-free full screen"),
            (self.toggle_left_sidebar_action, "View", "layers", "Show or hide the workflow sidebar"),
            (self.toggle_analysis_sidebar_action, "View", "sliders", "Show or hide contextual controls"),
            (self.reduced_motion_action, "View", "eye", "Reduce decorative interface motion"),
        )
        commands = [
            CommandItem(
                label=action.text().replace("&", "").rstrip("…"),
                category=category,
                handler=action.trigger,
                icon=icon,
                shortcut=action.shortcut().toString(),
                detail=detail,
                keywords=(action.toolTip(),),
                enabled=action.isEnabled,
            )
            for action, category, icon, detail in action_specs
        ]

        workspace_icons = {
            "home": "home",
            "preparation": "sliders",
            "peaks": "chart",
            "phase_structure": "crystal",
            "refinement_qpa": "layers",
            "validation_reports": "status",
        }
        for task in TASK_BY_KEY.values():
            workspace = WORKSPACE_BY_KEY[task.workspace]
            commands.append(
                CommandItem(
                    label=task.label,
                    category=f"Go to · {workspace.label}",
                    handler=lambda key=task.key: self._select_workflow_task(key),
                    icon=workspace_icons.get(task.workspace, "sparkles"),
                    detail=task.guidance,
                    keywords=(task.key.replace("_", " "), workspace.description),
                )
            )

        recent_store = getattr(self, "recent_projects_store", None)
        if recent_store is not None:
            for entry in recent_store.entries(limit=5):
                commands.append(
                    CommandItem(
                        label=f"Open {entry.name}",
                        category="Recent projects",
                        handler=lambda path=str(entry.path): self._open_recent_project(path),
                        icon="open",
                        detail=str(entry.path),
                        keywords=("recent", "project", entry.path.parent.name),
                        enabled=entry.exists,
                    )
                )

        commands.append(
            CommandItem(
                label="3D Crystal Studio",
                category="Visualize",
                handler=self.rietveld_widget.crystal_studio_button.click,
                icon="crystal",
                detail="Open cinematic and scientific crystal-structure rendering",
                keywords=("cif", "structure", "polyhedra", "hydrogen bonds"),
                enabled=self.rietveld_widget.crystal_studio_button.isEnabled,
            )
        )
        for theme_name, action in self.theme_actions.items():
            commands.append(
                CommandItem(
                    label=f"Use {theme_name} theme",
                    category="Appearance",
                    handler=action.trigger,
                    icon="sparkles",
                    detail="Switch the complete workbench color system",
                    keywords=("light", "dark", "color", "aura", "gold"),
                )
            )
        return commands

    def show_command_palette(self) -> None:
        if not hasattr(self, "command_palette"):
            self.command_palette = CommandPalette(self)
            self.command_palette.commandTriggered.connect(
                lambda label: self.statusBar().showMessage(f"Command: {label}", 2200)
            )
            self.cinematic_ui.decorate(self.command_palette)
        self.command_palette.set_theme(self.current_theme_name)
        self.command_palette.set_commands(self._command_center_items())
        self.command_palette.open_centered(
            reduced_motion=self.cinematic_ui.reduced_motion
        )

    # ---------- dataset helpers ----------








    # ---------- file actions ----------







    # ---------- processing ----------



































































































    # ---------- dataset editing ----------



    # ---------- complete scientific report ----------




    # ---------- export ----------









def run() -> int:
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyleSheet(build_stylesheet(DEFAULT_THEME_NAME))
    window = MainWindow()
    window.showMaximized()
    return app.exec()
