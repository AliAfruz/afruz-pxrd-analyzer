from __future__ import annotations

import sys
import html
import csv
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QSize, QUrl, QThread
from PySide6.QtGui import QAction, QActionGroup, QIcon, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFrame,
    QLayout,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QGroupBox,
    QScrollArea,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QCheckBox,
    QLineEdit,
    QInputDialog,
    QHeaderView,
    QSizePolicy,
    QProgressBar,
)

from .context_copy import (
    install_global_copy_support,
    install_label_copy_menu,
    install_table_copy_menu,
)
from .theme import DEFAULT_THEME_NAME, THEMES, build_stylesheet
from .cinematic_ui import AuraBackdrop, CinematicUiController
from .command_palette import CommandItem, CommandPalette
from .project_access import (
    DropImportOverlay,
    RecentProjectsPanel,
    RecentProjectsStore,
    can_accept_drop_paths,
    classify_drop_paths,
)
from .version import APP_NAME, APP_VERSION, APP_WINDOW_TITLE, project_window_title
from .widgets import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)
from .models import Dataset
from .application import ApplicationState, ProjectController, ProjectState
from .application.workflow_controller import WorkflowController
from .services import ExportService, ImportService, ReportService
from .workspaces import WorkspaceRegistry
from .processing import (
    ProcessingParameters,
    process_pattern,
    detect_peaks,
    smart_detect_peaks,
    normalize_peak_rows,
    active_peak_rows,
    create_manual_peak,
    merge_detected_with_manual,
)
from .advanced_background import (
    BACKGROUND_METHODS,
    BackgroundParameters,
    detect_background,
)
from .background_plot import AdvancedBackgroundPlotWidget
from .smart_smoothing import (
    SMOOTHING_METHODS,
    SmoothingParameters,
    smart_smooth_pattern,
)
from .smoothing_plot import AdvancedSmoothingPlotWidget
from .fitting import fit_detected_peaks
from .deconvolution_worker import DeconvolutionWorker
from .size_strain import analyze_size_strain
from .size_strain_plot import SizeStrainPlotWidget
from .cell_refinement_plot import CellRefinementPlotWidget
from .crystallography import (
    CIFImportError,
    CRYSTAL_SYSTEMS,
    calculate_powder_pattern,
    load_cif,
    match_observed_to_reference,
    refine_unit_cell,
)
from .text_export import write_table_txt, write_columns_txt
from .export_center import ExportCenterDialog
from .plot_widget import XRDPlotWidget
from .phase_identification import (
    identify_phases,
    reference_from_cif_pattern,
    reference_from_dataset,
)
from .phase_identification_plot import PhaseIdentificationPlotWidget
from .quantitative_phase import (
    QPA_MODES,
    QPA_WEIGHTING,
    QPAPhaseSpec,
    quantify_phases,
)
from .quantitative_phase_plot import QuantitativePhasePlotWidget
from .residual_stress import (
    ELASTIC_MODES,
    PEAK_SOURCE_MODES,
    REFERENCE_MODES,
    REGRESSION_MODES,
    ResidualStressError,
    analyze_sin2psi,
    extract_peak_observation,
    infer_psi_deg,
    two_theta_to_d,
)
from .residual_stress_plot import ResidualStressPlotWidget
from .workflow import (
    TASK_BY_KEY,
    WORKSPACE_BY_KEY,
    build_task_states,
    build_workflow_status,
    recommended_task_key,
    tasks_for_workspace,
)
# UnifiedWorkflowHeader compatibility: Phase 20 uses CleanApplicationHeader.
from .workflow_shell import (
    CleanApplicationHeader,
    ProgressDrawer,
    ProjectHomeWidget,
    WorkflowNavigationPanel,
)
from .analysis_panel import (
    AnalysisActionBar,
    AnalysisInspectorSection,
    AnalysisInspectorToolbar,
    AnalysisPanelDescriptor,
    AnalysisResultCard,
    InlineValidationCard,
    UnifiedAnalysisHeader,
    capture_widget_defaults,
    restore_widget_defaults,
)
from .scientific_state import (
    ScientificStateRegistry,
    STAGE_LABELS as SCIENTIFIC_STAGE_LABELS,
    STAGE_ORDER as SCIENTIFIC_STAGE_ORDER,
)
from .scientific_state_widget import ScientificStateDialog


def safe_project_component(name: str) -> str:
    import re
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._")
    return cleaned or "afruz_project"


def _state_attribute(container_name: str, attribute_name: str):
    """Expose central state through the legacy MainWindow API."""

    def getter(window):
        return getattr(getattr(window, container_name), attribute_name)

    def setter(window, value):
        setattr(getattr(window, container_name), attribute_name, value)

    return property(getter, setter)
