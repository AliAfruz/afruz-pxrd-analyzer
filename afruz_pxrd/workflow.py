from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping


@dataclass(frozen=True)
class WorkflowTask:
    key: str
    label: str
    workspace: str
    guided: bool
    guidance: str


@dataclass(frozen=True)
class WorkflowWorkspace:
    key: str
    label: str
    short_label: str
    description: str


@dataclass(frozen=True)
class WorkflowTaskState:
    """Task-level workflow state used by the Phase 20.1 guided shell."""

    key: str
    status: str
    enabled: bool
    completed: bool = False
    recommended: bool = False
    reason: str = ""
    missing: tuple[str, ...] = ()

    @property
    def symbol(self) -> str:
        return {
            "Complete": "✓",
            "Recommended": "→",
            "Available": "•",
            "Outdated": "⚠",
            "Invalid": "!",
            "Blocked": "×",
        }.get(self.status, "—")


WORKSPACES: tuple[WorkflowWorkspace, ...] = (
    WorkflowWorkspace(
        "home",
        "Project Home",
        "Home",
        "Review the selected dataset, workflow status, warnings and the recommended next action.",
    ),
    WorkflowWorkspace(
        "preparation",
        "Pattern Preparation",
        "Prepare",
        "Create one authoritative prepared pattern from the raw measurement.",
    ),
    WorkflowWorkspace(
        "peaks",
        "Peaks",
        "Peaks",
        "Curate one authoritative reflection list and perform peak-based analyses.",
    ),
    WorkflowWorkspace(
        "phase_structure",
        "Phase & Structure",
        "Phases",
        "Identify known phases or index and solve an unknown structure.",
    ),
    WorkflowWorkspace(
        "refinement_qpa",
        "Refinement & Quantification",
        "Refine",
        "Verify cells and structures, refine profiles and quantify phases.",
    ),
    WorkflowWorkspace(
        "validation_reports",
        "Validation & Reports",
        "Validate",
        "Audit robustness, preserve evidence and export reproducible reports.",
    ),
)


TASKS: tuple[WorkflowTask, ...] = (
    WorkflowTask("project_home", "Project overview", "home", True, "Review status and continue from the first incomplete scientific step."),
    WorkflowTask("raw_pattern", "Raw pattern", "home", True, "Inspect the imported measurement before applying any transformation."),
    WorkflowTask("background", "Background", "preparation", True, "Preview a peak-protected baseline before applying subtraction."),
    WorkflowTask("smoothing", "Smoothing", "preparation", True, "Use smoothing only when peak-shape diagnostics remain acceptable."),
    WorkflowTask("instrument", "Instrument calibration", "preparation", False, "Calibrate instrumental broadening and geometry using a suitable standard."),
    WorkflowTask("peak_list", "Peak list & fitting", "peaks", True, "Detect, add, exclude and freeze the authoritative peak list."),
    WorkflowTask("size_strain", "Size & strain", "peaks", False, "Estimate broadening-derived size and strain only after instrument correction."),
    WorkflowTask("residual_stress", "Residual stress", "peaks", False, "Build verified psi observations before sin-squared-psi regression."),
    WorkflowTask("phase_identification", "Known-phase identification", "phase_structure", True, "Compare curated peaks against the local reference library."),
    WorkflowTask("cif_cell", "CIF & unit cell", "phase_structure", False, "Inspect a known structure and refine its unit cell against observed peaks."),
    WorkflowTask("cif_library", "CIF library & structure match", "phase_structure", True, "Build a local CIF index, rank structure candidates and send selected CIFs to refinement."),
    WorkflowTask("unknown_phase", "Unknown phase", "phase_structure", True, "Create a residual, curate reflections and run native Afruz indexing."),
    WorkflowTask("solve_structure", "Solve structure", "phase_structure", True, "Prepare and validate an atomic model before Rietveld refinement."),
    WorkflowTask("pawley_lebail", "Pawley / Le Bail", "refinement_qpa", True, "Verify candidate cells without requiring an atomic structure."),
    WorkflowTask("rietveld", "Rietveld refinement", "refinement_qpa", True, "Refine only validated atomic models and review parameter correlations."),
    WorkflowTask("doping_series", "Doping-series comparison", "refinement_qpa", True, "Compare an ordered compositional series and optionally refine each pattern independently or sequentially."),
    WorkflowTask("multicomponent_refiner", "Intelligent multiphase refiner", "refinement_qpa", True, "Test user-selected CIF combinations, reject unsupported phases and export a Phase 19 report bundle."),
    WorkflowTask("qpa_preview", "Quantitative analysis", "refinement_qpa", False, "Use exploratory peak-based quantification only with explicit limitations."),
    WorkflowTask("validated_qpa", "Validated QPA", "refinement_qpa", True, "Report phase fractions from a validated structure-constrained workflow."),
    WorkflowTask("validation", "Validation", "validation_reports", True, "Audit assumptions, robustness and evidence before accepting results."),
    WorkflowTask("batch_reports", "Batch & reports", "validation_reports", True, "Apply a frozen recipe consistently and export reproducible outputs."),
    WorkflowTask("full_gui", "Full GUI workbench", "validation_reports", True, "Launch every major scientific engine from one GUI command center."),
    WorkflowTask("advanced", "Advanced analysis", "validation_reports", False, "Planned advanced engines remain separated from validated production workflows."),
)

WORKSPACE_BY_KEY = {workspace.key: workspace for workspace in WORKSPACES}
TASK_BY_KEY = {task.key: task for task in TASKS}

STATUS_ORDER = (
    "import",
    "preparation",
    "peaks",
    "phase",
    "refinement",
    "qpa",
    "validation",
)

STATUS_LABELS = {
    "import": "Import",
    "preparation": "Preparation",
    "peaks": "Peaks",
    "phase": "Phase",
    "refinement": "Refinement",
    "qpa": "QPA",
    "validation": "Validation",
}

# Stage prerequisites are intentionally project-level. Individual engines still
# perform their own detailed validation before starting.
TASK_STAGE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "project_home": (),
    "raw_pattern": ("import",),
    "background": ("import",),
    "smoothing": ("import",),
    "instrument": ("import",),
    "peak_list": ("preparation",),
    "size_strain": ("peaks",),
    "residual_stress": ("peaks",),
    "phase_identification": ("peaks",),
    "cif_cell": ("peaks",),
    "cif_library": ("peaks",),
    "unknown_phase": ("peaks",),
    "solve_structure": ("phase",),
    "pawley_lebail": ("phase",),
    "rietveld": ("phase",),
    "doping_series": ("import",),
    "multicomponent_refiner": ("peaks",),
    "qpa_preview": ("phase",),
    "validated_qpa": ("refinement",),
    "validation": ("refinement",),
    "batch_reports": ("import",),
    "full_gui": (),
    "advanced": (),
}

TASK_CONTEXT_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "size_strain": ("has_instrument_profile",),
    "cif_cell": ("has_reference_structure",),
    "solve_structure": ("has_candidate_cell",),
    "pawley_lebail": ("has_candidate_cell_or_structure",),
    "rietveld": ("has_atomic_structure",),
    "multicomponent_refiner": ("has_cif_candidates",),
    "validated_qpa": ("has_structure_refinement",),
}

CONTEXT_REQUIREMENT_LABELS: dict[str, str] = {
    "has_instrument_profile": "an active instrument-broadening profile",
    "has_reference_structure": "an imported or selected CIF structure",
    "has_candidate_cell": "an indexed candidate unit cell",
    "has_candidate_cell_or_structure": "an indexed cell or selected crystal structure",
    "has_atomic_structure": "a selected atomic structure",
    "has_cif_candidates": "at least one selected CIF candidate",
    "has_structure_refinement": "a current structure-constrained refinement",
}

TASK_COMPLETION_CONTEXT: dict[str, str] = {
    "background": "background_complete",
    "smoothing": "smoothing_complete",
    "instrument": "instrument_complete",
    "peak_list": "peak_list_complete",
    "size_strain": "size_strain_complete",
    "residual_stress": "residual_stress_complete",
    "phase_identification": "known_phase_complete",
    "cif_cell": "cell_refinement_complete",
    "cif_library": "cif_library_match_complete",
    "unknown_phase": "unknown_phase_complete",
    "solve_structure": "structure_complete",
    "pawley_lebail": "whole_pattern_complete",
    "rietveld": "rietveld_complete",
    "doping_series": "doping_series_complete",
    "multicomponent_refiner": "multicomponent_complete",
    "qpa_preview": "qpa_preview_complete",
    "validated_qpa": "validated_qpa_complete",
    "validation": "validation_complete",
    "batch_reports": "report_complete",
}

TASK_RESULT_STAGE: dict[str, str] = {
    "raw_pattern": "import",
    "background": "preparation",
    "smoothing": "preparation",
    "instrument": "preparation",
    "peak_list": "peaks",
    "size_strain": "peaks",
    "residual_stress": "peaks",
    "phase_identification": "phase",
    "cif_cell": "phase",
    "cif_library": "phase",
    "unknown_phase": "phase",
    "solve_structure": "phase",
    "pawley_lebail": "refinement",
    "rietveld": "refinement",
    "doping_series": "refinement",
    "multicomponent_refiner": "refinement",
    "qpa_preview": "qpa",
    "validated_qpa": "qpa",
    "validation": "validation",
    "batch_reports": "validation",
}


def tasks_for_workspace(workspace_key: str, mode: str = "Guided") -> list[WorkflowTask]:
    guided_only = str(mode).strip().lower() == "guided"
    return [
        task
        for task in TASKS
        if task.workspace == workspace_key and (task.guided or not guided_only)
    ]


def build_workflow_status(flags: Mapping[str, bool]) -> dict[str, str]:
    """Return ordered workflow states from simple scientific-completion flags."""
    states: dict[str, str] = {}
    blocked = False
    for key in STATUS_ORDER:
        complete = bool(flags.get(key, False))
        if complete:
            states[key] = "Complete"
        elif blocked:
            states[key] = "Not ready"
        else:
            states[key] = "Next"
            blocked = True
    return states


def _missing_stage_labels(task_key: str, states: Mapping[str, str]) -> tuple[str, ...]:
    missing: list[str] = []
    for stage in TASK_STAGE_REQUIREMENTS.get(task_key, ()):
        if states.get(stage) != "Complete":
            missing.append(STATUS_LABELS.get(stage, stage.title()))
    return tuple(missing)


def _missing_context_labels(task_key: str, context: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        CONTEXT_REQUIREMENT_LABELS.get(key, key.replace("_", " "))
        for key in TASK_CONTEXT_REQUIREMENTS.get(task_key, ())
        if not bool(context.get(key, False))
    )


def _task_is_complete(task_key: str, states: Mapping[str, str], context: Mapping[str, object]) -> bool:
    if task_key == "raw_pattern":
        return states.get("import") == "Complete"
    context_key = TASK_COMPLETION_CONTEXT.get(task_key)
    return bool(context_key and context.get(context_key, False))


def _candidate_recommendation(
    states: Mapping[str, str],
    context: Mapping[str, object],
    mode: str,
) -> str:
    if states.get("import") != "Complete":
        return "project_home"
    if states.get("preparation") in {"Next", "Outdated", "Invalid", "Not ready", None}:
        return "background"
    if states.get("peaks") in {"Next", "Outdated", "Invalid", "Not ready", None}:
        return "peak_list"
    if states.get("phase") in {"Next", "Outdated", "Invalid", "Not ready", None}:
        return "phase_identification"
    if states.get("refinement") in {"Next", "Outdated", "Invalid", "Not ready", None}:
        if context.get("has_atomic_structure"):
            return "rietveld"
        if context.get("has_candidate_cell_or_structure"):
            return "pawley_lebail"
        if context.get("has_cif_candidates"):
            return "multicomponent_refiner"
        return "cif_library"
    if states.get("qpa") in {"Next", "Outdated", "Invalid", "Not ready", None}:
        if context.get("has_structure_refinement"):
            return "validated_qpa"
        if context.get("has_atomic_structure"):
            return "rietveld"
        if context.get("has_candidate_cell"):
            return "solve_structure"
        return "cif_library"
    if states.get("validation") in {"Next", "Outdated", "Invalid", "Not ready", None}:
        return "validation"
    return "batch_reports"


def build_task_states(
    states: Mapping[str, str],
    context: Mapping[str, object] | None = None,
    *,
    mode: str = "Guided",
) -> dict[str, WorkflowTaskState]:
    """Evaluate every task with explicit prerequisites and actionable reasons.

    The function is UI-independent so workflow correctness can be regression
    tested without importing Qt.
    """

    context = dict(context or {})
    recommendation = _candidate_recommendation(states, context, mode)
    visible = {task.key for workspace in WORKSPACES for task in tasks_for_workspace(workspace.key, mode)}
    evaluated: dict[str, WorkflowTaskState] = {}

    for task in TASKS:
        missing_stages = _missing_stage_labels(task.key, states)
        missing_context = _missing_context_labels(task.key, context)
        missing = (*missing_stages, *missing_context)
        completed = _task_is_complete(task.key, states, context)
        result_stage = TASK_RESULT_STAGE.get(task.key)
        stage_state = states.get(result_stage, "") if result_stage else ""

        if completed and stage_state in {"Outdated", "Invalid"}:
            status = stage_state
            enabled = not missing
            reason = (
                "This result is outdated because an upstream scientific input changed. Re-run the task."
                if stage_state == "Outdated"
                else "This result is invalid under the current prerequisites. Review inputs and re-run it."
            )
        elif completed:
            status = "Complete"
            enabled = True
            reason = "A current result is available. You may inspect or re-run this task."
        elif missing:
            status = "Blocked"
            enabled = False
            reason = "Required first: " + "; ".join(missing) + "."
        else:
            status = "Available"
            enabled = True
            reason = task.guidance

        # Hidden expert tasks retain their state for persistence and diagnostics,
        # but can never become the Guided recommendation.
        is_recommended = task.key == recommendation and task.key in visible and status != "Complete"
        if is_recommended and enabled:
            if status == "Available":
                status = "Recommended"
                reason = "Recommended next action. " + task.guidance
            else:
                reason = "Recommended re-run. " + reason

        evaluated[task.key] = WorkflowTaskState(
            key=task.key,
            status=status,
            enabled=enabled,
            completed=completed,
            recommended=is_recommended and enabled,
            reason=reason,
            missing=tuple(missing),
        )

    # If the preferred task is blocked by a task-specific prerequisite, select
    # the first visible available task in the same scientific stage.
    preferred = evaluated.get(recommendation)
    if preferred is not None and not preferred.enabled:
        stage = TASK_RESULT_STAGE.get(recommendation)
        fallback = next(
            (
                task
                for task in TASKS
                if task.key in visible
                and TASK_RESULT_STAGE.get(task.key) == stage
                and evaluated[task.key].enabled
                and not evaluated[task.key].completed
            ),
            None,
        )
        if fallback is not None:
            current = evaluated[fallback.key]
            evaluated[fallback.key] = replace(
                current,
                status="Recommended",
                recommended=True,
                reason="Recommended next action. " + TASK_BY_KEY[fallback.key].guidance,
            )

    return evaluated


def recommended_task_key(
    states: Mapping[str, str],
    context: Mapping[str, object] | None = None,
    mode: str = "Guided",
    task_states: Mapping[str, WorkflowTaskState] | None = None,
) -> str:
    evaluated = dict(task_states or build_task_states(states, context, mode=mode))
    for task in TASKS:
        state = evaluated.get(task.key)
        if state is not None and state.recommended:
            return task.key
    candidate = _candidate_recommendation(states, dict(context or {}), mode)
    if candidate in evaluated and evaluated[candidate].enabled:
        return candidate
    return "project_home"
