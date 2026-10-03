from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, fields
from inspect import signature
from typing import Any

import numpy as np

from ..contracts import EngineIdentity, Provenance, ScientificResultContract
from ..contracts.store import contract_from_legacy
from ..crystallography import refine_unit_cell
from ..fitting import advanced_deconvolve_peaks, fit_detected_peaks
from ..instrument_calibration import calibrate_instrument, quality_check
from ..phase_identification import ReferenceEntry, identify_phases
from ..processing import (
    ProcessingParameters,
    detect_peaks,
    process_pattern,
    smart_detect_peaks,
)
from ..quantitative_phase import QPAPhaseSpec, quantify_phases
from ..rietveld_refinement import RietveldPhaseSpec, refine_rietveld
from ..validation_campaign import analyze_validation_campaign
from ..version import APP_VERSION
from ..whole_pattern_refinement import WholePatternPhaseSpec, refine_whole_pattern
from .base import (
    EngineDescriptor,
    EngineExecutionContext,
    EngineInputError,
    EngineRequest,
)
from .registry import EngineRegistry


NativeHandler = Callable[[EngineRequest, EngineExecutionContext], Mapping[str, Any]]


def _xy(request: EngineRequest) -> tuple[np.ndarray, np.ndarray]:
    request.require_inputs("x", "y")
    try:
        x = np.asarray(request.inputs["x"], dtype=float)
        y = np.asarray(request.inputs["y"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise EngineInputError("x and y must be numeric arrays") from exc
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise EngineInputError("x and y must be aligned one-dimensional arrays")
    if len(x) < 3:
        raise EngineInputError("At least three profile points are required")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise EngineInputError("x and y must contain only finite values")
    if np.any(np.diff(x) <= 0):
        raise EngineInputError("x must be strictly increasing")
    return x, y


def _sequence_input(request: EngineRequest, name: str) -> list[Any]:
    request.require_inputs(name)
    value = request.inputs[name]
    if not isinstance(value, (list, tuple)):
        raise EngineInputError(f"{name} must be a list")
    return list(value)


def _mapping_input(request: EngineRequest, name: str) -> dict[str, Any]:
    request.require_inputs(name)
    value = request.inputs[name]
    if not isinstance(value, Mapping):
        raise EngineInputError(f"{name} must be an object")
    return dict(value)


def _call_native(
    function: Callable[..., Any],
    positional: Sequence[Any],
    parameters: Mapping[str, Any],
    context: EngineExecutionContext,
) -> Any:
    kwargs = dict(parameters)
    function_signature = signature(function)
    if "progress_callback" in function_signature.parameters:
        kwargs["progress_callback"] = context.report_progress
    if "cancel_check" in function_signature.parameters:
        kwargs["cancel_check"] = context.cancelled
    try:
        function_signature.bind(*positional, **kwargs)
    except TypeError as exc:
        raise EngineInputError(
            f"Invalid inputs for {function.__name__}: {exc}"
        ) from exc
    context.raise_if_cancelled()
    result = function(*positional, **kwargs)
    context.raise_if_cancelled()
    return result


def _reference(value: Any) -> ReferenceEntry:
    if isinstance(value, ReferenceEntry):
        return value
    if not isinstance(value, Mapping):
        raise EngineInputError("Phase references must be ReferenceEntry values or objects")
    required = {row.name for row in fields(ReferenceEntry)}
    missing = required - set(value)
    if missing:
        raise EngineInputError(f"Phase reference is missing fields: {sorted(missing)}")
    return ReferenceEntry(**{name: value[name] for name in required})


def _whole_phase(value: Any) -> WholePatternPhaseSpec:
    if isinstance(value, WholePatternPhaseSpec):
        return value
    if not isinstance(value, Mapping) or "reference" not in value:
        raise EngineInputError("Whole-pattern phase specs require a reference")
    return WholePatternPhaseSpec(
        reference=_reference(value["reference"]),
        refine_cell=bool(value.get("refine_cell", True)),
        included=bool(value.get("included", True)),
    )


def _rietveld_phase(value: Any) -> RietveldPhaseSpec:
    if isinstance(value, RietveldPhaseSpec):
        return value
    if not isinstance(value, Mapping) or not isinstance(value.get("structure"), Mapping):
        raise EngineInputError("Rietveld phase specs require a structure object")
    preferred = value.get("preferred_orientation_hkl")
    return RietveldPhaseSpec(
        structure=dict(value["structure"]),
        name=value.get("name"),
        refine_cell=bool(value.get("refine_cell", True)),
        refine_biso=bool(value.get("refine_biso", False)),
        preferred_orientation_hkl=(
            None if preferred is None else tuple(int(row) for row in preferred)
        ),
        refine_preferred_orientation=bool(
            value.get("refine_preferred_orientation", False)
        ),
        initial_preferred_orientation_r=float(
            value.get("initial_preferred_orientation_r", 1.0)
        ),
        included=bool(value.get("included", True)),
    )


def _qpa_phase(value: Any) -> QPAPhaseSpec:
    if isinstance(value, QPAPhaseSpec):
        return value
    if not isinstance(value, Mapping) or "reference" not in value:
        raise EngineInputError("QPA phase specs require a reference")
    override = value.get("rir_override")
    return QPAPhaseSpec(
        reference=_reference(value["reference"]),
        zero_shift_deg=float(value.get("zero_shift_deg", 0.0)),
        rir_override=None if override is None else float(override),
    )


class NativeEngineAdapter:
    """Adapts an established numerical function to the Phase 6 protocol."""

    def __init__(self, descriptor: EngineDescriptor, handler: NativeHandler) -> None:
        self._descriptor = descriptor
        self._handler = handler

    @property
    def descriptor(self) -> EngineDescriptor:
        return self._descriptor

    def execute(
        self,
        request: EngineRequest,
        context: EngineExecutionContext,
    ) -> ScientificResultContract:
        if request.result_kind != self.descriptor.result_kind:
            raise EngineInputError(
                f"Engine {self.descriptor.engine_id} produces "
                f"{self.descriptor.result_kind}, not {request.result_kind}"
            )
        request.require_inputs(*self.descriptor.required_inputs)
        context.raise_if_cancelled()
        payload = self._handler(request, context)
        if not isinstance(payload, Mapping):
            raise EngineInputError(
                f"Engine {self.descriptor.engine_id} returned a non-object result"
            )
        contract = contract_from_legacy(
            request.result_kind,
            request.dataset_id,
            dict(payload),
        )
        contract.engine = self.descriptor.identity
        contract.input_signature = request.input_signature
        contract.parameters = {
            **contract.parameters,
            **request.parameters,
        }
        contract.provenance = Provenance(
            dependency_signatures={"engine_request": request.input_signature},
            notes=(
                f"Executed through engine '{self.descriptor.engine_id}'",
            ),
        )
        contract.validate_result()
        return contract


def _preprocessing(request: EngineRequest, context: EngineExecutionContext) -> dict:
    x, y = _xy(request)
    known = {row.name for row in fields(ProcessingParameters)}
    unknown = set(request.parameters) - known
    if unknown:
        raise EngineInputError(f"Unknown preprocessing parameters: {sorted(unknown)}")
    parameters = ProcessingParameters(**request.parameters)
    context.report_progress(0, 1, "Preparing diffraction pattern")
    processed, auxiliary = process_pattern(x, y, parameters)
    context.report_progress(1, 1, "Pattern preparation complete")
    background = (
        {
            "background": auxiliary["background"],
            "display_background": auxiliary.get("background_display"),
            "settings": auxiliary.get("background_parameters", {}),
            "diagnostics": auxiliary.get("background_diagnostics", {}),
        }
        if "background" in auxiliary
        else {}
    )
    smoothing = (
        {
            "smoothed": auxiliary["smoothed"],
            "source": auxiliary.get("smoothing_source"),
            "settings": auxiliary.get("smoothing_parameters", {}),
            "diagnostics": auxiliary.get("smoothing_diagnostics", {}),
        }
        if "smoothed" in auxiliary
        else {}
    )
    return {
        "processed": processed,
        "background": background,
        "smoothing": smoothing,
        "settings": asdict(parameters),
    }


def _peak_list(request: EngineRequest, context: EngineExecutionContext) -> dict:
    x, y = _xy(request)
    parameters = dict(request.parameters)
    method = str(parameters.pop("method", "regular")).strip().lower()
    context.report_progress(0, 1, "Detecting peaks")
    if method == "smart":
        sensitivity = str(parameters.pop("sensitivity", "Balanced"))
        if parameters:
            raise EngineInputError(f"Unknown smart peak parameters: {sorted(parameters)}")
        rows, diagnostics = smart_detect_peaks(x, y, sensitivity=sensitivity)
    elif method == "regular":
        rows = _call_native(detect_peaks, (x, y), parameters, context)
        diagnostics = {"method": "regular"}
    else:
        raise EngineInputError("Peak-list method must be 'regular' or 'smart'")
    context.report_progress(1, 1, f"Detected {len(rows)} peaks")
    return {
        "rows": rows,
        "meta": {"method": method, "diagnostics": diagnostics},
        "settings": request.parameters,
    }


def _peak_fitting(request: EngineRequest, context: EngineExecutionContext) -> dict:
    x, y = _xy(request)
    peaks = _sequence_input(request, "peaks")
    parameters = dict(request.parameters)
    advanced = bool(parameters.pop("advanced", False))
    function = advanced_deconvolve_peaks if advanced else fit_detected_peaks
    groups, diagnostics = _call_native(
        function,
        (x, y, peaks),
        parameters,
        context,
    )
    candidates = diagnostics.get("candidates", []) if isinstance(diagnostics, Mapping) else []
    return {
        "groups": groups,
        "candidates": candidates,
        "settings": {**request.parameters, "diagnostics": diagnostics},
    }


def _instrument(request: EngineRequest, context: EngineExecutionContext) -> dict:
    fit_groups = _sequence_input(request, "fit_groups")
    reference_peaks = _sequence_input(request, "reference_peaks")
    parameters = dict(request.parameters)
    qa_tolerance = float(parameters.pop("qa_tolerance_deg", 0.25))
    context.report_progress(0, 2, "Fitting instrument profile")
    profile = _call_native(
        calibrate_instrument,
        (fit_groups, reference_peaks),
        parameters,
        context,
    )
    qa_result = {}
    qa_groups = request.inputs.get("qa_fit_groups")
    if isinstance(qa_groups, (list, tuple)) and qa_groups:
        context.report_progress(1, 2, "Checking instrument profile")
        qa_result = quality_check(
            profile,
            list(qa_groups),
            reference_peaks,
            tolerance_deg=qa_tolerance,
        )
    context.report_progress(2, 2, "Instrument calibration complete")
    return {
        "profile": profile,
        "qa_result": qa_result,
        "settings": request.parameters,
        "warnings": profile.get("warnings", []),
    }


def _unit_cell(request: EngineRequest, context: EngineExecutionContext) -> dict:
    matches = _sequence_input(request, "matches")
    initial_cell = _mapping_input(request, "initial_cell")
    context.report_progress(0, 1, "Refining unit cell")
    result = _call_native(
        refine_unit_cell,
        (matches, initial_cell),
        request.parameters,
        context,
    )
    context.report_progress(1, 1, "Unit-cell refinement complete")
    return dict(result)


def _phase_identification(
    request: EngineRequest,
    context: EngineExecutionContext,
) -> dict:
    observed = _sequence_input(request, "observed_peaks")
    references = [_reference(row) for row in _sequence_input(request, "references")]
    context.report_progress(0, 1, "Screening phase references")
    result = _call_native(
        identify_phases,
        (observed, references),
        request.parameters,
        context,
    )
    context.report_progress(1, 1, "Phase identification complete")
    return dict(result)


def _whole_pattern(request: EngineRequest, context: EngineExecutionContext) -> dict:
    x, y = _xy(request)
    phase_specs = [_whole_phase(row) for row in _sequence_input(request, "phase_specs")]
    return dict(
        _call_native(
            refine_whole_pattern,
            (x, y, phase_specs),
            request.parameters,
            context,
        )
    )


def _rietveld(request: EngineRequest, context: EngineExecutionContext) -> dict:
    x, y = _xy(request)
    phase_specs = [_rietveld_phase(row) for row in _sequence_input(request, "phase_specs")]
    return dict(
        _call_native(
            refine_rietveld,
            (x, y, phase_specs),
            request.parameters,
            context,
        )
    )


def _qpa(request: EngineRequest, context: EngineExecutionContext) -> dict:
    x, y = _xy(request)
    phase_specs = [_qpa_phase(row) for row in _sequence_input(request, "phase_specs")]
    context.report_progress(0, 1, "Quantifying phases")
    result = _call_native(
        quantify_phases,
        (x, y, phase_specs),
        request.parameters,
        context,
    )
    context.report_progress(1, 1, "Phase quantification complete")
    return dict(result)


def _validation(request: EngineRequest, context: EngineExecutionContext) -> dict:
    records = _sequence_input(request, "records")
    context.report_progress(0, 1, "Analyzing validation campaign")
    result = _call_native(
        analyze_validation_campaign,
        (records,),
        request.parameters,
        context,
    )
    context.report_progress(1, 1, "Validation campaign complete")
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"records", "refinement_audits", "robustness_results"}
    }
    return {
        "rows": result.get("records", []),
        "summary": summary,
        "warnings": result.get("warnings", []),
        "status": result.get("status", ""),
    }


def create_default_engine_registry() -> EngineRegistry:
    """Return all ten native engines without constructing any Qt object."""
    definitions = (
        (
            "afruz.native.preprocessing",
            "preprocessing",
            "Native pattern preparation",
            ("x", "y"),
            _preprocessing,
            True,
            False,
        ),
        (
            "afruz.native.peak_list",
            "peak_list",
            "Native peak detection",
            ("x", "y"),
            _peak_list,
            True,
            False,
        ),
        (
            "afruz.native.peak_fitting",
            "peak_fitting",
            "Native peak-profile fitting",
            ("x", "y", "peaks"),
            _peak_fitting,
            True,
            True,
        ),
        (
            "afruz.native.instrument_calibration",
            "instrument_calibration",
            "Native instrument calibration",
            ("fit_groups", "reference_peaks"),
            _instrument,
            True,
            False,
        ),
        (
            "afruz.native.unit_cell_refinement",
            "unit_cell_refinement",
            "Native unit-cell refinement",
            ("matches", "initial_cell"),
            _unit_cell,
            True,
            False,
        ),
        (
            "afruz.native.phase_identification",
            "phase_identification",
            "Native phase identification",
            ("observed_peaks", "references"),
            _phase_identification,
            True,
            False,
        ),
        (
            "afruz.native.whole_pattern_refinement",
            "whole_pattern_refinement",
            "Native Pawley/Le Bail refinement",
            ("x", "y", "phase_specs"),
            _whole_pattern,
            True,
            True,
        ),
        (
            "afruz.native.rietveld_refinement",
            "rietveld_refinement",
            "Native Rietveld refinement",
            ("x", "y", "phase_specs"),
            _rietveld,
            True,
            True,
        ),
        (
            "afruz.native.qpa",
            "qpa",
            "Native RIR quantitative phase analysis",
            ("x", "y", "phase_specs"),
            _qpa,
            True,
            False,
        ),
        (
            "afruz.native.validation",
            "validation",
            "Native validation-campaign analysis",
            ("records",),
            _validation,
            True,
            False,
        ),
    )
    registry = EngineRegistry()
    for engine_id, kind, description, required, handler, progress, cancellation in definitions:
        registry.register(
            NativeEngineAdapter(
                EngineDescriptor(
                    engine_id=engine_id,
                    identity=EngineIdentity(description, APP_VERSION),
                    result_kind=kind,
                    description=description,
                    required_inputs=required,
                    supports_progress=progress,
                    supports_cancellation=cancellation,
                ),
                handler,
            ),
            default=True,
        )
    return registry
