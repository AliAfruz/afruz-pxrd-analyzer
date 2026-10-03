from __future__ import annotations

from threading import RLock
from time import perf_counter
from typing import Any, Mapping
import uuid

from ..contracts import Provenance, ResultContractStore
from ..contracts.base import utc_now
from .base import (
    CancellationCheck,
    EngineCancelled,
    EngineDescriptor,
    EngineExecutionContext,
    EngineExecutionFailure,
    EngineExecutionRecord,
    EngineInputError,
    EngineProgress,
    EngineRequest,
    EngineRun,
    EngineStatus,
    ProgressCallback,
)
from .registry import EngineRegistry


class ScientificEngineService:
    """Executes registered engines and records typed results without Qt."""

    def __init__(
        self,
        registry: EngineRegistry,
        result_store: ResultContractStore | None = None,
    ) -> None:
        if not isinstance(registry, EngineRegistry):
            raise TypeError("registry must be an EngineRegistry")
        self.registry = registry
        self.result_store = result_store
        self._history: list[EngineExecutionRecord] = []
        self._lock = RLock()

    def bind_result_store(self, result_store: ResultContractStore | None) -> None:
        if result_store is not None and not isinstance(
            result_store,
            ResultContractStore,
        ):
            raise TypeError("result_store must be a ResultContractStore or None")
        self.result_store = result_store

    def execute(
        self,
        request: EngineRequest,
        *,
        engine_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
        cancellation_check: CancellationCheck | None = None,
    ) -> EngineRun:
        if not isinstance(request, EngineRequest):
            raise EngineInputError("execute requires an EngineRequest")
        request.validate_signature()
        engine = self.registry.resolve(
            engine_id=engine_id,
            result_kind=request.result_kind,
        )
        descriptor = engine.descriptor
        execution_id = uuid.uuid4().hex
        started_at = utc_now()
        started_clock = perf_counter()
        progress_events: list[EngineProgress] = []

        def record_progress(completed: int, total: int, message: str) -> None:
            event = EngineProgress(completed, total, message)
            progress_events.append(event)
            if progress_callback is not None:
                progress_callback(completed, total, message)

        context = EngineExecutionContext(
            progress_callback=record_progress,
            cancellation_check=cancellation_check,
        )
        try:
            context.raise_if_cancelled()
            result = engine.execute(request, context)
            context.raise_if_cancelled()
            if result.KIND != request.result_kind:
                raise EngineInputError(
                    f"Engine returned {result.KIND} for a {request.result_kind} request"
                )
            if result.dataset_id != request.dataset_id:
                raise EngineInputError(
                    "Engine result dataset identity does not match request"
                )
            if result.engine != descriptor.identity:
                raise EngineInputError("Engine result identity does not match registry")
            if result.input_signature != request.input_signature:
                raise EngineInputError("Engine result input signature does not match request")
            result.validate_result()

            previous = (
                None
                if self.result_store is None
                else self.result_store.get(result.KIND, result.dataset_id)
            )
            source_ids = list(result.provenance.source_result_ids)
            if previous is not None and previous.result_id not in source_ids:
                source_ids.insert(0, previous.result_id)
            result.provenance = Provenance(
                source_result_ids=tuple(source_ids),
                dependency_signatures={
                    **result.provenance.dependency_signatures,
                    "engine_request": request.input_signature,
                },
                notes=result.provenance.notes,
            )
            if self.result_store is not None:
                self.result_store.record(result)
            record = self._record(
                execution_id=execution_id,
                request=request,
                descriptor=descriptor,
                status=EngineStatus.COMPLETED,
                started_at=started_at,
                elapsed=perf_counter() - started_clock,
                progress=progress_events,
                result_id=result.result_id,
                warnings=result.warnings,
            )
            return EngineRun(result=result, execution=record)
        except Exception as exc:
            status = (
                EngineStatus.CANCELLED
                if isinstance(exc, EngineCancelled)
                else EngineStatus.FAILED
            )
            record = self._record(
                execution_id=execution_id,
                request=request,
                descriptor=descriptor,
                status=status,
                started_at=started_at,
                elapsed=perf_counter() - started_clock,
                progress=progress_events,
                error=f"{type(exc).__name__}: {exc}",
            )
            message = (
                f"Engine execution cancelled: {descriptor.engine_id}"
                if status == EngineStatus.CANCELLED
                else f"Engine execution failed: {descriptor.engine_id}: {exc}"
            )
            raise EngineExecutionFailure(
                message,
                execution=record,
                cause=exc,
            ) from exc

    def execute_kind(
        self,
        result_kind: str,
        dataset_id: str,
        *,
        inputs: Mapping[str, Any],
        parameters: Mapping[str, Any] | None = None,
        engine_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
        cancellation_check: CancellationCheck | None = None,
    ) -> EngineRun:
        request = EngineRequest(
            result_kind=str(result_kind),
            dataset_id=str(dataset_id),
            inputs=dict(inputs),
            parameters=dict(parameters or {}),
        )
        return self.execute(
            request,
            engine_id=engine_id,
            progress_callback=progress_callback,
            cancellation_check=cancellation_check,
        )

    def history(self) -> tuple[EngineExecutionRecord, ...]:
        with self._lock:
            return tuple(self._history)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def _record(
        self,
        *,
        execution_id: str,
        request: EngineRequest,
        descriptor: EngineDescriptor,
        status: EngineStatus,
        started_at: str,
        elapsed: float,
        progress: list[EngineProgress],
        result_id: str | None = None,
        warnings: tuple[str, ...] = (),
        error: str | None = None,
    ) -> EngineExecutionRecord:
        record = EngineExecutionRecord(
            execution_id=execution_id,
            request_id=request.request_id,
            engine_id=descriptor.engine_id,
            engine=descriptor.identity,
            result_kind=request.result_kind,
            dataset_id=request.dataset_id,
            input_signature=request.input_signature,
            status=status,
            started_at=started_at,
            finished_at=utc_now(),
            elapsed_seconds=max(0.0, float(elapsed)),
            progress=tuple(progress),
            result_id=result_id,
            warnings=tuple(warnings),
            error=error,
        )
        with self._lock:
            self._history.append(record)
        return record
