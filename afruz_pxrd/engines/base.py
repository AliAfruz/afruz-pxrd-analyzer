from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
import uuid

from ..contracts import EngineIdentity, ScientificResultContract
from ..contracts.base import decode_value, encode_value, utc_now
from ..scientific_state import stable_signature


ENGINE_API_VERSION = 1
ProgressCallback = Callable[[int, int, str], None]
CancellationCheck = Callable[[], bool]


class EngineError(RuntimeError):
    """Base error for the GUI-independent scientific engine boundary."""


class EngineRegistrationError(EngineError):
    pass


class EngineNotFoundError(EngineError):
    pass


class EngineInputError(EngineError, ValueError):
    pass


class EngineCancelled(EngineError):
    pass


class EngineStatus(str, Enum):
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


@dataclass(frozen=True, kw_only=True)
class EngineRequest:
    """Versioned input envelope shared by every registered engine."""

    result_kind: str
    dataset_id: str
    inputs: dict[str, Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=utc_now)
    input_signature: str = ""

    REQUIRED_FIELDS = {
        "api_version",
        "request_id",
        "result_kind",
        "dataset_id",
        "inputs",
        "parameters",
        "created_at",
        "input_signature",
    }

    def __post_init__(self) -> None:
        for name in ("request_id", "result_kind", "dataset_id", "created_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EngineInputError(f"Engine request {name} is required")
        if not isinstance(self.inputs, dict):
            raise EngineInputError("Engine request inputs must be a dictionary")
        if not isinstance(self.parameters, dict):
            raise EngineInputError("Engine request parameters must be a dictionary")
        if not all(isinstance(key, str) and key for key in self.inputs):
            raise EngineInputError("Engine input names must be non-empty strings")
        if not all(isinstance(key, str) and key for key in self.parameters):
            raise EngineInputError("Engine parameter names must be non-empty strings")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise EngineInputError("Engine request created_at must be ISO-8601") from exc
        calculated_signature = self.calculated_signature()
        if self.input_signature and self.input_signature != calculated_signature:
            raise EngineInputError("Engine request input signature does not match inputs")
        signature = self.input_signature or calculated_signature
        object.__setattr__(self, "input_signature", signature)

    def calculated_signature(self) -> str:
        return stable_signature(
            {
                "result_kind": self.result_kind,
                "dataset_id": self.dataset_id,
                "inputs": self.inputs,
                "parameters": self.parameters,
            }
        )

    def validate_signature(self) -> None:
        if self.input_signature != self.calculated_signature():
            raise EngineInputError(
                "Engine request inputs changed after the signature was created"
            )

    def require_inputs(self, *names: str) -> None:
        missing = [name for name in names if name not in self.inputs]
        if missing:
            raise EngineInputError(
                f"{self.result_kind} request is missing inputs: {missing}"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate_signature()
        return {
            "api_version": ENGINE_API_VERSION,
            "request_id": self.request_id,
            "result_kind": self.result_kind,
            "dataset_id": self.dataset_id,
            "inputs": encode_value(self.inputs),
            "parameters": encode_value(self.parameters),
            "created_at": self.created_at,
            "input_signature": self.input_signature,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EngineRequest":
        if not isinstance(value, Mapping):
            raise EngineInputError("Engine request must be an object")
        missing = cls.REQUIRED_FIELDS - set(value)
        if missing:
            raise EngineInputError(f"Engine request is missing fields: {sorted(missing)}")
        try:
            api_version = int(value["api_version"])
        except (TypeError, ValueError) as exc:
            raise EngineInputError("Engine API version must be an integer") from exc
        if api_version != ENGINE_API_VERSION:
            raise EngineInputError(f"Unsupported engine API version {api_version}")
        inputs = decode_value(value["inputs"])
        parameters = decode_value(value["parameters"])
        if not isinstance(inputs, Mapping) or not isinstance(parameters, Mapping):
            raise EngineInputError("Engine inputs and parameters must be objects")
        return cls(
            result_kind=value["result_kind"],
            dataset_id=value["dataset_id"],
            inputs=dict(inputs),
            parameters=dict(parameters),
            request_id=value["request_id"],
            created_at=value["created_at"],
            input_signature=value["input_signature"],
        )


@dataclass(frozen=True)
class EngineProgress:
    completed: int
    total: int
    message: str
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if int(self.total) < 1:
            raise EngineInputError("Progress total must be positive")
        if not 0 <= int(self.completed) <= int(self.total):
            raise EngineInputError("Progress completed value is outside its range")

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": int(self.completed),
            "total": int(self.total),
            "message": str(self.message),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class EngineDescriptor:
    engine_id: str
    identity: EngineIdentity
    result_kind: str
    description: str
    required_inputs: tuple[str, ...]
    supports_progress: bool = False
    supports_cancellation: bool = False
    api_version: int = ENGINE_API_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.engine_id, str) or not self.engine_id.strip():
            raise EngineRegistrationError("Engine ID is required")
        if not isinstance(self.result_kind, str) or not self.result_kind.strip():
            raise EngineRegistrationError("Engine result kind is required")
        if not isinstance(self.identity, EngineIdentity):
            raise EngineRegistrationError("Engine identity is invalid")
        if not isinstance(self.required_inputs, tuple) or not all(
            isinstance(name, str) and name for name in self.required_inputs
        ):
            raise EngineRegistrationError(
                "Engine required inputs must be a tuple of non-empty strings"
            )
        if len(set(self.required_inputs)) != len(self.required_inputs):
            raise EngineRegistrationError("Engine required inputs contain duplicates")
        if int(self.api_version) != ENGINE_API_VERSION:
            raise EngineRegistrationError(
                f"Unsupported engine API version {self.api_version}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "identity": self.identity.to_dict(),
            "result_kind": self.result_kind,
            "description": self.description,
            "required_inputs": list(self.required_inputs),
            "supports_progress": self.supports_progress,
            "supports_cancellation": self.supports_cancellation,
            "api_version": self.api_version,
        }


class EngineExecutionContext:
    """Cooperative progress and cancellation surface passed to engines."""

    def __init__(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        cancellation_check: CancellationCheck | None = None,
    ) -> None:
        self._progress_callback = progress_callback
        self._cancellation_check = cancellation_check

    def cancelled(self) -> bool:
        return bool(self._cancellation_check and self._cancellation_check())

    def raise_if_cancelled(self) -> None:
        if self.cancelled():
            raise EngineCancelled("Scientific engine execution was cancelled")

    def report_progress(self, completed: int, total: int, message: str = "") -> None:
        progress = EngineProgress(int(completed), int(total), str(message))
        if self._progress_callback is not None:
            self._progress_callback(
                progress.completed,
                progress.total,
                progress.message,
            )


@runtime_checkable
class ScientificEngine(Protocol):
    @property
    def descriptor(self) -> EngineDescriptor:
        ...

    def execute(
        self,
        request: EngineRequest,
        context: EngineExecutionContext,
    ) -> ScientificResultContract:
        ...


@dataclass(frozen=True)
class EngineExecutionRecord:
    execution_id: str
    request_id: str
    engine_id: str
    engine: EngineIdentity
    result_kind: str
    dataset_id: str
    input_signature: str
    status: EngineStatus
    started_at: str
    finished_at: str
    elapsed_seconds: float
    progress: tuple[EngineProgress, ...] = ()
    result_id: str | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "request_id": self.request_id,
            "engine_id": self.engine_id,
            "engine": self.engine.to_dict(),
            "result_kind": self.result_kind,
            "dataset_id": self.dataset_id,
            "input_signature": self.input_signature,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": float(self.elapsed_seconds),
            "progress": [row.to_dict() for row in self.progress],
            "result_id": self.result_id,
            "warnings": list(self.warnings),
            "error": self.error,
        }


@dataclass(frozen=True)
class EngineRun:
    result: ScientificResultContract
    execution: EngineExecutionRecord


class EngineExecutionFailure(EngineError):
    def __init__(
        self,
        message: str,
        *,
        execution: EngineExecutionRecord,
        cause: Exception,
    ) -> None:
        super().__init__(message)
        self.execution = execution
        self.cause = cause
