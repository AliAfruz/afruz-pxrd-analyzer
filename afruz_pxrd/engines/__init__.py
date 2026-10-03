"""GUI-independent scientific engine API introduced in release 22.4.0."""

from .adapters import NativeEngineAdapter, create_default_engine_registry
from .base import (
    CancellationCheck,
    ENGINE_API_VERSION,
    EngineCancelled,
    EngineDescriptor,
    EngineError,
    EngineExecutionContext,
    EngineExecutionFailure,
    EngineExecutionRecord,
    EngineInputError,
    EngineNotFoundError,
    EngineProgress,
    EngineRegistrationError,
    EngineRequest,
    EngineRun,
    EngineStatus,
    ProgressCallback,
    ScientificEngine,
)
from .registry import EngineRegistry
from .service import ScientificEngineService

__all__ = [
    "ENGINE_API_VERSION",
    "CancellationCheck",
    "EngineCancelled",
    "EngineDescriptor",
    "EngineError",
    "EngineExecutionContext",
    "EngineExecutionFailure",
    "EngineExecutionRecord",
    "EngineInputError",
    "EngineNotFoundError",
    "EngineProgress",
    "EngineRegistrationError",
    "EngineRegistry",
    "EngineRequest",
    "EngineRun",
    "EngineStatus",
    "NativeEngineAdapter",
    "ProgressCallback",
    "ScientificEngine",
    "ScientificEngineService",
    "create_default_engine_registry",
]
