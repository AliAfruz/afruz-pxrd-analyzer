from __future__ import annotations

from collections.abc import Iterable
from threading import RLock

from .base import (
    EngineDescriptor,
    EngineNotFoundError,
    EngineRegistrationError,
    ScientificEngine,
)


class EngineRegistry:
    """Thread-safe registry of scientific engines and per-result defaults."""

    def __init__(self) -> None:
        self._engines: dict[str, ScientificEngine] = {}
        self._defaults: dict[str, str] = {}
        self._lock = RLock()

    def register(self, engine: ScientificEngine, *, default: bool = False) -> None:
        if not isinstance(engine, ScientificEngine):
            raise EngineRegistrationError(
                "Registered engines must implement descriptor and execute"
            )
        descriptor = engine.descriptor
        with self._lock:
            if descriptor.engine_id in self._engines:
                raise EngineRegistrationError(
                    f"Engine ID is already registered: {descriptor.engine_id}"
                )
            self._engines[descriptor.engine_id] = engine
            if default or descriptor.result_kind not in self._defaults:
                self._defaults[descriptor.result_kind] = descriptor.engine_id

    def register_many(
        self,
        engines: Iterable[ScientificEngine],
        *,
        defaults: bool = False,
    ) -> None:
        for engine in engines:
            self.register(engine, default=defaults)

    def unregister(self, engine_id: str) -> ScientificEngine:
        normalized_id = str(engine_id)
        with self._lock:
            try:
                engine = self._engines.pop(normalized_id)
            except KeyError as exc:
                raise EngineNotFoundError(f"Unknown engine ID: {engine_id}") from exc
            for kind, default_id in tuple(self._defaults.items()):
                if default_id == normalized_id:
                    replacement = next(
                        (
                            row.descriptor.engine_id
                            for row in self._engines.values()
                            if row.descriptor.result_kind == kind
                        ),
                        None,
                    )
                    if replacement is None:
                        self._defaults.pop(kind, None)
                    else:
                        self._defaults[kind] = replacement
            return engine

    def set_default(self, result_kind: str, engine_id: str) -> None:
        engine = self.resolve(engine_id=engine_id)
        if engine.descriptor.result_kind != str(result_kind):
            raise EngineRegistrationError(
                f"Engine '{engine_id}' produces {engine.descriptor.result_kind}, "
                f"not {result_kind}"
            )
        with self._lock:
            self._defaults[str(result_kind)] = str(engine_id)

    def resolve(
        self,
        *,
        engine_id: str | None = None,
        result_kind: str | None = None,
    ) -> ScientificEngine:
        with self._lock:
            selected_id = str(engine_id) if engine_id else None
            if selected_id is None and result_kind is not None:
                selected_id = self._defaults.get(str(result_kind))
            if selected_id is None:
                raise EngineNotFoundError(
                    "Specify an engine ID or a result kind with a registered default"
                )
            engine = self._engines.get(selected_id)
            if engine is None:
                raise EngineNotFoundError(f"Unknown engine ID: {selected_id}")
            if result_kind is not None and engine.descriptor.result_kind != result_kind:
                raise EngineNotFoundError(
                    f"Engine '{selected_id}' does not produce {result_kind}"
                )
            return engine

    def descriptors(self) -> tuple[EngineDescriptor, ...]:
        with self._lock:
            return tuple(
                engine.descriptor
                for _, engine in sorted(self._engines.items())
            )

    def defaults(self) -> dict[str, str]:
        with self._lock:
            return dict(self._defaults)

    def __len__(self) -> int:
        with self._lock:
            return len(self._engines)
