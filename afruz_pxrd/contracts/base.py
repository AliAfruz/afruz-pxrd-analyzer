from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Mapping
import uuid

import numpy as np


CONTRACT_SCHEMA_VERSION = 1


class ContractValidationError(ValueError):
    """Raised when a typed scientific result violates its public contract."""


class ValidationStatus(str, Enum):
    NOT_VALIDATED = "Not validated"
    VALID = "Valid"
    WARNING = "Warning"
    INVALID = "Invalid"
    EXPERIMENTAL = "Experimental"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return encode_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def decode_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get("__ndarray__") is True:
            required = {"dtype", "shape", "values"}
            missing = required - set(value)
            if missing:
                raise ContractValidationError(
                    f"Encoded array is missing fields: {sorted(missing)}"
                )
            array = np.asarray(value["values"], dtype=str(value["dtype"]))
            expected_shape = tuple(int(row) for row in value["shape"])
            if array.shape != expected_shape:
                raise ContractValidationError(
                    f"Encoded array shape {array.shape} does not match {expected_shape}"
                )
            return array
        return {str(key): decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    return value


def numeric_array(value: Any, *, field_name: str) -> np.ndarray:
    array = np.asarray(value if value is not None else [], dtype=float)
    if array.ndim != 1:
        raise ContractValidationError(f"{field_name} must be one-dimensional")
    if array.size and not np.all(np.isfinite(array)):
        raise ContractValidationError(f"{field_name} contains non-finite values")
    return array


def require_aligned_arrays(**arrays: np.ndarray) -> None:
    populated = {name: len(value) for name, value in arrays.items() if len(value)}
    if populated and len(set(populated.values())) != 1:
        raise ContractValidationError(
            "Profile arrays must have equal lengths: "
            + ", ".join(f"{name}={length}" for name, length in populated.items())
        )


@dataclass(frozen=True)
class EngineIdentity:
    name: str
    version: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ContractValidationError("Engine name is required")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ContractValidationError("Engine version is required")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EngineIdentity":
        if not isinstance(value, Mapping):
            raise ContractValidationError("engine must be an object")
        missing = {"name", "version"} - set(value)
        if missing:
            raise ContractValidationError(f"engine is missing fields: {sorted(missing)}")
        return cls(str(value["name"]), str(value["version"]))


@dataclass(frozen=True)
class Provenance:
    source_result_ids: tuple[str, ...] = ()
    dependency_signatures: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_result_ids, tuple) or not all(
            isinstance(row, str) for row in self.source_result_ids
        ):
            raise ContractValidationError("provenance source_result_ids must be strings")
        if not isinstance(self.dependency_signatures, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.dependency_signatures.items()
        ):
            raise ContractValidationError(
                "provenance dependency_signatures must map strings to strings"
            )
        if not isinstance(self.notes, tuple) or not all(
            isinstance(row, str) for row in self.notes
        ):
            raise ContractValidationError("provenance notes must be strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_result_ids": list(self.source_result_ids),
            "dependency_signatures": dict(self.dependency_signatures),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Provenance":
        if not isinstance(value, Mapping):
            raise ContractValidationError("provenance must be an object")
        return cls(
            tuple(str(row) for row in value.get("source_result_ids", [])),
            {
                str(key): str(item)
                for key, item in dict(value.get("dependency_signatures", {})).items()
            },
            tuple(str(row) for row in value.get("notes", [])),
        )


@dataclass(kw_only=True)
class ScientificResultContract:
    """Stable envelope shared by every scientific result family."""

    KIND: ClassVar[str] = "scientific_result"
    SCHEMA_VERSION: ClassVar[int] = CONTRACT_SCHEMA_VERSION

    dataset_id: str
    engine: EngineIdentity
    input_signature: str
    parameters: dict[str, Any] = field(default_factory=dict)
    uncertainties: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    validation_status: ValidationStatus = ValidationStatus.NOT_VALIDATED
    provenance: Provenance = field(default_factory=Provenance)
    result_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=utc_now)

    REQUIRED_ENVELOPE_FIELDS: ClassVar[set[str]] = {
        "kind",
        "schema_version",
        "result_id",
        "dataset_id",
        "engine",
        "input_signature",
        "parameters",
        "numerical_outputs",
        "uncertainties",
        "warnings",
        "validation_status",
        "created_at",
        "provenance",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.result_id, str) or not self.result_id.strip():
            raise ContractValidationError("Result identity is required")
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ContractValidationError("Dataset identity is required")
        if not isinstance(self.engine, EngineIdentity):
            raise ContractValidationError("engine must be an EngineIdentity")
        if not isinstance(self.input_signature, str) or not self.input_signature.strip():
            raise ContractValidationError("Input signature is required")
        if not isinstance(self.parameters, dict):
            raise ContractValidationError("parameters must be a dictionary")
        if not isinstance(self.uncertainties, dict):
            raise ContractValidationError("uncertainties must be a dictionary")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(row, str) for row in self.warnings
        ):
            raise ContractValidationError("warnings must be a tuple of strings")
        if not isinstance(self.validation_status, ValidationStatus):
            raise ContractValidationError("validation_status must be a ValidationStatus")
        if not isinstance(self.provenance, Provenance):
            raise ContractValidationError("provenance must be a Provenance")
        if not isinstance(self.created_at, str):
            raise ContractValidationError("created_at must be ISO-8601 text")
        try:
            datetime.fromisoformat(str(self.created_at).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractValidationError("created_at must be ISO-8601") from exc
        self.validate_result()

    def validate_result(self) -> None:
        """Subclasses validate their typed numerical payload."""

    def numerical_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def payload_kwargs(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.KIND,
            "schema_version": self.SCHEMA_VERSION,
            "result_id": self.result_id,
            "dataset_id": self.dataset_id,
            "engine": self.engine.to_dict(),
            "input_signature": self.input_signature,
            "parameters": encode_value(self.parameters),
            "numerical_outputs": encode_value(self.numerical_payload()),
            "uncertainties": encode_value(self.uncertainties),
            "warnings": list(self.warnings),
            "validation_status": self.validation_status.value,
            "created_at": self.created_at,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScientificResultContract":
        if not isinstance(value, Mapping):
            raise ContractValidationError(f"{cls.__name__} must be an object")
        missing = cls.REQUIRED_ENVELOPE_FIELDS - set(value)
        if missing:
            raise ContractValidationError(
                f"{cls.__name__} is missing fields: {sorted(missing)}"
            )
        if value["kind"] != cls.KIND:
            raise ContractValidationError(
                f"Expected result kind '{cls.KIND}', received '{value['kind']}'"
            )
        try:
            schema_version = int(value["schema_version"])
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("schema_version must be an integer") from exc
        if schema_version != cls.SCHEMA_VERSION:
            raise ContractValidationError(
                f"Unsupported {cls.KIND} schema version {value['schema_version']}"
            )
        outputs = decode_value(value["numerical_outputs"])
        if not isinstance(outputs, Mapping):
            raise ContractValidationError("numerical_outputs must be an object")
        try:
            status = ValidationStatus(str(value["validation_status"]))
        except ValueError as exc:
            raise ContractValidationError(
                f"Unknown validation status: {value['validation_status']}"
            ) from exc
        parameters = decode_value(value["parameters"])
        uncertainties = decode_value(value["uncertainties"])
        warnings = value["warnings"]
        if not isinstance(parameters, Mapping):
            raise ContractValidationError("parameters must be an object")
        if not isinstance(uncertainties, Mapping):
            raise ContractValidationError("uncertainties must be an object")
        if not isinstance(warnings, (list, tuple)) or not all(
            isinstance(row, str) for row in warnings
        ):
            raise ContractValidationError("warnings must be a list of strings")
        return cls(
            dataset_id=str(value["dataset_id"]),
            engine=EngineIdentity.from_dict(value["engine"]),
            input_signature=str(value["input_signature"]),
            parameters=dict(parameters),
            uncertainties=dict(uncertainties),
            warnings=tuple(warnings),
            validation_status=status,
            provenance=Provenance.from_dict(value["provenance"]),
            result_id=str(value["result_id"]),
            created_at=str(value["created_at"]),
            **cls.payload_kwargs(outputs),
        )
