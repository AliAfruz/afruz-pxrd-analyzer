from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable, Mapping
import uuid

try:  # NumPy is optional for the pure state model tests.
    import numpy as np
except Exception:  # pragma: no cover
    np = None


STATE_SCHEMA_VERSION = 1

STAGE_ORDER: tuple[str, ...] = (
    "import",
    "preparation",
    "peaks",
    "phase",
    "refinement",
    "qpa",
    "validation",
)

STAGE_LABELS: dict[str, str] = {
    "import": "Raw pattern",
    "preparation": "Prepared pattern",
    "peaks": "Master peak list",
    "phase": "Phase / structure",
    "refinement": "Refinement",
    "qpa": "Quantitative analysis",
    "validation": "Validation",
}

# A deliberately compact dependency graph. More detailed engines can place their
# own provenance inside node metadata while the project-level workflow remains
# understandable and stable.
DEFAULT_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "import": (),
    "preparation": ("import",),
    "peaks": ("preparation",),
    "phase": ("peaks",),
    "refinement": ("preparation", "phase"),
    "qpa": ("refinement",),
    "validation": ("refinement",),
}

VALID_STATUSES = {"Missing", "Current", "Outdated", "Invalid"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _jsonable(value: Any) -> Any:
    if np is not None and isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256(array.view(np.uint8)).hexdigest()
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": digest,
        }
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return repr(value)


def stable_signature(value: Any) -> str:
    """Return a stable SHA-256 digest for arrays and nested scientific records."""
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class ScientificStateNode:
    key: str
    revision: int = 0
    signature: str = ""
    status: str = "Missing"
    result_uid: str = ""
    dependencies: dict[str, int] = field(default_factory=dict)
    dependency_signatures: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""
    reason: str = ""
    invalidated_by: list[str] = field(default_factory=list)
    accepted_revision: int | None = None
    accepted_signature: str = ""
    accepted_at: str = ""

    @property
    def exists(self) -> bool:
        return self.revision > 0 and bool(self.signature)

    @property
    def is_current(self) -> bool:
        return self.status == "Current" and self.exists

    @property
    def is_accepted_current(self) -> bool:
        return bool(
            self.is_current
            and self.accepted_revision == self.revision
            and self.accepted_signature == self.signature
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], key: str | None = None) -> "ScientificStateNode":
        status = str(data.get("status", "Missing"))
        if status not in VALID_STATUSES:
            status = "Missing"
        return cls(
            key=str(key or data.get("key", "")),
            revision=max(0, int(data.get("revision", 0))),
            signature=str(data.get("signature", "")),
            status=status,
            result_uid=str(data.get("result_uid", "")),
            dependencies={str(k): int(v) for k, v in dict(data.get("dependencies", {})).items()},
            dependency_signatures={str(k): str(v) for k, v in dict(data.get("dependency_signatures", {})).items()},
            metadata=dict(data.get("metadata", {})),
            updated_at=str(data.get("updated_at", "")),
            reason=str(data.get("reason", "")),
            invalidated_by=[str(value) for value in data.get("invalidated_by", [])],
            accepted_revision=(
                None if data.get("accepted_revision") is None else int(data.get("accepted_revision"))
            ),
            accepted_signature=str(data.get("accepted_signature", "")),
            accepted_at=str(data.get("accepted_at", "")),
        )


class DatasetScientificState:
    def __init__(self, dataset_uid: str, nodes: Mapping[str, ScientificStateNode] | None = None):
        self.dataset_uid = str(dataset_uid)
        self.nodes: dict[str, ScientificStateNode] = {
            key: ScientificStateNode(key=key) for key in STAGE_ORDER
        }
        if nodes:
            for key, node in nodes.items():
                if key in self.nodes:
                    self.nodes[key] = node
        self.events: list[dict[str, Any]] = []

    def node(self, key: str) -> ScientificStateNode:
        if key not in self.nodes:
            raise KeyError(f"Unknown scientific-state stage: {key}")
        return self.nodes[key]

    def _dependency_snapshot(
        self,
        key: str,
        dependency_keys: Iterable[str] | None = None,
    ) -> tuple[dict[str, int], dict[str, str]]:
        revisions: dict[str, int] = {}
        signatures: dict[str, str] = {}
        keys = tuple(dependency_keys) if dependency_keys is not None else DEFAULT_DEPENDENCIES.get(key, ())
        for dependency in keys:
            node = self.node(dependency)
            revisions[dependency] = node.revision
            signatures[dependency] = node.signature
        return revisions, signatures

    def dependencies_current(self, key: str) -> bool:
        node = self.node(key)
        dependency_keys = tuple(node.dependencies) or DEFAULT_DEPENDENCIES.get(key, ())
        for dependency in dependency_keys:
            current = self.node(dependency)
            if not current.is_current:
                return False
            if node.dependencies.get(dependency) != current.revision:
                return False
            if node.dependency_signatures.get(dependency) != current.signature:
                return False
        return True

    def update_authoritative(
        self,
        key: str,
        value: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
        reason: str = "",
        accept: bool = False,
        dependency_keys: Iterable[str] | None = None,
    ) -> ScientificStateNode:
        return self.record_result(
            key,
            value,
            metadata=metadata,
            reason=reason,
            accept=accept,
            produced=True,
            dependency_keys=dependency_keys,
        )

    def observe_result(
        self,
        key: str,
        value: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
        reason: str = "Observed existing result",
        dependency_keys: Iterable[str] | None = None,
    ) -> ScientificStateNode:
        return self.record_result(
            key,
            value,
            metadata=metadata,
            reason=reason,
            produced=False,
            dependency_keys=dependency_keys,
        )

    def record_result(
        self,
        key: str,
        value: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
        reason: str = "",
        accept: bool = False,
        produced: bool = True,
        dependency_keys: Iterable[str] | None = None,
    ) -> ScientificStateNode:
        node = self.node(key)
        signature = stable_signature(value)
        changed = signature != node.signature or not node.exists
        dependency_revisions, dependency_signatures = self._dependency_snapshot(
            key,
            dependency_keys,
        )

        if changed:
            node.revision += 1
            node.signature = signature
            node.result_uid = uuid.uuid4().hex
            node.dependencies = dependency_revisions
            node.dependency_signatures = dependency_signatures
            node.metadata = dict(metadata or {})
            node.updated_at = utc_now()
            node.reason = reason
            node.invalidated_by = []
            node.status = "Current" if self._dependencies_available(key) else "Invalid"
            self._append_event("updated", key, node, reason)
            self.invalidate_downstream(key, reason=f"{STAGE_LABELS[key]} revision changed")
        else:
            if metadata:
                node.metadata.update(dict(metadata))
            # An explicit production event means this result was recomputed against
            # the current dependencies, even when numerical output happens to match.
            if produced:
                node.dependencies = dependency_revisions
                node.dependency_signatures = dependency_signatures
                node.updated_at = utc_now()
                node.reason = reason or node.reason
                node.invalidated_by = []
                node.status = "Current" if self._dependencies_available(key) else "Invalid"
                self._append_event("recomputed", key, node, reason)
            elif node.status == "Current" and not self.dependencies_current(key):
                node.status = "Outdated"
                node.invalidated_by = self._dependency_mismatches(key)

        if accept:
            self.accept(key)
        return node

    def _dependencies_available(self, key: str) -> bool:
        node = self.node(key)
        dependencies = tuple(node.dependencies) or DEFAULT_DEPENDENCIES.get(key, ())
        return all(self.node(dep).is_current for dep in dependencies)

    def _dependency_mismatches(self, key: str) -> list[str]:
        node = self.node(key)
        mismatches: list[str] = []
        dependencies = tuple(node.dependencies) or DEFAULT_DEPENDENCIES.get(key, ())
        for dependency in dependencies:
            current = self.node(dependency)
            if not current.is_current:
                mismatches.append(dependency)
            elif node.dependencies.get(dependency) != current.revision:
                mismatches.append(dependency)
            elif node.dependency_signatures.get(dependency) != current.signature:
                mismatches.append(dependency)
        return mismatches

    def invalidate_downstream(self, changed_key: str, *, reason: str = "") -> list[str]:
        invalidated: list[str] = []
        pending = [changed_key]
        seen: set[str] = set()
        while pending:
            upstream = pending.pop(0)
            for key, default_dependencies in DEFAULT_DEPENDENCIES.items():
                dependencies = set(default_dependencies) | set(self.node(key).dependencies)
                if key in seen or upstream not in dependencies:
                    continue
                seen.add(key)
                pending.append(key)
                node = self.node(key)
                if node.exists and node.status != "Outdated":
                    node.status = "Outdated"
                    node.invalidated_by = sorted(set([*node.invalidated_by, changed_key]))
                    node.reason = reason or f"Upstream stage {changed_key} changed"
                    invalidated.append(key)
                    self._append_event("outdated", key, node, node.reason)
        return invalidated

    def mark_missing(self, key: str, *, reason: str = "") -> None:
        node = self.node(key)
        if not node.exists:
            return
        node.status = "Missing"
        node.reason = reason
        node.updated_at = utc_now()
        self.invalidate_downstream(key, reason=reason or f"{STAGE_LABELS[key]} was removed")
        self._append_event("removed", key, node, reason)

    def accept(self, key: str) -> ScientificStateNode:
        node = self.node(key)
        if not node.is_current:
            raise ValueError(f"Cannot accept {STAGE_LABELS[key]} because it is {node.status.lower()}.")
        node.accepted_revision = node.revision
        node.accepted_signature = node.signature
        node.accepted_at = utc_now()
        self._append_event("accepted", key, node, "Accepted as current project result")
        return node

    def latest_current_key(self) -> str | None:
        current = [key for key in STAGE_ORDER if self.node(key).is_current]
        return current[-1] if current else None

    def reconcile(self) -> None:
        for key in STAGE_ORDER:
            node = self.node(key)
            if node.exists and node.status == "Current" and not self.dependencies_current(key):
                node.status = "Outdated"
                node.invalidated_by = self._dependency_mismatches(key)

    def workflow_flags(self) -> dict[str, bool]:
        self.reconcile()
        return {key: self.node(key).is_current for key in STAGE_ORDER}

    def counts(self) -> dict[str, int]:
        return {
            "current": sum(self.node(key).is_current for key in STAGE_ORDER),
            "outdated": sum(self.node(key).status == "Outdated" for key in STAGE_ORDER),
            "accepted": sum(self.node(key).is_accepted_current for key in STAGE_ORDER),
        }

    def _append_event(self, action: str, key: str, node: ScientificStateNode, reason: str) -> None:
        self.events.append(
            {
                "timestamp": utc_now(),
                "action": action,
                "stage": key,
                "revision": node.revision,
                "result_uid": node.result_uid,
                "reason": reason,
            }
        )
        if len(self.events) > 250:
            self.events = self.events[-250:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_uid": self.dataset_uid,
            "nodes": {key: node.to_dict() for key, node in self.nodes.items()},
            "events": list(self.events),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetScientificState":
        uid = str(data.get("dataset_uid", ""))
        nodes = {
            key: ScientificStateNode.from_dict(value, key=key)
            for key, value in dict(data.get("nodes", {})).items()
            if key in STAGE_ORDER and isinstance(value, Mapping)
        }
        state = cls(uid, nodes)
        state.events = [dict(row) for row in data.get("events", []) if isinstance(row, Mapping)][-250:]
        state.reconcile()
        return state


class ScientificStateRegistry:
    def __init__(self):
        self.by_dataset: dict[str, DatasetScientificState] = {}

    def ensure(self, dataset_uid: str) -> DatasetScientificState:
        uid = str(dataset_uid)
        if uid not in self.by_dataset:
            self.by_dataset[uid] = DatasetScientificState(uid)
        return self.by_dataset[uid]

    def remove(self, dataset_uid: str) -> None:
        self.by_dataset.pop(str(dataset_uid), None)

    def duplicate(self, source_uid: str, target_uid: str) -> DatasetScientificState:
        source = self.ensure(source_uid)
        payload = source.to_dict()
        payload["dataset_uid"] = str(target_uid)
        clone = DatasetScientificState.from_dict(payload)
        # The duplicate is a new scientific object. Keep revisions and provenance,
        # but assign new result identifiers so cross-dataset references cannot collide.
        for node in clone.nodes.values():
            if node.exists:
                node.result_uid = uuid.uuid4().hex
                node.metadata = {**node.metadata, "duplicated_from_dataset_uid": str(source_uid)}
        self.by_dataset[str(target_uid)] = clone
        return clone

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "datasets": {uid: state.to_dict() for uid, state in self.by_dataset.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ScientificStateRegistry":
        registry = cls()
        if not isinstance(data, Mapping):
            return registry
        datasets = data.get("datasets", {})
        if isinstance(datasets, Mapping):
            for uid, payload in datasets.items():
                if not isinstance(payload, Mapping):
                    continue
                merged = dict(payload)
                merged.setdefault("dataset_uid", str(uid))
                registry.by_dataset[str(uid)] = DatasetScientificState.from_dict(merged)
        return registry

    def prune(self, dataset_uids: Iterable[str]) -> None:
        allowed = {str(uid) for uid in dataset_uids}
        self.by_dataset = {
            uid: state for uid, state in self.by_dataset.items() if uid in allowed
        }
