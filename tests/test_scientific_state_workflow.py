from __future__ import annotations

import pytest

from afruz_pxrd.scientific_state import DatasetScientificState, ScientificStateRegistry
from afruz_pxrd.undo_redo import UndoRedoHistory
from afruz_pxrd.workflow import (
    build_task_states,
    build_workflow_status,
    recommended_task_key,
    tasks_for_workspace,
)


def test_scientific_state_invalidates_all_downstream_results():
    state = DatasetScientificState("dataset-1")
    chain = (
        ("import", None),
        ("preparation", ["import"]),
        ("peaks", ["preparation"]),
        ("phase", ["peaks"]),
        ("refinement", ["phase"]),
        ("qpa", ["refinement"]),
    )
    for key, dependencies in chain:
        state.record_result(
            key, {"result": key}, accept=True, dependency_keys=dependencies
        )

    state.update_authoritative(
        "preparation",
        {"result": "changed background"},
        accept=True,
        dependency_keys=["import"],
    )
    assert state.node("preparation").status == "Current"
    for key in ("peaks", "phase", "refinement", "qpa"):
        assert state.node(key).status == "Outdated"


def test_missing_result_cannot_be_accepted():
    with pytest.raises(ValueError, match="missing"):
        DatasetScientificState("dataset-1").accept("validation")


def test_scientific_state_registry_round_trip_and_duplicate_are_independent():
    registry = ScientificStateRegistry()
    registry.ensure("source").update_authoritative("import", {"points": 10}, accept=True)
    duplicate = registry.duplicate("source", "copy")
    duplicate.update_authoritative("import", {"points": 20}, accept=True)
    restored = ScientificStateRegistry.from_dict(registry.to_dict())

    assert restored.ensure("source").node("import").revision == 1
    assert restored.ensure("copy").node("import").revision == 2
    assert restored.ensure("source").node("import").signature != restored.ensure("copy").node("import").signature


def test_guided_workflow_recommends_first_ready_task():
    states = build_workflow_status(
        {
            "has_dataset": True,
            "is_prepared": False,
            "has_peaks": False,
            "has_phase": False,
            "has_refinement": False,
            "has_qpa": False,
            "has_validation": False,
        }
    )
    task_states = build_task_states(states, {}, mode="Guided")
    recommended = recommended_task_key(states, {}, "Guided", task_states)

    assert recommended
    assert task_states[recommended].enabled
    assert task_states[recommended].recommended
    assert tasks_for_workspace("preparation", "Guided")


def test_undo_redo_restores_deep_copied_snapshots():
    application_state = {"value": 1, "nested": ["original"]}

    def snapshot():
        return application_state.copy() | {"nested": list(application_state["nested"])}

    def restore(value):
        application_state.clear()
        application_state.update(value)

    history = UndoRedoHistory(snapshot_factory=snapshot, restore_callback=restore)
    history.checkpoint("Change value")
    application_state["value"] = 2
    application_state["nested"].append("edited")

    assert history.undo() == "Change value"
    assert application_state == {"value": 1, "nested": ["original"]}
    assert history.redo() == "Change value"
    assert application_state == {"value": 2, "nested": ["original", "edited"]}
