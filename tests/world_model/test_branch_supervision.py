"""Node labels and consequence labels stay separate from navigation imitation."""

from experiments.world_model.branch_supervision import (
    build_supervision,
    execution_relation,
    node_label,
)


def _frame(name: str, openings: int, position: list[float]) -> dict:
    return {"name": name, "opening_count": openings, "position": position}


def test_a_blocked_exit_does_not_become_a_link_or_an_instruction_label():
    relation, link = execution_relation("truncated_blocked")
    assert relation == "blocked"
    assert link == "no_topological_link"
    record = {
        "origin_id": "house|episode|3",
        "role": "fit",
        "scene": "house",
        "status": "truncated_blocked",
        "event_complete": False,
        "exit_id": 0,
        "repeat_index": 0,
        "policy_id": "geometric_opening_follower/v1",
        "evidence_dir": "origins/one/repeat0_exit0",
        "pre_decision": {"chosen_exit": {"exit_id": 0, "side": "left", "bearing_left_rad": 0.8}},
    }
    frames = [
        _frame("node", 2, [0.0, 0.0, 0.0]),
        _frame("branch-000", 2, [0.2, 0.0, 0.0]),
        _frame("branch-010", 0, [1.2, 0.0, 0.0]),
    ]
    nodes, consequences, counts = build_supervision([record], lambda _record: frames)
    assert consequences[0]["instruction_action_label"] is False
    assert consequences[0]["link"] == "no_topological_link"
    assert [row["label"] for row in nodes] == ["at_decision", "possible_decision", "leaving_decision"]
    assert counts["instruction_action_labels"] == 0
    assert all(row["official_id_used_as_class"] is False for row in nodes)


def test_one_opening_stays_a_boundary_and_a_missing_depth_stays_unknown():
    assert node_label(1, "transit") == "single_exit_or_boundary"
    assert node_label(0, "transit") == "no_forward_opening"
    assert node_label(None, "transit") == "unknown"


def test_a_confirmed_event_is_a_consequence_fact():
    relation, link = execution_relation("complete")
    assert relation == "followed_to_event"
    assert link == "observed_next_event"
