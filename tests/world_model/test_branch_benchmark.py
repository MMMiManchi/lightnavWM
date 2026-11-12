"""Same-state branch benchmark rules. These tests do not start Habitat."""

from __future__ import annotations

import math

from experiments.world_model.branch_benchmark import (
    POLICY_ID,
    candidate_origins,
    collection_order,
    event_status,
    far_enough,
    integrity_report,
    paired_exits,
    pilot_origins,
    roles_are_disjoint,
)
from experiments.world_model.branch_benchmark_entry import (
    _outcome_line,
    finished_executions,
    release_stale_claims,
    try_claim,
)


def _opening(exit_id: int, bearing: float, side: str) -> dict:
    return {
        "exit_id": exit_id,
        "bearing_left_rad": bearing,
        "side": side,
        "angular_width_rad": 0.4,
    }


def _node(scene: str, step: int, openings: list[dict]) -> dict:
    return {
        "scene": scene,
        "episode_uid": f"train|{scene}|1",
        "step_index": step,
        "observation_id": f"{scene}-{step}",
        "position": [0.0, 0.0, 0.0],
        "yaw": 0.0,
        "split": "formal_fit_20261028_04",
        "openings": openings,
    }


def _record(**overrides) -> dict:
    record = {
        "origin_id": "5LpN3gDmAk7|train|3",
        "role": "fit",
        "scene": "5LpN3gDmAk7",
        "exit_id": 0,
        "policy_id": POLICY_ID,
        "prefix_sha256": "abc",
        "lightnav_used": False,
        "pre_decision": {"observation_id": "node"},
        "status": "truncated_budget",
        "event_complete": False,
        "travel_m": 1.2,
    }
    record.update(overrides)
    return record


def test_a_new_origin_must_leave_places_already_used():
    assert far_enough([1.0, 0.0, 0.0], [[1.2, 0.0, 0.0]]) is False
    assert far_enough([2.0, 0.0, 0.0], [[1.0, 0.0, 0.0]]) is True


def test_house_roles_do_not_share_a_scene():
    assert roles_are_disjoint()


def test_two_openings_must_be_separated():
    close = paired_exits([
        _opening(0, 0.0, "center"),
        _opening(1, math.radians(10), "right"),
    ])
    apart = paired_exits([
        _opening(0, math.radians(40), "left"),
        _opening(1, math.radians(-40), "right"),
    ])
    assert close is None
    assert apart is not None
    assert [item["exit_id"] for item in apart] == [0, 1]


def test_pilot_takes_one_early_origin_from_each_of_two_fit_houses():
    nodes = [
        _node("5LpN3gDmAk7", 48, [_opening(0, 1.0, "left"), _opening(1, -1.0, "right")]),
        _node("5LpN3gDmAk7", 3, [_opening(0, 0.8, "left"), _opening(1, -0.2, "center")]),
        _node("kEZ7cmS4wCh", 17, [_opening(0, 1.2, "left"), _opening(1, -1.2, "right")]),
        _node("ZMojNkEp431", 4, [_opening(0, 1.0, "left"), _opening(1, -1.0, "right")]),
    ]
    origins = candidate_origins(nodes, ("5LpN3gDmAk7", "kEZ7cmS4wCh", "ZMojNkEp431"))
    pilot = pilot_origins(origins)
    assert [item["scene"] for item in pilot] == ["5LpN3gDmAk7", "kEZ7cmS4wCh"]
    assert pilot[0]["step_index"] == 3
    ordered = collection_order(origins)
    assert [item["origin_id"] for item in ordered[:2]] == [item["origin_id"] for item in pilot]
    assert ordered[-1]["scene"] == "ZMojNkEp431"


def test_fixed_budget_without_a_second_junction_view_is_not_an_event():
    assert event_status(
        restored=True, travel_m=2.0, opening_counts=[1, 1], stalled=False, steps=300, habitat_ended=False
    ) == "truncated_budget"
    assert event_status(
        restored=True, travel_m=1.2, opening_counts=[2, 2], stalled=False, steps=12, habitat_ended=False
    ) == "complete"
    assert event_status(
        restored=False, travel_m=0.0, opening_counts=[], stalled=False, steps=0, habitat_ended=False
    ) == "restore_failed"


def test_integrity_rejects_a_future_field_and_counts_paired_exits():
    rows = [
        _record(exit_id=0, status="complete", event_complete=True),
        _record(exit_id=1, status="complete", event_complete=True),
        _record(exit_id=0, pre_decision={"future_rgb": "leak"}),
    ]
    report = integrity_report(rows[:2])
    assert report["passed"] is True
    assert report["complete_transitions"] == 2
    assert report["origins_with_two_complete_exits"] == 1
    leaked = integrity_report(rows[2:])
    assert leaked["passed"] is False


def test_one_execution_has_one_claim_and_a_dead_claim_can_be_released(tmp_path):
    assert try_claim(tmp_path, "scene|episode|3", 1, 0) is True
    assert try_claim(tmp_path, "scene|episode|3", 1, 0) is False
    dead = tmp_path / "claims" / "scene_episode_9_e0_r1"
    dead.write_text(
        '{"pid": 99999999, "origin_id": "scene|episode|9", "exit_id": 0, "repeat_index": 1}',
        encoding="utf-8",
    )
    release_stale_claims(tmp_path, set())
    assert try_claim(tmp_path, "scene|episode|9", 0, 1) is True
    assert try_claim(tmp_path, "scene|episode|3", 1, 0) is False


def test_a_start_that_never_ran_can_still_be_reported():
    line = _outcome_line(
        {"role": "development", "scene": "ZMojNkEp431", "status": "error", "error": "partition mismatch"}
    )
    assert "没有执行" in line
    assert "partition mismatch" in line


def test_a_reset_error_can_be_collected_again_and_a_real_result_cannot():
    done = finished_executions([
        {"origin_id": "a", "exit_id": 0, "repeat_index": 0, "status": "error"},
        {"origin_id": "a", "exit_id": 1, "repeat_index": 0, "status": "truncated_blocked"},
    ])
    assert ("a", 0, 0) not in done
    assert ("a", 1, 0) in done
