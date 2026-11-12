"""CPU checks for the unwalked-exit walk. They do not start Habitat."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.world_model.branch_exit import (
    ANG_DPS,
    BRANCH_DISTANCE_M,
    BRANCH_STEPS,
    CAMERA_HEIGHT_M,
    EVAL_EPISODE_STEPS,
    FIT_SPLIT,
    FORWARD_MPS,
    LIN_MPS,
    STUCK_STEPS,
    arrival_stop,
    batch_passes,
    branch_stop,
    branch_velocity,
    exits_to_walk,
    follow_opening,
    is_arrival,
    motion_stalled,
    new_fit_houses,
    normalize_velocity,
    remaining_bearing,
    replay_actions,
    restore_matches,
    search_stop,
    unwalked_targets,
)
from experiments.world_model.ego_warp import intrinsics
from experiments.world_model.yaw_shift import HEIGHT, WIDTH


def _opening(exit_id: int, side: str = "right") -> dict:
    return {"exit_id": exit_id, "side": side, "bearing_left_rad": -0.4}


def _node(split: str, step: int, taken, openings: list[dict]) -> dict:
    return {
        "split": split,
        "scene": "VzqfbhrpDEA",
        "episode_uid": "R2R_VLNCE_v1-3_preprocessed|train|VzqfbhrpDEA|151",
        "observation_id": f"obs-{step}",
        "step_index": step,
        "position": [1.0, 0.1, 2.0],
        "yaw": 0.2,
        "taken_exit_id": taken,
        "openings": openings,
    }


def _target(episode_uid: str, step: int, exit_id: int) -> dict:
    return {"episode_uid": episode_uid, "step_index": step, "exit_id": exit_id}


def test_a_finished_exit_is_not_collected_again_and_one_episode_is_not_repeated():
    targets = [
        _target("ep-b", 8, 0),
        _target("ep-a", 6, 1),
        _target("ep-a", 90, 0),
        _target("ep-a", 40, 1),
    ]
    passes = batch_passes(targets, [("ep-a", 6, 1)])
    assert [
        [(item["episode_uid"], item["step_index"], item["exit_id"]) for item in batch]
        for batch in passes
    ] == [
        [("ep-a", 40, 1), ("ep-b", 8, 0)],
        [("ep-a", 90, 0)],
    ]


def test_new_houses_skip_dev_and_houses_already_walked():
    rows = [
        {"scene": "ur6pFq6Qu1A", "episode_id": "1", "partition": "train_fit"},
        {"scene": "5LpN3gDmAk7", "episode_id": "2", "partition": "train_fit"},
        {"scene": "17DRP5sb8fy", "episode_id": "9", "partition": "train_dev"},
        {"scene": "17DRP5sb8fy", "episode_id": "3", "partition": "train_fit"},
        {"scene": "1LXtFkjw3qL", "episode_id": "4", "partition": "train_fit"},
    ]
    chosen = new_fit_houses(rows, {"5LpN3gDmAk7"}, limit=8)
    assert [row["scene"] for row in chosen] == ["17DRP5sb8fy", "1LXtFkjw3qL"]
    assert chosen[0]["episode_id"] == "3"


def test_a_walk_uses_the_evaluation_step_budget_and_not_a_shorter_cap():
    assert search_stop(0, 0.0, 2) == "junction"
    assert search_stop(3, 0.2, 2) is None
    assert search_stop(80, 12.0, 1) is None
    assert search_stop(EVAL_EPISODE_STEPS, 1.0, 1) == "no_junction"
    openings = [
        {"exit_id": 0, "angular_width_rad": 0.2},
        {"exit_id": 1, "angular_width_rad": 0.8},
        {"exit_id": 2, "angular_width_rad": 0.5},
    ]
    assert [item["exit_id"] for item in exits_to_walk(openings)] == [1, 2]
    assert arrival_stop(3, 1.0, 2) == "next_decision"
    assert arrival_stop(3, 30.0, 1) is None
    assert arrival_stop(EVAL_EPISODE_STEPS, 1.0, 1) == "truncated_budget"
    assert is_arrival(arrival_stop(EVAL_EPISODE_STEPS, 1.0, 1)) is False


def test_the_shortest_train_fit_exit_is_first_and_dev_is_left_out():
    nodes = [
        _node("formal_dev_20261028_01", 1, 0, [_opening(1)]),
        _node(FIT_SPLIT, 1, None, [_opening(1)]),
        _node(FIT_SPLIT, 9, 0, [_opening(1, "left")]),
        _node(FIT_SPLIT, 6, 0, [_opening(0, "center"), _opening(1, "right")]),
    ]
    chosen = unwalked_targets(nodes)
    assert [item["step_index"] for item in chosen] == [6, 9]
    assert chosen[0]["exit_id"] == 1
    assert chosen[0]["side"] == "right"
    assert chosen[0]["taken_exit_id"] == 0


def _transition(t: int, executed: bool = True) -> dict:
    return {
        "episode_uid": "ep",
        "t": t,
        "action": {
            "execution_ack": {"executed": executed},
            "final_habitat_action": {"action": "velocity_control", "t": t},
        },
    }


def test_replay_uses_only_executed_steps_before_the_node():
    actions = replay_actions([_transition(1), _transition(0), _transition(2)], "ep", 2)
    assert [item["t"] for item in actions] == [0, 1]


def test_a_gap_or_a_skipped_command_is_not_replayed():
    with pytest.raises(RuntimeError):
        replay_actions([_transition(0), _transition(2)], "ep", 2)
    with pytest.raises(RuntimeError):
        replay_actions([_transition(0, executed=False)], "ep", 1)


def test_the_walk_turns_toward_the_opening_then_goes_forward():
    left = branch_velocity(0.5)
    right = branch_velocity(-0.5)
    ahead = branch_velocity(0.0)
    assert left["action_args"]["angular_velocity"] > 0
    assert right["action_args"]["angular_velocity"] < 0
    assert left["action_args"]["linear_velocity"] == -1.0
    assert ahead["action_args"]["linear_velocity"] == pytest.approx(
        normalize_velocity(FORWARD_MPS, LIN_MPS)
    )
    assert ahead["action_args"]["angular_velocity"] == 0.0
    assert normalize_velocity(0.0, LIN_MPS) == -1.0
    assert normalize_velocity(60.0, ANG_DPS) == pytest.approx(0.2)


def test_turning_left_brings_a_left_opening_toward_the_center():
    before = remaining_bearing(0.5, yaw_now=0.0, yaw_at_node=0.0)
    after = remaining_bearing(0.5, yaw_now=0.2, yaw_at_node=0.0)
    assert before == pytest.approx(0.5)
    assert after == pytest.approx(0.3)


def test_a_far_replay_is_not_treated_as_the_same_place():
    assert restore_matches([0.0, 0.0, 0.0], 0.0, [0.1, 0.0, 0.0], 0.1)
    assert not restore_matches([0.0, 0.0, 0.0], 0.0, [1.0, 0.0, 0.0], 0.0)


def test_stopping_early_is_not_an_arrival():
    assert branch_stop(BRANCH_STEPS, 0.2, 1) == "truncated_budget"
    assert branch_stop(3, BRANCH_DISTANCE_M, 1) == "truncated_distance"
    assert branch_stop(3, 0.2, 2) is None
    assert branch_stop(3, 1.0, 2) == "next_decision"
    assert branch_stop(3, 0.2, 1, stalled=True) == "truncated_blocked"
    assert is_arrival("truncated_budget") is False
    assert is_arrival("truncated_blocked") is False
    assert is_arrival("next_decision") is True
    assert math.isclose(BRANCH_DISTANCE_M, 2.0)
    assert motion_stalled([0.0, 0.0, 0.0]) is False
    assert motion_stalled([0.0] * STUCK_STEPS) is True
    assert motion_stalled([0.0, 0.0, 0.05, 0.0]) is False


def _room(distance_m: float = 8.0) -> np.ndarray:
    return np.full((HEIGHT, WIDTH), distance_m, dtype=np.float64)


def _block(depth: np.ndarray, columns: slice, distance_m: float = 0.35) -> None:
    depth[140:270, columns] = distance_m


def test_a_clear_opening_is_walked():
    command, mode = follow_opening("right", _room())
    assert mode == "drive"
    assert command["action_args"]["linear_velocity"] == pytest.approx(
        normalize_velocity(FORWARD_MPS, LIN_MPS)
    )


def test_a_chair_under_the_view_is_not_driven_into():
    depth = _room()
    _block(depth, slice(140, 340))
    command, mode = follow_opening("right", depth)
    assert mode == "turn"
    assert command["action_args"]["angular_velocity"] < 0
    assert command["action_args"]["linear_velocity"] == -1.0


def test_the_other_exit_is_not_used_to_get_around_the_chair():
    depth = _room()
    _block(depth, slice(140, WIDTH))
    assert follow_opening("right", depth) is None


def test_a_flat_floor_is_not_a_wall():
    _fx, fy, _cx, cy = intrinsics()
    depth = _room()
    for row in range(int(cy) + 1, HEIGHT):
        depth[row, :] = CAMERA_HEIGHT_M * fy / (row - cy)
    command, mode = follow_opening("right", depth)
    assert mode == "drive"
    assert command["action_args"]["linear_velocity"] > -0.95
