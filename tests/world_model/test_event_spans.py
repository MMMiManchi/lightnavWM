"""CPU checks for node-to-node spans. They do not read the dataset."""

from __future__ import annotations

import math

from experiments.world_model.event_spans import (
    places_with_two_exits,
    recollection_verdict,
    segment_episode,
    start_view_ids,
)


def _pose(index: int, x: float, z: float, yaw: float) -> dict:
    return {
        "scene": "scene",
        "episode_uid": "episode",
        "observation_id": f"obs-{index}",
        "position": (x, 0.0, z),
        "yaw": yaw,
    }


def test_a_straight_walk_is_one_span_to_the_episode_end():
    poses = [_pose(index, index * 0.2, 0.0, 0.0) for index in range(8)]
    spans = segment_episode(poses)
    assert len(spans) == 1
    assert spans[0]["end_kind"] == "episode_end"
    assert spans[0]["end_observation_id"] == "obs-7"
    assert spans[0]["end_view_ids"] == ["obs-7"]
    assert spans[0]["recorded_control_steps"] == 7


def test_a_quarter_turn_closes_a_node_before_the_episode_end():
    poses = [_pose(0, 0.0, 0.0, 0.0), _pose(1, 0.0, 0.0, math.pi / 2.0)]
    poses.extend(_pose(index, 0.2 * (index - 1), 0.0, math.pi / 2.0) for index in range(2, 6))
    spans = segment_episode(poses)
    assert spans[0]["end_kind"] == "turn"
    assert spans[0]["end_observation_id"] == "obs-2"
    assert spans[0]["end_view_ids"] == ["obs-0", "obs-1", "obs-2"]
    assert spans[-1]["end_kind"] == "episode_end"
    assert spans[-1]["end_view_ids"] == ["obs-5"]


def test_the_turn_bout_drops_the_corridor():
    poses = [_pose(index, 0.0, -0.25 * index, 0.0) for index in range(7)]
    poses.append(_pose(7, 0.0, -1.5, 0.4))
    poses.append(_pose(8, 0.0, -1.5, 0.8))
    poses.append(_pose(9, 0.0, -1.6, 0.8))
    spans = segment_episode(poses)
    assert spans[0]["end_kind"] == "turn"
    assert spans[0]["end_view_ids"] == ["obs-6", "obs-7", "obs-8", "obs-9"]
    assert "obs-0" not in spans[0]["end_view_ids"]


def test_a_slow_turn_keeps_the_spin_and_drops_the_corridor():
    poses = [_pose(index, 0.0, -0.25 * index, 0.0) for index in range(5)]
    yaw = 0.0
    for step in range(40):
        yaw += 0.02
        poses.append(_pose(5 + step, 0.0, -1.0, yaw))
    spans = segment_episode(poses)
    assert spans[0]["end_kind"] == "turn"
    assert spans[0]["end_view_ids"][0] == "obs-4"
    assert spans[0]["end_view_ids"][-1] == spans[0]["end_observation_id"]
    assert "obs-0" not in spans[0]["end_view_ids"]
    assert len(spans[0]["end_view_ids"]) >= 2


def test_the_next_span_reuses_the_settled_turn_views():
    poses = [_pose(0, 0.0, 0.0, 0.0), _pose(1, 0.0, 0.0, math.pi / 2.0)]
    poses.append(_pose(2, 0.2, 0.0, math.pi / 2.0))
    poses.append(_pose(3, 1.0, 0.0, math.pi / 2.0))
    poses.append(_pose(4, 1.0, 0.0, math.pi))
    poses.append(_pose(5, 1.2, 0.0, math.pi))
    spans = segment_episode(poses)
    starts = start_view_ids(spans)
    assert starts[0] == ["obs-0"]
    assert starts[1] == spans[0]["end_view_ids"]
    assert spans[0]["end_view_ids"][-1] == "obs-2"
    assert "obs-2" not in spans[1]["end_view_ids"]


def test_zero_motion_does_not_invent_nodes():
    poses = [_pose(index, 0.0, 0.0, 0.0) for index in range(6)]
    spans = segment_episode(poses)
    assert len(spans) == 1
    assert spans[0]["path_m"] == 0.0


def test_the_next_corner_of_the_same_walk_is_not_a_second_exit():
    spans = [
        {
            "scene": "scene",
            "episode_uid": "a",
            "start_position": [0.0, 0.0, 0.0],
            "start_yaw": 0.0,
            "start_observation_id": "obs-0",
            "end_observation_id": "obs-1",
            "start_is_episode_start": True,
        },
        {
            "scene": "scene",
            "episode_uid": "a",
            "start_position": [0.4, 0.0, 0.0],
            "start_yaw": math.pi / 2.0,
            "start_observation_id": "obs-1",
            "end_observation_id": "obs-2",
            "start_is_episode_start": False,
        },
    ]
    assert places_with_two_exits(spans) == []


def test_two_departure_headings_at_one_place_are_two_exits():
    shared = {
        "scene": "scene",
        "start_position": [0.0, 0.0, 0.0],
        "start_is_episode_start": False,
    }
    spans = [
        {
            **shared,
            "episode_uid": "a",
            "start_yaw": 0.0,
            "start_observation_id": "a0",
            "end_observation_id": "a1",
        },
        {
            **shared,
            "episode_uid": "b",
            "start_yaw": math.pi / 2.0,
            "start_observation_id": "b0",
            "end_observation_id": "b1",
        },
    ]
    places = places_with_two_exits(spans)
    assert len(places) == 1
    assert places[0]["exit_count"] == 2
    assert recollection_verdict(span_count=2, place_count=1)["call"] == "some_places_have_two_exits"
    assert recollection_verdict(span_count=4, place_count=0)["call"] == "taken_path_only"
    assert recollection_verdict(span_count=0, place_count=0)["doors_and_forks_labeled"] is False
