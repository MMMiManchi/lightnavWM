"""Official viewpoints label entries, visible exits, and returns. They are not the runtime input."""

from __future__ import annotations

import math

from experiments.world_model.official_nodes import (
    angle_candidate_exit_ids,
    annotate_walk,
    clearance_band,
    connected_groups,
    decision_places,
    depth_cone_status,
    detached_frame,
    exit_in_view,
    habitat_position,
    image_column,
    optical_axis_distance,
    rebuild_place,
    teacher_graph,
)


def _node(name: str, x: float, z: float, y: float = 0.0) -> dict:
    return {"id": name, "position": (x, y, z)}


def _place(viewpoint: str, exits: list[tuple[str, float, tuple[float, float, float]]]) -> dict:
    return {
        "place_id": f"house:{viewpoint}",
        "viewpoint_id": viewpoint,
        "position": [0.0, 0.0, 0.0],
        "member_positions": [[0.0, 0.0, 0.0]],
        "exits": [
            {"exit_id": exit_id, "bearing_rad": bearing, "toward_position": list(target)}
            for exit_id, bearing, target in exits
        ],
    }


def test_matterport_pose_lands_on_the_habitat_floor():
    pose = [0.0] * 16
    pose[3], pose[7], pose[11] = 1.0, 2.0, 1.6
    x, y, z = habitat_position(pose, 1.5)
    assert (x, z) == (1.0, -2.0)
    assert abs(y - 0.1) < 1e-9


def test_a_straight_hall_is_not_a_decision_and_a_junction_is():
    hall = [_node(name, x, 0.0) for name, x in (("A", 0.0), ("B", 2.0), ("C", 4.0), ("D", 6.0))]
    assert decision_places(hall, [("A", "B"), ("B", "C"), ("C", "D")]) == []
    junction = [
        _node("A0", -4.0, 0.0),
        _node("A", -2.0, 0.0),
        _node("B", 0.0, 0.0),
        _node("C", 2.0, 0.0),
        _node("C0", 4.0, 0.0),
        _node("D", 0.0, 2.0),
        _node("D0", 0.0, 4.0),
    ]
    links = [("A0", "A"), ("A", "B"), ("B", "C"), ("C", "C0"), ("B", "D"), ("D", "D0")]
    places = decision_places(junction, links)
    assert [place["viewpoint_id"] for place in places] == ["B"]
    assert len(places[0]["exits"]) == 3


def test_a_one_step_side_leaf_does_not_create_a_node():
    nodes = [
        _node("A0", -4.0, 0.0),
        _node("A", -2.0, 0.0),
        _node("B", 0.0, 0.0),
        _node("C", 2.0, 0.0),
        _node("C0", 4.0, 0.0),
        _node("D", 0.0, 1.0),
    ]
    links = [("A0", "A"), ("A", "B"), ("B", "C"), ("C", "C0"), ("B", "D")]
    assert decision_places(nodes, links) == []


def test_adjacent_forks_become_one_official_viewpoint():
    nodes = [
        _node("A0", -4.0, 0.0),
        _node("A", -2.0, 0.0),
        _node("B", 0.0, 0.0),
        _node("D", 0.0, 2.0),
        _node("D0", 0.0, 4.0),
        _node("E", 1.0, 0.0),
        _node("C", 3.0, 0.0),
        _node("C0", 5.0, 0.0),
        _node("F", 1.0, -2.0),
        _node("F0", 1.0, -4.0),
    ]
    links = [
        ("A0", "A"),
        ("A", "B"),
        ("B", "D"),
        ("D", "D0"),
        ("B", "E"),
        ("E", "C"),
        ("C", "C0"),
        ("E", "F"),
        ("F", "F0"),
    ]
    places = decision_places(nodes, links)
    assert len(places) == 1
    assert set(places[0]["member_viewpoint_ids"]) == {"B", "E"}


def test_a_square_loop_keeps_all_four_corners():
    nodes = [
        _node("A", 0.0, 0.0),
        _node("B", 4.0, 0.0),
        _node("C", 4.0, 4.0),
        _node("D", 0.0, 4.0),
    ]
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    graph = teacher_graph(nodes, edges)
    assert graph["cycle_rank"] == 1
    places = decision_places(nodes, edges)
    assert {place["viewpoint_id"] for place in places} == {"A", "B", "C", "D"}
    assert {place["kind"] for place in places} == {"turn"}


def test_yaw_zero_sees_the_forward_exit_only():
    forward = math.pi
    behind = 0.0
    assert exit_in_view(forward, yaw=0.0) is True
    assert exit_in_view(behind, yaw=0.0) is False


def test_a_shifted_camera_can_leave_the_anchor_cone():
    target = (-2.457456132866975, 0.0, -1.7207293090531386)
    at_anchor = angle_candidate_exit_ids((0.0, 0.0, 0.0), 0.0, [{"exit_id": "e1", "toward_position": target}])
    shifted = angle_candidate_exit_ids((1.4, 0.0, 0.0), 0.0, [{"exit_id": "e1", "toward_position": target}])
    assert at_anchor == ["e1"]
    assert shifted == []


def test_a_shared_prefix_is_not_rewritten():
    prefix = {"source_observation_id": "path#0", "position": [0.0, 0.0, 0.0], "has_rgb": False}
    first = detached_frame(prefix, "exit-0#0")
    second = detached_frame(prefix, "exit-1#0")
    first["source_observation_id"] = "changed"
    assert prefix["source_observation_id"] == "path#0"
    assert second["source_observation_id"] == "path#0"
    assert first["occurrence_id"] != second["occurrence_id"]


def test_entering_leaving_and_returning_are_labeled_from_the_walk():
    place = _place("node", [("ahead", math.pi, (0.0, 0.0, -2.0)), ("side", math.pi / 2.0, (2.0, 0.0, 0.0))])
    frames = [
        {"observation_id": "near", "position": [2.5, 0.0, 0.0], "yaw": 0.0},
        {"observation_id": "enter", "position": [0.0, 0.0, 0.0], "yaw": 0.0},
        {"observation_id": "stay", "position": [0.4, 0.0, 0.0], "yaw": 0.0},
        {"observation_id": "leave", "position": [5.0, 0.0, 0.0], "yaw": 0.0},
        {"observation_id": "return", "position": [0.0, 0.0, 0.0], "yaw": 0.0},
    ]
    rows, seen = annotate_walk(frames, [place])
    assert [row["phase"] for row in rows] == ["approach", "enter", "inside", "leave", "enter"]
    assert [row["entered_decision_region"] for row in rows] == [False, True, False, False, True]
    assert [row["revisit"] for row in rows] == [False, False, False, False, True]
    assert rows[1]["angle_candidate_exit_ids"] == ["ahead"]
    assert rows[1]["visibility"] == "unknown"
    assert rows[3]["in_decision_region"] is False
    assert rows[3]["place_id"] == "house:node"
    assert seen == {"house:node"}


def test_moving_into_the_next_region_is_another_entry():
    first = _place("one", [("ahead", math.pi, (0.0, 0.0, -2.0))])
    second = _place("two", [("ahead", math.pi, (4.0, 0.0, -2.0))])
    second["position"] = [4.0, 0.0, 0.0]
    second["member_positions"] = [[4.0, 0.0, 0.0]]
    frames = [
        {"observation_id": "a", "position": [0.0, 0.0, 0.0], "yaw": 0.0},
        {"observation_id": "b", "position": [4.0, 0.0, 0.0], "yaw": 0.0},
    ]
    rows, seen = annotate_walk(frames, [first, second])
    assert [row["phase"] for row in rows] == ["started_inside", "enter"]
    assert rows[1]["entered_decision_region"] is True
    assert seen == {"house:one", "house:two"}


def test_a_walk_that_starts_inside_is_not_an_entry():
    place = _place("node", [("ahead", math.pi, (0.0, 0.0, -2.0))])
    frames = [{"observation_id": "already", "position": [0.0, 0.0, 0.0], "yaw": 0.0}]
    rows, _seen = annotate_walk(frames, [place])
    assert rows[0]["phase"] == "started_inside"
    assert rows[0]["entered_decision_region"] is False


def test_a_nearby_place_across_a_wall_is_not_entered():
    place = _place("node", [("ahead", math.pi, (0.0, 0.0, -2.0))])
    frames = [{"observation_id": "beside", "position": [1.0, 0.0, 0.0], "yaw": 0.0}]
    rows, seen = annotate_walk(frames, [place], reachable=lambda _position, _member: False)
    assert rows[0]["phase"] == "outside"
    assert rows[0]["entered_decision_region"] is False
    assert seen == set()


def test_depth_column_uses_the_pinhole_and_the_optical_axis():
    assert image_column(math.radians(45), 480) == 101
    assert image_column(0.0, 480) == 240
    assert math.isclose(optical_axis_distance((0.0, 0.0, 0.0), (1.0, 0.0, -1.0), 0.0), 1.0)


def test_a_split_child_rebuilds_its_own_exits():
    nodes = [
        _node("A0", -4.0, 0.0),
        _node("A", -2.0, 0.0),
        _node("B", 0.0, 0.0),
        _node("D", 0.0, 2.0),
        _node("D0", 0.0, 4.0),
        _node("E", 1.0, 0.0),
        _node("C", 3.0, 0.0),
        _node("C0", 5.0, 0.0),
        _node("F", 1.0, -2.0),
        _node("F0", 1.0, -4.0),
    ]
    links = [
        ("A0", "A"),
        ("A", "B"),
        ("B", "D"),
        ("D", "D0"),
        ("B", "E"),
        ("E", "C"),
        ("C", "C0"),
        ("E", "F"),
        ("F", "F0"),
    ]
    graph = teacher_graph(nodes, links)
    parent = decision_places(nodes, links)[0]
    branch_nodes = {node for node, neighbors in graph["adjacency"].items() if len(neighbors) >= 3}
    child = rebuild_place(["B"], parent["kind"], graph["positions"], graph["adjacency"], branch_nodes)
    assert connected_groups(["B", "E"], lambda left, right: False) == [["B"], ["E"]]
    assert child is not None
    assert child["viewpoint_id"] == "B"
    assert child["member_viewpoint_ids"] == ["B"]
    assert "E" in {exit_row["toward_viewpoint_id"] for exit_row in child["exits"]}
    assert "E" not in {exit_row["toward_viewpoint_id"] for exit_row in parent["exits"]}
    blocked = rebuild_place(
        ["B"],
        parent["kind"],
        graph["positions"],
        graph["adjacency"],
        branch_nodes,
        walkable=lambda _origin, _target: False,
    )
    assert blocked is None


def test_clearance_and_depth_are_named_as_geometry():
    assert clearance_band(0.4) == "tight_passage"
    assert clearance_band(2.0) == "open_interior"
    assert clearance_band(1.0) == "intermediate"
    assert depth_cone_status([1.0] * 12, 4.0) == "blocked"
    assert depth_cone_status([5.0] * 12, 4.0) == "not_blocked"
    assert depth_cone_status([1.0], 4.0) == "unknown"


def test_an_earlier_walk_makes_the_same_official_node_known():
    place = _place("node", [("ahead", math.pi, (0.0, 0.0, -2.0))])
    frames = [{"observation_id": "again", "position": [0.0, 0.0, 0.0], "yaw": 0.0}]
    rows, _seen = annotate_walk(frames, [place], already_known={"house:node"})
    assert rows[0]["revisit"] is False
    assert rows[0]["known_place"] is True
