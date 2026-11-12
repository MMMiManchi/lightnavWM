"""Places line up across episodes. Loops stay loops. Unwalked exits stay unknown."""

from __future__ import annotations

import math

from experiments.world_model.scene_graph import build_graph, share_region


def _episode(uid: str, regions: list[str], positions: list[list[float]] | None = None) -> dict:
    if positions is None:
        positions = [[float(index), 0.0, 0.0] for index in range(len(regions))]
    return {
        "episode_uid": uid,
        "regions": regions,
        "observation_ids": [f"{uid}-{index}" for index in range(len(regions))],
        "positions": positions,
    }


def test_close_coordinates_do_not_share_a_place_across_a_wall():
    assert share_region(same_floor=True, same_island=True, segment_on_navmesh=False) is False
    assert share_region(same_floor=True, same_island=True, segment_on_navmesh=True) is True


def test_episodes_share_a_junction_and_a_loop_is_kept():
    graph = build_graph(
        [
            _episode("ep1", ["A", "B", "C"]),
            _episode("ep2", ["A", "D"]),
            _episode("ep3", ["D", "C", "A"]),
        ],
        {},
        {"B": ["turned"], "C": ["turned"], "D": ["turned"]},
    )
    regions = {place["region"] for place in graph["places"]}
    assert regions == {"A", "B", "C", "D"}
    assert any(place["region"] == "A" and "junction" in place["kinds"] for place in graph["places"])
    walked = {(edge["from_region"], edge["to_region"]) for edge in graph["edges"]}
    assert {("A", "B"), ("B", "C"), ("A", "D"), ("D", "C"), ("C", "A")} <= walked
    assert graph["cycles"]
    assert any(set(cycle) >= {"A", "D", "C"} for cycle in graph["cycles"])


def test_two_nearby_regions_stay_separate_and_an_unwalked_exit_has_no_destination():
    here = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]
    graph = build_graph(
        [
            _episode("left", ["L", "L2"], here),
            _episode("right", ["R"], [[0.2, 0.0, 0.0]]),
        ],
        {
            "L": [
                {"exit_id": "ahead", "bearing_rad": 0.0},
                {"exit_id": "side", "bearing_rad": math.pi / 2},
            ]
        },
        {"L": ["turned"], "L2": ["instruction_stop"], "R": ["turned"]},
    )
    assert {place["region"] for place in graph["places"]} == {"L", "L2", "R"}
    outcomes = {
        (row["exit_id"], row["outcome"], row["next_place_region"])
        for row in graph["branches"]
        if row["place_region"] == "L"
    }
    assert ("ahead", "walked", "L2") in outcomes
    assert ("side", "unknown", None) in outcomes
