"""Key places come from the route, not from how far apart they are."""

from __future__ import annotations

import math

from experiments.world_model.key_nodes import goal_stop, places_on_path, room_doors, walking_turns


def _line(start, step, count, y=0.0):
    return [[start[0] + step[0] * i, y, start[1] + step[1] * i] for i in range(count)]


def test_a_straight_hall_has_no_turn_and_a_close_corner_still_counts():
    straight = _line((0.0, 0.0), (0.2, 0.0), 20)
    assert walking_turns(straight) == []
    gentle = []
    for index in range(40):
        angle = math.radians(20.0) * index / 39
        gentle.append([math.sin(angle) * 6.0, 0.0, math.cos(angle) * 6.0])
    assert walking_turns(gentle) == []
    corner = _line((0.0, 0.0), (0.2, 0.0), 6) + _line((1.0, 0.2), (0.0, 0.2), 6)
    turns = walking_turns(corner)
    assert len(turns) == 1
    assert corner[turns[0]][0] == 1.0


def test_a_door_into_a_room_is_key_and_a_narrow_hall_is_not():
    positions = _line((0.0, 0.0), (0.2, 0.0), 16)
    assert room_doors(positions, [0.4] * len(positions)) == []
    clearance = [0.4] * 6 + [0.4, 0.5, 1.4, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6]
    doors = room_doors(positions, clearance)
    assert [door["kind"] for door in doors] == ["entered_room"]
    assert doors[0]["index"] < 8


def test_the_instruction_stop_counts_only_when_the_walk_reaches_it():
    positions = _line((0.0, 0.0), (0.5, 0.0), 6)
    assert goal_stop(positions, [20.0, 0.0, 0.0]) is None
    assert goal_stop(positions, [2.0, 0.0, 0.0]) == 4


def test_two_corners_in_the_same_spot_stay_one_place():
    positions = (
        _line((0.0, 0.0), (0.2, 0.0), 6)
        + _line((1.0, 0.2), (0.0, 0.2), 3)
        + _line((1.0, 0.6), (0.2, 0.0), 6)
    )
    places = places_on_path(positions, [1.5] * len(positions), None)
    assert len(places) == 1
    assert places[0]["kinds"] == ["turned"]
