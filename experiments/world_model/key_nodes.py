"""Key places on a walked route.

A key place is where the route turns onto another path, passes between a
passage and a room, or reaches the place the instruction said to stop.
How far apart two places are does not decide whether either one is key.
Half a meter is only used to read which way the path is going. Three
quarters of a meter only means it is the same place again.
"""

from __future__ import annotations

import math
from typing import Any

TURN_ONTO_PATH_RAD = math.radians(60.0)
DIRECTION_M = 0.5
DOOR_CLEARANCE_M = 0.75
ROOM_CLEARANCE_M = 1.2
SPACE_LOOK_M = 0.6
GOAL_REACH_M = 3.0
SAME_PLACE_M = 0.75
SAME_FLOOR_M = 0.8


def walking_turns(positions: list[list[float]]) -> list[int]:
    """Indices where the walked direction changes onto another path."""
    scored = []
    for index in range(len(positions)):
        behind = _point_along(positions, index, -1, DIRECTION_M)
        ahead = _point_along(positions, index, 1, DIRECTION_M)
        if behind is None or ahead is None:
            continue
        angle = _corner_angle(behind, positions[index], ahead)
        if angle >= TURN_ONTO_PATH_RAD:
            scored.append((index, angle))
    return [_peak(group) for group in _contiguous(scored)]


def room_doors(
    positions: list[list[float]], clearances: list[float | None]
) -> list[dict[str, Any]]:
    """Where the path passes between a tight passage and a room."""
    if len(positions) != len(clearances):
        raise ValueError("clearance must line up with the path")
    found = []
    for index in range(len(positions)):
        behind = _clearance_along(positions, clearances, index, -1, SPACE_LOOK_M)
        ahead = _clearance_along(positions, clearances, index, 1, SPACE_LOOK_M)
        here = clearances[index]
        if behind is None or ahead is None or here is None:
            continue
        if behind <= DOOR_CLEARANCE_M and ahead >= ROOM_CLEARANCE_M:
            found.append((index, "entered_room", here))
        elif behind >= ROOM_CLEARANCE_M and ahead <= DOOR_CLEARANCE_M:
            found.append((index, "left_room", here))
    doors = []
    for group in _contiguous_doors(found):
        kind = group[0][1]
        index = min(group, key=lambda item: item[2])[0]
        doors.append({"index": index, "kind": kind})
    return doors


def goal_stop(positions: list[list[float]], goal: list[float] | None) -> int | None:
    """The pose that reaches the instruction's stop, if the walk got there."""
    if goal is None or not positions:
        return None
    best = min(range(len(positions)), key=lambda index: _horizontal(positions[index], goal))
    if _horizontal(positions[best], goal) > GOAL_REACH_M:
        return None
    return best


def places_on_path(
    positions: list[list[float]],
    clearances: list[float | None],
    goal: list[float] | None,
) -> list[dict[str, Any]]:
    """One episode's key places, with a return kept as the earlier place."""
    events: list[dict[str, Any]] = []
    for index in walking_turns(positions):
        events.append({"index": index, "kinds": ["turned"]})
    for door in room_doors(positions, clearances):
        events.append({"index": door["index"], "kinds": [door["kind"]]})
    stop = goal_stop(positions, goal)
    if stop is not None:
        events.append({"index": stop, "kinds": ["instruction_stop"]})
    return _merge_events(positions, events)


def same_place(left: list[float], right: list[float]) -> bool:
    if abs(left[1] - right[1]) > SAME_FLOOR_M:
        return False
    return _horizontal(left, right) <= SAME_PLACE_M


def _point_along(positions: list[list[float]], index: int, sign: int, distance: float):
    travelled = 0.0
    cursor = index
    while True:
        nxt = cursor + sign
        if nxt < 0 or nxt >= len(positions):
            return None
        travelled += _horizontal(positions[cursor], positions[nxt])
        cursor = nxt
        if travelled >= distance:
            return positions[cursor]


def _clearance_along(
    positions: list[list[float]],
    clearances: list[float | None],
    index: int,
    sign: int,
    distance: float,
) -> float | None:
    travelled = 0.0
    cursor = index
    while True:
        nxt = cursor + sign
        if nxt < 0 or nxt >= len(positions):
            return None
        travelled += _horizontal(positions[cursor], positions[nxt])
        cursor = nxt
        if travelled >= distance:
            return clearances[cursor]


def _corner_angle(behind, here, ahead) -> float:
    incoming = (here[0] - behind[0], here[2] - behind[2])
    outgoing = (ahead[0] - here[0], ahead[2] - here[2])
    return abs(_angle_between(incoming, outgoing))


def _angle_between(left: tuple[float, float], right: tuple[float, float]) -> float:
    left_norm = math.hypot(*left)
    right_norm = math.hypot(*right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    cosine = (left[0] * right[0] + left[1] * right[1]) / (left_norm * right_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return math.acos(cosine)


def _contiguous(scored: list[tuple[int, float]]) -> list[list[tuple[int, float]]]:
    groups: list[list[tuple[int, float]]] = []
    for index, angle in scored:
        if groups and index <= groups[-1][-1][0] + 1:
            groups[-1].append((index, angle))
        else:
            groups.append([(index, angle)])
    return groups


def _contiguous_doors(found: list[tuple[int, str, float]]) -> list[list[tuple[int, str, float]]]:
    groups: list[list[tuple[int, str, float]]] = []
    for item in found:
        if groups and item[0] <= groups[-1][-1][0] + 1 and item[1] == groups[-1][-1][1]:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def _peak(group: list[tuple[int, float]]) -> int:
    return max(group, key=lambda item: item[1])[0]


def _merge_events(positions: list[list[float]], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(events, key=lambda item: int(item["index"]))
    places: list[dict[str, Any]] = []
    for event in ordered:
        position = positions[event["index"]]
        match = next((place for place in places if same_place(place["position"], position)), None)
        if match is None:
            places.append(
                {
                    "index": int(event["index"]),
                    "position": [float(value) for value in position],
                    "kinds": list(event["kinds"]),
                }
            )
            continue
        for kind in event["kinds"]:
            if kind not in match["kinds"]:
                match["kinds"].append(kind)
    return places


def _horizontal(left: list[float], right: list[float]) -> float:
    return math.hypot(left[0] - right[0], left[2] - right[2])
