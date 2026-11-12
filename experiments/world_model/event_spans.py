"""Turn existing trajectories into node-to-node spans.

A span runs from one pose to the next pose where measured heading has changed
by at least a quarter turn, or to the end of the episode. Control steps stay
inside the span. They are not the prediction target. Doorways and forks are
not labeled in these logs, so a straight walk through a door stays one span.

A turn node keeps the frames from the first step of that turn through the
settled frame. The straight walk before the turn is not part of the node.
An episode start or an episode end keeps its single frame.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from experiments.world_model.pose_math import body_frame_delta, wrap_pi

TURN_RAD = math.pi / 4.0
STRAIGHT_STEP_RAD = 0.03
WOBBLE_RESET_M = 0.5
SAME_PLACE_M = 0.75
DISTINCT_EXIT_RAD = 0.5


def segment_episode(poses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split one episode into spans. ``poses`` is in time order and has at least two rows."""
    if len(poses) < 2:
        raise ValueError("an episode needs a start and an end pose")
    spans: list[dict[str, Any]] = []
    origin = 0
    yaw_acc = 0.0
    path_acc = 0.0
    straight_path = 0.0
    bout_origin = 0
    step_yaw: list[float] = []
    for index in range(len(poses) - 1):
        forward, left, dyaw, _up = body_frame_delta(
            poses[index]["position"],
            poses[index]["yaw"],
            poses[index + 1]["position"],
            poses[index + 1]["yaw"],
        )
        step_path = math.hypot(forward, left)
        yaw_acc += dyaw
        path_acc += step_path
        step_yaw.append(abs(dyaw))
        if abs(dyaw) < STRAIGHT_STEP_RAD:
            straight_path += step_path
        else:
            straight_path = 0.0
        turn_closed = abs(dyaw) < STRAIGHT_STEP_RAD and abs(yaw_acc) >= TURN_RAD
        wobble = straight_path >= WOBBLE_RESET_M and abs(yaw_acc) < TURN_RAD
        if wobble:
            yaw_acc = 0.0
            straight_path = 0.0
            bout_origin = index + 1
            step_yaw = []
        if not turn_closed:
            continue
        view_ids = _turn_view_ids(poses, bout_origin, index + 1, step_yaw)
        spans.append(_span(poses, origin, index + 1, path_acc, "turn", view_ids))
        origin = index + 1
        yaw_acc = 0.0
        path_acc = 0.0
        straight_path = 0.0
        bout_origin = index + 1
        step_yaw = []
    if origin < len(poses) - 1:
        forward_path = 0.0
        for index in range(origin, len(poses) - 1):
            forward, left, _dyaw, _up = body_frame_delta(
                poses[index]["position"],
                poses[index]["yaw"],
                poses[index + 1]["position"],
                poses[index + 1]["yaw"],
            )
            forward_path += math.hypot(forward, left)
        end = len(poses) - 1
        spans.append(
            _span(poses, origin, end, forward_path, "episode_end", [poses[end]["observation_id"]])
        )
    return spans


def places_with_two_exits(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Places where two non-consecutive departures have different headings.

    The next span of the same walk starts where the previous span ended. That is
    one path turning a corner, not a second exit from a fork.
    """
    parent = list(range(len(spans)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index, left in enumerate(spans):
        for right_index in range(left_index + 1, len(spans)):
            right = spans[right_index]
            if not _same_decision_place(left, right):
                continue
            left_root, right_root = find(left_index), find(right_index)
            parent[right_root] = left_root
    groups: dict[int, list[dict[str, Any]]] = {}
    for index, span in enumerate(spans):
        groups.setdefault(find(index), []).append(span)
    places = []
    for cluster in groups.values():
        exits = _distinct_headings([float(span["start_yaw"]) for span in cluster])
        if exits < 2:
            continue
        places.append(
            {
                "scene": cluster[0]["scene"],
                "span_count": len(cluster),
                "exit_count": exits,
                "episode_count": len({span["episode_uid"] for span in cluster}),
                "includes_only_episode_starts": all(span["start_is_episode_start"] for span in cluster),
            }
        )
    return places


def start_view_ids(spans: list[dict[str, Any]]) -> list[list[str]]:
    """Views of the node a span leaves from.

    The first node of an episode is the single starting frame. Every later node
    is the turn that settled there, so it reuses that turn's views.
    """
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, span in enumerate(spans):
        groups[(str(span.get("split", "")), str(span["episode_uid"]))].append(index)
    result: list[list[str]] = [[] for _ in spans]
    for indexes in groups.values():
        for position, index in enumerate(indexes):
            span = spans[index]
            if position == 0:
                if not span["start_is_episode_start"]:
                    raise ValueError("the first span must be the episode start")
                result[index] = [span["start_observation_id"]]
                continue
            previous = spans[indexes[position - 1]]
            if previous["end_observation_id"] != span["start_observation_id"]:
                raise ValueError("spans are not a continuous walk")
            if previous["end_kind"] != "turn":
                raise ValueError("a later span must start at a settled turn")
            result[index] = list(previous["end_view_ids"])
    return result


def recollection_verdict(*, span_count: int, place_count: int) -> dict[str, Any]:
    """Say whether the walked turns are enough, and whether unseen exits are present."""
    if span_count < 0 or place_count < 0:
        raise ValueError("counts must be non-negative")
    if span_count == 0:
        call = "no_event_spans"
    elif place_count == 0:
        call = "taken_path_only"
    else:
        call = "some_places_have_two_exits"
    return {
        "call": call,
        "span_count": span_count,
        "places_with_two_exits": place_count,
        "doors_and_forks_labeled": False,
        "alternate_exit_observed": place_count > 0,
        "training_started": False,
        "habitat_started": False,
        "collection_started": False,
    }


def _turn_view_ids(
    poses: list[dict[str, Any]], bout_origin: int, end: int, step_yaw: list[float]
) -> list[str]:
    """Frames from the first turning step through the settled pose.

    ``step_yaw`` covers the steps since the last straight-walk reset. A slow
    turn never exceeds the straight-step yaw, so it keeps that whole short
    accumulation. The long corridor before the reset is already outside it.
    """
    if end <= bout_origin or len(step_yaw) != end - bout_origin:
        raise ValueError("turn steps do not match the bout")
    start = bout_origin
    for offset, magnitude in enumerate(step_yaw):
        if magnitude >= STRAIGHT_STEP_RAD:
            start = bout_origin + offset
            break
    view_ids = [poses[index]["observation_id"] for index in range(start, end + 1)]
    if view_ids[-1] != poses[end]["observation_id"]:
        raise ValueError("the settled frame must close the turn")
    return view_ids


def _span(
    poses: list[dict[str, Any]],
    start: int,
    end: int,
    path_m: float,
    end_kind: str,
    end_view_ids: list[str],
) -> dict[str, Any]:
    if not end_view_ids or end_view_ids[-1] != poses[end]["observation_id"]:
        raise ValueError("the node view must include its settled frame")
    yaw_change = wrap_pi(float(poses[end]["yaw"]) - float(poses[start]["yaw"]))
    return {
        "scene": poses[start]["scene"],
        "episode_uid": poses[start]["episode_uid"],
        "start_observation_id": poses[start]["observation_id"],
        "end_observation_id": poses[end]["observation_id"],
        "start_position": [float(part) for part in poses[start]["position"]],
        "end_position": [float(part) for part in poses[end]["position"]],
        "start_yaw": float(poses[start]["yaw"]),
        "end_yaw": float(poses[end]["yaw"]),
        "path_m": float(path_m),
        "net_yaw_rad": float(yaw_change),
        "recorded_control_steps": end - start,
        "end_kind": end_kind,
        "end_view_ids": list(end_view_ids),
        "start_is_episode_start": start == 0,
    }


def _same_decision_place(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["scene"] != right["scene"]:
        return False
    if _horizontal_m(left["start_position"], right["start_position"]) > SAME_PLACE_M:
        return False
    if left["episode_uid"] == right["episode_uid"] and _consecutive(left, right):
        return False
    return True


def _consecutive(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["end_observation_id"] == right["start_observation_id"]
        or right["end_observation_id"] == left["start_observation_id"]
    )


def _distinct_headings(headings: list[float]) -> int:
    kept: list[float] = []
    for heading in headings:
        if any(abs(wrap_pi(heading - previous)) <= DISTINCT_EXIT_RAD for previous in kept):
            continue
        kept.append(heading)
    return len(kept)


def _horizontal_m(left: list[float], right: list[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[2]) - float(right[2]))
