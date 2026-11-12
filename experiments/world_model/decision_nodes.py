"""Mark a node where one view shows two walkable openings.

The recorded depth is used only to see those openings. A corridor with one
opening is not a node. Rounded corners and slopes are not labeled: one depth
frame does not show them. Depth stays out of the navigator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from experiments.world_model.ego_warp import intrinsics
from experiments.world_model.event_spans import SAME_PLACE_M
from experiments.world_model.pose_math import body_frame_delta, wrap_pi
from experiments.world_model.yaw_shift import HEIGHT, WIDTH

HORIZON_ROW_START = 125
HORIZON_ROW_STOP = 136
OPEN_M = 3.0
WALL_M = 2.0
MIN_OPEN_DEG = 12.0
MIN_WALL_DEG = 8.0
SIDE_DEG = 15.0
WIDTH_RATIO = 1.25
REACH_MARGIN_M = 1.5
TAKEN_MATCH_DEG = 25.0
DEPARTURE_M = 0.4


def openings_from_depth(depth_m: np.ndarray) -> list[dict[str, Any]]:
    """Openings in one metric depth frame, ordered from body-left to body-right."""
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.shape != (HEIGHT, WIDTH) or not np.isfinite(depth).all():
        raise ValueError(f"depth must be finite and {HEIGHT}x{WIDTH}")
    if float(depth.min()) < 0.0:
        raise ValueError("depth must be meters, not a normalized map")
    column_depth = np.median(depth[HORIZON_ROW_START:HORIZON_ROW_STOP], axis=0)
    fx, _fy, cx, _cy = intrinsics()
    runs = _open_runs(column_depth)
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if not merged or _is_separator(column_depth, merged[-1][1] + 1, start - 1, fx, cx):
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], end)
    openings = []
    for start, end in merged:
        width = _span_rad(start, end, fx, cx)
        if math.degrees(width) < MIN_OPEN_DEG:
            continue
        open_columns = column_depth[start : end + 1]
        open_columns = open_columns[open_columns >= OPEN_M]
        if open_columns.size == 0:
            continue
        openings.append(_opening(start, end, width, float(np.median(open_columns)), fx, cx))
    openings.sort(key=lambda item: -float(item["bearing_left_rad"]))
    for index, opening in enumerate(openings):
        opening["exit_id"] = index
    return openings


def exit_contrast(openings: list[dict[str, Any]]) -> dict[str, int | None]:
    """Name the wider and longer-looking exit when the view has exactly two."""
    if len(openings) != 2:
        return {"wider_exit_id": None, "longer_looking_exit_id": None}
    left, right = openings
    return {
        "wider_exit_id": _contrast_id(
            left, right, "apparent_width_m", ratio=WIDTH_RATIO, margin=None
        ),
        "longer_looking_exit_id": _contrast_id(
            left, right, "visible_reach_m", ratio=None, margin=REACH_MARGIN_M
        ),
    }


def cluster_decision_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge decision views that stay within the same-place radius."""
    parent = list(range(len(frames)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index, left in enumerate(frames):
        for right_index in range(left_index + 1, len(frames)):
            if _horizontal(left["position"], frames[right_index]["position"]) > SAME_PLACE_M:
                continue
            parent[find(right_index)] = find(left_index)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, frame in enumerate(frames):
        groups[find(index)].append(frame)
    nodes = []
    for members in groups.values():
        members.sort(key=lambda frame: int(frame["step_index"]))
        representative = max(
            members,
            key=lambda frame: (
                len(frame["openings"]),
                sum(float(item["angular_width_rad"]) for item in frame["openings"]),
                -int(frame["step_index"]),
            ),
        )
        nodes.append({"members": members, "representative": representative})
    nodes.sort(key=lambda node: int(node["representative"]["step_index"]))
    return nodes


def departure_bearing(poses: list[dict[str, Any]], origin_index: int) -> float | None:
    """Body-left bearing of the first displacement that leaves the node."""
    if origin_index < 0 or origin_index >= len(poses):
        raise ValueError("departure needs a pose inside the episode")
    origin = poses[origin_index]
    for index in range(origin_index + 1, len(poses)):
        forward, left, _dyaw, _up = body_frame_delta(
            origin["position"], origin["yaw"], poses[index]["position"], poses[index]["yaw"]
        )
        if math.hypot(forward, left) < DEPARTURE_M:
            continue
        return math.atan2(left, forward)
    return None


def matched_exit(openings: list[dict[str, Any]], bearing_left_rad: float | None) -> int | None:
    """The visible exit the departure heads toward, or none if it is outside the view."""
    if bearing_left_rad is None or not openings:
        return None
    if not math.isfinite(bearing_left_rad):
        raise ValueError("departure bearing must be finite")
    best = min(
        openings,
        key=lambda item: abs(wrap_pi(float(item["bearing_left_rad"]) - bearing_left_rad)),
    )
    if abs(wrap_pi(float(best["bearing_left_rad"]) - bearing_left_rad)) > math.radians(TAKEN_MATCH_DEG):
        return None
    return int(best["exit_id"])


def _open_runs(column_depth: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, depth in enumerate(column_depth):
        if depth >= OPEN_M and start is None:
            start = index
        if depth < OPEN_M and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(column_depth) - 1))
    return runs


def _is_separator(column_depth: np.ndarray, start: int, end: int, fx: float, cx: float) -> bool:
    if start > end:
        return False
    if math.degrees(_span_rad(start, end, fx, cx)) < MIN_WALL_DEG:
        return False
    return float(np.median(column_depth[start : end + 1])) <= WALL_M


def _opening(start: int, end: int, width: float, reach: float, fx: float, cx: float) -> dict[str, Any]:
    center = (start + end) / 2.0
    bearing = -math.atan((center - cx) / fx)
    left_edge = math.atan((start - 0.5 - cx) / fx)
    right_edge = math.atan((end + 0.5 - cx) / fx)
    if bearing >= math.radians(SIDE_DEG):
        side = "left"
    elif bearing <= -math.radians(SIDE_DEG):
        side = "right"
    else:
        side = "center"
    return {
        "column_start": start,
        "column_end": end,
        "bearing_left_rad": bearing,
        "side": side,
        "angular_width_rad": width,
        "visible_reach_m": reach,
        "apparent_width_m": reach * (math.tan(right_edge) - math.tan(left_edge)),
    }


def _span_rad(start: int, end: int, fx: float, cx: float) -> float:
    return math.atan((end + 0.5 - cx) / fx) - math.atan((start - 0.5 - cx) / fx)


def _contrast_id(
    left: dict[str, Any],
    right: dict[str, Any],
    field: str,
    *,
    ratio: float | None,
    margin: float | None,
) -> int | None:
    left_value = float(left[field])
    right_value = float(right[field])
    if ratio is not None:
        if left_value >= right_value * ratio:
            return int(left["exit_id"])
        if right_value >= left_value * ratio:
            return int(right["exit_id"])
        return None
    if margin is None:
        raise ValueError("a contrast needs a ratio or a margin")
    if left_value >= right_value + margin:
        return int(left["exit_id"])
    if right_value >= left_value + margin:
        return int(right["exit_id"])
    return None


def _horizontal(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[2]) - float(right[2]))
