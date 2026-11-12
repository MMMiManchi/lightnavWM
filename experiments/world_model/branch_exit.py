"""Walk an opening that was seen at a marked node and not taken.

The replay prefix is the recorded velocity commands. The branch first turns
toward that opening. It then walks only where the body can pass. A horizon
gap can look over a chair; depth here only keeps the body out of that
furniture, on the side of the opening that was already chosen. A budget, a
distance cutoff, or a body that stops moving is a truncation, not an arrival.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from experiments.world_model.ego_warp import intrinsics
from experiments.world_model.pose_math import wrap_pi
from experiments.world_model.yaw_shift import HEIGHT, WIDTH

LIN_MPS = (0.0, 2.5)
ANG_DPS = (-300.0, 300.0)
FORWARD_MPS = 0.5
TURN_DPS = 60.0


def normalize_velocity(value: float, bounds: tuple[float, float]) -> float:
    """Map a physical speed into Habitat's [-1, 1]. -1 is the low end."""
    low, high = bounds
    span = high - low
    if span == 0.0:
        raise ValueError("velocity range must have a non-zero span")
    raw = 2.0 * (value - low) / span - 1.0
    if raw < -1.0:
        return -1.0
    if raw > 1.0:
        return 1.0
    return raw


FIT_SPLIT = "formal_fit_20261028_04"
RESTORE_PLACE_M = 0.30
RESTORE_YAW_RAD = 0.30
ALIGN_RAD = math.radians(12.0)
CORRECT_RAD = math.radians(3.0)
CORRECT_DPS = 30.0
BRANCH_STEPS = 80
BRANCH_DISTANCE_M = 2.0
NEXT_DECISION_MIN_M = 0.8
JUNCTION_LEAVE_M = 0.4
EVAL_EPISODE_STEPS = 500
NEW_HOUSE_COUNT = 8
CAMERA_HEIGHT_M = 0.50
BODY_LOW_M = 0.12
BODY_HIGH_M = 0.48
CORRIDOR_HALF_M = 0.15
CLEAR_DRIVE_M = 0.80
PASSABLE_M = 1.20
MIN_GAP_RAD = math.radians(12.0)
DETOUR_RAD = math.radians(80.0)
STUCK_STEPS = 4
STUCK_MOVE_M = 0.02


def unwalked_targets(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Train-fit openings other than the one the original walk left through."""
    chosen: list[dict[str, Any]] = []
    for node in nodes:
        if node.get("split") != FIT_SPLIT:
            continue
        taken = node.get("taken_exit_id")
        if taken is None:
            continue
        for opening in node["openings"]:
            if opening["exit_id"] == taken:
                continue
            chosen.append(
                {
                    "split": node["split"],
                    "scene": node["scene"],
                    "episode_uid": node["episode_uid"],
                    "observation_id": node["observation_id"],
                    "step_index": int(node["step_index"]),
                    "position": [float(value) for value in node["position"]],
                    "yaw": float(node["yaw"]),
                    "taken_exit_id": int(taken),
                    "exit_id": int(opening["exit_id"]),
                    "bearing_left_rad": float(opening["bearing_left_rad"]),
                    "side": opening["side"],
                }
            )
    chosen.sort(
        key=lambda item: (
            item["step_index"],
            item["scene"],
            item["episode_uid"],
            item["exit_id"],
        )
    )
    return chosen


def batch_passes(
    targets: list[dict[str, Any]], done: list[tuple[str, int, int]]
) -> list[list[dict[str, Any]]]:
    """Later exits of one episode go in later passes. A pass never repeats an episode."""
    finished = {(uid, int(step), int(exit_id)) for uid, step, exit_id in done}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        key = (target["episode_uid"], int(target["step_index"]), int(target["exit_id"]))
        if key in finished:
            continue
        grouped.setdefault(target["episode_uid"], []).append(target)
    for items in grouped.values():
        items.sort(key=lambda item: (int(item["step_index"]), int(item["exit_id"])))
    if not grouped:
        return []
    order = sorted(grouped)
    width = max(len(items) for items in grouped.values())
    return [
        [grouped[uid][index] for uid in order if index < len(grouped[uid])]
        for index in range(width)
    ]


def replay_actions(
    transitions: list[dict[str, Any]], episode_uid: str, step_index: int
) -> list[dict[str, Any]]:
    """Executed habitat actions that should return the robot to this node."""
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    rows = [row for row in transitions if row.get("episode_uid") == episode_uid]
    rows.sort(key=lambda row: int(row["t"]))
    chosen = [row for row in rows if int(row["t"]) < step_index]
    if [int(row["t"]) for row in chosen] != list(range(step_index)):
        raise RuntimeError("replay prefix is not a contiguous executed walk")
    actions = []
    for row in chosen:
        ack = row["action"]["execution_ack"]
        if ack.get("executed") is not True:
            raise RuntimeError(f"transition t={row['t']} was not executed")
        actions.append(row["action"]["final_habitat_action"])
    return actions


def remaining_bearing(original_bearing: float, yaw_now: float, yaw_at_node: float) -> float:
    """Opening bearing in the current body frame. Positive is still body-left."""
    turned = wrap_pi(yaw_now - yaw_at_node)
    return wrap_pi(original_bearing - turned)


def branch_velocity(bearing_left_rad: float) -> dict[str, Any]:
    """Turn toward the opening, then walk forward at half a meter per second."""
    if bearing_left_rad > ALIGN_RAD:
        linear = normalize_velocity(0.0, LIN_MPS)
        angular = normalize_velocity(TURN_DPS, ANG_DPS)
    elif bearing_left_rad < -ALIGN_RAD:
        linear = normalize_velocity(0.0, LIN_MPS)
        angular = normalize_velocity(-TURN_DPS, ANG_DPS)
    else:
        linear = normalize_velocity(FORWARD_MPS, LIN_MPS)
        if bearing_left_rad > CORRECT_RAD:
            angular = normalize_velocity(CORRECT_DPS, ANG_DPS)
        elif bearing_left_rad < -CORRECT_RAD:
            angular = normalize_velocity(-CORRECT_DPS, ANG_DPS)
        else:
            angular = 0.0
    return {
        "action": "velocity_control",
        "action_args": {"linear_velocity": linear, "angular_velocity": angular},
    }


def _depth_meters(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape != (HEIGHT, WIDTH) or not np.isfinite(depth).all():
        raise ValueError(f"depth must be finite and {HEIGHT}x{WIDTH}")
    if float(depth.min()) < 0.0:
        raise ValueError("depth must be meters")
    return depth


def _body_points(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-frame ``(x right, z forward, body mask)`` for one metric depth frame."""
    fx, fy, cx, cy = intrinsics()
    rows = np.arange(HEIGHT, dtype=np.float64)[:, None]
    cols = np.arange(WIDTH, dtype=np.float64)[None, :]
    x = (cols - cx) * depth / fx
    y = (rows - cy) * depth / fy
    height = CAMERA_HEIGHT_M - y
    body = (height >= BODY_LOW_M) & (height <= BODY_HIGH_M) & (depth >= 0.05)
    return x, depth, body


def forward_clearance(depth_m: np.ndarray) -> float:
    """Nearest body-height surface inside the robot's forward corridor, in meters."""
    depth = _depth_meters(depth_m)
    x, z, body = _body_points(depth)
    corridor = body & (np.abs(x) <= CORRIDOR_HALF_M)
    if not np.any(corridor):
        return 10.0
    return float(min(10.0, z[corridor].min()))


def _bearing_at(column: float) -> float:
    _fx, _fy, cx, _cy = intrinsics()
    return -math.atan((column - cx) / _fx)


def passable_gaps(depth_m: np.ndarray) -> list[dict[str, float]]:
    """Body-height gaps wide enough to enter, ordered from left to right."""
    depth = _depth_meters(depth_m)
    _x, z, body = _body_points(depth)
    nearest = np.min(np.where(body, z, np.inf), axis=0)
    gaps: list[dict[str, float]] = []
    start: int | None = None
    for column, distance in enumerate(nearest.tolist()):
        open_column = distance >= PASSABLE_M
        if open_column and start is None:
            start = column
        elif not open_column and start is not None:
            gaps.append(_gap(start, column - 1))
            start = None
    if start is not None:
        gaps.append(_gap(start, WIDTH - 1))
    return [gap for gap in gaps if gap["width_rad"] >= MIN_GAP_RAD]


def _gap(start: int, end: int) -> dict[str, float]:
    bearing = _bearing_at(0.5 * (start + end))
    return {
        "bearing_left_rad": bearing,
        "width_rad": abs(_bearing_at(start) - _bearing_at(end)),
    }


def _same_side(bearing: float, side: str) -> bool:
    if abs(bearing) <= ALIGN_RAD:
        return True
    if side == "right":
        return bearing < 0.0 and abs(bearing) <= DETOUR_RAD
    if side == "left":
        return bearing > 0.0 and abs(bearing) <= DETOUR_RAD
    if side == "center":
        return abs(bearing) <= DETOUR_RAD
    raise ValueError(f"unknown exit side {side}")


def follow_opening(side: str, depth_m: np.ndarray) -> tuple[dict[str, Any], str] | None:
    """Face a body-height gap on this exit's side, then walk. None means blocked."""
    gaps = [gap for gap in passable_gaps(depth_m) if _same_side(gap["bearing_left_rad"], side)]
    gap_bearing = None
    if gaps:
        gap_bearing = min(gaps, key=lambda item: abs(item["bearing_left_rad"]))["bearing_left_rad"]
    if forward_clearance(depth_m) >= CLEAR_DRIVE_M and (
        gap_bearing is None or abs(gap_bearing) <= ALIGN_RAD
    ):
        return branch_velocity(0.0 if gap_bearing is None else gap_bearing), "drive"
    if gap_bearing is None:
        return None
    return branch_velocity(gap_bearing), "turn"


def motion_stalled(forward_moves_m: list[float]) -> bool:
    """True when commanded forward motion has stopped for several steps."""
    if len(forward_moves_m) < STUCK_STEPS:
        return False
    return all(move < STUCK_MOVE_M for move in forward_moves_m[-STUCK_STEPS:])


def restore_error(
    position: list[float], yaw: float, saved_position: list[float], saved_yaw: float
) -> tuple[float, float]:
    place_m = math.hypot(position[0] - saved_position[0], position[2] - saved_position[2])
    yaw_rad = abs(wrap_pi(yaw - saved_yaw))
    return place_m, yaw_rad


def restore_matches(
    position: list[float], yaw: float, saved_position: list[float], saved_yaw: float
) -> bool:
    place_m, yaw_rad = restore_error(position, yaw, saved_position, saved_yaw)
    return place_m <= RESTORE_PLACE_M and yaw_rad <= RESTORE_YAW_RAD


def branch_stop(
    steps: int, travel_m: float, opening_count: int | None, *, stalled: bool = False
) -> str | None:
    """Why the branch walk ends. A truncation is not an arrival."""
    if stalled:
        return "truncated_blocked"
    if steps >= BRANCH_STEPS:
        return "truncated_budget"
    if travel_m >= BRANCH_DISTANCE_M:
        return "truncated_distance"
    if (
        opening_count is not None
        and travel_m >= NEXT_DECISION_MIN_M
        and opening_count >= 2
    ):
        return "next_decision"
    return None


def is_arrival(reason: str | None) -> bool:
    return reason == "next_decision"


def new_fit_houses(
    rows: list[dict[str, Any]],
    used_scenes: set[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """One train-fit episode from each house that has not been walked yet."""
    from experiments.world_model.latent_data import scene_partition

    if limit < 1:
        raise ValueError("house limit must be positive")
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    ordered = sorted(rows, key=lambda row: (str(row["scene"]), str(row["episode_id"])))
    for row in ordered:
        scene = str(row["scene"])
        if scene in used_scenes or scene in seen:
            continue
        if scene_partition(scene) != "train_fit" or row.get("partition") != "train_fit":
            continue
        chosen.append(row)
        seen.add(scene)
        if len(chosen) == limit:
            break
    return chosen


def exits_to_walk(openings: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
    """The widest openings at a junction, at most ``limit``."""
    if limit < 1:
        raise ValueError("exit limit must be positive")
    ranked = sorted(
        openings,
        key=lambda item: (-float(item["angular_width_rad"]), int(item["exit_id"])),
    )
    return ranked[:limit]


def search_stop(
    steps: int, travel_m: float, opening_count: int | None, *, stalled: bool = False
) -> str | None:
    """Look for the first junction until the evaluation episode budget."""
    at_junction = opening_count is not None and opening_count >= 2 and (
        steps == 0 or travel_m >= JUNCTION_LEAVE_M
    )
    if at_junction:
        return "junction"
    if stalled or steps >= EVAL_EPISODE_STEPS:
        return "no_junction"
    return None


def arrival_stop(
    steps: int, travel_m: float, opening_count: int | None, *, stalled: bool = False
) -> str | None:
    """Stop at the next junction, or at the same 500-step budget evaluation uses."""
    if stalled:
        return "truncated_blocked"
    if steps >= EVAL_EPISODE_STEPS:
        return "truncated_budget"
    if (
        opening_count is not None
        and travel_m >= NEXT_DECISION_MIN_M
        and opening_count >= 2
    ):
        return "next_decision"
    return None
