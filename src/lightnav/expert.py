"""Expert action labels for Go2-W correction data.

The label is one waypoint chunk in the same robot frame the checkpoint emits:
``[forward_m, lateral_m (+left), yaw_rad (+ccw)]``. Stopping is decided from the
remaining geodesic distance, not from the 3 m success radius. Inside that radius
the expert still walks the shortest path until it is within ``stop_within_m``.
If it has already been that close and is moving away, the label is stop.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from lightnav.traj_vocab import RVQBundle

# Successful eval stops land near 1 m (median 0.94 m). The 3 m radius is only
# the benchmark success line, so it is not the stop label.
DEFAULT_STOP_WITHIN_M = 1.0
DEFAULT_LEAVE_MARGIN_M = 0.3
# Matches velocity_control: 2.5 m/s * 0.1 s = 0.25 m per step.
DEFAULT_STEP_M = 0.25
# 300 deg/s * 0.1 s. The body cannot drive backward, so a path behind it is a turn.
MAX_YAW_STEP_RAD = math.radians(30.0)
FACE_PATH_YAW_RAD = math.radians(30.0)
# Following the shortest path reduces the remaining geodesic. This many steps
# with no further drop means the body is no longer getting closer.
DEFAULT_STALL_PATIENCE = 40
DEFAULT_STALL_IMPROVE_M = 0.05


def should_stop(
    distance_m: float,
    min_distance_m: float,
    *,
    stop_within_m: float = DEFAULT_STOP_WITHIN_M,
    leave_margin_m: float = DEFAULT_LEAVE_MARGIN_M,
) -> bool:
    """True when the expert's answer at this pose is the stop token."""
    if not math.isfinite(distance_m):
        return False
    if distance_m <= stop_within_m:
        return True
    return bool(
        math.isfinite(min_distance_m)
        and min_distance_m <= stop_within_m
        and distance_m > min_distance_m + leave_margin_m
    )


def _horizontal(point: np.ndarray) -> np.ndarray:
    out = np.asarray(point, dtype=np.float64).copy()
    out[1] = 0.0
    return out


def sample_polyline(
    points: Sequence[np.ndarray],
    *,
    step_m: float = DEFAULT_STEP_M,
    horizon: int = 10,
) -> List[Tuple[np.ndarray, float]]:
    """Sample ``horizon`` poses ``step_m`` apart along a world polyline.

    Each item is ``(xyz, yaw)``. Yaw is the heading of travel, using the Habitat
    convention ``atan2(-x, -z)`` so identity faces -Z at yaw 0. Past the end of
    the path the last pose repeats, which encodes as standing still.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if step_m <= 0:
        raise ValueError(f"step_m must be positive, got {step_m}")
    pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in points]
    if not pts:
        raise ValueError("polyline is empty")
    if len(pts) == 1:
        pts.append(pts[0].copy())

    seg_len: List[float] = []
    for a, b in zip(pts, pts[1:]):
        seg_len.append(float(np.linalg.norm(_horizontal(b) - _horizontal(a))))

    total = float(sum(seg_len))
    last_yaw = 0.0
    # Heading of the first non-zero segment, used before any motion.
    for a, b in zip(pts, pts[1:]):
        direction = _horizontal(b) - _horizontal(a)
        if np.linalg.norm(direction) > 1e-8:
            last_yaw = math.atan2(-direction[0], -direction[2])
            break

    samples: List[Tuple[np.ndarray, float]] = []
    for k in range(1, horizon + 1):
        dist = min(step_m * k, total)
        acc = 0.0
        pose = pts[-1]
        yaw = last_yaw
        for i, length in enumerate(seg_len):
            if acc + length >= dist - 1e-9 or i == len(seg_len) - 1:
                if length < 1e-8:
                    pose = pts[i + 1]
                else:
                    t = min(1.0, max(0.0, (dist - acc) / length))
                    pose = pts[i] * (1.0 - t) + pts[i + 1] * t
                direction = _horizontal(pts[i + 1]) - _horizontal(pts[i])
                if np.linalg.norm(direction) > 1e-8:
                    yaw = math.atan2(-direction[0], -direction[2])
                    last_yaw = yaw
                else:
                    yaw = last_yaw
                break
            acc += length
        samples.append((pose, yaw))
    return samples


def ego_waypoints(
    agent_pos: np.ndarray,
    agent_forward: np.ndarray,
    agent_left: np.ndarray,
    agent_yaw: float,
    samples: Sequence[Tuple[np.ndarray, float]],
) -> np.ndarray:
    """World poses -> ``(H, 3)`` absolute waypoints in the current robot frame."""
    pos = np.asarray(agent_pos, dtype=np.float64).reshape(3)
    forward = np.asarray(agent_forward, dtype=np.float64).reshape(3)
    left = np.asarray(agent_left, dtype=np.float64).reshape(3)
    out = np.zeros((len(samples), 3), dtype=np.float32)
    for i, (point, yaw) in enumerate(samples):
        delta = np.asarray(point, dtype=np.float64).reshape(3) - pos
        delta[1] = 0.0
        rel_yaw = (yaw - agent_yaw + math.pi) % (2.0 * math.pi) - math.pi
        out[i, 0] = float(np.dot(delta, forward))
        out[i, 1] = float(np.dot(delta, left))
        out[i, 2] = float(rel_yaw)
    return out


def face_path_waypoints(
    waypoints: np.ndarray,
    *,
    max_yaw_step: float = MAX_YAW_STEP_RAD,
    heading_tol: float = FACE_PATH_YAW_RAD,
) -> np.ndarray:
    """Turn in place when the next path point is not in front of the body.

    Linear speed is clipped at zero, so a backward chunk is not an executable
    action: the quantized token collapses to a no-op and the body never moves.
    Replacing that chunk with a yaw toward the path gives the next definite
    action, and the following step can drive forward.
    """
    out = np.asarray(waypoints, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] != 3 or out.shape[0] < 1:
        raise ValueError(f"waypoints must have shape (H, 3), got {out.shape}")
    forward = float(out[0, 0])
    target = float(out[0, 2])
    if forward > 0.05 or abs(target) <= heading_tol:
        return out
    step = math.copysign(min(abs(target), max_yaw_step), target)
    turned = np.zeros_like(out)
    for i in range(out.shape[0]):
        yaw = step * (i + 1)
        if abs(yaw) > abs(target):
            yaw = target
        turned[i, 2] = yaw
    return turned


def trim_stalled_end(
    distances: Sequence[float],
    *,
    patience: int = DEFAULT_STALL_PATIENCE,
    min_improve: float = DEFAULT_STALL_IMPROVE_M,
) -> int:
    """How many leading steps to keep before a no-progress tail.

    The index of the last drop of at least ``min_improve`` metres is kept,
    plus ``patience`` further steps for a turn or a short wobble. Steps after
    that repeat one place and are not training signal.
    """
    if patience < 0:
        raise ValueError(f"patience must be >= 0, got {patience}")
    best = float("inf")
    last = 0
    for index, distance in enumerate(distances):
        if not math.isfinite(distance):
            continue
        if distance < best - min_improve:
            best = float(distance)
            last = index
    if not distances:
        return 0
    return min(len(distances), last + patience + 1)


def trim_episode_rows(
    rows: Sequence[dict],
    *,
    patience: int = DEFAULT_STALL_PATIENCE,
    min_improve: float = DEFAULT_STALL_IMPROVE_M,
) -> List[dict]:
    """Drop steps that no longer get closer to the goal.

    A later real approach starts a new window. Stop labels stay even if they
    fall after the window, because leaving a place already reached is a stop.
    """
    if patience < 0:
        raise ValueError(f"patience must be >= 0, got {patience}")
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    best = float("inf")
    last = 0
    kept: List[dict] = []
    for index, row in enumerate(ordered):
        distance = float(row["distance_m"])
        if math.isfinite(distance) and distance < best - min_improve:
            best = distance
            last = index
        if index <= last + patience or row.get("stop"):
            kept.append(row)
    return kept


def expert_codes(
    bundle: RVQBundle,
    distance_m: float,
    min_distance_m: float,
    waypoints: Optional[np.ndarray],
    *,
    stop_within_m: float = DEFAULT_STOP_WITHIN_M,
    leave_margin_m: float = DEFAULT_LEAVE_MARGIN_M,
) -> List[int]:
    """Stop codes when close enough or leaving; otherwise the quantized chunk."""
    if should_stop(
        distance_m,
        min_distance_m,
        stop_within_m=stop_within_m,
        leave_margin_m=leave_margin_m,
    ):
        return bundle.stop_codes()
    if waypoints is None:
        raise ValueError("a continue label needs a waypoint chunk")
    return bundle.encode_waypoints(waypoints)
