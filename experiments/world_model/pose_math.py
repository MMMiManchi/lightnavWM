"""Body-frame yaw from an explicit xyzw quaternion.

Hamilton product, matching the collector's ``wm_quat_v1`` label. +yaw is a left
turn and yaw 0 faces world -Z. This is the formula unit tests lock. The Habitat
server may also report ``habitat_sim``'s rotation when that import works; a
disagreement is stored, not averaged.
"""

from __future__ import annotations

import math

YAW_METHOD = "wm_quat_v1"


def rotate_vector(xyzw: tuple[float, float, float, float], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z, w = (float(part) for part in xyzw)
    vx, vy, vz = (float(part) for part in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def yaw_from_xyzw(xyzw: tuple[float, float, float, float]) -> float:
    forward_x, _forward_y, forward_z = rotate_vector(xyzw, (0.0, 0.0, -1.0))
    return math.atan2(-forward_x, -forward_z)


def wrap_pi(delta: float) -> float:
    return (float(delta) + math.pi) % (2.0 * math.pi) - math.pi


def body_frame_delta(
    position_before_xyz_m: tuple[float, float, float],
    yaw_before_rad: float,
    position_after_xyz_m: tuple[float, float, float],
    yaw_after_rad: float,
) -> tuple[float, float, float, float]:
    """Horizontal motion of an agent, expressed in the body frame at the earlier pose.

    ``yaw`` uses the same convention as ``yaw_from_xyzw``: 0 faces world -Z, and a
    positive yaw is a left turn. The returned tuple is
    ``(forward_m, left_m, delta_yaw_rad, up_m)``. ``up_m`` is the world-Y change
    and is not part of the planar prediction target.
    """
    x0, y0, z0 = (float(part) for part in position_before_xyz_m)
    x1, y1, z1 = (float(part) for part in position_after_xyz_m)
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    yaw = float(yaw_before_rad)
    forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
    left_x, left_z = -math.cos(yaw), math.sin(yaw)
    forward = dx * forward_x + dz * forward_z
    left = dx * left_x + dz * left_z
    return forward, left, wrap_pi(float(yaw_after_rad) - yaw), dy
