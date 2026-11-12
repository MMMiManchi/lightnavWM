"""Depth warp from the commanded planar motion.

The stored depth is Habitat's normalized optical-axis z: meters are
``normalized * (max_depth - min_depth) + min_depth``. The camera is a pinhole
with square pixels, ``fx = (width / 2) / tan(hfov / 2)``, and the principal
point at ``(width / 2, height / 2)``. Pixel coordinates are integer indices.

Collision-free branch poses move ``v * dt`` along the starting forward axis and
yaw by ``w * dt``, with no lateral arc term. Positive yaw is a left turn and
moves scene content to the right. A point in the starting camera
(X right, Y down, Z forward) goes to

``x' = cos(yaw) * x + sin(yaw) * (z - forward)``
``z' = -sin(yaw) * x + cos(yaw) * (z - forward)``.
"""

from __future__ import annotations

import math

import numpy as np

from experiments.world_model.yaw_shift import (
    HEIGHT,
    HFOV_DEG,
    WIDTH,
    horizontal_shift_px,
    relative_improvement,
)

MIN_DEPTH_M = 0.0
MAX_DEPTH_M = 10.0
MIN_VALID_Z_M = 0.05
IMPROVEMENT_MIN = 0.10
COVERAGE_MIN = 0.50
STRAIGHT_RELATIVE_FLOOR = -0.10


def focal_length(width: int = WIDTH, hfov_deg: float = HFOV_DEG) -> float:
    if width < 2 or not 0.0 < hfov_deg < 180.0:
        raise ValueError("focal length needs a width and a horizontal field of view")
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def intrinsics(
    width: int = WIDTH, height: int = HEIGHT, hfov_deg: float = HFOV_DEG
) -> tuple[float, float, float, float]:
    if height < 2:
        raise ValueError("intrinsics need a height")
    focal = focal_length(width, hfov_deg)
    return focal, focal, width / 2.0, height / 2.0


def depth_to_meters(
    normalized: np.ndarray, *, min_depth: float = MIN_DEPTH_M, max_depth: float = MAX_DEPTH_M
) -> np.ndarray:
    """Invert Habitat's ``(clip(z) - min) / (max - min)`` normalization."""
    depth = np.asarray(normalized, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("depth must be a non-empty HxW array")
    if not np.isfinite(depth).all():
        raise ValueError("depth must be finite")
    if float(depth.min()) < -1e-3 or float(depth.max()) > 1.0 + 1e-3:
        raise ValueError("normalized depth must lie in [0, 1]")
    if not math.isfinite(min_depth) or not math.isfinite(max_depth) or max_depth <= min_depth:
        raise ValueError("depth range must be finite and positive")
    clipped = np.clip(depth, 0.0, 1.0)
    return clipped * (max_depth - min_depth) + min_depth


def command_motion(v_mps: float, w_rad_s: float, dt_s: float) -> tuple[float, float, float]:
    """Starting-frame ``(forward_m, left_m, yaw_rad)`` for one command interval."""
    if not all(math.isfinite(value) for value in (v_mps, w_rad_s, dt_s)) or dt_s <= 0.0:
        raise ValueError("command motion needs finite speeds and a positive interval")
    return v_mps * dt_s, 0.0, w_rad_s * dt_s


def nearest_shift(yaw_rad: float, width: int = WIDTH, hfov_deg: float = HFOV_DEG) -> int:
    """Same yaw/hfov*width shift as the pixel baseline, rounded to a column."""
    shift = horizontal_shift_px(yaw_rad, width, hfov_deg)
    rounded = int(np.rint(shift))
    if not -width < rounded < width:
        raise ValueError(f"yaw shift {rounded} leaves no overlap in width {width}")
    return rounded


def project_point(
    x: float,
    y: float,
    z: float,
    forward: float,
    left: float,
    yaw: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    if left != 0.0:
        raise ValueError("this camera motion has no lateral term")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    z_after_translation = z - forward
    x_new = cosine * x + sine * z_after_translation
    y_new = y
    z_new = -sine * x + cosine * z_after_translation
    if z_new <= 0.0:
        raise ValueError("projected point is not in front of the camera")
    return fx * x_new / z_new + cx, fy * y_new / z_new + cy, z_new


def warp_frame(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    forward: float,
    left: float,
    yaw: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_z: float = MIN_VALID_Z_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-splat ``rgb`` into the camera after the command. Nearer depth wins."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("warp expects uint8 HWC RGB")
    if left != 0.0:
        raise ValueError("this camera motion has no lateral term")
    height, width = rgb.shape[:2]
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.shape != (height, width):
        raise ValueError("depth and RGB must share height and width")
    if not all(math.isfinite(value) and value > 0.0 for value in (fx, fy, min_z)):
        raise ValueError("warp needs a positive focal length and near plane")
    columns, rows = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    valid = np.isfinite(depth) & (depth >= min_z)
    if not bool(valid.any()):
        raise ValueError("depth has no pixel beyond the near plane")
    z = depth[valid]
    x = (columns[valid] - cx) * z / fx
    y = (rows[valid] - cy) * z / fy
    cosine, sine = math.cos(yaw), math.sin(yaw)
    z_shifted = z - forward
    x_new = cosine * x + sine * z_shifted
    z_new = -sine * x + cosine * z_shifted
    front = z_new > min_z
    if not bool(front.any()):
        raise ValueError("command puts every valid pixel behind the camera")
    z_new = z_new[front]
    u = fx * x_new[front] / z_new + cx
    v = fy * y[front] / z_new + cy
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not bool(inside.any()):
        raise ValueError("command projects every valid pixel outside the frame")
    u = u[inside]
    v = v[inside]
    ui = ui[inside]
    vi = vi[inside]
    z_new = z_new[inside]
    colors = rgb[valid][front][inside]
    warped = np.zeros_like(rgb)
    mask = np.zeros((height, width), dtype=bool)
    _paint(vi.astype(np.int64) * width + ui.astype(np.int64), z_new, colors, warped, mask)
    _fill_sampling_gaps(u, v, ui, vi, z_new, colors, warped, mask)
    return warped, mask


def _paint(
    pixel: np.ndarray, z_values: np.ndarray, colors: np.ndarray, warped: np.ndarray, mask: np.ndarray
) -> None:
    """Write the nearest surface into pixels that do not yet have one."""
    if pixel.size == 0:
        return
    order = np.lexsort((z_values, pixel))
    _, first = np.unique(pixel[order], return_index=True)
    chosen = order[first]
    destination = pixel[chosen]
    empty = ~mask.reshape(-1)[destination]
    chosen = chosen[empty]
    if chosen.size == 0:
        return
    destination = pixel[chosen]
    warped.reshape(-1, 3)[destination] = colors[chosen]
    mask.reshape(-1)[destination] = True


def _fill_sampling_gaps(
    u: np.ndarray,
    v: np.ndarray,
    ui: np.ndarray,
    vi: np.ndarray,
    z_values: np.ndarray,
    colors: np.ndarray,
    warped: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Fill an empty neighbor when the projected point lies within one pixel of it.

    Occupied pixels are left alone, so a zero-motion warp stays a copy. The extra
    pixel only covers a nearest-neighbor sampling crack, not a disocclusion.
    """
    height, width = mask.shape
    offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]])
    neighbor_u = ui[:, None] + offsets[None, :, 0]
    neighbor_v = vi[:, None] + offsets[None, :, 1]
    distance = np.maximum(np.abs(u[:, None] - neighbor_u), np.abs(v[:, None] - neighbor_v))
    keep = distance <= 1.0 + 1e-6
    keep &= (neighbor_u >= 0) & (neighbor_u < width) & (neighbor_v >= 0) & (neighbor_v < height)
    if not bool(keep.any()):
        return
    source = np.broadcast_to(np.arange(ui.shape[0])[:, None], neighbor_u.shape)[keep]
    pixel = neighbor_v[keep].astype(np.int64) * width + neighbor_u[keep].astype(np.int64)
    _paint(pixel, z_values[source], colors[source], warped, mask)


def relative_or_tie(baseline: float, candidate: float) -> float:
    """Relative error drop. A zero baseline is a tie only when the candidate is also zero."""
    if (
        not math.isfinite(baseline)
        or not math.isfinite(candidate)
        or baseline < 0.0
        or candidate < 0.0
    ):
        raise ValueError("relative improvement needs finite non-negative errors")
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else -1.0
    return relative_improvement(baseline, candidate)


def fit_camera_ok(
    left_relative: float,
    right_relative: float,
    straight_relatives: list[float],
    turn_coverage: float,
) -> bool:
    """Turns must beat copy by more than 10%. Straight may not get more than 10% worse."""
    values = [left_relative, right_relative, turn_coverage, *straight_relatives]
    if any(not math.isfinite(value) for value in values) or len(straight_relatives) != 2:
        raise ValueError("fit check needs finite turn relatives, two straight relatives, and coverage")
    turns = left_relative > IMPROVEMENT_MIN and right_relative > IMPROVEMENT_MIN
    straight = all(value >= STRAIGHT_RELATIVE_FLOOR for value in straight_relatives)
    return bool(turns and straight and turn_coverage >= COVERAGE_MIN)


def ego_warp_verdict(
    *,
    fit_passed: bool,
    coverage: float | None = None,
    stop_relative: float | None = None,
    all_copy: float | None = None,
    all_yaw: float | None = None,
    all_depth: float | None = None,
    turn_copy: float | None = None,
    turn_yaw: float | None = None,
    turn_depth: float | None = None,
    forward_copy: float | None = None,
    forward_depth: float | None = None,
) -> dict:
    """Development gate. It is not consulted when the fit camera check failed."""
    labels = {
        "camera_model_failed_fit_check": "拟合分支上的相机核对没有通过",
        "warp_covers_too_little": "变换覆盖的像素不足一半",
        "warp_helps_without_motion": "停止命令上变换也降低了误差",
        "depth_warp_predicts": "深度自运动变换预测了开发集上的未来画面",
        "depth_warp_does_not_beat_baselines": "深度变换没有同时超过复制和偏航平移",
    }
    common = {
        "improvement_min": IMPROVEMENT_MIN,
        "coverage_min": COVERAGE_MIN,
        "development_is_blind_test": False,
        "states_are_formal_fit": True,
        "generalization": False,
        "navigation_connected": False,
        "habitat_started": False,
        "formal_cache_written": False,
        "training_started": False,
    }
    if not fit_passed:
        return {
            "call": "camera_model_failed_fit_check",
            "call_label": labels["camera_model_failed_fit_check"],
            "next_step": (
                "48 个 formal fit 分支上，深度变换没有在左右转都比复制低过 10%，"
                "或者直行变差超过 10%，或者转向的有效像素不足一半。"
                "先改相机或深度约定。开发集这次不算。"
            ),
            "fit_passed": False,
            **common,
        }
    numbers = {
        "coverage": coverage,
        "stop_relative": stop_relative,
        "all_copy": all_copy,
        "all_yaw": all_yaw,
        "all_depth": all_depth,
        "turn_copy": turn_copy,
        "turn_yaw": turn_yaw,
        "turn_depth": turn_depth,
        "forward_copy": forward_copy,
        "forward_depth": forward_depth,
    }
    if any(value is None or not math.isfinite(value) for value in numbers.values()):
        raise ValueError("a passing fit check needs finite development scores")
    if min(all_copy, all_yaw, turn_copy, turn_yaw, forward_copy) <= 0.0:
        raise ValueError("baselines must have positive MAE")
    beats = {
        "coverage_ok": coverage >= COVERAGE_MIN,
        "stop_holds": stop_relative <= IMPROVEMENT_MIN,
        "all_vs_copy": relative_improvement(all_copy, all_depth) > IMPROVEMENT_MIN,
        "all_vs_yaw": relative_improvement(all_yaw, all_depth) > IMPROVEMENT_MIN,
        "turn_vs_copy": relative_improvement(turn_copy, turn_depth) > IMPROVEMENT_MIN,
        "turn_vs_yaw": relative_improvement(turn_yaw, turn_depth) > IMPROVEMENT_MIN,
        "forward_vs_copy": relative_improvement(forward_copy, forward_depth) > IMPROVEMENT_MIN,
    }
    if not beats["coverage_ok"]:
        call = "warp_covers_too_little"
        next_step = (
            "有效像素平均不到一半。被丢掉的区域可能正是难预测的地方，"
            "误差下降不能算预测了未来画面。先查深度单位和投影，不训练。"
        )
    elif not beats["stop_holds"]:
        call = "warp_helps_without_motion"
        next_step = (
            "命令已经是停止，深度变换仍然把误差降低超过 10%。"
            "收益不是这条运动命令带来的。先不把几何变换当成预测。"
        )
    elif all(beats.values()):
        call = "depth_warp_predicts"
        next_step = (
            "在已登记的开发集上，深度自运动变换比复制当前帧、也比命令偏航平移，都低过 10%。"
            "它已经是一个能预测未来画面的几何基线。"
            "下一步只检查变换之后剩下的残差能不能预测，通过之前不训练。"
        )
    else:
        call = "depth_warp_does_not_beat_baselines"
        next_step = (
            "开发集上深度变换没有同时超过复制和转向时的偏航平移。"
            "先看直行和转向里没超过的那一项，再改那一项。不回到 SigLIP 网格。"
        )
    return {
        "call": call,
        "call_label": labels[call],
        "next_step": next_step,
        "fit_passed": True,
        "beats": beats,
        **common,
    }
