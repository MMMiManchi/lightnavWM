"""Global horizontal shift from the commanded yaw.

Positive yaw is a left turn. Turning left moves scene content to the right, so a
positive shift reads ``output[:, shift:] = image[:, :-shift]``. The shift is
``yaw / hfov * width``, the constant-pixel approximation of a 120 degree camera.
Pixels that enter from outside the frame are excluded, and the copy baseline
uses that same mask.
"""

from __future__ import annotations

import math

import numpy as np

WIDTH = 480
HEIGHT = 270
HFOV_DEG = 120.0
DT_S = 0.1
TURN_RATE_DPS = 60.0
IMPROVEMENT_MIN = 0.10
LEFT_ACTIONS = ("creep_left", "slow_left", "nominal_left")
RIGHT_ACTIONS = ("creep_right", "slow_right", "nominal_right")
STRAIGHT_ACTIONS = ("slow_straight", "nominal_straight")


def command_yaw_rad(w_dps: float, dt_s: float = DT_S) -> float:
    if not math.isfinite(w_dps) or not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("yaw needs a finite rate and a positive command interval")
    return math.radians(w_dps) * dt_s


def horizontal_shift_px(
    yaw_rad: float, width: int = WIDTH, hfov_deg: float = HFOV_DEG
) -> float:
    if not math.isfinite(yaw_rad) or width < 2 or not 0.0 < hfov_deg < 180.0:
        raise ValueError("shift needs a finite yaw, a width, and a horizontal field of view")
    return yaw_rad / math.radians(hfov_deg) * width


def shift_content_right(image: np.ndarray, shift_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Move image content right by ``shift_px`` columns. Negative moves it left."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("shift expects uint8 HWC RGB")
    height, width = image.shape[:2]
    if not -width < shift_px < width:
        raise ValueError(f"shift {shift_px} leaves no overlap in width {width}")
    warped = np.zeros_like(image)
    mask = np.zeros((height, width), dtype=bool)
    if shift_px >= 0:
        warped[:, shift_px:] = image[:, : width - shift_px]
        mask[:, shift_px:] = True
    else:
        warped[:, : width + shift_px] = image[:, -shift_px:]
        mask[:, : width + shift_px] = True
    return warped, mask


def masked_mae(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    if reference.shape != candidate.shape or mask.shape != reference.shape[:2]:
        raise ValueError("MAE inputs must share height, width, and channels")
    if mask.dtype != bool or not bool(mask.any()):
        raise ValueError("MAE mask must contain at least one pixel")
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return float(difference[mask].mean())


def relative_improvement(copy_mae: float, warp_mae: float) -> float:
    if not math.isfinite(copy_mae) or copy_mae <= 0.0 or not math.isfinite(warp_mae) or warp_mae < 0.0:
        raise ValueError("improvement needs a positive copy MAE and a finite warp MAE")
    return (copy_mae - warp_mae) / copy_mae


def integer_shift(yaw_rad: float, width: int = WIDTH, hfov_deg: float = HFOV_DEG) -> int:
    shift = horizontal_shift_px(yaw_rad, width, hfov_deg)
    rounded = int(round(shift))
    if abs(shift - rounded) > 1e-6:
        raise ValueError(f"yaw shift {shift} is not an integer pixel")
    return rounded


def yaw_shift_verdict(
    *,
    left_relative: float,
    right_relative: float,
    straight_command_relatives: list[float],
    straight_imposed_relatives: list[float],
) -> dict:
    """Turns must improve by more than 10%. The same shift must not help straight motion."""
    relatives = [left_relative, right_relative, *straight_command_relatives, *straight_imposed_relatives]
    if any(not math.isfinite(value) for value in relatives):
        raise ValueError("yaw-shift relatives must be finite")
    if len(straight_command_relatives) != len(STRAIGHT_ACTIONS):
        raise ValueError("straight command relatives must cover slow and nominal straight")
    if len(straight_imposed_relatives) != 2 * len(STRAIGHT_ACTIONS):
        raise ValueError("each straight action needs both imposed turn shifts")
    imposed_helps = any(value > IMPROVEMENT_MIN for value in straight_imposed_relatives)
    turns_improve = left_relative > IMPROVEMENT_MIN and right_relative > IMPROVEMENT_MIN
    straight_holds = all(value <= IMPROVEMENT_MIN for value in straight_command_relatives)
    if imposed_helps:
        call = "shift_helps_without_yaw"
        next_step = (
            "把转向用的像素平移加到直行上，误差也会降低。"
            "这次不能把收益记成命令偏航。先不改预测目标。"
        )
    elif turns_improve and straight_holds:
        call = "yaw_shift_explains_turns"
        next_step = (
            "0.1 秒转向的画面差主要由命令偏航对应的水平平移解释。"
            "不必再预测整幅 SigLIP 网格。若还要做世界模型，只预测平移之后剩下的残差，"
            "并且那个残差得先过可预测性检查。"
        )
    else:
        call = "yaw_shift_does_not_explain"
        next_step = (
            "按命令偏航平移后，左右转没有都比复制前一帧低过 10%。"
            "成片的变化不是这个全局平移。下一轮用现成光流做只读测量，不训练。"
        )
    labels = {
        "shift_helps_without_yaw": "平移在没有偏航时也降低误差",
        "yaw_shift_explains_turns": "命令偏航的水平平移解释了转向画面差",
        "yaw_shift_does_not_explain": "全局偏航平移没有解释转向画面差",
    }
    return {
        "call": call,
        "call_label": labels[call],
        "next_step": next_step,
        "improvement_min": IMPROVEMENT_MIN,
        "turns_improve": turns_improve,
        "straight_command_holds": straight_holds,
        "imposed_shift_helps_straight": imposed_helps,
        "development_is_blind_test": False,
        "states_are_formal_fit": True,
        "generalization": False,
        "navigation_connected": False,
        "habitat_started": False,
        "formal_cache_written": False,
        "training_started": False,
    }
