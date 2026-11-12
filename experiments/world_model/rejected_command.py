"""Copy the frame when the previous command was not executed.

A command was rejected when it asked for more than 0.05 m or 0.05 rad and the
measured body motion stayed under 0.02 m and 0.05 rad. The current step may use
that previous measurement. It may not use the current step's future pose.
"""

from __future__ import annotations

import math

from experiments.world_model.yaw_shift import relative_improvement

COMMAND_FORWARD_M = 0.05
COMMAND_YAW_RAD = 0.05
MEASURED_FORWARD_M = 0.02
MEASURED_YAW_RAD = 0.05
IMPROVEMENT_MIN = 0.10
OTHER_HARM_MAX = 0.01
STUCK_VS_COPY_MAX = 0.10


def previous_command_rejected(
    command_forward_m: float,
    command_yaw_rad: float,
    measured_forward_m: float,
    measured_yaw_rad: float,
) -> bool:
    values = (command_forward_m, command_yaw_rad, measured_forward_m, measured_yaw_rad)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("rejected-command inputs must be finite")
    asked = abs(command_forward_m) > COMMAND_FORWARD_M or abs(command_yaw_rad) > COMMAND_YAW_RAD
    stayed = abs(measured_forward_m) < MEASURED_FORWARD_M and abs(measured_yaw_rad) < MEASURED_YAW_RAD
    return bool(asked and stayed)


def rejection_verdict(
    *,
    stuck_warp: float,
    stuck_rule: float,
    stuck_copy: float,
    other_warp: float,
    other_rule: float,
    all_copy: float,
    all_yaw: float,
    all_rule: float,
) -> dict:
    """The stuck scene must become no worse than copy. The other scenes must stay put."""
    numbers = (
        stuck_warp,
        stuck_rule,
        stuck_copy,
        other_warp,
        other_rule,
        all_copy,
        all_yaw,
        all_rule,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in numbers):
        raise ValueError("rejection scores must be finite and non-negative")
    if min(stuck_warp, stuck_copy, other_warp, all_copy, all_yaw) <= 0.0:
        raise ValueError("baselines must have positive MAE")
    stuck_vs_warp = relative_improvement(stuck_warp, stuck_rule)
    stuck_vs_copy = relative_improvement(stuck_copy, stuck_rule)
    other_harm = (other_rule - other_warp) / other_warp
    beats = {
        "stuck_vs_warp": stuck_vs_warp > IMPROVEMENT_MIN,
        "stuck_not_worse_than_copy": stuck_vs_copy >= -STUCK_VS_COPY_MAX,
        "other_scenes_hold": other_harm <= OTHER_HARM_MAX,
        "all_vs_copy": relative_improvement(all_copy, all_rule) > IMPROVEMENT_MIN,
        "all_vs_yaw": relative_improvement(all_yaw, all_rule) > IMPROVEMENT_MIN,
    }
    common = {
        "improvement_min": IMPROVEMENT_MIN,
        "other_harm_max": OTHER_HARM_MAX,
        "beats": beats,
        "development_is_blind_test": False,
        "uses_current_future_pose": False,
        "generalization": False,
        "navigation_connected": False,
        "habitat_started": False,
        "formal_cache_written": False,
        "training_started": False,
    }
    if all(beats.values()):
        call = "rejected_command_explains_the_stuck_scene"
        label = "上一步命令被拒绝，解释了卡住的那个场景"
        next_step = (
            "上一步命令没有被执行时，这一步预测画面不变。"
            "卡住的场景不再比复制更差，其余场景的深度变换也保持住。"
            "下一步只检查这条修正之后剩下的残差能不能预测，通过之前不训练。"
        )
    elif not beats["other_scenes_hold"]:
        call = "rule_hurts_executed_scenes"
        label = "这条规则伤害了命令确实被执行的场景"
        next_step = (
            "用上一步是否被拒绝来决定复制，会让其余场景变差超过 1%。"
            "不能把这条规则当成修正。先看它误触发的那些步。"
        )
    else:
        call = "rejected_command_does_not_explain"
        label = "上一步被拒绝没有解释卡住的场景"
        next_step = (
            "即使上一步命令被拒绝时改为复制当前帧，卡住的场景仍没有回到复制的水平，"
            "或者全部八个场景不再同时超过复制和偏航平移。先不要训练。"
        )
    return {"call": call, "call_label": label, "next_step": next_step, **common}
