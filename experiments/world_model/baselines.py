"""Prediction baselines for the latent world model.

Baseline A copies the current latent. It is defined only when the two latent
vectors come from the same frozen encoder and have the same shape.

Baseline B integrates the command that was actually submitted to Habitat.
It reads ``final_habitat_action`` and the recorded control ranges. It does not
read the requested action, the decoded waypoint, or a privileged pose. Lateral
velocity is absent from that command, so the predicted left displacement is 0.
A pair of speeds below the Habitat stop thresholds predicts no motion, matching
``habitat.tasks.nav.nav.VelocityControlAction.step``, which returns before
integration in that case. The integration itself is first-order
``forward = v * dt``, ``yaw = w * dt``. Habitat afterwards calls
``integrate_transform`` and a navmesh step; those residuals are not modeled.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from experiments.world_model.pose_math import wrap_pi


def denormalize_raw(raw: float, low: float, high: float) -> float:
    """Inverse of ``velocity._normalize_to_raw`` and of Habitat's [-1, 1] scaling."""
    if not math.isfinite(raw) or not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("velocity values must be finite")
    if high == low:
        raise ValueError("velocity range span is zero")
    return float(low + (float(raw) + 1.0) / 2.0 * (float(high) - float(low)))


def copy_latent(latent_t: np.ndarray) -> np.ndarray:
    array = np.asarray(latent_t)
    if array.size == 0:
        raise ValueError("latent is empty")
    if not np.isfinite(array).all():
        raise ValueError("latent is not finite")
    return np.array(array, copy=True)


def latent_errors(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"latent shapes differ: {pred.shape} vs {true.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("latent errors require finite vectors")
    mse = float(np.mean((pred - true) ** 2))
    pred_norm = float(np.linalg.norm(pred))
    true_norm = float(np.linalg.norm(true))
    if pred_norm == 0.0 or true_norm == 0.0:
        cosine_distance = None
    else:
        cosine = float(np.dot(pred, true) / (pred_norm * true_norm))
        cosine_distance = float(1.0 - cosine)
    return {"mse": mse, "cosine_distance": cosine_distance}


def executed_twist(transition: dict[str, Any]) -> dict[str, Any]:
    """Physical twist of the command Habitat received, not the client's request."""
    action = transition["action"]
    ack = action["execution_ack"]
    final = action.get("final_habitat_action") or ack["final_habitat_action"]
    units = action.get("control_units") or ack.get("control_units")
    if units is None:
        raise ValueError("execution ack has no control_units")
    requested = action.get("client_request")
    if type(final) is int:
        if final != 0:
            raise ValueError(f"unsupported discrete action {final}")
        return _zero_twist(units, stop_reason="discrete_stop", requested=requested, final=final)
    if not isinstance(final, dict) or final.get("action") != "velocity_control":
        raise ValueError(f"unsupported final action {final!r}")
    args = final["action_args"]
    lin_raw = float(args["linear_velocity"])
    ang_raw = float(args["angular_velocity"])
    v_mps = denormalize_raw(lin_raw, *units["lin_vel_range_mps"])
    w_dps = denormalize_raw(ang_raw, *units["ang_vel_range_dps"])
    dt = action["execution_ack"].get("dt_command_s")
    if dt is None:
        raise ValueError("velocity command has no dt_command_s")
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError(f"dt_command_s must be positive, got {dt}")
    unit_dt = units.get("control_dt_s")
    if unit_dt is not None and abs(float(unit_dt) - float(dt)) > 1e-6:
        raise ValueError("dt_command_s disagrees with control_units.control_dt_s")
    min_lin = units.get("min_abs_lin_speed_mps")
    min_ang = units.get("min_abs_ang_speed_dps")
    if min_lin is None or min_ang is None:
        stop = v_mps == 0.0 and w_dps == 0.0
    else:
        stop = abs(v_mps) < float(min_lin) and abs(w_dps) < float(min_ang)
    w_rad_s = 0.0 if stop else math.radians(w_dps)
    return {
        "source": "final_habitat_action",
        "linear_raw": lin_raw,
        "angular_raw": ang_raw,
        "v_mps": 0.0 if stop else v_mps,
        "w_rad_s": w_rad_s,
        "dt_command_s": float(dt),
        "dt_sim_s": action.get("dt_sim_s"),
        "stop_threshold": stop,
        "stop_reason": "below_min_abs_speed" if stop else None,
        "request_differs": requested != final,
    }


def _zero_twist(units: dict[str, Any], stop_reason: str, requested: Any, final: Any) -> dict[str, Any]:
    dt = units.get("control_dt_s")
    return {
        "source": "final_habitat_action",
        "linear_raw": None,
        "angular_raw": None,
        "v_mps": 0.0,
        "w_rad_s": 0.0,
        "dt_command_s": None if dt is None else float(dt),
        "dt_sim_s": None,
        "stop_threshold": True,
        "stop_reason": stop_reason,
        "request_differs": requested != final,
    }


def kinematic_delta(transition: dict[str, Any]) -> dict[str, float | bool | None]:
    """Baseline B. Body-frame displacement from the submitted command."""
    twist = executed_twist(transition)
    dt = twist["dt_command_s"]
    if twist["stop_threshold"] or dt is None:
        forward = 0.0
        yaw = 0.0
    else:
        forward = float(twist["v_mps"]) * float(dt)
        yaw = float(twist["w_rad_s"]) * float(dt)
    return {
        "forward_m": forward,
        "left_m": 0.0,
        "delta_yaw_rad": wrap_pi(yaw),
        "stop_threshold": bool(twist["stop_threshold"]),
    }


def pose_errors(prediction: dict[str, float], measured: dict[str, float]) -> dict[str, float]:
    forward = float(measured["forward_m"]) - float(prediction["forward_m"])
    left = float(measured["left_m"]) - float(prediction["left_m"])
    yaw = wrap_pi(float(measured["delta_yaw_rad"]) - float(prediction["delta_yaw_rad"]))
    return {
        "translation_m": math.hypot(forward, left),
        "forward_error_m": forward,
        "left_error_m": left,
        "yaw_error_rad": yaw,
        "yaw_error_deg": math.degrees(yaw),
    }


def action_kind(twist: dict[str, Any], yaw_step_rad: float = 0.05) -> str:
    if twist["stop_threshold"]:
        return "stop_threshold"
    if abs(float(twist["w_rad_s"]) * float(twist["dt_command_s"])) >= yaw_step_rad:
        return "turning"
    return "mostly_forward"
