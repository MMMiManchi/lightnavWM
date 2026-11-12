"""Recognize Habitat's implicit zero-velocity STOP before dispatch."""

import math
from numbers import Integral

import yaml


def configured_stop_thresholds(config_path):
    """Read physical stop thresholds from the actual simulator configuration.

    Refuse a configuration without explicit thresholds: assuming the wrong
    defaults could let an intercepted STOP terminate the environment.
    """
    config = yaml.safe_load(config_path.read_text())
    try:
        action = config["habitat"]["task"]["actions"]["velocity_control"]
        linear = float(action["min_abs_lin_speed"])
        angular = float(action["min_abs_ang_speed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("sim config must explicitly declare velocity STOP thresholds") from exc
    if not all(math.isfinite(x) and x > 0 for x in (linear, angular)):
        raise ValueError("velocity STOP thresholds must be finite and positive")
    return {"linear_m_per_s": linear, "angular_deg_per_s": angular}


def is_implicit_stop(action, *, linear_range, angular_range, thresholds):
    """Mirror VelocityAction's physical-speed check, including discrete STOP."""
    if isinstance(action, Integral):
        return int(action) == 0
    if not isinstance(action, dict):
        return False
    if action.get("action") in (0, "stop"):
        return True
    if action.get("action") != "velocity_control":
        return False
    args = action["action_args"]
    raw = [float(args["linear_velocity"]), float(args["angular_velocity"])]
    if not all(math.isfinite(v) and -1 <= v <= 1 for v in raw):
        raise ValueError("velocity command must contain finite normalized controls")
    ranges = [linear_range, angular_range]
    if any(len(r) != 2 or not all(math.isfinite(float(v)) for v in r) or r[0] >= r[1] for r in ranges):
        raise ValueError("invalid velocity action range")
    physical = [lo+(v+1)/2*(hi-lo) for v, (lo, hi) in zip(raw, ranges, strict=True)]
    return (abs(physical[0]) < thresholds["linear_m_per_s"] and
            abs(physical[1]) < thresholds["angular_deg_per_s"])
