"""Inputs a future online policy is allowed to see.

Privileged files are not opened. Future observations are not included.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

FORBIDDEN_ONLINE_KEYS = {
    "distance_to_goal",
    "goal_distance",
    "goal_position",
    "geodesic_distance_m",
    "reference_path",
    "reference_path_ref",
    "success",
    "spl",
    "ndtw",
    "soft_spl",
    "oracle_success",
    "world_position_xyz_m",
    "world_rotation_xyzw",
    "derived_yaw_rad",
    "wm_privileged",
    "privileged",
    "pose",
    "collision",
    "metrics",
}

ONLINE_INFO_KEYS = ("habitat_time_step", "lin_vel_range", "ang_vel_range")


def online_info(info: dict[str, Any] | None) -> dict[str, Any]:
    source = info or {}
    redacted = {}
    for key in ONLINE_INFO_KEYS:
        if key in source:
            redacted[key] = source[key]
    leaked = FORBIDDEN_ONLINE_KEYS.intersection(redacted)
    if leaked:
        raise RuntimeError(f"online info leaked privileged keys: {sorted(leaked)}")
    return redacted


def online_observation(obs: dict[str, Any]) -> dict[str, Any]:
    if "rgb" not in obs:
        raise KeyError("online observation requires rgb")
    instruction = obs.get("instruction")
    return {"rgb": obs["rgb"], "instruction": instruction}


def decision_input(
    *,
    instruction_text: str,
    history_observation_ids: list[str],
    info: dict[str, Any] | None,
    obs_after_id: str | None = None,
) -> dict[str, Any]:
    if obs_after_id is not None and obs_after_id in history_observation_ids:
        raise RuntimeError("decision history includes the post-action observation")
    payload = {
        "instruction_text": instruction_text,
        "observation_ids": list(history_observation_ids),
        "info_keys": sorted(online_info(info)),
    }
    leaked = FORBIDDEN_ONLINE_KEYS.intersection(payload["info_keys"])
    if leaked:
        raise RuntimeError(f"decision input leaked privileged keys: {sorted(leaked)}")
    return payload


def assert_loader_ignores_privileged(run_dir: Path, reader) -> dict[str, Any]:
    """Call ``reader`` while failing if the privileged file is opened."""
    forbidden = (Path(run_dir) / "privileged_labels.jsonl").resolve()
    real_open = open

    def guarded(file, *args, **kwargs):
        target = file if not hasattr(file, "__fspath__") else Path(file)
        try:
            resolved = Path(target).resolve()
        except TypeError:
            resolved = None
        if resolved == forbidden:
            raise RuntimeError(f"online loader opened {forbidden}")
        return real_open(file, *args, **kwargs)

    import builtins

    previous = builtins.open
    builtins.open = guarded
    try:
        return reader()
    finally:
        builtins.open = previous
