"""Train-only transition collector.

The evaluation runner is not used. It cannot attach an action id, and a budget
STOP would otherwise sit next to the previous call's ``get_info``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.world_model import (
    COLLECTOR_VERSION,
    DATASET_NAME,
    END_RULE_VERSION,
    SCHEMA_VERSION,
    WAYPOINT_TIME_REASON,
)
from experiments.world_model.collection_gate import (
    CollectionGateError,
    assert_collectable,
    assert_dev_collectable,
    load_scene_inventory,
)
from experiments.world_model.campaign import register_campaign_run, validate_campaign_target
from experiments.world_model.controlled_trial import validate_controlled_trial
from experiments.world_model.group_manifest import (
    assign_groups,
    check_allowlist_against_train,
    load_split_rows,
    refuse_legacy_path,
)
from experiments.world_model.jsonutil import dumps, loads, sha256_file
from experiments.world_model.online_input import decision_input, online_info, online_observation
from experiments.world_model.precollection_protocol import (
    assert_collection_not_blocked,
    assert_dev_collection_authorized,
)
from experiments.world_model.recorder import RunRecorder
from experiments.world_model.validate import validate_run
from lightnav.velocity import first_waypoint_to_velocity_cmd

_ACT_TOKEN = re.compile(r"<act_l(\d+)_(\d+)>")
_END_PRIORITY = (
    "transport_error",
    "server_early_stop",
    "runner_budget",
    "decode_fallback",
    "model_stop",
    "habitat_stop",
    "habitat_limit",
    "manual_abort",
    "unknown",
)
FORMAL_MAX_STEPS = 500
FORMAL_WALL_CLOCK_BUDGET_S = 8 * 60 * 60
CONTROLLED_TRIAL_WALL_CLOCK_BUDGET_S = 1200


class WallClockBudgetExceeded(TimeoutError):
    pass


def resolve_wall_clock_budget(controlled_trial: bool, requested_s: float | None) -> float:
    cap = CONTROLLED_TRIAL_WALL_CLOCK_BUDGET_S if controlled_trial else FORMAL_WALL_CLOCK_BUDGET_S
    budget = cap if requested_s is None else float(requested_s)
    if not 0 < budget <= cap:
        raise ValueError(f"wall-clock budget must be in (0, {cap}] seconds")
    return budget


def _check_deadline(deadline_monotonic: float | None) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise WallClockBudgetExceeded("collector wall-clock budget exhausted")


def _tokens(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    found = [match.group(0) for match in _ACT_TOKEN.finditer(raw)]
    return found or None


def null_prediction() -> dict[str, Any]:
    return {
        "inference_called": False,
        "raw_text": None,
        "rvq_tokens": None,
        "rvq_codes": None,
        "decoded_waypoints": None,
        "selected_index": None,
        "selected_waypoint": None,
        "lateral_m_not_in_command": None,
        "waypoint_times": None,
        "waypoint_times_reason": WAYPOINT_TIME_REASON,
        "model_stop": None,
        "decode_status": "not_called",
    }


def prediction_record(info: dict[str, Any], rvq: Any) -> dict[str, Any]:
    raw = info.get("raw_text")
    traj = info.get("predicted_traj")
    if traj is None or info.get("decode_status") == "decode_failed":
        return {
            "inference_called": True,
            "raw_text": raw,
            "rvq_tokens": _tokens(raw),
            "rvq_codes": None,
            "decoded_waypoints": None,
            "selected_index": None,
            "selected_waypoint": None,
            "lateral_m_not_in_command": None,
            "waypoint_times": None,
            "waypoint_times_reason": WAYPOINT_TIME_REASON,
            "model_stop": False,
            "decode_status": "decode_failed",
        }
    waypoints = np.asarray(traj, dtype=np.float64)
    index = int(info["action_waypoint_index"])
    selected = np.asarray(info["action_waypoint"], dtype=np.float64)
    if selected.shape != (3,) or waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError("waypoint record has an unexpected shape")
    if index < 0 or index >= waypoints.shape[0] or not np.allclose(selected, waypoints[index]):
        raise ValueError("selected waypoint is not the indexed trajectory row")
    codes = info.get("rvq_codes")
    if "model_stop" in info:
        model_stop = bool(info["model_stop"])
    elif rvq is not None and codes is not None:
        model_stop = bool(rvq.is_stop(list(codes)))
    else:
        model_stop = False
    return {
        "inference_called": True,
        "raw_text": raw,
        "rvq_tokens": _tokens(raw),
        "rvq_codes": None if codes is None else [int(code) for code in codes],
        "decoded_waypoints": waypoints.tolist(),
        "selected_index": index,
        "selected_waypoint": selected.tolist(),
        "lateral_m_not_in_command": float(selected[1]),
        "waypoint_times": None,
        "waypoint_times_reason": WAYPOINT_TIME_REASON,
        "model_stop": model_stop,
        "decode_status": "decode_ok",
    }


def _primary(sources: list[str]) -> str:
    for name in _END_PRIORITY:
        if name in sources:
            return name
    return "unknown"


def _instruction_text(obs: dict[str, Any]) -> str:
    instruction = obs.get("instruction")
    if isinstance(instruction, dict):
        return str(instruction.get("text", ""))
    if instruction is None:
        return ""
    return str(instruction)


class ScriptedPolicy:
    """Deterministic stand-in. ``stop_action`` leaves the previous ``get_info`` in place."""

    def __init__(self, steps: list[dict[str, Any]]):
        self.steps = steps
        self.calls = 0
        self.dt = 0.1
        self.lin_vel_range = (0.0, 2.5)
        self.ang_vel_range = (-300.0, 300.0)
        self._info: dict[str, Any] = {"raw_text": "STALE_SHOULD_NOT_LEAK"}
        self.last_info_keys: list[str] = []

    def reset(self, obs: dict[str, Any]) -> None:
        self.calls = 0
        self._info = {"raw_text": "STALE_SHOULD_NOT_LEAK"}

    def stop_action(self) -> dict[str, Any]:
        return first_waypoint_to_velocity_cmd(
            np.zeros(3, dtype=np.float32), self.dt, self.lin_vel_range, self.ang_vel_range
        )

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = self.steps[self.calls]
        self.calls += 1
        self.last_info_keys = sorted((info or {}))
        kind = spec["kind"]
        if kind == "move":
            waypoint = np.asarray(spec.get("waypoint", [0.2, 0.05, 0.0]), dtype=np.float32)
            traj = np.zeros((10, 3), dtype=np.float32)
            traj[0] = waypoint
            self._info = {
                "raw_text": "<act_l0_1><act_l1_0><act_l2_0>",
                "predicted_traj": traj,
                "rvq_codes": [1, 0, 0],
                "action_waypoint_index": 0,
                "action_waypoint": waypoint,
                "model_stop": False,
                "decode_status": "decode_ok",
            }
            return first_waypoint_to_velocity_cmd(waypoint, self.dt, self.lin_vel_range, self.ang_vel_range)
        if kind == "model_stop":
            traj = np.zeros((10, 3), dtype=np.float32)
            self._info = {
                "raw_text": "<act_l0_0><act_l1_0><act_l2_0>",
                "predicted_traj": traj,
                "rvq_codes": [0, 0, 0],
                "action_waypoint_index": 0,
                "action_waypoint": traj[0],
                "model_stop": True,
                "decode_status": "decode_ok",
            }
            return self.stop_action()
        if kind == "decode_fail":
            self._info = {"raw_text": "not-an-action", "decode_status": "decode_failed", "predicted_traj": None}
            return self.stop_action()
        raise ValueError(kind)

    def get_info(self) -> dict[str, Any]:
        return dict(self._info)


def _client_seed(requested: int | None) -> dict[str, Any]:
    if requested is None:
        return {
            "python_random": {"applied": False, "value": None, "evidence": "no seed requested"},
            "numpy": {"applied": False, "value": None, "evidence": "no seed requested"},
            "inference": {
                "applied": False,
                "value": None,
                "reason": "greedy generation; no sampling seed interface is set",
            },
        }
    random.seed(int(requested))
    np.random.seed(int(requested))
    return {
        "python_random": {
            "value": int(requested),
            "applied": True,
            "evidence": "random.seed called in the collector process",
        },
        "numpy": {
            "value": int(requested),
            "applied": True,
            "evidence": "np.random.seed called in the collector process",
        },
        "inference": {
            "applied": False,
            "value": None,
            "reason": "greedy generation; no sampling seed interface is set",
        },
    }


def _git_fingerprint(root: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str:
        completed = subprocess.run(args, cwd=root, check=False, capture_output=True, text=True)
        return completed.stdout

    commit = run(["git", "rev-parse", "HEAD"]).strip()
    diff = run(["git", "diff", "HEAD"])
    status = run(["git", "status", "--short"])
    return {
        "commit": commit or None,
        "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "dirty": bool(status.strip()),
    }


_REAL_ENV_PREFIXES = ("habitat.", "habitat_sim", "lightnav_habitat")
_WRAPPER_PREFIX = "experiments.world_model.server_adapter"


def _is_habitat_env(env: Any, seen: set[int] | None = None) -> bool:
    """True when stepping this object would enter Habitat.

    A ``WorldModelVLNCEEnv`` whose ``_habitat_env`` is a test double stays
    false, so the offline session tests can still run. An uninitialized
    wrapper, or any object whose class lives in Habitat, is refused.
    """
    if env is None:
        return False
    seen = set() if seen is None else seen
    identity = id(env)
    if identity in seen:
        return True
    seen.add(identity)
    module = getattr(type(env), "__module__", "") or ""
    if module.startswith(_REAL_ENV_PREFIXES):
        return True
    if module.startswith(_WRAPPER_PREFIX):
        if not hasattr(env, "_habitat_env") or getattr(env, "_habitat_env", None) is None:
            return True
        return _is_habitat_env(env._habitat_env, seen)
    return False


def run_session(
    env: Any,
    policy: Any,
    recorder: RunRecorder,
    plan: list[dict[str, Any]],
    *,
    max_steps: int,
    requested_seed: int | None,
    collection_policy: str,
    controlled_trial: bool = False,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    if controlled_trial:
        validate_controlled_trial(
            repo_root=Path(__file__).resolve().parents[2],
            run_dir=recorder.run_dir,
            max_steps=max_steps,
            episodes=plan,
            require_absent=False,
        )
    if _is_habitat_env(env) and not controlled_trial:
        assert_collection_not_blocked()
    step_cap = FORMAL_MAX_STEPS
    if max_steps < 1 or max_steps > step_cap:
        raise ValueError(f"collector max_steps must be in 1..{step_cap} for this entry point")
    reset_keys: list[list[str]] = []
    server_seed = None
    resolved_split = None
    dataset_sha = None
    depth_contract = None
    history: dict[str, list[str]] = {}

    def save_observation(row: dict[str, Any], obs: dict[str, Any], info: dict[str, Any], step_index: int, terminal: bool) -> tuple[str, str | None]:
        sensor = obs.get("wm_sensor") or info.get("wm_sensor") or {}
        observation_id = f"obs-{row['episode_uid']}-{step_index}"
        rgb_record = None
        depth_record = None
        if sensor.get("rgb_missing"):
            rgb_record = None
        else:
            rgb_record = recorder.save_array(observation_id, "rgb", np.asarray(obs["rgb"]))
        if sensor.get("depth_missing"):
            depth_record = None
        elif "depth" in obs:
            depth_record = recorder.save_array(observation_id, "depth", np.asarray(obs["depth"]))
        privileged = info.get("wm_privileged") or {}
        pose = privileged.get("pose")
        label_id = None
        if privileged:
            label_id = f"pose-{observation_id}"
            recorder.add(
                "privileged_labels",
                {
                    "label_id": label_id,
                    "episode_uid": row["episode_uid"],
                    "observation_id": observation_id,
                    "pose": pose,
                    "pose_reason": privileged.get("pose_reason"),
                    "goal_position": info.get("goal_position"),
                    "geodesic_distance_m": float(info["distance_to_goal"]) if info.get("distance_to_goal") is not None and np.isfinite(info["distance_to_goal"]) else None,
                    "geodesic_distance_reason": "non_finite_habitat_distance_to_goal" if info.get("distance_to_goal") is not None and not np.isfinite(info["distance_to_goal"]) else None,
                    "source_geodesic_m": float(row["source_geodesic_m"]) if row.get("source_geodesic_m") is not None and np.isfinite(row["source_geodesic_m"]) else None,
                    "reference_path": info.get("reference_path"),
                    "collision": privileged.get("collision"),
                    "metrics": {
                        key: (float(info[key]) if np.isfinite(info[key]) else None)
                        for key in ("success", "spl", "ndtw", "oracle_success") if key in info
                    },
                    "metrics_non_finite": [key for key in ("success", "spl", "ndtw", "oracle_success") if key in info and info[key] is not None and not np.isfinite(info[key])],
                },
            )
        recorder.add(
            "observations",
            {
                "observation_id": observation_id,
                "episode_uid": row["episode_uid"],
                "step_index": step_index,
                "rgb": rgb_record,
                "rgb_missing": bool(sensor.get("rgb_missing")),
                "rgb_missing_reason": sensor.get("rgb_missing_reason"),
                "rgb_color_space": None if rgb_record is None else "RGB",
                "rgb_format": None if rgb_record is None else "numpy_npy",
                "depth": depth_record,
                "depth_missing": bool(sensor.get("depth_missing")) or depth_record is None,
                "depth_missing_reason": sensor.get("depth_missing_reason")
                or (None if depth_record is not None else "depth array was not in the returned observation"),
                "sim_time_s": info.get("wm_sim_time_s"),
                "sim_time_reason": info.get("wm_sim_time_reason", "clock_unavailable"),
                "arrival_monotonic_ns": time.monotonic_ns(),
                "available_at_decision": step_index,
                "terminal_observation": bool(terminal),
                "pose_label_ref": label_id,
            },
        )
        return observation_id, label_id

    for row in plan:
        _check_deadline(deadline_monotonic)
        obs, info = env.reset(seed=requested_seed)
        resolved = info.get("wm_resolved") or {}
        resolved_split = resolved.get("split", resolved_split)
        dataset_sha = resolved.get("dataset_sha256", dataset_sha)
        depth_contract = resolved.get("depth_sensor", depth_contract)
        server_seed = resolved.get("seed", server_seed)
        if resolved_split != "train":
            raise RuntimeError(f"server split is {resolved_split!r}")
        served = [tuple(item) for item in resolved.get("served_keys") or []]
        expected = [(item["scene"], item["episode_id"]) for item in plan]
        if served != expected:
            raise RuntimeError(f"served keys {served} != requested {expected}")
        reset_ack = info.get("wm_reset") or {}
        key = tuple(reset_ack.get("key") or [])
        if key != (row["scene"], row["episode_id"]):
            raise RuntimeError(f"reset {key} != plan {(row['scene'], row['episode_id'])}")
        reset_keys.append([row["scene"], row["episode_id"]])
        if reset_ack.get("applied_in_reset") is not False:
            raise RuntimeError("reset reported that it applied a seed; BaseHabitatEnv.reset does not")
        seen_text = _instruction_text(obs)
        if seen_text != row["instruction_text"]:
            raise RuntimeError("server instruction text does not match the train source")
        sensor = obs.get("wm_sensor") or {}
        obs_id, pose_id = save_observation(row, obs, info, 0, terminal=False)
        history[row["episode_uid"]] = [obs_id]
        completion = "completed"
        last_transition = None
        if sensor.get("rgb_missing"):
            completion = "infrastructure_error"
            recorder.add(
                "episodes",
                _episode_record(row, reset_ack, completion, last_transition, "rgb_missing"),
            )
            continue
        policy.reset(online_observation(obs))
        for step_index in range(max_steps):
            _check_deadline(deadline_monotonic)
            budget = step_index == max_steps - 1
            decision = decision_input(
                instruction_text=row["instruction_text"],
                history_observation_ids=list(history[row["episode_uid"]]),
                info=info,
            )
            started = time.monotonic()
            if budget:
                action = policy.stop_action()
                prediction = null_prediction()
                infer_s = None
                client_sources = ["runner_budget"]
            else:
                action = policy.act(online_observation(obs), online_info(info))
                _check_deadline(deadline_monotonic)
                infer_s = time.monotonic() - started
                rvq = getattr(getattr(policy, "agent", None), "rvq", None)
                prediction = prediction_record(policy.get_info(), rvq)
                client_sources = []
            if not prediction["inference_called"] and (
                prediction["raw_text"] is not None or prediction["model_stop"] is not None
            ):
                raise RuntimeError("budget stop copied a prediction")
            action_id = f"act-{uuid.uuid4()}"
            envelope = dict(action)
            envelope["wm_action_id"] = action_id
            client_request = {key: value for key, value in action.items()}
            try:
                _check_deadline(deadline_monotonic)
                rpc_started = time.monotonic()
                obs_after, reward, terminated, truncated, info_after = env.step(envelope)
                transport_s = time.monotonic() - rpc_started
            except WallClockBudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - the failure is stored
                getter = getattr(env, "execution_record", None)
                ack = getter(action_id) if getter else None
                if ack is None:
                    ack = {
                        "action_id": action_id,
                        "status": "unknown",
                        "executed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                _write_transition(
                    recorder,
                    row=row,
                    action_id=action_id,
                    step_index=step_index,
                    obs_before_id=obs_id,
                    pose_before_id=pose_id,
                    obs_after_id=None,
                    pose_after_id=None,
                    prediction=prediction,
                    client_request=client_request,
                    ack=ack,
                    infer_s=infer_s,
                    transport_s=None,
                    reward=None,
                    terminated=None,
                    truncated=None,
                    client_sources=client_sources + ["transport_error"],
                    decision=decision,
                    quality={"status": "quarantined", "errors": [f"step_exception: {exc}"]},
                )
                completion = "infrastructure_error"
                last_transition = action_id
                break
            ack = info_after.get("wm_execution")
            if not isinstance(ack, dict):
                ack = {"action_id": None, "status": "unknown", "executed": False}
            sources = list(client_sources)
            if ack.get("server_override"):
                sources.append("server_early_stop")
            if prediction.get("decode_status") == "decode_failed":
                sources.append("decode_fallback")
            if prediction.get("model_stop") is True:
                sources.append("model_stop")
            if ack.get("habitat_stop_called"):
                sources.append("habitat_stop")
            if truncated:
                sources.append("habitat_limit")
            if not sources:
                sources.append("unknown")
            after_sensor = obs_after.get("wm_sensor") or info_after.get("wm_sensor") or {}
            ended = bool(terminated or truncated or budget)
            after_id, after_pose = save_observation(row, obs_after, info_after, step_index + 1, terminal=ended)
            quality_errors = []
            if ack.get("status") != "executed" or ack.get("action_id") != action_id:
                quality_errors.append("ack_mismatch")
            if after_sensor.get("rgb_missing"):
                quality_errors.append("rgb_missing")
            if ack.get("sim_time_before_s") is None or ack.get("sim_time_after_s") is None or ack.get("dt_sim_s") is None:
                quality_errors.append("sim_time_unavailable")
            pose_after = (info_after.get("wm_privileged") or {}).get("pose") or {}
            if after_pose is None or pose_id is None or pose_after.get("world_position_xyz_m") is None:
                quality_errors.append("pose_missing")
            disagreement = pose_after.get("yaw_disagreement_rad")
            if disagreement is not None and abs(float(disagreement)) > 1e-4:
                quality_errors.append("yaw_convention_disagreement")
            blocking = [item for item in quality_errors if item not in ("pose_missing", "yaw_convention_disagreement")]
            if not quality_errors:
                status = "valid"
            elif not blocking:
                status = "partial"
            else:
                status = "quarantined"
            _write_transition(
                recorder,
                row=row,
                action_id=action_id,
                step_index=step_index,
                obs_before_id=obs_id,
                pose_before_id=pose_id,
                obs_after_id=after_id,
                pose_after_id=after_pose,
                prediction=prediction,
                client_request=client_request,
                ack=ack,
                infer_s=infer_s,
                transport_s=transport_s,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                client_sources=sources,
                decision=decision,
                quality={"status": status, "errors": quality_errors},
            )
            history[row["episode_uid"]].append(after_id)
            obs, info = obs_after, info_after
            obs_id, pose_id = after_id, after_pose
            last_transition = action_id
            if terminated or truncated:
                completion = "budget_truncated" if budget and not prediction.get("model_stop") else "completed"
                break
        else:
            completion = "budget_truncated"
        recorder.add("episodes", _episode_record(row, reset_ack, completion, last_transition, None))

    return {
        "reset_keys": reset_keys,
        "server_seed": server_seed,
        "resolved_split": resolved_split,
        "dataset_sha256": dataset_sha,
        "depth_sensor": depth_contract,
        "collection_policy": collection_policy,
    }


def _episode_record(row: dict[str, Any], reset_ack: dict[str, Any], completion: str, last_action: str | None, note: str | None) -> dict[str, Any]:
    return {
        "episode_uid": row["episode_uid"],
        "episode_id": row["episode_id"],
        "scene": row["scene"],
        "scene_id_raw": row["scene_id_raw"],
        "source_split": "train",
        "route_id_native": row.get("route_id_native"),
        "route_group_id": row.get("route_group_id"),
        "route_geometry_hash": row.get("route_geometry_hash"),
        "instruction_id_native": row.get("instruction_id_native"),
        "instruction_id_derived": row.get("instruction_id_derived"),
        "instruction_text": row["instruction_text"],
        "partition": row.get("partition"),
        "reset": {
            "key": reset_ack.get("key"),
            "applied_in_reset": reset_ack.get("applied_in_reset"),
            "reason": reset_ack.get("reason"),
        },
        "completion": {"state": completion, "last_action_id": last_action, "note": note},
    }


def _write_transition(
    recorder: RunRecorder,
    *,
    row: dict[str, Any],
    action_id: str,
    step_index: int,
    obs_before_id: str,
    pose_before_id: str | None,
    obs_after_id: str | None,
    pose_after_id: str | None,
    prediction: dict[str, Any],
    client_request: dict[str, Any],
    ack: dict[str, Any],
    infer_s: float | None,
    transport_s: float | None,
    reward: Any,
    terminated: Any,
    truncated: Any,
    client_sources: list[str],
    decision: dict[str, Any],
    quality: dict[str, Any],
) -> None:
    sources = list(dict.fromkeys(client_sources))
    recorder.add(
        "transitions",
        {
            "transition_id": f"tr-{row['episode_uid']}-{step_index}",
            "episode_uid": row["episode_uid"],
            "t": step_index,
            "obs_before_id": obs_before_id,
            "obs_after_id": obs_after_id,
            "pose_before_id": pose_before_id,
            "pose_after_id": pose_after_id,
            "prediction": prediction,
            "action": {
                "action_id": action_id,
                "client_request": client_request,
                "server_received": ack.get("client_request"),
                "server_override": ack.get("server_override"),
                "override_reason": ack.get("override_reason"),
                "final_habitat_action": ack.get("final_habitat_action"),
                "execution_ack": ack,
                "control_units": ack.get("control_units"),
                "dt_command_s": ack.get("dt_command_s"),
                "dt_command_reason": ack.get("dt_command_reason"),
                "dt_sim_s": ack.get("dt_sim_s"),
                "sim_time_before_s": ack.get("sim_time_before_s"),
                "sim_time_after_s": ack.get("sim_time_after_s"),
                "dt_sim_reason": ack.get("dt_sim_reason"),
                "dt_wall_s": ack.get("dt_wall_s"),
                "transport_wall_s": transport_s,
                "infer_s": infer_s,
            },
            "outcome": {
                "terminated": terminated,
                "truncated": truncated,
                "reward": None if reward is None else float(reward),
                "end_sources": sources,
                "primary_end_source": _primary(sources),
                "primary_rule": END_RULE_VERSION,
                "termination_reason": None,
            },
            "online_input": decision,
            "quality": quality,
        },
    )


class CollectClient:
    """ZMQ client that sends the reset seed and the action id. It does not use RemoteEnvClient."""

    def __init__(self, address: str, timeout_ms: int = 600000):
        import zmq

        self._zmq = zmq
        self.timeout_ms = int(timeout_ms)
        self.deadline_monotonic: float | None = None
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(address)

    def _rpc(self, command: str, data: Any) -> dict[str, Any]:
        timeout_ms = self.timeout_ms
        if self.deadline_monotonic is not None:
            remaining_ms = int((self.deadline_monotonic - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                raise WallClockBudgetExceeded("collector wall-clock budget exhausted")
            timeout_ms = min(timeout_ms, remaining_ms)
            self.socket.setsockopt(self._zmq.RCVTIMEO, timeout_ms)
            self.socket.setsockopt(self._zmq.SNDTIMEO, timeout_ms)
        self.socket.send_pyobj({"command": command, "data": data}, protocol=4)
        message = self.socket.recv_pyobj()
        if message.get("status") == "error":
            raise RuntimeError(message.get("message"))
        return message

    def reset(self, seed: int | None = None) -> tuple[Any, dict]:
        response = self._rpc("reset", {"seed": seed, "options": None})
        return response["obs"], response.get("info") or {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        response = self._rpc("step", action)
        return (
            response["obs"],
            response["reward"],
            response["terminated"],
            response["truncated"],
            response.get("info") or {},
        )

    def close(self) -> None:
        try:
            self._rpc("close", None)
        except Exception:
            pass
        self.socket.close(linger=0)
        self.context.term()


def _prepare_plan(
    allowlist_path: Path,
    dataset_root: Path,
    inventory_path: Path,
    *,
    partition: str = "train_fit",
) -> tuple[list[dict[str, Any]], dict[str, Any], str, dict[str, Any]]:
    refuse_legacy_path(allowlist_path)
    allowlist = loads(allowlist_path.read_text(encoding="utf-8"))
    _path, digest, rows = load_split_rows(dataset_root, "train")
    grouped = assign_groups(rows)
    plan = check_allowlist_against_train(allowlist, grouped, dataset_sha256=digest)
    inventory = load_scene_inventory(inventory_path)
    if partition == "train_fit":
        plan = assert_collectable(inventory, plan, dataset_sha256=digest)
    elif partition == "train_dev":
        plan = assert_dev_collectable(inventory, plan, dataset_sha256=digest)
    else:
        raise CollectionGateError("partition", f"unsupported collection partition {partition!r}")
    return plan, allowlist, digest, inventory


def parsed_habitat_split(config_path: Path) -> str:
    """Read ``habitat.dataset.split`` from the YAML structure, not from a text search."""
    if not config_path.is_file():
        raise CollectionGateError("config_missing", str(config_path))
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        split = payload["habitat"]["dataset"]["split"]
    except (TypeError, KeyError, AttributeError, yaml.YAMLError) as exc:
        raise CollectionGateError("config_split", f"cannot read habitat.dataset.split: {exc}") from exc
    if not isinstance(split, str):
        raise CollectionGateError("config_split", f"habitat.dataset.split is {split!r}")
    return split


def _require_train_config(config_path: Path) -> str:
    split = parsed_habitat_split(config_path)
    if split != "train":
        raise CollectionGateError("config_split", f"habitat.dataset.split is {split}")
    return sha256_file(config_path)


def collection_dry_run(
    *,
    allowlist_path: Path,
    dataset_root: Path,
    inventory_path: Path,
    run_dir: Path,
    config_path: Path,
    max_steps: int,
    partition: str = "train_fit",
) -> dict[str, Any]:
    """Check the collection plan without creating a run or starting Habitat."""
    if not 1 <= max_steps <= FORMAL_MAX_STEPS:
        raise CollectionGateError("max_steps", f"--max-steps must be between 1 and {FORMAL_MAX_STEPS}")
    refuse_legacy_path(run_dir)
    if run_dir.exists():
        raise CollectionGateError("run_dir_exists", f"refusing to use existing path {run_dir}")
    if not run_dir.parent.is_dir():
        raise CollectionGateError("run_dir_parent", f"parent directory does not exist: {run_dir.parent}")
    config_sha = _require_train_config(config_path)
    plan, allowlist, digest, inventory = _prepare_plan(
        allowlist_path, dataset_root, inventory_path, partition=partition
    )
    keys = [(row["scene"], row["episode_id"]) for row in plan]
    if len(keys) != len(set(keys)):
        raise CollectionGateError("allowlist", "duplicate episode in the checked plan")
    scenes = sorted({scene for scene, _episode_id in keys})
    return {
        "dry_run": True,
        "formal_collection_started": False,
        "formal_feature_cache": False,
        "training": False,
        "feature_layer_selected": None,
        "accepted_for_formal_training": False,
        "episodes": len(keys),
        "scenes": len(scenes),
        "scene_ids": scenes,
        "keys": [list(key) for key in keys],
        "route_group_ids": [row["route_group_id"] for row in plan],
        "max_steps": max_steps,
        "step_upper_bound": len(keys) * max_steps,
        "output_path": str(run_dir),
        "output_exists_after_check": run_dir.exists(),
        "dataset_sha256": digest,
        "allowlist_sha256": allowlist.get("allowlist_sha256"),
        "scene_inventory_path": str(inventory_path),
        "scene_inventory_sha256": sha256_file(inventory_path),
        "scene_split_version": inventory.get("scene_split_version"),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "config_split": "train",
        "collection_partition": partition,
        "training_use": "not_approved",
    }


def write_failure_record(recorder: RunRecorder, failure: dict[str, Any]) -> None:
    """Leave ``failure.json`` beside the existing run. Never overwrite or delete it."""
    message = dumps(failure)
    try:
        if not recorder.closed:
            recorder.close()
    except Exception as exc:
        sys.stderr.write(f"collector close failed: {type(exc).__name__}: {exc}\n{message}\n")
    path = recorder.run_dir / "failure.json"
    if path.exists():
        sys.stderr.write(f"failure.json already exists and was left unchanged\n{message}\n")
        return
    try:
        recorder.write_json("failure.json", failure)
    except Exception as exc:
        sys.stderr.write(f"failed to write failure.json: {type(exc).__name__}: {exc}\n{message}\n")


def abort_collection(
    recorder: RunRecorder,
    exc: BaseException,
    *,
    wall_clock: dict[str, Any] | None = None,
) -> int:
    """Record the exception that stopped collection and return a non-zero status."""
    failure = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    if wall_clock is not None:
        failure["wall_clock"] = wall_clock
    write_failure_record(recorder, failure)
    sys.stderr.write(dumps(failure) + "\n")
    return 1


def commit_validation(recorder: RunRecorder, report: dict[str, Any]) -> int:
    """Persist the validator report. A non-zero hard-error count becomes ``failure.json``."""
    if not (recorder.run_dir / "validation_report.json").is_file():
        recorder.write_json("validation_report.json", report)
    hard = report.get("hard_error_count")
    if hard != 0:
        write_failure_record(
            recorder,
            {
                "status": "validation_failed",
                "hard_error_count": hard,
                "errors": report.get("errors") or [],
            },
        )
        sys.stderr.write(dumps({"status": "validation_failed", "hard_error_count": hard}) + "\n")
        return 1
    return 0


def require_habitat_server(proc: Any) -> None:
    code = proc.poll()
    if code is not None:
        raise RuntimeError(f"habitat server exited {code}; see habitat_server.log")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect train-only LightNav transitions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allowlist", required=True)
    parser.add_argument("--scene-inventory", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--campaign-manifest", default=None)
    parser.add_argument("--partition", choices=("train_fit", "train_dev"), default="train_fit")
    parser.add_argument("--max-steps", type=int, default=FORMAL_MAX_STEPS)
    parser.add_argument("--wall-clock-budget-s", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--server", default=None)
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--habitat-python", default="/home/szgx/miniconda3/envs/habitat/bin/python")
    parser.add_argument("--port", type=int, default=5693)
    parser.add_argument("--dataset-root", default="data/datasets/R2R_VLNCE_v1-3_preprocessed")
    parser.add_argument("--policy", choices=("lightnav", "scripted"), default="lightnav")
    parser.add_argument("--model-path", default="checkpoints/LightNav-0")
    parser.add_argument("--backend", default="vllm_local")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--controlled-trial", action="store_true")
    args = parser.parse_args(argv)
    try:
        _require_train_config(Path(args.config))
    except (CollectionGateError, OSError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.dry_run:
        try:
            report = collection_dry_run(
                allowlist_path=Path(args.allowlist),
                dataset_root=Path(args.dataset_root),
                inventory_path=Path(args.scene_inventory),
                run_dir=Path(args.run_dir),
                config_path=Path(args.config),
                max_steps=args.max_steps,
                partition=args.partition,
            )
        except (CollectionGateError, RuntimeError, OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        print(dumps(report))
        return 0
    if args.max_steps < 1 or args.max_steps > FORMAL_MAX_STEPS:
        raise SystemExit("--max-steps exceeds this entry point's limit")
    try:
        wall_budget_s = resolve_wall_clock_budget(args.controlled_trial, args.wall_clock_budget_s)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_dir = Path(args.run_dir)
    refuse_legacy_path(run_dir)
    try:
        plan, allowlist, digest, inventory = _prepare_plan(
            Path(args.allowlist),
            Path(args.dataset_root),
            Path(args.scene_inventory),
            partition=args.partition,
        )
        if args.controlled_trial:
            validate_controlled_trial(
                repo_root=Path(".").resolve(),
                run_dir=run_dir,
                max_steps=args.max_steps,
                episodes=plan,
                require_absent=True,
            )
        else:
            if args.partition == "train_fit":
                assert_collection_not_blocked()
            else:
                assert_dev_collection_authorized()
            if args.partition == "train_fit" and (not args.campaign_id or not args.campaign_manifest):
                raise CollectionGateError(
                    "campaign_required",
                    "formal collection requires --campaign-id and --campaign-manifest for incremental accounting",
                )
            if args.partition == "train_fit":
                validate_campaign_target(Path(args.campaign_manifest), args.campaign_id, run_dir)
    except (CollectionGateError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if len({row["scene"] for row in plan}) < 1:
        raise SystemExit("allowlist is empty")
    if not args.launch_server and not args.server:
        raise SystemExit("pass --server or --launch-server")
    if args.controlled_trial:
        run_dir.parent.mkdir(parents=True, exist_ok=True)
    recorder = RunRecorder(run_dir)
    run_started = time.monotonic()
    run_started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_deadline = run_started + wall_budget_s
    previous_alarm = signal.getsignal(signal.SIGALRM)

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise WallClockBudgetExceeded("collector wall-clock budget exhausted")

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, wall_budget_s)
    server_proc = None
    env = None
    try:
        if args.launch_server:
            ready = run_dir / "habitat.ready"
            log_handle = open(run_dir / "habitat_server.log", "w", encoding="utf-8")
            server_env = os.environ.copy()
            habitat_cuda = os.environ.get("HABITAT_CUDA_VISIBLE_DEVICES")
            if habitat_cuda:
                server_env["CUDA_VISIBLE_DEVICES"] = habitat_cuda
                server_env["HABITAT_SIM_GPU_ID"] = "0"
            server_proc = subprocess.Popen(
                [
                    args.habitat_python,
                    str(Path("experiments/world_model/serve_collect.py").resolve()),
                    "--config",
                    args.config,
                    "--allowlist",
                    str(Path(args.allowlist).resolve()),
                    "--scene-inventory",
                    str(Path(args.scene_inventory).resolve()),
                    "--port",
                    str(args.port),
                    "--max-steps",
                    str(args.max_steps),
                    "--ready-file",
                    str(ready.resolve()),
                    "--dataset-root",
                    str(Path(args.dataset_root).resolve()),
                    "--partition",
                    args.partition,
                ]
                + (["--controlled-trial", "--trial-output-dir", str(run_dir.resolve())] if args.controlled_trial else [])
                + ([] if args.seed is None else ["--seed", str(args.seed)]),
                cwd=str(Path(".").resolve()),
                env=server_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            deadline = min(run_deadline, time.monotonic() + args.ready_timeout_s)
            while not ready.exists():
                require_habitat_server(server_proc)
                if time.monotonic() > deadline:
                    raise RuntimeError("habitat server did not become ready")
                time.sleep(1.0)
            address = f"tcp://127.0.0.1:{args.port}"
        else:
            address = args.server
        env = CollectClient(address)
        env.deadline_monotonic = run_deadline
        if args.policy == "scripted":
            policy = ScriptedPolicy([{"kind": "move"}] * max(1, args.max_steps - 1))
            collection_policy = "scripted_smoke"
        else:
            policy = None
            collection_policy = (
                "controlled_trial_lightnav_on_policy" if args.controlled_trial else "lightnav_on_policy"
            )

        def policy_factory(info: dict[str, Any]) -> Any:
            if args.policy == "scripted":
                return policy
            from lightnav.habitat.runner import (
                HabitatEvalConfig,
                _default_engine_factory,
                build_velocity_policy,
            )

            cfg = HabitatEvalConfig(
                model_path=args.model_path,
                backend=args.backend,
                episodes=len(plan),
                max_steps=args.max_steps,
                output_dir=str(run_dir / "unused_eval_output"),
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            engine, bundle = _default_engine_factory(cfg)
            return build_velocity_policy(cfg, engine, bundle, info)

        # Build the real policy on the first reset inside a thin wrapper.
        class _Lazy:
            def __init__(self) -> None:
                self.inner = policy

            def reset(self, obs: dict[str, Any]) -> None:
                if self.inner is None:
                    raise RuntimeError("policy was not built")
                self.inner.reset(obs)

            def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> dict[str, Any]:
                return self.inner.act(obs, info)

            def stop_action(self) -> dict[str, Any]:
                return self.inner.stop_action()

            def get_info(self) -> dict[str, Any]:
                return self.inner.get_info()

        lazy = _Lazy()
        if args.policy == "lightnav":
            # The velocity ranges arrive on the first reset. Build then, without
            # letting run_habitat_eval write an evaluation result file.
            original_reset = env.reset

            def reset_and_build(seed: int | None = None):
                obs, info = original_reset(seed)
                if lazy.inner is None:
                    lazy.inner = policy_factory(info)
                return obs, info

            env.reset = reset_and_build  # type: ignore[method-assign]
        summary = run_session(
            env,
            lazy if args.policy == "lightnav" else policy,
            recorder,
            plan,
            max_steps=args.max_steps,
            requested_seed=args.seed,
            collection_policy=collection_policy,
            controlled_trial=args.controlled_trial,
            deadline_monotonic=run_deadline,
        )
        client_seed = _client_seed(args.seed)
        server_seed = summary.get("server_seed") or {}
        status = server_seed.get("status") or "unknown"
        if status == "verified":
            status = "unknown"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "collector_version": COLLECTOR_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": {
                "name": DATASET_NAME,
                "split": "train",
                "dataset_sha256": digest,
                "allowlist_sha256": allowlist.get("allowlist_sha256"),
                "scene_inventory_sha256": sha256_file(Path(args.scene_inventory)),
                "scene_split_version": inventory.get("scene_split_version"),
            },
            "allowlist_keys": [[row["scene"], row["episode_id"]] for row in plan],
            "served_keys": summary["reset_keys"],
            "reset_keys": summary["reset_keys"],
            "seed": {
                "requested": args.seed,
                "status": status,
                "effective": {
                    "client": client_seed,
                    "server": server_seed.get("effective"),
                    "requested_copied_as_effective": False,
                },
                "gpu_determinism": "not_claimed",
            },
            "config": {
                "path": args.config,
                "sha256": sha256_file(Path(args.config)) if Path(args.config).is_file() else None,
                "max_steps": args.max_steps,
                "wall_clock_budget_s": wall_budget_s,
                "note": "yaml keeps the baseline 500-step habitat cap; this run's budget is max_steps",
            },
            "wall_clock": {
                "started_at": run_started_at,
                "budget_s": wall_budget_s,
                "elapsed_s": round(time.monotonic() - run_started, 6),
                "stop_reason": "run_completed",
            },
            "model": {"path": args.model_path if args.policy == "lightnav" else None, "backend": args.backend},
            "code": _git_fingerprint(Path(".").resolve()),
            "depth_sensor": summary.get("depth_sensor"),
            "collection_policy": collection_policy,
            "campaign_id": args.campaign_id if not args.controlled_trial else None,
            "collection_partition": args.partition,
            "training_use": "not_approved",
            "diagnostic_only": bool(args.controlled_trial),
            "doc_deltas": [
                "run_habitat_eval is not the collector loop: it sends no action_id and writes results.jsonl",
                "RemoteEnvClient.reset hardcodes seed None and is not used",
                "dt_sim_s is measured from Simulator.get_world_time across Env.step",
                "seed status is partial or unapplied, never verified",
                "per-reset seed is recorded as unapplied because BaseHabitatEnv.reset ignores it",
                "Habitat EpisodeIterator groups by scene by default; the collector disables that and checks iterator order",
            ],
        }
        # served_keys must be the server's list. reset_keys is the client's observed order.
        # They are equal only when every planned episode was reset. run_session already
        # required the server's served_keys to equal the plan, and reset_keys follows plan.
        manifest["served_keys"] = [[row["scene"], row["episode_id"]] for row in plan]
        recorder.close()
        recorder.write_json("manifest.json", manifest)
        report = validate_run(run_dir)
        status = commit_validation(recorder, report)
        if status == 0 and not args.controlled_trial and args.partition == "train_fit":
            try:
                register_campaign_run(Path(args.campaign_manifest), args.campaign_id, run_dir)
            except (OSError, ValueError, KeyError) as exc:
                sys.stderr.write(f"campaign registration failed; validated run is preserved but unregistered: {exc}\n")
                return 1
        print(
            dumps(
                {
                    "run_dir": str(run_dir),
                    "hard_error_count": report["hard_error_count"],
                    "counts": report["counts"],
                    "training_use": "not_approved",
                }
            )
        )
        return status
    except Exception as exc:
        return abort_collection(
            recorder,
            exc,
            wall_clock={
                "started_at": run_started_at,
                "budget_s": wall_budget_s,
                "elapsed_s": round(time.monotonic() - run_started, 6),
                "stop_reason": "wall_clock_budget" if isinstance(exc, WallClockBudgetExceeded) else "error",
            },
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if env is not None:
            env.close()
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
