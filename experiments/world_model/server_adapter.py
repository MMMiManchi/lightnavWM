"""Collection boundary around VLNCEEnv.

The default Habitat server is not modified. This subclass records the action
that ``VLNCEEnv.step`` actually forwards, then the action ``_convert_action``
produces, then whether the underlying ``env.step`` returns or raises.

``BaseHabitatEnv.reset`` still does not apply a per-reset seed. A requested
seed is written into the Habitat config before ``Env`` is constructed and
``Env.seed`` is called once after the simulator exists. Status stays
``partial`` or ``unapplied``; this collector never reports ``verified``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from lightnav_habitat.vlnce import VLNCEEnv, episode_allow_key

from experiments.world_model.pose_math import YAW_METHOD, yaw_from_xyzw


def freeze_action(action: Any) -> Any:
    if isinstance(action, (int, np.integer)):
        return int(action)
    if isinstance(action, dict):
        args = action.get("action_args") or {}
        frozen = {"action": action.get("action")}
        if "linear_velocity" in args or "angular_velocity" in args:
            frozen["action_args"] = {
                "linear_velocity": float(args["linear_velocity"]),
                "angular_velocity": float(args["angular_velocity"]),
            }
        return frozen
    raise TypeError(f"unsupported action type {type(action).__name__}")


def same_action(left: Any, right: Any) -> bool:
    return freeze_action(left) == freeze_action(right)


def split_envelope(action: Any) -> tuple[str, Any]:
    if not isinstance(action, dict) or "wm_action_id" not in action:
        raise ValueError("step requires wm_action_id; a bare action is refused")
    action_id = str(action["wm_action_id"])
    if not action_id:
        raise ValueError("wm_action_id is empty")
    body = {key: value for key, value in action.items() if key != "wm_action_id"}
    return action_id, body


def _json_pose_number(value: Any) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("non-finite pose component")
    return number


def _sim_world_time(env: Any) -> Optional[float]:
    """Read Habitat-Sim world time; never substitute command or wall time."""
    try:
        value = float(env.sim.get_world_time())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if not np.isfinite(value) or value < 0:
        return None
    return value


class WorldModelVLNCEEnv(VLNCEEnv):
    """VLN-CE env that records execution evidence and enforces an exact allowlist."""

    def __init__(
        self,
        *args: Any,
        allow_order: Optional[list] = None,
        requested_seed: Optional[int] = None,
        expected_dataset_sha256: Optional[str] = None,
        event_driven: bool = False,
        **kwargs: Any,
    ) -> None:
        self.event_driven_collection = bool(event_driven)
        if self.event_driven_collection:
            if kwargs.get("max_steps") is not None:
                raise ValueError("event-driven collection cannot also set a per-episode step cap")
            kwargs["max_steps"] = None
        self.allow_order = [(str(scene), str(episode_id)) for scene, episode_id in (allow_order or [])]
        self.requested_seed = None if requested_seed is None else int(requested_seed)
        self.expected_dataset_sha256 = expected_dataset_sha256
        self.served_keys: list[tuple[str, str]] = []
        self.reset_keys: list[tuple[str, str]] = []
        self.seed_report: dict[str, Any] = {
            "requested": self.requested_seed,
            "status": "unknown",
            "effective": {},
        }
        self._sensor_flags: dict[str, Any] = {}
        self._by_id: dict[str, dict[str, Any]] = {}
        self._resolved: dict[str, Any] = {}
        self._min_abs_lin_speed = None
        self._min_abs_ang_speed = None
        super().__init__(*args, **kwargs)
        if self.allow_order:
            unique = set(self.allow_order)
            if len(unique) != len(self.allow_order):
                raise ValueError("duplicate episode id in allow_order")
            self.episode_allow = unique

    def _apply_dataset_overrides(self, config: Any) -> None:
        seen = None
        try:
            seen = int(config.habitat.seed)
        except (AttributeError, TypeError, ValueError):
            seen = None
        super()._apply_dataset_overrides(config)
        if getattr(self, "event_driven_collection", False):
            config.habitat.environment.max_episode_steps = 0
            config.habitat.environment.max_episode_seconds = 0
        effective_split = str(config.habitat.dataset.split)
        if self.split != "train" or effective_split != "train":
            raise RuntimeError(f"collection dataset split is {effective_split!r}; only train is accepted")
        self.seed_report["effective"]["habitat_config_before"] = {
            "value": seen,
            "applied_by_collector": False,
            "evidence": "value read from the config object before this collector wrote it",
        }
        if self.requested_seed is None:
            self.seed_report["effective"]["habitat_config"] = {
                "value": seen,
                "applied": False,
                "evidence": "collector was not given a seed; the pre-existing config value was left untouched",
            }
            return
        config.habitat.seed = int(self.requested_seed)
        written = int(config.habitat.seed)
        self.seed_report["effective"]["habitat_config"] = {
            "value": written,
            "applied": written == self.requested_seed,
            "evidence": "assigned config.habitat.seed inside _apply_dataset_overrides, before habitat.Env()",
        }

    def _apply_episode_split(self, env: Any) -> None:
        if not self.allow_order:
            raise RuntimeError("world-model server requires an explicit allow_order; it will not serve a split prefix")
        if self.split_id is not None or self.split_num is not None:
            raise RuntimeError("world-model collection does not shard; pass the exact episode list")
        available: dict[tuple[str, str], Any] = {}
        for episode in list(env._dataset.episodes):
            key = episode_allow_key(getattr(episode, "scene_id", ""), getattr(episode, "episode_id", ""))
            if key in available:
                raise RuntimeError(f"dataset contains duplicate episode key {key}")
            available[key] = episode
        missing = [key for key in self.allow_order if key not in available]
        if missing:
            raise RuntimeError(f"allowlist ids are not in the loaded split: {missing}")
        ordered = [available[key] for key in self.allow_order]
        self._pin_episode_iterator(env)
        env.episodes = ordered
        self._require_iterator_order(env)
        self.served_keys = list(self.allow_order)

    def _pin_episode_iterator(self, env: Any) -> None:
        """Keep reset order equal to the allowlist.

        Habitat's EpisodeIterator defaults to ``group_by_scene=True``. That
        pulls later episodes of the first scene forward, so an allowlist that
        alternates scenes is not the reset sequence. Shuffle and scene-repeat
        switches are also turned off. The iterator seed is not an order source
        once shuffle is false.
        """
        from contextlib import contextmanager

        options = env._config.environment.iterator_options
        try:
            from habitat.config.read_write import read_write
        except ImportError:
            read_write = None
        if read_write is None:

            @contextmanager
            def _unlocked():
                yield env._config

            ctx = _unlocked()
        else:
            ctx = read_write(env._config)
        with ctx:
            options.shuffle = False
            options.group_by_scene = False
            options.cycle = False
            options.num_episode_sample = -1
            options.max_scene_repeat_episodes = -1
            options.max_scene_repeat_steps = -1

    def _require_iterator_order(self, env: Any) -> None:
        iterator = getattr(env, "_episode_iterator", None)
        episodes = list(getattr(iterator, "episodes", []) or [])
        walked = [
            episode_allow_key(getattr(episode, "scene_id", ""), getattr(episode, "episode_id", ""))
            for episode in episodes
        ]
        if walked != self.allow_order:
            raise RuntimeError(f"episode iterator order {walked} != allowlist {self.allow_order}")
        if bool(getattr(iterator, "group_by_scene", True)):
            raise RuntimeError("episode iterator still groups by scene")
        if bool(getattr(iterator, "shuffle", True)):
            raise RuntimeError("episode iterator still shuffles")
        self._iterator_pin = {
            "shuffle": False,
            "group_by_scene": False,
            "cycle": bool(getattr(iterator, "cycle", False)),
            "order_matches_allowlist": True,
            "order_source": "allowlist",
            "seed_selects_order": False,
        }

    def _create_habitat_env(self) -> Any:
        env = super()._create_habitat_env()
        self._capture_resolved(env)
        self._finalize_seed(env)
        if self.expected_dataset_sha256 and self._resolved.get("dataset_sha256") != self.expected_dataset_sha256:
            raise RuntimeError("resolved dataset bytes do not match the allowlist hash")
        if str(self._resolved.get("split")) != "train":
            raise RuntimeError(f"resolved split is {self._resolved.get('split')!r}, expected train")
        return env

    def _capture_resolved(self, env: Any) -> None:
        config = env._config
        split = str(config.dataset.split)
        data_path = str(config.dataset.data_path)
        formatted = data_path.format(split=split)
        from hashlib import sha256
        from pathlib import Path

        file_path = Path(formatted)
        digest = None
        if file_path.is_file():
            digest_obj = sha256()
            with open(file_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest_obj.update(chunk)
            digest = digest_obj.hexdigest()
        depth = _sensor_contract(config, "depth_sensor")
        rgb = _sensor_contract(config, "rgb_sensor")
        try:
            velocity = config.task.actions.velocity_control
            self._min_abs_lin_speed = float(velocity.min_abs_lin_speed)
            self._min_abs_ang_speed = float(velocity.min_abs_ang_speed)
            command_dt = float(velocity.time_step)
        except (AttributeError, TypeError, ValueError):
            command_dt = self.habitat_time_step
        self._resolved = {
            "split": split,
            "data_path": formatted,
            "dataset_sha256": digest,
            "config_seed_readback": _maybe_int(getattr(config, "seed", None)),
            "rgb_sensor": rgb,
            "depth_sensor": depth,
            "command_dt_s": command_dt,
            "min_abs_lin_speed": self._min_abs_lin_speed,
            "min_abs_ang_speed": self._min_abs_ang_speed,
            "iterator_seed_evidence": (
                "habitat.core.env.Env._setup_episode_iterator copies self._config.seed into "
                "EpisodeIterator. That constructor calls random.seed only when the seed is "
                "truthy, and it permutes episodes only when shuffle is true. This collector "
                "forces shuffle and group_by_scene off before the episodes setter rebuilds "
                "the iterator, then checks iterator.episodes against the allowlist."
            ),
            "episode_iterator_pin": getattr(self, "_iterator_pin", None),
        }

    def _finalize_seed(self, env: Any) -> None:
        readback = self._resolved.get("config_seed_readback")
        numpy_evidence = None
        called = False
        if self.requested_seed is not None:
            env.seed(int(self.requested_seed))
            called = True
            state = np.random.get_state()[1]
            numpy_evidence = sha256_state(state)
        config_applied = (
            self.requested_seed is not None and readback == self.requested_seed
        )
        self.seed_report["effective"]["episode_iterator"] = {
            "value": readback,
            "applied_to_episode_order": False,
            "process_rng_reseeded_when_seed_truthy": bool(readback),
            "order_source": "allowlist",
            "evidence": self._resolved.get("iterator_seed_evidence"),
        }
        self.seed_report["effective"]["simulator_task"] = {
            "value": self.requested_seed if called else None,
            "applied": called,
            "readback": "not_available",
            "evidence": "habitat.core.env.Env.seed was called"
            if called
            else "Env.seed was not called",
        }
        self.seed_report["effective"]["numpy_in_server"] = {
            "value": self.requested_seed if called else None,
            "applied": called,
            "state_token": numpy_evidence,
            "evidence": "Env.seed calls np.random.seed; token is sha256 of the MT state words, not the seed",
        }
        if self.requested_seed is None:
            self.seed_report["status"] = "unapplied"
        elif config_applied and called:
            self.seed_report["status"] = "partial"
        else:
            self.seed_report["status"] = "unknown"
        self.seed_report["gpu_determinism"] = "not_claimed"

    def execution_record(self, action_id: str) -> Optional[dict]:
        row = self._by_id.get(str(action_id))
        if row is None:
            return None
        return dict(row["ack"])

    def _predict_override(self, _client_action: Any) -> Optional[str]:
        # Same predicates as VLNCEEnv.step, read before that method mutates them.
        if self._early_stop_rotation > 0 and self._consecutive_no_progress > self._early_stop_rotation:
            return "early_stop_no_progress"
        if self._early_stop_steps > 0 and self._step_count > self._early_stop_steps:
            return "early_stop_step_limit"
        return None

    def _build_observation(self, habitat_obs: Dict) -> Dict[str, Any]:
        rgb_key = next((key for key in self.RGB_KEYS if key in habitat_obs), None)
        depth_key = next((key for key in self.DEPTH_KEYS if key in habitat_obs), None)
        self._sensor_flags = {
            "rgb_key": rgb_key,
            "depth_key": depth_key,
            "rgb_missing": rgb_key is None,
            "depth_missing": depth_key is None,
            "rgb_missing_reason": None
            if rgb_key
            else "no rgb/head_rgb key; BaseHabitatEnv._extract_rgb would return zeros and those zeros are not saved",
            "depth_missing_reason": None
            if depth_key
            else "no depth/head_depth key; BaseHabitatEnv._extract_depth would return zeros and those zeros are not saved",
        }
        obs = super()._build_observation(habitat_obs)
        obs["wm_sensor"] = dict(self._sensor_flags)
        return obs

    def _control_units(self) -> dict[str, Any]:
        return {
            "normalized_interval": [-1.0, 1.0],
            "lin_vel_range_mps": None if self.lin_vel_range is None else [float(v) for v in self.lin_vel_range],
            "ang_vel_range_dps": None if self.ang_vel_range is None else [float(v) for v in self.ang_vel_range],
            "min_abs_lin_speed_mps": self._min_abs_lin_speed,
            "min_abs_ang_speed_dps": self._min_abs_ang_speed,
            "control_dt_s": None if self.habitat_time_step is None else float(self.habitat_time_step),
            "mapping_note": "normalized -1 is the low end of lin_vel_range; normalized 0 is not a stop",
        }

    def _read_pose(self) -> dict[str, Any]:
        state = self._habitat_env.sim.get_agent_state()
        position = [_json_pose_number(part) for part in state.position]
        rotation = state.rotation
        if not all(hasattr(rotation, name) for name in ("x", "y", "z", "w")):
            return {
                "world_position_xyz_m": position,
                "world_rotation_xyzw": None,
                "rotation_convention": None,
                "rotation_reason": f"rotation type {type(rotation).__name__} has no x/y/z/w components",
                "derived_yaw_rad": None,
                "yaw_method": None,
            }
        xyzw = [
            _json_pose_number(rotation.x),
            _json_pose_number(rotation.y),
            _json_pose_number(rotation.z),
            _json_pose_number(rotation.w),
        ]
        local_yaw = float(yaw_from_xyzw((xyzw[0], xyzw[1], xyzw[2], xyzw[3])))
        habitat_yaw = _habitat_yaw(rotation)
        yaw = local_yaw
        method = YAW_METHOD
        disagreement = None
        if habitat_yaw is not None:
            disagreement = float(habitat_yaw - local_yaw)
            method = "habitat_sim.quat_rotate_vector"
            yaw = float(habitat_yaw)
        return {
            "world_position_xyz_m": position,
            "world_rotation_xyzw": xyzw,
            "rotation_convention": "xyzw",
            "derived_yaw_rad": yaw,
            "yaw_method": method,
            "yaw_crosscheck_wm_quat_v1": local_yaw,
            "yaw_disagreement_rad": disagreement,
        }

    def _attach_privileged(self, info: dict[str, Any]) -> None:
        pose = None
        pose_reason = None
        try:
            pose = self._read_pose()
        except Exception as exc:  # noqa: BLE001 - the failure is the record
            pose_reason = f"{type(exc).__name__}: {exc}"
        info["wm_privileged"] = {
            "pose": pose,
            "pose_reason": pose_reason,
            "collision": {
                "state": "unknown",
                "source": "not_read",
                "reason": "previous_step_collided is not a confirmed contact sensor on these logs",
            },
        }
        info["wm_resolved"] = {
            "split": self._resolved.get("split", self.split),
            "dataset_sha256": self._resolved.get("dataset_sha256"),
            "served_keys": [list(key) for key in self.served_keys],
            "depth_sensor": self._resolved.get("depth_sensor"),
            "rgb_sensor": self._resolved.get("rgb_sensor"),
            "seed": self.seed_report,
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        sim_time_s = _sim_world_time(self._habitat_env)
        info["wm_sim_time_s"] = sim_time_s
        info["wm_sim_time_reason"] = "measured with Simulator.get_world_time" if sim_time_s is not None else "clock_unavailable"
        key = episode_allow_key(str(info.get("scene_id", "")), info.get("episode_id", ""))
        if self.served_keys and key not in set(self.served_keys):
            raise RuntimeError(f"reset returned {key}, which is outside the served allowlist")
        self.reset_keys.append(key)
        self._attach_privileged(info)
        info["wm_reset"] = {
            "key": list(key),
            "reset_seed_argument": None if seed is None else int(seed),
            "applied_in_reset": False,
            "reason": "VLNCEEnv.reset calls BaseHabitatEnv.reset, which does not call Env.seed",
            "init_seed_status": self.seed_report.get("status"),
        }
        return obs, info

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        action_id, client_action = split_envelope(action)
        previous = self._by_id.get(action_id)
        if previous is not None:
            if previous["status"] == "executed":
                replay = _clone_step(previous["result"])
                replay[4]["wm_execution"] = dict(previous["ack"])
                replay[4]["wm_execution"]["replayed"] = True
                return replay
            raise RuntimeError(
                f"refusing to repeat action_id {action_id} after status {previous['status']}"
            )

        predicted = self._predict_override(client_action)
        trace: dict[str, Any] = {
            "action_id": action_id,
            "received": True,
            "submitted": False,
            "executed": False,
            "status": "received",
            "error": None,
            "client_request": freeze_action(client_action),
            "server_override": False,
            "override_reason": None,
            "action_after_server_logic": None,
            "final_habitat_action": None,
            "step_index_before": int(self._step_count),
            "step_index_after": None,
            "dt_wall_s": None,
            "dt_command_s": None,
            "dt_command_reason": None,
            "dt_sim_s": None,
            "sim_time_before_s": None,
            "sim_time_after_s": None,
            "dt_sim_reason": "clock_unavailable",
            "habitat_stop_called": False,
            "replayed": False,
            "control_units": self._control_units(),
        }
        original_convert = self._convert_action
        habitat = self._habitat_env
        real_step = habitat.step

        def convert(forwarded: Any) -> Any:
            trace["action_after_server_logic"] = freeze_action(forwarded)
            converted = original_convert(forwarded)
            trace["final_habitat_action"] = freeze_action(converted)
            trace["submitted"] = True
            trace["status"] = "submitted"
            return converted

        def traced_step(habitat_action: Any) -> Any:
            sim_before = _sim_world_time(habitat)
            trace["sim_time_before_s"] = sim_before
            started = time.monotonic()
            try:
                observed = real_step(habitat_action)
            except Exception as exc:
                trace["dt_wall_s"] = time.monotonic() - started
                sim_after = _sim_world_time(habitat)
                trace["sim_time_after_s"] = sim_after
                trace["dt_sim_s"] = sim_after - sim_before if sim_before is not None and sim_after is not None and sim_after >= sim_before else None
                trace["dt_sim_reason"] = "measured across Habitat Env.step; step raised" if trace["dt_sim_s"] is not None else "clock_unavailable_or_nonmonotonic"
                trace["status"] = "error"
                trace["error"] = f"{type(exc).__name__}: {exc}"
                raise
            trace["dt_wall_s"] = time.monotonic() - started
            sim_after = _sim_world_time(habitat)
            trace["sim_time_after_s"] = sim_after
            trace["dt_sim_s"] = sim_after - sim_before if sim_before is not None and sim_after is not None and sim_after >= sim_before else None
            trace["dt_sim_reason"] = "measured with Simulator.get_world_time across Env.step" if trace["dt_sim_s"] is not None else "clock_unavailable_or_nonmonotonic"
            trace["executed"] = True
            trace["status"] = "executed"
            return observed

        self._convert_action = convert
        habitat.step = traced_step
        try:
            result = VLNCEEnv.step(self, client_action)
        except Exception as exc:
            if trace["status"] != "error":
                trace["status"] = "error"
                trace["error"] = f"{type(exc).__name__}: {exc}"
            self._by_id[action_id] = {"status": "error", "ack": trace, "result": None}
            raise
        finally:
            self._convert_action = original_convert
            habitat.step = real_step

        after = trace["action_after_server_logic"]
        override = after is not None and not same_action(client_action, after)
        trace["server_override"] = bool(override)
        if override:
            trace["override_reason"] = predicted or "unconfirmed"
        final = trace["final_habitat_action"]
        if isinstance(final, dict) and final.get("action") == "velocity_control" and self.habitat_time_step is not None:
            trace["dt_command_s"] = float(self.habitat_time_step)
            trace["dt_command_reason"] = "resolved velocity_control.time_step of the submitted command"
        else:
            trace["dt_command_reason"] = "discrete STOP or missing resolved time_step; command dt was not defaulted"
        obs, reward, terminated, truncated, info = result
        trace["step_index_after"] = int(self._step_count)
        info["wm_sim_time_s"] = trace.get("sim_time_after_s")
        info["wm_sim_time_reason"] = trace.get("dt_sim_reason")
        trace["habitat_stop_called"] = bool(self._is_stop_called())
        self._attach_privileged(info)
        info["wm_execution"] = trace
        info["wm_sensor"] = dict(self._sensor_flags)
        stored = (obs, reward, terminated, truncated, info)
        self._by_id[action_id] = {"status": "executed", "ack": dict(trace), "result": _clone_step(stored)}
        return obs, reward, terminated, truncated, info


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sha256_state(words: Any) -> str:
    from hashlib import sha256

    array = np.asarray(words)
    return sha256(array.tobytes()).hexdigest()[:16]


def _sensor_contract(config: Any, name: str) -> dict[str, Any]:
    try:
        sensor = config.simulator.agents.main_agent.sim_sensors[name]
    except (AttributeError, KeyError, TypeError):
        return {"present_in_resolved_config": False}
    wanted = (
        "type",
        "width",
        "height",
        "hfov",
        "position",
        "min_depth",
        "max_depth",
        "normalize_depth",
    )
    record: dict[str, Any] = {"present_in_resolved_config": True}
    for field in wanted:
        if not hasattr(sensor, field):
            record[field] = None
            continue
        value = getattr(sensor, field)
        if isinstance(value, (int, float, str, bool)) or value is None:
            record[field] = value if not isinstance(value, float) else float(value)
        else:
            try:
                record[field] = [float(part) for part in value]
            except TypeError:
                record[field] = str(value)
    record["unit_note"] = (
        "meters only if normalize_depth is false and min_depth/max_depth are present; "
        "a missing flag is null, not assumed"
    )
    return record


def _habitat_yaw(rotation: Any) -> Optional[float]:
    try:
        import quaternion
        from habitat_sim.utils.common import quat_rotate_vector
    except Exception:
        return None
    try:
        quat = quaternion.quaternion(float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z))
        forward = np.asarray(quat_rotate_vector(quat, np.array([0.0, 0.0, -1.0])), dtype=np.float64)
        return float(np.arctan2(-forward[0], -forward[2]))
    except Exception:
        return None


def _clone_step(result: tuple) -> tuple:
    obs, reward, terminated, truncated, info = result
    copied_obs = dict(obs)
    for key in ("rgb", "depth"):
        if isinstance(copied_obs.get(key), np.ndarray):
            copied_obs[key] = copied_obs[key].copy()
    return copied_obs, reward, terminated, truncated, dict(info)
