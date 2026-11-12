"""CPU contract tests for train-only transition collection."""

from __future__ import annotations

import gzip
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.world_model import SCHEMA_VERSION
from experiments.world_model.collect import (
    FORMAL_MAX_STEPS,
    ScriptedPolicy,
    WallClockBudgetExceeded,
    _check_deadline,
    run_session,
)
from experiments.world_model.group_manifest import (
    assign_groups,
    build_allowlist,
    check_allowlist_against_train,
    load_split_rows,
    refuse_legacy_path,
)
from experiments.world_model.group_manifest import (
    main as group_main,
)
from experiments.world_model.jsonutil import dumps, loads, sha256_bytes
from experiments.world_model.online_input import assert_loader_ignores_privileged
from experiments.world_model.pose_math import wrap_pi, yaw_from_xyzw
from experiments.world_model.recorder import RunRecorder
from experiments.world_model.serve_collect import main as serve_main
from experiments.world_model.server_adapter import WorldModelVLNCEEnv
from experiments.world_model.validate import validate_run
from lightnav.habitat.runner import HabitatEvalConfig, run_habitat_eval
from lightnav.velocity import first_waypoint_to_velocity_cmd


def _episode_json(scene: str, episode_id: int, trajectory_id: int, text: str) -> dict:
    return {
        "episode_id": episode_id,
        "trajectory_id": trajectory_id,
        "scene_id": f"mp3d/{scene}/{scene}.glb",
        "start_position": [float(episode_id), 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "goals": [{"position": [1.0, 0.0, 1.0], "radius": 3.0}],
        "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        "info": {"geodesic_distance": 4.0},
        "instruction": {"instruction_text": text},
    }


def _write_split(root: Path, split: str, episodes: list[dict]) -> None:
    path = root / split / f"{split}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": episodes, "instruction_vocab": {"word_list": []}}, handle)


class _Rotation:
    x = 0.0
    y = 0.0
    z = 0.0
    w = 1.0


class FakeHabitat:
    def __init__(self, episodes: list[SimpleNamespace]):
        self.episodes = episodes
        self.index = 0
        self.current_episode = episodes[0]
        self.pos = np.array([0.0, 0.5, 0.0])
        self.step_calls = 0
        self.world_time = 0.0
        self.world_time_increment = 0.016
        self.fail = False
        self.rgb = True
        self.depth = True
        self.seed_calls: list[int] = []
        self.task = SimpleNamespace(is_stop_called=False)
        self.sim = self

    def seed(self, value: int) -> None:
        self.seed_calls.append(int(value))

    def get_world_time(self) -> float:
        return self.world_time

    def get_agent_state(self):
        return SimpleNamespace(position=self.pos.copy(), rotation=_Rotation())

    def geodesic_distance(self, _start, _goal) -> float:
        return 4.0

    def get_metrics(self) -> dict:
        return {"distance_to_goal": 4.0, "success": 0.0, "spl": 0.0, "ndtw": 0.0, "oracle_success": 0.0}

    def reset(self) -> dict:
        self.current_episode = self.episodes[self.index]
        self.index += 1
        self.pos = np.array([0.0, 0.5, 0.0])
        self.world_time = 0.0
        self.task.is_stop_called = False
        return self._obs()

    def step(self, action):
        self.step_calls += 1
        self.world_time += self.world_time_increment
        if self.fail:
            raise RuntimeError("sensor boom")
        stop = action == 0
        if isinstance(action, dict):
            linear = float(action["action_args"]["linear_velocity"])
            angular = float(action["action_args"]["angular_velocity"])
            stop = abs(linear + 1.0) < 1e-9 and abs(angular) < 1e-9
            if not stop:
                self.pos = self.pos + np.array([0.1, 0.0, 0.0])
        if stop:
            self.task.is_stop_called = True
        return self._obs()

    def _obs(self) -> dict:
        text = self.current_episode.instruction.instruction_text
        obs = {"instruction": {"text": text}}
        if self.rgb:
            image = np.zeros((4, 6, 3), dtype=np.uint8)
            image[0, 0, 0] = np.uint8(self.step_calls % 256)
            obs["rgb"] = image
        if self.depth:
            obs["depth"] = np.full((4, 6), 0.25, dtype=np.float32)
        return obs


def _plan_row(scene: str, episode_id: str, text: str, group: str) -> dict:
    return {
        "scene": scene,
        "scene_id_raw": f"mp3d/{scene}/{scene}.glb",
        "episode_id": episode_id,
        "episode_uid": f"R2R_VLNCE_v1-3_preprocessed|train|{scene}|{episode_id}",
        "instruction_text": text,
        "instruction_id_native": None,
        "instruction_id_derived": f"derived-{scene}-{episode_id}",
        "route_id_native": "4",
        "route_group_id": group,
        "route_geometry_hash": f"geom-{scene}",
        "partition": "train_fit",
        "source_split": "train",
        "source_geodesic_m": 4.0,
    }


def _namespace_episode(scene: str, episode_id: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        episode_id=episode_id,
        scene_id=f"mp3d/{scene}/{scene}.glb",
        trajectory_id="4",
        goals=[SimpleNamespace(position=[1.0, 0.0, 1.0])],
        reference_path=[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        instruction=SimpleNamespace(instruction_text=text, language=None),
    )


def _attach(episodes: list[SimpleNamespace], *, seed: int | None = 7, time_step: float = 0.2):
    order = []
    for episode in episodes:
        scene = episode.scene_id.split("/")[-2]
        order.append((scene, str(episode.episode_id)))
    env = WorldModelVLNCEEnv(
        config_path="unused.yaml",
        split="train",
        max_steps=20,
        episode_allow=set(order),
        allow_order=order,
        requested_seed=seed,
    )
    fake = FakeHabitat(episodes)
    env._habitat_env = fake
    env._initialized = True
    env.served_keys = order
    env.habitat_time_step = time_step
    env.lin_vel_range = (0.0, 2.5)
    env.ang_vel_range = (-300.0, 300.0)
    env._resolved = {"split": "train", "dataset_sha256": "abc", "depth_sensor": {"normalize_depth": None}}
    return env, fake


def _finish(recorder: RunRecorder, plan: list[dict], *, seed_status: str = "partial") -> dict:
    keys = [[row["scene"], row["episode_id"]] for row in plan]
    recorder.close()
    recorder.write_json(
        "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source": {"split": "train", "dataset_sha256": "abc"},
            "allowlist_keys": keys,
            "served_keys": keys,
            "reset_keys": keys,
            "seed": {"requested": 7, "status": seed_status, "effective": {"requested_copied_as_effective": False}},
        },
    )
    report = validate_run(recorder.run_dir)
    recorder.write_json("validation_report.json", report)
    return report


def _rows(run: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (run / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_yaw_identity_and_left_turn():
    assert yaw_from_xyzw((0.0, 0.0, 0.0, 1.0)) == pytest.approx(0.0)
    half = math.sqrt(2.0) / 2.0
    assert yaw_from_xyzw((0.0, half, 0.0, half)) == pytest.approx(math.pi / 2.0)
    assert wrap_pi(math.pi + 0.2) == pytest.approx(-math.pi + 0.2)


def test_train_allowlist_rejects_unknown_duplicate_and_val(tmp_path: Path):
    root = tmp_path / "data"
    train = [
        _episode_json("sceneA", 1, 10, "go left"),
        _episode_json("sceneB", 2, 11, "go right"),
        _episode_json("sceneA", 3, 10, "go left again"),
    ]
    _write_split(root, "train", train)
    _write_split(root, "val_unseen", [_episode_json("sceneA", 99, 10, "hidden")])
    path, digest, rows = load_split_rows(root, "train")
    grouped = assign_groups(rows)
    same_group = [row for row in grouped if row["route_group_id"] == grouped[0]["route_group_id"]]
    assert len(same_group) >= 2
    assert len({row["partition"] for row in same_group}) == 1
    chosen = [grouped[0], next(row for row in grouped if row["scene"] != grouped[0]["scene"])]
    allow = build_allowlist(grouped, chosen, dataset_json=path, dataset_sha256=digest)
    checked = check_allowlist_against_train(loads(dumps(allow)), grouped, dataset_sha256=digest)
    assert [(row["scene"], row["episode_id"]) for row in checked] == [
        (row["scene"], row["episode_id"]) for row in chosen
    ]

    unknown = loads(dumps(allow))
    unknown["episodes"].append(dict(unknown["episodes"][0], episode_id="999"))
    unknown["allowlist_sha256"] = sha256_bytes(dumps(unknown["episodes"]).encode())
    with pytest.raises(RuntimeError, match="unknown train episode"):
        check_allowlist_against_train(unknown, grouped, dataset_sha256=digest)

    duplicate = loads(dumps(allow))
    duplicate["episodes"].append(dict(duplicate["episodes"][0]))
    duplicate["allowlist_sha256"] = sha256_bytes(dumps(duplicate["episodes"]).encode())
    with pytest.raises(RuntimeError, match="duplicate"):
        check_allowlist_against_train(duplicate, grouped, dataset_sha256=digest)

    tainted = loads(dumps(allow))
    tainted["source_split"] = "val_unseen"
    with pytest.raises(RuntimeError, match="only train"):
        check_allowlist_against_train(tainted, grouped, dataset_sha256=digest)

    leaked = loads(dumps(allow))
    leaked["allowlist_sha256"] = sha256_bytes(dumps(leaked["episodes"]).encode())
    with pytest.raises(RuntimeError, match="val split"):
        check_allowlist_against_train(
            leaked,
            grouped,
            dataset_sha256=digest,
            forbidden_keys={("sceneA", leaked["episodes"][0]["episode_id"])},
        )

    with pytest.raises(SystemExit, match="train-only"):
        group_main(["--split", "val_unseen", "--dry-run"])
    before = sorted(path.as_posix() for path in root.rglob("*"))
    assert group_main(["--split", "train", "--dataset-root", str(root), "--dry-run"]) == 0
    assert sorted(path.as_posix() for path in root.rglob("*")) == before
    out = tmp_path / "pilot.json"
    code = group_main(
        ["--split", "train", "--dataset-root", str(root), "--pilot", "3", "--min-scenes", "2", "--out", str(out)]
    )
    if code == 0:
        assert out.is_file()
        with pytest.raises(SystemExit, match="overwrite"):
            group_main(
                [
                    "--split",
                    "train",
                    "--dataset-root",
                    str(root),
                    "--pilot",
                    "3",
                    "--min-scenes",
                    "2",
                    "--out",
                    str(out),
                ]
            )
    with pytest.raises(RuntimeError, match="legacy"):
        refuse_legacy_path(tmp_path / "r2r_probe" / "run")


def test_server_allowlist_must_match_loaded_split():
    env, _fake = _attach([_namespace_episode("sceneA", "1", "go left")])
    inner = SimpleNamespace(
        _dataset=SimpleNamespace(episodes=[SimpleNamespace(scene_id="mp3d/sceneA/sceneA.glb", episode_id="1")]),
        episodes=None,
    )
    env.allow_order = [("sceneA", "1"), ("missing", "9")]
    with pytest.raises(RuntimeError, match="not in the loaded split"):
        env._apply_episode_split(inner)
    with pytest.raises(ValueError, match="duplicate"):
        WorldModelVLNCEEnv(
            config_path="unused.yaml",
            split="train",
            allow_order=[("sceneA", "1"), ("sceneA", "1")],
        )


def test_allowlist_order_is_not_regrouped_by_scene():
    from lightnav_habitat.vlnce import episode_allow_key

    episodes = [
        SimpleNamespace(scene_id="mp3d/sceneA/sceneA.glb", episode_id="1"),
        SimpleNamespace(scene_id="mp3d/sceneB/sceneB.glb", episode_id="4"),
        SimpleNamespace(scene_id="mp3d/sceneA/sceneA.glb", episode_id="2"),
    ]
    options = SimpleNamespace(
        shuffle=True,
        group_by_scene=True,
        cycle=True,
        num_episode_sample=20,
        max_scene_repeat_episodes=1,
        max_scene_repeat_steps=10000,
    )

    class GroupingEnv:
        def __init__(self) -> None:
            self._dataset = SimpleNamespace(episodes=list(episodes))
            self._config = SimpleNamespace(environment=SimpleNamespace(iterator_options=options))
            self._episode_iterator = None

        @property
        def episodes(self):
            return self._dataset.episodes

        @episodes.setter
        def episodes(self, value):
            opts = self._config.environment.iterator_options
            value = list(value)
            if opts.group_by_scene:
                first: dict[str, int] = {}
                for episode in value:
                    first.setdefault(episode.scene_id, len(first))
                value = sorted(value, key=lambda episode: first[episode.scene_id])
            self._dataset.episodes = value
            self._episode_iterator = SimpleNamespace(
                episodes=value,
                shuffle=bool(opts.shuffle),
                group_by_scene=bool(opts.group_by_scene),
                cycle=bool(opts.cycle),
            )

    order = [("sceneA", "1"), ("sceneB", "4"), ("sceneA", "2")]
    env = WorldModelVLNCEEnv(config_path="unused.yaml", split="train", allow_order=order)
    inner = GroupingEnv()
    env._apply_episode_split(inner)
    walked = [
        episode_allow_key(episode.scene_id, episode.episode_id)
        for episode in inner._episode_iterator.episodes
    ]
    assert walked == order
    assert inner._episode_iterator.group_by_scene is False
    assert inner._episode_iterator.shuffle is False
    assert inner._episode_iterator.cycle is False
    assert options.num_episode_sample == -1
    assert options.max_scene_repeat_steps == -1


def test_reset_seed_is_not_applied_and_init_seed_is_partial():
    calls: list[int] = []

    class SeedEnv:
        def __init__(self) -> None:
            self._config = SimpleNamespace(seed=7)

        def seed(self, value: int) -> None:
            calls.append(int(value))

    env = WorldModelVLNCEEnv(config_path="unused.yaml", split="train", requested_seed=7)
    habitat = SimpleNamespace()
    habitat.seed = 100
    habitat.dataset = SimpleNamespace(split="val_unseen", data_path="keep", scenes_dir=None)
    habitat.simulator = SimpleNamespace(habitat_sim_v0=SimpleNamespace(gpu_device_id=1))
    config = SimpleNamespace(habitat=habitat)
    env._apply_dataset_overrides(config)
    assert config.habitat.seed == 7
    assert config.habitat.dataset.split == "train"
    env._resolved = {
        "config_seed_readback": 7,
        "iterator_seed_evidence": "config.seed copied by _setup_episode_iterator",
    }
    env._finalize_seed(SeedEnv())
    assert calls == [7]
    assert env.seed_report["status"] == "partial"
    assert env.seed_report["effective"]["simulator_task"]["readback"] == "not_available"

    unapplied = WorldModelVLNCEEnv(config_path="unused.yaml", split="train", requested_seed=None)
    unapplied._resolved = {"config_seed_readback": 100, "iterator_seed_evidence": "left at the config default"}
    unapplied._finalize_seed(SeedEnv())
    assert unapplied.seed_report["status"] == "unapplied"
    assert calls == [7]

    attached, fake = _attach([_namespace_episode("sceneA", "1", "go left")], seed=None)
    _obs, info = attached.reset(seed=123)
    assert info["wm_reset"]["applied_in_reset"] is False
    assert fake.seed_calls == []
    assert "does not call Env.seed" in info["wm_reset"]["reason"]


def test_zero_simulator_time_is_a_measured_value(tmp_path: Path):
    episode = _namespace_episode("sceneA", "1", "go left")
    plan = [_plan_row("sceneA", "1", "go left", "sceneA:4")]
    env, fake = _attach([episode])
    fake.world_time_increment = 0.0
    recorder = RunRecorder(tmp_path / "zero-clock")
    run_session(env, ScriptedPolicy([{"kind": "move"}]), recorder, plan, max_steps=2, requested_seed=7, collection_policy="scripted_test")
    report = _finish(recorder, plan)
    assert report["counts"]["valid"] == 2
    transitions = _rows(recorder.run_dir, "transitions.jsonl")
    assert transitions[0]["action"]["dt_sim_s"] == 0.0
    assert transitions[0]["action"]["dt_sim_reason"].startswith("measured with Simulator")


def test_missing_simulator_clock_quarantines_transition(tmp_path: Path):
    episode = _namespace_episode("sceneA", "1", "go left")
    plan = [_plan_row("sceneA", "1", "go left", "sceneA:4")]
    env, fake = _attach([episode])
    fake.sim = SimpleNamespace()
    recorder = RunRecorder(tmp_path / "missing-clock")
    run_session(env, ScriptedPolicy([{"kind": "move"}]), recorder, plan, max_steps=2, requested_seed=7, collection_policy="scripted_test")
    report = _finish(recorder, plan)
    assert report["hard_error_count"] == 0
    assert report["counts"]["quarantined"] == 2
    transitions = _rows(recorder.run_dir, "transitions.jsonl")
    assert transitions[0]["action"]["dt_sim_s"] is None
    assert "sim_time_unavailable" in transitions[0]["quality"]["errors"]


def test_move_budget_terminal_frame_and_no_stale_prediction(tmp_path: Path):
    episodes = [
        _namespace_episode("sceneA", "1", "go left"),
        _namespace_episode("sceneB", "2", "go right"),
    ]
    plan = [
        _plan_row("sceneA", "1", "go left", "sceneA:4"),
        _plan_row("sceneB", "2", "go right", "sceneB:4"),
    ]
    env, _fake = _attach(episodes, time_step=0.2)
    policy = ScriptedPolicy([{"kind": "move"}])
    recorder = RunRecorder(tmp_path / "run-happy")
    run_session(env, policy, recorder, plan, max_steps=2, requested_seed=7, collection_policy="scripted_test")
    report = _finish(recorder, plan)
    assert report["hard_error_count"] == 0, report["errors"]
    assert report["counts"] == {"valid": 4, "partial": 0, "quarantined": 0}
    assert report["collection_report"]["valid_action_kind_counts"] == {
        "stop_threshold": 2,
        "turning": 0,
        "mostly_forward": 2,
    }
    assert report["collection_report"]["primary_end_source_counts"] == {
        "runner_budget": 2,
        "unknown": 2,
    }
    assert report["collection_report"]["episode_completion_counts"] == {"budget_truncated": 2}
    transitions = _rows(recorder.run_dir, "transitions.jsonl")
    observations = _rows(recorder.run_dir, "observations.jsonl")
    assert {row["scene"] for row in _rows(recorder.run_dir, "episodes.jsonl")} == {"sceneA", "sceneB"}
    assert policy.last_info_keys == ["ang_vel_range", "habitat_time_step", "lin_vel_range"]
    move = next(row for row in transitions if row["episode_uid"].endswith("|1") and row["t"] == 0)
    expected = first_waypoint_to_velocity_cmd(
        np.array([0.2, 0.05, 0.0], dtype=np.float32), 0.1, (0.0, 2.5), (-300.0, 300.0)
    )
    assert move["action"]["client_request"]["action_args"]["linear_velocity"] == pytest.approx(
        expected["action_args"]["linear_velocity"]
    )
    assert move["action"]["final_habitat_action"] == move["action"]["client_request"]
    assert move["action"]["server_override"] is False
    assert move["prediction"]["lateral_m_not_in_command"] == pytest.approx(np.float32(0.05))
    assert len(move["action"]["client_request"]["action_args"]) == 2
    assert move["action"]["dt_command_s"] == pytest.approx(0.2)
    assert move["action"]["dt_sim_s"] == pytest.approx(0.016)
    assert move["action"]["sim_time_after_s"] - move["action"]["sim_time_before_s"] == pytest.approx(0.016)
    assert observations[0]["sim_time_s"] == pytest.approx(0.0)
    assert move["action"]["control_units"]["min_abs_lin_speed_mps"] is None
    assert move["action"]["execution_ack"]["status"] == "executed"
    labels = {row["label_id"]: row for row in _rows(recorder.run_dir, "privileged_labels.jsonl")}
    before = labels[move["pose_before_id"]]["pose"]["world_position_xyz_m"]
    after = labels[move["pose_after_id"]]["pose"]["world_position_xyz_m"]
    assert before[0] == pytest.approx(0.0)
    assert after[0] == pytest.approx(0.1)
    assert "distance_to_goal" not in move
    assert labels[move["pose_before_id"]]["geodesic_distance_m"] == pytest.approx(4.0)
    budget = next(row for row in transitions if row["t"] == 1)
    assert budget["prediction"]["inference_called"] is False
    assert budget["prediction"]["raw_text"] is None
    assert budget["prediction"]["decoded_waypoints"] is None
    assert budget["prediction"]["model_stop"] is None
    assert budget["prediction"]["decode_status"] == "not_called"
    assert policy.get_info()["raw_text"].startswith("<act")
    assert budget["action"]["final_habitat_action"]["action_args"]["linear_velocity"] == pytest.approx(-1.0)
    assert budget["outcome"]["primary_end_source"] == "runner_budget"
    assert "model_stop" not in budget["outcome"]["end_sources"]
    terminals = [row for row in observations if row["terminal_observation"]]
    assert len(terminals) == 2
    assert {row["step_index"] for row in terminals} == {2}
    assert budget["obs_after_id"] not in budget["online_input"]["observation_ids"]
    opened = assert_loader_ignores_privileged(
        recorder.run_dir, lambda: _rows(recorder.run_dir, "transitions.jsonl")[0]["online_input"]
    )
    assert "distance_to_goal" not in opened["info_keys"]


def test_formal_step_cap_and_wall_clock_deadline(tmp_path: Path):
    recorder = RunRecorder(tmp_path / "run-empty-plan")
    run_session(
        object(), object(), recorder, [], max_steps=FORMAL_MAX_STEPS,
        requested_seed=None, collection_policy="test",
    )
    recorder.close()
    with pytest.raises(ValueError, match="1..500"):
        run_session(
            object(), object(), RunRecorder(tmp_path / "run-too-many"), [],
            max_steps=FORMAL_MAX_STEPS + 1, requested_seed=None, collection_policy="test",
        )
    with pytest.raises(WallClockBudgetExceeded):
        _check_deadline(0.0)
    expired = RunRecorder(tmp_path / "run-expired-before-reset")
    with pytest.raises(WallClockBudgetExceeded):
        run_session(
            object(), object(), expired, [{"episode_uid": "ep"}], max_steps=1,
            requested_seed=None, collection_policy="test", deadline_monotonic=0.0,
        )
    expired.close()


def test_model_stop_decode_fallback_and_server_override(tmp_path: Path):
    def one(kind: str, name: str, *, override: bool = False) -> dict:
        env, _fake = _attach([_namespace_episode("sceneA", "1", "go left")])
        if override:
            original = env.reset

            def reset(seed=None):
                obs, info = original(seed=seed)
                env._early_stop_rotation = 1
                env._consecutive_no_progress = 5
                return obs, info

            env.reset = reset  # type: ignore[method-assign]
        plan = [_plan_row("sceneA", "1", "go left", "sceneA:4")]
        recorder = RunRecorder(tmp_path / name)
        run_session(
            env,
            ScriptedPolicy([{"kind": kind}]),
            recorder,
            plan,
            max_steps=2,
            requested_seed=7,
            collection_policy="scripted_test",
        )
        report = _finish(recorder, plan)
        assert report["hard_error_count"] == 0, report["errors"]
        return _rows(recorder.run_dir, "transitions.jsonl")[0]

    stopped = one("model_stop", "model-stop")
    assert stopped["prediction"]["model_stop"] is True
    assert stopped["prediction"]["decode_status"] == "decode_ok"
    assert stopped["outcome"]["primary_end_source"] == "model_stop"
    assert "decode_fallback" not in stopped["outcome"]["end_sources"]

    failed = one("decode_fail", "decode-fail")
    assert failed["prediction"]["decode_status"] == "decode_failed"
    assert failed["prediction"]["model_stop"] is False
    assert failed["prediction"]["decoded_waypoints"] is None
    assert failed["prediction"]["raw_text"] == "not-an-action"
    assert failed["outcome"]["primary_end_source"] == "decode_fallback"
    assert "model_stop" not in failed["outcome"]["end_sources"]
    assert failed["action"]["final_habitat_action"]["action_args"]["linear_velocity"] == pytest.approx(-1.0)

    replaced = one("move", "override", override=True)
    assert replaced["action"]["server_override"] is True
    assert replaced["action"]["override_reason"] == "early_stop_no_progress"
    assert replaced["action"]["final_habitat_action"] == 0
    assert replaced["action"]["client_request"]["action"] == "velocity_control"
    assert replaced["outcome"]["primary_end_source"] == "server_early_stop"
    assert replaced["action"]["dt_command_s"] is None


def test_step_exception_keeps_no_fabricated_next_frame(tmp_path: Path):
    env, fake = _attach([_namespace_episode("sceneA", "1", "go left")])
    fake.fail = True
    plan = [_plan_row("sceneA", "1", "go left", "sceneA:4")]
    recorder = RunRecorder(tmp_path / "boom")
    run_session(
        env,
        ScriptedPolicy([{"kind": "move"}]),
        recorder,
        plan,
        max_steps=2,
        requested_seed=7,
        collection_policy="scripted_test",
    )
    report = _finish(recorder, plan)
    assert report["hard_error_count"] == 0, report["errors"]
    assert report["counts"]["quarantined"] == 1
    assert report["counts"]["valid"] == 0
    row = _rows(recorder.run_dir, "transitions.jsonl")[0]
    assert row["obs_after_id"] is None
    assert row["quality"]["status"] == "quarantined"
    assert row["action"]["execution_ack"]["status"] == "error"
    assert row["action"]["execution_ack"]["executed"] is False
    assert "sensor boom" in row["action"]["execution_ack"]["error"]
    assert len(list((recorder.run_dir / "assets").glob("*_rgb.npy"))) == 1


def test_depth_or_rgb_missing_is_explicit(tmp_path: Path):
    plan = [_plan_row("sceneA", "1", "go left", "sceneA:4")]
    env, fake = _attach([_namespace_episode("sceneA", "1", "go left")])
    fake.depth = False
    recorder = RunRecorder(tmp_path / "nodepth")
    run_session(
        env,
        ScriptedPolicy([{"kind": "move"}]),
        recorder,
        plan,
        max_steps=1,
        requested_seed=7,
        collection_policy="scripted_test",
    )
    report = _finish(recorder, plan)
    assert report["hard_error_count"] == 0, report["errors"]
    observation = _rows(recorder.run_dir, "observations.jsonl")[0]
    assert observation["depth"] is None
    assert observation["depth_missing"] is True
    assert "zeros" in observation["depth_missing_reason"]
    assert list((recorder.run_dir / "assets").glob("*_depth.npy")) == []

    env, fake = _attach([_namespace_episode("sceneA", "1", "go left")])
    real_step = fake.step

    def step_without_rgb(action):
        observed = real_step(action)
        observed.pop("rgb", None)
        return observed

    fake.step = step_without_rgb
    recorder = RunRecorder(tmp_path / "norgb")
    run_session(
        env,
        ScriptedPolicy([{"kind": "move"}]),
        recorder,
        plan,
        max_steps=2,
        requested_seed=7,
        collection_policy="scripted_test",
    )
    report = _finish(recorder, plan)
    assert report["hard_error_count"] == 0, report["errors"]
    bad = _rows(recorder.run_dir, "transitions.jsonl")[0]
    assert bad["quality"]["status"] == "quarantined"
    assert "rgb_missing" in bad["quality"]["errors"]
    after = next(row for row in _rows(recorder.run_dir, "observations.jsonl") if row["observation_id"] == bad["obs_after_id"])
    assert after["rgb"] is None
    assert after["rgb_missing"] is True


def test_ack_mismatch_and_missing_ack_are_hard_errors(tmp_path: Path):
    plan = [_plan_row("sceneA", "1", "go left", "sceneA:4")]

    class Rewrite:
        def __init__(self, inner):
            self.inner = inner

        def reset(self, seed=None):
            return self.inner.reset(seed=seed)

        def step(self, action):
            obs, reward, terminated, truncated, info = self.inner.step(action)
            info = dict(info)
            info["wm_execution"] = dict(info["wm_execution"])
            info["wm_execution"]["action_id"] = "not-the-request"
            return obs, reward, terminated, truncated, info

        def execution_record(self, action_id):
            return self.inner.execution_record(action_id)

    env, _fake = _attach([_namespace_episode("sceneA", "1", "go left")])
    recorder = RunRecorder(tmp_path / "mismatch")
    run_session(
        Rewrite(env),
        ScriptedPolicy([{"kind": "move"}]),
        recorder,
        plan,
        max_steps=1,
        requested_seed=7,
        collection_policy="scripted_test",
    )
    report = _finish(recorder, plan)
    assert any(item["code"] == "action_id_mismatch" for item in report["errors"])

    class Drop:
        def __init__(self, inner):
            self.inner = inner

        def reset(self, seed=None):
            return self.inner.reset(seed=seed)

        def step(self, action):
            obs, reward, terminated, truncated, info = self.inner.step(action)
            info = dict(info)
            info.pop("wm_execution", None)
            return obs, reward, terminated, truncated, info

    env, _fake = _attach([_namespace_episode("sceneA", "1", "go left")])
    recorder = RunRecorder(tmp_path / "drop")
    run_session(
        Drop(env),
        ScriptedPolicy([{"kind": "move"}]),
        recorder,
        plan,
        max_steps=1,
        requested_seed=7,
        collection_policy="scripted_test",
    )
    report = _finish(recorder, plan)
    assert any(item["code"] == "ack_not_executed" for item in report["errors"])


def test_duplicate_action_id_replays_success_and_refuses_error():
    env, fake = _attach([_namespace_episode("sceneA", "1", "go left")])
    env.reset(seed=7)
    action = {
        "action": "velocity_control",
        "action_args": {"linear_velocity": 0.6, "angular_velocity": 0.0},
        "wm_action_id": "same",
    }
    env.step(action)
    calls = fake.step_calls
    replay = env.step(action)
    assert fake.step_calls == calls
    assert replay[4]["wm_execution"]["replayed"] is True
    with pytest.raises(ValueError, match="wm_action_id"):
        env.step({"action": "velocity_control", "action_args": {"linear_velocity": 0.0, "angular_velocity": 0.0}})
    fake.fail = True
    bad = dict(action, wm_action_id="bad")
    with pytest.raises(RuntimeError, match="sensor boom"):
        env.step(bad)
    with pytest.raises(RuntimeError, match="refusing to repeat"):
        env.step(bad)
    assert fake.step_calls == calls + 1


def test_image_hash_run_conflict_and_seed_claim(tmp_path: Path):
    plan = [_plan_row("sceneA", "1", "go left", "sceneA:4")]
    env, _fake = _attach([_namespace_episode("sceneA", "1", "go left")])
    recorder = RunRecorder(tmp_path / "images")
    run_session(
        env,
        ScriptedPolicy([{"kind": "move"}]),
        recorder,
        plan,
        max_steps=1,
        requested_seed=7,
        collection_policy="scripted_test",
    )
    report = _finish(recorder, plan)
    assert report["hard_error_count"] == 0
    target = next((recorder.run_dir / "assets").glob("*_rgb.npy"))
    blob = bytearray(target.read_bytes())
    blob[-1] ^= 0xFF
    target.write_bytes(blob)
    broken = validate_run(recorder.run_dir)
    assert any(item["code"] == "image_hash_mismatch" for item in broken["errors"])
    with pytest.raises(FileExistsError):
        RunRecorder(recorder.run_dir)
    partial = RunRecorder(tmp_path / "partial")
    assert any(item["code"] == "run_not_closed" for item in validate_run(partial.run_dir)["errors"])
    assert partial.run_dir.exists()

    copied = tmp_path / "verified-claim"
    shutil.copytree(tmp_path / "images", copied)
    manifest = json.loads((copied / "manifest.json").read_text(encoding="utf-8"))
    manifest["seed"]["status"] = "verified"
    (copied / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    claimed = validate_run(copied)
    assert any(item["code"] == "seed_claimed_verified" for item in claimed["errors"])

    shell = tmp_path / "group-conflict"
    shell.mkdir()
    (shell / "assets").mkdir()
    for name in ("observations", "transitions", "privileged_labels"):
        (shell / f"{name}.jsonl").write_text("", encoding="utf-8")
    episodes = [
        {
            "episode_uid": f"uid-{partition}",
            "source_split": "train",
            "partition": partition,
            "route_group_id": "same-group",
            "reset": {"applied_in_reset": False},
        }
        for partition in ("train_fit", "train_dev")
    ]
    (shell / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes) + "\n", encoding="utf-8")
    (shell / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "source": {"split": "train"},
                "allowlist_keys": [["sceneA", "1"]],
                "served_keys": [["sceneA", "1"]],
                "reset_keys": [["sceneA", "1"]],
                "seed": {"status": "partial", "requested": 1},
            }
        ),
        encoding="utf-8",
    )
    conflict = validate_run(shell)
    assert any(item["code"] == "group_partition_conflict" for item in conflict["errors"])


def test_collector_does_not_change_the_eval_loop(tmp_path: Path):
    policy_source = Path("src/lightnav/habitat/policy.py").read_text(encoding="utf-8")
    vlnce_source = Path("habitat_server/lightnav_habitat/vlnce.py").read_text(encoding="utf-8")
    baseline_yaml = Path("habitat_server/configs/vlnce_r2r_go2w.yaml").read_text(encoding="utf-8")
    assert "wm_action_id" not in policy_source
    assert "wm_execution" not in vlnce_source
    assert "split: val_unseen" in baseline_yaml
    assert "split: train" in Path("experiments/world_model/configs/collect_train_go2w.yaml").read_text(
        encoding="utf-8"
    )

    class EvalEnv:
        def __init__(self) -> None:
            self.actions: list = []
            self.pending = 0

        def reset(self):
            self.pending = 0
            return self._obs(), self._info()

        def step(self, action):
            self.actions.append(action)
            self.pending += 1
            return self._obs(), 0.0, self.pending >= 3, False, self._info()

        def close(self):
            return None

        @staticmethod
        def _obs():
            return {"rgb": np.zeros((2, 2, 3), dtype=np.uint8), "instruction": {"text": "go"}}

        @staticmethod
        def _info():
            return {
                "episode_id": "1",
                "scene_id": "sceneA",
                "habitat_time_step": 0.1,
                "lin_vel_range": [0.0, 2.5],
                "ang_vel_range": [-300.0, 300.0],
                "distance_to_goal": 1.0,
                "success": False,
                "spl": 0.0,
                "ndtw": 0.0,
            }

    class EvalPolicy:
        def __init__(self) -> None:
            self.acts = 0
            self.stops = 0

        def reset(self, _obs):
            return None

        def act(self, _obs, _info):
            self.acts += 1
            return {
                "action": "velocity_control",
                "action_args": {"linear_velocity": 0.25, "angular_velocity": 0.0},
            }

        def stop_action(self):
            self.stops += 1
            return {
                "action": "velocity_control",
                "action_args": {"linear_velocity": -1.0, "angular_velocity": 0.0},
            }

        def get_info(self):
            return {"raw_text": "unused"}

    env = EvalEnv()
    policy = EvalPolicy()

    def run_once(destination: Path) -> None:
        cfg = HabitatEvalConfig(model_path="unused", episodes=1, max_steps=3, output_dir=str(destination))
        run_habitat_eval(
            cfg,
            env_factory=lambda _cfg: env,
            policy_factory=lambda _cfg, _engine, _bundle, _info: policy,
            engine_factory=lambda _cfg: (object(), object()),
        )

    run_once(tmp_path / "eval-a")
    first = list(env.actions)
    run_once(tmp_path / "eval-b")
    assert policy.acts == 4
    assert policy.stops == 2
    assert env.actions[0] == first[0]
    assert all("wm_action_id" not in action for action in env.actions)
    with pytest.raises(SystemExit, match="3.9"):
        serve_main([])
