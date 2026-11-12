import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "habitat_server"))

from lightnav_habitat.base import BaseHabitatEnv

from experiments.world_model.branch_curation import FloorPolygon
from experiments.world_model.event_collection_control import (
    EventProgress,
    ResourcePause,
    StepAccounting,
    execute_once,
)
from experiments.world_model.server_adapter import WorldModelVLNCEEnv


class Native:
    def step(self, _action):
        return {}


class Wrapper(BaseHabitatEnv):
    def _create_habitat_env(self):
        return Native()

    def _extract_task_obs(self, obs):
        return {}

    def _convert_action(self, action):
        return action

    def _compute_reward(self, obs, info):
        return 0

    def _check_terminated(self, obs, info, action):
        return False

    def _build_observation(self, obs):
        return obs


def test_wrapper_really_has_no_fixed_cap_in_event_mode():
    env = Wrapper("unused", max_steps=None)
    env.initialize()
    for _ in range(1001):
        assert not env.step({})[3]
    bounded = Wrapper("unused", max_steps=3)
    bounded.initialize()
    assert not bounded.step({})[3]
    assert not bounded.step({})[3]
    assert bounded.step({})[3]


def config():
    return SimpleNamespace(habitat=SimpleNamespace(seed=11,
        simulator=SimpleNamespace(habitat_sim_v0=SimpleNamespace(gpu_device_id=0)),
        dataset=SimpleNamespace(split="train"),
        environment=SimpleNamespace(max_episode_steps=500, max_episode_seconds=10000)))


def test_native_habitat_limit_is_also_disabled_explicitly():
    env = WorldModelVLNCEEnv(config_path="unused", split="train", event_driven=True)
    cfg = config()
    env._apply_dataset_overrides(cfg)
    assert env.max_steps is None
    assert cfg.habitat.environment.max_episode_steps == cfg.habitat.environment.max_episode_seconds == 0
    bounded = WorldModelVLNCEEnv(config_path="unused", split="train")
    cfg = config()
    bounded._apply_dataset_overrides(cfg)
    assert bounded.max_steps == 500 and cfg.habitat.environment.max_episode_steps == 500
    with pytest.raises(ValueError):
        WorldModelVLNCEEnv(config_path="unused", event_driven=True, max_steps=300)


def rect(rid, left, right, code):
    return FloorPolygon(rid, code, 0, ((left, -2), (right, -2), (right, 2), (left, 2)))


def test_long_corridor_continues_past_300_and_stairs_are_not_a_terminal_room():
    polys = [rect(1, -2, 1, "h"), rect(2, 1, 50, "h"), rect(3, 50, 60, "s"), rect(4, 60, 100, "b")]
    monitor = EventProgress({"position": [0, 0.1, 0]}, polys)
    for i in range(1, 610):
        result = monitor.push({"position": [i / 10, 0.1, 0]}, True)
        if result:
            assert i > 600 and result["kind"] == "next_room"
            break
    else:
        pytest.fail("room event missing")


def test_repeated_frames_never_masquerade_as_a_return():
    monitor = EventProgress({"position": [0, 0.1, 0]}, [rect(1, -5, 5, "h")])
    for _ in range(600):
        assert monitor.push({"position": [0, 0.1, 0]}, True) is None


def test_optional_resource_budget_pauses_instead_of_manufacturing_a_terminal_event(tmp_path):
    budget = StepAccounting(tmp_path)
    budget.initialize(limit=2)
    budget.reserve("branch")
    budget.reserve("branch")
    with pytest.raises(ResourcePause):
        budget.reserve("branch")
    unlimited = StepAccounting(tmp_path / "unlimited")
    unlimited.initialize()
    for _ in range(305):
        unlimited.reserve("long-branch")


def test_bounded_retry_image_cache_keeps_a_separate_execution_id_guard():
    class Env:
        def __init__(self):
            self._by_id = {}

        def step(self, action):
            self._by_id[action["wm_action_id"]] = {"status": "executed"}
            return {}, 0, False, False, {"wm_execution": {"executed": True, "replayed": False}}

    env = Env()
    for i in range(100):
        execute_once(env, {}, str(i))
    assert len(env._by_id) == 64
    with pytest.raises(RuntimeError, match="already executed"):
        execute_once(env, {}, "0")
