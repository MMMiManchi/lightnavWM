import pytest

from experiments.world_model.viewpoint_branch_collection import (
    LOOK_DPS,
    execute,
    make_env,
    reset_singleton_episode,
    source_episode,
)


class FakeEnv:
    def __init__(self, replayed=False):
        self.replayed = replayed
        self.action_ids = []

    def step(self, action):
        self.action_ids.append(action["wm_action_id"])
        return {}, 0, False, False, {"wm_execution": {"executed": True, "replayed": self.replayed}}


def test_source_pan_returns_to_the_start_heading_without_fake_history():
    assert len(LOOK_DPS) == 8 and sum(LOOK_DPS) == 0


def test_each_source_observation_must_be_a_new_executed_action():
    env = FakeEnv()
    execute(env, {"action": "velocity_control"}, "candidate-a-look-0")
    execute(env, {"action": "velocity_control"}, "candidate-b-look-0")
    assert len(set(env.action_ids)) == 2
    with pytest.raises(RuntimeError, match="new executed-action"):
        execute(FakeEnv(replayed=True), {"action": "velocity_control"}, "duplicate")


def test_sealed_roles_fail_before_importing_or_initializing_a_simulator():
    with pytest.raises(ValueError, match="sealed"):
        make_env({"role": "sealed_test"}, 30)


def test_legacy_fit_labels_do_not_override_the_episode_route_gate():
    rows = [{"scene": "a", "episode_uid": "old-preferred", "episode_id": "1", "partition": "train_dev"},
            {"scene": "a", "episode_uid": "allowed", "episode_id": "2", "partition": "train_fit"}]
    assert source_episode(rows, "a", "fit", "old-preferred")["episode_uid"] == "allowed"


def test_repeated_sources_rewind_only_the_allowlisted_singleton():
    class Native:
        @property
        def episodes(self):
            return self._episodes

        @episodes.setter
        def episodes(self, values):
            self._episodes = values
            self.iterator = iter(values)

    class Wrapper:
        def __init__(self):
            self._habitat_env = Native()
            self._habitat_env.episodes = ["allowed-only"]

        def reset(self, seed):
            return next(self._habitat_env.iterator)

    env = Wrapper()
    assert reset_singleton_episode(env) == "allowed-only"
    assert reset_singleton_episode(env) == "allowed-only"
    env._habitat_env.episodes = ["allowed-only", "unapproved"]
    with pytest.raises(RuntimeError, match="exactly one"):
        reset_singleton_episode(env)
