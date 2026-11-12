import numpy as np
import pytest

from experiments.world_model.familiarity import (
    budget_memory_indices,
    choose_live_scenes,
    prior_episode_rows,
    seed_budget,
)
from experiments.world_model.scene_memory import PersistentSceneMemory


def data():
    pairs = [{"source_run": "run", "episode_uid": f"ep{i}", "scene": scene,
              "partition": "train_dev", "origin_index": step}
             for scene, i, step in [("a", 0, 1), ("a", 0, 2), ("b", 1, 1),
                                    ("a", 2, 1), ("a", 3, 1)]]
    order = [("run", f"ep{i}") for i in range(4)]
    return pairs, order


def test_nested_prefixes_never_include_current_episode_other_scene_or_future():
    pairs, order = data()
    assert prior_episode_rows(pairs, order, 3, 0) == ([], 0)
    assert prior_episode_rows(pairs, order, 3, 1) == ([0, 1], 1)
    assert prior_episode_rows(pairs, order, 4, 2) == ([0, 1, 3], 2)
    assert prior_episode_rows(pairs, order, 3, None) == ([0, 1], 1)
    with pytest.raises(ValueError):
        prior_episode_rows(pairs, order, 3, -1)


def test_more_exposure_can_help_source_coverage_without_fixing_direction():
    pairs, order = data()
    history = np.array([[[1., 0]], [[1., 0]], [[0., 1]], [[0., 1]], [[0., 1]]], dtype=np.float32)
    branch = np.array([[0., 1], [0., 1], [0., 1], [0., -1], [0., 1]], dtype=np.float32)
    first = budget_memory_indices(pairs, order, history, branch, [4], 1)
    many = budget_memory_indices(pairs, order, history, branch, [4], 2)
    assert first[2][0] == 0 and many[2][0] == 1
    assert many[3][0] == 0 and (many[0] == -1).all()


def test_live_scenes_chosen_by_coverage_and_acquisition_order_not_model_result():
    pairs, order = data()
    selected = choose_live_scenes(pairs, order, 1)
    assert selected[0]["scene"] == "a" and selected[0]["query_index"] == 4
    assert selected[0]["available_prior_episodes"] == 2


def test_repeating_identical_real_experience_does_not_increase_knowledge(tmp_path):
    pairs, order = data()
    for i, pair in enumerate(pairs):
        pair.update(sample_id=f"event-{i}", history_observation_ids=[f"source-{i}"], target_observation_id=f"outcome-{i}")
    arrays = {"history": np.ones((5, 1, 2), dtype=np.float32),
              "target": np.tile([0., 1.], (5, 1)), "branch": np.tile([0., 1.], (5, 1))}
    bank = PersistentSceneMemory(tmp_path / "memory.sqlite", contract_id="test", feature_dim=2)
    prefix = seed_budget(bank, pairs, arrays, order, 4, 2)
    assert prefix["pair_ids"] == [0, 1, 3] and bank.count("a") == 3
    seed_budget(bank, pairs, arrays, order, 4, 2)
    assert bank.count("a") == 3
    assert {row[0] for row in bank.connection.execute("select target_observation from experiences")} == {
        "outcome-0", "outcome-1", "outcome-3"}
    bank.close()
