import numpy as np
import pytest

from experiments.world_model.branch_training_data import paired_indices, prediction_metrics


def rows():
    return [{"scene": "house", "origin_id": "fork", "exit_id": i} for i in range(2)]


def test_pair_swap_checks_actual_exit_binding_and_ties_are_chance():
    target = np.eye(2, dtype=np.float32)
    labels = np.array([0, 1])
    correct = prediction_metrics(target, target, rows(), labels)
    wrong = prediction_metrics(target[::-1], target, rows(), labels)
    identical = np.ones((2, 2), dtype=np.float32) / np.sqrt(2)
    tied = prediction_metrics(identical, target, rows(), labels)
    assert correct["pair_assignment_accuracy"] == 1
    assert wrong["pair_assignment_accuracy"] == 0
    assert tied["pair_assignment_accuracy"] == 0.5
    assert tied["pair_margin_mean"] == pytest.approx(0)


def test_same_place_aliases_are_not_forced_to_disagree():
    assert paired_indices(rows(), np.array([3, 3])).shape == (0, 2)


def test_unrelated_sources_are_not_an_exit_swap_control():
    r = rows()
    r[1]["origin_id"] = "another-fork"
    assert len(paired_indices(r, np.array([0, 1]))) == 0
