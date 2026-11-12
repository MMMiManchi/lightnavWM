"""CPU checks for the next-node place comparison. They do not encode images."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.world_model.event_place import (
    cosine,
    mean_normalized,
    other_end_ids,
    place_verdict,
    sweep_verdict,
)


def test_cosine_is_one_for_the_same_vector_and_rejects_a_zero_vector():
    vector = np.array([3.0, 4.0], dtype=np.float32)
    assert cosine(vector, vector) == pytest.approx(1.0)
    assert cosine(vector, -vector) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cosine(vector, np.zeros(2))


def test_other_arrival_stays_in_the_split_and_is_not_the_span_itself():
    spans = [
        {"split": "dev", "episode_uid": "a", "start_observation_id": "a0", "end_observation_id": "a1"},
        {"split": "dev", "episode_uid": "b", "start_observation_id": "b0", "end_observation_id": "b1"},
        {"split": "fit", "episode_uid": "c", "start_observation_id": "c0", "end_observation_id": "c1"},
        {"split": "fit", "episode_uid": "d", "start_observation_id": "d0", "end_observation_id": "d1"},
    ]
    paired = other_end_ids(spans)
    assert paired[0] == "b1"
    assert paired[1] == "a1"
    assert paired[2] == "d1"
    assert paired[3] == "c1"


def test_mean_normalized_averages_unit_vectors_and_rejects_cancellation():
    stacked = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    mean = mean_normalized(stacked)
    assert mean == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError):
        mean_normalized(np.array([[1.0, 0.0], [-1.0, 0.0]]))


def test_sweep_verdict_keeps_the_margin_fixed_at_one_tenth():
    accepted = sweep_verdict([0.2, 0.0], [0.0, 0.0])
    assert accepted["call"] == "turn_sweep_separates_the_next_node"
    assert accepted["accepted"] is True
    assert accepted["margin"] == pytest.approx(0.10)
    assert accepted["branch_choice_tested"] is False
    rejected = sweep_verdict([0.19, 0.0], [0.0, 0.0])
    assert rejected["call"] == "turn_sweep_does_not_separate_the_next_node"
    assert rejected["median_delta"] == pytest.approx(0.095)


def test_verdict_uses_the_sign_of_the_median_gap():
    closer = place_verdict([0.8, 0.7], [0.2, 0.1])
    assert closer["call"] == "arrival_closer_than_another_place"
    assert closer["branch_choice_tested"] is False
    far = place_verdict([0.2, 0.1], [0.8, 0.7])
    assert far["call"] == "arrival_as_far_as_another_place"
    tied = place_verdict([0.4, 0.6], [0.5, 0.5])
    assert tied["call"] == "arrival_as_far_as_another_place"
