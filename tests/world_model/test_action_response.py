"""CPU checks for the fixed-observation action-swap metrics."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.world_model.action_response import (
    SPREAD_RATIO_RESPONSIVE,
    action_response_verdict,
    difference_directions,
    pairwise_l2,
)


def test_pairwise_l2_is_zero_for_identical_actions_and_exact_for_a_unit_shift():
    same = np.ones((3, 2, 2), dtype=np.float32)
    assert pairwise_l2(same) == 0.0
    pair = np.zeros((2, 2, 2), dtype=np.float32)
    pair[1] = 1.0
    assert pairwise_l2(pair) == pytest.approx(2.0)


def test_direction_is_undefined_when_the_prediction_does_not_move():
    true = np.zeros((2, 2, 2), dtype=np.float32)
    true[1] = 2.0
    predicted = np.zeros_like(true)
    result = difference_directions(predicted, true)
    assert result["comparable_pairs"] == 1
    assert result["responsive_pairs"] == 0
    assert result["direction_cosine_mean"] is None
    matched = difference_directions(true, true)
    assert matched["direction_cosine_mean"] == pytest.approx(1.0)
    assert pairwise_l2(true) / pairwise_l2(true) == pytest.approx(1.0)


def test_verdict_separates_sensitivity_from_direction_and_rejects_a_moving_no_action_model():
    assert SPREAD_RATIO_RESPONSIVE == 0.05
    seeds = (20261029, 20261030, 20261031)
    quiet = [{"seed": seed, "spread_ratio_median": 0.0, "direction_cosine_mean": None} for seed in seeds]
    still = [{"seed": seed, "spread_ratio_median": 0.001} for seed in seeds]
    verdict = action_response_verdict(quiet, still)
    assert verdict["probe_valid"] is True
    assert verdict["response_call"] == "insensitive"
    assert verdict["direction_call"] == "undefined"
    assert verdict["generalization"] is False
    assert verdict["habitat_started"] is False
    moving = [
        {"seed": seed, "spread_ratio_median": 0.2, "direction_cosine_mean": 0.4} for seed in seeds
    ]
    aligned = action_response_verdict(moving, still)
    assert aligned["response_call"] == "responsive"
    assert aligned["direction_call"] == "same_direction"
    opposite = [
        {"seed": seed, "spread_ratio_median": 0.2, "direction_cosine_mean": -0.1} for seed in seeds
    ]
    assert action_response_verdict(opposite, still)["direction_call"] == "not_aligned"
    leaked = [{"seed": seed, "spread_ratio_median": 0.02} for seed in seeds]
    assert action_response_verdict(quiet, leaked)["probe_valid"] is False
    with pytest.raises(ValueError, match="non-finite"):
        action_response_verdict(
            [{"seed": seeds[0], "spread_ratio_median": float("nan")}],
            still,
        )
