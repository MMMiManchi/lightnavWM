"""CPU checks for the residual-predictability gates. They do not encode or train."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.world_model.future_only import GRID
from experiments.world_model.residual_predictability import (
    beats_zero,
    linear_per_sample_mse,
    neighbor_correlation,
    predictability_verdict,
    readout_weight,
    scene_macro,
    scene_macro_rows,
    subspace_per_sample_mse,
    top_components,
    top_energy_share,
)


def test_energy_share_separates_a_spike_from_a_uniform_map():
    uniform = top_energy_share(np.ones(10))
    assert uniform["count"] == 1
    assert uniform["share"] == pytest.approx(0.1)
    spike = np.zeros(10)
    spike[-1] = 5.0
    assert top_energy_share(spike)["share"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-negative"):
        top_energy_share(np.array([1.0, -1.0]))


def test_neighbor_correlation_reads_a_ramp_and_a_checkerboard():
    ramp = np.arange(GRID, dtype=np.float64)[:, None] * np.ones(GRID)
    assert neighbor_correlation(ramp) > 0.9
    checkerboard = np.indices((GRID, GRID)).sum(axis=0) % 2
    assert neighbor_correlation(checkerboard) == pytest.approx(-1.0)
    assert neighbor_correlation(np.ones((GRID, GRID))) is None


def test_scene_macro_weights_scenes_once():
    assert scene_macro(np.array([0.0, 2.0, 4.0]), ["a", "a", "b"]) == pytest.approx(2.5)
    rows = scene_macro_rows(np.array([[0.0, 2.0], [2.0, 0.0], [4.0, 4.0]]), ["a", "a", "b"])
    assert rows == pytest.approx([2.5, 2.5])
    with pytest.raises(ValueError, match="scene macro"):
        scene_macro(np.array([1.0, 2.0]), ["only"])


def test_rank_one_subspace_and_linear_readout_match_direct_errors():
    torch.manual_seed(0)
    direction = torch.nn.functional.normalize(torch.randn(8), dim=0)
    signal = torch.tensor([3.0, -2.0, 4.0, -1.0])[:, None] * direction
    mean, components = top_components(signal, 1)
    subspace = subspace_per_sample_mse(signal, mean, components)
    assert torch.all(subspace < 1e-5)

    current = torch.randn(40, 6)
    residual = current[:, :1] * torch.tensor([0.2, -0.1, 0.4, 0.0, 0.3, -0.2])
    current_mean, current_components = top_components(current, 6)
    weight = readout_weight(current, residual, current_mean, current_components, ridge=0.0)
    recorded = linear_per_sample_mse(current, residual, current_mean, current_components, weight)
    scores = (current - current_mean) @ current_components.T
    design = torch.cat([scores, torch.ones(scores.shape[0], 1)], dim=1)
    direct = (residual - design @ weight).square().mean(dim=1)
    assert torch.allclose(recorded, direct, rtol=1e-4, atol=1e-5)
    assert float(recorded.mean()) < 1e-5


def _verdict(**overrides):
    values = {
        "zero_mse": 1.0,
        "mean_mse": 1.0,
        "subspace_mse": 1.0,
        "linear_mse": 1.0,
        "patch_top10_share": 0.10,
        "channel_top10_share": 0.10,
        "turn_neighbor_correlation": 0.1,
        "turn_energy": 1.0,
        "speed_energy": 1.0,
    }
    values.update(overrides)
    return predictability_verdict(**values)


def test_predictability_gates_follow_the_pre_registered_order():
    assert beats_zero(1.0, 0.995) is False
    assert beats_zero(1.0, 0.98) is True
    assert _verdict()["call"] == "no_stable_signal"
    geometric = _verdict(
        linear_mse=0.98,
        patch_top10_share=0.30,
        turn_neighbor_correlation=0.80,
        turn_energy=1.2,
        speed_energy=1.0,
    )
    assert geometric["call"] == "geometric_target_candidate"
    assert geometric["training_started"] is False
    assert geometric["generalization"] is False
    appearance = _verdict(
        subspace_mse=0.98,
        turn_neighbor_correlation=0.80,
        turn_energy=1.0,
        speed_energy=1.0,
    )
    assert appearance["call"] == "appearance_not_command"
    partial = _verdict(linear_mse=0.98, turn_neighbor_correlation=0.1, turn_energy=2.0)
    assert partial["call"] == "partial_linear_signal"
    direction = _verdict(mean_mse=0.98, turn_neighbor_correlation=0.1)
    assert direction["call"] == "stable_residual_direction"
    with pytest.raises(ValueError, match="speed energy"):
        _verdict(speed_energy=0.0)
