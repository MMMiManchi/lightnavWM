"""Residual initialization, spatial direction and causal memory isolation."""

import numpy as np
import pytest
import torch

from experiments.world_model.event_anchored import AnchoredEventConfig, AnchoredEventWorldModel
from experiments.world_model.event_residual import (
    ResidualEventConfig,
    ResidualEventWorldModel,
    direction_read,
)
from experiments.world_model.event_residual_data import (
    causal_memory_indices,
    direct_memory_prediction,
    materialize_memory,
)


def inputs():
    torch.manual_seed(4)
    return {"history": torch.nn.functional.normalize(torch.randn(2, 4, 12), dim=-1),
            "valid": torch.ones(2, 4, dtype=torch.bool), "branch": torch.tensor([[0.5, 0.866], [-0.5, 0.866]]),
            "columns": torch.randn(2, 27, 12), "memory_source": torch.randn(2, 3, 12),
            "memory_future": torch.randn(2, 3, 12), "memory_branch": torch.zeros(2, 3, 2),
            "memory_valid": torch.zeros(2, 3, dtype=torch.bool)}


@pytest.mark.parametrize("use_memory", [False, True])
def test_initial_prediction_preserves_current_and_empty_memory_stays_finite(use_memory):
    x = inputs()
    m = ResidualEventWorldModel(ResidualEventConfig(feature_dim=12, history_length=4, width=16, layers=1,
                                                   use_memory=use_memory)).eval()
    actual = m(**x)
    assert torch.allclose(actual, x["history"][:, -1], atol=1e-6)
    assert torch.isfinite(actual).all()


def test_direction_reads_positive_left_columns_and_control_is_invariant():
    columns = torch.zeros(2, 27, 2)
    columns[:, :13, 0] = 1
    columns[:, 13:, 1] = 1
    branch = torch.tensor([[0.707, 0.707], [-0.707, 0.707]])
    read = direction_read(columns, branch)
    assert read[0, 0] > read[0, 1]
    assert read[1, 1] > read[1, 0]
    assert torch.equal(direction_read(columns, branch, use_branch=False),
                       direction_read(columns, branch.flip(0), use_branch=False))
    with pytest.raises(ValueError, match="outside"):
        direction_read(columns, torch.tensor([[1., 0.], [1., 0.]]))


def test_direction_free_model_ignores_branch_even_after_learning_and_with_memory():
    x = inputs()
    x["memory_valid"][:] = True
    m = ResidualEventWorldModel(ResidualEventConfig(feature_dim=12, history_length=4, width=16, layers=1,
                                                   use_branch=False, use_memory=True)).eval()
    torch.nn.init.normal_(m.delta.weight, std=0.1)
    first = m(**x)
    x["branch"] = x["branch"].flip(0)
    x["memory_branch"] = torch.randn_like(x["memory_branch"])
    assert torch.equal(first, m(**x))


def test_prior_episode_memory_never_reads_current_future_or_another_scene():
    pairs = [{"source_run": "run", "episode_uid": ep, "scene": scene, "origin_index": i}
             for i, (ep, scene) in enumerate([("one", "A"), ("two", "A"), ("one", "A"), ("other", "B")])]
    h = np.array([[[1., 0.]]] * 4, dtype=np.float32)
    b = np.array([[0., 1.]] * 4, dtype=np.float32)
    order = [("run", "one"), ("run", "other"), ("run", "two")]
    selected = causal_memory_indices(pairs, h, b, order)
    assert (selected[[0, 2, 3]] == -1).all()
    assert set(selected[1][selected[1] >= 0]) == {0, 2}
    changed = [dict(p) for p in pairs]
    # Offline labels are irrelevant to memory lookup.
    for p in changed:
        p["offline_labels"] = {"target_place": "invented"}
    assert np.array_equal(selected, causal_memory_indices(changed, h, b, order))


def test_direct_history_baseline_uses_same_facts_without_predictor():
    h = np.array([[[1., 0.]], [[1., 0.]]], dtype=np.float32)
    t = np.array([[0., 1.], [-1., 0.]], dtype=np.float32)
    b = np.array([[0., 1.], [0., 1.]], dtype=np.float32)
    context = materialize_memory(np.array([[-1, -1], [0, -1]]), h, t, b)
    out = direct_memory_prediction(h[:, -1], b, context)
    assert np.array_equal(out[0], h[0, -1])
    assert out[1, 1] > out[1, 0]


def test_anchored_initialization_exactly_matches_equal_history_control():
    x = inputs()
    x["memory_source"] = x["history"][:, -1, None].repeat(1, 3, 1)
    x["memory_future"] = torch.nn.functional.normalize(x["memory_future"], dim=-1)
    x["memory_branch"] = x["branch"][:, None].repeat(1, 3, 1)
    x["memory_valid"][:] = True
    m = AnchoredEventWorldModel(AnchoredEventConfig(feature_dim=12, history_length=4, width=16, layers=1,
                                                   use_memory=True)).eval()
    actual = m(**x)
    expected = direct_memory_prediction(x["history"][:, -1].numpy(), x["branch"].numpy(),
                                        {k: v.numpy() for k, v in x.items() if k.startswith("memory_")})
    assert np.allclose(actual.detach().numpy(), expected, atol=1e-6)
    x["memory_valid"][:] = False
    torch.nn.init.normal_(m.delta.weight)
    # An unfamiliar branch explicitly abstains even after parameter updates.
    assert torch.allclose(m(**x), x["history"][:, -1], atol=1e-6)
