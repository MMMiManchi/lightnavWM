import torch

from experiments.world_model.event_tssm import (
    EventTSSM,
    TSSMConfig,
    branch_discrimination_loss,
    gaussian_kl,
    paired_future_loss,
)


def test_exit_loss_penalizes_collapsed_and_swapped_predictions_and_trains_both_exits():
    target = torch.eye(4)[:2]
    pairs = torch.tensor([[0, 1]])
    correct = branch_discrimination_loss(target, target, pairs)
    swapped = branch_discrimination_loss(target.flip(0), target, pairs)
    copy = target[0].repeat(2, 1).clone().requires_grad_()
    collapsed = branch_discrimination_loss(copy, target, pairs)
    assert correct < collapsed < swapped
    collapsed.backward()
    assert copy.grad[0].norm() > 0 and copy.grad[1].norm() > 0
    # A single correct exit cannot make the second one's retrieval correct.
    assert branch_discrimination_loss(target, target, pairs) < branch_discrimination_loss(copy.detach(), target, pairs)


def test_current_exit_visual_can_change_a_forecast_with_history_and_action_fixed():
    m = EventTSSM(TSSMConfig(feature_dim=12, width=16, stochastic_dim=4, layers=1, heads=4,
                            use_exit_visual=True)).eval()
    torch.nn.init.normal_(m.exit_encoder[-1].weight, std=.2)
    torch.nn.init.normal_(m.decoder[-1].weight, std=.05)
    h = torch.nn.functional.normalize(torch.randn(1, 8, 12), dim=-1)
    s = m.initialize(h)
    action = torch.tensor([[0., 1., 0., 1.]])
    a = m.transition(s, action, exit_visual=torch.eye(12)[:1])[1]["feature"]
    b = m.transition(s, action, exit_visual=torch.eye(12)[1:2])[1]["feature"]
    assert not torch.allclose(a, b) and s.tokens.shape[1] == 0


def model(**kwargs):
    torch.manual_seed(4)
    m = EventTSSM(TSSMConfig(feature_dim=12, width=16, stochastic_dim=4, layers=1, heads=4, **kwargs)).eval()
    # Exercise state dynamics as well as the zero-initialized copy baseline.
    torch.nn.init.normal_(m.decoder[-1].weight, std=0.05)
    h = torch.nn.functional.normalize(torch.randn(2, 8, 12), dim=-1)
    return m, h


def test_posterior_updates_are_carried_into_subsequent_predictions():
    m, h = model()
    state = m.initialize(h)
    action = torch.tensor([[0., 1., 0., 1.]] * 2)
    prior, _ = m.transition(state, action)
    a, _ = m.observe(prior, h[:, -1])
    b, _ = m.observe(prior, -h[:, -1])
    # Hold the visible feature equal: test latent feedback, not the residual skip.
    b.current = a.current
    assert not torch.allclose(m.transition(a, action)[0].deter, m.transition(b, action)[0].deter)
    assert state.tokens.shape[1] == 0 and prior.tokens.shape[1] == 1


def test_branches_fork_from_same_state_without_advancing_actual_state():
    m, h = model()
    s = m.initialize(h)
    a = torch.tensor([[.5, .866, 0., 1.]] * 2)
    b = a.clone()
    b[:, 0] *= -1
    p = m.transition(s, a)[1]["feature"]
    q = m.transition(s, b)[1]["feature"]
    assert not torch.allclose(p, q)
    assert s.tokens.shape[1] == 0


def test_no_branch_control_is_invariant():
    m, h = model(use_branch=False)
    s = m.initialize(h)
    a = torch.tensor([[.5, .866, 0., 1.]] * 2)
    b = a.clone()
    b[:, 0] *= -1
    torch.testing.assert_close(m.transition(s, a)[1]["feature"], m.transition(s, b)[1]["feature"])


def test_multi_step_imagination_does_not_receive_future_observations():
    m, h = model()
    state, features = m.imagine(m.initialize(h), torch.tensor([[[0., 1., 1., 0.]] * 3] * 2))
    assert features.shape == (2, 3, 12) and state.tokens.shape[1] == 3


def test_residual_is_not_limited_to_the_old_twenty_degree_cap():
    m, h = model()
    with torch.no_grad():
        m.decoder[-1].weight.zero_()
        m.decoder[-1].bias.fill_(0)
        m.decoder[-1].bias[0] = 20
    p = m.transition(m.initialize(h), torch.tensor([[0., 1., 0., 1.]] * 2))[1]["feature"]
    assert torch.any((p * h[:, -1]).sum(-1) < 0.5)


def test_same_place_history_can_change_prediction_and_kl_has_gradients():
    m, h = model()
    direction = torch.tensor([[0., 1., 0., 1.]] * 2)
    s = m.initialize(h)
    raw = torch.cat([h[:, -1], -h[:, -1], direction[:, :2]], dim=-1)[:, None]
    with_memory, p = m.transition(s, direction, memory=raw, memory_valid=torch.ones(2, 1, dtype=torch.bool))
    _, without = m.transition(s, direction)
    assert not torch.allclose(p["feature"], without["feature"])
    _, q = m.observe(with_memory, -h[:, -1])
    loss = gaussian_kl(q, p).mean() + paired_future_loss(p["feature"], -h[:, -1], torch.tensor([[0, 1]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
