"""Action-conditioned latent state transitions with observed posterior feedback."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TSSMConfig:
    feature_dim: int = 1152
    width: int = 128
    stochastic_dim: int = 32
    layers: int = 2
    heads: int = 4
    context_length: int = 16
    use_state: bool = True
    use_branch: bool = True
    use_exit_visual: bool = False


@dataclass
class TSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    current: torch.Tensor
    tokens: torch.Tensor
    anchor: torch.Tensor
    action: torch.Tensor

    def detach(self):
        return TSSMState(**{k: v.detach() for k, v in vars(self).items()})


class EventTSSM(nn.Module):
    def __init__(self, config=TSSMConfig()):
        super().__init__()
        self.config = config
        d, h, z = config.feature_dim, config.width, config.stochastic_dim
        self.observation = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU())
        self.initial_history = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(h, config.heads, h * 4, dropout=0.05,
                                       activation="gelu", batch_first=True, norm_first=True),
            1, enable_nested_tensor=False)
        self.initial_position = nn.Parameter(torch.randn(8, h) * 0.02)
        self.action_encoder = nn.Sequential(nn.Linear(4, h), nn.GELU(), nn.Linear(h, h))
        if config.use_exit_visual:
            self.exit_encoder = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Linear(h, h))
            nn.init.zeros_(self.exit_encoder[-1].weight)
            nn.init.zeros_(self.exit_encoder[-1].bias)
        self.state_action = nn.Linear(h + z + h, h)
        self.position = nn.Parameter(torch.randn(config.context_length, h) * 0.02)
        self.dynamics = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(h, config.heads, h * 4, dropout=0.05,
                                       activation="gelu", batch_first=True, norm_first=True),
            config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(h)
        self.prior = nn.Linear(h, z * 2)
        self.posterior = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, z * 2))
        self.memory_encoder = nn.Sequential(nn.Linear(d * 2 + 2, h), nn.GELU(), nn.Linear(h, h))
        self.decoder = nn.Sequential(nn.Linear(h + z + h, h * 2), nn.GELU(), nn.Linear(h * 2, d))
        # An unconstrained learned change around visible state, not a 20-degree
        # residual bound. Zero initialization starts at the copy-current baseline.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def configuration(self):
        return asdict(self.config)

    @staticmethod
    def distribution(values):
        mean, log_std = values.float().chunk(2, dim=-1)
        return {"mean": mean, "log_std": log_std.clamp(-5, 1)}

    @staticmethod
    def draw(stats, sample):
        return stats["mean"] + stats["log_std"].exp() * torch.randn_like(stats["mean"]) if sample else stats["mean"]

    def initialize(self, history, valid=None):
        if history.ndim != 3 or history.shape[1:] != (8, self.config.feature_dim):
            raise ValueError("initial history must contain eight feature slots")
        valid = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device) if valid is None else valid
        if not valid[:, -1].all():
            raise ValueError("current observation must be valid")
        encoded = self.observation(history)
        h = self.norm(self.initial_history(encoded + self.initial_position, src_key_padding_mask=~valid)[:, -1])
        current = history[:, -1]
        q = self.distribution(self.posterior(torch.cat([h, encoded[:, -1]], dim=-1)))
        return TSSMState(h, q["mean"], current, h.new_zeros(len(h), 0, self.config.width), current, torch.zeros_like(h))

    def memory_context(self, state, action, memory, memory_valid):
        h = state.deter
        if memory is None or not self.config.use_branch:
            return torch.zeros_like(h), state.current, h.new_zeros(len(h), dtype=torch.bool)
        d = self.config.feature_dim
        source, future, branch = memory[..., :d], memory[..., d:d * 2], memory[..., -2:]
        mask = memory_valid.bool() & (action[:, 3:4] > 0.5)
        ss = torch.sum(source * state.current[:, None], dim=-1)
        bs = torch.sum(branch * action[:, None, :2], dim=-1)
        scores = (ss + 0.25 * bs).masked_fill(~mask, -1e4)
        weights = scores.softmax(-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        context = torch.sum(self.memory_encoder(memory) * weights[..., None], dim=1)
        strong = mask & (ss >= 0.98) & (bs >= 0.98)
        sw = weights * strong
        sw = sw / sw.sum(-1, keepdim=True).clamp_min(1e-8)
        support = strong.any(-1)
        remembered = torch.sum(future * sw[..., None], dim=1)
        anchor = torch.where(support[:, None], F.normalize(remembered.float(), dim=-1), state.current)
        return context, anchor, support

    def decode(self, deter, stoch, action, anchor):
        delta = self.decoder(torch.cat([deter, stoch.to(deter.dtype), action], dim=-1)).float()
        return F.normalize(anchor.float() + delta, dim=-1)

    def transition(self, state, action, *, memory=None, memory_valid=None, sample=False, exit_visual=None):
        if action.shape != (len(state.deter), 4):
            raise ValueError("action is bearing sin/cos and declared observation/event option")
        if not self.config.use_branch:
            action = torch.cat([torch.zeros_like(action[:, :2]), action[:, 2:]], dim=-1)
        a = self.action_encoder(action)
        if self.config.use_exit_visual and exit_visual is not None:
            if exit_visual.shape != state.current.shape:
                raise ValueError("exit visual feature must match current feature dimensions")
            a = a + self.exit_encoder(exit_visual) * action[:, 3:4]
        h, z, previous = state.deter, state.stoch, state.tokens
        if not self.config.use_state:
            h, z, previous = self.observation(state.current), torch.zeros_like(z), previous[:, :0]
        context, anchor, supported = self.memory_context(state, action, memory, memory_valid)
        token = self.state_action(torch.cat([h, z.to(h.dtype), a], dim=-1)) + context
        tokens = torch.cat([previous, token[:, None]], dim=1)[:, -self.config.context_length:]
        length = tokens.shape[1]
        causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        next_h = self.norm(self.dynamics(tokens + self.position[:length], mask=causal)[:, -1])
        p = self.distribution(self.prior(next_h))
        next_z = self.draw(p, sample)
        feature = self.decode(next_h, next_z, a, anchor)
        return TSSMState(next_h, next_z, feature, tokens, anchor, a), {**p, "feature": feature, "memory_supported": supported}

    def observe(self, prior_state, actual_feature, *, sample=False):
        q = self.distribution(self.posterior(torch.cat([prior_state.deter, self.observation(actual_feature)], dim=-1)))
        z = self.draw(q, sample)
        reconstructed = self.decode(prior_state.deter, z, prior_state.action, prior_state.anchor)
        posterior = TSSMState(prior_state.deter, z, actual_feature, prior_state.tokens,
                              actual_feature, prior_state.action)
        return posterior, {**q, "feature": reconstructed}

    def imagine(self, state, actions, *, sample=False):
        if actions.ndim != 3 or actions.shape[1] < 1:
            raise ValueError("imagination needs a proposed action sequence")
        predictions = []
        for action in actions.unbind(1):
            state, prior = self.transition(state, action, sample=sample)
            predictions.append(prior["feature"])
        return state, torch.stack(predictions, dim=1)


def gaussian_kl(q, p):
    return (p["log_std"] - q["log_std"] +
            ((2 * q["log_std"]).exp() + (q["mean"] - p["mean"]).square()) /
            (2 * (2 * p["log_std"]).exp()) - 0.5).clamp_min(0.1).mean(-1)


def paired_future_loss(prediction, target, pairs, margin=0.05):
    if pairs.numel() == 0:
        return prediction.sum() * 0
    left, right = pairs.T
    gap = 0.5 * ((prediction[left] - prediction[right]) * (target[left] - target[right])).sum(-1)
    return F.relu(margin - gap).mean()


def branch_discrimination_loss(prediction, target, pairs, temperature=0.07):
    """Each exit must identify its own outcome, including both sides of a pair.

    Pair assignment alone can improve by moving just one prediction. Two
    row-wise retrieval losses prevent that shortcut; matching the observed
    difference also penalizes predictions which stay collapsed at the source.
    Pairs must share the *same* pre-decision history and have distinct outcomes.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if pairs.numel() == 0:
        return prediction.sum() * 0
    p, t = prediction[pairs], target[pairs]
    logits = torch.bmm(p, t.transpose(1, 2)) / temperature
    labels = torch.arange(2, device=p.device).expand(len(p), 2)
    retrieval = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
    difference = ((p[:, 0] - p[:, 1]) - (t[:, 0] - t[:, 1])).square().sum(-1).mean()
    return retrieval + difference


def sequence_objective(model, history, observations, valid, actions):
    state = model.initialize(history)
    total = history.new_zeros(len(history))
    counts = history.new_zeros(len(history))
    initial = state
    for t in range(observations.shape[1] - 1):
        active = valid[:, t + 1].float()
        if not bool(active.any()):
            break
        prior_state, p = model.transition(state, actions[:, t])
        state, q = model.observe(prior_state, observations[:, t + 1], sample=model.training)
        loss = 1 - (p["feature"] * observations[:, t + 1]).sum(-1)
        loss = loss + 0.5 * (1 - (q["feature"] * observations[:, t + 1]).sum(-1))
        loss = loss + 0.001 * gaussian_kl(q, p)
        total, counts = total + loss * active, counts + active
    sequential = (total / counts.clamp_min(1)).mean()
    overshooting = history.new_zeros(())
    terms = 0
    # Future observations do not enter these imagined trajectories.
    horizon = min(3, actions.shape[1])
    if horizon >= 2:
        _, imagined = model.imagine(initial, actions[:, :horizon])
        for t in range(1, horizon):
            active = valid[:, t + 1].float()
            if bool(active.any()):
                loss = 1 - (imagined[:, t] * observations[:, t + 1]).sum(-1)
                overshooting = overshooting + (loss * active).sum() / active.sum()
                terms += 1
    return sequential, overshooting / max(1, terms)
