"""Event-scale branch-conditioned latent dynamics, with a frozen feature target.

Only visual history and a selected relative direction are model inputs. The
posterior sees the observed future only during training; predict() uses prior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EventModelConfig:
    feature_dim: int = 1152
    history_length: int = 8
    width: int = 256
    layers: int = 4
    heads: int = 8
    stochastic_dim: int = 64
    use_branch: bool = True
    stochastic: bool = True


class EventLatentWorldModel(nn.Module):
    def __init__(self, config: EventModelConfig) -> None:
        super().__init__()
        self.config = config
        self.visual = nn.Linear(config.feature_dim, config.width)
        self.position = nn.Parameter(torch.randn(config.history_length, config.width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            config.width, config.heads, config.width * 4, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.history = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.width)
        self.branch = nn.Sequential(nn.Linear(2, config.width), nn.GELU(), nn.Linear(config.width, config.width))
        self.prior = nn.Linear(config.width * 2, config.stochastic_dim * 2)
        self.target_projection = nn.Linear(config.feature_dim, config.width)
        self.posterior = nn.Linear(config.width * 3, config.stochastic_dim * 2)
        latent_width = config.stochastic_dim if config.stochastic else 0
        self.decoder = nn.Sequential(
            nn.Linear(config.width * 2 + latent_width, config.width * 2),
            nn.GELU(), nn.Linear(config.width * 2, config.feature_dim),
        )

    def belief(self, history: torch.Tensor, valid: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        if history.shape[1:] != (self.config.history_length, self.config.feature_dim):
            raise ValueError("visual history shape does not match the model contract")
        if not valid[:, -1].all():
            raise ValueError("the current observation must be valid")
        tokens = self.visual(history) + self.position
        h = self.norm(self.history(tokens, src_key_padding_mask=~valid)[:, -1])
        b = self.branch(branch if self.config.use_branch else torch.zeros_like(branch))
        return torch.cat([h, b], dim=-1)

    def _distribution(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = values.float().chunk(2, dim=-1)
        return mean, log_std.clamp(-4.0, 2.0)

    def _decode(self, condition: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        parts = [condition] if latent is None else [condition, latent.to(condition.dtype)]
        return F.normalize(self.decoder(torch.cat(parts, dim=-1)).float(), dim=-1)

    def predict(self, history: torch.Tensor, valid: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        condition = self.belief(history, valid, branch)
        if not self.config.stochastic:
            return self._decode(condition)
        mean, _ = self._distribution(self.prior(condition))
        return self._decode(condition, mean)

    def imagine(self, history: torch.Tensor, valid: torch.Tensor, branch: torch.Tensor, samples: int = 4) -> torch.Tensor:
        condition = self.belief(history, valid, branch)
        if not self.config.stochastic:
            return self._decode(condition)[:, None]
        mean, log_std = self._distribution(self.prior(condition))
        outputs = [self._decode(condition, mean + log_std.exp() * torch.randn_like(mean)) for _ in range(samples)]
        return torch.stack(outputs, dim=1)

    def training_outputs(self, history, valid, branch, target) -> dict[str, torch.Tensor]:
        condition = self.belief(history, valid, branch)
        if not self.config.stochastic:
            prediction = self._decode(condition)
            return {"prior_prediction": prediction, "posterior_prediction": prediction, "kl": prediction.new_zeros(())}
        p_mean, p_log_std = self._distribution(self.prior(condition))
        target_token = self.target_projection(target)
        q_mean, q_log_std = self._distribution(self.posterior(torch.cat([condition, target_token], dim=-1)))
        sampled = q_mean + q_log_std.exp() * torch.randn_like(q_mean)
        kl_per_dimension = p_log_std - q_log_std + (
            (2 * q_log_std).exp() + (q_mean - p_mean).square()
        ) / (2 * (2 * p_log_std).exp()) - 0.5
        return {
            "prior_prediction": self._decode(condition, p_mean),
            "posterior_prediction": self._decode(condition, sampled),
            "kl": kl_per_dimension.clamp_min(0.1).mean(),
        }

    def configuration(self) -> dict:
        return asdict(self.config)


def multi_positive_contrastive(prediction, target, labels, temperature: float = 0.1):
    """Same real target place is a positive, including different views of it."""
    logits = prediction @ target.T / temperature
    positive = labels[:, None] == labels[None, :]
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)).mean()


def event_objective(outputs, target, labels, kl_weight: float = 1e-3):
    prior = (1 - (outputs["prior_prediction"] * target).sum(-1)).mean()
    posterior = (1 - (outputs["posterior_prediction"] * target).sum(-1)).mean()
    contrast = multi_positive_contrastive(outputs["prior_prediction"], target, labels)
    total = prior + posterior + 0.1 * contrast + kl_weight * outputs["kl"]
    return total, {"prior_cosine_loss": prior, "posterior_cosine_loss": posterior, "contrastive": contrast, "kl": outputs["kl"]}
