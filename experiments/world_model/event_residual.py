"""Residual event dynamics with causal, explicitly typed experience memory.

Targets stay in the audited, frozen SigLIP space. Patch columns are current
observations, not future observations or official graph features.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


def direction_read(columns: torch.Tensor, branch: torch.Tensor, *, use_branch: bool = True) -> torch.Tensor:
    """Read visible patch columns with the camera's pinhole projection (+left).

    A direction-free control reads the whole image and ignores the bearing.
    Columns are averaged vertically before this function, keeping left/right.
    """
    if not use_branch:
        return F.normalize(columns.float().mean(-2), dim=-1)
    angle = torch.atan2(branch[..., 0], branch[..., 1])
    if (angle.abs() >= math.pi / 3).any():
        raise ValueError("selected branch is outside the 120 degree camera")
    count = columns.shape[-2]
    center = (1 - torch.tan(angle) / math.tan(math.pi / 3)) * count / 2 - 0.5
    position = torch.arange(count, device=columns.device, dtype=torch.float32)
    weights = torch.softmax(-0.5 * ((position - center[..., None]) / 2.5).square(), dim=-1)
    return F.normalize(torch.einsum("...c,...cd->...d", weights, columns.float()), dim=-1)


@dataclass(frozen=True)
class ResidualEventConfig:
    feature_dim: int = 1152
    history_length: int = 8
    width: int = 128
    layers: int = 2
    heads: int = 4
    use_branch: bool = True
    use_memory: bool = False


class ResidualEventWorldModel(nn.Module):
    def __init__(self, config: ResidualEventConfig):
        super().__init__()
        self.config = config
        d, w = config.feature_dim, config.width
        self.visual = nn.Linear(d, w)
        self.position = nn.Parameter(torch.randn(config.history_length, w) * 0.02)
        layer = nn.TransformerEncoderLayer(w, config.heads, w * 2, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.history = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.spatial = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, w), nn.GELU())
        self.direction = nn.Sequential(nn.Linear(2, w), nn.GELU())
        self.condition = nn.Sequential(nn.LayerNorm(w * 3), nn.Linear(w * 3, w), nn.GELU())
        self.delta = nn.Linear(w, d)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.memory_query = nn.Linear(w, w, bias=False)
        self.memory_key = nn.Linear(d + 2, w, bias=False)
        self.memory_gate = nn.Sequential(nn.Linear(w + 2, w), nn.GELU(), nn.Linear(w, 1))
        nn.init.zeros_(self.memory_gate[-1].weight)
        nn.init.constant_(self.memory_gate[-1].bias, -2.0)

    def forward(self, history, valid, branch, columns, memory_source, memory_future,
                memory_branch, memory_valid, *, return_aux: bool = False):
        if history.shape[1:] != (self.config.history_length, self.config.feature_dim):
            raise ValueError("invalid visual history shape")
        if not valid[:, -1].all():
            raise ValueError("current observation must be valid")
        # Mask before projection too: invalid padding cannot affect attention.
        tokens = self.visual(history.masked_fill(~valid[..., None], 0)) + self.position
        temporal = self.history(tokens, src_key_padding_mask=~valid)[:, -1]
        bearing = branch if self.config.use_branch else torch.zeros_like(branch)
        spatial = self.spatial(direction_read(columns, branch, use_branch=self.config.use_branch))
        condition = self.condition(torch.cat([temporal, spatial, self.direction(bearing)], dim=-1))
        residual = self.delta(condition).float()
        current = history[:, -1].float()
        memory_weight = torch.zeros_like(current[:, :1])
        memory_delta = torch.zeros_like(current)
        if self.config.use_memory:
            sources, futures = memory_source.float(), memory_future.float()
            directions = memory_branch if self.config.use_branch else torch.zeros_like(memory_branch)
            keys = self.memory_key(torch.cat([sources.to(condition.dtype), directions.to(condition.dtype)], dim=-1))
            source_cos = torch.einsum("bd,bkd->bk", current, sources)
            branch_cos = torch.einsum("bd,bkd->bk", bearing.float(), directions.float())
            learned = torch.einsum("bd,bkd->bk", self.memory_query(condition), keys).float() / math.sqrt(self.config.width)
            logits = 10 * source_cos + 2 * branch_cos + learned
            # All-empty rows have zero attention, rather than softmax(-inf).
            weights = torch.softmax(logits.masked_fill(~memory_valid, -1e4), dim=-1) * memory_valid
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            observed = torch.einsum("bk,bkd->bd", weights, futures)
            similarity = torch.stack([(weights * source_cos).sum(-1), (weights * branch_cos).sum(-1)], dim=-1)
            memory_weight = torch.sigmoid(self.memory_gate(torch.cat([condition, similarity.to(condition.dtype)], dim=-1))).float()
            memory_weight = 0.75 * memory_weight * memory_valid.any(-1, keepdim=True)
            memory_delta = memory_weight * (observed - current)
        prediction = F.normalize(current + residual + memory_delta, dim=-1)
        if return_aux:
            return {"prediction": prediction, "residual": residual, "memory_weight": memory_weight, "condition": condition}
        return prediction

    def configuration(self):
        return asdict(self.config)


def residual_objective(outputs, target, labels, origin_labels, exit_labels):
    prediction = outputs["prediction"]
    logits = prediction @ target.T
    positives = labels[:, None] == labels[None, :]
    contrast = (torch.logsumexp(logits / 0.1, -1) -
                torch.logsumexp((logits / 0.1).masked_fill(~positives, -torch.inf), -1)).mean()
    different_exit = ((origin_labels[:, None] == origin_labels[None, :]) &
                      (exit_labels[:, None] != exit_labels[None, :]) & ~positives)
    margin = F.relu(logits - logits.diag()[:, None] + 0.05)
    hard = (margin * different_exit).sum() / different_exit.sum().clamp_min(1)
    cosine = (1 - (prediction * target).sum(-1)).mean()
    loss = cosine + 0.05 * contrast + 0.1 * hard + 0.01 * outputs["residual"].square().sum(-1).mean()
    return loss, {"cosine": cosine, "contrastive": contrast, "different_exit_margin": hard}
