"""Experience-conditioned dynamics anchored to the equal-history baseline.

Without real experience this model retains current state. This is an explicit
abstention, not a claim to know the connectivity of never-observed rooms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from experiments.world_model.event_residual import (
    ResidualEventConfig,
    ResidualEventWorldModel,
    direction_read,
)


@dataclass(frozen=True)
class AnchoredEventConfig(ResidualEventConfig):
    residual_scale: float = 0.02


class AnchoredEventWorldModel(ResidualEventWorldModel):
    def __init__(self, config: AnchoredEventConfig):
        super().__init__(config)
        self.attention_gain = nn.Parameter(torch.zeros(()))
        self.transport_gain = nn.Parameter(torch.zeros(()))
        nn.init.constant_(self.memory_gate[-1].bias, math.log(3))

    def forward(self, history, valid, branch, columns, memory_source, memory_future,
                memory_branch, memory_valid, *, return_aux=False):
        if history.shape[1:] != (self.config.history_length, self.config.feature_dim):
            raise ValueError("invalid visual history shape")
        if not valid[:, -1].all():
            raise ValueError("current observation must be valid")
        tokens = self.visual(history.masked_fill(~valid[..., None], 0)) + self.position
        temporal = self.history(tokens, src_key_padding_mask=~valid)[:, -1]
        bearing = branch if self.config.use_branch else torch.zeros_like(branch)
        spatial = self.spatial(direction_read(columns, branch, use_branch=self.config.use_branch))
        condition = self.condition(torch.cat([temporal, spatial, self.direction(bearing)], -1))
        current = history[:, -1].float()
        sources, futures = memory_source.float(), memory_future.float()
        directions = memory_branch if self.config.use_branch else torch.zeros_like(memory_branch)
        source_cos = torch.einsum("bd,bkd->bk", current, sources)
        branch_cos = torch.einsum("bd,bkd->bk", bearing.float(), directions.float())
        scores = 0.8 * source_cos + 0.2 * branch_cos
        masked = scores.masked_fill(~memory_valid, -1e4)
        best = masked.argmax(-1)
        row = torch.arange(len(current), device=current.device)
        confidence = ((source_cos.masked_fill(~memory_valid, 0).amax(-1) - 0.5) / 0.5).clamp(0, 1)
        if self.config.use_branch:
            floor = math.cos(math.radians(35))
            confidence = confidence * ((branch_cos[row, best] - floor) / (1 - floor)).clamp(0, 1)
        keys = self.memory_key(torch.cat([sources.to(condition.dtype), directions.to(condition.dtype)], -1))
        adjustment = torch.einsum("bd,bkd->bk", self.memory_query(condition), keys) / math.sqrt(self.config.width)
        logits = scores / 0.07 + self.attention_gain.tanh() * adjustment.float()
        weights = torch.softmax(logits.masked_fill(~memory_valid, -1e4), -1) * memory_valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        observed = torch.einsum("bk,bkd->bd", weights, futures)
        remembered_source = torch.einsum("bk,bkd->bd", weights, sources)
        similarity = torch.stack([(weights * source_cos).sum(-1), (weights * branch_cos).sum(-1)], -1)
        gate = torch.sigmoid(self.memory_gate(torch.cat([condition, similarity.to(condition.dtype)], -1))).float()
        has_memory = memory_valid.any(-1, keepdim=True) & self.config.use_memory
        weight = gate * confidence[:, None] * has_memory
        residual = self.config.residual_scale * self.delta(condition).float().tanh()
        if self.config.residual_scale:
            residual = residual + 0.25 * self.transport_gain.tanh() * (current - remembered_source)
        residual = residual * has_memory
        prediction = F.normalize(current + weight * (observed - current) + residual, dim=-1)
        if return_aux:
            return {"prediction": prediction, "residual": residual, "memory_weight": weight, "condition": condition}
        return prediction
