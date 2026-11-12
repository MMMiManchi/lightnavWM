"""Inference-only interface from frozen visual features to a future feature."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.world_model.event_latent import EventLatentWorldModel, EventModelConfig


class EventFeaturePredictor(nn.Module):
    """Inputs are raw features of past/current RGB and a selected sin/cos direction."""

    def __init__(self, model: EventLatentWorldModel, mean: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("feature_mean", mean.float())
        self.register_buffer("feature_scale", scale.float())
        if mean.shape != (model.config.feature_dim,) or scale.shape != mean.shape or not (scale > 0).all():
            raise ValueError("invalid frozen feature normalization")

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize((features.float()-self.feature_mean)/self.feature_scale, dim=-1)

    def forward(self, raw_history: torch.Tensor, valid: torch.Tensor, branch_direction: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize(raw_history).masked_fill(~valid[..., None], 0.)
        return self.model.predict(normalized, valid, branch_direction)


def load_event_predictor(bundle_path: Path, device: str = "cpu") -> EventFeaturePredictor:
    """Load the exported weights and fit-only normalization, never future labels."""
    bundle = torch.load(bundle_path, map_location=device, weights_only=True)
    model = EventLatentWorldModel(EventModelConfig(**bundle["model_config"]))
    model.load_state_dict(bundle["model_state_dict"])
    predictor = EventFeaturePredictor(model, bundle["feature_mean"], bundle["feature_scale"])
    return predictor.to(device).eval()
