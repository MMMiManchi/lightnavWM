"""Small residual predictor. It does not train on collected runs.

``z_hat = z_t + f(z_t, u_t)``. ``u_t`` is the physical twist of the command
submitted to Habitat: ``(v_mps, w_rad_s, dt_command_s)``. The decoded waypoint
and the client request are not inputs. Pose is a second head. The two losses
are returned separately and are not added inside this module.

No language encoder is wired in. Instruction text stays on the data row until
a frozen text encoder is identified. Depth is not an input.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualLatentPredictor(nn.Module):
    def __init__(self, latent_dim: int, control_dim: int = 3, hidden: int = 64) -> None:
        super().__init__()
        if latent_dim < 1 or control_dim != 3:
            raise ValueError("v1 control is exactly (v_mps, w_rad_s, dt_command_s)")
        self.latent_dim = latent_dim
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + control_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.delta_latent = nn.Linear(hidden, latent_dim)
        self.delta_pose = nn.Linear(hidden, 3)
        nn.init.zeros_(self.delta_latent.weight)
        nn.init.zeros_(self.delta_latent.bias)
        nn.init.zeros_(self.delta_pose.weight)
        nn.init.zeros_(self.delta_pose.bias)

    def forward(self, latent_t: torch.Tensor, control: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if control.shape[-1] != 3:
            raise ValueError(f"control must have 3 columns, got {tuple(control.shape)}")
        if latent_t.shape[-1] != self.latent_dim:
            raise ValueError("latent width does not match this predictor")
        hidden = self.trunk(torch.cat([latent_t, control], dim=-1))
        return latent_t + self.delta_latent(hidden), self.delta_pose(hidden)


def separate_losses(
    latent_hat: torch.Tensor,
    latent_next: torch.Tensor,
    pose_hat: torch.Tensor,
    pose_true: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Scale-separated terms. Callers choose weights; this function does not."""
    if latent_hat.shape != latent_next.shape:
        raise ValueError("latent prediction and target shapes differ")
    if pose_hat.shape != pose_true.shape or pose_hat.shape[-1] != 3:
        raise ValueError("pose tensors must be (..., 3)")
    yaw_error = pose_hat[..., 2] - pose_true[..., 2]
    return {
        "latent_mse": torch.mean((latent_hat - latent_next) ** 2),
        "translation_huber": nn.functional.smooth_l1_loss(pose_hat[..., :2], pose_true[..., :2]),
        "yaw_cosine": torch.mean(1.0 - torch.cos(yaw_error)),
    }
