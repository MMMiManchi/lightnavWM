"""Read-only checks for whether a patch residual is predictable.

The residual subspace is not a predictor: its coefficients are projections of
the residual itself. The linear readout is the predictor. It uses principal
components of the current patch grid and an intercept, estimated on fit only.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from experiments.world_model.future_only import GRID

RELATIVE_BAND = 0.01
COMPONENT_COUNT = 16
DESCRIPTIVE_COMPONENT_COUNTS = (1, 4, 16, 64)
TOP_FRACTION = 0.10
CONCENTRATION_SHARE = 0.25
COHERENCE_CORRELATION = 0.5
COMMAND_ENERGY_MARGIN = 0.10
READOUT_RIDGE = 1e-4
TURN_PAIRS = (
    ("creep_left", "creep_right"),
    ("slow_left", "slow_right"),
    ("nominal_left", "nominal_right"),
)
SPEED_PAIRS = (
    ("slow_straight", "nominal_straight"),
    ("creep_left", "slow_left"),
    ("slow_left", "nominal_left"),
    ("creep_left", "nominal_left"),
    ("creep_right", "slow_right"),
    ("slow_right", "nominal_right"),
    ("creep_right", "nominal_right"),
)


def beats_zero(zero_mse: float, predictor_mse: float, band: float = RELATIVE_BAND) -> bool:
    """True when the predictor is lower than zero by more than the band."""
    if not math.isfinite(zero_mse) or zero_mse <= 0.0:
        raise ValueError("zero residual MSE must be finite and positive")
    if not math.isfinite(predictor_mse) or predictor_mse < 0.0:
        raise ValueError("predictor MSE must be finite and non-negative")
    return (zero_mse - predictor_mse) / zero_mse > band


def relative_reduction(zero_mse: float, predictor_mse: float) -> float:
    if not math.isfinite(zero_mse) or zero_mse <= 0.0 or not math.isfinite(predictor_mse):
        raise ValueError("relative reduction needs a positive zero MSE and a finite predictor")
    return (zero_mse - predictor_mse) / zero_mse


def scene_macro_rows(values: np.ndarray, scenes: list[str]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != len(scenes) or len(scenes) == 0:
        raise ValueError("scene macro rows need one finite row per scene label")
    if not np.isfinite(array).all():
        raise ValueError("scene macro rows need one finite row per scene label")
    rows = []
    for name in sorted(set(scenes)):
        mask = np.asarray([scene == name for scene in scenes])
        rows.append(array[mask].mean(axis=0))
    return np.mean(rows, axis=0)


def scene_macro(values: np.ndarray, scenes: list[str]) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (len(scenes),) or not np.isfinite(array).all() or len(scenes) == 0:
        raise ValueError("scene macro needs one finite value per scene label")
    scores = []
    for name in sorted(set(scenes)):
        mask = np.asarray([scene == name for scene in scenes])
        scores.append(float(array[mask].mean()))
    return float(np.mean(scores))


def top_energy_share(energy: np.ndarray, fraction: float = TOP_FRACTION) -> dict:
    flat = np.asarray(energy, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all() or np.any(flat < 0.0):
        raise ValueError("energy must be finite and non-negative")
    total = float(flat.sum())
    if total <= 0.0 or not 0.0 < fraction < 1.0:
        raise ValueError("energy share needs a positive total and a fraction inside (0, 1)")
    count = int(math.floor(flat.size * fraction))
    if count < 1:
        raise ValueError("energy vector is too short for the requested fraction")
    ordered = np.sort(flat)[::-1]
    return {
        "count": count,
        "size": int(flat.size),
        "fraction": fraction,
        "share": float(ordered[:count].sum() / total),
    }


def neighbor_correlation(grid: np.ndarray) -> float | None:
    """Pearson correlation between each cell and the mean of its 4-neighbors."""
    array = np.asarray(grid, dtype=np.float64)
    if array.shape != (GRID, GRID) or not np.isfinite(array).all():
        raise ValueError(f"neighbor correlation expects a finite {GRID} by {GRID} map")
    values = np.empty(GRID * GRID, dtype=np.float64)
    neighbors = np.empty(GRID * GRID, dtype=np.float64)
    cursor = 0
    for row in range(GRID):
        for col in range(GRID):
            adjacent = []
            for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_row = row + delta_row
                next_col = col + delta_col
                if 0 <= next_row < GRID and 0 <= next_col < GRID:
                    adjacent.append(array[next_row, next_col])
            values[cursor] = array[row, col]
            neighbors[cursor] = float(np.mean(adjacent))
            cursor += 1
    if float(values.std()) == 0.0 or float(neighbors.std()) == 0.0:
        return None
    return float(np.corrcoef(values, neighbors)[0, 1])


def top_components(
    samples: torch.Tensor, count: int, relative_floor: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Largest principal directions of centered samples.

    Directions below ``relative_floor`` times the largest eigenvalue are omitted,
    so a returned basis can be shorter than ``count``.
    """
    if samples.ndim != 2 or samples.shape[0] < 2 or count < 1 or count >= samples.shape[0]:
        raise ValueError("component count must be positive and smaller than the sample count")
    if not math.isfinite(relative_floor) or relative_floor < 0.0:
        raise ValueError("relative floor must be finite and non-negative")
    if not torch.isfinite(samples).all():
        raise ValueError("non-finite samples")
    mean = samples.mean(dim=0)
    centered = samples - mean
    sample_count, width = centered.shape
    gram = torch.empty(sample_count, sample_count, dtype=torch.float64, device=samples.device)
    step = 256
    for start in range(0, sample_count, step):
        left = centered[start : start + step].double()
        blocks = []
        for other in range(0, sample_count, step):
            right = centered[other : other + step].double()
            blocks.append(left @ right.T)
        gram[start : start + left.shape[0]] = torch.cat(blocks, dim=1)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    descending = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[descending]
    eigenvectors = eigenvectors[:, descending]
    largest = float(eigenvalues[0])
    if largest <= 0.0:
        raise RuntimeError("samples have no variance")
    keep = int(torch.sum((eigenvalues > 0.0) & (eigenvalues >= relative_floor * largest)).item())
    keep = min(count, keep)
    if keep < 1:
        raise RuntimeError("no principal component passed the variance floor")
    chosen = eigenvalues[:keep]
    scaled = eigenvectors[:, :keep] / chosen.sqrt()
    directions = torch.empty(keep, width, dtype=torch.float32, device=samples.device)
    column_step = 65536
    for start in range(0, width, column_step):
        block = centered[:, start : start + column_step].double()
        directions[:, start : start + block.shape[1]] = (block.T @ scaled).T.float()
    norms = directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return mean, (directions / norms).contiguous()


def subspace_per_sample_mse(
    targets: torch.Tensor, mean: torch.Tensor, components: torch.Tensor
) -> torch.Tensor:
    """MSE of adding the target's own projection back onto the fit subspace."""
    centered = targets - mean
    coefficients = centered @ components.T
    captured = coefficients.square().sum(dim=1)
    total = centered.square().sum(dim=1)
    if torch.any(captured > total * (1.0 + 1e-3) + 1e-2):
        raise RuntimeError("subspace components are not orthonormal")
    return (total - captured).clamp_min(0) / targets.shape[-1]


def readout_weight(
    current: torch.Tensor,
    residual: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
    ridge: float = READOUT_RIDGE,
) -> torch.Tensor:
    """Map current-frame scores plus an intercept to the residual. Fit rows only."""
    scores = (current - mean) @ components.T
    ones = torch.ones(scores.shape[0], 1, dtype=scores.dtype, device=scores.device)
    design = torch.cat([scores, ones], dim=1)
    gram = design.T @ design
    scale = torch.trace(gram) / design.shape[1]
    penalty = ridge * scale
    system = gram + penalty * torch.eye(design.shape[1], dtype=gram.dtype, device=gram.device)
    cross = design.T @ residual
    return torch.linalg.solve(system, cross)


def linear_per_sample_mse(
    current: torch.Tensor,
    residual: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    scores = (current - mean) @ components.T
    ones = torch.ones(scores.shape[0], 1, dtype=scores.dtype, device=scores.device)
    design = torch.cat([scores, ones], dim=1)
    observed = residual.square().sum(dim=1)
    alignment = (design * (residual @ weight.T)).sum(dim=1)
    predicted = (design @ (weight @ weight.T) * design).sum(dim=1)
    return (observed - 2.0 * alignment + predicted).clamp_min(0) / residual.shape[-1]


def predictability_verdict(
    *,
    zero_mse: float,
    mean_mse: float,
    subspace_mse: float,
    linear_mse: float,
    patch_top10_share: float,
    channel_top10_share: float,
    turn_neighbor_correlation: float | None,
    turn_energy: float,
    speed_energy: float,
) -> dict:
    """Apply the diagnostic gates. Development numbers only; the 48 states are not a test."""
    mean_beats = beats_zero(zero_mse, mean_mse)
    subspace_beats = beats_zero(zero_mse, subspace_mse)
    linear_beats = beats_zero(zero_mse, linear_mse)
    for share in (patch_top10_share, channel_top10_share):
        if not math.isfinite(share) or not 0.0 <= share <= 1.0:
            raise ValueError("energy share must be finite and inside [0, 1]")
    patch_concentrated = patch_top10_share >= CONCENTRATION_SHARE
    channel_concentrated = channel_top10_share >= CONCENTRATION_SHARE
    if turn_neighbor_correlation is not None and not math.isfinite(turn_neighbor_correlation):
        raise ValueError("turn neighbor correlation is non-finite")
    turn_coherent = (
        turn_neighbor_correlation is not None and turn_neighbor_correlation >= COHERENCE_CORRELATION
    )
    if not math.isfinite(turn_energy) or not math.isfinite(speed_energy) or speed_energy <= 0.0:
        raise ValueError("turn and speed energies must be finite and speed energy must be positive")
    if turn_energy < 0.0:
        raise ValueError("turn energy must be non-negative")
    turn_specific = (turn_energy - speed_energy) / speed_energy > COMMAND_ENERGY_MARGIN
    concentrated = patch_concentrated or channel_concentrated
    any_beats = mean_beats or subspace_beats or linear_beats
    if not any_beats:
        call = "no_stable_signal"
        next_step = (
            "均值、残差子空间和当前帧线性读出都没有比零残差低过 1%。"
            "全网格残差没有稳定可预测成分。停止这条 SigLIP 残差，不建层级结构。"
        )
    elif linear_beats and concentrated and turn_coherent and turn_specific:
        call = "geometric_target_candidate"
        next_step = (
            "当前帧线性读出打赢零残差，能量集中，左右转的差成片且大于速度对。"
            "可以改成低维运动或几何目标。这一步还没有训练层级网络。"
        )
    elif any_beats and turn_coherent and not turn_specific:
        call = "appearance_not_command"
        next_step = (
            "变化在空间上成片，但左右转并不比速度变化更大。"
            "这是画面变化，不接导航，也不建层级网络。"
        )
    elif linear_beats:
        call = "partial_linear_signal"
        next_step = (
            "当前帧线性读出打赢了零残差，但能量集中和左右转成片没有同时成立。"
            "先不换目标，也不加容量。"
        )
    else:
        call = "stable_residual_direction"
        next_step = (
            "残差均值或子空间比零残差低过 1%，当前帧线性读出没有。"
            "先不把这当作可从图像预测的几何信号，也不建层级网络。"
        )
    labels = {
        "no_stable_signal": "全网格残差没有稳定可预测成分",
        "geometric_target_candidate": "低维几何目标可以进入设计",
        "appearance_not_command": "空间上成片，但不由左右转区分",
        "partial_linear_signal": "线性读出有收益，空间条件不齐",
        "stable_residual_direction": "残差有稳定方向，线性读出没有",
    }
    return {
        "call": call,
        "call_label": labels[call],
        "next_step": next_step,
        "mean_beats_zero": mean_beats,
        "subspace_beats_zero": subspace_beats,
        "linear_beats_zero": linear_beats,
        "patch_concentrated": patch_concentrated,
        "channel_concentrated": channel_concentrated,
        "turn_coherent": turn_coherent,
        "turn_specific": turn_specific,
        "relative_band": RELATIVE_BAND,
        "component_count": COMPONENT_COUNT,
        "top_fraction": TOP_FRACTION,
        "concentration_share": CONCENTRATION_SHARE,
        "coherence_correlation": COHERENCE_CORRELATION,
        "command_energy_margin": COMMAND_ENERGY_MARGIN,
        "development_is_blind_test": False,
        "states_are_formal_fit": True,
        "generalization": False,
        "navigation_connected": False,
        "habitat_started": False,
        "formal_cache_written": False,
        "training_started": False,
    }
