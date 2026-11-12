"""Future-only representation comparison.

Three inputs predict one future camera grid. The official pooler, the mean of
patch tokens, and the full patch grid do not share a tensor. G and M never
receive the current patch grid, including as a residual bypass. There is no
pose head, no history, and no depth.

SigLIP already added a position embedding before these tokens were cached.
This decoder still adds row and column indices of the processed image plane,
because that embedding is not an explicit coordinate input. Those indices are
not 3D world positions. Per-position loss supervises the future camera grid.
It does not establish that the same object stayed at the same index.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn

GRID = 27
TOKENS = GRID * GRID
FEATURE_DIM = 1152
BLOCK = 9
BLOCKS = 3
D_MODEL = 128
HEADS = 4
LAYERS = 2
FFN_WIDTH = D_MODEL * 4
MAX_PARAMETERS = 2_000_000
ARMS = ("pooler", "mean", "grid")
CANDIDATE_LAG = 5
RANDOM_SEED = 20261029
CHANGE_DIRECTION_MIN_NORM = 1e-6
CHANNEL_STD_FLOOR = 0.01
ACTION_CONTROL_RELATIVE_BAND = 0.01
NAVIGATION_CONNECTED = False
POSE_LOSS = False
DEVELOPMENT_IS_BLIND_TEST = False


def siglip_grid_side(image_size: int, patch_size: int) -> int:
    """Match SigLIP's valid patch convolution, which has no class token."""
    if image_size < patch_size or patch_size < 1:
        raise ValueError(f"invalid SigLIP grid inputs {image_size}, {patch_size}")
    return (image_size - patch_size) // patch_size + 1


def siglip_grid_report(image_size: int = 384, patch_size: int = 14) -> dict:
    """Record why 384 by 14 is a 27 grid and which pixels the convolution drops."""
    side = siglip_grid_side(image_size, patch_size)
    floordiv = image_size // patch_size
    if side != floordiv:
        raise ValueError(f"convolution grid {side} disagrees with floor division {floordiv}")
    return {
        "image_size": image_size,
        "patch_size": patch_size,
        "grid": side,
        "tokens": side * side,
        "order": "row-major Conv2d.flatten(2).transpose(1, 2)",
        "uncovered_pixels_after_last_patch": image_size - side * patch_size,
        "center_formula": "index * patch_size + (patch_size - 1) / 2",
        "coordinate_space": "resized image plane, not world meters",
        "class_token": False,
        "siglip_position_embedding_already_in_cached_tokens": True,
    }


def token_index(row: int, col: int, grid: int = GRID) -> int:
    if not 0 <= row < grid or not 0 <= col < grid:
        raise ValueError(f"patch coordinate {(row, col)} is outside {grid}")
    return row * grid + col


def token_coordinate(index: int, grid: int = GRID) -> tuple[int, int]:
    if not 0 <= index < grid * grid:
        raise ValueError(f"patch index {index} is outside {grid * grid}")
    return divmod(index, grid)


def patch_plane_centers(image_size: int = 384, patch_size: int = 14) -> np.ndarray:
    """Pixel centers on the resized image, in the same order as the cached tokens."""
    side = siglip_grid_side(image_size, patch_size)
    centers = np.empty((side * side, 2), dtype=np.float64)
    offset = (patch_size - 1) / 2
    for row in range(side):
        for col in range(side):
            centers[token_index(row, col, side)] = (
                row * patch_size + offset,
                col * patch_size + offset,
            )
    return centers


def conv_flatten_tokens(patch_grid: torch.Tensor) -> torch.Tensor:
    """Row-major order produced by ``Conv2d.flatten(2).transpose(1, 2)``."""
    if patch_grid.ndim != 3:
        raise ValueError("expected (height, width, channels)")
    channels_first = patch_grid.permute(2, 0, 1).unsqueeze(0)
    return channels_first.flatten(2).transpose(1, 2)[0]


def uses_memory_coordinates(arm: str) -> bool:
    if arm not in ARMS:
        raise ValueError(f"unknown representation {arm}")
    return arm == "grid"


def assert_split(fit_rows: list[dict], development_rows: list[dict]) -> dict[str, int]:
    """Development rows are a held-out development set, not a blind test."""
    fit_ids = [str(row["transition_id"]) for row in fit_rows]
    development_ids = [str(row["transition_id"]) for row in development_rows]
    if len(fit_ids) != len(set(fit_ids)) or len(development_ids) != len(set(development_ids)):
        raise ValueError("a split contains a duplicate transition id")
    if set(fit_ids) & set(development_ids):
        raise ValueError("a transition is in both fit and development")
    overlap = {
        "scene": {row["scene"] for row in fit_rows} & {row["scene"] for row in development_rows},
        "route_group_id": {row["route_group_id"] for row in fit_rows}
        & {row["route_group_id"] for row in development_rows},
    }
    bad = {name: sorted(values) for name, values in overlap.items() if values}
    if bad:
        raise ValueError(f"fit and development share {bad}")
    return {
        "fit_transitions": len(fit_ids),
        "development_transitions": len(development_ids),
        "fit_scenes": len({row["scene"] for row in fit_rows}),
        "development_scenes": len({row["scene"] for row in development_rows}),
        "development_is_blind_test": False,
    }


def assign_scene_folds(rows: list[dict], folds: int = 3) -> dict:
    """Seal a scene fold table. A route that crosses scenes blocks the split."""
    if folds < 2:
        raise ValueError("need at least two folds")
    route_scenes: dict[str, set[str]] = {}
    for row in rows:
        route_scenes.setdefault(str(row["route_group_id"]), set()).add(str(row["scene"]))
    crossed = {route: sorted(scenes) for route, scenes in route_scenes.items() if len(scenes) > 1}
    if crossed:
        raise ValueError(f"scene folds are blocked because routes cross scenes: {crossed}")
    scenes = sorted({str(row["scene"]) for row in rows})
    if len(scenes) < folds:
        raise ValueError("scene folds are blocked because there are fewer scenes than folds")
    assignment = {scene: index % folds for index, scene in enumerate(scenes)}
    sealed = []
    for fold in range(folds):
        validation = sorted(scene for scene, owner in assignment.items() if owner == fold)
        train = sorted(scene for scene, owner in assignment.items() if owner != fold)
        validation_routes = sorted(
            {str(row["route_group_id"]) for row in rows if str(row["scene"]) in set(validation)}
        )
        train_routes = sorted(
            {str(row["route_group_id"]) for row in rows if str(row["scene"]) in set(train)}
        )
        scene_overlap = sorted(set(validation) & set(train))
        route_overlap = sorted(set(validation_routes) & set(train_routes))
        if scene_overlap or route_overlap or not validation or not train:
            raise ValueError(f"fold {fold} does not isolate scenes and routes")
        sealed.append(
            {
                "fold": fold,
                "validation_scenes": validation,
                "train_scenes": train,
                "validation_routes": validation_routes,
                "train_routes": train_routes,
                "validation_transitions": sum(str(row["scene"]) in set(validation) for row in rows),
                "train_transitions": sum(str(row["scene"]) in set(train) for row in rows),
                "scene_overlap": scene_overlap,
                "route_overlap": route_overlap,
            }
        )
    return {
        "rule": "sort scene id ascending, fold = index modulo 3",
        "blocked": False,
        "random_frame_split": False,
        "folds": sealed,
    }


def stable_id_hash(ids: list[str]) -> str:
    body = "\n".join(sorted(set(ids))).encode()
    return hashlib.sha256(body).hexdigest()


def label_timing(rows: list[dict]) -> dict:
    """Check label order. Gaps are reported; non-finite or duplicate steps are refused."""
    same_frame = 0
    observation_chain_breaks = 0
    time_gaps = 0
    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        if not math.isfinite(float(row["t"])):
            raise ValueError("a label time is non-finite; it was not replaced")
        if row["obs_before_id"] == row["obs_after_id"]:
            same_frame += 1
        by_episode.setdefault(str(row["episode_uid"]), []).append(row)
    for group in by_episode.values():
        ordered = sorted(group, key=lambda item: (int(item["t"]), str(item["transition_id"])))
        seen: set[int] = set()
        for row in ordered:
            step = int(row["t"])
            if step in seen:
                raise ValueError("an episode repeats a time step; labels were not imputed")
            seen.add(step)
        for previous, row in zip(ordered, ordered[1:], strict=False):
            if int(row["t"]) != int(previous["t"]) + 1:
                time_gaps += 1
            elif row["obs_before_id"] != previous["obs_after_id"]:
                observation_chain_breaks += 1
    return {
        "episodes": len(by_episode),
        "same_frame_labels": same_frame,
        "time_gaps": time_gaps,
        "observation_chain_breaks": observation_chain_breaks,
        "labels_imputed": False,
    }


def build_memory(
    arm: str,
    *,
    patches: torch.Tensor | None,
    pooler: torch.Tensor | None,
) -> torch.Tensor:
    """Return the one tensor this arm is allowed to read."""
    if arm not in ARMS:
        raise ValueError(f"unknown representation {arm}")
    if arm == "pooler":
        if patches is not None:
            raise ValueError("pooler arm cannot read patch tokens")
        if pooler is None or pooler.ndim != 2 or pooler.shape[-1] != FEATURE_DIM:
            raise ValueError("pooler arm requires (batch, 1152)")
        return pooler[:, None, :]
    if pooler is not None:
        raise ValueError(f"{arm} arm cannot read the official pooler")
    if patches is None or patches.ndim != 3 or patches.shape[1:] != (TOKENS, FEATURE_DIM):
        raise ValueError("patch arms require (batch, 729, 1152)")
    if arm == "mean":
        return patches.mean(dim=1, keepdim=True)
    return patches


def executed_action_features(control: torch.Tensor, *, use_action: bool) -> torch.Tensor:
    """Six features from the executed command. The matched control uses zeros at train and infer."""
    if control.ndim != 2 or control.shape[-1] != 3:
        raise ValueError("control must be (batch, 3) as v_mps, w_rad_s, dt_command_s")
    if not use_action:
        return torch.zeros(control.shape[0], 6, device=control.device, dtype=control.dtype)
    velocity, yaw_rate, dt = control.unbind(-1)
    stop = (velocity.abs() < 0.01) & (yaw_rate.abs() < math.radians(0.5))
    velocity = torch.where(stop, torch.zeros_like(velocity), velocity)
    yaw_rate = torch.where(stop, torch.zeros_like(yaw_rate), yaw_rate)
    yaw = (yaw_rate * dt + torch.pi) % (2 * torch.pi) - torch.pi
    forward = velocity * dt
    return torch.stack(
        [
            velocity / 2.5,
            yaw_rate / math.radians(300.0),
            dt / 0.1,
            forward / 0.25,
            yaw / math.radians(30.0),
            stop.float(),
        ],
        dim=-1,
    )


class _CrossBlock(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(d_model, heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, FFN_WIDTH),
            nn.GELU(),
            nn.Linear(FFN_WIDTH, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _weights = self.cross(queries, memory, memory, need_weights=False)
        hidden = self.norm1(queries + attended)
        return self.norm2(hidden + self.ff(hidden))


ACTION_INJECTIONS = ("memory_token", "query_film")


class FutureGridDecoder(nn.Module):
    """Two cross-attention layers from a shared 27 by 27 query grid onto the memory.

    ``memory_token`` appends the command as one extra memory token.
    ``query_film`` does not. A zero-initialized film scales and shifts every query
    from the six command features, so the first forward is unmodulated.
    """

    def __init__(
        self,
        arm: str,
        use_action: bool,
        action_injection: str = "memory_token",
        d_model: int = D_MODEL,
        heads: int = HEADS,
        layers: int = LAYERS,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown representation {arm}")
        if action_injection not in ACTION_INJECTIONS:
            raise ValueError(f"unknown action injection {action_injection}")
        if layers != LAYERS or d_model != D_MODEL or heads != HEADS:
            raise ValueError("stage A uses one registered decoder width, depth, and head count")
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.arm = arm
        self.use_action = use_action
        self.action_injection = action_injection
        self.memory_tokens = TOKENS if arm == "grid" else 1
        self.memory_proj = nn.Linear(FEATURE_DIM, d_model)
        if action_injection == "memory_token":
            self.action_proj = nn.Linear(6, d_model)
        else:
            self.film = nn.Linear(6, 2 * d_model)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.row = nn.Embedding(GRID, d_model)
        self.col = nn.Embedding(GRID, d_model)
        self.query_content = nn.Parameter(torch.zeros(d_model))
        self.blocks = nn.ModuleList(_CrossBlock(d_model, heads) for _ in range(layers))
        self.out = nn.Linear(d_model, FEATURE_DIM)
        rows = torch.arange(GRID).repeat_interleave(GRID)
        cols = torch.arange(GRID).repeat(GRID)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        count = sum(parameter.numel() for parameter in self.parameters())
        if count > MAX_PARAMETERS:
            raise ValueError(f"{count} trainable parameters exceed {MAX_PARAMETERS}")

    def forward(self, memory: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        if memory.ndim != 3 or memory.shape[1:] != (self.memory_tokens, FEATURE_DIM):
            raise ValueError(
                f"{self.arm} memory must be (batch, {self.memory_tokens}, {FEATURE_DIM}), "
                f"got {tuple(memory.shape)}"
            )
        if control.shape != (memory.shape[0], 3):
            raise ValueError("control must be (batch, 3) and aligned with memory")
        features = executed_action_features(control, use_action=self.use_action)
        hidden = self.memory_proj(memory)
        if uses_memory_coordinates(self.arm):
            hidden = hidden + self.row(self.rows) + self.col(self.cols)
        queries = self.row(self.rows) + self.col(self.cols) + self.query_content
        queries = queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        if self.action_injection == "memory_token":
            action = self.action_proj(features)
            hidden = torch.cat([hidden, action[:, None, :]], dim=1)
        else:
            gamma, beta = self.film(features).chunk(2, dim=-1)
            queries = queries * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        for block in self.blocks:
            queries = block(queries, hidden)
        return self.out(queries)


def future_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error on the future camera grid. Pose is not an argument."""
    if predicted.shape != target.shape or predicted.shape[-2:] != (TOKENS, FEATURE_DIM):
        raise ValueError("future loss expects aligned (batch, 729, 1152) grids")
    return (predicted - target).square().mean()


def normalized_grid_mse(
    predicted: torch.Tensor, target: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    if scale.shape != (FEATURE_DIM,):
        raise ValueError("channel scale must be (1152,)")
    return future_loss(predicted / scale, target / scale)


def fit_channel_scale(future: np.ndarray) -> np.ndarray:
    """Channel standard deviation of future tokens. Fit rows only; floor 0.01."""
    array = np.asarray(future, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (TOKENS, FEATURE_DIM):
        raise ValueError("channel scale expects (samples, 729, 1152)")
    if not np.isfinite(array).all():
        raise ValueError("non-finite future tokens; scale was not imputed")
    deviation = np.ascontiguousarray(array).reshape(-1, FEATURE_DIM).std(axis=0)
    return np.maximum(deviation, CHANNEL_STD_FLOOR).astype(np.float32)


def decoder_flops(arm: str, batch: int = 1) -> int:
    """Multiply-add pairs, counted as two FLOPs each, for one forward."""
    if arm not in ARMS or batch < 1:
        raise ValueError("FLOPs require a known arm and a positive batch")
    memory = TOKENS if arm == "grid" else 1
    keys = memory + 1
    queries = TOKENS
    d_model = D_MODEL
    macs = batch * memory * FEATURE_DIM * d_model
    macs += batch * 6 * d_model
    for _ in range(LAYERS):
        macs += batch * (queries + 2 * keys) * d_model * d_model
        macs += 2 * batch * queries * keys * d_model
        macs += batch * queries * d_model * d_model
        macs += batch * queries * d_model * FFN_WIDTH
        macs += batch * queries * FFN_WIDTH * d_model
    macs += batch * queries * d_model * FEATURE_DIM
    return int(macs * 2)


def parameter_count(
    arm: str, use_action: bool = True, action_injection: str = "memory_token"
) -> int:
    model = FutureGridDecoder(arm, use_action=use_action, action_injection=action_injection)
    return sum(parameter.numel() for parameter in model.parameters())


def batch_plan(count: int, batch_size: int, updates: int, seed: int) -> list[np.ndarray]:
    """Identical truncated batches for every arm that shares count, batch, updates, and seed."""
    if count < batch_size or batch_size < 1 or updates < 1:
        raise ValueError("batch plan requires at least one full batch and one update")
    generator = torch.Generator()
    generator.manual_seed(seed)
    usable = (count // batch_size) * batch_size
    plan: list[np.ndarray] = []
    while len(plan) < updates:
        order = torch.randperm(count, generator=generator).numpy()[:usable]
        for start in range(0, usable, batch_size):
            plan.append(order[start : start + batch_size])
            if len(plan) == updates:
                break
    return plan


def spatial_descriptor(tokens: np.ndarray) -> np.ndarray:
    """Nine L2-normalized 3 by 3 block means, then one L2-normalized descriptor."""
    array = np.asarray(tokens, dtype=np.float32)
    if array.shape[-2:] != (TOKENS, FEATURE_DIM):
        raise ValueError("descriptor expects (..., 729, 1152)")
    lead = array.shape[:-2]
    grid = array.reshape(*lead, GRID, GRID, FEATURE_DIM)
    blocks = grid.reshape(*lead, BLOCKS, BLOCK, BLOCKS, BLOCK, FEATURE_DIM).mean(axis=(-4, -2))
    blocks = blocks.reshape(*lead, BLOCKS * BLOCKS, FEATURE_DIM)
    blocks = blocks / np.clip(np.linalg.norm(blocks, axis=-1, keepdims=True), 1e-8, None)
    flat = blocks.reshape(*lead, BLOCKS * BLOCKS * FEATURE_DIM)
    return flat / np.clip(np.linalg.norm(flat, axis=-1, keepdims=True), 1e-8, None)


def global_mean_descriptor(tokens: np.ndarray) -> np.ndarray:
    """L2-normalized mean over tokens. This is the old reference descriptor, not the primary one."""
    array = np.asarray(tokens, dtype=np.float32)
    if array.shape[-2:] != (TOKENS, FEATURE_DIM):
        raise ValueError("descriptor expects (..., 729, 1152)")
    mean = array.mean(axis=-2)
    return mean / np.clip(np.linalg.norm(mean, axis=-1, keepdims=True), 1e-8, None)


def _as_unit(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    return values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8, None)


def candidate_lists(meta: list[dict]) -> dict:
    """One causal candidate list per row, deduped by the raw RGB pixel hash."""
    by_episode: dict[str, list[int]] = {}
    for index, row in enumerate(meta):
        for key in ("episode_uid", "t", "transition_id", "rgb_pixel_sha256", "scene", "route_group_id"):
            if key not in row:
                raise ValueError(f"retrieval metadata is missing {key}")
        by_episode.setdefault(str(row["episode_uid"]), []).append(index)
    lists: list[list[int]] = [[] for _ in meta]
    removed = 0
    considered = 0
    excluded_before = 0
    excluded_after = 0
    for indexes in by_episode.values():
        ordered = sorted(indexes, key=lambda item: (int(meta[item]["t"]), str(meta[item]["transition_id"])))
        for index in ordered:
            raw = [
                item
                for item in ordered
                if int(meta[item]["t"]) <= int(meta[index]["t"]) - CANDIDATE_LAG
            ]
            considered += len(raw)
            if len(raw) < 2:
                excluded_before += 1
                continue
            kept: list[int] = []
            seen: set[str] = set()
            for item in raw:
                digest = str(meta[item]["rgb_pixel_sha256"])
                if digest in seen:
                    removed += 1
                    continue
                seen.add(digest)
                kept.append(item)
            if len(kept) < 2:
                excluded_after += 1
                continue
            lists[index] = kept
    eligible = sum(len(item) >= 2 for item in lists)
    return {
        "indices": lists,
        "queries": len(meta),
        "eligible": eligible,
        "excluded_before_dedup": excluded_before,
        "excluded_after_dedup": excluded_after,
        "duplicates_removed": removed,
        "candidates_considered": considered,
        "duplicate_rate": None if considered == 0 else removed / considered,
        "candidate_coverage": eligible / len(meta) if meta else None,
        "candidate_lag": CANDIDATE_LAG,
        "dedup": "admitted_rgb_pixel_sha256",
        "development_is_blind_test": False,
    }


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "recall_at_2": None, "mrr_oracle_top1": None}
    return {
        "n": len(rows),
        "recall_at_2": float(np.mean([row["recall_at_2"] for row in rows])),
        "mrr_oracle_top1": float(np.mean([row["mrr_oracle_top1"] for row in rows])),
    }


def retrieval_evaluation(
    queries: dict[str, np.ndarray],
    bank: np.ndarray,
    meta: list[dict],
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Score every query on the same candidate ids, descriptor, and oracle."""
    if not queries:
        raise ValueError("retrieval needs at least one query")
    if set(queries) & {"recent", "random"}:
        raise ValueError("recent and random are rankings of the shared list, not query features")
    prepared = candidate_lists(meta)
    bank_unit = _as_unit(bank)
    query_units = {name: _as_unit(value) for name, value in queries.items()}
    for name, value in query_units.items():
        if len(value) != len(meta) or len(bank_unit) != len(meta):
            raise ValueError(f"{name} query is not aligned with the candidate bank")
    generator = np.random.default_rng(random_seed)
    methods = [*queries, "recent", "random"]
    records: dict[str, list[dict]] = {name: [] for name in methods}
    for index, candidates in enumerate(prepared["indices"]):
        if len(candidates) < 2:
            continue
        candidate = np.asarray(candidates)
        bank_rows = bank_unit[candidate]
        oracle_scores = bank_rows @ bank_unit[index]
        oracle_order = np.argsort(-oracle_scores, kind="stable")
        oracle_top = set(oracle_order[:2].tolist())
        oracle_best = int(oracle_order[0])
        recent_key = np.array([-int(meta[item]["t"]) for item in candidates])
        rankings = {
            "recent": np.argsort(recent_key, kind="stable"),
            "random": generator.permutation(len(candidates)),
        }
        for name, vector in query_units.items():
            rankings[name] = np.argsort(-(bank_rows @ vector[index]), kind="stable")
        candidate_ids = [str(meta[item]["transition_id"]) for item in candidates]
        for name, order in rankings.items():
            picked = set(order[:2].tolist())
            rank = int(np.flatnonzero(order == oracle_best)[0]) + 1
            records[name].append(
                {
                    "transition_id": str(meta[index]["transition_id"]),
                    "scene": str(meta[index]["scene"]),
                    "route_group_id": str(meta[index]["route_group_id"]),
                    "candidate_ids": candidate_ids,
                    "recall_at_2": len(oracle_top & picked) / 2,
                    "mrr_oracle_top1": 1.0 / rank,
                }
            )
    reference_ids = [row["candidate_ids"] for row in records[methods[0]]]
    if any([row["candidate_ids"] for row in records[name]] != reference_ids for name in methods):
        raise ValueError("retrieval methods did not share candidate ids")
    by_scene = {
        scene: {
            name: _aggregate([row for row in records[name] if row["scene"] == scene]) for name in methods
        }
        for scene in sorted({row["scene"] for row in records[methods[0]]})
    }
    by_route = {
        route: {
            name: _aggregate([row for row in records[name] if row["route_group_id"] == route])
            for name in methods
        }
        for route in sorted({row["route_group_id"] for row in records[methods[0]]})
    }
    return {
        "methods": {name: _aggregate(records[name]) for name in methods},
        "by_scene": by_scene,
        "by_route": by_route,
        "candidates": {key: value for key, value in prepared.items() if key != "indices"},
        "random_seed": random_seed,
        "development_is_blind_test": False,
    }


def _row_norms(array: np.ndarray) -> np.ndarray:
    flat = np.asarray(array, dtype=np.float64).reshape(array.shape[0], -1)
    return np.linalg.norm(flat, axis=1)


def _token_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_unit = left / np.clip(np.linalg.norm(left, axis=-1, keepdims=True), 1e-8, None)
    right_unit = right / np.clip(np.linalg.norm(right, axis=-1, keepdims=True), 1e-8, None)
    return (left_unit * right_unit).sum(axis=-1).mean(axis=-1)


def grid_prediction_metrics(
    predicted: np.ndarray, target: np.ndarray, current: np.ndarray, scenes: list[str]
) -> dict:
    """Original-scale future-grid metrics. A ratio near zero change is left undefined."""
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    current = np.asarray(current)
    if predicted.shape != target.shape or predicted.shape != current.shape:
        raise ValueError("predicted, current, and future grids must share a shape")
    if predicted.shape[1:] != (TOKENS, FEATURE_DIM) or len(scenes) != len(predicted):
        raise ValueError("metrics expect one scene per (729, 1152) grid")
    count = len(predicted)
    per_sample = np.empty(count, dtype=np.float64)
    token_cosine = np.empty(count, dtype=np.float64)
    change_error = np.empty(count, dtype=np.float64)
    true_norm = np.empty(count, dtype=np.float64)
    pred_norm = np.empty(count, dtype=np.float64)
    direction_dot = np.empty(count, dtype=np.float64)
    cosine_to_current = np.empty(count, dtype=np.float64)
    cosine_current_future = np.empty(count, dtype=np.float64)
    for start in range(0, count, 16):
        stop = min(start + 16, count)
        piece_predicted = np.asarray(predicted[start:stop], dtype=np.float32)
        piece_target = np.asarray(target[start:stop], dtype=np.float32)
        piece_current = np.asarray(current[start:stop], dtype=np.float32)
        if (
            not np.isfinite(piece_predicted).all()
            or not np.isfinite(piece_target).all()
            or not np.isfinite(piece_current).all()
        ):
            raise ValueError("non-finite grid; metrics were not imputed")
        difference = piece_predicted - piece_target
        true_change = piece_target - piece_current
        pred_change = piece_predicted - piece_current
        per_sample[start:stop] = np.mean(difference * difference, axis=(1, 2), dtype=np.float64)
        token_cosine[start:stop] = _token_cosine(piece_predicted, piece_target)
        change_error[start:stop] = _row_norms(difference)
        true_norm[start:stop] = _row_norms(true_change)
        pred_norm[start:stop] = _row_norms(pred_change)
        direction_dot[start:stop] = np.sum(
            true_change.reshape(stop - start, -1) * pred_change.reshape(stop - start, -1), axis=1
        )
        cosine_to_current[start:stop] = _token_cosine(piece_predicted, piece_current)
        cosine_current_future[start:stop] = _token_cosine(piece_current, piece_target)
    defined = true_norm >= CHANGE_DIRECTION_MIN_NORM
    magnitude = np.full(count, np.nan)
    direction = np.full(count, np.nan)
    if defined.any():
        magnitude[defined] = pred_norm[defined] / true_norm[defined]
        direction[defined] = direction_dot[defined] / (true_norm[defined] * pred_norm[defined] + 1e-8)
    grouped: dict[str, list[float]] = {}
    grouped_cosine: dict[str, list[float]] = {}
    for value, cosine, scene in zip(per_sample, token_cosine, scenes, strict=True):
        grouped.setdefault(scene, []).append(float(value))
        grouped_cosine.setdefault(scene, []).append(float(cosine))
    return {
        "n": int(len(predicted)),
        "scenes": len(grouped),
        "mse": float(per_sample.mean()),
        "scene_macro_mse": float(np.mean([np.mean(values) for values in grouped.values()])),
        "token_cosine": float(token_cosine.mean()),
        "scene_macro_token_cosine": float(np.mean([np.mean(values) for values in grouped_cosine.values()])),
        "change_vector_error": float(change_error.mean()),
        "magnitude_ratio_mean": None if not defined.any() else float(np.nanmean(magnitude)),
        "magnitude_ratio_median": None if not defined.any() else float(np.nanmedian(magnitude)),
        "direction_cosine_mean": None if not defined.any() else float(np.nanmean(direction)),
        "direction_undefined": int((~defined).sum()),
        "cosine_prediction_to_current": float(cosine_to_current.mean()),
        "cosine_prediction_to_future": float(token_cosine.mean()),
        "cosine_current_to_future": float(cosine_current_future.mean()),
    }


def mean_change_prediction(
    train_current: np.ndarray, train_future: np.ndarray, current: np.ndarray
) -> np.ndarray:
    """Add the train-split mean token change. This reference sees the current grid."""
    if train_current.shape != train_future.shape or train_current.shape[1:] != (TOKENS, FEATURE_DIM):
        raise ValueError("mean change expects aligned training grids")
    future_total = np.zeros((TOKENS, FEATURE_DIM), dtype=np.float64)
    current_total = np.zeros((TOKENS, FEATURE_DIM), dtype=np.float64)
    for start in range(0, len(train_current), 32):
        future_total += np.sum(train_future[start : start + 32], axis=0, dtype=np.float64)
        current_total += np.sum(train_current[start : start + 32], axis=0, dtype=np.float64)
    delta = future_total / len(train_current) - current_total / len(train_current)
    return np.asarray(current, dtype=np.float32) + delta.astype(np.float32)


def synthetic_routing_check() -> dict:
    """Two layouts share a token mean and must not share a grid representation.

    The mean arm therefore emits one prediction for both samples. One prediction
    cannot place the active token at two different destination cells, so its
    loss on this pair cannot fall below 1. This checks the representation. It
    does not claim that a few optimizer steps solve correspondence.
    """
    current = torch.zeros(2, TOKENS, FEATURE_DIM)
    current[0, token_index(0, 0), 0] = 1.0
    current[1, token_index(5, 5), 0] = 1.0
    destination = (token_index(0, 1), token_index(5, 6))
    control = torch.tensor([[0.25, 0.0, 0.1], [0.25, 0.0, 0.1]])
    mean_memory = build_memory("mean", patches=current, pooler=None)
    grid_memory = build_memory("grid", patches=current, pooler=None)
    if not torch.equal(mean_memory[0], mean_memory[1]):
        raise AssertionError("the synthetic mean input unexpectedly depends on token position")
    if torch.allclose(grid_memory[0], grid_memory[1]):
        raise AssertionError("the synthetic grid input did not keep token position")
    torch.manual_seed(0)
    mean_model = FutureGridDecoder("mean", use_action=True)
    torch.manual_seed(0)
    grid_model = FutureGridDecoder("grid", use_action=True)
    mean_out = mean_model(mean_memory, control)
    grid_out = grid_model(grid_memory, control)
    if not torch.allclose(mean_out[0], mean_out[1]):
        raise AssertionError("the mean arm produced different outputs from one input")
    if torch.allclose(grid_out[0], grid_out[1]):
        raise AssertionError("the grid arm ignored the token layout")

    def route_loss(predicted: torch.Tensor) -> torch.Tensor:
        hit = (predicted[0, destination[0], 0] - 1.0).square() + (
            predicted[1, destination[1], 0] - 1.0
        ).square()
        miss = predicted[0, destination[1], 0].square() + predicted[1, destination[0], 0].square()
        return hit + miss

    loss = route_loss(mean_out)
    loss.backward()
    route_loss(grid_out).backward()
    for model in (mean_model, grid_model):
        gradients = [parameter.grad for parameter in model.parameters()]
        if any(grad is None or not torch.isfinite(grad).all() for grad in gradients):
            raise AssertionError("synthetic backward produced a non-finite gradient")
    return {
        "mean_outputs_tied": True,
        "grid_outputs_differ": True,
        "tied_output_loss_floor": 1.0,
        "mean_loss": float(loss.detach()),
    }


def parse_shard_spec(spec: str) -> tuple[int, int]:
    """Parse ``i/n`` into a zero-based shard index and a shard count."""
    left, separator, right = spec.partition("/")
    if separator != "/" or not left.isdigit() or not right.isdigit():
        raise ValueError(f"shard spec must look like 0/6, got {spec!r}")
    shard, count = int(left), int(right)
    if count < 1 or shard >= count:
        raise ValueError(f"shard {shard} is outside 0..{count - 1}")
    return shard, count


def shard_slots(job_count: int, shard: int, shards: int) -> list[int]:
    """Stable round-robin assignment. Every job index belongs to one shard."""
    if job_count < 0 or shards < 1 or shard < 0 or shard >= shards:
        raise ValueError(f"cannot assign {job_count} jobs to shard {shard}/{shards}")
    return [index for index in range(job_count) if index % shards == shard]


def _control_call(relatives: list[float], band: float) -> str:
    if all(value > band for value in relatives):
        return "action_lower"
    if all(value < -band for value in relatives):
        return "no_action_lower"
    if all(abs(value) <= band for value in relatives):
        return "within_band"
    return "mixed"


def residual_injection_verdict(rows: list[dict]) -> dict:
    """Scene-macro normalized residual MSE for the query-film control.

    Relative gap is (no-action - action) / action. The band is the same 1%.
    """
    expected = (20261029, 20261030, 20261031)
    labels = {
        "action_lower": "有动作更低",
        "no_action_lower": "无动作更低",
        "within_band": "1% 带内",
        "mixed": "方向不一致",
    }
    action: dict[int, float] = {}
    no_action: dict[int, float] = {}
    for row in rows:
        seed = int(row["seed"])
        mse = float(row["scene_macro_residual_mse"])
        if seed not in expected or not math.isfinite(mse) or mse <= 0.0:
            raise ValueError(f"seed {seed} residual MSE is missing, non-positive, or non-finite")
        bucket = action if bool(row["use_action"]) else no_action
        if seed in bucket:
            raise ValueError(f"duplicate residual row for seed {seed}")
        bucket[seed] = mse
    if set(action) != set(expected) or set(no_action) != set(expected):
        raise ValueError("residual rows are not the three registered seeds for both conditions")
    compared = []
    relatives = []
    for seed in expected:
        gap = (no_action[seed] - action[seed]) / action[seed]
        relatives.append(gap)
        compared.append(
            {
                "seed": seed,
                "action_mse": action[seed],
                "no_action_mse": no_action[seed],
                "relative_mse": gap,
            }
        )
    call = _control_call(relatives, ACTION_CONTROL_RELATIVE_BAND)
    return {
        "call": call,
        "call_label": labels[call],
        "relative_band": ACTION_CONTROL_RELATIVE_BAND,
        "metric": "scene_macro_normalized_residual_mse",
        "target": "future_patch_grid - current_patch_grid",
        "action_injection": "query_film",
        "seeds": compared,
        "action_lowers_residual_on_every_seed": call == "action_lower",
        "retrieval_recomputed": False,
        "evidence_grade": "不足",
        "development_is_blind_test": False,
        "navigation_connected": False,
        "habitat_started": False,
        "formal_cache_written": False,
        "stage_b_started": False,
        "stage_c_started": False,
        "stage_d_started": False,
    }


def action_control_verdict(action_rows: list[dict], no_action_rows: list[dict]) -> dict:
    """Compare matched trainings on the development set.

    Relative gap is (no-action metric - action metric) / action metric.
    Retrieval is not an input. Eight scenes stay evidence-poor.
    """
    expected = (20261029, 20261030, 20261031)
    labels = {
        "action_lower": "有动作更低",
        "no_action_lower": "无动作更低",
        "within_band": "1% 带内",
        "mixed": "方向不一致",
    }

    def index(rows: list[dict], label: str) -> dict:
        found = {}
        for row in rows:
            arm = str(row["arm"])
            seed = int(row["seed"])
            if arm not in ARMS or seed not in expected:
                raise ValueError(f"{label} row has arm {arm} and seed {seed}")
            key = (arm, seed)
            if key in found:
                raise ValueError(f"duplicate {label} row for {arm} seed {seed}")
            mse = float(row["scene_macro_mse"])
            change = float(row["change_vector_error"])
            if not math.isfinite(mse) or not math.isfinite(change) or mse <= 0.0 or change <= 0.0:
                raise ValueError(f"{label} {arm} seed {seed} metric is non-positive or non-finite")
            found[key] = {"scene_macro_mse": mse, "change_vector_error": change}
        needed = {(arm, seed) for arm in ARMS for seed in expected}
        if set(found) != needed:
            raise ValueError(f"{label} rows are not the three arms and three registered seeds")
        return found

    action = index(action_rows, "action")
    no_action = index(no_action_rows, "no-action")
    arms = {}
    for arm in ARMS:
        seeds = []
        mse_gaps = []
        change_gaps = []
        for seed in expected:
            left = action[(arm, seed)]
            right = no_action[(arm, seed)]
            mse_gap = (right["scene_macro_mse"] - left["scene_macro_mse"]) / left["scene_macro_mse"]
            change_gap = (
                right["change_vector_error"] - left["change_vector_error"]
            ) / left["change_vector_error"]
            mse_gaps.append(mse_gap)
            change_gaps.append(change_gap)
            seeds.append(
                {
                    "seed": seed,
                    "action_mse": left["scene_macro_mse"],
                    "no_action_mse": right["scene_macro_mse"],
                    "relative_mse": mse_gap,
                    "action_change_error": left["change_vector_error"],
                    "no_action_change_error": right["change_vector_error"],
                    "relative_change_error": change_gap,
                }
            )
        mse_call = _control_call(mse_gaps, ACTION_CONTROL_RELATIVE_BAND)
        change_call = _control_call(change_gaps, ACTION_CONTROL_RELATIVE_BAND)
        arms[arm] = {
            "mse_call": mse_call,
            "mse_call_label": labels[mse_call],
            "change_error_call": change_call,
            "change_error_call_label": labels[change_call],
            "seeds": seeds,
        }
    return {
        "relative_band": ACTION_CONTROL_RELATIVE_BAND,
        "metric": "scene_macro_mse",
        "grid_action_adds_future_grid_information": arms["grid"]["mse_call"] == "action_lower",
        "arms": arms,
        "retrieval_recomputed": False,
        "evidence_grade": "不足",
        "evidence_reason": "开发集只有 8 个场景，点估计不能排除零收益",
        "development_is_blind_test": False,
        "navigation_connected": False,
        "stage_b_started": False,
        "stage_c_started": False,
        "stage_d_started": False,
    }


def stage_a_verdict(seed_rows: list[dict]) -> dict:
    """Apply the registered development-set rules. Eight scenes stay evidence-poor."""
    expected = (20261029, 20261030, 20261031)
    seeds = tuple(sorted({int(row["seed"]) for row in seed_rows}))
    if seeds != expected:
        raise ValueError(f"verdict expects seeds {expected}, got {seeds}")
    representation = []
    retrieval = []
    for seed in expected:
        rows = {row["arm"]: row for row in seed_rows if int(row["seed"]) == seed}
        if set(rows) != set(ARMS):
            raise ValueError(f"seed {seed} is missing an arm")
        spatial = rows["grid"]
        representation.append(
            float(spatial["ratio_to_copy"]) < 1.0
            and float(spatial["ratio_to_copy"]) < float(rows["pooler"]["ratio_to_copy"])
            and float(spatial["ratio_to_copy"]) < float(rows["mean"]["ratio_to_copy"])
        )
        retrieval.append(float(spatial["recall_delta"]) >= 0.02 and float(spatial["mrr_delta"]) > 0.0)
    return {
        "representation_target_met": all(representation),
        "retrieval_target_met": all(retrieval),
        "evidence_grade": "不足",
        "evidence_reason": "开发集只有 8 个场景，点估计不能排除零收益",
        "development_is_blind_test": False,
        "navigation_connected": False,
        "matched_action_control_trained": False,
        "stage_b_started": False,
        "stage_c_started": False,
        "stage_d_started": False,
    }
