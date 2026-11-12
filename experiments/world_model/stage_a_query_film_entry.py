"""Residual grid target with the command modulating every query.

The decoder depth, data, batch, optimizer, and 600-update budget match the
memory-token residual run. This entry does not encode images, start Habitat,
rewrite the feature cache, or connect a prediction to LightNav.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "output/world_model_training/optimization_20261029_01"))

from experiments.world_model.future_only import (  # noqa: E402
    FEATURE_DIM,
    TOKENS,
    FutureGridDecoder,
    batch_plan,
    residual_injection_verdict,
    stable_id_hash,
)
from experiments.world_model.stage_a_future_entry import (  # noqa: E402
    PATCH_DIR,
    attach_identity,
    gather,
)

OUT = REPO / "output/world_model_training/stage_a_residual_query_film_20261102_01"
PREVIOUS = REPO / "output/world_model_training/stage_a_residual_target_20261102_01/summary.json"
SEEDS = (20261029, 20261030, 20261031)
FIT_IDS = "aa720086a6def94afb35883e3693b0416a6265122a39066152d53482fc659083"
DEV_IDS = "599a46f319a13bebbb0cb0a93252cc07c9b8cdc39a77ee69a868005af31ba480"
UPDATES = 600
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 0.01


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_arrays() -> dict:
    from experiment_common import APPROVALS, KEY, RUNS, load_data

    from experiments.world_model.latent_data import load_accepted_run_set

    tensors, meta, _info = load_data("cpu")
    bundle = load_accepted_run_set(
        [(run, APPROVALS / f"{run.name}.approval.json", KEY) for run in RUNS]
    )
    observations = {row["observation_id"]: row for row in bundle["observations"]}
    transitions = {row["transition_id"]: row for row in bundle["transitions"]}
    for side in ("fit", "dev"):
        attach_identity(meta[side], transitions, observations)
    index = json.loads((PATCH_DIR / "patch_index.json").read_text(encoding="utf-8"))
    row_of = {observation_id: position for position, observation_id in enumerate(index)}
    mapping = np.memmap(
        PATCH_DIR / "patches.f32.memmap",
        dtype=np.float32,
        mode="r",
        shape=(len(index), TOKENS, FEATURE_DIM),
    )
    arrays = {}
    for side in ("fit", "dev"):
        before = np.asarray([row_of[row["obs_before_id"]] for row in meta[side]], dtype=np.int64)
        after = np.asarray([row_of[row["obs_after_id"]] for row in meta[side]], dtype=np.int64)
        current = gather(mapping, before)
        future = gather(mapping, after)
        if not np.isfinite(current).all() or not np.isfinite(future).all():
            raise RuntimeError(f"non-finite cached tokens in {side}")
        arrays[side] = {
            "current": current,
            "future": future,
            "control": tensors[side][1].numpy(),
            "scenes": [str(row["scene"]) for row in meta[side]],
            "groups": [str(row["action_kind"]) for row in meta[side]],
            "transition_ids": [str(row["transition_id"]) for row in meta[side]],
        }
    if stable_id_hash(arrays["fit"]["transition_ids"]) != FIT_IDS:
        raise RuntimeError("fit transition ids do not match the sealed Stage A split")
    if stable_id_hash(arrays["dev"]["transition_ids"]) != DEV_IDS:
        raise RuntimeError("development transition ids do not match the sealed Stage A split")
    if len(arrays["fit"]["current"]) != 1816 or len(arrays["dev"]["current"]) != 3249:
        raise RuntimeError("sample counts do not match the sealed Stage A split")
    return arrays


def channel_scale(fit: dict) -> np.ndarray:
    residual = fit["future"] - fit["current"]
    deviation = residual.reshape(-1, FEATURE_DIM).std(axis=0)
    return np.maximum(deviation, 0.01).astype(np.float32)


def score(model: FutureGridDecoder, data: dict, scale: torch.Tensor, device: str) -> dict:
    model.eval()
    errors = []
    zeros = []
    pred_norm = []
    target_norm = []
    cosine = []
    scale_np = scale.detach().cpu().numpy()
    with torch.inference_mode():
        for start in range(0, len(data["current"]), 8):
            end = min(start + 8, len(data["current"]))
            current = torch.from_numpy(data["current"][start:end]).to(device)
            control = torch.from_numpy(data["control"][start:end]).to(device)
            target = torch.from_numpy(data["future"][start:end] - data["current"][start:end]).to(device)
            prediction = model(current, control)
            normalized = (prediction - target) / scale.view(1, 1, -1)
            errors.append(normalized.square().mean(dim=(1, 2)).cpu().numpy())
            zeros.append(np.square(target.cpu().numpy() / scale_np[None, None, :]).mean(axis=(1, 2)))
            pred_norm.append(prediction.flatten(1).norm(dim=1).cpu().numpy())
            target_norm.append(target.flatten(1).norm(dim=1).cpu().numpy())
            cosine.append(
                torch.nn.functional.cosine_similarity(
                    prediction.flatten(1), target.flatten(1), dim=1
                ).cpu().numpy()
            )
    values = np.concatenate(errors).astype(np.float64)
    zero_values = np.concatenate(zeros).astype(np.float64)
    direction = np.concatenate(cosine).astype(np.float64)
    if not np.isfinite(values).all() or not np.isfinite(zero_values).all():
        raise RuntimeError("residual score is non-finite")
    finite_direction = direction[np.isfinite(direction)]
    direction_mean = None if len(finite_direction) == 0 else float(finite_direction.mean())
    scenes = data["scenes"]
    scene_scores = [
        float(values[np.asarray([scene == name for scene in scenes])].mean())
        for name in sorted(set(scenes))
    ]
    zero_scene_scores = [
        float(zero_values[np.asarray([scene == name for scene in scenes])].mean())
        for name in sorted(set(scenes))
    ]
    grouped = {}
    for group in sorted(set(data["groups"])):
        mask = np.asarray([value == group for value in data["groups"]])
        grouped[group] = {
            "n": int(mask.sum()),
            "normalized_residual_mse": float(values[mask].mean()),
            "zero_residual_mse": float(zero_values[mask].mean()),
        }
    return {
        "n": int(len(values)),
        "normalized_residual_mse": float(values.mean()),
        "zero_residual_mse": float(zero_values.mean()),
        "scene_macro_residual_mse": float(np.mean(scene_scores)),
        "scene_macro_zero_residual_mse": float(np.mean(zero_scene_scores)),
        "predicted_change_norm_mean": float(np.concatenate(pred_norm).mean()),
        "true_change_norm_mean": float(np.concatenate(target_norm).mean()),
        "change_direction_cosine_mean": direction_mean,
        "nonfinite_direction_count": int(len(direction) - len(finite_direction)),
        "by_action_group": grouped,
    }


def train_one(arrays: dict, *, use_action: bool, seed: int, gpu: int) -> None:
    if gpu < 2:
        raise RuntimeError(f"gpu {gpu} is reserved; this run uses idle cards from 2 upward")
    device = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    fit, dev = arrays["fit"], arrays["dev"]
    scale_np = channel_scale(fit)
    scale = torch.from_numpy(scale_np).to(device)
    model = FutureGridDecoder("grid", use_action=use_action, action_injection="query_film").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    batches = batch_plan(len(fit["current"]), BATCH, UPDATES, seed)
    started = time.perf_counter()
    model.train()
    for update, rows in enumerate(batches, 1):
        current = torch.from_numpy(fit["current"][rows]).to(device)
        control = torch.from_numpy(fit["control"][rows]).to(device)
        target = torch.from_numpy(fit["future"][rows] - fit["current"][rows]).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(current, control)
        loss = ((prediction - target) / scale.view(1, 1, -1)).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at update {update}")
        loss.backward()
        optimizer.step()
        if update % 100 == 0:
            print(
                f"gpu={gpu} action={use_action} seed={seed} update={update}/{UPDATES} "
                f"loss={loss.item():.6f}",
                flush=True,
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = score(model, dev, scale, device)
    name = f"grid_{'action' if use_action else 'no_action'}_{seed}"
    checkpoint = OUT / "checkpoints" / f"{name}.pt"
    payload = {
        "arm": "grid",
        "seed": seed,
        "updates": UPDATES,
        "use_action": use_action,
        "action_injection": "query_film",
        "target": "future_patch_grid - current_patch_grid",
        "normalization": "per-channel std of train_fit residual targets, floor 0.01",
        "channel_scale": scale_np.tolist(),
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "navigation_connected": False,
    }
    with checkpoint.open("xb") as handle:
        torch.save(payload, handle)
    record = {
        "arm": "grid",
        "seed": seed,
        "gpu": gpu,
        "use_action": use_action,
        "action_injection": "query_film",
        "updates": UPDATES,
        "elapsed_s": elapsed,
        "fit_transitions": len(fit["current"]),
        "dev_transitions": len(dev["current"]),
        "fit_transition_ids_sha256": stable_id_hash(fit["transition_ids"]),
        "dev_transition_ids_sha256": stable_id_hash(dev["transition_ids"]),
        "checkpoint": str(checkpoint),
        "metrics": metrics,
    }
    write_json(OUT / "runs" / f"{name}.json", record)
    print(
        f"done gpu={gpu} action={use_action} seed={seed} "
        f"scene_macro={metrics['scene_macro_residual_mse']:.6f} elapsed={elapsed:.1f}s",
        flush=True,
    )


def prepare() -> None:
    OUT.mkdir()
    (OUT / "checkpoints").mkdir()
    (OUT / "logs").mkdir()
    (OUT / "runs").mkdir()
    write_json(
        OUT / "protocol.json",
        {
            "experiment": "stage_a_residual_query_film",
            "target": "future_patch_grid - current_patch_grid",
            "action_injection": "query_film",
            "unchanged": [
                "grid decoder depth and width",
                "fit 1816 and development 3249",
                "batch 128",
                "AdamW lr 1e-3 weight_decay 0.01",
                "600 updates",
                "matched no-action control",
            ],
            "selection": "fixed 600 updates; development is not used to pick a checkpoint",
            "rule": "action scene-macro normalized residual MSE lower than no-action by >1% on every seed",
            "seeds": list(SEEDS),
            "fit_transition_ids_sha256": FIT_IDS,
            "dev_transition_ids_sha256": DEV_IDS,
            "development_is_blind_test": False,
            "retrieval_recomputed": False,
            "navigation_connected": False,
            "habitat_started": False,
            "formal_cache_written": False,
        },
    )


def _cosine(value: float | None) -> str:
    if value is None:
        return "未定义"
    return f"{value:.4f}"


def render_report(verdict: dict, rows: list[dict], previous: dict | None) -> str:
    by_seed = {(row["use_action"], row["seed"]): row for row in rows}
    lines = [
        "# 阶段 A 残差目标，命令调制每个 query",
        "",
        "监督目标仍是未来 patch 网格减去当前 patch 网格。",
        "相对记忆 token 那次残差对照，只改了动作注入：命令不再作为额外记忆 token，而是缩放并平移每一个 query。",
        "解码器宽度和深度、数据、batch、优化器、600 updates 和无动作对照不变。",
        "开发集已用于此前阶段 A，不是盲测。没有重算检索，没有启动 Habitat，没有写入正式特征缓存，也没有接入 LightNav。",
        "",
    ]
    if previous is not None:
        lines.append(
            f"上一次记忆 token 残差对照的判定是 `{previous['verdict']}`。"
            "这次只看换注入方式之后，动作是否越过同一条 1% 线。"
        )
        lines.append("")
    lines.extend(
        [
            f"判定：{verdict['call_label']}（`{verdict['call']}`）。",
            "门槛要求三个 seed 的有动作场景宏平均归一化残差 MSE 都比无动作低超过 1%。",
            "",
            "| seed | 有动作 MSE | 无动作 MSE | 相对差 | 有动作预测幅度 | 真实幅度 | 方向余弦 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in verdict["seeds"]:
        action = by_seed[(True, item["seed"])]["metrics"]
        lines.append(
            f"| {item['seed']} | {item['action_mse']:.6f} | {item['no_action_mse']:.6f} | "
            f"{100 * item['relative_mse']:.2f}% | {action['predicted_change_norm_mean']:.2f} | "
            f"{action['true_change_norm_mean']:.2f} | {_cosine(action['change_direction_cosine_mean'])} |"
        )
    lines.extend(
        [
            "",
            "无动作模型的预测幅度和方向余弦：",
            "",
            "| seed | 预测幅度 | 方向余弦 | 零残差场景宏 MSE |",
            "|---:|---:|---:|---:|",
        ]
    )
    for seed in SEEDS:
        metrics = by_seed[(False, seed)]["metrics"]
        lines.append(
            f"| {seed} | {metrics['predicted_change_norm_mean']:.2f} | "
            f"{_cosine(metrics['change_direction_cosine_mean'])} | "
            f"{metrics['scene_macro_zero_residual_mse']:.6f} |"
        )
    lines.extend(["", "## 动作组", "", "| seed | 条件 | 类型 | n | MSE | 零残差 MSE |", "|---:|---|---|---:|---:|---:|"])
    for use_action, label in ((True, "有动作"), (False, "无动作")):
        for seed in SEEDS:
            grouped = by_seed[(use_action, seed)]["metrics"]["by_action_group"]
            for group, metrics in grouped.items():
                lines.append(
                    f"| {seed} | {label} | {group} | {metrics['n']} | "
                    f"{metrics['normalized_residual_mse']:.6f} | {metrics['zero_residual_mse']:.6f} |"
                )
    if verdict["call"] == "action_lower":
        closing = (
            "三个 seed 都越过 1%。这只说明残差误差用到了命令，不是导航结果。"
            "不重算检索，不进入后续阶段。"
        )
    elif verdict["call"] == "within_band":
        closing = (
            "换注入方式之后仍在 1% 带内。残差目标和 query 调制都没有让动作稳定降低误差。"
            "按预先约定，停止这条 0.1 秒 SigLIP 网格路线，不再加容量、步数或场景。"
        )
    elif verdict["call"] == "no_action_lower":
        closing = "无动作误差更低。动作通道没有把残差预测得更准。这条网格路线不再加下一项改动。"
    else:
        closing = "三个 seed 的方向不一致。先不改结构，也不补数据。"
    lines.extend(["", closing, ""])
    return "\n".join(lines)


def aggregate() -> None:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((OUT / "runs").glob("grid_*.json"))]
    verdict = residual_injection_verdict(
        [
            {
                "seed": row["seed"],
                "use_action": row["use_action"],
                "scene_macro_residual_mse": row["metrics"]["scene_macro_residual_mse"],
            }
            for row in rows
        ]
    )
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8")) if PREVIOUS.is_file() else None
    summary = {
        "experiment": "stage_a_residual_query_film",
        "protocol": json.loads((OUT / "protocol.json").read_text(encoding="utf-8")),
        "verdict": verdict,
        "runs": rows,
        "previous_memory_token_residual": None
        if previous is None
        else {"path": str(PREVIOUS), "verdict": previous["verdict"]},
    }
    write_json(OUT / "summary.json", summary)
    with (OUT / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(render_report(verdict, rows, previous))
    print(json.dumps(verdict, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-action", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare:
        prepare()
        return
    if args.aggregate:
        aggregate()
        return
    if args.gpu is None or args.seed is None:
        parser.error("--gpu and --seed are required for a training run")
    if not torch.cuda.is_available():
        raise SystemExit("query-film residual training requires CUDA")
    try:
        train_one(load_arrays(), use_action=not args.no_action, seed=args.seed, gpu=args.gpu)
    except Exception as exc:
        name = f"grid_{'no_action' if args.no_action else 'action'}_{args.seed}"
        failure = OUT / "runs" / f"failure_{name}.json"
        if not failure.exists():
            write_json(failure, {"error": f"{type(exc).__name__}: {exc}", "gpu": args.gpu, "seed": args.seed})
        raise


if __name__ == "__main__":
    main()
