"""Future-only pooled, mean, and grid comparison.

This reuses the existing patch cache. It does not encode images, start Habitat,
or connect a prediction to LightNav. Development results are not a blind test.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "output/world_model_training/optimization_20261029_01"))

from experiment_common import APPROVALS, KEY, RUNS, load_data  # noqa: E402

from experiments.world_model.feature_cache import SIGLIP_REVISION  # noqa: E402
from experiments.world_model.future_only import (  # noqa: E402
    ACTION_CONTROL_RELATIVE_BAND,
    ARMS,
    D_MODEL,
    FEATURE_DIM,
    TOKENS,
    FutureGridDecoder,
    action_control_verdict,
    assert_split,
    assign_scene_folds,
    batch_plan,
    decoder_flops,
    fit_channel_scale,
    global_mean_descriptor,
    grid_prediction_metrics,
    label_timing,
    mean_change_prediction,
    normalized_grid_mse,
    parameter_count,
    parse_shard_spec,
    retrieval_evaluation,
    shard_slots,
    siglip_grid_report,
    spatial_descriptor,
    stable_id_hash,
    stage_a_verdict,
    synthetic_routing_check,
)
from experiments.world_model.latent_data import load_accepted_run_set  # noqa: E402

ACTION_OUT = REPO / "output/world_model_training/stage_a_future_only_20261030_01"
NO_ACTION_OUT = REPO / "output/world_model_training/stage_a_no_action_20261102_01"
OUT = ACTION_OUT
USE_ACTION = True
PATCH_DIR = REPO / "output/world_model_training/spatial_vs_pooler_20261030_01"
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots"
    / SIGLIP_REVISION
)
SEEDS = (20261029, 20261030, 20261031)
UPDATES = 600
BATCH = 128
EVAL_EVERY = 50
LR = 1e-3
WEIGHT_DECAY = 0.01
EXPECTED = {"fit": 1816, "development": 3249}
ARM_LABEL = {"pooler": "G 官方 pooler", "mean": "M patch 均值", "grid": "S 空间 token"}


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(plain(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def plain(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_torch(path: Path, payload: dict) -> None:
    with path.open("xb") as handle:
        torch.save(payload, handle)


def memory_tensor(arm: str, arrays: dict, rows: np.ndarray) -> torch.Tensor:
    if arm == "pooler":
        values = arrays["pooler"][rows][:, None, :]
    elif arm == "mean":
        values = arrays["mean"][rows][:, None, :]
    elif arm == "grid":
        values = arrays["current"][rows]
    else:
        raise ValueError(arm)
    if arm != "grid" and values.shape[1:] != (1, FEATURE_DIM):
        raise ValueError("a global arm received a patch grid")
    return torch.from_numpy(np.ascontiguousarray(values))


def predict(model: FutureGridDecoder, arm: str, rows: np.ndarray, arrays: dict, device: str) -> np.ndarray:
    model.eval()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            part = rows[start : start + 16]
            memory = memory_tensor(arm, arrays, part).to(device)
            control = torch.from_numpy(np.ascontiguousarray(arrays["control"][part])).to(device)
            pieces.append(model(memory, control).float().cpu())
    model.train()
    predicted = torch.cat(pieces).numpy()
    if predicted.shape != (len(rows), TOKENS, FEATURE_DIM):
        raise RuntimeError(f"unexpected prediction shape {predicted.shape}")
    return predicted


def train_arm(
    arm: str,
    seed: int,
    train_rows: np.ndarray,
    val_rows: np.ndarray | None,
    arrays: dict,
    scenes: list[str] | None,
    device: str,
    selected_update: int | None,
    use_action: bool,
) -> dict:
    scale_np = fit_channel_scale(arrays["future"][train_rows])
    scale = torch.from_numpy(scale_np).to(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = FutureGridDecoder(arm, use_action=use_action).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    plan = batch_plan(len(train_rows), BATCH, UPDATES, seed)
    history = []
    selected_state = None
    started = time.perf_counter()
    model.train()
    for step, local in enumerate(plan, start=1):
        rows = train_rows[local]
        memory = memory_tensor(arm, arrays, rows).to(device)
        control = torch.from_numpy(np.ascontiguousarray(arrays["control"][rows])).to(device)
        target = torch.from_numpy(np.ascontiguousarray(arrays["future"][rows])).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = normalized_grid_mse(model(memory, control), target, scale)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{arm} seed {seed} produced a non-finite loss at update {step}")
        loss.backward()
        optimizer.step()
        if selected_update is not None and step == selected_update:
            selected_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if step % EVAL_EVERY != 0:
            continue
        row = {"update": step, "batch_loss": float(loss.detach().cpu())}
        if val_rows is not None and scenes is not None:
            predicted = predict(model, arm, val_rows, arrays, device)
            metrics = grid_prediction_metrics(
                predicted, arrays["future"][val_rows], arrays["current"][val_rows], scenes
            )
            copy = grid_prediction_metrics(
                arrays["current"][val_rows],
                arrays["future"][val_rows],
                arrays["current"][val_rows],
                scenes,
            )
            changed = mean_change_prediction(
                arrays["current"][train_rows], arrays["future"][train_rows], arrays["current"][val_rows]
            )
            drift = grid_prediction_metrics(changed, arrays["future"][val_rows], arrays["current"][val_rows], scenes)
            row.update(
                {
                    "scene_macro_mse": metrics["scene_macro_mse"],
                    "copy_scene_macro_mse": copy["scene_macro_mse"],
                    "ratio_to_copy": metrics["scene_macro_mse"] / copy["scene_macro_mse"],
                    "mean_change_scene_macro_mse": drift["scene_macro_mse"],
                    "token_cosine": metrics["token_cosine"],
                }
            )
            del predicted
            name = arm if use_action else f"{arm} no-action"
            print(
                f"{name} seed {seed} update {step} ratio {row['ratio_to_copy']:.4f}",
                flush=True,
            )
        history.append(row)
    if selected_update is not None and selected_state is None:
        raise RuntimeError(f"selected update {selected_update} was not saved")
    return {
        "model": model,
        "history": history,
        "scale": scale_np,
        "selected_state": selected_state,
        "elapsed_s": time.perf_counter() - started,
    }


def choose_updates(histories: list[dict]) -> dict[str, int]:
    chosen = {}
    for arm in ARMS:
        grouped: dict[int, list[float]] = {}
        for item in histories:
            if item["arm"] != arm:
                continue
            for row in item["history"]:
                grouped.setdefault(int(row["update"]), []).append(float(row["ratio_to_copy"]))
        if not grouped or any(len(values) != 9 for values in grouped.values()):
            raise RuntimeError(f"{arm} fold scores are not 3 folds by 3 seeds")
        update = min(grouped, key=lambda step: (float(np.mean(grouped[step])), step))
        chosen[arm] = int(update)
    return chosen


def load_head(device: str) -> torch.nn.Module:
    from transformers import SiglipVisionModel

    vision = SiglipVisionModel.from_pretrained(SNAPSHOT, local_files_only=True, torch_dtype=torch.float32)
    head = vision.head.to(device).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    del vision
    torch.cuda.empty_cache()
    return head


def pool_tokens(head: torch.nn.Module, tokens: np.ndarray, device: str) -> np.ndarray:
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(tokens), 8):
            batch = torch.from_numpy(np.ascontiguousarray(tokens[start : start + 8])).to(device)
            pieces.append(head(batch).float().cpu())
    return torch.cat(pieces).numpy()


def gather(mapping: np.memmap, ids: np.ndarray) -> np.ndarray:
    output = np.empty((len(ids), TOKENS, FEATURE_DIM), dtype=np.float32)
    for start in range(0, len(ids), 32):
        part = ids[start : start + 32]
        output[start : start + len(part)] = np.asarray(mapping[part])
    return output


def attach_identity(meta: list[dict], transitions: dict, observations: dict) -> None:
    for row in meta:
        transition = transitions[row["transition_id"]]
        if transition["obs_after_id"] != row["obs_after_id"]:
            raise RuntimeError(f"after-frame id mismatch for {row['transition_id']}")
        after = observations[transition["obs_after_id"]]["rgb"]
        row["obs_before_id"] = transition["obs_before_id"]
        row["rgb_pixel_sha256"] = str(after["pixel_sha256"])
        row["rgb_file_sha256"] = str(after["sha256"])


def zero_rows(array: np.ndarray) -> int:
    return int(np.all(array.reshape(len(array), -1) == 0, axis=1).sum())


def smoke(device: str, use_action: bool) -> dict:
    report = {}
    for arm in ARMS:
        torch.manual_seed(0)
        model = FutureGridDecoder(arm, use_action=use_action).to(device)
        width = TOKENS if arm == "grid" else 1
        memory = torch.randn(BATCH, width, FEATURE_DIM, device=device)
        control = torch.zeros(BATCH, 3, device=device)
        target = torch.randn(BATCH, TOKENS, FEATURE_DIM, device=device)
        scale = torch.ones(FEATURE_DIM, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = normalized_grid_mse(model(memory, control), target, scale)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        repeats = 3
        for _ in range(repeats):
            optimizer.zero_grad(set_to_none=True)
            loss = normalized_grid_mse(model(memory, control), target, scale)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        report[arm] = {
            "parameters": parameter_count(arm, use_action=use_action),
            "flops_one_sample": decoder_flops(arm, 1),
            "seconds_per_update_batch_128": (time.perf_counter() - started) / repeats,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "loss_finite": bool(torch.isfinite(loss).item()),
        }
        del model, memory, target, loss
        torch.cuda.empty_cache()
    return report


def evaluate_development(
    model_state: dict,
    arm: str,
    arrays: dict,
    meta: list[dict],
    device: str,
    head,
    use_action: bool,
) -> dict:
    model = FutureGridDecoder(arm, use_action=use_action).to(device)
    model.load_state_dict(model_state)
    rows = np.arange(len(meta))
    predicted = predict(model, arm, rows, arrays, device)
    scenes = [row["scene"] for row in meta]
    metrics = grid_prediction_metrics(predicted, arrays["future"], arrays["current"], scenes)
    pooled_predicted = pool_tokens(head, predicted, device)
    result = {
        "metrics": metrics,
        "pooled_mse_vs_frozen_head_target": float(np.mean((pooled_predicted - arrays["pooled_future"]) ** 2)),
    }
    if use_action:
        result["spatial_predicted"] = spatial_descriptor(predicted)
        result["global_predicted"] = global_mean_descriptor(predicted)
    del predicted, model
    torch.cuda.empty_cache()
    return result


def render_report(summary: dict) -> str:
    verdict = summary["verdict"]
    lines = [
        "# 阶段 A future-only 表示对照",
        "",
        "这是开发集上的 future-only 对照，不是盲测，也没有接入 LightNav。",
        "位姿头、短历史、深度、运动受限概率和匹配的无动作训练都没有跑。",
        "",
        "## 验证",
        "",
        f"- 网格：{summary['grid']['grid']}×{summary['grid']['grid']}，"
        f"卷积后图像边缘有 {summary['grid']['uncovered_pixels_after_last_patch']} 个像素不被任何 patch 覆盖。",
        "- 坐标是处理后图像平面的行、列索引，不是三维世界坐标。缓存 token 里已经包含 SigLIP 自己的位置嵌入。",
        f"- fit / development：{summary['counts']['fit_transitions']} / {summary['counts']['development_transitions']}。",
        f"- transition id sha256：`{summary['hashes']['fit_transition_ids']}`，"
        f"`{summary['hashes']['development_transition_ids']}`。",
        "- 三折按场景 id 排序后取模，训练折和验证折没有共享 scene 或 route。",
        f"- 冻结 pooling head 与已审计 pooler 的最大绝对差：{summary['head_check']['max_abs_diff']:.3e}。",
        "",
        "## 预算",
        "",
        f"可训练参数 {summary['smoke']['grid']['parameters']}，三组相同。"
        " 单样本 FLOPs（一次乘加算两次）"
        f" G/M {summary['smoke']['pooler']['flops_one_sample']}，"
        f"S {summary['smoke']['grid']['flops_one_sample']}。",
        "batch 128 的测得更新时间："
        + "，".join(
            f"{ARM_LABEL[arm]} {summary['smoke'][arm]['seconds_per_update_batch_128']:.3f} 秒"
            for arm in ARMS
        )
        + "。显存峰值："
        + "，".join(
            f"{ARM_LABEL[arm]} {summary['smoke'][arm]['peak_memory_bytes'] / (1024 ** 3):.2f} GiB"
            for arm in ARMS
        )
        + "。",
        "",
        "每个表示先在 fit 的三折上用三个 seed 训练，按场景宏平均 MSE 相对复制当前帧的比值选择更新数。"
        " 全 fit 再训练时，开发集只在这个预先选定的更新上评估。",
        "",
        "| 表示 | 选定更新 |",
        "| --- | ---: |",
    ]
    for arm in ARMS:
        lines.append(f"| {ARM_LABEL[arm]} | {summary['selected_updates'][arm]} |")
    lines.extend(["", "## 开发集未来网格", ""])
    lines.append("| 表示 | seed | 场景宏平均 MSE | 相对 copy | token cosine | 预测→未来 | 当前→未来 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["development"]:
        metrics = row["metrics"]
        lines.append(
            f"| {ARM_LABEL[row['arm']]} | {row['seed']} | {metrics['scene_macro_mse']:.4f} | "
            f"{row['ratio_to_copy']:.4f} | {metrics['token_cosine']:.4f} | "
            f"{metrics['cosine_prediction_to_future']:.4f} | {metrics['cosine_current_to_future']:.4f} |"
        )
    copy = summary["baselines"]["copy"]
    drift = summary["baselines"]["mean_change"]
    lines.extend(
        [
            "",
            f"复制当前帧的场景宏平均 MSE 是 {copy['scene_macro_mse']:.4f}。"
            f" fit 平均变化基线是 {drift['scene_macro_mse']:.4f}。",
            "G/M 的输入里没有当前 patch 网格；复制基线和平均变化基线在评估时看得到当前网格。",
            "",
            "## 检索",
            "",
            "主描述符是 3×3 块均值。下表是同一套候选、同一种 RGB 像素哈希去重。"
            " 全局均值描述符只作参考，不与主描述符的 Recall 相减。",
            "",
            "| 表示 | seed | 当前 Recall@2 | 预测 Recall@2 | 当前 MRR | 预测 MRR |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["development"]:
        spatial = row["spatial_retrieval"]["methods"]
        lines.append(
            f"| {ARM_LABEL[row['arm']]} | {row['seed']} | "
            f"{100 * spatial['current']['recall_at_2']:.2f}% | "
            f"{100 * spatial['predicted']['recall_at_2']:.2f}% | "
            f"{spatial['current']['mrr_oracle_top1']:.4f} | "
            f"{spatial['predicted']['mrr_oracle_top1']:.4f} |"
        )
    lines.extend(["", "## 判定", ""])
    lines.append(
        "表示数值目标"
        + ("达到。" if verdict["representation_target_met"] else "没有达到。")
        + "检索数值目标"
        + ("达到。" if verdict["retrieval_target_met"] else "没有达到。")
    )
    lines.append(f"证据等级：{verdict['evidence_grade']}。{verdict['evidence_reason']}。")
    lines.append("阶段 B、C、D 没有因为这次对照而启动。匹配的无动作训练对照也没有启动。")
    lines.append("")
    lines.append(summary["attribution"])
    lines.append("")
    return "\n".join(lines)


def attribution(development: list[dict]) -> str:
    spatial = [row for row in development if row["arm"] == "grid"]
    closer_to_current = all(
        row["metrics"]["cosine_prediction_to_current"] >= row["metrics"]["cosine_prediction_to_future"]
        for row in spatial
    )
    beats_copy = all(row["ratio_to_copy"] < 1 for row in spatial)
    if beats_copy and closer_to_current:
        return (
            "S 的网格 MSE 低于复制当前帧，但预测 token 仍不比当前帧更接近真实未来。"
            " 这部分误差下降更像把当前网格重建到未来网格上，不能当成动作后果已经学到。"
        )
    if beats_copy:
        return "S 的网格误差低于复制当前帧，并且预测 token 比当前帧更接近真实未来。这只说明未来特征预测有收益。"
    return "S 没有在三个 seed 上都低于复制当前帧。网格 MSE 没有给出继续扩展模型的理由。"


def load_action_reference(path: Path, hashes: dict) -> dict:
    if not path.is_file():
        raise RuntimeError(f"action reference is missing: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("navigation_connected") is not False:
        raise RuntimeError("action reference is marked connected to navigation")
    if summary["development_is_blind_test"] is not False:
        raise RuntimeError("action reference is marked as a blind test")
    if summary["verdict"]["matched_action_control_trained"] is not False:
        raise RuntimeError("action reference is not the with-action run")
    for key in ("fit_transition_ids", "development_transition_ids", "patch_index"):
        if summary["hashes"][key] != hashes[key]:
            raise RuntimeError(f"action reference {key} does not match this run")
    rows = []
    for row in summary["development"]:
        metrics = row["metrics"]
        rows.append(
            {
                "arm": row["arm"],
                "seed": int(row["seed"]),
                "scene_macro_mse": float(metrics["scene_macro_mse"]),
                "change_vector_error": float(metrics["change_vector_error"]),
                "direction_cosine_mean": float(metrics["direction_cosine_mean"]),
                "token_cosine": float(metrics["token_cosine"]),
                "ratio_to_copy": float(row["ratio_to_copy"]),
                "selected_update": int(row["selected_update"]),
            }
        )
    return {
        "rows": rows,
        "copy_scene_macro_mse": float(summary["baselines"]["copy"]["scene_macro_mse"]),
        "selected_updates": {arm: int(summary["selected_updates"][arm]) for arm in ARMS},
    }


def control_sentence(verdict: dict) -> str:
    grid = verdict["arms"]["grid"]
    call = grid["mse_call"]
    sentences = {
        "action_lower": (
            "空间 token 的有动作模型在三个 seed 上的场景宏平均 MSE 都低于无动作，"
            "而且相对差距都超过 1%。动作通道带了可预测的未来网格信息。"
        ),
        "no_action_lower": (
            "空间 token 关掉动作之后，三个 seed 的场景宏平均 MSE 都更低，相对差距都超过 1%。"
            "动作输入没有改善这份未来网格误差。"
        ),
        "within_band": (
            "空间 token 的有动作和无动作场景宏平均 MSE 在三个 seed 上都落在 1% 带内。"
            "这份误差下降不能记成动作后果。"
        ),
        "mixed": (
            "空间 token 的有动作和无动作差距在三个 seed 上方向不一致。"
            "动作通道没有稳定的未来网格误差收益。"
        ),
    }
    text = sentences[call]
    if grid["change_error_call"] != call:
        text += f" 变化向量误差的判定是{grid['change_error_call_label']}，与 MSE 判定不一致。"
    return text + " 检索没有重算。阶段 B、C、D 不因此启动。"


def render_no_action_report(summary: dict) -> str:
    verdict = summary["verdict"]
    lines = [
        "# 阶段 A 匹配的无动作对照",
        "",
        "动作通路在训练和推断时都是同一个全零的 6 维向量。参数量与有动作模型相同。",
        "这是开发集上的对照，不是盲测，也没有接入 LightNav。检索没有重算。",
        "有动作的数字来自已经完成的 future-only 运行，只读。",
    ]
    if summary.get("shard_count"):
        lines.append(
            f"36 次训练拆到 {summary['shard_count']} 张空闲 GPU 上，每次训练仍在一张卡上完成。"
        )
    lines.extend(
        [
            "",
            "## 验证",
        "",
        f"- fit / development：{summary['counts']['fit_transitions']} / {summary['counts']['development_transitions']}。",
        f"- transition id sha256：`{summary['hashes']['fit_transition_ids']}`，"
        f"`{summary['hashes']['development_transition_ids']}`。",
        "- 样本哈希与有动作那次一致。更新数按 fit 内的三折重选，不抄有动作运行的步数。",
        f"- 相对差距 = (无动作误差 - 有动作误差) / 有动作误差。带宽 {verdict['relative_band']:.0%}。",
        "",
        "## 选定更新",
        "",
        "| 表示 | 有动作 | 无动作 |",
        "| --- | ---: | ---: |",
        ]
    )
    reference_updates = summary["action_reference_updates"]
    for arm in ARMS:
        lines.append(
            f"| {ARM_LABEL[arm]} | {reference_updates[arm]} | {summary['selected_updates'][arm]} |"
        )
    lines.extend(
        [
            "",
            "## 开发集场景宏平均 MSE",
            "",
            "| 表示 | seed | 有动作 MSE | 无动作 MSE | 相对差距 | 有动作变化误差 | 无动作变化误差 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ARMS:
        for row in verdict["arms"][arm]["seeds"]:
            lines.append(
                f"| {ARM_LABEL[arm]} | {row['seed']} | {row['action_mse']:.4f} | "
                f"{row['no_action_mse']:.4f} | {100 * row['relative_mse']:.2f}% | "
                f"{row['action_change_error']:.1f} | {row['no_action_change_error']:.1f} |"
            )
    lines.extend(["", "## 判定", ""])
    for arm in ARMS:
        arm_verdict = verdict["arms"][arm]
        lines.append(
            f"- {ARM_LABEL[arm]}：MSE {arm_verdict['mse_call_label']}；"
            f"变化向量误差 {arm_verdict['change_error_call_label']}。"
        )
    lines.append(f"证据等级：{verdict['evidence_grade']}。{verdict['evidence_reason']}。")
    lines.append("")
    lines.append(summary["attribution"])
    lines.append("")
    return "\n".join(lines)


def _main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("this comparison requires CUDA")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    device = "cuda:0"
    if not USE_ACTION:
        print("matched no-action control; action features stay zero", flush=True)
    print("smoke forward and backward", flush=True)
    smoke_report = smoke(device, USE_ACTION)
    routing = synthetic_routing_check()
    print("load admitted rows and cached tokens", flush=True)
    tensors, meta, _info = load_data("cpu")
    bundle = load_accepted_run_set([(run, APPROVALS / f"{run.name}.approval.json", KEY) for run in RUNS])
    observations = {row["observation_id"]: row for row in bundle["observations"]}
    transitions = {row["transition_id"]: row for row in bundle["transitions"]}
    for side in ("fit", "dev"):
        attach_identity(meta[side], transitions, observations)
    counts = assert_split(meta["fit"], meta["dev"])
    if counts["fit_transitions"] != EXPECTED["fit"] or counts["development_transitions"] != EXPECTED["development"]:
        raise RuntimeError(f"unexpected split counts {counts}")
    timing = {"fit": label_timing(meta["fit"]), "development": label_timing(meta["dev"])}
    folds = assign_scene_folds(meta["fit"])
    index = json.loads((PATCH_DIR / "patch_index.json").read_text(encoding="utf-8"))
    row_of = {observation_id: position for position, observation_id in enumerate(index)}
    mapping = np.memmap(
        PATCH_DIR / "patches.f32.memmap", dtype=np.float32, mode="r", shape=(len(index), TOKENS, FEATURE_DIM)
    )
    arrays = {}
    for side in ("fit", "dev"):
        before = np.array([row_of[row["obs_before_id"]] for row in meta[side]])
        after = np.array([row_of[row["obs_after_id"]] for row in meta[side]])
        current = gather(mapping, before)
        future = gather(mapping, after)
        if not np.isfinite(current).all() or not np.isfinite(future).all():
            raise RuntimeError(f"{side} cached tokens are non-finite and were not replaced")
        pooler = tensors[side][0].numpy()
        pooler_future = tensors[side][2].numpy()
        arrays[side] = {
            "current": current,
            "future": future,
            "mean": current.mean(axis=1),
            "pooler": pooler,
            "pooler_future": pooler_future,
            "control": tensors[side][1].numpy(),
        }
    separation = float(np.max(np.abs(arrays["fit"]["pooler"] - arrays["fit"]["mean"])))
    if separation < 1e-3:
        raise RuntimeError("official pooler and patch mean are not distinct inputs")
    head = load_head(device)
    probe_at = np.linspace(0, len(meta["fit"]) - 1, 8).astype(int)
    current_probe = pool_tokens(head, arrays["fit"]["current"][probe_at], device)
    future_probe = pool_tokens(head, arrays["fit"]["future"][probe_at], device)
    current_diff = float(np.max(np.abs(current_probe - arrays["fit"]["pooler"][probe_at])))
    future_diff = float(np.max(np.abs(future_probe - arrays["fit"]["pooler_future"][probe_at])))
    head_diff = max(current_diff, future_diff)
    if head_diff > 1e-4:
        raise RuntimeError(f"frozen pooling head does not match the audited pooler ({head_diff})")
    pooled_future = pool_tokens(head, arrays["dev"]["future"], device)
    arrays["dev"]["pooled_future"] = pooled_future
    verification = {
        "grid": siglip_grid_report(),
        "routing": routing,
        "counts": counts,
        "timing": timing,
        "folds": folds,
        "zero_grids": {
            side: {"current": zero_rows(arrays[side]["current"]), "future": zero_rows(arrays[side]["future"])}
            for side in ("fit", "dev")
        },
        "pooler_minus_mean_max_abs": separation,
        "head_check": {
            "samples": 8,
            "current_max_abs_diff": current_diff,
            "future_max_abs_diff": future_diff,
            "max_abs_diff": head_diff,
            "threshold": 1e-4,
        },
        "hashes": {
            "fit_transition_ids": stable_id_hash([row["transition_id"] for row in meta["fit"]]),
            "development_transition_ids": stable_id_hash([row["transition_id"] for row in meta["dev"]]),
            "patch_index": stable_id_hash(index),
        },
        "smoke": smoke_report,
        "decoder": {"d_model": D_MODEL, "layers": 2, "heads": 4, "batch": BATCH, "updates": UPDATES},
        "action": "executed final_habitat_action v_mps, w_rad_s, dt_command_s; dt_sim is not an input",
        "matched_action_control_trained": False,
        "navigation_connected": False,
        "development_is_blind_test": False,
        "device": torch.cuda.get_device_name(0),
    }
    if not USE_ACTION:
        action_reference = load_action_reference(ACTION_OUT / "results.json", verification["hashes"])
        dev_scenes_check = [row["scene"] for row in meta["dev"]]
        copy_check = grid_prediction_metrics(
            arrays["dev"]["current"],
            arrays["dev"]["future"],
            arrays["dev"]["current"],
            dev_scenes_check,
        )
        copy_gap = abs(copy_check["scene_macro_mse"] - action_reference["copy_scene_macro_mse"])
        if copy_gap > 1e-4:
            raise RuntimeError(f"copy baseline drifted from the action run by {copy_gap}")
        verification["matched_action_control_trained"] = True
        verification["action"] = "matched control: the 6 action features are zeros at train and infer"
        verification["retrieval_recomputed"] = False
        verification["relative_mse_band"] = ACTION_CONTROL_RELATIVE_BAND
        verification["selection"] = (
            "same 3-fold scene/route rule inside fit; the update count is not copied from the action run"
        )
        verification["action_reference"] = {
            "directory": str(ACTION_OUT),
            "results": "results.json",
            "selected_updates": action_reference["selected_updates"],
            "copy_scene_macro_mse": action_reference["copy_scene_macro_mse"],
            "rows": action_reference["rows"],
            "rule": (
                "relative = (no_action - action) / action; "
                "action_lower requires every seed relative > 0.01; "
                "no_action_lower requires every seed relative < -0.01; "
                "within_band requires every absolute relative <= 0.01; otherwise mixed. "
                "grid_action_adds_future_grid_information follows grid scene-macro MSE only."
            ),
        }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "folds").mkdir()
    (OUT / "full").mkdir()
    write_json(OUT / "verification.json", verification)
    sealed = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    if sealed["folds"]["folds"] != plain(folds)["folds"]:
        raise RuntimeError("sealed fold table was not reread")
    histories = []
    for fold in folds["folds"]:
        train_rows = np.array(
            [i for i, row in enumerate(meta["fit"]) if row["scene"] in set(fold["train_scenes"])], dtype=np.int64
        )
        val_rows = np.array(
            [i for i, row in enumerate(meta["fit"]) if row["scene"] in set(fold["validation_scenes"])],
            dtype=np.int64,
        )
        if set(train_rows).intersection(val_rows):
            raise RuntimeError("a fold overlaps itself")
        val_scenes = [meta["fit"][int(index)]["scene"] for index in val_rows]
        for seed in SEEDS:
            for arm in ARMS:
                tag = arm if USE_ACTION else f"{arm} no-action"
                print(f"train {tag} fold {fold['fold']} seed {seed}", flush=True)
                result = train_arm(
                    arm, seed, train_rows, val_rows, arrays["fit"], val_scenes, device, None, USE_ACTION
                )
                record = {
                    "arm": arm,
                    "seed": seed,
                    "fold": fold["fold"],
                    "use_action": USE_ACTION,
                    "history": result["history"],
                    "elapsed_s": result["elapsed_s"],
                }
                histories.append(record)
                write_json(OUT / "folds" / f"{arm}_fold{fold['fold']}_seed{seed}.json", record)
                del result
                torch.cuda.empty_cache()
    selected = choose_updates(histories)
    write_json(OUT / "selected_updates.json", selected)
    reread_updates = json.loads((OUT / "selected_updates.json").read_text(encoding="utf-8"))
    development = []
    fit_rows = np.arange(len(meta["fit"]))
    dev_scenes = [row["scene"] for row in meta["dev"]]
    copy_metrics = grid_prediction_metrics(
        arrays["dev"]["current"], arrays["dev"]["future"], arrays["dev"]["current"], dev_scenes
    )
    drift_prediction = mean_change_prediction(
        arrays["fit"]["current"], arrays["fit"]["future"], arrays["dev"]["current"]
    )
    drift_metrics = grid_prediction_metrics(
        drift_prediction, arrays["dev"]["future"], arrays["dev"]["current"], dev_scenes
    )
    if USE_ACTION:
        spatial_bank = spatial_descriptor(arrays["dev"]["future"])
        spatial_current = spatial_descriptor(arrays["dev"]["current"])
        global_bank = global_mean_descriptor(arrays["dev"]["future"])
        global_current = global_mean_descriptor(arrays["dev"]["current"])
    for seed in SEEDS:
        for arm in ARMS:
            tag = arm if USE_ACTION else f"{arm} no-action"
            print(f"full fit {tag} seed {seed} keep update {reread_updates[arm]}", flush=True)
            result = train_arm(
                arm,
                seed,
                fit_rows,
                None,
                arrays["fit"],
                None,
                device,
                int(reread_updates[arm]),
                USE_ACTION,
            )
            payload = {
                "arm": arm,
                "seed": seed,
                "update": int(reread_updates[arm]),
                "model_state_dict": result["selected_state"],
                "channel_scale": result["scale"],
                "use_action": USE_ACTION,
                "navigation_connected": False,
            }
            save_torch(OUT / "full" / f"{arm}_{seed}_selected.pt", payload)
            payload["update"] = UPDATES
            payload["model_state_dict"] = {
                key: value.detach().cpu().clone() for key, value in result["model"].state_dict().items()
            }
            save_torch(OUT / "full" / f"{arm}_{seed}_last.pt", payload)
            evaluated = evaluate_development(
                result["selected_state"], arm, arrays["dev"], meta["dev"], device, head, USE_ACTION
            )
            ratio = evaluated["metrics"]["scene_macro_mse"] / copy_metrics["scene_macro_mse"]
            record = {
                "arm": arm,
                "seed": seed,
                "selected_update": int(reread_updates[arm]),
                "metrics": evaluated["metrics"],
                "ratio_to_copy": ratio,
                "pooled_mse_vs_frozen_head_target": evaluated["pooled_mse_vs_frozen_head_target"],
                "use_action": USE_ACTION,
                "train_elapsed_s": result["elapsed_s"],
            }
            if USE_ACTION:
                spatial = retrieval_evaluation(
                    {
                        "current": spatial_current,
                        "predicted": evaluated["spatial_predicted"],
                        "future_oracle": spatial_bank,
                    },
                    spatial_bank,
                    meta["dev"],
                )
                reference = retrieval_evaluation(
                    {
                        "current": global_current,
                        "predicted": evaluated["global_predicted"],
                        "future_oracle": global_bank,
                    },
                    global_bank,
                    meta["dev"],
                )
                if spatial["methods"]["future_oracle"]["recall_at_2"] != 1.0:
                    raise RuntimeError("future-oracle recall is not 1 on the shared descriptor")
                record["spatial_retrieval"] = spatial
                record["global_mean_retrieval"] = reference
            development.append(record)
            write_json(OUT / "full" / f"{arm}_{seed}_development.json", record)
            del result, evaluated
            torch.cuda.empty_cache()
    if USE_ACTION:
        verdict_rows = []
        for row in development:
            methods = row["spatial_retrieval"]["methods"]
            verdict_rows.append(
                {
                    "arm": row["arm"],
                    "seed": row["seed"],
                    "ratio_to_copy": row["ratio_to_copy"],
                    "recall_delta": methods["predicted"]["recall_at_2"] - methods["current"]["recall_at_2"],
                    "mrr_delta": methods["predicted"]["mrr_oracle_top1"] - methods["current"]["mrr_oracle_top1"],
                }
            )
        verdict = stage_a_verdict(verdict_rows)
        attribution_text = attribution(development)
    else:
        no_action_rows = [
            {
                "arm": row["arm"],
                "seed": row["seed"],
                "scene_macro_mse": row["metrics"]["scene_macro_mse"],
                "change_vector_error": row["metrics"]["change_vector_error"],
            }
            for row in development
        ]
        verdict = action_control_verdict(sealed["action_reference"]["rows"], no_action_rows)
        attribution_text = control_sentence(verdict)
    summary = {
        "grid": verification["grid"],
        "counts": counts,
        "hashes": verification["hashes"],
        "head_check": verification["head_check"],
        "smoke": smoke_report,
        "selected_updates": reread_updates,
        "baselines": {"copy": copy_metrics, "mean_change": drift_metrics},
        "development": development,
        "verdict": verdict,
        "attribution": attribution_text,
        "navigation_connected": False,
        "development_is_blind_test": False,
        "retrieval_recomputed": USE_ACTION,
    }
    if not USE_ACTION:
        summary["action_reference_updates"] = sealed["action_reference"]["selected_updates"]
    write_json(OUT / "results.json", summary)
    report = render_report(summary) if USE_ACTION else render_no_action_report(summary)
    with (OUT / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps(plain(verdict), ensure_ascii=False), flush=True)


def fold_jobs(meta_fit: list[dict], folds: dict) -> list[dict]:
    jobs = []
    for fold in folds["folds"]:
        train_rows = np.array(
            [i for i, row in enumerate(meta_fit) if row["scene"] in set(fold["train_scenes"])],
            dtype=np.int64,
        )
        val_rows = np.array(
            [i for i, row in enumerate(meta_fit) if row["scene"] in set(fold["validation_scenes"])],
            dtype=np.int64,
        )
        if set(train_rows).intersection(val_rows):
            raise RuntimeError("a fold overlaps itself")
        val_scenes = [meta_fit[int(index)]["scene"] for index in val_rows]
        for seed in SEEDS:
            for arm in ARMS:
                jobs.append(
                    {
                        "fold": int(fold["fold"]),
                        "seed": int(seed),
                        "arm": arm,
                        "train_rows": train_rows,
                        "val_rows": val_rows,
                        "val_scenes": val_scenes,
                    }
                )
    return jobs


def fold_record_ready(path: Path, arm: str, seed: int, fold: int) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    updates = [int(row["update"]) for row in record.get("history", []) if "ratio_to_copy" in row]
    return (
        record.get("arm") == arm
        and int(record.get("seed", -1)) == seed
        and int(record.get("fold", -1)) == fold
        and record.get("use_action") is False
        and updates == list(range(EVAL_EVERY, UPDATES + 1, EVAL_EVERY))
    )


def development_record_ready(path: Path, arm: str, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return False
    mse = metrics.get("scene_macro_mse")
    change = metrics.get("change_vector_error")
    return (
        record.get("arm") == arm
        and int(record.get("seed", -1)) == seed
        and record.get("use_action") is False
        and isinstance(mse, (int, float))
        and isinstance(change, (int, float))
        and float(mse) > 0.0
        and float(change) > 0.0
    )


def wait_for_files(paths: list[Path], label: str) -> None:
    while True:
        missing = [path.name for path in paths if not path.is_file()]
        if not missing:
            return
        print(f"waiting for {len(missing)} {label}", flush=True)
        time.sleep(20)


def _main_shard(shard: int, shards: int) -> None:
    """Train a disjoint slice of the sealed no-action run. One process uses one GPU."""
    if USE_ACTION:
        raise RuntimeError("shards are only for the matched no-action control")
    if not torch.cuda.is_available():
        raise SystemExit("this comparison requires CUDA")
    if (OUT / "results.json").is_file():
        print("results already written", flush=True)
        return
    sealed_path = OUT / "verification.json"
    if not sealed_path.is_file():
        raise RuntimeError("shard requires the sealed no-action verification")
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    if sealed.get("matched_action_control_trained") is not True or sealed.get("retrieval_recomputed") is not False:
        raise RuntimeError("sealed verification is not the matched no-action control")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    device = "cuda:0"
    print(f"shard {shard}/{shards} on {torch.cuda.get_device_name(0)}", flush=True)
    print("load admitted rows and cached tokens", flush=True)
    tensors, meta, _info = load_data("cpu")
    bundle = load_accepted_run_set([(run, APPROVALS / f"{run.name}.approval.json", KEY) for run in RUNS])
    observations = {row["observation_id"]: row for row in bundle["observations"]}
    transitions = {row["transition_id"]: row for row in bundle["transitions"]}
    for side in ("fit", "dev"):
        attach_identity(meta[side], transitions, observations)
    counts = assert_split(meta["fit"], meta["dev"])
    if counts["fit_transitions"] != EXPECTED["fit"] or counts["development_transitions"] != EXPECTED["development"]:
        raise RuntimeError(f"unexpected split counts {counts}")
    folds = assign_scene_folds(meta["fit"])
    if plain(folds)["folds"] != sealed["folds"]["folds"]:
        raise RuntimeError("shard fold table does not match the sealed verification")
    index = json.loads((PATCH_DIR / "patch_index.json").read_text(encoding="utf-8"))
    hashes = {
        "fit_transition_ids": stable_id_hash([row["transition_id"] for row in meta["fit"]]),
        "development_transition_ids": stable_id_hash([row["transition_id"] for row in meta["dev"]]),
        "patch_index": stable_id_hash(index),
    }
    if hashes != sealed["hashes"]:
        raise RuntimeError("shard data hashes do not match the sealed verification")
    row_of = {observation_id: position for position, observation_id in enumerate(index)}
    mapping = np.memmap(
        PATCH_DIR / "patches.f32.memmap", dtype=np.float32, mode="r", shape=(len(index), TOKENS, FEATURE_DIM)
    )
    arrays = {}
    for side in ("fit", "dev"):
        before = np.array([row_of[row["obs_before_id"]] for row in meta[side]])
        after = np.array([row_of[row["obs_after_id"]] for row in meta[side]])
        current = gather(mapping, before)
        future = gather(mapping, after)
        if not np.isfinite(current).all() or not np.isfinite(future).all():
            raise RuntimeError(f"{side} cached tokens are non-finite and were not replaced")
        arrays[side] = {
            "current": current,
            "future": future,
            "mean": current.mean(axis=1),
            "pooler": tensors[side][0].numpy(),
            "pooler_future": tensors[side][2].numpy(),
            "control": tensors[side][1].numpy(),
        }
    if float(np.max(np.abs(arrays["fit"]["pooler"] - arrays["fit"]["mean"]))) < 1e-3:
        raise RuntimeError("official pooler and patch mean are not distinct inputs")
    (OUT / "folds").mkdir(exist_ok=True)
    (OUT / "full").mkdir(exist_ok=True)
    jobs = fold_jobs(meta["fit"], folds)
    if len(jobs) != 27:
        raise RuntimeError(f"expected 27 fold jobs, got {len(jobs)}")
    for job in (jobs[slot] for slot in shard_slots(len(jobs), shard, shards)):
        path = OUT / "folds" / f"{job['arm']}_fold{job['fold']}_seed{job['seed']}.json"
        if fold_record_ready(path, job["arm"], job["seed"], job["fold"]):
            print(f"skip existing {path.name}", flush=True)
            continue
        if path.exists():
            path.unlink()
        print(f"train {job['arm']} no-action fold {job['fold']} seed {job['seed']}", flush=True)
        result = train_arm(
            job["arm"],
            job["seed"],
            job["train_rows"],
            job["val_rows"],
            arrays["fit"],
            job["val_scenes"],
            device,
            None,
            False,
        )
        write_json(
            path,
            {
                "arm": job["arm"],
                "seed": job["seed"],
                "fold": job["fold"],
                "use_action": False,
                "history": result["history"],
                "elapsed_s": result["elapsed_s"],
            },
        )
        del result
        torch.cuda.empty_cache()
    fold_paths = [
        OUT / "folds" / f"{job['arm']}_fold{job['fold']}_seed{job['seed']}.json" for job in jobs
    ]
    wait_for_files(fold_paths, "fold records")
    selected = choose_updates([json.loads(path.read_text(encoding="utf-8")) for path in fold_paths])
    selected_path = OUT / "selected_updates.json"
    if shard == 0 and not selected_path.exists():
        try:
            write_json(selected_path, selected)
        except FileExistsError:
            pass
    wait_for_files([selected_path], "selected updates")
    reread_updates = {arm: int(json.loads(selected_path.read_text(encoding="utf-8"))[arm]) for arm in ARMS}
    if reread_updates != selected:
        raise RuntimeError("selected updates do not match the fold scores")
    head = load_head(device)
    probe_at = np.linspace(0, len(meta["fit"]) - 1, 8).astype(int)
    current_probe = pool_tokens(head, arrays["fit"]["current"][probe_at], device)
    future_probe = pool_tokens(head, arrays["fit"]["future"][probe_at], device)
    head_diff = max(
        float(np.max(np.abs(current_probe - arrays["fit"]["pooler"][probe_at]))),
        float(np.max(np.abs(future_probe - arrays["fit"]["pooler_future"][probe_at]))),
    )
    if head_diff > 1e-4:
        raise RuntimeError(f"frozen pooling head does not match the audited pooler ({head_diff})")
    arrays["dev"]["pooled_future"] = pool_tokens(head, arrays["dev"]["future"], device)
    dev_scenes = [row["scene"] for row in meta["dev"]]
    copy_metrics = grid_prediction_metrics(
        arrays["dev"]["current"], arrays["dev"]["future"], arrays["dev"]["current"], dev_scenes
    )
    copy_gap = abs(copy_metrics["scene_macro_mse"] - float(sealed["action_reference"]["copy_scene_macro_mse"]))
    if copy_gap > 1e-4:
        raise RuntimeError(f"copy baseline drifted from the action run by {copy_gap}")
    fit_rows = np.arange(len(meta["fit"]))
    full_jobs = [{"seed": int(seed), "arm": arm} for seed in SEEDS for arm in ARMS]
    for job in (full_jobs[slot] for slot in shard_slots(len(full_jobs), shard, shards)):
        development_path = OUT / "full" / f"{job['arm']}_{job['seed']}_development.json"
        if development_record_ready(development_path, job["arm"], job["seed"]):
            print(f"skip existing {development_path.name}", flush=True)
            continue
        for name in (
            f"{job['arm']}_{job['seed']}_selected.pt",
            f"{job['arm']}_{job['seed']}_last.pt",
            development_path.name,
        ):
            stale = OUT / "full" / name
            if stale.exists():
                stale.unlink()
        update = reread_updates[job["arm"]]
        print(f"full fit {job['arm']} no-action seed {job['seed']} keep update {update}", flush=True)
        result = train_arm(job["arm"], job["seed"], fit_rows, None, arrays["fit"], None, device, update, False)
        payload = {
            "arm": job["arm"],
            "seed": job["seed"],
            "update": update,
            "model_state_dict": result["selected_state"],
            "channel_scale": result["scale"],
            "use_action": False,
            "navigation_connected": False,
        }
        save_torch(OUT / "full" / f"{job['arm']}_{job['seed']}_selected.pt", payload)
        payload["update"] = UPDATES
        payload["model_state_dict"] = {
            key: value.detach().cpu().clone() for key, value in result["model"].state_dict().items()
        }
        save_torch(OUT / "full" / f"{job['arm']}_{job['seed']}_last.pt", payload)
        evaluated = evaluate_development(
            result["selected_state"], job["arm"], arrays["dev"], meta["dev"], device, head, False
        )
        ratio = evaluated["metrics"]["scene_macro_mse"] / copy_metrics["scene_macro_mse"]
        write_json(
            development_path,
            {
                "arm": job["arm"],
                "seed": job["seed"],
                "selected_update": update,
                "metrics": evaluated["metrics"],
                "ratio_to_copy": ratio,
                "pooled_mse_vs_frozen_head_target": evaluated["pooled_mse_vs_frozen_head_target"],
                "use_action": False,
                "train_elapsed_s": result["elapsed_s"],
            },
        )
        del result, evaluated
        torch.cuda.empty_cache()
    if shard != 0:
        print(f"shard {shard} finished its jobs", flush=True)
        return
    development_paths = [
        OUT / "full" / f"{job['arm']}_{job['seed']}_development.json" for job in full_jobs
    ]
    wait_for_files(development_paths, "development records")
    development = [json.loads(path.read_text(encoding="utf-8")) for path in development_paths]
    drift_prediction = mean_change_prediction(
        arrays["fit"]["current"], arrays["fit"]["future"], arrays["dev"]["current"]
    )
    drift_metrics = grid_prediction_metrics(
        drift_prediction, arrays["dev"]["future"], arrays["dev"]["current"], dev_scenes
    )
    no_action_rows = [
        {
            "arm": row["arm"],
            "seed": row["seed"],
            "scene_macro_mse": row["metrics"]["scene_macro_mse"],
            "change_vector_error": row["metrics"]["change_vector_error"],
        }
        for row in development
    ]
    verdict = action_control_verdict(sealed["action_reference"]["rows"], no_action_rows)
    summary = {
        "grid": sealed["grid"],
        "counts": sealed["counts"],
        "hashes": sealed["hashes"],
        "head_check": sealed["head_check"],
        "smoke": sealed["smoke"],
        "selected_updates": reread_updates,
        "baselines": {"copy": copy_metrics, "mean_change": drift_metrics},
        "development": development,
        "verdict": verdict,
        "attribution": control_sentence(verdict),
        "navigation_connected": False,
        "development_is_blind_test": False,
        "retrieval_recomputed": False,
        "action_reference_updates": sealed["action_reference"]["selected_updates"],
        "shard_count": shards,
    }
    write_json(OUT / "results.json", summary)
    with (OUT / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(render_no_action_report(summary))
    print(json.dumps(plain(verdict), ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> None:
    global OUT, USE_ACTION
    args = list(sys.argv[1:] if argv is None else argv)
    shard = None
    if args == ["--matched-no-action"]:
        USE_ACTION = False
        OUT = NO_ACTION_OUT
    elif len(args) == 3 and args[0] == "--matched-no-action" and args[1] == "--shard":
        USE_ACTION = False
        OUT = NO_ACTION_OUT
        try:
            shard = parse_shard_spec(args[2])
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    elif args:
        raise SystemExit(
            "usage: python -m experiments.world_model.stage_a_future_entry "
            "[--matched-no-action [--shard i/n]]"
        )
    else:
        USE_ACTION = True
        OUT = ACTION_OUT
    try:
        if shard is None:
            _main()
        else:
            _main_shard(shard[0], shard[1])
    except Exception as exc:
        failure_name = "failure.json" if shard is None else f"failure_shard{shard[0]}.json"
        failure = OUT / failure_name
        if OUT.is_dir() and not failure.exists() and not (OUT / "results.json").exists():
            write_json(
                failure,
                {
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "navigation_connected": False,
                    "development_is_blind_test": False,
                },
            )
        raise


if __name__ == "__main__":
    main()
