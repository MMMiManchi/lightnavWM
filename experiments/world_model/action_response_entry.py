"""Read-only action swap on Stage A grid checkpoints.

Encodes saved branch PNGs with the audited SigLIP tower. It does not start
Habitat, write the formal feature cache, or train.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.world_model.action_response import (  # noqa: E402
    SEEDS,
    TRUE_DIFF_FLOOR,
    action_response_verdict,
    difference_directions,
    pairwise_l2,
)
from experiments.world_model.feature_cache import (  # noqa: E402
    AUDITED_PREPROCESSOR,
    OUTPUT_SHAPES,
    SIGLIP_REVISION,
    SIGLIP_WEIGHT_SHA256,
)
from experiments.world_model.future_only import (  # noqa: E402
    FEATURE_DIM,
    TOKENS,
    FutureGridDecoder,
)
from experiments.world_model.jsonutil import sha256_file  # noqa: E402

BRANCH = REPO / "output/world_model_diagnostics/action_branch_probe_20261030_05"
PATCH_DIR = REPO / "output/world_model_training/spatial_vs_pooler_20261030_01"
ACTION_CKPT = REPO / "output/world_model_training/stage_a_future_only_20261030_01/full"
NO_ACTION_CKPT = REPO / "output/world_model_training/stage_a_no_action_20261102_01/full"
POSE_ANALYSIS = BRANCH / "checkpoint_action_conditioned_analysis.json"
OUT = REPO / "output/world_model_diagnostics/stage_a_action_response_20261102_01"
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots"
    / SIGLIP_REVISION
)
DT = 0.1
CACHE_MATCH_MAX_ABS = 1e-4
ENCODE_BATCH = 4


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_png(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(f"branch frame is not uint8 RGB: {path}")
    return np.ascontiguousarray(array)


def load_states() -> list[dict]:
    states = []
    with (BRANCH / "branch_records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            branches = []
            for branch in row["branches"]:
                after = BRANCH / branch["replay_rgb_after_path"]
                if sha256_file(after) != branch["replay_rgb_after_sha256"]:
                    raise RuntimeError(f"after-frame hash mismatch: {after}")
                branches.append(
                    {
                        "name": branch["candidate"]["name"],
                        "v_mps": float(branch["candidate"]["v_mps"]),
                        "w_rad_s": math.radians(float(branch["candidate"]["w_dps"])),
                        "image": after,
                    }
                )
            before = BRANCH / row["replay_rgb_before_path"]
            if sha256_file(before) != row["replay_rgb_before_sha256"]:
                raise RuntimeError(f"before-frame hash mismatch: {before}")
            if len(branches) != 8 or len({item["name"] for item in branches}) != 8:
                raise RuntimeError(f"state {row['state_id']} does not have eight distinct actions")
            states.append(
                {
                    "state_id": row["state_id"],
                    "scene": row["scene"],
                    "observation_id": row["source_obs_before_id"],
                    "before": before,
                    "branches": branches,
                }
            )
    if len(states) != 48:
        raise RuntimeError(f"expected 48 branch states, found {len(states)}")
    return states


def encode_frames(images: list[np.ndarray], device: str) -> np.ndarray:
    from transformers import SiglipImageProcessor, SiglipVisionModel

    from experiments.world_model.cache_writer import verify_bound_files

    weight = SNAPSHOT / "model.safetensors"
    if verify_bound_files(weight)["weight_sha256"] != SIGLIP_WEIGHT_SHA256:
        raise RuntimeError("SigLIP weight hash does not match the audited contract")
    processor = SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
    model = SiglipVisionModel.from_pretrained(
        SNAPSHOT, local_files_only=True, torch_dtype=torch.float32
    ).to(device).eval()
    tokens = np.empty((len(images), TOKENS, FEATURE_DIM), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(images), ENCODE_BATCH):
            chunk = images[start : start + ENCODE_BATCH]
            pixels = processor(images=chunk, return_tensors="pt")["pixel_values"].to(device)
            output = model(pixel_values=pixels).last_hidden_state.float().cpu().numpy()
            if output.shape != (len(chunk), *OUTPUT_SHAPES["last_hidden_state"]):
                raise RuntimeError(f"unexpected patch output {output.shape}")
            if not np.isfinite(output).all():
                raise RuntimeError("encoder produced a non-finite token")
            tokens[start : start + len(chunk)] = output
            print(f"encoded {start + len(chunk)}/{len(images)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return tokens


def cached_before(states: list[dict]) -> np.ndarray:
    index = json.loads((PATCH_DIR / "patch_index.json").read_text(encoding="utf-8"))
    position = {observation_id: item for item, observation_id in enumerate(index)}
    mapping = np.memmap(
        PATCH_DIR / "patches.f32.memmap",
        dtype=np.float32,
        mode="r",
        shape=(len(index), TOKENS, FEATURE_DIM),
    )
    rows = []
    for state in states:
        if state["observation_id"] not in position:
            raise RuntimeError(f"branch observation is not in the patch cache: {state['observation_id']}")
        rows.append(np.asarray(mapping[position[state["observation_id"]]], dtype=np.float32))
    return np.stack(rows)


def predict_grid(
    checkpoint: Path,
    before: np.ndarray,
    controls: np.ndarray,
    device: str,
) -> np.ndarray:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if blob.get("arm") != "grid" or blob.get("navigation_connected") is not False:
        raise RuntimeError(f"checkpoint is not an unconnected grid model: {checkpoint}")
    model = FutureGridDecoder("grid", use_action=bool(blob["use_action"])).to(device).eval()
    model.load_state_dict(blob["model_state_dict"])
    predicted = np.empty((len(controls), TOKENS, FEATURE_DIM), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(controls), 16):
            stop = start + 16
            memory = torch.from_numpy(np.ascontiguousarray(before[start:stop])).to(device)
            control = torch.from_numpy(np.ascontiguousarray(controls[start:stop])).to(device)
            predicted[start:stop] = model(memory, control).float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return predicted


def score_checkpoint(
    name: str,
    seed: int,
    use_action: bool,
    predicted: np.ndarray,
    true: np.ndarray,
    current: np.ndarray,
    states: list[dict],
) -> dict:
    ratios = []
    comparable = 0
    responsive = 0
    cosine_weight = 0.0
    cosine_sum = 0.0
    absolute_pred = []
    absolute_current = []
    for state_index, state in enumerate(states):
        sl = slice(state_index * 8, (state_index + 1) * 8)
        pred_spread = pairwise_l2(predicted[sl])
        true_spread = pairwise_l2(true[sl])
        if true_spread >= TRUE_DIFF_FLOOR:
            ratios.append(pred_spread / true_spread)
        directions = difference_directions(predicted[sl], true[sl])
        comparable += int(directions["comparable_pairs"])
        responsive += int(directions["responsive_pairs"])
        if directions["direction_cosine_mean"] is not None:
            weight = int(directions["responsive_pairs"])
            cosine_sum += float(directions["direction_cosine_mean"]) * weight
            cosine_weight += weight
        for action in range(8):
            absolute_pred.append(_cosine(predicted[sl][action], true[sl][action]))
            absolute_current.append(_cosine(current[state_index], true[sl][action]))
        del state
    if not ratios:
        raise RuntimeError(f"{name} has no state with a real action spread")
    return {
        "name": name,
        "seed": seed,
        "use_action": use_action,
        "states": len(states),
        "states_with_true_spread": len(ratios),
        "spread_ratio_median": float(np.median(ratios)),
        "spread_ratio_mean": float(np.mean(ratios)),
        "comparable_pairs": comparable,
        "responsive_pairs": responsive,
        "direction_cosine_mean": None if cosine_weight == 0 else cosine_sum / cosine_weight,
        "token_cosine_prediction_to_true": float(np.mean(absolute_pred)),
        "token_cosine_current_to_true": float(np.mean(absolute_current)),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    flat_left = left.reshape(-1).astype(np.float64)
    flat_right = right.reshape(-1).astype(np.float64)
    return float(np.dot(flat_left, flat_right) / (np.linalg.norm(flat_left) * np.linalg.norm(flat_right)))


def render_report(summary: dict) -> str:
    verdict = summary["verdict"]
    lines = [
        "# 阶段 A 固定观测换动作探针",
        "",
        "这是 formal fit 里 48 个已有分支状态上的机制检查，不是场景隔离后的泛化。",
        "没有重训，没有启动 Habitat，没有写入正式特征缓存，也没有接入 LightNav。",
        "",
        f"前图编码与已有 patch 缓存的最大绝对差：{summary['cache_match_max_abs']:.3e}。",
        f"真实分支来自 `{summary['branch_dir']}`，48 个状态、8 个已执行命令、{summary['scenes']} 个场景。",
        "",
        "敏感性是预测网格随命令变化的幅度，除以真实未来网格随命令变化的幅度。",
        "方向只在真实差和预测差都超过 L2 1.0 的动作对上计算。敏感不等于预测对。",
        "",
        "| 模型 | seed | 幅度比中位数 | 幅度比均值 | 可比较动作对 | 有响应动作对 | 方向余弦 | 预测→真实 | 当前→真实 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["checkpoints"]:
        cosine = "未定义" if row["direction_cosine_mean"] is None else f"{row['direction_cosine_mean']:.4f}"
        lines.append(
            f"| {row['name']} | {row['seed']} | {row['spread_ratio_median']:.4f} | "
            f"{row['spread_ratio_mean']:.4f} | {row['comparable_pairs']} | {row['responsive_pairs']} | "
            f"{cosine} | {row['token_cosine_prediction_to_true']:.4f} | "
            f"{row['token_cosine_current_to_true']:.4f} |"
        )
    pose = summary["existing_pose_probe"]
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"探针有效：{verdict['probe_valid']}。响应：{verdict['response_label']}。方向：{verdict['direction_label']}。",
            "有响应的门槛是三个 seed 的幅度比中位数都至少 0.05。无动作模型的幅度比中位数必须低于 0.01，否则探针无效。",
            "",
            "已有位姿探针使用的是 v1 checkpoint，不是阶段 A 的网格模型，这次没有重跑。",
            f"384 条分支上 v1 位移误差 {pose['model_translation_vector_error_m']:.5f} 米，"
            f"命令积分 {pose['command_translation_vector_error_m']:.5f} 米。",
            "",
            summary["next_step"],
            "",
        ]
    )
    return "\n".join(lines)


def next_step(verdict: dict) -> str:
    if not verdict["probe_valid"]:
        return "无动作模型的输出也随命令变化，探针不能解释动作通路。先检查命令是否被错误置零。"
    if verdict["response_call"] == "insensitive":
        return (
            "有动作的网格模型在固定画面上几乎不随这 8 个命令变化。"
            "下一步改损失，预测未来特征减当前特征。解码器和数据先不动。"
        )
    if verdict["direction_call"] == "same_direction":
        return (
            "输出随命令变化，并且差的方向在这三个 seed 上都与真实分支同向。"
            "这仍只是 formal fit 的 48 个状态，不能写成泛化，也不能据此接入导航。"
        )
    if verdict["response_call"] == "responsive":
        return (
            "输出随命令变化，但差的方向没有在三个 seed 上都与真实分支同向。"
            "下一步才考虑用同状态多动作的差做监督。这 48 个状态不能同时用于训练和宣称泛化。"
        )
    return "三个 seed 的响应幅度不一致。先不要改结构或补数据，按 seed 查看幅度比。"


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("the probe requires CUDA")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    device = "cuda:0"
    states = load_states()
    print(f"load {len(states)} states", flush=True)
    images = []
    for state in states:
        images.append(load_png(state["before"]))
        images.extend(load_png(branch["image"]) for branch in state["branches"])
    tokens = encode_frames(images, device)
    current = tokens[0::9]
    true = np.stack([tokens[state * 9 + 1 : (state + 1) * 9] for state in range(len(states))])
    cached = cached_before(states)
    cache_gap = float(np.max(np.abs(current - cached)))
    print(f"cache match max abs {cache_gap:.3e}", flush=True)
    if cache_gap > CACHE_MATCH_MAX_ABS:
        raise RuntimeError(f"replay before-frames do not match the patch cache ({cache_gap})")
    controls = np.asarray(
        [
            [branch["v_mps"], branch["w_rad_s"], DT]
            for state in states
            for branch in state["branches"]
        ],
        dtype=np.float32,
    )
    repeated_before = np.repeat(current, 8, axis=0)
    flat_true = true.reshape(len(states) * 8, TOKENS, FEATURE_DIM)
    OUT.mkdir(parents=True, exist_ok=False)
    rows = []
    for use_action, root, label in (
        (True, ACTION_CKPT, "有动作"),
        (False, NO_ACTION_CKPT, "无动作"),
    ):
        for seed in SEEDS:
            path = root / f"grid_{seed}_selected.pt"
            print(f"predict {label} seed {seed}", flush=True)
            predicted = predict_grid(path, repeated_before, controls, device)
            rows.append(score_checkpoint(label, seed, use_action, predicted, flat_true, current, states))
            del predicted
    action_rows = [row for row in rows if row["use_action"]]
    no_action_rows = [row for row in rows if not row["use_action"]]
    verdict = action_response_verdict(action_rows, no_action_rows)
    pose_document = json.loads(POSE_ANALYSIS.read_text(encoding="utf-8"))
    pose = pose_document["summary"]["all"]
    summary = {
        "cache_match_max_abs": cache_gap,
        "branch_dir": str(BRANCH),
        "scenes": len({state["scene"] for state in states}),
        "checkpoints": rows,
        "verdict": verdict,
        "existing_pose_probe": {
            "checkpoint": pose_document["checkpoint"],
            "model_translation_vector_error_m": pose["model_translation_vector_error_m"],
            "command_translation_vector_error_m": pose["command_translation_vector_error_m"],
            "rerun": False,
        },
        "next_step": next_step(verdict),
        "habitat_started": False,
        "formal_cache_written": False,
        "training_started": False,
        "navigation_connected": False,
    }
    write_json(OUT / "results.json", summary)
    with (OUT / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(render_report(summary))
    print(json.dumps(verdict, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
