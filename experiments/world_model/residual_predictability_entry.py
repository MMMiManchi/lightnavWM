"""Diagnose whether the Stage A patch residual has a stable, predictable part.

This reads the existing patch cache and the saved branch PNGs. It does not
train, start Habitat, write the formal feature cache, or connect LightNav.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.world_model.action_response_entry import encode_frames, load_png, load_states  # noqa: E402
from experiments.world_model.future_only import FEATURE_DIM, GRID, TOKENS  # noqa: E402
from experiments.world_model.residual_predictability import (  # noqa: E402
    COMPONENT_COUNT,
    DESCRIPTIVE_COMPONENT_COUNTS,
    SPEED_PAIRS,
    TURN_PAIRS,
    linear_per_sample_mse,
    neighbor_correlation,
    predictability_verdict,
    readout_weight,
    relative_reduction,
    scene_macro,
    scene_macro_rows,
    subspace_per_sample_mse,
    top_components,
    top_energy_share,
)
from experiments.world_model.stage_a_query_film_entry import DEV_IDS, FIT_IDS, load_arrays  # noqa: E402

OUT = REPO / "output/world_model_diagnostics/stage_a_residual_predictability_20261102_01"
CHUNK = 8


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prepare() -> None:
    OUT.mkdir()
    write_json(
        OUT / "protocol.json",
        {
            "experiment": "stage_a_residual_predictability",
            "target": "future_patch_grid - current_patch_grid",
            "gate_split": "development scene macro",
            "development_is_blind_test": False,
            "relative_band": 0.01,
            "subspace_note": "coefficients are projections of the residual, so this is transfer of a subspace, not a prediction",
            "linear_readout": "intercept plus 16 principal components of the current patch grid, fit only",
            "readout_ridge": 1e-4,
            "component_count": COMPONENT_COUNT,
            "descriptive_component_counts": list(DESCRIPTIVE_COMPONENT_COUNTS),
            "concentration": "top floor(10%) of patches or channels holds at least 25% of scene-macro energy",
            "turn_pairs": [list(pair) for pair in TURN_PAIRS],
            "speed_pairs": [list(pair) for pair in SPEED_PAIRS],
            "coherence": "4-neighbor correlation of the mean left-right energy map is at least 0.5",
            "turn_specific": "mean left-right energy exceeds mean speed-pair energy by more than 10%",
            "branch_state_weight": "equal per state",
            "states_are_formal_fit": True,
            "generalization": False,
            "fit_transition_ids_sha256": FIT_IDS,
            "dev_transition_ids_sha256": DEV_IDS,
            "navigation_connected": False,
            "habitat_started": False,
            "formal_cache_written": False,
            "training_started": False,
        },
    )


def _scores(
    current: torch.Tensor,
    future: torch.Tensor,
    residual_mean: torch.Tensor,
    residual_components: torch.Tensor,
    current_mean: torch.Tensor,
    current_components: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, np.ndarray]:
    residual = future - current
    flat_current = current.flatten(1)
    flat_residual = residual.flatten(1)
    zero = flat_residual.square().mean(dim=1).detach().cpu().numpy()
    mean = (flat_residual - residual_mean).square().mean(dim=1).detach().cpu().numpy()
    subspace = {}
    for count in DESCRIPTIVE_COMPONENT_COUNTS:
        if count > residual_components.shape[0]:
            continue
        subspace[count] = (
            subspace_per_sample_mse(flat_residual, residual_mean, residual_components[:count])
            .detach()
            .cpu()
            .numpy()
        )
    linear = (
        linear_per_sample_mse(flat_current, flat_residual, current_mean, current_components, weight)
        .detach()
        .cpu()
        .numpy()
    )
    return {
        "zero": zero,
        "mean": mean,
        "subspace": subspace,
        "linear": linear,
        "position_energy": residual.square().sum(dim=-1).detach().cpu().numpy(),
        "channel_energy": residual.square().sum(dim=-2).detach().cpu().numpy(),
    }


def score_split(
    current_np: np.ndarray,
    future_np: np.ndarray,
    device: str,
    residual_mean: torch.Tensor,
    residual_components: torch.Tensor,
    current_mean: torch.Tensor,
    current_components: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, np.ndarray]:
    count = len(current_np)
    pieces = []
    for start in range(0, count, CHUNK):
        current = torch.from_numpy(current_np[start : start + CHUNK]).to(device)
        future = torch.from_numpy(future_np[start : start + CHUNK]).to(device)
        pieces.append(
            _scores(
                current,
                future,
                residual_mean,
                residual_components,
                current_mean,
                current_components,
                weight,
            )
        )
        print(f"scored {min(start + CHUNK, count)}/{count}", flush=True)
    subspace_keys = pieces[0]["subspace"]
    return {
        "zero": np.concatenate([item["zero"] for item in pieces]),
        "mean": np.concatenate([item["mean"] for item in pieces]),
        "linear": np.concatenate([item["linear"] for item in pieces]),
        "subspace": {
            count: np.concatenate([item["subspace"][count] for item in pieces]) for count in subspace_keys
        },
        "position_energy": np.concatenate([item["position_energy"] for item in pieces]),
        "channel_energy": np.concatenate([item["channel_energy"] for item in pieces]),
    }


def summarize_errors(scored: dict, scenes: list[str]) -> dict:
    zero = scene_macro(scored["zero"], scenes)
    summary = {
        "zero_mse": zero,
        "mean_mse": scene_macro(scored["mean"], scenes),
        "linear_mse": scene_macro(scored["linear"], scenes),
        "subspace_mse": {},
    }
    summary["mean_relative_reduction"] = relative_reduction(zero, summary["mean_mse"])
    summary["linear_relative_reduction"] = relative_reduction(zero, summary["linear_mse"])
    for count, values in scored["subspace"].items():
        mse = scene_macro(values, scenes)
        summary["subspace_mse"][str(count)] = mse
        summary[f"subspace_{count}_relative_reduction"] = relative_reduction(zero, mse)
    return summary


def concentration_report(scored: dict, scenes: list[str]) -> dict:
    position = scene_macro_rows(scored["position_energy"], scenes)
    channel = scene_macro_rows(scored["channel_energy"], scenes)
    position_share = top_energy_share(position)
    channel_share = top_energy_share(channel)
    total = float(position.sum())
    order = np.argsort(position)[::-1][:8]
    leaders = []
    for index in order:
        row, col = divmod(int(index), GRID)
        leaders.append(
            {"row": row, "col": col, "share": float(position[index] / total)}
        )
    energy_map = position.reshape(GRID, GRID)
    return {
        "patch_top10_share": position_share["share"],
        "patch_top10_count": position_share["count"],
        "channel_top10_share": channel_share["share"],
        "channel_top10_count": channel_share["count"],
        "residual_energy_neighbor_correlation": neighbor_correlation(energy_map),
        "leading_patches": leaders,
        "position_energy": position.tolist(),
    }


def action_geometry(tokens: np.ndarray, names: list[str]) -> dict:
    index = {name: position for position, name in enumerate(names)}
    missing = [name for pair in (*TURN_PAIRS, *SPEED_PAIRS) for name in pair if name not in index]
    if missing:
        raise RuntimeError(f"branch actions are missing: {missing}")

    def pair_energy(left: str, right: str) -> tuple[float, np.ndarray]:
        difference = tokens[:, index[left]] - tokens[:, index[right]]
        per_state = np.square(difference.astype(np.float64)).sum(axis=-1)
        return float(per_state.mean()), per_state.mean(axis=0)

    turn_maps = []
    turn_rows = []
    for left, right in TURN_PAIRS:
        energy, spatial = pair_energy(left, right)
        turn_maps.append(spatial)
        turn_rows.append({"pair": f"{left}/{right}", "energy": energy})
    speed_maps = []
    speed_rows = []
    for left, right in SPEED_PAIRS:
        energy, spatial = pair_energy(left, right)
        speed_maps.append(spatial)
        speed_rows.append({"pair": f"{left}/{right}", "energy": energy})
    turn_map = np.mean(turn_maps, axis=0).reshape(GRID, GRID)
    speed_map = np.mean(speed_maps, axis=0).reshape(GRID, GRID)
    return {
        "turn_pairs": turn_rows,
        "speed_pairs": speed_rows,
        "turn_energy": float(np.mean([row["energy"] for row in turn_rows])),
        "speed_energy": float(np.mean([row["energy"] for row in speed_rows])),
        "turn_neighbor_correlation": neighbor_correlation(turn_map),
        "speed_neighbor_correlation": neighbor_correlation(speed_map),
        "turn_energy_map": turn_map.reshape(-1).tolist(),
        "state_count": int(tokens.shape[0]),
        "state_weight": "equal",
        "states_are_formal_fit": True,
    }


def encode_branch_futures(device: str) -> tuple[np.ndarray, list[str], int]:
    states = load_states()
    names = [branch["name"] for branch in states[0]["branches"]]
    for state in states[1:]:
        if [branch["name"] for branch in state["branches"]] != names:
            raise RuntimeError(f"action order changed at {state['state_id']}")
    images = [load_png(branch["image"]) for state in states for branch in state["branches"]]
    encoded = encode_frames(images, device)
    if encoded.shape != (len(states) * len(names), TOKENS, FEATURE_DIM):
        raise RuntimeError(f"unexpected branch tokens {encoded.shape}")
    scenes = len({state["scene"] for state in states})
    return encoded.reshape(len(states), len(names), TOKENS, FEATURE_DIM), names, scenes


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _maybe(value: float | None) -> str:
    if value is None:
        return "未定义"
    return f"{value:.4f}"


def render_report(payload: dict) -> str:
    dev = payload["development"]
    fit = payload["fit"]
    concentration = payload["concentration"]
    geometry = payload["action_geometry"]
    verdict = payload["verdict"]
    lines = [
        "# 阶段 A 残差可预测性诊断",
        "",
        "这是只读诊断。开发集已经用于此前阶段 A，不是盲测。",
        "没有训练，没有启动 Habitat，没有写入正式特征缓存，也没有接入 LightNav。",
        "残差子空间用的是残差自己在 fit 方向上的投影，所以它只说明这些方向能不能迁到 dev，不是从图像做预测。",
        "从当前帧预测的是截距加上当前网格的 16 个主成分，系数只在 fit 上估计。",
        "",
        f"判定：{verdict['call_label']}（`{verdict['call']}`）。",
        verdict["next_step"],
        "",
        "## 开发集场景宏平均 MSE",
        "",
        "| 预测 | MSE | 相对零残差 |",
        "|---|---:|---:|",
        f"| 零残差 | {dev['zero_mse']:.6f} | 0.00% |",
        f"| fit 均值 | {dev['mean_mse']:.6f} | {_percent(dev['mean_relative_reduction'])} |",
        f"| 当前帧线性读出 | {dev['linear_mse']:.6f} | {_percent(dev['linear_relative_reduction'])} |",
    ]
    for count, mse in dev["subspace_mse"].items():
        reduction = dev[f"subspace_{count}_relative_reduction"]
        marker = "，门槛用这一行" if int(count) == COMPONENT_COUNT else ""
        lines.append(f"| 残差子空间 k={count}{marker} | {mse:.6f} | {_percent(reduction)} |")
    lines.extend(
        [
            "",
            "## fit 上的同一组数",
            "",
            "| 预测 | MSE | 相对零残差 |",
            "|---|---:|---:|",
            f"| 零残差 | {fit['zero_mse']:.6f} | 0.00% |",
            f"| fit 均值 | {fit['mean_mse']:.6f} | {_percent(fit['mean_relative_reduction'])} |",
            f"| 当前帧线性读出 | {fit['linear_mse']:.6f} | {_percent(fit['linear_relative_reduction'])} |",
        ]
    )
    for count, mse in fit["subspace_mse"].items():
        lines.append(
            f"| 残差子空间 k={count} | {mse:.6f} | {_percent(fit[f'subspace_{count}_relative_reduction'])} |"
        )
    lines.extend(
        [
            "",
            "## 能量位置",
            "",
            f"位置前 10%（{concentration['patch_top10_count']} 个）占场景宏能量 "
            f"{_percent(concentration['patch_top10_share'])}。",
            f"通道前 10%（{concentration['channel_top10_count']} 个）占 "
            f"{_percent(concentration['channel_top10_share'])}。",
            "集中的门槛是 25%。",
            f"残差能量图的四邻域相关：{_maybe(concentration['residual_energy_neighbor_correlation'])}。",
            "",
            "| 行 | 列 | 能量占比 |",
            "|---:|---:|---:|",
        ]
    )
    for patch in concentration["leading_patches"]:
        lines.append(f"| {patch['row']} | {patch['col']} | {_percent(patch['share'])} |")
    lines.extend(
        [
            "",
            "## 已执行分支上的左右转",
            "",
            f"{geometry['state_count']} 个状态来自 formal fit，按状态等权，不是泛化测试。"
            f"覆盖 {payload['branch_scenes']} 个场景。",
            f"左右转平均能量 {geometry['turn_energy']:.2f}，速度对平均能量 {geometry['speed_energy']:.2f}。",
            f"左右转能量图的四邻域相关：{_maybe(geometry['turn_neighbor_correlation'])}。",
            f"速度对能量图的四邻域相关：{_maybe(geometry['speed_neighbor_correlation'])}。",
            "成片的门槛是相关至少 0.5。左右转能量需要比速度对高过 10%，才算和转向有关。",
            "",
            "| 对比 | 能量 |",
            "|---|---:|",
        ]
    )
    for row in geometry["turn_pairs"] + geometry["speed_pairs"]:
        lines.append(f"| {row['pair']} | {row['energy']:.2f} |")
    lines.extend(["", verdict["next_step"], ""])
    return "\n".join(lines)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("residual diagnosis requires CUDA")
    prepare()
    device = "cuda:2"
    torch.cuda.set_device(2)
    torch.set_num_threads(4)
    print("load cached tokens", flush=True)
    arrays = load_arrays()
    fit_current = torch.from_numpy(arrays["fit"]["current"]).to(device).flatten(1)
    fit_residual = torch.from_numpy(arrays["fit"]["future"]).to(device).flatten(1) - fit_current
    arrays["fit"]["current"] = None
    arrays["fit"]["future"] = None
    print("principal components", flush=True)
    current_mean, current_components = top_components(fit_current, COMPONENT_COUNT)
    residual_mean, residual_components = top_components(fit_residual, max(DESCRIPTIVE_COMPONENT_COUNTS))
    if current_components.shape[0] < COMPONENT_COUNT or residual_components.shape[0] < COMPONENT_COUNT:
        raise RuntimeError("the 16-component gate is below the variance floor")
    weight = readout_weight(fit_current, fit_residual, current_mean, current_components)
    print("score fit", flush=True)
    fit_scored = _scores(
        fit_current.view(-1, TOKENS, FEATURE_DIM),
        (fit_current + fit_residual).view(-1, TOKENS, FEATURE_DIM),
        residual_mean,
        residual_components,
        current_mean,
        current_components,
        weight,
    )
    del fit_current, fit_residual
    torch.cuda.empty_cache()
    print("score development", flush=True)
    dev_scored = score_split(
        arrays["dev"]["current"],
        arrays["dev"]["future"],
        device,
        residual_mean,
        residual_components,
        current_mean,
        current_components,
        weight,
    )
    del residual_mean, residual_components, current_mean, current_components, weight
    torch.cuda.empty_cache()
    print("encode branch futures", flush=True)
    tokens, names, branch_scenes = encode_branch_futures(device)
    geometry = action_geometry(tokens, names)
    dev_summary = summarize_errors(dev_scored, arrays["dev"]["scenes"])
    fit_summary = summarize_errors(fit_scored, arrays["fit"]["scenes"])
    concentration = concentration_report(dev_scored, arrays["dev"]["scenes"])
    verdict = predictability_verdict(
        zero_mse=dev_summary["zero_mse"],
        mean_mse=dev_summary["mean_mse"],
        subspace_mse=dev_summary["subspace_mse"][str(COMPONENT_COUNT)],
        linear_mse=dev_summary["linear_mse"],
        patch_top10_share=concentration["patch_top10_share"],
        channel_top10_share=concentration["channel_top10_share"],
        turn_neighbor_correlation=geometry["turn_neighbor_correlation"],
        turn_energy=geometry["turn_energy"],
        speed_energy=geometry["speed_energy"],
    )
    payload = {
        "fit": fit_summary,
        "development": dev_summary,
        "concentration": concentration,
        "action_geometry": geometry,
        "branch_scenes": branch_scenes,
        "subspace_components_available": int(max(int(key) for key in dev_summary["subspace_mse"])),
        "current_components_used": COMPONENT_COUNT,
        "verdict": verdict,
        "fit_transition_ids_sha256": FIT_IDS,
        "dev_transition_ids_sha256": DEV_IDS,
    }
    write_json(OUT / "results.json", payload)
    with (OUT / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(render_report(payload))
    print(json.dumps({"call": verdict["call"], "next_step": verdict["next_step"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
