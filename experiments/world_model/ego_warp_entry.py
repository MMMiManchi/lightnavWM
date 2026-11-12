"""Read-only check: depth ego-motion warp against copy and yaw shift.

The 48 formal-fit branch states only check the camera. Development is scored
after that check passes. This does not train, start Habitat, write the feature
cache, or connect LightNav.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.world_model.action_response_entry import load_png, load_states  # noqa: E402
from experiments.world_model.baselines import action_kind, executed_twist  # noqa: E402
from experiments.world_model.ego_warp import (  # noqa: E402
    COVERAGE_MIN,
    IMPROVEMENT_MIN,
    MAX_DEPTH_M,
    MIN_DEPTH_M,
    MIN_VALID_Z_M,
    STRAIGHT_RELATIVE_FLOOR,
    command_motion,
    depth_to_meters,
    ego_warp_verdict,
    fit_camera_ok,
    intrinsics,
    nearest_shift,
    relative_or_tie,
    warp_frame,
)
from experiments.world_model.jsonutil import sha256_file  # noqa: E402
from experiments.world_model.residual_predictability import scene_macro  # noqa: E402
from experiments.world_model.yaw_shift import (  # noqa: E402
    DT_S,
    HEIGHT,
    HFOV_DEG,
    LEFT_ACTIONS,
    RIGHT_ACTIONS,
    STRAIGHT_ACTIONS,
    WIDTH,
    masked_mae,
    relative_improvement,
    shift_content_right,
)

OUT = REPO / "output/world_model_diagnostics/stage_a_ego_warp_20261102_01"
FIT = REPO / "output/world_model_data/formal_fit_20261028_04"
DEV = REPO / "output/world_model_data/formal_dev_20261028_01"
DEV_TRANSITION_COUNT = 3249
DEV_SCENE_COUNT = 8
FX, FY, CX, CY = intrinsics()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prepare() -> None:
    OUT.mkdir()
    write_json(
        OUT / "protocol.json",
        {
            "experiment": "stage_a_ego_warp",
            "question": "does commanded depth ego-motion predict the next RGB frame",
            "depth": "normalized optical-axis z; meters = normalized * (max - min) + min",
            "min_depth_m": MIN_DEPTH_M,
            "max_depth_m": MAX_DEPTH_M,
            "min_valid_z_m": MIN_VALID_Z_M,
            "camera": "pinhole, square pixels, principal point at width/2 and height/2",
            "width": WIDTH,
            "height": HEIGHT,
            "hfov_deg": HFOV_DEG,
            "fx": FX,
            "motion": "forward = v * dt_command, left = 0, yaw = w * dt_command",
            "motion_source": "collision-free branch poses, same integral as kinematic_delta",
            "yaw_baseline": "yaw / hfov * width, rounded to the nearest pixel",
            "mask": "nearest depth splat plus an empty neighbor within one pixel, intersected with the yaw-shift pixels; every method uses that intersection",
            "fit_check": "48 formal-fit branch states, equal weight per image, not a generalization test",
            "fit_gate": "left and right depth-vs-copy each above 10%; each straight relative at least -10%; turn coverage at least 50%",
            "development_scored_only_after_fit_passes": True,
            "development": "formal_dev_20261028_01 scene macro",
            "development_gate": "coverage at least 50%; stop does not improve more than 10%; all, turns, and forward each beat their baseline by more than 10%",
            "zero_baseline": "relative is 0 when both errors are 0, and -1 when only the baseline is 0",
            "improvement_min": IMPROVEMENT_MIN,
            "coverage_min": COVERAGE_MIN,
            "straight_relative_floor": STRAIGHT_RELATIVE_FLOOR,
            "uses_future_pose": False,
            "development_is_blind_test": False,
            "generalization": False,
            "navigation_connected": False,
            "habitat_started": False,
            "formal_cache_written": False,
            "training_started": False,
        },
    )


def index_observations(run: Path) -> dict[str, dict]:
    index = {}
    with (run / "observations.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            index[row["observation_id"]] = row
    return index


def load_checked(run: Path, meta: dict) -> np.ndarray:
    path = run / meta["path"]
    if sha256_file(path) != meta["sha256"]:
        raise RuntimeError(f"hash mismatch: {path}")
    array = np.load(path)
    if list(array.shape) != list(meta["shape"]) or str(array.dtype) != meta["dtype"]:
        raise RuntimeError(f"array contract mismatch: {path}")
    return array


def scene_id(episode_uid: str) -> str:
    parts = episode_uid.split("|")
    if len(parts) != 4 or not parts[2]:
        raise RuntimeError(f"episode uid has no scene: {episode_uid}")
    return parts[2]


def score_pair(
    before: np.ndarray, after: np.ndarray, depth_m: np.ndarray, forward: float, left: float, yaw: float
) -> dict[str, float]:
    warped, depth_mask = warp_frame(
        before, depth_m, forward, left, yaw, fx=FX, fy=FY, cx=CX, cy=CY
    )
    shifted, shift_mask = shift_content_right(before, nearest_shift(yaw))
    mask = depth_mask & shift_mask
    if not bool(mask.any()):
        raise RuntimeError("the shared mask is empty")
    return {
        "copy_mae": masked_mae(after, before, mask),
        "yaw_mae": masked_mae(after, shifted, mask),
        "depth_mae": masked_mae(after, warped, mask),
        "coverage": float(mask.mean()),
    }


def mean_field(rows: list[dict], field: str) -> float:
    return float(np.mean([row[field] for row in rows]))


def fit_summary(rows: list[dict]) -> dict:
    by_name = {}
    for name in (*LEFT_ACTIONS, *RIGHT_ACTIONS, *STRAIGHT_ACTIONS):
        chosen = [row for row in rows if row["name"] == name]
        if len(chosen) != 48:
            raise RuntimeError(f"{name} has {len(chosen)} rows, expected 48")
        copy_mae = mean_field(chosen, "copy_mae")
        depth_mae = mean_field(chosen, "depth_mae")
        yaw_mae = mean_field(chosen, "yaw_mae")
        by_name[name] = {
            "n": len(chosen),
            "copy_mae": copy_mae,
            "yaw_mae": yaw_mae,
            "depth_mae": depth_mae,
            "depth_vs_copy": relative_improvement(copy_mae, depth_mae),
            "depth_vs_yaw": relative_improvement(yaw_mae, depth_mae),
            "coverage": mean_field(chosen, "coverage"),
        }
    left_rows = [row for row in rows if row["name"] in LEFT_ACTIONS]
    right_rows = [row for row in rows if row["name"] in RIGHT_ACTIONS]
    turn_rows = left_rows + right_rows
    left_relative = relative_improvement(mean_field(left_rows, "copy_mae"), mean_field(left_rows, "depth_mae"))
    right_relative = relative_improvement(
        mean_field(right_rows, "copy_mae"), mean_field(right_rows, "depth_mae")
    )
    straight = [by_name[name]["depth_vs_copy"] for name in STRAIGHT_ACTIONS]
    coverage = mean_field(turn_rows, "coverage")
    return {
        "actions": by_name,
        "left_relative": left_relative,
        "right_relative": right_relative,
        "straight_relatives": straight,
        "turn_coverage": coverage,
        "passed": fit_camera_ok(left_relative, right_relative, straight, coverage),
    }


def group_macro(values: list[float], scenes: list[str], kinds: list[str], kind: str | None) -> dict:
    if kind is None:
        chosen_values = np.asarray(values, dtype=np.float64)
        chosen_scenes = scenes
    else:
        keep = [item == kind for item in kinds]
        chosen_values = np.asarray(values, dtype=np.float64)[np.asarray(keep)]
        chosen_scenes = [scene for scene, item in zip(scenes, keep) if item]
    if chosen_values.size == 0:
        raise RuntimeError(f"group {kind} is empty")
    return {
        "n": int(chosen_values.size),
        "scenes": len(set(chosen_scenes)),
        "scene_macro": scene_macro(chosen_values, chosen_scenes),
    }


def development_summary(rows: list[dict]) -> dict:
    if len(rows) != DEV_TRANSITION_COUNT:
        raise RuntimeError(f"expected {DEV_TRANSITION_COUNT} development transitions, found {len(rows)}")
    scenes = [row["scene"] for row in rows]
    if len(set(scenes)) != DEV_SCENE_COUNT:
        raise RuntimeError(f"expected {DEV_SCENE_COUNT} development scenes, found {len(set(scenes))}")
    kinds = [row["kind"] for row in rows]
    fields = {}
    for field in ("copy_mae", "yaw_mae", "depth_mae", "coverage"):
        overall = group_macro([row[field] for row in rows], scenes, kinds, None)
        fields[field] = {"all": overall["scene_macro"]}
        for kind in ("turning", "mostly_forward", "stop_threshold"):
            fields[field][kind] = group_macro([row[field] for row in rows], scenes, kinds, kind)
    groups = {}
    for kind in ("all", "turning", "mostly_forward", "stop_threshold"):
        if kind == "all":
            copy_mae = fields["copy_mae"]["all"]
            yaw_mae = fields["yaw_mae"]["all"]
            depth_mae = fields["depth_mae"]["all"]
            coverage = fields["coverage"]["all"]
            count = group_macro([row["copy_mae"] for row in rows], scenes, kinds, None)
        else:
            copy_mae = fields["copy_mae"][kind]["scene_macro"]
            yaw_mae = fields["yaw_mae"][kind]["scene_macro"]
            depth_mae = fields["depth_mae"][kind]["scene_macro"]
            coverage = fields["coverage"][kind]["scene_macro"]
            count = fields["copy_mae"][kind]
        groups[kind] = {
            "n": count["n"],
            "scenes": count["scenes"],
            "copy_mae": copy_mae,
            "yaw_mae": yaw_mae,
            "depth_mae": depth_mae,
            "coverage": coverage,
            "depth_vs_copy": relative_or_tie(copy_mae, depth_mae),
            "depth_vs_yaw": relative_or_tie(yaw_mae, depth_mae),
        }
    return groups


def run_fit(states: list[dict], observations: dict[str, dict]) -> tuple[list[dict], dict]:
    rows = []
    for state in states:
        observation = observations[state["observation_id"]]
        if observation["depth_missing"] or observation["rgb_missing"]:
            raise RuntimeError(f"branch observation is missing a frame: {state['observation_id']}")
        depth = depth_to_meters(load_checked(FIT, observation["depth"]))
        before = load_png(state["before"])
        if before.shape != (HEIGHT, WIDTH, 3) or depth.shape != (HEIGHT, WIDTH):
            raise RuntimeError(f"branch frame shape is not {HEIGHT}x{WIDTH}")
        for branch in state["branches"]:
            forward, left, yaw = command_motion(branch["v_mps"], branch["w_rad_s"], DT_S)
            scored = score_pair(before, load_png(branch["image"]), depth, forward, left, yaw)
            scored["name"] = branch["name"]
            rows.append(scored)
    return rows, fit_summary(rows)


def run_development(observations: dict[str, dict]) -> list[dict]:
    rows = []
    with (DEV / "transitions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            transition = json.loads(line)
            twist = executed_twist(transition)
            if twist["dt_command_s"] is None:
                raise RuntimeError(f"transition has no command interval: {transition['transition_id']}")
            forward, left, yaw = command_motion(
                float(twist["v_mps"]), float(twist["w_rad_s"]), float(twist["dt_command_s"])
            )
            before = observations[transition["obs_before_id"]]
            after = observations[transition["obs_after_id"]]
            if before["depth_missing"] or before["rgb_missing"] or after["rgb_missing"]:
                raise RuntimeError(f"development frame is missing: {transition['transition_id']}")
            scored = score_pair(
                load_checked(DEV, before["rgb"]),
                load_checked(DEV, after["rgb"]),
                depth_to_meters(load_checked(DEV, before["depth"])),
                forward,
                left,
                yaw,
            )
            scored["scene"] = scene_id(transition["episode_uid"])
            scored["kind"] = action_kind(twist)
            rows.append(scored)
            if len(rows) % 500 == 0:
                print(f"development {len(rows)}", flush=True)
    return rows


def fmt_mae(value: float) -> str:
    return f"{value:.4f}"


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(fit: dict, development: dict | None, verdict: dict) -> None:
    lines = [
        "# 阶段 A 深度自运动变换核对",
        "",
        "这是只读核对。深度是 Habitat 归一化后的光轴距离，米制为归一化值乘以 10。",
        "运动用命令而不是未来位姿：前向是 v 乘 dt，侧向是 0，偏航是 w 乘 dt。",
        "48 个 formal fit 分支只用来核对相机，按图像等权，不是泛化测试。",
        "开发集只有在这次核对通过之后才计分。比较用同一批有效像素。",
        "没有训练，没有启动 Habitat，没有写入正式特征缓存，也没有接入 LightNav。",
        "",
        f"判定：{verdict['call_label']}（`{verdict['call']}`）。",
        verdict["next_step"],
        "",
        "## 拟合分支上的相机核对",
        "",
        "| 动作 | n | 复制 MAE | 偏航平移 MAE | 深度变换 MAE | 相对复制 | 覆盖 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (*LEFT_ACTIONS, *RIGHT_ACTIONS, *STRAIGHT_ACTIONS):
        row = fit["actions"][name]
        lines.append(
            f"| {name} | {row['n']} | {fmt_mae(row['copy_mae'])} | {fmt_mae(row['yaw_mae'])} | "
            f"{fmt_mae(row['depth_mae'])} | {fmt_pct(row['depth_vs_copy'])} | {fmt_pct(row['coverage'])} |"
        )
    lines.extend(
        [
            "",
            f"左转合计相对复制 {fmt_pct(fit['left_relative'])}，右转合计 {fmt_pct(fit['right_relative'])}。",
            f"转向有效像素 {fmt_pct(fit['turn_coverage'])}。",
            "左右转都要高过 10%，直行不能变差超过 10%，转向覆盖不能低于 50%。",
            "",
        ]
    )
    if development is None:
        lines.append("拟合分支核对没有通过，开发集没有计分。")
    else:
        lines.extend(
            [
                "## 开发集",
                "",
                "| 分组 | n | 场景 | 复制 MAE | 偏航平移 MAE | 深度变换 MAE | 相对复制 | 相对偏航 | 覆盖 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        labels = {
            "all": "全部",
            "turning": "转向",
            "mostly_forward": "主要直行",
            "stop_threshold": "停止",
        }
        for kind, label in labels.items():
            row = development[kind]
            lines.append(
                f"| {label} | {row['n']} | {row['scenes']} | {fmt_mae(row['copy_mae'])} | "
                f"{fmt_mae(row['yaw_mae'])} | {fmt_mae(row['depth_mae'])} | "
                f"{fmt_pct(row['depth_vs_copy'])} | {fmt_pct(row['depth_vs_yaw'])} | "
                f"{fmt_pct(row['coverage'])} |"
            )
        lines.extend(["", "门槛是场景宏平均，并且每一项都要高过 10%。停止项不能高过 10%。", ""])
    lines.extend([verdict["next_step"], ""])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    prepare()
    print("indexing fit observations", flush=True)
    rows, fit = run_fit(load_states(), index_observations(FIT))
    del rows
    print(
        f"fit passed={fit['passed']} left={fit['left_relative']:.4f} right={fit['right_relative']:.4f}",
        flush=True,
    )
    if not fit["passed"]:
        verdict = ego_warp_verdict(fit_passed=False)
        write_json(OUT / "results.json", {"fit_check": fit, "development": None, "verdict": verdict})
        write_report(fit, None, verdict)
        print(verdict["call"], flush=True)
        return
    print("indexing development observations", flush=True)
    development_rows = run_development(index_observations(DEV))
    score_path = OUT / "development_scores.npz"
    if score_path.exists():
        raise FileExistsError(score_path)
    np.savez_compressed(
        score_path,
        copy_mae=np.array([row["copy_mae"] for row in development_rows], dtype=np.float64),
        yaw_mae=np.array([row["yaw_mae"] for row in development_rows], dtype=np.float64),
        depth_mae=np.array([row["depth_mae"] for row in development_rows], dtype=np.float64),
        coverage=np.array([row["coverage"] for row in development_rows], dtype=np.float64),
        scene=np.array([row["scene"] for row in development_rows]),
        kind=np.array([row["kind"] for row in development_rows]),
    )
    development = development_summary(development_rows)
    verdict = ego_warp_verdict(
        fit_passed=True,
        coverage=development["all"]["coverage"],
        stop_relative=development["stop_threshold"]["depth_vs_copy"],
        all_copy=development["all"]["copy_mae"],
        all_yaw=development["all"]["yaw_mae"],
        all_depth=development["all"]["depth_mae"],
        turn_copy=development["turning"]["copy_mae"],
        turn_yaw=development["turning"]["yaw_mae"],
        turn_depth=development["turning"]["depth_mae"],
        forward_copy=development["mostly_forward"]["copy_mae"],
        forward_depth=development["mostly_forward"]["depth_mae"],
    )
    write_json(OUT / "results.json", {"fit_check": fit, "development": development, "verdict": verdict})
    write_report(fit, development, verdict)
    print(verdict["call"], flush=True)


if __name__ == "__main__":
    main()
