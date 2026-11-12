"""Read-only check: copy the frame after a rejected command.

Uses the depth-warp scores already stored for the development set and the
recorded poses. It does not train, start Habitat, rewrite those scores, or
connect LightNav.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.world_model.baselines import executed_twist  # noqa: E402
from experiments.world_model.pose_math import body_frame_delta  # noqa: E402
from experiments.world_model.rejected_command import (  # noqa: E402
    COMMAND_FORWARD_M,
    COMMAND_YAW_RAD,
    IMPROVEMENT_MIN,
    MEASURED_FORWARD_M,
    MEASURED_YAW_RAD,
    OTHER_HARM_MAX,
    STUCK_VS_COPY_MAX,
    previous_command_rejected,
    rejection_verdict,
)
from experiments.world_model.residual_predictability import scene_macro  # noqa: E402
from experiments.world_model.yaw_shift import relative_improvement  # noqa: E402

OUT = REPO / "output/world_model_diagnostics/stage_a_rejected_command_20261102_01"
DEV = REPO / "output/world_model_data/formal_dev_20261028_01"
SCORES = REPO / "output/world_model_diagnostics/stage_a_ego_warp_20261102_01/development_scores.npz"
STUCK_SCENE = "2n8kARJN3HM"
DEV_TRANSITION_COUNT = 3249
DEV_SCENE_COUNT = 8


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prepare() -> None:
    OUT.mkdir()
    write_json(
        OUT / "protocol.json",
        {
            "experiment": "stage_a_rejected_command",
            "question": "does copying after a rejected previous command fix the stuck scene",
            "stuck_scene": STUCK_SCENE,
            "stuck_scene_reason": "position did not follow a forward command; center depth was several meters",
            "rule": "if the previous step in the same episode asked for motion and the measured body motion stayed near zero, predict the current frame; otherwise keep the depth warp",
            "command_forward_m": COMMAND_FORWARD_M,
            "command_yaw_rad": COMMAND_YAW_RAD,
            "measured_forward_m": MEASURED_FORWARD_M,
            "measured_yaw_rad": MEASURED_YAW_RAD,
            "uses_current_future_pose": False,
            "uses_previous_measured_pose": True,
            "same_pixels_as_depth_warp": True,
            "gate": "stuck scene improves on the depth warp by more than 10% and is not more than 10% worse than copy; the other seven scenes worsen by at most 1%; all eight still beat copy and yaw shift by more than 10%",
            "improvement_min": IMPROVEMENT_MIN,
            "other_harm_max": OTHER_HARM_MAX,
            "stuck_vs_copy_max": STUCK_VS_COPY_MAX,
            "development_is_blind_test": False,
            "generalization": False,
            "navigation_connected": False,
            "habitat_started": False,
            "formal_cache_written": False,
            "training_started": False,
        },
    )


def load_poses() -> dict[str, dict]:
    poses = {}
    with (DEV / "privileged_labels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            poses[row["label_id"]] = row["pose"]
    return poses


def measured_motion(poses: dict[str, dict], transition: dict) -> tuple[float, float]:
    before = poses[transition["pose_before_id"]]
    after = poses[transition["pose_after_id"]]
    forward, _left, yaw, _up = body_frame_delta(
        before["world_position_xyz_m"],
        before["derived_yaw_rad"],
        after["world_position_xyz_m"],
        after["derived_yaw_rad"],
    )
    return forward, yaw


def rejection_mask(transitions: list[dict], poses: dict[str, dict]) -> np.ndarray:
    measured = [measured_motion(poses, row) for row in transitions]
    commanded = []
    for row in transitions:
        twist = executed_twist(row)
        if twist["dt_command_s"] is None:
            raise RuntimeError(f"transition has no command interval: {row['transition_id']}")
        commanded.append(
            (float(twist["v_mps"]) * float(twist["dt_command_s"]), float(twist["w_rad_s"]) * float(twist["dt_command_s"]))
        )
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(transitions):
        groups[row["episode_uid"]].append(index)
    fire = np.zeros(len(transitions), dtype=bool)
    for indexes in groups.values():
        ordered = sorted(indexes, key=lambda index: transitions[index]["t"])
        times = [transitions[index]["t"] for index in ordered]
        if len(times) != len(set(times)):
            raise RuntimeError("an episode has two transitions at the same time")
        for previous, current in zip(ordered, ordered[1:]):
            fire[current] = previous_command_rejected(*commanded[previous], *measured[previous])
    return fire


def scene_of(transition: dict) -> str:
    parts = transition["episode_uid"].split("|")
    if len(parts) != 4 or not parts[2]:
        raise RuntimeError(f"episode uid has no scene: {transition['episode_uid']}")
    return parts[2]


def macro(values: np.ndarray, scenes: list[str]) -> float:
    return scene_macro(values, scenes)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(summary: dict, verdict: dict) -> None:
    lines = [
        "# 阶段 A 上一步命令被拒绝",
        "",
        "深度变换在命令真的执行时能预测下一帧。`2n8kARJN3HM` 里大部分步的命令平移没有发生，",
        "画面几乎不变，变换仍按命令挪动画面。这一步只使用上一步已经测到的位姿：",
        "上一步要求了运动但身体几乎没动，这一步就预测画面不变；否则仍用深度变换。",
        "不用这一步的未来位姿。没有训练，没有启动 Habitat，没有改写已有分数，也没有接入 LightNav。",
        "",
        f"判定：{verdict['call_label']}（`{verdict['call']}`）。",
        verdict["next_step"],
        "",
        "## 开发集",
        "",
        "| 范围 | n | 复制 MAE | 偏航平移 MAE | 深度变换 MAE | 修正后 MAE | 相对深度变换 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("stuck", STUCK_SCENE), ("other", "其余七个场景"), ("all", "全部八个场景")):
        row = summary[key]
        lines.append(
            f"| {label} | {row['n']} | {fmt(row['copy'])} | {fmt(row['yaw'])} | {fmt(row['warp'])} | "
            f"{fmt(row['rule'])} | {fmt_pct(row['rule_vs_warp'])} |"
        )
    lines.extend(
        [
            "",
            f"规则触发 {summary['fires']} 步，其中 {summary['stuck_fires']} 步在 `{STUCK_SCENE}`。",
            "卡住的场景要比深度变换低过 10%，并且不能比复制差过 10%。",
            "其余场景变差不能超过 1%。全部八个场景仍要比复制和偏航平移都低过 10%。",
            "",
            verdict["next_step"],
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    prepare()
    if not SCORES.is_file():
        raise RuntimeError("depth-warp scores are missing")
    stored = np.load(SCORES)
    transitions = [json.loads(line) for line in (DEV / "transitions.jsonl").open(encoding="utf-8")]
    if len(transitions) != DEV_TRANSITION_COUNT or len(stored["copy_mae"]) != DEV_TRANSITION_COUNT:
        raise RuntimeError("development scores and transitions do not have the registered length")
    scenes = [scene_of(row) for row in transitions]
    if scenes != stored["scene"].tolist():
        raise RuntimeError("development scores are not in transition order")
    if len(set(scenes)) != DEV_SCENE_COUNT or STUCK_SCENE not in set(scenes):
        raise RuntimeError("development scenes do not match the registered partition")
    fire = rejection_mask(transitions, load_poses())
    copy = stored["copy_mae"]
    yaw = stored["yaw_mae"]
    warp = stored["depth_mae"]
    rule = np.where(fire, copy, warp)
    stuck = np.array([scene == STUCK_SCENE for scene in scenes])
    other = ~stuck

    def pack(mask: np.ndarray) -> dict[str, float | int]:
        chosen_scenes = [scene for scene, keep in zip(scenes, mask) if keep]
        copy_mae = macro(copy[mask], chosen_scenes)
        yaw_mae = macro(yaw[mask], chosen_scenes)
        warp_mae = macro(warp[mask], chosen_scenes)
        rule_mae = macro(rule[mask], chosen_scenes)
        return {
            "n": int(mask.sum()),
            "scenes": len(set(chosen_scenes)),
            "copy": copy_mae,
            "yaw": yaw_mae,
            "warp": warp_mae,
            "rule": rule_mae,
            "rule_vs_warp": relative_improvement(warp_mae, rule_mae),
            "rule_vs_copy": relative_improvement(copy_mae, rule_mae),
        }

    summary = {
        "stuck": pack(stuck),
        "other": pack(other),
        "all": pack(np.ones(len(scenes), dtype=bool)),
        "fires": int(fire.sum()),
        "stuck_fires": int(fire[stuck].sum()),
    }
    verdict = rejection_verdict(
        stuck_warp=summary["stuck"]["warp"],
        stuck_rule=summary["stuck"]["rule"],
        stuck_copy=summary["stuck"]["copy"],
        other_warp=summary["other"]["warp"],
        other_rule=summary["other"]["rule"],
        all_copy=summary["all"]["copy"],
        all_yaw=summary["all"]["yaw"],
        all_rule=summary["all"]["rule"],
    )
    write_json(OUT / "results.json", {"summary": summary, "verdict": verdict})
    write_report(summary, verdict)
    print(verdict["call"], flush=True)


if __name__ == "__main__":
    main()
