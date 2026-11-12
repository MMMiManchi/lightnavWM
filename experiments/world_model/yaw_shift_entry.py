"""Read-only check: does a commanded-yaw pixel shift explain the branch frames?

Uses the saved PNGs from the existing branch probe. It does not train, start
Habitat, write the feature cache, or connect LightNav.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.world_model.action_response_entry import load_png, load_states  # noqa: E402
from experiments.world_model.yaw_shift import (  # noqa: E402
    DT_S,
    HEIGHT,
    HFOV_DEG,
    LEFT_ACTIONS,
    RIGHT_ACTIONS,
    STRAIGHT_ACTIONS,
    TURN_RATE_DPS,
    WIDTH,
    command_yaw_rad,
    integer_shift,
    masked_mae,
    relative_improvement,
    shift_content_right,
    yaw_shift_verdict,
)

OUT = REPO / "output/world_model_diagnostics/stage_a_yaw_shift_20261102_01"
TURN_SHIFT = integer_shift(command_yaw_rad(TURN_RATE_DPS))


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prepare() -> None:
    OUT.mkdir()
    write_json(
        OUT / "protocol.json",
        {
            "experiment": "stage_a_yaw_shift",
            "warp": "global horizontal shift of yaw / hfov * width",
            "sign": "positive yaw is a left turn and moves scene content right",
            "width": WIDTH,
            "height": HEIGHT,
            "hfov_deg": HFOV_DEG,
            "dt_s": DT_S,
            "turn_shift_px": TURN_SHIFT,
            "mask": "pixels shifted in from outside the frame are excluded; copy uses the same pixels",
            "improvement_min": 0.10,
            "gate": "left and right pooled MAE each improve by more than 10%; straight command shifts and both imposed turn shifts on straight stay at or below 10%",
            "state_weight": "equal per state",
            "states_are_formal_fit": True,
            "generalization": False,
            "navigation_connected": False,
            "habitat_started": False,
            "formal_cache_written": False,
            "training_started": False,
        },
    )


def score_pair(before: np.ndarray, after: np.ndarray, shift_px: int) -> tuple[float, float]:
    warped, mask = shift_content_right(before, shift_px)
    copy_mae = masked_mae(after, before, mask)
    warp_mae = masked_mae(after, warped, mask)
    return copy_mae, warp_mae


def require_frame(image: np.ndarray, path: Path) -> None:
    if image.shape != (HEIGHT, WIDTH, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"frame is not {WIDTH}x{HEIGHT} uint8 RGB: {path}")


def collect() -> tuple[dict, dict, int]:
    states = load_states()
    command: dict[str, list[tuple[float, float]]] = {}
    imposed: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for state in states:
        before = load_png(state["before"])
        require_frame(before, state["before"])
        for branch in state["branches"]:
            after = load_png(branch["image"])
            require_frame(after, branch["image"])
            name = branch["name"]
            shift = integer_shift(branch["w_rad_s"] * DT_S)
            command.setdefault(name, []).append(score_pair(before, after, shift))
            if name in STRAIGHT_ACTIONS:
                bucket = imposed.setdefault(name, {TURN_SHIFT: [], -TURN_SHIFT: []})
                for imposed_shift in (TURN_SHIFT, -TURN_SHIFT):
                    bucket[imposed_shift].append(score_pair(before, after, imposed_shift))
    return command, imposed, len({state["scene"] for state in states})


def pooled(rows: list[tuple[float, float]]) -> tuple[float, float, float]:
    copy_mae = float(np.mean([copy for copy, _warp in rows]))
    warp_mae = float(np.mean([warp for _copy, warp in rows]))
    return copy_mae, warp_mae, relative_improvement(copy_mae, warp_mae)


def action_rows(command: dict[str, list[tuple[float, float]]]) -> list[dict]:
    rows = []
    for name in (*LEFT_ACTIONS, *RIGHT_ACTIONS, *STRAIGHT_ACTIONS):
        copy_mae, warp_mae, relative = pooled(command[name])
        rows.append(
            {
                "action": name,
                "n": len(command[name]),
                "copy_mae": copy_mae,
                "warp_mae": warp_mae,
                "relative_improvement": relative,
            }
        )
    return rows


def imposed_rows(imposed: dict[str, dict[int, list[tuple[float, float]]]]) -> list[dict]:
    rows = []
    for name in STRAIGHT_ACTIONS:
        for shift in (TURN_SHIFT, -TURN_SHIFT):
            copy_mae, warp_mae, relative = pooled(imposed[name][shift])
            rows.append(
                {
                    "action": name,
                    "shift_px": shift,
                    "n": len(imposed[name][shift]),
                    "copy_mae": copy_mae,
                    "warp_mae": warp_mae,
                    "relative_improvement": relative,
                }
            )
    return rows


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_report(payload: dict) -> str:
    verdict = payload["verdict"]
    lines = [
        "# 阶段 A 命令偏航平移核对",
        "",
        "这是只读核对。48 个状态来自 formal fit，按状态等权，不是泛化测试。",
        "没有训练，没有启动 Habitat，没有写入正式特征缓存，也没有接入 LightNav。",
        "正偏航是左转。左转时画面内容向右移动，平移量是偏航角除以 120° 再乘以 480，正好 24 个像素。",
        "从画面外移入的像素不参与比较，复制前一帧也只用同一批像素。",
        "",
        f"判定：{verdict['call_label']}（`{verdict['call']}`）。",
        verdict["next_step"],
        "",
        "## 按命令偏航平移",
        "",
        "| 动作 | n | 复制 MAE | 平移 MAE | 相对改善 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["actions"]:
        lines.append(
            f"| {row['action']} | {row['n']} | {row['copy_mae']:.4f} | "
            f"{row['warp_mae']:.4f} | {_percent(row['relative_improvement'])} |"
        )
    lines.extend(
        [
            "",
            f"左转合计相对改善 {_percent(payload['left_relative'])}，"
            f"右转合计 {_percent(payload['right_relative'])}。",
            "门槛是左右转都要高过 10%。",
            "",
            "## 把 24 像素平移加到直行上",
            "",
            "| 动作 | 平移像素 | 复制 MAE | 平移 MAE | 相对改善 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["straight_imposed"]:
        lines.append(
            f"| {row['action']} | {row['shift_px']} | {row['copy_mae']:.4f} | "
            f"{row['warp_mae']:.4f} | {_percent(row['relative_improvement'])} |"
        )
    lines.extend(["", "直行上的这组改善不能高过 10%。", "", verdict["next_step"], ""])
    return "\n".join(lines)


def main() -> None:
    prepare()
    command, imposed, scenes = collect()
    expected = (*LEFT_ACTIONS, *RIGHT_ACTIONS, *STRAIGHT_ACTIONS)
    missing = [name for name in expected if name not in command]
    if missing:
        raise RuntimeError(f"branch actions are missing: {missing}")
    left_copy, left_warp, left_relative = pooled([row for name in LEFT_ACTIONS for row in command[name]])
    right_copy, right_warp, right_relative = pooled(
        [row for name in RIGHT_ACTIONS for row in command[name]]
    )
    command_rows = action_rows(command)
    extra_rows = imposed_rows(imposed)
    straight_command_rels = [
        row["relative_improvement"] for row in command_rows if row["action"] in STRAIGHT_ACTIONS
    ]
    verdict = yaw_shift_verdict(
        left_relative=left_relative,
        right_relative=right_relative,
        straight_command_relatives=straight_command_rels,
        straight_imposed_relatives=[row["relative_improvement"] for row in extra_rows],
    )
    payload = {
        "actions": command_rows,
        "straight_imposed": extra_rows,
        "left_copy_mae": left_copy,
        "left_warp_mae": left_warp,
        "left_relative": left_relative,
        "right_copy_mae": right_copy,
        "right_warp_mae": right_warp,
        "right_relative": right_relative,
        "branch_scenes": scenes,
        "turn_shift_px": TURN_SHIFT,
        "verdict": verdict,
    }
    write_json(OUT / "results.json", payload)
    with (OUT / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(render_report(payload))
    print(json.dumps({"call": verdict["call"], "next_step": verdict["next_step"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
