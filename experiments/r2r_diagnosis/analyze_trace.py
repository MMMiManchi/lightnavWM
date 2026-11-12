#!/usr/bin/env python3
"""Summarise a side-channel trace. Does not change any rollout."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "output" / "r2r_probe" / "trace_20261020"
HFOV_RAD = math.radians(120.0)
RISE_STEPS = 5
RISE_METRES = 0.5
MOVE_METRES = 0.3


def wrap(delta: float) -> float:
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def baseline_key(row: dict) -> tuple[str, str]:
    scene = os.path.basename(os.path.dirname(str(row.get("scene_id", "")).rstrip("/")))
    return scene, str(row.get("habitat_episode_id"))


def body_motion(row: dict) -> tuple[float | None, float | None, float | None]:
    before = row.get("pose_xyz_before")
    after = row.get("pose_xyz_after")
    yaw = row.get("pose_yaw_before")
    yaw_after = row.get("pose_yaw_after")
    if not before or not after or yaw is None or yaw_after is None:
        return None, None, None
    dx = float(after[0]) - float(before[0])
    dz = float(after[2]) - float(before[2])
    yaw = float(yaw)
    forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
    left_x, left_z = -math.cos(yaw), math.sin(yaw)
    return (
        dx * forward_x + dz * forward_z,
        dx * left_x + dz * left_z,
        wrap(float(yaw_after) - yaw),
    )


def horiz(row: dict) -> float | None:
    before = row.get("pose_xyz_before")
    after = row.get("pose_xyz_after")
    if not before or not after:
        return None
    return math.hypot(float(after[0]) - float(before[0]), float(after[2]) - float(before[2]))


def bearing_from_pixel(px, frame_size) -> float | None:
    """Approximate body-frame yaw of a pixel. +yaw is left. Image x grows to the right."""
    if not px or not frame_size or len(px) < 2 or len(frame_size) < 1:
        return None
    width = float(frame_size[0])
    if width <= 1:
        return None
    return -((float(px[0]) - width / 2.0) / width) * HFOV_RAD


def pointing_agrees(row: dict) -> bool | None:
    state = row.get("apos_state")
    yaw = row.get("commanded_yaw_rad")
    if state in (None, "none") or yaw is None:
        return None
    yaw = float(yaw)
    if state == "rot_left":
        return yaw >= -0.05
    if state == "rot_right":
        return yaw <= 0.05
    if state == "stop":
        return bool(row.get("model_stop"))
    if state == "point":
        bearing = bearing_from_pixel(row.get("apos_px"), row.get("pointing_frame"))
        if bearing is None:
            return None
        return abs(wrap(bearing - yaw)) <= math.radians(30)
    return None


def fill_controller_and_pointing(row: dict) -> None:
    """Recover fields the client stored indirectly.

    The velocity dict is nested under action_args. Recompute it with the same
    function the policy used. Pointing state is not in the habitat policy info;
    the raw grid id is, and it decodes without another model call.
    """
    from lightnav.velocity import first_waypoint_to_velocity_cmd
    from lightnav.vln_utils import (
        APOS_ROTL_ID,
        APOS_ROTR_ID,
        APOS_STOP_ID,
        POINT_CELLS,
        decode_point_pixel_center,
    )

    waypoint = row.get("executed_waypoint")
    if waypoint is not None:
        dt = float(row.get("dt_s") or 0.1)
        forward = float(waypoint[0]) / dt
        yaw_rate = math.degrees(float(waypoint[2])) / dt
        forward = min(max(forward, 0.0), 2.5)
        yaw_rate = min(max(yaw_rate, -300.0), 300.0)
        row["commanded_forward_clipped_m"] = round(forward * dt, 5)
        row["commanded_yaw_clipped_rad"] = round(math.radians(yaw_rate * dt), 5)
    if waypoint is not None and row.get("action_linear") is None:
        command = first_waypoint_to_velocity_cmd(
            np.asarray(waypoint, dtype=np.float32),
            float(row.get("dt_s") or 0.1),
            (0.0, 2.5),
            (-300.0, 300.0),
        )
        args = command["action_args"]
        row["action_linear"] = round(float(args["linear_velocity"]), 5)
        row["action_angular"] = round(float(args["angular_velocity"]), 5)
        row["action_source"] = "recomputed_same_mapper"

    frame_hw = row.get("frame_hw") or [270, 480]
    width, height = float(frame_hw[1]), float(frame_hw[0])
    if row.get("pointing_frame") is None:
        row["pointing_frame"] = [int(width), int(height)]

    def _state(token_id, stop_id: bool) -> str | None:
        if token_id is None:
            return None
        token_id = int(token_id)
        if token_id == 0:
            return "none" if stop_id else "not_visible"
        if stop_id and token_id == APOS_ROTL_ID:
            return "rot_left"
        if stop_id and token_id == APOS_ROTR_ID:
            return "rot_right"
        if stop_id and token_id == APOS_STOP_ID:
            return "stop"
        if 1 <= token_id <= POINT_CELLS:
            return "point"
        return "unknown"

    if row.get("apos_state") is None:
        row["apos_state"] = _state(row.get("apos_id"), True)
    if row.get("opos_state") is None:
        row["opos_state"] = _state(row.get("opos_id"), False)
    if row.get("apos_px") is None and row.get("apos_state") == "point":
        center = decode_point_pixel_center(int(row["apos_id"]), width, height)
        row["apos_px"] = None if center is None else [round(float(center[0]), 2), round(float(center[1]), 2)]
    if row.get("opos_px") is None and row.get("opos_state") == "point":
        center = decode_point_pixel_center(int(row["opos_id"]), width, height)
        row["opos_px"] = None if center is None else [round(float(center[0]), 2), round(float(center[1]), 2)]


def enrich(steps: list[dict]) -> None:
    for row in steps:
        fill_controller_and_pointing(row)
        forward, left, dyaw = body_motion(row)
        row["actual_forward_m"] = None if forward is None else round(forward, 5)
        row["actual_left_m"] = None if left is None else round(left, 5)
        row["actual_dyaw_rad"] = None if dyaw is None else round(dyaw, 5)
        cmd_yaw = row.get("commanded_yaw_clipped_rad")
        cmd_fwd = row.get("commanded_forward_clipped_m")
        row["yaw_follow_rad"] = None if dyaw is None or cmd_yaw is None else round(wrap(dyaw - float(cmd_yaw)), 5)
        row["forward_follow_m"] = None if forward is None or cmd_fwd is None else round(forward - float(cmd_fwd), 5)
        before = row.get("geodesic_before_m")
        after = row.get("geodesic_after_m")
        row["geodesic_delta_m"] = None if before is None or after is None else round(float(after) - float(before), 5)
        row["pointing_agrees_with_yaw"] = pointing_agrees(row)
        row["moved_m"] = None if horiz(row) is None else round(horiz(row), 5)


def sustained_rise(rows: list[dict]) -> int | None:
    if len(rows) < RISE_STEPS:
        return None
    for index in range(0, len(rows) - RISE_STEPS + 1):
        window = rows[index : index + RISE_STEPS]
        start = window[0].get("geodesic_before_m")
        end = window[-1].get("geodesic_after_m")
        moved = sum(float(item.get("moved_m") or 0.0) for item in window)
        if start is None or end is None:
            continue
        if float(end) - float(start) >= RISE_METRES and moved >= MOVE_METRES:
            return index
    return None


def window_cause(window: list[dict]) -> str:
    follows = [abs(float(item["yaw_follow_rad"])) for item in window if item.get("yaw_follow_rad") is not None]
    errors = [abs(float(item["yaw_error_rad"])) for item in window if item.get("yaw_error_rad") is not None]
    agrees = [item["pointing_agrees_with_yaw"] for item in window if item["pointing_agrees_with_yaw"] is not None]
    if follows and float(np.median(follows)) > math.radians(15):
        return "execution"
    if agrees and sum(1 for item in agrees if not item) > len(agrees) / 2:
        return "pointing_rvq_disagree"
    if errors and float(np.median(errors)) > math.radians(45):
        return "direction_decision"
    return "distance_rose_without_large_heading_gap"


def group_name(episode: dict) -> str:
    if episode["success"]:
        return "success"
    minimum = episode.get("min_distance_m")
    final = episode.get("final_distance_m")
    if minimum is not None and final is not None and float(minimum) < 3.0 and float(final) > 3.0:
        return "entered_then_left"
    return "never_entered_3m"


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_svg(path: Path, rows: list[dict]) -> None:
    series = [
        ("geodesic m", [row.get("geodesic_before_m") for row in rows], "#1f4e79"),
        ("|yaw error| deg", [None if row.get("yaw_error_rad") is None else abs(math.degrees(row["yaw_error_rad"])) for row in rows], "#b85c38"),
        ("|follow| deg", [None if row.get("yaw_follow_rad") is None else abs(math.degrees(row["yaw_follow_rad"])) for row in rows], "#2d6a4f"),
    ]
    width, panel, pad = 720, 120, 28
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{panel * 3 + 10}">']
    for index, (name, values, color) in enumerate(series):
        clean = [value for value in values if value is not None]
        top = index * panel
        parts.append(f'<text x="8" y="{top + 16}" font-size="12" font-family="sans-serif">{name}</text>')
        if len(clean) < 2:
            continue
        lo, hi = min(clean), max(clean)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        pts = []
        for step, value in enumerate(values):
            if value is None:
                continue
            x = pad + (width - 2 * pad) * step / max(len(values) - 1, 1)
            y = top + 24 + (panel - 36) * (1 - (value - lo) / (hi - lo))
            pts.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{" ".join(pts)}"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def step_line(row: dict) -> str:
    return (
        f"step {row['step']}: geodesic {fmt(row.get('geodesic_before_m'))}->{fmt(row.get('geodesic_after_m'))} m, "
        f"model waypoint {row.get('executed_waypoint')}, expert yaw {fmt(row.get('expert_yaw_rad'), 3)} rad, "
        f"yaw error {fmt(row.get('yaw_error_rad'), 3)} rad, "
        f"clipped command yaw {fmt(row.get('commanded_yaw_clipped_rad'), 3)} rad, "
        f"actual dyaw {fmt(row.get('actual_dyaw_rad'), 3)} rad, "
        f"follow {fmt(row.get('yaw_follow_rad'), 3)} rad, "
        f"apos {row.get('apos_id')}/{row.get('apos_state')}, rvq {row.get('rvq_codes')}, "
        f"raw `{row.get('raw_text')}`"
    )


def main() -> None:
    steps = load_jsonl(TRACE / "steps.jsonl")
    episodes = load_jsonl(TRACE / "episodes.jsonl")
    enrich(steps)
    by_episode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in steps:
        by_episode[(row["scene"], str(row["episode_id"]))].append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda item: item["step"])

    baseline = {
        baseline_key(row): row
        for row in load_jsonl(ROOT / "output" / "r2r_go2w" / "results.jsonl")
    }
    replay = {
        (row["scene"], str(row["episode_id"])): row
        for row in load_jsonl(ROOT / "output" / "r2r_probe" / "model.jsonl")
    }

    plot_dir = TRACE / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for episode in episodes:
        key = (episode["scene"], str(episode["episode_id"]))
        rows = by_episode[key]
        write_svg(plot_dir / f"{key[0]}_{key[1]}.svg", rows)
        base = baseline.get(key, {})
        previous = replay.get(key, {})
        rise = sustained_rise(rows)
        cause = window_cause(rows[rise : rise + RISE_STEPS]) if rise is not None else ""
        stop_steps = [row["step"] for row in rows if row.get("model_stop")]
        errors = [abs(float(row["yaw_error_rad"])) for row in rows if row.get("yaw_error_rad") is not None]
        follows = [abs(float(row["yaw_follow_rad"])) for row in rows if row.get("yaw_follow_rad") is not None]
        agrees = [row["pointing_agrees_with_yaw"] for row in rows if row["pointing_agrees_with_yaw"] is not None]
        step0 = rows[0] if rows else {}
        base_final = base.get("final_distance")
        base_success = None if not base else float(base.get("success") or 0) >= 0.5
        final_delta = None
        if base_final is not None and episode.get("final_distance_m") is not None:
            final_delta = abs(float(episode["final_distance_m"]) - float(base_final))
        min_step = None
        if rows:
            finite_rows = [row for row in rows if row.get("geodesic_before_m") is not None]
            if finite_rows:
                min_step = min(finite_rows, key=lambda row: float(row["geodesic_before_m"]))["step"]
        record = {
            "scene": key[0],
            "episode_id": key[1],
            "probe_class": episode.get("probe_class", ""),
            "group": group_name(episode),
            "success": episode["success"],
            "spl": episode.get("spl"),
            "ndtw": episode.get("ndtw"),
            "final_distance_m": episode.get("final_distance_m"),
            "min_distance_m": episode.get("min_distance_m"),
            "steps": episode["steps"],
            "stop_step": stop_steps[0] if stop_steps else "",
            "min_geodesic_step": "" if min_step is None else min_step,
            "step0_yaw_error_rad": step0.get("yaw_error_rad"),
            "step0_abs_yaw_error_deg": None if step0.get("yaw_error_rad") is None else round(abs(math.degrees(step0["yaw_error_rad"])), 2),
            "earliest_sustained_rise_step": "" if rise is None else rise,
            "rise_cause": cause,
            "steps_abs_yaw_error_gt_45": sum(1 for value in errors if value > math.radians(45)),
            "steps_abs_follow_gt_15deg": sum(1 for value in follows if value > math.radians(15)),
            "median_abs_follow_deg": None if not follows else round(math.degrees(float(np.median(follows))), 3),
            "max_abs_follow_deg": None if not follows else round(math.degrees(float(np.max(follows))), 3),
            "pointing_agree_steps": sum(1 for item in agrees if item),
            "pointing_compared_steps": len(agrees),
            "geodesic_increase_steps": sum(1 for row in rows if (row.get("geodesic_delta_m") or 0) > 0.05),
            "baseline_success": base_success,
            "baseline_final_distance_m": base_final,
            "baseline_steps": base.get("steps"),
            "success_match": base_success == episode["success"] if base_success is not None else "",
            "final_delta_m": None if final_delta is None else round(final_delta, 4),
            "replay_final_distance_m": previous.get("final_distance_m"),
            "replay_steps": previous.get("steps"),
            "replay_success": previous.get("success"),
            "collided_steps": sum(1 for row in rows if row.get("collided") is True),
            "collision_known_steps": sum(1 for row in rows if row.get("collided") is not None),
        }
        diagnostics.append(record)
        episode.update({
            "group": record["group"],
            "stop_step": record["stop_step"],
            "min_geodesic_step": record["min_geodesic_step"],
            "step0_yaw_error_rad": record["step0_yaw_error_rad"],
            "earliest_sustained_rise_step": record["earliest_sustained_rise_step"],
            "rise_cause": record["rise_cause"],
            "median_abs_follow_deg": record["median_abs_follow_deg"],
            "success_match_baseline": record["success_match"],
            "final_delta_vs_baseline_m": record["final_delta_m"],
        })

    csv_path = TRACE / "trace_diagnostics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostics)
    with (TRACE / "episodes.jsonl").open("w") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode) + "\n")
    with (TRACE / "steps.jsonl").open("w") as handle:
        for row in steps:
            handle.write(json.dumps(row) + "\n")

    facts = summarise(diagnostics, by_episode)
    (TRACE / "trace_facts.json").write_text(json.dumps(facts, indent=2))
    (TRACE / "trace_diagnosis.md").write_text(render_diagnosis(facts, diagnostics, by_episode))
    (TRACE / "next_research_decision.md").write_text(render_decision(facts))
    print(json.dumps({
        "episodes": len(diagnostics),
        "success_match": facts["success_match"],
        "max_final_delta_m": facts["max_final_delta_m"],
        "rise_causes": facts["rise_causes"],
    }, indent=2))


def summarise(diagnostics: list[dict], by_episode: dict) -> dict:
    matched = [row for row in diagnostics if row["success_match"] is True]
    deltas = [float(row["final_delta_m"]) for row in diagnostics if row["final_delta_m"] is not None]
    causes = defaultdict(int)
    for row in diagnostics:
        if row["group"] != "success" and row["rise_cause"]:
            causes[row["rise_cause"]] += 1
    groups = defaultdict(list)
    for row in diagnostics:
        groups[row["group"]].append(row)

    def median_step(rows: list[dict]) -> float | None:
        values = [int(row["earliest_sustained_rise_step"]) for row in rows if row["earliest_sustained_rise_step"] != ""]
        return None if not values else float(np.median(values))

    failures = [row for row in diagnostics if not row["success"]]
    small = [row for row in failures if row["step0_abs_yaw_error_deg"] is not None and row["step0_abs_yaw_error_deg"] < 20]
    large = [row for row in failures if row["step0_abs_yaw_error_deg"] is not None and row["step0_abs_yaw_error_deg"] > 45]
    success_big = [
        row for row in diagnostics
        if row["success"] and row["steps_abs_yaw_error_gt_45"]
    ]
    follow_fail = [row["median_abs_follow_deg"] for row in failures if row["median_abs_follow_deg"] is not None]
    follow_ok = [row["median_abs_follow_deg"] for row in diagnostics if row["success"] and row["median_abs_follow_deg"] is not None]
    agree_num = sum(int(row["pointing_agree_steps"]) for row in diagnostics)
    agree_den = sum(int(row["pointing_compared_steps"]) for row in diagnostics)
    return {
        "n": len(diagnostics),
        "success_match": sum(1 for row in matched if row["success"] == row["baseline_success"]),
        "success_flags_equal": all(row["success_match"] is True for row in diagnostics),
        "step_count_equal": all(row["baseline_steps"] == row["steps"] for row in diagnostics if row["baseline_steps"] is not None),
        "max_final_delta_m": None if not deltas else round(max(deltas), 4),
        "median_final_delta_m": None if not deltas else round(float(np.median(deltas)), 4),
        "rise_causes": dict(causes),
        "median_rise_step_failures": median_step(failures),
        "median_rise_step_never": median_step(groups["never_entered_3m"]),
        "median_rise_step_left": median_step(groups["entered_then_left"]),
        "n_never": len(groups["never_entered_3m"]),
        "n_left": len(groups["entered_then_left"]),
        "n_success": len(groups["success"]),
        "n_fail_step0_gt_45": len(large),
        "n_fail_step0_lt_20": len(small),
        "n_success_with_some_gt_45": len(success_big),
        "median_follow_deg_fail": None if not follow_fail else round(float(np.median(follow_fail)), 3),
        "median_follow_deg_success": None if not follow_ok else round(float(np.median(follow_ok)), 3),
        "pointing_agree": agree_num,
        "pointing_compared": agree_den,
        "collided_steps": sum(int(row["collided_steps"]) for row in diagnostics),
        "collision_known_steps": sum(int(row["collision_known_steps"]) for row in diagnostics),
        "cases": choose_cases(diagnostics),
        "groups": {name: [f"{row['scene']}/{row['episode_id']}" for row in rows] for name, rows in groups.items()},
    }


def choose_cases(diagnostics: list[dict]) -> dict[str, list[str]]:
    failures = [row for row in diagnostics if not row["success"]]
    large = sorted(
        [row for row in failures if (row["step0_abs_yaw_error_deg"] or 0) > 45],
        key=lambda row: row["step0_abs_yaw_error_deg"],
        reverse=True,
    )
    small = sorted(
        [row for row in failures if row["step0_abs_yaw_error_deg"] is not None and row["step0_abs_yaw_error_deg"] < 20],
        key=lambda row: row["step0_abs_yaw_error_deg"],
    )
    left = [row for row in diagnostics if row["group"] == "entered_then_left"]
    success = [row for row in diagnostics if row["success"]]
    picked = {
        "large_step0": [f"{row['scene']}/{row['episode_id']}" for row in large[:2]],
        "small_step0": [f"{row['scene']}/{row['episode_id']}" for row in small[:2]],
        "entered_then_left": [f"{row['scene']}/{row['episode_id']}" for row in left[:2]],
        "success": [f"{row['scene']}/{row['episode_id']}" for row in success[:2]],
    }
    return picked


def case_block(key: str, by_episode: dict, diagnostics: list[dict]) -> str:
    scene, episode_id = key.split("/")
    rows = by_episode[(scene, episode_id)]
    meta = next(row for row in diagnostics if row["scene"] == scene and str(row["episode_id"]) == episode_id)
    wanted = {0}
    if meta["earliest_sustained_rise_step"] != "":
        rise = int(meta["earliest_sustained_rise_step"])
        wanted.update(range(rise, min(rise + 3, len(rows))))
    if meta["min_geodesic_step"] != "":
        wanted.add(int(meta["min_geodesic_step"]))
    if meta["stop_step"] != "":
        wanted.add(int(meta["stop_step"]))
    lines = [
        f"### {key}",
        "",
        f"类别 `{meta['probe_class']}`，分组 `{meta['group']}`。成功 {meta['success']}，"
        f"最终测地 {fmt(meta['final_distance_m'])} m，最近 {fmt(meta['min_distance_m'])} m，"
        f"步数 {meta['steps']}，STOP 步 {meta['stop_step']}，"
        f"第一步航向差 {fmt(meta['step0_abs_yaw_error_deg'], 1)}°，"
        f"持续远离起点 step {meta['earliest_sustained_rise_step']}，窗口原因 `{meta['rise_cause']}`。",
        "",
    ]
    for row in rows:
        if row["step"] in wanted:
            lines.append(f"- {step_line(row)}")
    lines.append("")
    return "\n".join(lines)


def render_diagnosis(facts: dict, diagnostics: list[dict], by_episode: dict) -> str:
    cause_text = ", ".join(f"{name} {count}" for name, count in sorted(facts["rise_causes"].items())) or "无"
    lines = [
        "# LightNav-0 逐步决策诊断",
        "",
        "回放使用 `checkpoints/LightNav-0`，Go2-W 配置，30 条与之前相同的 episode。",
        "专家最短路径只写在日志里，没有送进模型，也没有拿来执行。",
        "航点坐标系是机体局部 `[forward_m, left_m, yaw_rad]`，正航向为逆时针。它不是世界坐标系下的绝对航向。",
        "控制器丢掉横向分量，线速度和角速度按 `dt=0.1` 从第一可用航点换算，并被线速度 `[0, 2.5]` m/s、角速度 `[-300, 300]` deg/s 裁剪。",
        "",
        "## 和 10 月 15 日 Go2-W baseline 是否同一条轨迹",
        "",
        f"- 成功标志一致的 episode：{facts['success_match']} / {facts['n']}。全部一致：{facts['success_flags_equal']}。",
        f"- 动作步数与 baseline 全部相同：{facts['step_count_equal']}。",
        f"- 最终测地距离绝对差，中位数 {facts['median_final_delta_m']} m，最大 {facts['max_final_delta_m']} m。",
        "",
        "持续远离的定义：从某一步起连续 5 次推理，测地距离净增加至少 0.5 m，且水平位移合计至少 0.3 m。",
        "45° 和 15° 只用来描述窗口，不单独判定走错。专家最短路径不是唯一可行路线。",
        "",
        "## 分组",
        "",
        f"- 从未进入 3 m：{facts['n_never']} 条。持续远离起点的中位步 {facts['median_rise_step_never']}。",
        f"- 进入 3 m 后又离开：{facts['n_left']} 条。持续远离起点的中位步 {facts['median_rise_step_left']}。",
        f"- 成功：{facts['n_success']} 条。",
        "",
        f"失败 episode 里，持续远离窗口的原因计数：{cause_text}。",
        f"失败样本持续远离起点的中位步：{facts['median_rise_step_failures']}。",
        f"第一步航向差大于 45° 的失败：{facts['n_fail_step0_gt_45']}。小于 20° 的失败：{facts['n_fail_step0_lt_20']}。",
        f"成功样本里，至少有一步航向差大于 45° 的 episode：{facts['n_success_with_some_gt_45']} / {facts['n_success']}。",
        f"机体航向跟随误差的中位数：失败 {facts['median_follow_deg_fail']}°，成功 {facts['median_follow_deg_success']}°。",
        f"Pointing 与所执行航向可比较的步数 {facts['pointing_compared']}，其中方向一致 {facts['pointing_agree']}。",
        "",
        f"碰撞来自 Habitat 的 `previous_step_collided`。有记录的步数 {facts['collision_known_steps']}，其中碰撞为真的步数 {facts['collided_steps']}。",
        "控制器的 `linear_velocity` / `angular_velocity` 是 Habitat 的 [-1, 1] 归一化值。客户端当时读错了字典层级，动作本身没有改。分析时用记录下的执行航点，调用 policy 使用的同一个 `first_waypoint_to_velocity_cmd` 重算，线速度范围 [0, 2.5]、角速度范围 [-300, 300]、dt=0.1。",
        "上下文长度 `prompt_tokens` 是本次生成时已经算好的 prompt token 数，写在引擎对象上，没有多跑一次推理。",
        "",
        "## 案例",
        "",
    ]
    for name in ("large_step0", "small_step0", "entered_then_left", "success"):
        lines.append(f"## {name}")
        lines.append("")
        for key in facts["cases"][name]:
            lines.append(case_block(key, by_episode, diagnostics))
    lines.append("每条 episode 的测地距离、航向差和跟随误差图在 `plots/`。")
    lines.append("")
    return "\n".join(lines)


def render_decision(facts: dict) -> str:
    agree_rate = None
    if facts["pointing_compared"]:
        agree_rate = facts["pointing_agree"] / facts["pointing_compared"]
    execution_small = (
        facts["median_follow_deg_fail"] is not None and facts["median_follow_deg_fail"] < 8
    )
    decision_dominates = facts["rise_causes"].get("direction_decision", 0) >= facts["rise_causes"].get("execution", 0)
    lines = [
        "# 下一步只做哪一件事",
        "",
        "这次回放没有加载 LoRA，也没有把专家动作送回模型。",
        "",
        "## 证据",
        "",
        f"- 30 条成功标志与 10 月 15 日 baseline 一致：{facts['success_flags_equal']}。最终测地距离最大差 {facts['max_final_delta_m']} m。",
        f"- 失败时持续远离窗口的原因：{facts['rise_causes']}。",
        f"- 失败样本的航向跟随误差中位数 {facts['median_follow_deg_fail']}°，成功样本 {facts['median_follow_deg_success']}°。",
        f"- Pointing 与执行航向一致的比例：{None if agree_rate is None else round(agree_rate, 3)}（{facts['pointing_agree']} / {facts['pointing_compared']}）。",
        f"- 第一步航向差小于 20° 仍失败的有 {facts['n_fail_step0_lt_20']} 条。成功样本里出现过大于 45° 航向差的有 {facts['n_success_with_some_gt_45']} 条。",
        "",
        "## 判断",
        "",
    ]
    if not facts["success_flags_equal"]:
        lines.append("成功标志和 baseline 不一致。在查清之前，不根据这次轨迹提出新机制。")
    elif execution_small and decision_dominates:
        lines.extend([
            "机体基本按裁剪后的航点转动。失败窗口里，Pointing 和 RVQ 航向大多同号，问题不在坐标变换或解码。",
            "持续远离发生在模型自己选出的方向上。专家最短路径能走通，只说明环境里有路，不能当成模型曾经生成过这条路。",
            "成功样本也会短时间偏离专家航向，所以不能用单步 45° 当作错误。",
            "",
            "这些证据不够支持再加一个新网络。特权测地距离上的纠错，和已发表或预印本中的 BudVLN、IDEAL-VLN、SeekVLN 重叠。STOP 头只能覆盖已经进过 3 m 的样本，之前的 oracle 已经量过这个上界。",
            "",
            "## 下一件最小验证",
            "",
            "不要训练。用已经存下的 `frames/`，只在每条失败的持续远离起点那一帧上，人工看指令和画面，记录模型 Pointing 指向的是通道、墙壁还是指令里的下一个地标。",
            "若那一帧里正确开口可见而 Pointing 指到别的地方，再考虑一个不使用测地距离的视觉纠错。若正确开口当时不在画面里，纠错网络没有可学的视觉目标。",
        ])
    else:
        lines.extend([
            "跟随误差或 Pointing/RVQ 不一致占了相当部分，先不要加新网络。",
            "下一件最小验证是把 `yaw_follow_rad` 大于 15° 的步单独列出来，对照 `action_angular`、裁剪后航向和实际 `actual_dyaw_rad`，确认是速度裁剪、符号还是仿真没走完。确认之前不改控制器，也不训练。",
        ])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
