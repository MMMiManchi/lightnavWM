#!/usr/bin/env python3
"""Cause table and representative world-frame trajectories from the screening CSVs."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output/r2r_probe/error_screening_full_episode_20261021"
TRACE = ROOT / "output/r2r_probe/trace_20261020/steps.jsonl"
TEMPORAL = ROOT / "output/r2r_probe/visual_audit_temporal_20261020/temporal_visual_audit.csv"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def load(name: str) -> list[dict]:
    with (OUT / name).open() as handle:
        return list(csv.DictReader(handle))


def write(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def visual_notes() -> dict[str, dict]:
    groups = defaultdict(list)
    if not TEMPORAL.exists():
        return {}
    with TEMPORAL.open() as handle:
        for row in csv.DictReader(handle):
            groups[row["episode_id"]].append(row["与专家方向的关系"])
    out = {}
    for key, labels in groups.items():
        n = len(labels)
        out[key] = {
            "n": n,
            "match": sum(1 for item in labels if item == "一致"),
            "mismatch": sum(1 for item in labels if item == "不一致"),
            "no_cue": sum(1 for item in labels if item == "无对照线索"),
            "unclear": sum(1 for item in labels if item == "无法判断"),
        }
    return out


def add(rows, episode, cause, applies, evidence, counter, span, confidence, missing):
    rows.append(
        {
            "scene": episode["scene"],
            "episode_id": episode["episode_id"],
            "failure_type": episode["failure_type"],
            "success": episode["success"],
            "cause": cause,
            "applies": applies,
            "evidence": evidence,
            "counterevidence": counter,
            "time_span": span,
            "confidence": confidence,
            "missing": missing,
        }
    )


def main() -> None:
    episodes = load("episode_level_screening.csv")
    timeline = load("episode_timeline.csv")
    by = defaultdict(list)
    for row in timeline:
        by[(row["scene"], row["episode_id"])].append(row)
    visuals = visual_notes()
    success_postpone = [
        int(row["far_turn_immediate_straight_steps"]) for row in episodes if row["success"] == "1"
    ]
    causes = []
    for episode in episodes:
        key = (episode["scene"], episode["episode_id"])
        ident = f"{episode['scene']}/{episode['episode_id']}"
        rows = by[key]
        rise = [row for row in rows if "geo_rise" in row["events"]]
        behind = [row for row in rise if "forward_while_path_behind" in row["events"]]
        stop_rise = [row for row in rise if row["expert_stop"] == "1"]
        span_behind = ""
        if behind:
            span_behind = f"{behind[0]['step']}-{behind[-1]['step']}"
        span_all = f"0-{rows[-1]['step']}"
        failed = episode["success"] == "0"
        oracle = episode["stop_oracle_success"] == "1"
        min_geo = float(episode["min_geodesic_m"])
        n_rise = int(episode["geo_rise_steps"])
        n_behind = int(episode["forward_while_path_behind"])

        add(
            causes, episode, "环境或配置导致路径不可行", "no",
            "逐步测地距离有限，Go2 重算网格上 3259/3259 步都能查到到终点的路径。专家回放 30/30 成功。",
            "",
            span_all, "高",
            "90 条无限测地距离不在这 30 条里，见 infinite_geodesic_episodes.csv。",
        )

        vis = visuals.get(ident)
        if vis is None:
            add(
                causes, episode, "视觉信息不足、遮挡或视角受限", "unknown",
                "本轮没有逐帧审查这条轨迹的图像。",
                "",
                span_all, "低",
                "连续 RGB 已做成回放，但没有形成逐帧视觉结论。",
            )
        else:
            weak = vis["no_cue"] + vis["unclear"]
            applies = "yes" if weak >= vis["match"] and weak > 0 else "no"
            add(
                causes, episode, "视觉信息不足、遮挡或视角受限", applies,
                f"先前直行段审查 {vis['n']} 帧：一致 {vis['match']}，不一致 {vis['mismatch']}，无对照 {vis['no_cue']}，无法判断 {vis['unclear']}。",
                "审查范围是直行段，不是整条 episode。",
                episode["straight_runs"] or span_all, "中",
                "直行段之外的帧本轮没有重新逐帧看。",
            )

        add(
            causes, episode, "指令地标或空间关系未被利用", "unknown",
            "没有做语言反事实，也没有把整段指令和连续画面逐句对齐。",
            "",
            span_all, "低",
            "需要连续画面和指令子句的对照，本轮没有完成。",
        )
        add(
            causes, episode, "目标区域定位", "no",
            f"opos 为像素点的步数是 {episode['opos_point_steps']}。R2R 的目标是区域，不是必须可见的物体。",
            "opos 长期为 0 不能当成没找到目标。",
            span_all, "中",
            "",
        )

        if failed and n_behind >= 5 and n_rise > 0 and n_behind >= 0.5 * n_rise:
            add(
                causes, episode, "远期轨迹选错方向", "yes",
                f"测地距离上升的 {n_rise} 步里，有 {n_behind} 步局部最短路径第一个 0.25 m 样本在机体后方（未截断航向绝对值大于 90°，前向不超过 0.05 m），模型仍以前向大于 0.1 m、航向绝对值小于 15° 前进。",
                "日志里的 ±30° 只是截断后的即时转向，这里用的是重算路径的未截断航向。",
                span_behind, "高",
                "机体扫过体积仍未知。",
            )
        else:
            add(
                causes, episode, "远期轨迹选错方向", "no",
                f"路径在身后而模型直行的上升步是 {n_behind}/{n_rise}。",
                "成功样本里也有短时出现，例如 98 有 12 步，但随后距离重新下降。",
                span_behind or span_all, "高" if n_rise == 0 or n_behind == 0 else "中",
                "",
            )

        near = int(episode["near_obstacle_steps"])
        add(
            causes, episode, "远期轨迹与可行通道几何冲突", "no",
            f"{near}/{episode['steps']} 步里，10 个预测点至少有一个中心到障碍边界的距离小于机体半径 0.215 m。6 条成功样本同样大量出现。",
            "它分不开成功和失败。previous_step_collided 全为 false，allow_sliding 为 true。这不是扫掠碰撞结论。",
            span_all, "高",
            "没有沿机体轮廓做扫掠。距离为 0 的点只说明查询结果落在边界上或内部，不能单独当成已经撞上。",
        )

        add(
            causes, episode, "转向时机是否不合理尚不确定", "no",
            f"路径在侧方（未截断航向 30° 到 90°）而模型直行的上升步只有 {episode['forward_while_path_side']}。上升主要发生在路径已经位于身后的步。",
            "不能把截断后的 30° 理解成前方不远处的一个小弯。",
            span_all, "高",
            "",
        )

        postpone = int(episode["far_turn_immediate_straight_steps"])
        add(
            causes, episode, "滚动预测中转向推迟或方向反复改变", "yes" if postpone >= 5 else "no",
            f"第 9 航点绝对值达到 15° 而第 0 航点小于 5° 的步数是 {postpone}，最长连续 {episode['longest_postpone']}。成功样本里这个数字是 {success_postpone}。",
            "成功轨迹里同样出现，所以它单独不能区分失败。",
            episode["postpone_runs"] or span_all, "中",
            "",
        )

        lat = float(episode["max_abs_executed_lateral_m"] or 0)
        add(
            causes, episode, "动作映射或横向分量丢失", "no",
            f"控制器丢掉横向。这条轨迹执行航点横向绝对值最大 {lat:.3f} m。航向跟随误差最大 0.00057°。",
            "厘米级横向丢失解释不了数米的测地距离增加。",
            span_all, "高",
            "",
        )

        if (not failed) and oracle:
            stop_apply, stop_conf = "no", "高"
            stop_ev = f"最近距离 {min_geo:.2f} m，最终仍小于 3 m。专家停止之后模型有时还在走，但停在了成功半径内。"
        elif failed and oracle and min_geo < 3.0:
            stop_apply, stop_conf = "yes", "高"
            stop_ev = f"最近距离 {min_geo:.2f} m，进入过 3 m。特权停止能把这条救回。上升步里专家已经要求停止的有 {len(stop_rise)} 步。"
        elif failed and n_rise <= 3 and min_geo >= 3.0 and rows[-1]["model_stop"] == "1":
            stop_apply, stop_conf = "yes", "中"
            stop_ev = f"测地距离大体下降，最近 {min_geo:.2f} m，在 3 m 外停下。单步上升超过 0.05 m 的只有 {n_rise} 步。"
        else:
            stop_apply, stop_conf = "no", "高"
            stop_ev = f"最近距离 {min_geo:.2f} m，特权停止不能把失败变成成功。" if failed else "成功。"
        add(
            causes, episode, "STOP与到达判断", stop_apply, stop_ev,
            "专家停止阈值是 1 m，官方成功是停下且测地距离小于 3 m。",
            f"{rows[-1]['step']}" if rows[-1]["model_stop"] == "1" else span_all,
            stop_conf, "",
        )

        add(
            causes, episode, "专家比较口径不一致", "yes",
            "非停止步上，重算的即时动作和日志一致。其中大量日志值是截断到 ±30° 的原地转向，未截断航向没有写在原日志里。",
            "本轮几何比较用的是重算折线，不是拿 ±30° 去比第 9 航点。",
            span_all, "高", "",
        )

        add(
            causes, episode, "数据不足无法判定", "yes" if vis is None else "no",
            "历史 RGB 没有保存。碰撞扫掠没有做。",
            "当前帧、位姿、10 个航点和重算路径是齐的。" if vis else "直行段有过视觉审查，整段仍没有逐帧结论。",
            span_all, "高" if vis is None else "中",
            "未逐帧审查的图像不能补视觉结论。",
        )

    write("cause_attribution.csv", causes)
    counts = defaultdict(int)
    for row in causes:
        if row["applies"] == "yes" and row["success"] == "0":
            counts[row["cause"]] += 1
    print("failure episodes with applies=yes")
    for key, value in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"{value:3} {key}")
    plot_representative()


def world_point(origin, yaw, forward_m, lateral_m):
    forward = np.array([-math.sin(yaw), -math.cos(yaw)])
    left = np.array([-math.cos(yaw), math.sin(yaw)])
    xz = np.array([origin[0], origin[2]])
    return xz + forward_m * forward + lateral_m * left


def plot_representative() -> None:
    import json

    wanted = [
        ("2azQ1b91cZZ", "10", "success"),
        ("2azQ1b91cZZ", "1049", "path behind from step 0"),
        ("X7HyMhZNoso", "1759", "path behind, far yaw right"),
        ("QUCTc6BB5sX", "455", "path behind, far yaw left"),
        ("2azQ1b91cZZ", "326", "long drift"),
        ("x8F5xyUWy9e", "585", "wrong way"),
        ("2azQ1b91cZZ", "214", "left after arrival"),
        ("EU6Fwq7SyZv", "972", "stopped outside 3 m"),
    ]
    steps = defaultdict(list)
    for line in TRACE.open():
        row = json.loads(line)
        key = (row["scene"], str(row["episode_id"]))
        if key in {(a, b) for a, b, _ in wanted}:
            steps[key].append(row)
    paths = {}
    with (OUT / "expert_path_reconstruction.csv").open() as handle:
        for row in csv.DictReader(handle):
            paths[(row["scene"], row["episode_id"], row["step"])] = row
    rises = {}
    with (OUT / "episode_level_screening.csv").open() as handle:
        for row in csv.DictReader(handle):
            rises[(row["scene"], row["episode_id"])] = row["first_sustained_rise_step"]

    font = ImageFont.truetype(FONT, 13)
    small = ImageFont.truetype(FONT, 11)
    cell_w, cell_h = 360, 300
    sheet = Image.new("RGB", (4 * cell_w, 2 * cell_h), (255, 255, 255))
    for index, (scene, eid, tag) in enumerate(wanted):
        rows = sorted(steps[(scene, eid)], key=lambda row: int(row["step"]))
        pts = [(float(row["pose_xyz_before"][0]), float(row["pose_xyz_before"][2])) for row in rows]
        last = rows[-1]
        pts.append((float(last["pose_xyz_after"][0]), float(last["pose_xyz_after"][2])))
        rise_step = rises[(scene, eid)]
        overlay_step = rise_step if rise_step != "" else str(len(rows) // 2)
        rec = paths.get((scene, eid, overlay_step))
        expert = []
        if rec and rec["path_world_xyz"]:
            for item in rec["path_world_xyz"].split(";"):
                x, _y, z = item.split(",")
                expert.append((float(x), float(z)))
        chunk = []
        src = next(row for row in rows if str(row["step"]) == overlay_step)
        origin = np.asarray(src["pose_xyz_before"], dtype=np.float64)
        yaw = float(src["pose_yaw_before"])
        for point in src["waypoints_robot"]:
            xz = world_point(origin, yaw, float(point[0]), float(point[1]))
            chunk.append((float(xz[0]), float(xz[1])))
        all_pts = pts + expert + chunk
        xs = [p[0] for p in all_pts]
        zs = [p[1] for p in all_pts]
        pad = 0.4
        minx, maxx = min(xs) - pad, max(xs) + pad
        minz, maxz = min(zs) - pad, max(zs) + pad
        span = max(maxx - minx, maxz - minz, 1.0)
        cx, cy = (minx + maxx) / 2, (minz + maxz) / 2
        minx, maxx = cx - span / 2, cx + span / 2
        minz, maxz = cy - span / 2, cy + span / 2

        def pix(x, z):
            # Same overhead flip as the replay map: camera-right is map-right.
            u = 28 + (maxx - x) / (maxx - minx) * (cell_w - 40)
            v = 36 + (maxz - z) / (maxz - minz) * (cell_h - 70)
            return u, v

        crop = Image.new("RGB", (cell_w, cell_h), (248, 248, 248))
        draw = ImageDraw.Draw(crop)
        draw.rectangle([0, 0, cell_w, 24], fill=(32, 32, 32))
        draw.text((6, 4), f"{scene}/{eid}  {tag}", fill=(255, 255, 255), font=small)
        if len(expert) >= 2:
            draw.line([pix(*p) for p in expert], fill=(30, 140, 70), width=2)
        draw.line([pix(*p) for p in pts], fill=(30, 70, 160), width=2)
        if len(chunk) >= 2:
            draw.line([pix(*p) for p in chunk], fill=(210, 90, 20), width=2)
        draw.ellipse([*pix(*pts[0]), *pix(*pts[0])], outline=(0, 0, 0))
        s = pix(*pts[0])
        draw.ellipse([s[0] - 4, s[1] - 4, s[0] + 4, s[1] + 4], fill=(20, 140, 60))
        e = pix(*pts[-1])
        draw.ellipse([e[0] - 4, e[1] - 4, e[0] + 4, e[1] + 4], fill=(180, 30, 30))
        draw.text((8, cell_h - 28), "blue actual  green expert path at rise  orange RVQ", fill=(60, 60, 60), font=small)
        draw.text((8, cell_h - 14), f"overlay step {overlay_step}", fill=(60, 60, 60), font=small)
        sheet.paste(crop, ((index % 4) * cell_w, (index // 4) * cell_h))
    sheet.save(OUT / "representative_trajectories.png")
    print("figure", sheet.size)


if __name__ == "__main__":
    main()
