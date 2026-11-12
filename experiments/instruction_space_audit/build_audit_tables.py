#!/usr/bin/env python3
"""Offline instruction-space audit from existing logs. Read-only on prior results."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/szgx/LightNav-0")
OUT = ROOT / "experiments/instruction_space_audit"
DATASET = ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
RESULTS = ROOT / "output/r2r_go2w/results.jsonl"
SCREEN = (
    ROOT
    / "output/r2r_probe/error_screening_full_episode_20261021/episode_level_screening.csv"
)
INFINITE = (
    ROOT
    / "output/r2r_probe/error_screening_full_episode_20261021/infinite_geodesic_episodes.csv"
)


def scene_stem(scene_id: str) -> str:
    return scene_id.split("/")[-1].replace(".glb", "")


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def yaw_of(vec: np.ndarray) -> float:
    return math.atan2(-float(vec[0]), -float(vec[2]))


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    qv = np.array([x, y, z], dtype=float)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def reference_headings(episode: dict) -> tuple[float | None, tuple | None]:
    """First reference-path segment and first later turn, in the start body frame.

    Positive is left. This is a diagnostic label, not a model input.
    """
    q = np.array(episode["start_rotation"], dtype=float)
    forward = quat_rotate(q, np.array([0.0, 0.0, -1.0]))
    forward[1] = 0.0
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        return None, None
    yaw0 = math.degrees(yaw_of(forward / norm))
    pts = [np.array(p, dtype=float) for p in episode["reference_path"]]
    prev = None
    walked = 0.0
    first_rel = None
    first_big = None
    for i in range(len(pts) - 1):
        delta = pts[i + 1] - pts[i]
        delta[1] = 0.0
        dist = float(np.linalg.norm(delta))
        if dist < 0.2:
            continue
        heading = math.degrees(yaw_of(delta / dist))
        rel = wrap(heading - yaw0)
        turn = 0.0 if prev is None else wrap(heading - prev)
        if first_rel is None:
            first_rel = rel
        walked += dist
        if first_big is None and abs(turn) >= 30.0:
            first_big = (walked, turn, rel)
        prev = heading
    return first_rel, first_big


def load_dataset() -> list[dict]:
    with gzip.open(DATASET, "rt", encoding="utf-8") as handle:
        return json.load(handle)["episodes"]


def load_results() -> dict[tuple[str, str], dict]:
    rows = {}
    with RESULTS.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[(scene_stem(row["scene_id"]), str(row["habitat_episode_id"]))] = row
    return rows


def success_flag(row: dict) -> int:
    return int(float(row["success"]) >= 0.5)


def load_infinite() -> set[tuple[str, str]]:
    with INFINITE.open() as handle:
        return {(row["scene"], str(row["habitat_episode_id"])) for row in csv.DictReader(handle)}


def load_screen() -> dict[tuple[str, str], dict]:
    with SCREEN.open() as handle:
        return {(row["scene"], str(row["episode_id"])): row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    episodes = load_dataset()
    results = load_results()
    screen = load_screen()
    no_path = load_infinite()
    by_episode = {}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for episode in episodes:
        scene = scene_stem(episode["scene_id"])
        by_episode[(scene, str(episode["episode_id"]))] = episode
        groups[(scene, str(episode["trajectory_id"]))].append(episode)

    triplet_rows = []
    failed_rows = []
    all_fail_rows = []
    outcome = Counter()
    n_success_hist = Counter()
    fail_episodes = 0
    fail_with_sibling = 0
    missing_result = 0
    for (scene, trajectory_id), members in sorted(groups.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        members = sorted(members, key=lambda episode: int(episode["episode_id"]))
        if len(members) != 3:
            raise SystemExit(f"unexpected group size {len(members)} for {scene} {trajectory_id}")
        packed = []
        successes = 0
        for episode in members:
            key = (scene, str(episode["episode_id"]))
            result = results.get(key)
            if result is None:
                missing_result += 1
                continue
            ok = success_flag(result)
            successes += ok
            packed.append((episode, result, ok))
        n_success_hist[successes] += 1
        if successes == 3:
            group_outcome = "all_success"
        elif successes == 0:
            group_outcome = "all_fail"
        else:
            group_outcome = "partial"
        outcome[group_outcome] += 1
        row = {
            "route_group_id": f"{scene}__{trajectory_id}",
            "scene": scene,
            "trajectory_id": trajectory_id,
            "n_success": successes,
            "group_outcome": group_outcome,
            "go2_path": "no_path" if all((scene, str(episode["episode_id"])) in no_path for episode in members) else "finite",
        }
        texts = []
        for index, (episode, result, ok) in enumerate(packed, start=1):
            text = " ".join(episode["instruction"]["instruction_text"].split())
            texts.append(text)
            row[f"episode_id_{index}"] = str(episode["episode_id"])
            row[f"success_{index}"] = ok
            row[f"final_distance_m_{index}"] = f"{float(result['final_distance']):.4f}"
            row[f"min_distance_m_{index}"] = f"{float(result['min_distance']):.4f}"
            row[f"steps_{index}"] = result["steps"]
            row[f"instruction_{index}"] = text
            if not ok:
                fail_episodes += 1
                sibling_ok = [item for item in packed if item[2]]
                if sibling_ok:
                    fail_with_sibling += 1
                failed_rows.append(
                    {
                        "episode_id": str(episode["episode_id"]),
                        "scene": scene,
                        "route_group_id": row["route_group_id"],
                        "trajectory_id": trajectory_id,
                        "instruction": text,
                        "final_distance_m": f"{float(result['final_distance']):.4f}",
                        "min_distance_m": f"{float(result['min_distance']):.4f}",
                        "steps": result["steps"],
                        "has_success_sibling": int(bool(sibling_ok)),
                        "sibling_success_episode_ids": " ".join(str(item[0]["episode_id"]) for item in sibling_ok),
                        "sibling_success_final_m": " ".join(f"{float(item[1]['final_distance']):.4f}" for item in sibling_ok),
                        "sibling_success_min_m": " ".join(f"{float(item[1]['min_distance']):.4f}" for item in sibling_ok),
                    }
                )
        triplet_rows.append(row)
        if group_outcome == "all_fail":
            all_fail_rows.append(row)

    sibling_only = [row for row in failed_rows if row["has_success_sibling"] == 1]
    stats = {
        "episodes": len(episodes),
        "routes": len(groups),
        "group_size_ok": all(len(v) == 3 for v in groups.values()),
        "matched_results": sum(1 for key in by_episode if key in results),
        "missing_result": missing_result,
        "success_episodes": sum(success_flag(row) for row in results.values()),
        "fail_episodes": fail_episodes,
        "all_success_routes": outcome["all_success"],
        "partial_routes": outcome["partial"],
        "all_fail_routes": outcome["all_fail"],
        "routes_with_1_success": n_success_hist[1],
        "routes_with_2_success": n_success_hist[2],
        "fail_with_success_sibling": fail_with_sibling,
        "fail_without_success_sibling": fail_episodes - fail_with_sibling,
        "no_path_episodes": len(no_path),
        "all_fail_routes_no_path": sum(1 for row in all_fail_rows if row["go2_path"] == "no_path"),
        "all_fail_routes_finite": sum(1 for row in all_fail_rows if row["go2_path"] == "finite"),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    triplet_fields = ["route_group_id", "scene", "trajectory_id", "n_success", "group_outcome", "go2_path"]
    for index in (1, 2, 3):
        triplet_fields += [
            f"episode_id_{index}",
            f"success_{index}",
            f"final_distance_m_{index}",
            f"min_distance_m_{index}",
            f"steps_{index}",
            f"instruction_{index}",
        ]
    write_csv(OUT / "route_triplets.csv", triplet_rows, triplet_fields)
    write_csv(
        OUT / "failed_with_success_sibling.csv",
        sibling_only,
        [
            "episode_id",
            "scene",
            "route_group_id",
            "trajectory_id",
            "instruction",
            "final_distance_m",
            "min_distance_m",
            "steps",
            "has_success_sibling",
            "sibling_success_episode_ids",
            "sibling_success_final_m",
            "sibling_success_min_m",
        ],
    )
    write_csv(OUT / "all_three_failed.csv", all_fail_rows, triplet_fields)
    write_reviews(by_episode, results, groups, screen, stats)
    print(json.dumps(stats, indent=2))


def write_reviews(by_episode, results, groups, screen, stats) -> None:
    # scene, episode. Hand labels only where a frame or a path/text check was actually made.
    selected = [
        ("2azQ1b91cZZ", "1266"),
        ("EU6Fwq7SyZv", "639"),
        ("oLBMNvg9in8", "846"),
        ("EU6Fwq7SyZv", "972"),
        ("x8F5xyUWy9e", "199"),
        ("x8F5xyUWy9e", "201"),
        ("x8F5xyUWy9e", "585"),
        ("8194nk5LbLH", "221"),
        ("2azQ1b91cZZ", "326"),
        ("2azQ1b91cZZ", "214"),
        ("QUCTc6BB5sX", "142"),
        ("2azQ1b91cZZ", "116"),
        ("2azQ1b91cZZ", "115"),
        ("2azQ1b91cZZ", "117"),
        ("QUCTc6BB5sX", "143"),
        ("QUCTc6BB5sX", "144"),
        ("TbHJrupSAjP", "172"),
        ("TbHJrupSAjP", "173"),
        ("TbHJrupSAjP", "174"),
        ("2azQ1b91cZZ", "215"),
        ("2azQ1b91cZZ", "216"),
        ("QUCTc6BB5sX", "875"),
        ("QUCTc6BB5sX", "455"),
        ("X7HyMhZNoso", "1759"),
        ("2azQ1b91cZZ", "1049"),
        ("2azQ1b91cZZ", "1087"),
        ("TbHJrupSAjP", "376"),
        ("x8F5xyUWy9e", "189"),
        ("X7HyMhZNoso", "901"),
        ("QUCTc6BB5sX", "569"),
        ("Z6MFQCViBuw", "393"),
        ("TbHJrupSAjP", "1332"),
        ("2azQ1b91cZZ", "10"),
        ("QUCTc6BB5sX", "98"),
        ("X7HyMhZNoso", "1090"),
        ("oLBMNvg9in8", "1072"),
        ("zsNo4HB9uLZ", "1040"),
        ("EU6Fwq7SyZv", "198"),
    ]
    discussed = {"1266", "639", "846", "972", "199", "585", "221", "326", "214", "142", "116"}
    review_rows = []
    evidence_rows = []
    for scene, episode_id in selected:
        episode = by_episode[(scene, episode_id)]
        result = results[(scene, episode_id)]
        route = f"{scene}__{episode['trajectory_id']}"
        members = groups[(scene, str(episode["trajectory_id"]))]
        n_success = 0
        for member in members:
            n_success += success_flag(results[(scene, str(member["episode_id"]))])
        ok = success_flag(result)
        text = " ".join(episode["instruction"]["instruction_text"].split())
        first_rel, first_big = reference_headings(episode)
        screen_row = screen.get((scene, episode_id))
        labels = hand_label(scene, episode_id, text, first_rel, first_big, result, screen_row, n_success)
        strata = []
        if episode_id in discussed:
            strata.append("已讨论案例")
        if n_success == 0:
            strata.append("三句都失败")
        elif ok == 0:
            strata.append("同路线有成功指令")
        else:
            strata.append("成功对照")
        if labels["frames_reviewed"] == "no":
            strata.append("未重看画面")
        review_rows.append(
            {
                "episode_id": episode_id,
                "scene": scene,
                "route_group_id": route,
                "sample_stratum": "|".join(strata),
                "instruction": text,
                "baseline_success": ok,
                "final_distance_m": f"{float(result['final_distance']):.4f}",
                "min_distance_m": f"{float(result['min_distance']):.4f}",
                "instruction_path_consistent": labels["instruction_path_consistent"],
                "direction_relation_consistent": labels["direction_relation_consistent"],
                "landmark_order_consistent": labels["landmark_order_consistent"],
                "target_grounding_correct": labels["target_grounding_correct"],
                "goal_visible_at_decision": labels["goal_visible_at_decision"],
                "failure_source": labels["failure_source"],
                "evidence": labels["evidence"],
                "reviewer_confidence": labels["reviewer_confidence"],
            }
        )
        evidence_rows.append(
            {
                "episode_id": episode_id,
                "scene": scene,
                "route_group_id": route,
                "A_current_vision": labels["A"],
                "A_changes_ranking": labels["A_changes"],
                "B_onboard_geometry": "尚无法验证",
                "B_privileged_path": labels["B"],
                "B_privileged_changes_ranking": labels["B_changes"],
                "C_other_route_memory": "尚无法验证",
                "C_same_route_history_upper_bound": "未执行，不与跨路线记忆混报",
                "D_language_trajectory": labels["D"],
                "D_changes_ranking": labels["D_changes"],
                "evidence_note": labels["abcd_note"],
            }
        )
    fields = [
        "episode_id",
        "scene",
        "route_group_id",
        "sample_stratum",
        "instruction",
        "baseline_success",
        "final_distance_m",
        "min_distance_m",
        "instruction_path_consistent",
        "direction_relation_consistent",
        "landmark_order_consistent",
        "target_grounding_correct",
        "goal_visible_at_decision",
        "failure_source",
        "evidence",
        "reviewer_confidence",
    ]
    write_csv(OUT / "manual_review.csv", review_rows, fields)
    write_csv(
        OUT / "evidence_abcd.csv",
        evidence_rows,
        [
            "episode_id",
            "scene",
            "route_group_id",
            "A_current_vision",
            "A_changes_ranking",
            "B_onboard_geometry",
            "B_privileged_path",
            "B_privileged_changes_ranking",
            "C_other_route_memory",
            "C_same_route_history_upper_bound",
            "D_language_trajectory",
            "D_changes_ranking",
            "evidence_note",
        ],
    )
    (OUT / "review_tally.json").write_text(
        json.dumps(tally(review_rows, evidence_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fmt_heading(first_rel, first_big) -> str:
    rel = "无" if first_rel is None else f"{first_rel:+.0f}度"
    if not first_big:
        big = "参考路径上没有超过30度的后续转折"
    else:
        big = f"走到约{first_big[0]:.1f}米时相对上一段转{first_big[1]:+.0f}度"
    return f"参考路径首段相对起步朝向{rel}（左为正），{big}。该角度只作诊断，不是模型输入。"


def screen_bits(screen_row: dict | None) -> str:
    if not screen_row:
        return "这条没有逐步轨迹。"
    return (
        f"逐步日志{screen_row['steps']}步，最近{screen_row['min_geodesic_m']}米，"
        f"结束{screen_row['final_geodesic_m']}米，测地距离上升{screen_row['geo_rise_steps']}步，"
        f"其中路径在身后仍直行{screen_row['forward_while_path_behind']}步，"
        f"低位移{screen_row['stuck_steps']}步，忽略专家停止{screen_row['expert_stop_ignored_steps']}步。"
    )


def unknown_visual(failure_source: str, evidence: str, confidence: str, **abcd) -> dict:
    base = {
        "frames_reviewed": "no",
        "instruction_path_consistent": "unknown",
        "direction_relation_consistent": "unknown",
        "landmark_order_consistent": "unknown",
        "target_grounding_correct": "unknown",
        "goal_visible_at_decision": "unknown",
        "failure_source": failure_source,
        "evidence": evidence,
        "reviewer_confidence": confidence,
        "A": "尚无法验证",
        "A_changes": "unknown",
        "B": "尚无法验证",
        "B_changes": "unknown",
        "D": "尚无法验证",
        "D_changes": "unknown",
        "abcd_note": "本轮没有打开该回合的画面，不用汇总数字代替视觉结论。",
    }
    base.update(abcd)
    return base


def hand_label(scene, episode_id, text, first_rel, first_big, result, screen_row, n_success) -> dict:
    heading = fmt_heading(first_rel, first_big)
    bits = screen_bits(screen_row)
    ok = success_flag(result)

    def done(**kwargs):
        row = unknown_visual("证据不足", "", "低")
        row.update(kwargs)
        return row

    if episode_id == "1266":
        return done(
            frames_reviewed="yes",
            instruction_path_consistent="unknown",
            direction_relation_consistent="unknown",
            landmark_order_consistent="no",
            target_grounding_correct="no",
            goal_visible_at_decision="no",
            failure_source="目标定位问题",
            reviewer_confidence="高",
            evidence=(
                "第77步测地距离3.81米，模型自己停止。apos=stop，opos在480x270画面底边[215,265]，对应相机前约0.5米的地面。"
                "这一帧是厨房和前方过道，画面里没有沙发。指令要求走到沙发再停。同路线1265成功，终点0.71米；不能因此把1266的指令判成写错。"
                + heading
            ),
            A="当前画面和原始输出不支持“已经到沙发”。opos落在近处地面，apos给出停止。",
            A_changes="yes",
            B="诊断用路径在停止步大约仍朝前（约-3度）。它会否定这次停止，但这是评测坐标，不是机器人当时拥有的地图。",
            B_changes="yes",
            D="语言和已查看画面冲突的是停止目标，不是一个已经核对过的转向词。没有证据要求改写这句指令。",
            D_changes="no",
            abcd_note="A来自trace_20261020第77步和该步原图。B是特权测地路径，单独列出。",
        )
    if episode_id == "639":
        return done(
            frames_reviewed="yes",
            instruction_path_consistent="ambiguous",
            direction_relation_consistent="ambiguous",
            landmark_order_consistent="ambiguous",
            target_grounding_correct="no",
            goal_visible_at_decision="yes",
            failure_source="路线或方向决策问题",
            reviewer_confidence="中",
            evidence=(
                "指令是下楼、右转、走向台球桌并在壁炉等待。第0步相机对着上方的楼梯结构。第0到9步是左转约174度，apos=rot_left，不是右转。"
                "第9步下方能看到台球桌，所以这个地标当时可见。第10步之后测地距离从11.61降到第45步的6.23，这段左转并没有一开始就走远。"
                "第45步画面是一间沙发厅，看不到台球桌或壁炉，opos仍是not_visible，之后距离又升到11.47。三句指令都失败，不能归到某一句写错。"
                + bits
            ),
            A="第9步的画面支持台球桌可见；第45步的画面和opos不支持已经到达壁炉。",
            A_changes="yes",
            B="特权路径在起步时在身后（约+179度），开头的左转把路径转进画面，不能用“禁止大转弯”否定它。后面距离上升的那些步，路径在身后时模型仍直行，特权航向会否定那些直行。",
            B_changes="yes",
            D="已查看的画面里台球桌存在。没有证据把这句改成另一条路线。",
            D_changes="no",
            abcd_note="没有把用户描述的“右转180度”写成日志结论；日志是左转174度。",
        )
    if episode_id == "846":
        return done(
            frames_reviewed="yes",
            instruction_path_consistent="no",
            direction_relation_consistent="no",
            landmark_order_consistent="unknown",
            target_grounding_correct="ambiguous",
            goal_visible_at_decision="yes",
            failure_source="疑似指令问题",
            reviewer_confidence="中",
            evidence=(
                "起步原图里壁炉在画面右侧，指令却写 with the fireplace on your left，并要求从左边的门离开。这和起步画面不一致。"
                "第10到17步在原地左转。最后第111步是模型自己停止，测地3.06米，没有到500步上限；该帧能看到楼梯，opos在画面底边。"
                "同路线短指令845成功，终点2.53米。结尾差6厘米仍是另一次停止问题，单改开头不能解释它。"
                + heading + bits
            ),
            A="起步画面不支持“壁炉在左侧”。结束画面支持已经靠近楼梯，但不支持把3.06米写成已经成功。",
            A_changes="yes",
            B="特权路径起步相对朝向约-111度，会否定“向左出门”作为第一条测地方向。它不能在3.06米时给出成功停止。",
            B_changes="yes",
            D="开头的左右关系和起步画面冲突，会改变对这一段的归类。不能外推成整句路线都不存在。",
            D_changes="yes",
            abcd_note="指令冲突只覆盖开头。结尾停止分开记录。",
        )
    if episode_id == "972":
        return done(
            frames_reviewed="yes",
            instruction_path_consistent="unknown",
            direction_relation_consistent="ambiguous",
            landmark_order_consistent="unknown",
            target_grounding_correct="unknown",
            goal_visible_at_decision="unknown",
            failure_source="停止决策问题",
            reviewer_confidence="中",
            evidence=(
                "第37步测地3.79米，apos=stop，opos=[235,205]。这一帧主要是木板墙和一个暗色开口，不能确认酒架或楼梯下方。"
                "指令没有明确转向词。同路线970、971成功。画面不足以把失败改判成指令写错。"
                + bits
            ),
            A="当前画面不能确认指令里的酒架，原始输出却停止了。",
            A_changes="yes",
            B="特权测地仍有3.79米，会否定这次成功停止。没有逐步局部地图。",
            B_changes="yes",
            D="没有足够的路标对照，不能声称发现了指令冲突。",
            D_changes="no",
            abcd_note="只看了停止帧，没有沿参考路径核对酒架是否存在。",
        )
    if episode_id == "199":
        return done(
            frames_reviewed="yes",
            instruction_path_consistent="ambiguous",
            direction_relation_consistent="unknown",
            landmark_order_consistent="unknown",
            target_grounding_correct="ambiguous",
            goal_visible_at_decision="ambiguous",
            failure_source="停止决策问题",
            reviewer_confidence="中",
            evidence=(
                "指令要求在右侧植物前停下。第20步测地3.19米，画面里有两盆植物。第29步在砖墙角落停止，植物仍在画面右侧，测地3.56米，apos=stop，opos在底边。"
                "命名植物可见，但数据集终点仍在3米以外。同一条路径199、200、201三句都失败，没有成功兄弟可以用来证明指令写错。"
                + bits
            ),
            A="画面支持植物可见，不支持已经到达3米内的数据集终点。",
            A_changes="no",
            B="特权距离会否定这次成功停止，但不能从这两帧判断植物是不是终点标记。",
            B_changes="yes",
            D="语言和植物画面相符，和终点坐标不相符。证据不够把整句判成写错。",
            D_changes="no",
            abcd_note="目标可见性标成ambiguous，因为植物在画面里，坐标仍在3米外。",
        )
    if episode_id == "585":
        rel_ok = first_rel is not None and 60 <= first_rel <= 120 and first_big is None
        return done(
            frames_reviewed="yes",
            instruction_path_consistent="yes" if rel_ok else "ambiguous",
            direction_relation_consistent="yes" if rel_ok else "ambiguous",
            landmark_order_consistent="unknown",
            target_grounding_correct="unknown",
            goal_visible_at_decision="no",
            failure_source="路线或方向决策问题",
            reviewer_confidence="高" if rel_ok else "低",
            evidence=(
                "指令只要求左转并停在泳池尽头的珠帘门口。模型转到约+90度后仍继续左转，累计约+271度，然后走进左侧门口。"
                "第4步身体已经约+90度，当时路径只剩约+13度，第9个航点仍是+90度。"
                "不能把这句改成右转。同路线584成功，不能当成这句缺少转向。"
                + heading + bits
            ),
            A="先前查看的第4、9、13步显示左转越过了泳池通道。本轮没有重开这些原图，方向结论沿用那次画面和航向日志。",
            A_changes="yes",
            B="特权路径在转过约90度后不再要求继续左转，会否定后面的左转。",
            B_changes="yes",
            D="参考路径的首段左转和“turn left”同向，后面没有超过30度的转折。一致性检查不应改这句指令。",
            D_changes="no",
            abcd_note="画面结论来自先前对585的逐步查看；本轮重新计算了参考路径角度。",
        )
    if episode_id == "221":
        return done(
            frames_reviewed="yes",
            instruction_path_consistent="yes",
            direction_relation_consistent="yes",
            landmark_order_consistent="no",
            target_grounding_correct="unknown",
            goal_visible_at_decision="yes",
            failure_source="路线或方向决策问题",
            reviewer_confidence="高",
            evidence=(
                "指令是先走向门，再左转上楼。起步画面右侧是玻璃门，路径航向约-52.5度，投影在画面右侧。模型第0到9步先原地左转约179度，没有先过那扇门。"
                "门当时可见，不是没看到开口。同路线220成功。这不能改写成指令漏写了左转时机。"
                + heading
            ),
            A="起步画面支持门在右侧可见，原始左转输出和“先过门”冲突。",
            A_changes="yes",
            B="特权路径指向右侧的门，会否定起步的原地左转。",
            B_changes="yes",
            D="指令顺序和参考路径的“先到门、再左转”一致，不应改指令。",
            D_changes="no",
            abcd_note="门的位置来自先前查看的起步帧和投影，不是本轮新截图。",
        )
    if episode_id == "326":
        return done(
            frames_reviewed="yes",
            instruction_path_consistent="ambiguous",
            direction_relation_consistent="ambiguous",
            landmark_order_consistent="ambiguous",
            target_grounding_correct="unknown",
            goal_visible_at_decision="unknown",
            failure_source="路线或方向决策问题",
            reviewer_confidence="中",
            evidence=(
                "先前查看的第94到104步，沙发在左、餐椅在右、拱门在左前方，第101步10个航点都大约右转15度，当时路径航向约-160度。"
                "指令写穿过两张沙发中间，没有写这次右转。参考路径开头几乎是直行。三句都失败，327只说走向正前方的照片，也失败。"
                "还不能确定指令漏写了必要的右转。"
                + heading + bits
            ),
            A="已查看的餐厅画面不支持把那次右转当成指令的下一句。",
            A_changes="yes",
            B="特权路径在该步指向大幅偏转，会否定15度右转后继续当前朝向。",
            B_changes="yes",
            D="没有完整核对路标顺序，不能把三句全失败当成指令错误。",
            D_changes="no",
            abcd_note="没有把第23步的厨房画面当成餐厅。",
        )
    if episode_id == "214":
        return done(
            frames_reviewed="yes",
            instruction_path_consistent="no",
            direction_relation_consistent="ambiguous",
            landmark_order_consistent="no",
            target_grounding_correct="no",
            goal_visible_at_decision="ambiguous",
            failure_source="疑似指令问题",
            reviewer_confidence="高",
            evidence=(
                "指令末尾要求在壁炉卧室里、左侧的床边等待。最近距离0.14米的画面是衣柜、梳妆台和浴室门口，没有床，也没有壁炉。床出现在起步画面的右侧前景。"
                "同坐标的215写抽屉和架子，216写敞开的门口，两句都成功。214到达坐标后又离开，结束于6.76米，所以还有一次停止失败。床这个目标本身和画面不符。"
                + heading
            ),
            A="到达坐标时的画面不支持床或壁炉。",
            A_changes="yes",
            B="特权停止规则会在0.14米处停下，只能避免离开，不能让床出现。",
            B_changes="yes",
            D="语言里的最终目标与同一终点坐标上看到的房间冲突，会把这句从模型漏停改判为指令目标不可靠。",
            D_changes="yes",
            abcd_note="终点画面来自214最近步的原图。215和216没有单独截取停止帧。",
        )
    if episode_id == "142":
        path_right = first_rel is not None and first_rel < -45
        return done(
            frames_reviewed="no",
            instruction_path_consistent="no" if path_right else "ambiguous",
            direction_relation_consistent="no" if path_right else "ambiguous",
            landmark_order_consistent="unknown",
            target_grounding_correct="unknown",
            goal_visible_at_decision="unknown",
            failure_source="疑似指令问题",
            reviewer_confidence="高" if path_right else "低",
            evidence=(
                "142要求先向前，进走廊后再左转。参考路径开头是右转，之后才左转。144写明右边的门再左转，全量评测成功，终点0.55米。"
                "142这次先左转约233度再平移，最近0.03米后又离开。漏写开头右转可以解释起步，不能单独解释到达后离开。"
                + heading
            ),
            A="起步航向日志支持转错了方向。本轮没有重开142的原图。",
            A_changes="unknown",
            B="特权路径开头向右，会否定起步的大角度左转。",
            B_changes="yes",
            D="指令的第一个转向词和参考路径首段相反，144补上这个右转后成功。会改变起步归类。",
            D_changes="yes" if path_right else "unknown",
            abcd_note="路径角度是本轮重算的。画面细节沿用先前回放。",
        )
    if episode_id == "116":
        return done(
            frames_reviewed="no",
            instruction_path_consistent="ambiguous",
            direction_relation_consistent="ambiguous",
            landmark_order_consistent="ambiguous",
            target_grounding_correct="unknown",
            goal_visible_at_decision="unknown",
            failure_source="路线或方向决策问题",
            reviewer_confidence="中",
            evidence=(
                "参考路径首段在起步朝向左侧约126度，但115、116、117三次实际都先右转（航向从60度降到约-80度），再左转。成功的115、117和失败的116在前20步同向。"
                "116的文本写了两次右转，成功的115写的是右转后再左转。116自己在第10步也开始左转，所以不能把失败归因于它照着第二个右转执行。"
                "第45步最近3.28米，当时仍在直行，opos不可见；第120步才停止，测地13.11米。分歧发生在开头之后，本轮没有重看第45步画面。"
                + heading
            ),
            A="尚无法验证。开头航向和成功兄弟相同，不能用开头画面解释失败。",
            A_changes="no",
            B="特权距离显示它在3.28米处没有停，之后走到13米。这能否定后半段，但是诊断信息。参考路径首段和三次实际右转并不相同，不能把参考路径当成唯一正确方向。",
            B_changes="yes",
            D="文本的第二个右转和成功回合的左转不同，但116自己也做了那个左转后仍然失败。一致性检查不能把这句改写成失败原因。",
            D_changes="no",
            abcd_note="同路线成功不能自动当成指令错误。这里开头动作和成功兄弟一致。",
        )
    if episode_id == "115":
        return done(
            frames_reviewed="no",
            instruction_path_consistent="yes",
            direction_relation_consistent="yes",
            landmark_order_consistent="yes",
            target_grounding_correct="unknown",
            goal_visible_at_decision="unknown",
            failure_source="不适用",
            reviewer_confidence="中",
            evidence="成功，终点0.85米。文本含右边的门和随后左转，与116同一参考路径的右然后左一致。未在本轮重看画面。" + heading,
            A="尚无法验证",
            A_changes="no",
            B="不需要用特权路径改正这句已经成功的指令。",
            B_changes="no",
            D="文本顺序和参考路径一致，不应改写。",
            D_changes="no",
            abcd_note="成功对照。目标像素没有新的人工核对。",
        )
    if episode_id == "117":
        return done(
            frames_reviewed="no",
            instruction_path_consistent="ambiguous",
            direction_relation_consistent="ambiguous",
            landmark_order_consistent="ambiguous",
            target_grounding_correct="unknown",
            goal_visible_at_decision="unknown",
            failure_source="不适用",
            reviewer_confidence="中",
            evidence=(
                "文本只写进卧室后左转进浴室，没有写开头的右转。实际先右转约125度并成功，终点0.53米。"
                "这是省略，不是一次失败。机械地补上转向会改写一句已经走对的指令。"
                + heading
            ),
            A="尚无法验证",
            A_changes="no",
            B="特权路径含开头右转，但模型已经自己做出这段右转。",
            B_changes="no",
            D="能发现省略，但不应把它当成需要纠正的错误指令。",
            D_changes="no",
            abcd_note="同路线历史上界如果拿115的专家路径教117，会把诊断信息泄漏进输入，本轮没有做。",
        )
    if episode_id == "144":
        path_right = first_rel is not None and first_rel < -45
        return done(
            frames_reviewed="no",
            instruction_path_consistent="yes" if path_right else "ambiguous",
            direction_relation_consistent="yes" if path_right else "ambiguous",
            landmark_order_consistent="unknown",
            target_grounding_correct="unknown",
            goal_visible_at_decision="unknown",
            failure_source="不适用",
            reviewer_confidence="中",
            evidence="成功，终点0.55米。文本写右边的门再左转，和参考路径先右后左一致。未重看画面。" + heading,
            D="文本和参考路径一致，不应改写。",
            D_changes="no",
            A_changes="no",
            B="不需要改这句成功指令。",
            B_changes="no",
            abcd_note="成功对照，用来对比142。",
        )
    if episode_id == "143":
        return done(
            frames_reviewed="no",
            instruction_path_consistent="ambiguous",
            direction_relation_consistent="ambiguous",
            landmark_order_consistent="unknown",
            target_grounding_correct="unknown",
            goal_visible_at_decision="unknown",
            failure_source="证据不足",
            reviewer_confidence="低",
            evidence=(
                "全量结果失败，终点10.04米，最近5.62米。文本写进走廊后在电话处左转，没有写144里的右边的门。"
                "这可能是省略，但本轮没有看143的画面，不能定成指令错误。"
                + heading
            ),
            D="文本比144少了开头右转，只能算可疑，不能算已经证明。",
            D_changes="unknown",
            abcd_note="有逐步日志，本轮未打开画面。",
        )
    if episode_id == "173":
        return done(
            frames_reviewed="no",
            instruction_path_consistent="ambiguous",
            direction_relation_consistent="ambiguous",
            landmark_order_consistent="unknown",
            target_grounding_correct="unknown",
            goal_visible_at_decision="unknown",
            failure_source="证据不足",
            reviewer_confidence="低",
            evidence=(
                "173写在楼梯底部向左转。参考路径首段只偏右约48度，随后要到约9.7米才有更大的右转。这个角度不够把“先左转”定成和路径相反。"
                "172写楼梯右侧并成功，终点1.01米。173终点8.83米。本轮没有看173的画面，不用172的成功代替173的指令审查。"
                + heading
            ),
            B="尚无法验证。48度的首段偏差不足以排除左转。",
            B_changes="unknown",
            D="文本和参考路径的差异没有大到可以改判。",
            D_changes="no",
            abcd_note="同路线成功只说明另一句走通了，不自动证明173写错。",
        )
    if episode_id == "172":
        return done(
            frames_reviewed="no",
            instruction_path_consistent="unknown",
            direction_relation_consistent="ambiguous",
            landmark_order_consistent="unknown",
            target_grounding_correct="unknown",
            goal_visible_at_decision="unknown",
            failure_source="不适用",
            reviewer_confidence="中",
            evidence="成功，终点1.01米。文本含两段向右，并要求停在双扇玻璃门。未重看画面，所以不把路标标成yes。" + heading,
            D="未逐段核对，不声称发现冲突，也不改这句。",
            D_changes="no",
            A_changes="no",
            B_changes="no",
            abcd_note="成功对照。",
        )
    if episode_id == "174":
        return done(
            frames_reviewed="no",
            failure_source="停止决策问题",
            reviewer_confidence="中",
            evidence=(
                "174最近0.98米，结束4.21米。它已经进过1米内，失败来自离开，不是没有走到。"
                "同路线172成功。本轮没有重开174到达时的画面，不能判断双扇门是否可见。"
                + bits + heading
            ),
            B="特权停止会在0.98米附近停下，从而避免离开。这是诊断标签。",
            B_changes="yes",
            D="没有新的文本冲突证据。到达后再离开更像停止，不像指令漏写右转。",
            D_changes="no",
            abcd_note="停止结论来自全量距离和逐步测地，不是新画面。",
        )
    if episode_id == "215":
        return done(
            frames_reviewed="no",
            instruction_path_consistent="ambiguous",
            direction_relation_consistent="ambiguous",
            landmark_order_consistent="unknown",
            target_grounding_correct="ambiguous",
            goal_visible_at_decision="unknown",
            failure_source="不适用",
            reviewer_confidence="中",
            evidence=(
                "成功，终点0.87米。文本最后是抽屉和架子。214在同一终点坐标的最近帧里看到了衣柜和抽屉，没有床。"
                "这支持214的床不可靠，但215自己的停止帧本轮没有打开。"
                + heading
            ),
            D="不改这句成功指令。它被用来对照214的目标词。",
            D_changes="no",
            A_changes="no",
            B_changes="no",
            abcd_note="目标词对照借用了214的终点画面，不是215的独立截图。",
        )
    if episode_id == "216":
        return done(
            frames_reviewed="no",
            failure_source="不适用",
            reviewer_confidence="中",
            evidence="成功，终点0.65米。文本是向右偏并在敞开的门口等待。214的终点画面里有门口。216自己的停止帧未打开。" + heading,
            D="不改这句成功指令。",
            D_changes="no",
            A_changes="no",
            B_changes="no",
            abcd_note="成功对照。",
        )
    if ok:
        return done(
            frames_reviewed="no",
            failure_source="不适用",
            reviewer_confidence="低",
            evidence=f"全量评测成功，终点{float(result['final_distance']):.2f}米，最近{float(result['min_distance']):.2f}米。本轮没有重看画面，不判断路标。" + bits,
            A_changes="no",
            B_changes="no",
            D="未核对文本和路径，不能声称一致，也不把成功指令改写。",
            D_changes="no",
            abcd_note="成功对照只证明这句在全量评测里成功。",
        )
    if episode_id == "455":
        return done(
            frames_reviewed="no",
            failure_source="路线或方向决策问题",
            reviewer_confidence="低",
            evidence=(
                "364步前进指令几乎没有位移，但逐步日志的碰撞标记是假，不能写成撞墙。"
                "距离上升的步里，特权路径多数在身后。画面没有重开，所以执行问题只记为未证实。"
                + bits
            ),
            B="特权路径会否定那些身后仍直行的步。机器人没有可用的局部地图日志。",
            B_changes="yes",
            D="未核对指令文本和路径。",
            D_changes="unknown",
            abcd_note="低位移本身不是碰撞证据。",
        )
    # Remaining failures: step logs exist for the screening set, frames not reopened.
    source = "证据不足"
    b_changes = "unknown"
    b_text = "尚无法验证"
    if screen_row:
        behind = int(screen_row["forward_while_path_behind"])
        ignored = int(screen_row["expert_stop_ignored_steps"])
        if screen_row["probe_class"] in {"left_after_1m", "left_after_3m"} or ignored > 0:
            source = "停止决策问题"
            b_text = "特权停止标签显示进过成功半径后又离开，或存在被忽略的专家停止。这不是机器人自己的地图。"
            b_changes = "yes"
        elif behind >= 10:
            source = "路线或方向决策问题"
            b_text = f"特权路径在{behind}个距离上升步位于身后，模型仍直行。可以否定那些直行，但是诊断信息。"
            b_changes = "yes"
        elif screen_row["probe_class"] == "stop_just_outside":
            source = "停止决策问题"
            b_text = "特权距离停在3到5米。没有新的画面证明路标已经出现。"
            b_changes = "yes"
    return done(
        frames_reviewed="no",
        failure_source=source,
        reviewer_confidence="低",
        evidence="本轮没有重开画面。下面只转述已有逐步日志，不把同路线成功兄弟当成指令错误。" + bits + heading,
        B=b_text,
        B_changes=b_changes,
        D="未做文本和路径的逐句核对。",
        D_changes="unknown",
        abcd_note="A保持尚无法验证。C的同路线历史上界没有运行。",
    )


def tally(review_rows: list[dict], evidence_rows: list[dict]) -> dict:
    fails = [row for row in review_rows if str(row["baseline_success"]) == "0"]
    inconsistent = [
        row["episode_id"]
        for row in fails
        if row["instruction_path_consistent"] == "no"
        or row["direction_relation_consistent"] == "no"
        or row["target_grounding_correct"] == "no"
    ]
    by_source = Counter(row["failure_source"] for row in fails)
    d_changes = [row["episode_id"] for row in evidence_rows if row["D_changes_ranking"] == "yes"]
    d_no = [row["episode_id"] for row in evidence_rows if row["D_changes_ranking"] == "no"]
    return {
        "review_n": len(review_rows),
        "review_fail_n": len(fails),
        "inconsistent_fail_ids": inconsistent,
        "failure_source": dict(by_source),
        "D_changes_yes": d_changes,
        "D_changes_no": d_no,
    }


if __name__ == "__main__":
    main()
