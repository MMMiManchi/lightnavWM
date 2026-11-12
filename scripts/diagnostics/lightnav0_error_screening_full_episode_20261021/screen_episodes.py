#!/usr/bin/env python3
"""Offline screening of the 30 traced R2R episodes.

Run with the Habitat 3.9 interpreter so the Go2-rebaked navmesh can be queried.
Does not roll out a policy and does not modify the model or the controller.

    PYTHONPATH=src /home/szgx/miniconda3/envs/habitat/bin/python \
        scripts/diagnostics/lightnav0_error_screening_full_episode_20261021/screen_episodes.py
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from habitat_sim import PathFinder, ShortestPath

from lightnav.expert import ego_waypoints, face_path_waypoints, sample_polyline

ROOT = Path(__file__).resolve().parents[3]
TRACE = ROOT / "output/r2r_probe/trace_20261020"
OUT = ROOT / "output/r2r_probe/error_screening_full_episode_20261021"
NAVMESH = OUT / "navmesh_go2w"
DATASET = ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
PAIRED = ROOT / "experiments/r2r_diagnosis/paired_episode_comparison.csv"
TEMPORAL = ROOT / "output/r2r_probe/visual_audit_temporal_20261020/temporal_visual_audit.csv"
GO2_RESULTS = ROOT / "output/r2r_go2w/results.jsonl"

GEO_MATCH_M = 0.05
LABEL_ATOL = 0.02
BODY_RADIUS_M = 0.215


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def frame_axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    forward = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float64)
    left = np.array([-math.cos(yaw), 0.0, math.sin(yaw)], dtype=np.float64)
    return forward, left


def robot_to_world(origin: np.ndarray, yaw: float, forward_m: float, lateral_m: float) -> np.ndarray:
    forward, left = frame_axes(yaw)
    return origin + forward_m * forward + lateral_m * left


def fmt_pts(points: list[np.ndarray]) -> str:
    return ";".join(f"{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}" for p in points)


def closest_on_polyline(points: list[np.ndarray], query: np.ndarray) -> tuple[float, float]:
    """Return (horizontal distance to the polyline, arc length of the closest point)."""
    if len(points) < 2:
        if not points:
            return float("inf"), 0.0
        d = query - points[0]
        d[1] = 0.0
        return float(np.linalg.norm(d)), 0.0
    best = float("inf")
    best_along = 0.0
    acc = 0.0
    q = query.copy()
    q[1] = 0.0
    for a, b in zip(points, points[1:]):
        aa = a.copy()
        bb = b.copy()
        aa[1] = 0.0
        bb[1] = 0.0
        seg = bb - aa
        length = float(np.linalg.norm(seg))
        if length < 1e-8:
            dist = float(np.linalg.norm(q - aa))
            along = acc
        else:
            t = float(np.clip(np.dot(q - aa, seg) / (length * length), 0.0, 1.0))
            proj = aa + t * seg
            dist = float(np.linalg.norm(q - proj))
            along = acc + t * length
        if dist < best:
            best = dist
            best_along = along
        acc += length
    return best, best_along


def load_goals() -> dict[tuple[str, str], np.ndarray]:
    with gzip.open(DATASET, "rt") as handle:
        data = json.load(handle)
    goals = {}
    starts = {}
    for episode in data["episodes"]:
        scene = episode["scene_id"].split("/")[-2]
        key = (scene, str(episode["episode_id"]))
        goals[key] = np.asarray(episode["goals"][0]["position"], dtype=np.float64)
        starts[key] = np.asarray(episode["start_position"], dtype=np.float64)
    return goals, starts


def load_pathfinders() -> dict[str, PathFinder]:
    finders = {}
    for path in sorted(NAVMESH.glob("*.navmesh")):
        finder = PathFinder()
        if not finder.load_nav_mesh(str(path)):
            raise RuntimeError(f"cannot load {path}")
        finders[path.stem] = finder
    return finders


def obstacle_distance(finder: PathFinder, point: np.ndarray) -> float | None:
    fn = getattr(finder, "distance_to_closest_obstacle", None)
    if fn is None:
        return None
    try:
        return float(fn(point, 2.0))
    except TypeError:
        try:
            return float(fn(point))
        except Exception:
            return None
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    goals, starts = load_goals()
    finders = load_pathfinders()
    paired = {}
    with PAIRED.open() as handle:
        for row in csv.DictReader(handle):
            paired[(row["scene"], row["habitat_episode_id"])] = row

    visual = defaultdict(list)
    if TEMPORAL.exists():
        with TEMPORAL.open() as handle:
            for row in csv.DictReader(handle):
                visual[row["episode_id"]].append(row)

    episodes = []
    steps_by = defaultdict(list)
    for line in (TRACE / "episodes.jsonl").open():
        episodes.append(json.loads(line))
    for line in (TRACE / "steps.jsonl").open():
        row = json.loads(line)
        steps_by[(row["scene"], str(row["episode_id"]))].append(row)
    for key in steps_by:
        steps_by[key].sort(key=lambda row: int(row["step"]))

    inventory = []
    timeline = []
    expert_rows = []
    geometry_rows = []
    per_episode = []

    for episode in episodes:
        scene = episode["scene"]
        eid = str(episode["episode_id"])
        key = (scene, eid)
        rows = steps_by[key]
        meta = paired.get(key, {})
        frame_dir = TRACE / "frames" / f"{scene}_{eid}"
        frame_files = sorted(frame_dir.glob("*.jpg")) if frame_dir.is_dir() else []
        expected = {f"{i:04d}.jpg" for i in range(len(rows))}
        present = {path.name for path in frame_files}
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        steps_idx = [int(row["step"]) for row in rows]
        contiguous = steps_idx == list(range(steps_idx[0], steps_idx[0] + len(steps_idx))) if steps_idx else False
        ten = sum(1 for row in rows if isinstance(row.get("waypoints_robot"), list) and len(row["waypoints_robot"]) == 10)
        inventory.append(
            {
                "scene": scene,
                "episode_id": eid,
                "probe_class": episode.get("probe_class") or meta.get("probe_class", ""),
                "failure_type": meta.get("failure_type", ""),
                "success": episode.get("success"),
                "n_steps": len(rows),
                "step_first": steps_idx[0] if steps_idx else "",
                "step_last": steps_idx[-1] if steps_idx else "",
                "steps_contiguous_from_zero": int(contiguous and steps_idx[:1] == [0]),
                "n_frames_on_disk": len(frame_files),
                "missing_frames": " ".join(missing),
                "extra_frames": " ".join(extra),
                "instruction_present": int(bool(rows and rows[0].get("instruction"))),
                "pose_present": int(all(row.get("pose_xyz_before") and row.get("pose_yaw_before") is not None for row in rows)),
                "waypoints_10_present": ten,
                "waypoints_10_missing": len(rows) - ten,
                "expert_present": sum(1 for row in rows if row.get("expert_waypoint")),
                "geodesic_present": sum(1 for row in rows if row.get("geodesic_before_m") is not None),
                "euclidean_present": sum(1 for row in rows if row.get("euclidean_horizontal_before_m") is not None),
                "apos_present": sum(1 for row in rows if row.get("apos_id") is not None),
                "opos_present": sum(1 for row in rows if row.get("opos_id") is not None),
                "history_frame_count_logged": int(bool(rows and "history_frames" in rows[0])),
                "history_images_stored": 0,
                "goal_in_dataset": int(key in goals),
                "go2_navmesh_present": int(scene in finders),
                "shipped_navmesh_present": int((ROOT / f"data/scene_datasets/mp3d/{scene}/{scene}.navmesh").exists()),
                "notes": "历史帧只记录了数量，没有保存历史 RGB。专家折线不在逐步日志里，由 Go2 重算导航网格补回。",
            }
        )

        finder = finders[scene]
        goal = goals[key]
        rise_run = 0
        straight_run = 0
        postpone_run = 0
        prev_wp9_sign = 0
        stats = defaultdict(int)
        cross_rise = []
        uncapped_rise = []
        lateral_exec = []
        obstacle_vals = []
        match_n = 0
        first_rise = ""
        longest_straight = 0
        longest_postpone = 0
        straight_span = []
        postpone_span = []

        for index, row in enumerate(rows):
            wp = np.asarray(row["waypoints_robot"], dtype=np.float64) if row.get("waypoints_robot") else None
            origin = np.asarray(row["pose_xyz_before"], dtype=np.float64)
            yaw = float(row["pose_yaw_before"])
            expert = row.get("expert_waypoint")
            expert_arr = np.asarray(expert, dtype=np.float64) if expert else None
            sp = ShortestPath()
            sp.requested_start = origin
            sp.requested_end = goal
            found = bool(finder.find_path(sp)) and sp.points is not None and len(sp.points) >= 2
            path_pts = [np.asarray(p, dtype=np.float64) for p in sp.points] if found else []
            geo_hat = float(sp.geodesic_distance) if found else float("inf")
            geo_logged = float(row["geodesic_before_m"])
            geo_err = geo_hat - geo_logged if found and math.isfinite(geo_logged) else float("nan")
            faced0 = None
            raw0 = None
            uncapped = float("nan")
            raw_forward = float("nan")
            if found:
                samples = sample_polyline(path_pts, step_m=0.25, horizon=10)
                forward, left = frame_axes(yaw)
                ego = ego_waypoints(origin, forward, left, yaw, samples)
                faced = face_path_waypoints(ego)
                raw0 = ego[0]
                faced0 = faced[0]
                uncapped = math.degrees(float(raw0[2]))
                raw_forward = float(raw0[0])
            label_match = False
            if faced0 is not None and expert_arr is not None:
                label_match = bool(np.allclose(faced0, expert_arr, atol=LABEL_ATOL))
            geo_match = bool(found and abs(geo_err) <= GEO_MATCH_M)
            valid = bool(found and geo_match and label_match)
            if valid:
                match_n += 1
                stats["valid"] += 1

            world_wps = []
            cross = []
            obst = []
            if wp is not None:
                for point in wp:
                    world = robot_to_world(origin, yaw, float(point[0]), float(point[1]))
                    world_wps.append(world)
                    if path_pts:
                        dist, _along = closest_on_polyline(path_pts, world)
                        cross.append(dist)
                    dist_obs = obstacle_distance(finder, world)
                    if dist_obs is not None and math.isfinite(dist_obs):
                        obst.append(dist_obs)
                        obstacle_vals.append(dist_obs)
            along_agent = closest_on_polyline(path_pts, origin)[1] if path_pts else float("nan")
            along_last = closest_on_polyline(path_pts, world_wps[-1])[1] if path_pts and world_wps else float("nan")

            cmd_yaw = math.degrees(float(row["commanded_yaw_rad"])) if row.get("commanded_yaw_rad") is not None else float("nan")
            cmd_fwd = float(row["commanded_forward_m"]) if row.get("commanded_forward_m") is not None else float("nan")
            act_fwd = float(row["actual_forward_m"]) if row.get("actual_forward_m") is not None else float("nan")
            moved = float(row["moved_m"]) if row.get("moved_m") is not None else float("nan")
            geo_delta = float(row["geodesic_delta_m"]) if row.get("geodesic_delta_m") is not None else float("nan")
            expert_yaw = math.degrees(float(expert_arr[2])) if expert_arr is not None else float("nan")
            expert_fwd = float(expert_arr[0]) if expert_arr is not None else float("nan")
            capped = bool(expert_arr is not None and abs(expert_fwd) <= 0.05 and abs(abs(expert_arr[2]) - math.radians(30)) < 1e-3)
            yaws = np.degrees(wp[:, 2]) if wp is not None else np.zeros(0)
            yaw0 = float(yaws[0]) if len(yaws) else float("nan")
            yaw9 = float(yaws[-1]) if len(yaws) else float("nan")
            lat0 = float(wp[0, 1]) if wp is not None else float("nan")
            if math.isfinite(lat0):
                lateral_exec.append(abs(lat0))

            events = []
            if math.isfinite(geo_delta) and geo_delta > 0.05:
                events.append("geo_rise")
                rise_run += 1
                stats["rise"] += 1
                if rise_run == 1:
                    events.append("rise_start")
                    if first_rise == "":
                        first_rise = int(row["step"])
                if valid and math.isfinite(uncapped):
                    uncapped_rise.append(uncapped)
                    if cross:
                        cross_rise.append(max(cross))
                if valid and raw_forward <= 0.05 and abs(uncapped) > 90 and cmd_fwd > 0.1 and abs(cmd_yaw) < 15:
                    stats["forward_while_path_behind"] += 1
                    events.append("forward_while_path_behind")
                if valid and raw_forward <= 0.05 and 30 < abs(uncapped) <= 90 and cmd_fwd > 0.1 and abs(cmd_yaw) < 15:
                    stats["forward_while_path_side"] += 1
                    events.append("forward_while_path_side")
            else:
                rise_run = 0
            if capped and cmd_fwd > 0.1 and abs(cmd_yaw) < 15:
                straight_run += 1
                stats["straight_while_capped"] += 1
                events.append("straight_while_capped")
                longest_straight = max(longest_straight, straight_run)
            else:
                if straight_run >= 5:
                    straight_span.append(f"{int(row['step']) - straight_run}-{int(row['step']) - 1}")
                straight_run = 0
            if math.isfinite(yaw9) and abs(yaw9) >= 15 and abs(yaw0) < 5:
                postpone_run += 1
                stats["far_turn_near_zero"] += 1
                events.append("far_turn_immediate_straight")
                longest_postpone = max(longest_postpone, postpone_run)
            else:
                if postpone_run >= 5:
                    postpone_span.append(f"{int(row['step']) - postpone_run}-{int(row['step']) - 1}")
                postpone_run = 0
            sign9 = 0 if not math.isfinite(yaw9) or abs(yaw9) < 15 else (1 if yaw9 > 0 else -1)
            if prev_wp9_sign and sign9 and sign9 != prev_wp9_sign:
                events.append("far_heading_flip")
                stats["far_flip"] += 1
            if sign9:
                prev_wp9_sign = sign9
            if cmd_fwd > 0.1 and math.isfinite(moved) and moved < 0.02:
                events.append("commanded_forward_but_barely_moved")
                stats["stuck"] += 1
            if abs(cmd_yaw) > 20 and cmd_fwd <= 0.05:
                events.append("turn_in_place")
                stats["spin"] += 1
            before_g = float(row["geodesic_before_m"])
            after_g = float(row["geodesic_after_m"]) if row.get("geodesic_after_m") is not None else float("nan")
            if before_g >= 3.0 and math.isfinite(after_g) and after_g < 3.0:
                events.append("entered_3m")
                stats["entered_3m"] += 1
            if before_g < 3.0 and math.isfinite(after_g) and after_g >= 3.0:
                events.append("left_3m")
                stats["left_3m"] += 1
            if row.get("model_stop"):
                events.append("model_stop")
            if row.get("expert_stop") and not row.get("model_stop"):
                events.append("expert_stop_ignored")
                stats["expert_stop_ignored"] += 1
            if obst and min(obst) < BODY_RADIUS_M:
                events.append("waypoint_center_within_body_radius_of_obstacle")
                stats["near_obstacle"] += 1

            expert_rows.append(
                {
                    "scene": scene,
                    "episode_id": eid,
                    "step": int(row["step"]),
                    "path_found": int(found),
                    "geodesic_logged_m": round(geo_logged, 5) if math.isfinite(geo_logged) else "",
                    "geodesic_recomputed_m": round(geo_hat, 5) if found else "",
                    "geodesic_error_m": round(geo_err, 5) if math.isfinite(geo_err) else "",
                    "geodesic_match": int(geo_match),
                    "label_match": int(label_match),
                    "reconstruction_valid": int(valid),
                    "logged_expert_forward_m": "" if expert_arr is None else round(float(expert_arr[0]), 5),
                    "logged_expert_yaw_deg": "" if expert_arr is None else round(math.degrees(float(expert_arr[2])), 3),
                    "recomputed_immediate_yaw_deg": "" if faced0 is None else round(math.degrees(float(faced0[2])), 3),
                    "uncapped_first_sample_forward_m": "" if raw0 is None else round(raw_forward, 4),
                    "uncapped_first_sample_yaw_deg": "" if raw0 is None else round(uncapped, 2),
                    "logged_label_is_capped_turn": int(capped),
                    "n_path_points": len(path_pts),
                    "path_world_xyz": fmt_pts(path_pts),
                }
            )
            geometry_rows.append(
                {
                    "scene": scene,
                    "episode_id": eid,
                    "step": int(row["step"]),
                    "reconstruction_valid": int(valid),
                    "cross_track_wp0_m": "" if not cross else round(cross[0], 4),
                    "cross_track_wp4_m": "" if len(cross) < 5 else round(cross[4], 4),
                    "cross_track_wp9_m": "" if not cross else round(cross[-1], 4),
                    "cross_track_max_m": "" if not cross else round(max(cross), 4),
                    "along_track_wp9_m": "" if not math.isfinite(along_last) else round(along_last, 4),
                    "along_track_agent_m": "" if not math.isfinite(along_agent) else round(along_agent, 4),
                    "min_obstacle_distance_m": "" if not obst else round(min(obst), 4),
                    "waypoints_closer_than_body_radius": sum(1 for value in obst if value < BODY_RADIUS_M),
                    "body_swept_collision": "unknown",
                    "geodesic_delta_m": "" if not math.isfinite(geo_delta) else round(geo_delta, 5),
                }
            )
            timeline.append(
                {
                    "scene": scene,
                    "episode_id": eid,
                    "step": int(row["step"]),
                    "frame_file": row.get("frame_file", ""),
                    "instruction": row.get("instruction", ""),
                    "history_frames": row.get("history_frames", ""),
                    "pose_x": origin[0],
                    "pose_y": origin[1],
                    "pose_z": origin[2],
                    "pose_yaw_deg": round(math.degrees(yaw), 3),
                    "commanded_forward_m": "" if not math.isfinite(cmd_fwd) else round(cmd_fwd, 4),
                    "commanded_yaw_deg": "" if not math.isfinite(cmd_yaw) else round(cmd_yaw, 3),
                    "actual_forward_m": row.get("actual_forward_m", ""),
                    "actual_left_m": row.get("actual_left_m", ""),
                    "actual_dyaw_deg": "" if row.get("actual_dyaw_rad") is None else round(math.degrees(float(row["actual_dyaw_rad"])), 3),
                    "moved_m": row.get("moved_m", ""),
                    "yaw_follow_deg": "" if row.get("yaw_follow_rad") is None else round(math.degrees(abs(float(row["yaw_follow_rad"]))), 5),
                    "lateral_ignored_m": row.get("lateral_ignored_m", ""),
                    "geodesic_before_m": row.get("geodesic_before_m", ""),
                    "geodesic_after_m": row.get("geodesic_after_m", ""),
                    "geodesic_delta_m": row.get("geodesic_delta_m", ""),
                    "euclidean_before_m": row.get("euclidean_horizontal_before_m", ""),
                    "euclidean_after_m": row.get("euclidean_horizontal_after_m", ""),
                    "expert_stop": int(bool(row.get("expert_stop"))),
                    "expert_forward_m": "" if not math.isfinite(expert_fwd) else round(expert_fwd, 4),
                    "expert_yaw_deg": "" if not math.isfinite(expert_yaw) else round(expert_yaw, 3),
                    "expert_label_capped": int(capped),
                    "uncapped_path_yaw_deg": "" if not math.isfinite(uncapped) else round(uncapped, 2),
                    "model_stop": int(bool(row.get("model_stop"))),
                    "apos_state": row.get("apos_state", ""),
                    "apos_id": row.get("apos_id", ""),
                    "opos_state": row.get("opos_state", ""),
                    "opos_id": row.get("opos_id", ""),
                    "wp0_forward_m": "" if wp is None else round(float(wp[0, 0]), 4),
                    "wp0_left_m": "" if wp is None else round(float(wp[0, 1]), 4),
                    "wp0_yaw_deg": "" if not math.isfinite(yaw0) else round(yaw0, 3),
                    "wp9_forward_m": "" if wp is None else round(float(wp[-1, 0]), 4),
                    "wp9_left_m": "" if wp is None else round(float(wp[-1, 1]), 4),
                    "wp9_yaw_deg": "" if not math.isfinite(yaw9) else round(yaw9, 3),
                    "waypoints_ego": ""
                    if wp is None
                    else ";".join(
                        f"{float(p[0]):.4f},{float(p[1]):.4f},{math.degrees(float(p[2])):.2f}"
                        for p in wp
                    ),
                    "max_abs_yaw_deg": "" if not len(yaws) else round(float(np.max(np.abs(yaws))), 3),
                    "collided": row.get("collided", ""),
                    "collision_source": row.get("collision_source", ""),
                    "reconstruction_valid": int(valid),
                    "events": "|".join(events),
                }
            )

        if straight_run >= 5:
            straight_span.append(f"{steps_idx[-1] - straight_run + 1}-{steps_idx[-1]}")
        if postpone_run >= 5:
            postpone_span.append(f"{steps_idx[-1] - postpone_run + 1}-{steps_idx[-1]}")
        final_geo = float(rows[-1]["geodesic_after_m"])
        min_geo = min(float(row["geodesic_before_m"]) for row in rows)
        min_geo = min(min_geo, final_geo)
        opos_point = sum(1 for row in rows if row.get("opos_state") == "point")
        per_episode.append(
            {
                "scene": scene,
                "episode_id": eid,
                "failure_type": meta.get("failure_type", ""),
                "probe_class": meta.get("probe_class", episode.get("probe_class", "")),
                "success": int(bool(episode.get("success"))),
                "expert_success": meta.get("expert_success", ""),
                "stop_oracle_success": meta.get("stop_success", ""),
                "steps": len(rows),
                "final_geodesic_m": round(final_geo, 4),
                "min_geodesic_m": round(min_geo, 4),
                "entered_3m": int(min_geo < 3.0),
                "first_sustained_rise_step": first_rise,
                "geo_rise_steps": stats["rise"],
                "straight_while_capped_steps": stats["straight_while_capped"],
                "longest_straight_while_capped": longest_straight,
                "straight_runs": " ".join(straight_span),
                "far_turn_immediate_straight_steps": stats["far_turn_near_zero"],
                "longest_postpone": longest_postpone,
                "postpone_runs": " ".join(postpone_span),
                "forward_while_path_behind": stats["forward_while_path_behind"],
                "forward_while_path_side": stats["forward_while_path_side"],
                "far_heading_flips": stats["far_flip"],
                "stuck_steps": stats["stuck"],
                "expert_stop_ignored_steps": stats["expert_stop_ignored"],
                "near_obstacle_steps": stats["near_obstacle"],
                "reconstruction_valid_steps": match_n,
                "reconstruction_valid_fraction": round(match_n / len(rows), 4),
                "opos_point_steps": opos_point,
                "max_abs_executed_lateral_m": "" if not lateral_exec else round(max(lateral_exec), 4),
                "p95_abs_executed_lateral_m": "" if not lateral_exec else round(float(np.quantile(lateral_exec, 0.95)), 4),
                "median_cross_track_on_rise_m": "" if not cross_rise else round(float(np.median(cross_rise)), 4),
                "prior_straight_run_frames_reviewed": len(visual.get(f"{scene}/{eid}", [])),
            }
        )

    # Infinite-geodesic episodes from the Go2-W eval, kept separate from the 30.
    infinite_rows = []
    for line in GO2_RESULTS.open():
        row = json.loads(line)
        distance = row["final_distance"]
        if isinstance(distance, float) and math.isfinite(distance):
            continue
        scene = row["scene_id"].split("/")[-2]
        eid = str(row["habitat_episode_id"])
        key = (scene, eid)
        finder = finders.get(scene)
        status = "navmesh_not_baked"
        geo = ""
        if finder is not None and key in starts and key in goals:
            sp = ShortestPath()
            sp.requested_start = starts[key]
            sp.requested_end = goals[key]
            found = bool(finder.find_path(sp)) and math.isfinite(float(sp.geodesic_distance))
            if found:
                status = "finite_on_rebaked_go2_navmesh"
                geo = round(float(sp.geodesic_distance), 4)
            else:
                status = "no_path_on_rebaked_go2_navmesh"
        infinite_rows.append(
            {
                "scene": scene,
                "habitat_episode_id": eid,
                "eval_steps": row["steps"],
                "eval_final_distance": "inf",
                "eval_min_distance": "inf",
                "eval_success": row["success"],
                "termination_reason": row["termination_reason"],
                "rebaked_start_to_goal": status,
                "rebaked_geodesic_m": geo,
            }
        )

    write_csv(OUT / "data_inventory.csv", inventory)
    write_csv(OUT / "episode_timeline.csv", timeline)
    write_csv(OUT / "expert_path_reconstruction.csv", expert_rows)
    write_csv(OUT / "trajectory_geometry.csv", geometry_rows)
    write_csv(OUT / "episode_level_screening.csv", per_episode)
    write_csv(OUT / "infinite_geodesic_episodes.csv", infinite_rows)

    valid = [row for row in expert_rows if row["path_found"] == 1]
    geo_ok = sum(int(row["geodesic_match"]) for row in valid)
    label_ok = sum(int(row["label_match"]) for row in valid)
    summary = {
        "episodes": len(episodes),
        "steps": len(timeline),
        "reconstruction_valid_steps": sum(int(row["reconstruction_valid"]) for row in expert_rows),
        "geodesic_match_steps": geo_ok,
        "label_match_steps": label_ok,
        "path_found_steps": len(valid),
        "infinite_episodes": len(infinite_rows),
        "infinite_no_path": sum(1 for row in infinite_rows if row["rebaked_start_to_goal"] == "no_path_on_rebaked_go2_navmesh"),
        "infinite_finite_on_rebake": sum(1 for row in infinite_rows if row["rebaked_start_to_goal"] == "finite_on_rebaked_go2_navmesh"),
        "inventory_missing_frames": sum(1 for row in inventory if row["missing_frames"]),
        "inventory_not_contiguous": sum(1 for row in inventory if not row["steps_contiguous_from_zero"]),
    }
    (OUT / "screening_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    for row in per_episode:
        print(
            f"{row['scene']}/{row['episode_id']} {row['failure_type']} "
            f"valid {row['reconstruction_valid_fraction']} rise {row['geo_rise_steps']} "
            f"behind {row['forward_while_path_behind']} side {row['forward_while_path_side']} "
            f"straight {row['straight_while_capped_steps']} postpone {row['far_turn_immediate_straight_steps']} "
            f"lat {row['max_abs_executed_lateral_m']} min {row['min_geodesic_m']}"
        )


if __name__ == "__main__":
    main()
