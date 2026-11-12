#!/usr/bin/env python3
"""Do other annotated routes already pass the privileged branch?

For each of the 18 diagnostic failures that never entered 3 m, take the first
logged step where the model drives forward while the Go2 shortest path is more
than 30 degrees off the body. Recompute that shortest path from the logged
pose. The correct-side point is the first place at least 0.75 m along it whose
heading differs from the body by at least 30 degrees.

The memory corpus is every other trajectory in the same scene. The two sibling
instructions of the same trajectory share one reference path, so they are
excluded. A second corpus keeps only trajectories where LightNav-0 succeeded
on at least one instruction, and still uses that trajectory's annotated
polyline. Reference polylines are densified to 0.25 m. This uses annotated
routes, not images and not the path the model actually drove.

Run with the Habitat 3.9 env. It only loads navmeshes.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import habitat_sim
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "output" / "r2r_probe" / "trace_20261020"
DIAG = TRACE / "trace_diagnostics.csv"
STEPS = TRACE / "steps.jsonl"
DATASET = ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
RESULTS = ROOT / "output" / "r2r_go2w" / "results.jsonl"
NAVMESH = ROOT / "output/r2r_probe/error_screening_full_episode_20261021/navmesh_go2w"
OUT = Path(__file__).resolve().parent / "scene_memory_coverage.csv"

ALONG_M = 0.75
TURN_RAD = math.radians(30.0)
DENSE_M = 0.25
FLOOR_M = 0.5
NEAR_M = 1.0
ALIGN_RAD = math.radians(45.0)


def wrap(delta: float) -> float:
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def heading(dx: float, dz: float) -> float:
    return math.atan2(-dx, -dz)


def forward_xz(yaw: float) -> tuple[float, float]:
    return -math.sin(yaw), -math.cos(yaw)


def densify(points: list[np.ndarray]) -> list[tuple[np.ndarray, float]]:
    samples: list[tuple[np.ndarray, float]] = []
    for a, b in zip(points, points[1:]):
        delta = b - a
        flat = math.hypot(float(delta[0]), float(delta[2]))
        seg_heading = heading(float(delta[0]), float(delta[2])) if flat > 1e-6 else 0.0
        steps = max(1, int(math.floor(flat / DENSE_M)))
        for i in range(steps):
            t = i / steps
            samples.append((a * (1.0 - t) + b * t, seg_heading))
    if points:
        samples.append((points[-1], samples[-1][1] if samples else 0.0))
    return samples


def walk(points: list[np.ndarray]) -> list[tuple[np.ndarray, float, float]]:
    out: list[tuple[np.ndarray, float, float]] = []
    acc = 0.0
    for a, b in zip(points, points[1:]):
        delta = b - a
        flat = math.hypot(float(delta[0]), float(delta[2]))
        if flat < 1e-6:
            continue
        seg_heading = heading(float(delta[0]), float(delta[2]))
        n = max(1, int(math.ceil(flat / DENSE_M)))
        for i in range(1, n + 1):
            dist = min(flat, DENSE_M * i)
            t = dist / flat
            out.append((a * (1.0 - t) + b * t, acc + dist, seg_heading))
        acc += flat
    return out


def nearest(
    point: np.ndarray,
    corpus: list[tuple[np.ndarray, float, int]],
) -> tuple[float, int]:
    best = float("inf")
    routes: set[int] = set()
    for sample, _seg_heading, route_id in corpus:
        if abs(float(sample[1] - point[1])) > FLOOR_M:
            continue
        dist = math.hypot(float(sample[0] - point[0]), float(sample[2] - point[2]))
        if dist < best:
            best = dist
        if dist <= NEAR_M:
            routes.add(route_id)
    return best, len(routes)


def nearest_aligned(
    point: np.ndarray,
    want_heading: float,
    corpus: list[tuple[np.ndarray, float, int]],
) -> tuple[float, int, bool, float]:
    best = float("inf")
    best_gap = float("inf")
    routes: set[int] = set()
    aligned_routes: set[int] = set()
    for sample, seg_heading, route_id in corpus:
        if abs(float(sample[1] - point[1])) > FLOOR_M:
            continue
        dist = math.hypot(float(sample[0] - point[0]), float(sample[2] - point[2]))
        gap = abs(wrap(seg_heading - want_heading))
        if dist < best:
            best = dist
        if dist <= NEAR_M:
            routes.add(route_id)
            best_gap = min(best_gap, gap)
            if gap <= ALIGN_RAD:
                aligned_routes.add(route_id)
    return best, len(routes), len(aligned_routes) > 0, best_gap


def main() -> None:
    with gzip.open(DATASET, "rt") as handle:
        episodes = json.load(handle)["episodes"]
    by_id: dict[tuple[str, str], dict] = {}
    routes: dict[str, dict[int, list[np.ndarray]]] = defaultdict(dict)
    for episode in episodes:
        scene = episode["scene_id"].split("/")[-2]
        key = (scene, str(episode["episode_id"]))
        by_id[key] = episode
        route = int(episode["trajectory_id"])
        routes[scene][route] = [np.asarray(p, dtype=np.float64) for p in episode["reference_path"]]

    success_routes: dict[str, set[int]] = defaultdict(set)
    with RESULTS.open() as handle:
        for line in handle:
            row = json.loads(line)
            if float(row.get("success") or 0) < 0.5:
                continue
            scene = row["scene_id"].split("/")[-2]
            episode = by_id.get((scene, str(row["habitat_episode_id"])))
            if episode is not None:
                success_routes[scene].add(int(episode["trajectory_id"]))

    corpus: dict[str, dict[int, list[tuple[np.ndarray, float]]]] = {}
    for scene, route_paths in routes.items():
        corpus[scene] = {route: densify(points) for route, points in route_paths.items()}

    diag = list(csv.DictReader(DIAG.open()))
    wanted = {
        (row["scene"], row["episode_id"]): row
        for row in diag
        if row["group"] == "never_entered_3m" and row["first_forward_while_path_off_step"]
    }
    steps: dict[tuple[str, str, int], dict] = {}
    with STEPS.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["scene"], str(row["episode_id"]))
            if key in wanted and int(row["step"]) == int(wanted[key]["first_forward_while_path_off_step"]):
                steps[(key[0], key[1], int(row["step"]))] = row

    finders: dict[str, habitat_sim.PathFinder] = {}
    records = []
    for (scene, episode_id), meta in sorted(wanted.items()):
        step = int(meta["first_forward_while_path_off_step"])
        logged = steps[(scene, episode_id, step)]
        episode = by_id[(scene, episode_id)]
        route = int(episode["trajectory_id"])
        pose = np.asarray(logged["pose_xyz_before"], dtype=np.float64)
        yaw = float(logged["pose_yaw_before"])
        expert_yaw = logged.get("expert_yaw_rad")
        goal = np.asarray(episode["goals"][0]["position"], dtype=np.float64)
        if scene not in finders:
            finder = habitat_sim.PathFinder()
            finder.load_nav_mesh(str(NAVMESH / f"{scene}.navmesh"))
            finders[scene] = finder
        finder = finders[scene]
        query = habitat_sim.ShortestPath()
        query.requested_start = finder.snap_point(pose.tolist())
        query.requested_end = finder.snap_point(goal.tolist())
        found = bool(finder.find_path(query))
        others = []
        for other_route, samples in corpus[scene].items():
            if other_route == route:
                continue
            for sample, seg_heading in samples:
                others.append((sample, seg_heading, other_route))
        own = [(sample, seg_heading, route) for sample, seg_heading in corpus[scene][route]]
        own_dist, _ = nearest(pose, own)
        pose_dist, pose_routes = nearest(pose, others)
        fx, fz = forward_xz(yaw)
        ahead = pose.copy()
        ahead[0] += fx * 1.0
        ahead[2] += fz * 1.0
        ahead_dist, ahead_routes = nearest(ahead, others)
        success_samples = [item for item in others if item[2] in success_routes[scene]]

        correct = None
        along = ""
        path_err = ""
        sign_ok = ""
        path_geo = ""
        if found and len(query.points) >= 2:
            path_geo = round(float(query.geodesic_distance), 3)
            samples = walk([np.asarray(p, dtype=np.float64) for p in query.points])
            chosen = None
            for point, dist, seg_heading in samples:
                err = wrap(seg_heading - yaw)
                if dist >= ALONG_M and abs(err) >= TURN_RAD:
                    chosen = (point, dist, err, seg_heading)
                    break
            if chosen is None and samples:
                point, dist, seg_heading = samples[-1]
                chosen = (point, dist, wrap(seg_heading - yaw), seg_heading)
            if chosen is not None:
                correct, along, path_err, seg_heading = chosen
                if expert_yaw is None or abs(float(expert_yaw)) < 1e-3:
                    sign_ok = "no_expert_sign"
                else:
                    sign_ok = "yes" if path_err * float(expert_yaw) > 0 else "no"
        correct_dist = ""
        correct_routes = ""
        aligned = ""
        heading_gap = ""
        success_dist = ""
        success_routes_near = ""
        success_ahead = ""
        if correct is not None:
            correct_dist, correct_routes, aligned_flag, gap = nearest_aligned(correct, seg_heading, others)
            correct_dist = round(correct_dist, 3)
            aligned = "yes" if aligned_flag else "no"
            heading_gap = "" if math.isinf(gap) else round(math.degrees(gap), 1)
            if success_samples:
                success_dist, success_routes_near, _, _ = nearest_aligned(correct, seg_heading, success_samples)
                success_dist = round(success_dist, 3)
                ahead_success, _, _, _ = nearest_aligned(ahead, yaw, success_samples)
                success_ahead = round(ahead_success, 3)
            else:
                success_dist = ""
                success_routes_near = 0
                success_ahead = ""
        records.append({
            "scene": scene,
            "episode_id": episode_id,
            "trajectory_id": route,
            "step": step,
            "other_routes_in_scene": len(corpus[scene]) - 1,
            "path_found": found,
            "geodesic_m": path_geo,
            "correct_along_m": "" if along == "" else round(float(along), 3),
            "path_minus_body_deg": "" if path_err == "" else round(math.degrees(float(path_err)), 1),
            "path_sign_matches_expert": sign_ok,
            "correct_to_other_route_m": correct_dist,
            "other_routes_within_1m": correct_routes,
            "nearest_route_heading_gap_deg": heading_gap,
            "other_route_same_direction": aligned,
            "success_routes_in_scene": len(success_routes[scene] - {route}),
            "correct_to_success_route_m": success_dist,
            "success_routes_within_1m": success_routes_near,
            "ahead_to_success_route_m": success_ahead,
            "pose_to_other_route_m": round(pose_dist, 3),
            "pose_routes_within_1m": pose_routes,
            "ahead_to_other_route_m": round(ahead_dist, 3),
            "ahead_routes_within_1m": ahead_routes,
            "pose_to_own_route_m": round(own_dist, 3),
            "expert_yaw_deg": "" if expert_yaw is None else round(math.degrees(float(expert_yaw)), 1),
            "executed_yaw_deg": round(math.degrees(float(logged["executed_waypoint"][2])), 1),
        })

    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    covered = [
        row for row in records
        if row["correct_to_other_route_m"] != "" and float(row["correct_to_other_route_m"]) <= NEAR_M
    ]
    print(json.dumps({
        "episodes": len(records),
        "path_found": sum(1 for row in records if row["path_found"]),
        "sign_match": sum(1 for row in records if row["path_sign_matches_expert"] == "yes"),
        "correct_within_1m": len(covered),
        "same_direction": sum(1 for row in covered if row["other_route_same_direction"] == "yes"),
    }, indent=2))


if __name__ == "__main__":
    main()
