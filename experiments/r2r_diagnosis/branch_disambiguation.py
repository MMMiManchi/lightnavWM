#!/usr/bin/env python3
"""Can other annotated routes pick the privileged branch at 18 fixed steps?

The 18 steps stay the ones in scene_memory_coverage.csv. The privileged branch
is an offline label: the shortest path uses the episode goal, and the goal is
not a retrieval feature.

Two candidate sets are scored separately.

Oracle pair: the label point and the point 1 m straight ahead are both known.
Distance retrieval picks whichever sits closer to memory. Direction retrieval
only counts memory segments whose travel heading is within 45 degrees of that
side. This asks whether geometry can rank two known branches. It does not ask
whether a robot could nominate the privileged branch.

Memory-only: headings come from other routes within 1.5 m of the body, clustered
at 30 degrees, plus straight ahead. The goal is not used to add a candidate.
Distance retrieval returns the cluster that passes closest to the body.
Direction retrieval returns the cluster whose own 1 m continuation has the
closest aligned memory segment.

Full bank: every other trajectory in the scene. Ordered bank: trajectories whose
first episode appears earlier in val_unseen.json.gz. Same-trajectory siblings
stay out of both banks. Neither bank is the path the model drove.

Run with the Habitat 3.9 env.
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
from check_scene_memory_coverage import (
    ALONG_M,
    DATASET,
    FLOOR_M,
    NAVMESH,
    RESULTS,
    STEPS,
    TURN_RAD,
    densify,
    forward_xz,
    heading,
    walk,
    wrap,
)

HERE = Path(__file__).resolve().parent
COVERAGE = HERE / "scene_memory_coverage.csv"
OUT = HERE / "branch_disambiguation.csv"

PROBE_M = 1.0
LOCAL_M = 1.5
CLUSTER_RAD = math.radians(30.0)
MATCH_RAD = math.radians(45.0)
TIE_M = 0.05
SUPPORT_M = 1.5


def circ_gap(a: float, b: float) -> float:
    return abs(wrap(a - b))


def load_routes():
    with gzip.open(DATASET, "rt") as handle:
        episodes = json.load(handle)["episodes"]
    by_id = {}
    order = defaultdict(list)
    paths = defaultdict(dict)
    first_index = defaultdict(dict)
    for index, episode in enumerate(episodes):
        scene = episode["scene_id"].split("/")[-2]
        route = int(episode["trajectory_id"])
        by_id[(scene, str(episode["episode_id"]))] = episode
        order[scene].append((index, str(episode["episode_id"]), route))
        if route not in paths[scene]:
            paths[scene][route] = [np.asarray(p, dtype=np.float64) for p in episode["reference_path"]]
            first_index[scene][route] = index
    success = defaultdict(set)
    with RESULTS.open() as handle:
        for line in handle:
            row = json.loads(line)
            if float(row.get("success") or 0) < 0.5:
                continue
            scene = row["scene_id"].split("/")[-2]
            episode = by_id.get((scene, str(row["habitat_episode_id"])))
            if episode is not None:
                success[scene].add(int(episode["trajectory_id"]))
    samples = {
        scene: {route: densify(points) for route, points in route_paths.items()}
        for scene, route_paths in paths.items()
    }
    return by_id, order, samples, first_index, success


def bank(scene, route, samples, allowed):
    rows = []
    for other, pts in samples[scene].items():
        if other == route or other not in allowed:
            continue
        for point, seg_heading in pts:
            rows.append((point, seg_heading, other))
    return rows


def min_dist(point, rows, want_heading=None):
    best = float("inf")
    routes = set()
    for sample, seg_heading, route_id in rows:
        if abs(float(sample[1] - point[1])) > FLOOR_M:
            continue
        if want_heading is not None and circ_gap(seg_heading, want_heading) > MATCH_RAD:
            continue
        dist = math.hypot(float(sample[0] - point[0]), float(sample[2] - point[2]))
        if dist < best:
            best = dist
        if dist <= SUPPORT_M:
            routes.add(route_id)
    return best, len(routes)


def clusters_near(pose, yaw, rows):
    near = []
    for sample, seg_heading, route_id in rows:
        if abs(float(sample[1] - pose[1])) > FLOOR_M:
            continue
        dist = math.hypot(float(sample[0] - pose[0]), float(sample[2] - pose[2]))
        if dist <= LOCAL_M:
            near.append((dist, wrap(seg_heading - yaw), route_id))
    near.sort()
    groups = []
    for dist, rel, route_id in near:
        placed = None
        for group in groups:
            if circ_gap(rel, group["rel"]) <= CLUSTER_RAD:
                placed = group
                break
        if placed is None:
            groups.append({"rel": rel, "dist": dist, "routes": {route_id}})
        else:
            placed["routes"].add(route_id)
            if dist < placed["dist"]:
                placed["dist"] = dist
    return groups


def choose_pair(correct_dist, ahead_dist):
    if math.isinf(correct_dist) and math.isinf(ahead_dist):
        return "abstain"
    if abs(correct_dist - ahead_dist) <= TIE_M:
        return "tie"
    return "correct" if correct_dist < ahead_dist else "straight"


def pick_cluster(groups, pose, yaw, rows, mode):
    if not groups:
        return "abstain", "", ""
    scored = []
    for group in groups:
        world = yaw + group["rel"]
        if mode == "distance":
            score = group["dist"]
            support = len(group["routes"])
        else:
            fx, fz = forward_xz(world)
            probe = pose.copy()
            probe[0] += fx * PROBE_M
            probe[2] += fz * PROBE_M
            score, support = min_dist(probe, rows, world)
            if score > SUPPORT_M:
                continue
        scored.append((score, group["rel"], support))
    if not scored:
        return "abstain", "", ""
    scored.sort()
    best = scored[0]
    if len(scored) > 1 and abs(scored[1][0] - best[0]) <= TIE_M:
        return "tie", round(math.degrees(best[1]), 1), best[2]
    return "chosen", round(math.degrees(best[1]), 1), best[2]


def matches(choice_deg, target_deg):
    if choice_deg == "":
        return ""
    return "yes" if circ_gap(math.radians(float(choice_deg)), math.radians(target_deg)) <= MATCH_RAD else "no"


def main() -> None:
    by_id, _order, samples, first_index, success = load_routes()
    coverage = list(csv.DictReader(COVERAGE.open()))
    logged = {}
    with STEPS.open() as handle:
        for line in handle:
            row = json.loads(line)
            logged[(row["scene"], str(row["episode_id"]), int(row["step"]))] = row

    finders = {}
    records = []
    for meta in coverage:
        scene = meta["scene"]
        episode_id = meta["episode_id"]
        step = int(meta["step"])
        route = int(meta["trajectory_id"])
        row = logged[(scene, episode_id, step)]
        episode = by_id[(scene, episode_id)]
        pose = np.asarray(row["pose_xyz_before"], dtype=np.float64)
        yaw = float(row["pose_yaw_before"])
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
        correct = None
        travel = None
        if found and len(query.points) >= 2:
            for point, dist, seg_heading in walk([np.asarray(p, dtype=np.float64) for p in query.points]):
                err = wrap(seg_heading - yaw)
                if dist + 1e-9 >= ALONG_M and abs(err) >= TURN_RAD:
                    correct = point
                    travel = err
                    break
        bearing = ""
        label_gap = ""
        if correct is not None:
            bearing = wrap(heading(float(correct[0] - pose[0]), float(correct[2] - pose[2])) - yaw)
            label_gap = round(math.degrees(circ_gap(bearing, travel)), 1)
            bearing_deg = round(math.degrees(bearing), 1)
            travel_deg = round(math.degrees(travel), 1)
        else:
            bearing_deg = ""
            travel_deg = ""

        fx, fz = forward_xz(yaw)
        ahead = pose.copy()
        ahead[0] += fx * PROBE_M
        ahead[2] += fz * PROBE_M
        current_index = first_index[scene][route]
        # The current trajectory's own first index is this route, which may be
        # a sibling that appears earlier. Prior routes are those first seen
        # before this episode's own index, not before the trajectory's first.
        own_index = next(
            index for index, eid, _route in _order[scene] if eid == episode_id
        )
        prior = {other for other, index in first_index[scene].items() if index < own_index and other != route}
        full = set(samples[scene]) - {route}
        success_prior = prior & success[scene]
        banks = {
            "full": bank(scene, route, samples, full),
            "ordered": bank(scene, route, samples, prior),
            "ordered_success": bank(scene, route, samples, success_prior),
        }
        record = {
            "scene": scene,
            "episode_id": episode_id,
            "trajectory_id": route,
            "step": step,
            "path_found": found,
            "travel_deg": travel_deg,
            "bearing_deg": bearing_deg,
            "bearing_travel_gap_deg": label_gap,
            "full_routes": len(full),
            "ordered_routes": len(prior),
            "ordered_success_routes": len(success_prior),
            "dataset_index": own_index,
        }
        for name, rows in banks.items():
            if correct is None:
                pair_dist = pair_dir = "no_label"
                d_correct = d_ahead = ""
            else:
                d_correct, _ = min_dist(correct, rows)
                d_ahead, _ = min_dist(ahead, rows)
                pair_dist = choose_pair(d_correct, d_ahead)
                d_correct_dir, _ = min_dist(correct, rows, yaw + travel)
                d_ahead_dir, _ = min_dist(ahead, rows, yaw)
                # An unmatched side is worse than any finite side, but a side
                # farther than the support radius cannot be selected.
                def usable(dist):
                    return dist if dist <= SUPPORT_M else float("inf")
                pair_dir = choose_pair(usable(d_correct_dir), usable(d_ahead_dir))
                d_correct = round(d_correct, 3)
                d_ahead = round(d_ahead, 3)
            groups = clusters_near(pose, yaw, rows)
            dist_status, dist_deg, dist_n = pick_cluster(groups, pose, yaw, rows, "distance")
            dir_status, dir_deg, dir_n = pick_cluster(groups, pose, yaw, rows, "direction")
            record.update({
                f"{name}_pair_distance": pair_dist,
                f"{name}_pair_direction": pair_dir,
                f"{name}_correct_dist_m": d_correct,
                f"{name}_ahead_dist_m": d_ahead,
                f"{name}_clusters": len(groups),
                f"{name}_distance_status": dist_status,
                f"{name}_distance_deg": dist_deg,
                f"{name}_distance_match_travel": matches(dist_deg, travel_deg) if dist_status == "chosen" else dist_status,
                f"{name}_distance_match_straight": matches(dist_deg, 0.0) if dist_status == "chosen" else dist_status,
                f"{name}_direction_status": dir_status,
                f"{name}_direction_deg": dir_deg,
                f"{name}_direction_match_travel": matches(dir_deg, travel_deg) if dir_status == "chosen" else dir_status,
                f"{name}_direction_match_straight": matches(dir_deg, 0.0) if dir_status == "chosen" else dir_status,
                f"{name}_direction_support": dir_n,
            })
        # current_index kept only to show the route-level index was not the cutoff.
        record["route_first_index"] = current_index
        records.append(record)

    fields = list(records[0].keys())
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    def count(key, value):
        return sum(1 for row in records if row[key] == value)

    print(json.dumps({
        "n": len(records),
        "label_gap_gt_45": sum(1 for row in records if row["bearing_travel_gap_deg"] != "" and float(row["bearing_travel_gap_deg"]) > 45),
        "full_pair_distance_correct": count("full_pair_distance", "correct"),
        "full_pair_distance_straight": count("full_pair_distance", "straight"),
        "full_pair_distance_tie": count("full_pair_distance", "tie"),
        "full_pair_direction_correct": count("full_pair_direction", "correct"),
        "full_pair_direction_straight": count("full_pair_direction", "straight"),
        "full_pair_direction_tie": count("full_pair_direction", "tie"),
        "full_pair_direction_abstain": count("full_pair_direction", "abstain"),
        "ordered_pair_distance_correct": count("ordered_pair_distance", "correct"),
        "ordered_pair_direction_correct": count("ordered_pair_direction", "correct"),
        "ordered_routes_zero": count("ordered_routes", 0) if False else sum(1 for row in records if int(row["ordered_routes"]) == 0),
        "full_distance_match": count("full_distance_match_travel", "yes"),
        "full_distance_straight": count("full_distance_match_straight", "yes"),
        "full_direction_match": count("full_direction_match_travel", "yes"),
        "full_direction_straight": count("full_direction_match_straight", "yes"),
        "full_direction_abstain": count("full_direction_status", "abstain"),
        "ordered_distance_match": count("ordered_distance_match_travel", "yes"),
        "ordered_direction_match": count("ordered_direction_match_travel", "yes"),
        "ordered_success_direction_match": count("ordered_success_direction_match_travel", "yes"),
    }, indent=2))


if __name__ == "__main__":
    main()
