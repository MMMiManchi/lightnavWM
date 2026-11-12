#!/usr/bin/env python3
"""Align the 75 turn-verb candidate routes with the shared reference path and the Go2 geodesic.

Run with the Habitat 3.9 interpreter. Does not roll out a policy.

    /home/szgx/miniconda3/envs/habitat/bin/python \
        experiments/instruction_space_audit/code_route_conflicts.py
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from habitat_sim import PathFinder, ShortestPath

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/instruction_space_audit"
DATASET = ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
NAVMESH = ROOT / "output/r2r_probe/error_screening_full_episode_20261021/navmesh_go2w"
CANDIDATES = OUT / "turn_command_conflicts.csv"
THRESHOLDS = (45.0, 30.0)

CMD = re.compile(
    r"\b(?:turn(?:\s+(?:to\s+)?(?:your\s+)?)?|take a|make a|hang a|veer|go)\s+"
    r"(?:a\s+)?(?:hard\s+|slight(?:ly)?\s+)?(left|right)\b",
    re.I,
)


def wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def yaw_of(vec: np.ndarray) -> float:
    return math.atan2(-float(vec[0]), -float(vec[2]))


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    qv = np.array([x, y, z], dtype=float)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def start_yaw_deg(episode: dict) -> float:
    forward = quat_rotate(np.array(episode["start_rotation"], dtype=float), np.array([0.0, 0.0, -1.0]))
    forward[1] = 0.0
    forward /= np.linalg.norm(forward)
    return math.degrees(yaw_of(forward))


def commands(text: str) -> tuple[str, ...]:
    return tuple(match.group(1).lower() for match in CMD.finditer(text))


def turn_sequence(points: list[np.ndarray], yaw0: float, threshold: float) -> list[str]:
    headings: list[float] = []
    for start, end in zip(points, points[1:]):
        delta = end - start
        delta[1] = 0.0
        length = float(np.linalg.norm(delta))
        if length < 0.25:
            continue
        headings.append(math.degrees(yaw_of(delta / length)))
    sequence: list[str] = []
    previous = yaw0
    accumulated = 0.0
    for heading in headings:
        accumulated += wrap_deg(heading - previous)
        previous = heading
        if abs(accumulated) >= threshold:
            sequence.append("left" if accumulated > 0 else "right")
            accumulated = 0.0
    return sequence


def is_subsequence(small: tuple[str, ...] | list[str], big: list[str]) -> bool:
    index = 0
    for item in big:
        if index < len(small) and small[index] == item:
            index += 1
    return index == len(small)


def align(instruction: tuple[str, ...], path: list[str]) -> str:
    if not instruction:
        return "no_explicit_turn"
    if not path:
        return "path_below_threshold"
    if list(instruction) == path:
        return "exact"
    if is_subsequence(instruction, path):
        return "subset"
    if is_subsequence(path, instruction):
        return "superset"
    return "sign_conflict"


def rollup(labels: list[str]) -> str:
    explicit = [label for label in labels if label != "no_explicit_turn"]
    if not explicit:
        return "no_explicit_turn"
    if "path_below_threshold" in explicit:
        return "path_below_threshold"
    conflicts = [label for label in explicit if label == "sign_conflict"]
    compatible = [label for label in explicit if label in {"exact", "subset", "superset"}]
    if conflicts and compatible:
        return "mixed"
    if conflicts:
        return "all_sign_conflict"
    return "wording_difference"


def outcome_pattern(success: list[int], labels: list[str]) -> str:
    conflict = [ok for ok, label in zip(success, labels) if label == "sign_conflict"]
    compatible = [ok for ok, label in zip(success, labels) if label in {"exact", "subset", "superset"}]
    if not conflict or not compatible:
        return "not_a_mixed_contrast"
    if all(ok == 0 for ok in conflict) and all(ok == 1 for ok in compatible):
        return "conflict_fails_match_succeeds"
    if all(ok == 1 for ok in conflict) and all(ok == 0 for ok in compatible):
        return "conflict_succeeds_match_fails"
    if len(set(conflict + compatible)) == 1:
        return "alignment_does_not_split_success"
    return "partial_split"


def load_episodes() -> dict[tuple[str, str], dict]:
    with gzip.open(DATASET, "rt", encoding="utf-8") as handle:
        episodes = json.load(handle)["episodes"]
    found = {}
    for episode in episodes:
        scene = episode["scene_id"].split("/")[-2]
        found[(scene, str(episode["episode_id"]))] = episode
    return found


def load_finders() -> dict[str, PathFinder]:
    finders = {}
    for path in sorted(NAVMESH.glob("*.navmesh")):
        finder = PathFinder()
        if not finder.load_nav_mesh(str(path)):
            raise RuntimeError(f"cannot load {path}")
        finders[path.stem] = finder
    return finders


def geodesic_points(finder: PathFinder, episode: dict) -> list[np.ndarray] | None:
    query = ShortestPath()
    query.requested_start = np.asarray(episode["start_position"], dtype=np.float32)
    query.requested_end = np.asarray(episode["goals"][0]["position"], dtype=np.float32)
    if not finder.find_path(query) or not query.points:
        return None
    return [np.asarray(point, dtype=np.float64) for point in query.points]


def main() -> None:
    episodes = load_episodes()
    finders = load_finders()
    candidates = list(csv.DictReader(CANDIDATES.open()))
    route_rows = []
    instruction_rows = []
    for route in candidates:
        scene = route["scene"]
        members = []
        for index in (1, 2, 3):
            episode = episodes[(scene, route[f"episode_id_{index}"])]
            members.append((index, episode, commands(route[f"instruction_{index}"])))
        yaw0 = start_yaw_deg(members[0][1])
        reference = [np.asarray(point, dtype=np.float64) for point in members[0][1]["reference_path"]]
        geodesic = geodesic_points(finders[scene], members[0][1])
        sequences = {
            "reference": {threshold: turn_sequence(reference, yaw0, threshold) for threshold in THRESHOLDS},
            "geodesic": {
                threshold: [] if geodesic is None else turn_sequence(geodesic, yaw0, threshold)
                for threshold in THRESHOLDS
            },
        }
        row = {
            "route_group_id": route["route_group_id"],
            "scene": scene,
            "trajectory_id": route["trajectory_id"],
            "stratum": route["group_outcome"],
            "go2_path": "no_path" if geodesic is None else "finite",
            "prior_go2_path_flag": route["go2_path"],
            "in_manual_review": route["in_manual_review"],
            "visual_review": "not_reviewed",
            "A_explicit_turn_text_conflict": "candidate",
            "B_landmark_target": "not_reviewed",
            "C_relative_direction": "not_reviewed",
            "D_order_relation": "not_reviewed",
            "semantic_conflict_confirmed": "unknown",
            "wording_difference_only": "unknown",
            "model_failure_explained_by_conflict": "unknown",
        }
        for source in ("reference", "geodesic"):
            for threshold in THRESHOLDS:
                key = f"{source}_turns_{int(threshold)}"
                row[key] = " ".join(sequences[source][threshold])
        for threshold in THRESHOLDS:
            labels = {
                "reference": [align(item[2], sequences["reference"][threshold]) for item in members],
                "geodesic": [align(item[2], sequences["geodesic"][threshold]) for item in members],
            }
            success = [int(route[f"success_{index}"]) for index, _, _ in members]
            row[f"vs_reference_{int(threshold)}"] = rollup(labels["reference"])
            row[f"vs_geodesic_{int(threshold)}"] = "no_path" if geodesic is None else rollup(labels["geodesic"])
            row[f"reference_pattern_{int(threshold)}"] = outcome_pattern(success, labels["reference"])
            row[f"geodesic_pattern_{int(threshold)}"] = (
                "no_path" if geodesic is None else outcome_pattern(success, labels["geodesic"])
            )
            row[f"path_sources_agree_{int(threshold)}"] = int(
                row[f"vs_reference_{int(threshold)}"] == row[f"vs_geodesic_{int(threshold)}"]
            )
            if threshold == 45.0:
                for index, episode, turns in members:
                    label_index = index - 1
                    instruction_rows.append(
                        {
                            "route_group_id": route["route_group_id"],
                            "scene": scene,
                            "trajectory_id": route["trajectory_id"],
                            "stratum": route["group_outcome"],
                            "episode_id": route[f"episode_id_{index}"],
                            "success": route[f"success_{index}"],
                            "turn_commands": " ".join(turns),
                            "vs_reference_45": labels["reference"][label_index],
                            "vs_geodesic_45": "no_path" if geodesic is None else labels["geodesic"][label_index],
                            "instruction": route[f"instruction_{index}"],
                        }
                    )
                    row[f"episode_id_{index}"] = route[f"episode_id_{index}"]
                    row[f"success_{index}"] = route[f"success_{index}"]
                    row[f"turns_{index}"] = " ".join(turns)
                    row[f"vs_reference_45_{index}"] = labels["reference"][label_index]
                    row[f"vs_geodesic_45_{index}"] = "no_path" if geodesic is None else labels["geodesic"][label_index]
        route_rows.append(row)

    route_fields = list(route_rows[0].keys())
    with (OUT / "route_level_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=route_fields)
        writer.writeheader()
        writer.writerows(route_rows)
    with (OUT / "route_instruction_alignment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(instruction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(instruction_rows)

    finite = [row for row in route_rows if row["go2_path"] == "finite"]
    print("routes", len(route_rows), "finite", len(finite))
    print("vs geodesic 45", Counter(row["vs_geodesic_45"] for row in finite))
    print("vs reference 45", Counter(row["vs_reference_45"] for row in finite))
    print("vs geodesic 30", Counter(row["vs_geodesic_30"] for row in finite))
    print("sources agree 45", Counter(row["path_sources_agree_45"] for row in finite))
    print("geodesic pattern 45", Counter(row["geodesic_pattern_45"] for row in finite))
    print("by stratum geodesic 45")
    for stratum in ("partial", "all_success", "all_fail"):
        subset = [row for row in finite if row["stratum"] == stratum]
        print(" ", stratum, len(subset), Counter(row["vs_geodesic_45"] for row in subset))
        print("   pattern", Counter(row["geodesic_pattern_45"] for row in subset))


if __name__ == "__main__":
    main()
