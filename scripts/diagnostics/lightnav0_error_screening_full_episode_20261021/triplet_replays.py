#!/usr/bin/env python3
"""Replays for instruction triples. Reads saved traces. Does not roll out."""

from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import habitat_sim
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("make_replays", HERE / "make_replays.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

OUT = ROOT / "output/r2r_probe/instruction_triplet_20261021"
NAVMESH = ROOT / "output/r2r_probe/error_screening_full_episode_20261021/navmesh_go2w"
OLD_STEPS = ROOT / "output/r2r_probe/trace_20261020/steps.jsonl"
NEW_STEPS = OUT / "steps.jsonl"
EPISODES = OUT / "episodes.jsonl"
REPLAY_DIR = OUT / "replay"

# The long instruction was already traced on 2026-10-20. Siblings are in NEW_STEPS.
HAVE = {
    ("2azQ1b91cZZ", "116"): "long fail",
    ("2azQ1b91cZZ", "214"): "long fail",
    ("QUCTc6BB5sX", "142"): "long fail",
    ("TbHJrupSAjP", "174"): "long fail",
}
WANT = [
    ("2azQ1b91cZZ", "115"),
    ("2azQ1b91cZZ", "116"),
    ("2azQ1b91cZZ", "117"),
    ("2azQ1b91cZZ", "214"),
    ("2azQ1b91cZZ", "215"),
    ("2azQ1b91cZZ", "216"),
    ("QUCTc6BB5sX", "142"),
    ("QUCTc6BB5sX", "143"),
    ("QUCTc6BB5sX", "144"),
    ("TbHJrupSAjP", "172"),
    ("TbHJrupSAjP", "173"),
    ("TbHJrupSAjP", "174"),
]


def load_steps(path: Path, keys: set[tuple[str, str]]) -> dict[tuple[str, str], list[dict]]:
    groups = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["scene"], str(row["episode_id"]))
            if key in keys:
                groups[key].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["step"]))
    return groups


def load_results() -> dict[tuple[str, str], dict]:
    found = {}
    if not EPISODES.exists():
        return found
    with EPISODES.open() as handle:
        for line in handle:
            row = json.loads(line)
            found[(row["scene"], str(row["episode_id"]))] = row
    return found


def as_timeline(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        yaw = row.get("commanded_yaw_rad")
        out.append(
            {
                "step": str(row["step"]),
                "instruction": row.get("instruction") or "",
                "frame_file": row["frame_file"],
                "geodesic_before_m": row.get("geodesic_before_m") or 0.0,
                "euclidean_before_m": row.get("euclidean_horizontal_before_m") or 0.0,
                "commanded_forward_m": row.get("commanded_forward_m") or 0.0,
                "commanded_yaw_deg": 0.0 if yaw is None else math.degrees(float(yaw)),
                "events": "",
            }
        )
    return out


def finder(scene: str, cache: dict):
    if scene not in cache:
        pathfinder = habitat_sim.PathFinder()
        mesh = NAVMESH / f"{scene}.navmesh"
        if not pathfinder.load_nav_mesh(str(mesh)):
            raise RuntimeError(f"cannot load {mesh}")
        cache[scene] = pathfinder
    return cache[scene]


def paths_for(pathfinder, rows: list[dict], goal: np.ndarray) -> dict[int, list[tuple[float, float, float]]]:
    goal32 = np.asarray(goal, dtype=np.float32)
    found = {}
    for row in rows:
        origin = np.asarray(row["pose_xyz_before"], dtype=np.float32)
        query = habitat_sim.ShortestPath()
        query.requested_start = origin
        query.requested_end = goal32
        if not pathfinder.find_path(query) or len(query.points) < 2:
            snapped = pathfinder.snap_point(origin)
            query.requested_start = np.asarray(snapped, dtype=np.float32)
            if not pathfinder.find_path(query) or len(query.points) < 2:
                found[int(row["step"])] = []
                continue
        found[int(row["step"])] = [tuple(float(v) for v in point) for point in query.points]
    return found


def main() -> None:
    keys = set(WANT)
    steps = load_steps(OLD_STEPS, keys)
    if NEW_STEPS.exists():
        fresh = load_steps(NEW_STEPS, keys)
        steps.update(fresh)
    results = load_results()
    goals = replay.load_goals()
    meshes = {}
    finders = {}
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for scene, eid in WANT:
        rows = steps.get((scene, eid))
        if not rows:
            print("missing", scene, eid)
            continue
        timeline = {(scene, eid): as_timeline(rows)}
        trace = {(scene, eid): {int(row["step"]): row for row in rows}}
        if scene not in meshes:
            meshes[scene] = replay.load_mesh(scene)
        goal = goals[(scene, eid)]
        paths = {(scene, eid): paths_for(finder(scene, finders), rows, goal)}
        result = results.get((scene, eid), {})
        success = result.get("success")
        final = result.get("final_distance_m")
        if success is None:
            final = float(rows[-1].get("geodesic_after_m") or rows[-1].get("geodesic_before_m") or 0)
            success = final < 3.0 and bool(rows[-1].get("model_stop"))
        tag = "OK" if success else "FAIL"
        words = len(rows[0].get("instruction", "").split())
        label = f"{eid}  {tag}  {words} words  final {float(final):.2f} m"
        # Put the outcome on the first instruction line. The renderer wraps it.
        timeline[(scene, eid)][0]["instruction"] = label + "   " + timeline[(scene, eid)][0]["instruction"]
        dest = REPLAY_DIR / f"{scene}_{eid}.mp4"
        frames = replay.render_episode(
            scene, eid, timeline, trace, paths, goal, meshes[scene], dest, None
        )
        index.append(
            {
                "scene": scene,
                "episode_id": eid,
                "success": int(bool(success)),
                "words": words,
                "final_m": None if final is None else round(float(final), 2),
                "frames": frames,
                "video": str(dest.relative_to(ROOT)),
            }
        )
        print(label, frames)
    with (REPLAY_DIR / "index.csv").open("w") as handle:
        handle.write("scene,episode_id,success,words,final_m,frames,video\n")
        for row in index:
            handle.write(
                f"{row['scene']},{row['episode_id']},{row['success']},{row['words']},"
                f"{row['final_m']},{row['frames']},{row['video']}\n"
            )


if __name__ == "__main__":
    main()
