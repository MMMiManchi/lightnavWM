"""CPU-only candidate mining in Habitat Python 3.9, preserving existing roles."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.branch_curation import load_floor_polygons  # noqa: E402
from experiments.world_model.branch_source_mining import RegionIndex, propose_pose  # noqa: E402


def main():
    import habitat_sim

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--legacy", type=Path, default=REPO / "output/world_model_training/event_latent_v1_20261104_01")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-scene", type=int, default=12)
    args = parser.parse_args()
    roles = defaultdict(set)
    for line in (args.legacy / "pairs.jsonl").read_text().splitlines():
        row = json.loads(line)
        roles[row["scene"]].add(row["partition"])
    if any(len(value) != 1 or not value <= {"train_fit", "train_dev"} for value in roles.values()):
        raise ValueError("legacy scene roles overlap or include a sealed role")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"worker_{args.worker:02d}.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    counts = []
    with destination.open("x") as stream:
        for scene in sorted(roles)[args.worker::args.workers]:
            role = "fit" if roles[scene] == {"train_fit"} else "development"
            root = REPO / "data/scene_datasets/mp3d" / scene
            house = root / f"{scene}.house"
            connectivity = REPO / "data/connectivity" / f"{scene}_connectivity.json"
            if not all(p.exists() for p in (house, connectivity, root / f"{scene}.navmesh")):
                counts.append({"scene": scene, "status": "missing_local_assets"})
                continue
            polygons = load_floor_polygons(house)
            floors = {poly.region_id: poly.floor_height for poly in polygons}
            panorama_regions = {}
            for line in house.read_text().splitlines():
                fields = line.split()
                if fields and fields[0] == "P" and len(fields[1]) == 32:
                    panorama_regions[fields[1]] = int(fields[3])
            finder = habitat_sim.PathFinder()
            if not finder.load_nav_mesh(str(root / f"{scene}.navmesh")):
                raise RuntimeError("local navmesh could not load")
            index = RegionIndex(polygons)
            proposed = []
            for node in json.loads(connectivity.read_text()):
                if not node.get("included"):
                    continue
                region = panorama_regions.get(node["image_id"])
                if region not in floors:
                    continue
                pose = node["pose"]
                desired = np.array([pose[3], floors[region] + 0.1, -pose[7]], dtype=np.float32)
                start = np.asarray(finder.snap_point(desired), dtype=np.float32)
                if not np.isfinite(start).all() or np.linalg.norm((start - desired)[[0, 2]]) > 0.25 or abs(start[1] - desired[1]) > 0.5:
                    continue
                candidate = propose_pose(finder, index, start, node["image_id"], scene, role)
                if candidate is not None:
                    proposed.append(candidate)
            selected = []
            for candidate in sorted(proposed, key=lambda c: c["ranking_score"]):
                if any(np.linalg.norm((np.array(candidate["position"]) - other["position"])[[0, 2]]) < 0.75
                       and abs(candidate["position"][1] - other["position"][1]) < 0.5 for other in selected):
                    continue
                selected.append(candidate)
                if len(selected) >= args.per_scene:
                    break
            for candidate in selected:
                stream.write(json.dumps(candidate) + "\n")
            stream.flush()
            counts.append({"scene": scene, "role": role, "proposed": len(proposed), "selected": len(selected)})
            print(json.dumps(counts[-1]), flush=True)
    (args.output / f"worker_{args.worker:02d}.meta.json").write_text(json.dumps({"scenes": counts,
        "selected_by_role": dict(Counter(json.loads(s)["role"] for s in destination.read_text().splitlines())),
        "sealed_houses_opened": 0}, indent=2))


if __name__ == "__main__":
    main()
