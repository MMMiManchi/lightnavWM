"""Scene-level train/dev inventory for ``wm-scene-split/0.1``.

The inventory is built only from the official train JSON. It does not collect
transitions and does not copy val_seen, val_unseen, or test episodes into either
side. ``require_formal_split`` remains the acceptance gate. A route-hash split
that puts one scene on both sides is reported and is not used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.world_model.group_manifest import assign_groups, load_split_rows
from experiments.world_model.jsonutil import dumps
from experiments.world_model.latent_data import (
    SCENE_SPLIT_VERSION,
    FormalSplitError,
    formal_report,
    scene_partition,
)

DEFAULT_ROOT = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed")
OTHER_SPLITS = ("val_seen", "val_unseen", "test")


def _episode_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_uid": row["episode_uid"],
        "scene": row["scene"],
        "episode_id": row["episode_id"],
        "route_group_id": row["route_group_id"],
        "source_split": row["source_split"],
    }


def build_train_scene_inventory(dataset_root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    path, digest, rows = load_split_rows(dataset_root, "train")
    grouped = assign_groups(rows)
    report = formal_report(_episode_view(row) for row in grouped)
    scenes: dict[str, dict[str, Any]] = {}
    route_sides: dict[str, set[str]] = defaultdict(set)
    for row in grouped:
        scene = str(row["scene"])
        side = scene_partition(scene)
        bucket = scenes.setdefault(
            scene,
            {
                "scene": scene,
                "scene_partition": side,
                "route_group_ids": set(),
                "n_episodes": 0,
            },
        )
        if bucket["scene_partition"] != side:
            raise FormalSplitError({"eligible": False, "reason": f"scene {scene} received two partitions"})
        bucket["route_group_ids"].add(str(row["route_group_id"]))
        bucket["n_episodes"] += 1
        route_sides[scene].add(str(row["partition"]))
    scene_rows = []
    for scene in sorted(scenes):
        bucket = scenes[scene]
        route_ids = sorted(bucket["route_group_ids"])
        scene_rows.append(
            {
                "scene": scene,
                "scene_partition": bucket["scene_partition"],
                "n_route_groups": len(route_ids),
                "n_episodes": bucket["n_episodes"],
                "route_group_ids": route_ids,
                "route_hash_partitions": sorted(route_sides[scene]),
            }
        )
    leaked_scenes = [row["scene"] for row in scene_rows if len(row["route_hash_partitions"]) > 1]
    train_scenes = {row["scene"] for row in scene_rows}
    train_keys = {(row["scene"], row["episode_id"]) for row in grouped}
    others = {}
    for split in OTHER_SPLITS:
        split_path = dataset_root / split / f"{split}.json.gz"
        if not split_path.is_file():
            others[split] = {"present": False}
            continue
        _, split_digest, split_rows = load_split_rows(dataset_root, split)
        split_scenes = {row["scene"] for row in split_rows}
        key_hits = []
        for row in split_rows:
            key = (row["scene"], row["episode_id"])
            if key not in train_keys:
                continue
            train_row = next(item for item in grouped if (item["scene"], item["episode_id"]) == key)
            key_hits.append(
                {
                    "scene": row["scene"],
                    "episode_id": row["episode_id"],
                    "same_trajectory_id": train_row["trajectory_id"] == row["trajectory_id"],
                    "same_instruction_text": train_row["instruction_text"] == row["instruction_text"],
                }
            )
        others[split] = {
            "present": True,
            "dataset_sha256": split_digest,
            "n_episodes": len(split_rows),
            "n_scenes": len(split_scenes),
            "scene_intersection_with_train": sorted(train_scenes & split_scenes),
            "episode_id_collisions": key_hits,
            "included_in_train_inventory": False,
        }
    counts = {key: report[key] for key in report if key not in ("episodes", "reason", "eligible", "scene_split_version")}
    return {
        "scene_split_version": SCENE_SPLIT_VERSION,
        "source_split": "train",
        "dataset_json": str(path),
        "dataset_sha256": digest,
        "eligible": report["eligible"],
        "reason": report["reason"],
        "counts": counts,
        "route_hash_scenes_on_both_sides": leaked_scenes,
        "other_splits": others,
        "pilot_run_excluded": "output/world_model_data/pilot_20261023_train3_r2",
        "collection_status": "not_collected",
        "scenes": scene_rows,
    }


def write_train_scene_inventory(out: Path, dataset_root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    destination = Path(out)
    if destination.exists():
        raise FormalSplitError({"eligible": False, "reason": f"refusing to overwrite {destination}"})
    inventory = build_train_scene_inventory(dataset_root)
    if not inventory["eligible"]:
        raise FormalSplitError({"eligible": False, "reason": inventory["reason"], "counts": inventory["counts"]})
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dumps(inventory) + "\n", encoding="utf-8")
    return inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the wm-scene-split/0.1 train inventory")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args(argv)
    if args.split != "train":
        raise SystemExit(f"refusing split {args.split!r}; the scene inventory reads only the train JSON")
    inventory = write_train_scene_inventory(args.out, args.dataset_root)
    print(
        dumps(
            {
                "wrote": str(args.out),
                "eligible": inventory["eligible"],
                "counts": inventory["counts"],
                "route_hash_scenes_on_both_sides": len(inventory["route_hash_scenes_on_both_sides"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
