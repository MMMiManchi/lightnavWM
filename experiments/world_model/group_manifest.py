"""Versioned train-split allowlist and scene/route groups.

``--episodes N`` is not a sampler. A pilot list is an explicit subset of the
official train JSON, checked against that file's bytes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from experiments.world_model import ALLOWLIST_VERSION, DATASET_NAME, GROUP_VERSION
from experiments.world_model.jsonutil import dumps, sha256_bytes, sha256_file
from experiments.world_model.latent_data import scene_partition

LEGACY_DIR_NAMES = ("r2r_probe", "go2w_expert")
GEOMETRY_DECIMALS = 4


def refuse_legacy_path(path: Path) -> None:
    parts = set(Path(path).resolve().parts)
    hit = parts.intersection(LEGACY_DIR_NAMES)
    if hit:
        raise RuntimeError(f"refusing to use legacy output {sorted(hit)}: {path}")


def canonical_scene(scene_id: str) -> str:
    text = str(scene_id).rstrip("/")
    parent = Path(text).parent.name
    if not parent or parent in (".", "/"):
        raise ValueError(f"cannot derive a scene id from {scene_id!r}")
    return parent


def _round_tree(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, GEOMETRY_DECIMALS)
    if isinstance(value, list):
        return [_round_tree(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _round_tree(item) for key, item in value.items()}
    if isinstance(value, int):
        return value
    return value


def geometry_hash(scene: str, start_position: Any, start_rotation: Any, reference_path: Any, goals: Any) -> str:
    payload = {
        "version": f"geom-round-{GEOMETRY_DECIMALS}",
        "scene": scene,
        "start_position": _round_tree(start_position),
        "start_rotation_json": _round_tree(start_rotation),
        "reference_path": _round_tree(reference_path),
        "goals": _round_tree(goals),
        "rotation_order": "unconfirmed_json_array",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def split_json_path(dataset_root: Path, split: str) -> Path:
    return dataset_root / split / f"{split}.json.gz"


def load_split_rows(dataset_root: Path, split: str) -> tuple[Path, str, list[dict[str, Any]]]:
    path = split_json_path(dataset_root, split)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    payload = json.loads(gzip.decompress(raw).decode("utf-8"))
    rows = []
    seen: set[tuple[str, str]] = set()
    for episode in payload["episodes"]:
        scene = canonical_scene(str(episode["scene_id"]))
        episode_id = str(episode["episode_id"])
        key = (scene, episode_id)
        if key in seen:
            raise RuntimeError(f"duplicate episode in {split}: {key}")
        seen.add(key)
        instruction = episode.get("instruction") or {}
        text = instruction.get("instruction_text")
        if not isinstance(text, str):
            raise RuntimeError(f"{key} has no instruction_text")
        native_instruction = instruction.get("instruction_id")
        trajectory = episode.get("trajectory_id")
        rows.append(
            {
                "scene": scene,
                "scene_id_raw": str(episode["scene_id"]),
                "episode_id": episode_id,
                "trajectory_id": None if trajectory is None else str(trajectory),
                "instruction_text": text,
                "instruction_id_native": None if not native_instruction else str(native_instruction),
                "start_position": episode.get("start_position"),
                "start_rotation_json": episode.get("start_rotation"),
                "goals": episode.get("goals"),
                "reference_path": episode.get("reference_path"),
                "source_geodesic_m": (episode.get("info") or {}).get("geodesic_distance"),
                "source_split": split,
            }
        )
    return path, digest, rows


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        self.parent.setdefault(key, key)

    def find(self, key: str) -> str:
        self.add(key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _partition(group_id: str) -> str:
    last = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[-2:], 16)
    return "train_dev" if last % 10 == 0 else "train_fit"


def assign_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by scene+trajectory, then merge identical rounded geometry."""
    nodes: list[dict[str, Any]] = []
    unions = _UnionFind()
    by_route: dict[str, str] = {}
    by_geom: dict[str, str] = {}
    for row in rows:
        route_native = row["trajectory_id"] or f"episode:{row['episode_id']}"
        member = f"{row['scene']}:{route_native}"
        geom = geometry_hash(
            row["scene"],
            row["start_position"],
            row["start_rotation_json"],
            row["reference_path"],
            row["goals"],
        )
        unions.add(member)
        if member in by_route:
            unions.union(member, by_route[member])
        else:
            by_route[member] = member
        if geom in by_geom:
            unions.union(member, by_geom[geom])
        else:
            by_geom[geom] = member
        nodes.append({**row, "member": member, "route_geometry_hash": geom})

    grouped = []
    for node in nodes:
        group_id = unions.find(node["member"])
        text_hash = hashlib.sha256(node["instruction_text"].encode("utf-8")).hexdigest()
        episode_uid = f"{DATASET_NAME}|{node['source_split']}|{node['scene']}|{node['episode_id']}"
        grouped.append(
            {
                **node,
                "episode_uid": episode_uid,
                "route_group_id": group_id,
                "route_id_native": node["trajectory_id"],
                "instruction_id_derived": f"{episode_uid}|{text_hash[:16]}",
                "partition": _partition(group_id),
            }
        )
    partitions: dict[str, set[str]] = {}
    for node in grouped:
        partitions.setdefault(node["route_group_id"], set()).add(node["partition"])
    conflicts = [key for key, parts in partitions.items() if len(parts) != 1]
    if conflicts:
        raise RuntimeError(f"group partition conflict: {conflicts[:5]}")
    return grouped


def index_by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["scene"], row["episode_id"]): row for row in rows}


def select_pilot(rows: list[dict[str, Any]], count: int, min_scenes: int) -> list[dict[str, Any]]:
    if count < min_scenes:
        raise ValueError("pilot count must be at least the scene minimum")
    chosen: list[dict[str, Any]] = []
    scenes: list[str] = []
    for row in rows:
        if row["partition"] != "train_fit":
            continue
        if row["scene"] in scenes:
            continue
        chosen.append(row)
        scenes.append(row["scene"])
        if len(scenes) >= min_scenes:
            break
    have = {(row["scene"], row["episode_id"]) for row in chosen}
    if len(chosen) < count:
        for row in rows:
            if row["partition"] != "train_fit" or row["scene"] not in scenes:
                continue
            key = (row["scene"], row["episode_id"])
            if key in have:
                continue
            chosen.append(row)
            have.add(key)
            if len(chosen) >= count:
                break
    if len(chosen) < count or len({row["scene"] for row in chosen}) < min_scenes:
        raise RuntimeError(
            f"train_fit does not contain {count} episodes across {min_scenes} scenes"
        )
    return chosen[:count]


def select_scene_partition_pilot(
    rows: list[dict[str, Any]],
    count: int,
    min_scenes: int,
    target_partition: str,
    excluded_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Choose a bounded pilot from the scene-disjoint world-model partition."""
    if target_partition not in ("train_fit", "train_dev"):
        raise ValueError(f"unsupported scene partition {target_partition!r}")
    if count < min_scenes:
        raise ValueError("pilot count must be at least the scene minimum")
    excluded_keys = excluded_keys or set()
    eligible = [
        row for row in rows
        if scene_partition(str(row["scene"])) == target_partition
        and (str(row["scene"]), str(row["episode_id"])) not in excluded_keys
    ]
    chosen: list[dict[str, Any]] = []
    scenes: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in eligible:
        if row["scene"] in scenes:
            continue
        chosen.append({**row, "partition": target_partition})
        scenes.append(row["scene"])
        seen.add((row["scene"], row["episode_id"]))
        if len(scenes) == min_scenes:
            break
    for row in eligible:
        key = (row["scene"], row["episode_id"])
        if key in seen:
            continue
        chosen.append({**row, "partition": target_partition})
        seen.add(key)
        if len(chosen) == count:
            break
    if len(chosen) < count or len(scenes) < min_scenes:
        raise RuntimeError(
            f"{target_partition} has only {len(chosen)} episodes across {len(scenes)} selected scenes; "
            f"requested {count} episodes and {min_scenes} scenes"
        )
    return chosen


def campaign_episode_keys(manifest_paths: list[Path]) -> set[tuple[str, str]]:
    """Load episode keys from immutable, validated train_fit campaign runs."""
    excluded: set[tuple[str, str]] = set()
    for manifest_path in manifest_paths:
        campaign = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if campaign.get("schema_version") != "wm-campaign/0.1":
            raise RuntimeError(f"unsupported campaign manifest: {manifest_path}")
        if campaign.get("scope") != "train_fit" or campaign.get("source_split") != "train":
            raise RuntimeError(f"campaign is not train_fit: {manifest_path}")
        for item in campaign.get("runs") or []:
            run_dir = Path(item["run_dir"])
            run_manifest_path = run_dir / "manifest.json"
            report_path = run_dir / "validation_report.json"
            if sha256_file(run_manifest_path) != item.get("manifest_sha256"):
                raise RuntimeError(f"registered run manifest changed: {run_manifest_path}")
            if sha256_file(report_path) != item.get("validation_report_sha256"):
                raise RuntimeError(f"registered validation report changed: {report_path}")
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                (run_manifest.get("source") or {}).get("split") != "train"
                or run_manifest.get("diagnostic_only") is True
                or run_manifest.get("collection_partition") != "train_fit"
                or report.get("hard_error_count") != 0
            ):
                raise RuntimeError(f"registered run is not a valid train_fit run: {run_dir}")
            excluded.update((str(scene), str(episode_id)) for scene, episode_id in run_manifest.get("allowlist_keys") or [])
    return excluded


def build_allowlist(
    rows: list[dict[str, Any]],
    chosen: list[dict[str, Any]],
    *,
    dataset_json: Path,
    dataset_sha256: str,
) -> dict[str, Any]:
    episodes = []
    seen: set[tuple[str, str]] = set()
    known = index_by_key(rows)
    for row in chosen:
        key = (row["scene"], row["episode_id"])
        if key in seen:
            raise RuntimeError(f"duplicate allowlist id {key}")
        seen.add(key)
        source = known.get(key)
        if source is None or source["source_split"] != "train":
            raise RuntimeError(f"allowlist id is not in the train source: {key}")
        episodes.append(
            {
                "scene": row["scene"],
                "episode_id": row["episode_id"],
                "route_id_native": row["route_id_native"],
                "partition": row["partition"],
                "episode_uid": row["episode_uid"],
                "route_group_id": row["route_group_id"],
            }
        )
    body = {
        "schema_version": ALLOWLIST_VERSION,
        "source_dataset": DATASET_NAME,
        "source_split": "train",
        "dataset_json": str(dataset_json),
        "dataset_sha256": dataset_sha256,
        "episodes": episodes,
    }
    encoded = dumps(body["episodes"]).encode()
    body["allowlist_sha256"] = sha256_bytes(encoded)
    return body


def check_allowlist_against_train(
    allowlist: dict[str, Any],
    train_rows: list[dict[str, Any]],
    *,
    dataset_sha256: str,
    forbidden_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if allowlist.get("schema_version") != ALLOWLIST_VERSION:
        raise RuntimeError(f"allowlist schema {allowlist.get('schema_version')!r} is not {ALLOWLIST_VERSION}")
    if allowlist.get("source_split") != "train":
        raise RuntimeError(f"allowlist split is {allowlist.get('source_split')!r}, only train is accepted")
    if allowlist.get("dataset_sha256") != dataset_sha256:
        raise RuntimeError("allowlist dataset hash does not match the train json.gz bytes")
    episodes = allowlist.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError("allowlist has no episodes")
    known = index_by_key(train_rows)
    ordered = []
    seen: set[tuple[str, str]] = set()
    for item in episodes:
        key = (str(item["scene"]), str(item["episode_id"]))
        if key in seen:
            raise RuntimeError(f"duplicate allowlist id {key}")
        seen.add(key)
        if forbidden_keys and key in forbidden_keys:
            raise RuntimeError(f"val split id leaked into the train allowlist: {key}")
        source = known.get(key)
        if source is None:
            raise RuntimeError(f"unknown train episode {key}")
        if source["source_split"] != "train":
            raise RuntimeError(f"split contamination at {key}: {source['source_split']}")
        if item.get("partition") in ("eval", "val_seen", "val_unseen"):
            raise RuntimeError(f"partition {item.get('partition')!r} is not a train partition")
        ordered.append(source | {"partition": item.get("partition", source["partition"])})
    encoded = dumps(episodes).encode()
    if sha256_bytes(encoded) != allowlist.get("allowlist_sha256"):
        raise RuntimeError("allowlist_sha256 does not match the episode list")
    return ordered


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenes = {row["scene"] for row in rows}
    groups = {row["route_group_id"] for row in rows}
    partitions: dict[str, int] = {}
    for row in rows:
        partitions[row["partition"]] = partitions.get(row["partition"], 0) + 1
    return {
        "episodes": len(rows),
        "scenes": len(scenes),
        "route_groups": len(groups),
        "partitions": partitions,
        "group_version": GROUP_VERSION,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a train-only world-model allowlist")
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--dataset-root",
        default="data/datasets/R2R_VLNCE_v1-3_preprocessed",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--min-scenes", type=int, default=2)
    parser.add_argument("--scene-partition", choices=("train_fit", "train_dev"), default="train_fit")
    parser.add_argument(
        "--exclude-campaign",
        action="append",
        default=[],
        help="exclude episodes registered in a prior train_fit campaign; repeatable",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.split != "train":
        raise SystemExit(f"refusing split {args.split!r}; world-model collection is train-only")
    root = Path(args.dataset_root)
    path, digest, rows = load_split_rows(root, "train")
    grouped = assign_groups(rows)
    summary = summarize(grouped)
    print(dumps({"dataset_json": str(path), "dataset_sha256": digest, **summary}))
    if args.dry_run and not args.pilot:
        return 0
    if not args.pilot:
        raise SystemExit("pass --dry-run or --pilot N --out PATH")
    if not args.out:
        raise SystemExit("--out is required when writing an allowlist")
    out = Path(args.out)
    refuse_legacy_path(out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    excluded_keys = campaign_episode_keys([Path(value) for value in args.exclude_campaign])
    if args.scene_partition == "train_fit":
        chosen = select_scene_partition_pilot(grouped, args.pilot, args.min_scenes, "train_fit", excluded_keys)
    else:
        chosen = select_scene_partition_pilot(grouped, args.pilot, args.min_scenes, "train_dev", excluded_keys)
    payload = build_allowlist(grouped, chosen, dataset_json=path, dataset_sha256=digest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(payload) + "\n", encoding="utf-8")
    print(dumps({"wrote": str(out), "episodes": len(payload["episodes"]), "allowlist_sha256": payload["allowlist_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
