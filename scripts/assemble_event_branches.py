"""Release audited v4 futures; keep censoring and physical returns separate."""

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world_model.branch_benchmark_entry import read_jsonl, write_json  # noqa: E402
from experiments.world_model.branch_curation import (  # noqa: E402
    CurationRules,
    branch_pair_summary,
    deduplicate_events,
    image_quality,
)
from experiments.world_model.jsonutil import sha256_file  # noqa: E402


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def spatial_clusters(origins):
    # Conservative connected components report nearby poses separately from
    # distinct physical choices; they are not a semantic topological map.
    remaining = list(origins)
    counts = Counter()
    while remaining:
        component = [remaining.pop()]
        expanded = True
        while expanded:
            expanded = False
            for row in remaining[:]:
                if any(row["scene"] == member["scene"] and row["role"] == member["role"]
                       and np.linalg.norm(np.array(row["position"]) - np.array(member["position"])) <= 1.5
                       for member in component):
                    component.append(row)
                    remaining.remove(row)
                    expanded = True
        counts[component[0]["role"]] += 1
    return dict(counts)


def build(source, output, supplements=()):
    output.mkdir(parents=True, exist_ok=False)
    tasks = read_jsonl(source / "tasks.jsonl")
    task_sources = {t["task_id"]: source for t in tasks}
    replacements = {}
    for supplement in supplements:
        added = read_jsonl(supplement / "tasks.jsonl")
        original = {t["task_id"]: t for t in tasks}
        ids = {t["task_id"] for t in added}
        origins_to_replace = {t["origin"]["origin_id"] for t in added}
        expected_ids = {t["task_id"] for t in tasks if t["origin"]["origin_id"] in origins_to_replace}
        if ids != expected_ids or not ids <= original.keys():
            raise ValueError("a controller correction must rerun every branch and repeat of each origin")
        for row in added:
            if row["origin"]["expected_rgb_sha256"] != original[row["task_id"]]["origin"]["expected_rgb_sha256"]:
                raise ValueError("controller correction changed the reviewed source")
            task_sources[row["task_id"]] = supplement
            replacements[row["task_id"]] = row
    tasks = [replacements.get(t["task_id"], t) for t in tasks]
    write_json(output / "selected_execution_sources.json", {
        "rule": "entire corrected origin replaces parent, regardless of outcome; no best-of selection",
        "tasks": {key: str(value) for key, value in task_sources.items()},
        "parent_raw_preserved": True})
    records, events, return_events, provenance = [], [], [], {}
    histories = defaultdict(set)
    for task in tasks:
        origin = task["origin"]
        if origin["role"] not in {"fit", "development"} or origin["scene"] == "Vvot9Ly1tCj":
            raise ValueError("sealed task refused")
        actual_source = task_sources[task["task_id"]]
        folder = actual_source / "executions" / task["task_id"]
        result_path = folder / "result.json"
        if not result_path.exists():
            raise ValueError(f"unexecuted task: {task['task_id']}")
        result = json.loads(result_path.read_text())
        records.append(result)
        provenance[str(result_path)] = sha256_file(result_path)
        commits = read_jsonl(folder / "committed.jsonl")
        if len(commits) != result["branch_steps"] or [r["index"] for r in commits] != list(range(1, len(commits) + 1)):
            raise ValueError("committed trajectory is not contiguous")
        if result["branch_steps"] > 1000:
            raise ValueError("authorized logical step ceiling exceeded")
        if not result["event_complete"]:
            continue
        if not result["source_restored"] or not result["checkpoint_reached"]:
            raise ValueError("future lacks same-source or chosen-entrance proof")
        history = read_jsonl(folder / "source_history.jsonl")
        refs = result["source_history_rgb"]
        if len(history) != len(refs) or len(refs) != 8 or history[-1]["name"] != "node":
            raise ValueError("causal source history is incomplete")
        if refs[-1]["sha256"] != origin["expected_rgb_sha256"]:
            raise ValueError("future is bound to a different reviewed source")
        histories[origin["origin_id"]].add(tuple(r["sha256"] for r in refs))
        for ref in refs:
            path = Path(ref["path"])
            if sha256_file(path) != ref["sha256"]:
                raise ValueError("source RGB changed")
            provenance[str(path)] = ref["sha256"]
        target_index = result["target_index"]
        frame = commits[target_index - 1]["frame"]
        target_path = Path(result["target_rgb"]["path"])
        if sha256_file(target_path) != result["target_rgb"]["sha256"] or frame["rgb_sha256"] != result["target_rgb"]["sha256"]:
            raise ValueError("actual future image hash changed")
        quality = image_quality(np.load(target_path, allow_pickle=False), CurationRules(maximum_black_fraction=0.20))
        if not quality["usable"]:
            raise ValueError("primary future failed registered image-quality rule")
        event = {"sample_id": task["task_id"], "role": origin["role"], "scene": origin["scene"],
            "origin_id": origin["origin_id"], "exit_id": task["exit"]["exit_id"], "repeat_index": task["repeat_index"],
            "inputs": {"history_rgb": refs, "history_valid": [True] * len(refs),
                "requested_skill": {"name": "follow_visible_exit_until_next_event", "policy_id": result["policy_id"],
                                    "bearing_left_rad": task["exit"]["bearing_left_rad"], "side": task["exit"]["side"]}},
            "target_kind": result["target_kind"],
            "target": {"rgb": {**result["target_rgb"], "quality": quality}, "horizon_s": result["horizon_s"],
                       "horizon_steps": target_index},
            "offline_labels": {"source_region": result["source_region"], "target_region": result["target_region"],
                "path_m": result["path_m"], "evidence_dir": str(folder), "source_batch": str(actual_source),
                "source_body_position": history[-1]["position"], "target_body_position": frame["position"],
                "intermediate_regions": result.get("intermediate_regions", []),
                "landmark_origin_id": result["event"].get("landmark_origin_id")},
            "actor_imitation_label": False, "is_human_verified_node_label": False}
        (return_events if result["target_kind"] == "observed_revisit" else events).append(event)
        for path in [folder / "committed.jsonl", folder / "source_history.jsonl", target_path]:
            provenance[str(path)] = sha256_file(path)
    if any(len(h) != 1 for h in histories.values()):
        raise ValueError("paired exits did not share identical source histories")
    primary = deduplicate_events(events)
    pairs = branch_pair_summary(primary)
    write_rows(output / "executions.jsonl", records)
    write_rows(output / "events_all_repeats.jsonl", events)
    write_rows(output / "physical_revisit_all_repeats.jsonl", return_events)
    write_rows(output / "branch_pairs.jsonl", pairs)
    nodes_path = source / "node_labels_review.jsonl"
    if nodes_path.exists():
        nodes = read_jsonl(nodes_path)
        if any(n["role"] not in {"fit", "development"} or n.get("training_mask") is not False for n in nodes):
            raise ValueError("node label role or silver supervision mask is invalid")
        shutil.copyfile(nodes_path, output / "node_labels.jsonl")
        provenance[str(nodes_path)] = sha256_file(nodes_path)
    stats = {}
    origins = json.loads((source / "landmarks.json").read_text())
    for role in ("fit", "development"):
        ee = [e for e in primary if e["role"] == role]
        rr = [r for r in records if r["role"] == role]
        write_rows(output / f"events_{role}.jsonl", ee)
        write_rows(output / f"physical_revisit_{role}.jsonl", deduplicate_events([e for e in return_events if e["role"] == role]))
        stats[role] = {"attempted_source_poses": len({r["origin_id"] for r in rr}),
            "attempted_houses": len({r["scene"] for r in rr}), "executions": len(rr),
            "statuses": dict(Counter(r["status"] for r in rr)), "primary_unique_events": len(ee),
            "primary_source_poses": len({e["origin_id"] for e in ee}), "primary_houses": len({e["scene"] for e in ee}),
            "target_kinds": dict(Counter(e["target_kind"] for e in ee)),
            "pair_states": dict(Counter(p["state"] for p in pairs if p["role"] == role)),
            "above_old_200_steps": sum(r["branch_steps"] > 200 for r in rr),
            "above_old_500_steps": sum(r["branch_steps"] > 500 for r in rr),
            "completed_above_200_steps": sum(r["event_complete"] and r["branch_steps"] > 200 for r in rr),
            "horizon_s": {"min": min((e["target"]["horizon_s"] for e in ee), default=None),
                          "max": max((e["target"]["horizon_s"] for e in ee), default=None)},
            "physical_revisits_including_repeats": sum(e["role"] == role for e in return_events)}
    all_collected = [json.loads((root / "executions" / t["task_id"] / "result.json").read_text())
                     for root in [source, *supplements] for t in read_jsonl(root / "tasks.jsonl")]
    write_json(output / "summary.json", {"roles": stats, "source_proximity_components_radius_1_5m": spatial_clusters(origins),
        "logical_branch_steps": sum(r["branch_steps"] for r in records),
        "physical_steps_including_source_and_restore": sum(r["env_steps"] for r in records),
        "collected_executions_including_superseded": len(all_collected),
        "physical_steps_including_superseded_source_and_restore": sum(r["env_steps"] for r in all_collected),
        "sealed_files_opened": 0, "source_histories_identical_within_origin": True,
        "world_model_trained": False, "LightNav_used": False, "node_labels_are_simulation_silver": True})
    write_json(output / "training_contract.json", {"inputs": ["history_rgb", "history_valid", "requested_skill"],
        "never_input": ["scene", "viewpoint_id", "episode_uid", "origin_id", "pose", "regions", "landmarks", "target", "horizon_s"],
        "event_targets": "actual room/next-reviewed-decision/outdoor observations", "normalization": "existing fit-only SigLIP statistics",
        "source_history": "8 actual causal pan observations over 0.7 seconds, identical before paired interventions",
        "same_region_is_automatic_negative": False, "repeats_count_as_independent_samples": False,
        "censored_is_success": False, "revisit_is_next_room": False, "actor_imitation_labels": False,
        "development_gradient_updates": False, "node_labels_are_human_gold": False,
        "legacy_region_entry_and_614_weak_events": "separate auxiliary sources, excluded from primary counts"})
    for path in [source / "protocol.json", source / "tasks.jsonl", Path(__file__).resolve()]:
        provenance[str(path)] = sha256_file(path)
    for supplement in supplements:
        for path in [supplement / "protocol.json", supplement / "tasks.jsonl"]:
            provenance[str(path)] = sha256_file(path)
    snapshot = output / "source_snapshot" / Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).resolve(), snapshot)
    write_json(output / "input_manifest.json", provenance)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--supplement", action="append", type=Path, default=[])
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve(), [p.resolve() for p in args.supplement])
