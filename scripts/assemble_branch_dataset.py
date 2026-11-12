"""Assemble actual region-entry targets, repeat groups and causal input histories.

Geometry candidate witnesses are never accepted as observed targets. Raw failed
walks are retained, and same-region outcomes are not automatic hard negatives.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.branch_benchmark_entry import read_jsonl, write_json  # noqa: E402
from experiments.world_model.branch_curation import (  # noqa: E402
    CurationRules,
    branch_pair_summary,
    deduplicate_events,
    image_quality,
)
from experiments.world_model.jsonutil import sha256_file  # noqa: E402


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def build(output, source, pilot, old_curated):
    output.mkdir(parents=True, exist_ok=False)
    tasks = read_jsonl(source / "tasks.jsonl")
    pilot_tasks = read_jsonl(pilot / "tasks.jsonl") if pilot else []
    records, events, provenance = [], [], {}
    source_histories = defaultdict(set)
    for root, jobs in [(source, tasks), (pilot, pilot_tasks)]:
        if root is None:
            continue
        for task in jobs:
            role = task["origin"]["role"]
            if role not in {"fit", "development"}:
                raise ValueError("sealed task in visible branch assembly")
            result_path = root / "executions" / task["task_id"] / "result.json"
            if not result_path.exists():
                raise ValueError(f"unexecuted task: {task['task_id']}")
            result = json.loads(result_path.read_text())
            provenance[str(result_path)] = sha256_file(result_path)
            records.append(result)
            if not result["event_complete"]:
                continue
            if not result["source_restored"] or not result["checkpoint_reached"]:
                raise ValueError("complete record lacks source/entrance proof")
            folder = Path(result["evidence_dir"])
            frames_path = folder / "observations.jsonl"
            commands_path = folder / "commands.jsonl"
            history_path = folder / "source_history.jsonl"
            frames = read_jsonl(frames_path)
            commands = read_jsonl(commands_path)
            history = read_jsonl(history_path)
            if len(frames) != result["branch_steps"] + 1 or len(commands) != result["branch_steps"]:
                raise ValueError("raw frame/command counts disagree")
            if result["target_index"] > result["branch_steps"] or result["target_index"] < 1:
                raise ValueError("future target is missing or not after the decision")
            target = frames[result["target_index"]]
            target_path = folder / "assets" / f"{target['name']}_rgb.npy"
            target_sha = sha256_file(target_path)
            if target_sha != target["rgb_sha256"]:
                raise ValueError("actual future image changed")
            references = result["source_history_rgb"]
            if len(references) != len(history) or history[-1]["name"] != "node":
                raise ValueError("source history is incomplete")
            if references[-1]["sha256"] != frames[0]["rgb_sha256"]:
                raise ValueError("history ends at a different source image")
            source_histories[result["origin_id"]].add(tuple(r["sha256"] for r in references))
            for ref in references:
                path = Path(ref["path"])
                if sha256_file(path) != ref["sha256"]:
                    raise ValueError("source history image changed")
                provenance[str(path)] = ref["sha256"]
            quality = image_quality(np.load(target_path, allow_pickle=False), CurationRules(maximum_black_fraction=0.20))
            event = {"sample_id": result["task_id"], "role": result["role"], "scene": result["scene"],
                "origin_id": result["origin_id"], "exit_id": result["exit_id"], "repeat_index": result["repeat_index"],
                "inputs": {"history_rgb": references, "history_valid": [True] * len(references),
                    "requested_skill": {"name": "follow_visible_exit_until_next_event", "policy_id": result["policy_id"],
                                        "bearing_left_rad": task["exit"]["bearing_left_rad"], "side": task["exit"]["side"]}},
                "target_kind": "first_confidently_entered_different_annotated_region",
                "target": {"rgb": {"path": str(target_path), "sha256": target_sha, "quality": quality},
                           "horizon_s": result["horizon_s"], "horizon_steps": result["target_index"]},
                "offline_labels": {"source_region": result["source_region"], "target_region": result["target_region"],
                    "path_m": result["path_m"], "source_body_position": frames[0]["position"],
                    "target_body_position": target["position"], "execution_end_index": result["branch_steps"],
                    "evidence_dir": str(folder), "source_batch": str(root)},
                "actor_imitation_label": False, "is_human_verified_node_label": False,
                "target_training_usable": quality["usable"]}
            events.append(event)
            for path in [frames_path, commands_path, history_path, target_path]:
                provenance[str(path)] = sha256_file(path)
    if any(len(values) != 1 for values in source_histories.values()):
        raise ValueError("branch interventions do not share identical pre-decision histories")
    primary = deduplicate_events([e for e in events if e["target_training_usable"]])
    weak = deduplicate_events([e for e in events if not e["target_training_usable"]])
    pairs = branch_pair_summary(primary)
    write_rows(output / "executions.jsonl", records)
    write_rows(output / "events_all_repeats.jsonl", events)
    write_rows(output / "branch_pairs.jsonl", pairs)
    for role in ("fit", "development"):
        write_rows(output / f"region_events_{role}.jsonl", [e for e in primary if e["role"] == role])
        write_rows(output / f"low_quality_auxiliary_{role}.jsonl", [e for e in weak if e["role"] == role])
    if old_curated:
        for role in ("fit", "development"):
            path = old_curated / f"region_events_{role}.jsonl"
            provenance[str(path)] = sha256_file(path)
        write_json(output / "additional_auxiliary_sources.json", {
            "old_region_events": {r: str(old_curated / f"region_events_{r}.jsonl") for r in ("fit", "development")},
            "old_weak_event_index": str(old_curated / "legacy_auxiliary_index.jsonl"),
            "merged_into_primary_counts": False})
    previews = [json.loads(path.read_text()) for path in (source / "previews").glob("*/preview.json")]
    preview_steps = sum(r["env_steps"] for r in previews)
    smoke_path = source / "source_smoke_summary.json"
    smoke_steps = json.loads(smoke_path.read_text())["environment_steps"] if smoke_path.exists() else 0
    source_records = [r for r in records if str(source) in r["evidence_dir"]]
    actual_steps = preview_steps + smoke_steps + sum(r["env_steps"] for r in source_records)
    budget = json.loads((source / "protocol.json").read_text())["environment_step_budget_including_previews"]
    if actual_steps > budget:
        raise ValueError("preview plus actual execution budget exceeded")
    stats = {}
    for role in ("fit", "development"):
        rr = [r for r in records if r["role"] == role]
        ee = [e for e in primary if e["role"] == role]
        stats[role] = {"houses_attempted": len({r["scene"] for r in rr}),
            "source_states_attempted": len({r["origin_id"] for r in rr}), "executions": len(rr),
            "statuses": dict(Counter(r["status"] for r in rr)),
            "complete_events_including_repeats": sum(r["event_complete"] for r in rr),
            "primary_deduplicated_events": len(ee), "primary_start_exit_combinations": len({(e["origin_id"], e["exit_id"]) for e in ee}),
            "primary_source_states": len({e["origin_id"] for e in ee}), "primary_houses": len({e["scene"] for e in ee}),
            "pair_states": dict(Counter(p["state"] for p in pairs if p["role"] == role)),
            "low_quality_auxiliary_events": sum(e["role"] == role for e in weak),
            "horizon_s": {"min": min((e["target"]["horizon_s"] for e in ee), default=None),
                          "max": max((e["target"]["horizon_s"] for e in ee), default=None)},
            "source_restore_failures": sum(not r["source_restored"] for r in rr)}
    write_json(output / "summary.json", {"roles": stats, "main_batch_environment_steps_including_previews": actual_steps,
        "main_batch_environment_budget": budget, "pilot_environment_steps": sum(r["env_steps"] for r in records if str(pilot) in r["evidence_dir"]),
        "sealed_trajectory_files_read": 0, "histories_identical_within_each_intervention_group": True,
        "target_quality_max_black_fraction": 0.20, "world_model_trained": False,
        "node_labels_are_reviewed_simulation_silver": True})
    write_json(output / "training_contract.json", {"event_input_fields": ["history_rgb", "history_valid", "requested_skill"],
        "never_feed": ["offline_labels", "scene", "origin_id", "offline_viewpoint_id", "episode_uid", "target", "file_paths_as_text", "horizon_s"],
        "target_is_actual_recorded_future": True, "geometric_witness_is_not_a_target": True,
        "history_is_actual_past": True, "actor_imitation_labels": False,
        "normalization": "inherit existing train_fit-only SigLIP coordinates", "development_gradient_updates": False,
        "same_region_or_repeat_targets_are_not_hard_negatives": True, "scenes_are_not_new_blind_houses": True})
    for path in [source / "protocol.json", source / "tasks.jsonl", Path(__file__).resolve()]:
        provenance[str(path)] = sha256_file(path)
    write_json(output / "input_manifest.json", provenance)
    print(json.dumps(stats), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--old-curated", type=Path)
    args = parser.parse_args()
    build(args.output.resolve(), args.source.resolve(), args.pilot.resolve() if args.pilot else None,
          args.old_curated.resolve() if args.old_curated else None)


if __name__ == "__main__":
    main()
