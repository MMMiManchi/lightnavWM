"""Prepare, execute and summarize a separately versioned branch recollection."""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.world_model.branch_benchmark_entry import (
    CONFIG,
    DATASET,
    FIT,
    INVENTORY,
    NODES,
    REPO,
    SOURCE_BY_SPLIT,
    _depth_m,
    _episode_rows,
    _pose,
    _save_frame,
    append_jsonl,
    read_jsonl,
    write_json,
)
from experiments.world_model.branch_curation import (
    CurationRules,
    assign_region,
    image_quality,
    load_floor_polygons,
    sample_id,
)
from experiments.world_model.branch_exit import follow_opening, motion_stalled, replay_actions
from experiments.world_model.branch_recollection import (
    POLICY_ID,
    RecollectionRules,
    causal_history,
    checkpoint_command,
    confirmed_event,
    exit_checkpoint,
    horizontal_distance,
    restored,
    validate_origin,
)
from experiments.world_model.jsonutil import sha256_file
from experiments.world_model.precollection_protocol import (
    assert_collection_not_blocked,
    assert_dev_collection_authorized,
)
from experiments.world_model.serve_collect import prepare_server_plan


def prepare(output: Path, origins_path: Path, source_batch: Path, budget=4000):
    output, source_batch = output.resolve(), source_batch.resolve()
    recipes = read_jsonl(origins_path)
    if not recipes:
        raise ValueError("no reviewed origins")
    if any(r.get("role") not in {"fit", "development"} for r in recipes):
        raise ValueError("sealed origins are never recollected here")
    wanted = {r["replay_recipe"]["restore_original_origin_id"] for r in recipes}
    nodes = {}
    for row in read_jsonl(NODES):
        key = f"{row['scene']}|{row['episode_uid']}|{row['step_index']}"
        if key in wanted:
            nodes[key] = row
    rules = RecollectionRules()
    manifest = json.loads((FIT / "manifest.json").read_text())
    if sha256_file(CONFIG) != manifest["config"]["sha256"]:
        raise ValueError("replay sensor/control config differs from original collection")
    output.mkdir(parents=True, exist_ok=False)
    sources = {str(origins_path.resolve()): sha256_file(origins_path), str(CONFIG): sha256_file(CONFIG),
               str(NODES): sha256_file(NODES), str(FIT / "manifest.json"): sha256_file(FIT / "manifest.json")}
    prepared = []
    for recipe in recipes:
        restore = recipe["replay_recipe"]
        parent = nodes[restore["restore_original_origin_id"]]
        expected_partition = "train_fit" if recipe["role"] == "fit" else "train_dev"
        command_path = Path(restore["then_replay_commands"]).resolve()
        if not command_path.is_relative_to(source_batch):
            raise ValueError("command prefix outside frozen source batch")
        source_root = SOURCE_BY_SPLIT[parent["split"]]
        actions = replay_actions(_episode_rows(source_root / "transitions.jsonl", parent["episode_uid"]),
                                 parent["episode_uid"], parent["step_index"])
        prefix = read_jsonl(command_path)[:restore["command_count"]]
        if len(prefix) != restore["command_count"]:
            raise ValueError("recollection prefix is incomplete")
        actions += [p["command"] for p in prefix]
        frames_path = command_path.with_name("observations.jsonl")
        frames = read_jsonl(frames_path)
        source = frames[restore["command_count"]]
        if source["rgb_sha256"] != restore["require_current_rgb_sha256"]:
            raise ValueError("recollection image does not match the recorded prefix")
        if len(actions) + rules.maximum_steps + 5 > 500:
            raise ValueError("replay and branch would exceed the server step limit")
        folder = output / "origins" / sample_id(recipe["new_origin_id"])
        folder.mkdir(parents=True)
        allowlist = command_path.parent.parent / "allowlist.json"
        plan, _order = prepare_server_plan(allowlist, DATASET, INVENTORY, partition=expected_partition)
        if any(e["scene"] != recipe["scene"] for e in plan["episodes"]):
            raise ValueError("source allowlist includes a different scene")
        shutil.copyfile(allowlist, folder / "allowlist.json")
        write_json(folder / "replay_actions.json", {"actions": actions})
        house = REPO / "data/scene_datasets/mp3d" / recipe["scene"] / f"{recipe['scene']}.house"
        polygons = load_floor_polygons(house)
        region = assign_region(source["position"], polygons, CurationRules())
        if region is None:
            raise ValueError("reviewed source has no unambiguous region; do not invent one")
        origin = {"origin_id": recipe["new_origin_id"], "role": recipe["role"], "scene": recipe["scene"],
                  "episode_uid": parent["episode_uid"], "step_index": len(actions),
                  "position": source["position"], "yaw": source["yaw"], "source_region": region,
                  "expected_rgb_sha256": source["rgb_sha256"], "exits": recipe["exit_proposals"],
                  "node_label_quality": "reviewed_visual_and_local_geometry_silver",
                  "seed": manifest["seed"]["requested"], "origin_directory": str(folder),
                  "policy_id": POLICY_ID, "house_partition": expected_partition,
                  "source_kind": "recorded_replay", "source_history_available": True}
        validate_origin(origin, rules)
        write_json(folder / "origin.json", origin)
        prepared.append(origin)
        for path in [command_path, frames_path, allowlist, source_root / "transitions.jsonl", house,
                     folder / "origin.json", folder / "allowlist.json", folder / "replay_actions.json"]:
            sources.setdefault(str(path), sha256_file(path))
    maximum_steps = sum((o["step_index"] + rules.maximum_steps) * 2 * rules.repeats for o in prepared)
    if maximum_steps > budget:
        raise ValueError(f"worst-case {maximum_steps} environment steps exceed declared budget {budget}")
    tasks = [{"task_id": sample_id(o["origin_id"], e["exit_id"], repeat), "origin": o,
              "exit": e, "repeat_index": repeat}
             for o in prepared for e in o["exits"] for repeat in range(rules.repeats)]
    protocol = {"rules": rules.record(), "policy_id": POLICY_ID, "origins": len(prepared),
                "tasks": len(tasks), "environment_step_budget": budget,
                "worst_case_environment_steps": maximum_steps,
                "source_batch": str(source_batch), "sealed_data_opened": False,
                "LightNav_used": False, "expert_task_route_required": False,
                "actor_imitation_labels": False, "geometry_is_offline_collection_and_label_only": True,
                "target_kind": rules.target_kind,
                "same_region_junctions_not_automatically_accepted": True,
                "failed_executions_are_not_retried_as_successes": True}
    write_json(output / "protocol.json", protocol)
    (output / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    for path in [Path(__file__).resolve(), REPO / "experiments/world_model/branch_recollection.py",
                 REPO / "scripts/recollect_reviewed_branches.py"]:
        sources[str(path)] = sha256_file(path)
        dest = output / "source_snapshot" / path.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    write_json(output / "input_manifest.json", sources)
    print(json.dumps(protocol), flush=True)


def _run_task(output, task, folder):
    if sys.version_info[:2] != (3, 9):
        raise RuntimeError("worker requires the installed Habitat Python 3.9")
    sys.path.insert(0, str(REPO / "habitat_server"))
    from experiments.world_model.server_adapter import WorldModelVLNCEEnv

    origin, exit_ = task["origin"], task["exit"]
    rules = RecollectionRules()
    validate_origin(origin, rules)
    if origin["role"] == "fit":
        assert_collection_not_blocked()
    else:
        assert_dev_collection_authorized()
    parent = Path(origin["origin_directory"])
    payload, order = prepare_server_plan(parent / "allowlist.json", DATASET, INVENTORY,
                                         partition=origin["house_partition"])
    actions = json.loads((parent / "replay_actions.json").read_text())["actions"]
    polygons = load_floor_polygons(REPO / "data/scene_datasets/mp3d" / origin["scene"] / f"{origin['scene']}.house")
    record = {"task_id": task["task_id"], "origin_id": origin["origin_id"], "role": origin["role"],
              "scene": origin["scene"], "episode_uid": origin["episode_uid"],
              "exit_id": exit_["exit_id"], "repeat_index": task["repeat_index"],
              "policy_id": POLICY_ID, "lightnav_used": False, "actor_imitation_label": False,
              "evidence_dir": str(folder), "status": "error", "event_complete": False,
              "replay_steps": 0, "branch_steps": 0, "env_steps": 0,
              "checkpoint_reached": False, "source_restored": False}
    env = None
    try:
        env = WorldModelVLNCEEnv(config_path=str(CONFIG), split="train", data_path=str(DATASET), gpu_id=0,
            max_steps=len(actions) + rules.maximum_steps + 5, episode_allow=set(order), allow_order=order,
            requested_seed=origin["seed"], expected_dataset_sha256=payload["dataset_sha256"],
            early_stop_rotation=0, early_stop_steps=0)
        obs, info = env.reset(seed=origin["seed"])
        if str(info.get("episode_id")) != origin["episode_uid"].split("|")[-1] or origin["scene"] not in str(info.get("scene_id", "")):
            raise RuntimeError("reset returned a different source episode")
        history = []
        history_steps = {max(1, len(actions) - i * rules.source_history_stride_steps)
                         for i in range(rules.source_history_frames)}
        for index, action in enumerate(actions):
            obs, _reward, terminated, truncated, info = env.step({"wm_action_id": f"replay-{index}", **action})
            record["replay_steps"] += 1
            record["env_steps"] += 1
            if terminated or truncated:
                raise RuntimeError("replay ended before the reviewed source")
            if index + 1 in history_steps:
                name = "node" if index == len(actions) - 1 else f"history-{index:03d}"
                frame = _save_frame(folder, name, obs, info)
                frame.update(replay_step=index + 1, rgb_path=str(folder / "assets" / f"{name}_rgb.npy"))
                append_jsonl(folder / "source_history.jsonl", frame)
                history.append(frame)
        node = history[-1]
        append_jsonl(folder / "observations.jsonl", node)
        position, yaw = _pose(info)
        record["source_restored"] = restored(origin, position, yaw, node["rgb_sha256"], rules)
        record["restored_rgb_sha256"] = node["rgb_sha256"]
        record["source_history_rgb"] = causal_history(history)
        if not record["source_restored"]:
            record["status"] = "restore_failed"
            return record
        checkpoint = exit_checkpoint(position, yaw, exit_["bearing_left_rad"], rules.checkpoint_m)
        frames = [node]
        drive_moves = []
        for step in range(rules.maximum_steps):
            if not record["checkpoint_reached"]:
                record["checkpoint_reached"] = horizontal_distance(position, checkpoint) <= rules.checkpoint_tolerance_m
            if not record["checkpoint_reached"]:
                command, mode = checkpoint_command(position, yaw, checkpoint), "entry_checkpoint"
            else:
                # The original side is no longer body-relative after turning into it.
                depth = _depth_m(obs)
                if depth is None:
                    record["status"] = "depth_unavailable"
                    break
                following = follow_opening("center", depth)
                if following is None:
                    record["status"] = "truncated_blocked"
                    break
                command, mode = following
            append_jsonl(folder / "commands.jsonl", {"step": step, "mode": mode, "command": command})
            previous = position
            obs, _reward, terminated, truncated, info = env.step({"wm_action_id": f"branch-{step}", **command})
            record["branch_steps"] += 1
            record["env_steps"] += 1
            position, yaw = _pose(info)
            frame = _save_frame(folder, f"branch-{step:03d}", obs, info)
            append_jsonl(folder / "observations.jsonl", frame)
            frames.append(frame)
            if float(command["action_args"]["linear_velocity"]) > -0.9:
                drive_moves.append(horizontal_distance(previous, position))
            else:
                drive_moves.clear()
            if not record["checkpoint_reached"]:
                record["checkpoint_reached"] = horizontal_distance(position, checkpoint) <= rules.checkpoint_tolerance_m
            event = confirmed_event(frames, polygons, record["checkpoint_reached"])
            if event is not None:
                target = frames[event["target_index"]]
                quality = image_quality(np.load(folder / "assets" / f"{target['name']}_rgb.npy"), CurationRules())
                record.update(status="complete" if quality["usable"] else "target_image_quarantined",
                              event_complete=quality["usable"], target_index=event["target_index"],
                              source_region=event["regions"][0], target_region=event["regions"][event["target_index"]],
                              horizon_s=event["horizon_s"], path_m=event["path_m"], target_quality=quality,
                              target_kind=rules.target_kind)
                break
            if motion_stalled(drive_moves):
                record["status"] = "truncated_blocked"
                break
            if terminated or truncated:
                record["status"] = "habitat_ended"
                break
            record["status"] = "truncated_budget" if step + 1 == rules.maximum_steps else "in_progress"
        return record
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["status"] = "error"
        raise
    finally:
        write_json(folder / "result.json", record)
        if env is not None:
            env.close()


def worker(output: Path, worker_index: int, workers: int):
    output = output.resolve()
    if not 0 <= worker_index < workers:
        raise ValueError("worker index out of range")
    for path, expected in json.loads((output / "input_manifest.json").read_text()).items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"prepared collection input changed: {path}")
    tasks = read_jsonl(output / "tasks.jsonl")[worker_index::workers]
    for task in tasks:
        folder = output / "executions" / task["task_id"]
        folder.mkdir(parents=True, exist_ok=False)
        result = _run_task(output, task, folder)
        print(json.dumps({k: result.get(k) for k in ("task_id", "status", "env_steps", "source_restored", "target_region")}), flush=True)


def summarize(output: Path):
    output = output.resolve()
    tasks = read_jsonl(output / "tasks.jsonl")
    results = []
    for task in tasks:
        path = output / "executions" / task["task_id"] / "result.json"
        if not path.exists():
            raise ValueError("collection has unfinished tasks")
        result = json.loads(path.read_text())
        results.append(result)
    protocol = json.loads((output / "protocol.json").read_text())
    steps = sum(r["env_steps"] for r in results)
    if steps > protocol["environment_step_budget"]:
        raise ValueError("registered environment budget exceeded")
    complete = [r for r in results if r["event_complete"]]
    outcomes = defaultdict(dict)
    examples = []
    for record in complete:
        task = next(t for t in tasks if t["task_id"] == record["task_id"])
        exit_ = task["exit"]
        folder = Path(record["evidence_dir"])
        frames = read_jsonl(folder / "observations.jsonl")
        target = frames[record["target_index"]]
        source = frames[0]
        examples.append({"sample_id": record["task_id"], "role": record["role"], "scene": record["scene"],
            "origin_id": record["origin_id"], "exit_id": record["exit_id"], "repeat_index": record["repeat_index"],
            "inputs": {"history_rgb": record["source_history_rgb"],
                       "requested_skill": {"policy_id": POLICY_ID, "bearing_left_rad": exit_["bearing_left_rad"], "side": exit_["side"]}},
            "target": {"rgb": {"path": str(folder / "assets" / f"{target['name']}_rgb.npy"), "sha256": target["rgb_sha256"]},
                       "horizon_s": record["horizon_s"], "kind": record["target_kind"]},
            "offline_labels": {"source_region": record["source_region"], "target_region": record["target_region"],
                               "evidence_dir": str(folder), "branch_steps": record["branch_steps"],
                               "source_rgb_sha256": source["rgb_sha256"]}, "actor_imitation_label": False})
        outcomes[record["origin_id"]].setdefault(record["exit_id"], set()).add(record["target_region"]["region_id"])
    paired = sum(len(v) == 2 for v in outcomes.values())
    different = sum(len(v) == 2 and len(set.union(*v.values())) > 1 for v in outcomes.values())
    summary = {"tasks": len(tasks), "statuses": dict(Counter(r["status"] for r in results)),
        "complete_events_including_repeats": len(complete),
        "unique_confirmed_start_exit_combinations": len({(r["origin_id"], r["exit_id"]) for r in complete}),
        "paired_origins": paired, "different_region_outcome_origins": different,
        "source_restore_failures": sum(not r["source_restored"] for r in results),
        "environment_steps": steps, "environment_step_budget": protocol["environment_step_budget"],
        "sealed_trajectory_files_read": 0, "node_labels_are_reviewed_silver": True,
        "is_complete_50_fit_20_dev_collection": False}
    (output / "transitions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (output / "event_examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in examples))
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
