"""Offline viewpoint candidates -> reviewed images -> actual branch outcomes.

Only existing fit/dev houses are opened. Geometry witnesses rank candidates;
they never replace actual recorded future observations or become student inputs.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.world_model.branch_benchmark_entry import (
    CONFIG,
    DATASET,
    INVENTORY,
    REPO,
    _depth_m,
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
from experiments.world_model.branch_exit import follow_opening, motion_stalled
from experiments.world_model.branch_recollection import (
    POLICY_ID,
    RecollectionRules,
    checkpoint_command,
    confirmed_event,
    exit_checkpoint,
    horizontal_distance,
)
from experiments.world_model.group_manifest import assign_groups, build_allowlist, load_split_rows
from experiments.world_model.jsonutil import sha256_file
from experiments.world_model.pose_math import wrap_pi
from experiments.world_model.precollection_protocol import (
    assert_collection_not_blocked,
    assert_dev_collection_authorized,
)
from experiments.world_model.serve_collect import prepare_server_plan

LOOK_DPS = (50, 50, -50, -50, -50, -50, 50, 50)
RULES = RecollectionRules(maximum_steps=200)


def source_episode(grouped, scene, role, preferred_uid):
    candidates = [r for r in grouped if r["scene"] == scene
                  and (role != "fit" or r["partition"] == "train_fit")]
    if not candidates:
        raise ValueError("house has no episode allowed by the existing collection gate")
    return min(candidates, key=lambda r: (r["episode_uid"] != preferred_uid, str(r["episode_id"])))


def _sources(output):
    return [Path(__file__).resolve(), REPO / "scripts/collect_viewpoint_branches.py",
            REPO / "experiments/world_model/branch_recollection.py"]


def prepare_previews(output, candidates, legacy):
    output = output.resolve()
    pooled = [r for f in sorted(candidates.glob("worker_??.jsonl")) for r in read_jsonl(f)]
    if not pooled:
        raise ValueError("candidate mining has no outputs")
    old = read_jsonl(legacy / "pairs.jsonl")
    roles, episodes = {}, {}
    for row in old:
        role = "fit" if row["partition"] == "train_fit" else "development"
        if row["partition"] not in {"train_fit", "train_dev"} or (row["scene"] in roles and roles[row["scene"]] != role):
            raise ValueError("historical roles are inconsistent")
        roles[row["scene"]] = role
        episodes.setdefault(row["scene"], row["episode_uid"])
    grouped_candidates = defaultdict(list)
    for row in pooled:
        if row["scene"] not in roles or roles[row["scene"]] != row["role"]:
            raise ValueError("candidate is outside the existing fit/dev scene pools")
        grouped_candidates[row["scene"]].append(row)
    if (output / "protocol.json").exists() or (output / "previews").exists():
        raise FileExistsError("preview protocol or actual observations already exist")
    output.mkdir(parents=True, exist_ok=True)
    path, dataset_hash, rows = load_split_rows(DATASET, "train")
    grouped = assign_groups(rows)
    previews = []
    inputs = {str(CONFIG): sha256_file(CONFIG), str(path): dataset_hash,
              str(legacy / "pairs.jsonl"): sha256_file(legacy / "pairs.jsonl")}
    for scene, choices in sorted(grouped_candidates.items()):
        role = roles[scene]
        episode = source_episode(grouped, scene, role, episodes[scene])
        episode_uid = episode["episode_uid"]
        folder = output / "scenes" / scene
        folder.mkdir(parents=True, exist_ok=True)
        allowlist = build_allowlist(grouped, [episode], dataset_json=path, dataset_sha256=dataset_hash)
        write_json(folder / "allowlist.json", allowlist)
        prepare_server_plan(folder / "allowlist.json", DATASET, INVENTORY,
                            partition="train_fit" if role == "fit" else "train_dev")
        limit = 3 if role == "fit" else 8
        for choice in sorted(choices, key=lambda r: r["ranking_score"])[:limit]:
            previews.append({**choice, "episode_uid": episode_uid, "allowlist": str(folder / "allowlist.json")})
        for file in [REPO / "data/scene_datasets/mp3d" / scene / f"{scene}.house",
                     REPO / "data/scene_datasets/mp3d" / scene / f"{scene}.navmesh"]:
            inputs[str(file)] = sha256_file(file)
    protocol = {"version": "viewpoint_branch_collection/v3", "rules": RULES.record(),
                "policy_id": POLICY_ID, "max_fit_origins": 50, "max_development_origins": 20,
                "preview_candidates": len(previews), "preview_worst_case_steps": len(previews) * len(LOOK_DPS),
                "environment_step_budget_including_previews": 60000,
                "collection_worst_case_steps": 70 * 4 * (len(LOOK_DPS) + RULES.maximum_steps),
                "look_degrees_per_second": list(LOOK_DPS), "config_sha256": inputs[str(CONFIG)],
                "seed": 11, "source_teleport_offline_only": True, "official_viewpoints_are_runtime_inputs": False,
                "graph_adjacency_used": False, "requires_current_RGB_architecture_review": True,
                "geometry_witness_is_training_target": False, "requires_actual_branch_execution": True,
                "sealed_data_opened": False, "actor_imitation_labels": False, "LightNav_used": False}
    if protocol["preview_worst_case_steps"] + protocol["collection_worst_case_steps"] > 60000:
        raise ValueError("registered preview and branch worst-case budget exceeds 60000")
    (output / "preview_tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in previews))
    write_json(output / "protocol.json", protocol)
    for file in _sources(output):
        inputs[str(file)] = sha256_file(file)
        dest = output / "source_snapshot" / file.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)
    write_json(output / "input_manifest.json", inputs)
    print(json.dumps({"preview_tasks": len(previews), "houses": len(grouped_candidates), "roles": dict(Counter(p["role"] for p in previews))}), flush=True)


def make_env(origin, max_steps, *, event_driven=False):
    if origin["role"] not in {"fit", "development"}:
        raise ValueError("sealed role refused before environment creation")
    if sys.version_info[:2] != (3, 9):
        raise RuntimeError("use the installed Habitat Python 3.9")
    sys.path.insert(0, str(REPO / "habitat_server"))
    from experiments.world_model.server_adapter import WorldModelVLNCEEnv

    if origin["role"] == "fit":
        assert_collection_not_blocked()
    else:
        assert_dev_collection_authorized()
    payload, order = prepare_server_plan(Path(origin["allowlist"]), DATASET, INVENTORY,
                                        partition="train_fit" if origin["role"] == "fit" else "train_dev")
    return WorldModelVLNCEEnv(config_path=str(CONFIG), split="train", data_path=str(DATASET), gpu_id=0,
        max_steps=max_steps, event_driven=event_driven,
        episode_allow=set(order), allow_order=order, requested_seed=11,
        expected_dataset_sha256=payload["dataset_sha256"], early_stop_rotation=0, early_stop_steps=0)


def execute(env, command, action_id):
    outcome = env.step({"wm_action_id": action_id, **command})
    ack = outcome[-1].get("wm_execution") or {}
    if ack.get("executed") is not True or ack.get("replayed") is True:
        raise RuntimeError("collector did not receive a new executed-action acknowledgement")
    return outcome


def reset_singleton_episode(env):
    native = env._habitat_env
    if native is not None:
        episodes = list(native.episodes)
        if len(episodes) != 1:
            raise RuntimeError("source reset requires exactly one allowlisted episode")
        # The collection adapter deliberately pins cycle=False. Rewind only the
        # already checked singleton; never enable unconstrained dataset cycling.
        native.episodes = episodes
    return env.reset(seed=11)


def place_and_observe(env, origin, folder, tag):
    reset_singleton_episode(env)
    reset_position = origin.get("reset_position", origin["position"])
    reset_yaw = origin.get("reset_yaw", origin["yaw"])
    sim = env._habitat_env.sim
    if not sim.set_agent_state(reset_position, [0.0, math.sin(reset_yaw / 2), 0.0, math.cos(reset_yaw / 2)]):
        raise RuntimeError("failed to place the offline source state")
    history = []
    for i, dps in enumerate(LOOK_DPS):
        command = {"action": "velocity_control", "action_args": {"linear_velocity": -1.0, "angular_velocity": dps / 300.0}}
        obs, _reward, terminated, truncated, info = execute(env, command, f"{tag}-look-{i}")
        if terminated or truncated:
            raise RuntimeError("source observation ended before branch choice")
        name = "node" if i + 1 == len(LOOK_DPS) else f"history-{i:03d}"
        frame = _save_frame(folder, name, obs, info)
        frame["observation_step"] = i + 1
        frame["rgb_path"] = str(folder / "assets" / f"{name}_rgb.npy")
        append_jsonl(folder / "source_history.jsonl", frame)
        append_jsonl(folder / "source_commands.jsonl", {"step": i, "command": command})
        history.append(frame)
    return obs, info, history


def preview_worker(output, worker, workers):
    output = output.resolve()
    tasks = read_jsonl(output / "preview_tasks.jsonl")
    scenes = sorted({r["scene"] for r in tasks})[worker::workers]
    for scene in scenes:
        rows = [r for r in tasks if r["scene"] == scene]
        env = make_env(rows[0], 30)
        try:
            for candidate in rows:
                folder = output / "previews" / candidate["candidate_id"]
                folder.mkdir(parents=True, exist_ok=False)
                result = {"candidate": candidate, "status": "error", "env_steps": 0, "error": None}
                try:
                    obs, info, history = place_and_observe(env, candidate, folder, candidate["candidate_id"])
                    result["env_steps"] = len(history)
                    position, yaw = _pose(info)
                    quality = image_quality(obs["rgb"], CurationRules(maximum_black_fraction=0.1))
                    polygons = load_floor_polygons(REPO / "data/scene_datasets/mp3d" / scene / f"{scene}.house")
                    region = assign_region(position, polygons, CurationRules())
                    clearances = []
                    for exit_ in candidate["exits"]:
                        goal = exit_checkpoint(position, yaw, wrap_pi(exit_["world_yaw"] - yaw))
                        end = np.asarray(env._habitat_env.sim.pathfinder.try_step_no_sliding(
                            np.asarray(position, dtype=np.float32), np.asarray(goal, dtype=np.float32)))
                        clearances.append(horizontal_distance(position, end))
                    candidate = {**candidate, "reset_position": candidate["position"], "reset_yaw": candidate["yaw"],
                                 "position": position, "yaw": yaw}
                    for exit_, clearance in zip(candidate["exits"], clearances):
                        exit_["bearing_left_rad"] = wrap_pi(exit_["world_yaw"] - yaw)
                        exit_["local_navmesh_1_5m_walked"] = clearance
                    result.update(candidate=candidate, source_region=region, source_quality=quality,
                                  source_rgb={"path": history[-1]["rgb_path"], "sha256": history[-1]["rgb_sha256"]},
                                  history_rgb=[{"path": f["rgb_path"], "sha256": f["rgb_sha256"]} for f in history],
                                  status="rendered" if quality["usable"] and region is not None and min(clearances) >= 1.35 else "source_rejected")
                except Exception as exc:
                    result["error"] = f"{type(exc).__name__}: {exc}"
                    result["env_steps"] = max(len(read_jsonl(folder / "source_commands.jsonl")), env._step_count)
                write_json(folder / "preview.json", result)
                print(json.dumps({"candidate": candidate["candidate_id"], "scene": scene, "status": result["status"]}), flush=True)
        finally:
            env.close()


def prepare_collection(output, reviews_path):
    output = output.resolve()
    reviews = json.loads(reviews_path.read_text())
    if not reviews.get("reviewer"):
        raise ValueError("explicit current-image review is required")
    accepted = []
    for cid, verdict in reviews["verdicts"].items():
        if verdict.get("label") != "decision":
            continue
        row = json.loads((output / "previews" / cid / "preview.json").read_text())
        if row["status"] != "rendered":
            raise ValueError("reviewed source failed image or local geometry checks")
        row["review"] = verdict
        accepted.append(row)
    counts = Counter(r["candidate"]["role"] for r in accepted)
    if counts["fit"] > 50 or counts["development"] > 20:
        raise ValueError("reviewed sources exceed the registered maximum")
    tasks = []
    for preview in accepted:
        origin = preview["candidate"]
        origin.update(origin_id=f"vp-reviewed:{origin['candidate_id']}", expected_rgb_sha256=preview["source_rgb"]["sha256"],
                      node_label_quality="reviewed_visual_and_local_geometry_silver")
        for exit_ in origin["exits"]:
            for repeat in range(2):
                tasks.append({"task_id": sample_id(origin["origin_id"], exit_["exit_id"], repeat),
                              "origin": origin, "exit": exit_, "repeat_index": repeat})
    path = output / "tasks.jsonl"
    if path.exists():
        raise FileExistsError(path)
    path.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    write_json(output / "reviewed_source_counts.json", {"origins": dict(counts), "tasks": len(tasks), "reviews_sha256": sha256_file(reviews_path)})
    print(json.dumps({"origins": dict(counts), "tasks": len(tasks)}), flush=True)


def collect_one(env, task, folder):
    origin, exit_ = task["origin"], task["exit"]
    record = {"task_id": task["task_id"], "origin_id": origin["origin_id"], "role": origin["role"],
              "scene": origin["scene"], "exit_id": exit_["exit_id"], "repeat_index": task["repeat_index"],
              "policy_id": POLICY_ID, "evidence_dir": str(folder), "status": "error", "event_complete": False,
              "env_steps": 0, "branch_steps": 0, "source_restored": False, "checkpoint_reached": False,
              "offline_source_placement": True, "actor_imitation_label": False, "LightNav_used": False}
    try:
        obs, info, history = place_and_observe(env, origin, folder, task["task_id"])
        record["env_steps"] = len(history)
        record["source_history_rgb"] = [{"path": f["rgb_path"], "sha256": f["rgb_sha256"]} for f in history]
        record["source_restored"] = history[-1]["rgb_sha256"] == origin["expected_rgb_sha256"]
        record["restored_rgb_sha256"] = history[-1]["rgb_sha256"]
        if not record["source_restored"]:
            record["status"] = "restore_failed"
            return record
        node = history[-1]
        append_jsonl(folder / "observations.jsonl", node)
        position, yaw = _pose(info)
        checkpoint = exit_checkpoint(position, yaw, exit_["bearing_left_rad"])
        polygons = load_floor_polygons(REPO / "data/scene_datasets/mp3d" / origin["scene"] / f"{origin['scene']}.house")
        frames, drive_moves = [node], []
        for step in range(RULES.maximum_steps):
            if not record["checkpoint_reached"]:
                record["checkpoint_reached"] = horizontal_distance(position, checkpoint) <= RULES.checkpoint_tolerance_m
            if not record["checkpoint_reached"]:
                command, mode = checkpoint_command(position, yaw, checkpoint), "entry_checkpoint"
            else:
                depth = _depth_m(obs)
                following = follow_opening("center", depth) if depth is not None else None
                if following is None:
                    record["status"] = "truncated_blocked"
                    break
                command, mode = following
            append_jsonl(folder / "commands.jsonl", {"step": step, "command": command, "mode": mode})
            previous = position
            obs, _reward, terminated, truncated, info = execute(env, command, f"{task['task_id']}-branch-{step}")
            record["env_steps"] += 1
            record["branch_steps"] += 1
            position, yaw = _pose(info)
            frame = _save_frame(folder, f"branch-{step:03d}", obs, info)
            append_jsonl(folder / "observations.jsonl", frame)
            frames.append(frame)
            if command["action_args"]["linear_velocity"] > -0.9:
                drive_moves.append(horizontal_distance(previous, position))
            else:
                drive_moves.clear()
            record["checkpoint_reached"] |= horizontal_distance(position, checkpoint) <= RULES.checkpoint_tolerance_m
            event = confirmed_event(frames, polygons, record["checkpoint_reached"])
            if event:
                target = frames[event["target_index"]]
                quality = image_quality(np.load(folder / "assets" / f"{target['name']}_rgb.npy"), CurationRules())
                record.update(status="complete" if quality["usable"] else "target_image_quarantined",
                    event_complete=quality["usable"], source_region=event["regions"][0],
                    target_region=event["regions"][event["target_index"]], target_index=event["target_index"],
                    horizon_s=event["horizon_s"], path_m=event["path_m"], target_kind=RULES.target_kind)
                break
            if motion_stalled(drive_moves):
                record["status"] = "truncated_blocked"
                break
            if terminated or truncated:
                record["status"] = "habitat_ended"
                break
            record["status"] = "truncated_budget"
        return record
    except Exception as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        record["env_steps"] = max(record["env_steps"], env._step_count)
        return record
    finally:
        write_json(folder / "result.json", record)


def collection_worker(output, worker, workers):
    output = output.resolve()
    tasks = read_jsonl(output / "tasks.jsonl")
    scenes = sorted({r["origin"]["scene"] for r in tasks})[worker::workers]
    for scene in scenes:
        group = [r for r in tasks if r["origin"]["scene"] == scene]
        env = make_env(group[0]["origin"], len(LOOK_DPS) + RULES.maximum_steps + 2)
        try:
            for task in group:
                folder = output / "executions" / task["task_id"]
                folder.mkdir(parents=True, exist_ok=False)
                result = collect_one(env, task, folder)
                print(json.dumps({k: result.get(k) for k in ("task_id", "scene", "status", "env_steps", "source_restored")}), flush=True)
        finally:
            env.close()
