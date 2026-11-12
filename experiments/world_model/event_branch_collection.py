"""Bounded-by-event branch collection, with an explicit 1000-step fallback.

The fallback produces a censored trajectory, never a successful future label.
All source placement and geometry queries are offline teacher operations.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import traceback
from collections import Counter, deque
from pathlib import Path

import numpy as np

from experiments.world_model.branch_benchmark_entry import (
    REPO,
    _depth_m,
    _pose,
    _save_frame,
    read_jsonl,
    write_json,
)
from experiments.world_model.branch_curation import (
    CurationRules,
    image_quality,
    load_floor_polygons,
    sample_id,
)
from experiments.world_model.branch_exit import branch_velocity, follow_opening
from experiments.world_model.branch_recollection import (
    checkpoint_command,
    exit_checkpoint,
    horizontal_distance,
)
from experiments.world_model.event_collection_control import (
    EventProgress,
    ResourcePause,
    StepAccounting,
    execute_once,
    turn_command,
)
from experiments.world_model.jsonutil import sha256_file
from experiments.world_model.pose_math import wrap_pi
from experiments.world_model.viewpoint_branch_collection import make_env, place_and_observe

POLICY = "reviewed_exit_then_reactive_follow_with_local_stall_recovery/v4.1"
DEFAULT_BRANCH_STEPS = 1000


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, data)
    temporary.replace(path)


def commit(path, row):
    # One committed row contains both the executed command and its observation.
    # A crash before this commit is recovered from the previous observed state.
    import os

    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare(output, preview_roots, reviews_path, max_branch_steps=DEFAULT_BRANCH_STEPS, label_landmarks_path=None):
    if not 1 <= max_branch_steps <= 1000:
        raise ValueError("the currently authorized maximum is 1000 branch steps")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    review = json.loads(reviews_path.read_text())
    if not review.get("reviewer") or review.get("not_yet_frozen_for_collection", False):
        raise ValueError("freeze an explicit current-image source review before collection")
    previews = {}
    for root in preview_roots:
        for path in root.glob("previews/*/preview.json"):
            row = json.loads(path.read_text())
            previews[row["candidate"]["candidate_id"]] = (row, path.resolve())
    tasks, origins, provenance = [], [], {str(reviews_path.resolve()): sha256_file(reviews_path)}
    for cid in review["accepted_candidate_ids"]:
        row, path = previews[cid]
        candidate = row["candidate"]
        if candidate["role"] not in {"fit", "development"} or candidate["scene"] == "Vvot9Ly1tCj":
            raise ValueError("sealed source refused")
        if row["status"] != "rendered":
            raise ValueError("accepted source failed preview checks")
        origin = {**candidate, "origin_id": f"vp-reviewed:{cid}",
                  "source_region_id": row["source_region"]["region_id"],
                  "expected_rgb_sha256": row["source_rgb"]["sha256"],
                  "node_label_quality": "reviewed_visual_and_local_geometry_silver"}
        origins.append(origin)
        provenance[str(path)] = sha256_file(path)
        for exit_ in origin["exits"]:
            for repeat in range(2):
                tasks.append({"task_id": sample_id(origin["origin_id"], exit_["exit_id"], repeat),
                              "origin": origin, "exit": exit_, "repeat_index": repeat})
    (output / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tasks))
    landmarks = origins
    if label_landmarks_path is not None:
        landmarks = json.loads(label_landmarks_path.read_text())
        if any(o["role"] not in {"fit", "development"} or o["scene"] == "Vvot9Ly1tCj" for o in landmarks):
            raise ValueError("sealed label landmark refused")
        provenance[str(label_landmarks_path.resolve())] = sha256_file(label_landmarks_path)
    write_json(output / "landmarks.json", landmarks)
    write_json(output / "protocol.json", {"version": "event_branch_collection/v4.1", "policy_id": POLICY,
        "maximum_branch_steps": max_branch_steps, "source_pan_steps_counted_as_branch": False,
        "native_step_limit": 0, "wrapper_step_limit": None,
        "fallback_status": "censored_max_steps", "resource_pause_is_success": False,
        "intermediate_region_codes": ["h", "s", "e"], "minimum_region_stability_frames": 5,
        "room_boundary_clearance_m": 0.3, "source_placement_offline_only": True,
        "reviewed_landmarks_are_controller_goals": False, "sealed_data_opened": False,
        "actor_imitation_labels": False, "node_labels": "reviewed_simulation_silver",
        "maximum_primary_target_black_fraction": 0.20,
        "roles": dict(Counter(o["role"] for o in origins)), "tasks": len(tasks)})
    for path in [Path(__file__).resolve(), REPO / "experiments/world_model/event_collection_control.py",
                 REPO / "experiments/world_model/server_adapter.py", REPO / "habitat_server/lightnav_habitat/base.py",
                 REPO / "experiments/world_model/viewpoint_branch_collection.py"]:
        provenance[str(path)] = sha256_file(path)
        dest = output / "source_snapshot" / path.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    write_json(output / "input_manifest.json", provenance)
    StepAccounting(output).initialize()
    print(json.dumps({"tasks": len(tasks), "roles": dict(Counter(o["role"] for o in origins))}), flush=True)


def entered_exit(position, start, heading, checkpoint):
    if horizontal_distance(position, checkpoint) <= 0.18:
        return True
    dx, dz = position[0] - start[0], position[2] - start[2]
    along = -dx * math.sin(heading) - dz * math.cos(heading)
    across = dx * math.cos(heading) - dz * math.sin(heading)
    return along >= 1.0 and abs(across) <= 0.45


def native_clearance(env, position, yaw, distance=2.0):
    goal = exit_checkpoint(position, yaw, 0, distance)
    end = env._habitat_env.sim.pathfinder.try_step_no_sliding(np.array(position, dtype=np.float32),
                                                            np.array(goal, dtype=np.float32))
    return horizontal_distance(position, end), float(end[1] - position[1])


def reactive_command(env, obs, position, yaw):
    depth = _depth_m(obs)
    # A stair riser is not an obstacle to this simulated walker. This local
    # native probe is a teacher controller aid, never a model input or target.
    distance, rise = native_clearance(env, position, yaw, 1.0)
    if distance >= 0.8 and abs(rise) >= 0.08:
        return branch_velocity(0), "follow_stairs_local_teacher"
    following = follow_opening("center", depth) if depth is not None else None
    if following is None and distance >= 0.35:
        return branch_velocity(0), "short_local_clearance_teacher"
    return following if following is not None else (None, "no_visible_opening")


def reset_local_recovery(state, position):
    anchor = state.get("recovery_anchor")
    if anchor is not None and np.linalg.norm(np.array(position) - np.array(anchor)) >= 0.7:
        state.update(recovery_count=0, recovery_anchor=position)


def rotation_coefficients(env):
    from habitat_sim.utils.common import quat_to_coeffs

    return quat_to_coeffs(env._habitat_env.sim.get_agent_state().rotation).tolist()


def collect_one(env, task, folder, root, landmarks, max_steps):
    origin, exit_ = task["origin"], task["exit"]
    accounting = StepAccounting(root)
    prior = read_jsonl(folder / "committed.jsonl")
    progress_path = folder / "progress.json"
    previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    attempt = previous.get("attempt", -1) + 1
    record = {"task_id": task["task_id"], "origin_id": origin["origin_id"], "role": origin["role"],
        "scene": origin["scene"], "exit_id": exit_["exit_id"], "repeat_index": task["repeat_index"],
        "policy_id": POLICY, "evidence_dir": str(folder), "status": "running", "event_complete": False,
        "branch_steps": len(prior), "maximum_branch_steps": max_steps, "attempt": attempt,
        "restore_overhead_steps": previous.get("restore_overhead_steps", 0), "source_restored": False,
        "actor_imitation_label": False, "LightNav_used": False, "checkpoint_reached": False}
    record["physical_steps_this_attempt"] = 0
    atomic_json(progress_path, record)
    try:
        restore_dir = folder if not (folder / "source_history.jsonl").exists() else folder / f"restore_{attempt:03d}"
        restore_dir.mkdir(parents=True, exist_ok=True)
        obs, info, history = place_and_observe(env, origin, restore_dir, f"{task['task_id']}-a{attempt}")
        record["physical_steps_this_attempt"] += len(history)
        for _ in history:
            accounting.reserve(task["task_id"])
        record["source_restored"] = history[-1]["rgb_sha256"] == origin["expected_rgb_sha256"]
        if not record["source_restored"]:
            raise RuntimeError("source RGB differs from the reviewed source")
        if restore_dir != folder:
            history = read_jsonl(folder / "source_history.jsonl")
            record["restore_overhead_steps"] += 8
        record["source_history_rgb"] = [{"path": h["rgb_path"], "sha256": h["rgb_sha256"]} for h in history]
        node = history[-1]
        record["restored_rgb_sha256"] = node["rgb_sha256"]
        polygons = load_floor_polygons(REPO / "data/scene_datasets/mp3d" / origin["scene"] / f"{origin['scene']}.house")
        monitor = EventProgress(node, polygons, landmarks, origin["origin_id"])
        start, heading = node["position"], node["yaw"] + exit_["bearing_left_rad"]
        checkpoint = exit_checkpoint(start, node["yaw"], exit_["bearing_left_rad"])
        state = {"phase": "entry", "scan_radians": 0.0, "recovery_count": 0, "scan_options": []}
        for row in prior:
            obs, _reward, ended, truncated, info = execute_once(env, row["command"],
                f"{task['task_id']}-a{attempt}-replay-{row['index']}", accounting, task["task_id"])
            record["physical_steps_this_attempt"] += 1
            record["restore_overhead_steps"] += 1
            if ended or truncated:
                raise RuntimeError("native episode ended during restoration")
            digest = hashlib.sha256(np.ascontiguousarray(obs["rgb"]).tobytes()).hexdigest()
            if digest != row["rgb_pixels_sha256"]:
                raise RuntimeError(f"deterministic replay RGB mismatch at {row['index']}")
            monitor.push(row["frame"], row["entrance_committed"])
            state = row["controller_after"]
            record["checkpoint_reached"] = row["entrance_committed"]
        if prior:
            record["resume_method"] = "verified_full_action_replay"
        position, yaw = _pose(info)
        recent_positions = deque(maxlen=60)
        for row in prior[-60:]:
            recent_positions.append(row["frame"]["position"])
        while len(prior) < max_steps:
            if (root / "PAUSE").exists() or shutil.disk_usage(folder).free < 8 * 1024**3:
                raise ResourcePause("requested pause or low disk; committed trajectory can be resumed")
            record["checkpoint_reached"] |= entered_exit(position, start, heading, checkpoint)
            reset_local_recovery(state, position)
            if record["checkpoint_reached"] and state["phase"] == "entry":
                state["phase"] = "follow"
            stagnant = (len(recent_positions) == 60 and
                        max(np.linalg.norm(np.array(p) - np.array(position)) for p in recent_positions) < 0.08)
            if state["phase"] == "escape" and stagnant:
                record["status"] = "controller_stalled"
                break
            command, mode = (checkpoint_command(position, yaw, checkpoint), "entry_checkpoint") if state["phase"] == "entry" else reactive_command(env, obs, position, yaw)
            if state["phase"] not in {"scan", "escape"} and (stagnant or command is None):
                if state["recovery_count"] >= 2 and stagnant:
                    record["status"] = "controller_stalled"
                    break
                state.update(phase="scan", scan_radians=0.0, scan_options=[],
                             recovery_count=state["recovery_count"] + 1, scan_previous_yaw=yaw,
                             recovery_anchor=position)
            if state["phase"] == "scan":
                clear, rise = native_clearance(env, position, yaw)
                if clear >= 0.6 and (record["checkpoint_reached"] or abs(wrap_pi(yaw - heading)) < math.radians(35)):
                    state["scan_options"].append({"yaw": yaw, "clearance_m": clear, "rise_m": rise})
                if state["scan_radians"] >= 2 * math.pi - 0.03:
                    if not state["scan_options"]:
                        record["status"] = "blocked_after_full_observation"
                        break
                    best = max(state["scan_options"], key=lambda x: x["clearance_m"] - 0.12 * abs(wrap_pi(x["yaw"] - heading)))
                    state.update(phase="escape", escape_yaw=best["yaw"], escape_start=position)
                    recent_positions.clear()
                else:
                    command, mode = turn_command(math.pi / 2), "recovery_full_observation"
            if state["phase"] == "escape":
                if horizontal_distance(position, state["escape_start"]) >= 0.7:
                    state["phase"] = "follow" if record["checkpoint_reached"] else "entry"
                    recent_positions.clear()
                command, mode = branch_velocity(wrap_pi(state["escape_yaw"] - yaw)), "recovery_escape"
            index = len(prior) + 1
            pending = {"index": index, "command": command, "mode": mode}
            atomic_json(folder / "pending_action.json", pending)
            obs, _reward, terminated, truncated, info = execute_once(env, command,
                f"{task['task_id']}-a{attempt}-branch-{index}", accounting, task["task_id"])
            record["physical_steps_this_attempt"] += 1
            position, new_yaw = _pose(info)
            if state["phase"] == "scan":
                state["scan_radians"] += abs(wrap_pi(new_yaw - yaw))
            yaw = new_yaw
            recent_positions.append(position)
            record["checkpoint_reached"] |= entered_exit(position, start, heading, checkpoint)
            frame = _save_frame(folder, f"branch-{index:05d}", obs, info)
            frame["rotation_xyzw"] = rotation_coefficients(env)
            row = {**pending, "frame": frame, "entrance_committed": record["checkpoint_reached"],
                   "rgb_pixels_sha256": hashlib.sha256(np.ascontiguousarray(obs["rgb"]).tobytes()).hexdigest(),
                   "controller_after": json.loads(json.dumps(state))}
            commit(folder / "committed.jsonl", row)
            prior.append(row)
            record["branch_steps"] = index
            event = monitor.push(frame, record["checkpoint_reached"])
            record.update(path_m=monitor.path_m, intermediate_regions=monitor.region_events,
                          controller=state, last_frame=frame)
            if event:
                quality = image_quality(obs["rgb"], CurationRules(maximum_black_fraction=0.20))
                record.update(status="complete" if quality["usable"] else "target_image_quarantined",
                    event_complete=quality["usable"], event=event, target_kind=event["kind"],
                    target_index=index, source_region=event["source_region"], target_region=event["target_region"],
                    horizon_s=event["control_horizon_s"], target_rgb={"path": str(folder / "assets" / f"{frame['name']}_rgb.npy"),
                                                                              "sha256": frame["rgb_sha256"]})
                break
            if terminated or truncated:
                raise RuntimeError("unexpected native termination before logical branch limit")
            atomic_json(progress_path, record)
        else:
            record["status"] = "censored_max_steps"
        record["resumable"] = record["status"] in {"censored_max_steps", "controller_stalled", "blocked_after_full_observation"}
    except ResourcePause as exc:
        record.update(status="paused_resource", resumable=True, error=str(exc))
    except Exception as exc:
        record.update(status="error", resumable=True, error=f"{type(exc).__name__}: {exc}")
        (folder / f"error_{attempt:03d}.txt").write_text(traceback.format_exc())
    finally:
        record["env_steps"] = 8 + record["branch_steps"] + record["restore_overhead_steps"]
        atomic_json(progress_path, record)
        atomic_json(folder / "result.json", record)
    return record


def worker(output, worker_id, workers, task_ids=None, resume=False):
    root = output.resolve()
    tasks = read_jsonl(root / "tasks.jsonl")
    if task_ids:
        tasks = [t for t in tasks if t["task_id"] in task_ids]
    scenes = sorted({t["origin"]["scene"] for t in tasks})[worker_id::workers]
    origins = json.loads((root / "landmarks.json").read_text())
    maximum = json.loads((root / "protocol.json").read_text())["maximum_branch_steps"]
    for scene in scenes:
        group = [t for t in tasks if t["origin"]["scene"] == scene]
        env = make_env(group[0]["origin"], None, event_driven=True)
        try:
            for task in group:
                folder = root / "executions" / task["task_id"]
                result_path = folder / "result.json"
                if result_path.exists():
                    status = json.loads(result_path.read_text())["status"]
                    if not resume or status != "paused_resource":
                        continue
                folder.mkdir(parents=True, exist_ok=True)
                result = collect_one(env, task, folder, root, [o for o in origins if o["scene"] == scene], maximum)
                print(json.dumps({k: result.get(k) for k in ("task_id", "scene", "status", "branch_steps", "target_kind", "source_restored")}), flush=True)
                if result["status"] == "paused_resource":
                    return
        finally:
            env.close()
