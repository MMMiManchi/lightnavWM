#!/usr/bin/env python3
"""Replay one Go2-W allowlist under a single intervention.

Modes
  model        baseline checkpoint, no extra decision
  oracle_stop  same actions, but stop on the first pose whose geodesic is < 3 m
  expert       execute the privileged shortest-path waypoint every step

oracle_stop and expert read simulator distance or the navmesh. They are upper
bounds, not deployable policies. The Go2-W LoRA checkpoint is not part of this
probe. Outputs go to a new directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lightnav.habitat.remote_env import RemoteEnvClient  # noqa: E402
from lightnav.velocity import first_waypoint_to_velocity_cmd  # noqa: E402


def episode_key(info: dict) -> tuple[str, str]:
    scene_id = str(info.get("scene_id", "")).rstrip("/")
    scene = os.path.basename(os.path.dirname(scene_id))
    return scene, str(info.get("episode_id", ""))


def finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def drive_waypoint(waypoints) -> np.ndarray:
    rows = np.asarray(waypoints, dtype=np.float32)
    for row in rows:
        if abs(float(row[0])) > 1e-6 or abs(float(row[2])) > 1e-6:
            return np.asarray(row, dtype=np.float32)
    return np.asarray(rows[0], dtype=np.float32)


def expert_action(info: dict) -> dict:
    label = info.get("expert") or {}
    dt = float(info.get("habitat_time_step") or 0.1)
    lin = tuple(float(x) for x in info["lin_vel_range"])
    ang = tuple(float(x) for x in info["ang_vel_range"])
    if label.get("stop") or not label.get("ok") or not label.get("waypoints"):
        waypoint = np.zeros(3, dtype=np.float32)
    else:
        waypoint = drive_waypoint(label["waypoints"])
    return first_waypoint_to_velocity_cmd(waypoint, dt, lin, ang)


def wrap_angle(delta: float) -> float:
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def heading_error(model_wp, expert_wp) -> float | None:
    if model_wp is None or expert_wp is None:
        return None
    return abs(wrap_angle(float(model_wp[2]) - float(expert_wp[2])))


def load_policy(args, info: dict):
    from lightnav.habitat.runner import (
        HabitatEvalConfig,
        _default_engine_factory,
        build_velocity_policy,
    )

    cfg = HabitatEvalConfig(
        server=args.server,
        model_path=args.model,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_mem,
        max_steps=500,
    )
    engine, bundle = _default_engine_factory(cfg)
    policy = build_velocity_policy(cfg, engine, bundle, info)
    return policy


def instruction_text(info: dict) -> str:
    raw = info.get("instruction")
    if isinstance(raw, dict):
        text = raw.get("instruction_text")
        if isinstance(text, list):
            return " ".join(str(part) for part in text)
        return "" if text is None else str(text)
    return "" if raw is None else str(raw)


def clip_body_step(waypoint, dt: float, lin: tuple[float, float], ang: tuple[float, float]) -> tuple[float, float]:
    """Forward metres and yaw radians the velocity mapper can actually request."""
    forward = float(waypoint[0]) / dt
    yaw_rate_deg = math.degrees(float(waypoint[2])) / dt
    forward = min(max(forward, lin[0]), lin[1])
    yaw_rate_deg = min(max(yaw_rate_deg, ang[0]), ang[1])
    return forward * dt, math.radians(yaw_rate_deg * dt)


def pointing_ids(raw_text: str) -> tuple[int | None, int | None]:
    import re

    apos = re.search(r"<apos_(\d+)>", raw_text or "")
    opos = re.search(r"<opos_(\d+)>", raw_text or "")
    return (int(apos.group(1)) if apos else None, int(opos.group(1)) if opos else None)


def run_trace(args, labels: dict) -> None:
    """Same actions as model mode. Writes one row per inference, then the episode result."""
    import imageio.v3 as iio

    out = Path(args.out)
    steps_path = Path(args.steps)
    frames_root = Path(args.frames)
    out.parent.mkdir(parents=True, exist_ok=True)
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)
    env = RemoteEnvClient(args.server)
    policy = None
    seen: set[tuple[str, str]] = set()
    with out.open("w") as episodes, steps_path.open("w") as steps_file:
        while True:
            obs, info = env.reset()
            key = episode_key(info)
            if key in seen:
                break
            seen.add(key)
            if policy is None:
                policy = load_policy(args, info)
            policy.reset(obs)
            instruction = instruction_text(info)
            frame_dir = frames_root / f"{key[0]}_{key[1]}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            min_distance = finite(info.get("distance_to_goal"))
            n_steps = 0
            t0 = time.perf_counter()
            for step in range(500):
                before = info.get("trace_state") or {}
                distance = finite(info.get("distance_to_goal"))
                if distance is not None:
                    min_distance = distance if min_distance is None else min(min_distance, distance)
                expert = info.get("expert") or {}
                expert_wp = drive_waypoint(expert["waypoints"]) if expert.get("waypoints") else None
                rgb = np.asarray(obs["rgb"])
                frame_file = frame_dir / f"{step:04d}.jpg"
                iio.imwrite(frame_file, rgb)
                t_act = time.perf_counter()
                action = policy.act(obs, info)
                infer_s = time.perf_counter() - t_act
                got = policy.get_info()
                model_wp = got.get("action_waypoint")
                raw_text = got.get("raw_text") or ""
                codes = got.get("rvq_codes")
                predicted = got.get("predicted_traj")
                pointing = got.get("pointing") or {}
                apos_id, opos_id = pointing_ids(raw_text)
                dt = float(info.get("habitat_time_step") or 0.1)
                lin = tuple(float(x) for x in info["lin_vel_range"])
                ang = tuple(float(x) for x in info["ang_vel_range"])
                clipped_forward = None
                clipped_yaw = None
                if model_wp is not None:
                    clipped_forward, clipped_yaw = clip_body_step(model_wp, dt, lin, ang)
                history = len(getattr(policy.agent, "_history_frame_ids", []))
                prompt_tokens = getattr(policy.engine, "_last_prompt_tokens", None)
                obs, _reward, terminated, truncated, info = env.step(action)
                after = info.get("trace_state") or {}
                after_distance = finite(info.get("distance_to_goal"))
                signed_error = None
                if model_wp is not None and expert_wp is not None:
                    signed_error = wrap_angle(float(model_wp[2]) - float(expert_wp[2]))
                row = {
                    "scene": key[0],
                    "episode_id": key[1],
                    "step": step,
                    "instruction": instruction,
                    "frame_file": str(frame_file),
                    "frame_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
                    "history_frames": history,
                    "prompt_tokens": None if prompt_tokens is None else int(prompt_tokens),
                    "raw_text": raw_text,
                    "apos_id": apos_id,
                    "apos_state": pointing.get("apos_state"),
                    "apos_px": pointing.get("apos_px"),
                    "opos_id": opos_id,
                    "opos_state": pointing.get("opos_state"),
                    "opos_px": pointing.get("opos_px"),
                    "pointing_frame": pointing.get("frame_size"),
                    "rvq_codes": codes,
                    "waypoints_robot": None if predicted is None else [
                        [round(float(v), 5) for v in row] for row in np.asarray(predicted)
                    ],
                    "waypoint_frame": "robot_local_forward_m_left_m_yaw_rad_ccw",
                    "executed_waypoint": None if model_wp is None else [round(float(v), 5) for v in model_wp],
                    "lateral_ignored_m": None if model_wp is None else round(float(model_wp[1]), 5),
                    "action_linear": None if action.get("linear_velocity") is None else round(float(action["linear_velocity"]), 5),
                    "action_angular": None if action.get("angular_velocity") is None else round(float(action["angular_velocity"]), 5),
                    "dt_s": dt,
                    "commanded_forward_m": None if model_wp is None else round(float(model_wp[0]), 5),
                    "commanded_yaw_rad": None if model_wp is None else round(float(model_wp[2]), 5),
                    "commanded_forward_clipped_m": None if clipped_forward is None else round(clipped_forward, 5),
                    "commanded_yaw_clipped_rad": None if clipped_yaw is None else round(clipped_yaw, 5),
                    "model_stop": bool(
                        model_wp is not None
                        and abs(float(model_wp[0])) <= 1e-6
                        and abs(float(model_wp[2])) <= 1e-6
                    ),
                    "infer_s": round(infer_s, 4),
                    "geodesic_before_m": distance,
                    "geodesic_after_m": after_distance,
                    "euclidean_horizontal_before_m": before.get("euclidean_horizontal_m"),
                    "euclidean_horizontal_after_m": after.get("euclidean_horizontal_m"),
                    "pose_xyz_before": before.get("xyz"),
                    "pose_yaw_before": before.get("yaw_rad"),
                    "pose_xyz_after": after.get("xyz"),
                    "pose_yaw_after": after.get("yaw_rad"),
                    "expert_stop": bool(expert.get("stop")),
                    "expert_waypoint": None if expert_wp is None else [round(float(v), 5) for v in expert_wp],
                    "expert_yaw_rad": None if expert_wp is None else round(float(expert_wp[2]), 5),
                    "yaw_error_rad": None if signed_error is None else round(signed_error, 5),
                    "collided": after.get("collided"),
                    "collision_source": after.get("collision_source"),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                steps_file.write(json.dumps(row) + "\n")
                steps_file.flush()
                n_steps += 1
                if terminated or truncated:
                    break
            elapsed = time.perf_counter() - t0
            final = finite(info.get("distance_to_goal"))
            if final is not None:
                min_distance = final if min_distance is None else min(min_distance, final)
            episode = {
                "scene": key[0],
                "episode_id": key[1],
                "probe_class": labels.get(key, ""),
                "mode": "trace",
                "instruction": instruction,
                "success": float(info.get("success") or 0) >= 0.5,
                "spl": finite(info.get("spl")),
                "ndtw": finite(info.get("ndtw")),
                "final_distance_m": final,
                "min_distance_m": min_distance,
                "steps": n_steps,
                "elapsed_s": round(elapsed, 2),
            }
            episodes.write(json.dumps(episode) + "\n")
            episodes.flush()
            print(
                f"trace {key[0]}/{key[1]} class={episode['probe_class']} "
                f"success={episode['success']} final={final} steps={n_steps}",
                flush=True,
            )
    env.close()


def wait_ready(path: Path) -> None:
    deadline = time.time() + 3600
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(2)
    raise TimeoutError(f"server did not write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("model", "oracle_stop", "expert", "trace"), required=True)
    parser.add_argument("--server", default="tcp://127.0.0.1:5671")
    parser.add_argument("--ready", required=True)
    parser.add_argument("--model", default="checkpoints/LightNav-0")
    parser.add_argument("--backend", default="vllm_local")
    parser.add_argument("--gpu-mem", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--classes", default="", help="probe_episodes.json, for labels only.")
    parser.add_argument("--steps", default="", help="Per-step jsonl. Used by trace mode.")
    parser.add_argument("--frames", default="", help="Directory for input-frame copies. Used by trace mode.")
    args = parser.parse_args()

    wait_ready(Path(args.ready))
    labels = {}
    if args.classes:
        for row in json.loads(Path(args.classes).read_text()):
            labels[(row["scene"], str(row["episode_id"]))] = row["probe_class"]

    if args.mode == "trace":
        if not args.steps or not args.frames:
            raise SystemExit("trace mode needs --steps and --frames")
        run_trace(args, labels)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = RemoteEnvClient(args.server)
    policy = None
    seen: set[tuple[str, str]] = set()
    with out.open("w") as handle:
        while True:
            obs, info = env.reset()
            key = episode_key(info)
            if key in seen:
                break
            seen.add(key)
            if policy is None and args.mode != "expert":
                policy = load_policy(args, info)
            if policy is not None:
                policy.reset(obs)
            steps = []
            min_distance = finite(info.get("distance_to_goal"))
            oracle_fired = False
            t0 = time.perf_counter()
            for step in range(500):
                distance = finite(info.get("distance_to_goal"))
                if distance is not None:
                    min_distance = distance if min_distance is None else min(min_distance, distance)
                expert = info.get("expert") or {}
                expert_wp = drive_waypoint(expert["waypoints"]) if expert.get("waypoints") else None
                model_wp = None
                raw_text = ""
                codes = None
                if args.mode == "expert":
                    action = expert_action(info)
                    source = "expert"
                elif args.mode == "oracle_stop" and distance is not None and distance < 3.0:
                    action = policy.stop_action()
                    source = "oracle_stop"
                    oracle_fired = True
                else:
                    t_act = time.perf_counter()
                    action = policy.act(obs, info)
                    infer_s = time.perf_counter() - t_act
                    got = policy.get_info()
                    model_wp = got.get("action_waypoint")
                    raw_text = got.get("raw_text") or ""
                    codes = got.get("rvq_codes")
                    source = "model"
                    if step == 0:
                        steps.append({
                            "step": 0,
                            "distance_m": distance,
                            "source": source,
                            "infer_s": round(infer_s, 4),
                            "model_waypoint": None if model_wp is None else [round(float(x), 4) for x in model_wp],
                            "expert_waypoint": None if expert_wp is None else [round(float(x), 4) for x in expert_wp],
                            "heading_error_rad": None if heading_error(model_wp, expert_wp) is None else round(heading_error(model_wp, expert_wp), 4),
                            "expert_stop": bool(expert.get("stop")),
                            "raw_text": raw_text[:400],
                            "rvq_codes": codes,
                        })
                if source != "model" or step != 0:
                    if not steps or steps[-1].get("step") != step:
                        steps.append({
                            "step": step,
                            "distance_m": distance,
                            "source": source,
                            "expert_stop": bool(expert.get("stop")),
                            "model_is_stop": bool(
                                model_wp is not None and abs(float(model_wp[0])) <= 1e-6 and abs(float(model_wp[2])) <= 1e-6
                            ),
                        })
                obs, _reward, terminated, truncated, info = env.step(action)
                if oracle_fired or terminated or truncated:
                    break
            elapsed = time.perf_counter() - t0
            final = finite(info.get("distance_to_goal"))
            if final is not None:
                min_distance = final if min_distance is None else min(min_distance, final)
            row = {
                "scene": key[0],
                "episode_id": key[1],
                "probe_class": labels.get(key, ""),
                "mode": args.mode,
                "success": float(info.get("success") or 0) >= 0.5,
                "spl": finite(info.get("spl")),
                "ndtw": finite(info.get("ndtw")),
                "final_distance_m": final,
                "min_distance_m": min_distance,
                "steps": len(steps),
                "oracle_fired": oracle_fired,
                "elapsed_s": round(elapsed, 2),
                "first_step": steps[0] if steps else None,
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(
                f"{args.mode} {key[0]}/{key[1]} class={row['probe_class']} "
                f"success={row['success']} final={final} steps={row['steps']}",
                flush=True,
            )
    env.close()


if __name__ == "__main__":
    main()
