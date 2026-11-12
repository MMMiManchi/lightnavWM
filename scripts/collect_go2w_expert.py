#!/usr/bin/env python3
"""Collect Go2-W expert labels that the trainer can read.

Each sample is one step of an episode the Go2-W body can actually route:

* frames already seen, saved as JPEGs
* the expert action as the same three ``<act_l*>`` tokens the checkpoint emits

The expert follows the navmesh shortest path. The stop token is used only when
the remaining geodesic distance is within 1 m, or when the agent has already
been that close and is moving away. The 3 m success radius is not a stop label.
Episodes whose geodesic distance is infinite (the body does not fit a route)
are skipped.

``--mode expert`` drives the expert and writes ``samples.jsonl``.
``--mode serve`` exposes the same labels over ZMQ so a model client can record
a correction wherever its action disagrees.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# This file is run with the Habitat 3.9 interpreter. Keep the import on `src/`.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "habitat_server"))

from lightnav_habitat.remote_server import RemoteEnvServer  # noqa: E402
from lightnav_habitat.vlnce import VLNCEEnv, episode_allow_key  # noqa: E402

from lightnav.expert import (  # noqa: E402
    DEFAULT_STALL_IMPROVE_M,
    DEFAULT_STALL_PATIENCE,
    DEFAULT_STEP_M,
    ego_waypoints,
    expert_codes,
    face_path_waypoints,
    sample_polyline,
    should_stop,
)
from lightnav.traj_vocab import load_rvq_bundle  # noqa: E402
from lightnav.velocity import first_waypoint_to_velocity_cmd  # noqa: E402
from lightnav.vln_utils import rvq_action_token  # noqa: E402

LOG = logging.getLogger("collect_go2w_expert")


def _finite_allowlist(results_path: Path, *, failures_only: bool) -> set[tuple[str, str]]:
    allow: set[tuple[str, str]] = set()
    with results_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            distance = row.get("final_distance", float("nan"))
            if not math.isfinite(distance):
                continue
            if failures_only and row.get("success", 0) >= 0.5:
                continue
            allow.add(episode_allow_key(row["scene_id"], row["habitat_episode_id"]))
    return allow


def _shard_allowlist(allow: set[tuple[str, str]], shard: int, shards: int) -> set[tuple[str, str]]:
    """Stable stride split so two collectors walk disjoint episodes."""
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError(f"shard {shard} out of range for {shards} shards")
    if shards == 1:
        return allow
    ordered = sorted(allow)
    return set(ordered[shard::shards])


def _agent_frame(rotation) -> tuple[np.ndarray, np.ndarray, float]:
    """Habitat rotation -> (forward, left, yaw). +yaw is a left turn, facing -Z at 0."""
    from habitat_sim.utils.common import quat_rotate_vector

    forward = np.asarray(quat_rotate_vector(rotation, np.array([0.0, 0.0, -1.0])), dtype=np.float64)
    left = np.asarray(quat_rotate_vector(rotation, np.array([-1.0, 0.0, 0.0])), dtype=np.float64)
    forward[1] = 0.0
    left[1] = 0.0
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    left /= max(float(np.linalg.norm(left)), 1e-8)
    yaw = math.atan2(-forward[0], -forward[2])
    return forward, left, yaw


def _goal_position(env: VLNCEEnv) -> np.ndarray:
    goal = env._habitat_env.current_episode.goals[0].position
    return np.asarray(goal, dtype=np.float64).reshape(3)


def _shortest_points(env: VLNCEEnv, position: np.ndarray, goal: np.ndarray) -> list[np.ndarray] | None:
    sim = env._habitat_env.sim
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = position
    path.requested_end = goal
    if not sim.pathfinder.find_path(path) or len(path.points) < 2:
        snapped = sim.pathfinder.snap_point(position)
        path.requested_start = snapped
        if not sim.pathfinder.find_path(path) or len(path.points) < 2:
            return None
    return [np.asarray(point, dtype=np.float64).reshape(3) for point in path.points]


class ExpertEnv:
    """VLN-CE env that attaches an expert label for the current observation."""

    def __init__(self, env: VLNCEEnv, bundle, stop_within_m: float, step_m: float):
        self.env = env
        self.bundle = bundle
        self.stop_within_m = stop_within_m
        self.step_m = step_m
        self.min_distance = float("inf")

    def initialize(self) -> None:
        self.env.initialize()

    def close(self) -> None:
        self.env.close()

    def _label(self, info: dict) -> dict:
        distance = float(info.get("distance_to_goal", float("inf")))
        if math.isfinite(distance):
            self.min_distance = min(self.min_distance, distance)
        state = self.env._habitat_env.sim.get_agent_state()
        position = np.asarray(state.position, dtype=np.float64).reshape(3)
        goal = _goal_position(self.env)
        if not math.isfinite(distance):
            return {"ok": False, "reason": "infinite_geodesic", "distance_m": distance}
        forward, left, yaw = _agent_frame(state.rotation)
        stop = should_stop(distance, self.min_distance, stop_within_m=self.stop_within_m)
        waypoints = None
        if not stop:
            points = _shortest_points(self.env, position, goal)
            if points is None:
                return {"ok": False, "reason": "no_path", "distance_m": distance}
            samples = sample_polyline(points, step_m=self.step_m, horizon=self.bundle.horizon)
            waypoints = face_path_waypoints(ego_waypoints(position, forward, left, yaw, samples))
        codes = expert_codes(
            self.bundle,
            distance,
            self.min_distance,
            waypoints,
            stop_within_m=self.stop_within_m,
        )
        tokens = "".join(rvq_action_token(level, code) for level, code in enumerate(codes))
        return {
            "ok": True,
            "stop": bool(self.bundle.is_stop(codes)),
            "codes": codes,
            "action": tokens,
            "distance_m": distance,
            "min_distance_m": self.min_distance,
            "waypoints": None if waypoints is None else waypoints.tolist(),
        }

    def reset(self, *, seed=None, options=None):
        self.min_distance = float("inf")
        obs, info = self.env.reset(seed=seed, options=options)
        info["expert"] = self._label(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["expert"] = self._label(info)
        return obs, reward, terminated, truncated, info

    def expert_action(self, label: dict) -> dict:
        """Velocity command that executes the expert label for one control step."""
        dt = float(self.env.habitat_time_step or 0.1)
        lin = tuple(self.env.lin_vel_range)
        ang = tuple(self.env.ang_vel_range)
        if label.get("stop") or not label.get("ok") or not label.get("waypoints"):
            waypoint = np.zeros(3, dtype=np.float32)
        else:
            # Drive the geometric chunk. The stored answer is still its RVQ
            # token, but a backward token decodes to a no-op on this body.
            waypoint = _select_drive_waypoint(np.asarray(label["waypoints"], dtype=np.float32))
        return first_waypoint_to_velocity_cmd(waypoint, dt, lin, ang)


def _select_drive_waypoint(waypoints: np.ndarray) -> np.ndarray:
    """First waypoint with forward or yaw motion; a pure-lateral row cannot drive."""
    for waypoint in waypoints:
        if abs(float(waypoint[0])) > 1e-6 or abs(float(waypoint[2])) > 1e-6:
            return np.asarray(waypoint, dtype=np.float32)
    return np.asarray(waypoints[0], dtype=np.float32)


def _write_jpeg(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80]):
        raise RuntimeError(f"failed to write {path}")


def _done_episodes(samples_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not samples_path.exists():
        return done
    with samples_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("episode_done"):
                done.add((row["scene"], str(row["episode_id"])))
    return done


def _episode_key(info: dict) -> tuple[str, str]:
    return episode_allow_key(info.get("scene_id", ""), info.get("episode_id", ""))


def collect_expert(expert: ExpertEnv, out_dir: Path, max_steps: int, limit: int) -> None:
    samples_path = out_dir / "samples.jsonl"
    frame_root = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = _done_episodes(samples_path)
    seen: set[tuple[str, str]] = set()
    n_saved = 0
    n_eps = 0
    with samples_path.open("a") as handle:
        while True:
            obs, info = expert.reset()
            key = _episode_key(info)
            if key in seen:
                break
            seen.add(key)
            scene, episode_id = key
            if key in done:
                continue
            label = info["expert"]
            if not label.get("ok"):
                handle.write(json.dumps({
                    "scene": scene,
                    "episode_id": episode_id,
                    "source": "expert",
                    "episode_done": True,
                    "skipped": label.get("reason", "unusable"),
                }) + "\n")
                handle.flush()
                LOG.info("skip %s/%s %s", scene, episode_id, label.get("reason"))
                continue
            ep_dir = frame_root / f"{scene}_{episode_id}"
            if ep_dir.exists():
                shutil.rmtree(ep_dir)
            instruction = obs["instruction"]["text"] if isinstance(obs["instruction"], dict) else str(obs["instruction"])
            if limit > 0 and n_eps >= limit:
                break
            n_eps += 1
            records = []
            still = 0
            previous = None
            stuck = False
            stalled = False
            best = float("inf")
            best_step = 0
            for step in range(max_steps):
                if not label.get("ok"):
                    break
                distance = float(label["distance_m"])
                if distance < best - DEFAULT_STALL_IMPROVE_M:
                    best = distance
                    best_step = step
                elif step - best_step > DEFAULT_STALL_PATIENCE and not label.get("stop"):
                    stalled = True
                    break
                frame_path = ep_dir / f"{step:04d}.jpg"
                _write_jpeg(frame_path, obs["rgb"])
                record = {
                    "source": "expert",
                    "scene": scene,
                    "episode_id": episode_id,
                    "instruction": instruction,
                    "step": step,
                    "frame_dir": str(ep_dir.relative_to(out_dir)),
                    "stop": label["stop"],
                    "codes": label["codes"],
                    "action": label["action"],
                    "distance_m": label["distance_m"],
                    "min_distance_m": label["min_distance_m"],
                    "episode_done": False,
                }
                signature = (round(float(label["distance_m"]), 3), tuple(label["codes"]))
                still = still + 1 if signature == previous else 0
                previous = signature
                action = expert.expert_action(label)
                obs, _reward, terminated, truncated, info = expert.step(action)
                label = info["expert"]
                finished = bool(terminated or truncated or record["stop"])
                record["episode_done"] = finished
                records.append(record)
                # A turn-in-place keeps the geodesic fixed but changes the token.
                # Twenty identical steps means the body never moved.
                if still >= 20:
                    stuck = True
                    break
                if finished:
                    break
            reached_cap = bool(records) and records[-1]["step"] >= max_steps - 1
            if records and not stuck and (records[-1]["episode_done"] or stalled or reached_cap):
                records[-1]["episode_done"] = True
                for record in records:
                    handle.write(json.dumps(record) + "\n")
                n_saved += len(records)
                LOG.info(
                    "expert %s/%s steps=%d saved_total=%d",
                    scene, episode_id, records[-1]["step"] + 1, n_saved,
                )
            else:
                if ep_dir.exists():
                    shutil.rmtree(ep_dir)
                reason = "stuck" if stuck else "incomplete"
                handle.write(json.dumps({
                    "scene": scene,
                    "episode_id": episode_id,
                    "source": "expert",
                    "episode_done": True,
                    "skipped": reason,
                }) + "\n")
                LOG.info("skip %s/%s %s", scene, episode_id, reason)
            handle.flush()
    LOG.info("expert collection finished episodes=%d samples=%d", n_eps, n_saved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("expert", "serve"), default="expert")
    parser.add_argument("--config", default="habitat_server/configs/vlnce_r2r_go2w.yaml")
    parser.add_argument("--results", default="output/r2r_go2w/results.jsonl")
    parser.add_argument("--bundle", default="checkpoints/LightNav-0/action_tokenizer")
    parser.add_argument("--out", default="output/go2w_expert")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--port", type=int, default=5666)
    parser.add_argument("--ready-file", default="",
                        help="Touched after the simulator is up, so a client can wait on it.")
    parser.add_argument("--stop-within-m", type=float, default=1.0)
    parser.add_argument("--step-m", type=float, default=DEFAULT_STEP_M)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--failures-only", action="store_true",
                        help="Serve or walk only finite-distance episodes the Go2-W eval failed.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many walked episodes (0 = all).")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.chdir(_ROOT)
    allow = _shard_allowlist(
        _finite_allowlist(Path(args.results), failures_only=args.failures_only),
        args.shard,
        args.shards,
    )
    LOG.info(
        "allowlist %d episodes failures_only=%s shard %d/%d",
        len(allow), args.failures_only, args.shard, args.shards,
    )
    bundle = load_rvq_bundle(args.bundle, horizon=10)
    env = VLNCEEnv(
        config_path=args.config,
        gpu_id=args.gpu_id,
        max_steps=args.max_steps,
        episode_allow=allow,
    )
    expert = ExpertEnv(env, bundle, args.stop_within_m, args.step_m)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "embodiment": "go2w",
        "body_height_m": 0.50,
        "body_radius_m": 0.215,
        "camera_height_m": 0.50,
        "image": [480, 270],
        "hfov_deg": 120,
        "stop_within_m": args.stop_within_m,
        "step_m": args.step_m,
        "horizon": 10,
        "prompt_style": "unified_traj",
        "action_tokens": "<act_l{level}_{code}>",
        "skipped": "infinite geodesic or no navmesh path",
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if args.mode == "expert":
        try:
            collect_expert(expert, out_dir, args.max_steps, args.limit)
        finally:
            expert.close()
        return
    server = RemoteEnvServer(expert, address=f"tcp://*:{args.port}")
    LOG.info("expert server on port %d", args.port)
    try:
        server.start(ready_file=args.ready_file or None)
    finally:
        expert.close()


if __name__ == "__main__":
    main()
