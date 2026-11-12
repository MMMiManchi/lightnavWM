#!/usr/bin/env python3
"""Serve a fixed R2R allowlist with privileged shortest-path labels.

Run with the Habitat 3.9 interpreter. This does not write the baseline results.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "habitat_server"))


def _load_collector():
    path = ROOT / "scripts" / "collect_go2w_expert.py"
    spec = importlib.util.spec_from_file_location("collect_go2w_expert", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collision(sim):
    """Read a collision flag if this Habitat build exposes one. Never invents it."""
    for name in ("previous_step_collided",):
        if hasattr(sim, name):
            return bool(getattr(sim, name)), name
    try:
        agent = sim.get_agent(0)
    except Exception:
        return None, "unavailable"
    for name in ("collided",):
        if hasattr(agent, name):
            return bool(getattr(agent, name)), f"agent.{name}"
    return None, "unavailable"


class PoseEnv:
    """Attach the current pose to info. Does not change the action the caller sends."""

    def __init__(self, expert, collector):
        self.expert = expert
        self.collector = collector

    def close(self) -> None:
        self.expert.close()

    def reset(self, *, seed=None, options=None):
        obs, info = self.expert.reset(seed=seed, options=options)
        info["trace_state"] = self._state()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.expert.step(action)
        info["trace_state"] = self._state()
        return obs, reward, terminated, truncated, info

    def _state(self) -> dict:
        import numpy as np

        vln = self.expert.env
        sim = vln._habitat_env.sim
        agent_state = sim.get_agent_state()
        position = np.asarray(agent_state.position, dtype=np.float64).reshape(3)
        _forward, _left, yaw = self.collector._agent_frame(agent_state.rotation)
        goal = self.collector._goal_position(vln)
        flat = position.copy()
        flat[1] = 0.0
        goal_flat = goal.copy()
        goal_flat[1] = 0.0
        collided, source = _collision(sim)
        return {
            "xyz": [round(float(v), 5) for v in position],
            "yaw_rad": round(float(yaw), 6),
            "goal_xyz": [round(float(v), 5) for v in goal],
            "euclidean_horizontal_m": round(float(np.linalg.norm(flat - goal_flat)), 5),
            "collided": collided,
            "collision_source": source,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow", required=True, help="JSON list of {scene, episode_id}.")
    parser.add_argument("--config", default="habitat_server/configs/vlnce_r2r_go2w.yaml")
    parser.add_argument("--port", type=int, default=5671)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--ready-file", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collector = _load_collector()
    keys = {
        (row["scene"], str(row["episode_id"]))
        for row in json.loads(Path(args.allow).read_text())
    }
    logging.info("probe allowlist %d episodes", len(keys))
    bundle = collector.load_rvq_bundle("checkpoints/LightNav-0/action_tokenizer", horizon=10)
    env = collector.VLNCEEnv(
        config_path=args.config,
        gpu_id=args.gpu_id,
        max_steps=500,
        episode_allow=keys,
    )
    expert = collector.ExpertEnv(env, bundle, 1.0, collector.DEFAULT_STEP_M)
    served = PoseEnv(expert, collector)
    ready = Path(args.ready_file)
    if ready.exists():
        ready.unlink()
    server = collector.RemoteEnvServer(served, address=f"tcp://*:{args.port}")
    try:
        server.start(ready_file=str(ready))
    finally:
        served.close()


if __name__ == "__main__":
    main()
