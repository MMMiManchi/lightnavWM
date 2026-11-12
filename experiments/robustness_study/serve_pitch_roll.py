#!/usr/bin/env python3
"""Serve R2R with per-step camera pitch and roll jitter.

Before the image is returned, pitch and roll are drawn independently from
Uniform(-1, 1) degrees and applied to the camera sensors. The agent body
pose is restored to the sensor spec afterwards, so the robot position,
heading and the action that was just executed are unchanged.

This is a diagnostic server. The official eval path is not modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "habitat_server"))

from lightnav_habitat.remote_server import RemoteEnvServer  # noqa: E402
from lightnav_habitat.vlnce import VLNCEEnv  # noqa: E402


class PitchRollJitter:
    """Re-render RGB after a small camera pitch and roll. Body pose stays put."""

    def __init__(self, env: VLNCEEnv, seed: int):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.draws = 0
        self.max_body_shift_m = 0.0
        self._logged_effect = False

    def initialize(self) -> None:
        self.env.initialize()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs, pitch, roll, shift = self._rerender(obs)
        info = dict(info)
        info["camera_pitch_deg"] = pitch
        info["camera_roll_deg"] = roll
        info["camera_body_shift_m"] = shift
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs, pitch, roll, shift = self._rerender(obs)
        info = dict(info)
        info["camera_pitch_deg"] = pitch
        info["camera_roll_deg"] = roll
        info["camera_body_shift_m"] = shift
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        logging.info(
            "pitch/roll draws %d, max body shift %.3e m",
            self.draws,
            self.max_body_shift_m,
        )
        self.env.close()

    def _rerender(self, obs: dict):
        import magnum as mn

        sim = self.env._habitat_env.sim
        agent = sim.get_agent(0)
        body = agent.scene_node
        position_before = np.array(body.absolute_translation, dtype=np.float64)
        rotation_before = _quat_xyzw(body.rotation)
        pitch = float(self.rng.uniform(-1.0, 1.0))
        roll = float(self.rng.uniform(-1.0, 1.0))
        sensors = list(agent._sensors.values())
        if not sensors:
            raise RuntimeError("agent has no sensors to jitter")
        for sensor in sensors:
            sensor.set_transformation_from_spec()
            sensor.node.rotate_x_local(mn.Deg(pitch))
            sensor.node.rotate_z_local(mn.Deg(roll))
        rendered = sim.sensor_suite.get_observations(sim.get_sensor_observations())
        for sensor in sensors:
            sensor.set_transformation_from_spec()
        position_after = np.array(body.absolute_translation, dtype=np.float64)
        rotation_after = _quat_xyzw(body.rotation)
        shift = float(np.linalg.norm(position_after - position_before))
        rot_delta = float(np.linalg.norm(np.subtract(rotation_after, rotation_before)))
        if shift > 1e-4 or rot_delta > 1e-4:
            raise RuntimeError(f"camera jitter moved the body: shift {shift} rot {rot_delta}")
        self.max_body_shift_m = max(self.max_body_shift_m, shift)
        rgb = np.asarray(rendered["rgb"])
        if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
            raise RuntimeError(f"unexpected rgb shape {rgb.shape}")
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        original = np.asarray(obs["rgb"])
        if original.shape[-1] == 4:
            original = original[..., :3]
        if not self._logged_effect and original.shape == rgb.shape:
            delta = float(np.mean(np.abs(original.astype(np.int16) - rgb.astype(np.int16))))
            logging.info(
                "first jitter pitch %.3f roll %.3f mean abs pixel delta %.3f",
                pitch,
                roll,
                delta,
            )
            self._logged_effect = True
        obs = dict(obs)
        obs["rgb"] = np.ascontiguousarray(rgb)
        self.draws += 1
        return obs, round(pitch, 4), round(roll, 4), shift


def _quat_xyzw(rotation) -> tuple[float, float, float, float]:
    vector = rotation.vector
    return (float(vector.x), float(vector.y), float(vector.z), float(rotation.scalar))


def smoke(env: PitchRollEnv) -> None:
    import magnum as mn

    obs, info = env.env.reset()
    sim = env.env._habitat_env.sim
    agent = sim.get_agent(0)
    base = np.asarray(obs["rgb"])
    if base.shape[-1] == 4:
        base = base[..., :3]
    sensor = next(iter(agent._sensors.values()))
    sensor.set_transformation_from_spec()
    sensor.node.rotate_x_local(mn.Deg(1.0))
    pitched = sim.sensor_suite.get_observations(sim.get_sensor_observations())["rgb"]
    sensor.set_transformation_from_spec()
    sensor.node.rotate_z_local(mn.Deg(1.0))
    rolled = sim.sensor_suite.get_observations(sim.get_sensor_observations())["rgb"]
    sensor.set_transformation_from_spec()
    for name, image in (("pitch1", pitched), ("roll1", rolled)):
        image = np.asarray(image)
        if image.shape[-1] == 4:
            image = image[..., :3]
        delta = np.abs(base.astype(np.int16) - image.astype(np.int16))
        print(
            f"{name} shape {image.shape} mean_abs {delta.mean():.3f} "
            f"frac_changed {(delta.sum(axis=2) > 0).mean():.3f}",
            flush=True,
        )
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow", required=True)
    parser.add_argument("--config", default="habitat_server/configs/vlnce_r2r.yaml")
    parser.add_argument("--port", type=int, default=5681)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    keys = {
        (row["scene"], str(row["episode_id"]))
        for row in json.loads(Path(args.allow).read_text())
    }
    logging.info("allow %d episodes, jitter seed %d", len(keys), args.seed)
    inner = VLNCEEnv(
        config_path=args.config,
        gpu_id=args.gpu_id,
        max_steps=500,
        episode_allow=keys,
    )
    wrapped = PitchRollJitter(inner, args.seed)
    if args.smoke:
        smoke(wrapped)
        return
    ready = Path(args.ready_file)
    if ready.exists():
        ready.unlink()
    server = RemoteEnvServer(wrapped, address=f"tcp://*:{args.port}")
    try:
        server.start(ready_file=str(ready))
    finally:
        wrapped.close()


if __name__ == "__main__":
    main()
