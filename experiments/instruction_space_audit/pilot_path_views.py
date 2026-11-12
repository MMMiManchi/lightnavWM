#!/usr/bin/env python3
"""Render reference-path keyframes for the six-route audit pilot.

Rules are fixed in random_audit_protocol.md. This does not roll out a policy.
Run with the Habitat 3.9 interpreter:

    CUDA_VISIBLE_DEVICES=2 /home/szgx/miniconda3/envs/habitat/bin/python \
        experiments/instruction_space_audit/pilot_path_views.py
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import habitat_sim
import imageio.v2 as imageio
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/instruction_space_audit"
DATASET = ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
FRAMES = OUT / "pilot_path_frames"
MID_SEGMENT_M = 3.5


def yaw_of(vec: np.ndarray) -> float:
    return math.degrees(math.atan2(-float(vec[0]), -float(vec[2])))


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    qv = np.array([x, y, z], dtype=float)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def start_yaw(episode: dict) -> float:
    forward = quat_rotate(np.array(episode["start_rotation"], dtype=float), np.array([0.0, 0.0, -1.0]))
    forward[1] = 0.0
    forward /= np.linalg.norm(forward)
    return yaw_of(forward)


def views_for(points: list[np.ndarray], episode: dict) -> list[dict]:
    rows = [
        {
            "frame_id": "start",
            "position": np.array(episode["start_position"], dtype=float),
            "yaw": start_yaw(episode),
            "arc_m": 0.0,
            "segment": -1,
            "use_dataset_quat": True,
        }
    ]
    arc = 0.0
    last_yaw = start_yaw(episode)
    for index, (start, end) in enumerate(zip(points, points[1:])):
        delta = end - start
        flat = delta.copy()
        flat[1] = 0.0
        flat_length = float(np.linalg.norm(flat))
        if flat_length > 1e-4:
            last_yaw = yaw_of(flat / flat_length)
        length = float(np.linalg.norm(delta))
        rows.append(
            {
                "frame_id": f"w{index}",
                "position": start.copy(),
                "yaw": last_yaw,
                "arc_m": arc,
                "segment": index,
                "use_dataset_quat": False,
            }
        )
        if length > MID_SEGMENT_M:
            rows.append(
                {
                    "frame_id": f"w{index}m",
                    "position": start + 0.5 * delta,
                    "yaw": last_yaw,
                    "arc_m": arc + 0.5 * length,
                    "segment": index,
                    "use_dataset_quat": False,
                }
            )
        arc += length
    rows.append(
        {
            "frame_id": "end",
            "position": points[-1].copy(),
            "yaw": last_yaw,
            "arc_m": arc,
            "segment": len(points) - 2,
            "use_dataset_quat": False,
        }
    )
    return rows


def make_sim(scene: str) -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = f"data/scene_datasets/mp3d/{scene}/{scene}.glb"
    sim_cfg.gpu_device_id = 0
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "color"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [270, 480]
    rgb.hfov = 120
    rgb.position = [0.0, 0.5, 0.0]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    agent_cfg.height = 0.5
    agent_cfg.radius = 0.215
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def main() -> None:
    sample = [
        row
        for row in csv.DictReader((OUT / "random_audit_sample.csv").open())
        if int(row["sample_index"]) <= 6
    ]
    with gzip.open(DATASET, "rt", encoding="utf-8") as handle:
        episodes = json.load(handle)["episodes"]
    by_id = {}
    for episode in episodes:
        scene = episode["scene_id"].split("/")[-2]
        by_id[(scene, str(episode["episode_id"]))] = episode

    planned = []
    for row in sample:
        episode = by_id[(row["scene"], row["episode_id_1"])]
        points = [np.asarray(point, dtype=float) for point in episode["reference_path"]]
        for view in views_for(points, episode):
            view["sample_index"] = row["sample_index"]
            view["route_group_id"] = row["route_group_id"]
            view["scene"] = row["scene"]
            view["episode"] = episode
            planned.append(view)

    FRAMES.mkdir(parents=True, exist_ok=True)
    manifest = []
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for view in planned:
        by_scene[view["scene"]].append(view)

    for scene, views in by_scene.items():
        sim = make_sim(scene)
        agent = sim.get_agent(0)
        try:
            for view in views:
                state = agent.get_state()
                state.position = view["position"]
                if view["use_dataset_quat"]:
                    state.rotation = quat_from_coeffs(view["episode"]["start_rotation"])
                else:
                    state.rotation = quat_from_angle_axis(
                        math.radians(view["yaw"]), np.array([0.0, 1.0, 0.0])
                    )
                agent.set_state(state)
                image = sim.get_sensor_observations()["color"][..., :3]
                name = f"{view['sample_index']}_{view['route_group_id']}_{view['frame_id']}.jpg"
                imageio.imwrite(FRAMES / name, image)
                manifest.append(
                    {
                        "sample_index": view["sample_index"],
                        "route_group_id": view["route_group_id"],
                        "scene": scene,
                        "frame_id": view["frame_id"],
                        "file": name,
                        "arc_m": f"{view['arc_m']:.2f}",
                        "segment": view["segment"],
                        "yaw_deg": f"{view['yaw']:.1f}",
                        "x": f"{view['position'][0]:.3f}",
                        "y": f"{view['position'][1]:.3f}",
                        "z": f"{view['position'][2]:.3f}",
                    }
                )
                print("wrote", name, flush=True)
        finally:
            sim.close()

    with (OUT / "pilot_path_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print("frames", len(manifest))


if __name__ == "__main__":
    main()
