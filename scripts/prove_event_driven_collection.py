#!/usr/bin/env python
"""Native Habitat regression: step 501 remains executable in collection mode."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "habitat_server"))

from experiments.world_model.branch_benchmark_entry import CONFIG, DATASET, INVENTORY, write_json
from experiments.world_model.event_collection_control import execute_once
from experiments.world_model.serve_collect import prepare_server_plan
from experiments.world_model.server_adapter import WorldModelVLNCEEnv
from experiments.world_model.viewpoint_branch_collection import reset_singleton_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidate = next(json.loads(line) for line in (args.previews / "preview_tasks.jsonl").read_text().splitlines()
                     if json.loads(line)["role"] == "fit")
    payload, order = prepare_server_plan(Path(candidate["allowlist"]), DATASET, INVENTORY, partition="train_fit")
    env = WorldModelVLNCEEnv(config_path=str(CONFIG), split="train", data_path=str(DATASET), gpu_id=0,
        event_driven=True, episode_allow=set(order), allow_order=order, requested_seed=11,
        expected_dataset_sha256=payload["dataset_sha256"], early_stop_rotation=0, early_stop_steps=0)
    try:
        env.reset(seed=11)
        native = env._habitat_env
        assert env.max_steps is None
        assert native._max_episode_steps == native._max_episode_seconds == 0
        command = {"action": "velocity_control", "action_args": {"linear_velocity": -1.0, "angular_velocity": 50 / 300}}
        for i in range(512):
            obs, _reward, terminated, truncated, _info = execute_once(env, command, f"native-unlimited-proof-{i}")
            if terminated or truncated:
                raise RuntimeError(f"unexpected end at step {i + 1}: terminated={terminated}, truncated={truncated}")
            if (i + 1) % 128 == 0:
                print(json.dumps({"executed_steps": i + 1, "terminated": terminated, "truncated": truncated}), flush=True)
        from habitat_sim.utils.common import quat_to_coeffs

        state = native.sim.get_agent_state()
        position, rotation = state.position.tolist(), quat_to_coeffs(state.rotation).tolist()
        rgb = obs["rgb"].copy()
        reset_singleton_episode(env)
        assert native.sim.set_agent_state(position, rotation, reset_sensors=True)
        restored_obs = env._build_observation(native.sim.get_sensor_observations())
        exact_snapshot = np.array_equal(rgb, restored_obs["rgb"])
        proof = {"scene": candidate["scene"], "executed_steps": 512, "passed_native_500_step_boundary": True,
                 "wrapper_max_steps": env.max_steps, "native_max_episode_steps": native._max_episode_steps,
                 "native_max_episode_seconds": native._max_episode_seconds, "retry_image_cache_size": len(env._by_id),
                 "snapshot_RGB_exact_after_reset": exact_snapshot, "training_examples": 0,
                 "sealed_data_opened": False}
        write_json(args.output / "proof.json", proof)
        print(json.dumps(proof), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
