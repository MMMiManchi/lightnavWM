"""Replay saved smoke controls to inspect an endpoint that was not logged.

Run with Habitat's Python 3.9. This does not rerun the policy and cannot turn an
unlogged original endpoint into an original measurement. Poses/goal distances
are diagnostic outputs only; they do not influence any action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=1)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 9):
        parser.error("run this replay with the installed Habitat Python 3.9")
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo), str(repo / "habitat_server")]
    import numpy as np
    from lightnav_habitat import vlnce_extensions  # noqa: F401
    from lightnav_habitat.vlnce import VLNCEEnv
    from PIL import Image

    source = args.source_smoke.resolve()
    manifest = json.loads((source / "source_manifest.json").read_text())
    config = repo / "experiments/world_model/configs/collect_train_go2w_radius010.yaml"
    if hashlib.sha256(config.read_bytes()).hexdigest() != manifest[str(config)]:
        raise ValueError("simulator configuration differs from the original smoke")
    previous = json.loads((source / "result.json").read_text())
    trace_path = source / "live_trace.json"
    trace = json.loads(trace_path.read_text())
    if len(trace) != previous["live"]["steps_executed"] or not 1 <= len(trace) <= 60:
        raise ValueError("saved control count differs from the reported real steps")
    for i, row in enumerate(trace):
        if row["step"] != i or not row.get("environment_step_returned") or row.get("action") is None:
            raise ValueError("trace is not a complete, actually dispatched control sequence")
    query = previous["query"].split("|")
    scene, episode_id = query[-3], query[-2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = VLNCEEnv(config_path=str(config), split="train", gpu_id=args.gpu_id,
                   max_steps=60, early_stop_rotation=0, early_stop_steps=0,
                   episode_allow={(scene, episode_id)})
    states = []

    def capture(obs, info, index):
        state = env._habitat_env.sim.get_agent_state()
        rotation = state.rotation
        rgb = np.asarray(obs["rgb"])
        record = {"after_actions": index, "position_xyz_m": np.asarray(state.position).tolist(),
                  "rotation_xyzw": [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)],
                  "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                  "distance_to_goal_m": float(info["distance_to_goal"]) if "distance_to_goal" in info else None,
                  "success": bool(info["success"]) if "success" in info else None}
        if index in (0, 40, 42, len(trace)):
            Image.fromarray(rgb).save(output / ("final.png" if index == len(trace) else f"after_{index:03d}.png"))
        states.append(record)

    try:
        obs, info = env.reset()
        if str(info.get("episode_id")) != episode_id:
            raise ValueError("replay server loaded another episode")
        instruction = obs.get("instruction")
        capture(obs, info, 0)
        terminated = truncated = False
        for i, row in enumerate(trace):
            obs, _reward, terminated, truncated, info = env.step(row["action"])
            capture(obs, info, i + 1)
            if (terminated or truncated) and i + 1 != len(trace):
                raise ValueError("replay ended before all original controls were applied")
        goal = env._habitat_env.current_episode.goals[0].position
        result = {"method": "open-loop replay of exactly the saved 55 environment controls; not a new model decision run",
                  "original_endpoint_was_not_logged": True, "source_smoke": str(source),
                  "source_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                  "same_config_hash_verified": True, "scene": scene, "episode_id": episode_id,
                  "original_instruction": instruction, "actions_replayed": len(trace),
                  "initial": states[0], "final": states[-1], "annotated_goal_position_xyz_m": list(goal),
                  "terminated": bool(terminated), "truncated": bool(truncated),
                  "replay_endpoint_is_not_an_original_logged_measurement": True,
                  "pose_and_goal_used_only_after_action_for_diagnostics": True}
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        (output / "state_trace.json").write_text(json.dumps(states, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
