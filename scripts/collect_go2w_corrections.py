#!/usr/bin/env python3
"""Walk the current model in the Go2-W body and keep the expert's action when they differ.

The Habitat side is ``collect_go2w_expert.py --mode serve --failures-only``. This
process only runs the checkpoint. A step is written when the model's RVQ codes
are not the expert's. The stored answer is the expert token string, so training
learns the correction rather than the mistake.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from lightnav.habitat.policy import extract_instruction  # noqa: E402
from lightnav.habitat.remote_env import RemoteEnvClient  # noqa: E402
from lightnav.habitat.runner import (  # noqa: E402
    HabitatEvalConfig,
    _default_engine_factory,
    build_velocity_policy,
)


def _episode_key(info: dict) -> tuple[str, str]:
    scene_id = str(info.get("scene_id", "")).rstrip("/")
    scene = os.path.basename(os.path.dirname(scene_id))
    return scene, str(info.get("episode_id", ""))


def _done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("episode_done"):
                done.add((row["scene"], str(row["episode_id"])))
    return done


def _wait_ready(ready: Path) -> None:
    deadline = time.time() + 3600
    while time.time() < deadline:
        if ready.exists():
            return
        time.sleep(2)
    raise TimeoutError(f"expert server did not write {ready}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="tcp://127.0.0.1:5666")
    parser.add_argument("--ready", default="output/go2w_expert/correction.ready")
    parser.add_argument("--model", default="checkpoints/LightNav-0")
    parser.add_argument("--out", default="output/go2w_expert")
    parser.add_argument("--corrections-name", default="corrections.jsonl")
    parser.add_argument("--done-from", action="append", default=[],
                        help="Extra jsonl whose finished episodes are skipped.")
    parser.add_argument("--gpu-mem", type=float, default=0.28)
    parser.add_argument("--backend", default="vllm_local")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / args.corrections_name
    frame_root = out_dir / "correction_frames"
    _wait_ready(Path(args.ready))

    cfg = HabitatEvalConfig(
        server=args.server,
        model_path=args.model,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_mem,
        max_steps=500,
    )
    engine, bundle = _default_engine_factory(cfg)
    env = RemoteEnvClient(args.server)
    done = _done(samples_path)
    for extra in args.done_from:
        done |= _done(Path(extra))
    seen: set[tuple[str, str]] = set()
    policy = None
    n_saved = 0
    with samples_path.open("a") as handle:
        while True:
            obs, info = env.reset()
            key = _episode_key(info)
            if key in seen:
                break
            seen.add(key)
            scene, episode_id = key
            if key in done:
                continue
            expert = info.get("expert") or {}
            if not expert.get("ok"):
                handle.write(json.dumps({
                    "source": "correction",
                    "scene": scene,
                    "episode_id": episode_id,
                    "episode_done": True,
                    "skipped": expert.get("reason", "unusable"),
                    "corrections": 0,
                }) + "\n")
                handle.flush()
                continue
            if policy is None:
                policy = build_velocity_policy(cfg, engine, bundle, info)
            policy.reset(obs)
            instruction = extract_instruction(obs)
            ep_dir = frame_root / f"{scene}_{episode_id}"
            if ep_dir.exists():
                shutil.rmtree(ep_dir)
            ep_dir.mkdir(parents=True)
            n_corr = 0
            for step in range(cfg.max_steps):
                iio.imwrite(ep_dir / f"{step:04d}.jpg", np.asarray(obs["rgb"]), quality=80)
                action = policy.act(obs, info)
                codes = (policy.get_info() or {}).get("rvq_codes")
                disagree = codes != expert.get("codes")
                if disagree and expert.get("ok"):
                    handle.write(json.dumps({
                        "source": "correction",
                        "scene": scene,
                        "episode_id": episode_id,
                        "instruction": instruction,
                        "step": step,
                        "frame_dir": str(ep_dir.relative_to(out_dir)),
                        "stop": expert["stop"],
                        "codes": expert["codes"],
                        "action": expert["action"],
                        "model_codes": codes,
                        "distance_m": expert["distance_m"],
                        "min_distance_m": expert["min_distance_m"],
                        "episode_done": False,
                    }) + "\n")
                    n_corr += 1
                    n_saved += 1
                obs, _reward, terminated, truncated, info = env.step(action)
                expert = info.get("expert") or {}
                if terminated or truncated or not expert.get("ok"):
                    break
            if n_corr == 0 and ep_dir.exists():
                shutil.rmtree(ep_dir)
            handle.write(json.dumps({
                "source": "correction",
                "scene": scene,
                "episode_id": episode_id,
                "episode_done": True,
                "corrections": n_corr,
            }) + "\n")
            handle.flush()
            print(f"correction {scene}/{episode_id} mismatches={n_corr} total={n_saved}", flush=True)
    env.close()
    print(f"correction collection finished samples={n_saved}", flush=True)


if __name__ == "__main__":
    main()
