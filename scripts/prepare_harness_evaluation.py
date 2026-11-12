"""Prepare isolated base/memory/predictive runs with one owner per scene."""

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.evaluation import scene_shards  # noqa: E402
from scripts.run_stateful_harness import load_tasks  # noqa: E402


def prepare(args):
    if not 1 <= args.workers <= 8 or not 1 <= args.port <= 65536 - args.workers:
        raise ValueError("expected 1–8 GPU workers and a valid port range")
    tasks = [task for path in args.tasks for task in load_tasks(path)]
    shards = scene_shards(tasks, args.workers)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"tasks": len(tasks), "scenes": len({t['scene'] for t in tasks}), "workers": args.workers,
                "same_scene_single_owner": True, "within_scene_order": "input order",
                "memory": "fresh per condition, serial cross-episode within scene",
                "online_learning": False, "max_steps": 500, "dataset_split": args.dataset_split,
                "conditions": ["base", "monitor", "memory", "predictive"], "world_model": str(args.world_model.resolve())}
    protocol["reference_namespace"] = hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:12]
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for condition in protocol["conditions"]:
        root = args.output / condition
        root.mkdir()
        commands = ["#!/usr/bin/env bash", "set -euo pipefail", "pids=()"]
        for worker, tasks in enumerate(shards):
            if not tasks:
                continue
            path = root / f"tasks_gpu{worker}.json"
            path.write_text(json.dumps(tasks, indent=2) + "\n")
            command = [str(REPO / ".venv/bin/python"), "-u", str(REPO / "scripts/run_stateful_harness.py"),
                "--tasks", str(path.resolve()), "--output", str((root / f"gpu{worker}").resolve()),
                "--experience", str((root / f"experience_gpu{worker}").resolve()),
                "--world-model", str(args.world_model.resolve()), "--dataset-split", args.dataset_split,
                "--run-id", protocol["reference_namespace"],
                "--sim-config", str(args.sim_config.resolve()), "--habitat-python", str(args.habitat_python),
                "--sim-gpu", str(worker), "--port", str(args.port + worker), "--max-steps", "500"]
            if condition == "base":
                command += ["--base-policy-only"]
            elif condition == "monitor":
                command += ["--disable-memory"]
            elif condition == "predictive":
                command += ["--prediction-overrides"]
            commands.append(f"CUDA_VISIBLE_DEVICES={worker} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                            f"VLLM_WORKER_MULTIPROC_METHOD=spawn {shlex.join(command)} > {shlex.quote(str((root / f'gpu{worker}.log').resolve()))} 2>&1 &")
            commands.append('pids+=("$!")')
        commands += ["status=0", 'for pid in "${pids[@]}"; do', '  wait "$pid" || status=1', "done", 'exit "$status"']
        (root / "run.sh").write_text("\n".join(commands) + "\n")
    parent = ["#!/usr/bin/env bash", "set -euo pipefail"]
    parent.extend(shlex.join(["bash", str((args.output / c / "run.sh").resolve())]) for c in protocol["conditions"])
    (args.output / "run_all.sh").write_text("\n".join(parent) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-model", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset-split", choices=["train", "val_seen", "val_unseen"], default="train")
    parser.add_argument("--port", type=int, default=6050)
    parser.add_argument("--sim-config", type=Path, default=REPO / "habitat_server/configs/vlnce_r2r_go2w.yaml")
    parser.add_argument("--habitat-python", type=Path, default=Path("/home/szgx/miniconda3/envs/habitat/bin/python3.9"))
    prepare(parser.parse_args())
