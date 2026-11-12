"""Collect scene-disjoint train_dev transitions using the shared collector."""

from __future__ import annotations

import argparse
from typing import Any

from experiments.world_model.collection_gate import CollectionGateError, assert_dev_collectable


def prepare_dev_plan(
    inventory: dict[str, Any],
    episodes: list[dict[str, Any]],
    *,
    dataset_sha256: str,
) -> list[dict[str, Any]]:
    """Return the dev rows. A train_fit row is rejected."""
    return assert_dev_collectable(inventory, episodes, dataset_sha256=dataset_sha256)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect train_dev episodes for world-model validation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allowlist", required=True)
    parser.add_argument("--scene-inventory", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", default="data/datasets/R2R_VLNCE_v1-3_preprocessed")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--wall-clock-budget-s", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--server", default=None)
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--habitat-python", default="/home/szgx/miniconda3/envs/habitat/bin/python")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument("--policy", choices=("lightnav", "scripted"), default="lightnav")
    parser.add_argument("--model-path", default="checkpoints/LightNav-0")
    parser.add_argument("--backend", default="vllm_local")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--ready-timeout-s", type=float, default=900.0)
    args = parser.parse_args(argv)
    from experiments.world_model.collect import main as collect_main

    forwarded = [
        "--config", args.config,
        "--allowlist", args.allowlist,
        "--scene-inventory", args.scene_inventory,
        "--run-dir", args.run_dir,
        "--dataset-root", args.dataset_root,
        "--max-steps", str(args.max_steps),
        "--port", str(args.port),
        "--partition", "train_dev",
        "--policy", args.policy,
        "--model-path", args.model_path,
        "--backend", args.backend,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--ready-timeout-s", str(args.ready_timeout_s),
    ]
    if args.wall_clock_budget_s is not None:
        forwarded.extend(("--wall-clock-budget-s", str(args.wall_clock_budget_s)))
    if args.seed is not None:
        forwarded.extend(("--seed", str(args.seed)))
    if args.server:
        forwarded.extend(("--server", args.server))
    if args.launch_server:
        forwarded.append("--launch-server")
    if args.launch_server:
        forwarded.extend(("--habitat-python", args.habitat_python))
    return collect_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
