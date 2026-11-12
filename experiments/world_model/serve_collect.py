"""Habitat-side collection server.

Run with the Habitat 3.9 interpreter. This does not replace ``lightnav_habitat.serve``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_allowlist(path: Path) -> tuple[dict, list]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_split") != "train":
        raise SystemExit(f"refusing allowlist split {payload.get('source_split')!r}")
    if payload.get("schema_version") != "wm-allowlist/0.1":
        raise SystemExit("allowlist schema is not wm-allowlist/0.1")
    order = []
    seen = set()
    for item in payload["episodes"]:
        key = (str(item["scene"]), str(item["episode_id"]))
        if key in seen:
            raise SystemExit(f"duplicate allowlist id {key}")
        seen.add(key)
        order.append(key)
    if not order:
        raise SystemExit("allowlist is empty")
    return payload, order


def prepare_server_plan(
    allowlist_path: Path,
    dataset_root: Path,
    inventory_path: Path,
    *,
    partition: str = "train_fit",
) -> tuple[dict, list]:
    """Check the inventory before Habitat is imported. A bad plan never reaches the simulator."""
    from experiments.world_model.collection_gate import (
        CollectionGateError,
        assert_collectable,
        assert_dev_collectable,
        load_scene_inventory,
    )
    from experiments.world_model.group_manifest import (
        assign_groups,
        check_allowlist_against_train,
        load_split_rows,
    )

    payload, order = _load_allowlist(allowlist_path)
    try:
        _dataset_path, digest, rows = load_split_rows(dataset_root, "train")
        plan = check_allowlist_against_train(payload, assign_groups(rows), dataset_sha256=digest)
        inventory = load_scene_inventory(inventory_path)
        if partition == "train_fit":
            assert_collectable(inventory, plan, dataset_sha256=digest)
        elif partition == "train_dev":
            assert_dev_collectable(inventory, plan, dataset_sha256=digest)
        else:
            raise CollectionGateError("partition", f"unsupported collection partition {partition!r}")
        checked = [(row["scene"], row["episode_id"]) for row in plan]
        if checked != order:
            raise CollectionGateError("allowlist", "served order does not match the checked train plan")
    except (CollectionGateError, RuntimeError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return payload, order


def main(argv: list | None = None) -> int:
    if sys.version_info[:2] != (3, 9):
        raise SystemExit(
            f"serve_collect must run on the Habitat Python 3.9 interpreter, not {sys.version.split()[0]}"
        )
    parser = argparse.ArgumentParser(description="Serve a strict train-only collection environment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allowlist", required=True)
    parser.add_argument("--scene-inventory", required=True)
    parser.add_argument("--port", type=int, required=True)
    ending = parser.add_mutually_exclusive_group(required=True)
    ending.add_argument("--max-steps", type=int)
    ending.add_argument("--event-driven", action="store_true",
                        help="Disable both per-episode step limits; collector decides actual event endings")
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--partition", choices=("train_fit", "train_dev"), default="train_fit")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--controlled-trial", action="store_true")
    parser.add_argument("--trial-output-dir", default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.event_driven and not 1 <= args.max_steps <= 500:
        raise SystemExit("--max-steps exceeds this server entry point's limit")
    if args.controlled_trial and args.event_driven:
        raise SystemExit("the archived controlled trial requires its original finite step budget")
    if args.controlled_trial and not args.trial_output_dir:
        raise SystemExit("--trial-output-dir is required for a controlled trial")
    if not args.controlled_trial and args.trial_output_dir:
        raise SystemExit("--trial-output-dir requires --controlled-trial")

    root = _repo_root()
    sys.path.insert(0, str(root))
    from experiments.world_model.controlled_trial import validate_controlled_trial

    payload, order = prepare_server_plan(
        Path(args.allowlist),
        Path(args.dataset_root),
        Path(args.scene_inventory),
        partition=args.partition,
    )
    if args.controlled_trial:
        try:
            validate_controlled_trial(
                repo_root=root,
                run_dir=Path(args.trial_output_dir),
                max_steps=args.max_steps,
                episodes=payload["episodes"],
                require_absent=False,
            )
        except (ValueError, FileExistsError) as exc:
            raise SystemExit(str(exc)) from exc
    from experiments.world_model.collection_gate import CollectionGateError
    from experiments.world_model.precollection_protocol import (
        assert_collection_not_blocked,
        assert_dev_collection_authorized,
    )

    if not args.controlled_trial:
        try:
            if args.partition == "train_fit":
                assert_collection_not_blocked()
            else:
                assert_dev_collection_authorized()
        except CollectionGateError as exc:
            raise SystemExit(str(exc)) from exc
    sys.path.insert(0, str(root / "habitat_server"))
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from lightnav_habitat.remote_server import RemoteEnvServer

    from experiments.world_model.server_adapter import WorldModelVLNCEEnv

    gpu_id = int(os.environ.get("HABITAT_SIM_GPU_ID", "0") if args.gpu_id is None else args.gpu_id)
    env = WorldModelVLNCEEnv(
        config_path=args.config,
        split="train",
        data_path=str(Path(args.dataset_root).resolve()),
        gpu_id=gpu_id,
        max_steps=args.max_steps,
        event_driven=args.event_driven,
        episode_allow=set(order),
        allow_order=order,
        requested_seed=args.seed,
        expected_dataset_sha256=payload.get("dataset_sha256"),
        early_stop_rotation=0,
        early_stop_steps=0,
    )
    server = RemoteEnvServer(env, address=f"tcp://*:{args.port}")
    server.start(ready_file=args.ready_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
