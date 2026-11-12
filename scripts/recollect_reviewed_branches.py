"""Separate reviewed-node trial; worker runs in the Habitat Python 3.9 env."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.branch_recollection_entry import prepare, summarize, worker  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "worker", "summarize"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--origins", type=Path, default=REPO / "output/world_model_data/branch_curated_20261105_01/release/recollection_origins.jsonl")
    parser.add_argument("--source-batch", type=Path, default=REPO / "output/world_model_data/branch_benchmark_20261105_01")
    parser.add_argument("--environment-step-budget", type=int, default=4000)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output, args.origins, args.source_batch, args.environment_step_budget)
    elif args.phase == "worker":
        worker(args.output, args.worker, args.workers)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
