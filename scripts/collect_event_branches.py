"""Collect until an observed event, with a 1000-step censoring fallback."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world_model.event_branch_collection import prepare, worker  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "worker"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--previews", action="append", type=Path, default=[])
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--label-landmarks", type=Path)
    parser.add_argument("--max-branch-steps", type=int, default=1000)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        if not args.previews or args.reviews is None:
            parser.error("prepare requires --previews and --reviews")
        prepare(args.output, args.previews, args.reviews, args.max_branch_steps, args.label_landmarks)
    else:
        if not 0 <= args.worker < args.workers:
            parser.error("worker must be in [0, workers)")
        worker(args.output, args.worker, args.workers, args.task_id, args.resume)


if __name__ == "__main__":
    main()
