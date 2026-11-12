"""Create current-view previews and collect reviewed offline viewpoint branches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.viewpoint_branch_collection import (  # noqa: E402
    collection_worker,
    prepare_collection,
    prepare_previews,
    preview_worker,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare-previews", "preview-worker", "prepare-collection", "collection-worker"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidates", type=Path, default=REPO / "output/world_model_data/branch_source_candidates_20261106_01")
    parser.add_argument("--legacy", type=Path, default=REPO / "output/world_model_training/event_latent_v1_20261104_01")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "prepare-previews":
        prepare_previews(args.output, args.candidates, args.legacy)
    elif args.phase == "preview-worker":
        preview_worker(args.output, args.worker, args.workers)
    elif args.phase == "prepare-collection":
        if args.reviews is None:
            parser.error("--reviews is required for prepare-collection")
        prepare_collection(args.output, args.reviews)
    else:
        collection_worker(args.output, args.worker, args.workers)


if __name__ == "__main__":
    main()
