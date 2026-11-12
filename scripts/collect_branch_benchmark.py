#!/usr/bin/env python3
"""Collect or recheck the same-state multi-exit event benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.world_model.branch_benchmark_entry import (
    collect,
    export_supervision,
    verify_output,
    write_arrival_supplement,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory for a collection run")
    parser.add_argument("--gpu", type=int, help="GPU index 0-7. Each process keeps this card.")
    parser.add_argument(
        "--prepare-arrivals",
        type=Path,
        help="Freeze junctions reached by finished exits as new origins. Does not start Habitat.",
    )
    parser.add_argument("--verify", type=Path, help="Existing run to recheck without Habitat")
    parser.add_argument(
        "--refresh-supervision",
        type=Path,
        help="Write node and consequence labels for an existing run. Does not start Habitat.",
    )
    args = parser.parse_args()
    if args.prepare_arrivals is not None:
        chosen = write_arrival_supplement(args.prepare_arrivals)
        print(f"arrival origins {len(chosen)}")
        return
    if args.refresh_supervision is not None:
        counts = export_supervision(args.refresh_supervision)
        print(f"node frames {counts['node_frames']} imitation labels {counts['instruction_action_labels']}")
        return
    if args.verify is not None:
        report = verify_output(args.verify)
        print(f"complete {report['complete_transitions']} passed {report['passed']}")
        return
    if args.output is None:
        raise SystemExit("pass --output for a new run, or --verify for an existing run")
    collect(args.output, None if args.gpu is None else str(args.gpu))


if __name__ == "__main__":
    main()
