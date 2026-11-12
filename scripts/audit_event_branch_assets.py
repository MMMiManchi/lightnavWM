"""Verify all retained branch RGB/depth hashes and immutable input snapshots."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.branch_benchmark_entry import read_jsonl, write_json  # noqa: E402
from experiments.world_model.jsonutil import sha256_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    files, inputs, snapshots = {}, {}, {}
    executions = 0
    for source in args.source:
        for task in read_jsonl(source / "tasks.jsonl"):
            if task["origin"]["role"] not in {"fit", "development"} or task["origin"]["scene"] == "Vvot9Ly1tCj":
                raise ValueError("sealed task refused")
            folder = source / "executions" / task["task_id"]
            result = json.loads((folder / "result.json").read_text())
            commits = read_jsonl(folder / "committed.jsonl")
            if len(commits) != result["branch_steps"] or result["branch_steps"] > 1000:
                raise ValueError("execution counts are inconsistent")
            frames = read_jsonl(folder / "source_history.jsonl") + [c["frame"] for c in commits]
            for restore in folder.glob("restore_*/source_history.jsonl"):
                frames.extend(read_jsonl(restore))
            for frame in frames:
                rgb = Path(frame.get("rgb_path", str(folder / "assets" / f"{frame['name']}_rgb.npy")))
                depth = rgb.with_name(rgb.name.replace("_rgb.npy", "_depth.npy"))
                files[str(rgb.resolve())] = frame["rgb_sha256"]
                files[str(depth.resolve())] = frame["depth_sha256"]
            executions += 1
        for name, expected in json.loads((source / "input_manifest.json").read_text()).items():
            path = Path(name)
            snapshot = source / "source_snapshot" / path.relative_to(REPO) if path.is_relative_to(REPO) else None
            if snapshot is not None and snapshot.exists():
                snapshots[str(snapshot.resolve())] = expected
            else:
                inputs[name] = expected
    for name, expected in json.loads((args.release / "input_manifest.json").read_text()).items():
        inputs[name] = expected
    failures = []
    def check(item):
        name, expected = item
        path = Path(name)
        if sha256_file(path) != expected:
            return name
        return None
    combined = {**files, **inputs, **snapshots}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(check, combined.items()):
            if result is not None:
                failures.append(result)
    report = {"executions_checked_including_superseded": executions, "asset_files_checked": len(files),
        "immutable_inputs_checked": len(inputs), "preserved_source_snapshots_checked": len(snapshots),
        "total_hash_checks": len(combined), "all_match": not failures, "failures": failures,
        "sealed_trajectory_or_image_files_opened": 0}
    write_json(args.output, report)
    print(json.dumps(report), flush=True)
    if failures:
        raise RuntimeError("asset/input hash mismatch")


if __name__ == "__main__":
    main()
