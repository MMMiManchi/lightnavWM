"""Second frozen teacher checks architecture; all admitted labels remain silver."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.branch_node_review import PROMPT, consensus_label  # noqa: E402
from scripts.curate_branch_benchmark import digest, read_rows, save, save_rows  # noqa: E402


def worker(args):
    import torch
    from PIL import Image

    from experiments.world_model.visual_harness import LocalQwenVision, parse_object

    output = args.output.resolve()
    nodes = read_rows(output.parent / "node_review_queue.jsonl")
    if not 0 <= args.worker < args.workers:
        raise ValueError("invalid worker")
    if not (output / "protocol.json").exists():
        raise ValueError("prepare refinement before workers")
    destination = output / f"worker_{args.worker:02d}.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(2)
    vision = LocalQwenVision(args.model.resolve(), device="cuda:0")
    with destination.open("x") as stream:
        for index, node in enumerate(nodes):
            if index % args.workers != args.worker:
                continue
            ref = node["inputs"]["history_rgb"][-1]
            if digest(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("node view changed")
            raw = vision(PROMPT, [("CURRENT", Image.fromarray(np.load(ref["path"], allow_pickle=False)))], max_new_tokens=420)
            try:
                value = parse_object(raw)
                if not isinstance(value, dict):
                    raise ValueError("not an object")
                confidence = float(value.get("confidence", 0.0))
                if not 0 <= confidence <= 1 or not isinstance(value.get("exits", []), list):
                    raise ValueError("invalid confidence or exits")
                error = None
            except (TypeError, ValueError) as exc:
                value, error = {"label": "unknown", "confidence": 0.0}, str(exc)
            stream.write(json.dumps({"sample_id": node["sample_id"], "input_sha256": ref["sha256"],
                                     "prediction": value, "raw_response": raw, "error": error}) + "\n")
            stream.flush()
            print(json.dumps({"worker": args.worker, "index": index, "label": value.get("label")}), flush=True)
    save(destination.with_suffix(".meta.json"), {"workers": args.workers, "worker": args.worker,
        "queue_sha256": digest(output.parent / "node_review_queue.jsonl"),
        "model_manifest_sha256": digest(args.model / "asset_manifest.json"),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest()})


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protocol.json", {"prompt": PROMPT, "workers": args.workers,
         "quality": "two_prompts_same_frozen_model_not_independent_ground_truth",
         "queue_sha256": digest(output.parent / "node_review_queue.jsonl"),
         "labels_authorize_strong_supervised_training": False,
         "labels_authorize_weak_auxiliary_pretraining": True})
    for path in (Path(__file__), REPO / "experiments/world_model/branch_node_review.py"):
        destination = output / "source_snapshot" / path.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


def summarize(args):
    output = args.output.resolve()
    second = {}
    for worker_id in range(args.workers):
        path = output / f"worker_{worker_id:02d}.jsonl"
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        if meta["queue_sha256"] != digest(output.parent / "node_review_queue.jsonl"):
            raise ValueError("refinement queue changed")
        for row in read_rows(path):
            if row["sample_id"] in second:
                raise ValueError("duplicate refinement sample")
            second[row["sample_id"]] = row
    rows = [row for role in ("fit", "development") for row in read_rows(output.parent / f"node_silver_{role}.jsonl")]
    if set(second) != {row["sample_id"] for row in rows}:
        raise ValueError("incomplete refinement")
    for row in rows:
        refined = second[row["sample_id"]]
        if refined["input_sha256"] != row["inputs"]["history_rgb"][-1]["sha256"]:
            raise ValueError("refinement image mismatch")
        label, reason = consensus_label(row["teacher"], refined["prediction"])
        row["first_pass_label"] = row["label"]
        row["label"], row["consensus_reason"], row["refinement"] = label, reason, refined
        row["training_mask"] = False
        row["weak_training_mask"] = label != "unknown"
        row["label_quality"] = "machine_consensus_silver_not_verified_ground_truth"
    frequencies = Counter((r["role"], r["scene"], r["label"]) for r in rows if r["weak_training_mask"])
    stats = {}
    for role in ("fit", "development"):
        selected = [r for r in rows if r["role"] == role]
        for row in selected:
            row["sample_weight"] = 0.1 / frequencies[(role, row["scene"], row["label"])] if row["weak_training_mask"] else 0.0
        save_rows(output / f"node_consensus_{role}.jsonl", selected)
        stats[role] = {"views": len(selected), "weak_labels": dict(Counter(r["label"] for r in selected if r["weak_training_mask"])),
                       "quarantined": sum(not r["weak_training_mask"] for r in selected),
                       "strong_verified_labels": 0}
    save(output / "counts.json", stats)
    save(output.parent / "node_release.json", {"use_files": [str(output / f"node_consensus_{r}.jsonl") for r in ("fit", "development")],
         "supersedes": ["node_silver_fit.jsonl", "node_silver_development.jsonl"],
         "first_pass_files_are_unfiltered_intermediates": True,
         "use_for_weak_auxiliary_pretraining_only": True, "strong_node_supervision_ready": False})
    print(json.dumps(stats))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "worker", "summarize"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=REPO / "checkpoints/world_model_planner/Qwen3-VL-4B-Instruct")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    {"prepare": prepare, "worker": worker, "summarize": summarize}[args.phase](args)


if __name__ == "__main__":
    main()
