"""Frozen current-view teacher drafts and indexed visual-review contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.branch_benchmark_entry import read_jsonl, write_json  # noqa: E402
from experiments.world_model.jsonutil import sha256_file  # noqa: E402
from experiments.world_model.viewpoint_branch_review import PROMPT, proposed_node_passes  # noqa: E402


def marked_image(preview):
    ref = preview["source_rgb"]
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("source image changed after preview")
    image = Image.fromarray(np.load(ref["path"], allow_pickle=False)).copy()
    draw = ImageDraw.Draw(image)
    for label, color, exit_ in zip(("A", "B"), ((0, 255, 255), (255, 0, 255)), preview["candidate"]["exits"]):
        u = 0.5 - math.tan(exit_["bearing_left_rad"]) / (2 * math.tan(math.radians(60)))
        x = int(round(u * (image.width - 1)))
        draw.line((x, 25, x, image.height - 2), fill=color, width=2)
        draw.rectangle((x - 8, 3, x + 10, 22), fill=color)
        draw.text((x - 3, 6), label, fill="black")
    return image


def worker(args):
    import torch

    from experiments.world_model.visual_harness import LocalQwenVision, parse_object

    torch.set_num_threads(2)
    output = args.output.resolve()
    previews = [json.loads(f.read_text()) for f in sorted((output / "previews").glob("*/preview.json"))]
    previews = [r for r in previews if r["status"] == "rendered"]
    destination = output / "source_judgements" / f"worker_{args.worker:02d}.jsonl"
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    vision = LocalQwenVision(args.model.resolve(), device="cuda:0")
    with destination.open("x") as stream:
        for i, preview in enumerate(previews):
            if i % args.workers != args.worker:
                continue
            raw = vision(PROMPT, [("CURRENT; proposed directions may be invalid", marked_image(preview))], max_new_tokens=340)
            try:
                value = parse_object(raw)
                passes = proposed_node_passes(value)
                if not isinstance(value, dict) or not 0 <= float(value.get("confidence", 0)) <= 1:
                    raise ValueError("invalid JSON schema/confidence")
                error = None
            except (TypeError, ValueError, AttributeError) as exc:
                value, passes, error = {}, False, str(exc)
            record = {"candidate_id": preview["candidate"]["candidate_id"], "role": preview["candidate"]["role"],
                      "scene": preview["candidate"]["scene"], "source_rgb_sha256": preview["source_rgb"]["sha256"],
                      "prediction": value, "provisional_pass": passes, "raw_response": raw, "error": error,
                      "is_node_ground_truth": False}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(json.dumps({"index": i, "role": record["role"], "provisional_pass": passes}), flush=True)
    write_json(destination.with_suffix(".meta.json"), {"prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
               "model_manifest_sha256": sha256_file(args.model / "asset_manifest.json"),
               "rendered_previews": len(previews), "current_only": True})


def summarize(args):
    output = args.output.resolve()
    files = sorted((output / "source_judgements").glob("worker_??.jsonl"))
    if len(files) != args.workers or not all(f.with_suffix(".meta.json").exists() for f in files):
        raise ValueError("teacher workers are incomplete")
    rows = [r for f in files for r in read_jsonl(f)]
    rendered = {json.loads(p.read_text())["candidate"]["candidate_id"] for p in (output / "previews").glob("*/preview.json")
                if json.loads(p.read_text())["status"] == "rendered"}
    if len({r["candidate_id"] for r in rows}) != len(rows) or {r["candidate_id"] for r in rows} != rendered:
        raise ValueError("teacher results do not exactly cover rendered sources")
    rows.sort(key=lambda r: (r["role"] != "fit", r["scene"], r["candidate_id"]))
    (output / "source_judgements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    review = [r for r in rows if r["provisional_pass"]]
    index = []
    for start in range(0, len(review), 20):
        group = review[start:start + 20]
        canvas = Image.new("RGB", (1280, 225 * ((len(group) + 3) // 4)), "white")
        draw = ImageDraw.Draw(canvas)
        for j, result in enumerate(group):
            preview = json.loads((output / "previews" / result["candidate_id"] / "preview.json").read_text())
            x, y = (j % 4) * 320, (j // 4) * 225
            canvas.paste(marked_image(preview).resize((316, 178)), (x + 2, y + 2))
            draw.text((x + 2, y + 182), f"{start+j:03d} {result['candidate_id'][:8]} {result['role']}", fill="black")
            draw.text((x + 2, y + 200), result["scene"], fill="black")
            index.append({"index": start + j, **result})
        canvas.save(output / f"source_review_{start//20:02d}.png")
    write_json(output / "source_review_index.json", {"rows": index, "all_are_provisional_machine_labels": True})
    stats = {role: {"judged": sum(r["role"] == role for r in rows),
                    "provisional_pass": sum(r["role"] == role and r["provisional_pass"] for r in rows)} for role in ("fit", "development")}
    write_json(output / "source_judgement_counts.json", stats)
    print(json.dumps(stats))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("worker", "summarize"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=REPO / "checkpoints/world_model_planner/Qwen3-VL-4B-Instruct")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "worker":
        worker(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
