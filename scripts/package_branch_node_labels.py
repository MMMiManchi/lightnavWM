"""Package explicitly reviewed current-view weak node labels, with uncertainty masks."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world_model.branch_benchmark_entry import write_json  # noqa: E402
from experiments.world_model.jsonutil import sha256_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previews", type=Path, action="append", required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.reviews.read_text())
    if not review.get("reviewer"):
        raise ValueError("explicit visual reviewer required")
    positives = set(review["accepted_candidate_ids"])
    ordinary = set(review["ordinary_candidate_ids"])
    if positives & ordinary:
        raise ValueError("conflicting visual labels")
    rows, manifest, seen = [], {str(args.reviews.resolve()): sha256_file(args.reviews)}, set()
    for root in args.previews:
        for path in sorted(root.glob("previews/*/preview.json")):
            data = json.loads(path.read_text())
            candidate = data["candidate"]
            cid = candidate["candidate_id"]
            if cid in seen:
                raise ValueError("candidate repeated across preview roots")
            seen.add(cid)
            if candidate["role"] not in {"fit", "development"} or candidate["scene"] == "Vvot9Ly1tCj":
                raise ValueError("sealed candidate refused")
            if data["status"] != "rendered":
                continue
            label = "decision" if cid in positives else "ordinary" if cid in ordinary else "unknown"
            refs = data["history_rgb"]
            for ref in refs:
                p = Path(ref["path"])
                if sha256_file(p) != ref["sha256"]:
                    raise ValueError("reviewed source image changed")
                manifest[str(p)] = ref["sha256"]
            rows.append({"sample_id": f"node:{cid}", "role": candidate["role"], "scene": candidate["scene"],
                "inputs": {"history_rgb": refs, "history_valid": [True] * len(refs)},
                "label": label, "training_mask": False, "weak_training_mask": label != "unknown",
                "label_quality": "Codex visual review with local geometry; simulation silver; no human gold",
                "label_scope": "visible immediate architectural choice in current observation, not all-around graph degree",
                "exits": [{"bearing_left_rad": e["bearing_left_rad"]} for e in candidate["exits"]] if label == "decision" else [],
                "exits_exhaustive": False, "requires_goal_route": False, "runtime_viewpoint_list_input": False,
                "reviewed_source": str(path.resolve())})
            manifest[str(path.resolve())] = sha256_file(path)
    if not (positives | ordinary) <= {r["sample_id"].removeprefix("node:") for r in rows}:
        raise ValueError("reviewed label has no rendered current-image evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write("".join(json.dumps(r) + "\n" for r in rows))
    manifest[str(Path(__file__).resolve())] = sha256_file(Path(__file__).resolve())
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps({role: dict(Counter(r["label"] for r in rows if r["role"] == role)) for role in ("fit", "development")}))


if __name__ == "__main__":
    main()
