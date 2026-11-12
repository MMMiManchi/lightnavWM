"""Report complete or still-running matched navigation ablations."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.evaluation import matched_navigation  # noqa: E402


def summarize(tasks, runs):
    expected = {(t["scene"], str(t["episode_id"])) for t in tasks}
    summaries, rows = {}, {}
    for label, folder in runs.items():
        paths = [folder / "results.json"] if (folder / "results.json").exists() else sorted(folder.glob("gpu*/results.json"))
        values = [row for path in paths for row in json.loads(path.read_text())]
        keys = [(r["scene"], str(r["dataset_episode_id"])) for r in values]
        if len(set(keys)) != len(keys) or set(keys) - expected:
            raise ValueError("duplicate or unexpected completed task")
        rows[label] = values
        summaries[label] = {"completed":len(values), "expected":len(expected), "complete":set(keys) == expected,
            "missing":sorted(expected - set(keys)),
            "sr_completed_only":sum(r["evaluation_only"]["success"] for r in values)/len(values) if values else None,
            "spl_completed_only":sum(r["evaluation_only"]["spl"] for r in values)/len(values) if values else None,
            **{key:sum(r.get(key, 0) for r in values) for key in ["memory_direct_queries", "world_model_queries",
                "retrieved_outcome_queries", "evidence_used_rounds", "prediction_controlled_rounds",
                "memory_controlled_rounds", "prediction_changed_retrieval_queries", "observed_route_spans", "verified_events"]}}
    base = rows.get("base", [])
    comparisons = {}
    for label, values in rows.items():
        if label != "base" and summaries[label]["complete"] and summaries.get("base", {}).get("complete"):
            comparisons[label] = matched_navigation(base, values)
    return {"conditions":summaries, "complete_set_comparisons":comparisons,
            "all_complete":all(r["complete"] for r in summaries.values()),
            "partial_scores_are_not_final":True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="label=run directory; use base for the reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {label:Path(path) for label, path in (item.split("=", 1) for item in args.run)}
    result = summarize(json.loads(args.tasks.read_text()), runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
