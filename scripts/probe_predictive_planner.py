"""Real VLM/per-candidate future-query probes over strictly causal memories.

Shared views/candidates and equal memory prefixes compare direct, predicted and
shuffled retrieval. This is not a navigation success or blind accuracy test.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.event_runtime import load_residual_predictor  # noqa: E402
from experiments.world_model.event_train import read_rows, save  # noqa: E402
from experiments.world_model.predictive_queries import BranchQueryService  # noqa: E402
from experiments.world_model.visual_harness import (  # noqa: E402
    LocalQwenVision,
    ObservationImages,
    QwenBranchProposer,
    QwenTaskPlanner,
    register_event_images,
)
from scripts.smoke_event_lightnav import raw_features, seed_prior_memory, select_checkpoint, serial  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--planner-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--sample-indices", type=int, nargs="+",
                        help="Explicit eligible train_dev sample indices, sorted by acquisition order.")
    args = parser.parse_args()
    if args.sample_indices is None and not 1 <= args.samples <= 5:
        parser.error("samples must be in 1..5")
    if args.sample_indices is not None and (not 1 <= len(args.sample_indices) <= 5 or
                                            len(set(args.sample_indices)) != len(args.sample_indices)):
        parser.error("sample-indices must contain 1..5 unique indices")
    torch.set_num_threads(4)
    output = args.output.resolve()
    root = args.training_run.resolve()
    config = json.loads((root / "config.json").read_text())
    checkpoint, _selected = select_checkpoint(root)
    predictor = load_residual_predictor(checkpoint)
    base = Path(config["base_run"])
    pairs = read_rows(base / "pairs.jsonl")
    source_features = raw_features(base)
    indices = np.load(root / "memory_indices_1.npy")
    order = {tuple(ep): i for i, ep in enumerate(json.loads((root / "episode_order.json").read_text()))}
    eligible = sorted([i for i, p in enumerate(pairs) if p["partition"] == "train_dev" and (indices[i] >= 0).any()],
                      key=lambda i: (order[(pairs[i]["source_run"], pairs[i]["episode_uid"])], pairs[i]["origin_index"]))
    if args.sample_indices is None:
        samples = eligible[:args.samples]
        selection = "first causal-memory dev queries in acquisition order"
    else:
        if not set(args.sample_indices).issubset(eligible):
            parser.error("sample-indices must refer to eligible causal-memory train_dev rows")
        samples = sorted(args.sample_indices, key=eligible.index)
        selection = "explicit eligible train_dev indices, ordered by causal acquisition"
    output.mkdir(parents=True, exist_ok=False)
    column_map = {}
    for path in sorted(root.glob("columns_*.npz")):
        chunk = np.load(path)
        column_map.update(zip(chunk["ids"].tolist(), chunk["columns"], strict=True))
    save(output / "protocol.json", {"samples": samples, "selection": selection,
              "tasks": ["Find a room containing a table.", "Find a room containing a bed."],
              "modes": ["direct", "predicted", "shuffled"], "future_gallery_not_an_input": True,
              "candidate_proposals": "current RGB VLM only; no official nodes/exits supplied",
              "memory_prefix_budget": "same visible actual records, same top3 limit for each mode",
              "planner_is_blind_to_retrieval_mode_label": True, "navigation_success_not_evaluated": True})
    visual = LocalQwenVision(args.planner_model, args.device)
    images = ObservationImages()
    register_event_images(images, base)
    proposer = QwenBranchProposer(visual, images)
    planner = QwenTaskPlanner(visual, images)
    results = []
    for sample in samples:
        folder = output / f"sample_{sample}"
        folder.mkdir()
        bank, _index, query, _arrays = seed_prior_memory(config, predictor, folder, chosen_index=sample)
        observation = query["history_observation_ids"][-1]
        images.get(observation).save(folder / "current.png")
        try:
            perception = proposer.propose(observation)
            save(folder / "perception.json", serial(asdict(perception)))
            raw = np.stack([source_features[i] if i else np.zeros(1152) for i in query["history_image_ids"]]).astype(np.float32)
            kwargs = {"source_observation": observation, "branches": perception.branches,
                      "raw_history": torch.from_numpy(raw)[None], "valid": torch.tensor([query["history_valid"]]),
                      "columns": torch.from_numpy(column_map[query["history_image_ids"][-1]][None])}
            for mode in ("direct", "predicted", "shuffled"):
                service = BranchQueryService(predictor, bank, memory_context=query["scene"], episode_id=query["episode_uid"],
                                             visible_sequence=bank.latest_sequence(query["scene"]), mode=mode)
                for task_index, instruction in enumerate(["Find a room containing a table.", "Find a room containing a bed."]):
                    context = service.build_context(instruction=instruction, **kwargs)
                    try:
                        decision = planner(context)
                        row = {"sample": sample, "mode": mode, "task": instruction, "candidate_count": len(perception.branches),
                               "decision": asdict(decision), "queries": service.last_trace, "planner": planner.last_trace}
                    except (ValueError, KeyError, TypeError) as error:
                        row = {"sample": sample, "mode": mode, "task": instruction, "error": str(error), "queries": service.last_trace}
                    save(folder / f"{mode}_task{task_index}.json", serial(row))
                    results.append(row)
                    print(json.dumps({"sample": sample, "mode": mode, "task": task_index,
                                      "action": row.get("decision", {}).get("action"), "error": row.get("error")}), flush=True)
        except (ValueError, KeyError, TypeError) as error:
            save(folder / "tool_error.json", {"error": str(error)})
            results.append({"sample": sample, "error": str(error)})
        finally:
            bank.close()
    successful = [r for r in results if "decision" in r]
    predicted_queries = [q for r in results if r.get("mode") == "predicted" for q in r.get("queries", [])]
    consumed_images = [ref for r in successful for ref in r["planner"]["image_observation_references"][1:]]
    save(output / "results.json", serial({"runs": results, "real_planner_calls": len(visual.calls),
              "model": str(args.planner_model.resolve()), "world_model": str(checkpoint.resolve()),
              "successful_decisions": len(successful), "supported_predicted_queries": sum(q["supported_by_history"] for q in predicted_queries),
              "prediction_vectors_used_for_actual_outcome_queries": any(q["retrieved"] for q in predicted_queries),
              "queried_actual_images_consumed_by_task_planner": bool(consumed_images), "navigation_gain_not_established": True}))
    save(output / "visual_calls.json", visual.calls)
    print(json.dumps({"state": "complete", "runs": len(results), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
