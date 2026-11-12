"""Prequential real-experience replay, separating memory from scene updates."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.branch_training_data import read_rows  # noqa: E402
from experiments.world_model.jsonutil import sha256_file  # noqa: E402
from experiments.world_model.scene_memory import PersistentSceneMemory  # noqa: E402
from experiments.world_model.tssm_runtime import TSSMSession, load_tssm_predictor  # noqa: E402


def weights_hash(model):
    import hashlib

    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def aggregate(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["category"]].append(row)
    result = {}
    for category, rows in groups.items():
        scenes = {r["scene"] for r in rows}
        result[category] = {"count": len(rows), "mean_cosine": float(np.mean([r["prediction_cosine_before_reveal"] for r in rows])),
            "scene_macro_cosine": float(np.mean([np.mean([r["prediction_cosine_before_reveal"] for r in rows if r["scene"] == s]) for s in scenes])),
            "used_observed_memory": sum(r["evidence_kind"] == "observed_memory_candidate" for r in rows)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update-steps", type=int, default=20)
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for path in [Path(__file__).resolve(), REPO / "experiments/world_model/tssm_runtime.py"]:
        target = output / "source_snapshot" / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        snapshots[str(path)] = {"sha256": sha256_file(path), "snapshot": str(target)}
    (output / "source_manifest.json").write_text(json.dumps(snapshots, indent=2) + "\n")
    torch.set_num_threads(2)
    predictor = load_tssm_predictor(args.model).eval()
    global_hash = weights_hash(predictor)
    base_hash = sha256_file(args.model)
    rows = [r for r in read_rows(data / "sequences.jsonl") if r["role"] == "development"]
    with np.load(data / "image_features.npz", allow_pickle=False) as source:
        images = dict(zip(source["image_sha256"].tolist(), source["pooler_raw"], strict=True))
    protocol = {"version": "event-tssm-prequential-replay/0.1", "model": str(args.model.resolve()),
        "checkpoint_sha256": base_hash, "data": str(data), "episodes": len(rows),
        "order": "declared scene, origin, exit, repeat; simulator wall-clock order not asserted",
        "prediction_saved_before_target_reveal": True, "global_parameters_frozen": True,
        "development_used_for_online_scene_adaptation": True,
        "online_targets_only_after_actual_observation": True, "cold_offline_validation_is_separate": True,
        "scene_updates_require_matching_previously_observed_source_and_direction": True,
        "repeat_view_is_identical": True, "cross_view_relocalization_tested": False,
        "new_branch_means_that_source_exit_has_no_prior_outcome_in_replay": True,
        "source_pose_and_official_node_ids_are_model_inputs": False,
        "LightNav_loaded": False, "new_simulator_runs": 0, "sealed_opened": False}
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    results = {}
    for name, memory, learning in [("frozen", False, False), ("memory_only", True, False),
                                    ("online_only", False, True), ("memory_and_online", True, True)]:
        folder = output / name
        folder.mkdir()
        bank = PersistentSceneMemory(folder / "facts.sqlite", contract_id=predictor.feature_contract_id,
                                     feature_dim=predictor.model.config.feature_dim)
        session = TSSMSession(predictor, bank, folder, base_hash=base_hash, use_memory=memory,
                              online_learning=learning, update_steps=args.update_steps)
        records, forecasts, targets = [], [], []
        seen_branches, seen_sources, seen_scenes = set(), set(), set()
        with (folder / "predictions.jsonl").open("x", buffering=1) as log:
            for i, row in enumerate(rows):
                source_key, branch_key = (row["scene"], row["origin_id"]), (row["scene"], row["origin_id"], row["exit_id"])
                category = ("known_branch_revisit" if branch_key in seen_branches else "unvisited_exit_at_known_source" if source_key in seen_sources
                            else "new_source_in_known_scene" if row["scene"] in seen_scenes else "first_scene_visit")
                history = np.stack([images[r["sha256"]] for r in row["history_rgb"]])[None]
                episode = "replay:" + row["sample_id"]
                session.begin_episode(row["scene"], episode, history, row["observations"][0]["path"],
                                      instruction="independent current task; history is environment evidence")
                ticket = session.forecast(row["sample_id"], row["direct_action"][:2])
                past = session.query_actual_memories(ticket)
                if any(m.sequence > ticket.visible_sequence or m.episode_id == episode for m in past):
                    raise ValueError("future memory leak")
                prediction = ticket.future_feature.copy()
                # Freeze the query before the real outcome becomes visible.
                log.write(json.dumps({"phase": "prediction_before_reveal", "episode": episode,
                    "category": category, "visible_sequence": ticket.visible_sequence,
                    "evidence_kind": ticket.evidence_kind, "adapter_version": ticket.adapter_version,
                    "adapter_applied": ticket.adapter_applied,
                    "retrieved_sequences": [m.sequence for m in past]}) + "\n")
                session.commit(ticket)
                for k in range(1, len(row["observations"]) - 1):
                    anchor = row["observations"][k]
                    session.observe_route(images[anchor["sha256"]], anchor["path"],
                        execution_ref=row["offline_only"]["evidence_dir"] + "/committed.jsonl", executed=True,
                        direction=row["actions"][k - 1][:2])
                actual = row["observations"][-1]
                raw_target = images[actual["sha256"]]
                with torch.inference_mode():
                    target = predictor.normalize(torch.from_numpy(raw_target[None]))[0].numpy()
                score = float(prediction @ target)
                feedback = session.observe_event(ticket, raw_target, target_observation=actual["path"],
                    execution_ref=row["offline_only"]["evidence_dir"] + "/committed.jsonl", executed=True, arrived=True)
                record = {"episode": episode, "scene": row["scene"], "sample_id": row["sample_id"],
                    "category": category, "prediction_cosine_before_reveal": score,
                    "evidence_kind": ticket.evidence_kind, "visible_sequence": ticket.visible_sequence,
                    "adapter_applied": ticket.adapter_applied,
                    "posterior_cache_steps_after_observation": session.state.tokens.shape[1],
                    "feedback_after_reveal": feedback}
                log.write(json.dumps({"phase": "actual_observed_then_updated", **record}) + "\n")
                records.append(record)
                forecasts.append(prediction)
                targets.append(target)
                seen_branches.add(branch_key)
                seen_sources.add(source_key)
                seen_scenes.add(row["scene"])
                if (i + 1) % 18 == 0:
                    print(json.dumps({"mode": name, "completed": i + 1, "episodes": len(rows)}), flush=True)
        session.end_episode()
        last = rows[-1]
        history = np.stack([images[r["sha256"]] for r in last["history_rgb"]])[None]
        session.begin_episode(last["scene"], "restart-before", history, "restart-source")
        expected = session.forecast("restart", last["direct_action"][:2]).future_feature.copy()
        session.end_episode()
        bank.close()
        bank = PersistentSceneMemory(folder / "facts.sqlite", contract_id=predictor.feature_contract_id,
                                     feature_dim=predictor.model.config.feature_dim)
        restarted = TSSMSession(predictor, bank, folder, base_hash=base_hash, use_memory=memory,
                                online_learning=learning, update_steps=args.update_steps)
        restarted.begin_episode(last["scene"], "restart-after", history, "restart-source")
        observed = restarted.forecast("restart", last["direct_action"][:2]).future_feature
        if not np.allclose(expected, observed, atol=1e-6):
            raise ValueError("scene experience or parameters did not survive restart")
        restarted.end_episode()
        bank.close()
        if weights_hash(predictor) != global_hash:
            raise ValueError("online scene updates changed the global world model")
        np.savez_compressed(folder / "predictions.npz", prediction=np.stack(forecasts), target=np.stack(targets))
        results[name] = {"categories": aggregate(records), "episodes": len(records),
            "accepted_updates": sum(bool(r["feedback_after_reveal"]["update"] and r["feedback_after_reveal"]["update"]["accepted_on_observed_replay"]) for r in records),
            "parameter_change_events": sum(bool(r["feedback_after_reveal"]["update"] and r["feedback_after_reveal"]["update"]["parameters_changed"]) for r in records),
            "global_weights_unchanged": True, "restart_verified": True}
        (folder / "summary.json").write_text(json.dumps(results[name], indent=2) + "\n")
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
