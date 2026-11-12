"""Release reviewed data and a concrete recollection queue, keeping drafts.

Reviews are explicit machine visual reviews, not human gold labels. Local
geometry can reject an opening proposal; a free ray alone never creates one.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.curate_branch_benchmark import digest, read_rows, save, save_rows  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    reviews = json.loads(args.reviews.read_text())
    source = data / "node_architecture_review"
    nodes = [n for role in ("fit", "development") for n in read_rows(source / f"node_consensus_{role}.jsonl")]
    judged = reviews["verdicts"]
    if not set(judged) <= {n["sample_id"] for n in nodes}:
        raise ValueError("review ids are not in this prepared queue")
    rays = {r["sample_id"]: r for r in json.loads((source / "reviewed_node_rays.json").read_text())}
    output.mkdir(parents=True, exist_ok=False)
    for node in nodes:
        verdict = judged.get(node["sample_id"], {"label": "unknown", "reason": "not_individually_reviewed"})
        label, reason = verdict["label"], verdict["reason"]
        if label not in {"decision", "ordinary", "unknown"}:
            raise ValueError("invalid review label")
        quality = node["inputs"]["history_rgb"][-1]["quality"]
        if quality["black_fraction"] > reviews["maximum_node_black_fraction"]:
            label, reason = "unknown", "substantial_missing_scan_geometry"
        if label == "decision":
            support = rays.get(node["sample_id"])
            if support is None or len(support["rays"]) < 2:
                raise ValueError("reviewed decision lacks geometry corroboration")
            if any(r["walked_m"][1] < 1.35 for r in support["rays"]):
                raise ValueError("reviewed decision includes a locally blocked proposed direction")
            bearings = [math.atan((0.5 - r["u"] / 1000) * 2 * math.tan(math.radians(60))) for r in support["rays"]]
            if abs(bearings[0] - bearings[1]) < math.radians(35):
                raise ValueError("reviewed directions are not sufficiently separated")
        node["draft_label"] = node["label"]
        node["label"], node["review_reason"] = label, reason
        node["label_quality"] = "Codex_visual_review_silver_not_human_gold"
        node["training_mask"] = False
        node["weak_training_mask"] = label != "unknown"
        node["reviewed_by"] = reviews["reviewer"]
        node["scope"] = "decision_evidence_visible_at_current_time_not_global_graph_degree"
    frequency = Counter((n["role"], n["scene"], n["label"], n["origin_id"]) for n in nodes if n["weak_training_mask"])
    origins = defaultdict(set)
    for node in nodes:
        if node["weak_training_mask"]:
            origins[(node["role"], node["scene"], node["label"])].add(node["origin_id"])
    stats = {}
    for role in ("fit", "development"):
        rows = [n for n in nodes if n["role"] == role]
        for n in rows:
            key = (role, n["scene"], n["label"])
            n["sample_weight"] = 0.1 / (len(origins[key]) * frequency[(*key, n["origin_id"])]) if n["weak_training_mask"] else 0.0
        save_rows(output / f"node_reviewed_{role}.jsonl", rows)
        stats[role] = {"views": len(rows), "reviewed_silver_labels": dict(Counter(n["label"] for n in rows if n["weak_training_mask"])),
                       "quarantined_or_unreviewed": sum(not n["weak_training_mask"] for n in rows),
                       "decision_houses": len({n["scene"] for n in rows if n["label"] == "decision"}),
                       "human_verified_labels": 0}
    executions = {e["sample_id"]: e for e in read_rows(data / "executions.jsonl")}
    recollect = []
    for node in nodes:
        if node["label"] != "decision" or node["role"] != "fit":
            continue
        execution = executions[node["occurrences"][0]]
        evidence = Path(execution["evidence_dir"])
        frames = read_rows(evidence / "observations.jsonl")
        if frames[node["frame_index"]]["rgb_sha256"] != node["inputs"]["history_rgb"][-1]["sha256"]:
            raise ValueError("new collection origin does not match its recorded replay prefix")
        proposals = []
        for i, ray in enumerate(rays[node["sample_id"]]["rays"]):
            bearing = math.atan((0.5 - ray["u"] / 1000) * 2 * math.tan(math.radians(60)))
            proposals.append({"exit_id": i, "bearing_left_rad": bearing, "image_bearing_u_0_1000": ray["u"],
                              "side": "center" if abs(bearing) < math.radians(20) else ("left" if bearing > 0 else "right"),
                              "local_navmesh_1_5m_walked": ray["walked_m"][1]})
        recollect.append({
            "priority": 1, "role": "fit", "scene": node["scene"], "new_origin_id": f"reviewed:{node['sample_id']}",
            "current_rgb": node["inputs"]["history_rgb"][-1], "exit_proposals": proposals,
            "replay_recipe": {"restore_original_origin_id": execution["origin_id"],
                              "then_replay_commands": str(evidence / "commands.jsonl"),
                              "command_count": node["frame_index"],
                              "require_current_rgb_sha256": node["inputs"]["history_rgb"][-1]["sha256"]},
            "reason": "independently visible architecture choices; no paired continuations were collected from this exact state",
            "needs_verified_end_event": True, "repeat_each_exit": 2,
            "official_viewpoint_or_adjacency_at_runtime": False, "task_route_label": False,
            "collection_started": False,
        })
    save_rows(output / "recollection_origins.jsonl", recollect)
    save(output / "node_counts.json", stats)
    shutil.copyfile(args.reviews, output / "review_decisions.json")
    save(output / "data_release.json", {
        "region_auxiliary_events": {r: str(data / f"region_events_{r}.jsonl") for r in ("fit", "development")},
        "node_weak_auxiliary_examples": {r: str(output / f"node_reviewed_{r}.jsonl") for r in ("fit", "development")},
        "features": str(data / "features/region_tensors.npz"),
        "image_feature_cache": str(data / "features/image_features.npz"),
        "legacy_weak_auxiliary_index": str(data / "legacy_auxiliary_index.jsonl"),
        "superseded_draft_node_files": ["node_silver_*.jsonl", "node_architecture_review/node_consensus_*.jsonl"],
        "node_strong_supervision_ready": False, "unseen_branch_prediction_training_ready": False,
        "small_region_auxiliary_pilot_ready": True, "actor_imitation": False,
        "inherited_history_limitation": "region samples have one recorded source frame, padded with a valid mask",
        "sealed": json.loads((data / "sealed_passthrough.json").read_text()),
        "new_collection_origins_prepared": len(recollect),
    })
    save(output / "next_collection_targets.json", {
        "status": "proposal_for_a_new_preregistered_round_not_a_claim_of_sufficiency",
        "preserve_current_sealed_test": True, "expert_goal_route_required": False,
        "repair_first": ["verify distinct architectural exits before designating a decision node",
                         "end on actual entered space or independently confirmed junction, not two depth gaps after 0.8m",
                         "accept censored outcomes without relabeling as a successful connection",
                         "retain loops, return visits and same-destination branches as valid experiences"],
        "diagnostic_pilot_target": {
            "fit": {"houses": 10, "independent_decision_states": 50, "exits_per_state": 2, "repeats_per_exit": 2,
                    "distinct_start_exit_interventions": 100, "recorded_completed_events": 200},
            "development": {"houses": 4, "independent_decision_states": 20, "exits_per_state": 2, "repeats_per_exit": 2,
                            "distinct_start_exit_interventions": 40, "recorded_completed_events": 80},
            "different_outcome_same_state_subset": {"fit_origins": 20, "development_origins": 10},
            "ordinary_and_observation_needed_views": "independently check and balance per house; keep ambiguous frames unknown",
            "scene_roles": "expand fit within historical fit houses; development within historical dev houses; do not move sealed or dev data into fit",
            "meaning": "minimum useful diagnostic round, not a guaranteed sample complexity for generalization"},
        "acceptance_depends_on": ["held-out-house event prediction better than current-frame copying and memory-only",
                                  "branch-sensitive predictions under identical source observations",
                                  "strictly past-memory retrieval, including absent answers",
                                  "node precision and recall including ordinary/occluded views",
                                  "new-goal cross-episode navigation without instruction drift"]})
    sources = [args.reviews.resolve(), source / "node_consensus_fit.jsonl", source / "node_consensus_development.jsonl",
               source / "reviewed_node_rays.json", source / "node_ray_audit.json", data / "counts.json",
               data / "executions.jsonl", data / "features/feature_contract.json", Path(__file__).resolve()]
    save(output / "input_manifest.json", {str(p): digest(p) for p in sources})
    snapshot = output / "source_snapshot" / Path(__file__).resolve().relative_to(REPO)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), snapshot)
    print(json.dumps({"node_counts": stats, "recollection_origins": len(recollect)}))


if __name__ == "__main__":
    main()
