"""Replay recorded queries against strictly completed prior episodes across shards."""

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.entrance_alignment import ImageDirectionTransport  # noqa: E402
from experiments.world_model.jsonutil import sha256_file  # noqa: E402
from experiments.world_model.memory_routing import MemoryRoutingPolicy  # noqa: E402
from experiments.world_model.scene_memory import PersistentSceneMemory  # noqa: E402
from experiments.world_model.task_planning import CurrentBranch  # noqa: E402
from experiments.world_model.visual_harness import ObservationImages  # noqa: E402


def run(args):
    facts, signature, refs = [], None, {}
    for path in sorted(args.source.glob("gpu*/experience/facts.sqlite")):
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        metadata = json.loads(db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()[0])
        if signature is not None and metadata != signature:
            raise ValueError("shards have different feature contracts")
        signature = metadata
        for row in db.execute("SELECT context,episode,event,source_observation,target_observation,execution_ref,source_feature,future_feature,branch FROM experiences WHERE outcome='arrived'"):
            context, episode, event, source, target, execution, raw_source, raw_future, branch = row
            controller = Path(execution).parent
            done, query = controller.parent / "result.json", controller / f"query-{int(source.rsplit(':', 1)[-1]):04d}.json"
            if not done.exists() or not query.exists():
                continue
            facts.append(dict(context=context, episode_id=episode, event_id=event, source_observation=source,
                target_observation=target, execution_ref=execution, source_feature=np.frombuffer(raw_source, dtype='<f4').copy(),
                observed_future=np.frombuffer(raw_future, dtype='<f4').copy(), branch_direction=np.frombuffer(branch, dtype='<f4').copy(),
                completed=done.stat().st_mtime, query_time=query.stat().st_mtime, query_file=query,
                shard=path.parent.parent.name))
        db.close()
        index = path.parent / "images/images.jsonl"
        for line in index.open():
            item = json.loads(line)
            refs[item["observation"]] = (index.parent / item["file"], item["sha256"])
    if not facts:
        raise ValueError("no completed actual event facts")
    args.output.mkdir(parents=True, exist_ok=False)
    bank = PersistentSceneMemory(args.output / "replay.sqlite", contract_id=signature["contract"], feature_dim=signature["dimension"])
    images = ObservationImages()
    for fact in facts:
        for key in ["source_observation", "target_observation"]:
            path, digest = refs[fact[key]]
            images.register_file(fact[key], path, digest)
    transport = ImageDirectionTransport(images)
    ordered, recorded, results = sorted(facts, key=lambda r:r["completed"]), set(), []
    for query in sorted(facts, key=lambda r:r["query_time"])[:args.maximum_queries]:
        for fact in ordered:
            key = (fact["context"], fact["episode_id"], fact["event_id"])
            if fact["completed"] >= query["query_time"] or key in recorded:
                continue
            arguments = {k:fact[k] for k in ["context", "episode_id", "event_id", "source_observation",
                "target_observation", "execution_ref", "source_feature", "observed_future", "branch_direction"]}
            bank.record_observed(**arguments, executed=True, arrived=True)
            bank.record_observed(**{**arguments, "context": fact["context"] + "::shard:" + fact["shard"]}, executed=True, arrived=True)
            recorded.add(key)
        bearing = math.atan2(*query["branch_direction"])
        if abs(bearing) >= math.pi/3:
            continue
        payload = json.loads(query["query_file"].read_text())
        depth = payload.get("source_actual_depth")
        if depth and sha256_file(query["query_file"].parent / depth["file"]) != depth["sha256"]:
            raise ValueError("recorded source depth changed")
        transport.current_depth = np.load(query["query_file"].parent / depth["file"], allow_pickle=False) if depth else None
        planning = json.loads((query["query_file"].parent / "planning.json").read_text())
        candidates = [b for row in planning if row.get("source_observation") == query["source_observation"]
                      for b in row.get("current_branches", [])]
        candidate = min(candidates, key=lambda b:abs(b["bearing_rad"]-bearing)) if candidates else {}
        region = tuple(candidate["bbox"]) if candidate.get("bbox") is not None else None
        branch = CurrentBranch("recorded", query["source_observation"], bearing, "actually executed recorded exit", image_region=region)
        transform = transport.for_branch(query["source_observation"], branch)
        routes = {}
        for name, policy in [("legacy_fragmented", MemoryRoutingPolicy(candidate_floor=.9, geometry_source_floor=.98)),
                              ("legacy", MemoryRoutingPolicy(candidate_floor=.9, geometry_source_floor=.98)),
                              ("geometry", MemoryRoutingPolicy())]:
            context = query["context"] + "::shard:" + query["shard"] if name == "legacy_fragmented" else query["context"]
            route = policy.resolve(bank, context=context, cutoff=bank.latest_sequence(context),
                episode_id=query["episode_id"], current=query["source_feature"], direction=query["branch_direction"], transform=transform)
            routes[name] = {"kind": route.kind, "reason": route.reason, "inspected": route.inspected,
                            "matches": [{"sequence":m.sequence, "source_similarity":m.source_similarity,
                                         "outcome_cosine_evaluation_only":float(m.observed_future @ query["observed_future"])}
                                        for m in route.matches]}
        results.append({"query_source": query["source_observation"], "context": query["context"],
                        "prior_facts": bank.count(query["context"]), "routes": routes})
        transport.last_trace.clear()
    summary = {"queries":len(results), "completed_facts":len(facts),
               "fragmented_legacy_direct":sum(r["routes"]["legacy_fragmented"]["kind"] == "memory" for r in results),
               "legacy_direct":sum(r["routes"]["legacy"]["kind"] == "memory" for r in results),
               "geometry_direct":sum(r["routes"]["geometry"]["kind"] == "memory" for r in results),
               "strict_completed_prior_episode_filter":True, "cross_shard_snapshot":True,
               "navigation_rerun":False, "source_banks_modified":False}
    (args.output / "results.json").write_text(json.dumps({"summary":summary, "queries":results}, indent=2)+"\n")
    bank.close()
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-queries", type=int, default=200)
    run(parser.parse_args())
