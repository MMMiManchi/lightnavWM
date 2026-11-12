"""Curate recorded fit/development branches without opening sealed trajectories.

Run --phase curate first. Optional frozen visual teacher workers read only the
prepared causal node views; --phase summarize combines their silver labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.branch_curation import (  # noqa: E402
    FORMAT_SOURCE,
    VISIBLE_ROLES,
    CurationRules,
    branch_pair_summary,
    deduplicate_events,
    image_quality,
    load_floor_polygons,
    region_entry,
    sample_id,
    student_inputs,
)

NODE_PROMPT = """Classify the CURRENT first-person image for visual navigation. Older images,
if supplied, occurred BEFORE the current image. No future image is supplied.
Use only visible evidence. Do not assume this is a junction or an expert route.
A passage is an immediate doorway/corridor with a plausible floor path. Gaps
between chairs, table legs, furniture, black unscanned areas, and a distant door
seen through a nearer door are NOT independent branches. Do not count the same
entrance twice. Do not infer exits behind the camera.
label=decision: at least TWO distinct, plausible immediate continuations require
a choice here. label=ordinary: a room interior or corridor with one clear
continuation and no visible immediate choice. label=observe: a blind corner,
blocked/dead-end view, or occlusion requires another observation. label=unknown:
the image does not support any of these reliably. Ordinary describes this view,
not proof that there is no junction outside the field of view.
Return JSON only: {"label":"decision|ordinary|observe|unknown",
"visible_passages":0,"confidence":0.0,"reason":"at most 30 words"}.
Confidence is your uncertainty, not a calibrated accuracy estimate.
"""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def save_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("evidence path escapes source batch")
    return path


def curate(args) -> None:
    batch, output = args.batch.resolve(), args.output.resolve()
    records = read_rows(batch / "transitions.jsonl")
    roles = defaultdict(set)
    for row in records:
        if row.get("scene") and row.get("role"):
            roles[row["scene"]].add(row["role"])
    if any(len(value) != 1 for value in roles.values()):
        raise ValueError("a house occurs in multiple roles")
    output.mkdir(parents=True, exist_ok=False)
    rules = CurationRules()
    protocol = {
        "rules": rules.record(), "source_batch": str(batch), "format_source": FORMAT_SOURCE,
        "visible_roles": sorted(VISIBLE_ROLES), "sealed_policy": "metadata_only_no_trajectory_access",
        "target_kind": "first_confidently_entered_different_annotated_region",
        "region_entry_proves_decision_node": False, "actor_imitation_labels": False,
        "node_teacher": {"prompt": NODE_PROMPT, "quality": "machine_silver_not_ground_truth",
                         "minimum_confidence": 0.8, "base_weight": 0.25},
        "history": "recorded RGB at or before the input observation; source events have one frame",
        "future_controls_as_student_inputs": False, "runtime_official_graph_lookup": False,
    }
    save(output / "protocol.json", protocol)
    tracked = {}

    def track(path):
        path = path.resolve()
        tracked.setdefault(str(path), digest(path))

    for name in ("transitions.jsonl", "protocol.json", "supplement_protocol.json", "split.json",
                 "sealed_manifest.json", "exposure.json", "input_hashes.json"):
        if (batch / name).exists():
            track(batch / name)
    for path in (Path(__file__).resolve(), REPO / "experiments/world_model/branch_curation.py"):
        track(path)
        dest = output / "source_snapshot" / path.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    polygons = {}
    assets = {}

    def image_ref(evidence, frame, kind="rgb"):
        if not re.fullmatch(r"node|branch-\d+", frame["name"]):
            raise ValueError("unexpected frame name")
        path = evidence / "assets" / f"{frame['name']}_{kind}.npy"
        expected = frame[f"{kind}_sha256"]
        if str(path) not in assets:
            actual = digest(path)
            if actual != expected:
                raise ValueError(f"image hash mismatch: {path}")
            track(path)
            pixels = np.load(path, allow_pickle=False)
            quality = image_quality(pixels, rules) if kind == "rgb" else {
                "usable": bool(pixels.ndim in {2, 3} and np.isfinite(pixels).all())}
            assets[str(path)] = {"path": str(path), "sha256": actual, "quality": quality}
        elif assets[str(path)]["sha256"] != expected:
            raise ValueError("inconsistent hash for repeated asset")
        return assets[str(path)]

    executions, events = [], []
    node_candidates = {}
    prep_errors = Counter()
    frame_counts = Counter()
    assigned_counts = Counter()
    task_keys = set()
    for row in records:
        # This check must precede resolving evidence, opening logs, or loading .house.
        role = row.get("role")
        if role not in VISIBLE_ROLES:
            continue
        if row.get("status") in {"error", "restore_failed"} or not row.get("evidence_dir"):
            prep_errors[role] += 1
            continue
        key = (row["origin_id"], row["exit_id"], row["repeat_index"])
        if key in task_keys:
            raise ValueError("duplicate committed execution; resolve before curation")
        task_keys.add(key)
        evidence = checked_path(batch, row["evidence_dir"])
        observations = evidence / "observations.jsonl"
        track(observations)
        frames = read_rows(observations)
        command_path = evidence / "commands.jsonl"
        commands = read_rows(command_path) if command_path.exists() else []
        if command_path.exists():
            track(command_path)
        if len(frames) != row["branch_steps"] + 1 or len(commands) != row["branch_steps"]:
            raise ValueError("recorded frame/command counts disagree")
        if [c.get("step") for c in commands] != list(range(len(commands))):
            raise ValueError("non-contiguous executed command steps")
        if frames[0]["rgb_sha256"] != row["pre_decision"]["rgb_sha256"]:
            raise ValueError("source observation does not match restored pre-decision view")
        scene = row["scene"]
        if scene not in polygons:
            house = args.scenes.resolve() / scene / f"{scene}.house"
            track(house)
            polygons[scene] = load_floor_polygons(house)
        entry = region_entry(frames, polygons[scene], rules)
        frame_counts[role] += len(frames)
        assigned_counts[role] += sum(region is not None for region in entry["regions"])
        execution_id = sample_id(*key)
        source = image_ref(evidence, frames[0])
        target_index = entry["target_index"]
        execution = {
            "sample_id": execution_id, "role": role, "scene": scene,
            "origin_id": row["origin_id"], "exit_id": row["exit_id"],
            "repeat_index": row["repeat_index"], "raw_status": row["status"],
            "raw_event_complete": row.get("event_complete") is True,
            "curated_status": entry["status"], "branch_steps": row["branch_steps"],
            "source_region": entry["regions"][0], "target_index": target_index,
            "evidence_dir": str(evidence), "source_quality": source["quality"],
            "censored_outcome": row["status"] if not row.get("event_complete") else None,
            "negative_next_region_label": False,
        }
        if target_index is not None:
            target = image_ref(evidence, frames[target_index])
            target_depth = image_ref(evidence, frames[target_index], "depth")
            if not all(a["quality"]["usable"] for a in (source, target, target_depth)):
                execution["curated_status"] = "region_entry_image_quarantined"
            elif source["sha256"] == target["sha256"]:
                execution["curated_status"] = "identical_source_target_quarantined"
            else:
                actual_prefix = commands[:target_index]
                event = {
                    "sample_id": execution_id, "role": role, "scene": scene,
                    "origin_id": row["origin_id"], "exit_id": row["exit_id"],
                    "repeat_index": row["repeat_index"], "target_kind": protocol["target_kind"],
                    "inputs": student_inputs(source, row["pre_decision"]["chosen_exit"], row["policy_id"]),
                    "target": {"rgb": target, "depth": target_depth,
                               "horizon_steps": target_index, "horizon_s": entry["horizon_s"]},
                    "offline_labels": {"source_region": entry["regions"][0],
                                       "target_region": entry["regions"][target_index],
                                       "path_m": entry["path_m"], "raw_final_status": row["status"],
                                       "recorded_command_prefix": actual_prefix,
                                       "source_body_position": frames[0]["position"],
                                       "target_body_position": frames[target_index]["position"]},
                    "provenance": {"evidence_dir": str(evidence), "observations_sha256": digest(observations)},
                    "admits_region_prediction_pilot": True, "is_decision_node_ground_truth": False,
                }
                events.append(event)
        executions.append(execution)
        indices = {0, len(frames) // 2, len(frames) - 1}
        if target_index is not None:
            indices.add(target_index)
        for index in sorted(indices):
            current = image_ref(evidence, frames[index])
            if not current["quality"]["usable"]:
                continue
            candidate_key = (role, current["sha256"])
            if candidate_key in node_candidates:
                node_candidates[candidate_key]["occurrences"].append(execution_id)
                continue
            history = [image_ref(evidence, frames[i]) for i in sorted({max(0, index - 10), max(0, index - 5), index})]
            history = [ref for ref in history if ref["quality"]["usable"]]
            node_candidates[candidate_key] = {
                "sample_id": sample_id(role, current["sha256"]), "role": role, "scene": scene,
                "origin_id": row["origin_id"], "frame_index": index,
                "inputs": {"history_rgb": history}, "occurrences": [execution_id],
                "label": "unreviewed", "training_mask": False,
                "offline_context": {"region": entry["regions"][index], "raw_status": row["status"]},
            }
    canonical = deduplicate_events(events)
    pairs = branch_pair_summary(canonical)
    sealed = [r for r in records if r.get("role") == "sealed_test"]
    sealed_meta = {
        "origin_count": len({r.get("origin_id") for r in sealed}), "execution_count": len(sealed),
        "raw_status_counts": dict(Counter(r.get("status") for r in sealed)),
        "semantic_curation_performed": False, "trajectory_files_opened": 0,
        "source_batch": str(batch), "role": "sealed_test", "kept_in_original_split": True,
    }
    nodes = sorted(node_candidates.values(), key=lambda n: n["sample_id"])
    save_rows(output / "executions.jsonl", executions)
    save_rows(output / "region_events_all_repeats.jsonl", events)
    save_rows(output / "branch_pairs.jsonl", pairs)
    save_rows(output / "node_review_queue.jsonl", nodes)
    for role in sorted(VISIBLE_ROLES):
        save_rows(output / f"region_events_{role}.jsonl", [e for e in canonical if e["role"] == role])
    save(output / "sealed_passthrough.json", sealed_meta)
    # Repair exposure accounting in the derived manifest, never in Cursor's raw files.
    legacy_path = args.legacy.resolve() / "pairs.jsonl"
    legacy = read_rows(legacy_path)
    track(legacy_path)
    excluded = {s for s, rr in roles.items() if not rr <= {"fit"}}
    # Historical format uses train_fit, not necessarily fit; keep both explicit.
    legacy_fit = [r for r in legacy if r.get("partition") in {"fit", "train_fit"} and r["scene"] not in excluded]
    save_rows(output / "legacy_auxiliary_index.jsonl", [
        {"sample_id": r["sample_id"], "scene": r["scene"], "partition": r["partition"],
         "source_pairs": str(legacy_path), "target_kind": "legacy_event_weak_labels",
         "primary_region_benchmark_member": False, "requires_separate_auxiliary_loss": True}
        for r in legacy_fit
    ])
    save(output / "prior_exposure.json", {
        "legacy_partition_scene_counts": {part: len({r["scene"] for r in legacy if r["partition"] == part})
                                          for part in sorted({r["partition"] for r in legacy})},
        "current_scenes": {s: {"role": next(iter(rr)), "prior_partitions": sorted({r["partition"] for r in legacy if r["scene"] == s})}
                           for s, rr in sorted(roles.items())},
        "legacy_fit_auxiliary_rows": len(legacy_fit), "sealed_is_new_house_blind_claim": False,
    })
    save(output / "training_contract.json", {
        "event_input_keys": ["history_rgb", "history_valid", "requested_skill"],
        "node_input_keys": ["history_rgb"], "read_pixel_arrays_only_from_rgb_refs": True,
        "never_feed": ["offline_labels", "offline_context", "target", "metadata", "scene", "origin_id",
                       "sample_id", "exit_id", "repeat_index", "paths_as_text", "horizon_s"],
        "actor_imitation": False, "source_history_frames": 1,
        "primary_target": protocol["target_kind"], "node_labels_are_silver": True,
        "normalization_fit_only": True, "use_sample_weight": True,
        "development_gradient_updates": False, "test_kept_sealed": True,
        "retrieval_negatives": "do not make repeats or same-region outcomes automatic hard negatives",
        "scenes_known_from_prior_experiments": True,
    })
    counts = {}
    for role in sorted(VISIBLE_ROLES):
        rr = [e for e in executions if e["role"] == role]
        ee = [e for e in events if e["role"] == role]
        cc = [e for e in canonical if e["role"] == role]
        counts[role] = {
            "scenes": len({r["scene"] for r in rr}), "origins": len({r["origin_id"] for r in rr}),
            "executions": len(rr), "recorded_frames": frame_counts[role],
            "uniquely_assigned_region_frames": assigned_counts[role], "preparation_errors": prep_errors[role],
            "curated_status_counts": dict(Counter(r["curated_status"] for r in rr)),
            "raw_candidate_complete": sum(r["raw_event_complete"] for r in rr),
            "confirmed_region_events_with_repeats": len(ee), "deduplicated_region_events": len(cc),
            "confirmed_requested_branches": len({(r["origin_id"], r["exit_id"]) for r in cc}),
            "event_origins": len({r["origin_id"] for r in cc}),
            "salvaged_from_censored_executions": sum(e["offline_labels"]["raw_final_status"] != "complete" for e in ee),
            "paired_origins": dict(Counter(p["state"] for p in pairs if p["role"] == role)),
            "node_review_views": sum(n["role"] == role for n in nodes),
            "horizon_s": {"min": min((e["target"]["horizon_s"] for e in cc), default=None),
                          "max": max((e["target"]["horizon_s"] for e in cc), default=None)},
            "events_by_scene": dict(Counter(e["scene"] for e in cc)),
        }
    save(output / "counts.json", counts)
    save(output / "input_manifest.json", tracked)
    if not all(digest(Path(path)) == expected for path, expected in tracked.items()):
        raise ValueError("source files changed during curation")
    write_report(output, counts, sealed_meta)
    print(json.dumps({"output": str(output), "counts": counts}, ensure_ascii=False), flush=True)


def write_report(output, counts, sealed_meta, teacher=None):
    rows = ["# Branch dataset curation", "", "Raw data are immutable. Fit/development only were reannotated.",
            "The target is a confirmed entry into a different annotated room-like region; it is not a decision-node label.",
            "", "| Role | Houses | Origins | Raw candidate complete | Confirmed entries incl. repeats | Deduplicated entries | Recovered from censored |", "|---|---:|---:|---:|---:|---:|---:|"]
    for role, c in counts.items():
        rows.append(f"| {role} | {c['scenes']} | {c['origins']} | {c['raw_candidate_complete']} | {c['confirmed_region_events_with_repeats']} | {c['deduplicated_region_events']} | {c['salvaged_from_censored_executions']} |")
    rows += ["", "Region metadata and recorded future commands are offline supervision, excluded from model inputs.",
             "Unconfirmed/blocked records are retained in executions.jsonl; lack of confirmation is not proof that no next place exists.",
             "Exact repeats are collapsed; sample weights balance houses, origins and requested exits.",
             "Same-region paired exits are not automatically contrastive negatives.",
             "Region-entry data cover only one event type. Within-room junctions/turns remain unverified.",
             f"Sealed: {sealed_meta['execution_count']} records remain in the original batch; zero sealed trajectory files opened.",
             "Old fit events are indexed as a separate weak auxiliary source, not added to the strong counts.",
             "", "## Coverage gaps", ""]
    for role, c in counts.items():
        rows.append(f"- {role}: branch pairs {json.dumps(c['paired_origins'])}; node review views {c['node_review_views']}.")
    rows += ["- Additional collection should target visually verified alternate exits at the same state, continue until a genuine room/junction event, and log censored outcomes.",
             "- Collect ordinary interiors, blind corners and occluded openings for the node detector; do not relabel depth-gap counts as verified nodes.",
             "- Expand fit/development houses with preregistered splits; keep current sealed houses and trajectories closed.",
             "- These counts support a small region-prediction pilot, not claims of generalization or unseen-branch competence."]
    if teacher is not None:
        rows += ["", "## Frozen visual teacher", "", "Machine silver labels, not independently verified ground truth.",
                 "Only current/past RGB was supplied. Teacher confidence is not calibrated accuracy.",
                 "```json", json.dumps(teacher, indent=2), "```"]
    (output / "report.md").write_text("\n".join(rows) + "\n")


def annotate(args) -> None:
    import torch
    from PIL import Image

    from experiments.world_model.visual_harness import LocalQwenVision, parse_object

    output = args.output.resolve()
    if not 0 <= args.worker < args.workers:
        raise ValueError("worker index out of range")
    queue = read_rows(output / "node_review_queue.jsonl")
    destination = output / "node_teacher" / f"worker_{args.worker:02d}.jsonl"
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(2)
    vision = LocalQwenVision(args.model.resolve(), device="cuda:0")
    with destination.open("x") as handle:
        for index, node in enumerate(queue):
            if index % args.workers != args.worker:
                continue
            images = []
            for i, ref in enumerate(node["inputs"]["history_rgb"]):
                path = Path(ref["path"])
                if digest(path) != ref["sha256"]:
                    raise ValueError("node image changed after curation")
                label = "CURRENT" if i == len(node["inputs"]["history_rgb"]) - 1 else f"PAST {i}"
                images.append((label, Image.fromarray(np.load(path, allow_pickle=False))))
            raw = vision(NODE_PROMPT, images, max_new_tokens=220)
            try:
                parsed = parse_object(raw)
                label = parsed.get("label", "unknown")
                confidence = float(parsed.get("confidence", 0.0))
                if label not in {"decision", "ordinary", "observe", "unknown"} or not 0 <= confidence <= 1:
                    raise ValueError("invalid teacher class/confidence")
                passages = parsed.get("visible_passages")
                if not isinstance(passages, int) or passages < 0 or (label == "decision" and passages < 2):
                    raise ValueError("inconsistent visible-passage count")
                error = None
            except (ValueError, TypeError, AttributeError) as exc:
                parsed, label, confidence, error = {}, "unknown", 0.0, str(exc)
            record = {"sample_id": node["sample_id"], "label": label, "confidence": confidence,
                      "parsed": parsed, "raw_response": raw, "error": error,
                      "input_rgb_sha256": [ref["sha256"] for ref in node["inputs"]["history_rgb"]],
                      "label_source": "frozen_Qwen3-VL-4B-Instruct_current_past_only"}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"worker": args.worker, "index": index, "label": label}), flush=True)
    save(destination.with_suffix(".meta.json"), {
        "model": str(args.model.resolve()), "model_asset_manifest_sha256": digest(args.model / "asset_manifest.json"),
        "prompt_sha256": hashlib.sha256(NODE_PROMPT.encode()).hexdigest(),
        "queue_sha256": digest(output / "node_review_queue.jsonl"),
        "script_sha256": digest(Path(__file__)), "workers": args.workers, "worker": args.worker,
    })


def summarize(args) -> None:
    output = args.output.resolve()
    queue = read_rows(output / "node_review_queue.jsonl")
    files = sorted((output / "node_teacher").glob("worker_??.jsonl"))
    if len(files) != args.workers:
        raise ValueError("teacher workers incomplete")
    predictions = {}
    for path in files:
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        if meta["queue_sha256"] != digest(output / "node_review_queue.jsonl"):
            raise ValueError("teacher queue changed")
        for row in read_rows(path):
            if row["sample_id"] in predictions:
                raise ValueError("duplicate teacher sample")
            predictions[row["sample_id"]] = row
    if set(predictions) != {row["sample_id"] for row in queue}:
        raise ValueError("teacher coverage does not match queue")
    combined = []
    for node in queue:
        pred = predictions[node["sample_id"]]
        if pred["input_rgb_sha256"] != [ref["sha256"] for ref in node["inputs"]["history_rgb"]]:
            raise ValueError("teacher inputs do not match causal images")
        node["label"] = pred["label"]
        node["teacher"] = pred
        node["label_quality"] = "machine_silver"
        node["training_mask"] = pred["label"] != "unknown" and pred["confidence"] >= 0.8 and not pred["error"]
        combined.append(node)
    frequencies = Counter((n["role"], n["scene"], n["label"]) for n in combined if n["training_mask"])
    stats = {}
    for role in sorted(VISIBLE_ROLES):
        rows = [n for n in combined if n["role"] == role]
        for row in rows:
            row["sample_weight"] = 0.25 / frequencies[(role, row["scene"], row["label"])] if row["training_mask"] else 0.0
        save_rows(output / f"node_silver_{role}.jsonl", rows)
        stats[role] = {"views": len(rows), "labels": dict(Counter(n["label"] for n in rows)),
                       "training_labels": dict(Counter(n["label"] for n in rows if n["training_mask"])),
                       "unmasked_views": sum(n["training_mask"] for n in rows)}
    save(output / "node_silver_counts.json", stats)
    write_report(output, json.loads((output / "counts.json").read_text()),
                 json.loads((output / "sealed_passthrough.json").read_text()), stats)
    print(json.dumps(stats, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("curate", "annotate", "summarize"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=Path, default=REPO / "output/world_model_data/branch_benchmark_20261105_01")
    parser.add_argument("--scenes", type=Path, default=REPO / "data/scene_datasets/mp3d")
    parser.add_argument("--legacy", type=Path, default=REPO / "output/world_model_training/event_latent_v1_20261104_01")
    parser.add_argument("--model", type=Path, default=REPO / "checkpoints/world_model_planner/Qwen3-VL-4B-Instruct")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    {"curate": curate, "annotate": annotate, "summarize": summarize}[args.phase](args)


if __name__ == "__main__":
    main()
