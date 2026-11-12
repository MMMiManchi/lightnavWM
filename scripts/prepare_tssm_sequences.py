"""Freeze continuous route sequences and encode only new observed anchor RGB."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.branch_curation import CurationRules, image_quality, sample_id  # noqa: E402
from experiments.world_model.branch_training_data import load_release, read_rows  # noqa: E402
from experiments.world_model.feature_cache import (  # noqa: E402
    AUDITED_PREPROCESSOR,
    SIGLIP_PREPROCESS_SHA256,
    SIGLIP_REVISION,
    SIGLIP_WEIGHT_SHA256,
)
from experiments.world_model.jsonutil import sha256_file  # noqa: E402
from experiments.world_model.tssm_sequences import route_actions, route_anchors  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(data, output):
    rows, arrays, bindings = load_release(data)
    output.mkdir(parents=True, exist_ok=False)
    selected = json.loads((data / "selected_execution_sources.json").read_text())["tasks"]
    images, sequences = {}, []
    for primary_index, event in enumerate(rows):
        for repeat in range(2):
            tid = sample_id(event["origin_id"], event["exit_id"], repeat)
            folder = Path(selected[tid]) / "executions" / tid
            result = json.loads((folder / "result.json").read_text())
            if not result["event_complete"] or result["target_rgb"]["sha256"] != event["target"]["rgb"]["sha256"]:
                raise ValueError("repeat is not a verified repeat of the primary event")
            history = read_rows(folder / "source_history.jsonl")
            commits = read_rows(folder / "committed.jsonl")
            if [c["index"] for c in commits] != list(range(1, result["branch_steps"] + 1)):
                raise ValueError("cannot splice missing executed observations")
            frames = [history[-1], *[c["frame"] for c in commits[:result["target_index"]]]]
            anchors, path = route_anchors(frames)
            refs, indices = [], []
            for index in anchors:
                frame = frames[index]
                rgb = Path(frame.get("rgb_path", str(folder / "assets" / f"{frame['name']}_rgb.npy")))
                if sha256_file(rgb) != frame["rgb_sha256"]:
                    raise ValueError("raw observed anchor changed")
                if index not in {0, len(frames) - 1} and not image_quality(np.load(rgb), CurationRules(maximum_black_fraction=0.2))["usable"]:
                    continue
                refs.append({"path": str(rgb), "sha256": frame["rgb_sha256"],
                             "kind": "reviewed_source" if index == 0 else event["target_kind"] if index == len(frames) - 1 else "observed_route_anchor"})
                indices.append(index)
            if len(refs) > 32:
                raise ValueError("sequence exceeds the registered 32 observed-state ceiling")
            history_refs = [{"path": h["rgb_path"], "sha256": h["rgb_sha256"]} for h in history]
            for ref in history_refs + refs:
                images.setdefault(ref["sha256"], {"path": ref["path"], "sha256": ref["sha256"]})
            bearing = event["inputs"]["requested_skill"]["bearing_left_rad"]
            sequences.append({"sample_id": tid, "primary_index": primary_index, "primary": tid == event["sample_id"],
                "role": event["role"], "scene": event["scene"], "origin_id": event["origin_id"],
                "exit_id": event["exit_id"], "repeat": repeat, "history_rgb": history_refs,
                "observations": refs, "actions": route_actions(len(refs) - 1, bearing).tolist(),
                "direct_action": [float(np.sin(bearing)), float(np.cos(bearing)), 0.0, 1.0],
                "weight": float(arrays["weights"][primary_index]) / 2,
                "target_label": int(arrays["target_labels"][primary_index]),
                "offline_only": {"evidence_dir": str(folder), "indices": indices,
                    "elapsed_seconds": [i * 0.1 for i in indices], "path_m": [float(path[i]) for i in indices]},
                "intermediate_anchors_are_decision_nodes": False})
            for p in [folder / "result.json", folder / "committed.jsonl", folder / "source_history.jsonl"]:
                bindings[str(p)] = sha256_file(p)
    sequences.sort(key=lambda r: (r["role"], r["scene"], r["origin_id"], r["exit_id"], r["repeat"]))
    for name, values in [("sequences.jsonl", sequences), ("images.jsonl", list(images.values()))]:
        (output / name).write_text("".join(json.dumps(v) + "\n" for v in values))
    save(output / "protocol.json", {"version": "event-tssm-sequences/0.1", "source_release": str(data),
        "stride_m": 2.0, "minimum_final_gap_m": 1.0, "primary_target_changed": False,
        "route_anchors_are_decision_nodes": False, "repeat_visits_are_independent_training_places": False,
        "cross_episode_order": "declared replay: scene, origin, exit, repeat; not claimed simulator wall-clock chronology",
        "future_executed_controls_used_as_prior_inputs": False, "sealed_opened": False})
    sources = [Path(__file__).resolve(), REPO / "experiments/world_model/tssm_sequences.py"]
    for p in sources:
        target = output / "source_snapshot" / p.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    save(output / "input_manifest.json", bindings)
    summary = {role: {"replay_episodes": sum(r["role"] == role for r in sequences),
        "independent_events": sum(r["role"] == role and r["primary"] for r in sequences),
        "state_counts": dict(Counter(len(r["observations"]) for r in sequences if r["role"] == role)),
        "episodes_with_multiple_transitions": sum(r["role"] == role and len(r["observations"]) >= 3 for r in sequences)}
        for role in ("fit", "development")}
    save(output / "summary.json", {"roles": summary, "unique_images": len(images)})
    print(json.dumps(summary), flush=True)


def encode(output, batch_size):
    import torch
    from transformers import SiglipImageProcessor, SiglipVisionModel

    protocol = json.loads((output / "protocol.json").read_text())
    source = Path(protocol["source_release"])
    refs = read_rows(output / "images.jsonl")
    encoder = Path.home() / ".cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots" / SIGLIP_REVISION
    if sha256_file(encoder / "model.safetensors") != SIGLIP_WEIGHT_SHA256 or sha256_file(AUDITED_PREPROCESSOR) != SIGLIP_PREPROCESS_SHA256:
        raise ValueError("audited visual encoder changed")
    with np.load(source / "features/image_features.npz", allow_pickle=False) as old:
        previous = dict(zip(old["image_sha256"].tolist(), old["pooler_raw"], strict=True))
        mean, scale = old["feature_mean"].copy(), old["feature_scale"].copy()
    pending = [r for r in refs if r["sha256"] not in previous]
    if pending:
        if not torch.cuda.is_available():
            raise RuntimeError("new anchor encoding requires CUDA")
        torch.set_num_threads(2)
        processor = SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
        model = SiglipVisionModel.from_pretrained(encoder, local_files_only=True, dtype=torch.float32).to("cuda:0").eval()
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            pixels = []
            for r in batch:
                if sha256_file(Path(r["path"])) != r["sha256"]:
                    raise ValueError("observed anchor image changed")
                pixels.append(np.load(r["path"], allow_pickle=False))
            with torch.inference_mode():
                raw = model(processor(images=pixels, return_tensors="pt")["pixel_values"].to("cuda:0")).pooler_output.float().cpu().numpy()
            previous.update({r["sha256"]: v for r, v in zip(batch, raw, strict=True)})
            if offset % 160 == 0:
                print(json.dumps({"new_images_encoded": offset + len(batch), "total_new": len(pending)}), flush=True)
    keys = [r["sha256"] for r in refs]
    raw = np.stack([previous[k] for k in keys])
    z = (raw - mean) / scale
    normalized = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    np.savez_compressed(output / "image_features.npz", image_sha256=np.array(keys), pooler_raw=raw,
                        pooler_normalized=normalized, feature_mean=mean, feature_scale=scale)
    sequences = read_rows(output / "sequences.jsonl")
    n, length, dim = len(sequences), max(len(r["observations"]) for r in sequences), 1152
    lookup = dict(zip(keys, normalized, strict=True))
    observations = np.zeros((n, length, dim), dtype=np.float32)
    history = np.zeros((n, 8, dim), dtype=np.float32)
    valid = np.zeros((n, length), dtype=bool)
    actions = np.zeros((n, length - 1, 4), dtype=np.float32)
    memory = np.zeros((n, 3, dim * 2 + 2), dtype=np.float32)
    memory_valid = np.zeros((n, 3), dtype=bool)
    past = defaultdict(list)
    memory_refs = []
    for i, row in enumerate(sequences):
        obs = np.stack([lookup[r["sha256"]] for r in row["observations"]])
        observations[i, :len(obs)], valid[i, :len(obs)] = obs, True
        history[i] = np.stack([lookup[r["sha256"]] for r in row["history_rgb"]])
        actions[i, :len(obs) - 1] = row["actions"]
        direction = np.array(row["direct_action"][:2])
        matches = [(float(obs[0] @ s), sid, s, f, b) for sid, s, f, b in past[row["scene"]]
                   if obs[0] @ s >= 0.9 and direction @ b >= 0.98]
        matches.sort(key=lambda x: -x[0])
        selected = matches[:3]
        for j, (_similarity, _sid, s, f, b) in enumerate(selected):
            memory[i, j], memory_valid[i, j] = np.r_[s, f, b], True
        memory_refs.append({"sample_id": row["sample_id"], "past_episode_ids": [v[1] for v in selected]})
        past[row["scene"]].append((row["sample_id"], obs[0], obs[-1], direction))
    np.savez_compressed(output / "tensors.npz", history=history, observations=observations, valid=valid,
        actions=actions, memory=memory, memory_valid=memory_valid,
        direct_action=np.array([r["direct_action"] for r in sequences], dtype=np.float32),
        fit=np.array([r["role"] == "fit" for r in sequences]), primary=np.array([r["primary"] for r in sequences]),
        labels=np.array([r["target_label"] for r in sequences]), weights=np.array([r["weight"] for r in sequences], dtype=np.float32),
        feature_mean=mean, feature_scale=scale)
    (output / "memory_provenance.jsonl").write_text("".join(json.dumps(r) + "\n" for r in memory_refs))
    save(output / "feature_contract.json", {"revision": SIGLIP_REVISION, "weights_sha256": SIGLIP_WEIGHT_SHA256,
        "normalization": "unchanged historical fit-only statistics then L2", "new_images": len(pending),
        "reused_images": len(refs) - len(pending), "shape": list(observations.shape),
        "past_memory_is_prior_real_observation": True, "global_training_uses_fit_only": True})
    print(json.dumps({"sequences": n, "max_states": length, "new_images": len(pending)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "encode"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.data is None:
            parser.error("prepare requires --data")
        prepare(args.data.resolve(), args.output.resolve())
    else:
        encode(args.output.resolve(), args.batch_size)
