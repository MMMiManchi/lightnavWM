"""Cache frozen, audited SigLIP features for the curated nonsealed data.

No world-model or navigation parameters are trained. Existing fit-only feature
normalization is inherited explicitly, so old memory and new targets agree.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.feature_cache import (  # noqa: E402
    AUDITED_PREPROCESSOR,
    SIGLIP_PREPROCESS_SHA256,
    SIGLIP_REVISION,
    SIGLIP_WEIGHT_SHA256,
)
from scripts.curate_branch_benchmark import digest, read_rows, save, save_rows  # noqa: E402


def main():
    import torch
    from transformers import SiglipImageProcessor, SiglipVisionModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--normalization-run", type=Path, default=REPO / "output/world_model_training/event_latent_v1_20261104_01")
    parser.add_argument("--encoder", type=Path, default=Path.home() / ".cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots" / SIGLIP_REVISION)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("frozen encoder requires CUDA; no CPU fallback")
    if not 1 <= args.batch_size <= 32:
        raise ValueError("batch size must be in 1..32")
    data, output = args.data.resolve(), args.output.resolve()
    if digest(args.encoder / "model.safetensors") != SIGLIP_WEIGHT_SHA256:
        raise ValueError("audited SigLIP weights changed")
    if digest(AUDITED_PREPROCESSOR) != SIGLIP_PREPROCESS_SHA256:
        raise ValueError("audited preprocessor changed")
    events = [r for role in ("fit", "development") for r in read_rows(data / f"region_events_{role}.jsonl")]
    nodes = read_rows(data / "node_review_queue.jsonl")
    images, roles = {}, defaultdict(set)
    for row in [*events, *nodes]:
        if row["role"] not in {"fit", "development"}:
            raise ValueError("unexpected sealed role in visible feature inputs")
        refs = row["inputs"]["history_rgb"] + ([row["target"]["rgb"]] if "target" in row else [])
        for ref in refs:
            images.setdefault(ref["sha256"], ref)
            roles[ref["sha256"]].add(row["role"])
    if any(len(value) > 1 for value in roles.values()):
        raise ValueError("identical image crosses fit/development roles")
    normalization = args.normalization_run.resolve() / "tensors.npz"
    contract_path = args.normalization_run.resolve() / "feature_contract.json"
    contract = json.loads(contract_path.read_text())
    if contract["revision"] != SIGLIP_REVISION or contract["normalization"] != "fit-only channel standardization then L2":
        raise ValueError("incompatible inherited normalization")
    with np.load(normalization, allow_pickle=False) as cached:
        mean, scale = cached["feature_mean"].copy(), cached["feature_scale"].copy()
    if mean.shape != (1152,) or scale.shape != (1152,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid inherited fit-only statistics")
    output.mkdir(parents=True, exist_ok=False)
    sources = [data / "protocol.json", data / "region_events_fit.jsonl", data / "region_events_development.jsonl",
               data / "node_review_queue.jsonl", normalization, contract_path, AUDITED_PREPROCESSOR, Path(__file__).resolve()]
    manifests = {str(p): digest(p) for p in sources}
    save(output / "input_manifest.json", manifests)
    snapshot = output / "source_snapshot" / Path(__file__).resolve().relative_to(REPO)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), snapshot)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    processor = SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
    model = SiglipVisionModel.from_pretrained(args.encoder, local_files_only=True, dtype=torch.float32).to("cuda:0").eval()
    ids = sorted(images)
    pooled, columns = [], []
    for start in range(0, len(ids), args.batch_size):
        selected = ids[start:start + args.batch_size]
        pixels = []
        for key in selected:
            ref = images[key]
            if digest(Path(ref["path"])) != key:
                raise ValueError("RGB asset changed before encoding")
            pixels.append(np.load(ref["path"], allow_pickle=False))
        inputs = processor(images=pixels, return_tensors="pt")["pixel_values"].to("cuda:0")
        with torch.inference_mode():
            features = model(inputs)
            p = features.pooler_output.float().cpu().numpy()
            c = features.last_hidden_state.reshape(len(selected), 27, 27, 1152).mean(dim=1).float().cpu().numpy()
        if p.shape != (len(selected), 1152) or not np.isfinite(p).all() or not np.isfinite(c).all():
            raise ValueError("invalid frozen encoder output")
        pooled.append(p)
        columns.append(c)
        print(json.dumps({"encoded": min(len(ids), start + args.batch_size), "total": len(ids)}), flush=True)
    raw = np.concatenate(pooled)
    spatial = np.concatenate(columns)
    standardized = (raw - mean) / scale
    normalized = standardized / np.maximum(np.linalg.norm(standardized, axis=1, keepdims=True), 1e-8)
    lookup = {key: i for i, key in enumerate(ids)}
    np.savez_compressed(output / "image_features.npz", image_sha256=np.array(ids), pooler_raw=raw,
                        column_tokens_raw=spatial, pooler_normalized=normalized,
                        feature_mean=mean, feature_scale=scale)
    history = np.zeros((len(events), 8, 1152), dtype=np.float32)
    valid = np.zeros((len(events), 8), dtype=bool)
    targets = []
    for i, event in enumerate(events):
        history[i, -1] = normalized[lookup[event["inputs"]["history_rgb"][-1]["sha256"]]]
        valid[i, -1] = True
        targets.append(normalized[lookup[event["target"]["rgb"]["sha256"]]])
    targets = np.stack(targets)
    branches = [event["inputs"]["requested_skill"]["bearing_left_rad"] for event in events]
    np.savez_compressed(output / "region_tensors.npz", sample_ids=np.array([e["sample_id"] for e in events]),
        history=history, valid=valid, target=targets,
        branch=np.array([[math.sin(b), math.cos(b)] for b in branches], dtype=np.float32),
        fit=np.array([e["role"] == "fit" for e in events]),
        weights=np.array([e["sample_weight"] for e in events], dtype=np.float32),
        feature_mean=mean, feature_scale=scale)
    save_rows(output / "samples.jsonl", [{"index": i, "sample_id": e["sample_id"], "role": e["role"],
               "scene": e["scene"], "target_kind": e["target_kind"]} for i, e in enumerate(events)])
    angles = np.degrees(np.arccos(np.clip(np.sum(history[:, -1] * targets, axis=1), -1, 1)))
    diagnostics = {role: {"count": sum(e["role"] == role for e in events),
        "mean_current_to_target_feature_angle_degrees": float(angles[[e["role"] == role for e in events]].mean())}
        for role in ("fit", "development")}
    save(output / "feature_contract.json", {"encoder": "google/siglip-so400m-patch14-384", "revision": SIGLIP_REVISION,
        "weights_sha256": SIGLIP_WEIGHT_SHA256, "preprocessor_sha256": SIGLIP_PREPROCESS_SHA256,
        "dtype": "float32", "images": len(ids), "pooler_shape": list(raw.shape), "column_tokens_shape": list(spatial.shape),
        "column_operation": "mean over 27 vertical patches; 27 horizontal columns retained",
        "pooler_normalization": "frozen historical train_fit channel statistics then L2",
        "normalization_run": str(args.normalization_run.resolve()), "normalization_fit_statistics_image_count": contract["fit_statistics_image_count"],
        "column_tokens_normalized": False, "history": "8 slots; only last slot valid, no invented source history",
        "branch": "[sin(bearing_left_rad), cos(bearing_left_rad)]", "world_model_trained": False,
        "primary_event_kind": "annotated_region_entry_not_verified_decision_node", "sealed_images_encoded": 0,
        "distance_diagnostics_not_accuracy_or_robot_turn_angle": diagnostics})
    if not all(digest(Path(path)) == expected for path, expected in manifests.items()):
        raise ValueError("feature inputs changed during encoding")
    print(json.dumps({"output": str(output), "images": len(ids), "events": len(events), "diagnostics": diagnostics}))


if __name__ == "__main__":
    main()
