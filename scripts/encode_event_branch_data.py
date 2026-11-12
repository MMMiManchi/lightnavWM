"""Freeze SigLIP features for all eight real pre-choice observations and futures."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.branch_benchmark_entry import read_jsonl, write_json  # noqa: E402
from experiments.world_model.branch_feature_pack import pack_events  # noqa: E402
from experiments.world_model.feature_cache import (  # noqa: E402
    AUDITED_PREPROCESSOR,
    SIGLIP_PREPROCESS_SHA256,
    SIGLIP_REVISION,
    SIGLIP_WEIGHT_SHA256,
)
from experiments.world_model.jsonutil import sha256_file  # noqa: E402


def main():
    import torch
    from transformers import SiglipImageProcessor, SiglipVisionModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--normalization-run", type=Path, default=REPO / "output/world_model_training/event_latent_v1_20261104_01")
    parser.add_argument("--encoder", type=Path, default=Path.home() / ".cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots" / SIGLIP_REVISION)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("frozen visual encoder requires CUDA")
    if not 1 <= args.batch_size <= 32:
        raise ValueError("batch size must be 1..32")
    if sha256_file(args.encoder / "model.safetensors") != SIGLIP_WEIGHT_SHA256 or sha256_file(AUDITED_PREPROCESSOR) != SIGLIP_PREPROCESS_SHA256:
        raise ValueError("audited encoder/preprocessor changed")
    events = [e for role in ("fit", "development") for e in read_jsonl(args.data / f"events_{role}.jsonl")]
    nodes = read_jsonl(args.data / "node_labels.jsonl")
    images, roles = {}, defaultdict(set)
    for row in [*events, *nodes]:
        if row["role"] not in {"fit", "development"}:
            raise ValueError("sealed role refused")
        refs = row["inputs"]["history_rgb"] + ([row["target"]["rgb"]] if "target" in row else [])
        for ref in refs:
            images.setdefault(ref["sha256"], ref)
            roles[ref["sha256"]].add(row["role"])
    if any(len(v) != 1 for v in roles.values()):
        raise ValueError("same RGB crosses training/development roles")
    contract_path = args.normalization_run / "feature_contract.json"
    contract = json.loads(contract_path.read_text())
    if contract["revision"] != SIGLIP_REVISION or contract["normalization"] != "fit-only channel standardization then L2":
        raise ValueError("incompatible normalization coordinates")
    stats_path = args.normalization_run / "tensors.npz"
    with np.load(stats_path, allow_pickle=False) as stats:
        mean, scale = stats["feature_mean"].copy(), stats["feature_scale"].copy()
    if mean.shape != (1152,) or scale.shape != (1152,) or np.any(scale <= 0) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("invalid inherited fit-only feature statistics")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [args.data / f"events_{r}.jsonl" for r in ("fit", "development")]
    sources += [args.data / "node_labels.jsonl", stats_path, contract_path, Path(__file__).resolve(), AUDITED_PREPROCESSOR]
    manifest = {str(p.resolve()): sha256_file(p) for p in sources}
    write_json(args.output / "input_manifest.json", manifest)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    processor = SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
    model = SiglipVisionModel.from_pretrained(args.encoder, local_files_only=True, dtype=torch.float32).to("cuda:0").eval()
    ids = sorted(images)
    pooled, columns = [], []
    for start in range(0, len(ids), args.batch_size):
        keys = ids[start:start + args.batch_size]
        arrays = []
        for key in keys:
            path = Path(images[key]["path"])
            if sha256_file(path) != key:
                raise ValueError("actual recorded image changed")
            arrays.append(np.load(path, allow_pickle=False))
        pixels = processor(images=arrays, return_tensors="pt")["pixel_values"].to("cuda:0")
        with torch.inference_mode():
            result = model(pixels)
        pooled.append(result.pooler_output.float().cpu().numpy())
        columns.append(result.last_hidden_state.reshape(len(keys), 27, 27, 1152).mean(dim=1).float().cpu().numpy())
        if start % (args.batch_size * 20) == 0:
            print(json.dumps({"encoded": min(start + len(keys), len(ids)), "total": len(ids)}), flush=True)
    raw, spatial = np.concatenate(pooled), np.concatenate(columns)
    z = (raw - mean) / scale
    vectors = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    if not np.isfinite(vectors).all() or not np.isfinite(spatial).all():
        raise ValueError("non-finite visual features")
    lookup = {key: i for i, key in enumerate(ids)}
    np.savez_compressed(args.output / "image_features.npz", image_sha256=np.array(ids),
        pooler_raw=raw, column_tokens_raw=spatial, pooler_normalized=vectors, feature_mean=mean, feature_scale=scale)
    packed = pack_events(events, vectors, lookup)
    np.savez_compressed(args.output / "event_tensors.npz", **packed, feature_mean=mean, feature_scale=scale)
    (args.output / "samples.jsonl").write_text("".join(json.dumps({"index": i, "sample_id": e["sample_id"],
        "role": e["role"], "scene": e["scene"], "target_kind": e["target_kind"]}) + "\n" for i, e in enumerate(events)))
    angles = np.degrees(np.arccos(np.clip(np.sum(packed["history"][:, -1] * packed["target"], axis=1), -1, 1)))
    diagnostics = {role: {"events": int(sum(e["role"] == role for e in events)),
        "mean_current_to_future_feature_angle_degrees": float(angles[[e["role"] == role for e in events]].mean())}
        for role in ("fit", "development")}
    write_json(args.output / "feature_contract.json", {"revision": SIGLIP_REVISION,
        "weights_sha256": SIGLIP_WEIGHT_SHA256, "preprocessor_sha256": SIGLIP_PREPROCESS_SHA256,
        "dtype": "float32", "unique_images": len(ids), "history_shape": list(packed["history"].shape),
        "target_shape": list(packed["target"].shape), "column_tokens_shape": list(spatial.shape),
        "history": "all eight actual causal frames retained; no invented repeats",
        "pooler_normalization": "inherited historical train_fit statistics then L2",
        "normalization_fit_statistics_image_count": contract["fit_statistics_image_count"],
        "node_features": "indexed by source RGB hash in image_features.npz; labels remain simulation silver",
        "feature_angles_are_not_prediction_accuracy": diagnostics, "world_model_trained": False,
        "LightNav_used": False, "sealed_images_encoded": 0})
    if any(sha256_file(Path(p)) != h for p, h in manifest.items()):
        raise ValueError("feature inputs changed during encoding")
    print(json.dumps({"events": len(events), "images": len(ids), "diagnostics": diagnostics}), flush=True)


if __name__ == "__main__":
    main()
