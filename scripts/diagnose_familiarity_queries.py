"""Post-run image-only diagnosis of empty memory, visual and direction misses.

Re-encodes saved current planning RGB, with the same frozen feature contract.
No true pose, target state or official graph is used. This never changes policy
actions or relaxes a retrieval gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.event_runtime import load_residual_predictor  # noqa: E402
from experiments.world_model.event_train import digest, save  # noqa: E402
from experiments.world_model.feature_cache import AUDITED_PREPROCESSOR, SIGLIP_WEIGHT_SHA256  # noqa: E402


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("real feature re-encoding requires CUDA")
    torch.set_num_threads(2)
    root, output = args.run_dir.resolve(), args.output.resolve()
    if not (root / "results.json").is_file():
        raise ValueError("diagnostics require a completely finished experiment")
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((root / "config.json").read_text())
    inputs, requests = {}, []
    empty_rounds = Counter()
    for result in sorted(root.glob("worker_*/*/result.json")):
        metadata = json.loads(result.read_text())
        folder = result.parent
        for row in json.loads((folder / "planning.json").read_text()):
            path = folder / "planning_images" / f"step_{row['step']:03d}.png"
            key = digest(path)
            inputs.setdefault(key, path)
            if not row.get("current_branches"):
                empty_rounds[metadata["condition"]] += 1
            requests.append((metadata, folder, row, key))
    from transformers import SiglipImageProcessor, SiglipVisionModel

    snapshot = Path(config["encoder_snapshot"])
    if digest(snapshot / "model.safetensors") != SIGLIP_WEIGHT_SHA256:
        raise ValueError("audited feature encoder changed")
    encoder = SiglipVisionModel.from_pretrained(snapshot, local_files_only=True, dtype=torch.float32).to(args.device).eval()
    processor = SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
    predictor = load_residual_predictor(Path(config["checkpoint"]))
    features = {}
    items = list(inputs.items())
    for start in range(0, len(items), 8):
        batch = items[start:start+8]
        pixels = [Image.open(path).convert("RGB") for _key, path in batch]
        values = encoder(processor(images=pixels, return_tensors="pt")["pixel_values"].to(args.device)).pooler_output.float().cpu()
        values = predictor.normalize(values).numpy()
        features.update((key, value) for (key, _path), value in zip(batch, values, strict=True))
    memories = {}
    rows, counts = [], defaultdict(Counter)
    mismatches = []
    for metadata, folder, request, key in requests:
        if folder not in memories:
            connection = sqlite3.connect(f"file:{folder / 'experience.sqlite'}?mode=ro", uri=True)
            raw = connection.execute("select sequence,source_feature,branch from experiences order by sequence").fetchall()
            connection.close()
            memories[folder] = raw
        raw = memories[folder]
        runtime = {q["candidate_id"]: q for q in request.get("queries", [])}
        for branch in request.get("current_branches", []):
            direction = np.array([math.sin(branch["bearing_rad"]), math.cos(branch["bearing_rad"])], dtype=np.float32)
            source = np.array([float(np.frombuffer(r[1], dtype="<f4") @ features[key]) for r in raw])
            angle = np.array([float(np.frombuffer(r[2], dtype="<f4") @ direction) for r in raw])
            source_ok = source >= .5
            compatible = source_ok & (angle >= math.cos(math.radians(35)))
            reason = ("empty_memory" if not raw else "source_visual_miss" if not source_ok.any()
                      else "direction_gate_reject" if not compatible.any() else "supported")
            actual = runtime.get(branch["id"], {}).get("supported_by_history")
            if actual is not None and actual != bool(compatible.any()):
                mismatches.append({"folder": str(folder), "step": request["step"], "candidate": branch["id"],
                                   "runtime": actual, "reencoded": bool(compatible.any())})
            row = {"scene": metadata["scene"], "partition": metadata["partition"], "condition": metadata["condition"],
                   "history_episodes": metadata["history_episodes"], "history_records": len(raw),
                   "step": request["step"], "candidate_id": branch["id"], "diagnosis": reason,
                   "source_compatible_count": int(source_ok.sum()), "source_and_direction_count": int(compatible.sum()),
                   "best_source_cosine": float(source.max()) if len(source) else None,
                   "runtime_supported": actual, "current_heading_alignment_not_verified": True}
            rows.append(row)
            counts[metadata["condition"]][reason] += 1
    save(output / "results.json", {"run": str(root), "queries": rows,
        "by_condition": dict(counts), "no_candidate_rounds": dict(empty_rounds),
        "unique_current_images_reencoded": len(inputs), "runtime_support_mismatches": mismatches,
        "thresholds": {"source_cosine": .5, "direction_degrees": 35},
        "post_run_only_not_a_policy_intervention": True, "world_pose_and_current_future_not_used": True})
    save(output / "input_images.json", {key: str(path) for key, path in inputs.items()})
    print(json.dumps({"counts": dict(counts), "unique_images": len(inputs), "mismatches": len(mismatches)}), flush=True)


if __name__ == "__main__":
    main()
