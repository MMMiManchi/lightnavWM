"""Finalize a trainable tensor contract and check gradients without optimizer updates."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.branch_benchmark_entry import read_jsonl, write_json  # noqa: E402
from experiments.world_model.branch_feature_pack import place_labels  # noqa: E402
from experiments.world_model.jsonutil import sha256_file  # noqa: E402


def main():
    import torch

    from experiments.world_model.event_latent import (
        EventLatentWorldModel,
        EventModelConfig,
        event_objective,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    root = args.data.resolve()
    events = [e for role in ("fit", "development") for e in read_jsonl(root / f"events_{role}.jsonl")]
    with np.load(root / "features/event_tensors.npz", allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    if arrays["sample_ids"].tolist() != [e["sample_id"] for e in events] or not arrays["valid"].all():
        raise ValueError("features lost source histories or sample identity")
    if not all(np.isfinite(arrays[key]).all() for key in ["history", "target", "branch", "weights"]):
        raise ValueError("non-finite training arrays")
    labels, keys = place_labels(events)
    arrays["target_labels"] = labels
    tensor_path = root / "features/training_tensors.npz"
    if tensor_path.exists():
        with np.load(tensor_path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(not np.array_equal(existing[k], v) for k, v in arrays.items()):
                raise ValueError("existing final tensors differ; create a new release")
    else:
        np.savez_compressed(tensor_path, **arrays)
    write_json(root / "features/target_label_keys.json", keys)
    torch.manual_seed(11)
    torch.set_num_threads(2)
    model = EventLatentWorldModel(EventModelConfig())
    fit_ids = np.flatnonzero(arrays["fit"])[:8]
    tensors = {key: torch.from_numpy(arrays[key][fit_ids]) for key in ["history", "valid", "branch", "target", "target_labels"]}
    outputs = model.training_outputs(tensors["history"], tensors["valid"], tensors["branch"], tensors["target"])
    loss, _parts = event_objective(outputs, tensors["target"], tensors["target_labels"])
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError("actual-data model loss or gradient is non-finite")
    model.eval()
    with torch.inference_mode():
        prediction = model.predict(tensors["history"], tensors["valid"], tensors["branch"])
    if prediction.shape != tensors["target"].shape or not torch.isfinite(prediction).all():
        raise ValueError("prior-only prediction contract failed")
    write_json(root / "training_smoke.json", {"fit_samples_used": len(fit_ids),
        "history_shape": list(tensors["history"].shape), "prediction_shape": list(prediction.shape),
        "finite_loss_and_gradients": True, "prior_predict_needs_future_input": False,
        "optimizer_steps": 0, "world_model_trained": False, "LightNav_loaded": False,
        "random_model_smoke_is_not_performance_validation": True})
    selected = [root / "summary.json", root / "integrity_audit.json", root / "training_contract.json",
        root / "events_fit.jsonl", root / "events_development.jsonl", root / "node_labels.jsonl",
        tensor_path, root / "features/feature_contract.json", root / "features/image_features.npz",
        root / "features/event_tensors.npz", root / "features/target_label_keys.json",
        root / "features/input_manifest.json", root / "training_smoke.json"]
    source_snapshots = {}
    for path in [Path(__file__).resolve(), REPO / "experiments/world_model/branch_feature_pack.py",
                 REPO / "experiments/world_model/event_latent.py"]:
        snapshot = root / "source_snapshot/finalization" / path.relative_to(REPO)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() and sha256_file(snapshot) != sha256_file(path):
            raise ValueError("finalization source changed; create a new release")
        shutil.copyfile(path, snapshot)
        selected.append(snapshot)
        source_snapshots[str(path)] = str(snapshot)
    write_json(root / "data_release.json", {"version": "event_branch_release/v4.1", "data_root": str(root),
        "training_tensors": str(tensor_path), "fit_mask": "fit", "gradient_updates_on_development": False,
        "target_labels_are_offline_supervision_not_model_inputs": True,
        "legacy_launcher_configuration_is_automatically_compatible": False,
        "legacy_auxiliary_sources_are_merged_into_primary": False, "sealed_data_opened": False,
        "world_model_trained": False, "source_snapshots": source_snapshots,
        "files": {str(p): sha256_file(p) for p in selected}})
    print(json.dumps({"fit_events": int(arrays["fit"].sum()), "development_events": int((~arrays["fit"]).sum()),
        "all_eight_history_frames_retained": True, "training_loss_and_gradients_finite": True, "optimizer_steps": 0}), flush=True)


if __name__ == "__main__":
    main()
