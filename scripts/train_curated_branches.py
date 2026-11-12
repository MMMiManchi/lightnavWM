"""Train a bounded eight-arm pilot on the immutable curated branch release."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.branch_training_data import (  # noqa: E402
    fixed_baselines,
    load_release,
    prediction_metrics,
)
from experiments.world_model.jsonutil import sha256_file  # noqa: E402
from experiments.world_model.precollection_protocol import assert_training_authorized  # noqa: E402


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(data, output, initial_checkpoint=None, initial_normalization=None):
    assert_training_authorized()
    rows, arrays, inputs = load_release(data)
    output.mkdir(parents=True, exist_ok=False)
    arms = []
    for seed in [11, 23, 47]:
        for conditional in [True, False]:
            arms.append({"name": "small_branch" if conditional else "small_no_branch", "seed": seed,
                "model": {"width": 128, "layers": 2, "heads": 4, "stochastic_dim": 32,
                          "use_branch": conditional, "stochastic": True}})
    for conditional in [True, False]:
        arms.append({"name": "reference_branch" if conditional else "reference_no_branch", "seed": 11,
            "model": {"width": 256, "layers": 4, "heads": 8, "stochastic_dim": 64,
                      "use_branch": conditional, "stochastic": True}})
    if initial_checkpoint is not None:
        import torch

        if initial_normalization is None:
            raise ValueError("pretrained initialization requires its bound normalization arrays")
        with np.load(initial_normalization, allow_pickle=False) as previous:
            if any(not np.array_equal(previous[k], arrays[k]) for k in ["feature_mean", "feature_scale"]):
                raise ValueError("pretrained model and new data use different feature coordinates")
        checkpoint = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        base_config = checkpoint["model_config"]
        arms = []
        for seed in [11, 23]:
            for conditional in [True, False]:
                arms.append({"name": "finetune_branch" if conditional else "finetune_no_branch", "seed": seed,
                    "model": {**base_config, "use_branch": conditional},
                    "initial_checkpoint": str(initial_checkpoint),
                    "initial_checkpoint_sha256": sha256_file(initial_checkpoint)})
        inputs[str(initial_checkpoint)] = sha256_file(initial_checkpoint)
        inputs[str(initial_normalization)] = sha256_file(initial_normalization)
    config = {"version": "curated_branch_training_pilot/v1", "data": str(data), "arms": arms,
        "maximum_updates": 2000, "batch_size": 32, "learning_rate": 0.0002,
        "weight_decay": 0.01, "evaluate_every": 50, "patience_evaluations": 10,
        "minimum_updates_before_stopping": 500, "selection": "development scene-macro cosine",
        "fit_events": int(arrays["fit"].sum()), "development_events": int((~arrays["fit"]).sum()),
        "future_targets_available_at_inference": False, "sealed_data_opened": False,
        "LightNav_loaded": False, "SigLIP_parameters_updated": False, "navigation_evaluation": False}
    if initial_checkpoint is not None:
        config.update(version="curated_branch_training_pilot/finetune-v1", maximum_updates=1000,
                      learning_rate=0.00002, patience_evaluations=8)
    save(output / "config.json", config)
    save(output / "authorization.json", {
        "source": "Direct user instruction on 2026-11-06: 现在这个数据量能否先训练一个版本看看有没有效果，然后在扩采数据",
        "scope": "world-model predictor training on the released fit split; development evaluation; GPUs 0-7",
        "source_data_modified": False, "sealed_test_opened": False})
    sources = [Path(__file__).resolve(), REPO / "experiments/world_model/branch_training_data.py",
               REPO / "experiments/world_model/event_latent.py", REPO / "experiments/world_model/event_inference.py"]
    source_manifest = {}
    for path in sources:
        dest = output / "source_snapshot" / path.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        source_manifest[str(path)] = {"sha256": sha256_file(path), "snapshot": str(dest)}
    save(output / "source_manifest.json", source_manifest)
    save(output / "input_manifest.json", inputs)
    save(output / "baselines.json", fixed_baselines(rows, arrays))
    shutil.copyfile(data / "features/feature_contract.json", output / "feature_contract.json")
    (output / "logs").mkdir()
    print(json.dumps({"fit": config["fit_events"], "development": config["development_events"], "arms": len(arms)}), flush=True)


def train(output, worker):
    import torch

    from experiments.world_model.event_latent import (
        EventLatentWorldModel,
        EventModelConfig,
        event_objective,
    )

    assert_training_authorized()
    if not torch.cuda.is_available():
        raise RuntimeError("world-model pilot requires CUDA; no silent fallback")
    config = json.loads((output / "config.json").read_text())
    if not 0 <= worker < len(config["arms"]):
        raise ValueError("invalid training arm")
    for name, record in json.loads((output / "source_manifest.json").read_text()).items():
        if sha256_file(Path(name)) != record["sha256"]:
            raise ValueError("training implementation changed after preparation")
    rows, arrays, _inputs = load_release(Path(config["data"]))
    arm = config["arms"][worker]
    destination = output / f"{arm['name']}_seed{arm['seed']}"
    destination.mkdir(exist_ok=False)
    random.seed(arm["seed"])
    np.random.seed(arm["seed"])
    torch.manual_seed(arm["seed"])
    torch.cuda.manual_seed_all(arm["seed"])
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    model = EventLatentWorldModel(EventModelConfig(**arm["model"])).to("cuda:0")
    if arm.get("initial_checkpoint"):
        path = Path(arm["initial_checkpoint"])
        if sha256_file(path) != arm["initial_checkpoint_sha256"]:
            raise ValueError("initial world-model checkpoint changed")
        previous = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(previous["model_state_dict"], strict=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    save(destination / "model_config.json", {**model.configuration(), "parameters": parameter_count})
    tensors = {k: torch.from_numpy(arrays[k]).to("cuda:0")
               for k in ["history", "valid", "target", "branch", "target_labels"]}
    indices = {"fit": np.flatnonzero(arrays["fit"]), "development": np.flatnonzero(~arrays["fit"])}
    ids_gpu = {k: torch.as_tensor(v, device="cuda:0") for k, v in indices.items()}
    metadata = {k: [rows[i] for i in v] for k, v in indices.items()}
    weights = arrays["weights"][indices["fit"]].astype(np.float64)
    weights /= weights.sum()
    rng = np.random.default_rng(arm["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    def evaluate():
        model.eval()
        predictions, metrics = {}, {}
        with torch.inference_mode():
            for role, ids in indices.items():
                j = ids_gpu[role]
                p = model.predict(tensors["history"][j], tensors["valid"][j], tensors["branch"][j]).float().cpu().numpy()
                predictions[role] = p
                metrics[role] = prediction_metrics(p, arrays["target"][ids], metadata[role], arrays["target_labels"][ids])
        return predictions, metrics

    best, stale, started = -np.inf, 0, time.monotonic()
    last_metrics = None

    def checkpoint_and_predictions(step, metrics, predictions):
        torch.save({"model_config": model.configuration(), "model_state_dict": model.state_dict(),
            "feature_mean": torch.from_numpy(arrays["feature_mean"]),
            "feature_scale": torch.from_numpy(arrays["feature_scale"]),
            "step": step, "evaluation": metrics, "arm": arm}, destination / "best.pt")
        for role, p in predictions.items():
            np.save(destination / f"best_{role}_predictions.npy", p)

    with (destination / "history.jsonl").open("x", buffering=1) as log:
        if arm.get("initial_checkpoint"):
            predictions, last_metrics = evaluate()
            best = last_metrics["development"]["scene_macro_cosine"]
            checkpoint_and_predictions(0, last_metrics, predictions)
            save(destination / "before_finetuning.json", last_metrics)
            log.write(json.dumps({"step": 0, "loss": None, "metrics": last_metrics, "elapsed_s": 0.0}) + "\n")
        for step in range(1, config["maximum_updates"] + 1):
            model.train()
            chosen = rng.choice(len(indices["fit"]), size=config["batch_size"], replace=True, p=weights)
            j = ids_gpu["fit"][torch.as_tensor(chosen, device="cuda:0")]
            optimizer.param_groups[0]["lr"] = config["learning_rate"] * min(1.0, step / 100)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.training_outputs(tensors["history"][j], tensors["valid"][j], tensors["branch"][j], tensors["target"][j])
                loss, _parts = event_objective(outputs, tensors["target"][j], tensors["target_labels"][j])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % config["evaluate_every"] == 0:
                predictions, last_metrics = evaluate()
                row = {"step": step, "loss": float(loss.detach()), "metrics": last_metrics,
                       "elapsed_s": time.monotonic() - started}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps({"worker": worker, "arm": arm["name"], "step": step,
                    "fit_cosine": last_metrics["fit"]["cosine"],
                    "dev_cosine": last_metrics["development"]["scene_macro_cosine"],
                    "dev_pair_accuracy": last_metrics["development"]["pair_assignment_accuracy"]}), flush=True)
                score = last_metrics["development"]["scene_macro_cosine"]
                if score > best + 1e-5:
                    best, stale = score, 0
                    checkpoint_and_predictions(step, last_metrics, predictions)
                else:
                    stale += 1
                if step >= config["minimum_updates_before_stopping"] and stale >= config["patience_evaluations"]:
                    break
    checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=True)
    save(destination / "result.json", {"arm": arm, "parameters": parameter_count,
        "best_step": checkpoint["step"], "best": checkpoint["evaluation"], "last": last_metrics,
        "updates_completed": step, "elapsed_s": time.monotonic() - started,
        "checkpoint": str(destination / "best.pt"), "fit_only_gradient_updates": True,
        "development_used_for_checkpoint_selection": True, "blind_test": False, "navigation_success_measured": False})


def summarize(output):
    config = json.loads((output / "config.json").read_text())
    results = [json.loads((output / f"{arm['name']}_seed{arm['seed']}" / "result.json").read_text())
               for arm in config["arms"]]
    groups = {}
    for name in dict.fromkeys(r["arm"]["name"] for r in results):
        group = [r for r in results if r["arm"]["name"] == name]
        metrics = {}
        for key in ["cosine", "scene_macro_cosine", "pair_assignment_accuracy", "pair_margin_mean", "predicted_pair_angle_degrees"]:
            values = [r["best"]["development"][key] for r in group]
            metrics[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
        groups[name] = {"seeds": [r["arm"]["seed"] for r in group], "metrics": metrics}
    save(output / "results.json", {"experiments": results, "groups": groups,
        "evaluation": "development diagnostic, including checkpoint selection; not a blind test or navigation metric",
        "paired_exit_test": "15 distinct-target origin pairs; identical histories; ties count 0.5",
        "blind_sealed_data_opened": False, "online_memory_or_adaptation_tested": False})
    representative = "finetune_branch" if any(a["name"] == "finetune_branch" for a in config["arms"]) else "small_branch"
    source = output / f"{representative}_seed11/best.pt"
    shutil.copyfile(source, output / "candidate_predictor.pt")
    save(output / "candidate.json", {"checkpoint": str(output / "candidate_predictor.pt"),
        "source_arm": f"{representative}_seed11", "selection": "fixed representative seed, not best seed selection",
        "automatically_enable_for_route_selection": False, "LightNav_connected": False,
        "feature_contract": str(output / "feature_contract.json"), "world_model_trained": True})
    print(json.dumps(groups), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "train", "summarize"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--initial-normalization", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase == "prepare":
        if args.data is None:
            parser.error("--data is required to prepare")
        prepare(args.data.resolve(), output,
                args.initial_checkpoint.resolve() if args.initial_checkpoint else None,
                args.initial_normalization.resolve() if args.initial_normalization else None)
    elif args.phase == "train":
        train(output, args.worker)
    else:
        summarize(output)


if __name__ == "__main__":
    main()
