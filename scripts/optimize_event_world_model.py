"""Bounded residual/spatial/memory optimization, selected on internal fit scenes.

The original development set is evaluated only after fixed refit. No downloads,
source record edits, official runtime ids, or CPU training fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.event_anchored import AnchoredEventConfig, AnchoredEventWorldModel  # noqa: E402
from experiments.world_model.event_guards import verify_prepared_references  # noqa: E402
from experiments.world_model.event_residual import (  # noqa: E402
    ResidualEventConfig,
    ResidualEventWorldModel,
    residual_objective,
)
from experiments.world_model.event_residual_data import (  # noqa: E402
    causal_memory_indices,
    direct_memory_prediction,
    materialize_memory,
)
from experiments.world_model.event_train import digest, metrics, read_rows, save  # noqa: E402
from experiments.world_model.feature_cache import AUDITED_PREPROCESSOR, SIGLIP_WEIGHT_SHA256  # noqa: E402
from experiments.world_model.precollection_protocol import assert_training_authorized  # noqa: E402

INPUTS = ("history", "valid", "branch", "columns", "memory_source", "memory_future", "memory_branch", "memory_valid")


def arms(variant="residual"):
    if variant == "anchored_memory":
        return [{"name": f"anchored_memory_{'branch' if branch else 'no_branch'}", "seed": seed,
                 "use_branch": branch, "use_memory": True, "residual_scale": 0.02}
                for seed in (20261104, 20261105, 20261106) for branch in (True, False)] + [
                    {"name": "anchored_gate_only", "seed": seed, "use_branch": True,
                     "use_memory": True, "residual_scale": 0.0} for seed in (20261104, 20261105)]
    return [{"name": f"residual_{'memory' if memory else 'cold'}_{'branch' if branch else 'no_branch'}",
             "seed": seed, "use_branch": branch, "use_memory": memory}
            for seed in (20261104, 20261105) for memory in (False, True) for branch in (True, False)]


def prepare(args):
    assert_training_authorized()
    root, base = args.output.resolve(), args.base_run.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = {"base_run": str(base), "run_dir": str(root), "workers": 8,
              "updates": 1200, "evaluate_every": 100, "patience": 5, "batch_size": 64,
              "learning_rate": 0.0001, "experiments": arms(args.variant), "model_kind": args.variant,
              "checkpoint_selection": "internal scene macro cosine + 0.1 * internal recall@1; includes step-zero control",
              "development_used_for_selection": False,
              "encoder_snapshot": json.loads((base / "config.json").read_text())["encoder_snapshot"]}
    if args.variant == "anchored_memory":
        config.update(updates=800, evaluate_every=50, patience=4, learning_rate=0.00003)
    save(root / "config.json", config)
    source_files = [Path(__file__), REPO / "experiments/world_model/event_residual.py",
                    REPO / "experiments/world_model/event_anchored.py",
                    REPO / "experiments/world_model/event_residual_data.py", REPO / "experiments/world_model/event_train.py",
                    REPO / "experiments/world_model/event_guards.py", AUDITED_PREPROCESSOR,
                    base / "pairs.jsonl", base / "images.jsonl", base / "tensors.npz", base / "feature_contract.json",
                    *sorted(base.glob("features_*.npz"))]
    save(root / "source_reference_check.json", verify_prepared_references(json.loads((base / "config.json").read_text())))
    pairs = read_rows(base / "pairs.jsonl")
    source_runs = json.loads((base / "config.json").read_text())["source_runs"]
    episode_order = []
    for run in source_runs:
        path = REPO / "output/world_model_data" / run / "episodes.jsonl"
        source_files.append(path)
        episode_order.extend((run, row["episode_uid"]) for row in read_rows(path))
    if len(episode_order) != len(set(episode_order)):
        raise ValueError("source episode order contains duplicate reset identities")
    save(root / "episode_order.json", episode_order)
    fit_scenes = sorted({p["scene"] for p in pairs if p["partition"] == "train_fit"},
                        key=lambda x: hashlib.sha256(f"residual-v2-internal:{x}".encode()).hexdigest())
    validation = fit_scenes[:8]
    save(root / "split.json", {"internal_validation_scenes": validation, "optimization_scenes": fit_scenes[8:],
                               "refit_scenes": sorted(fit_scenes), "original_development_is_blind": False,
                               "original_development_not_used_for_v2_stopping_or_model_choice": True})
    arrays = np.load(base / "tensors.npz")
    context_counts = {}
    for use_branch in (True, False):
        indices = causal_memory_indices(pairs, arrays["history"], arrays["branch"], episode_order, use_branch=use_branch)
        np.save(root / f"memory_indices_{int(use_branch)}.npy", indices)
        context_counts[str(use_branch)] = int((indices >= 0).any(-1).sum())
    images = {r["image_id"]: r for r in read_rows(base / "images.jsonl")}
    current = sorted({p["history_image_ids"][-1] for p in pairs})
    save(root / "current_images.json", [images[i] for i in current])
    save(root / "authorization.json", {"source": "2026-11-04 user: 那你继续优化，优化好之后再接入基座看看能否跑通",
                                       "gpu_scope": "previously authorized GPUs 0-7", "raw_records_modified": False})
    save(root / "protocol.json", {
        "hypotheses": ["copy-preserving residual dynamics", "current visible direction-aligned patch columns",
                       "learned reading of prior-episode actual experience triplets"],
        "controls": ["copy current", "nearest fit successor", "same-history retrieval without any world model",
                     "direction-free residual", "branch shuffle including spatial read and memory query"],
        "memory": "same scene namespace, only episodes finished before this one, top3 current-source/direction similarity; no future-query ranking during prediction",
        "target": "unchanged frozen 1152D v1 feature contract; actual first RGB at next event",
        "history": "unchanged causal 8-observation history; no official point/pose/id neural inputs",
        "normalization": "reuse fixed v1 fit-only statistics; internal validation images were included in these unsupervised statistics",
        "optimization": config, "memory_hit_samples_in_all_partitions": context_counts,
        "integration_gate": "positive internal validation gain and no original-development regression versus matching baselines; otherwise diagnostic-only",
        "navigation_effectiveness_not_implied": True})
    hashes = {str(p): digest(p) for p in source_files}
    save(root / "input_manifest.json", hashes)
    for path in source_files:
        if path.suffix == ".py":
            destination = root / "source_snapshot" / path.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            destination.chmod(0o444)
    print(json.dumps({"state": "prepared", "samples": len(pairs), "images": len(current), "internal_scenes": validation}), flush=True)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this entry point has no CPU training fallback")


def encode(config, worker):
    from transformers import SiglipImageProcessor, SiglipVisionModel

    require_cuda()
    root = Path(config["run_dir"])
    snapshot = Path(config["encoder_snapshot"])
    if digest(snapshot / "model.safetensors") != SIGLIP_WEIGHT_SHA256:
        raise ValueError("audited vision weights changed")
    model = SiglipVisionModel.from_pretrained(snapshot, local_files_only=True, dtype=torch.float32).cuda().eval()
    processor = SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
    torch.set_float32_matmul_precision("highest")
    items = json.loads((root / "current_images.json").read_text())[worker::config["workers"]]
    columns, pooled = [], []
    for start in range(0, len(items), 12):
        batch = items[start:start + 12]
        pixels = []
        for row in batch:
            if digest(Path(row["staged_path"])) != row["staged_sha256"]:
                raise ValueError("admitted source pixels changed")
            pixels.append(np.load(row["staged_path"], allow_pickle=False))
        data = processor(images=pixels, return_tensors="pt")["pixel_values"].cuda()
        with torch.inference_mode():
            out = model(data)
        columns.append(out.last_hidden_state.reshape(len(batch), 27, 27, 1152).mean(1).float().cpu().numpy())
        pooled.append(out.pooler_output.float().cpu().numpy())
    np.savez(root / f"columns_{worker}.npz", ids=np.array([r["image_id"] for r in items]),
             columns=np.concatenate(columns), pooled=np.concatenate(pooled))
    print(json.dumps({"state": "encoded", "worker": worker, "images": len(items)}), flush=True)


def data(config, use_branch):
    root, base = Path(config["run_dir"]), Path(config["base_run"])
    pairs = read_rows(base / "pairs.jsonl")
    arrays = dict(np.load(base / "tensors.npz"))
    column_map = {}
    for file in sorted(root.glob("columns_*.npz")):
        chunk = np.load(file)
        column_map.update(zip(chunk["ids"].tolist(), chunk["columns"], strict=True))
    arrays["columns"] = np.stack([column_map[p["history_image_ids"][-1]] for p in pairs])
    indices = np.load(root / f"memory_indices_{int(use_branch)}.npy")
    arrays.update(materialize_memory(indices, arrays["history"], arrays["target"], arrays["branch"]))
    arrays["memory_indices"] = indices
    validation = json.loads((root / "split.json").read_text())["internal_validation_scenes"]
    arrays["internal_validation"] = np.array([p["scene"] in validation for p in pairs])
    for field, key in [("origin_labels", "origin_place"), ("exit_labels", "chosen_exit")]:
        labels = sorted({p["offline_labels"][key] for p in pairs})
        arrays[field] = np.array([labels.index(p["offline_labels"][key]) for p in pairs], dtype=np.int64)
    return pairs, arrays


def evaluate(model, tensors, ids):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for chunk in np.array_split(ids, max(1, int(np.ceil(len(ids) / 128)))):
            index = torch.as_tensor(chunk, device="cuda")
            outputs.append(model(**{k: tensors[k][index] for k in INPUTS}).cpu().numpy())
    return np.concatenate(outputs)


def train(config, worker):
    assert_training_authorized()
    require_cuda()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    arm = config["experiments"][worker]
    root = Path(config["run_dir"])
    destination = root / f"{arm['name']}_seed{arm['seed']}"
    destination.mkdir(exist_ok=False)
    pairs, arrays = data(config, arm["use_branch"])
    tensors = {k: torch.from_numpy(v).cuda() for k, v in arrays.items() if k in (*INPUTS, "target", "target_labels", "origin_labels", "exit_labels")}
    internal_ids = np.flatnonzero(arrays["internal_validation"])
    fit_ids = np.flatnonzero(arrays["fit"])
    optimization_ids = np.flatnonzero(arrays["fit"] & ~arrays["internal_validation"])
    dev_ids = np.flatnonzero(~arrays["fit"])
    if config.get("model_kind") == "anchored_memory":
        model_config = AnchoredEventConfig(use_branch=arm["use_branch"], use_memory=arm["use_memory"],
                                          residual_scale=arm["residual_scale"])
        model_class = AnchoredEventWorldModel
    else:
        model_config = ResidualEventConfig(use_branch=arm["use_branch"], use_memory=arm["use_memory"])
        model_class = ResidualEventWorldModel

    def initialize():
        random.seed(arm["seed"])
        np.random.seed(arm["seed"])
        torch.manual_seed(arm["seed"])
        torch.cuda.manual_seed_all(arm["seed"])
        model = model_class(model_config).cuda()
        return model, torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)

    def score(prediction, ids):
        return metrics(prediction, arrays["target"][ids], arrays["target_labels"][ids], [pairs[i]["scene"] for i in ids])

    log = (destination / "history.jsonl").open("x", buffering=1)

    def optimize(model, optimizer, ids, updates, *, validate):
        rng = np.random.default_rng(arm["seed"])
        groups = defaultdict(list)
        for i in ids:
            groups[pairs[i]["offline_labels"]["origin_place"]].append(i)
        hard_pairs = [(i, j) for group in groups.values() for i in group for j in group if i < j and
                      pairs[i]["offline_labels"]["chosen_exit"] != pairs[j]["offline_labels"]["chosen_exit"]]
        probability = arrays["weights"][ids].astype(np.float64)
        probability /= probability.sum()
        best_step, best, stale = 0, -float("inf"), 0
        for step in range(updates + 1):
            if step:
                model.train()
                chosen = rng.choice(ids, size=config["batch_size"], replace=True, p=probability)
                if hard_pairs:
                    for k in range(16):
                        chosen[2*k:2*k+2] = hard_pairs[rng.integers(len(hard_pairs))]
                index = torch.as_tensor(chosen, device="cuda")
                batch = {k: tensors[k][index] for k in INPUTS}
                if arm["use_memory"]:
                    batch["memory_valid"] = batch["memory_valid"] & (torch.rand(len(chosen), 1, device="cuda") > 0.2)
                optimizer.param_groups[0]["lr"] = config["learning_rate"] * min(1.0, step / 50)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**batch, return_aux=True)
                loss, parts = residual_objective(outputs, tensors["target"][index], tensors["target_labels"][index],
                                                 tensors["origin_labels"][index], tensors["exit_labels"][index])
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite residual training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if validate and step % config["evaluate_every"] == 0:
                result = score(evaluate(model, tensors, internal_ids), internal_ids)
                value = result["scene_macro_cosine"] + 0.1 * result["recall_at_1"]
                row = {"step": step, "phase": "internal_selection", "metrics": result, "selection_score": value}
                if step:
                    row["training_parts"] = {k: float(v.detach()) for k, v in parts.items()}
                log.write(json.dumps(row) + "\n")
                print(json.dumps({"worker": worker, **row}), flush=True)
                if value > best:
                    best, best_step, stale = value, step, 0
                    torch.save({"state_dict": model.state_dict(), "step": step, "metrics": result}, destination / "internal_best.pt")
                else:
                    stale += 1
                if stale >= config["patience"]:
                    break
        return best_step

    started = time.time()
    model, optimizer = initialize()
    best_step = optimize(model, optimizer, optimization_ids, config["updates"], validate=True)
    # Refit once on all formal-fit scenes for the internally selected duration.
    model, optimizer = initialize()
    optimize(model, optimizer, fit_ids, best_step, validate=False)
    prediction = evaluate(model, tensors, dev_ids)
    development = score(prediction, dev_ids)
    baselines = {"copy_current": score(arrays["history"][dev_ids, -1], dev_ids)}
    direct = direct_memory_prediction(arrays["history"][:, -1], arrays["branch"], arrays, use_branch=arm["use_branch"])
    baselines["direct_same_history_without_world_model"] = score(direct[dev_ids], dev_ids)
    nearest = (arrays["history"][dev_ids, -1] @ arrays["history"][fit_ids, -1].T).argmax(-1)
    baselines["nearest_fit_successor"] = score(arrays["target"][fit_ids][nearest], dev_ids)
    # Swap only query directions; remembered actual execution directions stay fixed.
    swapped_branch = arrays["branch"].copy()
    swapped_branch[dev_ids] = np.roll(swapped_branch[dev_ids], 1, axis=0)
    order = [tuple(x) for x in json.loads((root / "episode_order.json").read_text())]
    swapped_indices = causal_memory_indices(pairs, arrays["history"], arrays["branch"], order,
                                            use_branch=arm["use_branch"], query_branch=swapped_branch)
    swapped_context = materialize_memory(swapped_indices, arrays["history"], arrays["target"], arrays["branch"])
    swapped = dict(tensors)
    swapped["branch"] = torch.from_numpy(swapped_branch).cuda()
    swapped.update({k: torch.from_numpy(v).cuda() for k, v in swapped_context.items()})
    shuffled = evaluate(model, swapped, dev_ids)
    # Save a self-contained inference bundle, with no gallery or official labels.
    bundle = {"schema": "wm-residual-event/0.1", "model_config": model.configuration(), "model_kind": config.get("model_kind", "residual"),
              "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
              "feature_mean": torch.from_numpy(arrays["feature_mean"]), "feature_scale": torch.from_numpy(arrays["feature_scale"]),
              "feature_contract": json.loads((Path(config["base_run"]) / "feature_contract.json").read_text()),
              "updates": best_step}
    torch.save(bundle, destination / "model.pt")
    np.save(destination / "dev_predictions.npy", prediction)
    np.save(destination / "shuffled_dev_predictions.npy", shuffled)
    internal = torch.load(destination / "internal_best.pt", map_location="cpu", weights_only=True)
    result = {"arm": arm, "internally_selected_updates": best_step, "internal": internal["metrics"],
              "development": development, "baselines": baselines, "shuffled_development": score(shuffled, dev_ids),
              "branch_response_l2": float(np.linalg.norm(prediction - shuffled, axis=-1).mean()),
              "optimization_samples": len(optimization_ids), "internal_samples": len(internal_ids), "refit_samples": len(fit_ids),
              "development_samples": len(dev_ids), "parameters": sum(p.numel() for p in model.parameters()),
              "elapsed_s": time.time() - started, "navigation_connected": False,
              "cross_episode_queries": int(arrays["memory_valid"][dev_ids].any(-1).sum())}
    save(destination / "result.json", result)
    log.close()
    print(json.dumps({"worker": worker, "state": "complete", "result": result}), flush=True)


def summarize(config):
    root = Path(config["run_dir"])
    results = [json.loads((root / f"{a['name']}_seed{a['seed']}" / "result.json").read_text()) for a in config["experiments"]]
    save(root / "results.json", {"runs": results, "selection": "internal validation only", "development_blind": False})
    for path, expected in json.loads((root / "input_manifest.json").read_text()).items():
        if digest(Path(path)) != expected:
            raise ValueError(f"source changed during optimization: {path}")
    print(json.dumps([{k: r[k] for k in ("arm", "internally_selected_updates", "internal", "development", "branch_response_l2")}
                      for r in results], indent=2), flush=True)


def launch(config_path):
    config = json.loads(config_path.read_text())
    root = Path(config["run_dir"])
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    try:
        for phase in ("encode", "train"):
            jobs, handles = [], []
            for worker in range(config["workers"]):
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                                   TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                handle = (log_dir / f"{phase}_gpu{worker}.log").open("x", buffering=1)
                handles.append(handle)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--phase", phase,
                           "--config", str(config_path), "--worker", str(worker)]
                jobs.append(subprocess.Popen(command, cwd=REPO, env=environment, stdout=handle, stderr=subprocess.STDOUT))
            while True:
                state = [job.poll() for job in jobs]
                save(root / "status.json", {"phase": phase, "exit_codes": state, "updated_unix_s": time.time()})
                if any(code not in (None, 0) for code in state):
                    for job in jobs:
                        if job.poll() is None:
                            job.terminate()
                    for job in jobs:
                        job.wait()
                    raise RuntimeError(f"{phase} failed: {state}; see {log_dir}")
                if all(code == 0 for code in state):
                    break
                time.sleep(3)
            for handle in handles:
                handle.close()
        summarize(config)
        save(root / "status.json", {"phase": "complete", "updated_unix_s": time.time()})
    except Exception as error:
        save(root / "status.json", {"phase": "failed", "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "encode", "train", "launch", "summarize"), required=True)
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--variant", choices=("residual", "anchored_memory"), default="residual")
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.base_run is None or args.output is None:
            parser.error("prepare requires --base-run and --output")
        prepare(args)
    else:
        if args.config is None:
            parser.error("this phase requires --config")
        config = json.loads(args.config.read_text())
        if args.phase == "launch":
            launch(args.config.resolve())
        elif args.phase in ("encode", "train"):
            globals()[args.phase](config, args.worker)
        else:
            summarize(config)


if __name__ == "__main__":
    main()
