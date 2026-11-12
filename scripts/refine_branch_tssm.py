"""Fit paired exit discrimination and evaluate retrieval before navigation."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.branch_training_data import paired_indices, prediction_metrics  # noqa: E402
from experiments.world_model.event_residual import direction_read  # noqa: E402
from experiments.world_model.event_tssm import (  # noqa: E402
    EventTSSM,
    branch_discrimination_loss,
    sequence_objective,
)
from experiments.world_model.jsonutil import sha256_file  # noqa: E402
from experiments.world_model.precollection_protocol import assert_training_authorized  # noqa: E402
from experiments.world_model.tssm_runtime import load_tssm_predictor  # noqa: E402
from scripts.train_event_tssm import load_data  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def discrimination_metrics(prediction, target, rows, labels):
    result = prediction_metrics(prediction, target, rows, labels)
    pairs = paired_indices(rows, labels)
    if len(pairs):
        p, t = prediction[pairs], target[pairs]
        scores = np.einsum("nid,njd->nij", p, t)
        margins = np.stack([scores[:, 0, 0] - scores[:, 0, 1], scores[:, 1, 1] - scores[:, 1, 0]], axis=1)
        result.update(exit_retrieval_accuracy=float(np.where(margins > 1e-6, 1.,
            np.where(margins < -1e-6, 0., .5)).mean()),
            both_exits_correct=float((margins > 1e-6).all(1).mean()))
    return result


def run(args):
    assert_training_authorized()
    if not torch.cuda.is_available():
        raise RuntimeError("branch refinement requires CUDA")
    if args.steps < 1 or args.pair_weight < 0 or args.sequence_weight < 0 or args.learning_rate <= 0:
        raise ValueError("invalid training budget")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows, arrays = load_data(args.data)
    predictor = load_tssm_predictor(args.initial_checkpoint, "cuda:0")
    columns = None
    if args.image_features is not None:
        with np.load(args.image_features, allow_pickle=False) as f:
            lookup = {key: i for i, key in enumerate(f["image_sha256"].tolist())}
            source_keys = [r["history_rgb"][-1]["sha256"] for r in rows]
            columns = torch.from_numpy(f["column_tokens_raw"][[lookup[k] for k in source_keys]]).cuda()
        visual_model = EventTSSM(replace(predictor.model.config, use_exit_visual=True)).cuda()
        incompatible = visual_model.load_state_dict(predictor.model.state_dict(), strict=False)
        if incompatible.unexpected_keys or any(not k.startswith("exit_encoder.") for k in incompatible.missing_keys):
            raise ValueError("unexpected visual world-model initialization")
        predictor.model = visual_model
    model = predictor.model.requires_grad_(True)
    fit = np.flatnonzero(arrays["fit"] & arrays["primary"])
    train_fit = np.flatnonzero(arrays["fit"])
    dev = np.flatnonzero(~arrays["fit"] & arrays["primary"])
    pairs = paired_indices([rows[i] for i in fit], arrays["labels"][fit])
    if not len(pairs) or not len(paired_indices([rows[i] for i in dev], arrays["labels"][dev])):
        raise ValueError("fit and development require paired distinct outcomes")
    for a, b in fit[pairs]:
        if not np.array_equal(arrays["history"][a], arrays["history"][b]):
            raise ValueError("paired exits must have identical pre-decision histories")
    # The dataset is already normalized using fit-only frozen statistics.
    np.testing.assert_array_equal(arrays["feature_mean"], predictor.feature_mean.cpu().numpy())
    np.testing.assert_array_equal(arrays["feature_scale"], predictor.feature_scale.cpu().numpy())
    h = torch.from_numpy(arrays["history"]).cuda()
    action = torch.from_numpy(arrays["direct_action"]).cuda()
    visual = direction_read(columns, action[:, :2], use_branch=model.config.use_branch) if columns is not None else None

    def predict(ids):
        return model.transition(model.initialize(h[ids]), action[ids],
            exit_visual=visual[ids] if visual is not None else None)[1]["feature"]
    target_np = arrays["observations"][np.arange(len(rows)), arrays["valid"].sum(1) - 1]
    target = torch.from_numpy(target_np).cuda()
    sequence_arrays = {key:torch.from_numpy(arrays[key]).cuda() for key in ["observations", "valid", "actions"]}
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    sources = [Path(__file__), REPO / "experiments/world_model/event_tssm.py"]
    for source in sources:
        shutil.copyfile(source, args.output / source.name)
    write(args.output / "manifest.json", {str(p): sha256_file(p) for p in
        [*sources, args.initial_checkpoint, args.data / "tensors.npz", args.data / "sequences.jsonl",
         *([args.image_features] if args.image_features is not None else [])]})
    # A direction-only regression is a genuine held-out baseline: no dev fit.
    design = np.c_[arrays["direct_action"][:, :2], np.ones(len(rows))]
    coefficients = np.linalg.solve(design[fit].T @ design[fit] + .1 * np.eye(3), design[fit].T @ target_np[fit])
    direction = design @ coefficients
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)

    def metrics(values, ids):
        return discrimination_metrics(values, target_np[ids], [rows[i] for i in ids], arrays["labels"][ids])

    baselines = {role: {"copy_current": metrics(arrays["history"][ids, -1], ids),
                        "direction_only": metrics(direction[ids], ids)} for role, ids in [("fit", fit), ("development", dev)]}
    write(args.output / "baselines.json", baselines)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    rng, best, best_step = np.random.default_rng(args.seed), -np.inf, 0
    log = (args.output / "history.jsonl").open("x", buffering=1)
    for step in range(args.steps + 1):
        # Shared state must be deterministic for paired interventions. eval()
        # disables dropout; it does not disable fit-only gradients.
        model.eval()
        if step:
            ids = rng.choice(train_fit, size=min(32, len(train_fit)), replace=False)
            chosen = fit[pairs[rng.integers(len(pairs), size=16)]].reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            p, pp = predict(ids), predict(chosen)
            pair_loss = branch_discrimination_loss(pp, target[chosen], torch.arange(len(chosen), device=h.device).reshape(-1, 2))
            loss = (1 - (p * target[ids]).sum(-1)).mean() + args.pair_weight * pair_loss
            if args.sequence_weight:
                sequence_ids = rng.choice(train_fit, size=4, replace=False)
                length = int(arrays["valid"][sequence_ids].sum(1).max())
                sequential, overshoot = sequence_objective(model, h[sequence_ids],
                    sequence_arrays["observations"][sequence_ids, :length], sequence_arrays["valid"][sequence_ids, :length],
                    sequence_arrays["actions"][sequence_ids, :length-1])
                loss = loss + args.sequence_weight * (sequential + .2 * overshoot)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite fit objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if step % 25 == 0 or step == args.steps:
            with torch.inference_mode():
                predictions = {role: predict(ids).cpu().numpy()
                               for role, ids in [("fit", fit), ("development", dev)]}
            results = {role: metrics(predictions[role], ids) for role, ids in [("fit", fit), ("development", dev)]}
            d = results["development"]
            # Rank primarily by exit retrieval, not a source-copy cosine.
            score = d["exit_retrieval_accuracy"] + .25 * d["both_exits_correct"] + .01 * d["scene_macro_cosine"]
            record = {"step": step, "metrics": results, "loss": float(loss.detach()) if step else None}
            log.write(json.dumps(record) + "\n")
            print(json.dumps({"step": step, "dev": d["exit_retrieval_accuracy"], "dev_both": d["both_exits_correct"],
                "dev_pair": d["pair_assignment_accuracy"], "dev_cosine": d["scene_macro_cosine"],
                "angle": d["predicted_pair_angle_degrees"]}), flush=True)
            if score > best:
                best, best_step = score, step
                bundle = {"schema": "event-tssm/0.1", "model_config": model.configuration(), "model_state_dict": model.state_dict(),
                          "feature_mean": predictor.feature_mean.cpu(), "feature_scale": predictor.feature_scale.cpu(),
                          "metrics": results, "step": step}
                torch.save(bundle, args.output / "candidate_tssm.pt")
                for role, values in predictions.items():
                    np.save(args.output / f"best_{role}_predictions.npy", values)
                write(args.output / "result.json", {"best_step": best_step, "metrics": results, "baselines": baselines,
                    "training_fit_samples": len(train_fit), "primary_fit_pairs": len(pairs),
                    "global_gradients_fit_only": True, "development_used_for_selection": True, "navigation_measured": False})
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--image-features", type=Path, help="Frozen actual current-image column tokens, indexed by RGB hash.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--pair-weight", type=float, default=2.)
    parser.add_argument("--sequence-weight", type=float, default=0., help="Preserve actual route posterior feedback while refining paired exits.")
    parser.add_argument("--learning-rate", type=float, default=.0002)
    run(parser.parse_args())
