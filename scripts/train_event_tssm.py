"""Train and compare a stateful event TSSM, paired exits, and causal replay."""

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
    paired_indices,
    prediction_metrics,
    read_rows,
)
from experiments.world_model.jsonutil import sha256_file  # noqa: E402
from experiments.world_model.precollection_protocol import assert_training_authorized  # noqa: E402


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_data(data):
    rows = read_rows(data / "sequences.jsonl")
    with np.load(data / "tensors.npz", allow_pickle=False) as f:
        a = {k: f[k].copy() for k in f.files}
    if len(rows) != len(a["fit"]) or np.any(a["valid"][:, 1:] & ~a["valid"][:, :-1]):
        raise ValueError("sequences must be contiguous before padding")
    if not np.array_equal(a["fit"], [r["role"] == "fit" for r in rows]):
        raise ValueError("partition mask mismatch")
    if any(r["role"] not in {"fit", "development"} or r["scene"] == "Vvot9Ly1tCj" for r in rows):
        raise ValueError("sealed sequence refused")
    if {r["scene"] for r in rows if r["role"] == "fit"} & {r["scene"] for r in rows if r["role"] == "development"}:
        raise ValueError("house split contamination")
    if not all(np.isfinite(a[k]).all() for k in ["observations", "history", "actions", "memory"]):
        raise ValueError("non-finite sequence features")
    return rows, a


def prepare(data, output):
    assert_training_authorized()
    rows, a = load_data(data)
    output.mkdir(parents=True, exist_ok=False)
    arms = []
    for seed in [11, 23]:
        for name, state, branch, pair in [("tssm_pair", True, True, 1.0),
                                          ("tssm_no_pair", True, True, 0.0),
                                          ("tssm_no_state", False, True, 1.0),
                                          ("tssm_no_branch", True, False, 0.0)]:
            arms.append({"name": name, "seed": seed, "use_state": state, "use_branch": branch, "pair_weight": pair})
    config = {"data": str(data), "arms": arms, "maximum_updates": 1000, "evaluate_every": 50,
              "minimum_updates": 300, "patience": 8, "learning_rate": 0.0002,
              "selection": "dev scene-macro cosine + 0.25 * paired assignment margin",
              "sequence_weight": 0.25, "overshoot_weight": 0.2, "memory_exposure": "past fit episodes only",
              "cold_evaluation_memory_empty": True, "LightNav_loaded": False, "sealed_opened": False}
    save(output / "config.json", config)
    bindings = {str(data / n): sha256_file(data / n) for n in ["protocol.json", "sequences.jsonl", "tensors.npz", "feature_contract.json", "memory_provenance.jsonl"]}
    save(output / "input_manifest.json", bindings)
    sources = [Path(__file__).resolve(), REPO / "experiments/world_model/event_tssm.py"]
    source_bindings = {}
    for p in sources:
        target = output / "source_snapshot" / p.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        source_bindings[str(p)] = {"sha256": sha256_file(p), "snapshot": str(target)}
    save(output / "source_manifest.json", source_bindings)
    save(output / "authorization.json", {"source": "2026-11-06 user: 那按照你说的继续优化",
        "scope": "stateful event dynamics, paired training, persistent scene memory and post-observation adaptation",
        "global_fit_only": True, "sealed_opened": False})
    primary = np.flatnonzero(a["primary"])
    obs_end = a["observations"][np.arange(len(rows)), a["valid"].sum(1) - 1]
    baseline_arrays = {"fit": a["fit"][primary], "weights": a["weights"][primary], "history": a["history"][primary],
                      "target": obs_end[primary], "branch": a["direct_action"][primary, :2], "target_labels": a["labels"][primary]}
    save(output / "baselines.json", fixed_baselines([rows[i] for i in primary], baseline_arrays))
    (output / "logs").mkdir()
    print(json.dumps({"arms": len(arms), "fit_independent": int((a["fit"] & a["primary"]).sum()),
                      "development_independent": int((~a["fit"] & a["primary"]).sum())}), flush=True)


def train(output, worker):
    import torch

    from experiments.world_model.event_latent import multi_positive_contrastive
    from experiments.world_model.event_tssm import (
        EventTSSM,
        TSSMConfig,
        gaussian_kl,
        paired_future_loss,
        sequence_objective,
    )

    assert_training_authorized()
    if not torch.cuda.is_available():
        raise RuntimeError("TSSM training requires CUDA")
    cfg = json.loads((output / "config.json").read_text())
    for name, record in json.loads((output / "source_manifest.json").read_text()).items():
        if sha256_file(Path(name)) != record["sha256"]:
            raise ValueError("implementation changed after registration")
    rows, a = load_data(Path(cfg["data"]))
    arm = cfg["arms"][worker]
    destination = output / f"{arm['name']}_seed{arm['seed']}"
    destination.mkdir(exist_ok=False)
    random.seed(arm["seed"])
    np.random.seed(arm["seed"])
    torch.manual_seed(arm["seed"])
    torch.cuda.manual_seed_all(arm["seed"])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = EventTSSM(TSSMConfig(use_state=arm["use_state"], use_branch=arm["use_branch"])).to("cuda:0")
    save(destination / "model_config.json", {**model.configuration(), "parameters": sum(p.numel() for p in model.parameters())})
    device_arrays = {k: torch.from_numpy(a[k]).to("cuda:0") for k in ["history", "observations", "valid", "actions", "memory", "memory_valid", "direct_action", "labels"]}
    end = a["valid"].sum(1) - 1
    targets = torch.from_numpy(a["observations"][np.arange(len(rows)), end]).to("cuda:0")
    fit = np.flatnonzero(a["fit"])
    long_fit = np.flatnonzero(a["fit"] & (a["valid"].sum(1) >= 3))
    cold_fit = np.flatnonzero(a["fit"] & a["primary"])
    dev = np.flatnonzero(~a["fit"] & a["primary"])
    local_pairs = paired_indices([rows[i] for i in cold_fit], a["labels"][cold_fit])
    pairs = cold_fit[local_pairs]
    weights = a["weights"][fit].astype(np.float64)
    weights /= weights.sum()
    rng = np.random.default_rng(arm["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.01)

    def evaluate():
        model.eval()
        metrics, predictions = {}, {}
        with torch.inference_mode():
            for role, ids in [("fit", cold_fit), ("development", dev)]:
                state = model.initialize(device_arrays["history"][ids])
                _, p = model.transition(state, device_arrays["direct_action"][ids])
                values = p["feature"].float().cpu().numpy()
                predictions[role] = values
                metrics[role] = prediction_metrics(values, targets[ids].cpu().numpy(), [rows[i] for i in ids], a["labels"][ids])
        return metrics, predictions

    def publish(step, metrics, predictions):
        torch.save({"schema": "event-tssm/0.1", "model_config": model.configuration(), "model_state_dict": model.state_dict(),
                    "feature_mean": torch.from_numpy(a["feature_mean"]), "feature_scale": torch.from_numpy(a["feature_scale"]),
                    "step": step, "metrics": metrics, "arm": arm}, destination / "best.pt")
        for role, values in predictions.items():
            np.save(destination / f"best_{role}_predictions.npy", values)

    best, stale, started = -np.inf, 0, time.monotonic()
    metrics, predictions = evaluate()
    best = metrics["development"]["scene_macro_cosine"] + 0.25 * metrics["development"]["pair_margin_mean"]
    publish(0, metrics, predictions)
    with (destination / "history.jsonl").open("x", buffering=1) as log:
        log.write(json.dumps({"step": 0, "metrics": metrics}) + "\n")
        for step in range(1, cfg["maximum_updates"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]["lr"] = cfg["learning_rate"] * min(1, step / 50)
            j = rng.choice(fit, size=24, replace=True, p=weights)
            source = model.initialize(device_arrays["history"][j])
            prior_state, p = model.transition(source, device_arrays["direct_action"][j],
                memory=device_arrays["memory"][j], memory_valid=device_arrays["memory_valid"][j])
            _, q = model.observe(prior_state, targets[j], sample=True)
            direct = (1 - (p["feature"] * targets[j]).sum(-1)).mean()
            direct = direct + 0.5 * (1 - (q["feature"] * targets[j]).sum(-1)).mean()
            direct = direct + 0.001 * gaussian_kl(q, p).mean()
            direct = direct + 0.1 * multi_positive_contrastive(p["feature"], targets[j], device_arrays["labels"][j])
            sj = rng.choice(long_fit if len(long_fit) else fit, size=4, replace=True)
            length = int(a["valid"][sj].sum(1).max())
            sequential, overshoot = sequence_objective(model, device_arrays["history"][sj],
                device_arrays["observations"][sj, :length], device_arrays["valid"][sj, :length],
                device_arrays["actions"][sj, :length - 1])
            pair_loss = direct.new_zeros(())
            if arm["pair_weight"] and len(pairs):
                chosen = pairs[rng.integers(len(pairs), size=8)].reshape(-1)
                model.eval()  # Paired inputs share deterministic state; gradients remain enabled.
                s = model.initialize(device_arrays["history"][chosen])
                _, pp = model.transition(s, device_arrays["direct_action"][chosen])
                local = torch.arange(len(chosen), device="cuda:0").reshape(-1, 2)
                pair_loss = paired_future_loss(pp["feature"], targets[chosen], local)
                model.train()
            loss = direct + cfg["sequence_weight"] * sequential + cfg["overshoot_weight"] * overshoot + arm["pair_weight"] * pair_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite stateful training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("non-finite TSSM gradients")
            optimizer.step()
            if step == 1 or step % cfg["evaluate_every"] == 0:
                metrics, predictions = evaluate()
                row = {"step": step, "loss": float(loss.detach()), "sequence_loss": float(sequential.detach()),
                       "overshoot_loss": float(overshoot.detach()), "pair_loss": float(pair_loss.detach()),
                       "metrics": metrics, "elapsed_s": time.monotonic() - started}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                v = metrics["development"]
                print(json.dumps({"arm": arm["name"], "step": step, "dev_cosine": v["scene_macro_cosine"],
                                  "dev_pair": v["pair_assignment_accuracy"], "fit_cosine": metrics["fit"]["cosine"]}), flush=True)
                score = v["scene_macro_cosine"] + 0.25 * v["pair_margin_mean"]
                if score > best + 1e-5:
                    best, stale = score, 0
                    publish(step, metrics, predictions)
                else:
                    stale += 1
                if step >= cfg["minimum_updates"] and stale >= cfg["patience"]:
                    break
    torch.save({"schema": "event-tssm/0.1", "model_config": model.configuration(), "model_state_dict": model.state_dict(),
                "feature_mean": torch.from_numpy(a["feature_mean"]), "feature_scale": torch.from_numpy(a["feature_scale"]),
                "step": step, "metrics": metrics, "arm": arm}, destination / "last.pt")
    best_checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=True)
    save(destination / "result.json", {"arm": arm, "best_step": best_checkpoint["step"], "best": best_checkpoint["metrics"],
        "last": metrics, "updates_completed": step, "elapsed_s": time.monotonic() - started,
        "offline_global_gradients_fit_only": True, "development_is_not_blind": True, "sealed_opened": False})


def summarize(output):
    config = json.loads((output / "config.json").read_text())
    results = [json.loads((output / f"{a['name']}_seed{a['seed']}" / "result.json").read_text()) for a in config["arms"]]
    save(output / "results.json", {"experiments": results, "development_used_for_selection": True,
                                  "navigation_success_measured": False, "sealed_opened": False})
    shutil.copyfile(output / "tssm_pair_seed11/best.pt", output / "candidate_tssm.pt")
    print(json.dumps([{**r["arm"], "step": r["best_step"], "dev_cosine": r["best"]["development"]["scene_macro_cosine"],
                       "dev_pair": r["best"]["development"]["pair_assignment_accuracy"]} for r in results]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "train", "summarize"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.data is None:
            parser.error("prepare requires --data")
        prepare(args.data.resolve(), args.output.resolve())
    elif args.phase == "train":
        train(args.output.resolve(), args.worker)
    else:
        summarize(args.output.resolve())
