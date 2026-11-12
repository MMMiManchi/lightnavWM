"""Admission and direction-sensitive evaluation for the curated branch release."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.world_model.branch_feature_pack import place_labels
from experiments.world_model.jsonutil import sha256_file


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_release(root: Path):
    root = root.resolve()
    release_path = root / "data_release.json"
    release = json.loads(release_path.read_text())
    if release["version"] != "event_branch_release/v4.1" or release["sealed_data_opened"]:
        raise ValueError("this pilot requires the admitted nonsealed v4.1 release")
    for name, expected in release["files"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"released input changed: {name}")
    rows = [e for role in ("fit", "development") for e in read_rows(root / f"events_{role}.jsonl")]
    with np.load(release["training_tensors"], allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    if arrays["sample_ids"].tolist() != [e["sample_id"] for e in rows]:
        raise ValueError("training tensors and sample identities disagree")
    expected_fit = np.array([e["role"] == "fit" for e in rows])
    if not np.array_equal(arrays["fit"], expected_fit) or not expected_fit.any() or expected_fit.all():
        raise ValueError("fit/development mask is invalid")
    if arrays["history"].shape != (len(rows), 8, 1152) or not arrays["valid"].all():
        raise ValueError("eight real history frames are required")
    if any(e["role"] not in {"fit", "development"} or e["scene"] == "Vvot9Ly1tCj" for e in rows):
        raise ValueError("sealed or unknown sample refused")
    if {e["scene"] for e in rows if e["role"] == "fit"} & {e["scene"] for e in rows if e["role"] == "development"}:
        raise ValueError("a house crosses fit/development")
    expected_labels, _ = place_labels(rows)
    if not np.array_equal(expected_labels, arrays["target_labels"]):
        raise ValueError("target alias grouping changed")
    for key in ["history", "target", "branch", "weights", "feature_mean", "feature_scale"]:
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"non-finite {key}")
    if np.any(arrays["weights"] <= 0) or np.any(arrays["feature_scale"] <= 0):
        raise ValueError("invalid sampling weights or normalization")
    for key in ["target", "branch"]:
        if not np.allclose(np.linalg.norm(arrays[key], axis=-1), 1.0, atol=1e-5):
            raise ValueError(f"{key} must be unit length")
    groups = defaultdict(list)
    for i, e in enumerate(rows):
        groups[e["origin_id"]].append(i)
    for ids in groups.values():
        if any(not np.array_equal(arrays["history"][ids[0]], arrays["history"][i]) for i in ids[1:]):
            raise ValueError("paired interventions have different input histories")
    return rows, arrays, {str(release_path): sha256_file(release_path), **release["files"]}


def paired_indices(rows, labels):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["origin_id"]].append(i)
    pairs = []
    for ids in groups.values():
        if len(ids) == 2 and rows[ids[0]]["exit_id"] != rows[ids[1]]["exit_id"] and labels[ids[0]] != labels[ids[1]]:
            pairs.append(ids)
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def prediction_metrics(prediction, target, rows, labels):
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid predicted features")
    cosine = np.sum(prediction * target, axis=-1)
    scenes = np.array([e["scene"] for e in rows])
    pairs = paired_indices(rows, labels)
    result = {"cosine": float(cosine.mean()),
              "scene_macro_cosine": float(np.mean([cosine[scenes == s].mean() for s in sorted(set(scenes))])),
              "samples": len(rows), "houses": len(set(scenes)), "distinct_target_pairs": len(pairs)}
    if len(pairs):
        a, b = pairs.T
        pd, td = prediction[a] - prediction[b], target[a] - target[b]
        # Identical history means this change can only come from the chosen exit.
        margin = np.sum(pd * td, axis=-1) / 2
        hit = np.where(margin > 1e-6, 1.0, np.where(margin < -1e-6, 0.0, 0.5))
        result.update(pair_assignment_accuracy=float(hit.mean()),
            pair_wins=int((margin > 1e-6).sum()), pair_ties=int((np.abs(margin) <= 1e-6).sum()),
            pair_margin_mean=float(margin.mean()), pair_margins=margin.tolist(),
            predicted_pair_angle_degrees=float(np.degrees(np.arccos(np.clip(np.sum(prediction[a] * prediction[b], axis=-1), -1, 1))).mean()),
            true_pair_angle_degrees=float(np.degrees(np.arccos(np.clip(np.sum(target[a] * target[b], axis=-1), -1, 1))).mean()))
    return result


def fixed_baselines(rows, arrays):
    fit_ids = np.flatnonzero(arrays["fit"])
    weights = arrays["weights"][fit_ids].astype(np.float64)
    mean = np.average(arrays["target"][fit_ids], weights=weights, axis=0)
    mean = (mean / np.linalg.norm(mean)).astype(np.float32)
    baselines = {}
    for role in ("fit", "development"):
        ids = np.flatnonzero(arrays["fit"] if role == "fit" else ~arrays["fit"])
        local = [rows[i] for i in ids]
        similarity = arrays["history"][ids, -1] @ arrays["history"][fit_ids, -1].T
        direction = arrays["branch"][ids] @ arrays["branch"][fit_ids].T
        if role == "fit":
            for j, i in enumerate(ids):
                similarity[j, [rows[k]["origin_id"] == rows[i]["origin_id"] for k in fit_ids]] = -np.inf
        predictions = {"copy_current": arrays["history"][ids, -1],
                       "fit_mean_future": np.broadcast_to(mean, (len(ids), len(mean))),
                       "nearest_fit_successor": arrays["target"][fit_ids[similarity.argmax(axis=1)]],
                       "nearest_fit_successor_direction": arrays["target"][fit_ids[(similarity + 0.15 * direction).argmax(axis=1)]]}
        baselines[role] = {name: prediction_metrics(p, arrays["target"][ids], local, arrays["target_labels"][ids])
                           for name, p in predictions.items()}
    return baselines
