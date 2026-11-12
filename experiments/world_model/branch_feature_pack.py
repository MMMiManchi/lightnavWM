"""Preserve every causal history frame when packaging frozen visual features."""

import math

import numpy as np


def place_labels(events):
    keys = []
    for event in events:
        labels = event["offline_labels"]
        if event["target_kind"] == "next_reviewed_decision":
            identity = labels.get("landmark_origin_id")
            if not identity:
                raise ValueError("reviewed decision target has no reviewed landmark identity")
            keys.append((event["scene"], "decision", str(identity)))
        else:
            keys.append((event["scene"], "region", str(labels["target_region"]["region_id"])))
    classes = {key: i for i, key in enumerate(sorted(set(keys)))}
    return np.array([classes[key] for key in keys], dtype=np.int64), sorted(set(keys))


def pack_events(events, vectors, lookup):
    width = vectors.shape[1]
    history = np.zeros((len(events), 8, width), dtype=np.float32)
    valid = np.zeros((len(events), 8), dtype=bool)
    targets = np.empty((len(events), width), dtype=np.float32)
    bearings = []
    for i, event in enumerate(events):
        refs = event["inputs"]["history_rgb"]
        flags = event["inputs"]["history_valid"]
        if not 1 <= len(refs) <= 8 or flags != [True] * len(refs):
            raise ValueError("history must contain only actual causal observations")
        for j, ref in enumerate(refs, 8 - len(refs)):
            history[i, j] = vectors[lookup[ref["sha256"]]]
            valid[i, j] = True
        targets[i] = vectors[lookup[event["target"]["rgb"]["sha256"]]]
        b = event["inputs"]["requested_skill"]["bearing_left_rad"]
        bearings.append([math.sin(b), math.cos(b)])
    return {"sample_ids": np.array([e["sample_id"] for e in events]), "history": history, "valid": valid,
            "target": targets, "branch": np.array(bearings, dtype=np.float32).reshape(-1, 2),
            "fit": np.array([e["role"] == "fit" for e in events], dtype=bool),
            "weights": np.array([e["sample_weight"] for e in events], dtype=np.float32)}
