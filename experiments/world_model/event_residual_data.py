"""Build strictly earlier-episode memory without using place ids as inputs."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def causal_memory_indices(pairs, history, branch, episode_order, *, use_branch=True, top_k=3, query_branch=None):
    """Episode order is acquired from source episode files, never target labels.

    Rows become visible only after the recorded episode ends. The current
    episode cannot read its own targets, including when sample rows interleave.
    The returned integer ids are provenance; neural inputs are real triplets.
    """
    query_branch = branch if query_branch is None else query_branch
    by_episode = defaultdict(list)
    for i, pair in enumerate(pairs):
        by_episode[(pair["source_run"], pair["episode_uid"])].append(i)
    if set(by_episode) - set(episode_order):
        raise ValueError("episode order does not cover all input pairs")
    selected = np.full((len(pairs), top_k), -1, dtype=np.int64)
    past = defaultdict(list)
    for episode in episode_order:
        members = by_episode.get(episode, [])
        if not members:
            continue
        scenes = {pairs[i]["scene"] for i in members}
        if len(scenes) != 1:
            raise ValueError("one recorded episode cannot span scene namespaces")
        scene = next(iter(scenes))
        available = np.array(past[scene], dtype=np.int64)
        for i in members:
            if not len(available):
                continue
            source_similarity = history[available, -1] @ history[i, -1]
            direction_similarity = branch[available] @ query_branch[i]
            keep = source_similarity >= 0.5
            if use_branch:
                keep &= direction_similarity >= math.cos(math.radians(35))
            positions = np.flatnonzero(keep)
            scores = 0.8 * source_similarity[positions]
            if use_branch:
                scores += 0.2 * direction_similarity[positions]
            order = positions[np.argsort(-scores, kind="stable")[:top_k]]
            selected[i, :len(order)] = available[order]
        past[scene].extend(sorted(members, key=lambda i: pairs[i]["origin_index"]))
    return selected


def materialize_memory(indices, history, target, branch):
    valid = indices >= 0
    safe = np.maximum(indices, 0)
    return {"memory_source": history[safe, -1] * valid[..., None],
            "memory_future": target[safe] * valid[..., None],
            "memory_branch": branch[safe] * valid[..., None], "memory_valid": valid}


def direct_memory_prediction(current, branch, context, *, use_branch=True):
    """No world model: same visible real history, current-visual/direction query."""
    source = context["memory_source"]
    valid = context["memory_valid"]
    cosine = np.einsum("bd,bkd->bk", current, source)
    direction = np.einsum("bd,bkd->bk", branch, context["memory_branch"])
    scores = 0.8 * cosine + (0.2 * direction if use_branch else 0)
    scores = np.where(valid, scores, -1e4)
    weights = np.exp((scores - scores.max(-1, keepdims=True)) / 0.07) * valid
    weights /= np.maximum(weights.sum(-1, keepdims=True), 1e-8)
    observed = np.einsum("bk,bkd->bd", weights, context["memory_future"])
    confidence = np.clip((np.max(np.where(valid, cosine, 0), -1) - 0.5) / 0.5, 0, 1)
    if use_branch:
        floor = math.cos(math.radians(35))
        best = np.argmax(scores, -1)
        confidence *= np.clip((direction[np.arange(len(best)), best] - floor) / (1 - floor), 0, 1)
    weight = 0.75 * confidence * valid.any(-1)
    prediction = (1 - weight[:, None]) * current + weight[:, None] * observed
    return prediction / np.maximum(np.linalg.norm(prediction, axis=-1, keepdims=True), 1e-8)
