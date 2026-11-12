"""Compare a node view with the view at the next node.

Both vectors are frozen SigLIP pooler outputs. The comparison asks whether the
arrival is closer to the start than to another node's arrival. It does not train
a predictor and does not score an exit that was never walked.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

LONG_PATH_M = 1.0
# Fixed after the single-frame median gap of 0.020. Do not refit it to a later run.
SWEEP_MARGIN = 0.10


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"cosine inputs differ: {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("cosine inputs must be finite")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        raise ValueError("cosine inputs must be non-zero")
    return float(np.dot(a, b) / denom)


def other_span_indexes(spans: list[dict[str, Any]]) -> list[int]:
    """Pair each span with the next span's arrival in the same split."""
    order = sorted(
        range(len(spans)),
        key=lambda index: (
            spans[index]["split"],
            spans[index]["episode_uid"],
            spans[index]["start_observation_id"],
        ),
    )
    paired = [-1] * len(spans)
    for position, index in enumerate(order):
        split = spans[index]["split"]
        banned = {spans[index]["start_observation_id"], spans[index]["end_observation_id"]}
        for step in range(1, len(order)):
            candidate_index = order[(position + step) % len(order)]
            candidate = spans[candidate_index]
            if candidate["split"] != split:
                continue
            if candidate["end_observation_id"] in banned:
                continue
            paired[index] = candidate_index
            break
        if paired[index] < 0:
            raise ValueError(f"span {index} has no other arrival in its split")
    return paired


def other_end_ids(spans: list[dict[str, Any]]) -> list[str]:
    """Pair each span with the next span's arrival in the same split."""
    return [spans[index]["end_observation_id"] for index in other_span_indexes(spans)]


def mean_normalized(vectors: np.ndarray) -> np.ndarray:
    """Mean of per-frame L2-normalized vectors. The mean itself is not re-normalized."""
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"a node descriptor needs a stack of vectors, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("node views must be finite")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("node views must be non-zero")
    mean = (array / norms[:, None]).mean(axis=0)
    if not np.isfinite(mean).all() or float(np.linalg.norm(mean)) == 0.0:
        raise ValueError("normalized views cancel")
    return mean


def place_verdict(start_end: list[float], start_other: list[float]) -> dict[str, Any]:
    if len(start_end) == 0 or len(start_end) != len(start_other):
        raise ValueError("place comparison needs paired cosines")
    if not all(math.isfinite(value) for value in (*start_end, *start_other)):
        raise ValueError("place comparison needs finite cosines")
    delta = [end - other for end, other in zip(start_end, start_other)]
    median_delta = _median(delta)
    call = (
        "arrival_closer_than_another_place"
        if median_delta > 0.0
        else "arrival_as_far_as_another_place"
    )
    return {
        "call": call,
        "count": len(start_end),
        "median_cosine_start_end": _median(start_end),
        "median_cosine_start_other": _median(start_other),
        "median_delta": median_delta,
        "training_started": False,
        "collection_started": False,
        "habitat_started": False,
        "formal_cache_written": False,
        "branch_choice_tested": False,
    }


def sweep_verdict(start_end: list[float], start_other: list[float]) -> dict[str, Any]:
    """Accept a turn-bout descriptor only when the median gap reaches ``SWEEP_MARGIN``."""
    if len(start_end) == 0 or len(start_end) != len(start_other):
        raise ValueError("sweep comparison needs paired cosines")
    if not all(math.isfinite(value) for value in (*start_end, *start_other)):
        raise ValueError("sweep comparison needs finite cosines")
    delta = [end - other for end, other in zip(start_end, start_other)]
    median_delta = _median(delta)
    accepted = median_delta >= SWEEP_MARGIN
    return {
        "call": (
            "turn_sweep_separates_the_next_node"
            if accepted
            else "turn_sweep_does_not_separate_the_next_node"
        ),
        "accepted": accepted,
        "margin": SWEEP_MARGIN,
        "count": len(start_end),
        "median_cosine_start_end": _median(start_end),
        "median_cosine_start_other": _median(start_other),
        "median_delta": median_delta,
        "count_closer": sum(value > 0.0 for value in delta),
        "count_reaching_margin": sum(value >= SWEEP_MARGIN for value in delta),
        "training_started": False,
        "collection_started": False,
        "habitat_started": False,
        "formal_cache_written": False,
        "branch_choice_tested": False,
    }


def _median(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
