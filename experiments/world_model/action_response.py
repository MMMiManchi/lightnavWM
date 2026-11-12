"""Metrics for a fixed-observation action swap. They do not train or encode."""

from __future__ import annotations

import numpy as np

TRUE_DIFF_FLOOR = 1.0
PRED_DIFF_FLOOR = 1.0
SPREAD_RATIO_RESPONSIVE = 0.05
NO_ACTION_SPREAD_RATIO_MAX = 0.01
SEEDS = (20261029, 20261030, 20261031)


def pairwise_l2(grids: np.ndarray) -> float:
    """Mean L2 between unordered pairs of flattened grids shaped (actions, 729, 1152)."""
    array = np.asarray(grids, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] < 2:
        raise ValueError("pairwise L2 expects at least two action grids")
    if not np.isfinite(array).all():
        raise ValueError("pairwise L2 received a non-finite grid")
    flat = array.reshape(array.shape[0], -1)
    total = 0.0
    count = 0
    for left in range(flat.shape[0]):
        for right in range(left + 1, flat.shape[0]):
            total += float(np.linalg.norm(flat[left] - flat[right]))
            count += 1
    return total / count


def difference_directions(
    predicted: np.ndarray,
    true: np.ndarray,
    *,
    true_floor: float = TRUE_DIFF_FLOOR,
    pred_floor: float = PRED_DIFF_FLOOR,
) -> dict:
    """Cosine of action-pair differences. A tiny predicted difference is not a direction."""
    left = np.asarray(predicted, dtype=np.float64)
    right = np.asarray(true, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 3 or left.shape[0] < 2:
        raise ValueError("difference directions expect aligned action grids")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("difference directions received a non-finite grid")
    predicted_flat = left.reshape(left.shape[0], -1)
    true_flat = right.reshape(right.shape[0], -1)
    comparable = 0
    responsive = 0
    cosines: list[float] = []
    for first in range(predicted_flat.shape[0]):
        for second in range(first + 1, predicted_flat.shape[0]):
            true_norm = float(np.linalg.norm(true_flat[first] - true_flat[second]))
            if true_norm < true_floor:
                continue
            comparable += 1
            predicted_delta = predicted_flat[first] - predicted_flat[second]
            predicted_norm = float(np.linalg.norm(predicted_delta))
            if predicted_norm < pred_floor:
                continue
            responsive += 1
            true_delta = true_flat[first] - true_flat[second]
            cosines.append(float(np.dot(predicted_delta, true_delta) / (predicted_norm * true_norm)))
    mean = None if not cosines else float(np.mean(cosines))
    return {
        "comparable_pairs": comparable,
        "responsive_pairs": responsive,
        "direction_cosine_mean": mean,
    }


def action_response_verdict(action_rows: list[dict], no_action_rows: list[dict]) -> dict:
    """Classify the probe. Fit-state mechanism is not a generalization result."""

    def ratios(rows: list[dict], label: str) -> dict[int, float]:
        found = {}
        for row in rows:
            seed = int(row["seed"])
            if seed not in SEEDS or seed in found:
                raise ValueError(f"{label} seed {seed} is missing or duplicated")
            ratio = float(row["spread_ratio_median"])
            if not np.isfinite(ratio) or ratio < 0.0:
                raise ValueError(f"{label} seed {seed} has a non-finite spread ratio")
            found[seed] = ratio
        if tuple(sorted(found)) != SEEDS:
            raise ValueError(f"{label} rows are not the three registered seeds")
        return found

    action = ratios(action_rows, "action")
    no_action = ratios(no_action_rows, "no-action")
    probe_valid = all(no_action[seed] < NO_ACTION_SPREAD_RATIO_MAX for seed in SEEDS)
    if not probe_valid:
        response = "invalid"
    elif all(action[seed] < SPREAD_RATIO_RESPONSIVE for seed in SEEDS):
        response = "insensitive"
    elif all(action[seed] >= SPREAD_RATIO_RESPONSIVE for seed in SEEDS):
        response = "responsive"
    else:
        response = "mixed"
    cosines = []
    for row in action_rows:
        value = row.get("direction_cosine_mean")
        cosines.append(None if value is None else float(value))
    if response != "responsive" or any(value is None or not np.isfinite(value) for value in cosines):
        direction = "undefined"
    elif all(value > 0.0 for value in cosines):
        direction = "same_direction"
    else:
        direction = "not_aligned"
    labels = {
        "invalid": "探针无效",
        "insensitive": "输出几乎不随命令变化",
        "responsive": "输出随命令变化",
        "mixed": "三个 seed 的响应不一致",
        "undefined": "方向未定义",
        "same_direction": "差的方向与真实分支同向",
        "not_aligned": "差的方向没有稳定同向",
    }
    return {
        "probe_valid": probe_valid,
        "response_call": response,
        "response_label": labels[response],
        "direction_call": direction,
        "direction_label": labels[direction],
        "spread_ratio_responsive": SPREAD_RATIO_RESPONSIVE,
        "no_action_spread_ratio_max": NO_ACTION_SPREAD_RATIO_MAX,
        "true_diff_floor": TRUE_DIFF_FLOOR,
        "pred_diff_floor": PRED_DIFF_FLOOR,
        "generalization": False,
        "states_are_formal_fit": True,
        "navigation_connected": False,
        "training_started": False,
        "habitat_started": False,
    }
