"""Nested real-episode experience budgets, independent of future query labels."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def ordered_scene_episodes(pairs, episode_order):
    rank = {tuple(ep): i for i, ep in enumerate(episode_order)}
    grouped = defaultdict(lambda: defaultdict(list))
    for i, pair in enumerate(pairs):
        key = (pair["source_run"], pair["episode_uid"])
        if key not in rank:
            raise ValueError("recorded episode is missing from acquisition order")
        grouped[pair["scene"]][key].append(i)
    return {scene: [(key, sorted(rows, key=lambda i: pairs[i]["origin_index"]))
                    for key, rows in sorted(episodes.items(), key=lambda x: rank[x[0]])]
            for scene, episodes in grouped.items()}


def prior_episode_rows(pairs, episode_order, query_index, budget):
    if budget is not None and budget < 0:
        raise ValueError("history budget cannot be negative")
    query = pairs[query_index]
    episodes = ordered_scene_episodes(pairs, episode_order)[query["scene"]]
    key = (query["source_run"], query["episode_uid"])
    position = next(i for i, (ep, _rows) in enumerate(episodes) if ep == key)
    visible = episodes[:position] if budget is None else episodes[:min(position, budget)]
    return [row for _ep, rows in visible for row in rows], len(visible)


def choose_live_scenes(pairs, episode_order, count=8):
    grouped = ordered_scene_episodes(pairs, episode_order)
    eligible = [s for s, eps in grouped.items() if len(eps) >= 3]
    eligible.sort(key=lambda s: (pairs[grouped[s][0][1][0]]["partition"] != "train_dev", s))
    if len(eligible) < count:
        raise ValueError("not enough scenes with two genuinely prior episodes")
    rows = []
    for scene in eligible[:count]:
        episodes = grouped[scene]
        (_run, uid), queries = episodes[-1]
        rows.append({"scene": scene, "partition": pairs[queries[0]]["partition"],
                     "episode_uid": uid, "episode_id": uid.split("|")[-1],
                     "query_index": queries[0], "available_prior_episodes": len(episodes)-1})
    return rows


def budget_memory_indices(pairs, episode_order, history, branch, query_ids, budget, top_k=3):
    indices = np.full((len(query_ids), top_k), -1, dtype=np.int64)
    episode_counts = np.zeros(len(query_ids), dtype=np.int64)
    source_counts, compatible_counts = episode_counts.copy(), episode_counts.copy()
    for row, query in enumerate(query_ids):
        available, episode_counts[row] = prior_episode_rows(pairs, episode_order, int(query), budget)
        if not available:
            continue
        available = np.array(available, dtype=np.int64)
        source = history[available, -1] @ history[query, -1]
        direction = branch[available] @ branch[query]
        source_ok = source >= .5
        compatible = source_ok & (direction >= math.cos(math.radians(35)))
        source_counts[row], compatible_counts[row] = source_ok.sum(), compatible.sum()
        positions = np.flatnonzero(compatible)
        positions = positions[np.argsort(-(.8*source[positions]+.2*direction[positions]), kind="stable")[:top_k]]
        indices[row, :len(positions)] = available[positions]
    return indices, episode_counts, source_counts, compatible_counts


def seed_budget(bank, pairs, arrays, episode_order, query_index, budget):
    visible, episodes = prior_episode_rows(pairs, episode_order, query_index, budget)
    query = pairs[query_index]
    for i in visible:
        pair = pairs[i]
        bank.record_observed(context=query["scene"], episode_id=pair["episode_uid"], event_id=pair["sample_id"],
            source_observation=pair["history_observation_ids"][-1], target_observation=pair["target_observation_id"],
            execution_ref=f"admitted-recorded-span:{pair['sample_id']}", source_feature=arrays["history"][i, -1],
            observed_future=arrays["target"][i], branch_direction=arrays["branch"][i], executed=True, arrived=True)
    return {"episodes": episodes, "records": len(visible), "pair_ids": visible,
            "source": "strictly prior admitted recorded event outcomes, not fresh unverified live arrivals"}
