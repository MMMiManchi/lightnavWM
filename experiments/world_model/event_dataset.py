"""Causal event pair construction. Official geometry is offline supervision only."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from experiments.world_model.pose_math import wrap_pi


def select_direction(origin: dict, departure: dict, place: dict, max_error_degrees: float = 35) -> dict | None:
    """Associate the observed departure with a candidate present at decision time.

    The future trajectory labels which candidate was taken. Its displacement,
    coordinates and command sequence are never returned as model inputs.
    """
    candidates = [e for e in place["exits"] if e["exit_id"] in origin.get("angle_candidate_exit_ids", [])]
    clear = {r["exit_id"] for r in origin.get("depth_exit_status", []) if r["status"] == "not_blocked"}
    dx = departure["position"][0] - origin["position"][0]
    dz = departure["position"][2] - origin["position"][2]
    if math.hypot(dx, dz) < 0.4 or origin.get("yaw") is None:
        return None
    walked = math.atan2(dx, dz)
    options = []
    for e in candidates:
        point = e["toward_position"]
        bearing = math.atan2(point[0]-origin["position"][0], point[2]-origin["position"][2])
        error = abs(wrap_pi(bearing-walked))
        if e["exit_id"] in clear and error <= math.radians(max_error_degrees):
            offset = wrap_pi(bearing-origin["yaw"]-math.pi)
            options.append((error, e["exit_id"], offset))
    if not options:
        return None
    error, exit_id, offset = min(options)
    return {"exit_id": exit_id, "direction": [math.sin(offset), math.cos(offset)], "matching_error_rad": error}


def construct_event_pairs(frames: list[dict], places: list[dict]) -> tuple[list[dict], dict]:
    """Use complete formal walks; never cross a reset, run, or episode."""
    by_place = {r["place_id"]: r for r in places}
    walks = defaultdict(list)
    for r in frames:
        if r["split"].startswith("formal_"):
            walks[(r["split"], r["episode_uid"])].append(r)
    pairs = []
    counts = Counter()
    for (split, uid), sequence in sorted(walks.items()):
        sequence.sort(key=lambda r:r["index"])
        if [r["index"] for r in sequence] != list(range(len(sequence))):
            raise ValueError("event walk has missing or duplicate steps")
        events = [r for r in sequence if r["phase"] in ("enter", "started_inside")]
        for a, b in zip(events, events[1:]):
            counts["adjacent_events"] += 1
            if a["place_id"] == b["place_id"] or b["index"]-a["index"] < 3:
                counts["same_place_or_too_short"] += 1
                continue
            if not a["has_rgb"] or not b["has_rgb"]:
                counts["missing_endpoint_rgb"] += 1
                continue
            span = sequence[a["index"]+1:b["index"]+1]
            departure = next((r for r in span if (
                not r["in_decision_region"] or r["place_id"] != a["place_id"]
            ) and math.hypot(r["position"][0]-a["position"][0], r["position"][2]-a["position"][2]) >= 0.4), None)
            if departure is None:
                counts["no_departure_evidence"] += 1
                continue
            choice = select_direction(a, departure, by_place[a["place_id"]])
            if choice is None:
                counts["no_matching_current_clear_direction"] += 1
                continue
            prior_events = [r["index"] for r in events if r["index"] < a["index"]][-4:]
            indices = [-1] * (4-len(prior_events)) + prior_events
            indices += [max(0, a["index"]-offset) for offset in (12, 8, 4, 0)]
            if any(i >= b["index"] for i in indices) or any(i > a["index"] for i in indices):
                raise ValueError("future history leakage")
            if any(i >= 0 and not sequence[i]["has_rgb"] for i in indices):
                counts["missing_history_rgb"] += 1
                continue
            pairs.append({
                "sample_id": f"{split}|{uid}|{a['index']}->{b['index']}",
                "source_run": split, "episode_uid": uid, "scene": a["scene"],
                "partition": "train_dev" if a["dev_excluded"] else "train_fit",
                "origin_index": a["index"], "target_index": b["index"],
                "history_indices": indices,
                "history_observation_ids": [sequence[i]["source_observation_id"] if i >= 0 else None for i in indices],
                "target_observation_id": b["source_observation_id"],
                "branch_direction": choice["direction"],
                "offline_labels": {"origin_place": a["place_id"], "target_place": b["place_id"],
                                   "chosen_exit": choice["exit_id"], "departure_evidence_index": departure["index"],
                                   "matching_error_rad": choice["matching_error_rad"],
                                   "branch_source": "offline_geometry_direction_teacher_not_a_runtime_graph"},
            })
    counts["eligible_pairs"] = len(pairs)
    return pairs, dict(counts)
