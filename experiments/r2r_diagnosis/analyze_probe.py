#!/usr/bin/env python3
"""Pair the finished 30-episode probe. Reads jsonl only; does not run Habitat."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "output" / "r2r_probe"
BASELINE = ROOT / "output" / "r2r_go2w" / "results.jsonl"
OUT = Path(__file__).resolve().parent

ROLLED = {
    "wrong_way": "一开始就走反",
    "big_drift": "走近后又走偏",
    "mild_drift": "走近后又走偏",
    "left_after_1m": "进入 3 米后又离开",
    "left_after_3m": "进入 3 米后又离开",
    "stop_just_outside": "停在最近距离 3–5 米",
    "stop_far_closest": "停在更远的位置",
    "success": "原本成功对照",
}
ORDER = [
    "一开始就走反",
    "走近后又走偏",
    "进入 3 米后又离开",
    "停在最近距离 3–5 米",
    "停在更远的位置",
    "原本成功对照",
]
APOS_RE = re.compile(r"<apos_(\d+)>")
OPOS_RE = re.compile(r"<opos_(\d+)>")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def baseline_index() -> dict[tuple[str, str], dict]:
    index = {}
    for row in load_jsonl(BASELINE):
        scene = Path(str(row["scene_id"])).parent.name
        index[(scene, str(row["habitat_episode_id"]))] = row
    return index


def pointing_state(raw: str) -> str:
    match = APOS_RE.search(raw or "")
    if not match:
        return "absent"
    token = int(match.group(1))
    if token == 1297:
        return "rot_left"
    if token == 1298:
        return "rot_right"
    if token == 1299:
        return "stop"
    if token == 0:
        return "none"
    return "point"


def main() -> None:
    modes = {}
    dup = {}
    for name in ("model", "oracle_stop", "expert"):
        rows = load_jsonl(PROBE / f"{name}.jsonl")
        keys = [(row["scene"], str(row["episode_id"])) for row in rows]
        dup[name] = len(keys) - len(set(keys))
        modes[name] = {(row["scene"], str(row["episode_id"])): row for row in rows}

    keys = set(modes["model"])
    paired = keys == set(modes["oracle_stop"]) == set(modes["expert"])
    base = baseline_index()
    missing_base = [key for key in keys if key not in base]

    paired_path = OUT / "paired_episode_comparison.csv"
    fields = [
        "episode_id",
        "scene",
        "habitat_episode_id",
        "failure_type",
        "probe_class",
        "baseline_success",
        "baseline_distance_m",
        "baseline_spl",
        "baseline_ndtw",
        "baseline_steps",
        "baseline_min_distance_m",
        "expert_success",
        "expert_distance_m",
        "expert_spl",
        "expert_ndtw",
        "expert_steps",
        "stop_success",
        "stop_distance_m",
        "stop_spl",
        "stop_ndtw",
        "stop_steps",
        "stop_oracle_fired",
        "stop_trigger_step",
        "stop_trigger_step_source",
        "original_eval_success",
        "original_eval_distance_m",
        "original_eval_spl",
        "original_eval_steps",
        "replay_matches_original_success",
        "first_heading_error_rad",
        "first_model_yaw_rad",
        "first_expert_yaw_rad",
        "first_apos_state",
        "first_raw_text",
        "path_length_m",
        "mean_infer_s",
    ]
    type_rows = []
    success_mismatch = []
    distance_deltas = []
    with paired_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(keys):
            model = modes["model"][key]
            expert = modes["expert"][key]
            stop = modes["oracle_stop"][key]
            original = base.get(key)
            first = model.get("first_step") or {}
            waypoint = first.get("model_waypoint") or [None, None, None]
            expert_wp = first.get("expert_waypoint") or [None, None, None]
            fired = bool(stop.get("oracle_fired"))
            trigger = stop["steps"] - 1 if fired else ""
            original_success = None
            if original is not None:
                original_success = float(original.get("success") or 0) >= 0.5
            match = original_success is not None and original_success == bool(model["success"])
            if original is not None and not match:
                success_mismatch.append(f"{key[0]}/{key[1]}")
            if original is not None and finite(original.get("final_distance")) is not None:
                distance_deltas.append(abs(float(model["final_distance_m"]) - float(original["final_distance"])))
            row = {
                "episode_id": f"{key[0]}/{key[1]}",
                "scene": key[0],
                "habitat_episode_id": key[1],
                "failure_type": ROLLED.get(model["probe_class"], model["probe_class"]),
                "probe_class": model["probe_class"],
                "baseline_success": int(bool(model["success"])),
                "baseline_distance_m": model["final_distance_m"],
                "baseline_spl": model["spl"],
                "baseline_ndtw": model["ndtw"],
                "baseline_steps": model["steps"],
                "baseline_min_distance_m": model["min_distance_m"],
                "expert_success": int(bool(expert["success"])),
                "expert_distance_m": expert["final_distance_m"],
                "expert_spl": expert["spl"],
                "expert_ndtw": expert["ndtw"],
                "expert_steps": expert["steps"],
                "stop_success": int(bool(stop["success"])),
                "stop_distance_m": stop["final_distance_m"],
                "stop_spl": stop["spl"],
                "stop_ndtw": stop["ndtw"],
                "stop_steps": stop["steps"],
                "stop_oracle_fired": int(fired),
                "stop_trigger_step": trigger,
                "stop_trigger_step_source": "derived_last_logged_action" if fired else "not_fired",
                "original_eval_success": "" if original_success is None else int(original_success),
                "original_eval_distance_m": "" if original is None else original.get("final_distance"),
                "original_eval_spl": "" if original is None else original.get("spl"),
                "original_eval_steps": "" if original is None else original.get("steps"),
                "replay_matches_original_success": int(match),
                "first_heading_error_rad": first.get("heading_error_rad", ""),
                "first_model_yaw_rad": waypoint[2],
                "first_expert_yaw_rad": expert_wp[2],
                "first_apos_state": pointing_state(first.get("raw_text") or ""),
                "first_raw_text": first.get("raw_text") or "",
                "path_length_m": "",
                "mean_infer_s": "",
            }
            writer.writerow(row)
            type_rows.append(row)

    by_type = {name: [] for name in ORDER}
    for row in type_rows:
        by_type[row["failure_type"]].append(row)
    type_path = OUT / "failure_type_comparison.csv"
    with type_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "failure_type",
                "n",
                "model_success",
                "expert_success",
                "stop_success",
                "stop_recovered",
                "expert_recovered",
                "sampled",
            ],
        )
        writer.writeheader()
        for name in ORDER:
            group = by_type[name]
            sampled = name != "停在更远的位置"
            writer.writerow({
                "failure_type": name,
                "n": len(group),
                "model_success": sum(int(row["baseline_success"]) for row in group),
                "expert_success": sum(int(row["expert_success"]) for row in group),
                "stop_success": sum(int(row["stop_success"]) for row in group),
                "stop_recovered": sum(
                    (not int(row["baseline_success"])) and int(row["stop_success"]) for row in group
                ),
                "expert_recovered": sum(
                    (not int(row["baseline_success"])) and int(row["expert_success"]) for row in group
                ),
                "sampled": int(sampled and len(group) > 0),
            })

    def pack(name: str) -> dict:
        rows = list(modes[name].values())
        first_infer = []
        for row in rows:
            step = row.get("first_step") or {}
            if step.get("infer_s") is not None:
                first_infer.append(float(step["infer_s"]))
        return {
            "n": len(rows),
            "success": sum(bool(row["success"]) for row in rows),
            "fail": sum(not bool(row["success"]) for row in rows),
            "sr": sum(bool(row["success"]) for row in rows) / len(rows),
            "spl": mean([finite(row["spl"]) for row in rows]),
            "ne_geodesic_m": mean([finite(row["final_distance_m"]) for row in rows]),
            "ndtw": mean([finite(row["ndtw"]) for row in rows]),
            "mean_decisions": mean([float(row["steps"]) for row in rows]),
            "mean_first_infer_s": mean(first_infer),
            "first_infer_n": len(first_infer),
            "oracle_fired": sum(bool(row.get("oracle_fired")) for row in rows),
            "success_with_final_ge_3": [
                f"{row['scene']}/{row['episode_id']}"
                for row in rows
                if row["success"] and finite(row["final_distance_m"]) is not None and row["final_distance_m"] >= 3.0
            ],
        }

    model_rows = {key: modes["model"][key] for key in keys}
    stop_gain = [
        f"{key[0]}/{key[1]}"
        for key in keys
        if (not model_rows[key]["success"]) and modes["oracle_stop"][key]["success"]
    ]
    stop_hurt = [
        f"{key[0]}/{key[1]}"
        for key in keys
        if model_rows[key]["success"] and (not modes["oracle_stop"][key]["success"])
    ]
    expert_gain = [
        f"{key[0]}/{key[1]}"
        for key in keys
        if (not model_rows[key]["success"]) and modes["expert"][key]["success"]
    ]
    expert_hurt = [
        f"{key[0]}/{key[1]}"
        for key in keys
        if model_rows[key]["success"] and (not modes["expert"][key]["success"])
    ]

    heading = []
    for key, row in model_rows.items():
        if row["probe_class"] == "success":
            continue
        err = (row.get("first_step") or {}).get("heading_error_rad")
        if err is None:
            continue
        heading.append((ROLLED[row["probe_class"]], abs(float(err))))

    summary = {
        "paired": paired,
        "duplicates": dup,
        "n": len(keys),
        "missing_baseline": [f"{a}/{b}" for a, b in missing_base],
        "success_flag_mismatches": success_mismatch,
        "abs_final_distance_delta_m": {
            "n": len(distance_deltas),
            "median": sorted(distance_deltas)[len(distance_deltas) // 2] if distance_deltas else None,
            "max": max(distance_deltas) if distance_deltas else None,
            "gt_0_5m": sum(delta > 0.5 for delta in distance_deltas),
            "gt_1m": sum(delta > 1.0 for delta in distance_deltas),
        },
        "modes": {name: pack(name) for name in ("model", "oracle_stop", "expert")},
        "stop_recovered": stop_gain,
        "stop_hurt": stop_hurt,
        "expert_recovered": expert_gain,
        "expert_hurt": expert_hurt,
        "failure_heading_gt_45deg": sum(err > math.radians(45) for _, err in heading),
        "failure_heading_n": len(heading),
        "apos_states": dict(Counter(
            pointing_state((row.get("first_step") or {}).get("raw_text") or "")
            for row in model_rows.values()
        )),
    }
    (OUT / "probe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
