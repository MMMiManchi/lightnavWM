#!/usr/bin/env python3
"""Recompute R2R failure classes from the saved episode logs.

The eval jsonl has one row per episode, not per step. This script checks that
the published counts are exclusive and exhaustive, estimates the stop-timing
upper bound, and writes a retrieval table. It does not rerun the baseline.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OFFICIAL = ROOT / "output" / "r2r_official" / "results.jsonl"
GO2 = ROOT / "output" / "r2r_go2w" / "results.jsonl"
EXPERT = ROOT / "output" / "go2w_expert" / "samples.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def key_of(row: dict) -> tuple[str, str]:
    scene = Path(str(row["scene_id"])).parent.name
    return scene, str(row["habitat_episode_id"])


def classify(row: dict) -> str:
    """Exclusive label. Thresholds match the published failure table."""
    minimum = finite(row.get("min_distance"))
    final = finite(row.get("final_distance"))
    initial = finite((row.get("termination_details") or {}).get("distance_to_goal_initial"))
    if minimum is None or final is None:
        return "unreachable"
    if minimum <= 3.0 and final > 3.0:
        return "left_after_1m" if minimum <= 1.0 else "left_after_3m"
    regret = final - minimum
    progress = (initial if initial is not None else minimum) - minimum
    if regret <= 0.5:
        return "stop_just_outside" if minimum <= 5.0 else "stop_far_closest"
    if progress < 1.0:
        return "wrong_way"
    if regret >= 3.0:
        return "big_drift"
    return "mild_drift"


ROLLED = {
    "wrong_way": "一开始就走反",
    "mild_drift": "走近后又走偏",
    "big_drift": "走近后又走偏",
    "left_after_1m": "进入 3 米后又离开",
    "left_after_3m": "进入 3 米后又离开",
    "stop_just_outside": "停在最近距离 3–5 米",
    "stop_far_closest": "停在更远的位置",
    "unreachable": "测地距离无穷",
}


def median(values: list[float]) -> float | None:
    nums = sorted(v for v in values if v is not None)
    if not nums:
        return None
    return nums[len(nums) // 2]


def mean(values: list[float]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def take_spread(rows: list[dict], count: int) -> list[dict]:
    rows = sorted(rows, key=lambda row: (row["scene"], row["episode_id"]))
    if len(rows) <= count:
        return rows
    return [rows[int(i * len(rows) / count)] for i in range(count)]


def expert_outcomes(path: Path) -> dict:
    if not path.exists():
        return {"present": False}
    last: dict[tuple[str, str], dict] = {}
    skipped = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("skipped"):
                skipped += 1
                continue
            if row.get("episode_done") and "distance_m" in row:
                last[(row["scene"], str(row["episode_id"]))] = row
    distances = [float(row["distance_m"]) for row in last.values()]
    return {
        "present": True,
        "episodes_with_final_pose": len(last),
        "skipped_rows": skipped,
        "final_lt_3m": sum(d < 3.0 for d in distances),
        "final_le_1m": sum(d <= 1.0 for d in distances),
        "final_median_m": median(distances),
        "note": (
            "These are poses along the privileged shortest path, not Habitat "
            "success flags. A final distance under 3 m means the path entered "
            "the success radius."
        ),
    }


def summarize(rows: list[dict]) -> dict:
    failures = [row for row in rows if float(row.get("success") or 0) < 0.5]
    counts = Counter(classify(row) for row in failures)
    rolled = Counter(ROLLED[classify(row)] for row in failures)
    finite_rows = [row for row in rows if finite(row.get("final_distance")) is not None]
    finals = [finite(row["final_distance"]) for row in finite_rows]
    minima = [finite(row["min_distance"]) for row in finite_rows]
    hindsight_success = [
        row for row in rows if bool(row.get("oracle_success")) and float(row.get("success") or 0) < 0.5
    ]
    return {
        "episodes": len(rows),
        "successes": sum(float(row.get("success") or 0) >= 0.5 for row in rows),
        "failures": len(failures),
        "sr": sum(float(row.get("success") or 0) >= 0.5 for row in rows) / len(rows),
        "oracle_success_rate": sum(bool(row.get("oracle_success")) for row in rows) / len(rows),
        "ne_final_m": mean(finals),
        "ne_if_stop_at_closest_m": mean(minima),
        "spl": mean([float(row["spl"]) for row in rows if finite(row.get("spl")) is not None]),
        "ndtw": mean([float(row["ndtw"]) for row in rows if finite(row.get("ndtw")) is not None]),
        "forced_stop_failures": sum(bool(row.get("forced_stop")) for row in failures),
        "failure_steps_ge_490": sum(int(row["steps"]) >= 490 for row in failures),
        "class_counts": dict(counts),
        "rolled_counts": dict(rolled),
        "hindsight_stop_recovers": len(hindsight_success),
        "hindsight_stop_matches_left_class": len(hindsight_success)
        == counts["left_after_1m"] + counts["left_after_3m"],
    }


def main() -> None:
    official = load_jsonl(OFFICIAL)
    go2 = load_jsonl(GO2)
    off_by = {key_of(row): row for row in official}
    go_by = {key_of(row): row for row in go2}
    if set(off_by) != set(go_by):
        raise SystemExit("official and go2 episode keys differ")

    index_path = OUT / "failure_index.jsonl"
    probe_pool: dict[str, list[dict]] = defaultdict(list)
    with index_path.open("w") as handle:
        for key, row in sorted(go_by.items()):
            other = off_by[key]
            label = classify(row)
            record = {
                "scene": key[0],
                "episode_id": key[1],
                "instruction": row.get("instruction", ""),
                "go2_success": float(row.get("success") or 0) >= 0.5,
                "official_success": float(other.get("success") or 0) >= 0.5,
                "go2_class": label,
                "go2_rolled": ROLLED[label],
                "official_class": classify(other),
                "official_rolled": ROLLED[classify(other)] if float(other.get("success") or 0) < 0.5 else "success",
                "go2_min_m": finite(row.get("min_distance")),
                "go2_final_m": finite(row.get("final_distance")),
                "go2_min_step": (row.get("termination_details") or {}).get("distance_to_goal_min_at_step"),
                "go2_steps": row.get("steps"),
                "go2_oracle_success": bool(row.get("oracle_success")),
                "go2_forced_stop": bool(row.get("forced_stop")),
                "official_min_m": finite(other.get("min_distance")),
                "official_final_m": finite(other.get("final_distance")),
                "official_steps": other.get("steps"),
                "shard": row.get("shard"),
                "logged_episode_id": row.get("episode_id"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not record["go2_success"]:
                probe_pool[label].append(record)
            elif record["go2_class"] != "unreachable":
                probe_pool["success"].append(record)

    probe = []
    for label, count in (
        ("wrong_way", 6),
        ("big_drift", 6),
        ("left_after_1m", 3),
        ("left_after_3m", 3),
        ("stop_just_outside", 6),
        ("success", 6),
    ):
        for row in take_spread(probe_pool[label], count):
            probe.append({"scene": row["scene"], "episode_id": row["episode_id"], "probe_class": label})
    (OUT / "probe_episodes.json").write_text(json.dumps(probe, indent=2) + "\n")

    inf_keys = [key for key, row in go_by.items() if classify(row) == "unreachable"]
    inf_official = Counter(
        "success" if float(off_by[key].get("success") or 0) >= 0.5 else ROLLED[classify(off_by[key])]
        for key in inf_keys
    )
    both_fail = 0
    only_go2 = 0
    only_official = 0
    for key, row in go_by.items():
        g = float(row.get("success") or 0) >= 0.5
        o = float(off_by[key].get("success") or 0) >= 0.5
        if not g and not o:
            both_fail += 1
        elif not g:
            only_go2 += 1
        elif not o:
            only_official += 1

    report = {
        "sources": {
            "official_results": str(OFFICIAL),
            "go2_results": str(GO2),
            "failure_index": str(index_path),
            "missing_from_results": [
                "per-step pose",
                "per-step pointing",
                "per-step RVQ codes",
                "decoded waypoints",
                "collision flag",
                "per-step inference latency",
            ],
            "present_in_results": [
                "episode id",
                "scene id",
                "instruction",
                "success",
                "oracle_success",
                "spl",
                "ndtw",
                "steps",
                "min_distance",
                "final_distance",
                "min distance step",
                "aggregated velocity action counts",
                "forced_stop",
            ],
        },
        "official": summarize(official),
        "go2": summarize(go2),
        "overlap": {
            "both_fail": both_fail,
            "only_go2_fail": only_go2,
            "only_official_fail": only_official,
            "go2_inf": len(inf_keys),
            "official_outcome_of_go2_inf": dict(inf_official),
        },
        "expert_shortest_path": expert_outcomes(EXPERT),
        "probe_episodes": len(probe),
        "class_rule": {
            "unreachable": "min or final distance is not finite",
            "left_after_1m": "min <= 1 and final > 3",
            "left_after_3m": "1 < min <= 3 and final > 3",
            "stop_just_outside": "final - min <= 0.5 and 3 < min <= 5",
            "stop_far_closest": "final - min <= 0.5 and min > 5",
            "wrong_way": "not the above, progress from start < 1 m, and final is > 0.5 m worse than min",
            "big_drift": "progress >= 1 m and final is >= 3 m worse than min",
            "mild_drift": "progress >= 1 m and final is 0.5–3 m worse than min",
        },
    }
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "official_rolled": report["official"]["rolled_counts"],
        "go2_rolled": report["go2"]["rolled_counts"],
        "official_sr": round(report["official"]["sr"], 4),
        "go2_sr": round(report["go2"]["sr"], 4),
        "hindsight_official": report["official"]["hindsight_stop_recovers"],
        "hindsight_go2": report["go2"]["hindsight_stop_recovers"],
        "hindsight_match": report["official"]["hindsight_stop_matches_left_class"],
        "expert": report["expert_shortest_path"],
        "probe": len(probe),
        "inf_official": report["overlap"]["official_outcome_of_go2_inf"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
