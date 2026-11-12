#!/usr/bin/env python3
"""Along a recorded trajectory, compare the original frame with ±1° image rotations.

The robot's logged path is not re-executed. History is the saved JPEG sequence.
The ViT cache is keyed only by frame index, so a rotated current frame must not
be left in that cache. Each extra decode drops those keys and then restores the
cache from before the rotation.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TRACE = ROOT / "output" / "r2r_probe" / "trace_20261020" / "steps.jsonl"
DIAG = ROOT / "output" / "r2r_probe" / "trace_20261020" / "trace_diagnostics.csv"
OUT_DIR = Path(__file__).resolve().parent / "replay_sensitivity"
ANGLES = (-1.0, 1.0)
YAW_RAD = 0.05
FORWARD_M = 0.05


def rotate_rgb(rgb: np.ndarray, degrees: float) -> np.ndarray:
    height, width = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return cv2.warpAffine(
        rgb, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def load_trace() -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with TRACE.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[(row["scene"], str(row["episode_id"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))
    return grouped


def load_diag() -> dict[tuple[str, str], dict]:
    found = {}
    with DIAG.open() as handle:
        for row in csv.DictReader(handle):
            found[(row["scene"], row["episode_id"])] = row
    return found


def waypoint_of(text: str, agent) -> tuple[list[int] | None, list[float] | None, str | None]:
    from lightnav.habitat.policy import select_action_waypoint
    from lightnav.vln_utils import parse_rvq_action_tokens

    try:
        waypoints, _ = agent.decode_waypoints(text)
        _, chosen = select_action_waypoint(np.asarray(waypoints, dtype=np.float32))
        codes = [int(c) for c in parse_rvq_action_tokens(text)]
    except ValueError as exc:
        return None, None, str(exc)
    return codes, [round(float(v), 5) for v in chosen], None


def meaningful(base: list[float] | None, other: list[float] | None) -> dict:
    if not base or not other:
        return {"comparable": False, "action_changed": False, "yaw_sign_changed": False, "stop_changed": False}
    df = float(other[0]) - float(base[0])
    dy = float(other[2]) - float(base[2])
    sign = (
        abs(other[2]) > YAW_RAD
        and abs(base[2]) > YAW_RAD
        and (other[2] > 0) != (base[2] > 0)
    )
    base_stop = abs(base[0]) <= 1e-3 and abs(base[2]) <= 1e-3
    other_stop = abs(other[0]) <= 1e-3 and abs(other[2]) <= 1e-3
    return {
        "comparable": True,
        "forward_delta_m": round(df, 5),
        "yaw_delta_rad": round(dy, 5),
        "action_changed": abs(df) > FORWARD_M or abs(dy) > YAW_RAD,
        "yaw_sign_changed": bool(sign),
        "stop_changed": base_stop != other_stop,
    }


def drop_frame(cache, frame_id: int) -> None:
    for key in list(cache._store):
        if key[0] == frame_id or key[1] == frame_id:
            del cache._store[key]


def restore_cache(cache, snapshot) -> None:
    cache._store.clear()
    for key, value in snapshot:
        cache._store[key] = value


def decode_view(policy, engine, rgb: np.ndarray, degrees: float) -> dict:
    from lightnav.inference.frame_preprocessing import rgb_frame_to_model_tensor

    agent = policy.agent
    frame_id = int(agent._history_frame_ids[-1])
    index = agent._buffer_len - 1
    cache = engine._vit_cache
    snapshot = list(cache._store.items())
    original = agent._sf_buffer[index].detach().clone()
    started = time.monotonic()
    try:
        if degrees != 0.0:
            drop_frame(cache, frame_id)
            agent._sf_buffer[index] = rgb_frame_to_model_tensor(
                rotate_rgb(rgb, degrees), agent.video_size
            )
        text, _ = engine.generate_from_frames(
            agent._get_video_tensor(),
            agent.instruction,
            predict_horizon=1,
            frame_ids=list(agent._history_frame_ids),
            task_type="vlnce_traj",
        )
    finally:
        agent._sf_buffer[index] = original
        if degrees != 0.0:
            restore_cache(cache, snapshot)
    elapsed = time.monotonic() - started
    codes, waypoint, error = waypoint_of(text, agent)
    return {
        "degrees": degrees,
        "raw_text": text,
        "rvq_codes": codes,
        "waypoint": waypoint,
        "error": error,
        "infer_s": round(elapsed, 4),
    }


def summarize(records: list[dict], diag: dict[tuple[str, str], dict]) -> dict:
    matched = [row for row in records if row["replay_rvq_match"]]
    changed = [row for row in matched if row["any_action_changed"]]

    def rate(rows, pred) -> dict:
        denom = len(rows)
        num = sum(1 for row in rows if pred(row))
        return {"n": num, "denom": denom, "rate": None if denom == 0 else round(num / denom, 4)}

    def bucket(step: int, nsteps: int) -> str:
        if step == 0:
            return "step0"
        if step >= nsteps - 5:
            return "last5"
        if step < 10:
            return "1_to_9"
        if step < 30:
            return "10_to_29"
        return "30_plus"

    by_group = defaultdict(list)
    by_bucket = defaultdict(list)
    near_rise = []
    for row in matched:
        meta = diag.get((row["scene"], row["episode_id"]), {})
        by_group[meta.get("group", "unknown")].append(row)
        by_bucket[bucket(int(row["step"]), int(meta.get("steps") or 0))].append(row)
        rise = meta.get("earliest_sustained_rise_step")
        if rise not in (None, ""):
            if abs(int(row["step"]) - int(float(rise))) <= 3:
                near_rise.append(row)

    episodes = []
    by_ep = defaultdict(list)
    for row in records:
        by_ep[(row["scene"], row["episode_id"])].append(row)
    for key, rows in by_ep.items():
        ok = [row for row in rows if row["replay_rvq_match"]]
        hits = [row for row in ok if row["any_action_changed"]]
        meta = diag.get(key, {})
        episodes.append(
            {
                "scene": key[0],
                "episode_id": key[1],
                "group": meta.get("group"),
                "probe_class": meta.get("probe_class"),
                "success": meta.get("success"),
                "steps_tested": len(rows),
                "replay_rvq_match": len(ok),
                "action_changed_steps": len(hits),
                "first_changed_step": None if not hits else min(int(row["step"]) for row in hits),
                "yaw_sign_steps": sum(row["any_yaw_sign_changed"] for row in ok),
                "both_rotations_agree_against_original": sum(row["rotations_agree_against_original"] for row in ok),
            }
        )

    return {
        "steps_tested": len(records),
        "replay_text_match": rate(records, lambda row: row["replay_text_match"]),
        "replay_rvq_match": rate(records, lambda row: row["replay_rvq_match"]),
        "action_changed_on_rvq_matched_steps": rate(matched, lambda row: row["any_action_changed"]),
        "yaw_sign_changed": rate(matched, lambda row: row["any_yaw_sign_changed"]),
        "stop_changed": rate(matched, lambda row: row["any_stop_changed"]),
        "rotations_agree_against_original": rate(matched, lambda row: row["rotations_agree_against_original"]),
        "by_group": {
            name: rate(rows, lambda row: row["any_action_changed"]) for name, rows in sorted(by_group.items())
        },
        "by_step_bucket": {
            name: rate(rows, lambda row: row["any_action_changed"]) for name, rows in sorted(by_bucket.items())
        },
        "within_3_of_sustained_rise": rate(near_rise, lambda row: row["any_action_changed"]),
        "episodes_with_any_change": sum(row["action_changed_steps"] > 0 for row in episodes),
        "episodes": episodes,
        "thresholds": {"yaw_rad": YAW_RAD, "forward_m": FORWARD_M, "angles_deg": list(ANGLES)},
    }


def main() -> None:
    from lightnav.habitat.runner import HabitatEvalConfig, _default_engine_factory, build_velocity_policy

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grouped = load_trace()
    diag = load_diag()
    print("episodes", len(grouped), "cuda", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    cfg = HabitatEvalConfig(
        server="tcp://127.0.0.1:9",
        model_path=str(ROOT / "checkpoints" / "LightNav-0"),
        backend="vllm_local",
        gpu_memory_utilization=0.40,
        max_steps=1,
    )
    t0 = time.monotonic()
    engine, bundle = _default_engine_factory(cfg)
    policy = build_velocity_policy(
        cfg,
        engine,
        bundle,
        {"habitat_time_step": 0.1, "lin_vel_range": [0.0, 2.5], "ang_vel_range": [-300.0, 300.0]},
    )
    if not policy.agent.slowfast:
        raise SystemExit("expected the SlowFast buffer; refusing a different history layout")
    print(f"engine_ready_s {time.monotonic() - t0:.1f}", flush=True)

    records: list[dict] = []
    out_path = OUT_DIR / "steps.jsonl"
    with out_path.open("w") as handle:
        for key, rows in grouped.items():
            instruction = rows[0]["instruction"]
            policy.reset({"instruction": instruction})
            ep_match = 0
            ep_change = 0
            for row in rows:
                rgb = np.asarray(iio.imread(ROOT / row["frame_file"]))
                policy.agent.observe(rgb)
                views = [decode_view(policy, engine, rgb, 0.0)]
                views.extend(decode_view(policy, engine, rgb, angle) for angle in ANGLES)
                base = views[0]
                diffs = [meaningful(base["waypoint"], view["waypoint"]) for view in views[1:]]
                rotations_same = (
                    len(views) == 3
                    and views[1]["rvq_codes"] is not None
                    and views[1]["rvq_codes"] == views[2]["rvq_codes"]
                    and views[1]["rvq_codes"] != base["rvq_codes"]
                )
                record = {
                    "scene": key[0],
                    "episode_id": key[1],
                    "step": int(row["step"]),
                    "logged_raw_text": row["raw_text"],
                    "logged_rvq_codes": row["rvq_codes"],
                    "replay_raw_text": base["raw_text"],
                    "replay_rvq_codes": base["rvq_codes"],
                    "replay_waypoint": base["waypoint"],
                    "replay_text_match": base["raw_text"] == row["raw_text"],
                    "replay_rvq_match": base["rvq_codes"] == row["rvq_codes"],
                    "views": views,
                    "diffs": diffs,
                    "any_action_changed": any(item.get("action_changed") for item in diffs),
                    "any_yaw_sign_changed": any(item.get("yaw_sign_changed") for item in diffs),
                    "any_stop_changed": any(item.get("stop_changed") for item in diffs),
                    "rotations_agree_against_original": rotations_same,
                }
                records.append(record)
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                ep_match += int(record["replay_rvq_match"])
                ep_change += int(record["any_action_changed"])
            print(
                f"{key[0]}/{key[1]} steps {len(rows)} rvq_match {ep_match}/{len(rows)} "
                f"action_changed {ep_change}",
                flush=True,
            )

    summary = summarize(records, diag)
    summary["engine_load_s"] = round(time.monotonic() - t0, 1)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
