#!/usr/bin/env python3
"""Open-loop check: same JPEG, repeated greedy decode, and image-plane rotation.

This does not change the Habitat camera, the robot heading, or the checkpoint.
Each trial resets the policy, so history and the ViT cache do not carry over.
The saved frames are JPEG, not the original simulator buffer.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TRACE = ROOT / "output" / "r2r_probe" / "trace_20261020" / "steps.jsonl"
OUT_DIR = Path(__file__).resolve().parent
CASES = [
    ("2azQ1b91cZZ", "10", "success"),
    ("X7HyMhZNoso", "1090", "success"),
    ("8194nk5LbLH", "221", "wrong_way"),
    ("QUCTc6BB5sX", "569", "wrong_way"),
    ("2azQ1b91cZZ", "326", "big_drift"),
    ("EU6Fwq7SyZv", "639", "big_drift"),
    ("2azQ1b91cZZ", "214", "left_after_1m"),
    ("2azQ1b91cZZ", "1087", "left_after_3m"),
    ("2azQ1b91cZZ", "1266", "stop_just_outside"),
    ("oLBMNvg9in8", "846", "stop_just_outside"),
]
CONDITIONS = [
    ("repeat", 0.0, 0),
    ("repeat", 0.0, 1),
    ("repeat", 0.0, 2),
    ("image_rotate", -1.0, 0),
    ("image_rotate", -0.5, 0),
    ("image_rotate", 0.5, 0),
    ("image_rotate", 1.0, 0),
]


def rotate_rgb(rgb: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate in the image plane about the frame center. Degrees, not radians."""
    if degrees == 0.0:
        return rgb
    height, width = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return cv2.warpAffine(
        rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def load_step0() -> dict[tuple[str, str], dict]:
    wanted = {(scene, episode) for scene, episode, _ in CASES}
    found = {}
    with TRACE.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["scene"], str(row["episode_id"]))
            if key in wanted and int(row["step"]) == 0:
                found[key] = row
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"missing step-0 rows: {sorted(missing)}")
    return found


def main() -> None:
    from lightnav.habitat.runner import HabitatEvalConfig, _default_engine_factory, build_velocity_policy
    from lightnav.inference.engine import _get_sampling_params

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "openloop_results.jsonl"
    summary_path = OUT_DIR / "openloop_summary.json"
    logged = load_step0()
    sampling = _get_sampling_params()
    print("sampling", sampling, flush=True)
    print("cuda", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

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
        {
            "habitat_time_step": 0.1,
            "lin_vel_range": [0.0, 2.5],
            "ang_vel_range": [-300.0, 300.0],
        },
    )
    load_s = time.monotonic() - t0
    print(f"engine_ready_s {load_s:.1f}", flush=True)

    import torch

    torch.cuda.reset_peak_memory_stats()
    records = []
    with rows_path.open("w") as handle:
        for scene, episode, label in CASES:
            source = logged[(scene, episode)]
            frame_path = ROOT / source["frame_file"]
            rgb = np.asarray(iio.imread(frame_path))
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise SystemExit(f"bad frame {frame_path} shape {rgb.shape}")
            instruction = source["instruction"]
            for kind, degrees, repeat in CONDITIONS:
                image = rotate_rgb(rgb, degrees)
                policy.reset({"instruction": instruction})
                started = time.monotonic()
                policy.act({"rgb": image})
                elapsed = time.monotonic() - started
                info = policy.get_info()
                waypoint = info.get("action_waypoint")
                record = {
                    "scene": scene,
                    "episode_id": episode,
                    "failure_type": label,
                    "condition": kind,
                    "image_rotation_deg": degrees,
                    "repeat": repeat,
                    "frame_file": source["frame_file"],
                    "frame_hw": list(rgb.shape[:2]),
                    "logged_raw_text": source["raw_text"],
                    "logged_rvq_codes": source["rvq_codes"],
                    "logged_apos_id": source["apos_id"],
                    "raw_text": info.get("raw_text"),
                    "rvq_codes": info.get("rvq_codes"),
                    "cluster_id": info.get("cluster_id"),
                    "action_waypoint": None if waypoint is None else [round(float(v), 5) for v in waypoint],
                    "prompt_tokens": getattr(engine, "_last_prompt_tokens", None),
                    "infer_s": round(elapsed, 4),
                    "matches_logged_text": info.get("raw_text") == source["raw_text"],
                }
                records.append(record)
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(
                    f"{scene}/{episode} {kind} {degrees:+.1f} r{repeat} "
                    f"{record['raw_text']} {elapsed:.3f}s",
                    flush=True,
                )

    by_episode: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        by_episode.setdefault((record["scene"], record["episode_id"]), []).append(record)

    episode_rows = []
    for key, items in by_episode.items():
        repeats = [item for item in items if item["condition"] == "repeat"]
        rotations = [item for item in items if item["condition"] == "image_rotate"]
        base = repeats[0]
        base_text = base["raw_text"]
        base_wp = base["action_waypoint"] or [None, None, None]

        def yaw_of(item: dict) -> float | None:
            waypoint = item["action_waypoint"]
            return None if not waypoint else float(waypoint[2])

        def changed(item: dict) -> dict:
            waypoint = item["action_waypoint"] or [None, None, None]
            return {
                "raw_text_changed": item["raw_text"] != base_text,
                "rvq_changed": item["rvq_codes"] != base["rvq_codes"],
                "l0_changed": (item["rvq_codes"] or [None])[0] != (base["rvq_codes"] or [None])[0],
                "yaw_sign_changed": (
                    yaw_of(item) is not None
                    and base_wp[2] is not None
                    and abs(yaw_of(item)) > 0.05
                    and abs(base_wp[2]) > 0.05
                    and (yaw_of(item) > 0) != (base_wp[2] > 0)
                ),
                "yaw_delta_rad": None if yaw_of(item) is None or base_wp[2] is None else round(yaw_of(item) - base_wp[2], 5),
                "forward_delta_m": None if waypoint[0] is None or base_wp[0] is None else round(float(waypoint[0]) - float(base_wp[0]), 5),
            }

        episode_rows.append(
            {
                "scene": key[0],
                "episode_id": key[1],
                "failure_type": items[0]["failure_type"],
                "repeat_unique_texts": len({item["raw_text"] for item in repeats}),
                "repeat_matches_logged": sum(item["matches_logged_text"] for item in repeats),
                "rotations_text_changed": sum(changed(item)["raw_text_changed"] for item in rotations),
                "rotations_l0_changed": sum(changed(item)["l0_changed"] for item in rotations),
                "rotations_yaw_sign_changed": sum(changed(item)["yaw_sign_changed"] for item in rotations),
                "rotation_details": [
                    {"deg": item["image_rotation_deg"], **changed(item), "raw_text": item["raw_text"]}
                    for item in rotations
                ],
            }
        )

    summary = {
        "n_episodes": len(episode_rows),
        "n_inferences": len(records),
        "engine_load_s": round(load_s, 1),
        "sampling": sampling,
        "perturbation": "image-plane rotation about the JPEG center, cv2 INTER_LINEAR, BORDER_REPLICATE; not a Habitat camera extrinsic",
        "repeat_episodes_with_any_text_change": sum(row["repeat_unique_texts"] > 1 for row in episode_rows),
        "repeat_matches_logged_count": sum(row["repeat_matches_logged"] for row in episode_rows),
        "repeat_possible_matches": 3 * len(episode_rows),
        "rotation_text_changes": sum(row["rotations_text_changed"] for row in episode_rows),
        "rotation_trials": 4 * len(episode_rows),
        "rotation_l0_changes": sum(row["rotations_l0_changed"] for row in episode_rows),
        "rotation_yaw_sign_changes": sum(row["rotations_yaw_sign_changed"] for row in episode_rows),
        "mean_infer_s": round(sum(item["infer_s"] for item in records) / len(records), 4),
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1),
        "episodes": episode_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", summary_path, flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
