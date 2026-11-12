#!/usr/bin/env python3
"""Record the step just before a sustained geodesic rise, and how the far waypoint faces.

A step counts as the path still being in front when the uncapped first-sample
heading is within ±90° and the expert label is not stop. The far waypoint is
the 10th integrated ego pose, not a future robot position.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output/r2r_probe/error_screening_full_episode_20261021"


def num(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def relation(path_yaw: float, wp9_yaw: float) -> str:
    if not math.isfinite(path_yaw):
        return "unknown"
    if abs(path_yaw) <= 15 or abs(wp9_yaw) <= 15:
        return "one_side_near_straight"
    if math.copysign(1.0, path_yaw) == math.copysign(1.0, wp9_yaw):
        return "same_side"
    return "opposite_side"


def main() -> None:
    timeline = defaultdict(list)
    with (OUT / "episode_timeline.csv").open() as handle:
        for row in csv.DictReader(handle):
            timeline[(row["scene"], row["episode_id"])].append(row)
    for rows in timeline.values():
        rows.sort(key=lambda row: int(row["step"]))
    episodes = list(csv.DictReader((OUT / "episode_level_screening.csv").open()))
    out_rows = []
    for episode in episodes:
        rows = timeline[(episode["scene"], episode["episode_id"])]
        rise = episode["first_sustained_rise_step"]
        chosen = None
        status = "no_sustained_rise"
        if rise != "":
            status = "no_front_step_before_rise"
            for row in rows:
                if int(row["step"]) >= int(rise):
                    break
                yaw = num(row["uncapped_path_yaw_deg"])
                if row["expert_stop"] != "1" and math.isfinite(yaw) and abs(yaw) <= 90:
                    chosen = row
                    status = "last_front_before_rise"
            if chosen is None:
                chosen = rows[int(rise)]
        else:
            chosen = rows[-1]
        path_yaw = num(chosen["uncapped_path_yaw_deg"])
        wp9 = num(chosen["wp9_yaw_deg"])
        in_fov = None
        fov_status = "no_sustained_rise"
        if rise != "":
            fov_status = "never_in_fov_before_rise"
            for row in rows:
                if int(row["step"]) >= int(rise):
                    break
                yaw = num(row["uncapped_path_yaw_deg"])
                # 120° horizontal field of view: |heading| > 60° falls outside the image.
                if row["expert_stop"] != "1" and math.isfinite(yaw) and abs(yaw) <= 60:
                    in_fov = row
                    fov_status = "last_in_fov_before_rise"
        fov_path = num(in_fov["uncapped_path_yaw_deg"]) if in_fov else float("nan")
        fov_wp9 = num(in_fov["wp9_yaw_deg"]) if in_fov else float("nan")
        out_rows.append(
            {
                "scene": episode["scene"],
                "episode_id": episode["episode_id"],
                "failure_type": episode["failure_type"],
                "success": episode["success"],
                "first_sustained_rise_step": rise,
                "window_status": status,
                "window_step": chosen["step"],
                "steps_until_rise": "" if rise == "" else int(rise) - int(chosen["step"]),
                "geodesic_m": chosen["geodesic_before_m"],
                "uncapped_path_yaw_deg": chosen["uncapped_path_yaw_deg"],
                "wp0_yaw_deg": chosen["wp0_yaw_deg"],
                "wp0_forward_m": chosen["wp0_forward_m"],
                "wp9_yaw_deg": chosen["wp9_yaw_deg"],
                "wp9_forward_m": chosen["wp9_forward_m"],
                "wp9_left_m": chosen["wp9_left_m"],
                "commanded_forward_m": chosen["commanded_forward_m"],
                "commanded_yaw_deg": chosen["commanded_yaw_deg"],
                "far_vs_path": relation(path_yaw, wp9),
                "in_fov_step": "" if in_fov is None else in_fov["step"],
                "in_fov_path_yaw_deg": "" if in_fov is None else in_fov["uncapped_path_yaw_deg"],
                "in_fov_wp9_yaw_deg": "" if in_fov is None else in_fov["wp9_yaw_deg"],
                "in_fov_far_vs_path": fov_status if in_fov is None else relation(fov_path, fov_wp9),
                "apos_state": chosen["apos_state"],
                "opos_state": chosen["opos_state"],
            }
        )
    dest = OUT / "onset_windows.csv"
    with dest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    from collections import Counter

    print(Counter((row["window_status"], row["far_vs_path"], row["success"]) for row in out_rows))


if __name__ == "__main__":
    main()
