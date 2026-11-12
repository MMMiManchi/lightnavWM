"""Rules for a separate recorded-node recollection trial.

The controller is an offline geometry teacher. Region ids and poses are label
or collection metadata; they are never navigation-policy student inputs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from experiments.world_model.branch_curation import CurationRules, region_entry
from experiments.world_model.branch_exit import branch_velocity
from experiments.world_model.pose_math import wrap_pi

POLICY_ID = "reviewed_exit_checkpoint_then_center_follow/v2"


@dataclass(frozen=True)
class RecollectionRules:
    version: str = "reviewed_branch_recollection/v2"
    checkpoint_m: float = 1.5
    checkpoint_tolerance_m: float = 0.18
    maximum_steps: int = 300
    repeats: int = 2
    minimum_exit_separation_degrees: float = 35.0
    source_history_frames: int = 8
    source_history_stride_steps: int = 4
    restoration_position_m: float = 0.02
    restoration_yaw_rad: float = 0.01
    source_rgb_hash_must_match: bool = True
    target_kind: str = "first_confidently_entered_different_annotated_region"

    def record(self):
        return asdict(self)


def exit_checkpoint(position, yaw, bearing, distance=1.5):
    heading = yaw + bearing
    return [position[0] - math.sin(heading) * distance, position[1],
            position[2] - math.cos(heading) * distance]


def horizontal_distance(a, b):
    return math.hypot(a[0] - b[0], a[2] - b[2])


def checkpoint_command(position, yaw, checkpoint):
    dx, dz = checkpoint[0] - position[0], checkpoint[2] - position[2]
    desired = math.atan2(-dx, -dz)
    return branch_velocity(wrap_pi(desired - yaw))


def validate_origin(origin, rules: RecollectionRules):
    if origin.get("role") not in {"fit", "development"}:
        raise ValueError("recollection never opens a sealed origin")
    if origin.get("node_label_quality") != "reviewed_visual_and_local_geometry_silver":
        raise ValueError("origin lacks explicit visual and local-geometry review")
    exits = origin.get("exits", [])
    if len(exits) != 2 or len({e["exit_id"] for e in exits}) != 2:
        raise ValueError("exactly two distinct reviewed exits required")
    gap = abs(wrap_pi(exits[0]["bearing_left_rad"] - exits[1]["bearing_left_rad"]))
    if gap < math.radians(rules.minimum_exit_separation_degrees):
        raise ValueError("reviewed directions overlap")
    if any(e.get("local_navmesh_1_5m_walked", 0) < 1.35 for e in exits):
        raise ValueError("reviewed entrance is locally blocked")
    if not origin.get("expected_rgb_sha256"):
        raise ValueError("missing restored image binding")


def restored(origin, position, yaw, rgb_sha256, rules: RecollectionRules):
    return (
        horizontal_distance(origin["position"], position) <= rules.restoration_position_m
        and abs(origin["position"][1] - position[1]) <= rules.restoration_position_m
        and abs(wrap_pi(origin["yaw"] - yaw)) <= rules.restoration_yaw_rad
        and rgb_sha256 == origin["expected_rgb_sha256"]
    )


def confirmed_event(frames, polygons, checkpoint_reached: bool):
    """No amount of depth gaps, elapsed time, or turning alone accepts a target."""
    entry = region_entry(frames, polygons, CurationRules())
    return entry if checkpoint_reached and entry["status"] == "confirmed_region_entry" else None


def causal_history(frames, maximum=8):
    selected = frames[-maximum:]
    if not selected or selected[-1]["name"] != "node":
        raise ValueError("source history must end at the decision observation")
    indices = [frame["replay_step"] for frame in selected]
    if indices != sorted(set(indices)):
        raise ValueError("source history repeats or reverses time")
    return [{"path": frame["rgb_path"], "sha256": frame["rgb_sha256"]} for frame in selected]
