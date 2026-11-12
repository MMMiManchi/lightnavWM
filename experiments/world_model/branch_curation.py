"""Offline region labels for recorded exploration, never a runtime node oracle.

The MP3D polygon annotation is independent of the collector's depth-gap rule.
Entering a room-like region is useful dynamics supervision, but does not by
itself prove that either endpoint is a navigation decision node.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VISIBLE_ROLES = frozenset({"fit", "development"})
FORMAT_SOURCE = "https://github.com/niessner/Matterport/blob/master/data_organization.md"


@dataclass(frozen=True)
class CurationRules:
    version: str = "recorded_region_entry/v1"
    stable_frames: int = 5
    target_clearance_m: float = 0.30
    minimum_path_m: float = 1.0
    floor_tolerance_below_m: float = 0.20
    maximum_body_height_m: float = 0.75
    maximum_black_fraction: float = 0.65
    minimum_rgb_std: float = 5.0
    simulator_step_s: float = 0.1

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FloorPolygon:
    region_id: int
    category_code: str
    floor_height: float
    points: tuple[tuple[float, float], ...]


def load_floor_polygons(path: Path) -> list[FloorPolygon]:
    """Read only R/S/V records; P is overloaded for portals and panoramas."""
    regions = {}
    surfaces = {}
    vertices: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "R":
            regions[int(fields[1])] = fields[5]
        elif fields[0] == "S" and fields[4] == "F":
            surfaces[int(fields[1])] = int(fields[2])
        elif fields[0] == "V" and fields[3] == "F":
            vertices[int(fields[2])].append(tuple(map(float, fields[4:7])))
    result = []
    for surface, region in sorted(surfaces.items()):
        points = vertices[surface]
        if len(points) < 3 or region not in regions:
            continue
        result.append(FloorPolygon(
            region, regions[region], sum(p[2] for p in points) / len(points),
            tuple((p[0], p[1]) for p in points),
        ))
    if not result:
        raise ValueError(f"no annotated floor polygons in {path}")
    return result


def boundary_distance(point, polygon) -> float:
    distance = math.inf
    for a, b in zip(polygon, (*polygon[1:], polygon[0])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = dx * dx + dy * dy
        t = min(1.0, max(0.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / norm)) if norm else 0.0
        distance = min(distance, math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy))
    return distance


def point_in_polygon(point, polygon) -> bool:
    """Ray crossing works for concave region polygons; a bounding box does not."""
    x, y = point
    inside = False
    for a, b in zip(polygon, (*polygon[1:], polygon[0])):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    return inside


def assign_region(position, polygons, rules: CurationRules) -> dict[str, Any] | None:
    # Inverse of official_nodes.habitat_position: Habitat (x,y,z) -> MP3D (x,-z,y).
    x, height, negative_y = map(float, position)
    point = (x, -negative_y)
    found: dict[int, dict[str, Any]] = {}
    for polygon in polygons:
        above_floor = height - polygon.floor_height
        if not -rules.floor_tolerance_below_m <= above_floor <= rules.maximum_body_height_m:
            continue
        clearance = boundary_distance(point, polygon.points)
        if clearance <= 1e-8 or point_in_polygon(point, polygon.points):
            candidate = {"region_id": polygon.region_id, "category_code": polygon.category_code,
                         "clearance_m": clearance, "height_above_floor_m": above_floor}
            previous = found.get(polygon.region_id)
            if previous is None or previous["clearance_m"] < clearance:
                found[polygon.region_id] = candidate
    # Overlapping rooms and exact shared boundaries stay unknown.
    return next(iter(found.values())) if len(found) == 1 else None


def region_entry(frames: list[dict], polygons, rules: CurationRules) -> dict[str, Any]:
    regions = [assign_region(frame["position"], polygons, rules) for frame in frames]
    if not regions or regions[0] is None:
        return {"status": "unassigned_source_region", "target_index": None, "regions": regions}
    source_id = regions[0]["region_id"]
    path = 0.0
    first_stable_destination = None
    for i, region in enumerate(regions):
        if i:
            a, b = frames[i - 1]["position"], frames[i]["position"]
            path += math.hypot(b[0] - a[0], b[2] - a[2])
        if i < rules.stable_frames - 1 or region is None or region["region_id"] == source_id:
            continue
        window = regions[i - rules.stable_frames + 1:i + 1]
        if not all(r and r["region_id"] == region["region_id"] for r in window):
            continue
        if first_stable_destination is None:
            first_stable_destination = region["region_id"]
        if first_stable_destination != region["region_id"]:
            # Do not silently skip a narrow, unconfirmed intermediate room.
            return {"status": "uncertain_intermediate_region", "target_index": None, "regions": regions}
        if region["clearance_m"] >= rules.target_clearance_m and path >= rules.minimum_path_m:
            return {"status": "confirmed_region_entry", "target_index": i, "regions": regions,
                    "path_m": path, "horizon_steps": i, "horizon_s": i * rules.simulator_step_s}
    return {"status": "unconfirmed_region_entry", "target_index": None, "regions": regions}


def image_quality(pixels, rules: CurationRules) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(pixels)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        return {"usable": False, "reason": "not_uint8_hwc_rgb"}
    black = float(np.mean(np.max(array, axis=-1) <= 4))
    std = float(np.std(array.astype(np.float32)))
    reasons = []
    if black > rules.maximum_black_fraction:
        reasons.append("mostly_unscanned_black")
    if std < rules.minimum_rgb_std:
        reasons.append("nearly_uniform")
    return {"usable": not reasons, "reason": reasons, "black_fraction": black, "rgb_std": std}


def student_inputs(source: dict, chosen: dict, policy_id: str) -> dict[str, Any]:
    """An explicit whitelist: no pose, room id, future controls, or task route."""
    return {
        "history_rgb": [source], "history_valid": [True],
        "requested_skill": {"name": "follow_visible_exit_until_next_event", "policy_id": policy_id,
                            "bearing_left_rad": float(chosen["bearing_left_rad"]), "side": chosen.get("side")},
    }


def deduplicate_events(events: list[dict]) -> list[dict]:
    """Collapse exact repeated targets, preserving genuinely different outcomes."""
    groups = defaultdict(list)
    for event in events:
        inputs = event["inputs"]
        key = (event["role"], event["origin_id"], event["exit_id"],
               inputs["history_rgb"][-1]["sha256"],
               inputs["requested_skill"]["bearing_left_rad"],
               inputs["requested_skill"]["policy_id"], event["target"]["rgb"]["sha256"])
        groups[key].append(event)
    canonical = []
    families = Counter((e["role"], e["origin_id"], e["exit_id"]) for e in events)
    for group in groups.values():
        first = min(group, key=lambda e: (e["repeat_index"], e["sample_id"]))
        first["duplicate_sample_ids"] = [e["sample_id"] for e in group if e is not first]
        first["repeat_count"] = len(group)
        first["family_weight"] = len(group) / families[(first["role"], first["origin_id"], first["exit_id"])]
        canonical.append(first)
    # Equal total mass per house, then per origin, then per requested branch.
    scene_origins = defaultdict(set)
    origin_exits = defaultdict(set)
    for event in canonical:
        scene_origins[(event["role"], event["scene"])].add(event["origin_id"])
        origin_exits[(event["role"], event["origin_id"])].add(event["exit_id"])
    for event in canonical:
        event["sample_weight"] = event["family_weight"] / (
            len(scene_origins[(event["role"], event["scene"])]) *
            len(origin_exits[(event["role"], event["origin_id"])])
        )
    return sorted(canonical, key=lambda e: e["sample_id"])


def branch_pair_summary(events: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for event in events:
        groups[(event["role"], event["origin_id"])].append(event)
    rows = []
    for (role, origin), group in sorted(groups.items()):
        by_exit = defaultdict(set)
        for event in group:
            by_exit[event["exit_id"]].add(event["offline_labels"]["target_region"]["region_id"])
        if len(by_exit) < 2:
            state = "only_one_confirmed_exit"
        elif any(len(regions) != 1 for regions in by_exit.values()):
            state = "variable_outcomes"
        elif len(set.union(*by_exit.values())) == 1:
            state = "same_confirmed_region"
        else:
            state = "different_confirmed_regions"
        rows.append({"role": role, "origin_id": origin, "state": state,
                     "regions_by_exit": {str(k): sorted(v) for k, v in by_exit.items()},
                     "sample_ids": [e["sample_id"] for e in group],
                     "automatically_use_as_hard_negatives": False})
    return rows


def sample_id(*parts) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:24]
