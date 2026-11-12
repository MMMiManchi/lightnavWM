"""Rank offline scan poses using local geometry, without using graph degree.

These are candidates, not node truth. Actual current RGB must still establish
two distinct architectural exits before a candidate becomes a training node.
"""

from __future__ import annotations

import math

import numpy as np

from experiments.world_model.branch_curation import sample_id
from experiments.world_model.pose_math import wrap_pi


class RegionIndex:
    """Vectorized labels with ambiguity preserved, using the same floor prisms."""

    def __init__(self, polygons):
        self.polygons = polygons

    def assign(self, positions):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        xy = positions[:, [0, 2]].copy()
        xy[:, 1] *= -1
        memberships = {}
        for polygon in self.polygons:
            points = np.array(polygon.points)
            active = ((positions[:, 1] - polygon.floor_height >= -0.2)
                      & (positions[:, 1] - polygon.floor_height <= 0.75)
                      & np.all(xy >= points.min(0) - 1e-8, axis=1)
                      & np.all(xy <= points.max(0) + 1e-8, axis=1))
            indices = np.flatnonzero(active)
            if not len(indices):
                continue
            test = xy[indices]
            inside = np.zeros(len(indices), dtype=bool)
            clearance = np.full(len(indices), np.inf)
            for a, b in zip(points, np.roll(points, -1, axis=0)):
                d = b - a
                norm = float(d @ d)
                t = np.clip(((test - a) @ d) / norm, 0, 1) if norm else np.zeros(len(indices))
                clearance = np.minimum(clearance, np.linalg.norm(test - a - t[:, None] * d, axis=1))
                if b[1] != a[1]:
                    crossings = (a[1] > test[:, 1]) != (b[1] > test[:, 1])
                    inside ^= crossings & (test[:, 0] < d[0] * (test[:, 1] - a[1]) / d[1] + a[0])
            accepted = inside | (clearance <= 1e-8)
            rid = polygon.region_id
            if rid not in memberships:
                memberships[rid] = np.full(len(positions), -1.0)
            memberships[rid][indices[accepted]] = np.maximum(
                memberships[rid][indices[accepted]], clearance[accepted])
        labels = np.full(len(positions), -1, dtype=np.int32)
        distances = np.full(len(positions), -1.0)
        count = np.zeros(len(positions), dtype=np.int32)
        for rid, values in memberships.items():
            active = values >= 0
            labels[active], distances[active] = rid, values[active]
            count[active] += 1
        labels[count != 1] = -1
        distances[count != 1] = -1
        return labels, distances


def first_ray_region(labels, clearances, source_region):
    first = next((int(r) for r in labels if r >= 0 and r != source_region), None)
    if first is None:
        return None
    for i in range(1, len(labels)):
        if labels[i - 1] == labels[i] == first and max(clearances[i - 1:i + 1]) >= 0.30:
            return first
        if labels[i] >= 0 and labels[i] not in {first, source_region}:
            return None
    return None


def propose_pose(finder, index, position, viewpoint_id, scene, role):
    start = np.asarray(position, dtype=np.float32)
    source_labels, source_clearance = index.assign([start])
    source = int(source_labels[0])
    if source < 0:
        return None
    angles = np.linspace(-math.pi, math.pi, 36, endpoint=False)
    distances = np.arange(0.5, 4.51, 0.5)
    points, walked = [], []
    for angle in angles:
        direction = np.array([-math.sin(angle), 0, -math.cos(angle)], dtype=np.float32)
        ray_points, ray_walked = [], []
        for distance in distances:
            end = np.asarray(finder.try_step_no_sliding(start, start + direction * distance))
            ray_points.append(end)
            ray_walked.append(float(np.linalg.norm((end - start)[[0, 2]])))
        points.extend(ray_points)
        walked.append(ray_walked)
    labels, clearances = index.assign(points)
    labels, clearances = labels.reshape(36, -1), clearances.reshape(36, -1)
    walked = np.array(walked)
    witnessed = []
    for i, angle in enumerate(angles):
        valid = walked[i] >= distances - 0.12
        if walked[i, 2] < 1.35:
            continue
        ray_labels = labels[i].copy()
        ray_labels[~valid] = -1
        witness = first_ray_region(ray_labels, clearances[i], source)
        if witness is not None:
            first = next(j for j, label in enumerate(ray_labels) if label == witness)
            witnessed.append({"world_yaw": float(angle), "witness_region": witness,
                              "witness_distance_m": float(distances[first]),
                              "local_navmesh_1_5m_walked": float(walked[i, 2])})
    best = None
    for i, a in enumerate(witnessed):
        for b in witnessed[i + 1:]:
            gap = abs(wrap_pi(a["world_yaw"] - b["world_yaw"]))
            if a["witness_region"] == b["witness_region"] or not math.radians(45) <= gap <= math.radians(100):
                continue
            yaw = wrap_pi(a["world_yaw"] + wrap_pi(b["world_yaw"] - a["world_yaw"]) / 2)
            # Require an independently free rear direction as well as two front outcomes.
            rear = [j for j, angle in enumerate(angles) if abs(wrap_pi(angle - yaw)) >= math.radians(110)
                    and walked[j, 4] >= 2.35]
            if not rear:
                continue
            score = a["witness_distance_m"] + b["witness_distance_m"] + 0.5 * float(source_clearance[0])
            if best is None or score < best[0]:
                best = score, yaw, a, b
    if best is None:
        return None
    score, yaw, a, b = best
    exits = []
    for i, ray in enumerate((a, b)):
        bearing = wrap_pi(ray["world_yaw"] - yaw)
        u = (0.5 - math.tan(bearing) / (2 * math.tan(math.radians(60)))) * 1000
        exits.append({"exit_id": i, "bearing_left_rad": bearing, "image_bearing_u_0_1000": u,
                      "side": "left" if bearing > 0 else "right", **ray})
    return {"candidate_id": sample_id(scene, viewpoint_id, round(yaw, 6)), "role": role, "scene": scene,
            "offline_viewpoint_id": viewpoint_id, "position": start.tolist(), "yaw": yaw,
            "source_region": source, "source_clearance_m": float(source_clearance[0]),
            "exits": exits, "ranking_score": score, "node_label": "unreviewed",
            "geometry_witness_is_training_target": False, "runtime_graph_lookup": False}
