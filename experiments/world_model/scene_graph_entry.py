"""Lay every recorded episode on one scene map and keep the loops.

A place is a walkable decision region. Two visits share it only when the
floor between them is open, not because the coordinates are close. An exit
that this visit did not walk stays unknown and is not given a destination.
Run with the Habitat 3.9 interpreter. It does not collect or train.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.world_model.decision_node_bev import MAP, _bounds, _mesh, _paint_floor, _projector
from experiments.world_model.event_spans_entry import SPLITS, _poses
from experiments.world_model.key_nodes import goal_stop, room_doors, walking_turns
from experiments.world_model.scene_graph import build_graph, share_region

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output/world_model_diagnostics/scene_graph_20261103_05"
FIT_HOUSE_DIRS = (
    REPO / "output/world_model_data/fit_houses_20261103_01",
    REPO / "output/world_model_data/fit_houses_20261103_02",
)
DATASET = REPO / "data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PANEL = 440
CELL_M = 1.5
PLACE_RADIUS_M = 1.5
FLOOR_GAP_M = 1.2
REVISIT_M = 2.0
COLORS = (
    (36, 84, 168),
    (20, 140, 80),
    (180, 60, 40),
    (120, 70, 160),
    (180, 120, 20),
    (20, 130, 140),
    (80, 80, 80),
    (160, 50, 110),
)
KIND_TEXT = {
    "turned": "拐上了另一条路",
    "entered_room": "走进了另一间房",
    "left_room": "从房间走到了外面",
    "instruction_stop": "到了指令要停的地方",
    "junction": "几条路线在这里分开或汇合",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "scenes").mkdir()
    _write_json(OUT / "protocol.json", _protocol())
    goals = _goals()
    poses_by_split = {split: _poses(root) for split, root in SPLITS.items()}
    for root in FIT_HOUSE_DIRS:
        poses_by_split[root.name] = _fit_house_episodes(root)
    by_scene: dict[str, list[tuple[str, str, list[dict]]]] = defaultdict(list)
    for split, episodes in poses_by_split.items():
        for episode_uid, poses in episodes.items():
            by_scene[episode_uid.split("|")[2]].append((split, episode_uid, poses))
    places_out = []
    occurrences_out = []
    branches_out = []
    edges_out = []
    cycle_count = 0
    for scene, episodes in sorted(by_scene.items()):
        finder = _finder(scene)
        mesh = _mesh(scene)
        assigned = []
        for split, episode_uid, poses in episodes:
            positions = [list(pose["position"]) for pose in poses]
            snapped = []
            for position in positions:
                point = _snap(finder, position)
                snapped.append(point if point is not None else [math.nan, math.nan, math.nan])
            assigned.append((split, episode_uid, poses, positions, snapped))
        built_episodes = []
        decisions = []
        for split, episode_uid, poses, positions, snapped in assigned:
            episode_id = episode_uid.split("|")[3]
            goal, _instruction = goals.get((scene, episode_id), (None, ""))
            clearances = [_clearance(finder, item) for item in snapped]
            decision = _decision_indexes(positions, clearances, goal)
            decisions.append(decision)
            built_episodes.append(
                {
                    "episode_uid": episode_uid,
                    "split": split,
                    "observation_ids": [pose["observation_id"] for pose in poses],
                    "positions": positions,
                    "snapped": snapped,
                }
            )
        clusters = _places(scene, finder, built_episodes, decisions)
        _assign(finder, clusters, built_episodes, decisions)
        clusters = _merge_revisits(finder, clusters, built_episodes)
        _assign(finder, clusters, built_episodes, decisions)
        clusters = _collapse_splits(finder, clusters, built_episodes)
        kinds_by_region = {cluster["region"]: cluster["kinds"] for cluster in clusters}
        exits = _exits(finder, built_episodes, set(kinds_by_region))
        graph = build_graph(built_episodes, exits, dict(kinds_by_region))
        place_ids = {
            place["region"]: f"{scene}:{index:03d}"
            for index, place in enumerate(graph["places"], start=1)
        }
        for place in graph["places"]:
            places_out.append(
                {
                    "place_id": place_ids[place["region"]],
                    "scene": scene,
                    "region": place["region"],
                    "kinds": place["kinds"],
                    "kinds_zh": [KIND_TEXT.get(kind, kind) for kind in place["kinds"]],
                }
            )
        split_of = {episode["episode_uid"]: episode["split"] for episode in built_episodes}
        for occurrence in graph["occurrences"]:
            occurrences_out.append(
                {
                    "occurrence_id": f"{place_ids[occurrence['place_region']]}|{occurrence['occurrence_id']}",
                    "place_id": place_ids[occurrence["place_region"]],
                    "scene": scene,
                    "split": split_of[occurrence["episode_uid"]],
                    "episode_uid": occurrence["episode_uid"],
                    "index": occurrence["index"],
                    "observation_id": occurrence["observation_id"],
                    "approach_bearing_rad": occurrence["approach_bearing_rad"],
                    "position": occurrence["position"],
                }
            )
        for edge in graph["edges"]:
            edges_out.append(
                {
                    "scene": scene,
                    "split": split_of[edge["episode_uid"]],
                    "episode_uid": edge["episode_uid"],
                    "from_place_id": place_ids[edge["from_region"]],
                    "to_place_id": place_ids[edge["to_region"]],
                    "from_occurrence_id": f"{place_ids[edge['from_region']]}|{edge['from_occurrence_id']}",
                    "observation_ids": edge["observation_ids"],
                }
            )
        for branch in graph["branches"]:
            branches_out.append(
                {
                    "scene": scene,
                    "place_id": place_ids[branch["place_region"]],
                    "occurrence_id": f"{place_ids[branch['place_region']]}|{branch['occurrence_id']}",
                    "exit_id": branch["exit_id"],
                    "bearing_rad": branch["bearing_rad"],
                    "walked": branch["walked"],
                    "next_place_id": None
                    if branch["next_place_region"] is None
                    else place_ids[branch["next_place_region"]],
                    "outcome": branch["outcome"],
                }
            )
        cycle_count += len(graph["cycles"])
        _draw(
            scene,
            mesh,
            built_episodes,
            graph,
            place_ids,
            {cluster["region"]: cluster["rep"] for cluster in clusters},
        )
    _write_jsonl(OUT / "places.jsonl", places_out)
    _write_jsonl(OUT / "occurrences.jsonl", occurrences_out)
    _write_jsonl(OUT / "edges.jsonl", edges_out)
    _write_jsonl(OUT / "branches.jsonl", branches_out)
    summary = _summary(places_out, occurrences_out, edges_out, branches_out, cycle_count)
    _write_json(OUT / "results.json", summary)
    (OUT / "report.md").write_text(_report(summary), encoding="utf-8")
    print(
        f"places {summary['places']} occurrences {summary['occurrences']} "
        f"walked exits {summary['walked_exits']} unknown exits {summary['unknown_exits']} "
        f"cycles {summary['cycles']}"
    )


def _protocol() -> dict:
    return {
        "place": "a walkable decision region shared across episodes only when the floor between the visits is open",
        "coordinate_closeness_is_not_enough": True,
        "occurrence": "one arrival of one episode, with its approach direction and the view at that time",
        "branch": "candidate exits at the place; only the exit this visit walked gets the next place",
        "unwalked_exit": "unknown, no invented destination",
        "cycles_kept": True,
        "place_radius_m": PLACE_RADIUS_M,
        "same_corridor_both_ways_is_not_a_junction": True,
        "a_bend_in_one_corridor_is_not_several_places": True,
        "corridor_points_stay_on_edges": True,
        "dev_used_for_training": False,
        "collection_started": False,
        "training_started": False,
    }


def _places(scene: str, finder, episodes, decisions) -> list[dict]:
    """One place per decision area. A hallway walked in both directions stays an edge."""
    cell_points: dict[tuple, list[list[float]]] = defaultdict(list)
    cell_bearings: dict[tuple, list[float]] = defaultdict(list)
    for episode in episodes:
        for key, point, bearing in _cell_visits(episode):
            cell_points[key].append(point)
            cell_bearings[key].append(bearing)
    forks: list[tuple[list[float], list[str]]] = []
    events: list[tuple[list[float], list[str]]] = []
    for key, bearings in cell_bearings.items():
        if _fork(bearings):
            forks.append((cell_points[key][0], ["junction"]))
    for episode, decision in zip(episodes, decisions):
        for index, kinds in decision.items():
            point = episode["snapped"][index]
            if _finite(point):
                events.append((point, list(kinds)))
    # The meeting of routes owns this patch of floor. A stop beside it joins
    # the meeting instead of pulling the meeting onto one arm.
    forks.sort(key=lambda item: (item[0][1], item[0][0], item[0][2]))
    events.sort(key=lambda item: (item[0][1], item[0][0], item[0][2]))
    seeds = forks + events
    clusters: list[dict] = []
    for point, kinds in seeds:
        cluster = _nearest_place(finder, clusters, point)
        if cluster is None:
            clusters.append(
                {
                    "region": f"{scene}-p{len(clusters) + 1:03d}",
                    "rep": point,
                    "kinds": list(kinds),
                }
            )
        else:
            for kind in kinds:
                if kind not in cluster["kinds"]:
                    cluster["kinds"].append(kind)
    return clusters


def _assign(finder, clusters, episodes, decisions) -> None:
    for episode, decision in zip(episodes, decisions):
        region_ids = []
        current = None
        for index, point in enumerate(episode["snapped"]):
            if current is not None and _reach(finder, point, current["rep"]) is not None:
                cluster = current
            else:
                cluster = _nearest_place(finder, clusters, point)
                current = cluster
            if cluster is None:
                region_ids.append(f"{episode['episode_uid']}-c-{index}")
            else:
                region_ids.append(cluster["region"])
                for kind in decision.get(index, []):
                    if kind not in cluster["kinds"]:
                        cluster["kinds"].append(kind)
        episode["regions"] = region_ids


def _merge_revisits(finder, clusters, episodes) -> list[dict]:
    """One route pacing the same short stretch is one place, not two."""
    by_region = {cluster["region"]: cluster for cluster in clusters}
    parent = {cluster["region"]: cluster["region"] for cluster in clusters}

    def find(region: str) -> str:
        while parent[region] != region:
            parent[region] = parent[parent[region]]
            region = parent[region]
        return region

    for episode in episodes:
        sequence = []
        for region in episode["regions"]:
            if region not in by_region:
                continue
            root = find(region)
            if not sequence or sequence[-1] != root:
                sequence.append(root)
        for index in range(len(sequence) - 2):
            left = find(sequence[index])
            right = find(sequence[index + 1])
            if find(sequence[index + 2]) != left or left == right:
                continue
            if not _within(finder, by_region[left]["rep"], by_region[right]["rep"], REVISIT_M):
                continue
            for kind in by_region[right]["kinds"]:
                if kind not in by_region[left]["kinds"]:
                    by_region[left]["kinds"].append(kind)
            if "junction" in by_region[right]["kinds"]:
                by_region[left]["rep"] = by_region[right]["rep"]
            parent[right] = left
    return [cluster for cluster in clusters if find(cluster["region"]) == cluster["region"]]


def _collapse_splits(finder, clusters, episodes) -> list[dict]:
    """A crossing split into two adjacent dots becomes the one meeting point."""
    by_region = {cluster["region"]: cluster for cluster in clusters}
    dropped = set()
    successors: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        sequence = []
        for region in episode["regions"]:
            if region not in by_region:
                continue
            if not sequence or sequence[-1] != region:
                sequence.append(region)
        for left, right in zip(sequence, sequence[1:]):
            if left != right:
                successors[left].add(right)
    for region, nxt in list(successors.items()):
        others = {item for item in nxt if item != region and item not in dropped}
        if len(others) != 1 or region in dropped:
            continue
        other = next(iter(others))
        if other not in by_region or other in dropped:
            continue
        if not _within(finder, by_region[region]["rep"], by_region[other]["rep"], REVISIT_M):
            continue
        kept = by_region[region]
        absorbed = by_region[other]
        for kind in absorbed["kinds"]:
            if kind not in kept["kinds"]:
                kept["kinds"].append(kind)
        kept["rep"] = [
            (kept["rep"][0] + absorbed["rep"][0]) / 2.0,
            (kept["rep"][1] + absorbed["rep"][1]) / 2.0,
            (kept["rep"][2] + absorbed["rep"][2]) / 2.0,
        ]
        dropped.add(other)
        for episode in episodes:
            episode["regions"] = [region if item == other else item for item in episode["regions"]]
    return [cluster for cluster in clusters if cluster["region"] not in dropped]


def _within(finder, left: list[float], right: list[float], limit: float) -> bool:
    straight = math.hypot(left[0] - right[0], left[2] - right[2])
    if straight > limit or abs(left[1] - right[1]) > 0.8:
        return False
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(left, dtype=np.float32)
    path.requested_end = np.asarray(right, dtype=np.float32)
    if not finder.find_path(path):
        return False
    geodesic = float(path.geodesic_distance)
    return share_region(
        same_floor=True,
        same_island=True,
        segment_on_navmesh=math.isfinite(geodesic) and geodesic <= straight * 1.5 + 0.4,
    )


def _nearest_place(finder, clusters: list[dict], point: list[float] | None) -> dict | None:
    if not _finite(point):
        return None
    best = None
    best_distance = PLACE_RADIUS_M
    for cluster in clusters:
        distance = _reach(finder, point, cluster["rep"])
        if distance is not None and distance <= best_distance:
            best = cluster
            best_distance = distance
    return best


def _cell_visits(episode: dict) -> list[tuple[tuple, list[float], float]]:
    """One departure direction each time a route passes through a cell.

    Sampling every step turns a bend into several directions. A pass has one.
    """
    snapped = episode["snapped"]
    positions = episode["positions"]
    visits = []
    key = None
    start = None
    for index in range(len(snapped) + 1):
        point = snapped[index] if index < len(snapped) else None
        this = _bucket(point) if _finite(point) else None
        if this == key:
            continue
        if key is not None and start is not None:
            bearing = _forward_bearing(positions, start, 0.8)
            if bearing is not None:
                visits.append((key, snapped[start], bearing))
        key = this
        start = index if this is not None else None
    return visits


def _fork(bearings: list[float]) -> bool:
    """A real split. Walking the same corridor back the other way is not one."""
    for left in range(len(bearings)):
        for right in range(left + 1, len(bearings)):
            delta = abs((bearings[left] - bearings[right] + math.pi) % (2.0 * math.pi) - math.pi)
            if math.radians(45.0) <= delta <= math.radians(135.0):
                return True
    return False


def _forward_bearing(positions: list[list[float]], index: int, distance_m: float) -> float | None:
    travelled = 0.0
    for nxt in range(index + 1, len(positions)):
        travelled += math.hypot(
            positions[nxt][0] - positions[nxt - 1][0],
            positions[nxt][2] - positions[nxt - 1][2],
        )
        if travelled >= distance_m:
            return math.atan2(positions[nxt][0] - positions[index][0], positions[nxt][2] - positions[index][2])
    return None


def _finite(point: list[float] | None) -> bool:
    return point is not None and all(math.isfinite(value) for value in point)


def _reach(finder, left: list[float], right: list[float]) -> float | None:
    straight = math.hypot(left[0] - right[0], left[2] - right[2])
    if straight > PLACE_RADIUS_M or abs(left[1] - right[1]) > 0.8:
        return None
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(left, dtype=np.float32)
    path.requested_end = np.asarray(right, dtype=np.float32)
    if not finder.find_path(path):
        return None
    geodesic = float(path.geodesic_distance)
    if not share_region(
        same_floor=True,
        same_island=True,
        segment_on_navmesh=math.isfinite(geodesic) and geodesic <= straight * 1.5 + 0.4,
    ):
        return None
    return geodesic


def _open_between(finder, left: list[float], right: list[float]) -> bool:
    if abs(left[1] - right[1]) > 0.8:
        return False
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(left, dtype=np.float32)
    path.requested_end = np.asarray(right, dtype=np.float32)
    if not finder.find_path(path):
        return share_region(same_floor=True, same_island=True, segment_on_navmesh=False)
    straight = math.hypot(left[0] - right[0], left[2] - right[2])
    geodesic = float(path.geodesic_distance)
    return share_region(
        same_floor=True,
        same_island=True,
        segment_on_navmesh=math.isfinite(geodesic) and geodesic <= straight * 1.5 + 0.4,
    )


def _bucket(point: list[float]) -> tuple:
    return (
        round(point[1] / 0.8),
        round(point[0] / CELL_M),
        round(point[2] / CELL_M),
    )


def _decision_indexes(positions, clearances, goal) -> dict[int, list[str]]:
    found: dict[int, list[str]] = defaultdict(list)
    for index in walking_turns(positions):
        found[index].append("turned")
    for door in room_doors(positions, clearances):
        found[door["index"]].append(door["kind"])
    stop = goal_stop(positions, goal)
    if stop is not None:
        found[stop].append("instruction_stop")
    return found


def _exits(finder, episodes, decision_regions: set[str]) -> dict[str, list[dict]]:
    centers: dict[str, list[float]] = {}
    successors: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        sequence = []
        for region, position in zip(episode["regions"], episode["positions"]):
            centers.setdefault(region, position)
            if not sequence or sequence[-1] != region:
                sequence.append(region)
        for left, right in zip(sequence, sequence[1:]):
            successors[left].add(right)
    wanted = set(decision_regions)
    for region, nxt in successors.items():
        if len(nxt) >= 2:
            wanted.add(region)
    exits = {}
    for region in wanted:
        origin = centers.get(region)
        if origin is None:
            continue
        sectors = _sectors(finder, origin)
        if sectors:
            exits[region] = sectors
    return exits


def _sectors(finder, origin: list[float]) -> list[dict]:
    open_angles = []
    for index in range(16):
        angle = -math.pi + index * (2.0 * math.pi / 16.0)
        target = [
            origin[0] + math.sin(angle) * 1.2,
            origin[1],
            origin[2] + math.cos(angle) * 1.2,
        ]
        snapped = _snap(finder, target)
        if snapped is not None and _open_between(finder, origin, snapped):
            open_angles.append(angle)
    if not open_angles:
        return []
    groups = [[open_angles[0]]]
    for angle in open_angles[1:]:
        if angle - groups[-1][-1] <= (2.0 * math.pi / 16.0) + 0.01:
            groups[-1].append(angle)
        else:
            groups.append([angle])
    if len(groups) > 1 and open_angles[0] + 2.0 * math.pi - open_angles[-1] <= (2.0 * math.pi / 16.0) + 0.05:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    sectors = []
    for index, group in enumerate(groups, start=1):
        sectors.append({"exit_id": f"e{index}", "bearing_rad": sum(group) / len(group)})
    return sectors


def _fit_house_episodes(root: Path) -> dict[str, list[dict]]:
    """Routes from the new-house walks. Each exit is its own pass through the junction."""
    houses = root / "houses"
    if not houses.is_dir():
        return {}
    episodes: dict[str, list[dict]] = {}
    for house in sorted(path for path in houses.iterdir() if path.is_dir()):
        scene, episode_id = house.name.rsplit("_", 1)
        search = _search_positions(house)
        exits = sorted(path for path in house.glob("exit_*") if path.is_dir())
        passes = []
        for exit_dir in exits:
            observations = exit_dir / "observations.jsonl"
            if not observations.is_file():
                continue
            tail = []
            for line in observations.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                tail.append(row["position"])
            if tail:
                passes.append((exit_dir.name, tail))
        if not passes and len(search) >= 2:
            passes.append(("search", []))
        for exit_name, tail in passes:
            poses = []
            for index, position in enumerate([*search, *tail]):
                poses.append(
                    {
                        "position": tuple(float(value) for value in position),
                        "observation_id": f"{house.name}-{exit_name}-{index}",
                    }
                )
            if len(poses) < 2:
                continue
            uid = f"R2R_VLNCE_v1-3_preprocessed|train|{scene}|{episode_id}|{exit_name}"
            episodes[uid] = poses
    return episodes


def _search_positions(house: Path) -> list[list[float]]:
    path = house / "path.jsonl"
    if path.is_file():
        found = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("phase") == "search":
                found.append([float(value) for value in row["position"]])
        if len(found) >= 2:
            return found
    node = house / "node.json"
    if not node.is_file():
        return []
    position = json.loads(node.read_text(encoding="utf-8"))["position"]
    return [[float(value) for value in position]]


def _floor_anchors(heights: list[float]) -> list[float]:
    """Heights on one ramp stay together. The next storey starts its own map."""
    anchors: list[float] = []
    for height in sorted(heights):
        if not anchors or height - anchors[-1] > FLOOR_GAP_M:
            anchors.append(height)
    return anchors


def _floor_anchor(height: float, anchors: list[float]) -> float:
    chosen = anchors[0]
    for anchor in anchors:
        if anchor <= height + 1e-6:
            chosen = anchor
    return chosen


def _draw(scene, mesh, episodes, graph, place_ids, reps) -> None:
    anchors = _floor_anchors(
        [position[1] for episode in episodes for position in episode["positions"]] or [0.0]
    )
    by_floor: dict[float, dict] = defaultdict(lambda: {"paths": [], "places": []})
    for episode in episodes:
        if not episode["positions"]:
            continue
        grouped: dict[float, list] = defaultdict(list)
        for position in episode["positions"]:
            grouped[_floor_anchor(position[1], anchors)].append(position)
        for floor, positions in grouped.items():
            if len(positions) < 2:
                continue
            by_floor[floor]["paths"].append({**episode, "positions": positions})
    seen = set()
    for occurrence in graph["occurrences"]:
        place_id = place_ids[occurrence["place_region"]]
        position = reps.get(occurrence["place_region"], occurrence["position"])
        floor = _floor_anchor(position[1], anchors)
        if (floor, place_id) in seen:
            continue
        seen.add((floor, place_id))
        by_floor[floor]["places"].append((place_id, position, occurrence["place_region"]))
    for floor, payload in by_floor.items():
        if not payload["paths"]:
            continue
        cloud = [
            (position[0], position[2])
            for episode in payload["paths"]
            for position in episode["positions"]
        ]
        cloud.extend((position[0], position[2]) for _place, position, _region in payload["places"])
        minx, minz, span = _bounds(cloud)
        pix = _projector(minx, minz, span)
        flat = [{"position": position} for episode in payload["paths"] for position in episode["positions"]]
        image = Image.new("RGB", (MAP + PANEL, MAP), (248, 247, 244))
        _paint_floor(image, mesh, pix, flat, minx, minz, span)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(FONT, 16)
        for index, episode in enumerate(payload["paths"]):
            color = COLORS[index % len(COLORS)]
            points = [pix(position[0], position[2]) for position in episode["positions"]]
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
        label_boxes = []
        for place_id, position, region in payload["places"]:
            u, v = pix(position[0], position[2])
            draw.ellipse([u - 8, v - 8, u + 8, v + 8], fill=(225, 120, 20), outline=(255, 255, 255))
            label = place_id.split(":")[-1]
            text_x, text_y = u + 10, v - 14
            for left, top, right, bottom in label_boxes:
                if text_x < right and text_x + 36 > left and text_y < bottom and text_y + 18 > top:
                    text_y = bottom + 2
            label_boxes.append((text_x, text_y, text_x + 36, text_y + 18))
            draw.text((text_x, text_y), label, fill=(20, 20, 20), font=font)
            drawn_bearings = set()
            for branch in graph["branches"]:
                if branch["place_region"] != region or branch["outcome"] != "unknown":
                    continue
                if branch["bearing_rad"] is None:
                    continue
                bearing_key = round(float(branch["bearing_rad"]), 2)
                if bearing_key in drawn_bearings:
                    continue
                drawn_bearings.add(bearing_key)
                end = [
                    position[0] + math.sin(branch["bearing_rad"]) * 1.3,
                    position[2] + math.cos(branch["bearing_rad"]) * 1.3,
                ]
                draw.line([pix(position[0], position[2]), pix(end[0], end[1])], fill=(140, 140, 140), width=2)
        image.save(OUT / "scenes" / f"{scene}_y{floor:.1f}.png")


def _summary(places, occurrences, edges, branches, cycles) -> dict:
    return {
        "places": len(places),
        "occurrences": len(occurrences),
        "edges": len(edges),
        "walked_exits": sum(1 for row in branches if row["outcome"] == "walked"),
        "unknown_exits": sum(1 for row in branches if row["outcome"] == "unknown"),
        "cycles": cycles,
        "scenes": len({row["scene"] for row in places}),
        "dev_used_for_training": False,
        "training_started": False,
    }


def _report(summary: dict) -> str:
    lines = [
        "这张图把同一栋房子里的各条路线叠在一起。",
        "地点是路上要做选择的区域：几条路在这里分开或汇合、拐上另一条路、进出房间，或者走到指令要停的地方。",
        "两次经过要能沿地面走通、并且落在同一个选择区域里，才共用一个地点。只是坐标靠近、中间隔着墙，不算同一处。同一条走廊上来回走，或者走廊只是弯着走，都不把这一段切成一串地点。",
        "每一次到达单独记：哪一条路线、走到第几步、从哪个方向来、当时看见哪一帧。",
        "出口只给这次真正走了的填写到达的下一个地点。没走的出口记为未知，不补一个终点。",
        "路线可以绕回旧地点，环路保留。",
        "检查用的路线只参与这张图，没有拿去训练。后采的房子也画在图上，同样还没有批准拿去训练。",
        "",
        f"- 地点 {summary['places']} 个，到达 {summary['occurrences']} 次，走过的连接 {summary['edges']} 条。",
        f"- 记下真实后果的出口 {summary['walked_exits']} 个，未知出口 {summary['unknown_exits']} 个。",
        f"- 绕回旧地点形成的环路 {summary['cycles']} 条。",
        "",
        "图在 scenes 里。彩色线是各条实际走过的路线。橙色点是地点。灰色短线是没走过的出口，短线尽头没有另一个点。",
    ]
    return "\n".join(lines) + "\n"


def _goals() -> dict[tuple[str, str], tuple[list[float] | None, str]]:
    payload = json.loads(gzip.decompress(DATASET.read_bytes()).decode("utf-8"))
    found = {}
    for episode in payload["episodes"]:
        scene = Path(str(episode["scene_id"])).parent.name
        goal = None
        goals = episode.get("goals") or []
        if goals:
            position = goals[0].get("position") if isinstance(goals[0], dict) else goals[0]
            if position is not None:
                goal = [float(value) for value in position]
        instruction = episode.get("instruction") or {}
        text = str(instruction.get("instruction_text") or "") if isinstance(instruction, dict) else ""
        found[(scene, str(episode["episode_id"]))] = (goal, text)
    return found


def _finder(scene: str):
    path = REPO / "data/scene_datasets/mp3d" / scene / f"{scene}.navmesh"
    finder = habitat_sim.PathFinder()
    if not finder.load_nav_mesh(str(path)):
        raise RuntimeError(f"cannot load {path}")
    return finder


def _snap(finder, position: list[float] | None) -> list[float] | None:
    if position is None:
        return None
    snapped = finder.snap_point(np.asarray(position, dtype=np.float32))
    point = [float(snapped[0]), float(snapped[1]), float(snapped[2])]
    if not all(math.isfinite(value) for value in point):
        return None
    if abs(point[1] - position[1]) > 0.8 or math.hypot(point[0] - position[0], point[2] - position[2]) > 1.0:
        return None
    return point


def _clearance(finder, point: list[float] | None) -> float | None:
    if point is None or not all(math.isfinite(value) for value in point):
        return None
    try:
        distance = float(finder.distance_to_closest_obstacle(np.asarray(point, dtype=np.float32), 4.0))
    except TypeError:
        distance = float(finder.distance_to_closest_obstacle(np.asarray(point, dtype=np.float32)))
    if not math.isfinite(distance):
        return None
    return distance


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
