"""Label collected walks with official decision viewpoints.

The connectivity graph is the answer key for training. It is not given to the
robot at runtime. Each frame is labeled with whether the walk has entered a
decision region, which of that region's exits are inside the camera, and
whether the place was already entered. Run the drawing step with the Habitat
3.9 interpreter. This does not collect or train.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.world_model.decision_node_bev import MAP, _bounds, _paint_floor, _projector
from experiments.world_model.ego_warp import depth_to_meters
from experiments.world_model.event_spans_entry import SPLITS, _poses
from experiments.world_model.key_nodes import room_doors
from experiments.world_model.official_nodes import (
    HFOV_DEG,
    MATCH_FLOOR_M,
    annotate_walk,
    clearance_band,
    cone_offset,
    connected_groups,
    decision_places,
    depth_cone_status,
    habitat_position,
    image_column,
    optical_axis_distance,
    rebuild_place,
    teacher_graph,
)
from experiments.world_model.pose_math import wrap_pi

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output/world_model_diagnostics/official_nodes_20261104_04"
PREVIOUS = REPO / "output/world_model_diagnostics/official_nodes_20261104_03"
RADIUS_DIRS = tuple(
    REPO / "output/world_model_data" / f"formal_fit_radius010_20261029_0{index}" for index in range(1, 7)
)
CONNECTIVITY = REPO / "data/connectivity"
FIT_HOUSE_DIRS = (
    REPO / "output/world_model_data/fit_houses_20261103_01",
    REPO / "output/world_model_data/fit_houses_20261103_02",
)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PANEL = 440
FLOOR_GAP_M = 1.2
NEAR_PATH_M = 6.0
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "scenes").mkdir()
    places_by_scene, teacher_rows, graph_stats, scene_graphs = _load_places()
    walks = _walks()
    places_out = []
    frames_out = []
    entries_out = []
    known: dict[str, set[str]] = defaultdict(set)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for walk in walks:
        grouped[walk["scene"]].append(walk)
    for scene, scene_walks in sorted(grouped.items()):
        finder = _finder(scene)
        places, rejected = _navmesh_places(finder, places_by_scene.get(scene, []), scene_graphs.get(scene))
        reachable = _reachability(finder)
        print(f"{scene} reachable {len(places)} unreachable {len(rejected)} walks {len(scene_walks)}", flush=True)
        for place in [*places, *rejected]:
            places_out.append({"scene": scene, "place_id": place["place_id"], **_place_record(place)})
        for walk in scene_walks:
            rows, entered = annotate_walk(walk["frames"], places, known[scene], HFOV_DEG, reachable=reachable)
            _add_clearance(finder, rows)
            _add_depth(rows, places)
            known[scene].update(entered)
            for row in rows:
                frames_out.append(
                    {
                        "scene": scene,
                        "split": walk["split"],
                        "episode_uid": walk["episode_uid"],
                        "dev_excluded": walk["dev_excluded"],
                        "geometry_family": walk.get("geometry_family", "go2w"),
                        "training_use": "not_approved",
                        **row,
                    }
                )
                if row["entered_decision_region"]:
                    entries_out.append(
                        {
                            "scene": scene,
                            "split": walk["split"],
                            "episode_uid": walk["episode_uid"],
                            "dev_excluded": walk["dev_excluded"],
                            "geometry_family": walk.get("geometry_family", "go2w"),
                            "training_use": "not_approved",
                            "occurrence_id": row["occurrence_id"],
                            "source_observation_id": row.get("source_observation_id"),
                            "index": row["index"],
                            "has_rgb": bool(row.get("has_rgb")),
                            "rgb_path": row.get("rgb_path"),
                            "rgb_sha256": row.get("rgb_sha256"),
                            "place_id": row["place_id"],
                            "viewpoint_id": row["viewpoint_id"],
                            "revisit": row["revisit"],
                            "known_place": row["known_place"],
                            "exit_ids": row["exit_ids"],
                            "phase": row["phase"],
                            "angle_candidate_exit_ids": row["angle_candidate_exit_ids"],
                            "depth_exit_status": row.get("depth_exit_status", []),
                            "clearance_band": row.get("clearance_band"),
                            "clearance_event": row.get("clearance_event"),
                            "visibility": "unknown",
                        }
                    )
    visual_frames = [row for row in frames_out if row.get("has_rgb")]
    visual_entries = [row for row in entries_out if row.get("has_rgb")]
    pose_entries = [row for row in entries_out if not row.get("has_rgb")]
    _write_json(OUT / "protocol.json", _protocol(len(places_by_scene), graph_stats))
    _write_jsonl(OUT / "places.jsonl", places_out)
    _write_jsonl(OUT / "frames.jsonl", frames_out)
    _write_jsonl(OUT / "visual_frames.jsonl", visual_frames)
    _write_jsonl(OUT / "entries.jsonl", entries_out)
    _write_jsonl(OUT / "visual_entries.jsonl", visual_entries)
    _write_jsonl(OUT / "pose_only_entries.jsonl", pose_entries)
    _write_jsonl(OUT / "teacher_edges.jsonl", teacher_rows)
    summary = _summary(places_out, frames_out, entries_out, graph_stats)
    _write_json(OUT / "results.json", summary)
    (OUT / "report.md").write_text(_report(summary), encoding="utf-8")
    drawn = defaultdict(list)
    for row in places_out:
        if row.get("navmesh_reachable") is not False:
            drawn[row["scene"]].append(row)
    _draw_all(grouped, drawn, entries_out)
    print(
        f"places {summary['decision_places']} visual entries {summary['visual_entries_fit']} "
        f"unique images {summary['visual_entry_images_fit']} pose-only entries {summary['pose_only_entries_fit']} "
        f"cycle rank {summary['cycle_rank']}"
    )


def _protocol(scenes_with_graph: int, graph_stats: dict) -> dict:
    return {
        "goal": "official viewpoints supervise a discoverer that builds its own map from continuous frames",
        "anchors": "forks, doorways, room boundaries, and clear turns, with geometry and reachability checked; neighbor count alone is not a node",
        "discoverer": "continuous camera frames supervise approach, entry, and departure, with corridor and room-interior negatives",
        "revisit": "the robot creates a node at a new place; a later pass uses vision and history to decide whether it is the same node",
        "runtime": "frames discover a node and its visible directions; world model and memory judge; LightNav executes; the new observation updates the robot's own map",
        "single_frame_classification_is_sufficient": False,
        "must_handle": [
            "incomplete view",
            "repeated triggers",
            "revisit from another heading",
            "adjacent places kept separate",
            "look again when the evidence is weak",
        ],
        "acceptance": "in houses held out of training, with no official id and no adjacency, decision regions are still found",
        "connectivity_graph_is_runtime_input": False,
        "neighbor_list_is_runtime_input": False,
        "official_id_is_runtime_input": False,
        "camera_hfov_deg": HFOV_DEG,
        "approach_radius_m": 3.0,
        "enter_radius_m": 1.5,
        "leave_radius_m": 2.0,
        "scenes_with_connectivity": scenes_with_graph,
        "dev_used_for_training": False,
        "training_started": False,
        "collection_started": False,
        "source_records_rewritten": False,
        "previous_version_not_for_training": str(PREVIOUS),
        "spanning_tree": False,
        "loops_kept": True,
        "cycle_rank": graph_stats["cycle_rank"],
        "angle_candidate_is_visibility_truth": False,
        "counts_authorize_training": False,
        "current_labels_cover": [
            "fork and sharp-turn anchors on a loop-preserving local graph",
            "navmesh snap; places that cannot be reached are not entry targets",
            "approach, enter, inside, leave, and started_inside phases",
            "clearance bands and doorway crossings on the walked path",
            "depth proxy in meters, sampled with a pinhole column and compared on the optical axis",
            "radius010 walks tagged separately from the go2w fit split",
            "a frame is inside a place only when the navmesh path to that place stays short",
            "split places keep only their own members, representative, and rebuilt exits",
        ],
        "depth_not_blocked_is_verified_visibility": False,
        "current_labels_missing": [
            "a doorway or room boundary that is not just a clearance change",
            "a node created by the robot rather than matched to an official id",
            "repeated-trigger suppression, a second heading, and abstain",
        ],
    }


def _load_places() -> tuple[dict[str, list[dict]], list[dict], dict, dict]:
    found = {}
    graphs = {}
    teacher_rows = []
    full_edges = 0
    local_edges = 0
    cycle_rank = 0
    for path in sorted(CONNECTIVITY.glob("*_connectivity.json")):
        scene = path.name.replace("_connectivity.json", "")
        nodes, edges = _read_graph(path)
        graph = teacher_graph(nodes, edges)
        places = decision_places(nodes, edges)
        for place in places:
            place["place_id"] = f"{scene}:{place['viewpoint_id']}"
        found[scene] = places
        graphs[scene] = {
            "positions": graph["positions"],
            "adjacency": graph["adjacency"],
            "branch_nodes": {node for node, neighbors in graph["adjacency"].items() if len(neighbors) >= 3},
        }
        local = set(graph["local_edge_list"])
        full_edges += graph["full_edges"]
        local_edges += graph["local_edges"]
        cycle_rank += graph["cycle_rank"]
        for left, right in graph["full_edge_list"]:
            teacher_rows.append(
                {
                    "scene": scene,
                    "left": left,
                    "right": right,
                    "in_local_graph": (left, right) in local or (right, left) in local,
                }
            )
    return found, teacher_rows, {
        "full_edges": full_edges,
        "local_edges": local_edges,
        "cycle_rank": cycle_rank,
    }, graphs


def _read_graph(path: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    index_to_id = {}
    nodes = []
    for index, item in enumerate(payload):
        if not item.get("included"):
            continue
        index_to_id[index] = item["image_id"]
        nodes.append({"id": item["image_id"], "position": habitat_position(item["pose"], item["height"])})
    edges = []
    for index, item in enumerate(payload):
        if index not in index_to_id:
            continue
        for other, flag in enumerate(item["unobstructed"]):
            if other <= index or not flag or other not in index_to_id:
                continue
            edges.append((index_to_id[index], index_to_id[other]))
    return nodes, edges


def _walks() -> list[dict]:
    walks = []
    for split, root in SPLITS.items():
        dev = split.startswith("formal_dev")
        images = _observation_images(root)
        for episode_uid, poses in _poses(root).items():
            scene = episode_uid.split("|")[2]
            frames = []
            for pose in poses:
                image = images.get(pose["observation_id"], {})
                frames.append(
                    _recorded_frame(
                        pose["position"],
                        pose["yaw"],
                        source_observation_id=pose["observation_id"],
                        source_kind="formal_rgb",
                        episode_uid=episode_uid,
                        image=image,
                    )
                )
            walks.append(_walk(split, scene, episode_uid, frames, dev))
    for root in FIT_HOUSE_DIRS:
        walks.extend(_fit_walks(root))
    for root in RADIUS_DIRS:
        if not (root / "transitions.jsonl").is_file():
            continue
        images = _observation_images(root)
        for episode_uid, poses in _poses(root).items():
            scene = episode_uid.split("|")[2]
            frames = [
                _recorded_frame(
                    pose["position"],
                    pose["yaw"],
                    source_observation_id=pose["observation_id"],
                    source_kind="radius010_rgb",
                    episode_uid=episode_uid,
                    image=images.get(pose["observation_id"], {}),
                )
                for pose in poses
            ]
            walk = _walk(root.name, scene, episode_uid, frames, False)
            walk["body_radius_m"] = 0.10
            walk["geometry_family"] = "radius010"
            walks.append(walk)
    return walks


def _fit_walks(root: Path) -> list[dict]:
    houses = root / "houses"
    if not houses.is_dir():
        return []
    walks = []
    for house in sorted(path for path in houses.iterdir() if path.is_dir()):
        scene, episode_id = house.name.rsplit("_", 1)
        search = _read_rows(house / "path.jsonl")
        search = [row for row in search if row.get("phase") in (None, "search")]
        parent = _read_rows(house / "observations.jsonl")
        passes = []
        for exit_dir in sorted(path for path in house.glob("exit_*") if path.is_dir()):
            tail = _read_rows(exit_dir / "observations.jsonl")
            if tail:
                passes.append((exit_dir.name, _fit_frames(house, exit_dir.name, search, parent, tail)))
        if not passes and search:
            passes.append(("search", _fit_frames(house, "search", search, parent, [])))
        if not passes and parent:
            passes.append(("start", _fit_frames(house, "start", [], parent, [])))
        for name, frames in passes:
            if not frames:
                continue
            uid = f"R2R_VLNCE_v1-3_preprocessed|train|{scene}|{episode_id}|{name}"
            for frame in frames:
                frame["episode_uid"] = uid
            walks.append(_walk(root.name, scene, uid, frames, False))
    return walks


def _fit_frames(house: Path, name: str, search: list[dict], parent: list[dict], tail: list[dict]) -> list[dict]:
    frames = []
    for index, row in enumerate(search):
        image = {}
        source_kind = "search_pose_only"
        source_id = f"{house.name}/path.jsonl#{index}"
        saved_name = "start" if index == 0 else ("node" if index == len(search) - 1 else None)
        if saved_name is not None:
            match = _saved_endpoint(parent, saved_name, row)
            if match is not None:
                image = _image_record(match, house / "assets")
                source_kind = "search_saved_endpoint"
                source_id = f"{house.name}/observations.jsonl#{saved_name}"
        frames.append(
            _recorded_frame(row["position"], row.get("yaw"), source_id, source_kind, "", image)
        )
    if tail:
        for index, row in enumerate(tail):
            frames.append(
                _recorded_frame(
                    row["position"],
                    row.get("yaw"),
                    f"{house.name}/{name}/observations.jsonl#{index}",
                    "branch_rgb",
                    "",
                    _image_record(row, house / name / "assets"),
                )
            )
    elif not search:
        for index, row in enumerate(parent):
            frames.append(
                _recorded_frame(
                    row["position"],
                    row.get("yaw"),
                    f"{house.name}/observations.jsonl#{index}",
                    "house_saved_rgb",
                    "",
                    _image_record(row, house / "assets"),
                )
            )
    return frames


def _saved_endpoint(parent: list[dict], name: str, row: dict) -> dict | None:
    for item in parent:
        if item.get("name") != name:
            continue
        if max(abs(float(left) - float(right)) for left, right in zip(item["position"], row["position"])) >= 1e-5:
            continue
        if abs(wrap_pi(float(item.get("yaw") or 0.0) - float(row.get("yaw") or 0.0))) >= 1e-5:
            continue
        return item
    return None


def _recorded_frame(position, yaw, source_observation_id: str, source_kind: str, episode_uid: str, image: dict) -> dict:
    has_rgb = bool(image.get("rgb_exists"))
    return {
        "episode_uid": episode_uid,
        "source_observation_id": source_observation_id,
        "observation_id": source_observation_id,
        "position": [float(value) for value in position],
        "yaw": None if yaw is None else float(yaw),
        "source_kind": source_kind,
        "has_rgb": has_rgb,
        "rgb_path": image.get("rgb_path") if has_rgb else None,
        "rgb_sha256": image.get("rgb_sha256") if has_rgb else None,
        "depth_path": image.get("depth_path"),
    }


def _observation_images(root: Path) -> dict[str, dict]:
    path = root / "observations.jsonl"
    found = {}
    if not path.is_file():
        return found
    for row in _read_rows(path):
        found[row["observation_id"]] = _image_record(row, root)
    return found


def _image_record(row: dict, base: Path) -> dict:
    rgb = row.get("rgb")
    found = {}
    if isinstance(rgb, dict) and not row.get("rgb_missing"):
        path = base / rgb["path"]
        found = {
            "rgb_path": str(path),
            "rgb_sha256": rgb.get("sha256"),
            "rgb_exists": path.is_file(),
        }
    depth = row.get("depth")
    if isinstance(depth, dict) and not row.get("depth_missing"):
        path = base / depth["path"]
        if path.is_file():
            found["depth_path"] = str(path)
    return found


def _walk(split: str, scene: str, episode_uid: str, frames: list[dict], dev: bool) -> dict:
    return {
        "split": split,
        "scene": scene,
        "episode_uid": episode_uid,
        "frames": frames,
        "dev_excluded": dev,
    }


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _place_record(place: dict) -> dict:
    return {
        "viewpoint_id": place["viewpoint_id"],
        "position": place["position"],
        "kind": place["kind"],
        "member_viewpoint_ids": place["member_viewpoint_ids"],
        "exits": place["exits"],
        "navmesh_reachable": place.get("navmesh_reachable"),
        "split_from_place_id": place.get("split_from_place_id"),
    }


def _finder(scene: str):
    path = REPO / "data/scene_datasets/mp3d" / scene / f"{scene}.navmesh"
    finder = habitat_sim.PathFinder()
    if not path.is_file() or not finder.load_nav_mesh(str(path)):
        return None
    return finder


def _snap(finder, position: list[float]) -> list[float] | None:
    snapped = finder.snap_point(np.asarray(position, dtype=np.float32))
    point = [float(snapped[0]), float(snapped[1]), float(snapped[2])]
    if not all(math.isfinite(value) for value in point):
        return None
    if abs(point[1] - position[1]) > 0.8 or math.hypot(point[0] - position[0], point[2] - position[2]) > 1.0:
        return None
    return point


def _same_region(finder, left: list[float], right: list[float]) -> bool:
    query = habitat_sim.ShortestPath()
    query.requested_start = np.asarray(left, dtype=np.float32)
    query.requested_end = np.asarray(right, dtype=np.float32)
    if not finder.find_path(query) or not math.isfinite(query.geodesic_distance):
        return False
    straight = math.hypot(left[0] - right[0], left[2] - right[2])
    return query.geodesic_distance <= straight * 1.5 + 0.4


def _navmesh_places(finder, places: list[dict], graph: dict | None) -> tuple[list[dict], list[dict]]:
    """Keep places the robot can stand on. A split gets its own members and exits."""
    if finder is None or graph is None:
        return [{**place, "navmesh_reachable": None} for place in places], []
    kept = []
    rejected = []
    for place in places:
        usable = []
        for member_id, member in zip(place["member_viewpoint_ids"], place["member_positions"]):
            snapped = _snap(finder, list(member))
            if snapped is not None:
                usable.append((member_id, snapped))
        if not usable:
            rejected.append({**place, "navmesh_reachable": False})
            continue
        groups = connected_groups(
            [member_id for member_id, _snapped in usable],
            lambda left, right: _same_region(finder, usable[left][1], usable[right][1]),
        )
        original_ids = set(place["member_viewpoint_ids"])
        if len(groups) == 1 and set(groups[0]) == original_ids:
            copy = dict(place)
            copy["navmesh_reachable"] = True
            kept.append(copy)
            continue
        for index, member_ids in enumerate(groups):
            rebuilt = rebuild_place(
                member_ids,
                place["kind"],
                graph["positions"],
                graph["adjacency"],
                graph["branch_nodes"],
                walkable=lambda origin, target: _short_path(finder, origin, target),
            )
            if rebuilt is None:
                rejected.append(
                    {
                        **place,
                        "place_id": f"{place['place_id']}#{index}",
                        "member_viewpoint_ids": member_ids,
                        "navmesh_reachable": False,
                        "split_from_place_id": place["place_id"],
                    }
                )
                continue
            same_representative = rebuilt["viewpoint_id"] == place["viewpoint_id"]
            rebuilt["place_id"] = place["place_id"] if len(groups) == 1 and same_representative else f"{place['place_id']}#{index}"
            rebuilt["navmesh_reachable"] = True
            rebuilt["split_from_place_id"] = place["place_id"]
            kept.append(rebuilt)
    return kept, rejected


def _reachability(finder):
    """True when the snapped path is not a long detour around a wall."""
    if finder is None:
        return None
    cache: dict[tuple, bool] = {}

    def check(origin, member) -> bool:
        key = (tuple(np.round(origin, 3)), tuple(np.round(member, 3)))
        if key not in cache:
            cache[key] = _short_path(finder, origin, member)
        return cache[key]

    return check


def _short_path(finder, origin, member) -> bool:
    start = _snap(finder, list(origin))
    end = _snap(finder, list(member))
    if start is None or end is None:
        return False
    return _same_region(finder, start, end)


def _clearance(finder, point: list[float]) -> float | None:
    if finder is None or not all(math.isfinite(value) for value in point):
        return None
    try:
        distance = float(finder.distance_to_closest_obstacle(np.asarray(point, dtype=np.float32), 4.0))
    except TypeError:
        distance = float(finder.distance_to_closest_obstacle(np.asarray(point, dtype=np.float32)))
    if not math.isfinite(distance):
        return None
    return distance


def _add_clearance(finder, rows: list[dict]) -> None:
    clearances = []
    for row in rows:
        value = _clearance(finder, row["position"])
        row["clearance_m"] = value
        row["clearance_band"] = clearance_band(value)
        clearances.append(value)
    if finder is None:
        return
    for event in room_doors([row["position"] for row in rows], clearances):
        rows[event["index"]]["clearance_event"] = event["kind"]


def _depth_band(depth: np.ndarray, column: int) -> list[float]:
    height, width = depth.shape[:2]
    left = max(0, column - 8)
    right = min(width, column + 9)
    top = int(height * 0.3)
    bottom = int(height * 0.7)
    return np.asarray(depth[top:bottom, left:right], dtype=np.float64).reshape(-1).tolist()


def _add_depth(rows: list[dict], places: list[dict]) -> None:
    by_id = {place["place_id"]: place for place in places}
    loaded: dict[str, np.ndarray] = {}
    for row in rows:
        candidates = row.get("angle_candidate_exit_ids") or []
        path = row.get("depth_path")
        place = by_id.get(row.get("place_id"))
        if row["phase"] not in ("enter", "inside", "started_inside") or not path or not candidates or place is None:
            row["depth_exit_status"] = []
            continue
        if path not in loaded:
            try:
                loaded[path] = depth_to_meters(np.load(path))
            except (OSError, ValueError):
                loaded[path] = None
        depth = loaded[path]
        if depth is None:
            row["depth_exit_status"] = [{"exit_id": exit_id, "status": "unknown"} for exit_id in candidates]
            continue
        exits = {item["exit_id"]: item for item in place["exits"]}
        camera = tuple(row["position"])
        yaw = row.get("yaw")
        statuses = []
        for exit_id in candidates:
            target = None if exit_id not in exits else exits[exit_id].get("toward_position")
            if yaw is None or target is None:
                statuses.append({"exit_id": exit_id, "status": "unknown"})
                continue
            column = image_column(cone_offset(camera, tuple(target), yaw), depth.shape[1])
            if column is None:
                statuses.append({"exit_id": exit_id, "status": "not_in_cone"})
                continue
            distance = optical_axis_distance(camera, tuple(target), yaw)
            statuses.append(
                {
                    "exit_id": exit_id,
                    "status": depth_cone_status(_depth_band(depth, column), distance),
                    "column": column,
                    "target_optical_m": distance,
                    "depth_units": "meters",
                    "projection": "pinhole",
                }
            )
        row["depth_exit_status"] = statuses


def _cohort(row: dict) -> str:
    split = row["split"]
    if split.startswith("formal_dev"):
        return "dev"
    if split.startswith("formal_fit_radius010"):
        return "radius010"
    if split.startswith("fit_houses"):
        return "fit_house"
    if split == "formal_fit_20261028_04":
        return "go2w_04"
    return "other"


def _depth_count(rows: list[dict], status: str) -> int:
    return sum(
        1
        for row in rows
        for item in row.get("depth_exit_status") or []
        if item.get("status") == status
    )


def _summary(places: list[dict], frames: list[dict], entries: list[dict], graph_stats: dict) -> dict:
    go2w_frames = [row for row in frames if _cohort(row) == "go2w_04"]
    go2w_entries = [row for row in entries if _cohort(row) == "go2w_04"]
    radius_entries = [row for row in entries if _cohort(row) == "radius010"]
    fit_house_entries = [row for row in entries if _cohort(row) == "fit_house"]
    fit_frames = go2w_frames
    fit_entries = go2w_entries
    visual_entries = [row for row in fit_entries if row.get("has_rgb")]
    by_walk = defaultdict(list)
    for row in fit_frames:
        by_walk[(row["split"], row["episode_uid"])].append(row)
    history = []
    for row in visual_entries:
        sequence = by_walk[(row["split"], row["episode_uid"])]
        window = sequence[max(0, row["index"] - 5) : row["index"] + 1]
        if len(window) == 6 and all(item.get("has_rgb") for item in window):
            history.append(row)
    occurrence_ids = [row["occurrence_id"] for row in frames]
    return {
        "decision_places": len(places),
        "fork_places": sum(1 for row in places if row["kind"] == "fork"),
        "turn_places": sum(1 for row in places if row["kind"] == "turn"),
        "scenes": len({row["scene"] for row in places}),
        "full_teacher_edges": graph_stats["full_edges"],
        "local_teacher_edges": graph_stats["local_edges"],
        "cycle_rank": graph_stats["cycle_rank"],
        "frames": len(frames),
        "visual_frames_fit": sum(1 for row in fit_frames if row.get("has_rgb")),
        "pose_only_frames_fit": sum(1 for row in fit_frames if not row.get("has_rgb")),
        "geometric_entries_fit": len(fit_entries),
        "visual_entries_fit": len(visual_entries),
        "visual_entry_images_fit": len({row["rgb_sha256"] for row in visual_entries}),
        "pose_only_entries_fit": sum(1 for row in fit_entries if not row.get("has_rgb")),
        "visual_entries_with_5_prior_rgb_frames": len(history),
        "visual_revisits_fit": sum(1 for row in visual_entries if row["revisit"]),
        "visual_known_place_entries_fit": sum(1 for row in visual_entries if row["known_place"]),
        "duplicate_occurrence_ids": len(occurrence_ids) - len(set(occurrence_ids)),
        "reachable_places": sum(1 for row in places if row.get("navmesh_reachable") is True),
        "unreachable_places": sum(1 for row in places if row.get("navmesh_reachable") is False),
        "approach_frames_go2w": sum(1 for row in go2w_frames if row.get("phase") == "approach"),
        "enter_frames_go2w": sum(1 for row in go2w_frames if row.get("phase") == "enter"),
        "leave_frames_go2w": sum(1 for row in go2w_frames if row.get("phase") == "leave"),
        "started_inside_frames_go2w": sum(1 for row in go2w_frames if row.get("phase") == "started_inside"),
        "tight_passage_frames_go2w": sum(1 for row in go2w_frames if row.get("clearance_band") == "tight_passage"),
        "open_interior_frames_go2w": sum(1 for row in go2w_frames if row.get("clearance_band") == "open_interior"),
        "doorway_events_go2w": sum(1 for row in go2w_frames if row.get("clearance_event")),
        "depth_blocked_go2w": _depth_count(go2w_frames, "blocked"),
        "depth_not_blocked_go2w": _depth_count(go2w_frames, "not_blocked"),
        "visual_entries_radius010": sum(1 for row in radius_entries if row.get("has_rgb")),
        "visual_entries_fit_house": sum(1 for row in fit_house_entries if row.get("has_rgb")),
        "dev_entries": sum(1 for row in entries if row["dev_excluded"]),
        "dev_used_for_training": False,
        "training_started": False,
        "counts_authorize_training": False,
        "visibility": "unknown",
    }


def _report(summary: dict) -> str:
    lines = [
        "要达成的任务是：锚点只留分岔、门口、房间交界和明显转弯，并核对几何和能不能走过去。邻居多不多不能单独决定一个点。",
        "节点发现器吃连续画面，监督接近、进入和离开。走廊和房间内部是负例。",
        "新地方由机器人自己建节点。再经过时用画面和历史判断是不是旧节点，地图是它自己积累的。",
        "运行时不提供官方编号和邻接：连续画面发现节点和可见方向，世界模型与记忆帮忙判断，LightNav 执行，真实观察写回自己的地图。证据不够就先观察再确认。",
        "单张照片分类不够。还要处理看不全、重复触发、换个方向回来，以及两个相邻地点被并成一个。",
        "验收是没参与训练的房子，不给官方编号和邻接，仍能发现决策区域。",
        "",
        "这一版修正第三版的三处标签：深度先从 0–1 还原成 0–10 米，再用透视投影取列，并和光轴距离比较；直线近但要绕很远的点不再算进入；拆开的节点只保留自己的成员，并重选代表点和出口。",
        "not_blocked 仍只是深度代理，不是看见了哪个出口。visibility 仍然是 unknown。起点已经在区域内记 started_inside，不算进入。",
        "radius010 的六批轨迹单独计数，不并进 formal_fit_20261028_04。下面的数量不是“够训练”的结论。",
        "",
        f"- 可站立的决策点 {summary['reachable_places']} 个，够不着的 {summary['unreachable_places']} 个。分岔 {summary['fork_places']} 个，拐弯 {summary['turn_places']} 个。",
        f"- 教师图同层连边 {summary['full_teacher_edges']} 条，按方向留近邻后 {summary['local_teacher_edges']} 条，回路秩 {summary['cycle_rank']}。",
        f"- go2w 这一批：接近 {summary['approach_frames_go2w']} 帧，进入 {summary['enter_frames_go2w']} 帧，离开 {summary['leave_frames_go2w']} 帧，开头就在里面 {summary['started_inside_frames_go2w']} 帧。",
        f"- go2w 窄通道 {summary['tight_passage_frames_go2w']} 帧，开阔处 {summary['open_interior_frames_go2w']} 帧，通道与房间的交界 {summary['doorway_events_go2w']} 处。",
        f"- go2w 深度方向：挡住 {summary['depth_blocked_go2w']} 条，深度够到目标距离 {summary['depth_not_blocked_go2w']} 条。",
        f"- go2w 几何进入 {summary['geometric_entries_fit']} 次。其中有图像的 {summary['visual_entries_fit']} 次，不同图像 {summary['visual_entry_images_fit']} 张。只有位置的 {summary['pose_only_entries_fit']} 次。",
        f"- radius010 有图像的进入 {summary['visual_entries_radius010']} 次。补采房子有图像的进入 {summary['visual_entries_fit_house']} 次。这两批不和 go2w 加在一起。",
        f"- 进入帧及其前 5 帧都有图像的 {summary['visual_entries_with_5_prior_rgb_frames']} 次。这只说明片段还能对上图像，不是模型窗口的规定长度。",
        f"- 有图像的同路返回 {summary['visual_revisits_fit']} 次，有图像且另一条路到过的 {summary['visual_known_place_entries_fit']} 次。",
        f"- occurrence_id 重复 {summary['duplicate_occurrence_ids']} 条。检查路线进入 {summary['dev_entries']} 次，不参与训练。",
        "",
        "图在 scenes 里。彩色线是走过的路线。实心橙点是路线走进的官方点，空心点是附近没走进的官方点。灰色短线是出口方向，不是已经核实的可见范围。",
    ]
    return "\n".join(lines) + "\n"


def _draw_all(grouped: dict[str, list[dict]], places_by_scene: dict[str, list[dict]], entries: list[dict]) -> None:
    entered = {(row["scene"], row["place_id"]) for row in entries}
    for scene, walks in sorted(grouped.items()):
        mesh_path = REPO / "data/scene_datasets/mp3d" / scene / f"{scene}.navmesh"
        if not mesh_path.is_file():
            continue
        finder = habitat_sim.PathFinder()
        if not finder.load_nav_mesh(str(mesh_path)):
            continue
        vertices = np.asarray(finder.build_navmesh_vertices(), dtype=np.float64)
        indices = np.asarray(finder.build_navmesh_vertex_indices(), dtype=np.int64).reshape(-1, 3)
        _draw_scene(scene, (vertices, indices), walks, places_by_scene.get(scene, []), entered)


def _draw_scene(scene, mesh, walks, places, entered: set[tuple[str, str]]) -> None:
    heights = [frame["position"][1] for walk in walks for frame in walk["frames"]]
    anchors = _floor_anchors(heights or [0.0])
    by_floor = defaultdict(lambda: {"paths": [], "places": []})
    for walk in walks:
        grouped = defaultdict(list)
        for frame in walk["frames"]:
            grouped[_floor_anchor(frame["position"][1], anchors)].append(frame["position"])
        for floor, positions in grouped.items():
            if len(positions) >= 2:
                by_floor[floor]["paths"].append(positions)
    for place in places:
        floor = _floor_anchor(place["position"][1], anchors)
        paths = by_floor[floor]["paths"]
        if not paths:
            continue
        near = _near_path(place["position"], paths)
        walked = (scene, place["place_id"]) in entered
        if near or walked:
            by_floor[floor]["places"].append((place, walked))
    for floor, payload in by_floor.items():
        if not payload["paths"]:
            continue
        cloud = [(position[0], position[2]) for path in payload["paths"] for position in path]
        cloud.extend((place["position"][0], place["position"][2]) for place, _walked in payload["places"])
        minx, minz, span = _bounds(cloud)
        pix = _projector(minx, minz, span)
        flat = [{"position": position} for path in payload["paths"] for position in path]
        image = Image.new("RGB", (MAP + PANEL, MAP), (248, 247, 244))
        _paint_floor(image, mesh, pix, flat, minx, minz, span)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(FONT, 16)
        for index, path in enumerate(payload["paths"]):
            points = [pix(position[0], position[2]) for position in path]
            if len(points) >= 2:
                draw.line(points, fill=COLORS[index % len(COLORS)], width=3)
        for number, (place, walked) in enumerate(payload["places"], start=1):
            u, v = pix(place["position"][0], place["position"][2])
            if walked:
                draw.ellipse([u - 8, v - 8, u + 8, v + 8], fill=(225, 120, 20), outline=(255, 255, 255))
                for exit_row in place["exits"]:
                    end = [
                        place["position"][0] + math.sin(exit_row["bearing_rad"]) * 1.3,
                        place["position"][2] + math.cos(exit_row["bearing_rad"]) * 1.3,
                    ]
                    draw.line([pix(place["position"][0], place["position"][2]), pix(end[0], end[1])], fill=(140, 140, 140), width=2)
            else:
                draw.ellipse([u - 7, v - 7, u + 7, v + 7], outline=(225, 120, 20), width=2)
            draw.text((u + 10, v - 14), f"{number:03d}", fill=(20, 20, 20), font=font)
        image.save(OUT / "scenes" / f"{scene}_y{floor:.1f}.png")


def _near_path(position: list[float], paths: list[list[list[float]]]) -> bool:
    for path in paths:
        for point in path:
            if abs(position[1] - point[1]) > MATCH_FLOOR_M:
                continue
            if math.hypot(position[0] - point[0], position[2] - point[2]) <= NEAR_PATH_M:
                return True
    return False


def _floor_anchors(heights: list[float]) -> list[float]:
    anchors = []
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


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
