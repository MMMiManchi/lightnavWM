"""Official viewpoints are training anchors. The robot builds its own map.

Anchors are forks, doorways, room boundaries, and clear turns, checked for
geometry and reachability. A count of neighbors is not enough. Training
supervises approach, entry, and departure on continuous frames, with corridor
and room-interior negatives, then whether a later view is a node the robot
already created. At runtime the graph, its ids, and its neighbor list are absent.
"""

from __future__ import annotations

import math
from collections import defaultdict

from experiments.world_model.pose_math import wrap_pi

ENTER_RADIUS_M = 1.5
LEAVE_RADIUS_M = 2.0
APPROACH_RADIUS_M = 3.0
TIGHT_CLEARANCE_M = 0.75
OPEN_CLEARANCE_M = 1.2
FLOOR_GAP_M = 1.2
MATCH_FLOOR_M = 0.8
MERGE_M = 1.5
TURN_CLEAR_M = 3.0
HFOV_DEG = 120.0
EXIT_GAP_RAD = math.radians(45.0)
MIN_SPLIT_RAD = math.radians(45.0)
MAX_CORNER_RAD = math.radians(135.0)
TURN_MIN_RAD = math.radians(60.0)
TURN_MAX_RAD = math.radians(120.0)
MIN_ARM_HOPS = 2
MIN_ARM_M = 3.0


def habitat_position(pose: list[float], height: float) -> tuple[float, float, float]:
    """Matterport camera pose to Habitat world coordinates.

    The pose is a row-major 4x4 matrix, z up. Habitat is y up, and a recorded
    robot position sits near the floor, so the camera height is removed.
    """
    matterport_x = float(pose[3])
    matterport_y = float(pose[7])
    matterport_z = float(pose[11])
    return (matterport_x, matterport_z - float(height), -matterport_y)


def camera_bearing(yaw: float) -> float:
    """Yaw 0 faces world -Z. The result uses atan2(dx, dz), the exit convention."""
    return wrap_pi(float(yaw) + math.pi)


def exit_in_view(exit_bearing: float, yaw: float, hfov_deg: float = HFOV_DEG) -> bool:
    """The bearing lies inside the camera cone. This is not an occlusion check."""
    half = math.radians(float(hfov_deg)) / 2.0
    return abs(wrap_pi(float(exit_bearing) - camera_bearing(yaw))) <= half


def cone_offset(camera: tuple[float, float, float], target: tuple[float, float, float], yaw: float) -> float:
    """Signed angle from the camera forward to a target. Positive is left."""
    bearing = math.atan2(float(target[0]) - camera[0], float(target[2]) - camera[2])
    return wrap_pi(bearing - camera_bearing(yaw))


def image_column(offset_rad: float, width: int, hfov_deg: float = HFOV_DEG) -> int | None:
    """Pinhole column for a cone offset. Positive offset is left. None outside the image."""
    half = math.radians(hfov_deg) / 2.0
    if abs(offset_rad) >= half or width < 2:
        return None
    focal = (width / 2.0) / math.tan(half)
    column = int(round(width / 2.0 - focal * math.tan(offset_rad)))
    if column < 0 or column >= width:
        return None
    return column


def optical_axis_distance(
    camera: tuple[float, float, float], target: tuple[float, float, float], yaw: float
) -> float:
    """Meters along the camera axis to a level target. This is the stored depth's axis."""
    offset = cone_offset(camera, target, yaw)
    horizontal = math.hypot(float(target[0]) - camera[0], float(target[2]) - camera[2])
    return horizontal * math.cos(offset)


def angle_candidate_exit_ids(
    camera: tuple[float, float, float],
    yaw: float | None,
    exits: list[dict],
    hfov_deg: float = HFOV_DEG,
) -> list[str]:
    """Exit ids whose target lies inside the cone from the current camera position.

    Occlusion is not tested. Callers must keep the result marked unknown.
    """
    if yaw is None:
        return []
    found = []
    for exit_row in exits:
        target = exit_row.get("toward_position")
        if target is None:
            continue
        bearing = math.atan2(float(target[0]) - camera[0], float(target[2]) - camera[2])
        if exit_in_view(bearing, yaw, hfov_deg):
            found.append(exit_row["exit_id"])
    return found


def detached_frame(frame: dict, occurrence_id: str) -> dict:
    """Copy one recorded step so a shared prefix cannot be rewritten by another branch."""
    copied = dict(frame)
    position = frame.get("position")
    if position is not None:
        copied["position"] = [float(value) for value in position]
    copied["occurrence_id"] = occurrence_id
    return copied


def teacher_graph(nodes: list[dict], edges: list[tuple[str, str]]) -> dict:
    """Full same-floor links, plus the local graph that still contains real loops."""
    positions, kept = _kept_edges(nodes, edges)
    adjacency = _local_adjacency(positions, kept)
    local_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    components = _component_count(adjacency)
    local_pairs = set()
    for left, neighbors in adjacency.items():
        for right in neighbors:
            local_pairs.add((left, right) if left < right else (right, left))
    return {
        "positions": positions,
        "full_edges": len(kept),
        "full_edge_list": kept,
        "local_edges": local_edges,
        "local_edge_list": sorted(local_pairs),
        "cycle_rank": local_edges - len(adjacency) + components,
        "adjacency": adjacency,
    }


def decision_places(nodes: list[dict], edges: list[tuple[str, str]]) -> list[dict]:
    """Keep official viewpoints where the scan graph actually branches or turns.

    A straight corridor, and a spur that points the same way as the corridor,
    are not decision regions. Nearby branch points on one patch become one
    viewpoint, the official one closest to their middle. A loop is kept.
    """
    graph = teacher_graph(nodes, edges)
    positions = graph["positions"]
    adjacency = graph["adjacency"]
    branch_nodes = {node for node, neighbors in adjacency.items() if len(neighbors) >= 3}
    forks = [node for node in branch_nodes if _real_arm_count(node, adjacency, positions, branch_nodes) >= 3]
    turns = [
        node
        for node, neighbors in adjacency.items()
        if len(neighbors) == 2
        and _sharp_corner(node, neighbors, positions)
        and _real_arm_count(node, adjacency, positions, branch_nodes) >= 2
        and not _near_any(node, forks, positions, TURN_CLEAR_M)
    ]
    places = []
    places.extend(_places_from(forks, adjacency, positions, branch_nodes, "fork"))
    places.extend(_places_from(turns, adjacency, positions, branch_nodes, "turn"))
    places.sort(key=lambda place: (place["position"][1], place["position"][0], place["position"][2], place["viewpoint_id"]))
    return places


def clearance_band(clearance_m: float | None) -> str:
    """Navmesh clearance. Tight and open are geometry, not a room name."""
    if clearance_m is None or not math.isfinite(clearance_m):
        return "unknown"
    if clearance_m <= TIGHT_CLEARANCE_M:
        return "tight_passage"
    if clearance_m >= OPEN_CLEARANCE_M:
        return "open_interior"
    return "intermediate"


def depth_cone_status(samples: list[float], target_distance_m: float) -> str:
    """Whether the depth in one image cone stops short of the exit target.

    ``not_blocked`` means the measured depth reaches the target distance.
    It does not prove which exit is seen.
    """
    finite = [value for value in samples if math.isfinite(value) and value > 0.05]
    if len(finite) < 10 or not math.isfinite(target_distance_m):
        return "unknown"
    finite.sort()
    median = finite[len(finite) // 2]
    if median + 0.5 < target_distance_m:
        return "blocked"
    return "not_blocked"


def annotate_walk(
    frames: list[dict],
    places: list[dict],
    already_known: set[str] | None = None,
    hfov_deg: float = HFOV_DEG,
    reachable=None,
) -> tuple[list[dict], set[str]]:
    """Label approach, entry, the stay, and departure. The first frame can already be inside.

    ``reachable(position, member)`` rejects a nearby member when the walk cannot
    reach it without a long detour. Straight-line distance alone can cross a wall.
    """
    known = set(already_known or ())
    seen: set[str] = set()
    current: dict | None = None
    visit_revisit = False
    visit_known = False
    rows = []
    for index, frame in enumerate(frames):
        position = tuple(frame["position"])
        if current is not None and _in_region(position, current, LEAVE_RADIUS_M, reachable):
            place = current
        else:
            place = _nearest_place(position, places, ENTER_RADIUS_M, reachable)
            current = place
        was_inside = bool(rows) and rows[-1]["in_decision_region"]
        now_inside = place is not None
        place_id = None if place is None else place["place_id"]
        previous_id = rows[-1]["place_id"] if was_inside else None
        if now_inside and place_id != previous_id:
            phase = "started_inside" if index == 0 else "enter"
        elif now_inside:
            phase = "inside"
        elif was_inside:
            phase = "leave"
            place_id = rows[-1]["place_id"]
            place = None
            current = None
        else:
            phase = _approach_phase(position, rows, places, reachable)
            place_id = None
        entered = phase == "enter"
        if phase in ("enter", "started_inside") and place is not None:
            visit_revisit = place_id in seen
            visit_known = place_id in known
            seen.add(place_id)
            exit_ids = [exit_row["exit_id"] for exit_row in place["exits"]]
            candidates = angle_candidate_exit_ids(position, frame.get("yaw"), place["exits"], hfov_deg)
        elif phase == "inside" and place is not None:
            exit_ids = [exit_row["exit_id"] for exit_row in place["exits"]]
            candidates = angle_candidate_exit_ids(position, frame.get("yaw"), place["exits"], hfov_deg)
        else:
            visit_revisit = False
            visit_known = False
            exit_ids = []
            candidates = []
        copied = detached_frame(frame, f"{frame.get('episode_uid', 'walk')}#{index}")
        copied.update(
            {
                "index": index,
                "phase": phase,
                "in_decision_region": phase in ("started_inside", "enter", "inside"),
                "entered_decision_region": entered,
                "place_id": place_id if phase != "approach" else _approach_place_id(position, places, reachable),
                "viewpoint_id": None if place is None else place["viewpoint_id"],
                "revisit": visit_revisit if phase in ("enter", "started_inside", "inside") else False,
                "known_place": visit_known if phase in ("enter", "started_inside", "inside") else False,
                "exit_ids": exit_ids,
                "angle_candidate_exit_ids": candidates,
                "visibility": "unknown",
            }
        )
        if phase == "inside":
            copied["revisit"] = rows[-1]["revisit"] if rows else False
            copied["known_place"] = rows[-1]["known_place"] if rows else False
        rows.append(copied)
    return rows, seen


def _approach_phase(
    position: tuple[float, float, float], rows: list[dict], places: list[dict], reachable=None
) -> str:
    place = _nearest_place(position, places, APPROACH_RADIUS_M, reachable)
    if place is None:
        return "outside"
    if not rows:
        return "approach"
    previous = tuple(rows[-1]["position"])
    now = min(_horizontal(position, tuple(member)) for member in place["member_positions"])
    before = min(_horizontal(previous, tuple(member)) for member in place["member_positions"])
    if now <= before:
        return "approach"
    return "outside"


def _places_from(
    seeds: list[str],
    adjacency: dict[str, set[str]],
    positions: dict,
    branch_nodes: set[str],
    kind: str,
) -> list[dict]:
    places = []
    for members in _merge(seeds, adjacency, positions):
        representative = _representative(members, positions)
        exits = _exits(representative, members, adjacency, positions, branch_nodes)
        if not _is_decision(exits):
            continue
        places.append(
            {
                "viewpoint_id": representative,
                "position": list(positions[representative]),
                "member_viewpoint_ids": sorted(members),
                "member_positions": [list(positions[member]) for member in sorted(members)],
                "kind": kind,
                "exits": exits,
            }
        )
    return places


def _kept_edges(nodes: list[dict], edges: list[tuple[str, str]]) -> tuple[dict, list[tuple[str, str]]]:
    positions = {node["id"]: tuple(node["position"]) for node in nodes}
    kept = []
    seen = set()
    for left, right in edges:
        if left not in positions or right not in positions or left == right:
            continue
        if abs(positions[left][1] - positions[right][1]) > FLOOR_GAP_M:
            continue
        key = (left, right) if left < right else (right, left)
        if key in seen:
            continue
        seen.add(key)
        kept.append(key)
    return positions, kept


def _local_adjacency(positions: dict, edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Keep the nearest viewpoint in each direction. A real loop remains."""
    incoming: dict[str, list[tuple[str, float]]] = {node: [] for node in positions}
    for left, right in edges:
        distance = _horizontal(positions[left], positions[right])
        incoming[left].append((right, distance))
        incoming[right].append((left, distance))
    adjacency = {node: set() for node in positions}
    for node, neighbors in incoming.items():
        pairs = [
            (_point_bearing(positions[node], positions[other]), other, distance) for other, distance in neighbors
        ]
        for group in _cluster([(bearing, other) for bearing, other, _distance in pairs]):
            nearest = min(group, key=lambda item: _distance_of(item[1], neighbors))
            adjacency[node].add(nearest[1])
            adjacency[nearest[1]].add(node)
    return adjacency


def _distance_of(other: str, neighbors: list[tuple[str, float]]) -> float:
    for name, distance in neighbors:
        if name == other:
            return distance
    return math.inf


def _component_count(adjacency: dict[str, set[str]]) -> int:
    seen = set()
    count = 0
    for start in adjacency:
        if start in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            for other in adjacency[node]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
    return count


def _merge(seeds: list[str], adjacency: dict[str, set[str]], positions: dict) -> list[list[str]]:
    seed_set = set(seeds)
    parent = {node: node for node in seeds}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for node in seeds:
        for other in adjacency[node]:
            if other not in seed_set:
                continue
            same_patch = _horizontal(positions[node], positions[other]) <= MERGE_M
            same_floor = abs(positions[node][1] - positions[other][1]) <= MATCH_FLOOR_M
            if same_patch and same_floor:
                parent[find(other)] = find(node)
    groups = defaultdict(list)
    for node in seeds:
        groups[find(node)].append(node)
    return list(groups.values())


def _representative(members: list[str], positions: dict) -> str:
    count = float(len(members))
    middle = (
        sum(positions[node][0] for node in members) / count,
        sum(positions[node][1] for node in members) / count,
        sum(positions[node][2] for node in members) / count,
    )
    return min(members, key=lambda node: (_horizontal(positions[node], middle), node))


def _exits(
    representative: str,
    members: list[str],
    adjacency: dict[str, set[str]],
    positions: dict,
    branch_nodes: set[str],
) -> list[dict]:
    member_set = set(members)
    outside = []
    for member in members:
        for other in adjacency[member]:
            if other in member_set:
                continue
            if _real_arm(member, other, adjacency, positions, branch_nodes - member_set):
                outside.append(other)
    pairs = [(_bearing(positions, representative, other), other) for other in outside]
    exits = []
    for index, group in enumerate(_cluster(pairs), start=1):
        bearing = math.atan2(
            sum(math.sin(angle) for angle, _other in group),
            sum(math.cos(angle) for angle, _other in group),
        )
        toward = min(group, key=lambda item: abs(wrap_pi(item[0] - bearing)))[1]
        exits.append(
            {
                "exit_id": f"e{index}",
                "bearing_rad": bearing,
                "toward_viewpoint_id": toward,
                "toward_position": list(positions[toward]),
            }
        )
    return exits


def _cluster(pairs: list[tuple[float, str]]) -> list[list[tuple[float, str]]]:
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda item: item[0])
    groups = [[ordered[0]]]
    for item in ordered[1:]:
        if item[0] - groups[-1][-1][0] <= EXIT_GAP_RAD:
            groups[-1].append(item)
        else:
            groups.append([item])
    wrapped = ordered[0][0] + 2.0 * math.pi - ordered[-1][0]
    if len(groups) > 1 and wrapped <= EXIT_GAP_RAD:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    return groups


def _is_decision(exits: list[dict]) -> bool:
    if len(exits) >= 3:
        return True
    if len(exits) != 2:
        return False
    delta = abs(wrap_pi(exits[0]["bearing_rad"] - exits[1]["bearing_rad"]))
    return MIN_SPLIT_RAD <= delta <= MAX_CORNER_RAD


def _sharp_corner(node: str, neighbors: set[str], positions: dict) -> bool:
    left, right = list(neighbors)
    delta = abs(wrap_pi(_bearing(positions, node, left) - _bearing(positions, node, right)))
    return TURN_MIN_RAD <= delta <= TURN_MAX_RAD


def _real_arm_count(node: str, adjacency: dict[str, set[str]], positions: dict, branch_nodes: set[str]) -> int:
    return sum(
        1
        for other in adjacency[node]
        if _real_arm(node, other, adjacency, positions, branch_nodes)
    )


def _real_arm(
    origin: str,
    nxt: str,
    adjacency: dict[str, set[str]],
    positions: dict,
    stop_nodes: set[str],
) -> bool:
    """A dangling scan point is not an exit. A corridor or another branch is."""
    length = _horizontal(positions[origin], positions[nxt])
    hops = 1
    previous, current = origin, nxt
    seen = {origin}
    for _step in range(80):
        if current in seen:
            return True
        seen.add(current)
        if current in stop_nodes or len(adjacency[current]) != 2:
            break
        left, right = list(adjacency[current])
        following = right if left == previous else left
        length += _horizontal(positions[current], positions[following])
        hops += 1
        previous, current = current, following
    return hops >= MIN_ARM_HOPS or length >= MIN_ARM_M or current in stop_nodes


def _near_any(node: str, others: list[str], positions: dict, limit: float) -> bool:
    here = positions[node]
    for other in others:
        there = positions[other]
        if _horizontal(here, there) <= limit and abs(here[1] - there[1]) <= MATCH_FLOOR_M:
            return True
    return False


def connected_groups(member_ids: list[str], linked) -> list[list[str]]:
    """Split ids into groups. ``linked(i, j)`` is true when the pair shares a short path."""
    parent = list(range(len(member_ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(member_ids)):
        for right in range(left + 1, len(member_ids)):
            if linked(left, right):
                parent[find(left)] = find(right)
    groups: dict[int, list[str]] = defaultdict(list)
    for index, member_id in enumerate(member_ids):
        groups[find(index)].append(member_id)
    return sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])


def rebuild_place(
    member_ids: list[str],
    kind: str,
    positions: dict,
    adjacency: dict[str, set[str]],
    branch_nodes: set[str],
    walkable=None,
) -> dict | None:
    """Representative and exits for one side of a split. Inherited exits are not reused."""
    members = [member_id for member_id in member_ids if member_id in positions]
    if not members:
        return None
    representative = _representative(members, positions)
    exits = _exits(representative, members, adjacency, positions, branch_nodes)
    if walkable is not None:
        origin = positions[representative]
        exits = [
            exit_row
            for exit_row in exits
            if walkable(origin, tuple(exit_row["toward_position"]))
        ]
    if not _is_decision(exits):
        return None
    return {
        "viewpoint_id": representative,
        "position": list(positions[representative]),
        "member_viewpoint_ids": sorted(members),
        "member_positions": [list(positions[member]) for member in sorted(members)],
        "kind": kind,
        "exits": exits,
    }


def _approach_place_id(
    position: tuple[float, float, float], places: list[dict], reachable=None
) -> str | None:
    place = _nearest_place(position, places, APPROACH_RADIUS_M, reachable)
    if place is None:
        return None
    return place["place_id"]


def _nearest_place(
    position: tuple[float, float, float], places: list[dict], radius: float, reachable=None
) -> dict | None:
    best = None
    best_distance = radius
    for place in places:
        for member in place["member_positions"]:
            if abs(position[1] - member[1]) > MATCH_FLOOR_M:
                continue
            distance = _horizontal(position, tuple(member))
            if distance > best_distance:
                continue
            if reachable is not None and not reachable(position, tuple(member)):
                continue
            best = place
            best_distance = distance
    return best


def _in_region(position: tuple[float, float, float], place: dict, radius: float, reachable=None) -> bool:
    for member in place["member_positions"]:
        if abs(position[1] - member[1]) > MATCH_FLOOR_M:
            continue
        if _horizontal(position, tuple(member)) > radius:
            continue
        if reachable is not None and not reachable(position, tuple(member)):
            continue
        return True
    return False


def _bearing(positions: dict, origin: str, target: str) -> float:
    return _point_bearing(positions[origin], positions[target])


def _point_bearing(origin: tuple[float, float, float], target: tuple[float, float, float]) -> float:
    return math.atan2(target[0] - origin[0], target[2] - origin[2])


def _horizontal(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.hypot(left[0] - right[0], left[2] - right[2])
