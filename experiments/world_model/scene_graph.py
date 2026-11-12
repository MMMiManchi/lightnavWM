"""A scene graph of places, visits, and walked exits.

Place identity comes from a walkable region, not from coordinates being close.
Two visits share a place only when they are already in the same region.
A region is a place when a route decides there: the walked paths leave toward
more than one next region, or the route turns, passes a room door, or reaches
the instruction stop. Corridor points in between stay on the edge.

The graph keeps cycles. An exit that nobody walked in this visit stays
unknown and does not receive a destination.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

MATCH_EXIT_RAD = math.radians(45.0)


def share_region(
    *,
    same_floor: bool,
    same_island: bool,
    segment_on_navmesh: bool,
) -> bool:
    """Close coordinates are not enough. The segment between them must be walkable."""
    return bool(same_floor and same_island and segment_on_navmesh)


def build_graph(
    episodes: list[dict[str, Any]],
    exits_by_region: dict[str, list[dict[str, Any]]],
    decision_regions: dict[str, list[str]],
) -> dict[str, Any]:
    """Build places, occurrences, walked edges, and per-visit branch outcomes."""
    compressed = [(episode, _compress(episode["regions"])) for episode in episodes]
    successors: dict[str, set[str]] = defaultdict(set)
    for _episode, sequence in compressed:
        for (left, _left_index), (right, _right_index) in zip(sequence, sequence[1:]):
            successors[left].add(right)
    place_regions = set(decision_regions)
    for region, nxt in successors.items():
        if len(nxt) >= 2:
            place_regions.add(region)
    occurrences = []
    edges = []
    branches = []
    for episode, sequence in compressed:
        hits = [(region, index) for region, index in sequence if region in place_regions]
        for visit_index, (region, index) in enumerate(hits):
            nxt = hits[visit_index + 1] if visit_index + 1 < len(hits) else None
            occurrence_id = f"{episode['episode_uid']}#{index}"
            bearing = _departure_bearing(episode["positions"], index)
            occurrences.append(
                {
                    "occurrence_id": occurrence_id,
                    "place_region": region,
                    "episode_uid": episode["episode_uid"],
                    "index": index,
                    "observation_id": episode["observation_ids"][index],
                    "approach_bearing_rad": _approach_bearing(episode["positions"], index),
                    "position": episode["positions"][index],
                }
            )
            if nxt is not None:
                edges.append(
                    {
                        "episode_uid": episode["episode_uid"],
                        "from_region": region,
                        "to_region": nxt[0],
                        "from_occurrence_id": occurrence_id,
                        "observation_ids": episode["observation_ids"][index : nxt[1] + 1],
                    }
                )
            taken = _match_exit(bearing, exits_by_region.get(region, []))
            for exit_index, exit_row in enumerate(exits_by_region.get(region, [])):
                walked = taken is not None and exit_index == taken and nxt is not None
                branches.append(
                    {
                        "occurrence_id": occurrence_id,
                        "place_region": region,
                        "exit_id": exit_row["exit_id"],
                        "bearing_rad": exit_row["bearing_rad"],
                        "walked": walked,
                        "next_place_region": nxt[0] if walked else None,
                        "outcome": "walked" if walked else "unknown",
                    }
                )
            if nxt is not None and taken is None:
                branches.append(
                    {
                        "occurrence_id": occurrence_id,
                        "place_region": region,
                        "exit_id": "walked",
                        "bearing_rad": bearing,
                        "walked": True,
                        "next_place_region": nxt[0],
                        "outcome": "walked",
                    }
                )
    places = []
    for region in sorted(place_regions):
        kinds = list(decision_regions.get(region, []))
        if len(successors.get(region, ())) >= 2 and "junction" not in kinds:
            kinds.append("junction")
        places.append({"region": region, "kinds": kinds})
    return {
        "places": places,
        "occurrences": occurrences,
        "edges": edges,
        "branches": branches,
        "cycles": directed_cycles([(edge["from_region"], edge["to_region"]) for edge in edges]),
    }


def directed_cycles(edges: list[tuple[str, str]]) -> list[list[str]]:
    """Directed cycles in the walked graph. A loop is kept, not opened into a tree."""
    graph: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        graph[left].add(right)
    cycles = []
    seen: set[tuple[str, ...]] = set()

    def walk(start: str, node: str, path: list[str]) -> None:
        if len(path) > 8:
            return
        for nxt in graph.get(node, ()):
            if nxt == start and len(path) > 1:
                body = path
                pivot = body.index(min(body))
                key = tuple(body[pivot:] + body[:pivot])
                if key not in seen:
                    seen.add(key)
                    cycles.append(body + [start])
            elif nxt not in path:
                walk(start, nxt, path + [nxt])

    for start in sorted(graph):
        walk(start, start, [start])
    return cycles


def _compress(regions: list[str]) -> list[tuple[str, int]]:
    sequence = []
    for index, region in enumerate(regions):
        if not sequence or sequence[-1][0] != region:
            sequence.append((region, index))
    return sequence


def _departure_bearing(positions: list[list[float]], index: int) -> float | None:
    travelled = 0.0
    for nxt in range(index + 1, len(positions)):
        travelled += _horizontal(positions[nxt - 1], positions[nxt])
        if travelled >= 0.4:
            return _bearing(positions[index], positions[nxt])
    return None


def _approach_bearing(positions: list[list[float]], index: int) -> float | None:
    travelled = 0.0
    cursor = index
    while cursor > 0:
        travelled += _horizontal(positions[cursor], positions[cursor - 1])
        cursor -= 1
        if travelled >= 0.4:
            return _bearing(positions[cursor], positions[index])
    return None


def _match_exit(bearing: float | None, exits: list[dict[str, Any]]) -> int | None:
    if bearing is None or not exits:
        return None
    best = None
    gap = MATCH_EXIT_RAD
    for index, exit_row in enumerate(exits):
        delta = abs(_wrap(bearing - float(exit_row["bearing_rad"])))
        if delta <= gap:
            gap = delta
            best = index
    return best


def _bearing(start: list[float], end: list[float]) -> float:
    return math.atan2(end[0] - start[0], end[2] - start[2])


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _horizontal(left: list[float], right: list[float]) -> float:
    return math.hypot(left[0] - right[0], left[2] - right[2])
