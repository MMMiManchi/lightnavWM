"""Select and check same-state, multi-exit event consequences.

Official ids can choose a replay prefix offline. They are not a runtime node
list. A fixed distance, a fixed duration, and a local stop are not arrivals.
"""

from __future__ import annotations

import math
from typing import Any

from experiments.world_model.pose_math import wrap_pi

POLICY_ID = "geometric_opening_follower/v1"
LIGHTNAV_USED = False
MIN_EXIT_GAP_RAD = math.radians(20.0)
NEXT_EVENT_MIN_M = 0.8
BRANCH_STEP_BUDGET = 300
SERVER_STEP_CAP = 500
FIT_HOUSES = ("5LpN3gDmAk7", "kEZ7cmS4wCh", "VzqfbhrpDEA", "jh4fc5c5qoQ")
DEVELOPMENT_HOUSES = ("ZMojNkEp431",)
SEALED_HOUSES = ("Vvot9Ly1tCj",)
ROLES = {
    "fit": FIT_HOUSES,
    "development": DEVELOPMENT_HOUSES,
    "sealed_test": SEALED_HOUSES,
}
COMPLETE = "complete"
FORBIDDEN_INPUT_FIELDS = {
    "future_rgb",
    "target_rgb",
    "target_distance_m",
    "official_runtime_id",
    "adjacency",
    "elapsed_s",
}


def role_for_scene(scene: str) -> str | None:
    for role, houses in ROLES.items():
        if scene in houses:
            return role
    return None


def roles_are_disjoint() -> bool:
    houses = [scene for group in ROLES.values() for scene in group]
    return len(houses) == len(set(houses)) and all(group for group in ROLES.values())


def _gap(left: float, right: float) -> float:
    return abs(wrap_pi(left - right))


def paired_exits(openings: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Two openings whose bearings differ by at least twenty degrees."""
    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for index, left in enumerate(openings):
        for right in openings[index + 1 :]:
            gap = _gap(float(left["bearing_left_rad"]), float(right["bearing_left_rad"]))
            if gap < MIN_EXIT_GAP_RAD:
                continue
            if best is None or gap > best[0]:
                best = (gap, left, right)
    if best is None:
        return None
    _gap_rad, left, right = best
    ordered = sorted((left, right), key=lambda item: -float(item["bearing_left_rad"]))
    return [
        {
            "exit_id": int(item["exit_id"]),
            "bearing_left_rad": float(item["bearing_left_rad"]),
            "side": item["side"],
            "angular_width_rad": float(item["angular_width_rad"]),
        }
        for item in ordered
    ]


def candidate_origins(nodes: list[dict[str, Any]], houses: tuple[str, ...]) -> list[dict[str, Any]]:
    """Decision views in the frozen houses that still fit inside the server step cap."""
    found = []
    order = {scene: index for index, scene in enumerate(houses)}
    for node in nodes:
        scene = node["scene"]
        if scene not in order:
            continue
        step_index = int(node["step_index"])
        if step_index + BRANCH_STEP_BUDGET + 5 > SERVER_STEP_CAP:
            continue
        exits = paired_exits(node.get("openings") or [])
        if exits is None:
            continue
        found.append(
            {
                "origin_id": f"{scene}|{node['episode_uid']}|{step_index}",
                "role": role_for_scene(scene),
                "scene": scene,
                "episode_uid": node["episode_uid"],
                "step_index": step_index,
                "observation_id": node["observation_id"],
                "position": [float(value) for value in node["position"]],
                "yaw": float(node["yaw"]),
                "split": node["split"],
                "exits": exits,
            }
        )
    found.sort(key=lambda item: (order[item["scene"]], item["step_index"], item["episode_uid"]))
    return found


def pilot_origins(origins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The earliest qualifying origin in each of the first two fit houses."""
    chosen = []
    for scene in FIT_HOUSES[:2]:
        match = next((item for item in origins if item["scene"] == scene), None)
        if match is not None:
            chosen.append(match)
    return chosen


def far_enough(position: list[float], others: list[list[float]], radius_m: float = 0.75) -> bool:
    """A new origin has to leave the places already used."""
    for other in others:
        if math.hypot(position[0] - other[0], position[2] - other[2]) <= radius_m:
            return False
    return True


def collection_order(origins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pilot origins first. The rest stay in frozen house order."""
    pilot = pilot_origins(origins)
    seen = {item["origin_id"] for item in pilot}
    return pilot + [item for item in origins if item["origin_id"] not in seen]


def event_status(
    *,
    restored: bool,
    travel_m: float,
    opening_counts: list[int | None],
    stalled: bool,
    steps: int,
    habitat_ended: bool,
) -> str:
    """Arrival needs two consecutive views of a new junction after leaving the start."""
    if not restored:
        return "restore_failed"
    if habitat_ended and steps < 1:
        return "habitat_ended"
    if len(opening_counts) >= 2:
        previous, current = opening_counts[-2], opening_counts[-1]
        if (
            travel_m >= NEXT_EVENT_MIN_M
            and previous is not None
            and current is not None
            and previous >= 2
            and current >= 2
        ):
            return COMPLETE
    if stalled:
        return "truncated_blocked"
    if habitat_ended:
        return "habitat_ended"
    if steps >= BRANCH_STEP_BUDGET:
        return "truncated_budget"
    return "truncated_unconfirmed"


def transition_is_complete(record: dict[str, Any]) -> bool:
    return record.get("status") == COMPLETE and record.get("event_complete") is True


def integrity_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the delivered rows. This does not decide that a walk reached an event."""
    failures: list[str] = []
    if not roles_are_disjoint():
        failures.append("house roles overlap")
    prefixes: dict[str, str] = {}
    policies: set[str] = set()
    for record in records:
        origin = record["origin_id"]
        role = record["role"]
        if role_for_scene(record["scene"]) != role:
            failures.append(f"{origin} role does not match its house")
        prefix = record.get("prefix_sha256")
        if origin in prefixes and prefixes[origin] != prefix:
            failures.append(f"{origin} does not share one replay prefix")
        elif prefix:
            prefixes[origin] = prefix
        policies.add(record.get("policy_id"))
        pre = record.get("pre_decision") or {}
        leaked = FORBIDDEN_INPUT_FIELDS.intersection(pre)
        if leaked:
            failures.append(f"{origin} pre-decision contains {sorted(leaked)}")
        if record.get("lightnav_used") is not False:
            failures.append(f"{origin} claims the LightNav policy")
        if transition_is_complete(record):
            if record.get("status") != COMPLETE:
                failures.append(f"{origin} complete flag disagrees with status")
            if float(record.get("travel_m") or 0.0) < NEXT_EVENT_MIN_M:
                failures.append(f"{origin} complete event did not leave the start")
    if len(policies) > 1:
        failures.append("more than one execution policy")
    complete = [row for row in records if transition_is_complete(row)]
    by_origin: dict[str, set[int]] = {}
    for row in complete:
        by_origin.setdefault(row["origin_id"], set()).add(int(row["exit_id"]))
    paired = sum(1 for exits in by_origin.values() if len(exits) >= 2)
    return {
        "records": len(records),
        "complete_transitions": len(complete),
        "origins_with_two_complete_exits": paired,
        "passed": not failures,
        "failures": failures,
    }
