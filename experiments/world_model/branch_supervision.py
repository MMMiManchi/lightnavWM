"""Split one exploration batch into node labels and consequence labels.

Neither stream is an expert action for the current instruction. A blocked walk
records the blockage and does not invent a connection to the next place.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

SKILL = "follow_visible_exit_until_next_event"


def _travel(start: list[float], position: list[float]) -> float:
    return float(math.hypot(position[0] - start[0], position[2] - start[2]))


def frame_phases(frames: list[dict[str, Any]], event_complete: bool) -> list[str]:
    """Mark the start, the departure, the travel, and a confirmed next event."""
    if not frames:
        return []
    start = frames[0]["position"]
    confirming = set()
    if event_complete and len(frames) >= 2:
        tail = frames[-2:]
        if all((frame.get("opening_count") or 0) >= 2 for frame in tail):
            confirming = {len(frames) - 2, len(frames) - 1}
    phases = []
    left = False
    for index, frame in enumerate(frames):
        if index == 0:
            phases.append("start")
            continue
        if index in confirming:
            phases.append("next_event")
            continue
        travel = _travel(start, frame["position"])
        openings = frame.get("opening_count")
        if not left and travel >= 0.4 and (openings or 0) < 2:
            left = True
            phases.append("leaving")
            continue
        phases.append("transit")
    return phases


def node_label(opening_count: int | None, phase: str) -> str:
    """Depth openings are a proxy. An official id is not the class."""
    if opening_count is None:
        return "unknown"
    if phase in {"start", "next_event"} and opening_count >= 2:
        return "at_decision"
    if phase == "leaving":
        return "leaving_decision"
    if opening_count >= 2:
        return "possible_decision"
    if opening_count == 1:
        return "single_exit_or_boundary"
    return "no_forward_opening"


def execution_relation(status: str | None) -> tuple[str, str]:
    """Requested skill versus what the body did. This is not a route label."""
    if status == "complete":
        return "followed_to_event", "observed_next_event"
    if status == "truncated_blocked":
        return "blocked", "no_topological_link"
    if status in {"restore_failed", "error"}:
        return "not_started", "no_topological_link"
    return "incomplete", "no_topological_link"


def consequence_from_execution(record: dict[str, Any]) -> dict[str, Any]:
    relation, link = execution_relation(record.get("status"))
    chosen = (record.get("pre_decision") or {}).get("chosen_exit") or {}
    return {
        "supervision": "future_consequence",
        "instruction_action_label": False,
        "follows_task_route": False,
        "requested_skill": {
            "skill": SKILL,
            "policy_id": record.get("policy_id"),
            "exit_id": record.get("exit_id"),
            "side": chosen.get("side"),
            "bearing_left_rad": chosen.get("bearing_left_rad"),
        },
        "execution_relation": relation,
        "link": link,
        "origin_id": record.get("origin_id"),
        "role": record.get("role"),
        "scene": record.get("scene"),
        "exit_id": record.get("exit_id"),
        "repeat_index": record.get("repeat_index"),
        "status": record.get("status"),
        "event_complete": record.get("event_complete") is True,
    }


def node_rows_from_frames(
    frames: list[dict[str, Any]],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    phases = frame_phases(frames, record.get("event_complete") is True)
    rows = []
    for frame, phase in zip(frames, phases):
        openings = frame.get("opening_count")
        rows.append(
            {
                "supervision": "node_discovery",
                "instruction_action_label": False,
                "label": node_label(openings, phase),
                "phase": phase,
                "label_source": "depth_opening_count",
                "official_id_used_as_class": False,
                "opening_count": openings,
                "frame": frame.get("name"),
                "origin_id": record.get("origin_id"),
                "role": record.get("role"),
                "scene": record.get("scene"),
                "exit_id": record.get("exit_id"),
                "repeat_index": record.get("repeat_index"),
            }
        )
    return rows


def build_supervision(
    records: list[dict[str, Any]],
    frames_for,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build both label streams. Frames are loaded by the caller."""
    nodes: list[dict[str, Any]] = []
    consequences: list[dict[str, Any]] = []
    for record in records:
        if "origin_id" not in record:
            continue
        consequences.append(consequence_from_execution(record))
        if record.get("status") in {None, "error", "restore_failed"}:
            continue
        if not record.get("evidence_dir"):
            continue
        nodes.extend(node_rows_from_frames(frames_for(record), record))
    return nodes, consequences, supervision_counts(nodes, consequences)


def supervision_counts(node_rows: list[dict[str, Any]], consequences: list[dict[str, Any]]) -> dict[str, Any]:
    visible = [row for row in node_rows if row.get("role") != "sealed_test"]
    visible_links = [row for row in consequences if row.get("role") != "sealed_test"]
    return {
        "node_frames": len(node_rows),
        "node_label_counts": dict(Counter(row["label"] for row in visible)),
        "phase_counts": dict(Counter(row["phase"] for row in visible)),
        "consequence_links": dict(Counter(row["link"] for row in visible_links)),
        "execution_relations": dict(Counter(row["execution_relation"] for row in visible_links)),
        "sealed_node_frames": sum(row.get("role") == "sealed_test" for row in node_rows),
        "sealed_consequences": sum(row.get("role") == "sealed_test" for row in consequences),
        "instruction_action_labels": sum(
            row.get("instruction_action_label") is True for row in node_rows + consequences
        ),
    }
