"""Event-driven collection: no per-branch horizon, resumable resource pauses."""

from __future__ import annotations

import fcntl
import json
import math
import os
from collections import deque
from pathlib import Path

import numpy as np

from experiments.world_model.branch_curation import CurationRules, assign_region

TRANSIT_CODES = frozenset({"h", "s", "e"})
OUTDOOR_CODES = frozenset({"p", "y"})
UNKNOWN_CODES = frozenset({"-", "x", ""})


class ResourcePause(RuntimeError):
    """The logical trajectory remains unfinished and can be resumed."""


class StepAccounting:
    """A shared accounting ledger; limit=None tracks work without limiting it."""

    def __init__(self, root: Path):
        self.root = root

    def initialize(self, limit=None, prior_steps=0):
        self.root.mkdir(parents=True, exist_ok=True)
        if limit is not None and limit < prior_steps:
            raise ValueError("resource allowance is below work already performed")
        with (self.root / "step_accounting.json").open("x") as stream:
            json.dump({"limit": limit, "reserved_steps": prior_steps, "prior_steps": prior_steps}, stream)

    def reserve(self, task_id):
        with (self.root / ".accounting.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / "step_accounting.json"
            record = json.loads(path.read_text())
            if record["limit"] is not None and record["reserved_steps"] >= record["limit"]:
                raise ResourcePause("optional batch allowance exhausted; trajectory saved for continuation")
            record["reserved_steps"] += 1
            record["last_task"] = task_id
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record) + "\n")
            os.replace(temporary, path)


def execute_once(env, command, action_id, accounting=None, task_id=None):
    seen = getattr(env, "_collection_seen_action_ids", None)
    if seen is None:
        seen = env._collection_seen_action_ids = set()
    if action_id in seen:
        raise RuntimeError("collection action id already executed")
    if accounting is not None:
        accounting.reserve(task_id)
    outcome = env.step({"wm_action_id": action_id, **command})
    ack = outcome[-1].get("wm_execution") or {}
    if ack.get("executed") is not True or ack.get("replayed") is True:
        raise RuntimeError("new executed-action acknowledgement missing")
    seen.add(action_id)
    # The generic server retains full observations for RPC retry. This local,
    # synchronous collector retains its own id guard and bounded image cache.
    cache = getattr(env, "_by_id", None)
    if cache is not None:
        while len(cache) > 64:
            oldest = next(iter(cache))
            if cache[oldest].get("status") != "executed":
                break
            del cache[oldest]
    return outcome


class EventProgress:
    """Incremental observed-state labels; hallways and stairs are intermediate."""

    def __init__(self, first_frame, polygons, landmarks=(), source_origin=None):
        self.rules = CurationRules()
        self.polygons = polygons
        self.start = np.array(first_frame["position"], dtype=float)
        self.previous = self.start.copy()
        self.source_region = assign_region(self.start, polygons, self.rules)
        if self.source_region is None:
            raise ValueError("event source region is ambiguous")
        self.window = deque(maxlen=5)
        self.landmark_window = deque(maxlen=3)
        self.path_m = 0.0
        self.farthest_m = 0.0
        self.steps = 0
        self.region_events = []
        self.last_stable_region = self.source_region["region_id"]
        self.landmarks = [p for p in landmarks if p["origin_id"] != source_origin
                          and np.linalg.norm(np.array(p["position"]) - self.start) >= 1.5]
        self.visits = {}

    def push(self, frame, entrance_committed):
        position = np.array(frame["position"], dtype=float)
        self.path_m += float(np.linalg.norm(position - self.previous))
        self.previous = position
        self.steps += 1
        distance = float(np.linalg.norm(position - self.start))
        self.farthest_m = max(self.farthest_m, distance)
        region = assign_region(position, self.polygons, self.rules)
        self.window.append(region)
        stable = (len(self.window) == 5 and region is not None
                  and all(r and r["region_id"] == region["region_id"] for r in self.window))
        if stable and region["region_id"] != self.last_stable_region:
            self.region_events.append({"index": self.steps, "region": region, "path_m": self.path_m})
            self.last_stable_region = region["region_id"]
        if not entrance_committed or self.path_m < 1.0:
            return None
        if stable and region["region_id"] != self.source_region["region_id"] and region["clearance_m"] >= 0.3:
            code = region["category_code"]
            if code not in TRANSIT_CODES | OUTDOOR_CODES | UNKNOWN_CODES:
                return self._event("next_room", region)
            if code in OUTDOOR_CODES and self.source_region["category_code"] not in OUTDOOR_CODES:
                return self._event("outdoor_place", region)
        # Reviewed locations are offline label landmarks, never navigation targets.
        match = next((p for p in self.landmarks if region is not None
                      and p.get("source_region_id") == region["region_id"]
                      and np.linalg.norm(np.array(p["position"]) - position) <= 0.60), None)
        self.landmark_window.append(match["origin_id"] if match else None)
        if (distance >= 1.5 and len(self.landmark_window) == 3 and match
                and all(x == match["origin_id"] for x in self.landmark_window)):
            return self._event("next_reviewed_decision", region, landmark_origin_id=match["origin_id"])
        # A physical revisit is a recorded execution outcome, not a claim that
        # the map has no other exits. Repeated frames or turning cannot trigger it.
        key = tuple(np.floor(position / 0.4).astype(int).tolist())
        old = self.visits.get(key)
        if old is not None and self.path_m - old["path_m"] >= 5.0 and self.farthest_m >= 1.5:
            return self._event("observed_revisit", region, topological_loop_claim=False)
        self.visits.setdefault(key, {"path_m": self.path_m, "index": self.steps})
        return None

    def _event(self, kind, region, **extra):
        return {"kind": kind, "target_index": self.steps, "source_region": self.source_region,
                "target_region": region, "path_m": self.path_m,
                "control_horizon_s": self.steps * 0.1, **extra}


def turn_command(error_rad):
    # Converge to an angle, rather than rotate for a fixed number of steps.
    dps = max(-60.0, min(60.0, math.degrees(error_rad) / 0.1))
    return {"action": "velocity_control", "action_args": {"linear_velocity": -1.0,
                                                           "angular_velocity": dps / 300.0}}
