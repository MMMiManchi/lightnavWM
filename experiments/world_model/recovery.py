"""Finite, evidence-scoped recovery; unknown is not a failed goal."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    reason: str
    angle: float = 0.0


class RecoveryManager:
    def __init__(self, *, observation_budget=8, unknown_stop_policy="defer", maximum_cycles=3):
        if unknown_stop_policy not in {"defer", "abstain"}:
            raise ValueError("unknown STOP policy must be defer or abstain")
        self.observation_budget, self.unknown_stop_policy = observation_budget, unknown_stop_policy
        self.maximum_cycles, self.version = maximum_cycles, 0
        self.phase_observations = self.total_observations = self.cycles = self.unknown_stops = 0
        self.explicit_rejection = False
        self.progress_reference, self.stage = None, 0
        self.history = []
        self.scan_offset = 0.0

    def observe_progress(self, feedback, stage):
        pose = feedback.get("local_pose")
        current = (feedback.get("pose_epoch"), pose[:2]) if pose is not None else None
        progressed = stage > self.stage
        if current is not None and self.progress_reference is not None and current[0] == self.progress_reference[0]:
            a, b = current[1], self.progress_reference[1]
            progressed |= math.hypot(a[0]-b[0], a[1]-b[1]) >= .6
        if progressed:
            self.phase_observations = self.cycles = self.unknown_stops = 0
            self.explicit_rejection = False
            self.scan_offset = 0.0
            self.progress_reference = current
            self.version += 1
        elif current is not None and (self.progress_reference is None or self.progress_reference[0] != current[0]):
            self.progress_reference = current
        self.stage = stage

    def evidence(self, verdict):
        if verdict == "not_reached":
            self.explicit_rejection = True
        elif verdict == "reached":
            self.explicit_rejection = False

    def observation(self, *, reason, target_bearing=None):
        if self.phase_observations >= self.observation_budget:
            return None
        n = self.phase_observations
        target = math.copysign(math.pi/6, target_bearing) if n == 0 and target_bearing else math.radians(30*(n//2+1))*(1 if n % 2 == 0 else -1)
        angle = math.atan2(math.sin(target-self.scan_offset), math.cos(target-self.scan_offset))
        if abs(angle) < math.radians(15):
            # Repeated requested bearings cannot provide a second view. In
            # Habitat a zero-velocity "look" also silently becomes STOP.
            angle = math.copysign(math.pi/6, 1 if n % 2 == 0 else -1)
            target = math.atan2(math.sin(self.scan_offset+angle), math.cos(self.scan_offset+angle))
        self.scan_offset = target
        self.phase_observations += 1
        self.total_observations += 1
        self.version += 1
        result = RecoveryDecision("observe", reason, angle)
        self.history.append({"version": self.version, "action": result.action, "reason": reason})
        return result

    def on_global_stop(self, *, verdict, target_bearing=None, can_backtrack=False):
        self.evidence(verdict)
        if verdict == "reached":
            return RecoveryDecision("await_confirmation", "positive goal evidence needs its temporal confirmation")
        self.unknown_stops += 1
        # The original policy keeps authority on uncertainty after real recovery
        # views. This is an unverified policy STOP, never a semantic completion.
        if (verdict == "unknown" and not self.explicit_rejection and self.unknown_stops >= 3 and
                self.phase_observations >= min(2, self.observation_budget)):
            return RecoveryDecision("defer_stop" if self.unknown_stop_policy == "defer" else "abstain",
                                    "original policy STOP remains unverified after bounded observations")
        if self.phase_observations < min(2*(self.cycles+1), self.observation_budget):
            return self.observation(reason="resolve_identity_or_position_uncertainty", target_bearing=target_bearing)
        if self.cycles < self.maximum_cycles:
            self.cycles += 1
            self.version += 1
            return RecoveryDecision("backtrack" if can_backtrack and self.cycles >= 2 else "relocalize",
                                    "additional views did not resolve the current hypothesis")
        return RecoveryDecision("abstain", "goal contradicted or unresolved after distinct recovery attempts")

    def snapshot(self):
        return {"version": self.version, "phase_observations": self.phase_observations,
            "total_observations": self.total_observations, "cycles": self.cycles,
            "unknown_stop_proposals": self.unknown_stops, "explicit_goal_contradiction": self.explicit_rejection,
            "unknown_stop_policy": self.unknown_stop_policy, "history": self.history[-12:]}
