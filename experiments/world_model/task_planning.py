"""Scene experience is evidence; the current task owns every route decision.

There is intentionally no default planner or nearest-experience action policy.
A planner must choose among branches grounded in the current observation and
return a decision bound to the current episode and original instruction.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable

from experiments.world_model.event_runtime import BranchEvidence, HarnessLightNavExecutor, LocalGoal
from experiments.world_model.scene_memory import MemoryMatch


@dataclass(frozen=True)
class EnvironmentEvidence:
    """Actual observations/outcomes, with old bearings in their original frame.

    Neither the previous task instruction nor a recommended next action is
    included. Historical bearings require alignment before reuse as a current
    direction; they are not current visible-exit candidates.
    """

    experiences: tuple[MemoryMatch, ...]


@dataclass(frozen=True)
class CurrentBranch:
    candidate_id: str
    source_observation: str
    bearing_rad: float
    description: str
    image_region: tuple[int, int, int, int] | None = None
    entry_point: tuple[float, float] | None = None

    def __post_init__(self):
        if not self.candidate_id or not self.source_observation or not self.description:
            raise ValueError("a current branch needs current visual grounding")
        if not math.isfinite(self.bearing_rad) or abs(self.bearing_rad) >= math.pi / 3:
            raise ValueError("a current branch must lie in the current camera cone")
        if self.image_region is not None:
            x1, y1, x2, y2 = self.image_region
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                raise ValueError("invalid normalized image region")
        if self.entry_point is not None and not all(math.isfinite(x) and 0 < x < 1000 for x in self.entry_point):
            raise ValueError("invalid current entrance point")


@dataclass(frozen=True)
class TaskPlanningContext:
    episode_id: str
    instruction: str
    source_observation: str
    visible_sequence: int
    environment: EnvironmentEvidence
    current_branches: tuple[CurrentBranch, ...] = ()
    predicted_branches: tuple[tuple[str, BranchEvidence], ...] = ()
    progress: str = ""
    allow_current_episode_history: bool = False

    def __post_init__(self):
        if not self.episode_id or not self.instruction.strip() or not self.source_observation:
            raise ValueError("current task and observation are required")
        candidates = {b.candidate_id for b in self.current_branches}
        if len(candidates) != len(self.current_branches):
            raise ValueError("duplicate current branch ids")
        if any(b.source_observation != self.source_observation for b in self.current_branches):
            raise ValueError("current branches cannot come from a stale observation")
        if any(m.sequence > self.visible_sequence or (m.episode_id == self.episode_id and not self.allow_current_episode_history)
               for m in self.environment.experiences):
            raise ValueError("environment evidence must be visible prior-episode experience")
        predicted_ids = [key for key, _ in self.predicted_branches]
        if len(set(predicted_ids)) != len(predicted_ids):
            raise ValueError("duplicate branch predictions")
        for key, prediction in self.predicted_branches:
            if key not in candidates or prediction.visible_sequence > self.visible_sequence:
                raise ValueError("prediction is not bound to a visible current branch/context")
            if any(sequence > self.visible_sequence for sequence in prediction.past_transition_sequences):
                raise ValueError("prediction references future experience")
            if any(m.sequence > self.visible_sequence or (m.episode_id == self.episode_id and not self.allow_current_episode_history)
                   for m in prediction.retrieved_evidence):
                raise ValueError("prediction retrieved invisible experience")
            if prediction.source_observation is not None and prediction.source_observation != self.source_observation:
                raise ValueError("prediction is from a stale observation")
            if prediction.branch_direction is not None:
                branch = next(b for b in self.current_branches if b.candidate_id == key)
                offset = math.atan2(*prediction.branch_direction)
                if abs(offset - branch.bearing_rad) > 1e-4:
                    raise ValueError("prediction belongs to another current direction")

    @property
    def task_key(self):
        return hashlib.sha256((self.episode_id + "\0" + self.instruction).encode()).hexdigest()


@dataclass(frozen=True)
class TaskBranchDecision:
    task_key: str
    candidate_id: str
    local_instruction: str
    evidence_sequences: tuple[int, ...] = ()
    action: str = "navigate"
    reason: str = ""
    observation_turn_rad: float = 0.0


TaskPlanner = Callable[[TaskPlanningContext], TaskBranchDecision | None]


class TaskGoalRouter:
    def __init__(self, executor: HarnessLightNavExecutor, episode_id: str,
                 planner: TaskPlanner | None = None):
        self.executor, self.episode_id, self.planner = executor, episode_id, planner
        self.last_status = "no_task_planner" if planner is None else "ready"
        self.last_decision = None

    def consider(self, context: TaskPlanningContext) -> LocalGoal | None:
        self.last_decision = None
        if context.episode_id != self.episode_id or context.instruction != self.executor.original_instruction:
            raise ValueError("planning context belongs to another episode or task")
        if self.planner is None:
            self.last_status = "no_task_planner"
            return None
        decision = self.planner(context)
        self.last_decision = decision
        if decision is None:
            self.last_status = "planner_abstained"
            return None
        if decision.task_key != context.task_key:
            raise ValueError("decision is from another episode or task")
        available = {m.sequence for m in context.environment.experiences}
        for _key, prediction in context.predicted_branches:
            available.update(m.sequence for m in prediction.retrieved_evidence
                             if m.sequence <= context.visible_sequence)
        if not set(decision.evidence_sequences).issubset(available):
            raise ValueError("decision cites unavailable experience")
        if decision.action == "observe":
            if not math.isfinite(decision.observation_turn_rad) or abs(decision.observation_turn_rad) > math.pi / 6 + 1e-6:
                raise ValueError("active observation exceeds the bounded turn")
            self.last_status = "observation_requested"
            return None
        if decision.action == "continue_global":
            self.executor.restore_global_goal()
            self.last_status = "continue_global_requested"
            return None
        if decision.action != "navigate":
            raise ValueError("unsupported task action")
        branch = next((b for b in context.current_branches if b.candidate_id == decision.candidate_id), None)
        if branch is None:
            raise ValueError("a remembered direction is not a grounded current branch")
        goal = LocalGoal(decision.local_instruction, context.source_observation,
                         decision.evidence_sequences, branch.bearing_rad)
        self.executor.commit_goal(goal, current_observation=context.source_observation,
                                  visible_sequence=context.visible_sequence)
        self.last_status = "task_goal_committed"
        return goal
