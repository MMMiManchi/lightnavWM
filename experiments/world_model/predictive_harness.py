"""A current-task planning round that consumes per-branch future queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.world_model.predictive_queries import BranchQueryService
from experiments.world_model.task_planning import TaskGoalRouter


@dataclass(frozen=True)
class PlanningRound:
    goal: object
    action: str
    observation_turn_rad: float
    trace: dict


class PredictiveTaskHarness:
    def __init__(self, *, executor, predictor, bank, memory_context, episode_id,
                 images, proposer, planner, query_mode="predicted", alignment_provider=None, progress_tracker=None,
                 query_service=None):
        self.executor, self.predictor = executor, predictor
        self.images, self.proposer, self.planner = images, proposer, planner
        self.router = TaskGoalRouter(executor, episode_id, planner)
        self.query = query_service or BranchQueryService(predictor, bank, memory_context=memory_context,
                    episode_id=episode_id, visible_sequence=bank.latest_sequence(memory_context), mode=query_mode,
                    alignment_provider=alignment_provider)
        self.progress_tracker=progress_tracker
        self.observations = []
        self.features = []
        self.columns = None
        self.planning_anchors = []
        self.feedback = []
        self.last_trace = {}

    def observe(self, reference, rgb, raw_feature, columns):
        self.images.put(reference, rgb)
        if self.observations and self.observations[-1] == reference:
            if not np.array_equal(self.features[-1], np.asarray(raw_feature)):
                raise ValueError("an observed frame cannot acquire different features")
            return
        if reference in self.observations:
            raise ValueError("cannot move episode history backward")
        self.observations.append(reference)
        self.features.append(np.asarray(raw_feature, dtype=np.float32).copy())
        self.columns = columns

    def report_execution(self, message):
        # Tool feedback records what happened, never asserts completed goals.
        self.feedback.append(str(message))

    def visual_history(self):
        if not self.features:
            raise ValueError("planning needs a current actual observation")
        dim = len(self.features[-1])
        prior = [i for i in self.planning_anchors if i < len(self.features)-1][-4:]
        rows = [np.zeros(dim, dtype=np.float32)]*(4-len(prior)) + [self.features[i] for i in prior]
        valid = [False]*(4-len(prior)) + [True]*len(prior)
        end = len(self.features)-1
        rows += [self.features[max(0, end-offset)] for offset in (12, 8, 4, 0)]
        valid += [True]*4
        device = self.predictor.feature_mean.device
        return torch.from_numpy(np.stack(rows))[None].to(device), torch.tensor([valid], device=device)

    def plan(self):
        reference = self.observations[-1]
        self.last_trace = {"source_observation": reference, "current_instruction": self.executor.original_instruction}
        perception = self.proposer.propose(reference)
        self.last_trace.update(perception_event=perception.event, perception_raw_output=perception.raw_response,
            current_branches=[{"id": b.candidate_id, "bearing_rad": b.bearing_rad,
                               "description": b.description, "bbox": b.image_region} for b in perception.branches])
        history, valid = self.visual_history()
        context = self.query.build_context(instruction=self.executor.original_instruction,
                    source_observation=reference, branches=perception.branches,
                    raw_history=history, valid=valid, columns=self.columns.to(history.device),
                    progress=((self.progress_tracker.context()+"\n") if self.progress_tracker is not None else "")+
                             ("\n".join(self.feedback[-8:]) or "No local subgoal completion has been verified."))
        self.last_trace.update(task_key=context.task_key, progress=context.progress, queries=self.query.last_trace)
        if hasattr(self.planner, "episode_history"):
            # A few actual past decision views, explicitly labeled by episode.
            self.planner.episode_history = tuple(self.observations[i] for i in self.planning_anchors[-2:])
        try:
            goal = self.router.consider(context)
        finally:
            self.last_trace["planner"] = getattr(self.planner, "last_trace", {})
        decision = self.router.last_decision
        trace = {**self.last_trace, "routing_status": self.router.last_status,
                 "selected_candidate": decision.candidate_id if decision is not None else None,
                 "used_evidence_sequences": list(decision.evidence_sequences) if decision is not None else [],
                 "action": decision.action if decision is not None else "continue_global",
                 "reason": decision.reason if decision is not None else "planner abstained",
                 "anchors_are_actual_decision_views_not_verified_topological_nodes": True}
        index = len(self.features)-1
        if not self.planning_anchors or self.planning_anchors[-1] != index:
            self.planning_anchors.append(index)
        return PlanningRound(goal, trace["action"], decision.observation_turn_rad if decision is not None else 0.0, trace)
