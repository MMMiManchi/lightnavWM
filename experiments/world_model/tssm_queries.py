"""Opt-in stateful world-model evidence for the existing task Harness."""

import math

import numpy as np
import torch

from experiments.world_model.event_runtime import BranchEvidence
from experiments.world_model.predictive_harness import PredictiveTaskHarness
from experiments.world_model.task_planning import (
    EnvironmentEvidence,
    TaskBranchDecision,
    TaskPlanningContext,
)


class TSSMBranchQueryService:
    def __init__(self, session, *, context, episode_id, alignment_provider=None):
        self.session, self.context, self.episode_id = session, context, episode_id
        self.last_trace, self.tickets = [], {}
        self.alignment_provider = alignment_provider
        self.requires_observation = False

    def build_context(self, *, instruction, source_observation, branches, raw_history, valid, columns=None, progress=""):
        branches = tuple(branches)
        self.current_branches = branches
        if len({b.candidate_id for b in branches}) != len(branches) or any(b.source_observation != source_observation for b in branches):
            raise ValueError("branch proposals must be distinct and grounded in this current view")
        if self.session.episode_id != self.episode_id or getattr(self.session, "context", None) != self.context:
            self.session.begin_episode(self.context, self.episode_id, raw_history, source_observation,
                                       instruction=instruction, valid=valid)
        if self.session.episode_id != self.episode_id or self.session.current_ref != source_observation:
            raise ValueError("actual movement feedback is required before planning at a new observation")
        bank = self.session.bank
        cutoff = bank.latest_sequence(self.context)
        current = self.session.state.current[0].cpu().numpy()
        environment = bank.nearest_observed_sources(context=self.context, before_sequence=cutoff,
            source_feature=current, exclude_episode=self.session.excluded_episode, source_floor=0.5)
        self.copy_matches = bank.query_actual_outcomes(context=self.context, before_sequence=cutoff,
            predicted_future=current, source_feature=current, branch_direction=[0, 1], exclude_episode=self.session.excluded_episode)
        if not self.session.use_memory:
            environment, self.copy_matches = [], []
        results, trace = [], []
        self.tickets = {}
        self.requires_observation = False
        for b in branches:
            direction = np.array([math.sin(b.bearing_rad), math.cos(b.bearing_rad)], dtype=np.float32)
            transform = self.alignment_provider.for_branch(source_observation, b) if self.alignment_provider is not None else None
            ticket = self.session.forecast(f"{source_observation}:{b.candidate_id}", direction, direction_transform=transform, columns=columns)
            evidence = self.session.query_actual_memories(ticket)
            self.tickets[b.candidate_id] = ticket
            self.requires_observation |= ticket.future_feature is None
            if ticket.future_feature is not None:
                result = BranchEvidence(ticket.future_feature, ticket.memory_sequences, tuple(evidence),
                    ticket.visible_sequence, bool(ticket.memory_sequences), "tssm_" + ticket.evidence_kind,
                    source_observation, tuple(float(x) for x in direction))
                results.append((b.candidate_id, result))
            trace.append({"candidate_id": b.candidate_id, "query_kind": "tssm_" + ticket.evidence_kind,
                "world_model_called": ticket.world_model_called, "route_reason": ticket.route_reason,
                "observation_required": ticket.future_feature is None,
                "memory_cutoff": ticket.visible_sequence, "adapter_version": ticket.adapter_version,
                "adapter_applied": ticket.adapter_applied,
                "actual_current_episode_history_allowed": self.session.current_episode_memory,
                "source_state_cache_steps": ticket.start_state.tokens.shape[1],
                "past_sequences": list(ticket.memory_sequences),
                "retrieved_sequences": [m.sequence for m in evidence],
                "copy_retrieved_sequences": [m.sequence for m in self.copy_matches],
                "prediction_changes_top_retrieval": bool(evidence and self.copy_matches and
                    evidence[0].sequence != self.copy_matches[0].sequence),
                "retrieved": [{"sequence": m.sequence, "actual_source": m.source_observation,
                    "outcome_kind": m.outcome_kind,
                    "actual_outcome": m.target_observation, "future_similarity": m.future_similarity,
                    "relation": "past_evidence_not_a_confirmed_current_connection"} for m in evidence]})
        self.last_trace = trace
        return TaskPlanningContext(self.episode_id, instruction, source_observation, cutoff,
            EnvironmentEvidence(tuple(environment)), branches, tuple(results), progress,
            allow_current_episode_history=self.session.current_episode_memory)

    def commit(self, candidate_id):
        self.session.commit(self.tickets[candidate_id])

    def observe_route(self, *args, **kwargs):
        return self.session.observe_route(*args, **kwargs)

    def observe_event(self, candidate_id, raw_target, **proof):
        return self.session.observe_event(self.tickets[candidate_id], raw_target, **proof)


class StatefulTSSMHarness(PredictiveTaskHarness):
    """Explicit opt-in bridge; actual milestone/event feedback drives updates."""

    def __init__(self, *, session, context, episode_id, executor, images, proposer, planner, progress_tracker=None,
                 alignment_provider=None, exploration_overrides=True, prediction_control=None):
        service = TSSMBranchQueryService(session, context=context, episode_id=episode_id, alignment_provider=alignment_provider)
        super().__init__(executor=executor, predictor=session.predictor, bank=session.bank,
            memory_context=context, episode_id=episode_id, images=images, proposer=proposer, planner=planner,
            progress_tracker=progress_tracker, query_service=service)
        self.committed_candidate = None
        self.monitor_candidate = None
        self.exploration_overrides = exploration_overrides
        self.prediction_control = prediction_control
        self.control_trace = {}
        self.router.planner = self._guarded_plan

    def _guarded_plan(self, context):
        self.control_trace = {}
        if self.query.requires_observation:
            return TaskBranchDecision(context.task_key, "", "", action="observe", observation_turn_rad=math.pi / 6,
                                      reason="Historical place/exit evidence is ambiguous; inspect the actual scene first.")
        decision = self.planner(context)
        if decision is not None and decision.action == "navigate" and not self.exploration_overrides:
            if decision.task_key != context.task_key or not any(b.candidate_id == decision.candidate_id for b in context.current_branches):
                raise ValueError("unbound exploration proposal")
            evidence = dict(context.predicted_branches).get(decision.candidate_id)
            if evidence is None or evidence.query_kind != "tssm_observed_memory_candidate":
                if self.prediction_control is not None:
                    self.control_trace = self.prediction_control.assess(context, decision, self.query.copy_matches,
                        self.query.session.state.current[0].cpu().numpy())
                    if self.control_trace["accepted"]:
                        return decision
                self.monitor_candidate = decision.candidate_id
                return TaskBranchDecision(context.task_key, "", "", action="continue_global",
                    reason="Unvalidated future is monitored during original-policy execution; it does not override the task.")
        return decision

    def visual_history(self):
        if not self.features:
            raise ValueError("planning needs an actual current observation")
        end = len(self.features) - 1
        prior = [i for i in self.planning_anchors if i < end][-4:]
        indices = sorted(set(prior + list(range(max(0, end - 3), end + 1))))[-8:]
        missing = 8 - len(indices)
        rows = [np.zeros_like(self.features[-1]) for _ in range(missing)] + [self.features[i] for i in indices]
        device = self.predictor.feature_mean.device
        return torch.from_numpy(np.stack(rows))[None].to(device), torch.tensor([[False] * missing + [True] * len(indices)], device=device)

    def plan(self):
        self.monitor_candidate = None
        result = super().plan()
        result.trace["prediction_control"] = dict(self.control_trace)
        result.trace["control_source"] = "prediction" if self.control_trace.get("accepted") else (
            "observed_memory" if result.goal is not None and not self.exploration_overrides else
            "exploration" if result.goal is not None else "original_policy")
        decision = self.router.last_decision
        if result.goal is not None and decision is not None and decision.action == "navigate":
            self.query.commit(decision.candidate_id)
            self.committed_candidate = decision.candidate_id
        elif self.monitor_candidate is not None:
            self.query.commit(self.monitor_candidate)
            self.committed_candidate = self.monitor_candidate
        return result

    def report_observed_milestone(self, raw_feature, reference, *, execution_ref, executed, direction=(0.0, 1.0)):
        # Called on actual route milestones, not every 0.1-second camera frame.
        return self.query.observe_route(raw_feature, reference, execution_ref=execution_ref,
                                        executed=executed, direction=direction)

    def report_verified_event(self, raw_target, **proof):
        if self.committed_candidate is None:
            raise ValueError("no current branch execution was committed")
        result = self.query.observe_event(self.committed_candidate, raw_target, **proof)
        self.committed_candidate = None
        return result

    def abandon_event(self, reason):
        if self.committed_candidate is not None:
            self.query.session.abort_event(self.query.tickets[self.committed_candidate], reason)
            self.committed_candidate = None
        self.executor.restore_global_goal()
