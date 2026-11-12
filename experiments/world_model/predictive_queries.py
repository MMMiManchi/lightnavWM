"""One causal query per current visible branch, consumed by task planning."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import numpy as np
import torch

from experiments.world_model.event_runtime import BranchEvidence, predict_branch_with_memory
from experiments.world_model.task_planning import EnvironmentEvidence, TaskPlanningContext


class BranchQueryService:
    def __init__(self, predictor, bank, *, memory_context, episode_id, visible_sequence, mode="predicted", alignment_provider=None):
        if mode not in ("predicted", "direct", "shuffled"):
            raise ValueError("unknown branch query mode")
        if predictor.feature_contract_id != bank.contract_id:
            raise ValueError("query and memory feature contracts differ")
        self.predictor, self.bank = predictor, bank
        self.memory_context, self.episode_id = memory_context, episode_id
        self.visible_sequence, self.mode = visible_sequence, mode
        self.last_trace = []
        self.alignment_provider = alignment_provider

    @torch.inference_mode()
    def build_context(self, *, instruction, source_observation, branches, raw_history, valid, columns, progress=""):
        branches = tuple(branches)
        if any(b.source_observation != source_observation for b in branches):
            raise ValueError("current visual candidates are stale")
        if len({b.candidate_id for b in branches}) != len(branches):
            raise ValueError("duplicate current visual candidates")
        current = self.predictor.normalize(raw_history[:, -1])[0].cpu().numpy()
        environment = self.bank.nearest_observed_sources(context=self.memory_context, before_sequence=self.visible_sequence,
                         source_feature=current, exclude_episode=self.episode_id, source_floor=.5)
        results = []
        directions = []
        for branch in branches:
            direction = np.array([math.sin(branch.bearing_rad), math.cos(branch.bearing_rad)], dtype=np.float32)
            directions.append(direction)
            tensor = torch.from_numpy(direction[None]).to(raw_history.device)
            transport = None if self.alignment_provider is None else self.alignment_provider.for_branch(source_observation,branch)
            if self.mode == "direct":
                matches = self.bank.retrieve(context=self.memory_context, before_sequence=self.visible_sequence,
                          source_feature=current, predicted_future=current, branch_direction=direction,
                          exclude_episode=self.episode_id, query_kind="source",direction_transform=transport)
                evidence = BranchEvidence(current.copy(), tuple(m.sequence for m in matches), tuple(matches),
                                          self.visible_sequence, bool(matches), "direct_history")
            else:
                evidence = predict_branch_with_memory(self.predictor, self.bank, context=self.memory_context,
                            episode_id=self.episode_id, visible_sequence=self.visible_sequence, raw_history=raw_history,
                            valid=valid, branch=tensor, columns=columns, outcome_key_query=True,direction_transform=transport)
            results.append(replace(evidence, source_observation=source_observation,
                                   branch_direction=tuple(float(x) for x in direction)))
        if self.mode == "shuffled" and len(results) > 1:
            original = [r.future_feature.copy() for r in results]
            for i, evidence in enumerate(results):
                query = original[(i+1) % len(results)]
                matches = self.bank.query_actual_outcomes(context=self.memory_context, before_sequence=self.visible_sequence,
                            predicted_future=query, source_feature=current, branch_direction=directions[i],
                            exclude_episode=self.episode_id) if evidence.supported_by_history else []
                results[i] = replace(evidence, future_feature=query, retrieved_evidence=tuple(matches),
                                     query_kind="shuffled_prediction_control")
        self.last_trace = [{"candidate_id": b.candidate_id, "current_bearing_rad": b.bearing_rad,
                            "query_kind": r.query_kind, "supported_by_history": r.supported_by_history,
                            "query_sha256": hashlib.sha256(r.future_feature.tobytes()).hexdigest(),
                            "past_transition_sequences": list(r.past_transition_sequences),
                            "retrieved": [{"sequence": m.sequence, "episode": m.episode_id,
                                           "actual_source": m.source_observation, "actual_outcome": m.target_observation,
                                           "score": m.score, "future_similarity": m.future_similarity,
                                           "relation": "retrieved_evidence_not_a_confirmed_current_connection"}
                                          for m in r.retrieved_evidence]}
                           for b, r in zip(branches, results, strict=True)]
        return TaskPlanningContext(self.episode_id, instruction, source_observation, self.visible_sequence,
                                   EnvironmentEvidence(tuple(environment)), branches,
                                   tuple((b.candidate_id, r) for b, r in zip(branches, results, strict=True)), progress)
