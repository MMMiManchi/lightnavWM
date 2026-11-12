"""Cross-episode history for a frozen future-feature predictor.

    predict -> execute -> observe actual outcome

No model parameters are updated here. The returned memory and fused feature
are explicitly separate from the base model prediction.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import numpy as np
import torch

from experiments.world_model.scene_memory import MemoryMatch, PersistentSceneMemory, unit_vector


@dataclass(frozen=True)
class MemoryPrediction:
    prediction_id: str
    base_future: np.ndarray
    future_feature: np.ndarray
    matches: tuple[MemoryMatch, ...]
    memory_weight: float
    visible_sequence: int

    def history_context(self) -> dict:
        """Typed experience triplets; they are not a fabricated continuous video."""
        dimension = len(self.base_future)
        return {
            "source_features":np.stack([m.source_feature for m in self.matches]) if self.matches else np.empty((0,dimension),dtype=np.float32),
            "executed_directions":np.stack([m.branch_direction for m in self.matches]) if self.matches else np.empty((0,2),dtype=np.float32),
            "observed_future_features":np.stack([m.observed_future for m in self.matches]) if self.matches else np.empty((0,dimension),dtype=np.float32),
            "record_sequences":np.array([m.sequence for m in self.matches],dtype=np.int64),
            "episode_ids_for_audit":tuple(m.episode_id for m in self.matches),
        }


class EpisodeMemorySession:
    def __init__(self, bank: PersistentSceneMemory, *, scope: str = "cross_episode"):
        if scope not in ("cross_episode", "all_past"):
            raise ValueError("unknown memory visibility scope")
        self.bank, self.scope = bank, scope
        self.context = self.episode_id = None
        self.episode_start_sequence = 0
        self.pending = {}

    def begin_episode(self, context: str, episode_id: str, instruction: str = "") -> None:
        if not context or not episode_id:
            raise ValueError("context and episode id are required")
        if self.context is not None:
            self.bank.end_episode(self.context,self.episode_id)
        self.context, self.episode_id = context, episode_id
        self.episode_start_sequence = self.bank.begin_episode(context,episode_id,instruction)
        self.pending.clear()

    def end_episode(self) -> None:
        if self.context is not None:
            self.bank.end_episode(self.context,self.episode_id)
        self.pending.clear()
        self.context = self.episode_id = None

    def predict_cached(
        self, base_future, current_feature, branch_direction, *, event_id: str, source_observation: str,
    ) -> MemoryPrediction:
        if self.context is None:
            raise ValueError("begin_episode must precede prediction")
        source = unit_vector(current_feature, self.bank.feature_dim)
        base = unit_vector(base_future, self.bank.feature_dim)
        branch = unit_vector(branch_direction, 2)
        cutoff = self.episode_start_sequence if self.scope == "cross_episode" else self.bank.latest_sequence(self.context)
        matches = self.bank.retrieve(
            context=self.context, before_sequence=cutoff, source_feature=source, predicted_future=base,
            branch_direction=branch, exclude_episode=self.episode_id if self.scope == "cross_episode" else None,
        )
        weight, fused = 0., base.copy()
        if matches:
            scores = np.array([m.score for m in matches])
            weights = np.exp((scores-scores.max())/.07)
            weights /= weights.sum()
            observed = sum(w*m.observed_future for w,m in zip(weights,matches,strict=True))
            confidence = max(0.,min(1.,(matches[0].source_similarity-.5)/.5))
            branch_floor = math.cos(math.radians(35))
            confidence *= max(0.,min(1.,(matches[0].branch_similarity-branch_floor)/(1-branch_floor)))
            weight = .75*confidence
            fused = unit_vector((1-weight)*base+weight*observed,self.bank.feature_dim)
        prediction_id = uuid.uuid4().hex
        self.pending[prediction_id] = {
            "context":self.context,"episode_id":self.episode_id,"event_id":event_id,
            "source_observation":source_observation,"source_feature":source,"branch_direction":branch,
        }
        return MemoryPrediction(prediction_id,base,fused,tuple(matches),weight,cutoff)

    def observe_real(
        self, prediction_id: str, observed_future, *, target_observation: str, execution_ref: str,
        executed_branch_direction, executed: bool, arrived: bool, outcome: str = "arrived",
    ) -> int:
        if prediction_id not in self.pending:
            raise ValueError("unknown, stale or already resolved prediction")
        original = dict(self.pending[prediction_id])
        # A controller can take a different exit than the requested one. Store
        # the observed execution, never silently attach it to the planned exit.
        original['branch_direction'] = unit_vector(executed_branch_direction,2)
        sequence = self.bank.record_observed(
            **original, observed_future=observed_future, target_observation=target_observation,
            execution_ref=execution_ref, executed=executed, arrived=arrived, outcome=outcome,
        )
        del self.pending[prediction_id]
        return sequence


def predict_with_scene_memory(
    predictor, session: EpisodeMemorySession, raw_history: torch.Tensor, valid: torch.Tensor,
    branch: torch.Tensor, *, event_id: str, source_observation: str,
) -> MemoryPrediction:
    """A real-time caller uses its normal encoder, then this history-aware wrapper."""
    if raw_history.shape[0] != 1:
        raise ValueError("one active scene/episode per memory session")
    predictor.eval()
    with torch.inference_mode():
        base = predictor(raw_history,valid,branch)[0].cpu().numpy()
        current = predictor.normalize(raw_history[:,-1])[0].cpu().numpy()
    return session.predict_cached(base,current,branch[0].detach().cpu().numpy(),
                                  event_id=event_id,source_observation=source_observation)
