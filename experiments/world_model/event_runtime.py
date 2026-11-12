"""Future-feature inference and an explicit Harness -> LightNav tool boundary.

This does not implement an autonomous node/exit proposer or a semantic planner.
The Harness must supply a grounded local goal and validate completion itself.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.world_model.event_anchored import AnchoredEventConfig, AnchoredEventWorldModel
from experiments.world_model.event_residual import ResidualEventConfig, ResidualEventWorldModel
from experiments.world_model.scene_memory import PersistentSceneMemory


class ResidualFeaturePredictor(nn.Module):
    def __init__(self, bundle):
        super().__init__()
        if bundle.get("schema") != "wm-residual-event/0.1":
            raise ValueError("unsupported event inference bundle")
        if bundle.get("model_kind") == "anchored_memory":
            self.model = AnchoredEventWorldModel(AnchoredEventConfig(**bundle["model_config"]))
        else:
            self.model = ResidualEventWorldModel(ResidualEventConfig(**bundle["model_config"]))
        self.model.load_state_dict(bundle["model_state_dict"])
        self.register_buffer("feature_mean", bundle["feature_mean"].float())
        self.register_buffer("feature_scale", bundle["feature_scale"].float())
        if not (self.feature_scale > 0).all():
            raise ValueError("invalid fixed feature normalization")
        self.feature_contract = bundle["feature_contract"]
        binding = (json.dumps(self.feature_contract, sort_keys=True).encode() +
                   self.feature_mean.cpu().numpy().tobytes() + self.feature_scale.cpu().numpy().tobytes())
        self.feature_contract_id = hashlib.sha256(binding).hexdigest()

    def normalize(self, features):
        return F.normalize((features.float() - self.feature_mean) / self.feature_scale, dim=-1)

    def forward(self, raw_history, valid, branch, columns, **canonical_memory):
        history = self.normalize(raw_history).masked_fill(~valid[..., None], 0)
        return self.model(history, valid, branch, columns, **canonical_memory)


def load_residual_predictor(path: Path, device="cpu"):
    return ResidualFeaturePredictor(torch.load(path, map_location="cpu", weights_only=True)).to(device).eval()


@dataclass(frozen=True)
class BranchEvidence:
    future_feature: np.ndarray
    past_transition_sequences: tuple[int, ...]
    retrieved_evidence: tuple
    visible_sequence: int
    supported_by_history: bool
    query_kind: str = "source_constrained_future"
    source_observation: str | None = None
    branch_direction: tuple[float, float] | None = None


@torch.inference_mode()
def predict_branch_with_memory(predictor, bank: PersistentSceneMemory, *, context, episode_id,
                               visible_sequence, raw_history, valid, branch, columns,
                               outcome_key_query=False, direction_transform=None):
    """Read past source/direction first; only then use prediction as a query.

    No target, world pose, place id or adjacency is accepted by this API.
    The bank must share the exported encoder/normalization contract.
    """
    if raw_history.shape[0] != 1:
        raise ValueError("one active episode per memory query")
    if bank.contract_id != predictor.feature_contract_id or bank.feature_dim != predictor.model.config.feature_dim:
        raise ValueError("memory does not use this predictor's fixed feature contract")
    current = predictor.normalize(raw_history[:, -1])[0].cpu().numpy()
    direction = branch[0].cpu().numpy()
    matches = bank.retrieve(context=context, before_sequence=visible_sequence,
                            source_feature=current, predicted_future=current, branch_direction=direction,
                            exclude_episode=episode_id, query_kind="source",direction_transform=direction_transform)
    dim = predictor.model.config.feature_dim
    count = 3
    sources = np.zeros((1, count, dim), dtype=np.float32)
    futures = np.zeros_like(sources)
    directions = np.zeros((1, count, 2), dtype=np.float32)
    mask = np.zeros((1, count), dtype=bool)
    for i, item in enumerate(matches):
        sources[0, i], futures[0, i], directions[0, i], mask[0, i] = (
            item.source_feature, item.observed_future,
            item.aligned_branch_direction if item.aligned_branch_direction is not None else item.branch_direction, True)
    device = raw_history.device
    memory = {k: torch.from_numpy(v).to(device) for k, v in {
        "memory_source": sources, "memory_future": futures,
        "memory_branch": directions, "memory_valid": mask}.items()}
    future = predictor(raw_history, valid, branch, columns, **memory)[0].cpu().numpy()
    if outcome_key_query:
        # The current model abstains without prior support: its copied current
        # feature must not be presented as an imagined unfamiliar future.
        evidence = bank.query_actual_outcomes(context=context, before_sequence=visible_sequence,
                    predicted_future=future, source_feature=current, branch_direction=direction,
                    exclude_episode=episode_id) if matches else []
    else:
        evidence = bank.retrieve(context=context, before_sequence=visible_sequence, source_feature=current,
                                 predicted_future=future, branch_direction=direction, exclude_episode=episode_id)
    return BranchEvidence(future, tuple(m.sequence for m in matches), tuple(evidence), visible_sequence, bool(matches),
                          "predicted_outcome_key" if outcome_key_query else "source_constrained_future")


@dataclass(frozen=True)
class LocalGoal:
    instruction: str
    source_observation: str
    evidence_sequences: tuple[int, ...]
    bearing_rad: float

    def __post_init__(self):
        if not self.instruction.strip() or not self.source_observation:
            raise ValueError("local goal needs an instruction and current visual evidence")
        if not math.isfinite(self.bearing_rad) or abs(self.bearing_rad) >= math.pi / 3:
            raise ValueError("local goal must be grounded in the current camera cone")


class HarnessLightNavExecutor:
    """Keep native episode history; update only the current local instruction."""

    def __init__(self, policy):
        self.policy = policy
        self.original_instruction = ""
        self.active_goal = None
        self.last_tool_status = "idle"

    def reset(self, observation):
        self.policy.reset(observation)
        self.original_instruction = self.policy.agent.instruction
        self.active_goal = None
        self.last_tool_status = "idle"

    def commit_goal(self, goal: LocalGoal, *, current_observation: str, visible_sequence: int):
        if goal.source_observation != current_observation:
            raise ValueError("local goal is based on a stale observation")
        if any(i > visible_sequence or i < 1 for i in goal.evidence_sequences):
            raise ValueError("local goal references future or invalid memory evidence")
        self.active_goal = goal
        # reset() would discard the episode's SlowFast memory and visual cache.
        self.policy.agent.instruction = goal.instruction

    def restore_global_goal(self):
        self.active_goal = None
        self.policy.agent.instruction = self.original_instruction

    def propose_action(self, observation, info=None):
        action = self.policy.act(observation, info)
        details = self.policy.get_info()
        return self._checked_proposal(action, details)

    def replan_observed_frame(self):
        """Reconsider a rejected local STOP without observing the frame twice."""
        from lightnav.habitat.policy import select_action_waypoint
        from lightnav.velocity import first_waypoint_to_velocity_cmd
        from lightnav.vln_utils import parse_rvq_action_tokens

        policy = self.policy
        text, _ = policy.engine.generate_from_frames(policy.agent._get_video_tensor(), policy.agent.instruction,
                    frame_ids=list(policy.agent._history_frame_ids), predict_horizon=1, task_type="vlnce_traj")
        try:
            trajectory, _ = policy.agent.decode_waypoints(text)
        except ValueError:
            self.last_tool_status = "decode_failed"
            return None, {"raw_text": text}
        index, waypoint = select_action_waypoint(trajectory)
        action = first_waypoint_to_velocity_cmd(waypoint, policy.dt, policy.lin_vel_range, policy.ang_vel_range)
        details = {"predicted_traj": trajectory, "raw_text": text, "action_waypoint_index": int(index),
                   "rvq_codes": [int(c) for c in parse_rvq_action_tokens(text)] if policy.method == "rvq" else None}
        return self._checked_proposal(action, details)

    def _checked_proposal(self, action, details):
        trajectory = details.get("predicted_traj")
        if trajectory is None:
            self.last_tool_status = "decode_failed"
            return None, details
        if np.allclose(trajectory, 0, atol=0.005):
            # A local stop is reported to Harness; it must not silently become
            # an environment STOP ending the global navigation task.
            self.last_tool_status = "stop_proposed"
            return None, details
        self.last_tool_status = "action_proposed"
        return action, details
