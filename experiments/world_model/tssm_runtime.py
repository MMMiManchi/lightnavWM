"""Stateful next-event queries, real-outcome memory, and scene-only updates."""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.world_model.event_residual import direction_read
from experiments.world_model.event_tssm import EventTSSM, TSSMConfig
from experiments.world_model.feature_cache import (
    SIGLIP_PREPROCESS_SHA256,
    SIGLIP_REVISION,
    SIGLIP_WEIGHT_SHA256,
)
from experiments.world_model.memory_routing import MemoryRoute, MemoryRoutingPolicy
from experiments.world_model.scene_memory import PersistentSceneMemory, unit_vector


class TSSMPredictor(nn.Module):
    def __init__(self, model, mean, scale):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.register_buffer("feature_mean", mean.float())
        self.register_buffer("feature_scale", scale.float())
        if mean.shape != (model.config.feature_dim,) or scale.shape != mean.shape or not (scale > 0).all():
            raise ValueError("invalid frozen normalization")
        body = json.dumps([SIGLIP_REVISION, SIGLIP_WEIGHT_SHA256, SIGLIP_PREPROCESS_SHA256]).encode()
        self.feature_contract_id = hashlib.sha256(body + mean.cpu().numpy().tobytes() + scale.cpu().numpy().tobytes()).hexdigest()

    def normalize(self, raw):
        if not torch.isfinite(raw).all():
            raise ValueError("non-finite observed visual feature")
        return F.normalize((raw.float() - self.feature_mean) / self.feature_scale, dim=-1)

    def forward(self, raw_history, branch, valid=None, columns=None):
        state = self.model.initialize(self.normalize(raw_history), valid)
        action = torch.cat([branch, torch.zeros_like(branch[:, :1]), torch.ones_like(branch[:, :1])], dim=-1)
        if self.model.config.use_exit_visual and columns is None:
            raise ValueError("this world model requires actual current patch columns")
        visual = direction_read(columns, branch, use_branch=self.model.config.use_branch) if columns is not None and self.model.config.use_exit_visual else None
        return self.model.transition(state, action, exit_visual=visual)[1]["feature"]


def load_tssm_predictor(path, device="cpu"):
    bundle = torch.load(path, map_location=device, weights_only=True)
    if bundle.get("schema") != "event-tssm/0.1":
        raise ValueError("unsupported stateful world-model checkpoint")
    model = EventTSSM(TSSMConfig(**bundle["model_config"]))
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    return TSSMPredictor(model, bundle["feature_mean"], bundle["feature_scale"]).to(device).eval()


class SceneDelta(nn.Module):
    def __init__(self, dimension, feature_dim, rank=16):
        super().__init__()
        self.down = nn.Linear(dimension, rank)
        self.up = nn.Linear(rank, feature_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs, basis):
        # No fixed angular or residual-amplitude ceiling.
        return F.normalize(basis + self.up(F.gelu(self.down(inputs))), dim=-1)


class SceneUpdates:
    def __init__(self, folder, *, context, base_hash, contract_id, dimension, feature_dim,
                 steps=20, learning_rate=0.003, rank=16):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / (hashlib.sha256(context.encode()).hexdigest() + ".pt")
        self.binding = {"context": context, "base_hash": base_hash, "contract": contract_id,
                        "dimension": dimension, "feature_dim": feature_dim, "rank": rank}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(11)
            self.adapter = SceneDelta(dimension, feature_dim, rank)
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.steps, self.version, self.optimizer_steps, self.replay = steps, 0, 0, []
        if self.path.exists():
            saved = torch.load(self.path, map_location="cpu", weights_only=True)
            if saved["binding"] != self.binding:
                raise ValueError("scene adapter belongs to another model or feature space")
            self.adapter.load_state_dict(saved["adapter"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.version, self.optimizer_steps, self.replay = saved["version"], saved["optimizer_steps"], saved["replay"]

    def parameter_hash(self):
        h = hashlib.sha256()
        for key, v in self.adapter.state_dict().items():
            h.update(key.encode())
            h.update(v.detach().numpy().tobytes())
        return h.hexdigest()

    def predict(self, inputs, basis):
        with torch.no_grad():
            return self.adapter(torch.tensor(inputs)[None], torch.tensor(basis)[None])[0].numpy().copy()

    def applicable(self, inputs):
        # Consult observed INPUT conditions only, never remembered target values.
        # Evidence about one exit does not validate another exit's scene delta.
        d = self.binding["feature_dim"]
        source, branch = inputs[:d], inputs[-4:-2]
        return any(float(source @ row[0][:d].numpy()) >= 0.98 and
                   float(branch @ row[0][-4:-2].numpy()) >= 0.98 for row in self.replay)

    def update_observed(self, inputs, basis, target):
        # Clone outside inference_mode so no inference tensor enters autograd.
        self.replay.append(tuple(torch.tensor(np.array(x, dtype=np.float32, copy=True)) for x in [inputs, basis, target]))
        self.replay = self.replay[-128:]
        x, b, y = [torch.stack([row[k] for row in self.replay]) for k in range(3)]
        with torch.no_grad():
            before = float((self.adapter(x, b) * y).sum(-1).mean())
        old_weights, old_optimizer = copy.deepcopy(self.adapter.state_dict()), copy.deepcopy(self.optimizer.state_dict())
        old_hash = self.parameter_hash()
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            p = self.adapter(x, b)
            loss = (1 - (p * y).sum(-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
            self.optimizer.step()
        with torch.no_grad():
            after = float((self.adapter(x, b) * y).sum(-1).mean())
        accepted = np.isfinite(after) and after >= before - 1e-7
        if accepted:
            self.version += 1
            self.optimizer_steps += self.steps
        else:
            self.adapter.load_state_dict(old_weights)
            self.optimizer.load_state_dict(old_optimizer)
        payload = {"binding": self.binding, "adapter": self.adapter.state_dict(),
                   "optimizer": self.optimizer.state_dict(), "version": self.version,
                   "optimizer_steps": self.optimizer_steps, "replay": self.replay}
        temporary = self.path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(self.path)
        return {"accepted_on_observed_replay": bool(accepted), "version": self.version,
                "optimizer_steps": self.optimizer_steps, "past_replay_cosine_before": before,
                "past_replay_cosine_after": after, "parameters_changed": old_hash != self.parameter_hash()}


@dataclass
class EventForecast:
    forecast_id: str
    episode_id: str
    context: str
    event_id: str
    source_observation: str
    generation: int
    visible_sequence: int
    direction: np.ndarray
    source_feature: np.ndarray
    base_feature: np.ndarray
    future_feature: np.ndarray
    adapter_inputs: np.ndarray
    adapter_version: int
    start_state: object
    prior_state: object
    memory_sequences: tuple
    evidence_kind: str
    adapter_applied: bool
    route_reason: str = ""
    memory_matches: tuple = ()
    world_model_called: bool = True
    exit_visual: np.ndarray | None = None


class TSSMSession:
    def __init__(self, predictor, bank: PersistentSceneMemory, folder: Path, *, base_hash,
                 use_memory=True, online_learning=False, update_steps=20, direction_transform=None,
                 routing_policy=None, current_episode_memory=False):
        self.predictor, self.bank, self.folder = predictor, bank, Path(folder)
        self.base_hash, self.use_memory, self.online_learning = base_hash, use_memory, online_learning
        self.update_steps, self.direction_transform = update_steps, direction_transform
        self.current_episode_memory = current_episode_memory
        self.routing_policy = routing_policy or MemoryRoutingPolicy()
        if bank.contract_id != predictor.feature_contract_id or bank.feature_dim != predictor.model.config.feature_dim:
            raise ValueError("persistent observations use another feature space")
        self.pending, self.active, self.state, self.episode_id = {}, None, None, None
        self.generation = 0

    @property
    def excluded_episode(self):
        return None if self.current_episode_memory else self.episode_id

    def begin_episode(self, context, episode_id, raw_history, source_observation, instruction="", valid=None):
        if not context or not episode_id or not source_observation:
            raise ValueError("episode requires context and an actual current observation")
        if self.episode_id is not None:
            self.end_episode()
        if any(r["episode_id"] == episode_id for r in self.bank.episode_history(context)):
            raise ValueError("new episode ids must be fresh; raw-motion state cannot silently resume")
        self.context, self.episode_id, self.current_ref = context, episode_id, source_observation
        self.bank.begin_episode(context, episode_id, instruction)
        self.generation += 1
        self.pending.clear()
        device = self.predictor.feature_mean.device
        with torch.inference_mode():
            history = self.predictor.normalize(torch.as_tensor(raw_history, dtype=torch.float32, device=device))
            if history.shape[0] != 1:
                raise ValueError("one live scene episode per session")
            if valid is not None:
                valid = torch.as_tensor(valid, dtype=torch.bool, device=device)
            self.state = self.predictor.model.initialize(history, valid)
        c = self.predictor.model.config
        self.updates = SceneUpdates(self.folder / "scene_adapters", context=context, base_hash=self.base_hash,
            contract_id=self.bank.contract_id, dimension=c.feature_dim * 2 + c.width + c.stochastic_dim + 4,
            feature_dim=c.feature_dim, steps=self.update_steps) if self.online_learning else None

    def end_episode(self):
        if self.episode_id is not None:
            self.bank.end_episode(self.context, self.episode_id)
        self.episode_id, self.state, self.active = None, None, None
        self.pending.clear()
        self.generation += 1

    def _memory_tensor(self, state, matches):
        d = self.predictor.model.config.feature_dim
        memory, mask = np.zeros((1, 3, d * 2 + 2), dtype=np.float32), np.zeros((1, 3), dtype=bool)
        for i, m in enumerate(matches[:3]):
            b = m.aligned_branch_direction if m.aligned_branch_direction is not None else m.branch_direction
            memory[0, i], mask[0, i] = np.r_[m.source_feature, m.observed_future, b], True
        device = state.current.device
        return torch.from_numpy(memory).to(device), torch.from_numpy(mask).to(device), matches

    @torch.inference_mode()
    def forecast(self, event_id, direction, *, direction_transform=None, columns=None):
        if self.state is None or self.active is not None or not event_id:
            raise ValueError("finish the active execution before proposing another event")
        direction = unit_vector(direction, 2)
        visual = None
        if self.predictor.model.config.use_exit_visual:
            if columns is None:
                raise ValueError("this world model requires actual current patch columns")
            raw_columns = torch.as_tensor(columns, device=self.state.current.device, dtype=torch.float32)
            visual = direction_read(raw_columns, torch.tensor(direction[None], device=raw_columns.device),
                                    use_branch=self.predictor.model.config.use_branch)
        cutoff = self.bank.latest_sequence(self.context)
        source = self.state.current[0].cpu().numpy().copy()
        route = self.routing_policy.resolve(self.bank, context=self.context, cutoff=cutoff,
            episode_id=self.excluded_episode, current=source, direction=direction,
            transform=direction_transform or self.direction_transform) if self.use_memory else MemoryRoute("predict", "memory_disabled")
        matches = route.matches
        if route.kind != "predict":
            # No prior, decoder or scene-adapter inference on a direct hit or
            # an ambiguous lookup. A dynamics prior can be deferred to feedback.
            future = matches[0].observed_future.copy() if route.kind == "memory" else None
            ticket = EventForecast(uuid.uuid4().hex, self.episode_id, self.context, event_id, self.current_ref,
                self.generation, cutoff, direction, source, future, future, None,
                self.updates.version if self.updates is not None else 0, self.state.detach(), None,
                tuple(m.sequence for m in matches), "observed_memory_candidate" if future is not None else "observation_required",
                False, route.reason, matches, False, visual[0].cpu().numpy().copy() if visual is not None else None)
            self.pending[ticket.forecast_id] = ticket
            return ticket
        memory, mask, _ = self._memory_tensor(self.state, matches)
        action = torch.tensor([[*direction, 0.0, 1.0]], device=self.state.current.device)
        prior, stats = self.predictor.model.transition(self.state, action, memory=memory, memory_valid=mask, exit_visual=visual)
        base = stats["feature"][0].cpu().numpy().copy()
        # Strong real-history matches are explicit remembered facts, not credit
        # assigned to an imagined unseen future or instructions to follow a route.
        remembered = bool(stats["memory_supported"][0])
        if remembered:
            base = prior.anchor[0].cpu().numpy().copy()
        inputs = torch.cat([self.state.current, stats["feature"], prior.deter, stats["mean"], action], dim=-1)[0].cpu().numpy().copy()
        apply_adapter = self.updates is not None and not remembered and self.updates.applicable(inputs)
        future = self.updates.predict(inputs, base) if apply_adapter else base.copy()
        ticket = EventForecast(uuid.uuid4().hex, self.episode_id, self.context, event_id, self.current_ref,
            self.generation, cutoff, direction, source, base, future, inputs,
            self.updates.version if self.updates is not None else 0, self.state.detach(), prior.detach(),
            tuple(m.sequence for m in matches), "observed_memory_candidate" if remembered else "model_prior",
            bool(apply_adapter), route.reason, matches, True, visual[0].cpu().numpy().copy() if visual is not None else None)
        self.pending[ticket.forecast_id] = ticket
        return ticket

    def _check(self, ticket):
        if (self.pending.get(ticket.forecast_id) is not ticket or ticket.episode_id != self.episode_id
                or ticket.context != self.context or (self.active is None and ticket.generation != self.generation)):
            raise ValueError("stale or unissued forecast cannot write observations")

    def commit(self, ticket):
        self._check(ticket)
        if ticket.future_feature is None:
            raise ValueError("ambiguous memory requires observation before execution")
        if self.active is not None:
            raise ValueError("an event execution is already active")
        self.active = ticket
        self.pending = {ticket.forecast_id: ticket}

    def abort_event(self, ticket, reason):
        self._check(ticket)
        if not reason or self.active is not ticket:
            raise ValueError("only an active event can be abandoned with a reason")
        self.pending.clear()
        self.active = None
        self.generation += 1

    @torch.inference_mode()
    def observe_route(self, raw_feature, reference, *, execution_ref, executed, direction=(0.0, 1.0)):
        if self.state is None or executed is not True or not reference or not execution_ref:
            raise ValueError("route feedback requires actual executed motion in a live episode")
        if self.active is None:
            self.pending.clear()
        value = self.predictor.normalize(torch.as_tensor(raw_feature, dtype=torch.float32,
            device=self.state.current.device)[None])
        action = torch.tensor([[*unit_vector(direction, 2), 1.0, 0.0]], device=value.device)
        prior, _ = self.predictor.model.transition(self.state, action)
        self.state, _ = self.predictor.model.observe(prior, value)
        self.current_ref = reference
        self.generation += 1

    def observe_event(self, ticket, raw_target, *, target_observation, execution_ref, executed, arrived,
                      executed_direction=None):
        self._check(ticket)
        if executed is not True or arrived is not True or not target_observation or not execution_ref or target_observation == ticket.source_observation:
            raise ValueError("only a verified actual next event can update persistent facts or parameters")
        if self.active is None:
            self.commit(ticket)
        if self.active is not ticket:
            raise ValueError("another branch was committed")
        actual = unit_vector(ticket.direction if executed_direction is None else executed_direction, 2)
        aligned = bool(np.allclose(actual, ticket.direction, atol=1e-5))
        inputs, basis, actual_prior = ticket.adapter_inputs, ticket.base_feature, ticket.prior_state
        with torch.inference_mode():
            value = self.predictor.normalize(torch.as_tensor(raw_target, dtype=torch.float32,
                device=self.state.current.device)[None])
            if not aligned or actual_prior is None:
                # Deferred memory-hit priors are computed only after execution.
                # They see the original source and actual direction, no target.
                action = torch.tensor([[*actual, 0.0, 1.0]], device=value.device)
                visual = torch.tensor(ticket.exit_visual[None], device=value.device) if aligned and ticket.exit_visual is not None else None
                actual_prior, p = self.predictor.model.transition(ticket.start_state, action, exit_visual=visual)
                basis = p["feature"][0].cpu().numpy().copy()
                inputs = torch.cat([ticket.start_state.current, p["feature"], actual_prior.deter, p["mean"], action], dim=-1)[0].cpu().numpy().copy()
            already_observed = self.current_ref == target_observation
            if already_observed and not torch.allclose(self.state.current, value, atol=1e-6):
                raise ValueError("one actual observation cannot acquire a different feature")
            if not already_observed and self.current_ref != ticket.source_observation:
                action = torch.tensor([[0., 1., 1., 0.]], device=value.device)
                actual_prior, _ = self.predictor.model.transition(self.state, action)
            if not already_observed:
                self.state, _ = self.predictor.model.observe(actual_prior, value)
            target = value[0].cpu().numpy().copy()
        sequence = self.bank.record_observed(context=self.context, episode_id=self.episode_id,
            event_id=ticket.event_id, source_observation=ticket.source_observation, target_observation=target_observation,
            execution_ref=execution_ref, source_feature=ticket.source_feature, observed_future=target,
            branch_direction=actual, executed=True, arrived=True)
        update = self.updates.update_observed(inputs, basis, target) if self.updates is not None else None
        self.current_ref, self.active = target_observation, None
        self.pending.clear()
        self.generation += 1
        return {"sequence": sequence, "planned_branch_verified": aligned,
                "pre_update_prediction_cosine": float(ticket.future_feature @ target) if aligned else None,
                "update": update, "state_cache_steps": self.state.tokens.shape[1],
                "deferred_prior_after_actual_execution": ticket.prior_state is None}

    def query_actual_memories(self, ticket, top_k=3):
        self._check(ticket)
        if not self.use_memory or ticket.evidence_kind == "observation_required":
            return ()
        if ticket.evidence_kind == "observed_memory_candidate":
            return ticket.memory_matches[:top_k]
        return self.bank.query_actual_outcomes(context=ticket.context, before_sequence=ticket.visible_sequence,
            predicted_future=ticket.future_feature, source_feature=ticket.source_feature,
            branch_direction=ticket.direction, exclude_episode=self.excluded_episode, top_k=top_k)
