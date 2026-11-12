"""Real observations persist independently of scene-specific gradient updates.

The visual space and global model stay fixed. A bounded scene residual learns
from scored, already observed transitions. No optimizer step runs inside an
episode; reserved support observations gate publication of a new model version.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SceneAdapterConfig:
    feature_dim: int = 1152
    condition_dim: int = 128
    rank: int = 16
    residual_limit: float = .35
    learning_rate: float = 3e-4
    update_steps: int = 40
    batch_size: int = 24
    replay_capacity: int = 256
    minimum_train: int = 3
    minimum_validation: int = 1


class SceneResidual(nn.Module):
    def __init__(self, config: SceneAdapterConfig):
        super().__init__()
        self.config = config
        size = config.feature_dim + config.condition_dim + 3
        self.down = nn.Linear(size, config.rank)
        self.up = nn.Linear(config.rank, config.feature_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs, basis):
        delta = self.up(F.gelu(self.down(inputs)))
        delta = self.config.residual_limit * delta / (1 + delta.norm(dim=-1, keepdim=True))
        return F.normalize(basis + delta, dim=-1)


@dataclass(frozen=True)
class PredictionTicket:
    sample_id: str
    episode_id: str
    source_observation: str
    source_step: int
    horizon: int
    model_version: int
    prediction: np.ndarray


def _array(value, size):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("invalid observed feature or input")
    return result.copy()


def verify_span(*, episode_id, source_step, target_step, horizon, commands,
                source_path, target_path, source_observation, target_observation):
    """Verify observation/control provenance, never assert topological arrival."""
    from PIL import Image

    if target_step-source_step != horizon or horizon < 1 or len(commands) != horizon:
        raise ValueError("executed span has the wrong horizon")
    if source_observation == target_observation or not episode_id:
        raise ValueError("an observed transition needs distinct references")
    for offset, command in enumerate(commands):
        if (command.get("episode_id") != episode_id or command.get("step") != source_step+offset
                or command.get("environment_step_returned") is not True or not command.get("action")):
            raise ValueError("missing or mismatched real execution acknowledgement")
    hashes = {}
    for label, path in [("source", source_path), ("target", target_path)]:
        path = Path(path)
        pixels = np.asarray(Image.open(path))
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
            raise ValueError("observed RGB is invalid")
        hashes[label] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"kind": "observed_control_span", "real_observation": True, "episode_id": episode_id,
            "source_step": source_step, "target_step": target_step, "horizon": horizon,
            "source_observation": source_observation, "target_observation": target_observation,
            "images": hashes, "commands": commands, "node_arrival_claimed": False}


class OnlineSceneLearner:
    def __init__(self, folder, *, scene_id, feature_contract, base_hash, target_kind,
                 config=SceneAdapterConfig(), device="cpu", learning=True):
        if target_kind not in ("next_event", "observed_control_span"):
            raise ValueError("unknown transition target kind")
        if not scene_id or not feature_contract or not base_hash:
            raise ValueError("scene and frozen-model/feature bindings are required")
        self.folder, self.config, self.device = Path(folder), config, torch.device(device)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.scene_id, self.target_kind, self.learning = scene_id, target_kind, learning
        self.binding = {"schema": "wm-online-scene/0.1", "scene_id": scene_id, "feature_contract": feature_contract,
                        "base_hash": base_hash, "target_kind": target_kind, "config": asdict(config)}
        self.db = sqlite3.connect(self.folder / "observed_replay.sqlite", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("create table if not exists metadata(key text primary key,value text not null)")
        binding = json.dumps(self.binding, sort_keys=True)
        with self.db:
            self.db.execute("insert or ignore into metadata values('binding',?)", (binding,))
        if self.db.execute("select value from metadata where key='binding'").fetchone()[0] != binding:
            self.db.close()
            raise ValueError("scene/feature/model/target binding differs")
        self.db.execute("""create table if not exists observations(
            sequence integer primary key autoincrement,sample text unique not null,episode text not null,
            source_ref text not null,target_ref text not null,inputs blob not null,basis blob not null,target blob not null,
            version integer not null,pre_update_cosine real not null,proof text not null)""")
        self.db.commit()
        with torch.random.fork_rng(devices=[]):
            # This initialization uses CPU RNG only. torch.manual_seed also
            # resets CUDA generators, including those used by live inference.
            torch.random.default_generator.manual_seed(20261105)
            self.adapter = SceneResidual(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        self.version, self.optimizer_steps = 0, 0
        active = self.db.execute("select value from metadata where key='active'").fetchone()
        if active:
            payload = torch.load(self.folder / active[0], map_location=self.device, weights_only=True)
            if payload["binding"] != self.binding:
                raise ValueError("scene checkpoint has a different binding")
            self.adapter.load_state_dict(payload["adapter"])
            self.optimizer.load_state_dict(payload["optimizer"])
            self.version, self.optimizer_steps = payload["version"], payload["optimizer_steps"]
        self.episode_id = None
        self.pending = {}
        self.last_update = None

    def begin_episode(self, episode_id):
        if self.episode_id is not None or not episode_id:
            raise ValueError("finish the previous episode first")
        self.episode_id = episode_id
        self.pending.clear()

    def parameters_hash(self):
        h = hashlib.sha256()
        for key, tensor in self.adapter.state_dict().items():
            h.update(key.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def _inputs(self, current, condition, branch, horizon):
        if not isinstance(horizon, int) or (horizon != 0 if self.target_kind == "next_event" else horizon < 1):
            raise ValueError("next-event horizon is unknown (0); fixed spans need a positive declared horizon")
        return np.concatenate([_array(current, self.config.feature_dim), _array(condition, self.config.condition_dim),
                               _array(branch, 2), np.array([horizon/100], dtype=np.float32)])

    def forecast(self, *, current, condition, branch, horizon, basis):
        inputs = self._inputs(current, condition, branch, horizon)
        basis = _array(basis, self.config.feature_dim)
        with torch.no_grad():
            value = self.adapter(torch.from_numpy(inputs)[None].to(self.device),
                                 torch.from_numpy(basis)[None].to(self.device))[0]
        return value.cpu().numpy().copy()

    def predict(self, *, sample_id, source_observation, source_step, current, condition, branch, horizon, basis):
        if self.episode_id is None or not sample_id or sample_id in self.pending:
            raise ValueError("prediction needs a live episode and a fresh sample id")
        if self.db.execute("select 1 from observations where sample=?", (sample_id,)).fetchone():
            raise ValueError("already observed samples cannot become new predictions")
        inputs = self._inputs(current, condition, branch, horizon)
        basis = _array(basis, self.config.feature_dim)
        future = self.forecast(current=current, condition=condition, branch=branch, horizon=horizon, basis=basis)
        ticket = PredictionTicket(sample_id, self.episode_id, source_observation, source_step, horizon,
                                  self.version, future)
        self.pending[sample_id] = (ticket, inputs, basis)
        return ticket

    def observe(self, ticket, target, proof):
        if ticket.episode_id != self.episode_id or ticket.sample_id not in self.pending:
            raise ValueError("outcome has no pending prediction in this episode")
        saved, inputs, basis = self.pending[ticket.sample_id]
        if (ticket is not saved or proof.get("real_observation") is not True
                or proof.get("episode_id") != ticket.episode_id
                or proof.get("source_observation") != ticket.source_observation
                or proof.get("source_step") != ticket.source_step
                or (proof.get("target_step", -1) <= ticket.source_step if self.target_kind == "next_event"
                    else proof.get("target_step", -1)-ticket.source_step != ticket.horizon)
                or proof.get("kind") != self.target_kind or not proof.get("target_observation")):
            raise ValueError("outcome provenance or prediction horizon differs")
        if self.target_kind == "observed_control_span":
            # Revalidate on ingestion; callers cannot substitute an unexecuted
            # proposal or silently relabel an interrupted span.
            validated = verify_span(episode_id=ticket.episode_id, source_step=ticket.source_step,
                target_step=proof["target_step"], horizon=ticket.horizon, commands=proof["commands"],
                source_path=proof["images"]["source"]["path"], target_path=proof["images"]["target"]["path"],
                source_observation=ticket.source_observation, target_observation=proof["target_observation"])
            if validated["images"] != proof["images"]:
                raise ValueError("observed images changed after verification")
        elif not proof.get("admitted_execution_ref"):
            raise ValueError("recorded event lacks execution admission")
        target = _array(target, self.config.feature_dim)
        norm = np.linalg.norm(target)
        if norm < 1e-8:
            raise ValueError("observed target cannot be zero")
        target /= norm
        score = float(ticket.prediction @ target)
        with self.db:
            self.db.execute("""insert into observations(sample,episode,source_ref,target_ref,inputs,basis,target,
                version,pre_update_cosine,proof) values(?,?,?,?,?,?,?,?,?,?)""", (ticket.sample_id,ticket.episode_id,
                ticket.source_observation,proof["target_observation"],inputs.tobytes(),basis.tobytes(),target.tobytes(),
                ticket.model_version,score,json.dumps(proof,sort_keys=True)))
        del self.pending[ticket.sample_id]
        return score

    def finish_episode(self):
        if self.episode_id is None:
            raise ValueError("no active episode")
        incomplete = list(self.pending)
        self.pending.clear()
        self.episode_id = None
        result = self.update() if self.learning else {"status": "frozen_control", "version": self.version}
        result["incomplete_predictions_not_trained"] = incomplete
        self.last_update = result
        return result

    def update(self):
        if self.episode_id is not None:
            raise ValueError("weights cannot change inside an episode")
        rows = self.db.execute("select sequence,inputs,basis,target from observations order by sequence").fetchall()
        if len(rows) > self.config.replay_capacity:
            rows = [rows[i] for i in np.linspace(0, len(rows)-1, self.config.replay_capacity, dtype=int)]
        train = [i for i, row in enumerate(rows) if row[0] % 4 != 0]
        validation = [i for i, row in enumerate(rows) if row[0] % 4 == 0]
        if len(train) < self.config.minimum_train or len(validation) < self.config.minimum_validation:
            return {"status": "insufficient_observed_support", "version": self.version,
                    "train": len(train), "validation": len(validation), "optimizer_steps": 0}
        values = [torch.from_numpy(np.stack([np.frombuffer(r[j], dtype=np.float32) for r in rows])).to(self.device)
                  for j in (1,2,3)]
        x, basis, target = values
        train_ids = torch.tensor(train, device=self.device)
        val_ids = torch.tensor(validation, device=self.device)

        def validation_loss():
            with torch.no_grad():
                return float((1-(self.adapter(x[val_ids], basis[val_ids])*target[val_ids]).sum(-1)).mean())

        before = validation_loss()
        best = before
        initial_hash = self.parameters_hash()
        best_weights = copy.deepcopy(self.adapter.state_dict())
        best_optimizer = copy.deepcopy(self.optimizer.state_dict())
        # A local generator makes background training independent of inference RNG.
        generator = torch.Generator(device=self.device).manual_seed(20261105+len(rows))
        for step in range(self.config.update_steps):
            chosen = train_ids[torch.randint(len(train), (min(self.config.batch_size,len(train)),),
                                               device=self.device, generator=generator)]
            prediction = self.adapter(x[chosen],basis[chosen])
            loss = (1-(prediction*target[chosen]).sum(-1)).mean()+.01*(prediction-basis[chosen]).square().sum(-1).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(),1.0)
            self.optimizer.step()
            self.optimizer_steps += 1
            if (step+1) % 10 == 0 or step+1 == self.config.update_steps:
                score = validation_loss()
                if score < best-1e-7:
                    best = score
                    best_weights = copy.deepcopy(self.adapter.state_dict())
                    best_optimizer = copy.deepcopy(self.optimizer.state_dict())
        self.adapter.load_state_dict(best_weights)
        self.optimizer.load_state_dict(best_optimizer)
        accepted = best < before-1e-7
        if accepted:
            self.version += 1
            models = self.folder / "models"
            models.mkdir(exist_ok=True)
            path = models / f"v{self.version:05d}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save({"binding": self.binding,"adapter": self.adapter.state_dict(),
                        "optimizer": self.optimizer.state_dict(),"version": self.version,
                        "optimizer_steps": self.optimizer_steps},temporary)
            os.replace(temporary,path)
            with self.db:
                self.db.execute("insert or replace into metadata values('active',?)",(str(path.relative_to(self.folder)),))
        return {"status": "published" if accepted else "validation_rejected", "version": self.version,
                "optimizer_steps": self.config.update_steps, "validation_before": before,"validation_after": best,
                "train": len(train),"validation": len(validation),
                "active_parameters_changed": initial_hash != self.parameters_hash()}

    def close(self):
        self.db.close()
