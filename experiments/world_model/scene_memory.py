"""Persistent, context-isolated memory of actually observed event outcomes."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def unit_vector(value, dimension: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (dimension,) or not np.isfinite(array).all():
        raise ValueError("invalid memory feature shape or values")
    norm = float(np.linalg.norm(array))
    if norm < 1e-8:
        raise ValueError("memory feature must be nonzero")
    return np.asarray(array / norm, dtype="<f4")


@dataclass(frozen=True)
class MemoryMatch:
    sequence: int
    episode_id: str
    event_id: str
    source_observation: str
    target_observation: str
    score: float
    source_similarity: float
    future_similarity: float
    branch_similarity: float
    source_feature: np.ndarray
    branch_direction: np.ndarray
    observed_future: np.ndarray
    aligned_branch_direction: np.ndarray | None = None
    alignment_info: dict | None = None
    outcome_kind: str = "arrived"


class PersistentSceneMemory:
    """SQLite is the durable journal; episode resets do not erase this bank.

    Context is a caller-provided physical workspace/session namespace. Official
    place ids, positions, adjacency and future observations are not query keys.
    """

    def __init__(self, path: Path, *, contract_id: str, feature_dim: int = 1152):
        if not contract_id or feature_dim < 1:
            raise ValueError("a fixed feature contract is required")
        path = Path(path)
        if (path.parent / "DO_NOT_ADMIT_EXPERIENCE.json").exists():
            raise ValueError("scene experience is quarantined; review its DO_NOT_ADMIT_EXPERIENCE.json before reuse")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_dim = feature_dim
        self.contract_id = contract_id
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        signature = json.dumps({"schema":"wm-scene-memory/0.1", "contract":contract_id, "dimension":feature_dim}, sort_keys=True)
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('contract', ?)", (signature,))
        if self.connection.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()[0] != signature:
            self.connection.close()
            raise ValueError("memory feature contract mismatch")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS experiences (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            context TEXT NOT NULL, episode TEXT NOT NULL, event TEXT NOT NULL,
            source_observation TEXT NOT NULL, target_observation TEXT NOT NULL,
            outcome TEXT NOT NULL, execution_ref TEXT NOT NULL,
            source_feature BLOB NOT NULL, future_feature BLOB NOT NULL, branch BLOB NOT NULL,
            UNIQUE(context, episode, event))""")
        self.connection.execute("CREATE INDEX IF NOT EXISTS context_time ON experiences(context, sequence)")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS episodes (
            context TEXT NOT NULL, episode TEXT NOT NULL, instruction TEXT NOT NULL,
            began_sequence INTEGER NOT NULL, ended_sequence INTEGER,
            PRIMARY KEY(context,episode))""")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def latest_sequence(self, context: str) -> int:
        row = self.connection.execute("SELECT COALESCE(MAX(sequence),0) FROM experiences WHERE context=?", (context,)).fetchone()
        return int(row[0])

    def count(self, context: str) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM experiences WHERE context=?", (context,)).fetchone()[0])

    def begin_episode(self, context: str, episode_id: str, instruction: str = "") -> int:
        cutoff = self.latest_sequence(context)
        with self.connection:
            existing = self.connection.execute("SELECT instruction FROM episodes WHERE context=? AND episode=?", (context,episode_id)).fetchone()
            if existing and existing[0] != instruction:
                raise ValueError("episode metadata cannot be reassigned")
            self.connection.execute("INSERT OR IGNORE INTO episodes VALUES (?,?,?,?,NULL)",
                                    (context,episode_id,instruction,cutoff))
        return cutoff

    def end_episode(self, context: str, episode_id: str) -> None:
        with self.connection:
            self.connection.execute("UPDATE episodes SET ended_sequence=? WHERE context=? AND episode=?",
                                    (self.latest_sequence(context),context,episode_id))

    def episode_history(self, context: str) -> list[dict]:
        rows = self.connection.execute("""SELECT episode,instruction,began_sequence,ended_sequence
            FROM episodes WHERE context=? ORDER BY rowid""", (context,))
        return [dict(zip(['episode_id','instruction','began_sequence','ended_sequence'],r,strict=True)) for r in rows]

    def experience_history(self, context: str, *, episode_id: str | None = None, before_sequence: int | None = None) -> list[dict]:
        cutoff = self.latest_sequence(context) if before_sequence is None else before_sequence
        sql = """SELECT sequence,episode,event,source_observation,target_observation,
            outcome,execution_ref,branch FROM experiences WHERE context=? AND sequence<=?"""
        values = [context,cutoff]
        if episode_id is not None:
            sql += " AND episode=?"
            values.append(episode_id)
        sql += " ORDER BY sequence"
        return [{'sequence':s,'episode_id':ep,'event_id':event,'source_observation':source,
                 'target_observation':target,'outcome':outcome,'execution_ref':ack,
                 'branch_direction':np.frombuffer(branch,dtype='<f4').tolist()}
                for s,ep,event,source,target,outcome,ack,branch in self.connection.execute(sql,values)]

    def record_observed(
        self, *, context: str, episode_id: str, event_id: str,
        source_observation: str, target_observation: str, execution_ref: str,
        source_feature, observed_future, branch_direction,
        executed: bool, arrived: bool, outcome: str = "arrived", provenance: str = "real_observation",
        measured_span: dict | None = None,
    ) -> int:
        if not all([context, episode_id, event_id, source_observation, target_observation, execution_ref]):
            raise ValueError("observed memory requires source and execution provenance")
        if provenance != "real_observation" or executed is not True:
            raise ValueError("predictions or unexecuted proposals are not real experiences")
        if outcome not in ("arrived", "blocked", "truncated", "observed_span") or (outcome == "arrived" and arrived is not True):
            raise ValueError("incomplete attempts cannot be relabeled as arrivals")
        if outcome == "observed_span":
            if measured_span is None or measured_span.get("source") != "actual_rgbd_pnp" or not measured_span.get("continuous"):
                raise ValueError("route observation needs continuous measured RGB-D motion")
            displacement = np.asarray(measured_span.get("source_camera_right_forward"), dtype=float)
            if displacement.shape != (2,) or not np.isfinite(displacement).all() or not .6 <= np.linalg.norm(displacement) <= 2.:
                raise ValueError("route observation needs 0.6–2 metres of measured displacement")
            expected = unit_vector([-displacement[0], displacement[1]], 2)
            if not np.allclose(expected, unit_vector(branch_direction, 2), atol=1e-5):
                raise ValueError("route direction disagrees with measured motion")
        source = unit_vector(source_feature, self.feature_dim).tobytes()
        future = unit_vector(observed_future, self.feature_dim).tobytes()
        branch = unit_vector(branch_direction, 2).tobytes()
        payload = (context, episode_id, event_id, source_observation, target_observation, outcome,
                   execution_ref, source, future, branch)
        with self.connection:
            previous = self.connection.execute("""SELECT sequence,source_observation,target_observation,outcome,
                execution_ref,source_feature,future_feature,branch FROM experiences
                WHERE context=? AND episode=? AND event=?""", payload[:3]).fetchone()
            if previous:
                if tuple(previous[1:]) != payload[3:]:
                    raise ValueError("an observed event cannot be rewritten with another outcome")
                return int(previous[0])
            cursor = self.connection.execute("""INSERT INTO experiences
                (context,episode,event,source_observation,target_observation,outcome,execution_ref,source_feature,future_feature,branch)
                VALUES (?,?,?,?,?,?,?,?,?,?)""", payload)
            return int(cursor.lastrowid)

    def retrieve(
        self, *, context: str, before_sequence: int, source_feature, predicted_future,
        branch_direction, exclude_episode: str | None = None, top_k: int = 3,
        source_floor: float = 0.5, branch_floor: float = math.cos(math.radians(35)),
        query_kind: str = "future",
        direction_transform=None,
    ) -> list[MemoryMatch]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if query_kind not in ("future", "source", "visual"):
            raise ValueError("unknown memory query kind")
        source = unit_vector(source_feature, self.feature_dim)
        future = unit_vector(predicted_future, self.feature_dim)
        branch = unit_vector(branch_direction, 2)
        sql = """SELECT sequence,episode,event,source_observation,target_observation,
            source_feature,future_feature,branch FROM experiences
            WHERE context=? AND sequence<=? AND outcome='arrived'"""
        arguments = [context, before_sequence]
        if exclude_episode is not None:
            sql += " AND episode<>?"
            arguments.append(exclude_episode)
        matches = []
        for sequence, episode, event, source_obs, target_obs, raw_source, raw_future, raw_branch in self.connection.execute(sql, arguments):
            s = np.frombuffer(raw_source, dtype="<f4")
            t = np.frombuffer(raw_future, dtype="<f4")
            b = np.frombuffer(raw_branch, dtype="<f4")
            source_similarity = float(source@s)
            aligned, alignment_info = None, None
            if source_similarity >= source_floor and query_kind != "visual" and direction_transform is not None:
                transformed = direction_transform(source_obs,b.copy())
                if transformed is None:
                    continue
                aligned,alignment_info = transformed
                aligned = unit_vector(aligned,2)
            branch_similarity = float(branch@(aligned if aligned is not None else b))
            if source_similarity < source_floor or (query_kind != "visual" and branch_similarity < branch_floor):
                continue
            future_similarity = float(future@t)
            if query_kind == "visual":
                score = source_similarity
            elif query_kind == "source":
                score = .8*source_similarity + .2*branch_similarity
            else:
                score = .5*source_similarity + .3*future_similarity + .2*branch_similarity
            matches.append(MemoryMatch(sequence, episode, event, source_obs, target_obs, score,
                                       source_similarity, future_similarity, branch_similarity,
                                       s.copy(),b.copy(),t.copy(),aligned,alignment_info))
        return sorted(matches, key=lambda r:(-r.score, r.sequence if query_kind != "future" else -r.sequence))[:top_k]

    def nearest_observed_sources(self, *, context, before_sequence, source_feature, exclude_episode,
                                 source_floor, top_k=3):
        """Propose visual revisits before a branch has been chosen.

        A visual match is a candidate, not proof of topological identity or of
        a currently traversable exit. Stored branch directions remain evidence.
        """
        return self.retrieve(context=context, before_sequence=before_sequence, source_feature=source_feature,
                             predicted_future=source_feature, branch_direction=[0, 1], exclude_episode=exclude_episode,
                             source_floor=source_floor, top_k=top_k, query_kind="visual")

    def query_actual_outcomes(self, *, context, before_sequence, predicted_future,
                              source_feature, branch_direction, exclude_episode,
                              top_k=3, minimum_similarity=-1.0):
        """Search actual outcome keys using a predicted future, not source ids.

        Source and direction similarities are returned for audit, not filters.
        Retrieved pictures are past evidence/analogies, not confirmed outcomes
        or connections for the current branch. Measured route spans retain their
        separate type; they never become next-event arrival labels.
        """
        if top_k < 1 or not -1 <= minimum_similarity <= 1:
            raise ValueError("invalid outcome-query limits")
        query = unit_vector(predicted_future, self.feature_dim)
        source = unit_vector(source_feature, self.feature_dim)
        branch = unit_vector(branch_direction, 2)
        sql = """SELECT sequence,episode,event,source_observation,target_observation,
                    source_feature,future_feature,branch,outcome FROM experiences
                    WHERE context=? AND sequence<=? AND outcome IN ('arrived','observed_span')"""
        arguments = [context, before_sequence]
        if exclude_episode is not None:
            sql += " AND episode<>?"
            arguments.append(exclude_episode)
        rows = self.connection.execute(sql + " ORDER BY sequence", arguments)
        matches = []
        for sequence, episode, event, source_obs, target_obs, raw_source, raw_future, raw_branch, outcome in rows:
            s = np.frombuffer(raw_source, dtype="<f4")
            t = np.frombuffer(raw_future, dtype="<f4")
            b = np.frombuffer(raw_branch, dtype="<f4")
            similarity = float(query @ t)
            if similarity >= minimum_similarity:
                matches.append(MemoryMatch(sequence, episode, event, source_obs, target_obs, similarity,
                                           float(source @ s), similarity, float(branch @ b),
                                           s.copy(), b.copy(), t.copy(), outcome_kind=outcome))
        return sorted(matches, key=lambda m: (-m.score, m.sequence))[:top_k]
