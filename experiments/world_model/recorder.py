"""Append-only run directory. An existing directory is refused, never cleared."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from experiments.world_model.group_manifest import refuse_legacy_path
from experiments.world_model.jsonutil import dumps, sha256_file

STREAMS = (
    "episodes",
    "observations",
    "transitions",
    "privileged_labels",
)


class RunRecorder:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        refuse_legacy_path(self.run_dir)
        self.run_dir.mkdir(parents=False, exist_ok=False)
        (self.run_dir / "assets").mkdir()
        self._handles = {
            name: open(self.run_dir / f"{name}.jsonl.partial", "w", encoding="utf-8")
            for name in STREAMS
        }
        self.closed = False

    def save_array(self, observation_id: str, kind: str, array: np.ndarray) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("recorder is closed")
        path = self.run_dir / "assets" / f"{observation_id}_{kind}.npy"
        if path.exists():
            raise FileExistsError(path)
        np.save(path, np.asarray(array))
        return {
            "path": str(path.relative_to(self.run_dir)),
            "sha256": sha256_file(path),
            "shape": [int(dim) for dim in np.asarray(array).shape],
            "dtype": str(np.asarray(array).dtype),
        }

    def add(self, stream: str, record: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("recorder is closed")
        if stream not in self._handles:
            raise KeyError(stream)
        handle = self._handles[stream]
        handle.write(dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def close(self) -> None:
        if self.closed:
            return
        for handle in self._handles.values():
            handle.close()
        for name in STREAMS:
            source = self.run_dir / f"{name}.jsonl.partial"
            target = self.run_dir / f"{name}.jsonl"
            if target.exists():
                raise FileExistsError(target)
            os.replace(source, target)
        self.closed = True

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        if "/" in name or name.startswith("."):
            raise ValueError(name)
        path = self.run_dir / name
        if path.exists():
            raise FileExistsError(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(dumps(payload) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path
