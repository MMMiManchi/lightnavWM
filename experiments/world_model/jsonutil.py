"""JSON helpers that refuse NaN/Inf instead of inventing a number."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_builtin(value: Any, *, where: str = "$") -> Any:
    """Convert numpy containers. Non-finite floats become None and are listed by the caller."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item, where=f"{where}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item, where=f"{where}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist(), where=where)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {where}")
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def dumps(value: Any) -> str:
    clean = to_builtin(value)
    return json.dumps(clean, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def loads(text: str) -> Any:
    return json.loads(text)
