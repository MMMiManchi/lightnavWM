"""Narrow validation for the one authorized Habitat diagnostic trial."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

TRIAL_NAME = "controlled_trial_20261027_01"
MAX_EPISODES = 6
MAX_STEPS = 200
REQUIRED_SCENES = 2


def trial_directory(repo_root: Path) -> Path:
    return (Path(repo_root) / "output" / "world_model" / TRIAL_NAME).resolve()


def validate_controlled_trial(
    *,
    repo_root: Path,
    run_dir: Path,
    max_steps: int,
    episodes: Iterable[dict[str, Any]],
    require_absent: bool,
) -> list[dict[str, Any]]:
    expected = trial_directory(repo_root)
    actual = Path(run_dir).resolve()
    if actual != expected:
        raise ValueError(f"controlled trial only permits {expected}")
    if require_absent and actual.exists():
        raise FileExistsError(f"refusing to overwrite {actual}")
    if not 1 <= int(max_steps) <= MAX_STEPS:
        raise ValueError(f"controlled-trial max_steps must be in 1..{MAX_STEPS}")
    rows = list(episodes)
    if not rows or len(rows) > MAX_EPISODES:
        raise ValueError(f"controlled trial requires 1..{MAX_EPISODES} episodes")
    scenes = {str(row.get("scene", "")) for row in rows}
    if len(scenes) != REQUIRED_SCENES:
        raise ValueError(f"controlled trial requires exactly {REQUIRED_SCENES} scenes")
    keys = [(str(row.get("scene", "")), str(row.get("episode_id", ""))) for row in rows]
    if len(set(keys)) != len(keys) or any(not scene or not episode for scene, episode in keys):
        raise ValueError("controlled-trial episode keys must be non-empty and unique")
    for row in rows:
        if row.get("source_split", "train") != "train":
            raise ValueError("controlled trial only accepts the official train split")
        if row.get("partition") != "train_fit":
            raise ValueError("controlled trial only accepts train_fit episodes")
    return rows
