from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.world_model import SCHEMA_VERSION
from experiments.world_model.campaign import (
    aggregate_campaign,
    register_campaign_run,
    validate_campaign_target,
)
from experiments.world_model.latent_data import scene_partition
from experiments.world_model.recorder import RunRecorder
from experiments.world_model.validate import validate_run


def _valid_scene() -> str:
    for index in range(100):
        scene = f"campaign-scene-{index}"
        if scene_partition(scene) == "train_fit":
            return scene
    raise AssertionError("could not find train_fit scene")


def _write_empty_valid_run(path: Path, campaign_id: str) -> Path:
    scene = _valid_scene()
    run = RunRecorder(path)
    run.add(
        "episodes",
        {
            "episode_uid": f"dataset|train|{scene}|1",
            "episode_id": "1",
            "scene": scene,
            "route_group_id": f"{scene}:route-1",
            "source_split": "train",
            "partition": "train_fit",
            "reset": {"key": [scene, "1"], "applied_in_reset": False},
            "completion": {"state": "completed"},
        },
    )
    run.close()
    run.write_json(
        "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "diagnostic_only": False,
            "source": {"split": "train"},
            "allowlist_keys": [[scene, "1"]],
            "served_keys": [[scene, "1"]],
            "reset_keys": [[scene, "1"]],
        },
    )
    run.write_json("validation_report.json", validate_run(path))
    return path


def test_campaign_revalidates_and_aggregates_runs_without_training_admission(tmp_path: Path):
    campaign_id = "train-fit-oct-2026"
    run_dir = _write_empty_valid_run(tmp_path / "run-001", campaign_id)
    manifest_path = tmp_path / "campaign" / "campaign_manifest.json"
    validate_campaign_target(manifest_path, campaign_id, run_dir)
    register_campaign_run(manifest_path, campaign_id, run_dir)

    report = aggregate_campaign(manifest_path)
    assert report["run_count"] == 1
    assert report["raw_coverage"]["episodes_per_scene"] == {_valid_scene(): 1}
    assert report["raw_coverage"]["non_budget_completed_episodes"] == 1
    assert report["raw_coverage_floor_met"] is False
    assert report["quota_eligible_transition_count"] == 0
    assert "dt_sim_s is null in wm-transition/0.1" not in report["quota_blockers"]
    assert report["quota_blockers"] == ["paired latent features are not authorized or available"]
    assert report["training_authorized"] is False
    assert report["training_admitted"] is False

    with pytest.raises(ValueError, match="already registered"):
        register_campaign_run(manifest_path, campaign_id, run_dir)


def test_campaign_rejects_scope_mismatch_before_collection(tmp_path: Path):
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps({"schema_version": "wm-campaign/0.1", "campaign_id": "other", "scope": "train_dev", "source_split": "train", "runs": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema or id mismatch"):
        validate_campaign_target(path, "train-fit", tmp_path / "run")
