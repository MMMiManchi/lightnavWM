"""Contract tests for the v1 latent world model. They do not train on the pilot."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.world_model.baselines import (
    copy_latent,
    denormalize_raw,
    executed_twist,
    kinematic_delta,
    latent_errors,
    pose_errors,
)
from experiments.world_model.latent_data import (
    MIN_DEV_SCENES,
    MIN_FIT_SCENES,
    FormalSplitError,
    assert_episode_assignment,
    end_source_counts,
    formal_report,
    load_accepted_run,
    read_recorded_run,
    require_formal_split,
    scene_partition,
    supervised_rows,
)
from experiments.world_model.pose_math import body_frame_delta, yaw_from_xyzw
from experiments.world_model.predictor import ResidualLatentPredictor, separate_losses
from lightnav.velocity import _normalize_to_raw

ACCEPTED = Path("output/world_model_data/pilot_20261023_train3_r2")
ABORTED = Path("output/world_model_data/pilot_20261023_train3")


def _episode(scene: str, episode_id: str, route: str | None = None) -> dict:
    return {
        "episode_uid": f"train|{scene}|{episode_id}",
        "scene": scene,
        "episode_id": episode_id,
        "route_group_id": route or f"{scene}:1",
        "source_split": "train",
        "partition": "train_fit",
    }


def _scenes(side: str, count: int) -> list[str]:
    found = []
    index = 0
    while len(found) < count:
        name = f"s{index}"
        if scene_partition(name) == side:
            found.append(name)
        index += 1
    return found


def test_body_frame_matches_yaw_convention():
    assert yaw_from_xyzw((0.0, 0.0, 0.0, 1.0)) == pytest.approx(0.0)
    forward, left, yaw, up = body_frame_delta((0.0, 0.0, 0.0), 0.0, (0.0, 0.0, -1.0), 0.0)
    assert (forward, left, yaw, up) == pytest.approx((1.0, 0.0, 0.0, 0.0))
    forward, left, yaw, up = body_frame_delta((0.0, 0.0, 0.0), 0.0, (-1.0, 0.0, 0.0), 0.0)
    assert (forward, left, yaw, up) == pytest.approx((0.0, 1.0, 0.0, 0.0))
    _forward, _left, yaw, _up = body_frame_delta((0.0, 0.0, 0.0), 3.0, (0.0, 0.0, 0.0), 3.0 + 3.5)
    assert yaw == pytest.approx(3.5 - 2.0 * np.pi)


def test_kinematic_baseline_uses_the_submitted_command():
    units = {
        "lin_vel_range_mps": [0.0, 2.5],
        "ang_vel_range_dps": [-300.0, 300.0],
        "min_abs_lin_speed_mps": 0.01,
        "min_abs_ang_speed_dps": 0.5,
        "control_dt_s": 0.1,
    }
    assert denormalize_raw(_normalize_to_raw(2.0, 0.0, 2.5), 0.0, 2.5) == pytest.approx(2.0)
    request = {"action": "velocity_control", "action_args": {"linear_velocity": 0.6, "angular_velocity": 0.0}}
    final = {"action": "velocity_control", "action_args": {"linear_velocity": -1.0, "angular_velocity": 0.0}}
    transition = {
        "action": {
            "client_request": request,
            "execution_ack": {"final_habitat_action": final, "control_units": units, "dt_command_s": 0.1},
            "dt_sim_s": None,
        }
    }
    twist = executed_twist(transition)
    assert twist["source"] == "final_habitat_action"
    assert twist["request_differs"] is True
    assert twist["stop_threshold"] is True
    assert twist["v_mps"] == 0.0
    delta = kinematic_delta(transition)
    assert (delta["forward_m"], delta["left_m"], delta["delta_yaw_rad"]) == (0.0, 0.0, 0.0)

    moving = {
        "action": {
            "client_request": request,
            "execution_ack": {
                "final_habitat_action": request,
                "control_units": units,
                "dt_command_s": 0.1,
            },
            "dt_sim_s": None,
        }
    }
    moved = kinematic_delta(moving)
    assert moved["forward_m"] == pytest.approx(0.2)
    assert moved["left_m"] == 0.0
    assert moved["delta_yaw_rad"] == pytest.approx(0.0)
    discrete = {
        "action": {
            "client_request": request,
            "execution_ack": {"final_habitat_action": 0, "control_units": units, "dt_command_s": None},
            "dt_sim_s": None,
        }
    }
    assert executed_twist(discrete)["stop_reason"] == "discrete_stop"
    assert kinematic_delta(discrete)["forward_m"] == 0.0


def test_copy_baseline_rejects_a_shape_change():
    latent = np.array([1.0, 0.0, 0.0])
    copied = copy_latent(latent)
    assert copied is not latent
    assert latent_errors(copied, np.array([1.0, 0.0, 0.0]))["mse"] == 0.0
    assert latent_errors(copied, np.array([0.0, 1.0, 0.0]))["cosine_distance"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="shapes"):
        latent_errors(copied, np.zeros(4))


def test_losses_stay_on_separate_scales_and_zero_init_copies_latent():
    torch.manual_seed(0)
    model = ResidualLatentPredictor(4, hidden=8)
    latent = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    control = torch.tensor([[1.5, -0.2, 0.1]])
    hat, pose = model(latent, control)
    assert torch.allclose(hat, latent)
    assert torch.allclose(pose, torch.zeros_like(pose))
    target_latent = latent + 0.5
    pose_a = torch.tensor([[0.1, 0.0, 0.3]])
    pose_b = torch.tensor([[4.0, -2.0, 1.2]])
    first = separate_losses(hat, target_latent, pose, pose_a)
    second = separate_losses(hat, target_latent, pose, pose_b)
    assert first["latent_mse"].item() == pytest.approx(second["latent_mse"].item())
    assert first["translation_huber"].item() != pytest.approx(second["translation_huber"].item())
    yaw_only = separate_losses(hat, target_latent, torch.zeros(1, 3), torch.tensor([[0.0, 0.0, np.pi]]))
    assert yaw_only["yaw_cosine"].item() == pytest.approx(2.0)


def test_formal_split_floors_and_route_collision():
    fit = _scenes("train_fit", MIN_FIT_SCENES)
    dev = _scenes("train_dev", MIN_DEV_SCENES)
    episodes = []
    for scene in fit:
        for route_index in range(3):
            episodes.append(_episode(scene, str(route_index), f"{scene}:{route_index}"))
    for scene in dev:
        for route_index in range(2):
            episodes.append(_episode(scene, str(route_index), f"{scene}:{route_index}"))
    report = formal_report(episodes)
    assert report["eligible"] is True
    assert report["train_dev_scenes"] >= MIN_DEV_SCENES
    require_formal_split(episodes)

    short = [_episode(scene, "1") for scene in fit[:2] + dev[:1]]
    blocked = formal_report(short)
    assert blocked["eligible"] is False
    with pytest.raises(FormalSplitError):
        require_formal_split(short)

    same_route = [_episode("sceneA", "1", "shared:4"), _episode("sceneB", "2", "shared:4")]
    with pytest.raises(FormalSplitError, match="route group"):
        assert_episode_assignment(
            same_route,
            {same_route[0]["episode_uid"]: "train_fit", same_route[1]["episode_uid"]: "train_dev"},
        )
    shared_scene = [_episode("sceneA", "1", "sceneA:1"), _episode("sceneA", "2", "sceneA:2")]
    with pytest.raises(FormalSplitError, match="scene"):
        assert_episode_assignment(
            shared_scene,
            {shared_scene[0]["episode_uid"]: "train_fit", shared_scene[1]["episode_uid"]: "train_dev"},
        )


def test_route_hash_and_scene_hash_are_different_rules():
    found = None
    for index in range(5000):
        scene = f"mp3d{index}"
        route = f"{scene}:1"
        if scene_partition(scene) != scene_partition(route):
            found = (scene, route)
            break
    assert found is not None


def test_accepted_pilot_is_blocked_and_its_rows_match_the_contract():
    if not ACCEPTED.is_dir():
        pytest.skip("accepted pilot run is not present")
    with pytest.raises(FormalSplitError, match="pilot_"):
        load_accepted_run(ACCEPTED)
    bundle = read_recorded_run(ACCEPTED)
    assert bundle["seed_status"] == "partial"
    assert bundle["manifest"]["source"]["split"] == "train"
    report = formal_report(bundle["episodes"])
    assert report["eligible"] is False
    assert report["train_dev_scenes"] == 0
    assert report["train_fit_scenes"] == 2
    assert report["train_fit_route_groups"] == 2
    with pytest.raises(FormalSplitError):
        require_formal_split(bundle["episodes"])

    with pytest.raises(FormalSplitError, match="consistent measured simulator clock"):
        supervised_rows(bundle)
    with pytest.raises(FormalSplitError):
        load_accepted_run(ABORTED)
    with pytest.raises(FormalSplitError):
        read_recorded_run(ABORTED)
    with pytest.raises(FormalSplitError):
        load_accepted_run(Path("output/r2r_probe"))


def test_failure_json_and_val_split_are_refused(tmp_path: Path):
    run = tmp_path / "partial_run"
    run.mkdir()
    (run / "failure.json").write_text("{}", encoding="utf-8")
    os.chmod(run / "failure.json", 0o444)
    os.chmod(run, 0o555)
    with pytest.raises(FormalSplitError, match="failure.json"):
        load_accepted_run(run)
    other = tmp_path / "val_run"
    other.mkdir()
    (other / "manifest.json").write_text(
        json.dumps({"schema_version": "wm-transition/0.2", "source": {"split": "val_unseen"}, "seed": {"status": "partial"}}),
        encoding="utf-8",
    )
    (other / "validation_report.json").write_text(json.dumps({"hard_error_count": 0}), encoding="utf-8")
    os.chmod(other / "manifest.json", 0o444)
    os.chmod(other / "validation_report.json", 0o444)
    os.chmod(other, 0o555)
    with pytest.raises(FormalSplitError, match="train"):
        load_accepted_run(other)
