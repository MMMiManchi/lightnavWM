"""Offline alignment of one transition to two feature rows. No encoder and no cache file."""

from __future__ import annotations

import pytest

from experiments.world_model.feature_cache import FeatureContractError
from experiments.world_model.feature_pairs import PAIRS_AUTHORIZE_TRAINING, align_transition_latents
from experiments.world_model.latent_data import scene_partition

_EXECUTED = {
    "action": "velocity_control",
    "action_args": {"linear_velocity": 1.0, "angular_velocity": 0.0},
}


def _transition(**overrides):
    row = {
        "transition_id": "tr-1",
        "episode_uid": "ep-1",
        "t": 3,
        "obs_before_id": "obs-before",
        "obs_after_id": "obs-after",
        "quality": {"status": "valid"},
        "action": {
            "action_id": "act-1",
            "dt_sim_s": 0.016,
            "sim_time_before_s": 1.0,
            "sim_time_after_s": 1.016,
            "client_request": {"action": "velocity_control", "action_args": {"linear_velocity": 0.0}},
            "final_habitat_action": _EXECUTED,
            "execution_ack": {
                "status": "executed",
                "executed": True,
                "action_id": "act-1",
                "final_habitat_action": _EXECUTED,
            },
        },
    }
    row.update(overrides)
    return row


def _observation(observation_id: str, step_index: int, **overrides):
    row = {
        "observation_id": observation_id,
        "episode_uid": "ep-1",
        "step_index": step_index,
        "rgb": {"path": f"assets/{observation_id}_rgb.npy", "sha256": f"img-{observation_id}"},
        "rgb_missing": False,
        "sim_time_s": 1.0 if step_index == 3 else 1.016,
    }
    row.update(overrides)
    return row


def _observations():
    return [_observation("obs-before", 3), _observation("obs-after", 4)]


def _episode(**overrides):
    row = {
        "episode_uid": "ep-1",
        "scene": "scene-a",
        "route_group_id": "route-a",
        "source_split": "train",
    }
    row.update(overrides)
    return row


def _align(transition, rows, **kwargs):
    observations = kwargs.pop("observations", _observations())
    episode = kwargs.pop("episode", _episode())
    return align_transition_latents(transition, rows, observations, episode, **kwargs)


def _feature(observation_id: str, step_index: int, *, output_name: str = "pooler_output", **overrides):
    shape = (1152,) if output_name == "pooler_output" else (729, 1152)
    row = {
        "observation_id": observation_id,
        "episode_uid": "ep-1",
        "step_index": step_index,
        "output_name": output_name,
        "revision": "rev",
        "weight_sha256": "w",
        "preprocess_sha256": "p",
        "feature_dtype": "float32",
        "feature_shape": shape,
        "feature_sha256": f"sha-{observation_id}-{output_name}",
        "image_sha256": f"img-{observation_id}",
        "source_split": "train",
        "source_run": "formal_run",
        "scene": "scene-a",
        "route_group_id": "route-a",
        "scene_partition": scene_partition("scene-a"),
        "image_path": f"assets/{observation_id}_rgb.npy",
    }
    row.update(overrides)
    return row


def test_alignment_names_the_same_output_and_refuses_training():
    assert PAIRS_AUTHORIZE_TRAINING is False
    aligned = _align(
        _transition(),
        [
            _feature("obs-before", 3),
            _feature("obs-after", 4),
            _feature("obs-before", 3, output_name="last_hidden_state", feature_sha256="other"),
        ],
        output_name="pooler_output",
    )
    assert aligned["transition_id"] == "tr-1"
    assert aligned["obs_before_id"] == "obs-before"
    assert aligned["obs_after_id"] == "obs-after"
    assert aligned["output_name"] == "pooler_output"
    assert aligned["action_id"] == "act-1"
    assert aligned["executed_action"] == _EXECUTED
    assert aligned["executed_action"]["action_args"]["linear_velocity"] == 1.0
    assert aligned["image_before_sha256"] == "img-obs-before"
    assert aligned["image_after_sha256"] == "img-obs-after"
    assert aligned["scene"] == "scene-a"
    assert aligned["route_group_id"] == "route-a"
    assert aligned["source_run"] == "formal_run"
    assert aligned["z_t_sha256"] == "sha-obs-before-pooler_output"
    assert aligned["z_t1_sha256"] == "sha-obs-after-pooler_output"
    assert aligned["admits_training"] is False
    assert "feature" not in aligned


def test_alignment_rejects_crossed_observations_and_bad_rows():
    rows = [_feature("obs-before", 3), _feature("obs-after", 4)]
    with pytest.raises(FeatureContractError, match="missing_feature"):
        _align(_transition(), rows, output_name="last_hidden_state")
    with pytest.raises(FeatureContractError, match="step_index"):
        _align(
            _transition(),
            [_feature("obs-before", 3), _feature("obs-after", 9)],
            output_name="pooler_output",
        )
    with pytest.raises(FeatureContractError, match="disagree"):
        _align(
            _transition(),
            [_feature("obs-before", 3), _feature("obs-after", 4, revision="other")],
            output_name="pooler_output",
        )
    with pytest.raises(FeatureContractError, match="cannot be paired"):
        _align(
            _transition(quality={"status": "quarantined"}),
            rows,
            output_name="pooler_output",
        )
    forged = _transition()
    forged["action"] = dict(forged["action"])
    forged["action"]["dt_sim_s"] = 0.1
    with pytest.raises(FeatureContractError, match="sim_time_mismatch"):
        _align(forged, rows, output_name="pooler_output")
    with pytest.raises(FeatureContractError, match="diagnostic"):
        _align(
            _transition(),
            [_feature("obs-before", 3, diagnostic_only=True), _feature("obs-after", 4)],
            output_name="pooler_output",
        )
    with pytest.raises(FeatureContractError, match="action_id_mismatch"):
        mismatched = _transition()
        mismatched["action"] = dict(mismatched["action"])
        mismatched["action"]["execution_ack"] = dict(mismatched["action"]["execution_ack"])
        mismatched["action"]["execution_ack"]["action_id"] = "other"
        _align(mismatched, rows, output_name="pooler_output")
    with pytest.raises(FeatureContractError, match="final_habitat_action"):
        missing_final = _transition()
        missing_final["action"] = dict(missing_final["action"])
        missing_final["action"]["execution_ack"] = dict(missing_final["action"]["execution_ack"])
        missing_final["action"]["execution_ack"]["final_habitat_action"] = None
        _align(missing_final, rows, output_name="pooler_output")
    with pytest.raises(FeatureContractError, match="duplicate_feature"):
        _align(
            _transition(),
            [_feature("obs-before", 3), _feature("obs-before", 3), _feature("obs-after", 4)],
            output_name="pooler_output",
        )
    with pytest.raises(FeatureContractError, match="same"):
        same = _transition(obs_after_id="obs-before")
        _align(same, rows, output_name="pooler_output")
    with pytest.raises(FeatureContractError, match="shape"):
        _align(
            _transition(),
            [_feature("obs-before", 3, feature_shape=(1,)), _feature("obs-after", 4)],
            output_name="pooler_output",
        )
    with pytest.raises(FeatureContractError, match="pilot"):
        _align(
            _transition(),
            [_feature("obs-before", 3, source_run="pilot_run"), _feature("obs-after", 4)],
            output_name="pooler_output",
        )
    with pytest.raises(FeatureContractError, match="feature_layer_selected"):
        _align(
            _transition(),
            [
                _feature("obs-before", 3, feature_layer_selected="pooler_output"),
                _feature("obs-after", 4),
            ],
            output_name="pooler_output",
        )
    with pytest.raises(FeatureContractError, match="does not match the episode"):
        _align(
            _transition(),
            rows,
            episode=_episode(scene="scene-b"),
            output_name="pooler_output",
        )
    with pytest.raises(FeatureContractError, match="scene_cross"):
        _align(
            _transition(),
            [
                _feature("obs-before", 3, scene_partition="not-a-partition"),
                _feature("obs-after", 4, scene_partition="not-a-partition"),
            ],
            output_name="pooler_output",
        )
    with pytest.raises(FeatureContractError, match="scene"):
        _align(
            _transition(),
            [_feature("obs-before", 3, scene="scene-b"), _feature("obs-after", 4)],
            output_name="pooler_output",
        )
    with pytest.raises(FeatureContractError, match="episode_uid"):
        _align(
            _transition(),
            [_feature("obs-before", 3, episode_uid="other"), _feature("obs-after", 4)],
            output_name="pooler_output",
        )
    with pytest.raises(FeatureContractError, match="image_path"):
        _align(
            _transition(),
            [_feature("obs-before", 3, image_path="assets/other.npy"), _feature("obs-after", 4)],
            output_name="pooler_output",
        )
    with pytest.raises(FeatureContractError, match="image_hash_mismatch"):
        _align(
            _transition(),
            [_feature("obs-before", 3, image_sha256="other"), _feature("obs-after", 4)],
            output_name="pooler_output",
        )
    forged_clock = _observation("obs-before", 3, sim_time_s=0.016)
    with pytest.raises(FeatureContractError, match="sim_time_mismatch"):
        _align(
            _transition(),
            rows,
            observations=[forged_clock, _observation("obs-after", 4)],
            output_name="pooler_output",
        )
