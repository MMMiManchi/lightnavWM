"""Align one transition with two feature rows. This module does not encode or cache.

``PAIRS_AUTHORIZE_TRAINING`` stays false. A structurally aligned pair is not a
training sample and does not admit a run. It also does not open image files or
replace the on-disk cache checker.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from experiments.world_model.feature_cache import (
    OUTPUT_SHAPES,
    FeatureContractError,
    _reject_row_source,
    scene_partition,
)

PAIRS_AUTHORIZE_TRAINING = False


def align_transition_latents(
    transition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    episode: Mapping[str, Any],
    *,
    output_name: str,
) -> dict[str, Any]:
    """Match one transition to two rows of the same ``output_name``.

    The join key is ``transition_id`` plus ``obs_before_id`` and
    ``obs_after_id``. Each feature row's ``image_sha256`` must equal that
    observation record's ``rgb.sha256``. Both rows must name the same run,
    scene, route group, and scene partition, and that scene and route must be
    the episode that owns the transition. ``scene_partition`` must be the
    registered scene hash, not a route-group label. The action in the result is
    ``final_habitat_action`` from the execution acknowledgement. The returned
    record names checksums only and does not open image files.
    """
    if PAIRS_AUTHORIZE_TRAINING:
        raise FeatureContractError("pair_authorization", "pair alignment cannot authorize training")
    if output_name not in OUTPUT_SHAPES:
        raise FeatureContractError("feature_dim_mismatch", output_name)
    transition_id = transition.get("transition_id")
    before_id = transition.get("obs_before_id")
    after_id = transition.get("obs_after_id")
    episode_uid = transition.get("episode_uid")
    if not transition_id or not before_id or not after_id or not episode_uid:
        raise FeatureContractError("missing_field", "transition is missing id, episode, or observation refs")
    if before_id == after_id:
        raise FeatureContractError("reference_mismatch", "obs_before_id and obs_after_id are the same")
    action_id, executed = _require_executed(transition)
    try:
        step = int(transition["t"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureContractError("missing_field", "transition t") from exc
    before = _select(rows, observation_id=str(before_id), output_name=output_name)
    after = _select(rows, observation_id=str(after_id), output_name=output_name)
    before_obs = _require_observation(
        observations,
        observation_id=str(before_id),
        episode_uid=str(episode_uid),
        step_index=step,
    )
    after_obs = _require_observation(
        observations,
        observation_id=str(after_id),
        episode_uid=str(episode_uid),
        step_index=step + 1,
    )
    _require_same_encoder(before, after)
    _require_bound(before, episode_uid=str(episode_uid), step_index=step, output_name=output_name)
    _require_bound(after, episode_uid=str(episode_uid), step_index=step + 1, output_name=output_name)
    _require_same_collection_identity(before, after)
    _require_registered_partition(before)
    _require_episode(
        episode,
        episode_uid=str(episode_uid),
        scene=str(before["scene"]),
        route_group_id=str(before["route_group_id"]),
    )
    _require_same_image(before, before_obs)
    _require_same_image(after, after_obs)
    try:
        sim_before = float(before_obs["sim_time_s"])
        sim_after = float(after_obs["sim_time_s"])
        action = transition.get("action") or {}
        dt_sim = float(action["dt_sim_s"])
        consistent_clock = (
            np.isfinite(sim_before) and np.isfinite(sim_after) and np.isfinite(dt_sim)
            and sim_after >= sim_before and abs(dt_sim - (sim_after - sim_before)) <= 1e-9
            and float(action["sim_time_before_s"]) == sim_before
            and float(action["sim_time_after_s"]) == sim_after
        )
    except (KeyError, TypeError, ValueError):
        consistent_clock = False
    if not consistent_clock:
        raise FeatureContractError("sim_time_mismatch", str(transition_id))
    return {
        "transition_id": str(transition_id),
        "episode_uid": str(episode_uid),
        "output_name": output_name,
        "obs_before_id": str(before_id),
        "obs_after_id": str(after_id),
        "action_id": action_id,
        "executed_action": executed,
        "image_before_sha256": before["image_sha256"],
        "image_after_sha256": after["image_sha256"],
        "z_t_sha256": before["feature_sha256"],
        "z_t1_sha256": after["feature_sha256"],
        "source_run": before["source_run"],
        "scene": before["scene"],
        "route_group_id": before["route_group_id"],
        "scene_partition": before["scene_partition"],
        "admits_training": False,
        "reason": "aligned features are not a training sample while the layer is unselected",
    }


def _require_executed(transition: Mapping[str, Any]) -> tuple[str, Any]:
    action = transition.get("action") or {}
    ack = action.get("execution_ack") or {}
    action_id = action.get("action_id")
    if not action_id or ack.get("action_id") != action_id:
        raise FeatureContractError("action_id_mismatch", str(transition.get("transition_id")))
    quality = (transition.get("quality") or {}).get("status")
    if quality != "valid" or ack.get("status") != "executed" or ack.get("executed") is not True:
        raise FeatureContractError("not_executed", f"{transition.get('transition_id')} cannot be paired")
    final = ack.get("final_habitat_action")
    if final is None:
        raise FeatureContractError("missing_field", "final_habitat_action")
    if action.get("final_habitat_action") != final:
        raise FeatureContractError(
            "reference_mismatch",
            "final_habitat_action disagrees with execution_ack",
        )
    return str(action_id), final


def _require_observation(
    observations: Sequence[Mapping[str, Any]],
    *,
    observation_id: str,
    episode_uid: str,
    step_index: int,
) -> Mapping[str, Any]:
    matched = [row for row in observations if str(row.get("observation_id")) == observation_id]
    if not matched:
        raise FeatureContractError("reference_mismatch", f"missing observation {observation_id}")
    if len(matched) > 1:
        raise FeatureContractError("reference_mismatch", f"duplicate observation {observation_id}")
    row = matched[0]
    if str(row.get("episode_uid")) != episode_uid:
        raise FeatureContractError("reference_mismatch", "observation episode_uid does not match")
    try:
        recorded_step = int(row["step_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureContractError("missing_field", "observation step_index") from exc
    if recorded_step != step_index:
        raise FeatureContractError("reference_mismatch", f"observation step_index {recorded_step} != {step_index}")
    if row.get("rgb_missing") is True:
        raise FeatureContractError("image_missing", observation_id)
    rgb = row.get("rgb")
    if not isinstance(rgb, Mapping) or not rgb.get("sha256"):
        raise FeatureContractError("image_missing", observation_id)
    try:
        sim_time = float(row["sim_time_s"])
        valid_time = np.isfinite(sim_time) and sim_time >= 0
    except (KeyError, TypeError, ValueError):
        valid_time = False
    if not valid_time:
        raise FeatureContractError("sim_time_unavailable", observation_id)
    return row


def _require_same_image(feature: Mapping[str, Any], observation: Mapping[str, Any]) -> None:
    rgb = observation["rgb"]
    if feature.get("image_sha256") != rgb.get("sha256"):
        raise FeatureContractError(
            "image_hash_mismatch",
            f"{observation.get('observation_id')} feature image does not match rgb.sha256",
        )
    if not feature.get("image_path") or feature.get("image_path") != rgb.get("path"):
        raise FeatureContractError(
            "reference_mismatch",
            f"{observation.get('observation_id')} feature image_path does not match the observation rgb path",
        )


def _select(
    rows: Sequence[Mapping[str, Any]],
    *,
    observation_id: str,
    output_name: str,
) -> Mapping[str, Any]:
    matched = [
        row
        for row in rows
        if str(row.get("observation_id")) == observation_id and str(row.get("output_name")) == output_name
    ]
    if not matched:
        raise FeatureContractError("missing_feature", f"{observation_id} {output_name}")
    if len(matched) > 1:
        raise FeatureContractError("duplicate_feature", f"{observation_id} {output_name}")
    return matched[0]


def _require_episode(
    episode: Mapping[str, Any],
    *,
    episode_uid: str,
    scene: str,
    route_group_id: str,
) -> None:
    if str(episode.get("episode_uid")) != episode_uid:
        raise FeatureContractError("reference_mismatch", "episode_uid does not match the transition")
    if episode.get("source_split") != "train":
        raise FeatureContractError("non_train_split", str(episode.get("source_split")))
    if str(episode.get("scene")) != scene or str(episode.get("route_group_id")) != route_group_id:
        raise FeatureContractError(
            "reference_mismatch",
            "feature scene or route does not match the episode",
        )


def _require_registered_partition(row: Mapping[str, Any]) -> None:
    expected = scene_partition(str(row.get("scene")))
    if row.get("scene_partition") != expected:
        raise FeatureContractError(
            "scene_cross",
            f"{row.get('scene')} is {row.get('scene_partition')} not {expected}",
        )


def _require_same_collection_identity(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    for field in ("source_run", "scene", "route_group_id", "scene_partition"):
        if not before.get(field) or before.get(field) != after.get(field):
            raise FeatureContractError("reference_mismatch", f"paired rows disagree on {field}")


def _require_same_encoder(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    for field in ("revision", "weight_sha256", "preprocess_sha256"):
        if before.get(field) != after.get(field) or not before.get(field):
            raise FeatureContractError("cache_config_mismatch", f"paired rows disagree on {field}")


def _require_bound(
    row: Mapping[str, Any],
    *,
    episode_uid: str,
    step_index: int,
    output_name: str,
) -> None:
    _reject_row_source(row)
    if row.get("feature_layer_selected") is not None:
        raise FeatureContractError("feature_layer_selected", "a paired row must not select a layer")
    if str(row.get("episode_uid")) != episode_uid:
        raise FeatureContractError("reference_mismatch", "feature episode_uid does not match the transition")
    if str(row.get("output_name")) != output_name:
        raise FeatureContractError("feature_dim_mismatch", str(row.get("output_name")))
    if row.get("feature_dtype") != "float32":
        raise FeatureContractError("cache_config_mismatch", "feature_dtype is not float32")
    try:
        actual = tuple(int(dim) for dim in row["feature_shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureContractError("feature_dim_mismatch", "feature_shape") from exc
    expected = OUTPUT_SHAPES[output_name]
    if actual != expected:
        raise FeatureContractError("feature_dim_mismatch", f"{output_name} shape {actual} != {expected}")
    try:
        recorded_step = int(row["step_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureContractError("missing_field", "feature step_index") from exc
    if recorded_step != step_index:
        raise FeatureContractError("reference_mismatch", f"step_index {recorded_step} != {step_index}")
    if not row.get("feature_sha256") or not row.get("image_sha256"):
        raise FeatureContractError("missing_field", "feature_sha256 or image_sha256")
