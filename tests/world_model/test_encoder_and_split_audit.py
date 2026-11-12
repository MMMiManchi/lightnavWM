"""Encoder-cache and scene-inventory checks. No weights are downloaded or trained."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.world_model.feature_cache import (
    AUDITED_PREPROCESSOR,
    FEATURE_CACHE_VERSION,
    OUTPUT_SHAPES,
    SIGLIP_CUDA,
    SIGLIP_ENCODER_ID,
    SIGLIP_MODEL_CLASS,
    SIGLIP_PREPROCESS_SHA256,
    SIGLIP_REVISION,
    SIGLIP_TORCH,
    SIGLIP_TRANSFORMERS,
    SIGLIP_WEIGHT_SHA256,
    FeatureContractError,
    check_encoder_manifest,
    check_feature_cache,
    feature_checksum,
)
from experiments.world_model.jsonutil import sha256_file
from experiments.world_model.latent_data import (
    FormalSplitError,
    assert_episode_assignment,
    scene_partition,
)
from experiments.world_model.scene_inventory import build_train_scene_inventory
from experiments.world_model.scene_inventory import main as inventory_main

TRAIN_JSON = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz")


def _formal_episodes() -> tuple[list[dict], dict]:
    fit: list[str] = []
    dev: list[str] = []
    index = 0
    while len(fit) < 8 or len(dev) < 3:
        scene = f"s{index}"
        side = scene_partition(scene)
        if side == "train_fit" and len(fit) < 8:
            fit.append(scene)
        elif side == "train_dev" and len(dev) < 3:
            dev.append(scene)
        index += 1
    episodes = []
    for scene, count in zip(fit, [13, 1, 1, 1, 1, 1, 1, 1]):
        for route_index in range(count):
            episodes.append(
                {
                    "episode_uid": f"{scene}-{route_index}",
                    "scene": scene,
                    "episode_id": str(route_index),
                    "route_group_id": f"{scene}:{route_index}",
                    "source_split": "train",
                }
            )
    for scene, count in zip(dev, [3, 1, 1]):
        for route_index in range(count):
            episodes.append(
                {
                    "episode_uid": f"{scene}-{route_index}",
                    "scene": scene,
                    "episode_id": str(route_index),
                    "route_group_id": f"{scene}:{route_index}",
                    "source_split": "train",
                }
            )
    return episodes, episodes[0]


def _manifest(tmp_path: Path, **overrides) -> dict:
    weight = tmp_path / "weights.safetensors"
    weight.write_bytes(b"not-a-real-model")
    manifest = {
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "encoder_id": SIGLIP_ENCODER_ID,
        "revision": SIGLIP_REVISION,
        "model_class": SIGLIP_MODEL_CLASS,
        "weight_path": str(weight),
        "weight_sha256": SIGLIP_WEIGHT_SHA256,
        "preprocess_path": str(AUDITED_PREPROCESSOR),
        "preprocess_sha256": SIGLIP_PREPROCESS_SHA256,
        "transformers_version": SIGLIP_TRANSFORMERS,
        "torch_version": SIGLIP_TORCH,
        "cuda_version": SIGLIP_CUDA,
        "scene_split_version": "wm-scene-split/0.1",
        "feature_dtype": "float32",
        "allow_processor_defaults": False,
        "feature_layer_selected": None,
    }
    manifest.update(overrides)
    return manifest


def _row(image: Path, episode: dict, **overrides) -> dict:
    feature = [0.0] * OUTPUT_SHAPES["pooler_output"][0]
    row = {
        "episode_uid": episode["episode_uid"],
        "step_index": 0,
        "observation_id": "obs-1",
        "image_path": image.name,
        "image_sha256": sha256_file(image),
        "source_split": "train",
        "source_run": "formal/train_run",
        "scene": episode["scene"],
        "route_group_id": episode["route_group_id"],
        "scene_partition": scene_partition(episode["scene"]),
        "revision": SIGLIP_REVISION,
        "weight_sha256": SIGLIP_WEIGHT_SHA256,
        "preprocess_sha256": SIGLIP_PREPROCESS_SHA256,
        "output_name": "pooler_output",
        "feature_dtype": "float32",
        "feature_shape": [1152],
        "feature": feature,
        "feature_sha256": feature_checksum(feature),
    }
    row.update(overrides)
    if "feature" in overrides and "feature_sha256" not in overrides:
        row["feature_sha256"] = feature_checksum(row["feature"])
    return row


def test_missing_weights_images_dimensions_nonfinite_and_cache_mismatch(tmp_path: Path):
    image = tmp_path / "rgb.npy"
    image.write_bytes(b"rgb")
    episodes, episode = _formal_episodes()
    manifest = _manifest(tmp_path)
    check_feature_cache([_row(image, episode)], manifest, image_root=tmp_path, split_episodes=episodes)

    missing = dict(manifest)
    missing["weight_path"] = str(tmp_path / "absent.safetensors")
    with pytest.raises(FeatureContractError, match="missing_weights"):
        check_encoder_manifest(missing)

    with pytest.raises(FeatureContractError, match="missing_image"):
        check_feature_cache(
            [_row(image, episode, image_path="gone.npy")],
            manifest,
            image_root=tmp_path,
            split_episodes=episodes,
        )

    with pytest.raises(FeatureContractError, match="feature_dim_mismatch"):
        check_feature_cache(
            [_row(image, episode, feature=[0.0, 1.0])],
            manifest,
            image_root=tmp_path,
            split_episodes=episodes,
        )

    values = [0.0] * 1152
    values[2] = float("nan")
    with pytest.raises(FeatureContractError, match="non_finite_feature"):
        check_feature_cache(
            [_row(image, episode, feature=values)],
            manifest,
            image_root=tmp_path,
            split_episodes=episodes,
        )

    with pytest.raises(FeatureContractError, match="cache_config_mismatch"):
        check_feature_cache(
            [_row(image, episode, preprocess_sha256="other")],
            manifest,
            image_root=tmp_path,
            split_episodes=episodes,
        )


def test_cache_rejects_pilot_non_train_scene_cross_privilege_and_a_selected_layer(tmp_path: Path):
    image = tmp_path / "rgb.npy"
    image.write_bytes(b"rgb")
    episodes, episode = _formal_episodes()
    manifest = _manifest(tmp_path)
    kwargs = {"image_root": tmp_path, "split_episodes": episodes}
    with pytest.raises(FeatureContractError, match="pilot_source"):
        check_feature_cache(
            [_row(image, episode, source_run="output/world_model_data/pilot_20261023_train3_r2")],
            manifest,
            **kwargs,
        )
    with pytest.raises(FeatureContractError, match="non_train_split"):
        check_feature_cache([_row(image, episode, source_split="val_unseen")], manifest, **kwargs)
    opposite = "train_dev" if scene_partition(episode["scene"]) == "train_fit" else "train_fit"
    with pytest.raises(FeatureContractError, match="scene_cross"):
        check_feature_cache(
            [_row(image, episode, scene_partition=opposite)],
            manifest,
            **kwargs,
        )
    other = next(item for item in episodes if item["scene"] != episode["scene"])
    with pytest.raises(FeatureContractError, match="scene_cross"):
        check_feature_cache(
            [
                _row(image, episode, observation_id="obs-1"),
                _row(image, other, observation_id="obs-2", route_group_id=episode["route_group_id"]),
            ],
            manifest,
            **kwargs,
        )
    with pytest.raises(FeatureContractError, match="privileged_field"):
        check_feature_cache([_row(image, episode, goal_position=[0.0, 0.0, 0.0])], manifest, **kwargs)
    selected = dict(manifest)
    selected["feature_layer_selected"] = "pooler_output"
    with pytest.raises(FeatureContractError, match="feature_layer_selected"):
        check_encoder_manifest(selected)


def test_scene_and_route_cross_and_non_train_split_are_rejected(tmp_path: Path):
    same_scene = [
        {"episode_uid": "a", "scene": "sceneA", "route_group_id": "sceneA:1"},
        {"episode_uid": "b", "scene": "sceneA", "route_group_id": "sceneA:2"},
    ]
    with pytest.raises(FormalSplitError, match="scene"):
        assert_episode_assignment(same_scene, {"a": "train_fit", "b": "train_dev"})
    same_route = [
        {"episode_uid": "a", "scene": "sceneA", "route_group_id": "shared"},
        {"episode_uid": "b", "scene": "sceneB", "route_group_id": "shared"},
    ]
    with pytest.raises(FormalSplitError, match="route group"):
        assert_episode_assignment(same_route, {"a": "train_fit", "b": "train_dev"})
    with pytest.raises(SystemExit, match="train"):
        inventory_main(["--split", "val_unseen", "--out", str(tmp_path / "nope.json")])


def test_siglip_384_valid_conv_matches_729_positions_without_using_checkpoint_weights():
    import torch
    from torch import nn
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings

    config = SiglipVisionConfig(
        image_size=384,
        patch_size=14,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
    )
    embeddings = SiglipVisionEmbeddings(config)
    assert embeddings.num_patches == 729
    assert embeddings.position_embedding.weight.shape == (729, 32)
    tokens = embeddings(torch.zeros(1, 3, 384, 384))
    assert tokens.shape == (1, 729, 32)

    probe = nn.Conv2d(3, 1, 14, stride=14, padding="valid", bias=True)
    probe.weight.data.fill_(1)
    probe.bias.data.zero_()
    trailing = torch.zeros(1, 3, 384, 384)
    trailing[:, :, 378:, :] = 1
    trailing[:, :, :, 378:] = 1
    assert tuple(probe(trailing).shape[-2:]) == (27, 27)
    assert int(probe(trailing).detach().count_nonzero()) == 0
    assert tuple(probe(torch.zeros(1, 3, 378, 378)).shape[-2:]) == (27, 27)


def test_official_train_inventory_meets_the_scene_floors():
    if not TRAIN_JSON.is_file():
        pytest.skip("official train json is not present")
    inventory = build_train_scene_inventory(TRAIN_JSON.parents[1])
    assert inventory["source_split"] == "train"
    assert inventory["eligible"] is True
    counts = inventory["counts"]
    assert counts["train_fit_scenes"] >= 8
    assert counts["train_fit_route_groups"] >= 20
    assert counts["train_dev_scenes"] >= 3
    assert counts["train_dev_route_groups"] >= 5
    assert counts["shared_scenes"] == []
    assert counts["shared_routes"] == []
    assert counts["train_fit_episodes"] != counts["train_fit_route_groups"]
    scenes = {row["scene"] for row in inventory["scenes"]}
    assert len(scenes) == len(inventory["scenes"])
    fit = {row["scene"] for row in inventory["scenes"] if row["scene_partition"] == "train_fit"}
    dev = {row["scene"] for row in inventory["scenes"] if row["scene_partition"] == "train_dev"}
    assert fit.isdisjoint(dev)
    for row in inventory["scenes"]:
        assert len(row["route_group_ids"]) == row["n_route_groups"]
        assert len(set(row["route_group_ids"])) == row["n_route_groups"]
    unseen = inventory["other_splits"]["val_unseen"]
    assert unseen["scene_intersection_with_train"] == []
    assert unseen["included_in_train_inventory"] is False
    collisions = inventory["other_splits"]["val_seen"]["episode_id_collisions"]
    assert collisions
    assert all(item["same_trajectory_id"] is False and item["same_instruction_text"] is False for item in collisions)
    assert inventory["collection_status"] == "not_collected"
    assert json.dumps(inventory).find("val_unseen")
