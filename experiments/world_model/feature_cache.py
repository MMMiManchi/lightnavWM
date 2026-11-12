"""Checks for a frozen SigLIP feature cache. This module does not encode images.

Schema ``wm-feature-cache/0.2`` binds every row to the audited vision tower.
``feature_layer_selected`` stays null: a row may store either candidate output,
and that does not select it. No formal cache is written here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from experiments.world_model.jsonutil import sha256_file
from experiments.world_model.latent_data import (
    FORBIDDEN_INPUT_KEYS,
    SCENE_SPLIT_VERSION,
    FormalSplitError,
    require_formal_split,
    scene_partition,
)

FEATURE_CACHE_VERSION = "wm-feature-cache/0.2"
SIGLIP_ENCODER_ID = "google/siglip-so400m-patch14-384"
SIGLIP_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
SIGLIP_MODEL_CLASS = "SiglipVisionModel"
SIGLIP_WEIGHT_SHA256 = "ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a"
SIGLIP_PREPROCESS_SHA256 = "f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8"
SIGLIP_TRANSFORMERS = "5.8.0"
SIGLIP_TORCH = "2.10.0+cu128"
SIGLIP_CUDA = "12.8"
AUDITED_PREPROCESSOR = (
    Path(__file__).resolve().parent / "contracts" / "siglip_so400m_patch14_384_preprocessor.json"
)
OUTPUT_SHAPES = {
    "pooler_output": (1152,),
    "last_hidden_state": (729, 1152),
}
POOLER_BYTES = 1152 * 4
TOKEN_BYTES = 729 * 1152 * 4
IDENTITY_FIELDS = {
    "encoder_id": SIGLIP_ENCODER_ID,
    "revision": SIGLIP_REVISION,
    "model_class": SIGLIP_MODEL_CLASS,
    "weight_sha256": SIGLIP_WEIGHT_SHA256,
    "preprocess_sha256": SIGLIP_PREPROCESS_SHA256,
    "transformers_version": SIGLIP_TRANSFORMERS,
    "torch_version": SIGLIP_TORCH,
    "cuda_version": SIGLIP_CUDA,
    "scene_split_version": SCENE_SPLIT_VERSION,
    "feature_dtype": "float32",
    "allow_processor_defaults": False,
}
REQUIRED_ROW_FIELDS = (
    "episode_uid",
    "step_index",
    "observation_id",
    "image_path",
    "image_sha256",
    "source_split",
    "source_run",
    "scene",
    "route_group_id",
    "scene_partition",
    "revision",
    "weight_sha256",
    "preprocess_sha256",
    "output_name",
    "feature_dtype",
    "feature_shape",
    "feature_sha256",
    "feature",
)


class FeatureContractError(RuntimeError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


def feature_checksum(values: list[float]) -> str:
    raw = np.asarray(values, dtype=np.float32).reshape(-1).tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _pilot_component(*values: object) -> str | None:
    for value in values:
        for part in Path(str(value)).parts:
            if part.startswith("pilot_"):
                return part
    return None


def _flat_feature(feature: Any) -> list[float]:
    if isinstance(feature, list) and feature and isinstance(feature[0], (list, tuple)):
        raise FeatureContractError("feature_dim_mismatch", "feature must be a flat vector")
    try:
        return [float(item) for item in feature]
    except (TypeError, ValueError) as exc:
        raise FeatureContractError("non_finite_feature", "feature is not a numeric vector") from exc


def check_encoder_manifest(manifest: dict[str, Any], *, verify_weight_hash: bool = False) -> None:
    if manifest.get("feature_cache_version") != FEATURE_CACHE_VERSION:
        raise FeatureContractError("cache_config_mismatch", "feature_cache_version is not wm-feature-cache/0.2")
    if "feature_layer_selected" not in manifest:
        raise FeatureContractError("missing_field", "feature_layer_selected")
    if manifest["feature_layer_selected"] is not None:
        raise FeatureContractError("feature_layer_selected", "the output layer stays unselected")
    for field, expected in IDENTITY_FIELDS.items():
        if manifest.get(field) != expected:
            raise FeatureContractError("cache_config_mismatch", f"{field} does not match the audited SigLIP contract")
    preprocess = Path(str(manifest.get("preprocess_path") or ""))
    if not preprocess.is_file():
        raise FeatureContractError("missing_preprocessor", str(preprocess))
    if sha256_file(preprocess) != SIGLIP_PREPROCESS_SHA256:
        raise FeatureContractError("cache_config_mismatch", "preprocess file sha256 does not match the audited JSON")
    weight = Path(str(manifest.get("weight_path") or ""))
    if not weight.is_file():
        raise FeatureContractError("missing_weights", str(weight))
    if verify_weight_hash and sha256_file(weight) != SIGLIP_WEIGHT_SHA256:
        raise FeatureContractError("cache_config_mismatch", "weight file sha256 does not match the audited checkpoint")


def _diagnostic_component(*values: object) -> str | None:
    for value in values:
        for part in Path(str(value)).parts:
            if part == "world_model_diagnostics" or part.startswith("siglip_repro_"):
                return part
    return None


def _reject_row_source(row: dict[str, Any]) -> None:
    if row.get("diagnostic_only") is True:
        raise FeatureContractError("diagnostic_only", "a diagnostic encoding is not a feature cache")
    leaked = sorted(set(FORBIDDEN_INPUT_KEYS).intersection(row))
    if leaked:
        raise FeatureContractError("privileged_field", " ".join(leaked))
    pilot = _pilot_component(row.get("image_path"), row.get("source_run"))
    if pilot:
        raise FeatureContractError("pilot_source", pilot)
    diagnostic = _diagnostic_component(row.get("image_path"), row.get("source_run"))
    if diagnostic:
        raise FeatureContractError("diagnostic_only", diagnostic)
    if row.get("source_split") != "train":
        raise FeatureContractError("non_train_split", str(row.get("source_split")))


def check_feature_row(
    row: dict[str, Any],
    manifest: dict[str, Any],
    *,
    image_root: Path,
    episodes_by_uid: dict[str, dict[str, Any]],
) -> None:
    check_encoder_manifest(manifest)
    for field in REQUIRED_ROW_FIELDS:
        if field not in row or row[field] in ("", None):
            raise FeatureContractError("missing_field", field)
    _reject_row_source(row)
    if row["revision"] != manifest["revision"] or row["weight_sha256"] != manifest["weight_sha256"]:
        raise FeatureContractError("cache_config_mismatch", "row revision or weight_sha256 does not match the manifest")
    if row["preprocess_sha256"] != manifest["preprocess_sha256"]:
        raise FeatureContractError("cache_config_mismatch", "row preprocess_sha256 does not match the manifest")
    if row["feature_dtype"] != "float32":
        raise FeatureContractError("cache_config_mismatch", "feature_dtype is not float32")
    episode = episodes_by_uid.get(str(row["episode_uid"]))
    if episode is None:
        raise FeatureContractError("formal_split", f"episode {row['episode_uid']} is not in the eligible split")
    if episode["scene"] != row["scene"] or episode["route_group_id"] != row["route_group_id"]:
        raise FeatureContractError("formal_split", "episode scene or route group does not match the split")
    if row["scene_partition"] != scene_partition(str(row["scene"])):
        raise FeatureContractError("scene_cross", str(row["scene"]))
    image = image_root / str(row["image_path"])
    if not image.is_file():
        raise FeatureContractError("missing_image", str(image))
    if sha256_file(image) != row["image_sha256"]:
        raise FeatureContractError("image_hash_mismatch", str(image))
    output_name = str(row["output_name"])
    if output_name not in OUTPUT_SHAPES:
        raise FeatureContractError("feature_dim_mismatch", output_name)
    expected_shape = OUTPUT_SHAPES[output_name]
    actual_shape = tuple(int(dim) for dim in row["feature_shape"])
    if actual_shape != expected_shape:
        raise FeatureContractError("feature_dim_mismatch", f"{output_name} shape {actual_shape} != {expected_shape}")
    values = _flat_feature(row["feature"])
    if len(values) != int(np.prod(expected_shape)):
        raise FeatureContractError(
            "feature_dim_mismatch",
            f"vector length {len(values)} != {int(np.prod(expected_shape))}",
        )
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise FeatureContractError("non_finite_feature", str(row["image_path"]))
    if row["feature_sha256"] != feature_checksum(values):
        raise FeatureContractError("cache_config_mismatch", "feature_sha256 does not match the float32 bytes")


def check_feature_cache(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    image_root: Path,
    split_episodes: list[dict[str, Any]],
    verify_weight_hash: bool = False,
) -> None:
    check_encoder_manifest(manifest, verify_weight_hash=verify_weight_hash)
    try:
        report = require_formal_split(split_episodes)
    except FormalSplitError as exc:
        raise FeatureContractError("formal_split", exc.report["reason"]) from exc
    episodes_by_uid = {str(episode["episode_uid"]): episode for episode in report["episodes"]}
    seen: set[tuple[str, str]] = set()
    route_scenes: dict[str, str] = {}
    for row in rows:
        _reject_row_source(row)
        route = str(row.get("route_group_id"))
        scene = str(row.get("scene"))
        if route in route_scenes and route_scenes[route] != scene:
            raise FeatureContractError("scene_cross", f"route group {route} spans scenes")
        route_scenes[route] = scene
        identity = (str(row.get("observation_id")), str(row.get("output_name")))
        if identity in seen:
            raise FeatureContractError("duplicate_feature", str(identity))
        seen.add(identity)
        check_feature_row(row, manifest, image_root=image_root, episodes_by_uid=episodes_by_uid)
