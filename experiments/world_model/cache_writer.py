"""Write one feature row only after the contract checks and the real files match.

Nothing is created if a check fails. An existing destination is left unchanged.
This module does not create a formal cache directory and does not train.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from experiments.world_model.feature_cache import (
    AUDITED_PREPROCESSOR,
    SIGLIP_PREPROCESS_SHA256,
    SIGLIP_WEIGHT_SHA256,
    FeatureContractError,
    check_feature_cache,
)
from experiments.world_model.jsonutil import dumps, sha256_file

FORMAL_CACHE_DIR_NAMES = frozenset({"formal_feature_cache", "world_model_diagnostics"})
FEATURE_CACHE_ROOT = Path(__file__).resolve().parents[2] / "output" / "world_model_cache"


def verify_bound_files(weight_path: Path, preprocess_path: Path = AUDITED_PREPROCESSOR) -> dict[str, str]:
    """Hash the files that will actually be read. The recorded sha is not enough."""
    weight = Path(weight_path)
    preprocess = Path(preprocess_path)
    if not preprocess.is_file():
        raise FeatureContractError("missing_preprocessor", str(preprocess))
    if preprocess.resolve() != AUDITED_PREPROCESSOR.resolve():
        raise FeatureContractError(
            "cache_config_mismatch",
            "preprocessor must be experiments/world_model/contracts/siglip_so400m_patch14_384_preprocessor.json",
        )
    preprocess_sha = sha256_file(preprocess)
    if preprocess_sha != SIGLIP_PREPROCESS_SHA256:
        raise FeatureContractError("cache_config_mismatch", "preprocessor bytes do not match the audited sha256")
    if not weight.is_file():
        raise FeatureContractError("missing_weights", str(weight))
    weight_sha = sha256_file(weight)
    if weight_sha != SIGLIP_WEIGHT_SHA256:
        raise FeatureContractError("cache_config_mismatch", "weight bytes do not match the audited sha256")
    return {"weight_sha256": weight_sha, "preprocess_sha256": preprocess_sha}


def _refuse_destination(destination: Path) -> None:
    path = Path(destination)
    if path.exists():
        raise FeatureContractError("cache_exists", f"refusing to overwrite {path}")
    blocked = FORMAL_CACHE_DIR_NAMES.intersection(path.parts)
    if blocked:
        raise FeatureContractError("formal_cache", f"refusing formal cache path {sorted(blocked)}")
    cache_root = FEATURE_CACHE_ROOT.resolve()
    if not path.resolve(strict=False).is_relative_to(cache_root):
        raise FeatureContractError("cache_path", f"feature rows must be under {cache_root}")
    if any(part.startswith("pilot_") for part in path.parts):
        raise FeatureContractError("pilot_source", str(path))


def commit_feature_row(
    destination: Path,
    row: dict[str, Any],
    manifest: dict[str, Any],
    *,
    image_root: Path,
    split_episodes: list[dict[str, Any]],
    weight_path: Path,
) -> None:
    """Validate, hash the real weight and preprocessor, then create ``destination``."""
    from experiments.world_model.precollection_protocol import assert_feature_extraction_not_authorized

    assert_feature_extraction_not_authorized()
    _refuse_destination(destination)
    if row.get("diagnostic_only") is True:
        raise FeatureContractError("diagnostic_only", "a diagnostic encoding is not a feature cache")
    bound = dict(manifest)
    bound["preprocess_path"] = str(AUDITED_PREPROCESSOR)
    bound["weight_path"] = str(weight_path)
    if bound.get("feature_layer_selected") is not None:
        raise FeatureContractError("feature_layer_selected", "the output layer stays unselected")
    check_feature_cache([row], bound, image_root=image_root, split_episodes=split_episodes)
    verify_bound_files(weight_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dumps(
        {
            "diagnostic_only": False,
            "feature_layer_selected": None,
            "preprocess_sha256": SIGLIP_PREPROCESS_SHA256,
            "weight_sha256": SIGLIP_WEIGHT_SHA256,
            "row": row,
        }
    )
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, (payload + "\n").encode("utf-8"))
    except Exception:
        os.close(fd)
        destination.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)
