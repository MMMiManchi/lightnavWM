"""Train the v1 predictor from approved, feature-paired train rows.

This module does not collect data or choose a feature layer. Callers must pass
an approved bundle returned by ``load_accepted_run_set`` and an explicit cache
output name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.world_model.feature_cache import (
    OUTPUT_SHAPES,
    FeatureContractError,
    _reject_row_source,
    check_feature_cache,
)
from experiments.world_model.feature_pairs import align_transition_latents
from experiments.world_model.latent_data import (
    FormalSplitError,
    admitted_training_rows,
    admitted_validation_rows,
)
from experiments.world_model.predictor import ResidualLatentPredictor, separate_losses


def materialize_feature_samples(
    bundle: dict[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
    manifest: dict[str, Any],
    *,
    image_root: Path,
    split_episodes: list[dict[str, Any]],
    output_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate cache and materialize scene-disjoint fit/dev tensors as arrays."""
    if not bundle.get("admitted_frames"):
        raise FormalSplitError({"eligible": False, "reason": "bundle did not come from approved run-set admission"})
    if output_name not in OUTPUT_SHAPES:
        raise FeatureContractError("feature_dim_mismatch", output_name)
    check_feature_cache(
        [dict(row) for row in feature_rows],
        manifest,
        image_root=image_root,
        split_episodes=split_episodes,
    )
    rows_by_observation: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in feature_rows:
        _reject_row_source(dict(row))
        rows_by_observation.setdefault(str(row["observation_id"]), {})[str(row["output_name"])] = row

    observations = {str(row["observation_id"]): row for row in bundle["observations"]}
    episodes = {str(row["episode_uid"]): row for row in bundle["episodes"]}
    transition_rows = {str(row["transition_id"]): row for row in bundle["transitions"]}
    supervised_by_id = {
        str(row["transition_id"]): row
        for row in admitted_training_rows(bundle) + admitted_validation_rows(bundle)
    }
    fit: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for transition_id, supervised in supervised_by_id.items():
        transition = transition_rows[transition_id]
        episode = episodes[str(transition["episode_uid"])]
        before_id = str(transition["obs_before_id"])
        after_id = str(transition["obs_after_id"])
        before = rows_by_observation.get(before_id, {}).get(output_name)
        after = rows_by_observation.get(after_id, {}).get(output_name)
        if before is None or after is None:
            raise FeatureContractError("missing_feature", f"{transition_id} {output_name}")
        aligned = align_transition_latents(
            transition,
            [before, after],
            list(observations.values()),
            episode,
            output_name=output_name,
        )
        if aligned["z_t_sha256"] != before["feature_sha256"] or aligned["z_t1_sha256"] != after["feature_sha256"]:
            raise FeatureContractError("cache_config_mismatch", transition_id)
        z_t = np.asarray(before["feature"], dtype=np.float32).reshape(OUTPUT_SHAPES[output_name])
        z_t1 = np.asarray(after["feature"], dtype=np.float32).reshape(OUTPUT_SHAPES[output_name])
        if output_name == "last_hidden_state":
            raise FeatureContractError(
                "unsupported_feature_shape",
                "v1 predictor currently consumes a vector; use pooler_output or add a spatial encoder head",
            )
        twist = supervised["model_input"]["executed_twist"]
        pose = supervised["offline_target"]["delta_pose"]
        sample = {
            "transition_id": transition_id,
            "scene": supervised["scene"],
            "route_group_id": supervised["route_group_id"],
            "z_t": z_t,
            "control": np.asarray(
                [twist["v_mps"], twist["w_rad_s"], twist["dt_command_s"]], dtype=np.float32
            ),
            "z_t1": z_t1,
            "pose": np.asarray([pose["forward_m"], pose["left_m"], pose["delta_yaw_rad"]], dtype=np.float32),
        }
        (fit if supervised["scene_partition"] == "train_fit" else dev).append(sample)
    if not fit or not dev:
        raise FormalSplitError({"eligible": False, "reason": "admitted fit or dev feature samples are empty"})
    return fit, dev


def train_predictor(
    bundle: dict[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
    manifest: dict[str, Any],
    *,
    image_root: Path,
    split_episodes: list[dict[str, Any]],
    output_name: str,
    output_path: Path,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fit on train_fit and select on train_dev; refuse overwrite and unapproved input."""
    from experiments.world_model.precollection_protocol import assert_training_authorized

    assert_training_authorized()
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    fit, dev = materialize_feature_samples(
        bundle,
        feature_rows,
        manifest,
        image_root=image_root,
        split_episodes=split_episodes,
        output_name=output_name,
    )
    model = ResidualLatentPredictor(latent_dim=1152).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def batch(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.as_tensor(np.stack([row[key] for row in rows]), dtype=torch.float32, device=device)
            for key in ("z_t", "control", "z_t1", "pose")
        )

    fit_batch = batch(fit)
    dev_batch = batch(dev)
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z_hat, pose_hat = model(fit_batch[0], fit_batch[1])
        losses = separate_losses(z_hat, fit_batch[2], pose_hat, fit_batch[3])
        objective = losses["latent_mse"] + losses["translation_huber"] + losses["yaw_cosine"]
        objective.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            z_dev, pose_dev = model(dev_batch[0], dev_batch[1])
            dev_losses = separate_losses(z_dev, dev_batch[2], pose_dev, dev_batch[3])
        history.append(
            {
                "epoch": epoch + 1,
                "train": {name: float(value.detach().cpu()) for name, value in losses.items()},
                "dev": {name: float(value.detach().cpu()) for name, value in dev_losses.items()},
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "latent_dim": 1152,
            "feature_output_name": output_name,
            "feature_cache_version": manifest.get("feature_cache_version"),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "train_samples": len(fit),
            "dev_samples": len(dev),
            "history": history,
        },
        destination,
    )
    return {"checkpoint": str(destination), "train_samples": len(fit), "dev_samples": len(dev), "history": history}
