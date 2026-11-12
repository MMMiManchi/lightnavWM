"""Startup gate: a collection plan may contain only train_fit scenes from the inventory.

The inventory file is the split authority. This module does not re-hash the
official JSON to decide which scene is train_fit. It checks the file, then
calls ``require_formal_split`` on the route groups stored in that file.
``train_dev`` scenes are rejected instead of being dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.world_model.jsonutil import loads
from experiments.world_model.latent_data import (
    SCENE_SPLIT_VERSION,
    FormalSplitError,
    require_formal_split,
    scene_partition,
)


class CollectionGateError(RuntimeError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


def _route_episodes(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = []
    seen_scenes: dict[str, str] = {}
    seen_routes: dict[str, str] = {}
    scenes = inventory.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise CollectionGateError("scene_inventory", "scenes are missing")
    for row in scenes:
        scene = str(row.get("scene") or "")
        side = row.get("scene_partition")
        routes = row.get("route_group_ids")
        if not scene or side not in ("train_fit", "train_dev") or not isinstance(routes, list) or not routes:
            raise CollectionGateError("scene_inventory", f"scene row is incomplete: {scene or row!r}")
        if scene_partition(scene) != side:
            raise CollectionGateError("scene_cross", f"{scene} is labeled {side}")
        if scene in seen_scenes:
            raise CollectionGateError("scene_cross", f"duplicate scene {scene}")
        seen_scenes[scene] = side
        if int(row.get("n_route_groups", -1)) != len(routes):
            raise CollectionGateError("scene_inventory", f"{scene} route count does not match route_group_ids")
        if int(row.get("n_episodes", 0)) < len(routes):
            raise CollectionGateError("scene_inventory", f"{scene} has fewer episodes than route groups")
        for route in routes:
            route = str(route)
            previous = seen_routes.get(route)
            if previous is not None and previous != scene:
                raise CollectionGateError("scene_cross", f"route group {route} spans scenes")
            seen_routes[route] = scene
            episodes.append(
                {
                    "episode_uid": route,
                    "scene": scene,
                    "episode_id": route,
                    "route_group_id": route,
                    "source_split": "train",
                }
            )
    return episodes


def validate_scene_inventory(inventory: dict[str, Any], *, dataset_sha256: str) -> dict[str, Any]:
    if inventory.get("scene_split_version") != SCENE_SPLIT_VERSION:
        raise CollectionGateError(
            "scene_inventory",
            f"scene_split_version {inventory.get('scene_split_version')!r} is not {SCENE_SPLIT_VERSION}",
        )
    if inventory.get("source_split") != "train":
        raise CollectionGateError("non_train_split", str(inventory.get("source_split")))
    if inventory.get("dataset_sha256") != dataset_sha256:
        raise CollectionGateError("dataset_hash", "inventory dataset_sha256 does not match the train json.gz")
    if inventory.get("eligible") is not True:
        raise CollectionGateError("formal_split", "inventory eligible flag is not true")
    try:
        report = require_formal_split(_route_episodes(inventory))
    except FormalSplitError as exc:
        raise CollectionGateError("formal_split", exc.report["reason"]) from exc
    counts = inventory.get("counts") or {}
    for key in ("train_fit_scenes", "train_fit_route_groups", "train_dev_scenes", "train_dev_route_groups"):
        if key in counts and int(counts[key]) != int(report[key]):
            raise CollectionGateError("scene_inventory", f"stored {key} does not match the route groups")
    return report


def assert_collectable(inventory: dict[str, Any], episodes: list[dict[str, Any]], *, dataset_sha256: str) -> list[dict[str, Any]]:
    """Return ``episodes`` only when every one is a train_fit route in the inventory."""
    validate_scene_inventory(inventory, dataset_sha256=dataset_sha256)
    fit_routes = {
        str(row["scene"]): {str(route) for route in row["route_group_ids"]}
        for row in inventory["scenes"]
        if row["scene_partition"] == "train_fit"
    }
    dev_scenes = {str(row["scene"]) for row in inventory["scenes"] if row["scene_partition"] == "train_dev"}
    if not episodes:
        raise CollectionGateError("allowlist", "collection plan is empty")
    accepted = []
    for episode in episodes:
        split = episode.get("source_split")
        if split != "train":
            raise CollectionGateError("non_train_split", str(split))
        scene = str(episode.get("scene"))
        route = str(episode.get("route_group_id"))
        if scene in dev_scenes or episode.get("partition") == "train_dev" or episode.get("scene_partition") == "train_dev":
            raise CollectionGateError("train_dev", f"{scene} is not collected")
        if scene not in fit_routes or route not in fit_routes[scene]:
            raise CollectionGateError("allowlist", f"{(scene, route)} is not a train_fit route in the inventory")
        accepted.append(episode)
    return accepted


def assert_dev_collectable(
    inventory: dict[str, Any],
    episodes: list[dict[str, Any]],
    *,
    dataset_sha256: str,
) -> list[dict[str, Any]]:
    """Return episodes from train_dev houses. The old route-group hash is not a veto."""
    validate_scene_inventory(inventory, dataset_sha256=dataset_sha256)
    dev_routes = {
        str(row["scene"]): {str(route) for route in row["route_group_ids"]}
        for row in inventory["scenes"]
        if row["scene_partition"] == "train_dev"
    }
    fit_scenes = {str(row["scene"]) for row in inventory["scenes"] if row["scene_partition"] == "train_fit"}
    if not episodes:
        raise CollectionGateError("allowlist", "dev collection plan is empty")
    accepted = []
    for episode in episodes:
        split = episode.get("source_split")
        if split != "train":
            raise CollectionGateError("non_train_split", str(split))
        scene = str(episode.get("scene"))
        route = str(episode.get("route_group_id"))
        if scene in fit_scenes or episode.get("scene_partition") == "train_fit":
            raise CollectionGateError("train_fit", f"{scene} is not collected by the dev entry")
        if scene not in dev_routes or route not in dev_routes[scene]:
            raise CollectionGateError("allowlist", f"{(scene, route)} is not a train_dev route in the inventory")
        accepted.append(episode)
    return accepted


def load_scene_inventory(path: Path) -> dict[str, Any]:
    inventory_path = Path(path)
    if not inventory_path.is_file():
        raise CollectionGateError("scene_inventory", f"missing inventory {inventory_path}")
    return loads(inventory_path.read_text(encoding="utf-8"))
