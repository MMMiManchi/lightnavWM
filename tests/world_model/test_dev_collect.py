"""The dev planner accepts only train_dev rows and does not start collection."""

from __future__ import annotations

import pytest

import experiments.world_model.dev_collect as dev_collect
import experiments.world_model.precollection_protocol as protocol
from experiments.world_model.collection_gate import (
    CollectionGateError,
    assert_collectable,
    assert_dev_collectable,
)
from experiments.world_model.dev_collect import main, prepare_dev_plan
from experiments.world_model.latent_data import scene_partition


def _sides() -> tuple[list[str], list[str]]:
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
    return fit, dev


def _inventory() -> tuple[dict, list[dict], list[dict]]:
    fit, dev = _sides()
    scenes = []
    fit_episodes = []
    dev_episodes = []
    for scene, count in zip(fit, [13, 1, 1, 1, 1, 1, 1, 1]):
        routes = [f"{scene}:fit{i}" for i in range(count)]
        scenes.append(
            {
                "scene": scene,
                "scene_partition": "train_fit",
                "n_route_groups": len(routes),
                "n_episodes": len(routes),
                "route_group_ids": routes,
            }
        )
        fit_episodes.append(
            {"scene": scene, "route_group_id": routes[0], "source_split": "train", "scene_partition": "train_fit"}
        )
    for scene, count in zip(dev, [3, 1, 1]):
        routes = [f"{scene}:dev{i}" for i in range(count)]
        scenes.append(
            {
                "scene": scene,
                "scene_partition": "train_dev",
                "n_route_groups": len(routes),
                "n_episodes": len(routes),
                "route_group_ids": routes,
            }
        )
        dev_episodes.append(
            {"scene": scene, "route_group_id": routes[0], "source_split": "train", "scene_partition": "train_dev"}
        )
    inventory = {
        "scene_split_version": "wm-scene-split/0.1",
        "source_split": "train",
        "dataset_sha256": "digest",
        "eligible": True,
        "scenes": scenes,
    }
    return inventory, fit_episodes, dev_episodes


def test_dev_plan_is_isolated_from_fit_and_does_not_run():
    inventory, fit_episodes, dev_episodes = _inventory()
    fit_plan = assert_collectable(inventory, fit_episodes, dataset_sha256="digest")
    dev_plan = prepare_dev_plan(inventory, dev_episodes, dataset_sha256="digest")
    assert {row["scene_partition"] for row in fit_plan} == {"train_fit"}
    assert {row["scene_partition"] for row in dev_plan} == {"train_dev"}
    assert {row["scene"] for row in fit_plan}.isdisjoint({row["scene"] for row in dev_plan})
    with pytest.raises(CollectionGateError, match="train_dev"):
        assert_collectable(inventory, dev_episodes, dataset_sha256="digest")
    with pytest.raises(CollectionGateError, match="train_fit"):
        assert_dev_collectable(inventory, fit_episodes, dataset_sha256="digest")
    with pytest.raises(SystemExit):
        main([])


def test_dev_main_delegates_to_shared_collector_with_dev_partition(monkeypatch):
    captured = {}

    def collect_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr("experiments.world_model.collect.main", collect_main)
    code = main([
        "--config", "config.yaml", "--allowlist", "allow.json",
        "--scene-inventory", "inventory.json", "--run-dir", "run",
    ])
    assert code == 0
    assert captured["argv"][captured["argv"].index("--partition") + 1] == "train_dev"
