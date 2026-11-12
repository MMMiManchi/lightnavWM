"""Collector startup must consume the scene inventory. No Habitat process is launched."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.world_model.collect import _prepare_plan, run_session
from experiments.world_model.collect import main as collect_main
from experiments.world_model.collection_gate import (
    CollectionGateError,
    assert_collectable,
    assert_dev_collectable,
    validate_scene_inventory,
)
from experiments.world_model.group_manifest import (
    assign_groups,
    build_allowlist,
    load_split_rows,
)
from experiments.world_model.jsonutil import dumps, loads, sha256_bytes, sha256_file
from experiments.world_model.latent_data import scene_partition
from experiments.world_model.recorder import RunRecorder
from experiments.world_model.serve_collect import prepare_server_plan
from tests.world_model.test_collection_contract import (
    _attach,
    _namespace_episode,
    _plan_row,
)

TRAIN_JSON = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz")
INVENTORY = Path("experiments/world_model/splits/wm_scene_split_0.1.json")
DATASET_ROOT = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed")


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


def _scene_row(scene: str, side: str, routes: list[str]) -> dict:
    return {
        "scene": scene,
        "scene_partition": side,
        "n_route_groups": len(routes),
        "n_episodes": len(routes),
        "route_group_ids": routes,
    }


def _inventory(digest: str, real_route: tuple[str, str] | None = None) -> tuple[dict, list[str], list[str]]:
    fit, dev = _sides()
    scenes = []
    for scene, count in zip(fit, [13, 1, 1, 1, 1, 1, 1, 1]):
        routes = [f"{scene}:pad{i}" for i in range(count)]
        if real_route is not None and scene == real_route[0]:
            routes[0] = real_route[1]
        scenes.append(_scene_row(scene, "train_fit", routes))
    for scene, count in zip(dev, [3, 1, 1]):
        scenes.append(_scene_row(scene, "train_dev", [f"{scene}:pad{i}" for i in range(count)]))
    inventory = {
        "scene_split_version": "wm-scene-split/0.1",
        "source_split": "train",
        "dataset_sha256": digest,
        "eligible": True,
        "scenes": scenes,
    }
    return inventory, fit, dev


def _episode_json(scene: str, episode_id: int, text: str) -> dict:
    return {
        "episode_id": episode_id,
        "trajectory_id": episode_id,
        "scene_id": f"mp3d/{scene}/{scene}.glb",
        "start_position": [float(episode_id), 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "goals": [{"position": [1.0, 0.0, 1.0], "radius": 3.0}],
        "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        "info": {"geodesic_distance": 4.0},
        "instruction": {"instruction_text": text},
    }


def _write_train(root: Path, episodes: list[dict]) -> str:
    path = root / "train" / "train.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": episodes, "instruction_vocab": {"word_list": []}}, handle)
    return sha256_file(path)


def _write_allowlist(path: Path, root: Path, chosen: list[dict]) -> None:
    _dataset, digest, rows = load_split_rows(root, "train")
    grouped = assign_groups(rows)
    payload = build_allowlist(grouped, chosen, dataset_json=_dataset, dataset_sha256=digest)
    path.write_text(dumps(payload) + "\n", encoding="utf-8")


def _row(scene: str, route: str, split: str = "train") -> dict:
    return {
        "scene": scene,
        "episode_id": "1",
        "episode_uid": f"synthetic|{split}|{scene}|1",
        "route_group_id": route,
        "source_split": split,
        "partition": "train_fit",
    }


def test_legal_train_inventory_accepts_only_its_train_fit_routes():
    inventory, fit, dev = _inventory("abc")
    report = validate_scene_inventory(inventory, dataset_sha256="abc")
    assert report["eligible"] is True
    accepted = assert_collectable(inventory, [_row(fit[0], f"{fit[0]}:pad0")], dataset_sha256="abc")
    assert accepted[0]["scene"] == fit[0]
    with pytest.raises(CollectionGateError, match="train_dev"):
        assert_collectable(inventory, [_row(dev[0], f"{dev[0]}:pad0")], dataset_sha256="abc")


def test_val_and_test_sources_are_refused():
    inventory, fit, _dev = _inventory("abc")
    route = f"{fit[0]}:pad0"
    for split in ("val_seen", "val_unseen", "test"):
        with pytest.raises(CollectionGateError, match="non_train_split"):
            assert_collectable(inventory, [_row(fit[0], route, split)], dataset_sha256="abc")


def test_scene_or_route_cross_schema_and_hash_are_refused():
    inventory, fit, _dev = _inventory("abc")
    crossed = loads(dumps(inventory))
    crossed["scenes"][0]["scene_partition"] = "train_dev"
    with pytest.raises(CollectionGateError, match="scene_cross"):
        validate_scene_inventory(crossed, dataset_sha256="abc")

    shared = loads(dumps(inventory))
    stolen = shared["scenes"][0]["route_group_ids"][0]
    shared["scenes"][1]["route_group_ids"].append(stolen)
    shared["scenes"][1]["n_route_groups"] += 1
    shared["scenes"][1]["n_episodes"] += 1
    with pytest.raises(CollectionGateError, match="scene_cross"):
        validate_scene_inventory(shared, dataset_sha256="abc")

    renamed = loads(dumps(inventory))
    renamed["scene_split_version"] = "wm-scene-split/9.9"
    with pytest.raises(CollectionGateError, match="scene_split_version"):
        validate_scene_inventory(renamed, dataset_sha256="abc")

    edited = loads(dumps(inventory))
    edited["dataset_sha256"] = "f" * 64
    with pytest.raises(CollectionGateError, match="dataset_hash"):
        validate_scene_inventory(edited, dataset_sha256="abc")

    short = loads(dumps(inventory))
    short["scenes"] = [row for row in short["scenes"] if row["scene"] != fit[-1]]
    with pytest.raises(CollectionGateError, match="formal_split"):
        validate_scene_inventory(short, dataset_sha256="abc")

    unmarked = loads(dumps(inventory))
    unmarked["eligible"] = False
    with pytest.raises(CollectionGateError, match="eligible"):
        validate_scene_inventory(unmarked, dataset_sha256="abc")


def test_prepare_plan_refuses_before_a_run_directory_exists(tmp_path: Path):
    fit, dev = _sides()
    root = tmp_path / "data"
    digest = _write_train(
        root,
        [
            _episode_json(fit[0], 1, "go to the door"),
            _episode_json(dev[0], 2, "stay in train_dev"),
        ],
    )
    _dataset, _digest, rows = load_split_rows(root, "train")
    grouped = assign_groups(rows)
    by_scene = {row["scene"]: row for row in grouped}
    inventory, _fit, _dev = _inventory(digest, real_route=(fit[0], by_scene[fit[0]]["route_group_id"]))
    dev_routes = next(row for row in inventory["scenes"] if row["scene"] == dev[0])["route_group_ids"]
    dev_routes[0] = by_scene[dev[0]]["route_group_id"]
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(dumps(inventory) + "\n", encoding="utf-8")
    allowlist_path = tmp_path / "allow.json"
    _write_allowlist(allowlist_path, root, [by_scene[fit[0]]])

    plan, _allowlist, plan_digest, _loaded = _prepare_plan(allowlist_path, root, inventory_path)
    assert plan_digest == digest
    assert [(row["scene"], row["episode_id"]) for row in plan] == [(fit[0], "1")]
    assert prepare_server_plan(allowlist_path, root, inventory_path)[1] == [(fit[0], "1")]

    dev_allowlist = tmp_path / "dev.json"
    _write_allowlist(dev_allowlist, root, [by_scene[dev[0]]])
    with pytest.raises(CollectionGateError, match="train_dev"):
        _prepare_plan(dev_allowlist, root, inventory_path)
    with pytest.raises(SystemExit, match="train_dev"):
        prepare_server_plan(dev_allowlist, root, inventory_path)

    contaminated = loads(allowlist_path.read_text(encoding="utf-8"))
    contaminated["episodes"][0]["partition"] = "val_unseen"
    contaminated["allowlist_sha256"] = sha256_bytes(dumps(contaminated["episodes"]).encode())
    bad_allowlist = tmp_path / "val.json"
    bad_allowlist.write_text(dumps(contaminated) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="val_unseen"):
        _prepare_plan(bad_allowlist, root, inventory_path)
    with pytest.raises(SystemExit, match="val_unseen"):
        prepare_server_plan(bad_allowlist, root, inventory_path)

    broken = loads(dumps(inventory))
    broken["scene_split_version"] = "wm-scene-split/0.0"
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(dumps(broken) + "\n", encoding="utf-8")
    run_dir = tmp_path / "must-not-exist"
    config = tmp_path / "train.yaml"
    config.write_text("habitat:\n  dataset:\n    split: train\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="scene_split_version"):
        collect_main(
            [
                "--config",
                str(config),
                "--allowlist",
                str(allowlist_path),
                "--scene-inventory",
                str(broken_path),
                "--run-dir",
                str(run_dir),
                "--dataset-root",
                str(root),
            ]
        )
    assert not run_dir.exists()


def test_reset_outside_allowlist_and_reordered_reset_write_no_transition(tmp_path: Path):
    env, fake = _attach([_namespace_episode("sceneB", "9", "other")])
    env.served_keys = [("sceneA", "1")]
    with pytest.raises(RuntimeError, match="outside the served allowlist"):
        env.reset()
    assert fake.step_calls == 0

    class Reordered:
        def reset(self, seed=None):
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            return (
                {"instruction": {"text": "go left"}, "rgb": image},
                {
                    "wm_resolved": {"split": "train", "served_keys": [["sceneA", "1"], ["sceneB", "2"]], "seed": {}},
                    "wm_reset": {"key": ["sceneB", "2"], "applied_in_reset": False},
                },
            )

        def step(self, _action):
            raise AssertionError("a reordered reset must not step")

    plan = [
        _plan_row("sceneA", "1", "go left", "sceneA:4"),
        _plan_row("sceneB", "2", "go right", "sceneB:4"),
    ]
    recorder = RunRecorder(tmp_path / "reordered")
    try:
        with pytest.raises(RuntimeError, match="reset"):
            run_session(
                Reordered(),
                object(),
                recorder,
                plan,
                max_steps=2,
                requested_seed=7,
                collection_policy="scripted_test",
            )
        assert (recorder.run_dir / "transitions.jsonl.partial").read_text(encoding="utf-8") == ""
        assert (recorder.run_dir / "observations.jsonl.partial").read_text(encoding="utf-8") == ""
        assert list((recorder.run_dir / "assets").iterdir()) == []
    finally:
        recorder.close()
    assert (recorder.run_dir / "transitions.jsonl").read_text(encoding="utf-8") == ""


def test_a_development_house_keeps_a_route_with_the_old_fit_hash():
    if not INVENTORY.is_file():
        pytest.skip("scene inventory is not on disk")
    inventory = loads(INVENTORY.read_text(encoding="utf-8"))
    episode = {
        "source_split": "train",
        "scene": "ZMojNkEp431",
        "episode_id": "136",
        "route_group_id": "ZMojNkEp431:97",
        "partition": "train_fit",
    }
    accepted = assert_dev_collectable(
        inventory,
        [episode],
        dataset_sha256=inventory["dataset_sha256"],
    )
    assert accepted[0]["route_group_id"] == "ZMojNkEp431:97"
    with pytest.raises(CollectionGateError, match="train_dev"):
        assert_collectable(inventory, [episode], dataset_sha256=inventory["dataset_sha256"])


def test_official_inventory_is_the_allow_set(tmp_path: Path):
    if not TRAIN_JSON.is_file() or not INVENTORY.is_file():
        pytest.skip("official train json or scene inventory is not on disk")
    inventory = loads(INVENTORY.read_text(encoding="utf-8"))
    digest = sha256_file(TRAIN_JSON)
    report = validate_scene_inventory(inventory, dataset_sha256=digest)
    assert report["train_fit_scenes"] >= 8
    assert report["train_dev_scenes"] >= 3
    dev_names = {row["scene"] for row in inventory["scenes"] if row["scene_partition"] == "train_dev"}
    fit_names = {row["scene"] for row in inventory["scenes"] if row["scene_partition"] == "train_fit"}
    assert "17DRP5sb8fy" in fit_names
    assert "1pXnuDYAj8r" in dev_names

    _path, loaded_digest, rows = load_split_rows(DATASET_ROOT, "train")
    assert loaded_digest == digest
    grouped = assign_groups(rows)
    fit_row = next(row for row in grouped if row["scene"] == "17DRP5sb8fy")
    dev_row = next(row for row in grouped if row["scene"] == "1pXnuDYAj8r")
    fit_routes = next(row["route_group_ids"] for row in inventory["scenes"] if row["scene"] == fit_row["scene"])
    assert fit_row["route_group_id"] in fit_routes

    allowlist = tmp_path / "fit.json"
    _write_allowlist(allowlist, DATASET_ROOT, [fit_row])
    plan, _payload, plan_digest, _loaded = _prepare_plan(allowlist, DATASET_ROOT, INVENTORY)
    assert plan_digest == digest
    assert [(row["scene"], row["episode_id"]) for row in plan] == [(fit_row["scene"], fit_row["episode_id"])]

    dev_allowlist = tmp_path / "dev.json"
    _write_allowlist(dev_allowlist, DATASET_ROOT, [dev_row])
    with pytest.raises(CollectionGateError, match="train_dev"):
        _prepare_plan(dev_allowlist, DATASET_ROOT, INVENTORY)
    with pytest.raises(SystemExit, match="train_dev"):
        prepare_server_plan(dev_allowlist, DATASET_ROOT, INVENTORY)
