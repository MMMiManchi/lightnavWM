"""Dry-run checks the collection plan and creates no run directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.world_model.collect import (
    CONTROLLED_TRIAL_WALL_CLOCK_BUDGET_S,
    FORMAL_WALL_CLOCK_BUDGET_S,
    collection_dry_run,
    resolve_wall_clock_budget,
)
from experiments.world_model.collect import main as collect_main
from experiments.world_model.group_manifest import assign_groups, load_split_rows
from experiments.world_model.jsonutil import dumps, loads, sha256_bytes
from tests.world_model.test_collection_contract import _attach, _namespace_episode
from tests.world_model.test_collection_gate import (
    _episode_json,
    _inventory,
    _sides,
    _write_allowlist,
    _write_train,
)


def _config(path: Path, split: str = "train") -> None:
    path.write_text(f"habitat:\n  dataset:\n    split: {split}\n", encoding="utf-8")


def test_formal_wall_clock_is_eight_hours_but_trial_remains_twenty_minutes():
    assert FORMAL_WALL_CLOCK_BUDGET_S == 8 * 60 * 60
    assert resolve_wall_clock_budget(False, None) == 8 * 60 * 60
    assert resolve_wall_clock_budget(False, 8 * 60 * 60) == 8 * 60 * 60
    assert CONTROLLED_TRIAL_WALL_CLOCK_BUDGET_S == 20 * 60
    assert resolve_wall_clock_budget(True, None) == 20 * 60
    with pytest.raises(ValueError, match="wall-clock budget"):
        resolve_wall_clock_budget(False, 8 * 60 * 60 + 1)
    with pytest.raises(ValueError, match="wall-clock budget"):
        resolve_wall_clock_budget(True, 20 * 60 + 1)


def _ready(tmp_path: Path, scenes: list[str] | None = None) -> dict[str, Path]:
    fit, dev = _sides()
    root = tmp_path / "data"
    digest = _write_train(
        root,
        [_episode_json(fit[0], 1, "go to the door"), _episode_json(dev[0], 2, "stay out")],
    )
    _dataset, _digest, rows = load_split_rows(root, "train")
    grouped = {row["scene"]: row for row in assign_groups(rows)}
    inventory, _fit, _dev = _inventory(digest, real_route=(fit[0], grouped[fit[0]]["route_group_id"]))
    dev_routes = next(row for row in inventory["scenes"] if row["scene"] == dev[0])["route_group_ids"]
    dev_routes[0] = grouped[dev[0]]["route_group_id"]
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(dumps(inventory) + "\n", encoding="utf-8")
    chosen = [grouped[scene] for scene in (scenes or [fit[0]])]
    allowlist = tmp_path / "allow.json"
    _write_allowlist(allowlist, root, chosen)
    config = tmp_path / "collect.yaml"
    _config(config)
    return {
        "root": root,
        "inventory": inventory_path,
        "allowlist": allowlist,
        "config": config,
        "fit": Path(fit[0]),
        "dev": Path(dev[0]),
    }


def test_legal_dry_run_prints_the_plan_and_creates_nothing(tmp_path: Path):
    ready = _ready(tmp_path)
    run_dir = tmp_path / "not-created"
    before = sorted(path.name for path in tmp_path.iterdir())
    report = collection_dry_run(
        allowlist_path=ready["allowlist"],
        dataset_root=ready["root"],
        inventory_path=ready["inventory"],
        run_dir=run_dir,
        config_path=ready["config"],
        max_steps=200,
    )
    assert report["dry_run"] is True
    assert report["formal_collection_started"] is False
    assert report["formal_feature_cache"] is False
    assert report["feature_layer_selected"] is None
    assert report["accepted_for_formal_training"] is False
    assert report["episodes"] == 1
    assert report["scenes"] == 1
    assert report["scene_ids"] == [ready["fit"].name]
    assert report["keys"] == [[ready["fit"].name, "1"]]
    assert report["max_steps"] == 200
    assert report["step_upper_bound"] == 200
    assert report["output_exists_after_check"] is False
    assert not run_dir.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_dry_run_refuses_non_train_cross_hash_and_existing_output(tmp_path: Path):
    ready = _ready(tmp_path)
    run_dir = tmp_path / "planned"
    dev_allowlist = tmp_path / "dev.json"
    _dataset, _digest, rows = load_split_rows(ready["root"], "train")
    grouped = {row["scene"]: row for row in assign_groups(rows)}
    _write_allowlist(dev_allowlist, ready["root"], [grouped[ready["dev"].name]])
    with pytest.raises(Exception, match="train_dev"):
        collection_dry_run(
            allowlist_path=dev_allowlist,
            dataset_root=ready["root"],
            inventory_path=ready["inventory"],
            run_dir=run_dir,
            config_path=ready["config"],
            max_steps=4,
        )
    assert not run_dir.exists()

    contaminated = loads(ready["allowlist"].read_text(encoding="utf-8"))
    contaminated["episodes"][0]["partition"] = "val_unseen"
    contaminated["allowlist_sha256"] = sha256_bytes(dumps(contaminated["episodes"]).encode())
    val_allowlist = tmp_path / "val.json"
    val_allowlist.write_text(dumps(contaminated) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="val_unseen"):
        collection_dry_run(
            allowlist_path=val_allowlist,
            dataset_root=ready["root"],
            inventory_path=ready["inventory"],
            run_dir=run_dir,
            config_path=ready["config"],
            max_steps=4,
        )

    crossed = loads(ready["inventory"].read_text(encoding="utf-8"))
    crossed["scenes"][0]["scene_partition"] = "train_dev"
    crossed_path = tmp_path / "crossed.json"
    crossed_path.write_text(dumps(crossed) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="scene_cross"):
        collection_dry_run(
            allowlist_path=ready["allowlist"],
            dataset_root=ready["root"],
            inventory_path=crossed_path,
            run_dir=run_dir,
            config_path=ready["config"],
            max_steps=4,
        )

    edited = loads(ready["inventory"].read_text(encoding="utf-8"))
    edited["dataset_sha256"] = "f" * 64
    edited_path = tmp_path / "edited.json"
    edited_path.write_text(dumps(edited) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="dataset_hash"):
        collection_dry_run(
            allowlist_path=ready["allowlist"],
            dataset_root=ready["root"],
            inventory_path=edited_path,
            run_dir=run_dir,
            config_path=ready["config"],
            max_steps=4,
        )
    assert not run_dir.exists()

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep\n", encoding="utf-8")
    with pytest.raises(Exception, match="run_dir_exists"):
        collection_dry_run(
            allowlist_path=ready["allowlist"],
            dataset_root=ready["root"],
            inventory_path=ready["inventory"],
            run_dir=occupied,
            config_path=ready["config"],
            max_steps=4,
        )
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "keep\n"

    val_config = tmp_path / "val.yaml"
    _config(val_config, "val_unseen")
    with pytest.raises(Exception, match="val_unseen"):
        collection_dry_run(
            allowlist_path=ready["allowlist"],
            dataset_root=ready["root"],
            inventory_path=ready["inventory"],
            run_dir=run_dir,
            config_path=val_config,
            max_steps=4,
        )


def test_main_dry_run_does_not_launch_or_write(tmp_path: Path):
    ready = _ready(tmp_path)
    run_dir = tmp_path / "absent-run"
    code = collect_main(
        [
            "--dry-run",
            "--launch-server",
            "--habitat-python",
            "/bin/false",
            "--config",
            str(ready["config"]),
            "--allowlist",
            str(ready["allowlist"]),
            "--scene-inventory",
            str(ready["inventory"]),
            "--run-dir",
            str(run_dir),
            "--dataset-root",
            str(ready["root"]),
            "--max-steps",
            "4",
        ]
    )
    assert code == 0
    assert not run_dir.exists()
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="run_dir_exists"):
        collect_main(
            [
                "--dry-run",
                "--config",
                str(ready["config"]),
                "--allowlist",
                str(ready["allowlist"]),
                "--scene-inventory",
                str(ready["inventory"]),
                "--run-dir",
                str(occupied),
                "--dataset-root",
                str(ready["root"]),
            ]
        )
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    assert not any(path.name.endswith(".jsonl") for path in tmp_path.rglob("*"))


def test_reset_outside_allowlist_fails_before_step():
    env, fake = _attach([_namespace_episode("sceneB", "9", "other")])
    env.served_keys = [("sceneA", "1")]
    with pytest.raises(RuntimeError, match="outside the served allowlist"):
        env.reset()
    assert fake.step_calls == 0
