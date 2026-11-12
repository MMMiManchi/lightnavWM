"""Collection failures stay on disk and are not training data."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.world_model.collect import (
    _require_train_config,
    abort_collection,
    commit_validation,
    parsed_habitat_split,
    require_habitat_server,
    run_session,
    write_failure_record,
)
from experiments.world_model.collect import main as collect_main
from experiments.world_model.collection_gate import CollectionGateError
from experiments.world_model.latent_data import FormalSplitError, load_accepted_run
from experiments.world_model.recorder import RunRecorder
from experiments.world_model.server_adapter import WorldModelVLNCEEnv
from experiments.world_model.validate import validate_run
from tests.world_model.test_collection_contract import (
    ScriptedPolicy,
    _attach,
    _namespace_episode,
    _plan_row,
)
from tests.world_model.test_training_admission import _freeze_tree


def test_validation_failure_is_recorded_and_refused(tmp_path: Path):
    recorder = RunRecorder(tmp_path / "checked")
    keep = recorder.run_dir / "keep.txt"
    keep.write_text("keep\n", encoding="utf-8")
    recorder.close()
    report = validate_run(recorder.run_dir)
    assert report["hard_error_count"] > 0
    assert commit_validation(recorder, report) == 1
    assert keep.read_text(encoding="utf-8") == "keep\n"
    failure = json.loads((recorder.run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "validation_failed"
    assert failure["hard_error_count"] == report["hard_error_count"]
    assert failure["errors"]
    assert (recorder.run_dir / "validation_report.json").is_file()
    with pytest.raises(FormalSplitError):
        load_accepted_run(recorder.run_dir)


def test_reset_step_and_image_errors_record_the_exception(tmp_path: Path):
    class Reordered:
        def reset(self, seed=None):
            return (
                {"instruction": {"text": "go left"}, "rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
                {
                    "wm_resolved": {"split": "train", "served_keys": [["sceneA", "1"]], "seed": {}},
                    "wm_reset": {"key": ["sceneB", "9"], "applied_in_reset": False},
                },
            )

        def step(self, _action):
            raise AssertionError("reset mismatch must not step")

    plan = [_plan_row("sceneA", "1", "go left", "sceneA:4")]
    reset_recorder = RunRecorder(tmp_path / "reset")
    with pytest.raises(RuntimeError) as reset_error:
        run_session(
            Reordered(),
            object(),
            reset_recorder,
            plan,
            max_steps=2,
            requested_seed=None,
            collection_policy="scripted_test",
        )
    assert abort_collection(reset_recorder, reset_error.value) == 1
    assert "reset" in (reset_recorder.run_dir / "failure.json").read_text(encoding="utf-8")
    assert (reset_recorder.run_dir / "transitions.jsonl").read_text(encoding="utf-8") == ""

    step_env, step_fake = _attach([_namespace_episode("sceneA", "1", "go left")])
    step_fake.fail = True
    step_recorder = RunRecorder(tmp_path / "step")
    run_session(
        step_env,
        ScriptedPolicy([{"kind": "move"}]),
        step_recorder,
        plan,
        max_steps=2,
        requested_seed=None,
        collection_policy="scripted_test",
    )
    step_recorder.close()
    row = json.loads((step_recorder.run_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["quality"]["status"] == "quarantined"
    assert "sensor boom" in row["action"]["execution_ack"]["error"]
    assert row["obs_after_id"] is None
    step_recorder.write_json(
        "manifest.json",
        {
            "schema_version": "wm-transition/0.2",
            "source": {"split": "train"},
            "seed": {"status": "partial"},
            "diagnostic_only": False,
        },
    )
    step_recorder.write_json("validation_report.json", {"hard_error_count": 0})
    _freeze_tree(step_recorder.run_dir)
    with pytest.raises(FormalSplitError, match="hard error|quarantine|approval"):
        load_accepted_run(step_recorder.run_dir)

    image_env, _image_fake = _attach([_namespace_episode("sceneA", "1", "go left")])
    image_recorder = RunRecorder(tmp_path / "image")
    target = image_recorder.run_dir / "assets" / f"obs-{plan[0]['episode_uid']}-0_rgb.npy"
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError) as image_error:
        run_session(
            image_env,
            ScriptedPolicy([{"kind": "move"}]),
            image_recorder,
            plan,
            max_steps=2,
            requested_seed=None,
            collection_policy="scripted_test",
        )
    assert abort_collection(image_recorder, image_error.value) == 1
    assert target.read_bytes() == b"keep"
    assert "FileExistsError" in (image_recorder.run_dir / "failure.json").read_text(encoding="utf-8")


def test_server_exit_existing_directory_and_unwritable_failure_record(tmp_path: Path, capsys, monkeypatch):
    class DeadServer:
        def poll(self):
            return 3

    with pytest.raises(RuntimeError, match="habitat server exited 3"):
        require_habitat_server(DeadServer())
    server_recorder = RunRecorder(tmp_path / "server")
    assert abort_collection(server_recorder, RuntimeError("habitat server exited 3; see habitat_server.log")) == 1
    assert "habitat server exited 3" in (server_recorder.run_dir / "failure.json").read_text(encoding="utf-8")

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        RunRecorder(occupied)
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    assert not (occupied / "failure.json").exists()

    existing = RunRecorder(tmp_path / "existing-failure")
    existing.close()
    (existing.run_dir / "failure.json").write_text('{"status":"original"}\n', encoding="utf-8")
    write_from = existing.run_dir / "keep.txt"
    write_from.write_text("keep\n", encoding="utf-8")
    write_failure_record(existing, {"status": "later", "error": "must not replace"})
    assert json.loads((existing.run_dir / "failure.json").read_text(encoding="utf-8"))["status"] == "original"
    assert write_from.read_text(encoding="utf-8") == "keep\n"

    broken = RunRecorder(tmp_path / "unwritable")
    (broken.run_dir / "keep.txt").write_text("keep\n", encoding="utf-8")

    def boom(_name, _payload):
        raise OSError("disk full")

    monkeypatch.setattr(broken, "write_json", boom)
    assert abort_collection(broken, RuntimeError("reset mismatch")) == 1
    captured = capsys.readouterr()
    assert "reset mismatch" in captured.err
    assert "disk full" in captured.err
    assert (broken.run_dir / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    assert not (broken.run_dir / "failure.json").exists()


def test_config_split_is_structural_and_the_runtime_override_stays_train(tmp_path: Path):
    trap = tmp_path / "trap.yaml"
    trap.write_text("# split: train\nhabitat:\n  dataset:\n    split: val_unseen\n", encoding="utf-8")
    assert parsed_habitat_split(trap) == "val_unseen"
    with pytest.raises(CollectionGateError, match="val_unseen"):
        _require_train_config(trap)
    absent = tmp_path / "not-created"
    with pytest.raises(SystemExit, match="val_unseen"):
        collect_main(
            [
                "--config",
                str(trap),
                "--allowlist",
                str(tmp_path / "missing-allow.json"),
                "--scene-inventory",
                str(tmp_path / "missing-inventory.json"),
                "--run-dir",
                str(absent),
                "--server",
                "tcp://127.0.0.1:9",
            ]
        )
    assert not absent.exists()

    real = parsed_habitat_split(Path("experiments/world_model/configs/collect_train_go2w.yaml"))
    assert real == "train"

    env = WorldModelVLNCEEnv.__new__(WorldModelVLNCEEnv)
    env.split = "train"
    env.data_path = "data/datasets/R2R_VLNCE_v1-3_preprocessed"
    env.scenes_dir = None
    env.gpu_id = 0
    env.requested_seed = None
    env.seed_report = {"effective": {}}
    env.allow_order = [("VzqfbhrpDEA", "10")]
    env.split_id = None
    env.split_num = None
    config = SimpleNamespace(
        habitat=SimpleNamespace(
            simulator=SimpleNamespace(habitat_sim_v0=SimpleNamespace(gpu_device_id=1)),
            dataset=SimpleNamespace(split="val_unseen", data_path="keep", scenes_dir=None),
            seed=100,
        )
    )
    env._apply_dataset_overrides(config)
    assert config.habitat.dataset.split == "train"
    assert config.habitat.dataset.data_path.endswith("{split}/{split}.json.gz")
    assert env.allow_order == [("VzqfbhrpDEA", "10")]
    env.split = "val_unseen"
    with pytest.raises(RuntimeError, match="val_unseen"):
        env._apply_dataset_overrides(config)
    env.allow_order = []
    with pytest.raises(RuntimeError, match="allow_order"):
        env._apply_episode_split(object())
