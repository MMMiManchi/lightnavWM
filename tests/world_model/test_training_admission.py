"""Formal training admission requires an external approval and a clean run."""

from __future__ import annotations

import errno
import io
import json
import os
import stat
import threading
from pathlib import Path

import numpy as np
import pytest

from experiments.world_model.jsonutil import sha256_file
from experiments.world_model.latent_data import (
    FormalSplitError,
    admitted_training_rows,
    admitted_validation_rows,
    approval_mac,
    issue_run_approval,
    load_accepted_run,
    load_accepted_run_set,
    read_admitted_rgb,
    read_recorded_run,
    scene_partition,
    supervised_rows,
)
from experiments.world_model.validate import (
    open_checked_nofollow,
    open_relative_nofollow,
    snapshot_run,
    validate_run,
)
from tests.world_model.test_encoder_and_split_audit import _formal_episodes

PLANNED = Path("output/world_model_data/pilot_bounded_acceptance_20261026T015701Z")
_UNITS = {
    "lin_vel_range_mps": [0.0, 2.5],
    "ang_vel_range_dps": [-300.0, 300.0],
    "min_abs_lin_speed_mps": 0.01,
    "min_abs_ang_speed_dps": 0.5,
    "control_dt_s": 0.1,
}


def _write_run(path: Path, episodes: list[dict], **manifest_updates) -> Path:
    path.mkdir(parents=True)
    manifest = {
        "schema_version": "wm-transition/0.2",
        "source": {
            "name": "R2R_VLNCE_v1-3_preprocessed",
            "split": "train",
            "dataset_sha256": "a" * 64,
            "scene_inventory_sha256": "b" * 64,
            "scene_split_version": "wm-scene-split/0.1",
        },
        "seed": {"status": "partial"},
        "training_use": "formal",
        "diagnostic_only": False,
    }
    manifest.update(manifest_updates)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "validation_report.json").write_text(json.dumps({"hard_error_count": 0}), encoding="utf-8")
    (path / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes),
        encoding="utf-8",
    )
    for name in ("observations", "transitions", "privileged_labels"):
        (path / f"{name}.jsonl").write_text("", encoding="utf-8")
    return path


def _rgb(run: Path, name: str) -> dict:
    array = np.zeros((2, 2, 3), dtype=np.uint8)
    array[0, 0] = (len(name), 2, 3)
    path = run / "assets" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return {
        "path": str(path.relative_to(run)),
        "sha256": sha256_file(path),
        "shape": [2, 2, 3],
        "dtype": "uint8",
    }


def _observation(run: Path, episode: dict, step: int) -> dict:
    name = f"obs-{episode['episode_uid']}-{step}_rgb.npy"
    return {
        "observation_id": f"obs-{episode['episode_uid']}-{step}",
        "episode_uid": episode["episode_uid"],
        "step_index": step,
        "rgb": _rgb(run, name),
        "rgb_missing": False,
        "sim_time_s": step * 0.016,
    }


def _pose(episode: dict, step: int) -> dict:
    return {
        "label_id": f"pose-{episode['episode_uid']}-{step}",
        "observation_id": f"obs-{episode['episode_uid']}-{step}",
        "episode_uid": episode["episode_uid"],
        "pose": {"world_position_xyz_m": [float(step), 0.0, 0.0], "derived_yaw_rad": 0.0},
    }


def _transition(episode: dict, step: int, *, status: str = "valid", errors: list | None = None) -> dict:
    final = {
        "action": "velocity_control",
        "action_args": {"linear_velocity": 0.2, "angular_velocity": 0.0},
    }
    executed = status == "valid"
    return {
        "transition_id": f"tr-{episode['episode_uid']}-{step}",
        "episode_uid": episode["episode_uid"],
        "t": step,
        "obs_before_id": f"obs-{episode['episode_uid']}-{step}",
        "obs_after_id": f"obs-{episode['episode_uid']}-{step + 1}",
        "pose_before_id": f"pose-{episode['episode_uid']}-{step}",
        "pose_after_id": f"pose-{episode['episode_uid']}-{step + 1}",
        "prediction": {
            "inference_called": False,
            "decode_status": "not_called",
            "model_stop": None,
            "raw_text": None,
            "rvq_tokens": None,
            "rvq_codes": None,
            "decoded_waypoints": None,
            "selected_index": None,
            "selected_waypoint": None,
            "waypoint_times": None,
        },
        "action": {
            "action_id": f"act-{episode['episode_uid']}-{step}",
            "client_request": final,
            "server_override": False,
            "final_habitat_action": final,
            "dt_sim_s": 0.016,
            "sim_time_before_s": step * 0.016,
            "sim_time_after_s": (step + 1) * 0.016,
            "execution_ack": {
                "action_id": f"act-{episode['episode_uid']}-{step}",
                "status": "executed" if executed else "error",
                "executed": executed,
                "final_habitat_action": final,
                "control_units": _UNITS,
                "dt_command_s": 0.1,
            },
        },
        "outcome": {"end_sources": [], "primary_end_source": "unknown"},
        "online_input": {
            "observation_ids": [f"obs-{episode['episode_uid']}-{step}"],
            "info_keys": [],
        },
        "quality": {"status": status, "errors": [] if errors is None else errors},
    }


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _prepare(
    run: Path,
    *,
    training_use: str = "not_approved",
    steps: int = 1,
    statuses: list[str] | None = None,
    episodes_override: list[dict] | None = None,
):
    episodes = episodes_override if episodes_override is not None else _formal_episodes()[0]
    episode = dict(episodes[0])
    episode["partition"] = "train_fit"
    episode["completion"] = {"state": "budget_truncated"}
    written = []
    for item in episodes:
        row = dict(item)
        row["partition"] = scene_partition(row["scene"])
        row["completion"] = {"state": "recorded"}
        row["instruction_text"] = "go to the door"
        written.append(row)
    episode = written[0]
    run.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "wm-transition/0.2",
        "source": {
            "name": "R2R_VLNCE_v1-3_preprocessed",
            "split": "train",
            "dataset_sha256": "a" * 64,
            "scene_inventory_sha256": "b" * 64,
            "scene_split_version": "wm-scene-split/0.1",
        },
        "seed": {"status": "partial", "requested": 1, "effective": {"requested_copied_as_effective": False}},
        "allowlist_keys": [[episode["scene"], episode["episode_id"]]],
        "served_keys": [[episode["scene"], episode["episode_id"]]],
        "reset_keys": [[episode["scene"], episode["episode_id"]]],
        "training_use": training_use,
        "diagnostic_only": False,
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    observations = [_observation(run, episode, step) for step in range(steps + 1)]
    poses = [_pose(episode, step) for step in range(steps + 1)]
    labels = statuses or ["valid"] * steps
    transitions = [
        _transition(episode, step, status=label, errors=["kept"] if label == "quarantined" else [])
        for step, label in enumerate(labels)
    ]
    _dump_jsonl(run / "episodes.jsonl", written)
    _dump_jsonl(run / "observations.jsonl", observations)
    _dump_jsonl(run / "transitions.jsonl", transitions)
    _dump_jsonl(run / "privileged_labels.jsonl", poses)
    report = validate_run(run)
    (run / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")
    return episode


def _chmod(path: Path, *, writable: bool) -> None:
    mode = stat.S_IMODE(path.lstat().st_mode)
    path.chmod(mode | 0o200 if writable else mode & ~0o222)


def _chmod_tree(root: Path, *, writable: bool) -> None:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        _chmod(current, writable=writable)
        for name in list(dirnames):
            if (current / name).is_symlink():
                dirnames.remove(name)
        for name in filenames:
            child = current / name
            if child.is_symlink() or not stat.S_ISREG(child.lstat().st_mode):
                continue
            _chmod(child, writable=writable)


def _freeze_tree(root: Path) -> None:
    _chmod_tree(root, writable=False)


def _thaw_tree(root: Path) -> None:
    _chmod_tree(root, writable=True)


def _rewrite_approval(path: Path, text: str) -> None:
    _chmod(path, writable=True)
    path.write_text(text, encoding="utf-8")
    _chmod(path, writable=False)


def _key(tmp_path: Path) -> Path:
    path = tmp_path / "approval.key"
    if path.exists() and not path.is_symlink():
        _chmod(path, writable=True)
    path.write_bytes(b"external-approval-key-32b!!")
    _chmod(path, writable=False)
    return path


def _sign(run: Path, tmp_path: Path, key: Path, *, name: str | None = None) -> Path:
    _freeze_tree(run)
    approval = tmp_path / "approvals" / f"{name or run.name}.json"
    issue_run_approval(run, approval, key)
    return approval


def _rows(run: Path, approval: Path, key: Path) -> list[dict]:
    bundle = load_accepted_run(run, approval_path=approval, approval_key_path=key)
    return supervised_rows(bundle)


def test_handwritten_formal_without_approval_is_refused(tmp_path: Path):
    run = tmp_path / "formal_string_only"
    _prepare(run, training_use="formal")
    _freeze_tree(run)
    with pytest.raises(FormalSplitError, match="approval record is missing"):
        _rows(run, tmp_path / "approvals" / "missing.json", _key(tmp_path))


def test_approval_for_another_run_is_refused(tmp_path: Path):
    key = _key(tmp_path)
    first = tmp_path / "run_a"
    second = tmp_path / "run_b"
    _prepare(first)
    _prepare(second)
    approval = _sign(second, tmp_path, key)
    _freeze_tree(first)
    with pytest.raises(FormalSplitError, match="run_name"):
        _rows(first, approval, key)


def test_approval_digest_or_identity_mismatch_is_refused(tmp_path: Path):
    key = _key(tmp_path)
    run = tmp_path / "digest_run"
    _prepare(run)
    approval = _sign(run, tmp_path, key)
    document = json.loads(approval.read_text(encoding="utf-8"))
    document["files"]["manifest.json"] = "0" * 64
    payload = {name: document[name] for name in ("schema_version", "run_name", "files")}
    document["mac"] = approval_mac(key.read_bytes(), payload)
    _rewrite_approval(approval, json.dumps(document))
    with pytest.raises(FormalSplitError, match="digest does not match"):
        _rows(run, approval, key)

    document["run_name"] = "some_other_run"
    document["mac"] = approval_mac(key.read_bytes(), {name: document[name] for name in ("schema_version", "run_name", "files")})
    _rewrite_approval(approval, json.dumps(document))
    with pytest.raises(FormalSplitError, match="run_name"):
        _rows(run, approval, key)


def test_missing_damaged_or_modified_approval_is_refused(tmp_path: Path):
    key = _key(tmp_path)
    run = tmp_path / "approval_file_run"
    _prepare(run)
    approval = _sign(run, tmp_path, key)
    missing = tmp_path / "approvals" / "absent.json"
    with pytest.raises(FormalSplitError, match="approval record is missing"):
        _rows(run, missing, key)
    _rewrite_approval(approval, "{")
    with pytest.raises(FormalSplitError, match="damaged"):
        _rows(run, approval, key)
    approval.unlink()
    _sign(run, tmp_path, key)
    document = json.loads(approval.read_text(encoding="utf-8"))
    document["mac"] = "0" * 64
    _rewrite_approval(approval, json.dumps(document))
    with pytest.raises(FormalSplitError, match="modified"):
        _rows(run, approval, key)


def test_signed_clean_run_loads_every_row(tmp_path: Path):
    key = _key(tmp_path)
    run = tmp_path / "clean_run"
    _prepare(run, training_use="not_approved")
    approval = _sign(run, tmp_path, key)
    rows = _rows(run, approval, key)
    assert len(rows) == 1
    assert rows[0]["quality"] == "valid"
    bundle = load_accepted_run(run, approval_path=approval, approval_key_path=key)
    sealed_rgb = bundle["observations"][0]["rgb"]
    assert sealed_rgb["path"] == sealed_rgb["relative_path"]
    pixels = read_admitted_rgb(rows[0]["model_input"]["rgb_t"]).tobytes()
    image = next(iter((run / "assets").glob("*.npy")))
    _thaw_tree(run)
    image.write_bytes(b"replaced-after-admission")
    assert read_admitted_rgb(rows[0]["model_input"]["rgb_t"]).tobytes() == pixels
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_use"] == "not_approved"


def test_approved_fit_and_dev_runs_are_admitted_as_separate_row_sets(tmp_path: Path):
    key = _key(tmp_path)
    episodes, _ = _formal_episodes()
    fit_episodes = [row for row in episodes if scene_partition(row["scene"]) == "train_fit"]
    dev_episodes = [row for row in episodes if scene_partition(row["scene"]) == "train_dev"]
    fit_run = tmp_path / "fit_run"
    dev_run = tmp_path / "dev_run"
    _prepare(fit_run, episodes_override=fit_episodes)
    _prepare(dev_run, episodes_override=dev_episodes)
    fit_approval = _sign(fit_run, tmp_path, key, name="fit")
    dev_approval = _sign(dev_run, tmp_path, key, name="dev")

    bundle = load_accepted_run_set(
        [(fit_run, fit_approval, key), (dev_run, dev_approval, key)]
    )
    assert bundle["formal_split"]["eligible"] is True
    fit_rows = admitted_training_rows(bundle)
    dev_rows = admitted_validation_rows(bundle)
    assert fit_rows and all(row["scene_partition"] == "train_fit" for row in fit_rows)
    assert dev_rows and all(row["scene_partition"] == "train_dev" for row in dev_rows)


def test_approved_fit_only_run_set_cannot_claim_formal_training_readiness(tmp_path: Path):
    key = _key(tmp_path)
    episodes, _ = _formal_episodes()
    fit_episodes = [row for row in episodes if scene_partition(row["scene"]) == "train_fit"]
    fit_run = tmp_path / "fit_only_run"
    _prepare(fit_run, episodes_override=fit_episodes)
    approval = _sign(fit_run, tmp_path, key)
    with pytest.raises(FormalSplitError, match="train_dev has 0 scenes"):
        load_accepted_run_set([(fit_run, approval, key)])


def test_approved_runs_with_different_dataset_snapshots_cannot_be_combined(tmp_path: Path):
    key = _key(tmp_path)
    episodes, _ = _formal_episodes()
    fit_episodes = [row for row in episodes if scene_partition(row["scene"]) == "train_fit"]
    dev_episodes = [row for row in episodes if scene_partition(row["scene"]) == "train_dev"]
    fit_run = tmp_path / "fit_dataset_a"
    dev_run = tmp_path / "dev_dataset_b"
    _prepare(fit_run, episodes_override=fit_episodes)
    _prepare(dev_run, episodes_override=dev_episodes)
    manifest_path = dev_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["dataset_sha256"] = "c" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validate = validate_run(dev_run)
    (dev_run / "validation_report.json").write_text(json.dumps(validate), encoding="utf-8")
    fit_approval = _sign(fit_run, tmp_path, key, name="fit-a")
    dev_approval = _sign(dev_run, tmp_path, key, name="dev-b")

    with pytest.raises(FormalSplitError, match="incompatible dataset or split identities"):
        load_accepted_run_set([(fit_run, fit_approval, key), (dev_run, dev_approval, key)])


def test_one_quarantined_transition_refuses_the_whole_run(tmp_path: Path):
    key = _key(tmp_path)
    run = tmp_path / "mixed_run"
    _prepare(run, steps=2, statuses=["valid", "quarantined"])
    approval = _sign(run, tmp_path, key)
    with pytest.raises(FormalSplitError, match="quarantined.*refuses the whole run"):
        _rows(run, approval, key)


def test_missing_quality_is_not_treated_as_valid(tmp_path: Path):
    key = _key(tmp_path)
    run = tmp_path / "missing_quality"
    _prepare(run)
    rows = [json.loads(line) for line in (run / "transitions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["quality"] = {}
    _dump_jsonl(run / "transitions.jsonl", rows)
    report = validate_run(run)
    assert report["hard_error_count"] > 0
    (run / "validation_report.json").write_text(json.dumps({"schema_version": "wm-transition/0.2", "hard_error_count": 0, "transitions": 1, "episodes": report["episodes"], "observations": report["observations"], "privileged_labels": report["privileged_labels"], "counts": {"valid": 1, "partial": 0, "quarantined": 0}}), encoding="utf-8")
    approval = _sign(run, tmp_path, key)
    with pytest.raises(FormalSplitError, match="unknown_quality|quality"):
        _rows(run, approval, key)


def test_all_quarantined_transitions_refuse_the_whole_run(tmp_path: Path):
    key = _key(tmp_path)
    run = tmp_path / "all_quarantine"
    _prepare(run, steps=2, statuses=["quarantined", "quarantined"])
    approval = _sign(run, tmp_path, key)
    with pytest.raises(FormalSplitError, match="quarantined.*refuses the whole run"):
        _rows(run, approval, key)


def test_zero_hard_errors_and_empty_transitions_are_refused(tmp_path: Path):
    key = _key(tmp_path)
    run = tmp_path / "empty_transitions"
    _prepare(run)
    (run / "transitions.jsonl").write_text("", encoding="utf-8")
    report = validate_run(run)
    assert report["hard_error_count"] == 0
    assert report["transitions"] == 0
    (run / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")
    approval = _sign(run, tmp_path, key)
    with pytest.raises(FormalSplitError, match="transition record is empty"):
        _rows(run, approval, key)


def test_report_transition_count_must_match_the_jsonl(tmp_path: Path):
    key = _key(tmp_path)
    run = tmp_path / "count_mismatch"
    _prepare(run)
    report = json.loads((run / "validation_report.json").read_text(encoding="utf-8"))
    report["transitions"] = 99
    (run / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")
    approval = _sign(run, tmp_path, key)
    with pytest.raises(FormalSplitError, match="report transitions count does not match"):
        _rows(run, approval, key)


def test_damaged_jsonl_dangling_refs_and_bad_images_are_refused(tmp_path: Path):
    key = _key(tmp_path)
    damaged = tmp_path / "damaged_jsonl"
    _prepare(damaged)
    (damaged / "transitions.jsonl").write_text("{\n", encoding="utf-8")
    approval = _sign(damaged, tmp_path, key)
    with pytest.raises(FormalSplitError, match="jsonl_unreadable"):
        _rows(damaged, approval, key)

    dangling = tmp_path / "dangling_obs"
    _prepare(dangling)
    rows = [json.loads(line) for line in (dangling / "transitions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["obs_before_id"] = "missing-obs"
    _dump_jsonl(dangling / "transitions.jsonl", rows)
    approval = _sign(dangling, tmp_path, key)
    with pytest.raises(FormalSplitError, match="temporal_break"):
        _rows(dangling, approval, key)

    missing_image = tmp_path / "missing_image"
    _prepare(missing_image)
    next(iter((missing_image / "assets").glob("*.npy"))).unlink()
    approval = _sign(missing_image, tmp_path, key)
    with pytest.raises(FormalSplitError, match="image_missing"):
        _rows(missing_image, approval, key)

    bad_hash = tmp_path / "bad_hash"
    _prepare(bad_hash)
    image = next(iter((bad_hash / "assets").glob("*.npy")))
    blob = bytearray(image.read_bytes())
    blob[-1] ^= 0xFF
    image.write_bytes(blob)
    approval = _sign(bad_hash, tmp_path, key)
    with pytest.raises(FormalSplitError, match="image_hash_mismatch"):
        _rows(bad_hash, approval, key)

    duplicate = tmp_path / "duplicate_id"
    _prepare(duplicate, steps=2)
    rows = [json.loads(line) for line in (duplicate / "transitions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[1]["transition_id"] = rows[0]["transition_id"]
    _dump_jsonl(duplicate / "transitions.jsonl", rows)
    report = validate_run(duplicate)
    assert report["hard_error_count"] == 0
    (duplicate / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")
    approval = _sign(duplicate, tmp_path, key)
    with pytest.raises(FormalSplitError, match="duplicate or missing transition id"):
        _rows(duplicate, approval, key)


def test_manifest_or_report_edits_after_approval_are_refused(tmp_path: Path):
    key = _key(tmp_path)
    manifest_run = tmp_path / "edited_manifest"
    _prepare(manifest_run)
    approval = _sign(manifest_run, tmp_path, key)
    manifest = json.loads((manifest_run / "manifest.json").read_text(encoding="utf-8"))
    manifest["training_use"] = "formal"
    _thaw_tree(manifest_run)
    (manifest_run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _freeze_tree(manifest_run)
    with pytest.raises(FormalSplitError, match="digest does not match"):
        _rows(manifest_run, approval, key)

    report_run = tmp_path / "edited_report"
    _prepare(report_run)
    approval = _sign(report_run, tmp_path, key)
    report = json.loads((report_run / "validation_report.json").read_text(encoding="utf-8"))
    report["hard_error_count"] = 0
    report["note"] = "edited after approval"
    _thaw_tree(report_run)
    (report_run / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")
    _freeze_tree(report_run)
    with pytest.raises(FormalSplitError, match="digest does not match"):
        _rows(report_run, approval, key)


def test_refusals_do_not_return_partial_training_rows(tmp_path: Path):
    key = _key(tmp_path)
    run = tmp_path / "no_partial"
    _prepare(run, steps=2, statuses=["valid", "quarantined"])
    approval = _sign(run, tmp_path, key)
    with pytest.raises(FormalSplitError, match="refuses the whole run"):
        bundle = load_accepted_run(run, approval_path=approval, approval_key_path=key)
        raise AssertionError(supervised_rows(bundle))


def test_path_blocks_and_failure_files_still_refuse(tmp_path: Path):
    episodes, _episode = _formal_episodes()
    forged = _write_run(tmp_path / "pilot_bounded_acceptance_20261026T015701Z", episodes)
    with pytest.raises(FormalSplitError, match="pilot_bounded_acceptance"):
        load_accepted_run(forged)
    old_pilot = _write_run(tmp_path / "pilot_20261023_train3_r2", episodes)
    with pytest.raises(FormalSplitError, match="pilot_"):
        load_accepted_run(old_pilot)
    new_acceptance = _write_run(tmp_path / "acceptance_batch_9", episodes, training_use="not_approved")
    with pytest.raises(FormalSplitError, match="acceptance_batch_9"):
        load_accepted_run(new_acceptance)
    with pytest.raises(FormalSplitError, match="pilot_bounded_acceptance_20261026T015701Z"):
        load_accepted_run(PLANNED)
    assert not PLANNED.exists()

    failed = _write_run(tmp_path / "failed_run", episodes)
    (failed / "failure.json").write_text('{"status":"failed"}\n', encoding="utf-8")
    _freeze_tree(failed)
    with pytest.raises(FormalSplitError, match="failure.json"):
        load_accepted_run(failed)
    invalid = _write_run(tmp_path / "invalid_run", episodes)
    (invalid / "validation_report.json").write_text(json.dumps({"hard_error_count": 2}), encoding="utf-8")
    _freeze_tree(invalid)
    with pytest.raises(FormalSplitError, match="hard error"):
        load_accepted_run(invalid)

    diagnostic = _write_run(tmp_path / "diagnostic_run", episodes, diagnostic_only=True)
    _freeze_tree(diagnostic)
    with pytest.raises(FormalSplitError, match="diagnostic_only"):
        load_accepted_run(diagnostic)


def _forge_clean_report(run: Path) -> None:
    fresh = json.loads((run / "validation_report.json").read_text(encoding="utf-8"))
    fresh["hard_error_count"] = 0
    fresh["errors"] = []
    fresh["counts"] = {"valid": fresh["transitions"], "partial": 0, "quarantined": 0}
    (run / "validation_report.json").write_text(json.dumps(fresh), encoding="utf-8")


def test_approval_key_and_record_boundaries(tmp_path: Path):
    run = tmp_path / "key_boundary"
    _prepare(run)
    approval = _sign(run, tmp_path, _key(tmp_path))
    via_link = tmp_path / "approvals" / "via_link.json"
    via_link.symlink_to(approval)
    with pytest.raises(FormalSplitError, match="without following links"):
        _rows(run, via_link, _key(tmp_path))
    empty = tmp_path / "empty.key"
    empty.write_bytes(b"")
    _chmod(empty, writable=False)
    short = tmp_path / "short.key"
    short.write_bytes(b"0123456789abcde")
    _chmod(short, writable=False)
    wrong = tmp_path / "wrong.key"
    wrong.write_bytes(b"different-approval-key-32b!!")
    _chmod(wrong, writable=False)
    with pytest.raises(FormalSplitError, match="too short"):
        _rows(run, approval, empty)
    with pytest.raises(FormalSplitError, match="too short"):
        _rows(run, approval, short)
    with pytest.raises(FormalSplitError, match="modified"):
        _rows(run, approval, wrong)

    document = json.loads(approval.read_text(encoding="utf-8"))
    document["comment"] = "not a grant"
    payload = {name: document[name] for name in ("schema_version", "run_name", "files")}
    document["mac"] = approval_mac(_key(tmp_path).read_bytes(), payload)
    _rewrite_approval(approval, json.dumps(document))
    with pytest.raises(FormalSplitError, match="unknown fields"):
        _rows(run, approval, _key(tmp_path))

    approval.unlink()
    _sign(run, tmp_path, _key(tmp_path))
    document = json.loads(approval.read_text(encoding="utf-8"))
    document["files"] = ["manifest.json"]
    document["mac"] = approval_mac(
        _key(tmp_path).read_bytes(),
        {name: document[name] for name in ("schema_version", "run_name", "files")},
    )
    _rewrite_approval(approval, json.dumps(document))
    with pytest.raises(FormalSplitError, match="damaged"):
        _rows(run, approval, _key(tmp_path))


def test_changes_after_approval_refuse_the_whole_run(tmp_path: Path):
    key = _key(tmp_path)
    image_run = tmp_path / "image_replaced"
    _prepare(image_run, steps=2)
    approval = _sign(image_run, tmp_path, key)
    image = next(iter((image_run / "assets").glob("*.npy")))
    blob = bytearray(image.read_bytes())
    blob[-1] ^= 0xFF
    _thaw_tree(image_run)
    image.write_bytes(blob)
    _freeze_tree(image_run)
    with pytest.raises(FormalSplitError, match="image_hash_mismatch|digest does not match"):
        bundle = load_accepted_run(image_run, approval_path=approval, approval_key_path=key)
        raise AssertionError(bundle)

    jsonl_run = tmp_path / "jsonl_replaced"
    _prepare(jsonl_run)
    approval = _sign(jsonl_run, tmp_path, key)
    rows = [json.loads(line) for line in (jsonl_run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["instruction_text"] = "replaced after approval"
    _thaw_tree(jsonl_run)
    _dump_jsonl(jsonl_run / "episodes.jsonl", rows)
    _freeze_tree(jsonl_run)
    with pytest.raises(FormalSplitError, match="digest does not match"):
        load_accepted_run(jsonl_run, approval_path=approval, approval_key_path=key)

    extra_run = tmp_path / "extra_file"
    _prepare(extra_run)
    approval = _sign(extra_run, tmp_path, key)
    _thaw_tree(extra_run)
    (extra_run / "notes.txt").write_text("not in the original snapshot\n", encoding="utf-8")
    _freeze_tree(extra_run)
    with pytest.raises(FormalSplitError, match="digest does not match"):
        load_accepted_run(extra_run, approval_path=approval, approval_key_path=key)


def test_returned_samples_stay_on_the_approved_bytes(tmp_path: Path):
    key = _key(tmp_path)
    run = tmp_path / "sealed_run"
    _prepare(run)
    approval = _sign(run, tmp_path, key)
    bundle = load_accepted_run(run, approval_path=approval, approval_key_path=key)
    rows = supervised_rows(bundle)
    original = read_admitted_rgb(rows[0]["model_input"]["rgb_t"]).copy()
    instruction = bundle["episodes"][0]["instruction_text"]
    _thaw_tree(run)
    for image in (run / "assets").glob("*.npy"):
        image.write_bytes(b"not-the-approved-npy")
    episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    episodes[0]["instruction_text"] = "changed after the snapshot was returned"
    _dump_jsonl(run / "episodes.jsonl", episodes)
    assert bundle["episodes"][0]["instruction_text"] == instruction
    assert read_admitted_rgb(rows[0]["model_input"]["rgb_t"]).tobytes() == original.tobytes()
    frame = rows[0]["model_input"]["rgb_t"]
    with pytest.raises(FormalSplitError, match="filesystem path"):
        read_admitted_rgb(frame["relative_path"])
    forged = dict(frame)
    forged["pixels"] = np.zeros_like(original)
    forged["pixel_sha256"] = "0" * 64
    with pytest.raises(FormalSplitError, match="not an admitted frame"):
        read_admitted_rgb(forged)
    assert read_admitted_rgb(frame).tobytes() == original.tobytes()


def test_paths_symlinks_and_special_files_are_refused(tmp_path: Path):
    """Linux directory fds and O_NOFOLLOW. This does not claim a live cross-file race."""
    key = _key(tmp_path)
    outside = tmp_path / "outside.npy"
    np.save(outside, np.ones((2, 2, 3), dtype=np.uint8))
    escaped = tmp_path / "escaped_path"
    _prepare(escaped)
    observations = [
        json.loads(line) for line in (escaped / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in observations:
        row["rgb"]["path"] = "../outside.npy"
        row["rgb"]["sha256"] = sha256_file(outside)
    _dump_jsonl(escaped / "observations.jsonl", observations)
    for image in (escaped / "assets").glob("*.npy"):
        image.unlink()
    _forge_clean_report(escaped)
    approval = _sign(escaped, tmp_path, key)
    with pytest.raises(FormalSplitError, match="path_escape"):
        load_accepted_run(escaped, approval_path=approval, approval_key_path=key)

    absolute = tmp_path / "absolute_path"
    _prepare(absolute)
    observations = [
        json.loads(line) for line in (absolute / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in observations:
        row["rgb"]["path"] = str(outside)
        row["rgb"]["sha256"] = sha256_file(outside)
    _dump_jsonl(absolute / "observations.jsonl", observations)
    for image in (absolute / "assets").glob("*.npy"):
        image.unlink()
    _forge_clean_report(absolute)
    approval = _sign(absolute, tmp_path, key)
    with pytest.raises(FormalSplitError, match="path_escape"):
        load_accepted_run(absolute, approval_path=approval, approval_key_path=key)

    linked_target = tmp_path / "elsewhere" / "clean_run"
    _prepare(linked_target)
    approval = _sign(linked_target, tmp_path, key)
    link = tmp_path / "clean_run"
    link.symlink_to(linked_target, target_is_directory=True)
    with pytest.raises(FormalSplitError, match="without following links"):
        load_accepted_run(link, approval_path=approval, approval_key_path=key)

    file_link = tmp_path / "file_link"
    _prepare(file_link)
    approval = _sign(file_link, tmp_path, key)
    _thaw_tree(file_link)
    asset = next(iter((file_link / "assets").glob("*.npy")))
    asset.unlink()
    asset.symlink_to(outside)
    _freeze_tree(file_link)
    with pytest.raises(FormalSplitError, match="symlink"):
        load_accepted_run(file_link, approval_path=approval, approval_key_path=key)

    directory_link = tmp_path / "directory_link"
    _prepare(directory_link)
    approval = _sign(directory_link, tmp_path, key)
    _thaw_tree(directory_link)
    hidden = tmp_path / "hidden_assets"
    hidden.mkdir()
    (hidden / "note.txt").write_text("outside\n", encoding="utf-8")
    (directory_link / "assets" / "linked").symlink_to(hidden, target_is_directory=True)
    _freeze_tree(directory_link)
    with pytest.raises(FormalSplitError, match="symlink"):
        load_accepted_run(directory_link, approval_path=approval, approval_key_path=key)

    pipe_run = tmp_path / "fifo_run"
    _prepare(pipe_run)
    approval = _sign(pipe_run, tmp_path, key)
    _thaw_tree(pipe_run)
    os.mkfifo(pipe_run / "assets" / "pipe")
    _freeze_tree(pipe_run)
    with pytest.raises(FormalSplitError, match="non-regular"):
        load_accepted_run(pipe_run, approval_path=approval, approval_key_path=key)

    linked_file = tmp_path / "hardlink_run"
    _prepare(linked_file)
    approval = _sign(linked_file, tmp_path, key)
    _thaw_tree(linked_file)
    os.link(outside, linked_file / "assets" / "alias.npy")
    _freeze_tree(linked_file)
    with pytest.raises(FormalSplitError, match="hard link"):
        load_accepted_run(linked_file, approval_path=approval, approval_key_path=key)


def test_duplicate_ids_and_broken_references_refuse_every_row(tmp_path: Path):
    key = _key(tmp_path)
    episodes_run = tmp_path / "duplicate_episode"
    _prepare(episodes_run, steps=2)
    rows = [json.loads(line) for line in (episodes_run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[1]["episode_uid"] = rows[0]["episode_uid"]
    _dump_jsonl(episodes_run / "episodes.jsonl", rows)
    _forge_clean_report(episodes_run)
    approval = _sign(episodes_run, tmp_path, key)
    with pytest.raises(FormalSplitError, match="duplicate_episode"):
        bundle = load_accepted_run(episodes_run, approval_path=approval, approval_key_path=key)
        raise AssertionError(len(supervised_rows(bundle)))

    labels_run = tmp_path / "duplicate_label"
    _prepare(labels_run, steps=2)
    poses = [
        json.loads(line) for line in (labels_run / "privileged_labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    poses[1]["label_id"] = poses[0]["label_id"]
    _dump_jsonl(labels_run / "privileged_labels.jsonl", poses)
    _forge_clean_report(labels_run)
    approval = _sign(labels_run, tmp_path, key)
    with pytest.raises(FormalSplitError, match="duplicate_label"):
        load_accepted_run(labels_run, approval_path=approval, approval_key_path=key)

    reference_run = tmp_path / "reference_mismatch"
    _prepare(reference_run)
    poses = [
        json.loads(line)
        for line in (reference_run / "privileged_labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    poses[0]["observation_id"] = "other-observation"
    _dump_jsonl(reference_run / "privileged_labels.jsonl", poses)
    _forge_clean_report(reference_run)
    approval = _sign(reference_run, tmp_path, key)
    with pytest.raises(FormalSplitError, match="reference_mismatch"):
        load_accepted_run(reference_run, approval_path=approval, approval_key_path=key)


def test_sealed_pixels_survive_public_mutation(tmp_path: Path):
    key = _key(tmp_path)
    run = tmp_path / "immutable_pixels"
    _prepare(run, steps=2)
    approval = _sign(run, tmp_path, key)
    rows = _rows(run, approval, key)
    first = rows[0]["model_input"]["rgb_t"]
    second = rows[0]["model_input"]["rgb_t1"]
    approved_first = read_admitted_rgb(first).tobytes()
    approved_second = read_admitted_rgb(second).tobytes()

    subscript = read_admitted_rgb(first)
    subscript[0, 0, 0] = np.uint8(9)
    subscript.setflags(write=True)
    subscript[0, 0, 0] = np.uint8(4)
    public = first["pixels"]
    public.setflags(write=True)
    public[0, 0, 0] = np.uint8(3)
    with pytest.raises(FormalSplitError, match="rejects public modification"):
        first["pixels"] = np.zeros_like(public)
    with pytest.raises(FormalSplitError, match="rejects public modification"):
        first["pixel_sha256"] = "f" * 64
    assert read_admitted_rgb(first).tobytes() == approved_first
    assert read_admitted_rgb(second).tobytes() == approved_second
    again = read_admitted_rgb(first)
    assert not np.shares_memory(again, subscript)
    assert not np.shares_memory(again, public)
    assert not np.shares_memory(read_admitted_rgb(first), read_admitted_rgb(second))


def test_parent_symlinks_are_refused_and_real_paths_open(tmp_path: Path):
    key = _key(tmp_path)
    real = tmp_path / "real_parent" / "clean_run"
    _prepare(real)
    approval = _sign(real, tmp_path, key)
    loaded = read_admitted_rgb(_rows(real, approval, key)[0]["model_input"]["rgb_t"])
    assert loaded.shape == (2, 2, 3)
    assert str(loaded.dtype) == "uint8"

    parent_link = tmp_path / "link_parent"
    parent_link.symlink_to(tmp_path / "real_parent", target_is_directory=True)
    with pytest.raises(FormalSplitError, match="without following links"):
        load_accepted_run(parent_link / "clean_run", approval_path=approval, approval_key_path=key)

    ancestor = tmp_path / "real_ancestor" / "mid" / "clean_run"
    _prepare(ancestor)
    ancestor_approval = _sign(ancestor, tmp_path, key, name="ancestor_clean_run")
    ancestor_link = tmp_path / "link_ancestor"
    ancestor_link.symlink_to(tmp_path / "real_ancestor", target_is_directory=True)
    with pytest.raises(FormalSplitError, match="without following links"):
        load_accepted_run(
            ancestor_link / "mid" / "clean_run",
            approval_path=ancestor_approval,
            approval_key_path=key,
        )

    dotted = Path(f"{real}/../clean_run")
    with pytest.raises(ValueError, match="without following links"):
        snapshot_run(dotted)


def test_stat_then_replace_with_symlink_is_rejected_by_relative_open(tmp_path: Path):
    """Interleaving is explicit, not a racing thread.

    1. Hold the parent directory fd and lstat the child name as a real directory.
    2. Rename that directory aside and put a symlink at the same name.
    3. Call the production relative open with O_NOFOLLOW. It must fail.
    This does not enter the inode check, because the open itself fails.
    The window inside one open() syscall is not covered.
    """
    parent = tmp_path / "race_parent"
    child = parent / "child"
    child.mkdir(parents=True)
    held = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        info = os.stat("child", dir_fd=held, follow_symlinks=False)
        assert stat.S_ISDIR(info.st_mode)
        aside = tmp_path / "child_real"
        os.rename(child, aside)
        (parent / "child").symlink_to(aside, target_is_directory=True)
        with pytest.raises(OSError):
            open_relative_nofollow(held, "child", directory=True)
    finally:
        os.close(held)


def test_replacement_between_lstat_and_open_is_refused(tmp_path: Path):
    """Interleaving is the production hook, not a racing thread.

    ``open_checked_nofollow`` lstats, runs ``after_lstat``, then opens with
    ``O_NOFOLLOW`` and requires the same inode. The hook renames the original
    entry aside and creates a different directory or file at that name. The
    window inside one syscall, and a writer changing bytes of an already-open
    inode, are not covered.
    """
    parent = tmp_path / "swap_parent"
    child = parent / "child"
    child.mkdir(parents=True)
    (child / "marker.txt").write_text("original")
    held = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        def swap_directory() -> None:
            os.rename(child, tmp_path / "child_real")
            (parent / "child").mkdir()
            (parent / "child" / "marker.txt").write_text("replaced")

        before = set(os.listdir("/proc/self/fd"))
        with pytest.raises(ValueError, match="replaced before open"):
            open_checked_nofollow(held, "child", directory=True, after_lstat=swap_directory)
        assert set(os.listdir("/proc/self/fd")) == before
        assert (tmp_path / "child_real" / "marker.txt").read_text() == "original"
        assert (parent / "child" / "marker.txt").read_text() == "replaced"
    finally:
        os.close(held)

    file_parent = tmp_path / "file_parent"
    file_parent.mkdir()
    original = file_parent / "frame.npy"
    original.write_bytes(b"approved-bytes")
    file_fd = os.open(file_parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        def swap_file() -> None:
            os.rename(original, tmp_path / "frame_real.npy")
            (file_parent / "frame.npy").write_bytes(b"replacement-bytes")

        with pytest.raises(ValueError, match="replaced before open"):
            open_checked_nofollow(file_fd, "frame.npy", directory=False, after_lstat=swap_file)
        assert (tmp_path / "frame_real.npy").read_bytes() == b"approved-bytes"
        assert (file_parent / "frame.npy").read_bytes() == b"replacement-bytes"
    finally:
        os.close(file_fd)


def test_formal_reads_refuse_writable_inputs(tmp_path: Path):
    """Writable formal inputs are refused. A frozen read blocks a new write open.

    The hook runs on the first ``os.read`` of the frozen load and tries
    ``O_WRONLY`` on an image. That open fails, and the admitted pixels match the
    file from before the attempt. This does not cover a writable descriptor
    opened before the mode change, the owner restoring the write bit, root, or
    another mount. ``st_size`` and ``st_mtime`` are not compared.
    """
    key = _key(tmp_path)
    writable = tmp_path / "writable_run"
    _prepare(writable)
    with pytest.raises(FormalSplitError, match="not frozen"):
        issue_run_approval(writable, tmp_path / "approvals" / "writable_run.json", key)
    assert not (tmp_path / "approvals" / "writable_run.json").exists()

    run = tmp_path / "frozen_during_read"
    _prepare(run)
    image = next(iter((run / "assets").glob("*.npy")))
    before = image.read_bytes()
    approval = _sign(run, tmp_path, key)
    loose = tmp_path / "loose.key"
    loose.write_bytes(key.read_bytes())
    with pytest.raises(FormalSplitError, match="not frozen"):
        load_accepted_run(run, approval_path=approval, approval_key_path=loose)
    _chmod(approval, writable=True)
    with pytest.raises(FormalSplitError, match="not frozen"):
        load_accepted_run(run, approval_path=approval, approval_key_path=key)
    _chmod(approval, writable=False)

    state = {"errno": None, "tried": False}
    real_read = os.read

    def hooked(fd: int, size: int) -> bytes:
        if not state["tried"]:
            state["tried"] = True
            try:
                writer = os.open(image, os.O_WRONLY)
            except OSError as exc:
                state["errno"] = exc.errno
            else:
                os.write(writer, b"mutated")
                os.close(writer)
        return real_read(fd, size)

    import experiments.world_model.validate as validate

    validate.os.read = hooked
    try:
        rows = _rows(run, approval, key)
    finally:
        validate.os.read = real_read
    assert state["errno"] == errno.EACCES
    assert image.read_bytes() == before
    assert read_admitted_rgb(rows[0]["model_input"]["rgb_t"]).tobytes() == np.load(io.BytesIO(before)).tobytes()


def test_approval_and_key_parent_symlinks_are_refused(tmp_path: Path):
    """Credential parent symlinks use the same walk as the run.

    The final component is a regular file. A full-path ``O_NOFOLLOW`` open would
    follow the parent; this walk refuses it and does not create an approval in
    the link target. Replacing a directory entry between lstat and open is
    covered by ``test_replacement_between_lstat_and_open``, which calls
    ``open_checked_nofollow`` for one component. That is the opener used here.
    """
    key = _key(tmp_path)
    run = tmp_path / "credential_run"
    _prepare(run)
    approval = _sign(run, tmp_path, key)
    assert len(_rows(run, approval, key)) == 1

    parent_link = tmp_path / "link_approvals"
    parent_link.symlink_to(approval.parent, target_is_directory=True)
    with pytest.raises(FormalSplitError, match="without following links"):
        load_accepted_run(run, approval_path=parent_link / approval.name, approval_key_path=key)

    real_keys = tmp_path / "real_keys"
    real_keys.mkdir()
    key_file = real_keys / "k"
    key_file.write_bytes(key.read_bytes())
    _chmod(key_file, writable=False)
    key_link = tmp_path / "link_keys"
    key_link.symlink_to(real_keys, target_is_directory=True)
    with pytest.raises(FormalSplitError, match="without following links"):
        load_accepted_run(run, approval_path=approval, approval_key_path=key_link / "k")

    ancestor = tmp_path / "real_cred_root" / "mid"
    ancestor.mkdir(parents=True)
    ancestor_key = ancestor / "k"
    ancestor_key.write_bytes(key.read_bytes())
    _chmod(ancestor_key, writable=False)
    ancestor_link = tmp_path / "link_cred_root"
    ancestor_link.symlink_to(tmp_path / "real_cred_root", target_is_directory=True)
    with pytest.raises(FormalSplitError, match="without following links"):
        load_accepted_run(
            run,
            approval_path=approval,
            approval_key_path=ancestor_link / "mid" / "k",
        )

    real_dir = tmp_path / "real_issue_dir"
    real_dir.mkdir()
    issue_link = tmp_path / "issue_link"
    issue_link.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(FormalSplitError, match="without following links"):
        issue_run_approval(run, issue_link / "new.json", key)
    assert not (real_dir / "new.json").exists()


def _identity(path: Path) -> tuple[int, int, bytes]:
    info = path.lstat()
    if stat.S_ISREG(info.st_mode) and not path.is_symlink():
        payload = path.read_bytes()
    else:
        payload = b""
    return info.st_ino, stat.S_IMODE(info.st_mode), payload


def test_issue_run_approval_never_replaces_an_existing_record(tmp_path: Path):
    """Production issue_run_approval publishes with renameat2(RENAME_NOREPLACE).

    Failure injections patch the production write, fsync, and rename calls used
    by that entry. They are not a scheduler race except the two-thread case,
    which waits inside rename so both callers have already written a temp file.
    """
    import experiments.world_model.validate as validate

    key = _key(tmp_path)
    run = tmp_path / "publish_run"
    _prepare(run)
    approval = _sign(run, tmp_path, key)
    original = _identity(approval)
    with pytest.raises(FormalSplitError, match="already exists"):
        issue_run_approval(run, approval, key)
    assert _identity(approval) == original
    rows = _rows(run, approval, key)
    assert len(rows) == 1

    def refuse_existing(target: Path) -> None:
        before = _identity(target)
        with pytest.raises(FormalSplitError, match="already exists"):
            issue_run_approval(run, target, key)
        assert _identity(target) == before
        assert not [path for path in target.parent.iterdir() if path.name.endswith(".partial")]

    link_target = tmp_path / "kept.json"
    link_target.write_bytes(b"keep-target")
    link = tmp_path / "approvals" / "link.json"
    link.symlink_to(link_target)
    refuse_existing(link)
    assert link.is_symlink()
    assert link_target.read_bytes() == b"keep-target"

    directory = tmp_path / "approvals" / "dir.json"
    directory.mkdir()
    refuse_existing(directory)
    assert directory.is_dir()

    pipe = tmp_path / "approvals" / "pipe.json"
    os.mkfifo(pipe)
    refuse_existing(pipe)
    assert stat.S_ISFIFO(pipe.lstat().st_mode)

    source = tmp_path / "hardlink-src.json"
    source.write_bytes(b"linked-bytes")
    linked = tmp_path / "approvals" / "linked.json"
    os.link(source, linked)
    refuse_existing(linked)
    assert linked.stat().st_nlink == 2
    assert linked.read_bytes() == b"linked-bytes"

    blocked = tmp_path / "blocked_approvals"
    blocked.mkdir()
    keeper = blocked / "keeper.txt"
    keeper.write_bytes(b"keeper")
    _chmod(keeper, writable=False)
    _chmod(blocked, writable=False)
    with pytest.raises(FormalSplitError, match="temporary file failed"):
        issue_run_approval(run, blocked / "new.json", key)
    assert keeper.read_bytes() == b"keeper"
    assert not (blocked / "new.json").exists()
    assert not [path for path in blocked.iterdir() if path.name.endswith(".partial")]

    real_write = validate.os.write

    def fail_write(fd: int, data: bytes) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    validate.os.write = fail_write
    try:
        with pytest.raises(FormalSplitError, match="write failed"):
            issue_run_approval(run, tmp_path / "approvals" / "write.json", key)
    finally:
        validate.os.write = real_write
    assert not (tmp_path / "approvals" / "write.json").exists()
    assert not [path for path in (tmp_path / "approvals").iterdir() if path.name.endswith(".partial")]

    real_fsync = validate.os.fsync

    def fail_sync(fd: int) -> None:
        raise OSError(errno.EIO, "Input/output error")

    validate.os.fsync = fail_sync
    try:
        with pytest.raises(FormalSplitError, match="sync failed"):
            issue_run_approval(run, tmp_path / "approvals" / "sync.json", key)
    finally:
        validate.os.fsync = real_fsync
    assert not (tmp_path / "approvals" / "sync.json").exists()
    assert not [path for path in (tmp_path / "approvals").iterdir() if path.name.endswith(".partial")]

    real_rename = validate.rename_noreplace

    def fail_publish(old_dir: int, old_name: str, new_dir: int, new_name: str) -> None:
        raise OSError(errno.EIO, "Input/output error")

    validate.rename_noreplace = fail_publish
    try:
        with pytest.raises(FormalSplitError, match="publish failed"):
            issue_run_approval(run, tmp_path / "approvals" / "publish.json", key)
    finally:
        validate.rename_noreplace = real_rename
    assert not (tmp_path / "approvals" / "publish.json").exists()
    assert not [path for path in (tmp_path / "approvals").iterdir() if path.name.endswith(".partial")]

    def plant_destination(old_dir: int, old_name: str, new_dir: int, new_name: str) -> None:
        planted = os.open(new_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644, dir_fd=new_dir)
        os.write(planted, b"planted-approval")
        os.close(planted)
        real_rename(old_dir, old_name, new_dir, new_name)

    validate.rename_noreplace = plant_destination
    try:
        with pytest.raises(FormalSplitError, match="already exists"):
            issue_run_approval(run, tmp_path / "approvals" / "raced.json", key)
    finally:
        validate.rename_noreplace = real_rename
    raced = tmp_path / "approvals" / "raced.json"
    assert raced.read_bytes() == b"planted-approval"
    assert not [path for path in (tmp_path / "approvals").iterdir() if path.name.endswith(".partial")]

    def swap_temp_then_fail(old_dir: int, old_name: str, new_dir: int, new_name: str) -> None:
        other = old_name + ".other"
        other_fd = os.open(other, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644, dir_fd=old_dir)
        os.write(other_fd, b"not-our-temp")
        os.close(other_fd)
        os.rename(old_name, old_name + ".moved", src_dir_fd=old_dir, dst_dir_fd=old_dir)
        os.rename(other, old_name, src_dir_fd=old_dir, dst_dir_fd=old_dir)
        raise OSError(errno.EEXIST, "File exists")

    validate.rename_noreplace = swap_temp_then_fail
    try:
        with pytest.raises(FormalSplitError, match="already exists"):
            issue_run_approval(run, tmp_path / "approvals" / "cleanup.json", key)
    finally:
        validate.rename_noreplace = real_rename
    assert not (tmp_path / "approvals" / "cleanup.json").exists()
    swapped = [
        path
        for path in (tmp_path / "approvals").iterdir()
        if path.name.endswith(".partial") and path.read_bytes() == b"not-our-temp"
    ]
    assert len(swapped) == 1

    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def publish_once() -> None:
        def wait_then_rename(old_dir: int, old_name: str, new_dir: int, new_name: str) -> None:
            barrier.wait(timeout=5)
            real_rename(old_dir, old_name, new_dir, new_name)

        validate.rename_noreplace = wait_then_rename
        try:
            issue_run_approval(run, tmp_path / "approvals" / "concurrent.json", key)
            outcomes.append("ok")
        except FormalSplitError as exc:
            outcomes.append(str(exc))

    first = threading.Thread(target=publish_once)
    second = threading.Thread(target=publish_once)
    first.start()
    second.start()
    first.join()
    second.join()
    validate.rename_noreplace = real_rename
    assert outcomes.count("ok") == 1
    assert any("already exists" in item for item in outcomes)
    concurrent = tmp_path / "approvals" / "concurrent.json"
    assert stat.S_IMODE(concurrent.stat().st_mode) & 0o222 == 0
    assert not [
        path
        for path in (tmp_path / "approvals").iterdir()
        if path.name.startswith(".concurrent.json.") and path.name.endswith(".partial")
    ]
    loaded = load_accepted_run(run, approval_path=concurrent, approval_key_path=key)
    assert loaded["admitted_frames"]


def _jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_runner_budget_cannot_be_labeled_as_a_complete_trajectory(tmp_path: Path):
    run = tmp_path / "budget_label"
    _prepare(run)
    episodes = _jsonl_rows(run / "episodes.jsonl")
    episodes[0]["completion"] = {"state": "completed"}
    _write_jsonl(run / "episodes.jsonl", episodes)
    transitions = _jsonl_rows(run / "transitions.jsonl")
    transitions[0]["outcome"] = {
        "end_sources": ["runner_budget"],
        "primary_end_source": "runner_budget",
    }
    _write_jsonl(run / "transitions.jsonl", transitions)
    report = validate_run(run)
    assert any(item["code"] == "budget_labeled_complete" for item in report["errors"])
    with pytest.raises(FormalSplitError, match="runner_budget"):
        supervised_rows(read_recorded_run(run))


def test_command_time_cannot_be_stored_as_simulation_time(tmp_path: Path):
    run = tmp_path / "sim_time"
    _prepare(run)
    transitions = _jsonl_rows(run / "transitions.jsonl")
    transitions[0]["action"]["dt_sim_s"] = transitions[0]["action"]["execution_ack"]["dt_command_s"]
    _write_jsonl(run / "transitions.jsonl", transitions)
    report = validate_run(run)
    assert any(item["code"] == "sim_time_mismatch" for item in report["errors"])
    with pytest.raises(FormalSplitError, match="consistent measured simulator clock"):
        supervised_rows(read_recorded_run(run))


def test_diagnostic_read_cannot_enter_training(tmp_path: Path):
    key = _key(tmp_path)
    run = tmp_path / "training_gate"
    _prepare(run)
    approval = _sign(run, tmp_path, key)
    admitted = load_accepted_run(run, approval_path=approval, approval_key_path=key)
    rows = admitted_training_rows(admitted)
    assert not isinstance(rows[0]["model_input"]["rgb_t"], (str, Path))
    diagnostic = read_recorded_run(run)
    assert not diagnostic.get("admitted_frames")
    with pytest.raises(FormalSplitError, match="load_accepted_run"):
        admitted_training_rows(diagnostic)


def test_directory_sync_failure_does_not_roll_back_or_replace(tmp_path: Path):
    """A directory fsync after a successful no-replace publish is not a rollback.

    The approval bytes are already at the final name. The caller sees the sync
    error, a second issue is refused, and the published bytes still load.
    """
    import experiments.world_model.validate as validate

    key = _key(tmp_path)
    run = tmp_path / "dirsync_run"
    _prepare(run)
    _freeze_tree(run)
    approval = tmp_path / "approvals" / "dirsync_run.json"
    real_fsync = validate.os.fsync
    calls = {"count": 0}

    def fail_directory_sync(fd: int) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            real_fsync(fd)
            return
        raise OSError(errno.EIO, "Input/output error")

    validate.os.fsync = fail_directory_sync
    try:
        with pytest.raises(FormalSplitError, match="sync failed"):
            issue_run_approval(run, approval, key)
    finally:
        validate.os.fsync = real_fsync
    published = approval.read_bytes()
    assert published
    assert stat.S_IMODE(approval.stat().st_mode) & 0o222 == 0
    before = _identity(approval)
    with pytest.raises(FormalSplitError, match="already exists"):
        issue_run_approval(run, approval, key)
    assert _identity(approval) == before
    loaded = load_accepted_run(run, approval_path=approval, approval_key_path=key)
    assert loaded["admitted_frames"]


def test_missing_noreplace_syscall_fails_closed(tmp_path: Path):
    import ctypes

    import experiments.world_model.validate as validate

    key = _key(tmp_path)
    run = tmp_path / "noreplace_missing"
    _prepare(run)
    _freeze_tree(run)
    approval = tmp_path / "approvals" / "noreplace_missing.json"
    previous = validate._renameat2

    def unavailable(*_args: object) -> int:
        ctypes.set_errno(errno.ENOSYS)
        return -1

    validate._renameat2 = unavailable
    try:
        with pytest.raises(FormalSplitError, match="no-replace publish is unavailable"):
            issue_run_approval(run, approval, key)
    finally:
        validate._renameat2 = previous
    assert not approval.exists()
    assert not [
        path
        for path in approval.parent.iterdir()
        if path.name.endswith(".partial")
    ]

