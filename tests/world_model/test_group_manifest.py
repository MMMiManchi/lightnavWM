from __future__ import annotations

from experiments.world_model.group_manifest import campaign_episode_keys, select_scene_partition_pilot
from experiments.world_model.latent_data import scene_partition
from experiments.world_model.jsonutil import dumps, sha256_file


def test_scene_partition_pilot_is_scene_disjoint_and_labels_allowlist_rows():
    rows = []
    fit_count = 0
    dev_count = 0
    index = 0
    while fit_count < 3 or dev_count < 3:
        scene = f"scene-{index}"
        side = scene_partition(scene)
        for episode in range(3):
            rows.append({"scene": scene, "episode_id": str(episode), "route_group_id": f"{scene}:{episode}"})
        if side == "train_fit":
            fit_count += 1
        else:
            dev_count += 1
        index += 1

    selected = select_scene_partition_pilot(rows, count=4, min_scenes=2, target_partition="train_dev")
    assert len(selected) == 4
    assert len({row["scene"] for row in selected}) >= 2
    assert all(row["partition"] == "train_dev" for row in selected)
    assert all(scene_partition(row["scene"]) == "train_dev" for row in selected)


def test_scene_partition_pilot_excludes_registered_campaign_episodes():
    rows = []
    for index in range(100):
        scene = f"scene-{index}"
        for episode in range(3):
            rows.append({"scene": scene, "episode_id": str(episode), "route_group_id": f"{scene}:{episode}"})
    candidate = next(row for row in rows if scene_partition(row["scene"]) == "train_fit")
    excluded = {(candidate["scene"], candidate["episode_id"])}

    selected = select_scene_partition_pilot(
        rows, count=2, min_scenes=1, target_partition="train_fit", excluded_keys=excluded
    )

    assert len(selected) == 2
    assert all((row["scene"], row["episode_id"]) not in excluded for row in selected)


def test_campaign_episode_keys_checks_registered_run_hashes(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_manifest = {
        "source": {"split": "train"},
        "diagnostic_only": False,
        "collection_partition": "train_fit",
        "allowlist_keys": [["scene-a", "1"]],
    }
    run_path = run_dir / "manifest.json"
    run_path.write_text(dumps(run_manifest), encoding="utf-8")
    report_path = run_dir / "validation_report.json"
    report_path.write_text(dumps({"hard_error_count": 0}), encoding="utf-8")
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        dumps({
            "schema_version": "wm-campaign/0.1",
            "scope": "train_fit",
            "source_split": "train",
            "runs": [{
                "run_dir": str(run_dir),
                "manifest_sha256": sha256_file(run_path),
                "validation_report_sha256": sha256_file(report_path),
            }],
        }),
        encoding="utf-8",
    )

    assert campaign_episode_keys([campaign_path]) == {("scene-a", "1")}

    run_manifest["allowlist_keys"] = [["scene-b", "2"]]
    run_path.write_text(dumps(run_manifest), encoding="utf-8")
    try:
        campaign_episode_keys([campaign_path])
    except RuntimeError as exc:
        assert "manifest changed" in str(exc)
    else:
        raise AssertionError("changed campaign run manifest must be refused")
