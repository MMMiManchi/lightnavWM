"""Recorded region supervision must not become an oracle at policy input."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace

import numpy as np
import pytest

from experiments.world_model.branch_curation import (
    CurationRules,
    FloorPolygon,
    assign_region,
    branch_pair_summary,
    deduplicate_events,
    image_quality,
    point_in_polygon,
    region_entry,
    student_inputs,
)
from scripts.curate_branch_benchmark import curate


def rectangle(region, left, right):
    return FloorPolygon(region, "r", 0.0, ((left, -1), (right, -1), (right, 1), (left, 1)))


def frames(xs):
    return [{"position": [x, 0.1, 0]} for x in xs]


def test_coordinate_transform_and_floor_separate_overlapping_levels():
    polygon = FloorPolygon(3, "k", 2.0, ((1, 3), (4, 3), (4, 6), (1, 6)))
    rules = CurationRules()
    assert assign_region([2, 2.1, -4], [polygon], rules)["region_id"] == 3
    assert assign_region([2, 0.1, -4], [polygon], rules) is None
    assert assign_region([2, 2.1, 4], [polygon], rules) is None


def test_concave_floor_is_not_its_bounding_box():
    l_shape = ((0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3))
    assert point_in_polygon((0.5, 2), l_shape)
    assert not point_in_polygon((2, 2), l_shape)


def test_shared_boundary_and_overlapping_regions_remain_unknown():
    polygons = [rectangle(1, 0, 1), rectangle(2, 1, 3)]
    assert assign_region([1, 0.1, 0], polygons, CurationRules()) is None
    assert assign_region([1.5, 0.1, 0], [rectangle(1, 0, 2), rectangle(2, 1, 3)], CurationRules()) is None


def test_many_depth_gaps_or_a_long_same_room_walk_do_not_make_an_event():
    observations = frames(np.linspace(0.2, 4, 50))
    for frame in observations:
        frame["opening_count"] = 4
    result = region_entry(observations, [rectangle(1, 0, 5)], CurationRules())
    assert result["target_index"] is None


def test_first_deep_stable_room_entry_is_selected_from_a_long_failed_walk():
    observations = frames([0.2, 0.5, 1.05, 1.10, 1.15, 1.20, 1.4, 1.5, 1.6])
    result = region_entry(observations, [rectangle(1, 0, 1), rectangle(2, 1, 3)], CurationRules())
    assert result["target_index"] == 6
    assert result["path_m"] == pytest.approx(1.2)
    assert result["regions"][6]["region_id"] == 2


def test_short_boundary_chatter_does_not_confirm_a_room():
    observations = frames([0.2, 1.4, 1.5, 0.7, 0.8, 1.4, 1.5])
    assert region_entry(observations, [rectangle(1, 0, 1), rectangle(2, 1, 3)], CurationRules())["target_index"] is None


def test_narrow_unconfirmed_intermediate_region_cannot_be_silently_skipped():
    observations = frames([0.5] + [1.05] * 5 + [1.5] * 5)
    result = region_entry(observations, [rectangle(1, 0, 1), rectangle(2, 1, 1.1), rectangle(3, 1.1, 3)], CurationRules())
    assert result["status"] == "uncertain_intermediate_region"


def test_model_inputs_are_a_whitelist_of_current_sensors_and_requested_skill():
    chosen = {"bearing_left_rad": 0.7, "side": "left", "target_region": "bathroom", "position": [1, 2, 3]}
    inputs = student_inputs({"path": "current.npy", "sha256": "abc"}, chosen, "fixed/v1")
    assert set(inputs) == {"history_rgb", "history_valid", "requested_skill"}
    assert set(inputs["requested_skill"]) == {"name", "policy_id", "bearing_left_rad", "side"}
    assert "bathroom" not in json.dumps(inputs)


def event(repeat, target, exit_id=0, region=2):
    return {"role": "fit", "scene": "s", "origin_id": "o", "exit_id": exit_id,
            "repeat_index": repeat, "sample_id": f"{repeat}_{target}_{exit_id}",
            "inputs": student_inputs({"sha256": "src"}, {"bearing_left_rad": 0.4, "side": "left"}, "v1"),
            "target": {"rgb": {"sha256": target}}, "offline_labels": {"target_region": {"region_id": region}}}


def test_identical_repeats_collapse_but_stochastic_targets_remain_weighted():
    result = deduplicate_events([event(0, "a"), event(1, "a"), event(2, "b")])
    assert len(result) == 2
    assert sum(r["sample_weight"] for r in result) == pytest.approx(1)
    assert sorted(r["repeat_count"] for r in result) == [1, 2]


def test_two_exits_reaching_one_region_are_not_counterfactual_negatives():
    pairs = branch_pair_summary([event(0, "a", 0), event(0, "b", 1)])
    assert pairs[0]["state"] == "same_confirmed_region"
    assert not pairs[0]["automatically_use_as_hard_negatives"]


def test_black_or_uniform_image_is_quarantined():
    assert not image_quality(np.zeros((10, 10, 3), dtype=np.uint8), CurationRules())["usable"]


def test_full_curation_does_not_touch_sealed_evidence_or_mix_old_dev(tmp_path):
    batch = tmp_path / "batch"
    evidence = batch / "visible"
    assets = evidence / "assets"
    assets.mkdir(parents=True)
    observations = frames([0.2, 0.5, 1.4, 1.5, 1.6, 1.7, 1.8])
    for index, frame in enumerate(observations):
        frame["name"] = "node" if not index else f"branch-{index - 1:03d}"
        for kind in ("rgb", "depth"):
            path = assets / f"{frame['name']}_{kind}.npy"
            pixels = np.arange(192, dtype=np.uint8).reshape(8, 8, 3) + index if kind == "rgb" else np.ones((8, 8), dtype=np.float32)
            np.save(path, pixels)
            frame[f"{kind}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (evidence / "observations.jsonl").write_text("".join(json.dumps(f) + "\n" for f in observations))
    (evidence / "commands.jsonl").write_text("".join(json.dumps({"step": i}) + "\n" for i in range(6)))
    record = {"role": "fit", "scene": "fit_house", "origin_id": "origin", "exit_id": 0, "repeat_index": 0,
              "status": "truncated_blocked", "event_complete": False, "branch_steps": 6, "policy_id": "v1",
              "evidence_dir": "visible", "pre_decision": {"rgb_sha256": observations[0]["rgb_sha256"],
              "chosen_exit": {"bearing_left_rad": 0.4, "side": "left"}}}
    sealed = {"role": "sealed_test", "scene": "sealed_house", "origin_id": "sealed_origin",
              "status": "complete", "evidence_dir": "DO_NOT_OPEN_MISSING"}
    (batch / "transitions.jsonl").write_text(json.dumps(record) + "\n" + json.dumps(sealed) + "\n")
    scenes = tmp_path / "scenes"
    house = scenes / "fit_house" / "fit_house.house"
    house.parent.mkdir(parents=True)
    lines = []
    for region, left, right in ((1, 0, 1), (2, 1, 3)):
        lines += [f"R {region} 0 0 0 r 0 0 0", f"S {region} {region} 0 F 0 0 0"]
        lines += [f"V {i} {region} F {x} {y} 0" for i, (x, y) in enumerate(((left, -1), (right, -1), (right, 1), (left, 1)))]
    house.write_text("\n".join(lines))
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "pairs.jsonl").write_text(json.dumps({"sample_id": "olddev", "scene": "old", "partition": "train_dev"}) + "\n")
    output = tmp_path / "curated"
    curate(Namespace(batch=batch, output=output, scenes=scenes, legacy=legacy))
    counts = json.loads((output / "counts.json").read_text())
    assert counts["fit"]["salvaged_from_censored_executions"] == 1
    assert counts["fit"]["deduplicated_region_events"] == 1
    assert not (output / "legacy_auxiliary_index.jsonl").read_text()
    seal = json.loads((output / "sealed_passthrough.json").read_text())
    assert seal["execution_count"] == 1 and seal["trajectory_files_opened"] == 0
    assert "sealed_house.house" not in (output / "input_manifest.json").read_text()
    assert (batch / "transitions.jsonl").read_text().count("DO_NOT_OPEN_MISSING") == 1
