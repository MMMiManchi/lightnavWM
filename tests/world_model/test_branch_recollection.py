import math

import pytest

from experiments.world_model.branch_curation import FloorPolygon
from experiments.world_model.branch_recollection import (
    RecollectionRules,
    causal_history,
    checkpoint_command,
    confirmed_event,
    exit_checkpoint,
    restored,
    validate_origin,
)


def origin():
    return {"role": "fit", "node_label_quality": "reviewed_visual_and_local_geometry_silver",
            "position": [0, 0, 0], "yaw": 0, "expected_rgb_sha256": "same",
            "exits": [{"exit_id": 0, "bearing_left_rad": 0.5, "local_navmesh_1_5m_walked": 1.5},
                      {"exit_id": 1, "bearing_left_rad": -0.5, "local_navmesh_1_5m_walked": 1.5}]}


def test_camera_left_and_habitat_yaw_agree():
    assert exit_checkpoint([0, 0, 0], 0, 0) == pytest.approx([0, 0, -1.5])
    assert exit_checkpoint([0, 0, 0], 0, math.pi / 2) == pytest.approx([-1.5, 0, 0])
    assert exit_checkpoint([0, 0, 0], math.pi / 2, 0) == pytest.approx([-1.5, 0, 0])
    turn = checkpoint_command([0, 0, 0], 0, [-1, 0, 0])
    assert turn["action_args"]["angular_velocity"] > 0


def test_depth_proxy_or_a_prior_review_does_not_admit_sealed_data():
    validate_origin(origin(), RecollectionRules())
    with pytest.raises(ValueError):
        validate_origin(dict(origin(), role="sealed_test"), RecollectionRules())
    with pytest.raises(ValueError):
        validate_origin(dict(origin(), node_label_quality="depth_openings"), RecollectionRules())


def test_matching_coordinates_do_not_override_a_different_source_image():
    assert restored(origin(), [0, 0, 0], 0, "same", RecollectionRules())
    assert not restored(origin(), [0, 0, 0], 0, "different", RecollectionRules())


def test_same_room_motion_and_depth_gaps_never_accept_a_next_event():
    polygon = FloorPolygon(1, "h", 0, ((-5, -5), (5, -5), (5, 5), (-5, 5)))
    frames = [{"position": [i / 10, 0.1, 0], "opening_count": 5} for i in range(30)]
    assert confirmed_event(frames, [polygon], True) is None


def test_a_real_region_entry_still_requires_the_requested_entrance():
    polygons = [FloorPolygon(1, "h", 0, ((0, -1), (1, -1), (1, 1), (0, 1))),
                FloorPolygon(2, "k", 0, ((1, -1), (3, -1), (3, 1), (1, 1)))]
    frames = [{"position": [x, 0.1, 0]} for x in [0.2, 0.5, 1.3, 1.4, 1.5, 1.6, 1.7]]
    assert confirmed_event(frames, polygons, False) is None
    assert confirmed_event(frames, polygons, True)["target_index"] == 6


def test_history_is_actual_past_and_ends_before_the_branch_action():
    frames = [{"name": "history", "replay_step": i, "rgb_path": f"{i}.npy", "rgb_sha256": str(i)} for i in range(9)]
    frames[-1]["name"] = "node"
    history = causal_history(frames)
    assert len(history) == 8 and history[0]["sha256"] == "1" and history[-1]["sha256"] == "8"
    with pytest.raises(ValueError):
        causal_history([frames[-1], frames[0]])
