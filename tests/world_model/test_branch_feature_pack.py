import numpy as np
import pytest

from experiments.world_model.branch_feature_pack import pack_events, place_labels


def event(history):
    return {"sample_id": "x", "inputs": {"history_rgb": [{"sha256": str(i)} for i in history],
        "history_valid": [True] * len(history), "requested_skill": {"bearing_left_rad": 0}},
        "target": {"rgb": {"sha256": "8"}}, "role": "fit", "sample_weight": 1}


def test_all_eight_actual_frames_survive_packing():
    vectors = np.arange(27, dtype=np.float32).reshape(9, 3)
    packed = pack_events([event(range(8))], vectors, {str(i): i for i in range(9)})
    np.testing.assert_array_equal(packed["history"][0], vectors[:8])
    assert packed["valid"].all()
    np.testing.assert_array_equal(packed["target"][0], vectors[8])


def test_short_history_is_padded_without_inventing_observations():
    vectors = np.arange(27, dtype=np.float32).reshape(9, 3)
    packed = pack_events([event([1, 2])], vectors, {str(i): i for i in range(9)})
    assert packed["valid"].tolist() == [[False] * 6 + [True, True]]
    with pytest.raises(ValueError):
        pack_events([event(range(9))], vectors, {str(i): i for i in range(9)})


def test_place_positives_do_not_merge_houses_or_distinct_hallway_decisions():
    a = {"scene": "A", "target_kind": "next_room", "offline_labels": {"target_region": {"region_id": 1}}}
    b = {**a, "scene": "B"}
    c = {**a, "target_kind": "next_reviewed_decision", "offline_labels": {"landmark_origin_id": "node1"}}
    d = {**c, "offline_labels": {"landmark_origin_id": "node2"}}
    labels, _ = place_labels([a, dict(a), b, c, d])
    assert labels[0] == labels[1]
    assert len(set(labels.tolist())) == 4
