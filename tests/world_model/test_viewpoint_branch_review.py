from experiments.world_model.viewpoint_branch_review import proposed_node_passes


def test_confident_free_floor_or_mirror_is_not_an_architectural_branch():
    p = {"decision_here": True, "distinct_architectural_exits": True, "scene_state": "decision", "confidence": 0.99,
         "A": {"kind": "doorway", "immediate": True}, "B": {"kind": "open_floor", "immediate": True}}
    assert not proposed_node_passes(p)
    p["B"]["kind"] = "mirror"
    assert not proposed_node_passes(p)
    p["B"]["kind"] = "corridor"
    assert proposed_node_passes(p)
    p["distinct_architectural_exits"] = "true"
    assert not proposed_node_passes(p)
