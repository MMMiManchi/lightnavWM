from experiments.world_model.branch_node_review import (
    consensus_label,
    supported_exits,
    two_distinct_exits,
)


def exit_(left, right, point):
    return {"kind": "doorway", "clear_architecture": True,
            "bbox": [left, 200, right, 800], "floor_entry": [point, 700]}


def test_duplicate_or_nested_doorways_cannot_be_two_exits():
    assert not two_distinct_exits([exit_(50, 900, 100), exit_(200, 700, 600)])


def test_close_distant_doors_do_not_mark_the_current_view_as_a_decision():
    assert not two_distinct_exits([exit_(400, 480, 440), exit_(500, 600, 550)])
    assert two_distinct_exits([exit_(50, 350, 200), exit_(650, 950, 800)])


def test_windows_missing_geometry_and_off_image_points_are_rejected():
    invalid = [dict(exit_(50, 300, 100), kind="window"),
               dict(exit_(50, 300, 100), clear_architecture=False), exit_(50, 300, 900)]
    assert not supported_exits({"exits": invalid})


def test_confidence_or_teacher_agreement_alone_is_insufficient():
    first = {"label": "decision", "confidence": 0.99}
    assert consensus_label(first, dict(first, exits=[]))[0] == "unknown"
    assert consensus_label(first, {"label": "ordinary", "confidence": 0.99, "exits": []})[0] == "unknown"
    assert consensus_label(first, dict(first, exits=[exit_(50, 350, 200), exit_(650, 950, 800)]))[0] == "decision"
