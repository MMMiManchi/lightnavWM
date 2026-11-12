"""CPU checks for the previous-command rejection rule. They do not read frames."""

from __future__ import annotations

import pytest

from experiments.world_model.rejected_command import previous_command_rejected, rejection_verdict


def test_only_an_unexecuted_motion_command_counts_as_rejected():
    assert previous_command_rejected(0.15, 0.0, 0.0, 0.0)
    assert previous_command_rejected(0.0, 0.10, 0.0, 0.0)
    assert not previous_command_rejected(0.0, 0.0, 0.0, 0.0)
    assert not previous_command_rejected(0.15, 0.0, 0.15, 0.0)
    assert not previous_command_rejected(0.15, 0.0, 0.0, 0.10)
    assert not previous_command_rejected(0.05, 0.0, 0.0, 0.0)


def test_verdict_requires_the_stuck_scene_and_the_other_scenes():
    passed = dict(
        stuck_warp=16.0,
        stuck_rule=3.0,
        stuck_copy=3.0,
        other_warp=5.0,
        other_rule=5.0,
        all_copy=19.0,
        all_yaw=17.0,
        all_rule=6.0,
    )
    assert rejection_verdict(**passed)["call"] == "rejected_command_explains_the_stuck_scene"
    harmed = dict(passed)
    harmed["other_rule"] = 5.2
    assert rejection_verdict(**harmed)["call"] == "rule_hurts_executed_scenes"
    short = dict(passed)
    short["stuck_rule"] = 15.0
    assert rejection_verdict(**short)["call"] == "rejected_command_does_not_explain"
    worse_than_copy = dict(passed)
    worse_than_copy["stuck_rule"] = 4.0
    assert rejection_verdict(**worse_than_copy)["call"] == "rejected_command_does_not_explain"
    assert rejection_verdict(**passed)["training_started"] is False
    assert rejection_verdict(**passed)["uses_current_future_pose"] is False


def test_exact_boundary_does_not_count_as_a_clear_improvement():
    borderline = dict(
        stuck_warp=10.0,
        stuck_rule=9.0,
        stuck_copy=3.0,
        other_warp=5.0,
        other_rule=5.0,
        all_copy=19.0,
        all_yaw=17.0,
        all_rule=6.0,
    )
    assert rejection_verdict(**borderline)["call"] == "rejected_command_does_not_explain"
    with pytest.raises(ValueError):
        previous_command_rejected(float("nan"), 0.0, 0.0, 0.0)
