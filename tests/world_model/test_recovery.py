import math

from experiments.world_model.recovery import RecoveryManager


def test_unknown_global_stop_can_defer_without_claiming_semantic_completion():
    r = RecoveryManager(observation_budget=8)
    assert r.on_global_stop(verdict="unknown").action == "observe"
    assert r.on_global_stop(verdict="unknown").action == "observe"
    assert r.on_global_stop(verdict="unknown").action == "defer_stop"


def test_explicit_negative_evidence_cannot_be_overridden_by_unknown_stop_deference():
    r = RecoveryManager(observation_budget=8)
    r.on_global_stop(verdict="not_reached")
    decisions = [r.on_global_stop(verdict="unknown", can_backtrack=True) for _ in range(6)]
    assert all(d.action != "defer_stop" for d in decisions)
    assert any(d.action in {"relocalize", "backtrack"} for d in decisions)


def test_observation_budget_resets_on_measured_progress_not_camera_changes():
    r = RecoveryManager(observation_budget=2)
    r.observe_progress({"local_pose": [0, 0, 0], "pose_epoch": 0}, 0)
    r.observation(reason="uncertain")
    r.observation(reason="uncertain")
    r.observe_progress({"local_pose": [0, 0, 1.0], "pose_epoch": 0}, 0)
    assert r.phase_observations == 2
    r.observe_progress({"local_pose": [0, .8, 1.0], "pose_epoch": 0}, 0)
    assert r.phase_observations == 0 and r.total_observations == 2


def test_repeated_scan_target_requests_a_distinct_nonzero_observation():
    r = RecoveryManager(observation_budget=8)
    r.phase_observations = 3
    r.scan_offset = -math.radians(60)
    decision = r.observation(reason="repeat target")
    assert abs(decision.angle) >= math.radians(15)
    assert not math.isclose(r.scan_offset, -math.radians(60))
