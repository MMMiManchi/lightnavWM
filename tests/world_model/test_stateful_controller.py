import json
from types import SimpleNamespace

import numpy as np
import torch

from experiments.world_model.live_images import PersistentObservationImages
from experiments.world_model.stateful_controller import StatefulEpisodeController
from experiments.world_model.stop_guard import is_implicit_stop
from experiments.world_model.task_planning import CurrentBranch, TaskBranchDecision
from experiments.world_model.task_progress import StageEvidence, TaskStage
from tests.world_model.test_tssm_runtime import setup

MOVE = {"action": "velocity_control", "action_args": {"linear_velocity": 1.0, "angular_velocity": 0.0}}


class Policy:
    dt, lin_vel_range, ang_vel_range = 0.1, (0.0, 1.0), (-180.0, 180.0)

    def __init__(self):
        self.observed = 0

    def observe(self, *_):
        self.observed += 1

    def stop_action(self):
        return {"action": "stop"}


class Executor:
    original_instruction = "stop beside the blue table"

    def __init__(self):
        self.policy, self.active_goal = Policy(), None
        self.last_tool_status = "idle"
        self.proposal, self.replanned, self.replans = None, MOVE, 0

    def commit_goal(self, goal, **_):
        self.active_goal = goal

    def restore_global_goal(self):
        self.active_goal = None

    def propose_action(self, obs, info):
        self.policy.observe(obs, info)
        self.last_tool_status = "stop_proposed" if self.proposal is None else "action_proposed"
        return self.proposal, {}

    def replan_observed_frame(self):
        self.replans += 1
        self.last_tool_status = "stop_proposed" if self.replanned is None else "action_proposed"
        return self.replanned, {}


class Verifier:
    def __init__(self, positive=False, event=False):
        self.positive = positive
        self.last_trace = {"event_kind": "next_room"} if event else {}

    def __call__(self, tracker, *args):
        return StageEvidence(tracker.active.stage_id, tracker.anchor, args[-1], self.positive,
                             self.positive, 0.9, "actual visual transition" if self.positive else "target unconfirmed")


class SyntheticCrossing:
    """Isolate controller writeback tests from the separately tested RGB-D gate."""
    def __init__(self, *_a, **_k):
        pass

    def observe(self, _feedback):
        pass

    def eligible(self, _step):
        return True

    def snapshot(self):
        return {"test_fixture_crossing": True, "motion_chain_valid": True}


def test_uncertain_native_stop_is_latched_through_inspection_and_return(tmp_path):
    c, bank, target = make_controller(tmp_path, navigate=False, policy_stop_roundtrip=True, unknown_stop_policy="defer")
    c.executor.replanned = None
    observation = {"rgb": np.asarray(c.images.get(c.initial))}
    action, trace = c.next_action(0, observation, {})
    assert trace["policy_stop_latched"] and not trace.get("stop_dispatched")
    assert action["action_args"]["linear_velocity"] == -1  # raw -1 maps to zero m/s
    c.executed(0, action, rgb_changed=True)
    c.observe(1, observation["rgb"], target, torch.zeros(1, 27, 12))
    c.looking["remaining"] = 0
    action, trace = c.next_action(1, observation, {})
    assert action["action_args"]["linear_velocity"] == -1
    assert action["action_args"]["angular_velocity"] < 0
    c.executed(1, action, rgb_changed=True)
    c.observe(2, observation["rgb"], target, torch.zeros(1, 27, 12))
    c.looking["remaining"] = 0
    action, trace = c.next_action(2, observation, {})
    assert action == {"action": "stop"} and trace["stop_dispatched"]
    assert trace["policy_STOP_deferred"] and trace["policy_stop_latched"]
    assert c.executor.replans == 0 and bank.count("house") == 0
    bank.close()


def make_controller(tmp_path, *, navigate=True, event=False, goal=False, budget=2, decomposition_reply=None, **options):
    _p, bank, session, history, target = setup(tmp_path, learning=True)
    images, executor = PersistentObservationImages(tmp_path / "images"), Executor()

    def vision(*_args, **_kwargs):
        return json.dumps(decomposition_reply or {"stages": [{"instruction": executor.original_instruction, "kind": "goal"}]})

    proposer = SimpleNamespace(propose=lambda ref: SimpleNamespace(event="doorway", raw_response="actual visible door",
        branches=(CurrentBranch("door", ref, 0.1, "visible doorway"),)))

    def planner(context):
        return TaskBranchDecision(context.task_key, "door" if navigate else "", "cross this visible doorway" if navigate else "",
                                  action="navigate" if navigate else "continue_global")

    settings = dict(exploration_overrides=True, unknown_stop_policy="abstain", maximum_recovery_cycles=0,
                    require_measured_progress=False)
    settings.update(options)
    c = StatefulEpisodeController(session=session, executor=executor, vision=vision, images=images,
        context="house", episode_id="live-1", output=tmp_path / "episode", initial_rgb=np.zeros((12, 12, 3), dtype=np.uint8),
        planning_interval=50, verification_interval=50, observation_budget=budget,
        proposer=proposer, planner=planner, progress_verifier=Verifier(goal), event_verifier=Verifier(event, event=True), **settings)
    c.observe(0, np.zeros((12, 12, 3), dtype=np.uint8), history[0, -1], torch.zeros(1, 27, 12))
    return c, bank, target


def advance(c, step, feature, action=MOVE):
    c.executed(step, action, rgb_changed=True)
    c.observe(step + 1, np.full((12, 12, 3), step + 1, dtype=np.uint8), feature, torch.zeros(1, 27, 12))


def test_unverified_local_stop_restores_current_task_without_writing_an_arrival(tmp_path):
    c, bank, target = make_controller(tmp_path)
    c.plan(0)
    for step in range(3):
        advance(c, step, target)
    before = c.executor.policy.observed
    action, trace = c.next_action(3, {"rgb": np.zeros((12, 12, 3), dtype=np.uint8)}, {"distance_to_goal": 0.01})
    assert action == MOVE and trace["local_stop_intercepted"]
    assert c.executor.active_goal is None and c.pending is None
    assert c.executor.replans == 1 and c.executor.policy.observed == before + 1
    assert bank.count("house") == 0 and c.session.updates.version == 0
    assert not c.progress.accept_global_stop()
    c.finish("test_end")
    bank.close()


def test_global_stop_requires_goal_evidence_and_budget_exhaustion_is_not_success(tmp_path):
    c, bank, target = make_controller(tmp_path, navigate=False, budget=1)
    c.executor.replanned = None
    obs = {"rgb": np.zeros((12, 12, 3), dtype=np.uint8)}
    action, first = c.next_action(0, obs, {"success": True, "distance_to_goal": 0.01})
    assert first["global_stop_intercepted"] and action["action"] != "stop"
    advance(c, 0, target, action)
    for step in range(1, 10):
        action, second = c.next_action(step, obs, {"success": True})
        if action is None:
            break
        assert action["action"] != "stop"
        advance(c, step, target, action)
    assert action is None and second["end_reason"] == "unverified_STOP_no_supported_continuation"
    assert not second.get("stop_dispatched") and not c.progress.complete and bank.count("house") == 0
    c.finish(second["end_reason"])
    bank.close()


def test_unmarked_zero_velocity_cannot_bypass_the_global_stop_gate(tmp_path):
    c, bank, _target = make_controller(tmp_path, navigate=False)
    zero = {"action": "velocity_control", "action_args": {"linear_velocity": -1., "angular_velocity": 0.}}
    action, trace = c._result(zero, {"step": 0, "action_source": "LightNav_current_task"})
    assert trace['implicit_stop_intercepted'] and trace['action_source'] == 'Harness_active_observation'
    assert not trace.get('stop_dispatched')
    assert not is_implicit_stop(action, linear_range=c.executor.policy.lin_vel_range,
                                angular_range=c.executor.policy.ang_vel_range, thresholds=c.stop_thresholds)
    assert c.looking and c.recovery.total_observations == 1
    c.finish('test_end')
    bank.close()


def test_zero_velocity_with_no_observation_budget_abstains_instead_of_stopping(tmp_path):
    c, bank, _target = make_controller(tmp_path, navigate=False, budget=0)
    zero = {"action": "velocity_control", "action_args": {"linear_velocity": -1., "angular_velocity": 0.}}
    action, trace = c._result(zero, {"step": 0, "action_source": "LightNav_current_task"})
    assert action is None and trace['implicit_stop_intercepted']
    assert trace['end_reason'] == 'implicit_stop_blocked_without_observation_budget'
    assert not trace.get('stop_dispatched') and bank.count('house') == 0
    c.finish('test_end')
    bank.close()


def test_actual_next_event_needs_two_views_then_writes_memory_and_parameters(tmp_path):
    c, bank, target = make_controller(tmp_path, event=True, exit_guard_factory=SyntheticCrossing,
                                     require_measured_progress=True)
    observe = c.feedback.observe
    def measured_observe(step, reference, rgb, depth, receipt=None):
        record = observe(step, reference, rgb, depth, receipt)
        record["measurement"] = {"valid": True, "translation_m": .15, "yaw_left_rad": 0.}
        record["local_pose"] = [0., .15*step, 0.]
        return record
    c.feedback.observe = measured_observe
    c.plan(0)
    for step in range(3):
        advance(c, step, target)
    assert not c.check_event(3) and bank.count("house") == 0
    for step in range(3, 6):
        advance(c, step, target)
    assert c.check_event(6) and bank.count("house") == 1
    assert c.session.updates.version == 1 and c.pending is None and c.executor.active_goal is None
    assert not c.progress.complete  # next-room completion is not the blue-table task goal
    c.plan(6)
    assert c.pending["tracker"].observations[c.current_reference] == 6
    c.finish("test_end")
    saved = json.loads((c.output / "events.json").read_text())[-1]
    assert saved["event_supported"] and len(saved["supporting_views"]) == 2
    assert saved["measured_motion_support"]["source"] == "actual_rgbd_camera_motion_not_issued_controls"
    bank.close()


def test_verified_global_stop_is_dispatched_without_a_policy_stop_proposal(tmp_path):
    c, bank, target = make_controller(tmp_path, navigate=False, goal=True)
    for step in range(3):
        advance(c, step, target)
    c.check_progress(3)
    assert not c.progress.complete
    for step in range(3, 6):
        advance(c, step, target)
    c.check_progress(6)
    assert c.progress.complete
    action, trace = c.next_action(6, {"rgb": np.zeros((12, 12, 3), dtype=np.uint8)}, {})
    assert action == {"action": "stop"} and trace["stop_dispatched"]
    assert bank.count("house") == 0
    c.finish("stop_dispatched")
    saved = json.loads((c.output / "progress.json").read_text())
    accepted = next(row["accepted_stage"] for row in saved if row.get("advanced"))
    assert len(accepted["supporting_views"]) == 2
    bank.close()


def test_missing_rgbd_motion_cannot_mark_a_visually_supported_goal_complete(tmp_path):
    c, bank, target = make_controller(tmp_path, navigate=False, goal=True, require_measured_progress=True)
    for step in range(6):
        advance(c, step, target)
        if step in [2, 5]:
            c.check_progress(step+1)
    assert not c.progress.complete and not c.progress.accept_global_stop()
    assert c.progress.last_status == "measured_motion_unavailable_need_new_anchor"
    assert bank.count("house") == 0
    c.finish("test_end")
    bank.close()


def test_invalid_task_decomposition_cannot_certify_the_full_instruction(tmp_path):
    invalid = {"stages": [{"instruction": "invented destination", "kind": "goal"}]}
    c, bank, target = make_controller(tmp_path, navigate=False, goal=True, decomposition_reply=invalid)
    assert c.decomposition["error"] is not None
    assert c.progress.instruction == c.executor.original_instruction
    for step in range(6):
        advance(c, step, target)
        if step in [2, 5]:
            c.check_progress(step+1)
    assert not c.progress.complete and not c.progress.accept_global_stop()
    assert c.progress_trace[-1]["status"] == "task_decomposition_unverified"
    assert bank.count("house") == 0
    c.finish("test_end")
    bank.close()


def test_original_policy_stop_can_be_deferred_without_writing_a_verified_arrival(tmp_path):
    c, bank, target = make_controller(tmp_path, navigate=False, budget=8,
                                     unknown_stop_policy="defer", maximum_recovery_cycles=3)
    c.executor.replanned = None
    obs = {"rgb": np.zeros((12, 12, 3), dtype=np.uint8)}
    for step in range(20):
        action, trace = c.next_action(step, obs, {"distance_to_goal": 0.01, "success": True})
        if trace.get("policy_STOP_deferred"):
            break
        assert action["action"] != "stop"
        advance(c, step, target, action)
    assert action == {"action": "stop"} and trace["policy_STOP_deferred"]
    assert not trace["semantic_arrival_confirmed"] and not c.progress.complete
    assert bank.count("house") == 0 and c.session.updates.version == 0
    c.finish("unverified_policy_stop")
    bank.close()


def test_pending_intermediate_stage_gets_a_second_view_before_unknown_stop_fallback(tmp_path):
    c, bank, target = make_controller(tmp_path, navigate=False, goal=True, budget=8,
                                     unknown_stop_policy="defer", maximum_recovery_cycles=3)
    c.progress.stages = (TaskStage("route", "pass the landmark", "actually pass the landmark", "navigate"),
                         *c.progress.stages)
    c.recovery.unknown_stops = c.recovery.phase_observations = 2
    for step in range(3):
        advance(c, step, target)
    c.check_progress(3)
    assert c.progress.confirmation is not None and c.progress.active_index == 0
    obs = {"rgb": np.zeros((12, 12, 3), dtype=np.uint8)}
    action, trace = c.next_action(3, obs, {})
    assert action["action"] != "stop" and trace["recovery_action"] == "await_confirmation"
    assert not trace.get("policy_STOP_deferred", False)
    for step in range(3, 12):
        advance(c, step, target, action)
        action, _trace = c.next_action(step+1, obs, {})
        if c.progress.active_index:
            assert action is not None and action["action"] != "stop"
            assert not _trace.get("policy_STOP_deferred", False)
            break
        assert action is not None and action["action"] != "stop"
    assert c.progress.active_index == 1 and not c.progress.complete
    assert bank.count("house") == 0
    c.finish("test_end")
    bank.close()


def test_zero_observation_budget_can_still_defer_an_unverified_original_stop(tmp_path):
    c, bank, target = make_controller(tmp_path, navigate=False, goal=True, budget=0,
                                     unknown_stop_policy="defer", maximum_recovery_cycles=0)
    c.progress.stages = (TaskStage("route", "pass the landmark", "actually pass the landmark", "navigate"),
                         *c.progress.stages)
    for step in range(3):
        advance(c, step, target)
    c.check_progress(3)
    c.recovery.unknown_stops = 2
    action, trace = c.next_action(3, {"rgb": np.zeros((12, 12, 3), dtype=np.uint8)}, {})
    assert action == {"action": "stop"} and trace["policy_STOP_deferred"]
    assert not c.progress.complete and bank.count("house") == 0
    c.finish("test_end")
    bank.close()


def test_passive_option_monitors_original_policy_and_only_actual_event_updates_memory(tmp_path):
    c, bank, target = make_controller(tmp_path, event=True, exploration_overrides=False,
                                      exit_guard_factory=SyntheticCrossing)
    c.executor.proposal = MOVE
    result = c.plan(0)
    assert result.goal is None and c.pending["command"].mode == "monitor"
    assert c.executor.active_goal is None
    for step in range(6):
        action, trace = c.next_action(step, {"rgb": np.zeros((12, 12, 3), dtype=np.uint8)}, {})
        assert action == MOVE and trace["action_source"] == "LightNav_current_task"
        advance(c, step, target, action)
        if step == 2:
            assert not c.check_event(3)
    assert c.check_event(6) and bank.count("house") == 1
    assert c.session.updates.version == 1 and not c.progress.complete
    c.finish("test_end")
    bank.close()


def test_late_execution_receipt_is_not_assigned_to_a_new_option(tmp_path):
    c, bank, _target = make_controller(tmp_path)
    c.plan(0)
    first = c.pending["command"].command_id
    c.abandon("replanned")
    c.plan(0)
    assert c.pending["command"].command_id != first
    c.executed(0, MOVE, rgb_changed=True, command_id=first)
    assert c.pending is None and not c.commands[-1]["matched_current_command"]
    assert c.commands[-1]["command_id"] == first and bank.count("house") == 0
    c.finish("test_end")
    bank.close()


def test_confident_visual_event_without_measured_selected_exit_crossing_cannot_write(tmp_path):
    c, bank, target = make_controller(tmp_path, event=True, exploration_overrides=False)
    c.plan(0)
    for step in range(6):
        advance(c, step, target)
        c.check_event(step+1)
    assert bank.count("house") == 0 and c.session.updates.version == 0
    assert not any(r["event_supported"] for r in c.event_trace)
    assert "source_depth_or_aperture_box_unavailable" in c.event_trace[-1]["exit_crossing_guard"]["reason"]
    c.finish("test_end")
    bank.close()


def test_recovery_returns_toward_actual_local_anchor_and_aborts_when_odometry_frame_changes(tmp_path):
    c, bank, _target = make_controller(tmp_path, navigate=False)
    c.feedback.pose_valid = True
    c.feedback.pose[:] = [0., .6, 0.]
    c.feedback.anchors.append({"reference": "actually_seen_past_view", "xy": np.array([0., .2]), "epoch": 0})
    c.feedback.records[-1]["clearance_m"] = 1.
    obs, trace = {"rgb": np.zeros((12, 12, 3), dtype=np.uint8)}, {}
    assert c._start_backtrack()
    turn = c._backtrack(obs, {}, trace)
    assert turn["action_args"]["linear_velocity"] == -1.  # turn before moving toward an anchor behind us
    assert abs(turn["action_args"]["angular_velocity"]) > 0
    c.feedback.pose[2] = np.pi
    forward = c._backtrack(obs, {}, {})
    assert abs(forward["action_args"]["angular_velocity"]) < 1e-6
    assert abs((forward["action_args"]["linear_velocity"]+1)/2-.15) < 1e-6
    c.feedback.epoch += 1
    assert c._backtrack(obs, {}, {}) is None and c.backtracking is None
    assert bank.count("house") == 0 and not c.progress.complete
    c.finish("test_end")
    bank.close()


def test_unknown_passive_exit_is_refreshed_from_new_view_without_fabricating_an_event(tmp_path):
    c, bank, target = make_controller(tmp_path, exploration_overrides=False)
    c.planning_interval = 6
    c.executor.proposal = MOVE
    c.plan(0)
    old = c.pending["command"].command_id
    for step in range(6):
        advance(c, step, target)
    action, _trace = c.next_action(6, {"rgb": np.full((12, 12, 3), 6, dtype=np.uint8)}, {})
    assert action == MOVE and c.pending["command"].mode == "monitor"
    assert c.pending["command"].command_id != old
    assert c.pending["source"] == c.current_reference
    assert c.abandoned[-1]["reason"] == "unmeasurable_exit_hypothesis_refresh_at_new_view"
    assert bank.count("house") == 0 and c.session.updates.version == 0
    c.finish("test_end")
    bank.close()
