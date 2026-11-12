"""Real-history feature inputs and Harness/LightNav execution contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.world_model.event_anchored import AnchoredEventConfig, AnchoredEventWorldModel
from experiments.world_model.event_runtime import (
    HarnessLightNavExecutor,
    LocalGoal,
    ResidualFeaturePredictor,
    predict_branch_with_memory,
)
from experiments.world_model.scene_memory import PersistentSceneMemory
from experiments.world_model.task_planning import (
    CurrentBranch,
    EnvironmentEvidence,
    TaskBranchDecision,
    TaskGoalRouter,
    TaskPlanningContext,
)


class FakePolicy:
    def __init__(self):
        self.agent = SimpleNamespace(instruction="", frames=[])
        self.resets = 0
        self.stop = False

    def reset(self, observation):
        self.resets += 1
        self.agent.instruction = observation["instruction"]
        self.agent.frames.clear()

    def act(self, observation, info=None):
        self.agent.frames.append(observation["rgb"])
        return {"action": "velocity_control"}

    def get_info(self):
        return {"predicted_traj": np.zeros((10, 3)) if self.stop else np.ones((10, 3))}


def test_goal_change_preserves_native_history_and_observes_once():
    policy = FakePolicy()
    executor = HarnessLightNavExecutor(policy)
    executor.reset({"instruction": "original"})
    policy.agent.frames.append("past")
    goal = LocalGoal("use the left door", "now", (1,), 0.3)
    executor.commit_goal(goal, current_observation="now", visible_sequence=1)
    assert policy.resets == 1 and policy.agent.frames == ["past"]
    assert policy.agent.instruction == "use the left door"
    assert executor.original_instruction == "original"
    action, _ = executor.propose_action({"rgb": "current"})
    assert action is not None and policy.agent.frames == ["past", "current"]
    executor.restore_global_goal()
    assert policy.agent.instruction == "original"
    assert policy.resets == 1


def test_stale_or_future_evidence_is_not_executable():
    executor = HarnessLightNavExecutor(FakePolicy())
    goal = LocalGoal("go left", "old", (2,), 0.3)
    with pytest.raises(ValueError, match="stale"):
        executor.commit_goal(goal, current_observation="new", visible_sequence=2)
    with pytest.raises(ValueError, match="future"):
        executor.commit_goal(goal, current_observation="old", visible_sequence=1)


def test_local_stop_is_a_tool_result_not_an_environment_stop():
    policy = FakePolicy()
    policy.stop = True
    executor = HarnessLightNavExecutor(policy)
    action, _ = executor.propose_action({"rgb": "now"})
    assert action is None and executor.last_tool_status == "stop_proposed"


def test_rejected_local_stop_replans_without_adding_the_frame_again():
    calls = []
    def generate(video, instruction, **kwargs):
        calls.append((instruction, kwargs["frame_ids"]))
        return "<traj_1>", 0.1
    policy = FakePolicy()
    policy.method = "flat"
    policy.dt, policy.lin_vel_range, policy.ang_vel_range = 0.1, (0., 2.5), (-300., 300.)
    policy.engine = SimpleNamespace(generate_from_frames=generate)
    policy.agent._history_frame_ids = [0]
    policy.agent._get_video_tensor = lambda: torch.zeros(1, 3, 16, 16)
    policy.agent.decode_waypoints = lambda text: (np.array([[0.1, 0, 0]], dtype=np.float32), {})
    executor = HarnessLightNavExecutor(policy)
    executor.original_instruction = "finish the original route"
    executor.restore_global_goal()
    before = list(policy.agent._history_frame_ids)
    action, details = executor.replan_observed_frame()
    assert action is not None and executor.last_tool_status == "action_proposed"
    assert policy.agent._history_frame_ids == before
    assert calls == [("finish the original route", [0])]
    assert details["action_waypoint_index"] == 0


def test_runtime_prediction_requires_fixed_normalization_and_only_visible_memory(tmp_path):
    config = AnchoredEventConfig(feature_dim=4, history_length=4, width=8, heads=2, layers=1, use_memory=True)
    model = AnchoredEventWorldModel(config)
    bundle = {"schema": "wm-residual-event/0.1", "model_kind": "anchored_memory", "model_config": model.configuration(),
              "model_state_dict": model.state_dict(), "feature_mean": torch.zeros(4), "feature_scale": torch.ones(4),
              "feature_contract": {"encoder": "test"}}
    predictor = ResidualFeaturePredictor(bundle).eval()
    memory = PersistentSceneMemory(tmp_path / "mem.sqlite", contract_id=predictor.feature_contract_id, feature_dim=4)
    def record(ep, event):
        return memory.record_observed(context="A", episode_id=ep, event_id=event,
                    source_observation="actual-before", target_observation="actual-after", execution_ref="ack",
                    source_feature=[1, 0, 0, 0], observed_future=[0, 1, 0, 0], branch_direction=[0, 1],
                    executed=True, arrived=True)
    first = record("previous", "one")
    record("current", "two")
    record("later", "three")
    args = dict(context="A", episode_id="current", visible_sequence=first,
                raw_history=torch.tensor([[[1., 0, 0, 0]] * 4]), valid=torch.ones(1, 4, dtype=torch.bool),
                branch=torch.tensor([[0., 1.]]), columns=torch.ones(1, 27, 4))
    result = predict_branch_with_memory(predictor, memory, **args)
    assert result.past_transition_sequences == (first,)
    assert result.future_feature[1] > result.future_feature[0]
    assert {m.episode_id for m in result.retrieved_evidence} == {"previous"}
    assert memory.count("A") == 3  # Predictions were not written as observations.
    memory.contract_id = "another-normalization"
    with pytest.raises(ValueError, match="contract"):
        predict_branch_with_memory(predictor, memory, **args)
    memory.close()


def planning_fixture(tmp_path):
    bank = PersistentSceneMemory(tmp_path / "environment.sqlite", contract_id="test", feature_dim=4)
    bank.begin_episode("house", "old_task", "an unrelated previous instruction")
    bank.record_observed(context="house", episode_id="old_task", event_id="real_walk",
                         source_observation="previous-view", target_observation="actual-room-view", execution_ref="ack",
                         source_feature=[1, 0, 0, 0], observed_future=[0, 1, 0, 0],
                         branch_direction=[0.5, 0.8660254], executed=True, arrived=True)
    bank.end_episode("house", "old_task")
    evidence = bank.nearest_observed_sources(context="house", before_sequence=bank.latest_sequence("house"),
                                             source_feature=[1, 0, 0, 0], exclude_episode="new", source_floor=.7)
    bank.close()
    return EnvironmentEvidence(tuple(evidence))


def test_environment_memory_alone_never_changes_the_current_goal(tmp_path):
    environment = planning_fixture(tmp_path)
    policy = FakePolicy()
    executor = HarnessLightNavExecutor(policy)
    executor.reset({"instruction": "go to the bedroom"})
    context = TaskPlanningContext("new", executor.original_instruction, "now", 1, environment)
    router = TaskGoalRouter(executor, "new")
    assert router.consider(context) is None
    assert router.last_status == "no_task_planner"
    assert executor.active_goal is None
    assert policy.agent.instruction == "go to the bedroom"
    assert policy.resets == 1


def test_same_environment_can_support_different_current_task_decisions(tmp_path):
    environment = planning_fixture(tmp_path)
    branches = (CurrentBranch("left", "now", .4, "currently visible left entrance"),
                CurrentBranch("right", "now", -.4, "currently visible right entrance"))
    # A controlled planner stub tests the routing contract, not learned semantic
    # understanding. Environmental facts remain identical between the tasks.
    choices = []
    for episode, instruction, selected in [("kitchen_task", "go to the kitchen", "left"),
                                           ("bedroom_task", "go to the bedroom", "right")]:
        policy = FakePolicy()
        executor = HarnessLightNavExecutor(policy)
        executor.reset({"instruction": instruction})
        context = TaskPlanningContext(episode, instruction, "now", 1, environment, branches)
        def planner(current, chosen=selected):
            assert current.environment is environment
            assert current.instruction == instruction
            return TaskBranchDecision(current.task_key, chosen, f"enter the {chosen} entrance", (1,))
        goal = TaskGoalRouter(executor, episode, planner).consider(context)
        choices.append(goal.bearing_rad)
        assert executor.original_instruction == instruction
        assert policy.resets == 1
    assert choices == [.4, -.4]
    assert environment.experiences[0].branch_direction[0] > 0  # Old direction did not dictate the second choice.


def test_a_decision_from_a_previous_task_cannot_control_the_new_episode(tmp_path):
    environment = planning_fixture(tmp_path)
    branches = (CurrentBranch("left", "now", .4, "visible entrance"),)
    old = TaskPlanningContext("old_request", "go to the kitchen", "now", 1, environment, branches)
    cached = TaskBranchDecision(old.task_key, "left", "go left", (1,))
    policy = FakePolicy()
    executor = HarnessLightNavExecutor(policy)
    executor.reset({"instruction": "go to the bedroom"})
    current = TaskPlanningContext("new_request", executor.original_instruction, "now", 1, environment, branches)
    router = TaskGoalRouter(executor, "new_request", lambda _: cached)
    with pytest.raises(ValueError, match="another episode or task"):
        router.consider(current)
    assert policy.agent.instruction == "go to the bedroom"


def test_past_direction_is_not_a_current_visible_exit_candidate(tmp_path):
    environment = planning_fixture(tmp_path)
    policy = FakePolicy()
    executor = HarnessLightNavExecutor(policy)
    executor.reset({"instruction": "go to the bedroom"})
    context = TaskPlanningContext("new", executor.original_instruction, "now", 1, environment)
    router = TaskGoalRouter(executor, "new",
                           lambda c: TaskBranchDecision(c.task_key, "remembered_left", "go left", (1,)))
    with pytest.raises(ValueError, match="grounded current branch"):
        router.consider(context)
    assert policy.agent.instruction == executor.original_instruction


def test_a_new_goal_can_choose_an_unvisited_current_branch_without_history(tmp_path):
    policy = FakePolicy()
    executor = HarnessLightNavExecutor(policy)
    executor.reset({"instruction": "explore the visible doorway"})
    branches = (CurrentBranch("new_door", "now", -.2, "currently visible unexplored doorway"),)
    context = TaskPlanningContext("new", executor.original_instruction, "now", 0, EnvironmentEvidence(()), branches)
    router = TaskGoalRouter(executor, "new",
                           lambda c: TaskBranchDecision(c.task_key, "new_door", "enter the visible doorway"))
    goal = router.consider(context)
    assert goal.bearing_rad == -.2 and goal.evidence_sequences == ()
