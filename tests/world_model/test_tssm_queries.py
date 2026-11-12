import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.world_model.event_tssm import EventTSSM, TSSMConfig
from experiments.world_model.scene_memory import PersistentSceneMemory
from experiments.world_model.task_planning import CurrentBranch, TaskBranchDecision
from experiments.world_model.tssm_queries import StatefulTSSMHarness, TSSMBranchQueryService
from experiments.world_model.tssm_runtime import TSSMPredictor, TSSMSession


def test_stateful_evidence_enters_existing_task_context_without_choosing_old_route(tmp_path):
    model = EventTSSM(TSSMConfig(feature_dim=12, width=16, stochastic_dim=4, layers=1, heads=4))
    predictor = TSSMPredictor(model, torch.zeros(12), torch.ones(12)).eval()
    bank = PersistentSceneMemory(tmp_path / "facts.sqlite", contract_id=predictor.feature_contract_id, feature_dim=12)
    source, left, right = np.eye(12, dtype=np.float32)[:3]
    for event, angle, target in [("left", .5, left), ("right", -.5, right)]:
        bank.record_observed(context="house", episode_id="previous", event_id=event,
            source_observation="old-source", target_observation=event, execution_ref="executed",
            source_feature=source, observed_future=target, branch_direction=[math.sin(angle), math.cos(angle)],
            executed=True, arrived=True)
    session = TSSMSession(predictor, bank, tmp_path, base_hash="frozen")
    service = TSSMBranchQueryService(session, context="house", episode_id="current")
    branches = tuple(CurrentBranch(name, "now", angle, "visible doorway") for name, angle in [("L", .5), ("R", -.5)])
    inputs = dict(source_observation="now", branches=branches, raw_history=torch.tensor(source[None, None].repeat(8, 1)),
                  valid=torch.ones(1, 8, dtype=torch.bool))
    context = service.build_context(instruction="current goal differs from the old task", **inputs)
    result = dict(context.predicted_branches)
    assert result["L"].retrieved_evidence[0].target_observation == "left"
    assert result["R"].retrieved_evidence[0].target_observation == "right"
    assert bank.count("house") == 2 and session.state.tokens.shape[1] == 0
    service.commit("L")
    service.observe_event("L", left, target_observation="actual-left", execution_ref="commands", executed=True, arrived=True)
    assert session.state.tokens.shape[1] == 1
    with pytest.raises(ValueError, match="feedback"):
        service.build_context(instruction="new goal", **inputs)
    bank.close()


def test_harness_choice_commits_a_forecast_and_real_feedback_updates_it(tmp_path):
    model = EventTSSM(TSSMConfig(feature_dim=12, width=16, stochastic_dim=4, layers=1, heads=4))
    predictor = TSSMPredictor(model, torch.zeros(12), torch.ones(12)).eval()
    bank = PersistentSceneMemory(tmp_path / "facts.sqlite", contract_id=predictor.feature_contract_id, feature_dim=12)
    session = TSSMSession(predictor, bank, tmp_path, base_hash="frozen")
    branch = CurrentBranch("left", "now", .5, "visible left doorway")
    executor = SimpleNamespace(original_instruction="follow this current task", commit_goal=lambda *args, **kwargs: None)
    images = SimpleNamespace(put=lambda *args: None)
    proposer = SimpleNamespace(propose=lambda ref: SimpleNamespace(event="decision", raw_response="current view", branches=(branch,)))
    def planner(c):
        return TaskBranchDecision(c.task_key, "left", "enter the visible doorway")
    harness = StatefulTSSMHarness(session=session, context="house", episode_id="current", executor=executor,
                                 images=images, proposer=proposer, planner=planner)
    harness.observe("now", np.zeros((2, 2, 3)), np.eye(12, dtype=np.float32)[0], torch.ones(1, 27, 12))
    assert harness.visual_history()[1].sum() == 1
    result = harness.plan()
    assert result.goal is not None and session.active is not None and bank.count("house") == 0
    harness.report_verified_event(np.eye(12, dtype=np.float32)[1], target_observation="after",
                                   execution_ref="commands", executed=True, arrived=True)
    assert bank.count("house") == 1 and session.state.tokens.shape[1] == 1
    bank.close()


def test_ambiguous_memory_overrides_navigation_with_a_bounded_observation(tmp_path):
    from unittest.mock import Mock

    from tests.world_model.test_memory_routing import fact
    from tests.world_model.test_tssm_runtime import setup

    predictor, bank, session, history, target = setup(tmp_path)
    fact(bank, history[0, -1], target, direction=(math.sin(.5), math.cos(.5)))
    fact(bank, history[0, -1], np.eye(12, dtype=np.float32)[2], event="conflicting",
         direction=(math.sin(.5), math.cos(.5)))
    branch = CurrentBranch("left", "now", .5, "visible left doorway")
    executor = SimpleNamespace(original_instruction="current task", commit_goal=Mock())
    planner = Mock(side_effect=AssertionError("ambiguous history must trigger observation first"))
    harness = StatefulTSSMHarness(session=session, context="house", episode_id="current", executor=executor,
        images=SimpleNamespace(put=lambda *args: None), planner=planner,
        proposer=SimpleNamespace(propose=lambda ref: SimpleNamespace(event="doorway", raw_response="current", branches=(branch,))))
    harness.observe("now", np.zeros((2, 2, 3)), history[0, -1], torch.zeros(1, 27, 12))
    result = harness.plan()
    assert result.goal is None and result.action == "observe" and abs(result.observation_turn_rad) <= math.pi / 6
    planner.assert_not_called()
    executor.commit_goal.assert_not_called()
    assert session.active is None and bank.count("house") == 2
    bank.close()


def test_live_current_episode_recall_uses_only_already_observed_facts(tmp_path):
    from dataclasses import replace

    from tests.world_model.test_tssm_runtime import setup

    _, bank, session, history, target = setup(tmp_path)
    session.current_episode_memory = True
    session.begin_episode("house", "current", history, "source")
    ticket = session.forecast("executed", [0, 1])
    session.commit(ticket)
    session.observe_event(ticket, target, target_observation="arrived", execution_ref="actual_commands",
                          executed=True, arrived=True)
    service = TSSMBranchQueryService(session, context="house", episode_id="current")
    context = service.build_context(instruction="current task", source_observation="arrived",
        branches=(CurrentBranch("next", "arrived", 0., "visible exit"),), raw_history=torch.tensor(history), valid=None)
    evidence = dict(context.predicted_branches)["next"]
    assert context.allow_current_episode_history and evidence.retrieved_evidence[0].episode_id == "current"
    assert evidence.visible_sequence == 1
    with pytest.raises(ValueError, match="invisible"):
        replace(context, allow_current_episode_history=False)
    bank.record_observed(context="house", episode_id="later", event_id="future", source_observation="a",
        target_observation="b", execution_ref="actual_later_commands", source_feature=target,
        observed_future=target, branch_direction=[0, 1], executed=True, arrived=True)
    assert all(m.sequence == 1 for m in service.session.query_actual_memories(service.tickets["next"]))
    bank.close()
