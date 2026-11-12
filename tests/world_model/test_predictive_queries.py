"""Future query -> actual outcome images -> current-task choice dataflow."""

import json
import math

import numpy as np
import pytest
import torch

from experiments.world_model.event_anchored import AnchoredEventConfig, AnchoredEventWorldModel
from experiments.world_model.event_runtime import HarnessLightNavExecutor, ResidualFeaturePredictor
from experiments.world_model.predictive_queries import BranchQueryService
from experiments.world_model.scene_memory import PersistentSceneMemory
from experiments.world_model.task_planning import (
    CurrentBranch,
    EnvironmentEvidence,
    TaskBranchDecision,
    TaskGoalRouter,
    TaskPlanningContext,
)
from experiments.world_model.visual_harness import (
    ObservationImages,
    QwenBranchProposer,
    QwenTaskPlanner,
)


def setup_query(tmp_path):
    torch.manual_seed(2)
    config = AnchoredEventConfig(feature_dim=4, history_length=8, width=8, heads=2, layers=1,
                                 use_memory=True, residual_scale=0)
    model = AnchoredEventWorldModel(config)
    predictor = ResidualFeaturePredictor({"schema": "wm-residual-event/0.1", "model_kind": "anchored_memory",
                "model_config": model.configuration(), "model_state_dict": model.state_dict(),
                "feature_mean": torch.zeros(4), "feature_scale": torch.ones(4), "feature_contract": {"encoder": "test"}}).eval()
    bank = PersistentSceneMemory(tmp_path / "scene.sqlite", contract_id=predictor.feature_contract_id, feature_dim=4)
    for event, angle, target in [("red", .5, [0, 1, 0, 0]), ("blue", -.5, [0, 0, 1, 0])]:
        bank.record_observed(context="house", episode_id="previous", event_id=event,
                source_observation=f"{event}-source", target_observation=f"{event}-actual", execution_ref=f"actual-{event}",
                source_feature=[1, 0, 0, 0], observed_future=target,
                branch_direction=[math.sin(angle), math.cos(angle)], executed=True, arrived=True)
    branches = (CurrentBranch("left", "now", .5, "current left doorway", (50, 100, 400, 950)),
                CurrentBranch("right", "now", -.5, "current right doorway", (600, 100, 950, 950)))
    inputs = {"source_observation": "now", "branches": branches,
              "raw_history": torch.tensor([[[1., 0, 0, 0]]*8]), "valid": torch.ones(1, 8, dtype=torch.bool),
              "columns": torch.ones(1, 27, 4)}
    return predictor, bank, inputs


def test_predicted_future_retrieves_actual_outcome_keys_and_respects_visibility(tmp_path):
    predictor, bank, inputs = setup_query(tmp_path)
    cutoff = bank.latest_sequence("house")
    service = BranchQueryService(predictor, bank, memory_context="house", episode_id="current", visible_sequence=cutoff)
    context = service.build_context(instruction="find the red room", **inputs)
    results = dict(context.predicted_branches)
    assert results["left"].retrieved_evidence[0].target_observation == "red-actual"
    assert results["right"].retrieved_evidence[0].target_observation == "blue-actual"
    assert all(row["query_kind"] == "predicted_outcome_key" for row in service.last_trace)
    assert all(m.sequence <= cutoff for r in results.values() for m in r.retrieved_evidence)
    assert bank.count("house") == 2  # No prediction became a fact.
    bank.close()


def test_unfamiliar_state_is_unknown_not_a_copied_future_lookup(tmp_path):
    predictor, bank, inputs = setup_query(tmp_path)
    inputs["raw_history"] = torch.tensor([[[0., 0, 0, 1]]*8])
    context = BranchQueryService(predictor, bank, memory_context="house", episode_id="current",
                                visible_sequence=2).build_context(instruction="find a room", **inputs)
    assert all(not r.supported_by_history and not r.retrieved_evidence for _, r in context.predicted_branches)
    bank.close()


def test_future_key_search_is_not_limited_to_similar_current_sources(tmp_path):
    _predictor, bank, _inputs = setup_query(tmp_path)
    result = bank.query_actual_outcomes(context="house", before_sequence=1, predicted_future=[0, 1, 0, 0],
                        source_feature=[0, 0, 0, 1], branch_direction=[0, -1], exclude_episode="current")
    assert result[0].target_observation == "red-actual" and result[0].source_similarity == 0
    assert result[0].score == pytest.approx(1)
    assert not bank.query_actual_outcomes(context="house", before_sequence=2, predicted_future=[0, 1, 0, 0],
                        source_feature=[1, 0, 0, 0], branch_direction=[0, 1], exclude_episode="previous")
    bank.close()


class ColorEvidenceBackend:
    """A deterministic test oracle for plumbing, not semantic-model accuracy."""

    def __call__(self, prompt, images):
        context = json.loads(prompt.split("CONTEXT:\n", 1)[1])
        red_goal = "red" in context["current_task_instruction"]
        selected = None
        for branch in context["current_branches"]:
            evidence = branch["past_evidence"][0]
            pixel = images[evidence["actual_outcome_image_index"]][1].getpixel((0, 0))
            if (pixel[0] > pixel[2]) == red_goal:
                selected = (branch, evidence)
                break
        assert selected is not None
        branch, evidence = selected
        return json.dumps({"action": "navigate", "candidate_id": branch["candidate_id"],
                           "local_instruction": "enter the selected current doorway",
                           "evidence_sequences": [evidence["sequence"]], "reason": "test goal/outcome match"})


def test_changing_prediction_changes_queries_and_task_choice_without_changing_memory(tmp_path):
    predictor, bank, inputs = setup_query(tmp_path)
    images = ObservationImages()
    for ref, color in [("now", [10, 10, 10]), ("red-source", [30, 30, 30]), ("blue-source", [30, 30, 30]),
                       ("red-actual", [255, 0, 0]), ("blue-actual", [0, 0, 255])]:
        images.put(ref, np.broadcast_to(np.array(color, dtype=np.uint8), (32, 48, 3)))
    choices = {}
    for mode, task in [("predicted", "find red room"), ("predicted", "find blue room"), ("shuffled", "find red room")]:
        service = BranchQueryService(predictor, bank, memory_context="house", episode_id="current", visible_sequence=2, mode=mode)
        context = service.build_context(instruction=task, **inputs)
        planner = QwenTaskPlanner(ColorEvidenceBackend(), images)
        decision = planner(context)
        choices[(mode, task)] = decision.candidate_id
        assert len(planner.last_trace["consumed_prediction_candidates"]) == 2
    assert choices[("predicted", "find red room")] == "left"
    assert choices[("predicted", "find blue room")] == "right"
    assert choices[("shuffled", "find red room")] == "right"
    assert bank.count("house") == 2
    bank.close()


def test_rgb_proposer_grounds_directions_in_current_image_not_old_actions():
    images = ObservationImages()
    images.put("now", np.zeros((270, 480, 3), dtype=np.uint8))
    response = {"event": "junction", "branches": [
        {"description": "left opening", "bbox": [20, 100, 420, 950], "entry_point": [200, 850], "confidence": .8},
        {"description": "right opening", "bbox": [580, 100, 980, 950], "entry_point": [800, 850], "confidence": .8}]}
    perception = QwenBranchProposer(lambda *_: json.dumps(response), images).propose("now")
    assert len(perception.branches) == 2
    assert perception.branches[0].bearing_rad > 0 and perception.branches[1].bearing_rad < 0
    assert all(b.source_observation == "now" for b in perception.branches)


def test_visual_proposer_repairs_missing_fields_without_accepting_the_bad_response():
    images=ObservationImages()
    images.put('now',np.zeros((270,480,3),dtype=np.uint8))
    prompts=[]
    def backend(prompt,_images):
        prompts.append(prompt)
        branch={'description':'current visible doorway','bbox':[200,100,800,950],'entry_point':[500,800]}
        if len(prompts)>1:
            branch['confidence']=.8
        return json.dumps({'event':'doorway','branches':[branch]})
    perception=QwenBranchProposer(backend,images).propose('now')
    assert len(prompts)==2 and 'confidence' in prompts[1]
    assert len(perception.branches)==1 and perception.branches[0].entry_point==(500.,800.)


def test_nested_distant_entrance_is_not_an_independent_immediate_branch():
    images = ObservationImages()
    images.put("now", np.zeros((270, 480, 3), dtype=np.uint8))
    response = {"event": "invalid literal enum", "branches": [
        {"description": "immediate entrance", "bbox": [280, 0, 630, 999], "entry_point": [450, 900], "confidence": .8},
        {"description": "same opening edge", "bbox": [550, 0, 630, 999], "entry_point": [580, 900], "confidence": .6}]}
    perception = QwenBranchProposer(lambda *_: json.dumps(response), images).propose("now")
    assert len(perception.branches) == 1 and perception.event == "uncertain"


def test_planner_cannot_cite_an_unshown_past_outcome(tmp_path):
    predictor, bank, inputs = setup_query(tmp_path)
    images = ObservationImages()
    for ref in ["now", "red-source", "red-actual", "blue-source", "blue-actual"]:
        images.put(ref, np.zeros((32, 48, 3), dtype=np.uint8))
    context = BranchQueryService(predictor, bank, memory_context="house", episode_id="current", visible_sequence=2).build_context(
        instruction="find a room", **inputs)
    planner = QwenTaskPlanner(lambda *_: json.dumps({"action": "continue_global", "evidence_sequences": [999]}), images)
    with pytest.raises(ValueError, match="not shown"):
        planner(context)
    bank.close()

def test_navigation_citations_are_bound_to_the_selected_branch(tmp_path):
    predictor, bank, inputs = setup_query(tmp_path)
    images = ObservationImages()
    for ref in ["now", "red-source", "red-actual", "blue-source", "blue-actual"]:
        images.put(ref, np.zeros((32, 48, 3), dtype=np.uint8))
    context = BranchQueryService(predictor, bank, memory_context="house", episode_id="current",
                                 visible_sequence=2).build_context(instruction="find a room", **inputs)
    matches = dict(context.predicted_branches)
    left_sequence = matches["left"].retrieved_evidence[0].sequence
    right_sequence = matches["right"].retrieved_evidence[0].sequence
    prompts = []

    def backend(prompt, _images):
        prompts.append(prompt)
        sequence = right_sequence if len(prompts) == 1 else left_sequence
        return json.dumps({"action": "navigate", "candidate_id": "left",
                           "local_instruction": "enter the visible left doorway",
                           "evidence_sequences": [sequence], "reason": "test branch citation"})

    planner = QwenTaskPlanner(backend, images)
    decision = planner(context)
    assert decision.candidate_id == "left"
    assert decision.evidence_sequences == (left_sequence,)


def test_invalid_image_index_citation_gets_one_explicit_model_repair():
    images = ObservationImages()
    images.put("now", np.zeros((32, 48, 3), dtype=np.uint8))
    prompts = []

    def backend(prompt, _images):
        prompts.append(prompt)
        return json.dumps({"action": "continue_global", "evidence_sequences": [0] if len(prompts) == 1 else []})

    planner = QwenTaskPlanner(backend, images)
    context = TaskPlanningContext("ep", "find the exit", "now", 0, EnvironmentEvidence(()))
    decision = planner(context)
    assert decision.evidence_sequences == () and len(prompts) == 2
    assert "rejected by the tool validator" in prompts[1]
    assert planner.last_trace["validation_attempts"][0]["error"] == "planner cited evidence that was not shown"


def test_decision_history_uses_only_actual_prior_anchors_and_does_not_duplicate_current(tmp_path):
    from experiments.world_model.predictive_harness import PredictiveTaskHarness

    predictor, bank, _inputs = setup_query(tmp_path)
    harness = PredictiveTaskHarness(executor=None, predictor=predictor, bank=bank, memory_context="house",
        episode_id="current", images=ObservationImages(), proposer=None, planner=None)
    pixels = np.zeros((32, 48, 3), dtype=np.uint8)
    for i in range(18):
        harness.observe(f"live:{i}", pixels, np.array([i, 0, 0, 0], dtype=np.float32), torch.ones(1, 27, 4))
        if i % 8 == 0:
            harness.planning_anchors.append(i)
    history, valid = harness.visual_history()
    assert history[0, :, 0].tolist() == [0, 0, 8, 16, 5, 9, 13, 17]
    assert valid[0].tolist() == [False, True, True, True, True, True, True, True]
    harness.observe("live:17", pixels, np.array([17, 0, 0, 0], dtype=np.float32), torch.ones(1, 27, 4))
    assert len(harness.observations) == 18
    with pytest.raises(ValueError, match="backward"):
        harness.observe("live:0", pixels, np.zeros(4, dtype=np.float32), torch.ones(1, 27, 4))
    bank.close()


def test_current_task_router_validates_observation_requests_and_can_restore_global():
    class Policy:
        pass
    policy = Policy()
    policy.agent = type("Agent", (), {"instruction": "original"})()
    executor = HarnessLightNavExecutor(policy)
    executor.original_instruction = "original"
    context = TaskPlanningContext("ep", "original", "now", 0, EnvironmentEvidence(()))
    router = TaskGoalRouter(executor, "ep", lambda c: TaskBranchDecision(c.task_key, "", "", (), "observe", "look", .7))
    with pytest.raises(ValueError, match="bounded"):
        router.consider(context)
    policy.agent.instruction = "an old local instruction"
    router.planner = lambda c: TaskBranchDecision(c.task_key, "", "", (), "continue_global")
    assert router.consider(context) is None and policy.agent.instruction == "original"
