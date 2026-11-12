"""Causal input, direction matching, and prior/posterior regression checks."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.world_model.event_dataset import construct_event_pairs, select_direction
from experiments.world_model.event_guards import validate_pair_references
from experiments.world_model.event_inference import EventFeaturePredictor, load_event_predictor
from experiments.world_model.event_latent import (
    EventLatentWorldModel,
    EventModelConfig,
    event_objective,
    multi_positive_contrastive,
)


def place():
    return {"place_id":"A", "exits":[
        {"exit_id":"front", "toward_position":[0.,0.,-3.]},
        {"exit_id":"left", "toward_position":[-3.,0.,0.]},
    ]}


def frame(index, x=0., z=0., *, uid="ep", region="A", phase="inside"):
    return {"split":"formal_fit_example", "episode_uid":uid, "index":index,
            "scene":"house", "position":[x,0.,z], "yaw":0., "has_rgb":True,
            "phase":phase, "place_id":region, "in_decision_region":phase in ("enter","inside","started_inside"),
            "source_observation_id":f"{uid}-{index}", "dev_excluded":False,
            "angle_candidate_exit_ids":["front"],
            "depth_exit_status":[{"exit_id":"front","status":"not_blocked"}]}


def test_branch_is_one_of_the_current_candidates_not_a_future_vector():
    origin = frame(0)
    selected = select_direction(origin,frame(3,x=-.1,z=-1.),place())
    assert selected["exit_id"] == "front"
    assert selected["direction"] == pytest.approx([0.,1.])
    assert "position" not in selected
    assert select_direction(origin,frame(3,x=-1.,z=0.),place()) is None


def test_a_blocked_or_unobserved_candidate_is_not_training_condition():
    origin = frame(0)
    origin["depth_exit_status"][0]["status"] = "blocked"
    assert select_direction(origin,frame(3,z=-1.),place()) is None
    origin["depth_exit_status"][0]["status"] = "not_blocked"
    origin["angle_candidate_exit_ids"] = []
    assert select_direction(origin,frame(3,z=-1.),place()) is None


def test_temporal_inputs_end_at_origin_and_do_not_cross_episodes():
    sequence = [frame(i,z=-i*.2,phase="started_inside" if i==0 else "inside") for i in range(8)]
    sequence.append(frame(8,z=-1.6,region="B",phase="enter"))
    sequence.append(frame(0,uid="other",region="B",phase="enter"))
    pairs, _ = construct_event_pairs(sequence,[place()])
    assert len(pairs) == 1
    p = pairs[0]
    assert p["origin_index"] == 0 and p["target_index"] == 8
    assert max(p["history_indices"]) == 0
    assert p["target_observation_id"] not in p["history_observation_ids"]
    assert all(oid is None or oid.startswith("ep-") for oid in p["history_observation_ids"])


def test_missing_steps_are_rejected():
    with pytest.raises(ValueError,match="missing or duplicate"):
        construct_event_pairs([frame(0),frame(2)],[place()])


def test_history_observation_id_cannot_point_to_future_even_if_index_claims_past():
    p = {"origin_index":0,"target_index":8,"episode_uid":"ep",
         "history_indices":[-1]*4+[0]*4,"history_observation_ids":[None]*4+["now"]*4,
         "history_image_ids":[None]*4+["img0"]*4,"target_observation_id":"future","target_image_id":"img8"}
    observations = {"now":{"episode_uid":"ep","step_index":0,"rgb":{"sha256":"img0"}},
                    "future":{"episode_uid":"ep","step_index":8,"rgb":{"sha256":"img8"}}}
    validate_pair_references(p,observations)
    observations['now']['step_index'] = 7
    with pytest.raises(ValueError,match="history reference"):
        validate_pair_references(p,observations)


def model_inputs(use_branch=True, stochastic=True):
    torch.manual_seed(7)
    config = EventModelConfig(feature_dim=12,history_length=4,width=16,layers=1,heads=4,
                              stochastic_dim=4,use_branch=use_branch,stochastic=stochastic)
    model = EventLatentWorldModel(config).eval()
    history = torch.randn(3,4,12)
    valid = torch.ones(3,4,dtype=torch.bool)
    valid[:,0] = False
    branch = torch.tensor([[0.,1.],[1.,0.],[-1.,0.]])
    return model, history, valid, branch


def test_prior_prediction_does_not_use_the_future_target():
    model,h,v,b = model_inputs()
    before = model.predict(h,v,b)
    out = model.training_outputs(h,v,b,torch.randn(3,12))
    after = model.predict(h,v,b)
    assert torch.equal(before,after)
    assert torch.allclose(before,out["prior_prediction"])
    assert torch.allclose(before.norm(dim=-1),torch.ones(3),atol=1e-6)


def test_no_branch_control_is_invariant_to_branch_changes():
    model,h,v,b = model_inputs(use_branch=False)
    assert torch.equal(model.predict(h,v,b),model.predict(h,v,b.roll(1,0)))
    conditional,h,v,b = model_inputs(use_branch=True)
    assert not torch.allclose(conditional.predict(h,v,b),conditional.predict(h,v,b.roll(1,0)))


def test_padded_history_has_no_influence():
    model,h,v,b = model_inputs()
    before = model.predict(h,v,b)
    changed = h.clone()
    changed[:,0] = 1000
    assert torch.allclose(before,model.predict(changed,v,b),atol=1e-6)


@pytest.mark.parametrize("stochastic",[True,False])
def test_objective_has_finite_backward_and_prior_can_be_exported(stochastic):
    model,h,v,b = model_inputs(stochastic=stochastic)
    target = torch.nn.functional.normalize(torch.randn(3,12),dim=-1)
    outputs = model.training_outputs(h,v,b,target)
    loss, terms = event_objective(outputs,target,torch.tensor([0,1,1]))
    loss.backward()
    assert math.isfinite(loss.item())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert terms["prior_cosine_loss"].item() >= 0
    assert model.imagine(h,v,b).shape == (3,4 if stochastic else 1,12)


def test_same_real_target_place_is_not_a_false_negative():
    vectors = torch.nn.functional.normalize(torch.randn(4,12),dim=-1)
    loss = multi_positive_contrastive(vectors,vectors,torch.ones(4,dtype=torch.long))
    assert loss.item() == pytest.approx(0.,abs=1e-6)


def test_exported_predictor_binds_normalization_and_needs_no_future(tmp_path):
    model,h,v,b = model_inputs()
    mean = torch.randn(12)
    scale = torch.ones(12)*2
    predictor = EventFeaturePredictor(model,mean,scale).eval()
    expected = predictor(h,v,b)
    path = tmp_path/'predictor.pt'
    torch.save({'model_config':model.configuration(),'model_state_dict':model.state_dict(),
                'feature_mean':mean,'feature_scale':scale},path)
    restored = load_event_predictor(path)
    assert torch.allclose(restored(h,v,b),expected,atol=1e-6)
    assert torch.allclose(restored(h,v,b).norm(dim=-1),torch.ones(3),atol=1e-6)
