import numpy as np
import pytest
import torch

from experiments.world_model.event_tssm import EventTSSM, TSSMConfig
from experiments.world_model.scene_memory import PersistentSceneMemory
from experiments.world_model.tssm_runtime import TSSMPredictor, TSSMSession


def setup(tmp_path, *, memory=True, learning=False):
    torch.manual_seed(3)
    model = EventTSSM(TSSMConfig(feature_dim=12, width=16, stochastic_dim=4, layers=1, heads=4))
    p = TSSMPredictor(model, torch.zeros(12), torch.ones(12)).eval()
    bank = PersistentSceneMemory(tmp_path / "facts.sqlite", contract_id=p.feature_contract_id, feature_dim=12)
    session = TSSMSession(p, bank, tmp_path, base_hash="fixed-model", use_memory=memory,
                          online_learning=learning, update_steps=20)
    history = np.tile(np.eye(12, dtype=np.float32)[0], (1, 8, 1))
    target = np.eye(12, dtype=np.float32)[1]
    return p, bank, session, history, target


def observe(session, ticket, target, **changes):
    args = dict(target_observation="actual-after", execution_ref="executed-commands", executed=True, arrived=True)
    args.update(changes)
    return session.observe_event(ticket, target, **args)


def test_past_real_memory_survives_episode_and_process_reset(tmp_path):
    p, bank, s, h, target = setup(tmp_path)
    s.begin_episode("house", "ep1", h, "source")
    before = s.forecast("branch", [0, 1])
    assert bank.count("house") == 0
    observe(s, before, target)
    s.end_episode()
    bank.close()
    bank = PersistentSceneMemory(tmp_path / "facts.sqlite", contract_id=p.feature_contract_id, feature_dim=12)
    s = TSSMSession(p, bank, tmp_path, base_hash="fixed-model")
    s.begin_episode("house", "ep2", h, "new-source")
    known = s.forecast("known", [0, 1])
    unknown = s.forecast("unvisited", [0, -1])
    assert known.future_feature @ target > .99 and known.memory_sequences
    assert not unknown.memory_sequences
    assert s.state.tokens.shape[1] == 0
    assert s.query_actual_memories(known)[0].sequence <= known.visible_sequence
    bank.close()


def test_only_post_observation_updates_change_scene_parameters(tmp_path):
    p, bank, s, h, target = setup(tmp_path, memory=False, learning=True)
    fixed = {k: v.clone() for k, v in p.state_dict().items()}
    s.begin_episode("house", "ep1", h, "source")
    old = s.updates.parameter_hash()
    ticket = s.forecast("branch", [0, 1])
    assert s.updates.parameter_hash() == old
    result = observe(s, ticket, target)
    assert result["update"]["parameters_changed"]
    assert result["update"]["past_replay_cosine_after"] > result["update"]["past_replay_cosine_before"]
    assert all(torch.equal(v, p.state_dict()[k]) for k, v in fixed.items())
    s.end_episode()
    s.begin_episode("house", "ep2", h, "revisit")
    assert s.updates.parameter_hash() != old and s.updates.version == 1
    assert s.forecast("revisit", [0, 1]).future_feature @ target > ticket.future_feature @ target
    unseen = s.forecast("not-yet-walked", [0, -1])
    assert not unseen.adapter_applied
    np.testing.assert_allclose(unseen.future_feature, unseen.base_feature)
    bank.close()


def test_route_posterior_feedback_does_not_forge_decision_memories(tmp_path):
    _p, bank, s, h, target = setup(tmp_path)
    s.begin_episode("house", "ep1", h, "source")
    left, right = s.forecast("left", [0, 1]), s.forecast("right", [1, 0])
    s.commit(left)
    s.observe_route(target, "mid-route", execution_ref="motion", executed=True)
    assert bank.count("house") == 0 and s.state.tokens.shape[1] == 1
    with pytest.raises(ValueError, match="stale"):
        observe(s, right, target)
    observe(s, left, target)
    assert bank.count("house") == 1 and s.state.tokens.shape[1] == 2
    bank.close()


def test_final_frame_is_not_assimilated_twice(tmp_path):
    _p, bank, s, h, target = setup(tmp_path)
    s.begin_episode("house", "ep1", h, "source")
    t = s.forecast("branch", [0, 1])
    s.commit(t)
    s.observe_route(target, "actual-after", execution_ref="motion", executed=True)
    before = s.state.tokens.shape[1]
    observe(s, t, target)
    assert s.state.tokens.shape[1] == before
    bank.close()


def test_actual_global_motion_invalidates_uncommitted_queries_without_writing_facts(tmp_path):
    _p, bank, s, h, target = setup(tmp_path)
    s.begin_episode("house", "ep1", h, "source")
    t = s.forecast("proposed", [0, 1])
    s.observe_route(target, "actual-milestone", execution_ref="executed-global-motion", executed=True)
    assert s.state.tokens.shape[1] == 1 and bank.count("house") == 0
    with pytest.raises(ValueError, match="stale"):
        observe(s, t, target)
    bank.close()


@pytest.mark.parametrize("changes", [{"executed": False}, {"arrived": False}, {"execution_ref": ""}])
def test_unverified_future_cannot_become_memory_or_training(tmp_path, changes):
    _p, bank, s, h, target = setup(tmp_path, learning=True)
    s.begin_episode("house", "ep1", h, "source")
    t = s.forecast("branch", [0, 1])
    with pytest.raises(ValueError):
        observe(s, t, target, **changes)
    assert bank.count("house") == 0 and s.updates.version == 0
    bank.close()


def test_actual_direction_is_used_and_other_context_does_not_inherit_memory(tmp_path):
    _p, bank, s, h, target = setup(tmp_path)
    s.begin_episode("house", "ep1", h, "source")
    t = s.forecast("planned", [0, 1])
    r = observe(s, t, target, executed_direction=[0, -1])
    assert not r["planned_branch_verified"] and r["pre_update_prediction_cosine"] is None
    s.begin_episode("house", "ep2", h, "revisit")
    assert not s.forecast("planned", [0, 1]).memory_sequences
    assert s.forecast("actual", [0, -1]).memory_sequences
    s.begin_episode("another-house", "ep3", h, "elsewhere")
    assert not s.forecast("unknown", [0, -1]).memory_sequences
    with pytest.raises(ValueError, match="stale"):
        observe(s, t, target)
    bank.close()
