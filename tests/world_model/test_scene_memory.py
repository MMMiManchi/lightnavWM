"""Cross-episode persistence, real-outcome provenance and causal visibility."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.world_model.event_inference import EventFeaturePredictor
from experiments.world_model.event_latent import EventLatentWorldModel, EventModelConfig
from experiments.world_model.event_memory import EpisodeMemorySession, predict_with_scene_memory
from experiments.world_model.scene_memory import PersistentSceneMemory


def bank(tmp_path):
    return PersistentSceneMemory(tmp_path/'memory.sqlite',contract_id='frozen-features-1',feature_dim=4)


def observed(memory, **changes):
    row = dict(context='house',episode_id='ep1',event_id='event',source_observation='before',
               target_observation='after',execution_ref='ack',source_feature=[1,0,0,0],
               observed_future=[0,1,0,0],branch_direction=[0,1],executed=True,arrived=True)
    row.update(changes)
    return memory.record_observed(**row)


def query(memory, **changes):
    row = dict(context='house',before_sequence=memory.latest_sequence('house'),
               source_feature=[1,0,0,0],predicted_future=[1,0,0,0],branch_direction=[0,1])
    row.update(changes)
    return memory.retrieve(**row)


def test_memory_and_episode_instructions_survive_restart(tmp_path):
    m = bank(tmp_path)
    m.begin_episode('house','ep1','Go through the door')
    observed(m)
    m.end_episode('house','ep1')
    m.close()
    m = bank(tmp_path)
    assert len(query(m,exclude_episode='ep2')) == 1
    assert m.episode_history('house')[0]['instruction'] == 'Go through the door'
    assert m.episode_history('house')[0]['ended_sequence'] == 1
    assert m.experience_history('house',episode_id='ep1')[0]['source_observation'] == 'before'
    m.close()


def test_quarantined_experience_cannot_be_loaded_for_retrieval_or_scene_updates(tmp_path):
    m = bank(tmp_path)
    observed(m)
    m.close()
    before = (tmp_path/'memory.sqlite').read_bytes()
    (tmp_path/'DO_NOT_ADMIT_EXPERIENCE.json').write_text('{"reason":"reviewed wrong-exit attribution"}')
    with pytest.raises(ValueError,match='quarantined'):
        bank(tmp_path)
    assert (tmp_path/'memory.sqlite').read_bytes() == before


def test_contexts_and_feature_contracts_are_isolated(tmp_path):
    m = bank(tmp_path)
    observed(m)
    assert query(m,context='another_house') == []
    m.close()
    with pytest.raises(ValueError,match='contract mismatch'):
        PersistentSceneMemory(tmp_path/'memory.sqlite',contract_id='changed-normalization',feature_dim=4)


def test_future_insertions_and_current_episode_are_not_visible(tmp_path):
    m = bank(tmp_path)
    first = observed(m)
    cutoff = first
    observed(m,episode_id='ep_later',event_id='future')
    assert {r.episode_id for r in query(m,before_sequence=cutoff)} == {'ep1'}
    assert query(m,before_sequence=cutoff,exclude_episode='ep1') == []
    m.close()


@pytest.mark.parametrize('change',[
    {'provenance':'prediction'}, {'executed':False}, {'arrived':False},
    {'target_observation':''}, {'execution_ref':''}, {'observed_future':[np.nan,1,0,0]},
])
def test_unobserved_or_unverified_outcomes_cannot_become_facts(tmp_path,change):
    m = bank(tmp_path)
    with pytest.raises(ValueError):
        observed(m,**change)
    assert m.count('house') == 0
    m.close()


def test_truncated_observations_are_kept_without_becoming_next_event_targets(tmp_path):
    m = bank(tmp_path)
    observed(m,outcome='truncated',arrived=False)
    assert m.count('house') == 1
    assert query(m) == []
    m.close()


def test_an_observed_event_is_idempotent_and_cannot_be_rewritten(tmp_path):
    m = bank(tmp_path)
    assert observed(m) == observed(m)
    with pytest.raises(ValueError,match='cannot be rewritten'):
        observed(m,observed_future=[0,0,1,0])
    assert m.count('house') == 1
    m.close()


def test_opposite_branch_does_not_match_the_same_visual_place(tmp_path):
    m = bank(tmp_path)
    observed(m)
    assert query(m,branch_direction=[0,-1]) == []
    m.close()


def test_visual_revisit_proposal_does_not_require_a_preselected_direction(tmp_path):
    m = bank(tmp_path)
    observed(m, branch_direction=[0, -1])
    visual = m.nearest_observed_sources(context='house', before_sequence=m.latest_sequence('house'),
                                       source_feature=[1, 0, 0, 0], exclude_episode='new', source_floor=.7)
    assert len(visual) == 1 and visual[0].branch_direction[1] == -1
    # It remains inappropriate evidence for a forward branch.
    assert query(m, branch_direction=[0, 1]) == []
    m.close()


def test_cross_episode_prediction_only_changes_after_real_history(tmp_path):
    m = bank(tmp_path)
    session = EpisodeMemorySession(m)
    session.begin_episode('house','ep1')
    a = session.predict_cached([1,0,0,0],[1,0,0,0],[0,1],event_id='a',source_observation='a-before')
    assert not a.matches and np.array_equal(a.future_feature,a.base_future)
    assert m.count('house') == 0
    session.observe_real(a.prediction_id,[0,1,0,0],target_observation='a-after',execution_ref='ack',executed_branch_direction=[0,1],executed=True,arrived=True)
    same_episode = session.predict_cached([1,0,0,0],[1,0,0,0],[0,1],event_id='b',source_observation='b-before')
    assert not same_episode.matches
    session.begin_episode('house','ep2')
    b = session.predict_cached([1,0,0,0],[1,0,0,0],[0,1],event_id='c',source_observation='c-before')
    assert b.matches[0].episode_id == 'ep1'
    assert b.future_feature[1] > .9
    context = b.history_context()
    assert context['source_features'].shape == (1,4)
    assert context['executed_directions'].shape == (1,2)
    assert context['observed_future_features'].shape == (1,4)
    assert context['episode_ids_for_audit'] == ('ep1',)
    assert context['record_sequences'].max() <= b.visible_sequence
    assert m.count('house') == 1
    m.close()


def test_stale_decision_cannot_write_into_a_new_episode(tmp_path):
    m = bank(tmp_path)
    session = EpisodeMemorySession(m)
    session.begin_episode('house','ep1')
    a = session.predict_cached([1,0,0,0],[1,0,0,0],[0,1],event_id='a',source_observation='before')
    session.begin_episode('house','ep2')
    with pytest.raises(ValueError,match='stale'):
        session.observe_real(a.prediction_id,[0,1,0,0],target_observation='after',execution_ref='ack',executed_branch_direction=[0,1],executed=True,arrived=True)
    assert m.count('house') == 0
    m.close()


def test_real_predictor_reads_cross_episode_history_without_updating_parameters(tmp_path):
    torch.manual_seed(8)
    model = EventLatentWorldModel(EventModelConfig(feature_dim=12,history_length=4,width=16,layers=1,heads=4,stochastic_dim=4))
    predictor = EventFeaturePredictor(model,torch.zeros(12),torch.ones(12)).eval()
    old_weights = {k:v.clone() for k,v in predictor.state_dict().items()}
    m = PersistentSceneMemory(tmp_path/'integration.sqlite',contract_id='test',feature_dim=12)
    session = EpisodeMemorySession(m)
    history = torch.randn(1,4,12)
    valid = torch.ones(1,4,dtype=torch.bool)
    branch = torch.tensor([[0.,1.]])
    session.begin_episode('house','ep1')
    first = predict_with_scene_memory(predictor,session,history,valid,branch,event_id='one',source_observation='before1')
    target = np.zeros(12,dtype=np.float32)
    target[0] = 1
    session.observe_real(first.prediction_id,target,target_observation='after1',execution_ref='ack1',executed_branch_direction=[0,1],executed=True,arrived=True)
    session.begin_episode('house','ep2')
    second = predict_with_scene_memory(predictor,session,history,valid,branch,event_id='two',source_observation='before2')
    assert second.matches[0].episode_id == 'ep1'
    assert np.allclose(first.base_future,second.base_future)
    assert not np.allclose(second.base_future,second.future_feature)
    assert all(torch.equal(old_weights[k],v) for k,v in predictor.state_dict().items())
    m.close()


def test_controller_deviation_is_stored_under_the_actual_direction(tmp_path):
    m = bank(tmp_path)
    session = EpisodeMemorySession(m)
    session.begin_episode('house','ep1')
    p = session.predict_cached([1,0,0,0],[1,0,0,0],[0,1],event_id='one',source_observation='before')
    session.observe_real(p.prediction_id,[0,1,0,0],target_observation='after',execution_ref='ack',
                         executed_branch_direction=[0,-1],executed=True,arrived=True)
    session.begin_episode('house','ep2')
    planned = session.predict_cached([1,0,0,0],[1,0,0,0],[0,1],event_id='two',source_observation='before2')
    actual = session.predict_cached([1,0,0,0],[1,0,0,0],[0,-1],event_id='three',source_observation='before3')
    assert not planned.matches and actual.matches
    m.close()
