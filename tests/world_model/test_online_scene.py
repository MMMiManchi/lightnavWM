
import numpy as np
import pytest
import torch
from PIL import Image

from experiments.world_model.online_scene import OnlineSceneLearner, SceneAdapterConfig, verify_span


def learner(folder, **kwargs):
    config = SceneAdapterConfig(feature_dim=4, condition_dim=2, rank=4, learning_rate=.02, update_steps=40)
    return OnlineSceneLearner(folder, scene_id="house", feature_contract="fixed", base_hash="base", target_kind="next_event",
                              config=config, **kwargs)


def prediction(model, index):
    return model.predict(sample_id=f"event{index}", source_observation=f"source{index}", source_step=index*4,
        current=[1,0,0,0], condition=[1,0], branch=[0,1], horizon=0, basis=[1,0,0,0])


def proof(ticket):
    return {"kind":"next_event", "real_observation":True,"episode_id":ticket.episode_id,
            "source_observation":ticket.source_observation,"target_observation":f"target-{ticket.sample_id}",
            "source_step":ticket.source_step,"target_step":ticket.source_step+4,
            "admitted_execution_ref":f"actual-recorded:{ticket.sample_id}"}


def test_writing_real_memory_does_not_itself_update_parameters(tmp_path):
    model=learner(tmp_path)
    model.begin_episode("one")
    before=model.parameters_hash()
    ticket=prediction(model,0)
    model.observe(ticket,[.8,.6,0,0],proof(ticket))
    assert model.parameters_hash()==before and model.version==0
    assert model.db.execute("select count(*) from observations").fetchone()[0]==1
    with pytest.raises(ValueError,match="inside"):
        model.update()
    model.close()


def test_gradients_update_scene_only_after_episode_and_checkpoint_restores(tmp_path):
    model=learner(tmp_path)
    model.begin_episode("one")
    before=model.parameters_hash()
    for i in range(8):
        ticket=prediction(model,i)
        assert ticket.model_version==0
        model.observe(ticket,[.8,.6,0,0],proof(ticket))
    assert model.parameters_hash()==before
    result=model.finish_episode()
    assert result["optimizer_steps"]==40 and result["active_parameters_changed"]
    assert result["validation_after"]<result["validation_before"]
    assert model.version==1
    state=model.parameters_hash()
    model.close()
    reloaded=learner(tmp_path)
    assert reloaded.parameters_hash()==state and reloaded.version==1
    reloaded.begin_episode("two")
    ticket=prediction(reloaded,9)
    assert ticket.prediction[1]>0 and ticket.model_version==1
    reloaded.close()


def test_future_or_mismatched_proof_cannot_train_a_prediction(tmp_path):
    model=learner(tmp_path)
    model.begin_episode("one")
    ticket=prediction(model,0)
    fake=proof(ticket)
    fake["real_observation"]=False
    with pytest.raises(ValueError,match="provenance"):
        model.observe(ticket,[0,1,0,0],fake)
    fake=proof(ticket)
    fake["target_step"]=ticket.source_step
    with pytest.raises(ValueError,match="provenance"):
        model.observe(ticket,[0,1,0,0],fake)
    model.finish_episode()
    assert model.db.execute("select count(*) from observations").fetchone()[0]==0
    with pytest.raises(ValueError):
        model.observe(ticket,[0,1,0,0],proof(ticket))
    model.close()


def test_checkpoint_cannot_be_rebound_to_other_scene_or_feature_space(tmp_path):
    model=learner(tmp_path)
    model.close()
    with pytest.raises(ValueError,match="binding"):
        OnlineSceneLearner(tmp_path,scene_id="other",feature_contract="fixed",base_hash="base",target_kind="next_event",
                           config=SceneAdapterConfig(feature_dim=4,condition_dim=2,rank=4,learning_rate=.02))


def test_control_span_checks_actual_dispatches_and_keeps_arrival_unknown(tmp_path):
    for name in ["source","target"]:
        Image.fromarray(np.zeros((32,48,3),dtype=np.uint8)).save(tmp_path/f"{name}.png")
    commands=[{"step":i,"episode_id":"one","environment_step_returned":True,"action":{"move":1}} for i in range(2)]
    kwargs=dict(episode_id="one",source_step=0,target_step=2,horizon=2,commands=commands,
                source_path=tmp_path/'source.png',target_path=tmp_path/'target.png',source_observation="s",target_observation="t")
    observed=verify_span(**kwargs)
    assert observed["node_arrival_claimed"] is False
    commands[1]["environment_step_returned"]=False
    with pytest.raises(ValueError,match="acknowledgement"):
        verify_span(**kwargs)


def test_no_history_does_not_mask_the_scene_residual(tmp_path):
    model=learner(tmp_path)
    with torch.no_grad():
        model.adapter.up.bias.copy_(torch.tensor([0.,1,0,0]))
    result=model.forecast(current=[1,0,0,0],condition=[1,0],branch=[0,1],horizon=0,basis=[1,0,0,0])
    assert result[1]>0  # No historical-memory mask can turn learned parameters off.
    model.close()


def test_true_future_event_duration_cannot_leak_into_the_prediction(tmp_path):
    model=learner(tmp_path)
    with pytest.raises(ValueError,match='horizon'):
        model.forecast(current=[1,0,0,0],condition=[1,0],branch=[0,1],horizon=12,basis=[1,0,0,0])
    model.close()


def test_scene_initialization_preserves_inference_rngs(tmp_path,monkeypatch):
    before=torch.random.get_rng_state().clone()
    def forbidden_cuda_seed(_seed):
        pytest.fail('CPU scene initialization must not reset live CUDA generators')
    monkeypatch.setattr(torch.cuda,'manual_seed_all',forbidden_cuda_seed)
    model=learner(tmp_path)
    assert torch.equal(before,torch.random.get_rng_state())
    model.close()
