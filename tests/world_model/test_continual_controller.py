import json
from types import SimpleNamespace

import numpy as np
import torch

from experiments.world_model.continual_controller import ContinualEpisodeController
from experiments.world_model.event_anchored import AnchoredEventConfig, AnchoredEventWorldModel
from experiments.world_model.event_runtime import HarnessLightNavExecutor, ResidualFeaturePredictor
from experiments.world_model.online_scene import OnlineSceneLearner, SceneAdapterConfig


class ControlledVision:
    def __call__(self,prompt,images,**kwargs):
        if prompt.startswith('Split this'):
            return json.dumps({'stages':[{'instruction':'reach the table','criterion':'table visible in front','kind':'goal'}]})
        if prompt.startswith('Inspect only'):
            return json.dumps({'event':'doorway','branches':[{'description':'visible door','bbox':[200,100,800,950],
                'entry_point':[500,800],'confidence':.9}]})
        if prompt.startswith('Choose the next'):
            context=json.loads(prompt.split('CONTEXT:\n')[1])
            return json.dumps({'action':'navigate','candidate_id':context['current_branches'][0]['candidate_id'],
                               'local_instruction':'go through the visible door','evidence_sequences':[]})
        return json.dumps({'completed':False,'target_visible':False,'confidence':.8,'reason':'target unknown'})


def setup(tmp_path,episode):
    config=AnchoredEventConfig(feature_dim=4,width=8,heads=2,layers=1,use_memory=True,residual_scale=0)
    model=AnchoredEventWorldModel(config)
    predictor=ResidualFeaturePredictor({'schema':'wm-residual-event/0.1','model_kind':'anchored_memory',
        'model_config':model.configuration(),'model_state_dict':model.state_dict(),'feature_mean':torch.zeros(4),
        'feature_scale':torch.ones(4),'feature_contract':{'name':'test'}}).eval()
    learner=OnlineSceneLearner(tmp_path/'scene',scene_id='house',feature_contract=predictor.feature_contract_id,
        base_hash='frozen',target_kind='observed_control_span',config=SceneAdapterConfig(feature_dim=4,condition_dim=8,rank=4))
    learner.begin_episode(episode)
    policy=SimpleNamespace(agent=SimpleNamespace(instruction='reach the table'))
    executor=HarnessLightNavExecutor(policy)
    executor.original_instruction='reach the table'
    pixels=np.random.default_rng(2).integers(0,256,(270,480,3),dtype=np.uint8)
    controller=ContinualEpisodeController(executor=executor,predictor=predictor,learner=learner,
        vision=ControlledVision(),episode_id=episode,output=tmp_path/episode,initial_rgb=pixels)
    return controller,learner,pixels


def test_actual_executed_span_writes_back_and_is_readable_next_episode(tmp_path):
    controller,learner,pixels=setup(tmp_path,'first')
    feature=np.array([1,0,0,0],dtype=np.float32)
    columns=torch.ones(1,27,4)
    controller.observe(0,pixels,feature,columns)
    assert controller.plan(0).goal is not None
    assert learner.db.execute('select count(*) from observations').fetchone()[0]==0
    source=controller.executor.active_goal.source_observation
    for step in range(6):
        controller.executed(step,{'move':1},rgb_changed=True,local_goal_ref=source)
        controller.observe(step+1,pixels,feature,columns)
    assert len(controller.verified_spans)==1
    assert controller.verified_spans[0]['node_arrival_claimed'] is False
    learner.finish_episode()
    learner.close()
    later,reopened,pixels=setup(tmp_path,'second')
    assert later.query.visible_sequence==1 and len(later.query.records)==1
    later.observe(0,pixels,feature,columns)
    later.plan(0)
    assert later.query.last_trace[0]['supported_by_history']
    assert later.query.last_trace[0]['retrieved'][0]['episode']=='first'
    reopened.close()


def test_interrupted_local_goal_cannot_be_recorded_as_its_executed_outcome(tmp_path):
    controller,learner,pixels=setup(tmp_path,'first')
    feature=np.array([1,0,0,0],dtype=np.float32)
    columns=torch.ones(1,27,4)
    controller.observe(0,pixels,feature,columns)
    controller.plan(0)
    for step in range(6):
        controller.executed(step,{'move':1},rgb_changed=True,local_goal_ref=None)
        controller.observe(step+1,pixels,feature,columns)
    assert not controller.verified_spans and controller.abandoned
    assert learner.db.execute('select count(*) from observations').fetchone()[0]==0
    learner.finish_episode()
    learner.close()
