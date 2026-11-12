import json

import numpy as np
import pytest

from experiments.world_model.task_progress import (
    QwenProgressVerifier,
    QwenTaskDecomposer,
    StageEvidence,
    TaskProgress,
    TaskStage,
)
from experiments.world_model.visual_harness import ObservationImages


def make_progress():
    return TaskProgress(episode_id='new-task',instruction='exit then stop at table',initial_observation='start',
        stages=[TaskStage('exit','exit room','door frame is behind us','navigate'),
                TaskStage('goal','stop at table','table visibly in front at stopping location','goal')])


def support(progress,reference,**kwargs):
    return StageEvidence(progress.active.stage_id,progress.anchor,reference,True,kwargs.get('target_visible',False),.9,'visible before and after change')


def advance_views(progress,start,first,second,visible=False):
    progress.observe(first,start)
    assert not progress.consider(support(progress,first,target_visible=visible))
    for i in range(start,start+3):
        progress.executed(i,environment_step_returned=True,rgb_changed=True,action={'move':1})
    progress.observe(second,start+3)
    return progress.consider(support(progress,second,target_visible=visible))


def test_stage_order_and_final_target_need_actual_temporally_separated_support():
    progress=make_progress()
    assert not progress.accept_global_stop()
    assert advance_views(progress,0,'exit-a','exit-b')
    assert progress.active.kind=='goal' and not progress.accept_global_stop()
    assert not advance_views(progress,4,'goal-a','goal-b',visible=False)
    assert advance_views(progress,8,'goal-c','goal-d',visible=True)
    assert progress.accept_global_stop()


def test_one_image_or_unchanged_views_cannot_complete_a_stage():
    progress=make_progress()
    progress.observe('one',0)
    progress.consider(support(progress,'one'))
    assert not progress.consider(support(progress,'one'))
    for i in range(3):
        progress.executed(i,environment_step_returned=True,rgb_changed=False,action={'move':1})
    progress.observe('same-scene',3)
    assert not progress.consider(support(progress,'same-scene'))
    assert progress.active_index==0


def test_old_episode_and_later_stage_evidence_cannot_skip_current_instruction():
    progress=make_progress()
    with pytest.raises(ValueError,match='grounded'):
        progress.consider(StageEvidence('goal','old','not-seen',True,True,.99,'claims arrival'))
    assert progress.active_index==0


def test_an_actual_active_look_can_provide_the_second_goal_view():
    stage=TaskStage('goal','stop at table','table visibly in front','goal')
    progress=TaskProgress(episode_id='ep',instruction='stop at table',stages=[stage],initial_observation='start',
        motion_contract={'dt_seconds':.1,'lin_vel_range':[0,2.5],'ang_vel_range':[-300,300]})
    progress.observe('seen',0)
    progress.consider(support(progress,'seen',target_visible=True))
    progress.executed(0,environment_step_returned=True,rgb_changed=True,
                      action={'action':'velocity_control','action_args':{'linear_velocity':-1.,'angular_velocity':1.}})
    progress.observe('looked',1)
    assert progress.consider(support(progress,'looked',target_visible=True))
    assert progress.accept_global_stop()


def test_commanded_turn_without_measured_motion_does_not_certify_arrival():
    stage=TaskStage('goal','stop at table','table visibly in front','goal')
    progress=TaskProgress(episode_id='ep',instruction=stage.instruction,stages=[stage],initial_observation='start',
        motion_contract={'dt_seconds':.1,'lin_vel_range':[0,2.5],'ang_vel_range':[-300,300]},
        require_measured_motion=True)
    turn={'action':'velocity_control','action_args':{'linear_velocity':-1.,'angular_velocity':1.}}
    progress.executed(0,environment_step_returned=True,rgb_changed=True,action=turn)
    progress.observe('seen',1)
    progress.record_observed_motion('seen',{'reference':'seen','pose_epoch':0,'local_pose':[0.,0.,0.],
                                            'measurement':{'valid':True}})
    assert not progress.consider(support(progress,'seen',target_visible=True))
    progress.executed(1,environment_step_returned=True,rgb_changed=True,action=turn)
    progress.observe('still',2)
    progress.record_observed_motion('still',{'reference':'still','pose_epoch':0,'local_pose':[0.,0.,0.],
                                             'measurement':{'valid':True}})
    assert not progress.consider(support(progress,'still',target_visible=True))
    assert progress.last_status=='insufficient_measured_camera_motion'
    progress.executed(2,environment_step_returned=True,rgb_changed=True,action=turn)
    progress.observe('actually-turned',3)
    progress.record_observed_motion('actually-turned',{'reference':'actually-turned','pose_epoch':0,
        'local_pose':[0.,0.,.32],'measurement':{'valid':True}})
    assert progress.consider(support(progress,'actually-turned',target_visible=True))
    assert progress.completed[-1]['measured_motion_support']['yaw_rad']==pytest.approx(.32)


def test_progress_reanchors_after_a_broken_measured_motion_chain():
    stage=TaskStage('goal','stop at table','table visibly in front','goal')
    progress=TaskProgress(episode_id='ep',instruction=stage.instruction,stages=[stage],initial_observation='start',
                          require_measured_motion=True)
    progress.observe('old',1)
    progress.record_observed_motion('old',{'reference':'old','pose_epoch':0,'local_pose':[0.,0.,0.],
                                           'measurement':{'valid':True}})
    assert not progress.consider(support(progress,'old',target_visible=True))
    progress.executed(1,environment_step_returned=True,rgb_changed=True,action={'move':1})
    progress.observe('rebased',2)
    progress.record_observed_motion('rebased',{'reference':'rebased','pose_epoch':1,'local_pose':[0.,0.,0.],
                                               'measurement':{'valid':True}})
    assert not progress.consider(support(progress,'rebased',target_visible=True))
    assert progress.last_status=='measured_motion_unavailable_need_new_anchor'
    progress.executed(2,environment_step_returned=True,rgb_changed=True,action={'move':1})
    progress.observe('new-view',3)
    progress.record_observed_motion('new-view',{'reference':'new-view','pose_epoch':1,'local_pose':[0.,0.,.32],
                                                'measurement':{'valid':True}})
    assert progress.consider(support(progress,'new-view',target_visible=True))


@pytest.mark.parametrize(('rgb_changed','expected'), [(True,True),(False,False)])
def test_measured_translation_requires_a_distinct_actual_view(rgb_changed,expected):
    stage=TaskStage('goal','stop at table','table visibly in front','goal')
    progress=TaskProgress(episode_id='ep',instruction=stage.instruction,stages=[stage],initial_observation='start',
                          require_measured_motion=True)
    progress.observe('seen',1)
    progress.record_observed_motion('seen',{'reference':'seen','pose_epoch':0,'local_pose':[0.,0.,0.],
                                            'measurement':{'valid':True}})
    assert not progress.consider(support(progress,'seen',target_visible=True))
    for step in range(1,4):
        progress.executed(step,environment_step_returned=True,rgb_changed=rgb_changed,action={'move':1})
    progress.observe('after',4)
    progress.record_observed_motion('after',{'reference':'after','pose_epoch':0,'local_pose':[0.,.12,0.],
                                             'measurement':{'valid':True}})
    assert progress.consider(support(progress,'after',target_visible=True)) is expected


@pytest.mark.parametrize(('instruction','kind'), [
    ('walk across the open area','navigate'),
    ('turn right into the room','turn'),
    ('enter the room with the television','goal'),
])
def test_walking_stage_needs_translation_even_when_second_view_is_a_real_turn(instruction,kind):
    stage=TaskStage('move',instruction,'the stage is complete',kind)
    stages=[stage] if kind=='goal' else [stage,TaskStage('goal','stop','stop','goal')]
    progress=TaskProgress(episode_id='ep',instruction=instruction,stages=stages,initial_observation='start',
                          require_measured_motion=True)
    for step,ref,pose in [(0,'start',[0.,0.,0.]),(1,'seen',[0.,0.,0.]),(2,'turned',[0.,0.,.35])]:
        if step:
            progress.executed(step-1,environment_step_returned=True,rgb_changed=True,action={'move':1})
            progress.observe(ref,step)
        progress.record_observed_motion(ref,{'reference':ref,'pose_epoch':0,'local_pose':pose,
                                              'measurement':{'valid':True}})
        if step:
            assert not progress.consider(support(progress,ref,target_visible=True))
    assert progress.active_index==0
    assert progress.last_status=='stage_translation_not_measured'


def test_stage_translation_can_precede_the_two_confirmation_views():
    stage=TaskStage('walk','walk across the open area','the stage is complete','navigate')
    progress=TaskProgress(episode_id='ep',instruction=stage.instruction,
        stages=[stage,TaskStage('goal','stop','stop','goal')],initial_observation='start',require_measured_motion=True)
    for step,ref,pose in [(0,'start',[0.,0.,0.]),(1,'seen',[0.,.46,0.]),(2,'turned',[0.,.46,.35])]:
        if step:
            progress.executed(step-1,environment_step_returned=True,rgb_changed=True,action={'move':1})
            progress.observe(ref,step)
        progress.record_observed_motion(ref,{'reference':ref,'pose_epoch':0,'local_pose':pose,
                                              'measurement':{'valid':True}})
        if step==1:
            assert not progress.consider(support(progress,ref))
        elif step==2:
            assert progress.consider(support(progress,ref))
    assert progress.completed[-1]['measured_stage_translation']['translation_m']==pytest.approx(.46)


def test_stage_translation_does_not_cross_a_broken_motion_epoch():
    stage=TaskStage('walk','walk across the open area','the stage is complete','navigate')
    progress=TaskProgress(episode_id='ep',instruction=stage.instruction,
        stages=[stage,TaskStage('goal','stop','stop','goal')],initial_observation='start',require_measured_motion=True)
    for step,ref,epoch,pose in [(0,'start',0,[0.,0.,0.]),(1,'seen',1,[0.,0.,0.]),
                                (2,'turned',1,[0.,0.,.35])]:
        if step:
            progress.executed(step-1,environment_step_returned=True,rgb_changed=True,action={'move':1})
            progress.observe(ref,step)
        progress.record_observed_motion(ref,{'reference':ref,'pose_epoch':epoch,'local_pose':pose,
                                              'measurement':{'valid':True}})
        if step:
            assert not progress.consider(support(progress,ref))
    assert progress.last_status=='stage_translation_not_measured'


def test_normalized_controls_are_decoded_before_judging_turn_direction():
    progress=make_progress()
    progress.motion_contract={'dt_seconds':.1,'lin_vel_range':(0,2.5),'ang_vel_range':(-300,300)}
    progress.executed(0,environment_step_returned=True,rgb_changed=True,
        action={'action':'velocity_control','action_args':{'linear_velocity':-1.,'angular_velocity':-.5}})
    motion=progress.executions[0]['commanded_motion']
    assert motion['forward_metres']==0 and motion['yaw_degrees_left_positive']==-15
    assert motion['evidence_kind']=='issued_control_not_measured_camera_motion'


def test_intermediate_verifier_does_not_receive_a_later_goal():
    progress=make_progress()
    progress.observe('current',3)
    images=ObservationImages()
    for ref in ['start','current']:
        images.put(ref,np.zeros((32,48,3),dtype=np.uint8))
    prompts=[]
    def backend(prompt,_images):
        prompts.append(prompt)
        return json.dumps({'completed':True,'target_visible':False,'confidence':.9,'reason':'outside the initial room'})
    evidence=QwenProgressVerifier(backend,images)(progress,'current')
    assert evidence.completed
    assert 'table' not in prompts[0] and 'FIRST-PERSON' in prompts[0]
    assert '"completed":false' not in prompts[0] and '"confidence":0.0' not in prompts[0]


def test_final_verifier_demands_the_current_target_without_default_negative_values():
    stage=TaskStage('goal','stop beside the blue table','blue table beside the camera','goal')
    progress=TaskProgress(episode_id='ep',instruction=stage.instruction,stages=[stage],initial_observation='start')
    progress.observe('current',3)
    images=ObservationImages()
    for ref in ['start','current']:
        images.put(ref,np.zeros((32,48,3),dtype=np.uint8))
    prompts=[]
    def backend(prompt,_images):
        prompts.append(prompt)
        return json.dumps({'reason':'target absent','completed':False,'target_visible':False,'confidence':.95})
    evidence=QwenProgressVerifier(backend,images)(progress,'current')
    assert not evidence.completed and not evidence.target_visible
    assert 'CURRENT view' in prompts[0] and 'blue table' in prompts[0]
    assert 'There are no default values' in prompts[0]


def test_stage_roles_are_normalized_without_invented_visible_body_requirements():
    raw={'stages':[{'instruction':'exit the room','criterion':'person touching a handle','kind':'goal'},
                   {'instruction':'stop at the table','criterion':'both hands visible','kind':'navigate'},
                   {'instruction':'stop at the table','criterion':'duplicated','kind':'goal'}]}
    stages=QwenTaskDecomposer(lambda *_a,**_k:json.dumps(raw))('Exit the room then stop at the table.')
    assert len(stages)==2 and [s.kind for s in stages]==['navigate','goal']
    assert 'handle' not in stages[0].criterion and 'hands' not in stages[1].criterion
