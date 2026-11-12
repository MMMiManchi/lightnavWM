"""Observed-span writeback, online scene weights, alignment and task progress."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import torch
from PIL import Image

from experiments.world_model.continual_queries import ControlSpanQueries
from experiments.world_model.entrance_alignment import ImageDirectionTransport
from experiments.world_model.online_scene import verify_span
from experiments.world_model.predictive_harness import PredictiveTaskHarness
from experiments.world_model.task_progress import (
    QwenProgressVerifier,
    QwenTaskDecomposer,
    TaskProgress,
    TaskStage,
)
from experiments.world_model.visual_harness import (
    ObservationImages,
    QwenBranchProposer,
    QwenTaskPlanner,
)


class ContinualEpisodeController:
    def __init__(self,*,executor,predictor,learner,vision,episode_id,output,initial_rgb,horizon=6):
        self.executor,self.predictor,self.learner=executor,predictor,learner
        self.episode_id,self.output,self.horizon=episode_id,output,horizon
        self.output.mkdir(parents=True,exist_ok=False)
        (self.output/'frames').mkdir()
        (self.output/'predictions').mkdir()
        self.images=ObservationImages()
        initial=f'{episode_id}:view:0'
        self.images.put(initial,initial_rgb)
        decomposer=QwenTaskDecomposer(vision)
        try:
            stages=decomposer(executor.original_instruction)
            decomposition_error=None
        except (ValueError,TypeError,KeyError) as error:
            stages=(TaskStage('stage-0',executor.original_instruction,'All requested target and stopping relations visibly established','goal'),)
            decomposition_error=str(error)
        policy=executor.policy
        motion_contract=({'dt_seconds':policy.dt,'lin_vel_range':policy.lin_vel_range,'ang_vel_range':policy.ang_vel_range}
                         if all(hasattr(policy,k) for k in ['dt','lin_vel_range','ang_vel_range']) else None)
        self.progress=TaskProgress(episode_id=episode_id,instruction=executor.original_instruction,
            stages=stages,initial_observation=initial,motion_contract=motion_contract)
        self.decomposition={'stages':[asdict(s) for s in stages],'raw':decomposer.last_raw,'error':decomposition_error}
        self.verifier=QwenProgressVerifier(vision,self.images)
        self.transport=ImageDirectionTransport(self.images)
        self.query=ControlSpanQueries(predictor=predictor,learner=learner,images=self.images,
            alignment_provider=self.transport,episode_id=episode_id,horizon=horizon)
        self.harness=PredictiveTaskHarness(executor=executor,predictor=predictor,bank=None,
            memory_context=learner.scene_id,episode_id=episode_id,images=self.images,
            proposer=QwenBranchProposer(vision,self.images),planner=QwenTaskPlanner(vision,self.images),
            progress_tracker=self.progress,query_service=self.query)
        self.pending=None
        self.verified_spans=[]
        self.abandoned=[]
        self.progress_trace=[]
        self.planning_trace=[]
        self.commands=[]
        self.current_reference=None

    def observe(self,step,rgb,feature,columns,depth=None):
        reference=f'{self.episode_id}:view:{step}'
        path=self.output/'frames'/f'{step:04d}.png'
        Image.fromarray(np.asarray(rgb)).save(path)
        self.harness.observe(reference,rgb,feature,columns)
        self.progress.observe(reference,step)
        self.current_reference=reference
        self.transport.current_depth=depth
        device=self.predictor.feature_mean.device
        current=self.predictor.normalize(torch.from_numpy(np.asarray(feature,dtype=np.float32))[None].to(device))[0].cpu().numpy()
        if self.pending is not None and step>=self.pending['ticket'].source_step+self.horizon:
            pending=self.pending
            self.pending=None
            ticket=pending['ticket']
            commands=pending['commands']
            if step==ticket.source_step+self.horizon and len(commands)==self.horizon and all(
                    c['local_goal_ref']==ticket.source_observation for c in commands):
                proof=verify_span(episode_id=self.episode_id,source_step=ticket.source_step,target_step=step,
                    horizon=self.horizon,commands=commands,source_path=pending['source_path'],target_path=path,
                    source_observation=ticket.source_observation,target_observation=reference)
                proof.update(skill_id='frozen_lightnav_visible_entrance',requested_local_instruction=pending['instruction'],
                             branch_condition_role='requested_current_entrance_under_frozen_closed_loop_policy')
                cosine=self.learner.observe(ticket,current,proof)
                self.verified_spans.append({'source_step':ticket.source_step,'target_step':step,
                    'prediction_version':ticket.model_version,'pre_update_cosine':cosine,'node_arrival_claimed':False})
            else:
                self.abandoned.append({'sample':ticket.sample_id,'reason':'execution_interrupted_or_horizon_changed'})
        return current

    def check_progress(self,step):
        if self.progress.complete:
            return
        try:
            evidence=self.verifier(self.progress,self.current_reference)
            advanced=self.progress.consider(evidence)
            self.progress_trace.append({'step':step,**self.verifier.last_trace,'status':self.progress.last_status,
                                        'advanced':advanced})
            if advanced:
                self.executor.restore_global_goal()
                self.harness.report_execution('Current task stage advanced from repeated actual visual evidence: '+self.progress.context())
        except (ValueError,TypeError,KeyError) as error:
            self.progress_trace.append({'step':step,'error':str(error),'advanced':False})

    def plan(self,step):
        self.transport.last_trace=[]
        try:
            result=self.harness.plan()
            self.planning_trace.append({'step':step,**result.trace,'alignment':self.transport.last_trace.copy()})
            if result.goal is not None:
                if self.pending is not None:
                    self.abandoned.append({'sample':self.pending['ticket'].sample_id,'reason':'new_local_goal_before_horizon'})
                decision=self.harness.router.last_decision
                ticket=self.query.start_prediction(decision.candidate_id,sample_id=f'{self.episode_id}:span:{step}:{len(self.planning_trace)}',
                    source_observation=self.current_reference,source_step=step)
                np.savez_compressed(self.output/'predictions'/f'{step:04d}-{len(self.planning_trace):02d}.npz',
                                    future=ticket.prediction,model_version=ticket.model_version,source_step=step)
                self.pending={'ticket':ticket,'commands':[],'source_path':self.output/'frames'/f'{step:04d}.png',
                              'instruction':result.goal.instruction}
                self.harness.report_execution(f'At step {step} issued local entrance task: {result.goal.instruction}; completion unverified.')
            return result
        except (ValueError,TypeError,KeyError) as error:
            self.executor.restore_global_goal()
            self.planning_trace.append({'step':step,**self.harness.last_trace,'error':str(error),
                                       'alignment':self.transport.last_trace.copy()})
            return None

    def executed(self,step,action,*,rgb_changed,local_goal_ref):
        record={'episode_id':self.episode_id,'step':step,'action':action,'environment_step_returned':True,
                'rgb_changed':bool(rgb_changed),'local_goal_ref':local_goal_ref}
        self.commands.append(record)
        self.progress.executed(step,environment_step_returned=True,rgb_changed=rgb_changed,action=action)
        if self.pending is not None:
            self.pending['commands'].append(record)

    def persist(self):
        for name,value in [('decomposition',self.decomposition),('planning',self.planning_trace),
                           ('progress',self.progress_trace),('verified_spans',self.verified_spans),
                           ('abandoned_spans',self.abandoned),('executed_commands',self.commands)]:
            (self.output/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        (self.output/'progress_state.json').write_text(self.progress.context()+'\n')
