"""Typed fixed-control-span forecasts from newly observed, persisted experience.

Old next-event labels are never relabeled as fixed-duration outcomes. Online
scene weights can propose a future query without a matching source record;
such retrieval remains an analogy, not a newly confirmed physical connection.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch

from experiments.world_model.event_runtime import BranchEvidence
from experiments.world_model.scene_memory import MemoryMatch
from experiments.world_model.task_planning import EnvironmentEvidence, TaskPlanningContext


class ControlSpanQueries:
    def __init__(self,*,predictor,learner,images,alignment_provider,episode_id,horizon=12):
        if learner.target_kind!='observed_control_span':
            raise ValueError('fixed control queries require their own typed observed-span memory')
        if learner.binding['feature_contract']!=predictor.feature_contract_id:
            raise ValueError('query and online replay feature contracts differ')
        self.predictor,self.learner,self.images=predictor,learner,images
        self.alignment_provider=alignment_provider
        self.episode_id,self.horizon=episode_id,horizon
        self.visible_sequence=int(learner.db.execute('select coalesce(max(sequence),0) from observations').fetchone()[0])
        self.records=[]
        for row in learner.db.execute('select sequence,sample,episode,source_ref,target_ref,inputs,target,proof from observations where sequence<=?',(self.visible_sequence,)):
            sequence,sample,episode,source,target,inputs,future,proof=row
            proof=json.loads(proof)
            if episode==episode_id or proof.get('horizon')!=horizon or proof.get('skill_id')!='frozen_lightnav_visible_entrance':
                continue
            for kind,reference in [('source',source),('target',target)]:
                record=proof['images'][kind]
                self.images.register_file(reference,record['path'],record['sha256'])
            inputs=np.frombuffer(inputs,dtype=np.float32).copy()
            match=MemoryMatch(sequence,episode,sample,source,target,0.,0.,0.,0.,
                inputs[:predictor.model.config.feature_dim],inputs[-3:-1],np.frombuffer(future,dtype=np.float32).copy())
            self.records.append(match)
        self.last_trace=[]
        self.last_inputs={}

    @torch.inference_mode()
    def build_context(self,*,instruction,source_observation,branches,raw_history,valid,columns,progress=''):
        from dataclasses import replace

        current=self.predictor.normalize(raw_history[:,-1])[0].cpu().numpy()
        device=raw_history.device
        dim=len(current)
        self.last_trace=[]
        self.last_inputs={}
        predictions=[]
        environment={}
        for branch in branches:
            if branch.source_observation!=source_observation:
                raise ValueError('span forecast needs a current entrance')
            direction=np.array([math.sin(branch.bearing_rad),math.cos(branch.bearing_rad)],dtype=np.float32)
            transport=self.alignment_provider.for_branch(source_observation,branch)
            compatible=[]
            for record in self.records:
                source_cos=float(current@record.source_feature)
                if source_cos<.5:
                    continue
                environment[record.sequence]=replace(record,score=source_cos,source_similarity=source_cos)
                aligned=transport(record.source_observation,record.branch_direction)
                if aligned is None:
                    continue
                transported,information=aligned
                branch_cos=float(direction@transported)
                if branch_cos<math.cos(math.radians(35)):
                    continue
                compatible.append(replace(record,score=.8*source_cos+.2*branch_cos,source_similarity=source_cos,
                    branch_similarity=branch_cos,aligned_branch_direction=transported,alignment_info=information))
            selected=sorted(compatible,key=lambda m:(-m.score,m.sequence))[:3]
            basis=current.copy()
            if selected:
                scores=np.array([r.score for r in selected])
                weights=np.exp((scores-scores.max())/.07)
                weights/=weights.sum()
                remembered=sum(w*r.observed_future for w,r in zip(weights,selected,strict=True))
                confidence=np.clip((max(r.source_similarity for r in selected)-.5)/.5,0,1)
                basis=(1-.5*confidence)*current+.5*confidence*remembered
                basis/=max(np.linalg.norm(basis),1e-8)
            empty={'memory_source':torch.zeros(1,3,dim,device=device),
                   'memory_future':torch.zeros(1,3,dim,device=device),
                   'memory_branch':torch.zeros(1,3,2,device=device),
                   'memory_valid':torch.zeros(1,3,dtype=torch.bool,device=device)}
            result=self.predictor(raw_history,valid,torch.from_numpy(direction[None]).to(device),columns,
                                  **empty,return_aux=True)
            condition=result['condition'][0].cpu().numpy()
            inputs=dict(current=current,condition=condition,branch=direction,horizon=self.horizon,basis=basis)
            future=self.learner.forecast(**inputs)
            self.last_inputs[branch.candidate_id]=inputs
            usable=bool(selected) or self.learner.version>0
            ranked=sorted([replace(m,score=float(future@m.observed_future),
                future_similarity=float(future@m.observed_future),source_similarity=float(current@m.source_feature),
                branch_similarity=float(direction@m.branch_direction)) for m in self.records],
                key=lambda m:(-m.score,m.sequence))[:3] if usable else []
            evidence=BranchEvidence(future,tuple(m.sequence for m in selected),tuple(ranked),self.visible_sequence,
                bool(selected),'adapted_control_span' if self.learner.version else 'observed_control_span_prior',
                source_observation,tuple(float(v) for v in direction))
            predictions.append((branch.candidate_id,evidence))
            self.last_trace.append({'candidate_id':branch.candidate_id,'query_kind':evidence.query_kind,
                'target_kind':'observed_control_span','horizon':self.horizon,'model_version':self.learner.version,
                'supported_by_history':bool(selected),'scene_model_hypothesis_without_direct_match':not selected and self.learner.version>0,
                'past_transition_sequences':list(evidence.past_transition_sequences),
                'query_sha256':hashlib.sha256(future.tobytes()).hexdigest(),
                'retrieved':[{'sequence':m.sequence,'episode':m.episode_id,'actual_source':m.source_observation,
                              'actual_outcome':m.target_observation,'score':m.score} for m in ranked]})
        return TaskPlanningContext(self.episode_id,instruction,source_observation,self.visible_sequence,
            EnvironmentEvidence(tuple(environment.values())),tuple(branches),tuple(predictions),progress)

    def start_prediction(self,candidate_id,*,sample_id,source_observation,source_step):
        return self.learner.predict(sample_id=sample_id,source_observation=source_observation,
                                   source_step=source_step,**self.last_inputs[candidate_id])
