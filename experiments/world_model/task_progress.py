"""Ordered task stages with current-episode visual evidence and STOP gating.

The verifier proposes visual support; repeated, motion-separated observations
are required to advance. This is not privileged goal-distance verification.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass

from experiments.world_model.visual_harness import parse_object


@dataclass(frozen=True)
class TaskStage:
    stage_id: str
    instruction: str
    criterion: str
    kind: str


@dataclass(frozen=True)
class StageEvidence:
    stage_id: str
    before_observation: str
    current_observation: str
    completed: bool
    target_visible: bool
    confidence: float
    reason: str
    target_identified: bool | None = None
    identity_reference: str | None = None
    verdict: str = "unknown"


class TaskProgress:
    def __init__(self, *, episode_id, instruction, stages, initial_observation, motion_contract=None, initial_step=0,
                 require_current_target=True, require_measured_motion=False):
        if not episode_id or not instruction or not stages or not initial_observation:
            raise ValueError('progress needs a current task, stages and actual initial view')
        if (len({s.stage_id for s in stages})!=len(stages) or stages[-1].kind!='goal'
                or any(s.kind=='goal' for s in stages[:-1])):
            raise ValueError('stages need distinct ids and a final goal')
        if any(s.kind not in ('navigate','turn','goal') or not s.criterion or not s.instruction for s in stages):
            raise ValueError('task stage lacks an explicit observation criterion')
        self.episode_id,self.instruction=episode_id,instruction
        self.stages=tuple(stages)
        self.active_index=0
        self.anchor=initial_observation
        if not isinstance(initial_step, int) or initial_step < 0:
            raise ValueError('initial observation requires its actual nonnegative step')
        self.observations={initial_observation:initial_step}
        self.executions={}
        self.confirmation=None
        self.completed=[]
        self.last_status='awaiting_visual_evidence'
        self.require_current_target=bool(require_current_target)
        self.require_measured_motion=bool(require_measured_motion)
        self.observed_motion={}
        self.motion_contract=None
        if motion_contract is not None:
            dt=float(motion_contract['dt_seconds'])
            ranges=[tuple(float(v) for v in motion_contract[name]) for name in ['lin_vel_range','ang_vel_range']]
            if not math.isfinite(dt) or dt<=0 or any(len(r)!=2 or not all(math.isfinite(v) for v in r) or r[0]>=r[1] for r in ranges):
                raise ValueError('invalid executed velocity contract')
            self.motion_contract={'dt_seconds':dt,'lin_vel_range':ranges[0],'ang_vel_range':ranges[1]}

    @property
    def active(self):
        return self.stages[min(self.active_index,len(self.stages)-1)]

    @property
    def complete(self):
        return self.active_index==len(self.stages)

    def observe(self,reference,step):
        if reference in self.observations and self.observations[reference]!=step:
            raise ValueError('one actual observation cannot move in time')
        if step<max(self.observations.values()):
            raise ValueError('progress observation is stale')
        self.observations[reference]=step

    def executed(self,step,*,environment_step_returned,rgb_changed,action):
        if environment_step_returned is not True or not action:
            raise ValueError('progress requires actual execution acknowledgement')
        record={'step':step,'rgb_changed':bool(rgb_changed),'action':action}
        if self.motion_contract is not None and action.get('action')=='velocity_control':
            args=action['action_args']
            raw=[float(args[k]) for k in ['linear_velocity','angular_velocity']]
            if any(not math.isfinite(v) or not -1<=v<=1 for v in raw):
                raise ValueError('executed velocity commands must be normalized to [-1,1]')
            v,w=[lo+(value+1)/2*(hi-lo) for value,(lo,hi) in zip(raw,
                [self.motion_contract['lin_vel_range'],self.motion_contract['ang_vel_range']],strict=True)]
            dt=self.motion_contract['dt_seconds']
            record['commanded_motion']={'dt_seconds':dt,'forward_metres':v*dt,'yaw_degrees_left_positive':w*dt,
                'evidence_kind':'issued_control_not_measured_camera_motion'}
        if step in self.executions and self.executions[step]!=record:
            raise ValueError('execution acknowledgement cannot change')
        self.executions[step]=record

    def record_observed_motion(self,reference,record):
        if reference not in self.observations or record.get('reference')!=reference:
            raise ValueError('measured motion must be bound to an actual current observation')
        epoch=record.get('pose_epoch')
        if not isinstance(epoch,int) or epoch<0:
            raise ValueError('invalid measured-motion epoch')
        pose=record.get('local_pose')
        if pose is not None:
            if record.get('measurement',{}).get('valid') is not True or len(pose)!=3:
                raise ValueError('local pose requires a valid measured camera motion')
            pose=[float(v) for v in pose]
            if not all(math.isfinite(v) for v in pose):
                raise ValueError('invalid measured local pose')
        entry={'pose_epoch':epoch,'local_pose':pose}
        if reference in self.observed_motion and self.observed_motion[reference]!=entry:
            raise ValueError('an actual observation cannot be assigned a different measured pose')
        self.observed_motion[reference]=entry

    def _requires_stage_translation(self):
        # A turn or a changed viewpoint can confirm a second observation, but
        # cannot by itself establish that a walking/entering stage happened.
        text=self.active.instruction.lower()
        return (self.active.kind=='navigate' or bool(re.search(
            r'\b(?:enter|go|walk|move|pass|cross|proceed|drive|travel|approach|leave|exit)\b',text))
            or bool(re.search(r'\bturn\b.*\binto\b',text)))

    def _stage_translation(self,current_reference):
        """Measure from the earliest available pose in this stage and epoch."""
        after=self.observed_motion.get(current_reference)
        if after is None or after['local_pose'] is None:
            return None
        start_step=self.observations[self.anchor]
        end_step=self.observations[current_reference]
        candidates=[(self.observations[ref],ref,pose) for ref,pose in self.observed_motion.items()
                    if start_step<=self.observations[ref]<=end_step and pose['local_pose'] is not None
                    and pose['pose_epoch']==after['pose_epoch']]
        if not candidates:
            return None
        _,reference,before=min(candidates,key=lambda row:row[0])
        a,b=before['local_pose'],after['local_pose']
        return {'start_reference':reference,'end_reference':current_reference,
                'translation_m':math.hypot(b[0]-a[0],b[1]-a[1]),'pose_epoch':after['pose_epoch'],
                'source':'actual_rgbd_camera_motion_not_issued_controls'}

    def consider(self,evidence:StageEvidence):
        if self.complete:
            self.last_status='all_stages_visually_supported'
            return False
        if (evidence.stage_id!=self.active.stage_id or evidence.before_observation!=self.anchor
                or evidence.current_observation not in self.observations):
            raise ValueError('progress evidence is not grounded in this active stage/current episode')
        if not math.isfinite(evidence.confidence) or not 0<=evidence.confidence<=1:
            raise ValueError('invalid verifier proposal confidence')
        step=self.observations[evidence.current_observation]
        if step!=max(self.observations.values()):
            raise ValueError('progress evidence must use the latest actual observation')
        # A model's self-reported confidence is only an admission heuristic.
        positive=evidence.completed and evidence.confidence>=.8 and bool(evidence.reason)
        identity_supported=evidence.target_visible
        if not self.require_current_target and evidence.target_identified is True:
            ref=evidence.identity_reference
            identity_supported=identity_supported or (ref in self.observations and self.observations[ref]<=step)
        if self.active.kind=='goal':
            positive=positive and identity_supported
        if not positive:
            self.confirmation=None
            self.last_status='not_visually_established'
            return False
        if self.confirmation is None:
            self.confirmation=(step,evidence.current_observation)
            self.last_status='needs_second_motion_separated_view'
            return False
        prior,reference=self.confirmation
        between=[x for t,x in self.executions.items() if prior<=t<step]
        motion_support=None
        if self.require_measured_motion:
            before=self.observed_motion.get(reference)
            after=self.observed_motion.get(evidence.current_observation)
            if (before is None or after is None or before['local_pose'] is None or after['local_pose'] is None
                    or before['pose_epoch']!=after['pose_epoch']):
                self.confirmation=(step,evidence.current_observation)
                self.last_status='measured_motion_unavailable_need_new_anchor'
                return False
            a,b=before['local_pose'],after['local_pose']
            translation=math.hypot(b[0]-a[0],b[1]-a[1])
            yaw=abs(math.atan2(math.sin(b[2]-a[2]),math.cos(b[2]-a[2])))
            active_look=yaw>=math.radians(15)
            motion_support={'translation_m':translation,'yaw_rad':yaw,'pose_epoch':after['pose_epoch'],
                            'source':'actual_rgbd_camera_motion_not_issued_controls'}
            if not active_look and translation<.10:
                self.last_status='insufficient_measured_camera_motion'
                return False
        else:
            active_look=any(abs(x.get('commanded_motion',{}).get('yaw_degrees_left_positive',0))>=15 for x in between)
        separation=1 if active_look else 3
        if (step-prior<separation or reference==evidence.current_observation or len(between)<separation
                or not any(x['rgb_changed'] for x in between)):
            self.last_status='insufficient_new_execution_evidence'
            return False
        stage_motion=None
        if self.require_measured_motion and self._requires_stage_translation():
            stage_motion=self._stage_translation(evidence.current_observation)
            if stage_motion is None or stage_motion['translation_m']<.10:
                self.last_status='stage_translation_not_measured'
                return False
        self.completed.append({'stage':asdict(self.active),'supporting_views':[reference,evidence.current_observation],
                               'evidence':asdict(evidence),'target_identity_supported':bool(identity_supported),
                               'measured_motion_support':motion_support,
                               'measured_stage_translation':stage_motion,
                               'support_level':('temporal_visual_and_measured_motion_support_not_ground_truth'
                                                if self.require_measured_motion else 'temporal_visual_support_not_ground_truth')})
        self.active_index+=1
        self.anchor=evidence.current_observation
        self.confirmation=None
        self.last_status='stage_advanced' if not self.complete else 'all_stages_visually_supported'
        return True

    def accept_global_stop(self):
        return self.complete and self.completed[-1]['target_identity_supported']

    def context(self):
        return json.dumps({'current_episode':self.episode_id,'original_instruction':self.instruction,
            'completed_stages':[x['stage']['instruction'] for x in self.completed],
            'active_stage':None if self.complete else asdict(self.active),'complete':self.complete,
            'status':self.last_status,'support_kind':'visual_tool_with_temporal_checks_not_privileged_goal_measurement'},ensure_ascii=False)


class QwenTaskDecomposer:
    def __init__(self,vision,*,require_verbatim=False):
        self.vision=vision
        self.last_raw=''
        self.require_verbatim=require_verbatim

    def __call__(self,instruction):
        prompt='''Split this CURRENT robot navigation instruction into 1..8 ordered atomic stages. The RGB camera is FIRST PERSON; the robot/person, hands and feet are not visible. Preserve route constraints and landmarks in order. Never require seeing a person or touching handles. Use navigate or turn, and label ONLY the final stage goal. Do not duplicate the final stage. Return JSON {"stages":[{"instruction":"one original subtask","criterion":"first-person visible evidence for this subtask only","kind":"navigate"}]}. Intermediate criteria must not demand a later destination. No invented landmarks or extra stopping requirements.
CURRENT INSTRUCTION: '''+instruction
        if self.require_verbatim:
            prompt+='\nEvery stage instruction MUST quote a contiguous literal span of the original instruction. Split explicit clauses only; do not invent a room center, threshold or extra turn.'
        self.last_raw=self.vision(prompt,[],max_new_tokens=700)
        rows=parse_object(self.last_raw)['stages']
        if not 1<=len(rows)<=8:
            raise ValueError('invalid number of current task stages')
        compact=[]
        for row in rows:
            if not compact or re.sub(r'\W','',str(row['instruction']).lower())!=re.sub(r'\W','',str(compact[-1]['instruction']).lower()):
                compact.append(row)
        stages=[]
        position=0
        original=' '.join(re.findall(r'\w+',instruction.lower()))
        covered=set()
        for i,row in enumerate(compact):
            text=str(row['instruction']).strip()
            if not text:
                raise ValueError('empty task stage')
            if self.require_verbatim:
                literal=' '.join(re.findall(r'\w+',text.lower()))
                found=original.find(literal,position)
                if found<0 or (found and original[found-1]!=' ') or (found+len(literal)<len(original) and original[found+len(literal)]!=' '):
                    raise ValueError('decomposition invented or reordered an instruction clause')
                covered.update(range(found,found+len(literal)))
                position=found+len(literal)
            kind='goal' if i==len(compact)-1 else ('turn' if row.get('kind')=='turn' else 'navigate')
            if self.require_verbatim and i<len(compact)-1 and re.match(r'(?:(?:and|then|next|finally)\s+)*turn\b',text,re.IGNORECASE):
                kind='turn'
            # Keep the original subtask text; don't inherit free-form criteria
            # that add visible-body requirements or a later target.
            criterion='First-person before/after views establish this subtask only: '+text
            if kind=='goal':
                stop=re.search(r'\bstop\b.*',instruction,flags=re.IGNORECASE)
                criterion+=' Requested final relation: '+(stop.group() if stop else text)
            stages.append(TaskStage(f'stage-{i}',text,criterion,kind))
        if self.require_verbatim:
            for word in re.finditer(r'\w+',original):
                if word.group() not in {'and','then','next','finally','please'} and not set(range(*word.span())).issubset(covered):
                    raise ValueError('decomposition omitted an instruction constraint')
            if len(stages)==1 and re.search(r'\bthen\b',instruction,flags=re.IGNORECASE):
                raise ValueError('an explicit task sequence cannot be verified as one undivided stage')
        stages=tuple(stages)
        return stages


class QwenProgressVerifier:
    def __init__(self,vision,images):
        self.vision,self.images=vision,images
        self.last_trace={}

    def __call__(self,progress:TaskProgress,current_reference):
        payload={'active_stage':asdict(progress.active),
                 'completed_stages':[x['stage']['instruction'] for x in progress.completed],
                 'recent_actual_dispatches':[{'step':r['step'],'rgb_changed':r['rgb_changed'],
                     'action_type':r['action'].get('action','unknown'),'commanded_motion':r.get('commanded_motion')}
                     for r in list(progress.executions.values())[-12:]],
                 'before_is_stage_anchor_not_an_expected_future':True}
        scope=('Judge only this INTERMEDIATE FIRST-PERSON navigation subtask. Completion means the camera has achieved the stated local movement. An initial room can be out of view after leaving it. Later destinations are irrelevant. '
               if progress.active.kind!='goal' else
               'Judge only this FINAL FIRST-PERSON navigation subtask. Completion requires identifying the actual requested target and establishing the requested stopping relation in the CURRENT view. Movement or generic furniture does not establish arrival. An unseen target is unknown, not complete. ')
        prompt=scope+'''The robot is the camera and is never visible. Compare the ACTUAL STAGE START with the ACTUAL CURRENT VIEW, using the intervening images and actual controls to establish the transition. A commanded yaw in DEGREES is LEFT when positive, RIGHT when negative; forward distance is in METRES. These are issued commands, not measured camera motion; require visual support. Missing commanded motion means the units are unavailable. Unchanged initial views cannot establish movement. A policy STOP or familiarity alone is not evidence. Write concrete visual evidence first, then a consistent decision; admit uncertainty.
Return one JSON object with these fields: reason (string describing visual evidence or uncertainty), completed (boolean, true exactly when completion is visually supported), target_visible (boolean, true only when the final requested target is identifiable in the current view; set false for an intermediate stage), confidence (number from 0 to 1 expressing confidence in your verdict; this is an uncalibrated estimate). There are no default values.
CONTEXT: '''+json.dumps(payload)
        image_items=[('ACTUAL STAGE START',self.images.get(progress.anchor))]
        current_step=progress.observations[current_reference]
        used={progress.anchor,current_reference}
        for offset in [6,1]:
            candidates=[(step,ref) for ref,step in progress.observations.items() if step<=current_step-offset and ref not in used]
            if candidates:
                step,reference=max(candidates)
                used.add(reference)
                image_items.append((f'ACTUAL EARLIER EPISODE VIEW at step {step}',self.images.get(reference)))
        image_items.append(('ACTUAL CURRENT VIEW',self.images.get(current_reference)))
        raw=self.vision(prompt,image_items)
        value=parse_object(raw)
        if not isinstance(value['completed'],bool) or not isinstance(value['target_visible'],bool):
            raise ValueError('verifier flags must be booleans')
        evidence=StageEvidence(progress.active.stage_id,progress.anchor,current_reference,value['completed'],
                               value['target_visible'],float(value['confidence']),str(value['reason']))
        self.last_trace={'context':payload,'raw_output':raw,'evidence':asdict(evidence)}
        return evidence
