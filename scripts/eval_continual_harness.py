"""Fresh multi-episode LightNav runs with real writeback, SGD and visual progress.

Eight workers: four fixed scenes x memory-only / memory+scene-SGD. Earlier
episodes are newly executed by this policy; fixed-duration outcomes never
claim arrival at official nodes. Pose/goal metrics are evaluation-only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

from experiments.world_model.continual_controller import ContinualEpisodeController  # noqa: E402
from experiments.world_model.event_runtime import HarnessLightNavExecutor, load_residual_predictor  # noqa: E402
from experiments.world_model.event_train import digest, read_rows, save  # noqa: E402
from experiments.world_model.familiarity import choose_live_scenes, ordered_scene_episodes  # noqa: E402
from experiments.world_model.online_input import online_info, online_observation  # noqa: E402
from experiments.world_model.online_scene import OnlineSceneLearner, SceneAdapterConfig  # noqa: E402
from scripts.smoke_event_lightnav import build_base, select_checkpoint, serial  # noqa: E402


def prepare(args):
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    training=args.training_run.resolve()
    previous=json.loads((training/'config.json').read_text())
    base=Path(previous['base_run'])
    pairs=read_rows(base/'pairs.jsonl')
    order=json.loads((training/'episode_order.json').read_text())
    grouped=ordered_scene_episodes(pairs,order)
    selected=choose_live_scenes(pairs,order,count=4)
    dataset=REPO/'data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz'
    raw=json.loads(gzip.decompress(dataset.read_bytes()))['episodes']
    dataset_rank={(Path(p['scene_id']).parent.name,str(p['episode_id'])):i for i,p in enumerate(raw)}
    for scene in selected:
        representatives=[]
        seen=set()
        for key,rows in grouped[scene['scene']]:
            route=pairs[rows[0]]['route_group_id']
            if route not in seen:
                representatives.append(key)
                seen.add(route)
        chosen=representatives[:args.episodes] if len(representatives)>=args.episodes else [e for e,_r in grouped[scene['scene']]][:args.episodes]
        tasks=[{'scene':scene['scene'],'episode_id':uid.split('|')[-1]} for _run,uid in chosen]
        scene['tasks']=sorted(tasks,key=lambda t:dataset_rank[(t['scene'],t['episode_id'])])
    checkpoint,_=select_checkpoint(training)
    model_path=REPO/'checkpoints/LightNav-0'
    controller_identity={str(p.relative_to(model_path)):digest(p) for p in
                         [model_path/'config.json',model_path/'eval_config.json',*sorted(model_path.glob('*.safetensors'))]}
    sim_config=REPO/'experiments/world_model/configs/collect_train_go2w_radius010.yaml'
    control_contract={'schema':'lightnav-observed-span/0.1','skill_id':'frozen_lightnav_visible_entrance',
        'horizon':6,'simulation_config_sha256':digest(sim_config),
        'waypoint_conversion_sha256':digest(REPO/'src/lightnav/velocity.py')}
    model_binding=hashlib.sha256(json.dumps({'predictor':digest(checkpoint),'controller':controller_identity,
        'control_contract':control_contract},sort_keys=True).encode()).hexdigest()
    config={'output':str(root),'training_run':str(training),'base_run':str(base),'checkpoint':str(checkpoint),
            'model_binding':model_binding,'controller_weights':controller_identity,'model_path':str(model_path),
            'encoder_snapshot':previous['encoder_snapshot'],'planner_model':str(REPO/'checkpoints/world_model_planner/Qwen3-VL-4B-Instruct'),
            'habitat_python':'/home/szgx/miniconda3/envs/habitat/bin/python',
            'sim_config':str(sim_config),'control_contract':control_contract,
            'base_port':5790,'scenes':selected,'episodes':args.episodes,'live_steps':args.live_steps,
            'horizon':6,'planning_interval':12,'progress_interval':6,'observation_budget':2,
            'experience_source':'fresh actually executed closed-loop spans; no old teacher memory seeded',
            'target_kind':'observed_control_span','world_pose_goal_metrics_not_policy_inputs':True,
            'scene_selection':'dev-first lexical first four eligible scenes; prefer distinct recorded route groups for task coverage',
            'study':'engineering paired pilot, not blind navigation benchmark or a significance test'}
    save(root/'config.json',config)
    modules=['online_scene','continual_controller','continual_queries','entrance_alignment','task_progress',
             'predictive_harness','predictive_queries','visual_harness','scene_memory','event_runtime','event_anchored','event_residual',
             'task_planning','online_input','feature_cache']
    sources=[Path(__file__),checkpoint,sim_config,REPO/'src/lightnav/velocity.py',dataset,
             *[REPO/f'experiments/world_model/{m}.py' for m in modules]]
    save(root/'input_manifest.json',{str(p):digest(p) for p in sources})
    for p in sources:
        if p.suffix in ('.py','.yaml'):
            target=root/'source_snapshot'/p.relative_to(REPO)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
    print(json.dumps({'scenes':selected,'runs':8*args.episodes,'config':str(root/'config.json')}),flush=True)


def episode(config,selection,task,number,env,factory,predictor,encoder,processor,vision,learner,output):
    from PIL import Image

    from lightnav.velocity import first_waypoint_to_velocity_cmd

    obs,info=env.reset()
    if str(info.get('episode_id'))!=task['episode_id'] or task['scene'] not in info.get('scene_id',''):
        raise ValueError('fresh episode order differs from registered tasks')
    executor=HarnessLightNavExecutor(factory(online_info(info)))
    executor.reset(online_observation(obs))
    episode_id=f'episode-{number}'
    learner.begin_episode(episode_id)
    model_before=learner.parameters_hash()
    version_before=learner.version
    memory_before=int(learner.db.execute('select count(*) from observations').fetchone()[0])
    controller=ContinualEpisodeController(executor=executor,predictor=predictor,learner=learner,vision=vision,
        episode_id=episode_id,output=output/f'episode_{number:02d}',initial_rgb=np.asarray(obs['rgb']),horizon=config['horizon'])
    initial=hashlib.sha256(np.asarray(obs['rgb']).tobytes()).hexdigest()
    initial_distance=info.get('distance_to_goal')
    steps=[]
    observations_used=0
    stop_reason='client_budget'
    calls_start=len(vision.calls)
    for step in range(config['live_steps']):
        rgb=np.asarray(obs['rgb'])
        with torch.inference_mode():
            view=encoder(processor(images=[rgb],return_tensors='pt')['pixel_values'].cuda())
        feature=view.pooler_output.float().cpu().numpy()[0]
        columns=view.last_hidden_state.reshape(1,27,27,1152).mean(1).float().cpu()
        depth=np.asarray(obs['depth']) if 'depth' in obs else None
        if depth is not None:
            # Installed Habitat config: linear depth, min=0/max=10, normalized.
            # Common scaling cancels in PnP; clipped far readings are invalid.
            depth=np.where(depth>=.999,np.nan,depth)
        controller.observe(step,rgb,feature,columns,depth)
        if step and step%config['progress_interval']==0:
            controller.check_progress(step)
        record={'step':step,'model_version':learner.version,'task_stage':controller.progress.active.stage_id}
        action=None
        if controller.progress.accept_global_stop():
            executor.policy.observe(online_observation(obs),online_info(info))
            action=executor.policy.stop_action()
            record.update(stop_source='Harness_temporal_visual_goal_support',stop_dispatched=True)
        elif step%config['planning_interval']==0:
            planned=controller.plan(step)
            if (planned is not None and planned.action=='observe' and abs(planned.observation_turn_rad)>1e-6
                    and observations_used<config['observation_budget']):
                observations_used+=1
                executor.restore_global_goal()
                executor.policy.observe(online_observation(obs),online_info(info))
                action=first_waypoint_to_velocity_cmd(np.array([0.,0.,planned.observation_turn_rad]),
                    executor.policy.dt,executor.policy.lin_vel_range,executor.policy.ang_vel_range)
                record['action_source']='Harness_requested_observation'
        if action is None:
            action,details=executor.propose_action(online_observation(obs),online_info(info))
            record['rvq_codes']=details.get('rvq_codes')
            if action is None and executor.active_goal is not None and executor.last_tool_status=='stop_proposed':
                record.update(stop_proposal_source='LightNav_local',local_stop_intercepted=True)
                before=len(executor.policy.agent._history_frame_ids)
                executor.restore_global_goal()
                controller.harness.report_execution(f'At step {step} local STOP was unverified; original task restored.')
                action,details=executor.replan_observed_frame()
                assert before==len(executor.policy.agent._history_frame_ids)
            if action is None and executor.last_tool_status=='stop_proposed':
                if not controller.progress.complete:
                    prior_stage=controller.progress.active_index
                    controller.check_progress(step)
                    if controller.progress.active_index!=prior_stage and not controller.progress.complete:
                        controller.check_progress(step)
                record.update(stop_proposal_source='LightNav_original_task',stop_guard='TaskProgress',
                              progress_at_stop=json.loads(controller.progress.context()))
                if controller.progress.accept_global_stop():
                    action=executor.policy.stop_action()
                    record.update(stop_source='LightNav_original_task',stop_dispatched=True)
                elif observations_used<config['observation_budget']:
                    observations_used+=1
                    executor.restore_global_goal()
                    angle=-np.pi/6 if 'right' in controller.progress.active.instruction.lower() else np.pi/6
                    action=first_waypoint_to_velocity_cmd(np.array([0.,0.,angle]),executor.policy.dt,
                        executor.policy.lin_vel_range,executor.policy.ang_vel_range)
                    record.update(global_stop_intercepted=True,action_source='Harness_bounded_observation_after_STOP')
                else:
                    planned=controller.plan(step)
                    if planned is not None and planned.goal is not None:
                        action,details=executor.replan_observed_frame()
                    if action is None:
                        record.update(global_stop_intercepted=True,client_abstained=True)
                        stop_reason='unverified_stop_no_supported_continuation'
                        steps.append(record)
                        break
            elif action is None:
                record['tool_failure']=executor.last_tool_status
                steps.append(record)
                stop_reason='tool_failure'
                break
        local_ref=executor.active_goal.source_observation if executor.active_goal is not None else None
        old=hashlib.sha256(rgb.tobytes()).hexdigest()
        action=serial(action)
        obs,_reward,terminated,truncated,info=env.step(action)
        changed=old!=hashlib.sha256(np.asarray(obs['rgb']).tobytes()).hexdigest()
        controller.executed(step,action,rgb_changed=changed,local_goal_ref=local_ref)
        record.update(action=action,environment_step_returned=True,terminated=terminated,truncated=truncated,
                      rgb_changed=changed,native_history_frames=len(executor.policy.agent._history_frame_ids))
        record['evaluation_only']=serial({k:info[k] for k in ['distance_to_goal','success','spl'] if k in info})
        steps.append(record)
        controller.persist()
        save(controller.output/'steps.json',steps)
        if step%12==0:
            save(output/'progress.json',{'episode':number,'step':step,'model_version':learner.version,
                'memory_before':memory_before,'new_verified_spans':len(controller.verified_spans),
                'stage':controller.progress.active_index})
        if terminated or truncated:
            stop_reason='dispatched_stop' if record.get('stop_dispatched') else 'environment_limit'
            break
    executed=sum(r.get('environment_step_returned',False) for r in steps)
    if controller.current_reference!=f'{episode_id}:view:{executed}':
        with torch.inference_mode():
            view=encoder(processor(images=[np.asarray(obs['rgb'])],return_tensors='pt')['pixel_values'].cuda())
        controller.observe(executed,np.asarray(obs['rgb']),view.pooler_output.float().cpu().numpy()[0],
            view.last_hidden_state.reshape(1,27,27,1152).mean(1).float().cpu(),
            np.asarray(obs['depth']) if 'depth' in obs else None)
    Image.fromarray(np.asarray(obs['rgb'])).save(controller.output/'final.png')
    controller.persist()
    save(controller.output/'steps.json',steps)
    save(controller.output/'visual_calls.json',vision.calls[calls_start:])
    update=learner.finish_episode()
    queries=[q for row in controller.planning_trace for q in row.get('queries',[])]
    row={'scene':selection['scene'],'partition':selection['partition'],'episode_index':number,
         'dataset_episode_id':task['episode_id'],'learning':learner.learning,'executed_steps':executed,
         'initial_rgb_sha256':initial,'initial_distance_evaluation_only':initial_distance,
         'memory_before':memory_before,'memory_after':int(learner.db.execute('select count(*) from observations').fetchone()[0]),
         'model_version_before':version_before,'model_version_after':learner.version,
         'parameters_changed':model_before!=learner.parameters_hash(),'update':update,
         'verified_observed_spans':len(controller.verified_spans),'query_count':len(queries),
         'history_supported_queries':sum(q['supported_by_history'] for q in queries),
         'adapted_hypothesis_queries':sum(q.get('scene_model_hypothesis_without_direct_match',False) for q in queries),
         'stage_advances':controller.progress.active_index,'stage_count':len(controller.progress.stages),
         'stop_interceptions':sum(r.get('global_stop_intercepted',False) or r.get('local_stop_intercepted',False) for r in steps),
         'observation_turns':observations_used,'stop_reason':stop_reason,
         'evaluation_only':serial({k:info[k] for k in ['distance_to_goal','success','spl'] if k in info}),
         'world_pose_goal_not_input':True,'node_arrivals_not_asserted_from_fixed_spans':True}
    save(controller.output/'result.json',row)
    print(json.dumps(row),flush=True)
    return row


def worker(config,number):
    from transformers import SiglipImageProcessor, SiglipVisionModel

    from experiments.world_model.feature_cache import AUDITED_PREPROCESSOR, SIGLIP_WEIGHT_SHA256
    from experiments.world_model.visual_harness import LocalQwenVision
    from lightnav.habitat.remote_env import RemoteEnvClient

    if not torch.cuda.is_available():
        raise RuntimeError('real continual navigation requires CUDA')
    torch.set_num_threads(2)
    output=Path(config['output'])/f'worker_{number:02d}'
    output.mkdir(exist_ok=False)
    selection=config['scenes'][number%4]
    learning=number>=4
    save(output/'episode_allowlist.json',selection['tasks'])
    server_env=os.environ.copy()
    server_env.pop('CUDA_VISIBLE_DEVICES',None)
    server_env.update(PYTHONPATH=str(REPO/'habitat_server'),HABITAT_SIM_GPU_ID=str(number))
    port=config['base_port']+number
    ready=output/'habitat.ready'
    stream=(output/'habitat.log').open('w')
    server=subprocess.Popen([config['habitat_python'],'-u','-m','lightnav_habitat.serve','--task','vlnce',
        '--config',config['sim_config'],'--split','train','--port',str(port),'--max-steps',str(config['live_steps']),
        '--episode-ids',str(output/'episode_allowlist.json'),'--ready-file',str(ready)],env=server_env,cwd=REPO,
        stdout=stream,stderr=subprocess.STDOUT)
    env=None
    try:
        deadline=time.monotonic()+180
        while not ready.exists():
            if server.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('simulator not ready; inspect retained habitat.log')
            time.sleep(.5)
        _engine,_bundle,factory=build_base(SimpleNamespace(model_path=Path(config['model_path']),backend='vllm_local'))
        predictor=load_residual_predictor(Path(config['checkpoint']),'cuda:0')
        snapshot=Path(config['encoder_snapshot'])
        if digest(snapshot/'model.safetensors')!=SIGLIP_WEIGHT_SHA256:
            raise ValueError('frozen feature encoder changed')
        encoder=SiglipVisionModel.from_pretrained(snapshot,local_files_only=True,dtype=torch.float32).cuda().eval()
        processor=SiglipImageProcessor.from_json_file(str(AUDITED_PREPROCESSOR))
        vision=LocalQwenVision(Path(config['planner_model']))
        env=RemoteEnvClient(f'tcp://127.0.0.1:{port}',timeout_ms=120000,recv_retries=0)
        results=[]
        for index,task in enumerate(selection['tasks']):
            learner=OnlineSceneLearner(output/'scene_model',scene_id=selection['scene'],
                feature_contract=predictor.feature_contract_id,base_hash=config['model_binding'],
                target_kind='observed_control_span',config=SceneAdapterConfig(),device='cuda:0',learning=learning)
            try:
                results.append(episode(config,selection,task,index,env,factory,predictor,encoder,processor,vision,learner,output))
            finally:
                learner.close()
            save(output/'results.json',results)
        save(output/'complete.json',{'worker':number,'state':'complete'})
    finally:
        if env is not None:
            env.close()
        if server.poll() is None:
            server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        stream.close()


def summarize(config):
    root=Path(config['output'])
    rows=[r for i in range(8) for r in json.loads((root/f'worker_{i:02d}/results.json').read_text())]
    manifest=json.loads((root/'input_manifest.json').read_text())
    if not all(digest(Path(p))==h for p,h in manifest.items()):
        raise ValueError('registered global inputs changed during the live trial')
    for scene in range(4):
        a=json.loads((root/f'worker_{scene:02d}/results.json').read_text())
        b=json.loads((root/f'worker_{scene+4:02d}/results.json').read_text())
        assert all(x['initial_rgb_sha256']==y['initial_rgb_sha256'] for x,y in zip(a,b,strict=True))
    summary={'episodes':rows,'new_real_observed_spans':sum(r['verified_observed_spans'] for r in rows),
             'real_optimizer_steps':sum(r['update'].get('optimizer_steps',0) for r in rows),
             'published_updates':sum(r['update']['status']=='published' for r in rows),
             'later_episodes_using_updated_scene_weights':sum(r['model_version_before']>0 for r in rows),
             'frozen_global_inputs_unchanged':True,'paired_initial_RGB_identical':True,
             'engineering_pilot_not_formal_navigation_gain':True}
    save(root/'results.json',summary)
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','launch','worker','summarize'],required=True)
    parser.add_argument('--training-run',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--episodes',type=int,default=3)
    parser.add_argument('--live-steps',type=int,default=120)
    args=parser.parse_args()
    if args.phase=='prepare':
        if args.training_run is None or args.output is None or not 1<=args.episodes<=3 or not 30<=args.live_steps<=200:
            parser.error('prepare requires training-run, new output, episodes 1..3 and live-steps 30..200')
        prepare(args)
        return
    if args.config is None or not 0<=args.worker<8:
        parser.error('config and worker 0..7 required')
    config=json.loads(args.config.read_text())
    if args.phase=='worker':
        worker(config,args.worker)
    elif args.phase=='summarize':
        summarize(config)
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count()<8:
            raise RuntimeError('this registered pilot uses all eight GPUs')
        root=Path(config['output'])
        jobs=[]
        streams=[]
        try:
            for i in range(8):
                environment=os.environ.copy()
                environment.update(CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn')
                stream=(root/f'worker_{i:02d}.log').open('w')
                streams.append(stream)
                jobs.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--phase','worker',
                    '--config',str(args.config.resolve()),'--worker',str(i)],env=environment,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT))
            codes=[job.wait() for job in jobs]
            if any(codes):
                raise RuntimeError(f'workers failed {codes}; keep all logs for review')
            summarize(config)
        finally:
            for job in jobs:
                if job.poll() is None:
                    job.terminate()
            for stream in streams:
                stream.close()


if __name__=='__main__':
    main()
