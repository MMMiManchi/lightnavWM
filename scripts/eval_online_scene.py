"""Causal four-arm scene adaptation on real admitted event sequences, eight GPUs.

Every episode is forecast with a fixed scene-model version. Actual outcomes
are scored and journaled, then reserved-support-gated SGD can change the model
for later episodes. This is recorded online-learning replay, not new navigation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.event_residual_data import causal_memory_indices, materialize_memory  # noqa: E402
from experiments.world_model.event_runtime import load_residual_predictor  # noqa: E402
from experiments.world_model.event_train import digest, metrics, read_rows, save  # noqa: E402
from experiments.world_model.familiarity import ordered_scene_episodes  # noqa: E402
from experiments.world_model.online_scene import OnlineSceneLearner, SceneAdapterConfig  # noqa: E402
from scripts.smoke_event_lightnav import raw_features, select_checkpoint  # noqa: E402

ARMS = (("frozen",False,False),("memory_only",True,False),
        ("adapt_only",False,True),("memory_and_adapt",True,True))


def worker(root, number):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("scene adaptation requires CUDA; no CPU fallback")
    config=json.loads((root/'config.json').read_text())
    base=Path(config['base_run'])
    output=root/f'worker_{number:02d}'
    output.mkdir(exist_ok=False)
    pairs=read_rows(base/'pairs.jsonl')
    arrays=np.load(base/'tensors.npz')
    grouped=ordered_scene_episodes(pairs,config['episode_order'])
    scenes=sorted(grouped)[number::8]
    ids=np.array([i for i,p in enumerate(pairs) if p['scene'] in scenes])
    local={int(i):j for j,i in enumerate(ids)}
    predictor=load_residual_predictor(Path(config['checkpoint']),'cuda:0')
    features=raw_features(base)
    columns={}
    for path in sorted(Path(config['training_run']).glob('columns_*.npz')):
        with np.load(path) as data:
            columns.update(zip(data['ids'].tolist(),data['columns'],strict=True))
    raw=np.stack([[features[k] if k else np.zeros(1152,dtype=np.float32)
                   for k in pairs[i]['history_image_ids']] for i in ids]).astype(np.float32)
    spatial=np.stack([columns[pairs[i]['history_image_ids'][-1]] for i in ids]).astype(np.float32)
    indices=causal_memory_indices(pairs,arrays['history'],arrays['branch'],[tuple(e) for e in config['episode_order']])
    frozen={}
    condition=[]
    with torch.inference_mode():
        for use_memory in [False,True]:
            selected=indices[ids] if use_memory else np.full((len(ids),3),-1,dtype=np.int64)
            memory=materialize_memory(selected,arrays['history'],arrays['target'],arrays['branch'])
            outputs=[]
            for start in range(0,len(ids),48):
                sl=slice(start,start+48)
                args=[torch.from_numpy(x[sl]).cuda() for x in
                      [raw,arrays['valid'][ids],arrays['branch'][ids],spatial]]
                result=predictor(*args,**{k:torch.from_numpy(v[sl]).cuda() for k,v in memory.items()},return_aux=True)
                outputs.append(result['prediction'].cpu().numpy())
                if use_memory:
                    condition.append(result['condition'].cpu().numpy())
            frozen[use_memory]=np.concatenate(outputs)
    condition=np.concatenate(condition)
    del raw,spatial,columns,features
    predictions=np.zeros((4,len(ids),1152),dtype=np.float32)
    versions=np.zeros((4,len(ids)),dtype=np.int64)
    audit=[]
    learner_config=SceneAdapterConfig(update_steps=config['update_steps'])
    for scene in scenes:
        for arm_index,(arm,use_memory,learning) in enumerate(ARMS):
            folder=output/scene/arm
            updates=[]
            for (run,uid),members in grouped[scene]:
                learner=OnlineSceneLearner(folder,scene_id=scene,feature_contract=predictor.feature_contract_id,
                    base_hash=config['checkpoint_sha256'],target_kind='next_event',config=learner_config,
                    device='cuda:0',learning=learning)
                learner.begin_episode(f'{run}|{uid}')
                for index in members:
                    j=local[index]
                    pair=pairs[index]
                    ticket=learner.predict(sample_id=pair['sample_id'],source_observation=pair['history_observation_ids'][-1],
                        source_step=pair['origin_index'],current=arrays['history'][index,-1],condition=condition[j],
                        branch=arrays['branch'][index],horizon=0,basis=frozen[use_memory][j])
                    predictions[arm_index,j]=ticket.prediction
                    versions[arm_index,j]=ticket.model_version
                    proof={'kind':'next_event','real_observation':True,'episode_id':f'{run}|{uid}',
                           'source_observation':ticket.source_observation,'target_observation':pair['target_observation_id'],
                           'source_step':pair['origin_index'],'target_step':pair['target_index'],
                           'admitted_execution_ref':f"admitted-recorded-span:{pair['sample_id']}"}
                    # This is the first use of this query's actual outcome.
                    learner.observe(ticket,arrays['target'][index],proof)
                update=learner.finish_episode()
                updates.append({'episode':uid,**update})
                learner.close()  # Prove both parameters and replay survive reopening.
            audit.append({'scene':scene,'arm':arm,'episodes':len(updates),
                          'published_updates':sum(u['status']=='published' for u in updates),
                          'optimizer_steps':sum(u.get('optimizer_steps',0) for u in updates),
                          'updates':updates})
            print(json.dumps({k:audit[-1][k] for k in ['scene','arm','published_updates','optimizer_steps']}),flush=True)
    np.savez_compressed(output/'predictions.npz',ids=ids,predictions=predictions,versions=versions)
    save(output/'learning_audit.json',audit)
    save(output/'complete.json',{'worker':number,'state':'complete'})


def summarize(root):
    config=json.loads((root/'config.json').read_text())
    base=Path(config['base_run'])
    pairs=read_rows(base/'pairs.jsonl')
    arrays=np.load(base/'tensors.npz')
    chunks=[np.load(root/f'worker_{i:02d}/predictions.npz') for i in range(8)]
    ids=np.concatenate([c['ids'] for c in chunks])
    order=np.argsort(ids)
    assert np.array_equal(ids[order],np.arange(len(pairs)))
    rows=[]
    for partition in ['train_dev','train_fit']:
        keep=np.array([p['partition']==partition for p in pairs])
        target=arrays['target'][keep]
        labels=arrays['target_labels'][keep]
        scenes=[p['scene'] for p in pairs if p['partition']==partition]
        for arm_index,(arm,_m,_a) in enumerate(ARMS):
            values=np.concatenate([c['predictions'][arm_index] for c in chunks])[order][keep]
            versions=np.concatenate([c['versions'][arm_index] for c in chunks])[order][keep]
            rows.append({'partition':partition,'arm':arm,'queries':int(keep.sum()),
                         'queries_after_published_update':int((versions>0).sum()),
                         'metrics':metrics(values,target,labels,scenes)})
    audit=[r for i in range(8) for r in json.loads((root/f'worker_{i:02d}/learning_audit.json').read_text())]
    frozen=json.loads((root/'input_manifest.json').read_text())
    assert all(digest(Path(p))==h for p,h in frozen.items())
    save(root/'results.json',{'metrics':rows,'published_scene_updates':sum(r['published_updates'] for r in audit),
        'real_optimizer_steps':sum(r['optimizer_steps'] for r in audit),'global_inputs_and_weights_unchanged':True,
        'episode_forecasts_precede_own_target_learning':True,'recorded_replay_not_new_navigation':True,
        'current_future_duration_not_input_to_event_predictor':True,
        'development_previously_examined_not_blind':True})
    print(json.dumps({'state':'complete','rows':rows}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',type=int)
    parser.add_argument('--update-steps',type=int,default=40)
    args=parser.parse_args()
    root=args.output.resolve()
    if args.worker is not None:
        if not 0<=args.worker<8:
            parser.error('worker must be 0..7')
        worker(root,args.worker)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count()<8:
        raise RuntimeError('the registered parallel replay requires eight CUDA devices')
    if not 10<=args.update_steps<=200:
        parser.error('update steps must be 10..200')
    root.mkdir(parents=True,exist_ok=False)
    training=args.training_run.resolve()
    training_config=json.loads((training/'config.json').read_text())
    checkpoint,_=select_checkpoint(training)
    config={'training_run':str(training),'base_run':training_config['base_run'],'checkpoint':str(checkpoint),
            'checkpoint_sha256':digest(checkpoint),'episode_order':json.loads((training/'episode_order.json').read_text()),
            'update_steps':args.update_steps,'adapter':asdict(SceneAdapterConfig(update_steps=args.update_steps)),
            'arms':[x[0] for x in ARMS],
            'event_horizon':0,'fixed_hyperparameters_not_selected_on_development':True}
    save(root/'config.json',config)
    sources=[Path(__file__),REPO/'experiments/world_model/online_scene.py',
             REPO/'experiments/world_model/event_anchored.py',REPO/'experiments/world_model/event_residual.py',
             REPO/'experiments/world_model/event_runtime.py',REPO/'experiments/world_model/event_residual_data.py',
             REPO/'experiments/world_model/familiarity.py',checkpoint,Path(config['base_run'])/'tensors.npz',
             Path(config['base_run'])/'pairs.jsonl',training/'episode_order.json']
    save(root/'input_manifest.json',{str(p):digest(p) for p in sources})
    for path in sources:
        if path.suffix=='.py':
            destination=root/'source_snapshot'/path.relative_to(REPO)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination)
    jobs=[]
    streams=[]
    try:
        for i in range(8):
            environment=os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            stream=(root/f'worker_{i:02d}.log').open('w')
            streams.append(stream)
            jobs.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                '--training-run',str(training),'--output',str(root),'--worker',str(i)],
                env=environment,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT))
        codes=[j.wait() for j in jobs]
        if any(codes):
            raise RuntimeError(f'worker failures {codes}; inspect retained logs')
        summarize(root)
    finally:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
        for stream in streams:
            stream.close()


if __name__=='__main__':
    main()
