"""Causal replay of recorded episodes with persistent real-experience memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

from experiments.world_model.event_memory import EpisodeMemorySession  # noqa: E402
from experiments.world_model.event_train import metrics  # noqa: E402
from experiments.world_model.scene_memory import PersistentSceneMemory  # noqa: E402


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root,output = args.run_dir.resolve(),args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    source_paths=[root/'pairs.jsonl',root/'tensors.npz',root/'feature_contract.json',Path(__file__),
                  REPO/'output/world_model_data/formal_dev_20261028_01/episodes.jsonl',
                  REPO/'experiments/world_model/scene_memory.py',REPO/'experiments/world_model/event_memory.py',
                  REPO/'experiments/world_model/event_train.py']
    source_paths += sorted(root.glob('tssm_branch_seed*/best_prior_predictions.npy'))
    input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    arrays=np.load(root/'tensors.npz')
    pairs=[json.loads(s) for s in (root/'pairs.jsonl').read_text().splitlines()]
    episodes=[json.loads(s) for s in (REPO/'output/world_model_data/formal_dev_20261028_01/episodes.jsonl').read_text().splitlines()]
    dev_ids=np.flatnonzero(~arrays['fit'])
    dev_local={int(global_id):local for local,global_id in enumerate(dev_ids)}
    by_episode=defaultdict(list)
    for i in dev_ids:
        by_episode[pairs[i]['episode_uid']].append(int(i))
    ordered=[r for r in episodes if r['episode_uid'] in by_episode]
    assert len(ordered)==len(by_episode)
    contract=hashlib.sha256(arrays['feature_mean'].tobytes()+arrays['feature_scale'].tobytes()+
                            (root/'feature_contract.json').read_bytes()).hexdigest()
    config={'base_run':str(root),'feature_contract_id':contract,'episode_order':'actual recorded episode manifest order',
            'memory_query':'0.5 source visual cosine + 0.3 predicted/observed future cosine + 0.2 direction cosine',
            'source_cosine_floor':.5,'direction_gate_degrees':35,'top_k':3,'fusion_weight_cap':.75,
            'neural_parameters_updated':False,'official_place_ids_used_for_retrieval':False,
            'episode_boundary_preserves_scene_memory':True,'evaluation':'development replay, not real new navigation or a blind test'}
    save(output/'protocol.json',config)
    methods=['episode_only_history','cross_episode_only_history','all_past_history']
    seeds=sorted({e['seed'] for e in json.loads((root/'config.json').read_text())['experiments'] if e['name']=='tssm_branch'})
    all_results=[]
    target=arrays['target'][dev_ids]
    labels=arrays['target_labels'][dev_ids]
    scenes=[pairs[i]['scene'] for i in dev_ids]
    for seed in seeds:
        base=np.load(root/f'tssm_branch_seed{seed}/best_prior_predictions.npy')
        all_results.append({'seed':seed,'method':'frozen_world_model','metrics':metrics(base,target,labels,scenes)})
        for method in methods:
            db=output/f'{method}_seed{seed}.sqlite'
            predictions=np.zeros_like(base)
            audit=[]
            previous_episode_count=defaultdict(int)
            previous_annotations=defaultdict(list)
            for ep in ordered:
                uid,scene=ep['episode_uid'],ep['scene']
                # Scene id is only the dataset's workspace-boundary grouping;
                # no point/coordinate/id is used by the visual retrieval API.
                context=scene if method!='episode_only_history' else f'{scene}|episode:{uid}'
                memory=PersistentSceneMemory(db,contract_id=contract)
                session=EpisodeMemorySession(memory,scope='cross_episode' if method=='cross_episode_only_history' else 'all_past')
                session.begin_episode(context,uid,ep['instruction_text'])
                starting_count=memory.count(context)
                episode_annotations=[]
                previous_edges={(r['offline_labels']['origin_place'],r['offline_labels']['chosen_exit']) for r in previous_annotations[scene]}
                previous_routes={r['route_group_id'] for r in previous_annotations[scene]}
                previous_current_images={r['history_image_ids'][-1] for r in previous_annotations[scene]}
                previous_future_images={r['target_image_id'] for r in previous_annotations[scene]}
                for i in sorted(by_episode[uid],key=lambda i:pairs[i]['origin_index']):
                    pair=pairs[i]
                    local=dev_local[i]
                    decision=session.predict_cached(base[local],arrays['history'][i,-1],arrays['branch'][i],
                                                    event_id=pair['sample_id'],source_observation=pair['history_observation_ids'][-1])
                    assert all(m.sequence<=decision.visible_sequence for m in decision.matches)
                    if method=='cross_episode_only_history':
                        assert all(m.episode_id!=uid for m in decision.matches)
                    predictions[local]=decision.future_feature
                    # The actual future is first used after the prediction, for
                    # scoring and then for recording the observed execution.
                    cosine=float(decision.future_feature@arrays['target'][i])
                    sequence=session.observe_real(decision.prediction_id,arrays['target'][i],
                                target_observation=pair['target_observation_id'],execution_ref=f"approved-recorded-span:{pair['sample_id']}",
                                executed_branch_direction=arrays['branch'][i],
                                executed=True,arrived=True)
                    assert sequence>decision.visible_sequence
                    audit.append({'sample_id':pair['sample_id'],'episode_uid':uid,'scene_namespace':scene,
                                  'previous_episodes':previous_episode_count[scene],
                                  'prior_episode_records':starting_count,
                                  'known_origin_exit_in_prior_episode':(pair['offline_labels']['origin_place'],pair['offline_labels']['chosen_exit']) in previous_edges,
                                  'route_seen_in_prior_episode':pair['route_group_id'] in previous_routes,
                                  'current_image_seen_in_prior_episode':pair['history_image_ids'][-1] in previous_current_images,
                                  'target_image_seen_in_prior_episode':pair['target_image_id'] in previous_future_images,
                                  'memory_weight':decision.memory_weight,'cosine_after_real_arrival':cosine,
                                  'visible_sequence':decision.visible_sequence,'recorded_after_prediction_sequence':sequence,
                                  'retrieved':[{'sequence':m.sequence,'episode_uid':m.episode_id,'event_id':m.event_id,
                                                'source_observation':m.source_observation,'target_observation':m.target_observation,
                                                'source_similarity':m.source_similarity,'score':m.score} for m in decision.matches]})
                    episode_annotations.append(pair)
                session.end_episode()
                expected_count=starting_count+len(by_episode[uid])
                assert memory.count(context)==expected_count
                memory.close()
                # Reopen on every episode: persistence cannot be an in-process cache.
                reopened=PersistentSceneMemory(db,contract_id=contract)
                assert reopened.count(context)==expected_count
                reopened.close()
                previous_episode_count[scene]+=1
                previous_annotations[scene].extend(episode_annotations)
            np.save(output/f'{method}_seed{seed}_predictions.npy',predictions)
            (output/f'{method}_seed{seed}_audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
            by_sample={r['sample_id']:r for r in audit}
            later=np.array([by_sample[pairs[i]['sample_id']]['previous_episodes']>0 for i in dev_ids])
            known=np.array([by_sample[pairs[i]['sample_id']]['known_origin_exit_in_prior_episode'] for i in dev_ids])
            same_route=np.array([by_sample[pairs[i]['sample_id']]['route_seen_in_prior_episode'] for i in dev_ids])
            same_image=np.array([by_sample[pairs[i]['sample_id']]['current_image_seen_in_prior_episode'] for i in dev_ids])
            cos=(predictions*target).sum(-1)
            basecos=(base*target).sum(-1)
            subgroup={}
            pred_hits=labels[(predictions@target.T).argmax(1)]==labels
            base_hits=labels[(base@target.T).argmax(1)]==labels
            for name,mask in [('first_episode',~later),('later_episodes',later),('prior_known_origin_exit',known),
                              ('later_new_route',later & ~same_route),('later_same_route',later & same_route),
                              ('later_new_current_view',later & ~same_image)]:
                subgroup[name]={'queries':int(mask.sum()),'cosine':float(cos[mask].mean()) if mask.any() else None,
                                 'base_cosine':float(basecos[mask].mean()) if mask.any() else None,
                                 'recall_at_1':float(pred_hits[mask].mean()) if mask.any() else None,
                                 'base_recall_at_1':float(base_hits[mask].mean()) if mask.any() else None}
            all_results.append({'seed':seed,'method':method,'metrics':metrics(predictions,target,labels,scenes),
                                'queries_with_memory':sum(bool(r['retrieved']) for r in audit),
                                'queries_with_cross_episode_hits':sum(any(m['episode_uid']!=r['episode_uid'] for m in r['retrieved']) for r in audit),
                                'subgroups':subgroup,'episodes':len(ordered),'real_records':len(audit),
                                'source':'only earlier actual recorded event outcomes; no parameter updates'})
    means={}
    for method in ['frozen_world_model',*methods]:
        rows=[r for r in all_results if r['method']==method]
        means[method]={key:float(np.mean([r['metrics'][key] for r in rows])) for key in ['cosine','scene_macro_cosine','recall_at_1','recall_at_5','mrr']}
    save(output/'results.json',{'means':means,'runs':all_results,'neural_parameters_updated':False,
                               'evaluation':'18 recorded episodes / 6 development scenes / 44 transitions, replayed per checkpoint seed',
                               'cross_episode_memory_persists_after_restart':True})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in input_hashes.items())
    save(output/'input_manifest.json',input_hashes)
    snapshot=output/'source_snapshot'
    for p in source_paths:
        if p.suffix=='.py':
            destination=snapshot/p.relative_to(REPO)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,destination)
            destination.chmod(0o444)
    print(json.dumps(means,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
