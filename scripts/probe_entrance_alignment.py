"""Real-image camera-alignment diagnostic; pose selects/scores cases only.

No pose, scene/node id, future state or graph is passed to the aligner. Diagnostic
images are used only to evaluate geometry, never as scene-adapter training data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

from experiments.world_model.entrance_alignment import EntranceAligner, wrapped  # noqa: E402
from experiments.world_model.event_train import digest, save  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=Path,default=REPO/'output/world_model_diagnostics/official_nodes_20261104_04/frames.jsonl')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=40)
    args=parser.parse_args()
    if not 10<=args.limit<=100:
        parser.error('limit must be 10..100')
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    frames=[json.loads(s) for s in args.frames.read_text().splitlines()]
    groups=defaultdict(list)
    first_scene={}
    for row in frames:
        if row.get('yaw') is None or not row.get('position') or not row.get('rgb_path'):
            continue
        if not Path(row['rgb_path']).is_file():
            continue
        position=row['position']
        key=(row['scene'],row['episode_uid'],round(position[0],1),round(position[1],1),round(position[2],1))
        groups[key].append(row)
        first_scene.setdefault(row['scene'],row)
    positives=[]
    for group in groups.values():
        for j,current in enumerate(group):
            for source in group[:j]:
                distance=float(np.linalg.norm(np.array(current['position'])-np.array(source['position'])))
                offset=wrapped(source['yaw']-current['yaw'])
                if distance<=.02 and math.radians(8)<=abs(offset)<=math.radians(65):
                    positives.append((source,current,offset,distance))
                    break
            if len(positives)>=args.limit:
                break
        if len(positives)>=args.limit:
            break
    scenes=sorted(first_scene)
    negatives=[(first_scene[scenes[i]],first_scene[scenes[i+1]]) for i in range(min(20,len(scenes)-1))]
    save(output/'protocol.json',{'positive_count':len(positives),'negative_count':len(negatives),
        'selection':'first stored near-stationary real views, translation<=2cm, heading change 8..65 degrees',
        'pose_is_evaluation_only':True,'diagnostic_images_not_used_to_train_world_model':True,
        'frames_sha256':digest(args.frames)})
    aligner=EntranceAligner()
    results=[]
    for source,current,expected,distance in positives:
        for row in [source,current]:
            if row.get('rgb_sha256') and digest(Path(row['rgb_path']))!=row['rgb_sha256']:
                raise ValueError('diagnostic image changed')
        old=np.load(source['rgb_path'],allow_pickle=False)
        now=np.load(current['rgb_path'],allow_pickle=False)
        # Pixels are the only inputs; pose remains outside this call.
        estimate=aligner.estimate(old,now,entry_point=(500,700))
        error=abs(wrapped(estimate.bearing_offset-expected)) if estimate.accepted else None
        results.append({'source':source['source_observation_id'],'current':current['source_observation_id'],
            'expected_offset_degrees_evaluation_only':math.degrees(expected),'translation_m_evaluation_only':distance,
            'estimate':asdict(estimate),'angular_error_degrees':math.degrees(error) if error is not None else None})
    negative_results=[]
    for source,current in negatives:
        estimate=aligner.estimate(np.load(source['rgb_path'],allow_pickle=False),
            np.load(current['rgb_path'],allow_pickle=False),entry_point=(500,700))
        negative_results.append({'source_scene':source['scene'],'other_scene':current['scene'],'estimate':asdict(estimate)})
    accepted=[r for r in results if r['estimate']['accepted']]
    save(output/'results.json',{'positive_cases':results,'negative_cases':negative_results,
        'positive_accepted':len(accepted),'positive_total':len(results),
        'median_error_degrees':float(np.median([r['angular_error_degrees'] for r in accepted])) if accepted else None,
        'accepted_within_5_degrees':sum(r['angular_error_degrees']<=5 for r in accepted),
        'false_accepts_other_scenes':sum(r['estimate']['accepted'] for r in negative_results),
        'this_does_not_validate_translation_or_physical_connection':True})
    print(json.dumps({k:v for k,v in json.loads((output/'results.json').read_text()).items() if not k.endswith('_cases')}))


if __name__=='__main__':
    main()
