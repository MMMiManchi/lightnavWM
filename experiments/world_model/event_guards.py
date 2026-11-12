"""Verify event temporal references against approved raw observation metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def validate_pair_references(pair: dict, observations: dict[str, dict]) -> None:
    origin, target = pair['origin_index'], pair['target_index']
    if target <= origin or len(pair['history_indices']) != 8:
        raise ValueError('invalid event temporal extent')
    for index, oid, image_id in zip(pair['history_indices'], pair['history_observation_ids'], pair['history_image_ids'], strict=True):
        if index < 0:
            if oid is not None or image_id is not None:
                raise ValueError('padded history contains an observation')
            continue
        row = observations[oid]
        if index > origin or row['episode_uid'] != pair['episode_uid'] or row['step_index'] != index:
            raise ValueError('history reference is future or belongs to another episode')
        if row['rgb']['sha256'] != image_id:
            raise ValueError('history image binding differs from the approved source')
    row = observations[pair['target_observation_id']]
    if row['episode_uid'] != pair['episode_uid'] or row['step_index'] != target or row['rgb']['sha256'] != pair['target_image_id']:
        raise ValueError('target binding differs from the approved source')


def verify_prepared_references(config: dict) -> dict:
    root = Path(config['run_dir'])
    repo = Path(__file__).resolve().parents[2]
    hashes = json.loads((root/'input_manifest.json').read_text())
    for name, expected in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'prepared input changed: {name}')
    observations = {}
    for run in config['source_runs']:
        folder = 'motion_supplement_20261030_01' if 'radius010' in run else 'first_v1_20261029'
        approval = json.loads((repo/'output/world_model_approvals'/folder/f'{run}.approval.json').read_text())
        path = repo/'output/world_model_data'/run/'observations.jsonl'
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != approval['files']['observations.jsonl']:
            raise ValueError('source observation metadata changed after admission')
        observations[run] = {r['observation_id']:r for r in (json.loads(s) for s in body.splitlines() if s.strip())}
    pairs = [json.loads(s) for s in (root/'pairs.jsonl').read_text().splitlines() if s.strip()]
    for pair in pairs:
        validate_pair_references(pair,observations[pair['source_run']])
    return {'verified_pairs':len(pairs),'history_is_causal':True,'source_images_bound':True}
