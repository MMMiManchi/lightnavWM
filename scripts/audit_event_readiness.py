"""Audit event coverage, adaptation reach and strictly causal retrieval support.

CPU only. Existing features/predictions are read; no model trains, no Habitat
starts, and old raw data, model weights and experiment results stay immutable.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.world_model.event_readiness import (  # noqa: E402
    adaptation_effect,
    branch_coverage,
    causal_gallery_report,
)
from experiments.world_model.event_train import digest, read_rows, save  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--replay-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, replay, output = [p.resolve() for p in [args.base_run, args.replay_run, args.output]]
    config = json.loads((replay/'config.json').read_text())
    if Path(config['base_run']).resolve() != base:
        raise ValueError('replay uses a different event dataset')
    recorded = json.loads((replay/'input_manifest.json').read_text())
    for path in [base/'pairs.jsonl', base/'tensors.npz']:
        if recorded.get(str(path)) != digest(path):
            raise ValueError('event inputs differ from those used by the completed replay')
    files = sorted(replay.glob('worker_??/predictions.npz'))
    if len(files) != 8 or config['arms'] != ['frozen', 'memory_only', 'adapt_only', 'memory_and_adapt']:
        raise ValueError('expected the completed registered eight-worker four-arm replay')
    pairs = read_rows(base/'pairs.jsonl')
    with np.load(base/'tensors.npz', allow_pickle=False) as data:
        current, target = data['history'][:, -1].copy(), data['target'].copy()
    ids, values = [], []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            ids.append(data['ids'].copy())
            values.append(data['predictions'].copy())
    ids = np.concatenate(ids)
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.arange(len(pairs))):
        raise ValueError('replay is incomplete or repeats sample indices')
    predictions = np.concatenate(values, axis=1)[:, order]
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), REPO/'experiments/world_model/event_readiness.py', base/'pairs.jsonl',
               base/'tensors.npz', replay/'config.json', replay/'input_manifest.json', replay/'results.json', *files]
    hashes = {str(p): digest(p) for p in sources}
    save(output/'input_manifest.json', hashes)
    for path in sources:
        if path.suffix == '.py':
            destination = output/'source_snapshot'/path.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    coverage = branch_coverage(pairs)
    effect = adaptation_effect(pairs, current, target, predictions[1], predictions[3])
    galleries = {arm: causal_gallery_report(pairs, config['episode_order'], target, predictions[index])
                 for index, arm in [(1, 'memory_only'), (3, 'memory_and_adapt')]}
    radius = float(config['adapter']['residual_limit'])
    report = {'coverage': coverage, 'development_adaptation': effect,
        'strictly_past_actual_gallery': galleries,
        'scene_adapter_limit': {'residual_norm_bound': radius,
            'maximum_feature_angular_change_degrees': math.degrees(math.asin(radius)),
            'meaning': 'relative to unit base prediction, not a robot rotation angle'},
        'readiness': {'event_prediction_accepted': False, 'old_development_is_blind': False,
            'new_blind_queries_created': False, 'new_counterfactual_rollouts_collected': False,
            'unbounded_event_models_already_exist': True,
            'blockers': ['insufficient identical-observation branch interventions',
                         'small residual adaptation adds no closed-gallery correct queries',
                         'closed-gallery scoring cannot establish open-memory success',
                         'fresh preregistered event-level tests and fixed-option outcomes still required']}}
    if not all(digest(Path(p)) == h for p, h in hashes.items()):
        raise ValueError('audit inputs changed during analysis')
    save(output/'results.json', report)
    print(json.dumps({'output': str(output), 'development_adaptation': {
        k: v for k, v in effect.items() if k not in ['memory_correct_queries', 'adapted_correct_queries']},
        'coverage': {k: {n: v[n] for n in ['origin_labels', 'origin_exit_count_histogram',
            'identical_history_multi_exit_groups', 'clean_discriminative_history_groups',
            'identical_input_multiple_target_groups']} for k, v in coverage.items()},
        'causal_gallery': {k: {n: v[n] for n in ['queries', 'empty_gallery', 'exact_labeled_target_available',
            'exact_labeled_target_absent', 'nearest_record_correct_when_target_available']} for k, v in galleries.items()}},
        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
