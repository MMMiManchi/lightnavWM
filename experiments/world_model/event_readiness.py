"""Offline evidence for event prediction, distinct from memory calibration.

Place/exit labels are used ONLY to audit support and score retrieval. They are
never predictor inputs. Matching labels at a place do not establish identical
observations, and identical observations do not establish identical policies.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def angles_degrees(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError('angles require equally shaped matrices')
    norms = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    if not np.isfinite(first).all() or not np.isfinite(second).all() or (norms <= 1e-12).any():
        raise ValueError('angles require finite nonzero vectors')
    return np.degrees(np.arccos(np.clip((first*second).sum(-1)/norms, -1, 1)))


def branch_coverage(pairs):
    reports = {}
    for partition in ['all', *sorted({p['partition'] for p in pairs})]:
        selected = [(i, p) for i, p in enumerate(pairs) if partition == 'all' or p['partition'] == partition]
        origins, histories = defaultdict(set), defaultdict(list)
        for index, pair in selected:
            labels = pair['offline_labels']
            origins[(pair['scene'], labels['origin_place'])].add(labels['chosen_exit'])
            # Branch labels alone cannot make different camera views a paired
            # intervention. Keep the full visual history and its padding mask.
            key = (pair['scene'], tuple(pair['history_image_ids']), tuple(pair['history_valid']))
            histories[key].append((index, pair))
        detailed, ambiguous = [], []
        same_pixels_multi_exit = 0
        clean_discriminative = 0
        for (scene, _images, _valid), members in histories.items():
            by_direction = defaultdict(set)
            exits, locations = set(), set()
            for _index, pair in members:
                branch = np.asarray(pair['branch_direction'], dtype=np.float64)
                if branch.shape != (2,) or not np.isfinite(branch).all() or np.linalg.norm(branch) < 1e-8:
                    raise ValueError('branch needs a finite nonzero direction')
                direction = tuple(np.round(branch/np.linalg.norm(branch), 6))
                labels = pair['offline_labels']
                by_direction[direction].add(labels['target_place'])
                exits.add((labels['origin_place'], labels['chosen_exit']))
                locations.add(labels['origin_place'])
            conflicts = [direction for direction, targets in by_direction.items() if len(targets) > 1]
            if conflicts:
                ambiguous.append({'scene': scene, 'pair_indices': [i for i, _p in members],
                                  'conflicting_direction_groups': len(conflicts)})
            if len(exits) < 2:
                continue
            same_pixels_multi_exit += 1
            distinct_targets = set().union(*by_direction.values())
            clean = (len(locations) == 1 and len(by_direction) >= 2 and len(distinct_targets) >= 2 and not conflicts)
            clean_discriminative += clean
            detailed.append({'scene': scene, 'pair_indices': [i for i, _p in members],
                'observed_directions': [{'sin_cos': list(d), 'target_places': sorted(t)} for d, t in by_direction.items()],
                'has_distinct_observed_outcomes': len(distinct_targets) >= 2,
                'same_input_has_multiple_target_places': bool(conflicts),
                'clean_discriminative_by_existing_labels': clean})
        reports[partition] = {'transitions': len(selected), 'origin_labels': len(origins),
            'origin_exit_count_histogram': dict(sorted(Counter(map(len, origins.values())).items())),
            'identical_history_groups': len(histories),
            'identical_history_multi_exit_groups': same_pixels_multi_exit,
            'clean_discriminative_history_groups': clean_discriminative,
            'identical_input_multiple_target_groups': len(ambiguous),
            'multi_exit_history_groups': detailed, 'ambiguous_input_groups': ambiguous,
            'qualification': 'pixel/input equality and offline labels only; fixed policy and full state equivalence still require verification'}
    return reports


def causal_gallery_report(pairs, episode_order, target, prediction, *, partition='train_dev'):
    """Score only actual outcomes from strictly earlier episodes in this scene.

    No acceptance threshold is chosen from these queries. If the exact labeled
    destination is absent, report missing support, not a prediction failure or
    automatic semantic unknown: a different place can still be a useful analogy.
    """
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or target.shape[0] != len(pairs) or target.ndim != 2:
        raise ValueError('predictions and actual outcomes must cover all pairs')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('retrieval features must be finite')
    grouped = defaultdict(list)
    for index, pair in enumerate(pairs):
        grouped[(pair['source_run'], pair['episode_uid'])].append(index)
    order = [tuple(ep) for ep in episode_order]
    if len(order) != len(set(order)) or set(grouped)-set(order):
        raise ValueError('acquisition order is incomplete or duplicates episodes')
    past, rows = defaultdict(list), []
    for episode in order:
        members = grouped.get(episode, [])
        if not members:
            continue
        scenes = {pairs[i]['scene'] for i in members}
        if len(scenes) != 1:
            raise ValueError('one episode spans scene namespaces')
        scene = next(iter(scenes))
        gallery = past[scene]
        for index in members:
            pair = pairs[index]
            if pair['partition'] != partition:
                continue
            truth = pair['offline_labels']['target_place']
            positives = [i for i in gallery if pairs[i]['offline_labels']['target_place'] == truth]
            top = None
            if gallery:
                top = gallery[int((target[gallery] @ prediction[index]).argmax())]
            rows.append({'pair_index': index, 'sample_id': pair['sample_id'], 'scene': scene,
                'gallery_records': len(gallery), 'actual_gallery_pair_indices': list(gallery),
                'exact_labeled_target_in_gallery': bool(positives),
                'top_pair_index': top, 'nearest_record_correct': bool(top in positives) if positives else None,
                'top_cosine': float(target[top] @ prediction[index]) if top is not None else None})
        # All predictions in this episode see the same prefix, regardless of
        # the layout of sample rows or when later target labels are inspected.
        past[scene].extend(members)
    supported = [r for r in rows if r['exact_labeled_target_in_gallery']]
    return {'queries': len(rows), 'empty_gallery': sum(not r['gallery_records'] for r in rows),
        'exact_labeled_target_available': len(supported), 'exact_labeled_target_absent': len(rows)-len(supported),
        'nearest_record_correct_when_target_available': sum(r['nearest_record_correct'] for r in supported),
        'conditional_recall_at_1': sum(r['nearest_record_correct'] for r in supported)/len(supported) if supported else None,
        'open_set_abstention_calibrated': False,
        'interpretation': 'strictly earlier same-scene actual records; missing exact target is support coverage, not proof of bad prediction',
        'rows': rows}


def adaptation_effect(pairs, current, target, memory, adapted, *, partition='train_dev'):
    indices = np.array([i for i, p in enumerate(pairs) if p['partition'] == partition], dtype=np.int64)
    if not len(indices):
        raise ValueError('partition contains no event queries')
    current, target, memory, adapted = [np.asarray(x)[indices] for x in [current, target, memory, adapted]]
    original = angles_degrees(current, target)
    change = angles_degrees(memory, adapted)
    unchanged = np.max(np.abs(memory-adapted), axis=-1) < 1e-7
    change[unchanged] = 0
    labels = np.array([pairs[i]['offline_labels']['target_place'] for i in indices])
    first = (labels[(memory @ target.T).argmax(-1)] == labels)
    second = (labels[(adapted @ target.T).argmax(-1)] == labels)
    return {'queries': len(indices), 'unchanged_predictions': int(unchanged.sum()),
        'changed_predictions': int((~unchanged).sum()), 'maximum_adapter_change_feature_degrees': float(change.max()),
        'mean_current_to_target_feature_degrees': float(original.mean()),
        'memory_correct_queries': [pairs[i]['sample_id'] for i in indices[first]],
        'adapted_correct_queries': [pairs[i]['sample_id'] for i in indices[second]],
        'gained_queries': [pairs[i]['sample_id'] for i in indices[second & ~first]],
        'lost_queries': [pairs[i]['sample_id'] for i in indices[first & ~second]],
        'identical_correct_query_set': bool(np.array_equal(first, second)),
        'closed_gallery_is_only_an_offline_score': True,
        'closed_gallery_not_accessible_to_runtime_predictor': True}
