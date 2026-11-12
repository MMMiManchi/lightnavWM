import numpy as np
import pytest

from experiments.world_model.event_readiness import (
    angles_degrees,
    branch_coverage,
    causal_gallery_report,
)


def pair(episode, branch, target, *, image='same', scene='house'):
    return {'sample_id': f'{scene}-{episode}-{branch}-{target}', 'source_run': 'run', 'episode_uid': episode,
        'scene': scene, 'partition': 'train_dev', 'history_image_ids': [None, image], 'history_valid': [False, True],
        'branch_direction': [0., 1.] if branch == 'forward' else [1., 0.],
        'offline_labels': {'origin_place': 'junction', 'chosen_exit': branch, 'target_place': target}}


def test_place_label_coverage_does_not_imply_matching_visual_inputs():
    pairs = [pair('a', 'forward', 'bathroom'), pair('b', 'left', 'hallway', image='different_view')]
    report = branch_coverage(pairs)['train_dev']
    assert report['origin_exit_count_histogram'] == {2: 1}
    assert report['identical_history_multi_exit_groups'] == 0


def test_same_target_exits_are_not_discriminative_negatives_and_ambiguity_is_reported():
    pairs = [pair('a', 'forward', 'bathroom'), pair('b', 'left', 'bathroom')]
    report = branch_coverage(pairs)['train_dev']
    assert report['identical_history_multi_exit_groups'] == 1
    assert report['clean_discriminative_history_groups'] == 0
    pairs.append(pair('c', 'forward', 'hallway'))
    report = branch_coverage(pairs)['train_dev']
    assert report['identical_input_multiple_target_groups'] == 1
    assert report['clean_discriminative_history_groups'] == 0


def test_distinct_outcomes_with_same_history_and_different_options_qualify_only_as_label_controls():
    report = branch_coverage([pair('a', 'forward', 'bathroom'), pair('b', 'left', 'hallway')])['train_dev']
    assert report['clean_discriminative_history_groups'] == 1
    assert 'fixed policy' in report['qualification']


def test_causal_gallery_never_reads_own_episode_or_other_scene_and_reports_absent_target():
    pairs = [pair('one', 'forward', 'bathroom'), pair('two', 'left', 'hallway'),
             pair('one', 'left', 'hallway'), pair('other', 'forward', 'hallway', scene='other-house'),
             pair('three', 'left', 'hallway')]
    target = np.array([[1., 0.], [0., 1.], [0., 1.], [0., 1.], [0., 1.]])
    report = causal_gallery_report(pairs, [('run', k) for k in ['one', 'other', 'two', 'three']], target, target)
    rows = {r['pair_index']: r for r in report['rows']}
    assert rows[0]['actual_gallery_pair_indices'] == rows[2]['actual_gallery_pair_indices'] == []
    assert rows[3]['actual_gallery_pair_indices'] == []
    assert rows[1]['actual_gallery_pair_indices'] == [0, 2]
    assert rows[4]['actual_gallery_pair_indices'] == [0, 2, 1]
    assert rows[0]['nearest_record_correct'] is None
    assert rows[1]['nearest_record_correct'] is True
    assert not report['open_set_abstention_calibrated']


def test_angles_are_feature_geometry_with_zero_changes_and_no_nan_fallback():
    np.testing.assert_allclose(angles_degrees([[1., 0.], [1., 0.]], [[1., 0.], [0., 1.]]), [0., 90.])
    with pytest.raises(ValueError, match='nonzero'):
        angles_degrees([[0., 0.]], [[1., 0.]])
