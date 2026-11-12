import numpy as np

from experiments.world_model.tssm_sequences import route_actions, route_anchors


def test_route_states_preserve_the_actual_event_and_are_spaced_by_motion():
    frames = [{"position": [i / 20, 0, 0]} for i in range(201)]
    ids, path = route_anchors(frames)
    assert ids[0] == 0 and ids[-1] == 200
    assert len(ids) > 2
    assert np.diff(path[ids]).min() >= 1


def test_turning_frames_do_not_become_fake_intermediate_nodes():
    ids, _ = route_anchors([{"position": [0, 0, 0]} for _ in range(700)])
    assert ids == [0, 699]


def test_future_control_trace_is_not_an_action_condition():
    actions = route_actions(4, -0.5)
    assert actions.shape == (4, 4)
    assert np.all(actions[:, 2:] == [1, 0])
    np.testing.assert_allclose(actions[1:, :2], [[0, 1]] * 3)
