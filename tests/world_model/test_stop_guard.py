import pytest

from experiments.world_model.stop_guard import configured_stop_thresholds, is_implicit_stop


def test_habitat_scaled_near_zero_controls_are_implicit_stop():
    thresholds = {'linear_m_per_s': .01, 'angular_deg_per_s': .5}
    settings = {'linear_range': (0., 2.5), 'angular_range': (-300., 300.), 'thresholds': thresholds}
    def velocity(linear, angular):
        return {'action': 'velocity_control', 'action_args': {'linear_velocity': linear, 'angular_velocity': angular}}
    assert is_implicit_stop(velocity(-1., 0.), **settings)
    assert is_implicit_stop(velocity(-.999, .0008), **settings)
    assert not is_implicit_stop(velocity(-1., .005), **settings)
    assert not is_implicit_stop(velocity(-.98, 0.), **settings)
    assert is_implicit_stop(0, **settings)


def test_stop_thresholds_are_read_from_the_actual_config_and_missing_values_fail_closed(tmp_path):
    config = tmp_path/'sim.yaml'
    config.write_text('habitat:\n  task:\n    actions:\n      velocity_control:\n'
                      '        min_abs_lin_speed: 0.01\n        min_abs_ang_speed: 0.5\n')
    assert configured_stop_thresholds(config) == {'linear_m_per_s': .01, 'angular_deg_per_s': .5}
    config.write_text('habitat: {task: {actions: {velocity_control: {}}}}\n')
    with pytest.raises(ValueError, match='explicitly declare'):
        configured_stop_thresholds(config)
