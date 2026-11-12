"""Expert stop rule and polyline sampling. No simulator."""

import math

import numpy as np

from lightnav.expert import (
    ego_waypoints,
    face_path_waypoints,
    sample_polyline,
    should_stop,
    trim_episode_rows,
    trim_stalled_end,
)


def test_stop_is_not_the_3m_success_line():
    assert should_stop(2.5, 2.5) is False
    assert should_stop(1.0, 1.0) is True
    assert should_stop(0.4, 0.4) is True
    # Was inside 1 m and is now leaving.
    assert should_stop(1.5, 0.8) is True
    # Still outside the stop radius: keep walking, even if distance grew a bit.
    assert should_stop(4.0, 3.2) is False
    assert should_stop(float("inf"), 1.0) is False


def test_polyline_samples_are_a_quarter_meter_apart():
    points = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -2.5])]
    samples = sample_polyline(points, step_m=0.25, horizon=10)
    assert len(samples) == 10
    # Identity faces -Z, so walking toward -Z is straight ahead.
    forward = np.array([0.0, 0.0, -1.0])
    left = np.array([-1.0, 0.0, 0.0])
    wp = ego_waypoints(np.zeros(3), forward, left, 0.0, samples)
    np.testing.assert_allclose(wp[:, 0], 0.25 * np.arange(1, 11), atol=1e-5)
    np.testing.assert_allclose(wp[:, 1], 0.0, atol=1e-5)
    assert abs(wp[-1, 2]) < 1e-5


def test_a_left_turn_has_positive_yaw():
    # From the origin facing -Z, the path goes forward 0.25 m then toward -X (left).
    points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -0.25]),
        np.array([-1.0, 0.0, -0.25]),
    ]
    samples = sample_polyline(points, step_m=0.25, horizon=4)
    forward = np.array([0.0, 0.0, -1.0])
    left = np.array([-1.0, 0.0, 0.0])
    wp = ego_waypoints(np.zeros(3), forward, left, 0.0, samples)
    assert wp[2, 1] > 0.2  # later points sit to the left
    assert wp[2, 2] > 0.5  # heading has turned ccw
    assert math.isfinite(float(wp.sum()))


def test_a_path_behind_the_body_becomes_a_turn():
    # Facing -Z, the route leaves toward +Z. The body cannot drive backward.
    behind = np.zeros((10, 3), dtype=np.float32)
    behind[:, 0] = -0.25 * np.arange(1, 11)
    behind[:, 2] = math.pi
    turned = face_path_waypoints(behind)
    assert np.all(turned[:, 0] == 0.0)
    assert turned[0, 2] > 0.4
    assert abs(turned[-1, 2] - math.pi) < 1e-5
    ahead = behind.copy()
    ahead[:, 0] *= -1
    ahead[:, 2] = 0.0
    np.testing.assert_allclose(face_path_waypoints(ahead), ahead)


def test_a_stall_drops_the_tail_but_keeps_a_real_stop():
    closing = [10 - 0.2 * i for i in range(40)]
    # Index 40 is the step that first reaches 2.0 m; forty more steps stay.
    assert trim_stalled_end(closing + [2.0] * 400) == 81
    assert trim_stalled_end(closing) == len(closing)
    stop_rows = [
        {"step": i, "distance_m": 5 - 0.1 * i, "stop": i == 40}
        for i in range(41)
    ]
    kept = trim_episode_rows(stop_rows)
    assert len(kept) == 41
    assert kept[-1]["stop"] is True


def test_a_later_stop_is_kept_without_the_stuck_middle():
    rows = [{"step": i, "distance_m": 2.0, "stop": i == 200} for i in range(201)]
    rows[0]["distance_m"] = 5.0
    kept = trim_episode_rows(rows)
    assert kept[-1]["step"] == 200
    assert kept[-1]["stop"] is True
    assert len(kept) == 43
    assert all(row["step"] <= 41 or row["stop"] for row in kept)
