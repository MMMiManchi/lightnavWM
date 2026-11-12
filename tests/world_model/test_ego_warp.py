"""CPU checks for the commanded depth warp. They do not read frames."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.world_model.ego_warp import (
    command_motion,
    depth_to_meters,
    ego_warp_verdict,
    fit_camera_ok,
    focal_length,
    nearest_shift,
    project_point,
    relative_or_tie,
    warp_frame,
)
from experiments.world_model.yaw_shift import DT_S, HFOV_DEG, TURN_RATE_DPS, WIDTH


def test_command_and_camera_match_the_collision_free_branch_motion():
    forward, left, yaw = command_motion(1.5, 0.0, DT_S)
    assert (forward, left, yaw) == pytest.approx((0.15, 0.0, 0.0))
    forward, left, yaw = command_motion(0.0, math.radians(TURN_RATE_DPS), DT_S)
    assert left == 0.0
    assert yaw == pytest.approx(math.radians(6.0))
    assert nearest_shift(yaw, WIDTH, HFOV_DEG) == 24
    assert nearest_shift(0.0) == 0
    focal = focal_length(WIDTH, HFOV_DEG)
    assert focal == pytest.approx(240.0 / math.sqrt(3.0))
    meters = depth_to_meters(np.array([[0.0, 0.2], [1.0, 0.5]], dtype=np.float32))
    assert np.allclose(meters, [[0.0, 2.0], [10.0, 5.0]])


def test_left_yaw_moves_the_center_pixel_right_and_zero_motion_is_identity():
    fx = fy = 4.0
    cx, cy = 4.0, 2.0
    yaw = math.radians(6.0)
    u, v, z_new = project_point(0.0, 0.0, 2.0, 0.0, 0.0, yaw, fx, fy, cx, cy)
    assert u == pytest.approx(cx + fx * math.tan(yaw))
    assert v == pytest.approx(cy)
    assert z_new == pytest.approx(2.0 * math.cos(yaw))
    rgb = np.zeros((5, 9, 3), dtype=np.uint8)
    rgb[2, 4] = 255
    depth = np.zeros((5, 9), dtype=np.float64)
    depth[2, 4] = 2.0
    turned = math.radians(20.0)
    u, v, _z_new = project_point(0.0, 0.0, 2.0, 0.0, 0.0, turned, fx, fy, cx, cy)
    warped, mask = warp_frame(rgb, depth, 0.0, 0.0, turned, fx=fx, fy=fy, cx=cx, cy=cy)
    ui, vi = int(np.rint(u)), int(np.rint(v))
    assert ui > 4
    assert warped[vi, ui, 0] == 255
    assert mask[vi, ui]
    image = np.random.default_rng(0).integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    plane = np.full((6, 8), 2.0)
    copied, full = warp_frame(image, plane, 0.0, 0.0, 0.0, fx=4.0, fy=4.0, cx=4.0, cy=3.0)
    assert full.all()
    assert np.array_equal(copied, image)


def test_a_one_pixel_sampling_gap_is_filled_without_changing_identity():
    rgb = np.zeros((3, 7, 3), dtype=np.uint8)
    rgb[1, 2] = (10, 0, 0)
    rgb[1, 3] = (200, 0, 0)
    depth = np.zeros((3, 7), dtype=np.float64)
    depth[1, 2] = 2.0
    depth[1, 3] = 2.0
    warped, mask = warp_frame(rgb, depth, 1.0, 0.0, 0.0, fx=2.0, fy=2.0, cx=3.0, cy=1.0)
    assert mask[1, 1] and mask[1, 2] and mask[1, 3]
    assert warped[1, 1, 0] == 10
    assert warped[1, 3, 0] == 200
    assert int(warped[1, 2, 0]) in (10, 200)


def test_nearer_surface_wins_the_same_pixel():
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    rgb[1, 1] = (10, 0, 0)
    rgb[1, 2] = (200, 0, 0)
    depth = np.zeros((3, 3), dtype=np.float64)
    depth[1, 1] = 2.0
    depth[1, 2] = 4.0
    warped, mask = warp_frame(
        rgb, depth, -6.0, 0.0, 0.0, fx=1.0, fy=1.0, cx=1.0, cy=1.0, min_z=0.05
    )
    assert mask[1, 1]
    assert warped[1, 1, 0] == 10


def test_a_zero_baseline_is_a_tie_only_when_the_candidate_is_zero():
    assert relative_or_tie(0.0, 0.0) == 0.0
    assert relative_or_tie(0.0, 1.0) == -1.0
    assert relative_or_tie(10.0, 8.0) == pytest.approx(0.2)


def test_verdict_stops_before_development_when_the_fit_camera_check_fails():
    assert fit_camera_ok(0.11, 0.11, [-0.10, 0.0], 0.50)
    assert not fit_camera_ok(0.10, 0.11, [0.0, 0.0], 0.80)
    assert not fit_camera_ok(0.20, 0.20, [-0.11, 0.0], 0.80)
    assert not fit_camera_ok(0.20, 0.20, [0.0, 0.0], 0.49)
    failed = ego_warp_verdict(fit_passed=False)
    assert failed["call"] == "camera_model_failed_fit_check"
    assert failed["training_started"] is False
    scores = dict(
        fit_passed=True,
        coverage=0.80,
        stop_relative=0.0,
        all_copy=20.0,
        all_yaw=18.0,
        all_depth=14.0,
        turn_copy=25.0,
        turn_yaw=19.0,
        turn_depth=15.0,
        forward_copy=12.0,
        forward_depth=9.0,
    )
    assert ego_warp_verdict(**scores)["call"] == "depth_warp_predicts"
    thin = dict(scores)
    thin["coverage"] = 0.40
    assert ego_warp_verdict(**thin)["call"] == "warp_covers_too_little"
    stopped = dict(scores)
    stopped["stop_relative"] = 0.11
    assert ego_warp_verdict(**stopped)["call"] == "warp_helps_without_motion"
    short = dict(scores)
    short["turn_depth"] = 18.0
    assert ego_warp_verdict(**short)["call"] == "depth_warp_does_not_beat_baselines"
