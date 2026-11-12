"""CPU checks for the commanded-yaw pixel shift. They do not read branch frames."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.world_model.yaw_shift import (
    DT_S,
    HFOV_DEG,
    TURN_RATE_DPS,
    WIDTH,
    command_yaw_rad,
    horizontal_shift_px,
    integer_shift,
    masked_mae,
    relative_improvement,
    shift_content_right,
    yaw_shift_verdict,
)


def test_sixty_degrees_per_second_is_a_24_pixel_shift():
    yaw = command_yaw_rad(TURN_RATE_DPS, DT_S)
    assert yaw == pytest.approx(math.radians(6.0))
    assert horizontal_shift_px(yaw, WIDTH, HFOV_DEG) == pytest.approx(24.0)
    assert integer_shift(yaw) == 24
    assert integer_shift(command_yaw_rad(0.0)) == 0
    assert integer_shift(command_yaw_rad(-TURN_RATE_DPS)) == -24


def test_positive_shift_moves_content_right_and_ignores_the_empty_border():
    before = np.zeros((8, 32, 3), dtype=np.uint8)
    before[:, 10] = 255
    after = np.zeros_like(before)
    after[:, 16] = 255
    warped, mask = shift_content_right(before, 6)
    assert mask[:, :6].sum() == 0
    assert mask[:, 6:].all()
    assert masked_mae(after, warped, mask) == 0.0
    assert masked_mae(after, before, mask) > 0.0
    unchanged, full = shift_content_right(before, 0)
    assert full.all()
    assert np.array_equal(unchanged, before)


def test_yaw_shift_verdict_requires_both_turn_directions_and_a_straight_control():
    explained = yaw_shift_verdict(
        left_relative=0.2,
        right_relative=0.2,
        straight_command_relatives=[0.0, 0.0],
        straight_imposed_relatives=[0.0, 0.0, -0.1, -0.1],
    )
    assert explained["call"] == "yaw_shift_explains_turns"
    assert explained["training_started"] is False
    assert explained["habitat_started"] is False
    artifact = yaw_shift_verdict(
        left_relative=0.2,
        right_relative=0.2,
        straight_command_relatives=[0.0, 0.0],
        straight_imposed_relatives=[0.11, 0.0, 0.0, 0.0],
    )
    assert artifact["call"] == "shift_helps_without_yaw"
    missed = yaw_shift_verdict(
        left_relative=0.2,
        right_relative=0.05,
        straight_command_relatives=[0.0, 0.0],
        straight_imposed_relatives=[0.0, 0.0, 0.0, 0.0],
    )
    assert missed["call"] == "yaw_shift_does_not_explain"
    assert relative_improvement(1.0, 0.9) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="both imposed"):
        yaw_shift_verdict(
            left_relative=0.2,
            right_relative=0.2,
            straight_command_relatives=[0.0, 0.0],
            straight_imposed_relatives=[0.0],
        )
