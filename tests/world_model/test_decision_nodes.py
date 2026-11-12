"""CPU checks for decision-node marking. They do not read the dataset."""

from __future__ import annotations

import math

import numpy as np

from experiments.world_model.decision_nodes import (
    cluster_decision_frames,
    departure_bearing,
    exit_contrast,
    matched_exit,
    openings_from_depth,
)
from experiments.world_model.yaw_shift import HEIGHT, WIDTH


def _depth(columns: np.ndarray) -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    image[125:136, :] = columns[None, :]
    return image


def test_a_single_corridor_is_not_a_decision():
    columns = np.full(WIDTH, 0.8)
    columns[140:341] = 8.0
    openings = openings_from_depth(_depth(columns))
    assert len(openings) == 1
    assert openings[0]["side"] == "center"


def test_a_near_wall_marks_two_exits_and_their_visible_difference():
    columns = np.full(WIDTH, 4.0)
    columns[:200] = 8.0
    columns[200:240] = 1.0
    openings = openings_from_depth(_depth(columns))
    assert [item["side"] for item in openings] == ["left", "right"]
    contrast = exit_contrast(openings)
    assert contrast["wider_exit_id"] == 0
    assert contrast["longer_looking_exit_id"] == 0
    assert exit_contrast(openings[:1]) == {"wider_exit_id": None, "longer_looking_exit_id": None}


def test_a_thin_pole_or_a_far_gap_stays_one_opening():
    pole = np.full(WIDTH, 8.0)
    pole[238:241] = 0.4
    assert len(openings_from_depth(_depth(pole))) == 1
    far = np.full(WIDTH, 8.0)
    far[200:240] = 2.5
    assert len(openings_from_depth(_depth(far))) == 1


def test_nearby_views_of_one_junction_are_one_node():
    frames = [
        _frame(0, 0.0, openings=2, angle=1.0),
        _frame(1, 0.2, openings=1, angle=0.4),
        _frame(2, 3.0, openings=2, angle=0.8),
    ]
    nodes = cluster_decision_frames(frames)
    assert len(nodes) == 2
    assert nodes[0]["representative"]["step_index"] == 0
    assert len(nodes[0]["members"]) == 2
    assert nodes[1]["representative"]["step_index"] == 2


def test_the_taken_exit_is_the_opening_the_walk_enters():
    poses = [
        {"position": (0.0, 0.0, 0.0), "yaw": 0.0},
        {"position": (-0.5, 0.0, -0.1), "yaw": 0.0},
    ]
    bearing = departure_bearing(poses, 0)
    assert bearing is not None and bearing > 0.0
    openings = [
        {"exit_id": 0, "bearing_left_rad": math.pi / 2.0},
        {"exit_id": 1, "bearing_left_rad": 0.0},
    ]
    assert matched_exit(openings, bearing) == 0
    assert matched_exit(openings, -math.pi / 2.0) is None
    assert departure_bearing([poses[0]], 0) is None


def _frame(step: int, x: float, *, openings: int, angle: float) -> dict:
    return {
        "step_index": step,
        "position": (x, 0.0, 0.0),
        "openings": [{"angular_width_rad": angle / openings} for _ in range(openings)],
    }
