from types import SimpleNamespace

import numpy as np
import pytest

from experiments.world_model.exit_crossing import ObservedExitCrossing, aperture_plane


def source_depth():
    depth = np.full((120, 200), 2.)
    depth[:, 85:115] = 4.
    return depth


def feedback(step, x=0., z=0., yaw=0., epoch=0):
    return {"step": step, "reference": f"actual:{step}", "local_pose": [x, z, yaw], "pose_epoch": epoch}


def guard(pose=None):
    branch = SimpleNamespace(source_observation="actual:0", image_region=[300, 200, 700, 900])
    return ObservedExitCrossing(branch, source_depth(), pose or feedback(0), step=0, scale=1.)


def test_actual_aperture_plane_uses_boundary_depth_not_the_distant_interior():
    plane, error = aperture_plane(source_depth(), [300, 200, 700, 900], scale=1.)
    assert error is None and plane["distance_m"] == pytest.approx(2.)
    assert plane["interior_depth_beyond_plane_m"] == pytest.approx(2.)


def test_wall_or_missing_depth_cannot_prove_an_open_passage():
    assert aperture_plane(None, [300, 200, 700, 900])[0] is None
    assert aperture_plane(np.full((120, 200), 2.), [300, 200, 700, 900], scale=1.)[0] is None


def test_forward_motion_before_threshold_and_pure_rotation_are_not_crossing():
    gate = guard()
    for step in range(1, 5):
        gate.observe(feedback(step, z=step*.25, yaw=step*.2))
        assert not gate.eligible(step)


def test_continuous_actual_crossing_through_selected_aperture_is_admitted():
    gate = guard()
    for step in range(1, 10):
        gate.observe(feedback(step, z=step*.25))
    assert gate.eligible(9)
    assert gate.snapshot()["crossing"]["actual_reference"] == "actual:9"
    assert not gate.eligible(10)  # a future/stale confirmation cannot reuse it


def test_crossing_the_plane_outside_the_selected_exit_is_rejected():
    gate = guard()
    for step in range(1, 10):
        gate.observe(feedback(step, x=step*.3, z=step*.25))
    assert not gate.eligible(9)
    assert gate.reason == "measured_plane_crossing_outside_selected_aperture"


@pytest.mark.parametrize("change", ["missing_pose", "new_epoch", "missing_step"])
def test_lost_source_to_current_motion_chain_cannot_be_reconstructed_from_commands(change):
    gate = guard()
    item = feedback(1)
    if change == "missing_pose":
        item["local_pose"] = None
    elif change == "new_epoch":
        item["pose_epoch"] = 1
    else:
        item["step"] = 2
    gate.observe(item)
    for step in range(2, 12):
        gate.observe(feedback(step, z=step*.25))
    assert not gate.eligible(11) and not gate.valid


def test_crossing_respects_source_camera_rotation_without_a_world_goal_pose():
    gate = guard(feedback(0, x=3., z=5., yaw=np.pi/2))
    for step in range(1, 10):
        gate.observe(feedback(step, x=3.-step*.25, z=5., yaw=np.pi/2))
    assert gate.eligible(9)
