import cv2
import numpy as np

from experiments.world_model.execution_feedback import ExecutionFeedback, camera_motion


def test_rgbd_geometry_recovers_actual_translation_and_yaw():
    k = np.array([[220., 0, 160], [0, 220, 120], [0, 0, 1]])
    source = np.array([(x, y) for x in range(50, 280, 25) for y in range(40, 210, 25)], dtype=float)
    xyz = (np.linalg.inv(k) @ np.c_[source, np.ones(len(source))].T).T * 2
    rotation, _ = cv2.Rodrigues(np.array([0., .08, 0.]))
    translation = np.array([.03, 0, .10])
    moved = (rotation @ (xyz-translation).T).T
    projected = (k @ moved.T).T
    target = projected[:, :2]/projected[:, 2:]
    result = camera_motion(source, target, np.full((240, 320), 2.), k)
    assert result["valid"]
    np.testing.assert_allclose([result["right_m"], result["forward_m"], result["yaw_left_rad"]], [.03, .10, .08], atol=.002)


def test_missing_depth_and_issued_forward_commands_do_not_forge_measured_progress():
    f = ExecutionFeedback()
    pixels = np.zeros((40, 40, 3), dtype=np.uint8)
    f.observe(0, "start", pixels, None)
    for step in range(1, 10):
        f.observe(step, f"frame-{step}", pixels, None, {"action": "move_forward"})
    context = f.context(commanded_forward=10.)
    assert context["measured_net_translation_m"] is None
    assert not context["stalled"] and context["local_pose"] is None
    assert f.backtrack_anchor() is None
