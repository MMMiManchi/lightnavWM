"""Conservative source-aperture crossing from actual depth and local RGB-D motion."""

import math

import numpy as np


def aperture_plane(depth, box, *, scale=10.0, hfov=120.0):
    """Fit a horizontal jamb segment; an unmeasurable opening stays unknown.

    Depth through the opening is NOT its threshold depth. We require two
    supported boundary strips and a deeper interior before estimating a plane.
    Coordinates are source-camera right/forward, never simulator world poses.
    """
    if depth is None or box is None or len(box) != 4:
        return None, "source_depth_or_aperture_box_unavailable"
    image = np.asarray(depth, dtype=float).squeeze()*scale
    if image.ndim != 2:
        return None, "invalid_depth_shape"
    x1, y1, x2, y2 = [float(v)/1000 for v in box]
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None, "invalid_aperture_box"
    h, w = image.shape
    focal = w/(2*math.tan(math.radians(hfov)/2))

    def points(a, b, top=.2, bottom=.8):
        left, right = max(0, int(w*a)), min(w, int(w*b))
        start, end = int(h*(y1+top*(y2-y1))), int(h*(y1+bottom*(y2-y1)))
        z = image[start:end, left:right]
        valid = np.isfinite(z) & (z > .05) & (z < 9.99)
        if z.size < 12 or valid.sum() < max(12, .6*z.size):
            return None
        u = np.broadcast_to(np.arange(left, right)[None], z.shape)
        return np.c_[((u-w/2)/focal*z)[valid], z[valid]]

    a = points(x1, x1+.08*(x2-x1))
    b = points(x2-.08*(x2-x1), x2)
    interior = points(x1+.35*(x2-x1), x1+.65*(x2-x1), .3, .7)
    if a is None or b is None or interior is None:
        return None, "boundary_or_interior_depth_not_supported"
    left, right = np.median(a, axis=0), np.median(b, axis=0)
    width = float(np.linalg.norm(right-left))
    if not .45 <= width <= 4.0:
        return None, "aperture_width_not_supported"
    tangent = (right-left)/width
    normal = np.array([-tangent[1], tangent[0]])
    center = (left+right)/2
    if np.dot(normal, center) < 0:
        normal *= -1
    distance = float(np.dot(center, normal))
    residual = np.abs((np.r_[a, b]-center)@normal)
    if not .3 <= distance <= 8 or np.quantile(residual, .9) > .25:
        return None, "jamb_plane_not_supported"
    gap = float(np.median((interior-center)@normal))
    if gap < .25:
        return None, "interior_does_not_support_an_open_passage"
    return {"center_right_forward_m": center.tolist(), "tangent": tangent.tolist(),
        "normal": normal.tolist(), "width_m": width, "distance_m": distance,
        "interior_depth_beyond_plane_m": gap,
        "evidence_kind": "estimated_source_RGBD_aperture_not_ground_truth"}, None


class ObservedExitCrossing:
    def __init__(self, branch, depth, source_feedback, *, step, scale=10.0, hfov=120.0):
        self.geometry, self.reason = aperture_plane(depth, branch.image_region, scale=scale, hfov=hfov)
        self.source = branch.source_observation
        self.epoch = source_feedback["pose_epoch"]
        pose = source_feedback.get("local_pose")
        # An initial actual camera defines an arbitrary local origin. Later
        # invalid odometry cannot invent another source-to-current transform.
        self.source_pose = np.zeros(3) if step == 0 and pose is None else None if pose is None else np.array(pose)
        if self.source_pose is None:
            self.reason = "source_local_motion_frame_unavailable"
        self.valid = self.geometry is not None and self.source_pose is not None
        self.last_step, self.previous = step, np.zeros(2)
        self.crossing, self.crossed = None, False
        self.last_side = None

    def observe(self, feedback):
        if not self.valid:
            return
        step = feedback["step"]
        pose = feedback.get("local_pose")
        if step != self.last_step+1 or pose is None or feedback["pose_epoch"] != self.epoch:
            self.valid, self.reason = False, "source_to_current_motion_chain_unavailable"
            return
        delta = np.array(pose[:2])-self.source_pose[:2]
        yaw = self.source_pose[2]
        current = np.array([math.cos(yaw)*delta[0]+math.sin(yaw)*delta[1],
                            -math.sin(yaw)*delta[0]+math.cos(yaw)*delta[1]])
        normal, center, tangent = [np.array(self.geometry[k]) for k in
                                   ["normal", "center_right_forward_m", "tangent"]]
        before, after = float((self.previous-center)@normal), float((current-center)@normal)
        if before <= 0 < after:
            position = self.previous+(current-self.previous)*(-before/(after-before))
            if abs(float((position-center)@tangent)) > self.geometry["width_m"]/2-.1:
                self.valid, self.reason = False, "measured_plane_crossing_outside_selected_aperture"
                return
            self.crossing = {"actual_reference": feedback["reference"], "step": step,
                             "source_camera_right_forward_m": position.tolist()}
        if after < -.15:
            self.crossing, self.crossed = None, False
        if self.crossing is not None and after >= .15:
            self.crossed = True
        self.last_step, self.previous, self.last_side = step, current, after
        self.reason = "measured_selected_aperture_crossing" if self.crossed else "selected_aperture_not_yet_crossed"

    def eligible(self, step):
        return bool(self.valid and self.crossed and self.last_step == step)

    def snapshot(self):
        return {"source_observation": self.source, "geometry": self.geometry,
            "motion_chain_valid": self.valid, "pose_epoch": self.epoch,
            "selected_aperture_crossed": self.crossed and self.valid,
            "last_step": self.last_step, "signed_distance_beyond_plane_m": self.last_side,
            "crossing": self.crossing, "reason": self.reason,
            "required_before_visual_event_writeback": True}
