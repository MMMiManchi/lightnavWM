"""Local RGB-D motion estimates and acknowledged skill execution, without world poses."""

import math
from collections import deque

import cv2
import numpy as np


def camera_motion(source, target, depth, intrinsics):
    """Old-camera 3D -> current pixels; return measured camera displacement."""
    h, w = depth.shape
    uv = np.rint(source).astype(int)
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[valid]
    source, target = source[valid], target[valid]
    z = depth[uv[:, 1], uv[:, 0]]
    good = np.isfinite(z) & (z > 0.05) & (z < 9.99)
    source, target, z = source[good], target[good], z[good]
    if len(z) < 16:
        return {"valid": False, "reason": "insufficient_actual_depth_matches"}
    xyz = (np.linalg.inv(intrinsics) @ np.c_[source, np.ones(len(z))].T).T * z[:, None]
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(xyz, target, intrinsics, None,
        iterationsCount=100, reprojectionError=3.0, confidence=0.99, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 12 or len(inliers) / len(z) < 0.5:
        return {"valid": False, "reason": "motion_geometry_not_supported"}
    ids = inliers[:, 0]
    rotation, _ = cv2.Rodrigues(rvec)
    delta = -rotation.T @ tvec[:, 0]
    projected, _ = cv2.projectPoints(xyz[ids], rvec, tvec, intrinsics, None)
    error = float(np.median(np.linalg.norm(projected[:, 0] - target[ids], axis=-1)))
    yaw = math.atan2(float(rotation[0, 2]), float(rotation[2, 2]))
    if error > 3 or np.linalg.norm(delta) > 0.5 or abs(yaw) > math.pi / 2:
        return {"valid": False, "reason": "implausible_or_inaccurate_motion"}
    return {"valid": True, "source": "actual_rgbd_pnp", "right_m": float(delta[0]),
        "forward_m": float(delta[2]), "translation_m": float(np.linalg.norm(delta[[0, 2]])),
        "yaw_left_rad": yaw, "inliers": len(ids), "median_error_px": error}


class ExecutionFeedback:
    def __init__(self, *, depth_scale=10.0, hfov_degrees=120.0, window=12):
        cv2.setNumThreads(1)
        self.depth_scale, self.hfov = depth_scale, hfov_degrees
        self.detector = cv2.SIFT_create(nfeatures=1200)
        self.previous, self.records = None, deque(maxlen=window)
        self.pose = np.zeros(3, dtype=float)  # local right, forward, yaw-left; never a world goal pose
        self.epoch, self.pose_valid = 0, False
        self.anchors = deque(maxlen=12)

    @staticmethod
    def clearance(depth):
        if depth is None:
            return None
        h, w = depth.shape
        crop = depth[int(h*.25):int(h*.65), int(w*.4):int(w*.6)]
        values = crop[np.isfinite(crop) & (crop > 0.05)]
        return float(np.median(values)) if len(values) > crop.size*.25 else None

    def observe(self, step, reference, rgb, depth, receipt=None):
        rgb = np.asarray(rgb)
        depth = None if depth is None else np.asarray(depth, dtype=float).squeeze() * self.depth_scale
        points, descriptors = self.detector.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
        keypoints = np.array([p.pt for p in points], dtype=float).reshape(-1, 2)
        current = (step, reference, rgb.copy(), depth, keypoints, descriptors)
        measurement = {"valid": False, "reason": "initial_observation" if self.previous is None else "depth_or_features_unavailable"}
        broken = self.previous is not None and step != self.previous[0] + 1
        if self.previous is not None and receipt is not None and not broken:
            old = self.previous
            if old[3] is not None and old[5] is not None and descriptors is not None:
                pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(old[5], descriptors, k=2)
                good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < .72*n.distance]
                if len(good) >= 16:
                    h, w = rgb.shape[:2]
                    focal = w / (2*math.tan(math.radians(self.hfov)/2))
                    k = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=float)
                    measurement = camera_motion(old[4][[m.queryIdx for m in good]], keypoints[[m.trainIdx for m in good]], old[3], k)
                else:
                    measurement = {"valid": False, "reason": "insufficient_actual_image_matches"}
        if measurement["valid"]:
            yaw = self.pose[2]
            right, forward = measurement["right_m"], measurement["forward_m"]
            self.pose[:2] += [math.cos(yaw)*right-math.sin(yaw)*forward,
                             math.sin(yaw)*right+math.cos(yaw)*forward]
            self.pose[2] += measurement["yaw_left_rad"]
            self.pose_valid = True
        elif self.previous is not None:
            if self.pose_valid:
                self.epoch += 1
                self.pose[:] = 0
                self.anchors.clear()
            self.pose_valid = False
        record = {"step": step, "reference": reference, "measurement": measurement,
            "pose_epoch": self.epoch, "local_pose": self.pose.tolist() if self.pose_valid else None,
            "clearance_m": self.clearance(depth), "continuity_broken": broken,
            "actual_execution_acknowledged": receipt is not None,
            "issued_action": receipt.get("action") if receipt else None}
        self.records.append(record)
        if self.pose_valid and (not self.anchors or np.linalg.norm(self.pose[:2]-self.anchors[-1]["xy"]) > .3):
            self.anchors.append({"reference": reference, "xy": self.pose[:2].copy(), "epoch": self.epoch})
        self.previous = current
        return record

    def context(self, *, commanded_forward=0.0):
        records = list(self.records)
        valid = [r for r in records if r["measurement"]["valid"]]
        same = [r for r in valid if r["pose_epoch"] == self.epoch]
        net = float(np.linalg.norm(np.array(same[-1]["local_pose"][:2])-same[0]["local_pose"][:2])) if len(same) >= 2 else None
        stalled = len(records) >= 8 and len(valid) >= len(records)*.7 and commanded_forward >= .2 and net is not None and net < .04
        return {"actual_motion_source": "RGB-D geometry; issued controls are separate",
            "clearance_is_front_sensor_depth_not_goal_distance": True,
            "recent_measurements": records[-6:], "measured_net_translation_m": net,
            "valid_fraction": len(valid)/max(len(records), 1), "stalled": stalled,
            "clearance_m": records[-1]["clearance_m"] if records else None,
            "local_pose": self.pose.tolist() if self.pose_valid else None, "pose_epoch": self.epoch,
            "continuity_broken": bool(records and records[-1]["continuity_broken"])}

    def backtrack_anchor(self):
        if not self.pose_valid:
            return None
        candidates = [a for a in self.anchors if a["epoch"] == self.epoch and
                      .35 < np.linalg.norm(self.pose[:2]-a["xy"]) < 1.5]
        return candidates[-1] if candidates else None
