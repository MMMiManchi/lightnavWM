"""Remember actual route views without promoting them to next-room arrivals."""

import math

import numpy as np
import torch


class ObservedRouteJournal:
    def __init__(self, session, execution_ref):
        self.session, self.execution_ref = session, str(execution_ref)
        self.anchor = None
        self.records = []

    def observe(self, reference, raw_feature, feedback):
        pose = feedback.get("local_pose")
        if (pose is None or not feedback.get("measurement", {}).get("valid") or
                feedback.get("measurement", {}).get("source") != "actual_rgbd_pnp" or
                not feedback.get("actual_execution_acknowledged") or feedback.get("continuity_broken")):
            self.anchor = None
            return
        current = (reference, np.asarray(raw_feature).copy(), np.asarray(pose), feedback["pose_epoch"])
        if self.anchor is None or self.anchor[3] != current[3]:
            self.anchor = current
            return
        old = self.anchor
        delta = current[2][:2] - old[2][:2]
        distance = float(np.linalg.norm(delta))
        if distance < .6:
            return
        self.anchor = current
        if distance > 2.:
            return
        yaw = old[2][2]
        right = math.cos(yaw) * delta[0] + math.sin(yaw) * delta[1]
        forward = -math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1]
        measured = {"source": "actual_rgbd_pnp", "continuous": True,
                    "source_camera_right_forward": [right, forward], "pose_epoch": current[3],
                    "source_local_pose": old[2].tolist(), "target_local_pose": current[2].tolist()}
        predictor = self.session.predictor
        with torch.inference_mode():
            raw = torch.tensor(np.stack([old[1], current[1]]), device=predictor.feature_mean.device)
            source, target = predictor.normalize(raw).cpu().numpy()
        sequence = self.session.bank.record_observed(context=self.session.context, episode_id=self.session.episode_id,
            event_id=f"route:{old[0]}:{reference}", source_observation=old[0], target_observation=reference,
            execution_ref=self.execution_ref, source_feature=source, observed_future=target,
            branch_direction=[-right, forward], executed=True, arrived=False, outcome="observed_span", measured_span=measured)
        self.records.append({"sequence": sequence, "source_observation": old[0], "target_observation": reference,
                             "outcome_kind": "observed_span", "measured_span": measured,
                             "next_event_arrival_claimed": False, "parameters_updated": False})
