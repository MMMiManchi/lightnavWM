"""Continuous observed route states, kept separate from decision-node labels."""

from __future__ import annotations

import numpy as np


def route_anchors(frames, *, stride_m=2.0, minimum_tail_m=1.0, minimum_steps=10):
    if len(frames) < 2:
        raise ValueError("a sequence needs an observed future")
    positions = np.array([f["position"] for f in frames], dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError("non-finite offline route geometry")
    path = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    chosen = [0]
    for i in range(1, len(frames) - 1):
        if (i - chosen[-1] >= minimum_steps and len(frames) - 1 - i >= minimum_steps
                and path[i] - path[chosen[-1]] >= stride_m
                and path[-1] - path[i] >= minimum_tail_m):
            chosen.append(i)
    chosen.append(len(frames) - 1)
    return chosen, path


def route_actions(count, bearing):
    if count < 1:
        raise ValueError("a route needs a transition")
    # [sin bearing, cos bearing, route-observation option, next-event option].
    # Future executed controls or the eventual target pose are never conditions.
    result = np.tile([0.0, 1.0, 1.0, 0.0], (count, 1)).astype(np.float32)
    result[0, :2] = [np.sin(bearing), np.cos(bearing)]
    return result
