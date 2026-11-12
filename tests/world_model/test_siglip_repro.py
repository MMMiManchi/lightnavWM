"""The cross-process numeric bar is fixed before any GPU measurement."""

from __future__ import annotations

import numpy as np

from experiments.world_model.siglip_repro_probe import (
    CROSS_PROCESS_COSINE_MIN,
    CROSS_PROCESS_MAX_ABS_ATOL,
    CROSS_PROCESS_MSE_ATOL,
    compare_features,
)


def test_predeclared_tolerance_rejects_a_nonzero_difference():
    assert CROSS_PROCESS_MAX_ABS_ATOL == 0.0
    assert CROSS_PROCESS_MSE_ATOL == 0.0
    assert CROSS_PROCESS_COSINE_MIN == 1.0
    values = np.arange(4, dtype=np.float32).reshape(2, 2)
    exact = compare_features(values, values.copy())
    assert exact["passed"] is True
    assert exact["max_abs"] == 0.0
    assert exact["mse"] == 0.0
    assert exact["cosine"] == 1.0

    shifted = values.copy()
    shifted.reshape(-1)[0] += np.float32(1e-8)
    drifted = compare_features(values, shifted)
    assert drifted["passed"] is False
    assert drifted["max_abs"] > 0.0
    assert drifted["suggested_review_atol_not_used_for_pass"] == 1e-5

    wrong_shape = compare_features(values, values.reshape(-1))
    assert wrong_shape["passed"] is False
    nonfinite = values.copy()
    nonfinite.reshape(-1)[1] = np.float32("nan")
    assert compare_features(values, nonfinite)["passed"] is False
