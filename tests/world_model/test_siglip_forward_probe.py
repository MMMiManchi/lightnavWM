"""CPU checks for the SigLIP probe gates. The 3.5 GB forward is not run here."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.world_model.siglip_forward_probe import (
    ProbeError,
    assert_audited_preprocessor,
    assert_processor_contract,
    require_pinned_gpu,
    tensor_summary,
)


class _Processor:
    def __init__(self, height: int):
        self.size = {"height": height, "width": height}
        self.image_mean = (0.5, 0.5, 0.5)
        self.image_std = (0.5, 0.5, 0.5)
        self.do_center_crop = None
        self.do_resize = True
        self.resample = 3
        self.do_rescale = True
        self.rescale_factor = 1.0 / 255.0
        self.do_normalize = True


def test_probe_rejects_unpinned_gpu_wrong_preprocessor_and_default_224(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ProbeError, match="CUDA_VISIBLE_DEVICES"):
        require_pinned_gpu(2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert require_pinned_gpu(2) == "2"
    bad = tmp_path / "preprocessor_config.json"
    bad.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProbeError, match="sha256"):
        assert_audited_preprocessor(bad)
    with pytest.raises(ProbeError, match="384"):
        assert_processor_contract(_Processor(224))
    assert assert_processor_contract(_Processor(384))["height"] == 384


def test_pooler_predictor_is_much_smaller_than_a_flattened_token_vector():
    from experiments.world_model.predictor import ResidualLatentPredictor

    count = sum(parameter.numel() for parameter in ResidualLatentPredictor(1152).parameters())
    assert count == 153_219
    assert 1152 * 4 == 4608
    assert 729 * 1152 * 4 == 3_359_232


def test_tensor_summary_counts_nonfinite_values_without_putting_them_in_the_range():
    values = torch.tensor([0.0, 2.0, float("nan")])
    summary = tensor_summary(values)
    assert summary["shape"] == [3]
    assert summary["finite_count"] == 2
    assert summary["nonfinite_count"] == 1
    assert summary["min"] == 0.0
    assert summary["max"] == 2.0
