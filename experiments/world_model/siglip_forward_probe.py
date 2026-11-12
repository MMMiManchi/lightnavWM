"""One-image SigLIP vision-tower probe. This does not write a feature cache.

The process must be started with ``CUDA_VISIBLE_DEVICES`` set to the single
chosen GPU. It loads ``SiglipVisionModel`` only, checks the audited weight and
preprocessor hashes, and exits if either check fails. It does not try another
device or another encoder.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Any

from experiments.world_model.jsonutil import dumps, sha256_file, to_builtin

AUDITED_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
AUDITED_WEIGHT_SHA256 = "ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a"
AUDITED_PREPROCESSOR_SHA256 = "f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8"
LIGHTNAV_CHECKPOINT_SHA256 = "ffc4a925378a881afa761865048eb8d07c55cacf5eaf66548b6641c39f67af18"
MIN_FREE_BYTES = 8 * 1024**3
DEFAULT_SNAPSHOT = Path(
    "~/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots"
).expanduser() / AUDITED_REVISION


class ProbeError(RuntimeError):
    pass


def require_pinned_gpu(index: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(index):
        raise ProbeError(
            f"CUDA_VISIBLE_DEVICES must be exactly {index!r} so this process cannot see another GPU; got {visible!r}"
        )
    return visible


def assert_audited_preprocessor(path: Path) -> str:
    if not path.is_file():
        raise ProbeError(f"audited preprocessor is missing: {path}")
    digest = sha256_file(path)
    if digest != AUDITED_PREPROCESSOR_SHA256:
        raise ProbeError(f"preprocessor sha256 {digest} does not match the audited revision file")
    return digest


def _size_hw(size: Any) -> tuple[int, int]:
    if isinstance(size, dict):
        return int(size["height"]), int(size["width"])
    return int(size.height), int(size.width)


def assert_processor_contract(processor: Any) -> dict[str, Any]:
    height, width = _size_hw(processor.size)
    mean = tuple(float(value) for value in processor.image_mean)
    std = tuple(float(value) for value in processor.image_std)
    center_crop = getattr(processor, "do_center_crop", None)
    contract = {
        "height": height,
        "width": width,
        "do_center_crop": center_crop,
        "do_resize": bool(processor.do_resize),
        "resample": int(processor.resample),
        "do_rescale": bool(processor.do_rescale),
        "rescale_factor": float(processor.rescale_factor),
        "do_normalize": bool(processor.do_normalize),
        "image_mean": mean,
        "image_std": std,
    }
    if (height, width) != (384, 384):
        raise ProbeError(f"refusing processor size {(height, width)}; the audited size is 384")
    if center_crop:
        raise ProbeError("refusing a processor that center-crops")
    if not processor.do_resize or int(processor.resample) != 3:
        raise ProbeError("processor resize does not match the audited bicubic resize")
    if not processor.do_rescale or abs(float(processor.rescale_factor) - (1.0 / 255.0)) > 1e-15:
        raise ProbeError("processor rescale does not match 1/255")
    if mean != (0.5, 0.5, 0.5) or std != (0.5, 0.5, 0.5) or not processor.do_normalize:
        raise ProbeError("processor normalization does not match mean/std 0.5")
    return contract


def tensor_summary(values: Any) -> dict[str, Any]:
    import torch

    flat = values.detach().float().cpu().reshape(-1)
    finite = torch.isfinite(flat)
    data = flat[finite]
    summary: dict[str, Any] = {
        "shape": [int(dim) for dim in values.shape],
        "dtype": str(values.dtype).removeprefix("torch."),
        "device": str(values.device),
        "numel": int(values.numel()),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
        "requires_grad": bool(values.requires_grad),
    }
    if data.numel():
        summary.update(
            {
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "std": float(data.std(unbiased=False)),
            }
        )
    return summary


def max_abs_diff(left: Any, right: Any) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(dumps(payload) + "\n", encoding="utf-8")


def _prepare_output(out: Path, preprocessor: Path) -> Path:
    if out.exists():
        raise ProbeError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    processor_dir = out / "processor"
    processor_dir.mkdir()
    shutil.copyfile(preprocessor, processor_dir / "preprocessor_config.json")
    _write(
        out / "preprocessor_source.json",
        {
            "sha256": AUDITED_PREPROCESSOR_SHA256,
            "byte_count": preprocessor.stat().st_size,
            "origin": (
                "google/siglip-so400m-patch14-384 revision "
                f"{AUDITED_REVISION} preprocessor_config.json"
            ),
            "local_snapshot_contains_file": False,
            "copied_from": str(preprocessor),
            "copied_into_snapshot": False,
            "network_fetch_this_run": False,
        },
    )
    return processor_dir


def _forward_once(model: Any, pixels: Any, device: str) -> dict[str, Any]:
    import torch

    torch.cuda.reset_peak_memory_stats(device)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    starter.record()
    with torch.inference_mode():
        outputs = model(pixel_values=pixels)
    ender.record()
    torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - wall_start
    return {
        "outputs": outputs,
        "cuda_event_ms": float(starter.elapsed_time(ender)),
        "wall_s": wall_s,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one offline SigLIP vision forward")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from experiments.world_model.precollection_protocol import (
        assert_feature_extraction_not_authorized,
    )

    assert_feature_extraction_not_authorized()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    report: dict[str, Any] = {
        "status": "failed",
        "formal_feature_cache": False,
        "formal_collection": False,
        "training": False,
        "network_fetch_this_run": False,
        "task_prompt_sha256": LIGHTNAV_CHECKPOINT_SHA256,
        "task_prompt_sha256_is_siglip_weight": False,
        "audited_weight_sha256": AUDITED_WEIGHT_SHA256,
    }
    try:
        visible = require_pinned_gpu(args.gpu)
        revision = (args.snapshot.parent.parent / "refs" / "main").read_text(encoding="utf-8").strip()
        if revision != AUDITED_REVISION or args.snapshot.name != AUDITED_REVISION:
            raise ProbeError(f"snapshot revision {revision} is not {AUDITED_REVISION}")
        weight = args.snapshot / "model.safetensors"
        if not weight.is_file():
            raise ProbeError(f"missing weight file: {weight}")
        weight_sha = sha256_file(weight)
        report["weight_sha256"] = weight_sha
        report["weight_bytes"] = weight.stat().st_size
        if weight_sha != AUDITED_WEIGHT_SHA256:
            raise ProbeError("SigLIP weight sha256 does not match the encoder audit")
        preprocessor_sha = assert_audited_preprocessor(args.preprocessor)
        image_sha = sha256_file(args.image)
        if image_sha != args.image_sha256:
            raise ProbeError("image sha256 does not match the observation record")
        processor_dir = _prepare_output(args.out, args.preprocessor)
        report["preprocessor_sha256"] = preprocessor_sha
        report["image"] = {
            "path": str(args.image),
            "sha256": image_sha,
            "source_split": "train",
            "pilot_run": "output/world_model_data/pilot_20261023_train3_r2",
            "used_as_formal_sample": False,
        }

        import torch
        from transformers import SiglipImageProcessor, SiglipVisionModel

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ProbeError(f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}")
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        report["gpu"] = {
            "cuda_visible_devices": visible,
            "torch_index": 0,
            "name": torch.cuda.get_device_name(0),
            "free_bytes_before_load": int(free_bytes),
            "total_bytes": int(total_bytes),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
        }
        if free_bytes < MIN_FREE_BYTES:
            raise ProbeError(f"GPU {args.gpu} has {free_bytes} free bytes; not switching devices")

        import numpy as np

        image = np.load(args.image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ProbeError(f"image is not HWC uint8 RGB, got dtype {image.dtype} shape {image.shape}")
        processor = SiglipImageProcessor.from_pretrained(processor_dir, local_files_only=True)
        contract = assert_processor_contract(processor)
        first = processor(images=image, return_tensors="pt")["pixel_values"]
        second = processor(images=image, return_tensors="pt")["pixel_values"]
        if tuple(first.shape) != (1, 3, 384, 384) or first.dtype != torch.float32:
            raise ProbeError(f"pixel tensor is {tuple(first.shape)} {first.dtype}")
        if not torch.isfinite(first).all():
            raise ProbeError("pixel tensor is not finite")
        report["processor_contract"] = contract
        report["pixel_values"] = tensor_summary(first)
        report["preprocessor_repeat"] = {
            "label": "same image processed twice; not a feature cache",
            "max_abs_diff": max_abs_diff(first, second),
        }

        model = SiglipVisionModel.from_pretrained(args.snapshot, local_files_only=True)
        if hasattr(model, "text_model"):
            raise ProbeError("full SigLIP model was loaded; this probe allows only the vision tower")
        model.eval()
        device = "cuda:0"
        torch.cuda.reset_peak_memory_stats(device)
        model.to(device)
        torch.cuda.synchronize(device)
        pixels = first.to(device)
        report["load"] = {
            "class_name": type(model).__name__,
            "training": bool(model.training),
            "parameter_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        primary = _forward_once(model, pixels, device)
        repeat = _forward_once(model, pixels, device)
        last = primary["outputs"].last_hidden_state
        pooled = primary["outputs"].pooler_output
        report["forward"] = {
            "label": "primary forward",
            "cuda_event_ms": primary["cuda_event_ms"],
            "wall_s": primary["wall_s"],
            "peak_allocated_bytes": primary["peak_allocated_bytes"],
            "peak_reserved_bytes": primary["peak_reserved_bytes"],
            "oom": False,
        }
        report["last_hidden_state"] = tensor_summary(last)
        report["pooler_output"] = tensor_summary(pooled)
        report["repeatability_probe"] = {
            "label": "second forward of the same pixel tensor; not a feature cache",
            "cuda_event_ms": repeat["cuda_event_ms"],
            "wall_s": repeat["wall_s"],
            "last_hidden_state_max_abs_diff": max_abs_diff(last, repeat["outputs"].last_hidden_state),
            "pooler_output_max_abs_diff": max_abs_diff(pooled, repeat["outputs"].pooler_output),
        }
        report["feature_layer_selected"] = None
        report["status"] = "completed"
    except ProbeError as exc:
        report["error"] = str(exc)
        if args.out.exists():
            _write(args.out / "forward_report.json", report)
        print(dumps({"status": report["status"], "error": str(exc)}))
        return 2
    except Exception as exc:
        oom = exc.__class__.__name__ == "OutOfMemoryError"
        report["error"] = f"{exc.__class__.__name__}: {exc}"
        report.setdefault("forward", {})
        report["forward"]["oom"] = oom
        if args.out.exists():
            _write(args.out / "forward_report.json", report)
        print(dumps({"status": report["status"], "error": report["error"], "oom": oom}))
        return 3

    _write(args.out / "forward_report.json", report)
    (args.out / "probe.log").write_text(dumps(to_builtin(report)) + "\n", encoding="utf-8")
    print(dumps({"status": "completed", "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
