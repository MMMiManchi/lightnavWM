"""Compare two processes encoding the same RGB on one GPU.

The numeric bar is fixed before the run: max absolute error 0, MSE 0, and
cosine similarity 1. A same-process repeat on this stack was bitwise identical,
and this check has no separate noise model. A nonzero difference fails. The
report may mention 1e-5 as a later review value; that value is not a pass.

Outputs stay under ``output/world_model_diagnostics`` and are marked
``diagnostic_only``. They are not a feature cache and are not training data.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.world_model.feature_cache import AUDITED_PREPROCESSOR
from experiments.world_model.jsonutil import dumps, sha256_file
from experiments.world_model.siglip_forward_probe import (
    AUDITED_REVISION,
    AUDITED_WEIGHT_SHA256,
    DEFAULT_SNAPSHOT,
    ProbeError,
    assert_audited_preprocessor,
    assert_processor_contract,
    require_pinned_gpu,
)

CROSS_PROCESS_MAX_ABS_ATOL = 0.0
CROSS_PROCESS_MSE_ATOL = 0.0
CROSS_PROCESS_COSINE_MIN = 1.0
SUGGESTED_REVIEW_ATOL = 1e-5
PILOT_RUN = Path("output/world_model_data/pilot_20261023_train3_r2")
OUTPUTS = ("pooler_output", "last_hidden_state")


def compare_features(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Apply the predeclared tolerances. This function does not choose a new bar."""
    result: dict[str, Any] = {
        "left_shape": [int(dim) for dim in left.shape],
        "right_shape": [int(dim) for dim in right.shape],
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "left_finite": bool(np.isfinite(left).all()),
        "right_finite": bool(np.isfinite(right).all()),
        "predeclared_max_abs_atol": CROSS_PROCESS_MAX_ABS_ATOL,
        "predeclared_mse_atol": CROSS_PROCESS_MSE_ATOL,
        "predeclared_cosine_min": CROSS_PROCESS_COSINE_MIN,
        "suggested_review_atol_not_used_for_pass": SUGGESTED_REVIEW_ATOL,
    }
    if left.shape != right.shape or left.dtype != right.dtype or left.dtype != np.float32:
        result["passed"] = False
        result["reason"] = "shape or dtype mismatch"
        return result
    if not result["left_finite"] or not result["right_finite"]:
        result["passed"] = False
        result["reason"] = "non-finite values"
        return result
    diff = left.astype(np.float64) - right.astype(np.float64)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    mse = float(np.mean(diff * diff)) if diff.size else 0.0
    left_norm = float(np.linalg.norm(left.astype(np.float64)))
    right_norm = float(np.linalg.norm(right.astype(np.float64)))
    cosine = None
    if np.array_equal(left, right) and left_norm > 0.0:
        cosine = 1.0
    elif left_norm > 0.0 and right_norm > 0.0:
        cosine = float(
            np.dot(left.astype(np.float64).ravel(), right.astype(np.float64).ravel()) / (left_norm * right_norm)
        )
    result.update({"max_abs": max_abs, "mse": mse, "cosine": cosine})
    result["passed"] = bool(
        max_abs <= CROSS_PROCESS_MAX_ABS_ATOL
        and mse <= CROSS_PROCESS_MSE_ATOL
        and cosine is not None
        and cosine >= CROSS_PROCESS_COSINE_MIN
    )
    result["reason"] = "within predeclared tolerance" if result["passed"] else "outside predeclared tolerance"
    return result


def select_diagnostic_frames(run_dir: Path = PILOT_RUN) -> list[dict[str, str]]:
    """One existing RGB from each of two train scenes. These frames stay diagnostic."""
    chosen: dict[str, dict[str, str]] = {}
    observations = run_dir / "observations.jsonl"
    if not observations.is_file():
        raise ProbeError(f"diagnostic RGB source is missing: {observations}")
    for line in observations.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("step_index") != 0 or not row.get("rgb"):
            continue
        scene = str(row["episode_uid"]).split("|")[2]
        if scene in chosen:
            continue
        path = run_dir / row["rgb"]["path"]
        digest = sha256_file(path)
        if digest != row["rgb"]["sha256"]:
            raise ProbeError(f"RGB bytes do not match the observation record: {path}")
        chosen[scene] = {"scene": scene, "path": str(path), "sha256": digest}
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        raise ProbeError("need RGB from two different scenes")
    return [chosen[scene] for scene in sorted(chosen)]


def _runtime_record() -> dict[str, Any]:
    import torch
    import transformers

    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_index": 0,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _encode(image_path: Path, snapshot: Path, processor_dir: Path) -> dict[str, np.ndarray]:
    import torch
    from transformers import SiglipImageProcessor, SiglipVisionModel

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ProbeError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")
    image = np.load(image_path)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ProbeError(f"image is not HWC uint8 RGB: {image.dtype} {image.shape}")
    processor = SiglipImageProcessor.from_pretrained(processor_dir, local_files_only=True)
    assert_processor_contract(processor)
    pixels = processor(images=image, return_tensors="pt")["pixel_values"]
    if tuple(pixels.shape) != (1, 3, 384, 384) or pixels.dtype != torch.float32:
        raise ProbeError(f"pixel tensor is {tuple(pixels.shape)} {pixels.dtype}")
    model = SiglipVisionModel.from_pretrained(snapshot, local_files_only=True)
    if hasattr(model, "text_model"):
        raise ProbeError("full SigLIP model was loaded; this probe allows only the vision tower")
    model.eval()
    device = "cuda:0"
    model.to(device)
    with torch.inference_mode():
        outputs = model(pixel_values=pixels.to(device))
    encoded = {
        "pooler_output": outputs.pooler_output.detach().float().cpu().numpy()[0],
        "last_hidden_state": outputs.last_hidden_state.detach().float().cpu().numpy()[0],
    }
    if encoded["pooler_output"].shape != (1152,) or encoded["last_hidden_state"].shape != (729, 1152):
        raise ProbeError(f"unexpected shapes {[array.shape for array in encoded.values()]}")
    if not np.isfinite(encoded["pooler_output"]).all() or not np.isfinite(encoded["last_hidden_state"]).all():
        raise ProbeError("encoder output is not finite")
    return encoded


def _worker(args: argparse.Namespace) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    require_pinned_gpu(args.gpu)
    weight = args.snapshot / "model.safetensors"
    if sha256_file(weight) != AUDITED_WEIGHT_SHA256 or args.snapshot.name != AUDITED_REVISION:
        raise ProbeError("worker weight or revision does not match the audited snapshot")
    if sha256_file(args.image) != args.image_sha256:
        raise ProbeError("worker image bytes do not match the requested sha256")
    assert_audited_preprocessor(args.preprocessor)
    if args.out.exists():
        raise ProbeError(f"refusing to overwrite {args.out}")
    processor_dir = args.out.parent / f"processor-{os.getpid()}"
    if processor_dir.exists():
        raise ProbeError(f"refusing to reuse {processor_dir}")
    processor_dir.mkdir()
    try:
        (processor_dir / "preprocessor_config.json").write_bytes(args.preprocessor.read_bytes())
        arrays = _encode(args.image, args.snapshot, processor_dir)
        np.savez(args.out, **{name: arrays[name] for name in OUTPUTS})
        meta = {
            "diagnostic_only": True,
            "formal_feature_cache": False,
            "feature_layer_selected": None,
            "pid": os.getpid(),
            "image_sha256": args.image_sha256,
            "weight_sha256": AUDITED_WEIGHT_SHA256,
            "preprocess_sha256": sha256_file(args.preprocessor),
            "runtime": _runtime_record(),
        }
        args.out.with_suffix(".json").write_text(dumps(meta) + "\n", encoding="utf-8")
    finally:
        for child in processor_dir.glob("*"):
            child.unlink()
        processor_dir.rmdir()
    return 0


def _launch_worker(gpu: int, image: dict[str, str], snapshot: Path, out: Path, label: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    command = [
        sys.executable,
        "-m",
        "experiments.world_model.siglip_repro_probe",
        "--worker",
        "--gpu",
        str(gpu),
        "--snapshot",
        str(snapshot),
        "--preprocessor",
        str(AUDITED_PREPROCESSOR),
        "--image",
        image["path"],
        "--image-sha256",
        image["sha256"],
        "--out",
        str(out / f"{label}.npz"),
    ]
    completed = subprocess.run(command, env=env, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ProbeError(f"{label} exited {completed.returncode}: {detail[-2000:]}")


def _driver(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise ProbeError(f"refusing to overwrite {args.out}")
    frames = select_diagnostic_frames()
    args.out.mkdir(parents=True)
    report: dict[str, Any] = {
        "diagnostic_only": True,
        "formal_feature_cache": False,
        "formal_collection": False,
        "training": False,
        "feature_layer_selected": None,
        "used_as_formal_sample": False,
        "gpu_index": args.gpu,
        "cross_gpu": False,
        "tolerance_basis": (
            "Same-process repeat encoding on this software stack was bitwise identical. "
            "Two processes on one GPU therefore pass only when max absolute error, MSE, "
            "and cosine similarity meet 0, 0, and 1. 1e-5 is recorded as a possible later "
            "review value and does not make a nonzero result a pass."
        ),
        "predeclared_max_abs_atol": CROSS_PROCESS_MAX_ABS_ATOL,
        "predeclared_mse_atol": CROSS_PROCESS_MSE_ATOL,
        "predeclared_cosine_min": CROSS_PROCESS_COSINE_MIN,
        "suggested_review_atol_not_used_for_pass": SUGGESTED_REVIEW_ATOL,
        "frames": frames,
        "status": "failed",
    }
    try:
        for index, frame in enumerate(frames):
            _launch_worker(args.gpu, frame, args.snapshot, args.out, f"process_a_scene_{index}")
            _launch_worker(args.gpu, frame, args.snapshot, args.out, f"process_b_scene_{index}")
            left = np.load(args.out / f"process_a_scene_{index}.npz")
            right = np.load(args.out / f"process_b_scene_{index}.npz")
            report.setdefault("comparisons", []).append(
                {
                    "scene": frame["scene"],
                    "image_sha256": frame["sha256"],
                    "outputs": {name: compare_features(left[name], right[name]) for name in OUTPUTS},
                }
            )
        passed = all(
            item["passed"]
            for comparison in report["comparisons"]
            for item in comparison["outputs"].values()
        )
        report["status"] = "passed" if passed else "failed"
    except ProbeError as exc:
        report["status"] = "not_executed" if "not switching" in str(exc) else "failed"
        report["error"] = str(exc)
    (args.out / "repro_report.json").write_text(dumps(report) + "\n", encoding="utf-8")
    print(dumps({"status": report["status"], "out": str(args.out)}))
    return 0 if report["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-process SigLIP diagnostic")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--preprocessor", type=Path, default=AUDITED_PREPROCESSOR)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--image-sha256")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    from experiments.world_model.precollection_protocol import (
        assert_feature_extraction_not_authorized,
    )

    assert_feature_extraction_not_authorized()
    if args.worker:
        return _worker(args)
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out = Path("output/world_model_diagnostics") / f"siglip_repro_{stamp}"
    return _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
